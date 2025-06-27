from enum import Enum
from xdsl.dialects.experimental import fir
from xdsl.utils.hints import isa
from xdsl.ir import Block, Region
from xdsl.dialects import builtin, func, arith, scf, cf, math
from ftn.dialects import ftn_relative_cf

from ftn.transforms.to_core.misc.fortran_code_description import ProgramState
from ftn.transforms.to_core.misc.ssa_context import SSAValueCtx

from ftn.transforms.to_core.utils import create_index_constant

import ftn.transforms.to_core.expressions as expressions
import ftn.transforms.to_core.statements as statements

LoopStepDirection = Enum("LoopStepDirection", ["INCREMENT", "DECREMENT", "UNKNOWN"])


def get_loop_arg_val_if_known(ssa):
    # Grabs out the value corresponding the to loops input SSA
    # if this can be found statically, otherwise return None
    ssa_base = ssa
    if isa(ssa_base.owner, arith.IndexCastOp):
        ssa_base = ssa_base.owner.input

    if isa(ssa_base.owner, arith.ConstantOp):
        assert isa(ssa_base.type, builtin.IndexType) or isa(
            ssa_base.type, builtin.IntegerType
        )
        val = ssa_base.owner.value.value.data
        return val
    else:
        return None


def determine_loop_step_direction(step_ssa):
    # Determines the loop step direction
    step_val = get_loop_arg_val_if_known(step_ssa)
    if step_val is not None:
        if step_val > 0:
            return LoopStepDirection.INCREMENT
        else:
            return LoopStepDirection.DECREMENT
    else:
        return LoopStepDirection.UNKNOWN


def handle_conditional_true_or_false_region(
    program_state: ProgramState, ctx: SSAValueCtx, region: Region
):
    arg_types = []

    if len(region.blocks) > 0:
        for arg in region.blocks[0].args:
            arg_types.append(arg.type)

    new_block = Block(arg_types=arg_types)

    if len(region.blocks) > 0:
        for fir_arg, std_arg in zip(region.blocks[0].args, new_block.args):
            ctx[fir_arg] = std_arg

        region_body_ops = []
        for single_op in region.blocks[0].ops:
            region_body_ops += statements.translate_stmt(program_state, ctx, single_op)

        assert isa(region_body_ops[-1], scf.YieldOp)
        new_block.add_ops(region_body_ops)
    else:
        new_block.add_op(scf.YieldOp())

    return new_block


def generate_index_inversion_at_start_of_loop(
    index_ssa, lower_ssa, upper_ssa, target_type
):
    # This is the index inversion required at the start of the loop if working backwards
    inversion_ops = []
    reduce_idx_from_start = arith.SubiOp(index_ssa, lower_ssa)
    invert_idx = arith.SubiOp(upper_ssa, reduce_idx_from_start)
    inversion_ops += [reduce_idx_from_start, invert_idx]
    if isa(target_type, builtin.IntegerType):
        index_cast = arith.IndexCastOp(invert_idx.results[0], target_type)
        inversion_ops.append(index_cast)
    return inversion_ops


def generate_convert_step_to_absolute(step_ssa):
    # Generates the MLIR to convert an index ssa to it's absolute (positive) form
    assert isa(step_ssa.type, builtin.IndexType)
    cast_int = arith.IndexCastOp(step_ssa, builtin.i64)
    step_absolute = math.AbsIOp(cast_int.results[0])
    cast_abs = arith.IndexCastOp(step_absolute.results[0], builtin.IndexType())
    return [cast_int, step_absolute, cast_abs], cast_abs.results[0]


def translate_do_loop(program_state: ProgramState, ctx: SSAValueCtx, op: fir.DoLoopOp):
    if ctx.contains(op.results[1]):
        for fir_result in op.results[1:]:
            assert ctx.contains(fir_result)
        return []

    lower_bound_ops = expressions.translate_expr(program_state, ctx, op.lowerBound)
    upper_bound_ops = expressions.translate_expr(program_state, ctx, op.upperBound)
    step_ops = expressions.translate_expr(program_state, ctx, op.step)
    initarg_ops = expressions.translate_expr(program_state, ctx, op.initArgs)

    lower_bound = ctx[op.lowerBound]
    upper_bound = ctx[op.upperBound]

    assert len(op.regions) == 1
    assert len(op.regions[0].blocks) == 1

    arg_types = []
    for arg in op.regions[0].blocks[0].args:
        arg_types.append(arg.type)

    new_block = Block(arg_types=arg_types)

    for fir_arg, std_arg in zip(op.regions[0].blocks[0].args, new_block.args):
        ctx[fir_arg] = std_arg

    # Determine the step direction (increment, decrement or unknown)
    step_direction = determine_loop_step_direction(ctx[op.step])

    loop_body_ops = []
    if step_direction == LoopStepDirection.DECREMENT:
        # If this is stepping down, then we assume the loop is do high, low, step
        # as this follows Fortran semantics, for instance do 10,1,-1 would count
        # down from 10, whereas do 1,10,-1 would not execute any iterations
        # scf.for always counts up, therefore we assume that it is high to low
        # and these need swapped around
        t = upper_bound
        upper_bound = lower_bound
        lower_bound = t

        # The loop counter is incrementing, we need to invert this based on the upper bound
        # to get the value as if the loop was actually counting downwards. Note that there are
        # two values each iteration, the actual loop index driven by scf.for (which is an index)
        # and an i32 integer which is the index we are updating from one iteration to the next
        # it is the later that is written to the loop variable. This integer value tracks
        # the scf index (at the end of the loop it is incremented based on the index)
        loop_body_ops += generate_index_inversion_at_start_of_loop(
            new_block.args[0],
            lower_bound,
            upper_bound,
            op.regions[0].blocks[0].args[1].type,
        )
        del ctx[op.regions[0].blocks[0].args[1]]
        ctx[op.regions[0].blocks[0].args[1]] = loop_body_ops[-1].results[0]

        # The step must be positive
        step_abs_ops, step_abs_ssa = generate_convert_step_to_absolute(ctx[op.step])
        step_ops += step_abs_ops
        del ctx[op.step]
        ctx[op.step] = step_abs_ssa
    elif step_direction == LoopStepDirection.UNKNOWN:
        # We don't know if the step is positive or not, as this is from an input
        # variable. The same as the above, but support for decrement is driven
        # by conditionals
        loop_one_const = create_index_constant(1)
        loop_idx_cmp = arith.CmpiOp(
            ctx[op.step], loop_one_const, 2
        )  # checking if step is less than 1

        reduction_ops = generate_index_inversion_at_start_of_loop(
            new_block.args[0],
            lower_bound,
            upper_bound,
            op.regions[0].blocks[0].args[1].type,
        )
        # We are going to wrap this in a conditional, therefore yield the result of the index inversion
        reduction_ops.append(scf.YieldOp(reduction_ops[-1]))
        # Wrap in a conditional, if so then invert the index, otherwise just send the index through,
        # see above explanation for more details on this step
        scf_if = scf.IfOp(
            loop_idx_cmp.results[0],
            [ctx[op.regions[0].blocks[0].args[1]].type],
            reduction_ops,
            [scf.YieldOp(ctx[op.regions[0].blocks[0].args[1]])],
        )
        loop_body_ops += [loop_one_const, loop_idx_cmp, scf_if]
        del ctx[op.regions[0].blocks[0].args[1]]
        ctx[op.regions[0].blocks[0].args[1]] = scf_if.results[0]

        # This does the swapping between the lower and upper bounds, as above if we count down
        # then indexes will be high to low, i.e. do high, low, step but these need swapped
        # for the scf loop. This is driven by the conditional on the step
        outer_const = create_index_constant(1)
        outer_idx_cmp = arith.CmpiOp(ctx[op.step], outer_const, 2)
        nscf_if = scf.IfOp(
            outer_idx_cmp.results[0],
            [upper_bound.type, lower_bound.type],
            [scf.YieldOp(upper_bound, lower_bound)],
            [scf.YieldOp(lower_bound, upper_bound)],
        )
        lower_bound = nscf_if.results[0]
        upper_bound = nscf_if.results[1]
        initarg_ops += [outer_const, outer_idx_cmp, nscf_if]

        # Regardless of whether the step is positive or not then we convert
        # it to positive, as if it is already then it doesn't change anything
        # and it's not worth the conditional check
        step_abs_ops, step_abs_ssa = generate_convert_step_to_absolute(ctx[op.step])
        step_ops += step_abs_ops
        del ctx[op.step]
        ctx[op.step] = step_abs_ssa

    for loop_op in op.regions[0].blocks[0].ops:
        loop_body_ops += statements.translate_stmt(program_state, ctx, loop_op)

    # We need to add one to the upper index, as scf.for is not inclusive on the
    # top bound, whereas fir for loops are
    one_val_op = create_index_constant(1)
    add_op = arith.AddiOp(upper_bound, one_val_op)
    initarg_ops += [one_val_op, add_op]
    upper_bound = add_op.results[0]

    # The fir result has both the index and iterargs, whereas the yield has only
    # the iterargs. Therefore need to rebuild the yield with the first argument (the index)
    # removed from it
    yield_op = loop_body_ops[-1]
    assert isa(yield_op, scf.YieldOp)
    new_yieldop = scf.YieldOp(*yield_op.arguments[1:])
    del loop_body_ops[-1]
    loop_body_ops.append(new_yieldop)

    new_block.add_ops(loop_body_ops)

    scf_for_loop = scf.ForOp(
        lower_bound, upper_bound, ctx[op.step], [ctx[op.initArgs]], new_block
    )

    for index, scf_result in enumerate(scf_for_loop.results):
        ctx[op.results[index + 1]] = scf_result

    return lower_bound_ops + upper_bound_ops + step_ops + initarg_ops + [scf_for_loop]


def translate_return(program_state: ProgramState, ctx: SSAValueCtx, op: func.ReturnOp):
    ssa_to_return = []
    args_ops = []
    for arg in op.arguments:
        args_ops += expressions.translate_expr(program_state, ctx, arg)
        ssa_to_return.append(ctx[arg])
    new_return = func.ReturnOp(*ssa_to_return)
    return args_ops + [new_return]


def translate_result(program_state: ProgramState, ctx: SSAValueCtx, op: fir.ResultOp):
    ops_list = []
    ssa_list = []
    for operand in op.operands:
        expr_ops = expressions.translate_expr(program_state, ctx, operand)
        ops_list += expr_ops
        ssa_list.append(ctx[operand])
    yield_op = scf.YieldOp(*ssa_list)
    return ops_list + [yield_op]


def translate_iterate_while(
    program_state: ProgramState, ctx: SSAValueCtx, op: fir.IterateWhileOp
):
    # FIR's iterate while is like the do loop as it has a numeric counter but it also has an i1
    # flag to drive whether to continue (flag=true) or exit (flag=false). We map this to scf.While
    # however it's a bit of a different operation, so we need to more manual things here to achieve this
    if ctx.contains(op.results[1]):
        for fir_result in op.results[1:]:
            assert ctx.contains(fir_result)
        return []

    lower_bound_ops = expressions.translate_expr(program_state, ctx, op.lowerBound)
    upper_bound_ops = expressions.translate_expr(program_state, ctx, op.upperBound)
    step_ops = expressions.translate_expr(program_state, ctx, op.step)
    iterate_in_ops = expressions.translate_expr(program_state, ctx, op.iterateIn)
    initarg_ops = expressions.translate_expr(program_state, ctx, op.initArgs)

    zero_const = create_index_constant(0)
    # Will be true if smaller than zero, this is needed because if counting backwards
    # then check it is larger or equal to the upper bound, otherwise it's smaller or equals
    step_op_lt_zero = arith.CmpiOp(ctx[op.step], zero_const, 2)
    step_zero_check_ops = [zero_const, step_op_lt_zero]

    assert len(op.regions) == 1
    assert len(op.regions[0].blocks) == 1

    arg_types = [builtin.IndexType(), builtin.i1, ctx[op.initArgs].type]

    before_block = Block(arg_types=arg_types)

    # Build the index check, the one to use depends on whether the step is positive or not
    true_cmp_op = arith.CmpiOp(before_block.args[0], ctx[op.upperBound], 5)
    true_cmp_block = [true_cmp_op, scf.YieldOp(true_cmp_op)]
    false_cmp_op = arith.CmpiOp(before_block.args[0], ctx[op.upperBound], 3)
    false_cmp_block = [false_cmp_op, scf.YieldOp(false_cmp_op)]
    index_comparison = scf.IfOp(
        step_op_lt_zero, builtin.i1, true_cmp_block, false_cmp_block
    )

    # True if both are true, false otherwise (either the counter or bool can quit out of loop)
    or_comparison = arith.AndIOp(index_comparison, before_block.args[1])
    condition_op = scf.ConditionOp(or_comparison, *before_block.args)

    before_block.add_ops([index_comparison, or_comparison, condition_op])

    after_block = Block(arg_types=arg_types)

    for fir_arg, std_arg in zip(op.regions[0].blocks[0].args, after_block.args):
        ctx[fir_arg] = std_arg

    loop_body_ops = []
    for loop_op in op.regions[0].blocks[0].ops:
        loop_body_ops += statements.translate_stmt(program_state, ctx, loop_op)

    # This updates the loop counter
    update_loop_idx = arith.AddiOp(after_block.args[0], ctx[op.step])

    # Now we grab out the loop update to return, the true or false flag, and the initarg
    yield_op = loop_body_ops[-1]
    assert isa(yield_op, scf.YieldOp)
    ssa_args = [update_loop_idx.results[0], yield_op.arguments[0], after_block.args[2]]

    # Rebuilt yield with these new SSAs
    new_yieldop = scf.YieldOp(*ssa_args)
    del loop_body_ops[-1]
    # Add new yield and the update loop counter to the block
    after_block.add_ops(loop_body_ops + [update_loop_idx, new_yieldop])

    while_return_types = [builtin.IndexType(), builtin.i1, ctx[op.initArgs]]

    scf_while_loop = scf.WhileOp(
        [ctx[op.lowerBound], ctx[op.iterateIn], ctx[op.initArgs]],
        arg_types,
        [before_block],
        [after_block],
    )

    # It is correct to have them this way round, in fir it's i1, index whereas here
    # we have index, i1, index (and we ignore the last index)
    ctx[op.results[0]] = scf_while_loop.results[1]
    ctx[op.results[1]] = scf_while_loop.results[0]

    return (
        lower_bound_ops
        + upper_bound_ops
        + step_ops
        + step_zero_check_ops
        + iterate_in_ops
        + initarg_ops
        + [scf_while_loop]
    )


def translate_conditional(program_state: ProgramState, ctx: SSAValueCtx, op: fir.IfOp):
    # Each function automatically deallocates scope local allocatable arrays at the end,
    # check to see if that is the purpose of this conditional. If so then just ignore it
    is_final_auto_free = check_if_condition_is_end_fn_allocatable_automatic_free(
        op.condition.owner
    )
    if is_final_auto_free:
        return []

    conditional_expr_ops = expressions.translate_expr(program_state, ctx, op.condition)

    true_block = handle_conditional_true_or_false_region(
        program_state, ctx, op.regions[0]
    )
    false_block = handle_conditional_true_or_false_region(
        program_state, ctx, op.regions[1]
    )

    scf_if = scf.IfOp(ctx[op.condition], [], [true_block], [false_block])

    return conditional_expr_ops + [scf_if]


def translate_branch(program_state: ProgramState, ctx: SSAValueCtx, op: cf.BranchOp):
    current_fn_identifier = program_state.getCurrentFnState().fn_identifier
    target_block_index = program_state.function_definitions[
        current_fn_identifier
    ].blocks.index(op.successor)

    ops_list = []
    block_ssas = []
    for arg in op.arguments:
        ops_list += expressions.translate_expr(program_state, ctx, arg)
        block_ssas.append(ctx[arg])
    relative_branch = ftn_relative_cf.BranchOp(
        current_fn_identifier, target_block_index, *block_ssas
    )
    ops_list.append(relative_branch)
    return ops_list


def translate_conditional_branch(
    program_state: ProgramState, ctx: SSAValueCtx, op: cf.ConditionalBranchOp
):
    current_fn_identifier = program_state.getCurrentFnState().fn_identifier
    then_block_index = program_state.function_definitions[
        current_fn_identifier
    ].blocks.index(op.then_block)
    else_block_index = program_state.function_definitions[
        current_fn_identifier
    ].blocks.index(op.else_block)

    ops_list = []
    ops_list += expressions.translate_expr(program_state, ctx, op.cond)

    then_block_ssas = []
    else_block_ssas = []
    for arg in op.then_arguments:
        ops_list += expressions.translate_expr(program_state, ctx, arg)
        then_block_ssas.append(ctx[arg])
    for arg in op.else_arguments:
        ops_list += expressions.translate_expr(program_state, ctx, arg)
        else_block_ssas.append(ctx[arg])

    relative_cond_branch = ftn_relative_cf.ConditionalBranchOp(
        current_fn_identifier,
        ctx[op.cond],
        then_block_index,
        then_block_ssas,
        else_block_index,
        else_block_ssas,
    )
    ops_list.append(relative_cond_branch)

    return ops_list


def check_if_condition_is_end_fn_allocatable_automatic_free(
    condition_op: arith.CmpiOp | arith.CmpfOp,
):
    if isa(condition_op, arith.CmpiOp):
        if isa(condition_op.lhs.owner, fir.ConvertOp):
            return isa(condition_op.lhs.owner.value.type, fir.HeapType) and isa(
                condition_op.lhs.owner.results[0].type, builtin.IntegerType
            )
    return False

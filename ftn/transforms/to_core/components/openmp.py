from xdsl.ir import Block, Region
from xdsl.utils.hints import isa
from xdsl.dialects import builtin, omp

from ftn.transforms.to_core.misc.fortran_code_description import ProgramState
from ftn.transforms.to_core.misc.ssa_context import SSAValueCtx

from ftn.transforms.to_core.utils import create_index_constant

import ftn.transforms.to_core.expressions as expressions
import ftn.transforms.to_core.statements as statements


def handle_var_operand_field(program_state, ctx, var_operands):
    arg_types = []
    vars_ops = []
    vars_ssa = []

    if len(var_operands) > 0:
        for operand in var_operands:
            vars_ops += expressions.translate_expr(program_state, ctx, operand)
            vars_ssa.append(ctx[operand])
            arg_types.append(ctx[operand].type)

    return vars_ops, vars_ssa, arg_types


def handle_opt_operand_field(program_state, ctx, opt_operand):
    arg_types = []
    vars_ops = []
    vars_ssa = []

    if opt_operand is not None:
        vars_ops += expressions.translate_expr(program_state, ctx, opt_operand)
        vars_ssa = ctx[opt_operand]
        arg_types.append(ctx[opt_operand].type)

    return vars_ops, vars_ssa, arg_types


def translate_private(program_state: ProgramState, ctx: SSAValueCtx, op: omp.PrivateOp):
    if len(op.alloc_region.blocks) > 0:
        alloc_region_ops = []
        for single_op in op.alloc_region.blocks[0].ops:
            alloc_region_ops += statements.translate_stmt(program_state, ctx, single_op)
        alloc_region_blocks = [Block(alloc_region_ops)]
    else:
        alloc_region_blocks = []

    if len(op.copy_region.blocks) > 0:
        copy_region_ops = []
        for single_op in op.copy_region.blocks[0].ops:
            copy_region_ops += statements.translate_stmt(program_state, ctx, single_op)
        copy_region_blocks = [Block(copy_region_ops)]
    else:
        copy_region_blocks = []

    if len(op.dealloc_region.blocks) > 0:
        dealloc_region_ops = []
        for single_op in op.dealloc_region.blocks[0].ops:
            dealloc_region_ops += statements.translate_stmt(
                program_state, ctx, single_op
            )
        dealloc_region_blocks = [Block(dealloc_region_ops)]
    else:
        dealloc_region_blocks = []

    return omp.PrivateOp.build(
        regions=[
            Region(alloc_region_blocks),
            Region(copy_region_blocks),
            Region(dealloc_region_blocks),
        ],
        properties={
            "sym_name": op.sym_name,
            "type": op.var_type,
            "data_sharing_type": op.data_sharing_type,
        },
    )


def translate_omp_mapinfo(
    program_state: ProgramState, ctx: SSAValueCtx, op: omp.MapInfoOp
):
    if ctx.contains(op.results[0]):
        return []

    var_ptr_ops = expressions.translate_expr(program_state, ctx, op.var_ptr)
    var_ptr_ssa = ctx[op.var_ptr]
    var_ptr_type = ctx[op.var_ptr].type

    var_ptr_ptr_ops = []
    var_ptr_ptr_ssa = []
    if op.var_ptr_ptr is not None:
        var_ptr_ptr_ops = expressions.translate_expr(program_state, ctx, op.var_ptr_ptr)
        var_ptr_ptr_ssa = [ctx[op.var_ptr_ptr]]

    members_ops = []
    members_ssa = []
    for arg in op.members:
        members_ops += expressions.translate_expr(program_state, ctx, arg)
        members_ssa.append(ctx[arg])

    bounds_ops = []
    bounds_ssa = []
    for arg in op.bounds:
        bounds_ops += expressions.translate_expr(program_state, ctx, arg)
        bounds_ssa.append(bounds_ops[-1].results[0])

    mapinfo_op = omp.MapInfoOp.build(
        operands=[var_ptr_ssa, var_ptr_ptr_ssa, members_ssa, bounds_ssa],
        properties={
            "map_type": op.map_type,
            "name": op.var_name,
            "var_type": var_ptr_type,
        },
        result_types=[var_ptr_type],
    )

    ctx[op.results[0]] = mapinfo_op.results[0]

    return var_ptr_ops + var_ptr_ptr_ops + members_ops + bounds_ops + [mapinfo_op]


def translate_omp_bounds(
    program_state: ProgramState, ctx: SSAValueCtx, op: omp.MapBoundsOp
):
    if ctx.contains(op.results[0]):
        return []

    lower_ops = expressions.translate_expr(program_state, ctx, op.lower_bound)
    lower_ssa = ctx[op.lower_bound]

    upper_ops = expressions.translate_expr(program_state, ctx, op.upper_bound)
    upper_ssa = ctx[op.upper_bound]

    extent_ops = expressions.translate_expr(program_state, ctx, op.extent)
    extent_ssa = ctx[op.extent]

    stride_ops = expressions.translate_expr(program_state, ctx, op.stride)
    stride_ssa = ctx[op.stride]

    start_ops = expressions.translate_expr(program_state, ctx, op.start_idx)
    start_ssa = ctx[op.start_idx]

    bounds_op = omp.MapBoundsOp.build(
        operands=[lower_ssa, upper_ssa, extent_ssa, stride_ssa, start_ssa],
        properties={"stride_in_bytes": op.stride_in_bytes},
        result_types=[omp.MapBoundsType()],
    )

    ctx[op.results[0]] = bounds_op.results[0]

    return lower_ops + upper_ops + extent_ops + stride_ops + start_ops + [bounds_op]


def translate_omp_wsloop(
    program_state: ProgramState, ctx: SSAValueCtx, op: omp.WsLoopOp
):
    arg_types = []

    allocate_vars_ops, allocate_vars_ssa, allocate_vars_types = (
        handle_var_operand_field(program_state, ctx, op.allocate_vars)
    )
    arg_types += allocate_vars_types

    allocator_vars_ops, allocator_vars_ssa, allocator_vars_types = (
        handle_var_operand_field(program_state, ctx, op.allocator_vars)
    )
    arg_types += allocator_vars_types

    linear_vars_ops, linear_vars_ssa, linear_vars_types = handle_var_operand_field(
        program_state, ctx, op.linear_vars
    )
    arg_types += linear_vars_types

    linear_step_vars_ops, linear_step_vars_ssa, linear_step_vars_types = (
        handle_var_operand_field(program_state, ctx, op.linear_step_vars)
    )
    arg_types += linear_step_vars_types

    private_vars_ops, private_vars_ssa, private_vars_types = handle_var_operand_field(
        program_state, ctx, op.private_vars
    )
    arg_types += private_vars_types

    reduction_vars_ops, reduction_vars_ssa, reduction_vars_types = (
        handle_var_operand_field(program_state, ctx, op.reduction_vars)
    )
    arg_types += reduction_vars_types

    schedule_chunk_ops, schedule_chunk_ssa, schedule_chunk_types = (
        handle_opt_operand_field(program_state, ctx, op.schedule_chunk)
    )
    arg_types += schedule_chunk_types

    new_block = Block(arg_types=arg_types)

    for fir_arg, std_arg in zip(op.body.blocks[0].args, new_block.args):
        ctx[fir_arg] = std_arg

    region_body_ops = []
    top_level_ops = []
    for single_op in op.body.blocks[0].ops:
        region_body_ops = statements.translate_stmt(program_state, ctx, single_op)
        if len(region_body_ops) > 1:
            # The loopnest op will pull down the lower, upper and step constants into this region, however they need to sit above simd as otherwise verification will fail
            # therefore extract out all but the last operation (the loop nest)
            top_level_ops = region_body_ops[0:-1]
            region_body_ops = [region_body_ops[-1]]

    assert len(region_body_ops) == 1
    assert isa(region_body_ops[0], omp.LoopNestOp) or isa(
        region_body_ops[0], omp.SIMDOp
    )

    new_block.add_ops(region_body_ops)

    new_props = {}
    for key, value in op.properties.items():
        if key != "operandSegmentSizes":
            new_props[key] = value

    omp_wsloop = omp.WsLoopOp.build(
        operands=[
            allocate_vars_ssa,
            allocator_vars_ssa,
            linear_vars_ssa,
            linear_step_vars_ssa,
            private_vars_ssa,
            reduction_vars_ssa,
            schedule_chunk_ssa,
        ],
        regions=[Region([new_block])],
        properties=new_props,
    )

    return (
        allocate_vars_ops
        + allocator_vars_ops
        + linear_vars_ops
        + linear_step_vars_ops
        + private_vars_ops
        + reduction_vars_ops
        + schedule_chunk_ops
        + top_level_ops
        + [omp_wsloop]
    )


def translate_omp_parallel(
    program_state: ProgramState, ctx: SSAValueCtx, op: omp.ParallelOp
):
    arg_types = []

    allocate_vars_ops, allocate_vars_ssa, allocate_vars_types = (
        handle_var_operand_field(program_state, ctx, op.allocate_vars)
    )
    arg_types += allocate_vars_types

    allocators_vars_ops, allocators_vars_ssa, allocators_vars_types = (
        handle_var_operand_field(program_state, ctx, op.allocators_vars)
    )
    arg_types += allocators_vars_types

    if_expr_ops, if_expr_ssa, if_expr_types = handle_opt_operand_field(
        program_state, ctx, op.if_expr
    )
    arg_types += if_expr_types

    num_threads_ops, num_threads_ssa, num_threads_types = handle_opt_operand_field(
        program_state, ctx, op.num_threads
    )
    arg_types += num_threads_types

    private_vars_ops, private_vars_ssa, private_vars_types = handle_var_operand_field(
        program_state, ctx, op.private_vars
    )
    arg_types += private_vars_types

    reduction_vars_ops, reduction_vars_ssa, reduction_vars_types = (
        handle_var_operand_field(program_state, ctx, op.reduction_vars)
    )
    arg_types += reduction_vars_types

    new_block = Block(arg_types=arg_types)

    for fir_arg, std_arg in zip(op.region.blocks[0].args, new_block.args):
        ctx[fir_arg] = std_arg

    region_body_ops = []
    for single_op in op.region.blocks[0].ops:
        region_body_ops += statements.translate_stmt(program_state, ctx, single_op)

    new_block.add_ops(region_body_ops)

    new_props = {}
    for key, value in op.properties.items():
        if key != "operandSegmentSizes":
            new_props[key] = value

    omp_parallel = omp.ParallelOp.build(
        operands=[
            allocate_vars_ssa,
            allocators_vars_ssa,
            if_expr_ssa,
            num_threads_ssa,
            private_vars_ssa,
            reduction_vars_ssa,
        ],
        regions=[Region([new_block])],
        properties=new_props,
    )

    return (
        allocate_vars_ops
        + allocators_vars_ops
        + if_expr_ops
        + num_threads_ops
        + private_vars_ops
        + reduction_vars_ops
        + [omp_parallel]
    )


def translate_omp_loopnest(
    program_state: ProgramState, ctx: SSAValueCtx, op: omp.LoopNestOp
):
    arg_types = []

    lb_ops = expressions.translate_expr(program_state, ctx, op.lowerBound[0])
    ub_ops = expressions.translate_expr(program_state, ctx, op.upperBound[0])
    step_ops = expressions.translate_expr(program_state, ctx, op.step[0])

    # A loopnest has a single argument type of i32, which is the loop iteration counter
    new_block = Block(arg_types=[builtin.i32])

    for fir_arg, std_arg in zip(op.body.blocks[0].args, new_block.args):
        ctx[fir_arg] = std_arg

    region_body_ops = []
    for single_op in op.body.blocks[0].ops:
        region_body_ops += statements.translate_stmt(program_state, ctx, single_op)

    new_block.add_ops(region_body_ops)

    new_props = {}
    for key, value in op.properties.items():
        if key != "operandSegmentSizes":
            new_props[key] = value

    loopnest_op = omp.LoopNestOp.build(
        operands=[lb_ops, ub_ops, step_ops],
        regions=[Region([new_block])],
        properties=new_props,
    )

    return lb_ops + ub_ops + step_ops + [loopnest_op]


def translate_omp_team(program_state: ProgramState, ctx: SSAValueCtx, op: omp.TeamsOp):
    arg_ssa = []
    arg_ops = []

    if op.num_teams_lower is not None:
        ops = expressions.translate_expr(program_state, ctx, op.num_teams_lower)
        arg_ops += ops
        arg_ssa.append([ops[-1].results[0]])
    else:
        arg_ssa.append([])

    if op.num_teams_upper is not None:
        ops = expressions.translate_expr(program_state, ctx, op.num_teams_upper)
        arg_ops += ops
        arg_ssa.append([ops[-1].results[0]])
    else:
        arg_ssa.append([])

    arg_ssa += [[], [], [], [], []]

    new_block = Block()

    region_body_ops = []
    for single_op in op.body.blocks[0].ops:
        region_body_ops += statements.translate_stmt(program_state, ctx, single_op)

    new_block.add_ops(region_body_ops)

    new_props = {}
    for key, value in op.properties.items():
        if key != "operandSegmentSizes":
            new_props[key] = value

    teams_op = omp.TeamsOp.build(
        operands=arg_ssa, regions=[Region([new_block])], properties=new_props
    )
    return arg_ops + [teams_op]


def translate_omp_simd(program_state: ProgramState, ctx: SSAValueCtx, op: omp.SIMDOp):
    arg_types = []

    aligned_vars_ops, aligned_vars_ssa, aligned_var_types = handle_var_operand_field(
        program_state, ctx, op.aligned_vars
    )
    arg_types += aligned_var_types

    if_expr_ops, if_expr_ssa, if_expr_types = handle_opt_operand_field(
        program_state, ctx, op.if_expr
    )
    arg_types += if_expr_types

    linear_vars_ops, linear_vars_ssa, linear_vars_types = handle_var_operand_field(
        program_state, ctx, op.linear_vars
    )
    arg_types += linear_vars_types

    linear_step_vars_ops, linear_step_vars_ssa, linear_step_vars_types = (
        handle_var_operand_field(program_state, ctx, op.linear_step_vars)
    )
    arg_types += linear_step_vars_types

    nontemporal_vars_ops, nontemporal_vars_ssa, nontemporal_vars_types = (
        handle_opt_operand_field(program_state, ctx, op.nontemporal_vars)
    )
    arg_types += nontemporal_vars_types

    private_vars_ops, private_vars_ssa, private_vars_types = handle_opt_operand_field(
        program_state, ctx, op.private_vars
    )
    arg_types += private_vars_types

    reduction_vars_ops, reduction_vars_ssa, reduction_vars_types = (
        handle_opt_operand_field(program_state, ctx, op.reduction_vars)
    )
    arg_types += reduction_vars_types

    new_block = Block(arg_types=arg_types)

    for fir_arg, std_arg in zip(op.body.blocks[0].args, new_block.args):
        ctx[fir_arg] = std_arg

    region_body_ops = []
    top_level_ops = []
    for single_op in op.body.blocks[0].ops:
        region_body_ops = statements.translate_stmt(program_state, ctx, single_op)
        if len(region_body_ops) > 1:
            # The loopnest op will pull down the lower, upper and step constants into this region, however they need to sit above simd as otherwise verification will fail
            # therefore extract out all but the last operation (the loop nest)
            top_level_ops = region_body_ops[0:-1]
            region_body_ops = [region_body_ops[-1]]

    assert len(region_body_ops) == 1
    assert isa(region_body_ops[0], omp.LoopNestOp) or isa(
        region_body_ops[0], omp.WsLoopOp
    )

    new_block.add_ops(region_body_ops)

    new_props = {}
    for key, value in op.properties.items():
        if key != "operandSegmentSizes":
            new_props[key] = value

    simd_op = omp.SIMDOp.build(
        operands=[
            aligned_vars_ssa,
            if_expr_ssa,
            linear_vars_ssa,
            linear_step_vars_ssa,
            nontemporal_vars_ssa,
            private_vars_ssa,
            reduction_vars_ssa,
        ],
        regions=[Region([new_block])],
        properties=new_props,
    )

    return (
        aligned_vars_ops
        + if_expr_ops
        + linear_vars_ops
        + linear_step_vars_ops
        + nontemporal_vars_ops
        + private_vars_ops
        + reduction_vars_ops
        + top_level_ops
        + [simd_op]
    )


def translate_omp_target(
    program_state: ProgramState, ctx: SSAValueCtx, op: omp.TargetOp
):
    map_var_ops = []
    map_var_ssa = []
    arg_types = []
    for arg in op.map_vars:
        v_ops = expressions.translate_expr(program_state, ctx, arg)
        map_var_ops += v_ops
        map_var_ssa.append(map_var_ops[-1].results[0])
        arg_types.append(map_var_ops[-1].results[0].type)

    new_block = Block(arg_types=arg_types)

    for fir_arg, std_arg in zip(op.region.blocks[0].args, new_block.args):
        ctx[fir_arg] = std_arg

    region_body_ops = []
    for single_op in op.region.blocks[0].ops:
        region_body_ops += statements.translate_stmt(program_state, ctx, single_op)

    new_block.add_ops(region_body_ops)

    new_props = {}
    for key, value in op.properties.items():
        if key != "operandSegmentSizes":
            new_props[key] = value

    # For the moment we ignore a large number of operands passed to the target call
    # consider handling these in the future

    target_op = omp.TargetOp.build(
        operands=[[], [], [], [], [], [], [], [], [], map_var_ssa, [], []],
        regions=[Region([new_block])],
        properties=new_props,
    )

    return map_var_ops + [target_op]


def translate_omp_yield(program_state: ProgramState, ctx: SSAValueCtx, op: omp.YieldOp):
    ops_list = []
    ssa_list = []
    for operand in op.arguments:
        expr_ops = expressions.translate_expr(program_state, ctx, operand)
        ops_list += expr_ops
        ssa_list.append(ctx[operand])
    return [omp.YieldOp(*ssa_list)]

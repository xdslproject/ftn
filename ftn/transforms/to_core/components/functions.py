from xdsl.ir import SSAValue, BlockArgument
from xdsl.irdl import Operand
from xdsl.utils.hints import isa
from util.visitor import Visitor
from xdsl.context import Context
from xdsl.dialects.experimental import fir, hlfir
from xdsl.ir import Operation, SSAValue, OpResult, Attribute, Block, Region

from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter,
    op_type_rewrite_pattern,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
)
from xdsl.passes import ModulePass
from xdsl.dialects import builtin, func, llvm, arith, memref, scf, cf, linalg, omp, math
from ftn.dialects import ftn_relative_cf

from ftn.transforms.to_core.misc.fortran_code_description import (
    ProgramState,
    ArrayDescription,
    ArgIntent,
)
from ftn.transforms.to_core.misc.ssa_context import SSAValueCtx

from ftn.transforms.to_core.utils import clean_func_name

import ftn.transforms.to_core.components.intrinsics as ftn_intrinsics
import ftn.transforms.to_core.components.types as ftn_types
import ftn.transforms.to_core.expressions as expressions
import ftn.transforms.to_core.statements as statements


def translate_function(program_state: ProgramState, ctx: SSAValueCtx, fn: func.FuncOp):
    within_module = fn.sym_name.data.startswith("_QM")
    fn_identifier = clean_func_name(fn.sym_name.data)

    if within_module:
        module_name = fn_identifier.split("P")[0]
        fn_name = fn_identifier.split("P")[1]
    else:
        module_name = None
        fn_name = fn_identifier

    if fn_name in ftn_intrinsics.FortranIntrinsicsHandleExplicitly.keys():
        return None

    body = Region()
    if len(fn.body.blocks) > 0:
        # This is a function with a body, the input types come from the block as
        # we will manipulate these to pass constants if possible
        program_state.enterFunction(fn_name, fn_identifier, module_name)
        fn_in_arg_types = []
        for idx, arg in enumerate(fn.args):
            fir_type = arg.type
            if (
                program_state.function_definitions[fn_identifier].args[idx].is_scalar
                and program_state.function_definitions[fn_identifier].args[idx].intent
                == ArgIntent.IN
            ):
                # This is a scalar in, therefore it's just the constant type (don't encode as a memref)
                if isa(fir_type, fir.ReferenceType):
                    fn_in_arg_types.append(fir_type.type)
                else:
                    fn_in_arg_types.append(arg.type)
            else:
                converted_type = ftn_types.convert_fir_type_to_standard(fir_type)
                if (
                    isa(converted_type, memref.MemRefType)
                    and program_state.function_definitions[fn_identifier]
                    .args[idx]
                    .is_allocatable
                ):
                    converted_type = memref.MemRefType(converted_type, shape=[])

                fn_in_arg_types.append(converted_type)

        for idx, block in enumerate(fn.body.blocks):
            if idx == 0:
                # If this is the first block, then it is the function arguments
                new_block = Block(arg_types=fn_in_arg_types)
            else:
                # Otherwise the arg types are the same as the blocks
                new_block = Block(arg_types=block.args)

            for fir_arg, std_arg in zip(block.args, new_block.args):
                ctx[fir_arg] = std_arg

            ops_list = []
            for op in block.ops:
                ops_list += statements.translate_stmt(program_state, ctx, op)

            new_block.add_ops(ops_list)
            body.add_block(new_block)
        program_state.leaveFunction()
    else:
        # This is the definition of an external function, need to resolve input types
        fn_in_arg_types = []
        for t in fn.function_type.inputs.data:
            fn_in_arg_types.append(ftn_types.convert_fir_type_to_standard_if_needed(t))

    # Perform some conversion on return types to standard
    return_types = []
    for rt in fn.function_type.outputs.data:
        if not isa(rt, builtin.NoneType):
            # Ignore none types, these are simply omitted
            return_types.append(ftn_types.convert_fir_type_to_standard_if_needed(rt))

    fn_identifier = fn.sym_name.data

    new_fn_type = builtin.FunctionType.from_lists(fn_in_arg_types, return_types)

    new_func = func.FuncOp(
        fn_identifier,
        new_fn_type,
        body,
        fn.sym_visibility,
        arg_attrs=fn.arg_attrs,
        res_attrs=fn.res_attrs,
    )
    return new_func


def handle_call_argument(
    program_state: ProgramState,
    ctx: SSAValueCtx,
    fn_name: str,
    arg: Operand,
    arg_index: int,
):
    if isa(arg.owner, hlfir.AssociateOp):
        # This is a scalar that we are passing by value
        if arg.owner.uniq_name is not None:
            assert arg.owner.uniq_name.data == "adapt.valuebyref"
        else:
            # This might be added as an attribute instead, allow this
            assert "adapt.valuebyref" in arg.owner.attributes
        arg_defn = program_state.function_definitions[fn_name].args[arg_index]
        # For now we just work with scalars here, could pass arrays by literal too
        assert arg_defn.is_scalar
        if arg_defn.is_scalar and arg_defn.intent == ArgIntent.IN:
            # This is a scalar with intent in, therefore just pass the constant
            ops_list = expressions.translate_expr(program_state, ctx, arg.owner.source)
            ctx[arg] = ctx[arg.owner.source]
            return ops_list, False
        else:
            # Otherwise we need to pack the constant into a memref and pass this
            assert isa(arg_defn.arg_type, fir.ReferenceType)
            ops_list = expressions.translate_expr(program_state, ctx, arg.owner.source)
            memref_alloca_op = memref.AllocaOp.get(
                ftn_types.convert_fir_type_to_standard(arg_defn.arg_type.type), shape=[]
            )

            storage_op = memref.StoreOp.get(
                ctx[arg.owner.source], memref_alloca_op.results[0], []
            )
            ctx[arg] = memref_alloca_op.results[0]
            return ops_list + [memref_alloca_op, storage_op], True
    else:
        # Here passing a variable (array or scalar variable). This is a little confusing, as we
        # allow the translate_expr to handle it, but if the function accepts an integer due to
        # scalar and intent(in), then we need to load the memref.
        ops_list = expressions.translate_expr(program_state, ctx, arg)
        if not program_state.function_definitions[fn_name].is_definition_only:
            arg_defn = program_state.function_definitions[fn_name].args[arg_index]
            if (
                arg_defn.is_scalar
                and arg_defn.intent == ArgIntent.IN
                and isa(ctx[arg].type, memref.MemRefType)
            ):
                # The function will accept a constant, but we are currently passing a memref
                # therefore need to load the value and pass this

                load_op = memref.LoadOp.get(ctx[arg], [])

                # arg is already in our ctx from above, so remove it and add in the load as
                # we want to reference that instead
                del ctx[arg]
                ctx[arg] = load_op.results[0]
                ops_list += [load_op]
            elif (
                not arg_defn.is_scalar
                and not arg_defn.is_allocatable
                and isa(ctx[arg].type, memref.MemRefType)
                and isa(ctx[arg].type.element_type, memref.MemRefType)
            ):
                load_op = memref.LoadOp.get(ctx[arg], [])
                del ctx[arg]
                ctx[arg] = load_op.results[0]
                ops_list += [load_op]
        elif fn_name == "_FortranAProgramStart" and arg_index == 3:
            # This is a hack, Flang currently generates incorrect typing for passing memory to the program initialisation
            # routine. As func.call verifies this we can not get away with it, so must extract the llvm pointer
            # from the memref and pass this directly
            assert isa(ctx[arg].type, memref.MemRefType)
            extract_ptr_as_idx_op = memref.ExtractAlignedPointerAsIndexOp.get(ctx[arg])
            i64_idx_op = arith.IndexCastOp(
                extract_ptr_as_idx_op.results[0], builtin.i64
            )
            ptr_op = llvm.IntToPtrOp(i64_idx_op.results[0])
            ops_list += [extract_ptr_as_idx_op, i64_idx_op, ptr_op]
            del ctx[arg]
            ctx[arg] = ptr_op.results[0]
        return ops_list, False


def translate_call(program_state: ProgramState, ctx: SSAValueCtx, op: fir.CallOp):
    if len(op.results) > 0 and ctx.contains(op.results[0]):
        return []

    fn_name = clean_func_name(op.callee.string_value())

    if fn_name in ftn_intrinsics.FortranIntrinsicsHandleExplicitly.keys():
        return ftn_intrinsics.FortranIntrinsicsHandleExplicitly[fn_name](
            program_state, ctx, op
        )

    # Create a new context scope, as we might overwrite some SSAs with packing
    call_ctx = SSAValueCtx(ctx)

    arg_ops = []
    are_temps_allocated = False
    for idx, arg in enumerate(op.args):
        # This is more complex, as constants are passed directly or packed in a temporary
        specific_arg_ops, temporary_allocated = handle_call_argument(
            program_state, call_ctx, fn_name, arg, idx
        )
        if temporary_allocated:
            are_temps_allocated = True
        arg_ops += specific_arg_ops

    arg_ssa = []
    for arg in op.args:
        arg_ssa.append(call_ctx[arg])

    return_types = []
    return_ssas = []
    for ret in op.results:
        # Ignore none types, these are just omitted
        if not isa(ret.type, builtin.NoneType):
            return_types.append(
                ftn_types.convert_fir_type_to_standard_if_needed(ret.type)
            )
            return_ssas.append(ret)

    call_op = func.CallOp(op.callee, arg_ssa, return_types)

    if (
        program_state.function_definitions[fn_name].is_definition_only
        and not are_temps_allocated
    ):
        for idx, ret_ssa in enumerate(return_ssas):
            ctx[ret_ssa] = call_op.results[idx]
        return arg_ops + [call_op]
    else:
        alloc_scope_return = memref.AllocaScopeReturnOp.build(
            operands=[call_op.results]
        )
        alloca_scope_op = memref.AllocaScopeOp.build(
            regions=[Region([Block(arg_ops + [call_op, alloc_scope_return])])],
            result_types=[return_types],
        )

        for idx, ret_ssa in enumerate(return_ssas):
            ctx[ret_ssa] = alloca_scope_op.results[idx]

        return [alloca_scope_op]

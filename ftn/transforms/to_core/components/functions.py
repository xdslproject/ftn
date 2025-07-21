from xdsl.irdl import Operand
from xdsl.utils.hints import isa
from xdsl.dialects.experimental import fir, hlfir
from xdsl.ir import Block, Region
from xdsl.dialects import builtin, func, llvm, arith, memref

from ftn.transforms.to_core.misc.fortran_code_description import (
    ProgramState,
    ArrayDescription,
    ArgIntent,
)
from ftn.transforms.to_core.misc.ssa_context import SSAValueCtx

from ftn.transforms.to_core.utils import (
    clean_func_name,
    create_index_constant,
    generate_extract_ptr_from_memref,
)

import ftn.transforms.to_core.components.intrinsics as ftn_intrinsics
import ftn.transforms.to_core.components.ftn_types as ftn_types
import ftn.transforms.to_core.components.memory as ftn_memory
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
    ptr_unpack_args = []
    if len(fn.body.blocks) > 0:
        # This is a function with a body, the input types come from the block as
        # we will manipulate these to pass constants if possible
        program_state.enterFunction(fn_name, fn_identifier, module_name)
        fn_in_arg_types = []
        for idx, arg in enumerate(fn.args):
            fir_type = arg.type

            if ftn_types.does_type_represent_ftn_pointer(fir_type):
                # If we are passing a Fortran pointer then we need to handle this differently, actually pass
                # the LLVM pointer of this and reconstruct, to access the same underlying memref
                converted_type = llvm.LLVMPointerType.opaque()
                ptr_unpack_args.append((idx, fir_type))
            else:
                converted_type = ftn_types.convert_fir_type_to_standard(fir_type)
                if (
                    isa(converted_type, builtin.MemRefType)
                    and program_state.function_definitions[fn_identifier]
                    .args[idx]
                    .is_allocatable
                ):
                    converted_type = builtin.MemRefType(converted_type, shape=[])

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
            for ptr_idx, fir_type in ptr_unpack_args:
                # This has been passed as a pointer, therefore pack into a memref
                seq_type = ftn_types.get_type_from_chain(fir_type, fir.SequenceType)
                base_type = ftn_types.convert_fir_type_to_standard(seq_type)

                build_ops, build_ssa = ftn_memory.generate_memref_from_llvm_ptr(
                    new_block.args[ptr_idx], [], base_type
                )

                del ctx[block.args[ptr_idx]]
                ctx[block.args[ptr_idx]] = build_ssa

                ops_list += build_ops

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

        # We need to pack the constant into a memref and pass this
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
            if ftn_types.does_type_represent_ftn_pointer(arg_defn.arg_type):
                # If we are passing a pointer then grab the underlying LLVM pointer from the memref
                extract_ops, extract_ssa = generate_extract_ptr_from_memref(ctx[arg])
                ops_list += extract_ops
                del ctx[arg]
                ctx[arg] = extract_ssa
            else:
                if (
                    not arg_defn.is_scalar
                    and not arg_defn.is_allocatable
                    and isa(ctx[arg].type, builtin.MemRefType)
                    and isa(ctx[arg].type.element_type, builtin.MemRefType)
                ):
                    load_op = memref.LoadOp.get(ctx[arg], [])
                    del ctx[arg]
                    ctx[arg] = load_op.results[0]
                    ops_list += [load_op]

        else:
            # We have partial argument information, assumes all scalar and intent inout
            arg_defn = program_state.function_definitions[fn_name].args[arg_index]
            if (
                arg_defn.arg_type is not fir.ReferenceType
                and arg_defn.arg_type is not fir.SequenceType
            ):
                # The function argument is not a reference or sequence (array) so likely a scalar
                if isa(ctx[arg].type, builtin.MemRefType):
                    if isa(arg_defn.arg_type, fir.ReferenceType):
                        # It is expecting an LLVM pointer
                        extract_ops, extract_ssa = generate_extract_ptr_from_memref(
                            ctx[arg]
                        )
                        del ctx[arg]
                        ctx[arg] = extract_ssa
                        ops_list += extract_ops
                    else:
                        # We will extract the memref
                        load_idx_ssa = []
                        if len(ctx[arg].type.shape) != 0:
                            assert len(ctx[arg].type.shape) == 1
                            assert ctx[arg].type.shape.data[0].data == 1
                            accessor_op = create_index_constant(0)
                            ops_list.append(accessor_op)
                            load_idx_ssa.append(accessor_op.results[0])

                        # The passed argument is a memref, we therefore need to extract this
                        load_op = memref.LoadOp.get(ctx[arg], load_idx_ssa)
                        del ctx[arg]
                        ctx[arg] = load_op.results[0]
                        ops_list += [load_op]
        if fn_name == "_FortranAProgramStart" and arg_index == 3:
            # This is a hack, Flang currently generates incorrect typing for passing memory to the program initialisation
            # routine. As func.call verifies this we can not get away with it, so must extract the llvm pointer
            # from the memref and pass this directly
            extract_ops, extract_ssa = generate_extract_ptr_from_memref(ctx[arg])
            ops_list += extract_ops
            del ctx[arg]
            ctx[arg] = extract_ssa
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

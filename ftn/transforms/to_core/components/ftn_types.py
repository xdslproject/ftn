from typing import cast
from xdsl.dialects.experimental import fir
from xdsl.ir import Attribute, TypeAttribute
from xdsl.utils.hints import isa
from xdsl.dialects import builtin, llvm, arith, memref

from ftn.transforms.to_core.misc.fortran_code_description import ProgramState
from ftn.transforms.to_core.misc.ssa_context import SSAValueCtx

import ftn.transforms.to_core.expressions as expressions


def contains_type(type_chain, type_to_match):
    return get_type_from_chain(type_chain, type_to_match) != None


def get_type_from_chain(type_chain, type_to_match):
    if isa(type_chain, type_to_match):
        return type_chain
    if (
        isa(type_chain, fir.ReferenceType)
        or isa(type_chain, fir.BoxType)
        or isa(type_chain, fir.SequenceType)
        or isa(type_chain, fir.HeapType)
        or isa(type_chain, fir.PointerType)
        or isa(type_chain, fir.LLVMPointerType)
    ):
        return get_type_from_chain(type_chain.type, type_to_match)
    else:
        return None


def does_type_represent_ftn_pointer(type_chain):
    return contains_type(type_chain, fir.ReferenceType) and contains_type(
        type_chain, fir.PointerType
    )


def convert_fir_type_to_standard_if_needed(fir_type):
    if isa(fir_type, fir.ReferenceType):
        return llvm.LLVMPointerType.opaque()
    else:
        return convert_fir_type_to_standard(fir_type)


def convert_fir_type_to_standard(fir_type: Attribute, ref_as_mem_ref: bool = True) -> TypeAttribute:
    if isa(fir_type, fir.ReferenceType):
        if ref_as_mem_ref:
            base_t = convert_fir_type_to_standard(fir_type.type, ref_as_mem_ref)
            if isa(base_t, builtin.MemRefType):
                return base_t
            else:
                return builtin.MemRefType(
                    base_t, [], builtin.NoneAttr(), builtin.NoneAttr()
                )
        else:
            return llvm.LLVMPointerType.opaque()
    elif isa(fir_type, fir.BoxType):
        return convert_fir_type_to_standard(fir_type.type, ref_as_mem_ref)
    elif isa(fir_type, fir.SequenceType):
        base_t = convert_fir_type_to_standard(fir_type.type)
        dim_sizes: list[int] = []
        for shape_el in fir_type.shape:
            if isa(shape_el, builtin.IntegerAttr):
                dim_sizes.append(shape_el.value.data)
            else:
                dim_sizes.append(-1)
        # Reverse the sizes to go from Fortran to C allocation semantics
        dim_sizes.reverse()
        return builtin.MemRefType(
            base_t, dim_sizes, builtin.NoneAttr(), builtin.NoneAttr()
        )
    elif isa(fir_type, fir.LogicalType):
        return builtin.i1
    elif isa(fir_type, fir.BoxCharType):
        return llvm.LLVMStructType.from_type_list(
            [llvm.LLVMPointerType.opaque(), builtin.i64]
        )
    elif isa(fir_type, builtin.TupleType):
        new_types:list[TypeAttribute] = []
        for ty in fir_type.types:
            new_types.append(convert_fir_type_to_standard(ty, ref_as_mem_ref))
        return builtin.TupleType(new_types)
    elif isa(fir_type, builtin.NoneType):
        return builtin.i32
    else:
        return cast(TypeAttribute, fir_type)


def translate_convert(program_state: ProgramState, ctx: SSAValueCtx, op: fir.ConvertOp):
    if ctx.contains(op.results[0]):
        return []
    value_ops = expressions.translate_expr(program_state, ctx, op.value)
    in_type = op.value.type
    out_type = op.results[0].type
    new_conv = None
    if (
        isa(in_type, builtin.Float32Type)
        and isa(out_type, builtin.Float64Type)
        or isa(in_type, builtin.Float16Type)
        and isa(out_type, builtin.Float64Type)
        or isa(in_type, builtin.Float16Type)
        and isa(out_type, builtin.Float32Type)
    ):
        new_conv = arith.ExtFOp(ctx[op.value], out_type)
        ctx[op.results[0]] = new_conv.results[0]

    if (
        isa(in_type, builtin.Float64Type)
        and isa(out_type, builtin.Float32Type)
        or isa(in_type, builtin.Float64Type)
        and isa(out_type, builtin.Float16Type)
        or isa(in_type, builtin.Float32Type)
        and isa(out_type, builtin.Float16Type)
    ):
        new_conv = arith.TruncFOp(ctx[op.value], out_type)
        ctx[op.results[0]] = new_conv.results[0]

    if (isa(in_type, builtin.IndexType) and isa(out_type, builtin.IntegerType)) or (
        isa(in_type, builtin.IntegerType) and isa(out_type, builtin.IndexType)
    ):
        if isa(ctx[op.value].type, builtin.IndexType) or isa(
            out_type, builtin.IndexType
        ):
            new_conv = arith.IndexCastOp(ctx[op.value], out_type)
            ctx[op.results[0]] = new_conv.results[0]
        else:
            ctx[op.results[0]] = ctx[op.value]
            return []

    if isa(in_type, builtin.IntegerType) and isa(out_type, builtin.AnyFloat):
        new_conv = arith.SIToFPOp(ctx[op.value], out_type)
        ctx[op.results[0]] = new_conv.results[0]

    if isa(in_type, builtin.AnyFloat) and isa(out_type, builtin.IntegerType):
        new_conv = arith.FPToSIOp(ctx[op.value], out_type)
        ctx[op.results[0]] = new_conv.results[0]

    if new_conv is not None:
        new_conv = [new_conv]

    if isa(in_type, builtin.IntegerType) and isa(out_type, builtin.IntegerType):
        in_width = in_type.width.data
        out_width = out_type.width.data
        if in_width < out_width:
            new_conv = arith.ExtUIOp(ctx[op.value], out_type)
            ctx[op.results[0]] = new_conv.results[0]
            new_conv = [new_conv]
        elif in_width > out_width:
            new_conv = arith.TruncIOp(ctx[op.value], out_type)
            ctx[op.results[0]] = new_conv.results[0]
            new_conv = [new_conv]
        else:
            # They are the same, ignore and use the input directly
            new_conv = []
            ctx[op.results[0]] = ctx[op.value]

    if (
        isa(in_type, fir.ReferenceType)
        and isa(out_type, fir.ReferenceType)
        or (isa(in_type, fir.BoxType) and isa(out_type, fir.BoxType))
    ):
        if isa(out_type.type, builtin.IntegerType) and out_type.type.width.data == 8:
            # Converting to an LLVM pointer
            # The element type is an LLVM array, we hard code this to be size 1 here which is OK as it just needs to
            # grab the starting pointer to this
            get_element_ptr = llvm.GEPOp(
                ctx[op.value],
                [0, 0],
                result_type=llvm.LLVMPointerType.opaque(),
                pointee_type=llvm.LLVMArrayType.from_size_and_type(
                    1, builtin.IntegerType(8)
                ),
            )
            ctx[op.results[0]] = get_element_ptr.results[0]
            new_conv = [get_element_ptr]
        elif isa(out_type.type, fir.SequenceType) and isa(
            in_type.type, fir.SequenceType
        ):
            # Converting between two shapes in the array
            assert out_type.type.type == in_type.type.type
            shape_size = []
            for s in out_type.type.shape.data:
                if isa(s, fir.DeferredAttr):
                    shape_size.append(-1)
                else:
                    shape_size.append(s.value.data)
            # Reverse shape_size to get it from Fortran allocation to C/MLIR allocation
            shape_size.reverse()
            target_type = builtin.MemRefType(
                convert_fir_type_to_standard(out_type.type.type), shape_size
            )
            cast_op = memref.CastOp.get(ctx[op.value], target_type)

            ctx[op.results[0]] = cast_op.results[0]
            new_conv = [cast_op]
        elif isa(out_type.type, fir.BoxType) and isa(in_type.type, fir.BoxType):
            new_conv = []
            ctx[op.results[0]] = ctx[op.value]

    if isa(in_type, fir.HeapType) and isa(out_type, fir.ReferenceType):
        # When passing arrays to subroutines will box_addr to a heaptype, then convert
        # to a reference type. Both these contain arrays, therefore set this to
        # short circuit to the type of the arg (effectively this is a pass through)
        assert isa(in_type.type, fir.SequenceType)
        assert isa(out_type.type, fir.SequenceType)
        # Assert that what we will forward to is in-fact a memref type
        assert isa(ctx[op.value].type, builtin.MemRefType)
        ctx[op.results[0]] = ctx[op.value]
        new_conv = []

    if isa(out_type, fir.BoxType) and isa(in_type, fir.BoxType):
        new_conv = []
        ctx[op.results[0]] = ctx[op.value]

    if isa(in_type, fir.PointerType) and isa(out_type, fir.ReferenceType):
        assert isa(in_type.type, fir.SequenceType)
        assert isa(out_type.type, fir.SequenceType)
        ctx[op.results[0]] = ctx[op.value]
        new_conv = []

    if isa(in_type, fir.LogicalType):
        assert out_type == builtin.i1
        ctx[op.results[0]] = ctx[op.value]
        new_conv = []

    if in_type == builtin.i1:
        assert isa(out_type, fir.LogicalType)
        ctx[op.results[0]] = ctx[op.value]
        new_conv = []

    if new_conv is None:
        raise Exception(f"Could not convert between `{in_type}' and `{out_type}`")
    return value_ops + new_conv


def translate_string_literal(
    program_state: ProgramState, ctx: SSAValueCtx, op: fir.StringLitOp
):
    str_type = llvm.LLVMArrayType.from_size_and_type(
        op.size.value.data, builtin.IntegerType(8)
    )
    str_global_op = llvm.GlobalOp(
        str_type,
        "temporary_identifier",
        "internal",
        0,
        True,
        value=op.value,
        unnamed_addr=0,
    )
    return [str_global_op]

from functools import reduce
from xdsl.dialects.experimental import fir, hlfir
from xdsl.ir import BlockArgument
from xdsl.irdl import Operand
from xdsl.utils.hints import isa
from xdsl.ir import OpResult, Block
from xdsl.dialects import builtin, llvm, arith, memref, scf

from ftn.transforms.to_core.misc.fortran_code_description import (
    ProgramState,
    ArrayDescription,
)
from ftn.transforms.to_core.misc.ssa_context import SSAValueCtx

from ftn.transforms.to_core.utils import (
    create_index_constant,
    generate_dereference_memref,
    check_if_has_type,
    remove_tuple_type_from_memref,
    generate_extract_ptr_from_memref,
)

import ftn.transforms.to_core.components.ftn_types as ftn_types
import ftn.transforms.to_core.expressions as expressions
import ftn.transforms.to_core.statements as statements


def generate_memref_from_llvm_ptr(llvm_ptr_in_ssa, dim_sizes, target_type):
    # Builds a memref from an LLVM pointer. This is required if we are working with
    # global arrays, as they are llvm.array, and the pointer is grabbed from that and
    # then the memref constructed
    ptr_type = llvm.LLVMPointerType.opaque()

    offsets = [1]
    if len(dim_sizes) > 1:
        for d in dim_sizes[:-1]:
            offsets.append(d * offsets[-1])

    offsets.reverse()

    # Reverse the indicies as Fortran and C/MLIR are opposite in terms of
    # the order of the contiguous dimension (F is least, whereas C/MLIR is highest)
    dim_sizes = dim_sizes.copy()
    dim_sizes.reverse()

    array_type = llvm.LLVMArrayType.from_size_and_type(
        builtin.IntAttr(len(dim_sizes)), builtin.i64
    )
    if len(dim_sizes) > 0:
        struct_type = llvm.LLVMStructType.from_type_list(
            [ptr_type, ptr_type, builtin.i64, array_type, array_type]
        )
    else:
        struct_type = llvm.LLVMStructType.from_type_list(
            [ptr_type, ptr_type, builtin.i64]
        )

    undef_memref_struct_op = llvm.UndefOp.create(result_types=[struct_type])
    insert_alloc_ptr_op = llvm.InsertValueOp.create(
        properties={"position": builtin.DenseArrayBase.from_list(builtin.i64, [0])},
        operands=[undef_memref_struct_op.results[0], llvm_ptr_in_ssa],
        result_types=[struct_type],
    )
    insert_aligned_ptr_op = llvm.InsertValueOp.create(
        properties={"position": builtin.DenseArrayBase.from_list(builtin.i64, [1])},
        operands=[insert_alloc_ptr_op.results[0], llvm_ptr_in_ssa],
        result_types=[struct_type],
    )

    offset_op = arith.ConstantOp.from_int_and_width(0, 64)
    insert_offset_op = llvm.InsertValueOp.create(
        properties={"position": builtin.DenseArrayBase.from_list(builtin.i64, [2])},
        operands=[insert_aligned_ptr_op.results[0], offset_op.results[0]],
        result_types=[struct_type],
    )

    ops_to_add = [
        undef_memref_struct_op,
        insert_alloc_ptr_op,
        insert_aligned_ptr_op,
        offset_op,
        insert_offset_op,
    ]

    memref_create_ssa = ops_to_add[-1].results[0]

    if len(dim_sizes) > 0:
        for idx, dim in enumerate(dim_sizes):
            size_op = arith.ConstantOp.from_int_and_width(dim, 64)
            insert_size_op = llvm.InsertValueOp.create(
                properties={
                    "position": builtin.DenseArrayBase.from_list(builtin.i64, [3, idx])
                },
                operands=[ops_to_add[-1].results[0], size_op.results[0]],
                result_types=[struct_type],
            )

            # One for dimension stride
            stride_op = arith.ConstantOp.from_int_and_width(offsets[idx], 64)
            insert_stride_op = llvm.InsertValueOp.create(
                properties={
                    "position": builtin.DenseArrayBase.from_list(builtin.i64, [4, idx])
                },
                operands=[insert_size_op.results[0], stride_op.results[0]],
                result_types=[struct_type],
            )
            memref_create_ssa = insert_stride_op.results[0]

            ops_to_add += [size_op, insert_size_op, stride_op, insert_stride_op]

    target_memref_type = memref.MemRefType(
        ftn_types.convert_fir_type_to_standard(target_type), dim_sizes
    )

    unrealised_conv_cast_op = builtin.UnrealizedConversionCastOp.create(
        operands=[memref_create_ssa], result_types=[target_memref_type]
    )
    ops_to_add.append(unrealised_conv_cast_op)
    return ops_to_add, unrealised_conv_cast_op.results[0]


def define_scalar_var(
    program_state: ProgramState, ctx: SSAValueCtx, op: hlfir.DeclareOp
):
    if ctx.contains(op.results[0]):
        assert ctx.contains(op.results[1])
        return []
    if isa(op.memref, OpResult):
        allocation_op = op.memref.owner
        if isa(allocation_op, fir.AllocaOp):
            assert isa(allocation_op.results[0].type, fir.ReferenceType)
            assert allocation_op.results[0].type.type == allocation_op.in_type
            memref_alloca_op = memref.AllocaOp.get(
                ftn_types.convert_fir_type_to_standard(allocation_op.in_type), shape=[]
            )
            ctx[op.results[0]] = memref_alloca_op.results[0]
            ctx[op.results[1]] = memref_alloca_op.results[0]
            return [memref_alloca_op]
        elif isa(allocation_op, fir.AddressOfOp):
            expr_ops = expressions.translate_expr(program_state, ctx, op.memref)
            ctx[op.results[0]] = ctx[allocation_op.results[0]]
            ctx[op.results[1]] = ctx[allocation_op.results[0]]
            return expr_ops
        elif isa(allocation_op, fir.UnboxcharOp):
            if isa(allocation_op.boxchar, BlockArgument):
                # If it is a block argument then set to be that
                ctx[op.results[0]] = ctx[allocation_op.boxchar]
                ctx[op.results[1]] = ctx[allocation_op.boxchar]
                return []
            else:
                # Otherwise process the argument and set to be the result
                expr_ops = expressions.translate_expr(
                    program_state, ctx, allocation_op.boxchar
                )
                ctx[op.results[0]] = ctx[allocation_op.results[0]]
                ctx[op.results[1]] = ctx[allocation_op.results[0]]
                return expr_ops
        else:
            raise Exception(
                f"Could not define scalar from allocation operation `{allocation_op.name}'"
            )
    elif isa(op.memref, BlockArgument):
        ctx[op.results[0]] = ctx[op.memref]
        ctx[op.results[1]] = ctx[op.memref]
        return []


def dims_has_static_size(dims):
    for dim in dims:
        if not isa(dim, int):
            return False
    return True


def gather_static_shape_dims_from_shape(shape_op: fir.ShapeOp):
    # fir.Shape is for default, 1 indexed arrays
    dim_sizes = []
    dim_starts = []
    assert shape_op.extents is not None
    for extent in shape_op.extents:
        if isa(extent.owner, arith.ConstantOp):
            assert isa(extent.owner.result.type, builtin.IndexType)
            dim_sizes.append(extent.owner.value.value.data)
        else:
            dim_sizes.append(None)
    dim_starts = [1] * len(dim_sizes)
    return dim_sizes, dim_starts, dim_sizes


def gather_static_shape_dims_from_shapeshift(shape_op: fir.ShapeShiftOp):
    # fir.ShapeShift is for arrays indexed on a value other than 1
    dim_sizes = []
    dim_starts = []
    dim_ends = []
    assert shape_op.pairs is not None
    # Now iterate in pairs of low, high e.g. (low, high), (low, high) etc
    paired_vals = list(zip(shape_op.pairs[::2], shape_op.pairs[1::2]))
    for low_arg, high_arg in paired_vals:
        if isa(low_arg.owner, arith.ConstantOp):
            assert isa(low_arg.owner.result.type, builtin.IndexType)
            dim_starts.append(low_arg.owner.value.value.data)
        else:
            dim_starts.append(low_arg)

        if isa(high_arg.owner, arith.ConstantOp):
            assert isa(high_arg.owner.result.type, builtin.IndexType)
            dim_sizes.append(high_arg.owner.value.value.data)
        else:
            dim_sizes.append(high_arg)

        if dim_starts[-1] is int and dim_sizes[-1] is int:
            dim_ends.append((dim_sizes[-1] + dim_starts[-1]) - 1)
        else:
            dim_ends.append(None)

    return dim_sizes, dim_starts, dim_ends


def define_stack_array_var(
    program_state: ProgramState,
    ctx: SSAValueCtx,
    op: hlfir.DeclareOp,
    dim_sizes: list,
    dim_starts: list,
    dim_ends: list,
):
    if ctx.contains(op.results[0]):
        assert ctx.contains(op.results[1])
        return []

    # It might be one of tree things - allocated from a global, an alloca in fir
    # for stack local variable, or an array function argument that is statically sized
    assert (
        isa(op.memref.owner, fir.AddressOfOp)
        or isa(op.memref.owner, fir.AllocaOp)
        or (op.memref.owner, Block)
    )

    if isa(op.memref.owner, Block):
        # It is an array fn argument that is statically sized
        fir_array_type = op.memref.type
    else:
        # Global or fn stack local variab;e
        fir_array_type = op.memref.owner.results[0].type

    assert isa(fir_array_type, fir.ReferenceType)
    fir_array_type = fir_array_type.type

    assert isa(fir_array_type, fir.SequenceType)
    # Ensure collected dimensions and the addressof type dimensions are consistent
    for type_size, dim_size in zip(fir_array_type.shape, dim_sizes):
        assert isa(type_size.type, builtin.IntegerType)
        assert type_size.value.data == dim_size

    array_name = op.uniq_name.data
    # Store information about the array - the size, and lower and upper bounds as we need this when accessing elements
    program_state.getCurrentFnState().array_info[array_name] = ArrayDescription(
        array_name, dim_sizes, dim_starts, dim_ends
    )

    if isa(op.memref.owner, Block):
        # This is a statically sized array passed to the function
        # so just point to this
        ctx[op.results[0]] = ctx[op.memref]
        ctx[op.results[1]] = ctx[op.memref]
        return []
    elif isa(op.memref.owner, fir.AddressOfOp):
        memref_lookup = memref.GetGlobalOp(
            op.memref.owner.symbol.string_value(),
            ftn_types.convert_fir_type_to_standard(fir_array_type),
        )
        ctx[op.results[0]] = memref_lookup.results[0]
        ctx[op.results[1]] = memref_lookup.results[0]
        return [memref_lookup]
    elif isa(op.memref.owner, fir.AllocaOp):
        # Issue an allocation on the stack
        # Reverse the indicies as Fortran and C/MLIR are opposite in terms of
        # the order of the contiguous dimension (F is least, whereas C/MLIR is highest)
        dim_sizes_reversed = dim_sizes.copy()
        dim_sizes_reversed.reverse()
        memref_alloca_op = memref.AllocaOp.get(
            ftn_types.convert_fir_type_to_standard(fir_array_type.type),
            shape=dim_sizes_reversed,
        )
        ctx[op.results[0]] = memref_alloca_op.results[0]
        ctx[op.results[1]] = memref_alloca_op.results[0]
        return [memref_alloca_op]
    else:
        assert False


def translate_boxdims(program_state: ProgramState, ctx: SSAValueCtx, op: fir.BoxDimsOp):
    # Grabs dimensions from a box for a given dimension which is provided. This returns the triplet of lower bound, extent, and stride for 'dim' dimension
    # we currently hardcode lower bound and stride and in the future should extend to provide these properly
    if ctx.contains(op.results[0]):
        return []

    val_load_ops = expressions.translate_expr(program_state, ctx, op.val)
    var_ssa = ctx[op.val]

    idx_ops = expressions.translate_expr(program_state, ctx, op.dim)
    idx_ssa = idx_ops[-1].results[0]

    get_rank = memref.RankOp.from_memref(var_ssa)
    lb_op = create_index_constant(1)

    # sub_idx=arith.SubiOp(idx_ssa, lb_op)
    sub_zero_idx = arith.SubiOp(get_rank.results[0], lb_op)
    sub_rank = arith.SubiOp(sub_zero_idx, idx_ssa)

    ub_op = memref.DimOp.from_source_and_index(var_ssa, sub_rank)
    extent_op = create_index_constant(1)

    ctx[op.results[0]] = lb_op.results[0]
    ctx[op.results[1]] = ub_op.results[0]
    ctx[op.results[2]] = extent_op.results[0]
    return (
        val_load_ops
        + idx_ops
        + [lb_op, get_rank, sub_zero_idx, sub_rank, extent_op, ub_op]
    )


def translate_boxoffset(
    program_state: ProgramState, ctx: SSAValueCtx, op: fir.BoxOffsetOp
):
    # Grabs the underlying LLVM pointer from the box which is provided. This acts as a way to retrieve a pointer
    # to the memory whilst still maintaining abstraction around FIR memory. This is needed when interfacing with
    # external APIs, such as OpenMP and OpenACC

    if ctx.contains(op.results[0]):
        return []

    val_load_ops = expressions.translate_expr(program_state, ctx, op.val)
    var_ssa = ctx[op.val]

    extract_ops, extract_ssa = generate_extract_ptr_from_memref(var_ssa)
    ctx[op.results[0]] = extract_ssa

    return val_load_ops + extract_ops


def translate_alloca(program_state: ProgramState, ctx: SSAValueCtx, op: fir.AllocaOp):
    if ctx.contains(op.results[0]):
        return []

    # If any use of the result is the declareop, then ignore this as
    # we will handle it elsewhere - otherwise this is used internally
    for use in op.results[0].uses:
        if isa(use.operation, hlfir.DeclareOp):
            return []

    memref_alloca_op = memref.AllocaOp.get(
        ftn_types.convert_fir_type_to_standard(op.in_type), shape=[]
    )
    ctx[op.results[0]] = memref_alloca_op.results[0]
    return [memref_alloca_op]


def translate_declare(
    program_state: ProgramState, ctx: SSAValueCtx, op: hlfir.DeclareOp
):
    # If already seen then simply ignore
    if ctx.contains(op.results[0]):
        for res in op.results:
            assert ctx.contains(res)
        return []

    if (
        op.fortran_attrs is not None
        and len(op.fortran_attrs.data) == 1
        and op.fortran_attrs.data[0] == fir.FortranVariableFlags.HOSTASSOC
    ):
        return []

    if (
        isa(op.memref.owner, fir.UnboxcharOp)
        and isa(op.results[0].type, fir.BoxCharType)
        and isa(op.results[1].type, fir.ReferenceType)
    ):
        assert len(op.typeparams) == 1
        assert isa(op.typeparams[0].owner, fir.UnboxcharOp)

        memref_handle_ops = expressions.translate_expr(program_state, ctx, op.memref)

        ctx[op.results[0]] = ctx[op.memref.owner.boxchar]
        ctx[op.results[1]] = ctx[op.memref]
        return memref_handle_ops

    if (
        isa(op.results[0].type, fir.ReferenceType)
        and isa(op.results[0].type.type, fir.BoxType)
        and not isa(op.memref, BlockArgument)
    ):
        # This is allocating either an allocatable array, or a pointer but
        # how we do these is the same in the resulting IR. The last part ensures
        # is not a block argument (which would already be allocated and passed into a block e.g. a function)

        if isa(op.results[0].type.type.type, fir.HeapType):
            # This is an allocatable array
            assert fir.FortranVariableFlags.ALLOCATABLE in op.fortran_attrs.flags
        elif isa(op.results[0].type.type.type, fir.PointerType):
            # This is a pointer
            assert fir.FortranVariableFlags.POINTER in op.fortran_attrs.flags
            program_state.getCurrentFnState().pointers.append(op.uniq_name.data)

        assert isa(op.results[0].type.type.type.type, fir.SequenceType)
        num_dims = len(op.results[0].type.type.type.type.shape)

        if isa(op.memref.owner, fir.AddressOfOp):
            memref_lookup = memref.GetGlobalOp(
                op.memref.owner.symbol.string_value(),
                builtin.MemRefType(
                    ftn_types.convert_fir_type_to_standard(
                        op.memref.owner.results[0].type
                    ),
                    shape=[],
                ),
            )
            ctx[op.results[0]] = memref_lookup.results[0]
            ctx[op.results[1]] = memref_lookup.results[0]
            return [memref_lookup]
        else:
            alloc_memref_container = memref.AllocaOp.get(
                builtin.MemRefType(
                    op.results[0].type.type.type.type.type, shape=num_dims * [-1]
                ),
                shape=[],
            )

            ctx[op.results[0]] = alloc_memref_container.results[0]
            ctx[op.results[1]] = alloc_memref_container.results[0]
            return [alloc_memref_container]

    if op.results[0].uses.get_length() == 0 and op.results[1].uses.get_length() == 0:
        # Some declare ops are never actually used in the code, Flang seems to generate
        # the declare for global arrays regardless in some functions and therefore
        # we ignore them if the declare doesn't have any uses
        return []

    # Passing an allocatable with annonymous dimension e.g. memref<?,i32> it doesn't know the size

    if (
        op.shape is None
        and not isa(op.results[0].type, fir.BoxType)
        and (
            isa(op.results[0].type, fir.ReferenceType)
            and not isa(op.results[0].type.type, fir.BoxType)
        )
    ):
        # Ensure it doesn't have a shape, and there isn't a boxtype (which caries the shape)
        # this means that it is a scalar (TODO: could also check the inner type)
        return define_scalar_var(program_state, ctx, op)
    else:
        if op.shape is not None:
            # There is a shape we can use in determining the size
            if isa(op.shape.owner, fir.ShapeOp):
                dim_sizes, dim_starts, dim_ends = gather_static_shape_dims_from_shape(
                    op.shape.owner
                )
            elif isa(op.shape.owner, fir.ShapeShiftOp):
                dim_sizes, dim_starts, dim_ends = (
                    gather_static_shape_dims_from_shapeshift(op.shape.owner)
                )
            else:
                assert False

            static_size = dims_has_static_size(dim_sizes)
            if static_size:
                return define_stack_array_var(
                    program_state, ctx, op, dim_sizes, dim_starts, dim_ends
                )
            elif isa(op.memref, BlockArgument) and isa(
                op.results[1].type, fir.ReferenceType
            ):
                # This is an array passed into a function
                shape_expr_list = []
                for ds in dim_starts:
                    if not isa(ds, int) and ds is not None:
                        shape_expr_list += expressions.translate_expr(
                            program_state, ctx, ds
                        )
                ctx[op.results[0]] = ctx[op.memref]
                ctx[op.results[1]] = ctx[op.memref]
                array_name = op.uniq_name.data
                # Store information about the array - the size, and lower and upper bounds as we need this when accessing elements
                program_state.getCurrentFnState().array_info[array_name] = (
                    ArrayDescription(array_name, dim_sizes, dim_starts, dim_ends)
                )
                return shape_expr_list
            else:
                assert False
        else:
            # There is no shape, we need to grab this from the memref using operations
            assert isa(op.memref, BlockArgument)
            assert isa(op.results[0].type, fir.BoxType) or isa(
                op.results[0].type, fir.ReferenceType
            )
            sequence_type = ftn_types.get_type_from_chain(
                op.results[0].type, fir.SequenceType
            )
            assert isa(sequence_type, fir.SequenceType)
            num_dims = len(sequence_type.shape)

            one_op = create_index_constant(1)
            ops_list = [one_op]
            size_ssas = []
            end_ssas = []

            dim_memref_ssa = ctx[op.memref]
            if isa(dim_memref_ssa.type.element_type, builtin.MemRefType):
                # If its an allocatable we need to load the memref that it contains
                load_op = memref.LoadOp.get(dim_memref_ssa, [])
                dim_memref_ssa = load_op.results[0]
                ops_list.append(load_op)
            for dim in range(num_dims):
                dim_idx_op = create_index_constant(dim)
                get_dim_op = memref.DimOp.from_source_and_index(
                    dim_memref_ssa, dim_idx_op
                )
                add_arith_op = arith.AddiOp(get_dim_op.results[0], one_op.results[0])
                ops_list += [dim_idx_op, get_dim_op, add_arith_op]
                size_ssas.append(get_dim_op.results[0])
                end_ssas.append(add_arith_op.results[0])
            array_name = op.uniq_name.data
            ctx[op.results[0]] = ctx[op.memref]
            ctx[op.results[1]] = ctx[op.memref]
            program_state.getCurrentFnState().array_info[array_name] = ArrayDescription(
                array_name, size_ssas, [1] * len(size_ssas), end_ssas
            )
            return ops_list
    assert False


def translate_reassoc(
    program_state: ProgramState,
    ctx: SSAValueCtx,
    op: fir.NoReassocOp | hlfir.NoReassocOp,
):
    if ctx.contains(op.results[0]):
        return []
    if isa(op, fir.NoReassocOp):
        expr_list = expressions.translate_expr(program_state, ctx, op.val)
    elif isa(op, hlfir.NoReassocOp):
        expr_list = expressions.translate_expr(program_state, ctx, op.var)

    ctx[op.results[0]] = ctx[op.var]
    return expr_list


def translate_absent(program_state: ProgramState, ctx: SSAValueCtx, op: fir.AbsentOp):
    if ctx.contains(op.results[0]):
        return []

    null_ptr = llvm.ZeroOp()
    ctx[op.results[0]] = null_ptr.results[0]
    return [null_ptr]


def translate_zerobits(
    program_state: ProgramState, ctx: SSAValueCtx, op: fir.ZeroBitsOp
):
    # This often appears in global regions for array declaration, if so then we need to
    # handle differently as can not use a memref in LLVM global operations
    result_type = op.results[0].type
    if isa(result_type, fir.SequenceType):
        base_type = result_type.type
        array_sizes = []
        for d in result_type.shape.data:
            assert isa(d, builtin.IntegerAttr)
            array_sizes.append(d.value.data)
    else:
        base_type = ftn_types.convert_fir_type_to_standard(result_type)
        if check_if_has_type(builtin.TupleType, base_type):
            # Flang will generate a fir.ref of tuple for a zero bits fed into the program entry point
            # this can not be represented in core MLIR dialects, so we strip out the tuple component
            base_type = remove_tuple_type_from_memref(base_type)
        array_sizes = [1]

    if program_state.isInGlobal():
        tgt_type = ftn_types.convert_fir_type_to_standard(result_type)
        if isa(tgt_type, builtin.MemRefType):
            # If this is a memref type, then use memref.global here, we don't need to update the context
            # as it is used directly as a global region (it isn't embedded in anything)
            assert isa(op.parent.parent.parent, fir.GlobalOp)

            converted_result_type = ftn_types.convert_fir_type_to_standard(result_type)
            if isa(result_type, fir.HeapType) or isa(result_type, fir.PointerType):
                # This is an allocatable array if it's heaptype or a pointer if its
                # a pointer type, (otherwise this is just an arraytype)
                # therefore wrap this in a memref type due to allocatable nature
                converted_result_type = builtin.MemRefType(converted_result_type, [])
            global_memref = memref.GlobalOp.get(
                op.parent.parent.parent.sym_name,
                converted_result_type,
                builtin.UnitAttr(),
            )

            return [global_memref]
        else:
            # Otherwise is not a memref, so need to allocate as LLVM compatible operation
            total_size = reduce((lambda x, y: x * y), array_sizes)
            llvm_array_type = llvm.LLVMArrayType.from_size_and_type(
                total_size, base_type
            )
            zero_op = llvm.ZeroOp.build(result_types=[llvm_array_type])

            ctx[op.results[0]] = zero_op.results[0]
            return [zero_op]
    else:
        memref_alloc = memref.AllocOp.get(base_type, shape=array_sizes)
        ctx[op.results[0]] = memref_alloc.results[0]
        return [memref_alloc]


def translate_freemem(program_state: ProgramState, ctx: SSAValueCtx, op: fir.FreememOp):
    assert isa(op.heapref.owner, fir.BoxAddrOp)
    assert isa(op.heapref.owner.val.owner, fir.LoadOp)
    memref_ssa = op.heapref.owner.val.owner.memref

    ops_list = []
    assert isa(ctx[memref_ssa].type, builtin.MemRefType)
    if isa(ctx[memref_ssa].type.element_type, builtin.MemRefType):
        load_op, load_ssa = generate_dereference_memref(ctx[memref_ssa])
        ops_list.append(load_op)
    else:
        load_ssa = ctx[memref_ssa]

    ops_list.append(memref.DeallocOp.get(load_ssa))
    return ops_list


def translate_address_of(
    program_state: ProgramState, ctx: SSAValueCtx, op: fir.AddressOfOp
):
    if ctx.contains(op.results[0]):
        return []

    assert isa(op.results[0].type, fir.ReferenceType)
    global_lookup = llvm.AddressOfOp(op.symbol, llvm.LLVMPointerType.opaque())

    ctx[op.results[0]] = global_lookup.results[0]
    return [global_lookup]


def translate_emboxchar(program_state, ctx, op: fir.EmboxcharOp):
    if ctx.contains(op.results[0]):
        return []

    struct_type = llvm.LLVMStructType.from_type_list(
        [llvm.LLVMPointerType.opaque(), builtin.i64]
    )

    char_ptr_ops_list = expressions.translate_expr(program_state, ctx, op.memref)

    undef_memref_struct_op = llvm.UndefOp.create(result_types=[struct_type])

    insert_char_ptr_op = llvm.InsertValueOp.create(
        properties={"position": builtin.DenseArrayBase.from_list(builtin.i64, [0])},
        operands=[undef_memref_struct_op.results[0], ctx[op.memref]],
        result_types=[struct_type],
    )

    size_ops_list = expressions.translate_expr(program_state, ctx, op.len)
    if isa(ctx[op.len].type, builtin.IndexType):
        conv_i64 = arith.IndexCastOp(ctx[op.len], builtin.i64)
        size_ops_list += [conv_i64]
        del ctx[op.len]
        ctx[op.len] = conv_i64.results[0]

    insert_len_op = llvm.InsertValueOp.create(
        properties={"position": builtin.DenseArrayBase.from_list(builtin.i64, [1])},
        operands=[insert_char_ptr_op.results[0], ctx[op.len]],
        result_types=[struct_type],
    )

    ctx[op.results[0]] = insert_len_op.results[0]

    return (
        char_ptr_ops_list
        + size_ops_list
        + [undef_memref_struct_op, insert_char_ptr_op, insert_len_op]
    )


def translate_unboxchar(program_state, ctx, op: fir.UnboxcharOp):
    if ctx.contains(op.results[0]):
        assert ctx.contains(op.results[1])
        return []

    boxchar_ops_list = expressions.translate_expr(program_state, ctx, op.boxchar)

    extract_char_ptr = llvm.ExtractValueOp(
        builtin.DenseArrayBase.from_list(builtin.i64, [0]),
        ctx[op.boxchar],
        llvm.LLVMPointerType.opaque(),
    )
    extract_char_len = llvm.ExtractValueOp(
        builtin.DenseArrayBase.from_list(builtin.i64, [1]), ctx[op.boxchar], builtin.i64
    )

    ctx[op.results[0]] = extract_char_ptr.results[0]
    ctx[op.results[1]] = extract_char_len.results[0]

    return boxchar_ops_list + [extract_char_ptr, extract_char_len]


def translate_elemental(program_state, ctx, op: hlfir.ElementalOp):
    if ctx.contains(op.results[0]):
        return []

    assert isa(op.shape.owner, fir.ShapeOp)
    sizes = list(op.shape.owner.extents)

    size_ops = []
    for size_specific in sizes:
        size_ops += expressions.translate_expr(program_state, ctx, size_specific)

    # We need to reverse the size to get the ordering correct, from Fortran to C style
    sizes.reverse()

    memref_shape = [
        -1 if isa(f, fir.DeferredAttr) else f.value.data
        for f in op.results[0].type.shape
    ]

    dynamic_sizes = []

    for idx, s in enumerate(memref_shape):
        if s == -1:
            dynamic_sizes.append(ctx[sizes[idx]])

    memref_alloca_op = memref.AllocOp(
        dynamic_sizes, [], ftn_types.convert_fir_type_to_standard(op.results[0].type)
    )

    loop_ops = generate_scf_loop_for_elemental_dimension(
        program_state,
        ctx,
        sizes,
        [],
        op.regions[0].blocks[0].args,
        op.regions[0].blocks[0].ops,
        memref_alloca_op,
        0,
        len(sizes),
    )
    ctx[op.results[0]] = memref_alloca_op.results[0]

    return size_ops + [memref_alloca_op] + loop_ops


def generate_scf_loop_for_elemental_dimension(
    program_state,
    ctx,
    sizes,
    loop_index_args,
    elemental_block_args,
    elemental_ops,
    memref_alloca_op,
    loop_idx,
    total_loops,
):
    new_block = Block(arg_types=[builtin.IndexType()])
    # Need to add one to the index as hlfir elemental loops start
    # from index 1, whereas scf starts from zero
    one_const = create_index_constant(1)
    add_one_to_idx = arith.AddiOp(new_block.args[0], one_const)
    ctx[elemental_block_args[loop_idx]] = add_one_to_idx.results[0]

    loop_index_args.append(new_block.args[0])

    loop_body_ops = [one_const, add_one_to_idx]

    if loop_idx == len(sizes) - 1:
        # Insert in the actual compute loop
        for loop_op in elemental_ops:
            if isa(loop_op, hlfir.YieldElementOp):
                expr_lhs_ops = expressions.translate_expr(
                    program_state, ctx, loop_op.element_value
                )
                # The elemental load orders input load indicies in C rather than F
                # order, these are already reversed in the loading of them elsewhere
                # in this transformtion, therefore need to do another reverse to
                # convert back into C form. We do this here and also for the
                # statements translated in the else block too
                invert_load_indexes(expr_lhs_ops, total_loops)
                storage_op = memref.StoreOp.get(
                    ctx[loop_op.element_value],
                    memref_alloca_op.results[0],
                    loop_index_args,
                )
                loop_body_ops += expr_lhs_ops + [storage_op]
            else:
                specific_loop_body_ops = statements.translate_stmt(
                    program_state, ctx, loop_op
                )
                invert_load_indexes(specific_loop_body_ops, total_loops)
                loop_body_ops += specific_loop_body_ops
    else:
        loop_body_ops += generate_scf_loop_for_elemental_dimension(
            program_state,
            ctx,
            sizes,
            loop_index_args,
            elemental_block_args,
            elemental_ops,
            memref_alloca_op,
            loop_idx + 1,
            total_loops,
        )

    new_block.add_ops(loop_body_ops + [scf.YieldOp()])

    lower_bound = create_index_constant(0)
    step = create_index_constant(1)
    scf_for_loop = scf.ForOp(lower_bound, ctx[sizes[loop_idx]], step, [], new_block)

    return [lower_bound, step, scf_for_loop]


def invert_load_indexes(ops, num_dims):
    # Invert load indexes, this is needed if they are in the wrong order, the
    # elemental has these in C order so it is correct but elsewhere our transform
    # assumes F order and-so reverses them. Therefore this will re-reverse them
    idx_to_invert = []
    for idx, op in enumerate(ops):
        if isa(op, memref.LoadOp):
            if len(op.indices) == num_dims:
                idx_to_invert.append(idx)

    for idx in idx_to_invert:
        index_ssa = list(ops[idx].indices)
        index_ssa.reverse()
        new_load_op = memref.LoadOp.get(ops[idx].memref, index_ssa)
        ops[idx].results[0].replace_by(new_load_op.results[0])
        ops[idx] = new_load_op


def translate_destroy(program_state, ctx, op: hlfir.DestroyOp):
    expr_lhs_ops = expressions.translate_expr(program_state, ctx, op.expr)
    dealloc_op = memref.DeallocOp.get(ctx[op.expr])

    return expr_lhs_ops + [dealloc_op]

from functools import reduce
from xdsl.dialects.experimental import fir, hlfir
from xdsl.ir import BlockArgument
from xdsl.irdl import Operand
from xdsl.utils.hints import isa
from xdsl.ir import OpResult, Block
from xdsl.dialects import builtin, llvm, arith, memref

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
)

import ftn.transforms.to_core.components.ftn_types as ftn_types
import ftn.transforms.to_core.expressions as expressions


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
            expr_ops = expressions.translate_expr(
                program_state, ctx, allocation_op.boxchar
            )
            ctx[op.results[0]] = ctx[allocation_op.results[0]]
            ctx[op.results[1]] = ctx[allocation_op.results[0]]
            return expr_ops
        else:
            assert False
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

    lb_op = create_index_constant(1)
    ub_op = memref.DimOp.from_source_and_index(var_ssa, idx_ssa)
    extent_op = create_index_constant(1)

    ctx[op.results[0]] = lb_op.results[0]
    ctx[op.results[1]] = ub_op.results[0]
    ctx[op.results[2]] = extent_op.results[0]
    return val_load_ops + idx_ops + [lb_op, extent_op, ub_op]


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

    assert isa(var_ssa.type, builtin.MemRefType)
    extract_ptr_as_idx_op = memref.ExtractAlignedPointerAsIndexOp.get(var_ssa)
    i64_idx_op = arith.IndexCastOp(extract_ptr_as_idx_op.results[0], builtin.i64)
    ptr_op = llvm.IntToPtrOp(i64_idx_op.results[0])
    ctx[op.results[0]] = ptr_op.results[0]

    return val_load_ops + [extract_ptr_as_idx_op, i64_idx_op, ptr_op]


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
        return []

    if (
        isa(op.results[0].type, fir.ReferenceType)
        and isa(op.results[0].type.type, fir.BoxType)
        and not isa(op.memref, BlockArgument)
    ):
        # This is an allocatable array, we will handle this on the allocation, last part ensures
        # is not a block argument (which would already be allocated and passed into a block e.g. a function)
        assert isa(op.results[0].type.type.type, fir.HeapType)
        assert isa(op.results[0].type.type.type.type, fir.SequenceType)
        num_dims = len(op.results[0].type.type.type.type.shape)
        alloc_memref_container = memref.AllocaOp.get(
            builtin.MemRefType(
                op.results[0].type.type.type.type.type, shape=num_dims * [-1]
            ),
            shape=[],
        )
        ctx[op.results[0]] = alloc_memref_container.results[0]
        ctx[op.results[1]] = alloc_memref_container.results[0]
        return [alloc_memref_container]

    if len(op.results[0].uses) == 0 and len(op.results[1].uses) == 0:
        # Some declare ops are never actually used in the code, Flang seems to generate
        # the declare for global arrays regardless in some functions and therefore
        # we ignore them if the declare doesn't have any uses
        return []

    # Passing an allocatable with annonymous dimension e.g. memref<?,i32> it doesn't know the size

    if op.shape is None and not isa(op.results[0].type, fir.BoxType):
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
            assert isa(op.results[0].type, fir.BoxType) and isa(
                op.results[0].type.type, fir.SequenceType
            )
            num_dims = len(op.results[0].type.type.shape)

            one_op = create_index_constant(1)
            ops_list = [one_op]
            size_ssas = []
            end_ssas = []
            for dim in range(num_dims):
                dim_idx_op = create_index_constant(dim)
                get_dim_op = memref.DimOp.from_source_and_index(
                    ctx[op.memref], dim_idx_op
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
            global_memref = memref.GlobalOp.get(
                op.parent.parent.parent.sym_name,
                ftn_types.convert_fir_type_to_standard(result_type),
                None,
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

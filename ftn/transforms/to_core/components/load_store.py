from xdsl.ir import BlockArgument
from xdsl.utils.hints import isa
from xdsl.dialects.experimental import fir, hlfir
from xdsl.ir import Operation, OpResult
from xdsl.dialects import builtin, llvm, arith, memref, linalg

from ftn.transforms.to_core.misc.fortran_code_description import (
    ProgramState,
    ArrayDescription,
)
from ftn.transforms.to_core.misc.ssa_context import SSAValueCtx

from ftn.transforms.to_core.utils import (
    create_index_constant,
    generate_dereference_memref,
)

import ftn.transforms.to_core.components.ftn_types as ftn_types
import ftn.transforms.to_core.expressions as expressions


def remove_array_size_convert(op):
    # This is for array size, they are often converted from
    # integer to index, so work back to find the original integer
    if isa(op, fir.ConvertOp):
        assert isa(op.results[0].type, builtin.IndexType)
        assert isa(op.value.type, builtin.IntegerType)
        return op.value.owner
    return op


def array_access_components(
    program_state: ProgramState, ctx: SSAValueCtx, op: hlfir.DesignateOp
):
    # This will generate the required operations and SSA for index accesses to an array, whether
    # this is storage or loading in the wider context. It will offset depending upon the logical
    # start index to the physical start index of 0
    if isa(op.memref.owner, hlfir.DeclareOp):
        array_name = op.memref.owner.uniq_name.data
    elif isa(op.memref.owner, fir.LoadOp):
        assert isa(op.memref.owner.memref.owner, hlfir.DeclareOp)
        array_name = op.memref.owner.memref.owner.uniq_name.data
    else:
        assert False

    is_pointer = array_name in program_state.getCurrentFnState().pointers

    ops_list = []
    indexes_ssa = []
    for idx, index in enumerate(op.indices):
        ops = expressions.translate_expr(program_state, ctx, index)
        ops_list += ops
        if not isa(ctx[index].type, builtin.IndexType):
            assert isa(ctx[index].type, builtin.IntegerType)
            convert_op = arith.IndexCastOp(ctx[index], builtin.IndexType())
            ops_list.append(convert_op)
            index_ssa = convert_op.results[0]
        else:
            index_ssa = ctx[index]

        if is_pointer:
            # If this is a pointer then we assume starting from one as is the default in Fortran
            # this should be extended in the future, would need to use dynamic sizing variables
            # to keep track of this
            offset_const = arith.ConstantOp.create(
                properties={"value": builtin.IntegerAttr.from_index_int_value(1)},
                result_types=[builtin.IndexType()],
            )
            subtract_op = arith.SubiOp(index_ssa, offset_const)
            ops_list += [offset_const, subtract_op]
            indexes_ssa.append(subtract_op.results[0])
        else:
            dim_start = (
                program_state.getCurrentFnState().array_info[array_name].dim_starts[idx]
            )
            if isa(dim_start, int):
                assert dim_start >= 0
                if dim_start > 0:
                    # If zero start then we are good, otherwise need to zero index this
                    offset_const = arith.ConstantOp.create(
                        properties={
                            "value": builtin.IntegerAttr.from_index_int_value(dim_start)
                        },
                        result_types=[builtin.IndexType()],
                    )
                    subtract_op = arith.SubiOp(index_ssa, offset_const)
                    ops_list += [offset_const, subtract_op]
                    indexes_ssa.append(subtract_op.results[0])
                else:
                    indexes_ssa.append(index_ssa)
            elif isa(dim_start, OpResult):
                # This is not a constant literal in the code, therefore use the variable that drives this
                # which was generated previously, so just link to this
                assert ctx[dim_start] is not None
                subtract_op = arith.SubiOp(index_ssa, ctx[dim_start])
                ops_list.append(subtract_op)
                indexes_ssa.append(subtract_op.results[0])
            else:
                assert False
    return ops_list, indexes_ssa


def generate_var_dim_size_load(ctx: SSAValueCtx, op: fir.LoadOp):
    # Generates operations to load the size of a dimension from a variable
    # and ensure that it is typed as an index
    var_ssa = ctx[op.memref]
    ops_list = []
    if isa(var_ssa.type, builtin.MemRefType):
        # If this is a memref then we need to load it to
        # retrieve the index value
        load_op = memref.LoadOp.get(var_ssa, [])
        ops_list += [load_op]
        var_ssa = load_op.results[0]
    if not isa(var_ssa.type, builtin.IndexType):
        assert isa(var_ssa.type, builtin.IntegerType)
        convert_op = arith.IndexCastOp(var_ssa, builtin.IndexType())
        ops_list.append(convert_op)
        return convert_op.results[0], ops_list
    else:
        return var_ssa, ops_list


def handle_array_size_lu_bound(
    program_state: ProgramState, ctx: SSAValueCtx, bound_op: Operation, ssa
):
    # Handles extracting the literal size of a lower or upper array size bound
    # or the corresponding ssa and ops if it is driven by a variable
    bound_val = load_ssa = load_ops = None
    bound_op = remove_array_size_convert(bound_op)
    if isa(bound_op, arith.ConstantOp):
        bound_val = bound_op.value.value.data
    elif isa(bound_op, fir.LoadOp):
        load_ssa, load_ops = generate_var_dim_size_load(ctx, bound_op)
    else:
        # Otherwise this is a general expression, therefore translate that and
        # handle it generally
        load_ops = expressions.translate_expr(program_state, ctx, ssa)
        load_ssa = load_ops[-1].results[0]

    return bound_val, load_ssa, load_ops


def handle_array_to_array_assignment(source_ssa, target_ssa, ctx):
    # Handle array to array assignment, it might be actual arrays,
    # or constant e.g. a = (/ 1, 2, 3 /), or the result of
    # an intrinsic such as transpose or matmul
    expr_lhs_ops = []
    expr_rhs_ops = []

    assert isa(ctx[source_ssa].type, builtin.MemRefType)
    assert isa(ctx[target_ssa].type, builtin.MemRefType)

    # If these are allocatables or pointers then need dereferencing to get
    # the underlying memref
    if isa(ctx[target_ssa].type.element_type, builtin.MemRefType):
        load_op, lhs_load_ssa = generate_dereference_memref(ctx[target_ssa])
        expr_lhs_ops.append(load_op)
    else:
        lhs_load_ssa = ctx[target_ssa]

    if isa(ctx[source_ssa].type.element_type, builtin.MemRefType):
        load_op, rhs_load_ssa = generate_dereference_memref(ctx[source_ssa])
        expr_rhs_ops.append(load_op)
    else:
        rhs_load_ssa = ctx[source_ssa]

    # Check the two memrefs are compatible, this means element type is
    # the same and number of dimensions too
    memref_comparison = ftn_types.compare_memrefs(lhs_load_ssa.type, rhs_load_ssa.type)
    assert memref_comparison is not ftn_types.MemrefComparison.INCOMPATIBLE

    if memref_comparison == ftn_types.MemrefComparison.CONVERTABLE:
        # The two memrefs are largely compatible, but one has deferred size,
        # therefore convert LHS type (source) into RHS type (target)
        cast_op = memref.CastOp.get(
            lhs_load_ssa,
            rhs_load_ssa.type,
        )
        copy_op = memref.CopyOp(cast_op.results[0], rhs_load_ssa)
        return expr_lhs_ops + expr_rhs_ops + [cast_op, copy_op]
    elif memref_comparison == ftn_types.MemrefComparison.SAME:
        copy_op = memref.CopyOp(lhs_load_ssa, rhs_load_ssa)
        return expr_lhs_ops + expr_rhs_ops + [copy_op]
    else:
        assert False


def translate_assign(program_state: ProgramState, ctx: SSAValueCtx, op: hlfir.AssignOp):
    expr_lhs_ops = expressions.translate_expr(program_state, ctx, op.lhs)
    if isa(op.rhs.owner, hlfir.DeclareOp):
        expr_rhs_ops = expressions.translate_expr(program_state, ctx, op.rhs)
        # Scalar value or assign entire array to another
        if isa(ctx[op.rhs].type, builtin.MemRefType):
            if isa(op.rhs.owner.results[0].type, fir.BoxType) or (
                isa(op.rhs.owner.results[0].type, fir.ReferenceType)
                and (isa(op.rhs.owner.results[0].type.type, fir.BoxType))
                or isa(op.rhs.owner.results[0].type.type, fir.SequenceType)
            ):
                # Assigning an array to an array, this will result in a memref.copy operation
                if isa(ctx[op.lhs].type, builtin.MemRefType):
                    return (
                        expr_lhs_ops
                        + expr_rhs_ops
                        + handle_array_to_array_assignment(op.rhs, op.lhs, ctx)
                    )
                else:
                    # Assign a scalar across the array, requiring a broadcast
                    if isa(ctx[op.rhs].type.element_type, builtin.MemRefType):
                        load_op, lhs_load_ssa = generate_dereference_memref(ctx[op.rhs])
                        expr_lhs_ops.append(load_op)
                    else:
                        lhs_load_ssa = ctx[op.rhs]
                    memref_alloca_op = memref.AllocaOp.get(ctx[op.lhs].type, shape=[])
                    memref_store = memref.StoreOp.get(
                        ctx[op.lhs], memref_alloca_op.results[0], []
                    )

                    result_type = builtin.TensorType(
                        lhs_load_ssa.type.element_type, lhs_load_ssa.type.shape
                    )
                    linalg_bcast = linalg.BroadcastOp(
                        memref_alloca_op.results[0],
                        lhs_load_ssa,
                        builtin.DenseArrayBase.from_list(
                            builtin.i64, list(range(len(lhs_load_ssa.type.shape)))
                        ),
                    )
                    return (
                        expr_lhs_ops
                        + expr_rhs_ops
                        + [memref_alloca_op, memref_store, linalg_bcast]
                    )
            else:
                assert isa(op.rhs.owner.results[0].type, fir.ReferenceType)

                storage_op = memref.StoreOp.get(ctx[op.lhs], ctx[op.rhs], [])
                return expr_lhs_ops + expr_rhs_ops + [storage_op]
        elif isa(ctx[op.rhs].type, llvm.LLVMPointerType):
            storage_op = llvm.StoreOp(ctx[op.lhs], ctx[op.rhs])
            return expr_lhs_ops + expr_rhs_ops + [storage_op]
        else:
            assert False
    elif isa(op.rhs.owner, hlfir.DesignateOp):
        # Array value
        assert op.rhs.owner.indices is not None
        ops_list, indexes_ssa = array_access_components(
            program_state, ctx, op.rhs.owner
        )
        if isa(op.rhs.owner.memref.owner, hlfir.DeclareOp):
            memref_reference = op.rhs.owner.memref
        elif isa(op.rhs.owner.memref.owner, fir.LoadOp):
            memref_reference = op.rhs.owner.memref.owner.memref
        else:
            assert False

        assert isa(ctx[memref_reference].type, builtin.MemRefType)
        if isa(ctx[memref_reference].type.element_type, builtin.MemRefType):
            load_op, load_ssa = generate_dereference_memref(ctx[memref_reference])
            ops_list.append(load_op)
        else:
            load_ssa = ctx[memref_reference]
        # Reverse the indicies as Fortran and C/MLIR are opposite in terms of
        # the order of the contiguous dimension (F is least, whereas C/MLIR is highest)
        indexes_ssa_reversed = indexes_ssa.copy()
        indexes_ssa_reversed.reverse()
        storage_op = memref.StoreOp.get(ctx[op.lhs], load_ssa, indexes_ssa_reversed)
        ops_list.append(storage_op)
        return expr_lhs_ops + ops_list
    elif isa(op.lhs.owner, fir.LoadOp) and isa(op.rhs.owner, fir.LoadOp):
        # Pointer assigned to a pointer
        assert isa(op.lhs.owner.memref.owner, hlfir.DeclareOp) and isa(
            op.rhs.owner.memref.owner, hlfir.DeclareOp
        )
        assert (
            fir.FortranVariableFlags.POINTER
            in op.lhs.owner.memref.owner.fortran_attrs.flags
        )
        assert (
            fir.FortranVariableFlags.POINTER
            in op.rhs.owner.memref.owner.fortran_attrs.flags
        )
        expr_rhs_ops = expressions.translate_expr(program_state, ctx, op.rhs)
        return (
            expr_lhs_ops
            + expr_rhs_ops
            + handle_array_to_array_assignment(op.rhs, op.lhs, ctx)
        )
    elif isa(op.lhs.owner, hlfir.DeclareOp) and isa(op.rhs.owner, fir.LoadOp):
        # Assigning an array to a pointer
        expr_rhs_ops = expressions.translate_expr(program_state, ctx, op.rhs)
        return (
            expr_lhs_ops
            + expr_rhs_ops
            + handle_array_to_array_assignment(op.rhs, op.lhs, ctx)
        )
    else:
        assert False


def generate_allocatable_array_allocate(
    program_state: ProgramState, ctx: SSAValueCtx, op: fir.StoreOp
):
    # Allocates memory to an allocatable array
    assert isa(op.value.owner.memref.owner.results[0].type, fir.HeapType)
    assert isa(op.value.owner.memref.owner.results[0].type.type, fir.SequenceType)
    base_type = op.value.owner.memref.owner.results[0].type.type.type
    for dim_shape in op.value.owner.memref.owner.results[0].type.type.shape.data:
        assert isa(dim_shape, fir.DeferredAttr)

    assert len(op.value.owner.shape) == 1
    default_start_idx_mode = isa(op.value.owner.shape[0].owner, fir.ShapeOp)

    dim_sizes = []
    dim_starts = []
    dim_ends = []
    dim_ssas = []
    ops_list = []
    for shape in op.value.owner.memref.owner.shape:
        assert isa(shape.owner, arith.SelectOp)
        # Flang adds a guard to ensure that non-negative size is used, hence select
        # the actual provided size is the lhs
        size_op = remove_array_size_convert(shape.owner.lhs.owner)
        if isa(size_op, arith.ConstantOp):
            # Default sized
            dim_sizes.append(size_op.value.value.data)
            const_op = create_index_constant(size_op.value.value.data)
            ops_list.append(const_op)
            dim_ssas.append(const_op.results[0])
        elif isa(size_op, arith.AddiOp):
            # Start dim is offset, this does the substract, then add one
            # so we need to work back to calculate
            assert isa(size_op.rhs.owner, arith.ConstantOp)
            assert size_op.rhs.owner.value.value.data == 1
            assert isa(size_op.lhs.owner, arith.SubiOp)

            upper_bound_val, upper_load_ssa, upper_load_ops = (
                handle_array_size_lu_bound(
                    program_state,
                    ctx,
                    size_op.lhs.owner.lhs.owner,
                    size_op.lhs.owner.lhs,
                )
            )
            if upper_bound_val is not None:
                assert upper_load_ssa is None
                assert upper_load_ops is None
                dim_ends.append(upper_bound_val)
            else:
                assert upper_load_ssa is not None
                assert upper_load_ops is not None
                ops_list += upper_load_ops
                dim_ends.append(upper_load_ssa)
                ctx[upper_load_ssa] = upper_load_ssa

            lower_bound_val, lower_load_ssa, lower_load_ops = (
                handle_array_size_lu_bound(
                    program_state,
                    ctx,
                    size_op.lhs.owner.rhs.owner,
                    size_op.lhs.owner.rhs,
                )
            )
            if lower_bound_val is not None:
                assert lower_load_ssa is None
                assert lower_load_ops is None
                dim_starts.append(lower_bound_val)
            else:
                assert lower_load_ssa is not None
                assert lower_load_ops is not None
                ops_list += lower_load_ops
                dim_starts.append(lower_load_ssa)
                ctx[lower_load_ssa] = lower_load_ssa

            if lower_bound_val is not None and upper_bound_val is not None:
                # Constant based on literal dimension size, we know the value so put in directly
                dim_sizes.append((upper_bound_val - lower_bound_val) + 1)
                const_op = create_index_constant(
                    (upper_bound_val - lower_bound_val) + 1
                )
                ops_list.append(const_op)
                dim_ssas.append(const_op.results[0])
            else:
                if upper_load_ssa is None:
                    upper_const_op = create_index_constant(upper_bound_val)
                    ops_list.append(upper_const_op)
                    upper_load_ssa = upper_const_op.results[0]
                if lower_load_ssa is None:
                    lower_const_op = create_index_constant(lower_bound_val)
                    ops_list.append(lower_const_op)
                    lower_load_ssa = lower_const_op.results[0]

                one_const_op = create_index_constant(1)
                sub_op = arith.SubiOp(upper_load_ssa, lower_load_ssa)
                add_op = arith.AddiOp(sub_op, one_const_op)
                ops_list += [one_const_op, sub_op, add_op]
                dim_ssas.append(add_op.results[0])
                dim_sizes.append(add_op.results[0])

        elif isa(size_op, fir.LoadOp):
            # Sized based off a variable rather than constant, therefore need
            # to load this and convert to an index if it is an integer
            load_ssa, load_ops = generate_var_dim_size_load(ctx, size_op)
            ops_list += load_ops
            dim_ssas.append(load_ssa)
            dim_sizes.append(load_ssa)
        else:
            assert False

    if default_start_idx_mode:
        assert len(dim_starts) == 0
        dim_starts = [1] * len(dim_sizes)
        dim_ends = dim_sizes

    assert len(dim_sizes) == len(dim_starts) == len(dim_ends)

    # Now create memref, passing -1 as shape will make this deferred size
    # Reverse the indicies as Fortran and C/MLIR are opposite in terms of
    # the order of the contiguous dimension (F is least, whereas C/MLIR is highest)
    dim_ssa_reversed = dim_ssas.copy()
    dim_ssa_reversed.reverse()
    memref_allocation_op = memref_alloca_op = memref.AllocOp.get(
        base_type, shape=[-1] * len(dim_ssas), dynamic_sizes=dim_ssa_reversed
    )
    ops_list.append(memref_allocation_op)

    store_op = memref.StoreOp.get(
        memref_allocation_op.results[0], ctx[op.memref.owner.results[0]], []
    )
    ops_list.append(store_op)
    # ctx[op.memref.owner.results[0]]=memref_allocation_op.results[0]
    # ctx[op.memref.owner.results[1]]=memref_allocation_op.results[0]

    fn_name = program_state.getCurrentFnState().fn_name
    array_name = op.memref.owner.uniq_name.data

    # Store information about the array - the size, and lower and upper bounds as we need this when accessing elements
    program_state.getCurrentFnState().array_info[array_name] = ArrayDescription(
        array_name, dim_sizes, dim_starts, dim_ends
    )
    return ops_list


def handle_pointer_assignment(
    program_state: ProgramState, ctx: SSAValueCtx, source_op, target_op
):
    assert isa(target_op.owner, hlfir.DeclareOp) and isa(
        source_op.owner, hlfir.DeclareOp
    )
    assert fir.FortranVariableFlags.POINTER in target_op.owner.fortran_attrs.flags
    # The source can be either a target or a pointer
    assert (
        fir.FortranVariableFlags.TARGET in source_op.owner.fortran_attrs.flags
        or fir.FortranVariableFlags.POINTER in source_op.owner.fortran_attrs.flags
    )

    expr_ops = expressions.translate_expr(program_state, ctx, source_op)
    assert isa(ctx[source_op].type, builtin.MemRefType)

    if isa(ctx[source_op].type.element_type, builtin.MemRefType):
        # This memref itself is pointing to memory (an allocatable or pointer)
        # therefore we need to dereference this
        load_op, source_ssa = generate_dereference_memref(ctx[source_op])
        ops = [load_op]
    else:
        source_ssa = ctx[source_op]
        ops = []

    if any(i.data != -1 for i in ctx[source_op].type.shape.data):
        # The source type has explicit dimension sizes, by definition a pointer must be unknown
        # dimension sizes so we need to convert
        num_dims = len(ctx[source_op].type.shape.data)
        cast_op = memref.CastOp.get(
            source_ssa,
            builtin.MemRefType(source_ssa.type.element_type, shape=num_dims * [-1]),
        )
        source_ssa = cast_op.results[0]
        ops.append(cast_op)

    store_op = memref.StoreOp.get(source_ssa, ctx[target_op], [])
    return ops + [store_op]


def translate_store(program_state: ProgramState, ctx: SSAValueCtx, op: fir.StoreOp):
    # This is used for internal program components, such as loop indexes and storing
    # allocated memory to an allocatable
    if isa(op.value.owner, fir.EmboxOp):
        if isa(op.memref.owner, hlfir.DeclareOp):
            # We know the target (memref) is a declaration, so it's just now to understand
            # what the LHS (source) is to determine the action to undertake
            if isa(op.value.owner.memref.owner, fir.AllocmemOp):
                # The source is a chunk of memory, this is therefore allocating
                # into an allocatable array
                assert (
                    fir.FortranVariableFlags.ALLOCATABLE
                    in op.memref.owner.fortran_attrs.flags
                )
                return generate_allocatable_array_allocate(program_state, ctx, op)
            elif isa(op.value.owner.memref.owner, fir.ZeroBitsOp):
                pass
            elif isa(op.value.owner.memref.owner, hlfir.DeclareOp):
                # The source is a declaration too, this is assigning one declaration to
                # another, which is a variable to a pointer
                return handle_pointer_assignment(
                    program_state, ctx, op.value.owner.memref, op.memref
                )
            elif (
                isa(op.value.owner.memref.owner, fir.BoxAddrOp)
                and isa(op.value.owner.memref.owner.val.owner, fir.LoadOp)
                and isa(
                    op.value.owner.memref.owner.val.owner.memref.owner, hlfir.DeclareOp
                )
            ):
                # This is dereferencing a box and then loading the declaration, typically done when
                # we are assigning an allocatable to a pointer
                return handle_pointer_assignment(
                    program_state,
                    ctx,
                    op.value.owner.memref.owner.val.owner.memref,
                    op.memref,
                )
            else:
                assert False
    elif isa(op.value.owner, fir.ReboxOp):
        if isa(op.memref.owner, hlfir.DeclareOp):
            if isa(op.value.owner.box.owner, fir.LoadOp) and isa(
                op.value.owner.box.owner.memref.owner, hlfir.DeclareOp
            ):
                return handle_pointer_assignment(
                    program_state,
                    ctx,
                    op.value.owner.box.owner.memref,
                    op.memref,
                )
    elif isa(op.value.owner, hlfir.DeclareOp):
        # Flang generates i and j loop iterations in a strange way, we handle them here
        assert isa(op.memref.owner, fir.CoordinateOfOp)
        assert isa(op.memref.owner.ref.owner, fir.AllocaOp)
        assert isa(op.memref.owner.ref.owner.in_type, builtin.TupleType)
        assert len(op.memref.owner.ref.owner.in_type.types.data) > 0
        assert isa(op.memref.owner.ref.owner.in_type.types.data[0], fir.ReferenceType)
        base_type = op.memref.owner.ref.owner.in_type.types.data[0].type
        memref_allocation_op = memref_alloca_op = memref.AllocaOp.get(base_type)
        return [memref_allocation_op]
    else:
        expr_ops = expressions.translate_expr(program_state, ctx, op.value)

        if isa(op.memref.owner, hlfir.DeclareOp):
            storage_op = memref.StoreOp.get(ctx[op.value], ctx[op.memref], [])
        elif isa(op.memref.owner, fir.AllocaOp):
            assert ctx[op.memref] is not None
            storage_op = memref.StoreOp.get(ctx[op.value], ctx[op.memref], [])
        else:
            raise Exception(
                f"Could not translate store acting on lhs operand `{op.value.owner}'"
            )
        return expr_ops + [storage_op]


def translate_load(program_state: ProgramState, ctx: SSAValueCtx, op: fir.LoadOp):
    if ctx.contains(op.results[0]):
        return []

    # If this is a block argument, then it might be a scalar if it's in only. Therefore
    # check to see if this is a block argument and whether type is not memref, if so
    # just link directly. Otherwise it must be a memref
    if isa(ctx[op.memref], BlockArgument) and not isa(
        ctx[op.memref].type, builtin.MemRefType
    ):
        if isa(ctx[op.memref].type, llvm.LLVMPointerType):
            # If this is an LLVM pointer then load it
            load_op = llvm.LoadOp(
                ctx[op.memref],
                ftn_types.convert_fir_type_to_standard(op.results[0].type),
            )
            ctx[op.results[0]] = load_op.results[0]
            return [load_op]
        else:
            ctx[op.results[0]] = ctx[op.memref]
            return []
    elif isa(op.memref.owner, hlfir.DeclareOp):
        # Scalar value
        if isa(ctx[op.memref].type, builtin.MemRefType):
            # This is held in a memref, it is a variable in the user's code,
            # therefore load it up
            assert isa(op.memref.owner.results[0].type, fir.ReferenceType)

            load_op = memref.LoadOp.get(ctx[op.memref], [])
            ctx[op.results[0]] = load_op.results[0]
            return [load_op]
        elif isa(ctx[op.memref].type, llvm.LLVMPointerType):
            # This is referenced by an LLVM pointer, it is because it has been loaded by an addressof
            # operation, most likely because that loads in a global. Regardless, we issue and LLVM
            # load operation to load the value
            assert isa(op.memref.owner.results[0].type, fir.ReferenceType)
            # As LLVM pointer types are opaque, we need to grab the element type from
            # the declaration fir.reference type
            load_op = llvm.LoadOp(
                ctx[op.memref],
                ftn_types.convert_fir_type_to_standard(
                    op.memref.owner.results[0].type.type
                ),
            )
            ctx[op.results[0]] = load_op.results[0]
            return [load_op]
        else:
            assert False
    elif isa(op.memref.owner, hlfir.DesignateOp):
        # Array value
        assert op.memref.owner.indices is not None
        assert isa(op.memref.owner.memref.owner, hlfir.DeclareOp) or isa(
            op.memref.owner.memref.owner, fir.LoadOp
        )
        ops_list, indexes_ssa = array_access_components(
            program_state, ctx, op.memref.owner
        )

        if isa(op.memref.owner.memref.owner, hlfir.DeclareOp):
            src_ssa = op.memref.owner.memref
        elif isa(op.memref.owner.memref.owner, fir.LoadOp):
            src_ssa = op.memref.owner.memref.owner.memref
        else:
            assert False

        assert isa(ctx[src_ssa].type, builtin.MemRefType)
        if isa(ctx[src_ssa].type.element_type, builtin.MemRefType):
            load_op, load_ssa = generate_dereference_memref(ctx[src_ssa])
            ops_list.append(load_op)
        else:
            load_ssa = ctx[src_ssa]

        # Reverse the indicies as Fortran and C/MLIR are opposite in terms of
        # the order of the contiguous dimension (F is least, whereas C/MLIR is highest)
        indexes_ssa_reversed = indexes_ssa.copy()
        indexes_ssa_reversed.reverse()
        load_op = memref.LoadOp.get(load_ssa, indexes_ssa_reversed)
        ops_list.append(load_op)
        ctx[op.results[0]] = load_op.results[0]
        return ops_list
    elif isa(op.memref.owner, fir.AllocaOp):
        # This is used for loading an internal variable
        assert isa(op.memref.owner.results[0].type, fir.ReferenceType)

        load_op = memref.LoadOp.get(ctx[op.memref], [])
        ctx[op.results[0]] = load_op.results[0]
        return [load_op]
    else:
        assert False


def translate_apply(program_state, ctx, op: hlfir.ApplyOp):
    # A bit of a confusing name, the apply operation will load values, based
    # on provided index(es) from an hlfir.ExprType. Effectively, this is
    # loading values from a temporary created by an elemental.
    if ctx.contains(op.results[0]):
        return []

    ops_list = []

    ops_list += expressions.translate_expr(program_state, ctx, op.expr)

    assert isa(op.expr.type, hlfir.ExprType)
    assert isa(ctx[op.expr].type, builtin.MemRefType)

    load_index_ssa = []
    for idx in op.indices:
        ops_list += expressions.translate_expr(program_state, ctx, idx)
        # Our handling of the elemental operation increments scf loop index by one
        # so it starts from 1 as per Fortran. This works fine generally, as for memory
        # loads Fortran applies the subi to minus one, however for the apply operator
        # it assumes indexes start at 1. Therefore if the index has had this increment
        # applied to it, then just point directly to the scf loop index so that we are
        # starting counting from zero
        if (
            isa(ctx[idx].owner, arith.AddiOp)
            and isa(ctx[idx].owner.lhs, BlockArgument)
            and isa(ctx[idx].owner.rhs.owner, arith.ConstantOp)
            and isa(ctx[idx].owner.rhs.owner.value.type, builtin.IndexType)
            and ctx[idx].owner.rhs.owner.value.value.data == 1
        ):
            load_index_ssa.append(ctx[idx].owner.lhs)
        else:
            load_index_ssa.append(ctx[idx])
    load_index_ssa.reverse()
    load_op = memref.LoadOp.get(ctx[op.expr], load_index_ssa)
    ops_list.append(load_op)

    ctx[op.results[0]] = load_op.results[0]
    return ops_list


def translate_designate_op(program_state, ctx, op: hlfir.DesignateOp):
    """
    We ignore this for now, it tends to be used to extract part of an array or a reference to a value,
    with the array and indicies provided as arguments. Could implement by extracting value and then
    wrapping in memref and grabbing LLVM pointer from that, or alternatively get LLVM pointer of
    original memref and do pointer arithmetic on it (second option is better!)
    """
    if ctx.contains(op.results[0]):
        return []

    raise Exception(
        "Designate Op is not supported at the moment, need to complete this in the load store"
    )

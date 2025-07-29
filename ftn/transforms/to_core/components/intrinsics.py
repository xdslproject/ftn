from xdsl.dialects.experimental import fir, hlfir
from xdsl.utils.hints import isa
from xdsl.ir import Block, Region
from xdsl.dialects import builtin, arith, memref, linalg

from ftn.transforms.to_core.misc.fortran_code_description import ProgramState
from ftn.transforms.to_core.misc.ssa_context import SSAValueCtx

from ftn.transforms.to_core.utils import (
    generate_dereference_memref,
    create_index_constant,
)

import ftn.transforms.to_core.expressions as expressions


def handle_create_temporary_linalg_output_memref(
    result_type, element_type, input_ssas, input_dims_to_read
):
    output_shape = [
        -1 if isa(s, fir.DeferredAttr) else s.value for s in result_type.shape
    ]

    ops_list = []
    dynamic_sizes = []
    if -1 in output_shape:
        # If we have deferred sizes then grab the output sizes from the input array sizes
        # Ensure all elements are -1
        assert len(set(output_shape)) == 1
        for input_ssa, input_dim_to_read in zip(input_ssas, input_dims_to_read):
            dim_idx = create_index_constant(input_dim_to_read)
            dim_load_size = memref.DimOp.from_source_and_index(input_ssa, dim_idx)
            dynamic_sizes.append(dim_load_size.results[0])
            ops_list += [dim_idx, dim_load_size]

    output_memref_op = memref.AllocOp.get(
        element_type,
        shape=output_shape,
        dynamic_sizes=dynamic_sizes,
    )

    return output_memref_op.results[0], ops_list + [output_memref_op]


def translate_matmul(program_state: ProgramState, ctx: SSAValueCtx, op: hlfir.MatmulOp):
    if ctx.contains(op.results[0]):
        return []

    assert isa(op.results[0].type, hlfir.ExprType)

    lhs_ops_list = expressions.translate_expr(program_state, ctx, op.lhs)
    rhs_ops_list = expressions.translate_expr(program_state, ctx, op.rhs)

    if isa(ctx[op.lhs].type.element_type, builtin.MemRefType):
        load_op, lhs_load_ssa = generate_dereference_memref(ctx[op.lhs])
        lhs_ops_list.append(load_op)
    else:
        lhs_load_ssa = ctx[op.lhs]

    if isa(ctx[op.rhs].type.element_type, builtin.MemRefType):
        load_op, rhs_load_ssa = generate_dereference_memref(ctx[op.rhs])
        rhs_ops_list.append(load_op)
    else:
        rhs_load_ssa = ctx[op.rhs]

    memref_ssa, allocation_ops = handle_create_temporary_linalg_output_memref(
        op.results[0].type,
        lhs_load_ssa.type.element_type,
        [lhs_load_ssa, rhs_load_ssa],
        [0, 1],
    )

    assert isa(lhs_load_ssa.type, builtin.MemRefType)
    assert isa(rhs_load_ssa.type, builtin.MemRefType)

    matmul_op = linalg.MatmulOp((lhs_load_ssa, rhs_load_ssa), [memref_ssa])

    ctx[op.results[0]] = memref_ssa

    return lhs_ops_list + rhs_ops_list + allocation_ops + [matmul_op]


def translate_dotproduct(
    program_state: ProgramState, ctx: SSAValueCtx, op: hlfir.DotProductOp
):
    if ctx.contains(op.results[0]):
        return []
    lhs_ops_list = expressions.translate_expr(program_state, ctx, op.lhs)
    rhs_ops_list = expressions.translate_expr(program_state, ctx, op.rhs)

    if isa(ctx[op.lhs].type.element_type, builtin.MemRefType):
        load_op, lhs_load_ssa = generate_dereference_memref(ctx[op.lhs])
        lhs_ops_list.append(load_op)
    else:
        lhs_load_ssa = ctx[op.lhs]

    if isa(ctx[op.rhs].type.element_type, builtin.MemRefType):
        load_op, rhs_load_ssa = generate_dereference_memref(ctx[op.rhs])
        rhs_ops_list.append(load_op)
    else:
        rhs_load_ssa = ctx[op.rhs]

    output_memref_op = memref.AllocaOp.get(op.results[0].type, shape=[])
    dot_op = linalg.DotOp((lhs_load_ssa, rhs_load_ssa), [output_memref_op])
    extract_op = memref.LoadOp.get(output_memref_op, [])

    ctx[op.results[0]] = extract_op.results[0]

    return lhs_ops_list + rhs_ops_list + [output_memref_op, dot_op, extract_op]


def translate_transpose(
    program_state: ProgramState, ctx: SSAValueCtx, op: hlfir.TransposeOp
):
    if ctx.contains(op.results[0]):
        return []

    array_ops_list = expressions.translate_expr(program_state, ctx, op.array)

    if isa(ctx[op.array].type.element_type, builtin.MemRefType):
        load_op, array_load_ssa = generate_dereference_memref(ctx[op.array])
        array_ops_list.append(load_op)
    else:
        array_load_ssa = ctx[op.array]

    memref_ssa, allocation_ops = handle_create_temporary_linalg_output_memref(
        op.results[0].type,
        array_load_ssa.type.element_type,
        [array_load_ssa, array_load_ssa],
        [0, 1],
    )

    transpose_op = linalg.TransposeOp(
        array_load_ssa,
        memref_ssa,
        builtin.DenseArrayBase.from_list(builtin.i64, [1, 0]),
    )

    ctx[op.results[0]] = memref_ssa

    return array_ops_list + allocation_ops + [transpose_op]


def translate_sum(program_state: ProgramState, ctx: SSAValueCtx, op: hlfir.SumOp):
    if ctx.contains(op.results[0]):
        return []

    ops_list = expressions.translate_expr(program_state, ctx, op.array)

    if isa(ctx[op.array].type.element_type, builtin.MemRefType):
        load_op, array_load_ssa = generate_dereference_memref(ctx[op.array])
        ops_list.append(load_op)
    else:
        array_load_ssa = ctx[op.array]

    if op.dim is not None:
        dim_ops_list = expressions.translate_expr(program_state, ctx, op.dim)
        # Just support constant expression as the dimension as need to know
        # this statically for the linalg.reduce operation
        assert len(dim_ops_list) == 1
        assert isa(dim_ops_list[0], arith.ConstantOp)
        assert dim_ops_list[0].value.type == builtin.i32
        reduction_dimensions = [
            (len(array_load_ssa.type.shape) - 1)
            - (dim_ops_list[0].value.value.data - 1)
        ]
    else:
        reduction_dimensions = list(range(len(array_load_ssa.type.shape)))

    input_array_shape = [s.data for s in array_load_ssa.type.shape]

    if len(reduction_dimensions) == len(array_load_ssa.type.shape):
        memref_shape = []
        memref_dynamic_sizes = []
    elif len(reduction_dimensions) == 1:
        if -1 in input_array_shape:
            assert len(set(input_array_shape)) == 1
            memref_shape = [-1] * (len(array_load_ssa.type.shape) - 1)
            memref_dynamic_sizes = []
            if len(array_load_ssa.type.shape) > 1:
                for dim in list(range(len(array_load_ssa.type.shape))):
                    if dim not in reduction_dimensions:
                        dim_const_op = create_index_constant(dim)
                        dim_op = memref.DimOp.from_source_and_index(
                            array_load_ssa, dim_const_op
                        )
                        ops_list += [dim_const_op, dim_op]
                        memref_dynamic_sizes.append(dim_op.results[0])
        else:
            memref_shape = list(array_load_ssa.type.shape)
            del memref_shape[reduction_dimensions[0]]
            memref_dynamic_sizes = []
    else:
        assert False

    if isa(op.results[0].type, hlfir.ExprType):
        # If this is an exprtype then it is an array output, put onto the heap
        base_type = op.results[0].type.elementType
        output_memref_op = memref.AllocOp.get(
            base_type, shape=memref_shape, dynamic_sizes=memref_dynamic_sizes
        )
    else:
        # For a singleton output use the stack
        base_type = op.results[0].type
        output_memref_op = memref.AllocaOp.get(
            base_type, shape=memref_shape, dynamic_sizes=memref_dynamic_sizes
        )

    block = Block(arg_types=[base_type, base_type])
    if isa(base_type, builtin.IntegerType):
        zero_const = arith.ConstantOp.from_int_and_width(0, base_type)
        add_op = arith.AddiOp(block.args[0], block.args[1])
    elif isa(base_type, builtin.AnyFloat):
        zero_const = arith.ConstantOp(builtin.FloatAttr(0.0, base_type))
        add_op = arith.AddfOp(block.args[0], block.args[1])
    else:
        assert False

    yield_op = linalg.YieldOp(add_op)

    # We need to initialise the output memref to zero
    if len(output_memref_op.results[0].type.shape) == 0:
        # If it's a singleton then simply do a memref store
        initialise_output_memref_ops = [
            memref.StoreOp.get(zero_const, output_memref_op.results[0], [])
        ]
    else:
        # Otherwise we need to broadcast the constant to all elements
        const_memref = memref.AllocaOp.get(zero_const.results[0].type, shape=[])
        set_zero_const_memref = memref.StoreOp.get(
            zero_const, const_memref.results[0], []
        )
        initialise_output_memref = linalg.BroadcastOp(
            const_memref.results[0],
            output_memref_op.results[0],
            builtin.DenseArrayBase.from_list(
                builtin.i64, list(range(len(output_memref_op.results[0].type.shape)))
            ),
        )
        initialise_output_memref_ops = [
            const_memref,
            set_zero_const_memref,
            initialise_output_memref,
        ]

    block.add_ops([add_op, yield_op])

    reduce_op = linalg.ReduceOp(
        array_load_ssa,
        output_memref_op.results[0],
        builtin.DenseArrayBase.from_list(builtin.i64, reduction_dimensions),
        Region([block]),
    )

    reduction_ops = [reduce_op]
    if len(output_memref_op.results[0].type.shape) == 0:
        # If it is a singleton then extract that
        extract_op = memref.LoadOp.get(output_memref_op, [])
        reduction_ops.append(extract_op)
        ctx[op.results[0]] = extract_op.results[0]
    else:
        # Otherwise the return is the output array
        ctx[op.results[0]] = output_memref_op.results[0]

    return (
        ops_list
        + [output_memref_op, zero_const]
        + initialise_output_memref_ops
        + reduction_ops
    )


def handle_movealloc_intrinsic_call(
    program_state: ProgramState, ctx: SSAValueCtx, op: fir.CallOp
):
    src_ssa = op.args[1]
    dst_ssa = op.args[0]

    src_list = expressions.translate_expr(program_state, ctx, src_ssa)
    dst_list = expressions.translate_expr(program_state, ctx, dst_ssa)

    assert isa(ctx[src_ssa].type, builtin.MemRefType)
    assert isa(ctx[dst_ssa].type, builtin.MemRefType)
    load_op = memref.LoadOp.get(ctx[src_ssa], [])
    store_op = memref.StoreOp.get(load_op.results[0], ctx[dst_ssa], [])
    return [load_op, store_op]


FortranIntrinsicsHandleExplicitly = {
    "_FortranAMoveAlloc": handle_movealloc_intrinsic_call
}

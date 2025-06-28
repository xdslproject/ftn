from xdsl.dialects.experimental import fir, hlfir
from xdsl.utils.hints import isa
from xdsl.ir import Block, Region
from xdsl.dialects import builtin, arith, memref, linalg

from ftn.transforms.to_core.misc.fortran_code_description import ProgramState
from ftn.transforms.to_core.misc.ssa_context import SSAValueCtx

from ftn.transforms.to_core.utils import generate_dereference_memref

import ftn.transforms.to_core.expressions as expressions


def translate_matmul(program_state: ProgramState, ctx: SSAValueCtx, op: hlfir.MatmulOp):
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

    output_memref_op = memref.AllocaOp.get(
        lhs_load_ssa.type.element_type, shape=lhs_load_ssa.type.shape
    )
    matmul_op = linalg.MatmulOp(
        (lhs_load_ssa, rhs_load_ssa), [output_memref_op.results[0]]
    )

    ctx[op.results[0]] = output_memref_op.results[0]

    return lhs_ops_list + rhs_ops_list + [output_memref_op, matmul_op]


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

    ops_list = []
    if isa(ctx[op.array].type.element_type, builtin.MemRefType):
        load_op, array_load_ssa = generate_dereference_memref(ctx[op.array])
        ops_list.append(load_op)
    else:
        array_load_ssa = ctx[op.array]

    output_memref_op = memref.AllocaOp.get(
        array_load_ssa.type.element_type, shape=array_load_ssa.type.shape
    )

    transpose_op = linalg.TransposeOp(
        array_load_ssa,
        output_memref_op,
        builtin.DenseArrayBase.create_dense_int_or_index(builtin.i32, [1, 0]),
    )

    ctx[op.results[0]] = output_memref_op.results[0]

    return ops_list + [output_memref_op, transpose_op]


def translate_sum(program_state: ProgramState, ctx: SSAValueCtx, op: hlfir.SumOp):
    if ctx.contains(op.results[0]):
        return []

    array_ops_list = expressions.translate_expr(program_state, ctx, op.array)

    ops_list = []
    if isa(ctx[op.array].type.element_type, builtin.MemRefType):
        load_op, array_load_ssa = generate_dereference_memref(ctx[op.array])
        ops_list.append(load_op)
    else:
        array_load_ssa = ctx[op.array]

    output_memref_op = memref.AllocaOp.get(op.results[0].type, shape=[])

    block = Block(arg_types=[op.results[0].type, op.results[0].type])
    if isa(op.results[0].type, builtin.IntegerType):
        zero_const = arith.ConstantOp.from_int_and_width(0, op.results[0].type)
        add_op = arith.AddiOp(block.args[0], block.args[1])
    elif isa(op.results[0].type, builtin.AnyFloat):
        if isa(op.results[0].type, builtin.Float16Type):
            width = 16
        elif isa(op.results[0].type, builtin.Float32Type):
            width = 32
        elif isa(op.results[0].type, builtin.Float64Type):
            width = 64
        else:
            assert False
        zero_const = arith.ConstantOp.from_float_and_width(0.0, width)
        add_op = arith.AddfOp(block.args[0], block.args[1])
    else:
        assert False

    yield_op = linalg.YieldOp(add_op)

    # We need to initialise the output memref to zero
    initialise_output_memref = memref.StoreOp.get(
        zero_const, output_memref_op.results[0], []
    )

    block.add_ops([add_op, yield_op])

    reduce_op = linalg.ReductionOp(
        [array_load_ssa],
        [output_memref_op],
        builtin.DenseArrayBase.create_dense_int_or_index(builtin.i32, [0]),
        Region([block]),
    )
    extract_op = memref.LoadOp.get(output_memref_op, [])

    ctx[op.results[0]] = extract_op.results[0]

    return ops_list + [
        output_memref_op,
        zero_const,
        initialise_output_memref,
        reduce_op,
        extract_op,
    ]


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

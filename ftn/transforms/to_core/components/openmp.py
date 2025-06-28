from xdsl.ir import Block, Region
from xdsl.dialects import omp

from ftn.transforms.to_core.misc.fortran_code_description import ProgramState
from ftn.transforms.to_core.misc.ssa_context import SSAValueCtx

from ftn.transforms.to_core.utils import create_index_constant

import ftn.transforms.to_core.expressions as expressions
import ftn.transforms.to_core.statements as statements


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


def translate_omp_parallel(
    program_state: ProgramState, ctx: SSAValueCtx, op: omp.TeamsOp
):
    arg_ssa = []
    arg_ops = []

    if op.if_expr_var is not None:
        ops = expressions.translate_expr(program_state, ctx, op.if_expr_var)
        arg_ops += ops
        arg_ssa.append([ops[-1].results[0]])
    else:
        arg_ssa.append([])

    if op.num_threads_var is not None:
        ops = expressions.translate_expr(program_state, ctx, op.num_threads_var)
        arg_ops += ops
        arg_ssa.append([ops[-1].results[0]])
    else:
        arg_ssa.append([])

    arg_ssa += [[], [], []]

    new_block = Block()

    region_body_ops = []
    for single_op in op.region.blocks[0].ops:
        region_body_ops += statements.translate_stmt(program_state, ctx, single_op)

    new_block.add_ops(region_body_ops)

    return arg_ops + [
        omp.ParallelOp.build(
            operands=arg_ssa, regions=[Region([new_block])], properties={}
        )
    ]


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


def translate_omp_simdloop(
    program_state: ProgramState, ctx: SSAValueCtx, op: omp.SIMDOp
):
    arg_types = []

    lb_ops = expressions.translate_expr(program_state, ctx, op.lowerBound[0])
    ub_ops = expressions.translate_expr(program_state, ctx, op.upperBound[0])
    step_ops = expressions.translate_expr(program_state, ctx, op.step[0])

    arg_types = [
        lb_ops[-1].results[0].type,
        ub_ops[-1].results[0].type,
        step_ops[-1].results[0].type,
    ]

    new_block = Block(arg_types=arg_types)

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

    simd_op = omp.SIMDOp.build(
        operands=[lb_ops, ub_ops, step_ops, [], [], []],
        regions=[Region([new_block])],
        properties=new_props,
    )

    return lb_ops + ub_ops + step_ops + [simd_op]


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

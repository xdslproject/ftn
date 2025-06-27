from abc import ABC
from enum import Enum
import itertools
import copy
from functools import reduce
from typing import TypeVar, cast
from dataclasses import dataclass
from xdsl.dialects.experimental import fir, hlfir
from dataclasses import dataclass, field
from typing import Dict, Optional
from xdsl.ir import SSAValue, BlockArgument
from xdsl.irdl import Operand
from xdsl.utils.hints import isa
from util.visitor import Visitor
from xdsl.context import Context
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

from ftn.transforms.to_core.misc.fortran_code_description import ProgramState
from ftn.transforms.to_core.misc.ssa_context import SSAValueCtx

from ftn.transforms.to_core.utils import create_index_constant

import ftn.transforms.to_core.expressions as expressions
import ftn.transforms.to_core.statements as statements


def translate_omp_mapinfo(
    program_state: ProgramState, ctx: SSAValueCtx, op: omp.MapInfoOp
):
    var_ptr_ops = []
    var_ptr_ssa = []
    var_ptr_type = []
    for arg in op.var_ptr:
        var_ptr_ops += expressions.translate_expr(program_state, ctx, arg)
        var_ptr_ssa.append(ctx[arg])
        var_ptr_type.append(ctx[arg].type)

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
        operands=[var_ptr_ssa, members_ssa, bounds_ssa],
        properties={
            "map_type": op.map_type,
            "var_name": op.var_name,
            "var_type": var_ptr_type[0],
        },
        result_types=var_ptr_type,
    )

    return var_ptr_ops + members_ops + bounds_ops + [mapinfo_op]


def translate_omp_bounds(
    program_state: ProgramState, ctx: SSAValueCtx, op: omp.BoundsOp
):
    lower_ops = []
    lower_ssa = []
    for arg in op.lower:
        lower_ops += expressions.translate_expr(program_state, ctx, arg)
        lower_ssa.append(ctx[arg])

    upper_ops = []
    upper_ssa = []
    for arg in op.upper:
        pass  # upper_ops+=translate_expr(program_state, ctx, arg)
        # upper_ssa.append(ctx[arg])

    extent_ops = []
    extent_ssa = []
    for arg in op.extent:
        extent_ops += expressions.translate_expr(program_state, ctx, arg)
        extent_ssa.append(ctx[arg])

    stride_ops = []
    stride_ssa = []
    for arg in op.stride:
        pass  # stride_ops+=translate_expr(program_state, ctx, arg)
        # stride_ssa.append(ctx[arg])

    start_ops = []
    start_ssa = []
    for arg in op.start:
        pass  # start_ops+=translate_expr(program_state, ctx, arg)
        # start_ssa.append(ctx[arg])

    bounds_op = omp.BoundsOp.build(
        operands=[lower_ssa, upper_ssa, extent_ssa, stride_ssa, start_ssa],
        properties={"stride_in_bytes": op.stride_in_bytes},
        result_types=[omp.DataBoundsTy()],
    )

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
    program_state: ProgramState, ctx: SSAValueCtx, op: omp.SIMDLoopOp
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

    simd_op = omp.SIMDLoopOp.build(
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

    target_op = omp.TargetOp.build(
        operands=[[], [], [], map_var_ssa],
        regions=[Region([new_block])],
        properties=new_props,
    )

    return map_var_ops + [target_op]

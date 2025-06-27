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
import ftn.transforms.to_core.expressions as expressions

def translate_select(program_state: ProgramState, ctx: SSAValueCtx, op: arith.SelectOp):
    if ctx.contains(op.results[0]):
        return []

    cond_ops_list = expressions.translate_expr(program_state, ctx, op.cond)
    lhs_ops_list = expressions.translate_expr(program_state, ctx, op.lhs)
    rhs_ops_list = expressions.translate_expr(program_state, ctx, op.rhs)

    select_op = arith.SelectOp(ctx[op.cond], ctx[op.lhs], ctx[op.rhs])

    ctx[op.results[0]] = select_op.results[0]

    return cond_ops_list + lhs_ops_list + rhs_ops_list + [select_op]

def translate_cmp(
    program_state: ProgramState, ctx: SSAValueCtx, op: arith.CmpiOp | arith.CmpfOp
):
    if ctx.contains(op.results[0]):
        return []

    lhs_expr_ops = expressions.translate_expr(program_state, ctx, op.lhs)
    rhs_expr_ops = expressions.translate_expr(program_state, ctx, op.rhs)

    if isa(op, arith.CmpiOp):
        comparison_op = arith.CmpiOp(ctx[op.lhs], ctx[op.rhs], op.predicate.value.data)
    elif isa(op, arith.CmpfOp):
        comparison_op = arith.CmpfOp(ctx[op.lhs], ctx[op.rhs], op.predicate.value.data)
    else:
        assert False

    ctx[op.results[0]] = comparison_op.results[0]
    return lhs_expr_ops + rhs_expr_ops + [comparison_op]

def translate_constant(
    program_state: ProgramState, ctx: SSAValueCtx, op: arith.ConstantOp
):
    if ctx.contains(op.results[0]):
        return []
    new_const = arith.ConstantOp(op.value, op.results[0].type)
    ctx[op.results[0]] = new_const.results[0]
    return [new_const]

def translate_float_unary_arithmetic(
    program_state: ProgramState, ctx: SSAValueCtx, op: Operation
):
    if ctx.contains(op.results[0]):
        return []
    operand_ops = expressions.translate_expr(program_state, ctx, op.operand)
    operand_ssa = ctx[op.operand]
    fast_math_attr = op.fastmath
    result_type = op.results[0].type
    unary_arith_op = None

    if isa(op, arith.NegfOp):
        unary_arith_op = arith.NegfOp(operand_ssa, fast_math_attr)
    else:
        raise Exception(f"Could not translate `{op}' as a unary float operation")

    assert unary_arith_op is not None
    ctx[op.results[0]] = unary_arith_op.results[0]
    return operand_ops + [unary_arith_op]


def translate_float_binary_arithmetic(
    program_state: ProgramState, ctx: SSAValueCtx, op: Operation
):
    if ctx.contains(op.results[0]):
        return []
    lhs_ops = expressions.translate_expr(program_state, ctx, op.lhs)
    rhs_ops = expressions.translate_expr(program_state, ctx, op.rhs)
    lhs_ssa = ctx[op.lhs]
    rhs_ssa = ctx[op.rhs]
    fast_math_attr = op.fastmath
    result_type = op.results[0].type
    bin_arith_op = None

    if isa(op, arith.AddfOp):
        bin_arith_op = arith.AddfOp(lhs_ssa, rhs_ssa, fast_math_attr, result_type)
    elif isa(op, arith.SubfOp):
        bin_arith_op = arith.SubfOp(lhs_ssa, rhs_ssa, fast_math_attr, result_type)
    elif isa(op, arith.MulfOp):
        bin_arith_op = arith.MulfOp(lhs_ssa, rhs_ssa, fast_math_attr, result_type)
    elif isa(op, arith.DivfOp):
        bin_arith_op = arith.DivfOp(lhs_ssa, rhs_ssa, fast_math_attr, result_type)
    elif isa(op, arith.MaximumfOp):
        bin_arith_op = arith.MaximumfOp(lhs_ssa, rhs_ssa, fast_math_attr, result_type)
    elif isa(op, arith.MaxnumfOp):
        bin_arith_op = arith.MaxnumfOp(lhs_ssa, rhs_ssa, fast_math_attr, result_type)
    elif isa(op, arith.MinimumfOp):
        bin_arith_op = arith.MinimumfOp(lhs_ssa, rhs_ssa, fast_math_attr, result_type)
    elif isa(op, arith.MinnumfOp):
        bin_arith_op = arith.MinnumfOp(lhs_ssa, rhs_ssa, fast_math_attr, result_type)
    else:
        raise Exception(f"Could not translate `{op}' as a binary float operation")

    assert bin_arith_op is not None
    ctx[op.results[0]] = bin_arith_op.results[0]
    return lhs_ops + rhs_ops + [bin_arith_op]


def translate_integer_binary_arithmetic(
    program_state: ProgramState, ctx: SSAValueCtx, op: Operation
):
    if ctx.contains(op.results[0]):
        return []
    lhs_ops = expressions.translate_expr(program_state, ctx, op.lhs)
    rhs_ops = expressions.translate_expr(program_state, ctx, op.rhs)
    lhs_ssa = ctx[op.lhs]
    rhs_ssa = ctx[op.rhs]
    bin_arith_op = None

    if isa(op, arith.AddiOp):
        bin_arith_op = arith.AddiOp(lhs_ssa, rhs_ssa)
    elif isa(op, arith.SubiOp):
        bin_arith_op = arith.SubiOp(lhs_ssa, rhs_ssa)
    elif isa(op, arith.MuliOp):
        bin_arith_op = arith.MuliOp(lhs_ssa, rhs_ssa)
    elif isa(op, arith.DivUIOp):
        bin_arith_op = arith.DivUIOp(lhs_ssa, rhs_ssa)
    elif isa(op, arith.DivSIOp):
        bin_arith_op = arith.DivSIOp(lhs_ssa, rhs_ssa)
    elif isa(op, arith.FloorDivSIOp):
        bin_arith_op = arith.FloorDivSIOp(lhs_ssa, rhs_ssa)
    elif isa(op, arith.CeilDivSIOp):
        bin_arith_op = arith.CeilDivSIOp(lhs_ssa, rhs_ssa)
    elif isa(op, arith.CeilDivUIOp):
        bin_arith_op = arith.CeilDivUIOp(lhs_ssa, rhs_ssa)
    elif isa(op, arith.RemUIOp):
        bin_arith_op = arith.RemUIOp(lhs_ssa, rhs_ssa)
    elif isa(op, arith.RemSIOp):
        bin_arith_op = arith.RemSIOp(lhs_ssa, rhs_ssa)
    elif isa(op, arith.MinUIOp):
        bin_arith_op = arith.MinUIOp(lhs_ssa, rhs_ssa)
    elif isa(op, arith.MaxUIOp):
        bin_arith_op = arith.MaxUIOp(lhs_ssa, rhs_ssa)
    elif isa(op, arith.MinSIOp):
        bin_arith_op = arith.MinSIOp(lhs_ssa, rhs_ssa)
    elif isa(op, arith.MaxSIOp):
        bin_arith_op = arith.MaxSIOp(lhs_ssa, rhs_ssa)
    elif isa(op, arith.AndIOp):
        bin_arith_op = arith.AndIOp(lhs_ssa, rhs_ssa)
    elif isa(op, arith.OrIOp):
        bin_arith_op = arith.OrIOp(lhs_ssa, rhs_ssa)
    elif isa(op, arith.XOrIOp):
        bin_arith_op = arith.XOrIOp(lhs_ssa, rhs_ssa)
    elif isa(op, arith.ShLIOp):
        bin_arith_op = arith.ShLIOp(lhs_ssa, rhs_ssa)
    elif isa(op, arith.ShRUIOp):
        bin_arith_op = arith.ShRUIOp(lhs_ssa, rhs_ssa)
    elif isa(op, arith.ShRSIOp):
        bin_arith_op = arith.ShRSIOp(lhs_ssa, rhs_ssa)
    elif isa(op, arith.AddUIExtendedOp):
        bin_arith_op = arith.AddUIExtendedOp(lhs_ssa, rhs_ssa)
    else:
        raise Exception(f"Could not translate `{op}' as a binary integer operation")

    assert bin_arith_op is not None
    ctx[op.results[0]] = bin_arith_op.results[0]
    return lhs_ops + rhs_ops + [bin_arith_op]


def translate_math_operation(
    program_state: ProgramState, ctx: SSAValueCtx, op: Operation
):
    if ctx.contains(op.results[0]):
        return []

    expr_ops = []
    if isa(op, math.FmaOp):
        expr_ops += expressions.translate_expr(program_state, ctx, op.a)
        expr_ops += expressions.translate_expr(program_state, ctx, op.b)
        expr_ops += expressions.translate_expr(program_state, ctx, op.c)
    else:
        if hasattr(op, "operand"):
            expr_ops += expressions.translate_expr(program_state, ctx, op.operand)
        if hasattr(op, "lhs"):
            expr_ops += expressions.translate_expr(program_state, ctx, op.lhs)
        if hasattr(op, "rhs"):
            expr_ops += expressions.translate_expr(program_state, ctx, op.rhs)

    math_op = None
    if isa(op, math.AbsFOp):
        math_op = op.__class__(ctx[op.operand], op.fastmath)
    elif isa(op, math.AbsIOp):
        math_op = op.__class__(ctx[op.operand])
    elif isa(op, math.Atan2Op):
        math_op = op.__class__(ctx[op.lhs], ctx[op.rhs], op.fastmath)
    elif isa(op, math.AtanOp):
        math_op = op.__class__(ctx[op.operand], op.fastmath)
    elif isa(op, math.CbrtOp):
        math_op = op.__class__(ctx[op.operand], op.fastmath)
    elif isa(op, math.CeilOp):
        math_op = op.__class__(ctx[op.operand], op.fastmath)
    elif isa(op, math.CopySignOp):
        math_op = op.__class__(ctx[op.lhs], ctx[op.rhs], op.fastmath)
    elif isa(op, math.CosOp):
        math_op = op.__class__(ctx[op.operand], op.fastmath)
    elif isa(op, math.CountLeadingZerosOp):
        math_op = op.__class__(ctx[op.operand])
    elif isa(op, math.CountTrailingZerosOp):
        math_op = op.__class__(ctx[op.operand])
    elif isa(op, math.CtPopOp):
        math_op = op.__class__(ctx[op.operand])
    elif isa(op, math.ErfOp):
        math_op = op.__class__(ctx[op.operand], op.fastmath)
    elif isa(op, math.Exp2Op):
        math_op = op.__class__(ctx[op.operand], op.fastmath)
    elif isa(op, math.ExpM1Op):
        math_op = op.__class__(ctx[op.operand], op.fastmath)
    elif isa(op, math.ExpOp):
        math_op = op.__class__(ctx[op.operand], op.fastmath)
    elif isa(op, math.FPowIOp):
        math_op = op.__class__(ctx[op.lhs], ctx[op.rhs], op.fastmath)
    elif isa(op, math.FloorOp):
        math_op = op.__class__(ctx[op.operand], op.fastmath)
    elif isa(op, math.FmaOp):
        math_op = op.__class__(ctx[op.operand])
    elif isa(op, math.IPowIOp):
        math_op = op.__class__(ctx[op.lhs], ctx[op.rhs])
    elif isa(op, math.Log10Op):
        math_op = op.__class__(ctx[op.operand], op.fastmath)
    elif isa(op, math.Log1pOp):
        math_op = op.__class__(ctx[op.operand], op.fastmath)
    elif isa(op, math.Log2Op):
        math_op = op.__class__(ctx[op.operand], op.fastmath)
    elif isa(op, math.LogOp):
        math_op = op.__class__(ctx[op.operand], op.fastmath)
    elif isa(op, math.PowFOp):
        math_op = op.__class__(ctx[op.lhs], ctx[op.rhs], op.fastmath)
    elif isa(op, math.RoundEvenOp):
        math_op = op.__class__(ctx[op.operand], op.fastmath)
    elif isa(op, math.RoundOp):
        math_op = op.__class__(ctx[op.operand], op.fastmath)
    elif isa(op, math.RsqrtOp):
        math_op = op.__class__(ctx[op.operand], op.fastmath)
    elif isa(op, math.SinOp):
        math_op = op.__class__(ctx[op.operand], op.fastmath)
    elif isa(op, math.SqrtOp):
        math_op = op.__class__(ctx[op.operand], op.fastmath)
    elif isa(op, math.TanOp):
        math_op = op.__class__(ctx[op.operand], op.fastmath)
    elif isa(op, math.TanhOp):
        math_op = op.__class__(ctx[op.operand], op.fastmath)
    elif isa(op, math.TruncOp):
        math_op = op.__class__(ctx[op.operand], op.fastmath)
    else:
        raise Exception(f"Could not translate `{op}' as a math operation")

    assert math_op is not None
    ctx[op.results[0]] = math_op.results[0]
    return expr_ops + [math_op]

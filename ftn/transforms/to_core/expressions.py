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

import ftn.transforms.to_core.components.intrinsics as ftn_intrinsics
import ftn.transforms.to_core.components.maths as ftn_maths
import ftn.transforms.to_core.components.types as ftn_types
import ftn.transforms.to_core.components.load_store as ftn_load_store
import ftn.transforms.to_core.components.memory as ftn_memory
import ftn.transforms.to_core.components.functions as ftn_functions
import ftn.transforms.to_core.components.control_flow as ftn_ctrl_flow

def translate_expr(program_state: ProgramState, ctx: SSAValueCtx, ssa_value: SSAValue):
    if isa(ssa_value, BlockArgument):
        return []
    else:
        ops = try_translate_expr(program_state, ctx, ssa_value.owner)
        if ops is not None:
            return ops

        raise Exception(f"Could not translate `{ssa_value.owner}' as an expression")


def try_translate_expr(program_state: ProgramState, ctx: SSAValueCtx, op: Operation):
    if isa(op, arith.ConstantOp):
        return ftn_maths.translate_constant(program_state, ctx, op)
    elif (
        isa(op, arith.AddiOp)
        or isa(op, arith.SubiOp)
        or isa(op, arith.MuliOp)
        or isa(op, arith.DivUIOp)
        or isa(op, arith.DivSIOp)
        or isa(op, arith.FloorDivSIOp)
        or isa(op, arith.CeilDivSIOp)
        or isa(op, arith.CeilDivUIOp)
        or isa(op, arith.RemUIOp)
        or isa(op, arith.RemSIOp)
        or isa(op, arith.MinUIOp)
        or isa(op, arith.MaxUIOp)
        or isa(op, arith.MinSIOp)
        or isa(op, arith.MaxSIOp)
        or isa(op, arith.AndIOp)
        or isa(op, arith.OrIOp)
        or isa(op, arith.XOrIOp)
        or isa(op, arith.ShLIOp)
        or isa(op, arith.ShRUIOp)
        or isa(op, arith.ShRSIOp)
        or isa(op, arith.AddUIExtendedOp)
    ):
        return ftn_maths.translate_integer_binary_arithmetic(program_state, ctx, op)
    elif (
        isa(op, arith.AddfOp)
        or isa(op, arith.SubfOp)
        or isa(op, arith.MulfOp)
        or isa(op, arith.DivfOp)
        or isa(op, arith.MaximumfOp)
        or isa(op, arith.MaxnumfOp)
        or isa(op, arith.MinimumfOp)
        or isa(op, arith.MinnumfOp)
    ):
        return ftn_maths.translate_float_binary_arithmetic(program_state, ctx, op)
    elif isa(op, arith.NegfOp):
        return ftn_maths.translate_float_unary_arithmetic(program_state, ctx, op)
    elif isa(op, fir.LoadOp):
        return ftn_load_store.translate_load(program_state, ctx, op)
    elif isa(op, fir.ConvertOp):
        return ftn_types.translate_convert(program_state, ctx, op)
    elif isa(op, fir.DoLoopOp):
        # Do loop can be either an expression or statement
        return ftn_ctrl_flow.translate_do_loop(program_state, ctx, op)
    elif isa(op, fir.IterateWhileOp):
        return translate_iterate_while(program_state, ctx, op)
    elif isa(op, hlfir.DeclareOp):
        return ftn_memory.translate_declare(program_state, ctx, op)
    elif isa(op, arith.CmpiOp) or isa(op, arith.CmpfOp):
        return ftn_maths.translate_cmp(program_state, ctx, op)
    elif isa(op, fir.CallOp):
        return ftn_functions.translate_call(program_state, ctx, op)
    elif isa(op, fir.StringLitOp):
        return ftn_types.translate_string_literal(program_state, ctx, op)
    elif isa(op, fir.AddressOfOp):
        return ftn_memory.translate_address_of(program_state, ctx, op)
    elif isa(op, hlfir.NoReassocOp) or isa(op, fir.NoReassocOp):
        return ftn_memory.translate_reassoc(program_state, ctx, op)
    elif isa(op, fir.ZeroBitsOp):
        return ftn_memory.translate_zerobits(program_state, ctx, op)
    elif isa(op, fir.BoxAddrOp):
        # Ignore box address, just process argument and link to results of that
        expr_list = translate_expr(program_state, ctx, op.val)
        ctx[op.results[0]] = ctx[op.val]
        return expr_list
    elif isa(op, arith.SelectOp):
        return ftn_maths.translate_select(program_state, ctx, op)
    elif isa(op, fir.EmboxOp) or isa(op, fir.EmboxcharOp):
        expr_ops = translate_expr(program_state, ctx, op.memref)
        ctx[op.results[0]] = ctx[op.memref.owner.results[0]]
        return expr_ops
    elif isa(op, hlfir.AssociateOp):
        expr_ops = translate_expr(program_state, ctx, op.source)
        ctx[op.results[0]] = ctx[op.source.owner.results[0]]
        return expr_ops
    elif isa(op, hlfir.AsExprOp):
        expr_ops = translate_expr(program_state, ctx, op.var)
        ctx[op.results[0]] = ctx[op.var]
        return expr_ops
    elif isa(op, fir.ReboxOp):
        expr_ops = translate_expr(program_state, ctx, op.box)
        ctx[op.results[0]] = ctx[op.box]
        return expr_ops
    elif isa(op, hlfir.DotProductOp):
        return ftn_intrinsics.translate_dotproduct(program_state, ctx, op)
    elif isa(op, hlfir.CopyInOp):
        return translate_copyin(program_state, ctx, op)
    elif isa(op, fir.AbsentOp):
        return memory.translate_absent(program_state, ctx, op)
    elif isa(op, hlfir.SumOp):
        return ftn_intrinsics.translate_sum(program_state, ctx, op)
    elif isa(op, hlfir.TransposeOp):
        return ftn_intrinsics.translate_transpose(program_state, ctx, op)
    elif isa(op, hlfir.MatmulOp):
        return translate_matmul(program_state, ctx, op)
    #elif isa(op, omp.BoundsOp):
    #    return translate_omp_bounds(program_state, ctx, op)
    #elif isa(op, omp.MapInfoOp):
    #    return translate_omp_mapinfo(program_state, ctx, op)
    else:
        for math_op in math.Math.operations:
            # Check to see if this is a math operation
            if isa(op, math_op):
                return ftn_maths.translate_math_operation(program_state, ctx, op)
        return None

def translate_copyin(program_state: ProgramState, ctx: SSAValueCtx, op: hlfir.CopyInOp):
    if ctx.contains(op.results[0]):
        return []

    expr_ops = translate_expr(program_state, ctx, op.var)
    ctx[op.results[0]] = ctx[op.var]
    return expr_ops


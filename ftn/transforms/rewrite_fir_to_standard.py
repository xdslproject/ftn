from abc import ABC
from typing import TypeVar, cast
from dataclasses import dataclass
from xdsl.dialects.experimental import fir
from xdsl.utils.hints import isa
from xdsl.dialects.memref import MemRefType
from xdsl.ir import Operation, SSAValue, OpResult, Attribute, MLContext, Block, Region

from xdsl.pattern_rewriter import (RewritePattern, PatternRewriter,
                                   op_type_rewrite_pattern,
                                   PatternRewriteWalker,
                                   GreedyRewritePatternApplier)
from xdsl.passes import ModulePass
from xdsl.dialects import builtin, func, llvm, arith

class RewriteFirConvert(RewritePattern):
  """
  Rewrites FIR convert operation into standard dialect equivalent
  """
  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: fir.Convert, rewriter: PatternRewriter, /):
    in_type=op.value.type
    out_type=op.results[0].type
    new_conv=None
    if isinstance(in_type, builtin.Float32Type) and isinstance(out_type,
      builtin.Float64Type) or isinstance(in_type, builtin.Float16Type) and isinstance(out_type,
      builtin.Float64Type) or isinstance(in_type, builtin.Float16Type) and isinstance(out_type,
      builtin.Float32Type) :
      new_conv=arith.ExtFOp(op.value, out_type)

    if isinstance(in_type, builtin.Float64Type) and isinstance(out_type,
      builtin.Float32Type) or isinstance(in_type, builtin.Float64Type) and isinstance(out_type,
      builtin.Float16Type) or isinstance(in_type, builtin.Float32Type) and isinstance(out_type,
      builtin.Float16Type) :
      new_conv=arith.TruncFOp(op.value, out_type)

    if isinstance(in_type, builtin.IndexType) and isinstance(out_type, builtin.IntegerType):
      new_conv=arith.IndexCastOp(op.value, out_type)

    if isinstance(in_type, builtin.IntegerType) and isinstance(out_type, builtin.AnyFloat):
      new_conv=arith.SIToFPOp(op.value, out_type)

    if isinstance(in_type, builtin.AnyFloat) and isinstance(out_type, builtin.IntegerType):
      new_conv=arith.FPToSIOp(op.value, out_type)

    if new_conv is not None: rewriter.replace_matched_op(new_conv)


@dataclass(frozen=True)
class RewriteFIRToStandard(ModulePass):
  """
  This is the entry point for the transformation pass which will then apply the rewriter
  """
  name = 'rewrite-fir-to-standard'

  def apply(self, ctx: MLContext, module: builtin.ModuleOp):
    walker = PatternRewriteWalker(GreedyRewritePatternApplier([
              RewriteFirConvert(),
    ]),
                                   apply_recursively=True)
    walker.rewrite_module(module.regions[0].blocks[0].ops.first)

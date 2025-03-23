from abc import ABC
from enum import Enum
from typing import TypeVar, cast
from dataclasses import dataclass
import itertools
from xdsl.utils.hints import isa
from xdsl.dialects import memref, scf, linalg, func
from xdsl.ir import Operation, SSAValue, OpResult, Attribute, MLContext, Block, Region

from xdsl.pattern_rewriter import (RewritePattern, PatternRewriter,
                                   op_type_rewrite_pattern,
                                   PatternRewriteWalker,
                                   GreedyRewritePatternApplier)
from xdsl.passes import ModulePass
from xdsl.dialects import builtin, func, llvm, arith
from util.visitor import Visitor

ReductionOperation = Enum('ReductionOp', ['ADD', 'MUL'])

class HandleLinalgReduction(RewritePattern):
  
  def get_reduction_op(self, op):
    if isa(op, arith.Addi):
      return ReductionOperation.ADD

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: linalg.ReductionOp, rewriter: PatternRewriter, /):    
    reduction_type=self.get_reduction_op(op.body.blocks[0].ops.first)
    init_call_op=func.Call("xrt_init", [], [])
    
    exit(0)

@dataclass(frozen=True)
class LinalgToXrt(ModulePass):
  """
  This is the entry point for the transformation pass which will then apply the rewriter
  """
  name = 'linalg-to-xrt'

  def apply(self, ctx: MLContext, module: builtin.ModuleOp):    

    walker = PatternRewriteWalker(GreedyRewritePatternApplier([
              HandleLinalgReduction(),
    ]), apply_recursively=False, walk_reverse=True)
    walker.rewrite_module(module)

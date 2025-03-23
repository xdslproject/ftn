from abc import ABC
from typing import TypeVar, cast
from dataclasses import dataclass
import itertools
from xdsl.utils.hints import isa
from xdsl.dialects import memref, scf, omp
from xdsl.ir import Operation, SSAValue, OpResult, Attribute, MLContext, Block, Region

from xdsl.pattern_rewriter import (RewritePattern, PatternRewriter,
                                   op_type_rewrite_pattern,
                                   PatternRewriteWalker,
                                   GreedyRewritePatternApplier)
from xdsl.passes import ModulePass
from xdsl.dialects import builtin, func, llvm, arith
from util.visitor import Visitor

@dataclass(frozen=True)
class IsolateTarget(ModulePass):
  """
  This is the entry point for the transformation pass which will then apply the rewriter
  """
  name = 'isolate-target'

  def apply(self, ctx: MLContext, module: builtin.ModuleOp):
    for mod in module.regions[0].block.ops:
      assert isa(mod, builtin.ModuleOp)
      if "target" not in mod.attributes:
        mod.detach()              

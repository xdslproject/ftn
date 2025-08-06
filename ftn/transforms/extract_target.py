from abc import ABC
from ast import Module
from hmac import new
from typing import TypeVar, cast
from dataclasses import dataclass, field
import itertools
from xdsl.utils.hints import isa
from xdsl.dialects import memref, scf, omp
from xdsl.context import Context
from xdsl.ir import Operation, SSAValue, OpResult, Attribute, Block, Region

from xdsl.pattern_rewriter import (RewritePattern, PatternRewriter,
                                   op_type_rewrite_pattern,
                                   PatternRewriteWalker,
                                   GreedyRewritePatternApplier)
from xdsl.passes import ModulePass
from xdsl.dialects import builtin, func, llvm, arith
from xdsl.rewriter import InsertPoint

@dataclass
class RewriteTarget(RewritePattern):
  module : builtin.ModuleOp
  target_ops: list[Operation] = field(default_factory=list)

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: omp.TargetOp, rewriter: PatternRewriter, /):
    arg_types = []
    for var in op.has_device_addr_vars:
      assert isinstance(var.type, builtin.MemRefType)
      var_type = var.type
      arg_types.append(var_type)

    # The target function is extracted to a function in a different module. This function will 
    # be called in the original module instead. The function signature is necessary to have the 
    # function in the symbol table.
    call_dev_func = func.CallOp("tt_device", op.has_device_addr_vars, [])
    dev_func_signature = func.FuncOp.external("tt_device", arg_types, [])
    rewriter.insert_op(call_dev_func, InsertPoint.before(op))
    rewriter.insert_op(dev_func_signature, InsertPoint.at_start(self.module.body.block))

    dev_func_block = Block(arg_types=arg_types)
    op.detach()

    rewriter.insert_op(op, InsertPoint.at_start(dev_func_block))
    rewriter.insert_op(func.ReturnOp(), InsertPoint.at_end(dev_func_block))
    for block_arg,operand in zip(dev_func_block.args, op.has_device_addr_vars):
      operand.replace_by_if(block_arg, lambda use: use.operation == op)

    dev_func = func.FuncOp.from_region("tt_device", arg_types, [], Region(dev_func_block))


    self.target_ops = [dev_func]


@dataclass(frozen=True)
class ExtractTarget(ModulePass):
  """
  This is the entry point for the transformation pass which will then apply the rewriter
  """
  name = 'extract-target'

  def apply(self, ctx: Context, module: builtin.ModuleOp):
    rw_target= RewriteTarget(module)
    walker = PatternRewriteWalker(GreedyRewritePatternApplier([
              rw_target,
    ]), apply_recursively=False, walk_reverse=True)

    walker.rewrite_module(module)

    # NOTE: The region recieving the block must be empty. Otherwise, the single block region rule of
    # the module will not be satisfied.
    containing_mod=builtin.ModuleOp(Region())
    module.regions[0].move_blocks(containing_mod.regions[0])

    new_module=builtin.ModuleOp(rw_target.target_ops, {"target": builtin.StringAttr("tt_device")})

    block = Block()
    block.add_ops([new_module, containing_mod])
    module.regions[0].add_block(block)

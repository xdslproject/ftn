from abc import ABC
from ast import FunctionType, Module
from hmac import new
from json import load
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
from ftn.util.visitor import Visitor
from xdsl.rewriter import InsertPoint

@dataclass
class TargetFuncToHLS(RewritePattern):
    def deref_args(self, func_op: func.FuncOp, rewriter : PatternRewriter):
        """Dereference the arguments of a function operation."""
        new_input_types = []

        for arg in func_op.body.block.args:
            if isa(arg.type, builtin.MemRefType):
                deref_type = arg.type.element_type
                func_op.replace_argument_type(arg, deref_type, rewriter)
                new_input_types.append(deref_type)

    def forward_map_info(self, map_info: omp.MapInfoOp, rewriter: PatternRewriter):
        map_info.omp_ptr.replace_by(map_info.var_ptr)

    def deref_scalar_memops(self, scalar_ssa: SSAValue, rewriter: PatternRewriter):
        for use in scalar_ssa.uses:
            if isinstance(use.operation, memref.LoadOp):
                load_op = use.operation
                # NOTE: the operand of the load operation is not a memref anymore, we have dereferenced it, 
                # so we forward it.
                load_op.res.replace_by(load_op.memref)
                rewriter.erase_op(use.operation)
            elif isinstance(use.operation, memref.StoreOp):
                rewriter.erase_op(use.operation)

    def deref_memref_memops(self, memref_ssa: SSAValue, rewriter: PatternRewriter):
        for use in memref_ssa.uses:
            if isinstance(use.operation, memref.LoadOp):
                # The first load was used to load the pointer to the array. The index to retrieve an element from the array is applied 
                # in the next load. Since we have dereferenced the first pointer, we need to end up with a single load that accesses
                # the array directly.
                ptr_load_op = use.operation

                # FIXME: this is assuming each dereferencing load only has one use
                for ptr_use in ptr_load_op.res.uses:
                    if isinstance(ptr_use.operation, memref.LoadOp):
                        array_load_op = ptr_use.operation
                        array_idx = array_load_op.indices

                        new_load_op = memref.LoadOp.get(memref_ssa, array_idx)
                        array_load_op.res.replace_by(ptr_load_op.res)
                        rewriter.replace_op(ptr_load_op, new_load_op)
                        rewriter.erase_op(array_load_op)

                    elif isinstance(ptr_use.operation, memref.StoreOp):
                        array_store_op = ptr_use.operation
                        array_idx = array_store_op.indices
                        new_store_op = memref.StoreOp.get(array_store_op.value, memref_ssa, array_idx)
                        rewriter.insert_op(new_store_op, InsertPoint.before(ptr_load_op))
                        rewriter.erase_op(array_store_op)
                        rewriter.erase_op(ptr_load_op)


    def remove_target(self, target_op: omp.TargetOp, target_func: func.FuncOp, rewriter: PatternRewriter):
        """Remove the target operation from the module."""
        for operand in target_op.map_vars:
            operand_idx = target_op.operands.index(operand)
            block_arg = target_op.region.block.args[operand_idx]
            block_arg.replace_by(operand)

            if not isinstance(operand.type, builtin.MemRefType):
                self.deref_scalar_memops(operand, rewriter)
            else:
                self.deref_memref_memops(operand, rewriter)

        target_op_terminator = target_op.region.block.last_op
        assert target_op_terminator
        rewriter.erase_op(target_op_terminator)

        target_func_terminator = target_func.body.block.last_op
        assert target_func_terminator
        for block in reversed(target_op.region.blocks):
            rewriter.inline_block(block, InsertPoint.before(target_func_terminator))

        rewriter.erase_op(target_op)

    def remove_remaining_omp_ops(self, target_func: func.FuncOp, rewriter: PatternRewriter):
        """Remove any remaining OpenMP operations in the target function."""
        for op in target_func.walk():
            if isinstance(op, omp.MapInfoOp):
                rewriter.erase_op(op)

        for op in target_func.walk():
            if isinstance(op, omp.MapBoundsOp):
                rewriter.erase_op(op)

    @op_type_rewrite_pattern
    def match_and_rewrite(self, module: builtin.ModuleOp, rewriter: PatternRewriter, /):
        if "target" not in module.attributes:
            return

        target_name = module.attributes["target"].data
        target_func = [op for op in module.walk() if isinstance(op, func.FuncOp) and op.sym_name.data == target_name][0]

        self.deref_args(target_func, rewriter)
        for map_info in target_func.walk():
            if not isinstance(map_info, omp.MapInfoOp):
                continue

            self.forward_map_info(map_info, rewriter)

        target_op = [op for op in target_func.walk() if isinstance(op, omp.TargetOp)][0]
        self.remove_target(target_op, target_func, rewriter)
        self.remove_remaining_omp_ops(target_func, rewriter)


@dataclass(frozen=True)
class TargetToHLSPass(ModulePass):
  """
  This is the entry point for the transformation pass which will then apply the rewriter
  """
  name = 'target-to-hls'

  def apply(self, ctx: Context, module: builtin.ModuleOp):
    walker = PatternRewriteWalker(GreedyRewritePatternApplier([
              TargetFuncToHLS(),
    ]), apply_recursively=False, walk_reverse=True)

    walker.rewrite_module(module)
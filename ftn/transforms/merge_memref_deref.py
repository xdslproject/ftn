from abc import ABC
from typing import TypeVar, cast
from dataclasses import dataclass
import itertools
from xdsl.context import Context
from xdsl.utils.hints import isa
from xdsl.dialects import memref, scf
from xdsl.ir import Operation, SSAValue, OpResult, Attribute, Block, Region

from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter,
    op_type_rewrite_pattern,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
)
from xdsl.passes import ModulePass
from xdsl.dialects import builtin, func, llvm, arith
from util.visitor import Visitor


class GatherReferencedMemrefs(Visitor):
    def __init__(self):
        self.referenced_memrefs = []

    def traverse_alloca(self, alloca_op: memref.AllocaOp):
        assert isa(alloca_op.results[0].type, memref.MemRefType)
        if isa(alloca_op.results[0].type.element_type, memref.MemRefType):
            self.referenced_memrefs.append(alloca_op.results[0])


class CheckMemrefsForReassign(Visitor):
    def __init__(self, referenced_memrefs):
        self.referenced_memrefs = referenced_memrefs
        self.reassigned_memrefs = []

    def traverse_store(self, store_op: memref.StoreOp):
        assert isa(store_op.memref.type, memref.MemRefType)
        if isa(store_op.memref.type.element_type, memref.MemRefType):
            if store_op.memref in self.referenced_memrefs:
                if store_op.memref not in self.reassigned_memrefs:
                    self.reassigned_memrefs.append(store_op.memref)


class GatherAccessedRefMemrefs(Visitor):
    def __init__(self, referenced_memrefs):
        self.referenced_memrefs = referenced_memrefs
        self.accessed_ref_memrefs = []

    def traverse_load(self, load_op: memref.LoadOp):
        assert isa(load_op.memref.type, memref.MemRefType)
        if isa(load_op.memref.type.element_type, memref.MemRefType):
            if load_op.memref in self.referenced_memrefs:
                if load_op.memref not in self.accessed_ref_memrefs:
                    self.accessed_ref_memrefs.append(load_op.memref)


class CheckForLoop(Visitor):
    def __init__(self, referenced_memrefs):
        self.referenced_memrefs = referenced_memrefs

    def traverse_for(self, for_op: scf.ForOp):
        check_for_reassign = CheckMemrefsForReassign(self.referenced_memrefs)
        check_for_reassign.traverse(for_op)

        loaded_ref_memrefs = GatherAccessedRefMemrefs(self.referenced_memrefs)
        loaded_ref_memrefs.traverse(for_op)

        load_to_mem = {}
        top_level_ops = []

        for loaded_memref in loaded_ref_memrefs.accessed_ref_memrefs:
            if loaded_memref not in check_for_reassign.reassigned_memrefs:
                top_level_memref = memref.Load.get(loaded_memref, [])
                load_to_mem[loaded_memref] = top_level_memref.results[0]
                top_level_ops.append(top_level_memref)

        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    RewriteReferenceLoads(load_to_mem),
                ]
            ),
            apply_recursively=False,
        )
        walker.rewrite_module(for_op)
        for_op.parent.insert_ops_before(top_level_ops, for_op)


class RewriteFor(RewritePattern):
    def __init__(self, referenced_memrefs):
        self.referenced_memrefs = referenced_memrefs

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ForOp, rewriter: PatternRewriter, /):
        for_loop_analyser = CheckForLoop(self.referenced_memrefs)
        for_loop_analyser.traverse(op)


class RewriteReferenceLoads(RewritePattern):
    def __init__(self, load_to_mem):
        self.load_to_mem = load_to_mem

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.LoadOp, rewriter: PatternRewriter, /):
        assert isa(op.memref.type, memref.MemRefType)
        if (
            isa(op.memref.type.element_type, memref.MemRefType)
            and op.memref in self.load_to_mem.keys()
        ):
            rewriter.replace_matched_op([], [self.load_to_mem[op.memref]])


@dataclass(frozen=True)
class MergeMemRefDeref(ModulePass):
    """
    This is the entry point for the transformation pass which will then apply the rewriter
    """

    name = "merge-memref-deref"

    def apply(self, ctx: Context, module: builtin.ModuleOp):
        memref_visitor = GatherReferencedMemrefs()
        memref_visitor.traverse(module)

        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    RewriteFor(memref_visitor.referenced_memrefs),
                ]
            ),
            apply_recursively=False,
            walk_reverse=True,
        )
        walker.rewrite_module(module)

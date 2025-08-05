from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, omp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from ftn.dialects import device


@dataclass(frozen=True)
class LowerTargetOp(RewritePattern):
    target: str = None

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: omp.TargetOp, rewriter: PatternRewriter):
        assert self.target is not None

        rewriter.replace_op(op.region.block.ops.last, device.KernelTerminatorOp())

        kernel = device.KernelCreate(
            self.target,
            rewriter.move_region_contents_to_new_regions(op.region),
            [],
            [],
            op.has_device_addr_vars,
        )
        launch = device.KernelLaunch(kernel)
        wait = device.KernelWait(kernel)

        rewriter.replace_matched_op([kernel, launch, wait])


class OmpTargetToKernelPass(ModulePass):
    name = "omp-target-to-kernel"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        assert "omp.target_triples" in op.attributes

        assert len(op.attributes["omp.target_triples"]) == 1
        target = op.attributes["omp.target_triples"].data[0]

        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerTargetOp(target),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)

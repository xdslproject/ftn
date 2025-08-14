from dataclasses import dataclass, field
from xdsl.context import Context
from xdsl.ir import Operation, Block, Region

from xdsl.pattern_rewriter import (RewritePattern, PatternRewriter,
                                   op_type_rewrite_pattern,
                                   PatternRewriteWalker,
                                   GreedyRewritePatternApplier)
from xdsl.passes import ModulePass
from xdsl.dialects import builtin, func
from xdsl.rewriter import InsertPoint
from ftn.dialects import device

@dataclass
class RewriteTarget(RewritePattern):
  module : builtin.ModuleOp
  target_ops: list[Operation] = field(default_factory=list)

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: device.KernelCreate, rewriter: PatternRewriter, /):
    arg_types = []
    for var in op.mapped_data:
      assert isinstance(var.type, builtin.MemRefType)
      var_type = var.type
      arg_types.append(var_type)

    assert op.body
    dev_func_body = rewriter.move_region_contents_to_new_regions(op.body)
    dev_func = func.FuncOp.from_region(
        "tt_device",
        arg_types,
        [],
        dev_func_body
    )

    ## Fix type of the block arguments (there is a mismatch between mapped_data and block args)
    n_args = len(dev_func_body.block.args)
    for block_arg, arg_type in zip(dev_func_body.block.args, arg_types):
      new_block_arg = dev_func_body.block.insert_arg(arg_type, len(dev_func_body.block.args))
      block_arg.replace_by(new_block_arg)

    for arg in dev_func_body.block.args[:n_args]:
      rewriter.erase_block_argument(arg)

    # kernel_create cannot have both a pointer to a device_function and a body.
    op.device_function = builtin.SymbolRefAttr(dev_func.sym_name)

    assert dev_func_body.block.last_op is not None, "The last operation in the device function block must not be None"
    rewriter.erase_op(dev_func_body.block.last_op)

    rewriter.insert_op(func.ReturnOp(), InsertPoint.at_end(dev_func_body.block))

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

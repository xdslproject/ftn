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

    # The target function is extracted to a function in a different module. This function will 
    # be called in the original module instead. The function signature is necessary to have the 
    # function in the symbol table.
    call_dev_func = func.CallOp("tt_device", op.mapped_data, [device.KernelHandle()])
    dev_func_signature = func.FuncOp.external("tt_device", arg_types, [device.KernelHandle()])
    rewriter.insert_op(call_dev_func, InsertPoint.before(op))
    rewriter.insert_op(dev_func_signature, InsertPoint.at_start(self.module.body.block))

    dev_func_block = op.body.block

    # Fix type of the block arguments (there is a mismatch between mapped_data and block args)
    n_args = len(dev_func_block.args)
    for block_arg, arg_type in zip(dev_func_block.args, arg_types):
      new_block_arg = dev_func_block.insert_arg(arg_type, len(dev_func_block.args))
      block_arg.replace_by(new_block_arg)

    for arg in dev_func_block.args[:n_args]:
      rewriter.erase_block_argument(arg)

    op.res.replace_by(call_dev_func.results[0])
    op.body.detach_block(op.body.block)
    rewriter.erase_matched_op()
    assert dev_func_block.last_op is not None, "The last operation in the device function block must not be None"
    rewriter.erase_op(dev_func_block.last_op)

    dev_func_body = Region([dev_func_block])

    rewriter.insert_op(func.ReturnOp(), InsertPoint.at_end(dev_func_body.block))
    for block_arg,operand in zip(dev_func_block.args, op.mapped_data):
      operand.replace_by_if(block_arg, lambda use: use.operation == op)

    dev_func = func.FuncOp.from_region("tt_device", arg_types, [], dev_func_body)


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

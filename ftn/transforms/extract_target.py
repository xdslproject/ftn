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
from ftn.util.visitor import Visitor
from xdsl.rewriter import InsertPoint

@dataclass
class RewriteTarget(RewritePattern):
  module : builtin.ModuleOp
  target_ops: list[Operation] = field(default_factory=list)

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: omp.TargetOp, rewriter: PatternRewriter, /):
    arg_types=[]
    arg_ssa=[]

    loc_idx=0

    locations={}

    memref_dim_ops=[]
    # Grab bounds and info, then at end the terminator
    for var in op.map_vars:
      var_op=var.owner
      arg_types.append(var_op.var_ptr.type)
      arg_ssa.append(var_op.var_ptr)
      locations[var_op]=loc_idx
      loc_idx+=1
      if isa(var_op.var_ptr.type, builtin.MemRefType):
        memref_type=var_op.var_ptr.type
        src_memref=var_op.var_ptr
        if isa(memref_type.element_type, builtin.MemRefType):
          assert len(memref_type.shape) == 0
          memref_type=var_op.var_ptr.type.element_type
          memref_loadop=memref.LoadOp.get(src_memref, [])
          src_memref=memref_loadop.results[0]
          memref_dim_ops.append(memref_loadop)
        for idx, s in enumerate(memref_type.shape):
          assert isa(s, builtin.IntAttr)
          if (s.data == -1):
            # Need to pass the dimension shape size in explicitly as it is deferred
            const_op=arith.ConstantOp.from_int_and_width(idx, builtin.IndexType())
            dim_size=memref.DimOp.from_source_and_index(src_memref, const_op)
            memref_dim_ops+=[const_op, dim_size]
            arg_ssa.append(dim_size.results[0])
            arg_types.append(dim_size.results[0].type)
            loc_idx+=1

      if len(var_op.bounds) > 0:
        bound_op=var_op.bounds[0].owner
        arg_types.append(bound_op.lower_bound.type)
        arg_ssa.append(bound_op.lower_bound)
        arg_types.append(bound_op.upper_bound.type)
        arg_ssa.append(bound_op.upper_bound)
        locations[bound_op]=loc_idx
        # Adding both lower and upper bound
        loc_idx+=2
      else:
        pass#self.target_ops+=[var_op]

    new_block = Block(arg_types=arg_types)

    new_mapinfo_ssa=[]
    for var in op.map_vars:
      var_op=var.owner

      map_bounds=[]
      if len(var_op.bounds) > 0:
        bound_op=var_op.bounds[0].owner
        res_types=[]
        for res in bound_op.results: res_types.append(res.type)
        new_bounds_op=omp.MapBoundsOp.build(operands=[[new_block.args[locations[bound_op]]], [new_block.args[locations[bound_op]+1]], [], [], []],
                      properties={"stride_in_bytes": bound_op.stride_in_bytes},
                      result_types=res_types)

        new_block.add_op(new_bounds_op)
        map_bounds=[new_bounds_op.results[0]]

      res_types=[]
      for res in var_op.results: res_types.append(res.type)
      mapinfo_op=omp.MapInfoOp.build(operands=[new_block.args[locations[var_op]], [], [], map_bounds],
                      properties={"map_type": var_op.map_type, "name": var_op.var_name, "var_type": var_op.var_type, "map_capture_type": omp.VariableCaptureKindAttr(omp.VariableCaptureKind.BY_REF)},
                      result_types=res_types)
      new_mapinfo_ssa.append(mapinfo_op.results[0])

      new_block.add_op(mapinfo_op)

    reg=op.region
    op.detach_region(reg)

    new_omp_target_op=omp.TargetOp.build(operands=[[],[],[],[],[],[],[],[],[], new_mapinfo_ssa, [], []], regions=[reg])
    new_block.add_op(new_omp_target_op)
    new_block.add_op(func.ReturnOp())

    new_fn_type=builtin.FunctionType.from_lists(arg_types, [])

    body = Region()
    body.add_block(new_block)

    new_func=func.FuncOp("tt_device", new_fn_type, body)
    new_func_signature=func.FuncOp.external("tt_device", new_fn_type.inputs.data, new_fn_type.outputs.data)

    self.target_ops=[new_func]

    call_fn=func.CallOp.create(properties={"callee": builtin.SymbolRefAttr("tt_device")}, operands=arg_ssa, result_types=[])
    op.parent.insert_ops_before(memref_dim_ops+[call_fn], op)
    rewriter.insert_op(new_func_signature, InsertPoint.at_start(self.module.body.block))

    op.parent.detach_op(op)

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

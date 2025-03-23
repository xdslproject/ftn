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

class RewriteTarget(RewritePattern):
  def __init__(self):
    self.target_ops=[]

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: omp.TargetOp, rewriter: PatternRewriter, /):
    arg_types=[]
    arg_ssa=[]

    idx=0

    locations={}

    # Grab bounds and info, then at end the terminator
    for var in op.map_vars:
      var_op=var.owner
      var_op.parent.detach_op(var_op)
      arg_types.append(var_op.var_ptr[0].type)
      arg_ssa.append(var_op.var_ptr[0])
      locations[var_op]=idx
      idx+=1
      if len(var_op.bounds) > 0:
        bound_op=var_op.bounds[0].owner
        bound_op.parent.detach_op(bound_op)
        #self.target_ops+=[bound_op, var_op]
        arg_types.append(bound_op.lower[0].type)
        arg_ssa.append(bound_op.lower[0])
        locations[bound_op]=idx
        idx+=1
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
        new_bounds_op=omp.BoundsOp.build(operands=[[new_block.args[locations[bound_op]]], [], [], [], []],
                      properties={"stride_in_bytes": bound_op.stride_in_bytes},
                      result_types=res_types)

        new_block.add_op(new_bounds_op)        
        map_bounds=[new_bounds_op.results[0]]

      res_types=[]
      for res in var_op.results: res_types.append(res.type)
      mapinfo_op=omp.MapInfoOp.build(operands=[[new_block.args[locations[var_op]]], [], map_bounds],
                      properties={"map_type": var_op.map_type, "var_name": var_op.var_name, "var_type": var_op.var_type},
                      result_types=res_types)
      new_mapinfo_ssa.append(mapinfo_op.results[0])

      new_block.add_op(mapinfo_op)
              
    reg=op.region
    op.detach_region(reg)

    new_omp_target_op=omp.TargetOp.build(operands=[[],[],[], new_mapinfo_ssa], regions=[reg])        
    new_block.add_op(new_omp_target_op)
    new_block.add_op(func.Return())
    
    new_fn_type=builtin.FunctionType.from_lists(arg_types, [])

    body = Region()
    body.add_block(new_block)
    
    new_func=func.FuncOp("tt_device", new_fn_type, body)
    
    self.target_ops=[new_func]
    
    call_fn=func.Call.create(properties={"callee": builtin.SymbolRefAttr("tt_device")}, operands=arg_ssa, result_types=[])
    op.parent.insert_op_before(call_fn, op)
    
    op.parent.detach_op(op)

@dataclass(frozen=True)
class ExtractTarget(ModulePass):
  """
  This is the entry point for the transformation pass which will then apply the rewriter
  """
  name = 'extract-target'

  def apply(self, ctx: MLContext, module: builtin.ModuleOp):
    rw_target= RewriteTarget()
    walker = PatternRewriteWalker(GreedyRewritePatternApplier([
              rw_target,
    ]), apply_recursively=False, walk_reverse=True)

    walker.rewrite_module(module)

    containing_mod=builtin.ModuleOp([])
    module.regions[0].move_blocks(containing_mod.regions[0])

    new_module=builtin.ModuleOp(rw_target.target_ops, {"target": builtin.StringAttr("tt_device")})

    block = Block()
    block.add_ops([new_module, containing_mod])
    module.regions[0].add_block(block)

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, SSAValue, Region, Block
from xdsl.dialects.builtin import IntegerAttr, StringAttr, ArrayAttr
from xdsl.pattern_rewriter import (GreedyRewritePatternApplier,
                                   PatternRewriter, PatternRewriteWalker,
                                   RewritePattern, op_type_rewrite_pattern)

from ftn.dialects import ftn_dag
from util.visitor import Visitor
    
class ApplyProcedureUnBinder(RewritePattern):
  @op_type_rewrite_pattern
  def match_and_rewrite(self, call_expr: ftn_dag.CallExpr, rewriter: PatternRewriter):    
    for op in call_expr.bound_function_instance.blocks[0].ops:
      op.detach()
      
    call_expr.attributes["bound_variables"]=ArrayAttr([])    
    
def unbind_procedures(ctx: ftn_dag.MLContext, module: ModuleOp) -> ModuleOp:
  walker = PatternRewriteWalker(GreedyRewritePatternApplier([
        ApplyProcedureUnBinder(),]), apply_recursively=False)
  walker.rewrite_module(module)
  return module
  

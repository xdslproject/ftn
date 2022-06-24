from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, SSAValue, Region, Block
from xdsl.dialects.builtin import IntegerAttr, StringAttr, ArrayAttr
from xdsl.pattern_rewriter import (GreedyRewritePatternApplier,
                                   PatternRewriter, PatternRewriteWalker,
                                   RewritePattern, op_type_rewrite_pattern)

from ftn.dialects import ftn_dag
from util.visitor import Visitor

class CollectAllRoutines(Visitor):
  all_routines={}
  
  def visit_routine(self, routine: ftn_dag.Routine):
    self.all_routines[routine.routine_name.data]=routine
    
class ReplaceDummyWithConcrete(RewritePattern):
  def __init__(self, dummy_to_concrete):
    self.dummy_to_concrete=dummy_to_concrete
    
  @op_type_rewrite_pattern
  def match_and_rewrite(self, expr_name: ftn_dag.ExprName, rewriter: PatternRewriter):
    var_name=expr_name.var.var_name.data    
    if var_name in self.dummy_to_concrete:
      rewriter.replace_op(expr_name, self.dummy_to_concrete[var_name].clone())
    
class ApplyProcedureBinder(RewritePattern):
  def __init__(self, routines):
    self.routines=routines
    
  def get_concrete_var_name(self, op):
    if isinstance(op, ftn_dag.ExprName):
      return op.var.var_name
    elif isinstance(op, ftn_dag.ArrayAccess):
      return get_concrete_var_name(op.var.blocks[0].ops[0])
    elif isinstance(op, ftn_dag.MemberAccess):
      return op.var.var_name
    else:
      return None
  
  @op_type_rewrite_pattern
  def match_and_rewrite(self, call_expr: ftn_dag.CallExpr, rewriter: PatternRewriter):
    procedure_bodies=[]  
    concrete_var_names=[] 
    if call_expr.func.data in self.routines:
      for op in self.routines[call_expr.func.data].routine_body.blocks[0].ops:
        procedure_bodies.append(op.clone())
        
      call_to_procedure_dummy_mapping={}      
      for concrete, dummy in zip(call_expr.args.blocks[0].ops, self.routines[call_expr.func.data].args.data):
        concrete_var_name=self.get_concrete_var_name(concrete)
        if concrete_var_name is not None: concrete_var_names.append(concrete_var_name)         
        call_to_procedure_dummy_mapping[dummy.var_name.data]=concrete      
      
      walker = PatternRewriteWalker(GreedyRewritePatternApplier([
        ReplaceDummyWithConcrete(call_to_procedure_dummy_mapping),]), apply_recursively=False)
      for proc in procedure_bodies:
        walker.rewrite_module(proc)
      
      rewriter.insert_op_at_pos(procedure_bodies, call_expr.bound_function_instance.blocks[0], 0)
      call_expr.attributes["bound_variables"]=ArrayAttr(concrete_var_names)    
    
def bind_procedures(ctx: ftn_dag.MLContext, module: ModuleOp) -> ModuleOp:
  visitor = CollectAllRoutines()
  visitor.traverse(module)
  
  walker = PatternRewriteWalker(GreedyRewritePatternApplier([
        ApplyProcedureBinder(visitor.all_routines),]), apply_recursively=False)
  walker.rewrite_module(module)
  return module
  
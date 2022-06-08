from __future__ import annotations
from xdsl.dialects.builtin import StringAttr, ModuleOp, IntegerAttr, ArrayAttr
from xdsl.ir import Operation, Attribute, ParametrizedAttribute, Region, Block, SSAValue, MLContext

from util.list_ops import flatten
from psy.dialects import psy_dag, psy_ast, psy_type
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


@dataclass
class SSAValueCtx:
    """
    Context that relates identifiers from the AST to SSA values used in the flat representation.
    """
    dictionary: Dict[str, SSAValue] = field(default_factory=dict)
    parent_scope: Optional[SSAValueCtx] = None

    def __getitem__(self, identifier: str) -> Optional[SSAValue]:
        """Check if the given identifier is in the current scope, or a parent scope"""
        ssa_value = self.dictionary.get(identifier, None)        
        if ssa_value:
            return ssa_value
        elif self.parent_scope:
            return self.parent_scope[identifier]
        else:
            return None

    def __setitem__(self, identifier: str, ssa_value: SSAValue):
        """Relate the given identifier and SSA value in the current scope"""
        if identifier in self.dictionary:
            raise Exception()
        else:
            self.dictionary[identifier] = ssa_value


def psy_ast_to_psy_dag(ctx: MLContext, input_module: ModuleOp):
    input_program = input_module.ops[0]
    assert isinstance(input_program, psy_ast.FileContainer) or isinstance(input_program, psy_ast.Container)
    res_module = translate_program(input_program)    
    # Now we need to gather the containers and inform the top level routine about these so it can generate the use
    applyModuleUseToFloatingRegions(res_module, collectContainerNames(res_module))
    res_module.regions[0].move_blocks(input_module.regions[0])
    # Create program entry point
    #wrap_top_levelcall_from_main(ctx, input_module)
    
def collectContainerNames(module: ModuleOp):
  container_names=[]
  for op in module.ops:
    if isinstance(op, psy_dag.Container):
      container_names.append(op.container_name)
  return container_names

def applyModuleUseToFloatingRegions(module: ModuleOp, container_names):
  for op in module.ops:
    if isinstance(op, psy_dag.Routine):
      op.attributes["required_module_uses"]=ArrayAttr(container_names)
    
def wrap_top_levelcall_from_main(ctx: MLContext, module: ModuleOp):
  found_routine=find_floating_region(module)
  assert found_routine is not None
  
  body = Region()
  block = Block()
  
  callexpr = psy_dag.CallExpr.get(found_routine.routine_name, [], True)
  block.add_ops([callexpr])
  body.add_block(block)
  main = psy_dag.Routine.get("main", "void", [], [], body, True)  
  module.regions[0].blocks[0].add_ops([main])
    
def find_floating_region(module: ModuleOp):
  for op in module.ops:
    if isinstance(op, psy_dag.Routine):
      return op
  return None    

def translate_program(p: psy_ast.FileContainer) -> ModuleOp:
    # create an empty global context
    global_ctx = SSAValueCtx()   
    if isinstance(p, psy_ast.FileContainer):
      containers: List[psy_dag.Container] = [translate_container(global_ctx, container) for container in p.programs.blocks[0].ops]
    elif isinstance(p, psy_ast.Container):
      containers: List[psy_dag.Container] = [translate_container(global_ctx, p)]
    else:
      containers: List[psy_dag.Container] = []
            
    return ModuleOp.from_region_or_ops(containers)
  
def translate_container(ctx: SSAValueCtx, op: Operation) -> Operation:
  
  if isinstance(op, psy_ast.Container):
    imports = Region()
    imports_block = Block()
    imports_block.add_ops(translate_import_stmt(ctx, import_statement) for import_statement in op.imports.blocks[0].ops)
    imports.add_block(imports_block)
    
    routines = Region()
    routines_block = Block()
    routines_block.add_ops(translate_fun_def(ctx, routine) for routine in op.routines.blocks[0].ops)
    routines.add_block(routines_block)
    
    return psy_dag.Container.create(attributes={"container_name": op.container_name, "default_visibility": op.default_visibility,
                                                "public_routines": op.public_routines, "private_routines": op.private_routines}, regions=[imports, routines])
  elif isinstance(op, psy_ast.Routine):
    return translate_fun_def(ctx, op)
  
def translate_import_stmt(ctx: SSAValueCtx, op: Operation) -> List[Operation]:
  return psy_dag.Import.create(attributes={"import_name": op.attributes["import_name"], "specific_procedures": op.attributes["specific_procedures"]})
  
def translate_def_or_stmt(ctx: SSAValueCtx, op: Operation) -> List[Operation]:
    """
    Translate an operation that can either be a definition or statement
    """
    # first try to translate op as a definition:
    #   if op is a definition this will return a list of translated Operations
    ops = try_translate_def(ctx, op)
    if ops is not None:
        return ops
    # op has not been a definition, try to translate op as a statement:
    #   if op is a statement this will return a list of translated Operations
    ops = try_translate_stmt(ctx, op)
    if ops is not None:
      return ops
    # operation must have been translated by now
    raise Exception(f"Could not translate `{op}' as a definition or statement")


def try_translate_def(ctx: SSAValueCtx,
                      op: Operation) -> Optional[List[Operation]]:
    """
    Tries to translate op as a definition.
    Returns a list of the translated Operations if op is a definition, returns None otherwise.
    """
    if isinstance(op, psy_ast.Routine):
        return [translate_fun_def(ctx, op)]
    elif isinstance(op, psy_ast.VarDef):
        return translate_var_def(ctx, op)
    else:
        return None


def translate_def(ctx: SSAValueCtx, op: Operation) -> List[Operation]:
    """
    Translates op as a definition.
    Returns a list of the translated Operations if op is a definition, fails otherwise.
    """
    ops = try_translate_def(ctx, op)
    if ops is None:
        raise Exception(f"Could not translate `{op}' as a definition")
    else:
        return ops  
  
def translate_fun_def(ctx: SSAValueCtx,
                      routine_def: psy_ast.Routine) -> Operation:    
    routine_name = routine_def.attributes["routine_name"]  
    
    #return_type = try_translate_type(fun_def.return_type.blocks[0].ops[0])
    #if return_type is None:
    #    return_type = choco_type.none_type

    # create a new nested scope and
    # relate parameter identifiers with SSA values of block arguments
    body = Region()
    
    c = SSAValueCtx(dictionary={}, parent_scope=ctx)
    
    imports = []
    for import_statement in routine_def.imports.blocks[0].ops:
      imports.append(translate_import_stmt(ctx, import_statement))

    declarations=[]
    for op in routine_def.local_var_declarations.blocks[0].ops:
      declarations.append(translate_def_or_stmt(c, op))
    
    exec_statements=[]
    for op in routine_def.routine_body.blocks[0].ops:
      exec_statements.append(translate_def_or_stmt(c, op))
      
    # Do arguments last, as these are already created in the declarations lookup
    arguments=[]
    for op in routine_def.args.data:
      token=c[op.data]
      arguments.append(token)
          
    return psy_dag.Routine.get(routine_name, "void", arguments, imports, declarations, exec_statements)
    
def try_translate_type(ctx: SSAValueCtx, op: Operation) -> Optional[Attribute]:
    """Tries to translate op as a type, returns None otherwise."""    
    if isinstance(op, psy_ast.IntegerType):               
      return psy_type.int_type
    elif isinstance(op, psy_ast.Float32Type):              
      return psy_type.float_type
    elif isinstance(op, psy_ast.DerivedType):
      return psy_type.DerivedType([try_translate_type(op.type)])
    elif isinstance(op, psy_ast.ArrayType):
      transformed_shape=[]
      for member in op.shape.data:
        if isinstance(member, psy_ast.AnonymousAttr):
          transformed_shape.append(psy_dag.AnonymousAttr())
        else:
          transformed_shape.append(try_translate_expr(ctx, member))
      return psy_type.ArrayType([ArrayAttr(transformed_shape), try_translate_type(ctx, op.element_type)])

    return None


def translate_var_def(ctx: SSAValueCtx,
                      var_def: psy_ast.VarDef) -> List[Operation]:
   
    var_name = var_def.attributes["var_name"]
    assert isinstance(var_name, StringAttr)
    type = try_translate_type(ctx, var_def.attributes["type"])    
    
    tkn=psy_dag.Token([var_name, type])

    vardef=psy_dag.VarDef.create(attributes={"var": tkn, "is_proc_argument": var_def.attributes["is_proc_argument"], 
                                             "is_constant": var_def.attributes["is_constant"], 
                                             "intent": var_def.attributes["intent"]}, result_types=[type])    

    # relate variable identifier and SSA value by adding it into the current context    
    ctx[var_name.data] = tkn 

    return vardef

def try_translate_expr(
        ctx: SSAValueCtx,
        op: Operation) -> Optional[Tuple[List[Operation], SSAValue]]:
    """
    Tries to translate op as an expression.
    If op is an expression, returns a list of the translated Operations
    and the ssa value representing the translated expression.
    Returns None otherwise.
    """    
    if isinstance(op, psy_ast.Literal):        
        return translate_literal(op)
    if isinstance(op, psy_ast.ExprName):
        if ctx[op.id.data] is None:
          print("Missing "+op.id.data)
        return psy_dag.ExprName.create(attributes={"id": op.attributes["id"], "var": ctx[op.id.data]})
    if isinstance(op, psy_ast.BinaryExpr):        
        return translate_binary_expr(ctx, op)
    if isinstance(op, psy_ast.CallExpr):
        print("No call expression here!")
        return translate_call_expr(ctx, op)
    if isinstance(op, psy_ast.MemberAccess):
        return translate_member_access_expr(ctx, op)
    if isinstance(op, psy_ast.ArrayAccess):
        return translate_array_access_expr(ctx, op)

    assert False, "Unknown Expression "+str(op)


def translate_expr(ctx: SSAValueCtx,
                   op: Operation) -> Tuple[List[Operation], SSAValue]:
    """
    Translates op as an expression.
    If op is an expression, returns a list of the translated Operations
    and the ssa value representing the translated expression.
    Fails otherwise.
    """    
    res = try_translate_expr(ctx, op)
    if res is None:
        raise Exception(f"Could not translate `{op}' as an expression")
    else:
        ops = res
        return ops
      
def translate_array_access_expr(ctx: SSAValueCtx, op: psy_ast.ArrayAccess) -> List[Operation]:
  index_accessors=[]
  for entry in op.accessors.blocks[0].ops:
    index_accessors.append(translate_expr(ctx, entry))
  return psy_dag.ArrayAccess.get(ctx[op.var_name.data], index_accessors)
      
def translate_member_access_expr(ctx: SSAValueCtx, op: psy_ast.MemberAccess, first_in_chain=True) -> List[Operation]:  
  entry = op.member.blocks[0].ops[0]  
  if isinstance(entry, psy_ast.MemberAccess):
    members=translate_member_access_expr(ctx, entry, False)
  elif isinstance(entry, psy_ast.ExprName):
    members=[entry.id.data]
  else:
    members=[]
  if (first_in_chain):    
    return psy_dag.MemberAccess.get(ctx[op.id.data], [StringAttr(member) for member in members])
  else:    
    members.insert(0, op.attributes["id"].data)
    return members
      
def translate_literal(op: psy_ast.Literal) -> Operation:
    value = op.attributes["value"]

    if isinstance(value, IntegerAttr):
        return psy_dag.Literal.create(attributes={"value": value},
                                         result_types=[psy_type.int_type])  
        
    if isinstance(value, psy_ast.FloatAttr):
        attr = psy_dag.FloatAttr(value.data)  
        return psy_dag.Literal.create(attributes={"value": attr},
                                         result_types=[psy_type.float_type])  

    if isinstance(value, StringAttr):
        return psy_dag.Literal.create(attributes={"value": value},
                                         result_types=[psy_type.str_type])

    raise Exception(f"Could not translate `{op}' as a literal")
  
def translate_binary_expr(
        ctx: SSAValueCtx,
        binary_expr: psy_ast.BinaryExpr) -> List[Operation]:
    lhs = translate_expr(ctx, binary_expr.lhs.blocks[0].ops[0])
    rhs = translate_expr(ctx, binary_expr.rhs.blocks[0].ops[0])

    if binary_expr.op.data in ['!=', '==', '<', '<=', '>', '>=', 'is']:
        result_type = psy_type.bool_type

    attr = binary_expr.op
    assert isinstance(attr, Attribute)      

    return psy_dag.BinaryExpr.get(attr,lhs, rhs)
    
# This function could be avoided if we could remove the result type via rewriting in a separate pass
def translate_call_expr_stmt(ctx: SSAValueCtx,
                             call_expr: psy_ast.CallExpr, statement) -> List[Operation]:
    
    result_list=[]
    for arg in call_expr.args.blocks[0].ops:
      result_list.append(translate_expr(ctx, arg))

    name = call_expr.attributes["func"]    
    return psy_dag.CallExpr.get(name.data, result_list, statement)
  
def try_translate_stmt(ctx: SSAValueCtx,
                       op: Operation) -> Optional[List[Operation]]:
    """
    Tries to translate op as a statement.
    If op is an expression, returns a list of the translated Operations.
    Returns None otherwise.
    """
    if isinstance(op, psy_ast.Assign):
        return translate_assign(ctx, op)
    if isinstance(op, psy_ast.CallExpr):
        return translate_call_expr_stmt(ctx, op, True)
    if isinstance(op, psy_ast.If):
        return translate_if(ctx, op)
    if isinstance(op, psy_ast.Do):
        return translate_do(ctx, op)

    res = try_translate_expr(ctx, op)
    if res is None:
        return None
    else:
        return res[0]


def translate_stmt(ctx: SSAValueCtx, op: Operation) -> List[Operation]:
    """
    Translates op as a statement.
    If op is an expression, returns a list of the translated Operations.
    Fails otherwise.
    """
    ops = try_translate_stmt(ctx, op)
    if ops is None:
        raise Exception(f"Could not translate `{op}' as a statement")
    else:
        return ops
      
def translate_if(ctx: SSAValueCtx, if_stmt: psy_ast.If) -> List[Operation]:
    cond = translate_expr(ctx, if_stmt.cond.blocks[0].ops[0])

    ops: List[Operation] = []
    for op in if_stmt.then.blocks[0].ops:
        stmt_ops = translate_stmt(ctx, op)
        ops.append(stmt_ops)
    then = Region.from_operation_list(ops)

    ops: List[Operation] = []
    for op in if_stmt.orelse.blocks[0].ops:
        stmt_ops = translate_stmt(ctx, op)
        ops.append(stmt_ops)
    orelse = Region.from_operation_list(ops)
    
    return psy_dag.If.get(cond, then, orelse)
  
def translate_do(ctx: SSAValueCtx,
                  for_stmt: psy_ast.Do) -> List[Operation]:

    ops: List[Operation] = []
    for op in for_stmt.body.blocks[0].ops:
        stmt_ops = translate_stmt(ctx, op)
        ops.append(stmt_ops)
    body = Region.from_operation_list(ops)

    iterator = ctx[for_stmt.iter_name.data]
    return psy_dag.Do.get(for_stmt.iter_name.data, translate_expr(ctx, for_stmt.start.blocks[0].ops[0]), translate_expr(ctx, for_stmt.stop.blocks[0].ops[0]), translate_expr(ctx, for_stmt.step.blocks[0].ops[0]), body)  


def translate_assign(ctx: SSAValueCtx,
                     assign: psy_ast.Assign) -> List[Operation]:
    lhs=translate_expr(ctx, assign.lhs.op)
    rhs=translate_expr(ctx, assign.rhs.op)
    return psy_dag.Assign.get(lhs, rhs) 
        
from __future__ import annotations
from xdsl.dialects.builtin import StringAttr, ModuleOp, IntegerAttr
from xdsl.ir import Operation, Attribute, ParametrizedAttribute, Region, Block, SSAValue, MLContext

from util.list_ops import flatten
from psy.dialects import psy_flat, psy_ast, psy_type
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


def psy_ast_to_psy_flat(ctx: MLContext, input_module: ModuleOp):
    input_program = input_module.ops[0]
    assert isinstance(input_program, psy_ast.FileContainer)
    res_module = translate_program(input_program)
    res_module.regions[0].move_blocks(input_module.regions[0])
    wrap_top_levelcall_from_main(ctx, input_module)


def wrap_top_levelcall_from_main(ctx: MLContext, module: ModuleOp):
  found_routine=find_floating_region(module)
  assert found_routine is not None
  
  body = Region()
  block = Block()
  
  callexpr = psy_flat.CallExpr.create(attributes={"func_name": found_routine.routine_name}, operands=[])
  block.add_ops([callexpr])
  body.add_block(block)
  main = psy_flat.Routine.create(attributes={
          "routine_name": StringAttr.from_str("_main"),
          "return_type": psy_type.none_type
          }, regions=[body])
  module.regions[0].blocks[0].add_ops([main])
    
def find_floating_region(module: ModuleOp):
  for op in module.ops:
    if isinstance(op, psy_flat.Routine):
      return op
  return None

def translate_program(p: psy_ast.FileContainer) -> ModuleOp:
    # create an empty global context
    global_ctx = SSAValueCtx()   
    containers: List[List[Operation]] = [
      translate_container(global_ctx, container) for container in p.programs.blocks[0].ops
    ]
            
    return ModuleOp.from_region_or_ops(containers)
  
def translate_container(ctx: SSAValueCtx, op: Operation) -> Operation:
  
  if isinstance(op, psy_ast.Container):    
    body = Region()
    block = Block()
    block.add_ops(translate_fun_def(ctx, routine.blocks[0].ops[0]) for routine in op.regions)

    body.add_block(block)
    return psy_flat.Container.create(attributes={"container_name": op.container_name}, regions=[body])
  elif isinstance(op, psy_ast.Routine):
    return translate_fun_def(ctx, op)

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

    def get_param(op: Operation) -> Tuple[str, Attribute]:
        assert isinstance(op, psy_ast.TypedVar)
        var_name = op.attributes.get('var_name')
        assert isinstance(var_name, StringAttr)
        name = var_name.data
        type_name = op.regions[0].blocks[0].ops[0]
        type = try_translate_type(type_name)
        assert type is not None
        return name, type

    params = [get_param(op) for op in routine_def.params.blocks[0].ops]
    param_names: List[str] = [p[0] for p in params]
    param_types: List[Attribute] = [p[1] for p in params]
    #return_type = try_translate_type(fun_def.return_type.blocks[0].ops[0])
    #if return_type is None:
    #    return_type = choco_type.none_type

    body = Region()
    block = Block.from_arg_types(param_types)
    # create a new nested scope and
    # relate parameter identifiers with SSA values of block arguments
    c = SSAValueCtx(dictionary=dict(zip(param_names, block.args)),
                    parent_scope=ctx)
    # use the nested scope when translate the body of the function
    block.add_ops(
        flatten([
            translate_def_or_stmt(c, op)
            for op in routine_def.local_var_declarations.blocks[0].ops
       ]))
    
    block.add_ops(
        flatten([
            translate_def_or_stmt(c, op)
            for op in routine_def.routine_body.blocks[0].ops
        ]))
    body.add_block(block)

    return psy_flat.Routine.create(attributes={
        "routine_name": routine_name,
        "return_type": routine_def.return_type
    }, regions=[body])


def try_translate_type(op: Operation) -> Optional[Attribute]:
    """Tries to translate op as a type, returns None otherwise."""    
    if isinstance(op, psy_ast.IntegerType):               
      return psy_type.int_type
    elif isinstance(op, psy_ast.Float32Type):              
      return psy_type.float_type

    return None


def translate_var_def(ctx: SSAValueCtx,
                      var_def: psy_ast.VarDef) -> List[Operation]:
   
    var_name = var_def.attributes["var_name"]
    assert isinstance(var_name, StringAttr)
    type = try_translate_type(var_def.attributes["type"])

    flat_var_def = psy_flat.VarDef.create(attributes={"var_name": var_name, "type":type}, operands=[], result_types=[type])

    # relate variable identifier and SSA value by adding it into the current context
    ctx[var_name.data] = flat_var_def.results[0]

    return [flat_var_def]


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
        op = translate_literal(op)
        return [op], op.results[0]
    if isinstance(op, psy_ast.ExprName):
        ssa_value = ctx[op.id.data]
        assert isinstance(ssa_value, SSAValue)
        return [], ssa_value    
    if isinstance(op, psy_ast.BinaryExpr):
        return translate_binary_expr(ctx, op)
    if isinstance(op, psy_ast.CallExpr):
        print("No call expression here!")
        return translate_call_expr(ctx, op)

    assert False, "Unknown Expression"


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
        ops, ssa_value = res
        return ops, ssa_value


def translate_literal(op: psy_ast.Literal) -> Operation:
    value = op.attributes["value"]

    if isinstance(value, IntegerAttr):
        return psy_flat.Literal.create(attributes={"value": value},
                                         result_types=[psy_type.int_type])  
        
    if isinstance(value, psy_ast.FloatAttr):
        attr = psy_flat.FloatAttr(value.data)  
        return psy_flat.Literal.create(attributes={"value": attr},
                                         result_types=[psy_type.float_type])  

    if isinstance(value, StringAttr):
        return psy_flat.Literal.create(attributes={"value": value},
                                         result_types=[psy_type.str_type])

    raise Exception(f"Could not translate `{op}' as a literal")

def translate_binary_expr(
        ctx: SSAValueCtx,
        binary_expr: psy_ast.BinaryExpr) -> Tuple[List[Operation], SSAValue]:
    lhs, lhs_ssa_value = translate_expr(ctx, binary_expr.lhs.blocks[0].ops[0])
    rhs, rhs_ssa_value = translate_expr(ctx, binary_expr.rhs.blocks[0].ops[0])
    result_type = rhs_ssa_value.typ
    if binary_expr.op.data != "is":
        assert lhs_ssa_value.typ == rhs_ssa_value.typ

    if binary_expr.op.data in ['!=', '==', '<', '<=', '>', '>=', 'is']:
        result_type = psy_type.bool_type

    attr = binary_expr.op
    assert isinstance(attr, Attribute)

    # Special case when the binary operation has a different execution order
    #if binary_expr.op.data in ['or', 'and']:
    #    lhs.append(Yield.get(lhs_ssa_value))
    #    rhs.append(Yield.get(rhs_ssa_value))
    #    flat_binary_expr = choco_flat.EffectfulBinaryExpr.build(
    #        attributes={"op": attr},
    #        regions=[lhs, rhs],
    #        result_types=[result_type])
    #    return [flat_binary_expr], flat_binary_expr.results[0]

    flat_binary_expr = psy_flat.BinaryExpr.create(
        attributes={"op": attr},
        operands=[lhs_ssa_value, rhs_ssa_value],
        result_types=[result_type])
    return lhs + rhs + [flat_binary_expr], flat_binary_expr.results[0]

#def translate_call_expr(
#        ctx: SSAValueCtx,
#        call_expr: psy_ast.CallExpr) -> Tuple[List[Operation], SSAValue]:
#    ops: List[Operation] = []
#    args: List[SSAValue] = []

#    for arg in call_expr.args.blocks[0].ops:
#        op, arg = translate_expr(ctx, arg)
#        ops += op
#        args.append(arg)

#    name = call_expr.attributes["func"]
#    call = psy_flat.CallExpr.create(
#        attributes={"func_name": name},
#        operands=args,
#        result_types=[call_expr.attributes["type"]])
#    ops.append(call)
#    return ops, call.results[0]


# This function could be avoided if we could remove the result type via rewriting in a separate pass
def translate_call_expr_stmt(ctx: SSAValueCtx,
                             call_expr: choco_ast.CallExpr) -> List[Operation]:
    ops: List[Operation] = []
    args: List[SSAValue] = []

    for arg in call_expr.args.blocks[0].ops:
        op, arg = translate_expr(ctx, arg)
        ops += op
        args.append(arg)

    name = call_expr.attributes["func"]
    call = psy_flat.CallExpr.create(attributes={"func_name": name}, operands=args)
    ops.append(call)
    return ops

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
        return translate_call_expr_stmt(ctx, op)
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
    cond, cond_name = translate_expr(ctx, if_stmt.cond.blocks[0].ops[0])

    ops: List[Operation] = []
    for op in if_stmt.then.blocks[0].ops:
        stmt_ops = translate_stmt(ctx, op)
        ops += stmt_ops
    then = Region.from_operation_list(ops)

    ops: List[Operation] = []
    for op in if_stmt.orelse.blocks[0].ops:
        stmt_ops = translate_stmt(ctx, op)
        ops += stmt_ops
    orelse = Region.from_operation_list(ops)

    new_op = psy_flat.If.build(operands=[cond_name], regions=[then, orelse])
    return cond + [new_op]
  
def translate_do(ctx: SSAValueCtx,
                  for_stmt: psy_ast.Do) -> List[Operation]:
    start, start_name = translate_expr(ctx, for_stmt.start.blocks[0].ops[0])
    stop, stop_name = translate_expr(ctx, for_stmt.stop.blocks[0].ops[0])
    step, step_name = translate_expr(ctx, for_stmt.step.blocks[0].ops[0])

    ops: List[Operation] = []
    for op in for_stmt.body.blocks[0].ops:
        stmt_ops = translate_stmt(ctx, op)
        ops += stmt_ops
    body = Region.from_operation_list(ops)

    iterator = ctx[for_stmt.iter_name.data]
    new_op = psy_flat.Do.build(operands=[iterator, start_name, stop_name, step_name],
                                  regions=[body])
    return start + stop + step + [new_op]  


def split_multi_assign(
        assign: psy_ast.Assign) -> Tuple[List[Operation], Operation]:
    """Get the list of targets of a multi assign, as well as the expression value."""
    if isinstance(assign.rhs.op, psy_ast.Assign):
        targets, value = split_multi_assign(assign.rhs.op)
        return [assign.target.op] + targets, value
    return [assign.lhs.op], assign.rhs.op


def translate_assign(ctx: SSAValueCtx,
                     assign: psy_ast.Assign) -> List[Operation]:
    targets, value = split_multi_assign(assign)
    value_flat, value_var = translate_expr(ctx, value)

    translated_targets = [translate_expr(ctx, target) for target in targets]
    targets_flat = [
        target_op for target in translated_targets for target_op in target[0]
    ]
    targets_var = [target[1] for target in translated_targets]

    assigns: List[Operation] = [
        psy_flat.Assign.build(operands=[target_var, value_var])
        for target_var in targets_var
    ]
    return value_flat + targets_flat + assigns

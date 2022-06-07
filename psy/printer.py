from psy.dialects import *
from xdsl.dialects.builtin import StringAttr, IntegerAttr
import sys

incr=0

def print_op(op, stream=sys.stdout):
    global incr
    if isinstance(op, psy_dag.FileContainer):
      pass
    elif isinstance(op, psy_dag.Container):
      print_container(op)
    elif isinstance(op, psy_dag.Routine):      
      print_routine(op)
    elif isinstance(op, psy_dag.VarDef):
      print_vardef(op)
    elif isinstance(op, psy_dag.CallExpr):
      print_callexpr(op)
    elif isinstance(op, psy_dag.Literal):
      print_literal(op)
    elif isinstance(op, psy_dag.BinaryExpr):      
      print_op(op.lhs.blocks[0].ops[0])
      print(op.attributes["op"].data, end="")
      print_op(op.rhs.blocks[0].ops[0])
    elif isinstance(op, psy_dag.Assign):
      print_assign(op)
    elif isinstance(op, psy_dag.ExprName):          
      print(op.var.var_name.data, end="")          
    elif isinstance(op, psy_dag.If):
      print_if(op)
    elif isinstance(op, psy_dag.Do):
      print_do(op)
    elif isinstance(op, psy_dag.Import):
      print_import(op)
    elif isinstance(op, psy_dag.MemberAccess):
      print_memberaccess(op)
    else:
        raise Exception(f"Trying to print unknown operation '{op.name}'")
      
def print_callexpr(op):
  if op.isstatement.data: print_indent()
  print(f"call {op.func.data}(", end='')
  needs_comma=False
  for arg in op.args.blocks[0].ops:
    if (needs_comma): print(", ", end="")
    needs_comma=True
    print_op(arg)
  print(")", end="")
  if op.isstatement.data: print("") 
      
def print_memberaccess(op):
  print(f"{op.var.var_name.data}", end="")
  for member in op.fields.data:
    print(f"%{member.data}", end="")
      
def print_import(op):
  print(f"use {op.import_name.data}", end="")
  if len(op.specific_procedures.data) > 1:
    print(", only : ", end="")
    needs_comma=False
    for proc in op.specific_procedures.data:
      if (needs_comma): print(", ", end="")
      needs_comma=True
      print(proc.data, end="")      
  print("") # force a newline
      
def print_vardef(op):
  print_indent()
  if isinstance(op.var.type, psy_type.DerivedType):
    type_str=f"type({op.var.type.parameters[0].data})"
  else:
    type_str=f"{op.var.type.parameters[0].data}"  
  if op.is_proc_argument.data:    
    type_str+=f", intent({op.intent.data})"
  print(f"{type_str} :: {op.var.var_name.data}")   
      
def print_container(op):
  global incr
  print_indent()
  print(f"module {op.container_name.data}")
  incr+=2
  for import_stmt in op.imports.blocks[0].ops:
    print_indent()
    print_op(import_stmt)
  print_indent()
  print("implicit none")
  print_indent()
  print("public\n\ncontains")      
  for block in op.routines.blocks:
    print_op(block.ops[0])        
  incr-=2
  print_indent()
  print(f"end module {op.container_name.data}\n")
      
def print_literal(op):
  global incr
  literal_val=op.attributes["value"]
  if isinstance(literal_val, IntegerAttr):
    print(op.attributes["value"].parameters[0].data, end="")
  elif isinstance(literal_val, psy_dag.FloatAttr):
    print(op.attributes["value"].data, end="")      
      
def print_assign(op):
  global incr
  print_indent()      
  print_op(op.lhs.blocks[0].ops[0])
  print("=", end="")
  print_op(op.rhs.blocks[0].ops[0])
  print("") # For a new line      
      
def print_if(op):
  global incr
  print_indent()
  print("if (", end="")
  print_op(op.cond.blocks[0].ops[0])
  print(") then")      
  incr+=2
  print_op(op.then.blocks[0].ops[0])
  incr-=2      
  if len(op.orelse.blocks[0].ops) > 0:
    print_indent()
    print("else")
    incr+=2
    print_op(op.orelse.blocks[0].ops[0])
    incr-=2
  print_indent()
  print("end if")

def print_do(op):
  global incr
  print_indent()
  print(f"do {op.iter_name.data}=", end="")
  print_op(op.start.blocks[0].ops[0])
  print(", ", end="")
  print_op(op.stop.blocks[0].ops[0])
  print(", ", end="")
  print_op(op.step.blocks[0].ops[0])
  print("")
  incr+=2
  print_op(op.body.blocks[0].ops[0])
  incr-=2
  print_indent()
  print("end do")
    
def print_routine(op):
  global incr
  print("")
  print_indent()
  if op.program_entry_point.data:
    print(f"program {op.routine_name.data}")
  else:
    print(f"subroutine {op.routine_name.data}(", end="")
    needs_comma=False
    for arg in op.args.data:
      if (needs_comma): print(", ", end="")
      needs_comma=True
      print(arg.var_name.data, end="")
    print(")")
  incr+=2
  for import_stmt in op.imports.blocks[0].ops:
    print_indent()
    print_op(import_stmt)
  for block in op.local_var_declarations.blocks[0].ops:
    print_op(block)
  if len(op.local_var_declarations.blocks[0].ops) > 0: print("")
  for block in op.routine_body.blocks[0].ops:
    print_op(block)
  incr-=2
  print_indent()
  if op.program_entry_point.data:
    print(f"end program {op.routine_name.data}")
  else:
    print(f"end subroutine {op.routine_name.data}")
    
def print_indent():
  print(" "*incr, end="")


def print_program(instructions, stream=sys.stdout):    
  for op in instructions:
    print_op(op, stream=stream)
    
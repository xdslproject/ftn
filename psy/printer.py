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
      print_indent()
      print(f"{op.type.parameters[0].data} :: {op.var_name.data}")      
    elif isinstance(op, psy_dag.CallExpr):
      if op.isstatement.data: print_indent()
      print(f"call {op.func.data}()", end='')
      if op.isstatement.data: print("")
    elif isinstance(op, psy_dag.Literal):
      print_literal(op)
    elif isinstance(op, psy_dag.BinaryExpr):      
      print_op(op.lhs.blocks[0].ops[0])
      print(op.attributes["op"].data, end="")
      print_op(op.rhs.blocks[0].ops[0])
    elif isinstance(op, psy_dag.Assign):
      print_assign(op)
    elif isinstance(op, psy_dag.ExprName):
      if len(op.operands) > 0:        
        print(f"{op.operands[0].name.data}", end="")
      else:
        print(f"{op.id.data}", end="")         
    elif isinstance(op, psy_dag.If):
      print_if(op)
    elif isinstance(op, psy_dag.Do):
      print_do(op)
    else:
        raise Exception(f"Trying to print unknown operation '{op.name}'")
      
def print_container(op):
  global incr
  print_indent()
  print(f"module {op.container_name.data}")
  incr+=2
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
    print(f"subroutine {op.routine_name.data}()")
  incr+=2
  for entry in op.attributes["required_module_uses"].data:
    print_indent()
    print("use "+entry.data)
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
    
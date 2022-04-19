from psy.dialects import *
import sys


def print_op(op, stream=sys.stdout):
    attr = op.attributes

    if isinstance(op, psy_flat.FileContainer):
      pass
    elif isinstance(op, psy_flat.Container):
       print(f"module {op.container_name.data}\nimplicit none\npublic\n\ncontains")
       for block in op.routines.blocks:
         print_op(block.ops[0])
       print(f"end module {op.container_name.data}\n")
    elif isinstance(op, psy_flat.Routine):
      print(f"subroutine {op.routine_name.data}()")
      for block in op.routine_body.blocks[0].ops:
         print_op(block)
      print(f"end subroutine {op.routine_name.data}")
    elif isinstance(op, psy_flat.VarDef):
      print(op.var_name.data)
      #print(op.type)
    elif isinstance(op, psy_flat.CallExpr):
      print(f"call {op.func_name.data}()", end='')
    elif isinstance(op, psy_flat.Literal):
      pass#print(op.attributes["value"].parameters[0].data)
    elif isinstance(op, psy_flat.BinaryExpr):
      print(attr)
      print(op.attributes["op"].data)
    elif isinstance(op, psy_flat.Assign):
      print(op.operands[0])
    else:
        raise Exception(f"Trying to print unknown operation '{op.name}'")
    

    #print("", file=stream)


def print_program(instructions, stream=sys.stdout):    
  for op in instructions:
    print_op(op, stream=stream)
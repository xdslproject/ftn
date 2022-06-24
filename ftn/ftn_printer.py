from ftn.dialects import *
from xdsl.dialects.builtin import StringAttr, IntegerAttr
import sys

class FortranPrinter():
  def __init__(self):
    self.incr=0
    
  def print_op(self, op, stream=sys.stdout):    
    if isinstance(op, ftn_dag.FileContainer):
      pass
    elif isinstance(op, ftn_dag.Container):
      self.print_container(op)
    elif isinstance(op, ftn_dag.Routine):      
      self.print_routine(op)
    elif isinstance(op, ftn_dag.VarDef):
      self.print_vardef(op)
    elif isinstance(op, ftn_dag.CallExpr):
      self.print_callexpr(op)
    elif isinstance(op, ftn_dag.Literal):
      self.print_literal(op)
    elif isinstance(op, ftn_dag.BinaryExpr):      
      self.print_op(op.lhs.blocks[0].ops[0])
      print(op.attributes["op"].data, end="")
      self.print_op(op.rhs.blocks[0].ops[0])
    elif isinstance(op, ftn_dag.Assign):
      self.print_assign(op)
    elif isinstance(op, ftn_dag.ExprName):          
      print(op.var.var_name.data, end="")          
    elif isinstance(op, ftn_dag.If):
      self.print_if(op)
    elif isinstance(op, ftn_dag.Do):
      self.print_do(op)
    elif isinstance(op, ftn_dag.Import):
      self.print_import(op)
    elif isinstance(op, ftn_dag.MemberAccess):
      self.print_memberaccess(op)
    elif isinstance(op, ftn_dag.ArrayAccess):
      self.print_arrayaccess(op)
    else:
        raise Exception(f"Trying to print unknown operation '{op.name}'")
      
  def print_arrayaccess(self, op):
    self.print_op(op.var.blocks[0].ops[0])
    print("(", end="")
    needs_comma=False
    for member in op.accessors.blocks[0].ops:
      if (needs_comma): print(", ", end="")
      needs_comma=True
      self.print_op(member)
    print(")", end="")        
      
  def print_callexpr(self, op):
    if op.isstatement.data: self.print_indent()
    print(f"call {op.func.data}(", end='')    
    for index, arg in enumerate(op.args.blocks[0].ops):
      if (index > 0): print(", ", end="")      
      self.print_op(arg)
    print(")", end="")
    if op.isstatement.data: print("") 
        
  def print_memberaccess(self, op):
    print(f"{op.var.var_name.data}", end="")
    for member in op.fields.data:
      print(f"%{member.data}", end="")
        
  def print_import(self, op):
    print(f"use {op.import_name.data}", end="")
    if len(op.specific_procedures.data) > 1:
      print(", only : ", end="")      
      for index, proc in enumerate(op.specific_procedures.data):
        if (index > 0): print(", ", end="")        
        print(proc.data, end="")      
    print("") # force a newline
        
  def print_vardef(self, op):
    self.print_indent()
    type_str=self.generate_typestring(op.var.type)
    type_str+=self.generate_vardeclaration_extra_type_info(op)
    print(f"{type_str} :: {op.var.var_name.data}")
    
  def generate_vardeclaration_extra_type_info(self, var_def):
    extra_info=""
    if var_def.is_proc_argument.data:    
      extra_info+=f", intent({var_def.intent.data})"
    return extra_info
    
  def generate_typestring(self, type):  
    if isinstance(type, ftn_type.DerivedType):
      type_str=f"type({type.parameters[0].data})"
    elif isinstance(type, ftn_type.ArrayType):
      type_str=self.generate_typestring(type.element_type)
      type_str+=", dimension("      
      for index, dim_size in enumerate(type.shape.data):
        if isinstance(dim_size, ftn_dag.AnonymousAttr):
          if (index > 0): type_str+=(",")          
          type_str+=":"
      type_str+=")"
    else:
      type_str=f"{type.parameters[0].data}"
      if (len(type.parameters[1].data) > 0):
        type_str+=f"(kind={type.parameters[1].data})"
      elif (type_str=="float" or type_str=="integer") and type.parameters[2].data != 4:
        type_str+=f"({type.parameters[2].data})"
    return type_str  
        
  def print_container(self, op):    
    self.print_indent()
    print(f"module {op.container_name.data}")
    self.incr+=2
    for import_stmt in op.imports.blocks[0].ops:
      self.print_indent()
      self.print_op(import_stmt)
    print("")
    self.print_indent()
    print("implicit none")
    self.print_indent()
    print(f"{op.default_visibility.data}\n")
    self.print_container_level_routine_visibility("public", op.public_routines.data)
    self.print_container_level_routine_visibility("private", op.private_routines.data)  
    print("contains")      
    for block in op.routines.blocks[0].ops:
      self.print_op(block)        
    self.incr-=2
    self.print_indent()
    print(f"end module {op.container_name.data}\n")
    
  def print_container_level_routine_visibility(self, visibility, vis_list):
    if len(vis_list) > 0:
      self.print_indent()
      print(f"{visibility} :: ", end="")      
      for index, member in enumerate(vis_list):
        if (index > 0): print(", ", end="")        
        print(member.data, end="")
      print("\n")
        
  def print_literal(self, op):    
    literal_val=op.attributes["value"]
    if isinstance(literal_val, IntegerAttr):
      print(op.attributes["value"].parameters[0].data, end="")
    elif isinstance(literal_val, ftn_dag.FloatAttr):
      print(op.attributes["value"].data, end="")      
        
  def print_assign(self, op):    
    self.print_indent()
    self.print_op(op.lhs.blocks[0].ops[0])
    print("=", end="")
    self.print_op(op.rhs.blocks[0].ops[0])
    print("") # For a new line      
        
  def print_if(self, op):    
    self.print_indent()
    print("if (", end="")
    self.print_op(op.cond.blocks[0].ops[0])
    print(") then")      
    self.incr+=2
    self.print_op(op.then.blocks[0].ops[0])
    self.incr-=2      
    if len(op.orelse.blocks[0].ops) > 0:
      self.print_indent()
      print("else")
      self.incr+=2
      self.print_op(op.orelse.blocks[0].ops[0])
      self.incr-=2
    self.print_indent()
    print("end if")
  
  def print_do(self, op):    
    self.print_indent()
    
    print("do ", end="")
    self.print_op(op.iterator.blocks[0].ops[0])
    print("=", end="")
    self.print_op(op.start.blocks[0].ops[0])
    print(", ", end="")
    self.print_op(op.stop.blocks[0].ops[0])
    print(", ", end="")
    self.print_op(op.step.blocks[0].ops[0])
    print("")
    self.incr+=2
    self.print_op(op.body.blocks[0].ops[0])
    self.incr-=2
    self.print_indent()
    print("end do")
      
  def print_routine(self, op):
    print("")
    self.print_out_routine(op)
    
  def print_out_routine(self, op):
    self.print_indent()
    if op.program_entry_point.data:
      print(f"program {op.routine_name.data}")
    else:
      print(f"subroutine {op.routine_name.data}(", end="")      
      for index, arg in enumerate(op.args.data):
        if (index > 0): print(", ", end="")
        print(arg.var_name.data, end="")
      print(")")
    self.incr+=2
    for import_stmt in op.imports.blocks[0].ops:
      self.print_indent()
      self.print_op(import_stmt)
    for block in op.local_var_declarations.blocks[0].ops:
      self.print_op(block)
    if len(op.local_var_declarations.blocks[0].ops) > 0: print("")
    for block in op.routine_body.blocks[0].ops:
      self.print_op(block)
    self.incr-=2
    self.print_indent()
    if op.program_entry_point.data:
      print(f"end program {op.routine_name.data}")
    else:
      print(f"end subroutine {op.routine_name.data}")
      
  def print_indent(self):
    print(" "*self.incr, end="")

def print_fortran(instructions, stream=sys.stdout):
  fortran_printer=FortranPrinter()
  for op in instructions:
    fortran_printer.print_op(op, stream=stream)
    
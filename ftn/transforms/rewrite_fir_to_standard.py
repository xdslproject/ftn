from abc import ABC
from enum import Enum
import itertools
import copy
from functools import reduce
from typing import TypeVar, cast
from dataclasses import dataclass
from xdsl.dialects.experimental import fir, hlfir
from dataclasses import dataclass, field
from typing import Dict, Optional
from xdsl.ir import SSAValue, BlockArgument
from xdsl.irdl import Operand
from xdsl.utils.hints import isa
from util.visitor import Visitor
from xdsl.ir import Operation, SSAValue, OpResult, Attribute, MLContext, Block, Region

from xdsl.pattern_rewriter import (RewritePattern, PatternRewriter,
                                   op_type_rewrite_pattern,
                                   PatternRewriteWalker,
                                   GreedyRewritePatternApplier)
from xdsl.passes import ModulePass
from xdsl.dialects import builtin, func, llvm, arith, memref, scf, cf, linalg
from xdsl.dialects.experimental import math
from ftn.dialects import ftn_relative_cf

ArgIntent = Enum('ArgIntent', ['IN', 'OUT', 'INOUT', 'UNKNOWN'])
LoopStepDirection = Enum('LoopStepDirection', ['INCREMENT', 'DECREMENT', 'UNKNOWN'])

def clean_func_name(func_name: str):
  if "_QP" in func_name:
    return func_name.split("_QP")[1]
  elif "_QM" in func_name:
    return func_name.split("_QM")[1]
  else:
    return func_name

class ArrayDescription:
  def __init__(self, name, dim_sizes, dim_starts, dim_ends):
    self.name=name
    self.dim_sizes=dim_sizes
    self.dim_starts=dim_starts
    self.dim_ends=dim_ends

class ComponentState:
  def __init__(self, fn_name=None, module_name=None, fn_identifier=None):
    self.fn_name=fn_name
    self.module_name=module_name
    self.fn_identifier=fn_identifier
    self.array_info={}

class ArgumentDefinition:
  def __init__(self, name, is_scalar, arg_type, intent, is_allocatable):
    self.name=name
    self.is_scalar=is_scalar
    self.intent=intent
    self.arg_type=arg_type
    self.is_allocatable=is_allocatable

class FunctionDefinition:
  def __init__(self, name, return_type, is_definition_only, blocks):
    self.name=name
    self.return_type=return_type
    self.args=[]
    self.is_definition_only=is_definition_only
    self.blocks=blocks

  def add_arg_def(self, arg_def):
    self.args.append(arg_def)

class GlobalFIRComponent:
  def __init__(self, sym_name, type, fir_mlir):
    self.sym_name=sym_name
    self.type=type
    self.fir_mlir=fir_mlir
    self.standard_mlir=None

class ProgramState:
  def __init__(self):
    self.function_definitions={}
    self.global_state=ComponentState()
    self.function_state=None
    self.fir_global_constants={}
    self.is_in_global=False

  def enterGlobal(self):
    assert self.function_state is None
    self.is_in_global=True

  def leaveGlobal(self):
    assert self.function_state is None
    self.is_in_global=False

  def isInGlobal(self):
    return self.is_in_global

  def addFunctionDefinition(self, name, fn_def):
    assert name not in self.function_definitions.keys()
    self.function_definitions[name]=fn_def

  def enterFunction(self, fn_name, function_identifier, module_name=None):
    assert self.function_state is None
    self.function_state=ComponentState(fn_name, module_name, function_identifier)

  def getCurrentFnState(self):
    assert self.function_state is not None
    return self.function_state

  def leaveFunction(self):
    self.function_state=None

class GatherFIRGlobals(Visitor):
  def __init__(self, program_state):
    self.program_state=program_state

  def traverse_global(self, global_op: fir.Global):
    if global_op.constant is not None:
      gfir=GlobalFIRComponent(global_op.sym_name.data, global_op.type, global_op)
      self.program_state.fir_global_constants[global_op.sym_name.data]=gfir

class GatherFunctionInformation(Visitor):
  def __init__(self, program_state):
    self.program_state=program_state

  def get_declare_from_arg_uses(self, arg_uses):
    for use in arg_uses:
      # It can unbox characters into a declare, if so then just follow it through
      if isa(use.operation, fir.Unboxchar):
        ub_dec=self.get_declare_from_arg_uses(use.operation.results[0].uses)
        if ub_dec is not None: return ub_dec
      if isa(use.operation, hlfir.DeclareOp):
        return use.operation
    return None

  def map_ftn_attrs_to_intent(self, ftn_attrs):
    if ftn_attrs is not None:
      for attr in ftn_attrs.data:
        if attr == fir.FortranVariableFlags.INTENT_IN:
          return ArgIntent.IN
        elif attr == fir.FortranVariableFlags.INTENT_INOUT:
          return ArgIntent.INOUT
        elif attr == fir.FortranVariableFlags.INTENT_OUT:
          return ArgIntent.OUT
    return ArgIntent.UNKNOWN

  def check_if_has_allocatable_attr(self, op: hlfir.DeclareOp):
    if "fortran_attrs" in op.properties.keys():
      attrs=op.properties["fortran_attrs"]
      assert isa(attrs, fir.FortranVariableFlagsAttr)
      for attr in attrs.data:
        if attr == fir.FortranVariableFlags.ALLOCATABLE:
          return True
    return False

  def get_base_type(self, t):
    if isa(t, fir.ReferenceType):
      return self.get_base_type(t.type)
    elif isa(t, fir.BoxType):
      return self.get_base_type(t.type)
    elif isa(t, fir.HeapType):
      return self.get_base_type(t.type)
    return t

  def traverse_func_op(self, func_op: func.FuncOp):
    fn_name=clean_func_name(func_op.sym_name.data)
    return_type=None
    if len(func_op.function_type.outputs.data) > 0:
      return_type=func_op.function_type.outputs.data[0]
    fn_def=FunctionDefinition(fn_name, return_type, len(func_op.body.blocks) == 0, list(func_op.body.blocks))
    if len(func_op.body.blocks) != 0:
      # This has concrete implementation (e.g. is not a function definition)
      assert len(func_op.body.blocks) >= 1
      # Even if the body has more than one block, we only care about the first block as that is
      # the entry point from the function, so it has the function arguments in it
      for block_arg in func_op.body.blocks[0].args:
        declare_op=self.get_declare_from_arg_uses(block_arg.uses)
        assert declare_op is not None
        arg_type=declare_op.results[0].type
        base_type=self.get_base_type(arg_type)
        is_scalar=declare_op.shape is None and not isa(base_type, fir.SequenceType)
        arg_name=declare_op.uniq_name.data
        is_allocatable=self.check_if_has_allocatable_attr(declare_op)
        # This is a bit strange, in a module we have modulenamePprocname, however
        # flang then uses modulenameFprocname for array literal string names
        #assert fn_name.replace("P", "F")+"E" in arg_name
        #arg_name=arg_name.split(fn_name.replace("P", "F")+"E")[1]
        arg_intent=self.map_ftn_attrs_to_intent(declare_op.fortran_attrs)
        arg_def=ArgumentDefinition(arg_name, is_scalar, arg_type, arg_intent, is_allocatable)
        fn_def.add_arg_def(arg_def)
    self.program_state.addFunctionDefinition(fn_name, fn_def)

class GatherFunctions(Visitor):
  def __init__(self):
    self.functions={}

  def traverse_func_op(self, func_op: func.FuncOp):
    fn_name=func_op.sym_name.data
    if isa(fn_name, builtin.StringAttr): fn_name=fn_name.data
    fn_name=clean_func_name(fn_name)
    self.functions[fn_name]=func_op

@dataclass
class SSAValueCtx:
    """
    Context that relates identifiers from the AST to SSA values used in the flat representation.
    """
    dictionary: Dict[str, SSAValue] = field(default_factory=dict)
    parent_scope = None

    def __init__(self, parent_scope=None):
      self.parent_scope=parent_scope
      self.dictionary = {}

    def __getitem__(self, identifier: str) -> Optional[SSAValue]:
        """Check if the given identifier is in the current scope, or a parent scope"""
        ssa_value = self.dictionary.get(identifier, None)
        if ssa_value:
            return ssa_value
        elif self.parent_scope:
            return self.parent_scope[identifier]
        else:
            return None

    def __delitem__(self, identifier: str):
        if identifier in self.dictionary:
          del self.dictionary[identifier]

    def __setitem__(self, identifier: str, ssa_value: SSAValue):
        """Relate the given identifier and SSA value in the current scope"""
        if identifier in self.dictionary:
            raise Exception()
        else:
            self.dictionary[identifier] = ssa_value

    def contains(self, identifier):
        if identifier in self.dictionary: return True
        if self.parent_scope is not None:
          return self.parent_scope.contains(identifier)
        else:
          return False


def translate_program(program_state: ProgramState, input_module: builtin.ModuleOp) -> builtin.ModuleOp:
    # create an empty global context
    global_ctx = SSAValueCtx()
    body = Region()
    block = Block()
    for fn in input_module.ops:
      if isa(fn, func.FuncOp):
        fn_op=translate_function(program_state, global_ctx, fn)
        if fn_op is not None: block.add_op(fn_op)
      elif isa(fn, fir.Global):
        global_op=translate_global(program_state, global_ctx, fn)
        if global_op is not None:
          block.add_op(global_op)
      else:
        assert False
    body.add_block(block)
    return builtin.ModuleOp(body)

def translate_global(program_state, global_ctx, global_op: fir.Global):
  if global_op.sym_name.data == "_QQEnvironmentDefaults":return None
  assert len(global_op.regions) == 1
  assert len(global_op.regions[0].blocks) == 1

  ops_list=[]
  program_state.enterGlobal()
  for op in global_op.regions[0].blocks[0].ops:
    ops_list+=translate_stmt(program_state, global_ctx, op)

  program_state.leaveGlobal()

  if isa(global_op.type, fir.CharacterType):
    assert len(ops_list)==1
    rebuilt_global=llvm.GlobalOp(ops_list[0].global_type, global_op.sym_name, ops_list[0].linkage,
                      ops_list[0].addr_space.value.data, global_op.constant, value=ops_list[0].value,
                      unnamed_addr=ops_list[0].unnamed_addr.value.data)
    return rebuilt_global
  elif isa(global_op.type, fir.IntegerType) or isa(global_op.type, builtin.AnyFloat) or isa(global_op.type, fir.SequenceType):
    assert len(ops_list)==1
    global_contained_op=ops_list[0]
    assert (isa(global_contained_op, arith.Constant) or isa(global_contained_op, llvm.ZeroOp))
    return_op=llvm.ReturnOp.build(operands=[global_contained_op.results[0]])

    return llvm.GlobalOp(global_contained_op.results[0].type, global_op.sym_name, "internal",
                      constant=global_op.constant,
                      body=Region([Block([global_contained_op, return_op])]))
  else:
    raise Exception(f"Could not translate global region of type `{global_op.type}'")

def convert_fir_type_to_standard(fir_type, ref_as_mem_ref=True):
  if isa(fir_type, fir.ReferenceType):
    if ref_as_mem_ref:
      base_t=convert_fir_type_to_standard(fir_type.type, ref_as_mem_ref)
      if isa(base_t, memref.MemRefType):
        return base_t
      else:
        return memref.MemRefType(base_t, [], builtin.NoneAttr(), builtin.NoneAttr())
    else:
      return llvm.LLVMPointerType.opaque()
  elif isa(fir_type, fir.BoxType):
    return convert_fir_type_to_standard(fir_type.type, ref_as_mem_ref)
  elif isa(fir_type, fir.SequenceType):
    base_t=convert_fir_type_to_standard(fir_type.type)
    dim_sizes=[]
    for shape_el in fir_type.shape:
      if isa(shape_el, builtin.IntegerAttr):
        dim_sizes.append(shape_el.value.data)
      else:
        dim_sizes.append(-1)
    # Reverse the sizes to go from Fortran to C allocation semantics
    dim_sizes.reverse()
    return memref.MemRefType(base_t, dim_sizes, builtin.NoneAttr(), builtin.NoneAttr())
  elif isa(fir_type, fir.LogicalType):
    return builtin.i1
  else:
    return fir_type

def translate_function(program_state: ProgramState, ctx: SSAValueCtx, fn: func.FuncOp):
  within_module=fn.sym_name.data.startswith("_QM")
  fn_identifier=clean_func_name(fn.sym_name.data)

  if within_module:
    module_name=fn_identifier.split("P")[0]
    fn_name=fn_identifier.split("P")[1]
  else:
    module_name=None
    fn_name=fn_identifier

  if fn_name in FortranIntrinsicsHandleExplicitly.keys(): return None

  body = Region()
  if len(fn.body.blocks) > 0:
    # This is a function with a body, the input types come from the block as
    # we will manipulate these to pass constants if possible
    program_state.enterFunction(fn_name, fn_identifier, module_name)
    fn_in_arg_types=[]
    for idx, arg in enumerate(fn.args):
      fir_type=arg.type
      if (program_state.function_definitions[fn_identifier].args[idx].is_scalar and
          program_state.function_definitions[fn_identifier].args[idx].intent == ArgIntent.IN):
        # This is a scalar in, therefore it's just the constant type (don't encode as a memref)
        if isa(fir_type, fir.ReferenceType):
          fn_in_arg_types.append(fir_type.type)
        else:
          fn_in_arg_types.append(arg.type)
      else:
        converted_type=convert_fir_type_to_standard(fir_type)
        if (isa(converted_type, memref.MemRefType) and
            program_state.function_definitions[fn_identifier].args[idx].is_allocatable):
          converted_type=memref.MemRefType(converted_type, shape=[])

        fn_in_arg_types.append(converted_type)

    for idx, block in enumerate(fn.body.blocks):
      if idx == 0:
        # If this is the first block, then it is the function arguments
        new_block = Block(arg_types=fn_in_arg_types)
      else:
        # Otherwise the arg types are the same as the blocks
        new_block = Block(arg_types=block.args)

      for fir_arg, std_arg in zip(block.args, new_block.args):
        ctx[fir_arg]=std_arg

      ops_list=[]
      for op in block.ops:
        ops_list+=translate_stmt(program_state, ctx, op)

      new_block.add_ops(ops_list)
      body.add_block(new_block)
    program_state.leaveFunction()
  else:
    # This is the definition of an external function, need to resolve input types
    fn_in_arg_types=[]
    for t in fn.function_type.inputs.data:
      fn_in_arg_types.append(convert_fir_type_to_standard_if_needed(t))

  # Perform some conversion on return types to standard
  return_types=[]
  for rt in fn.function_type.outputs.data:
    if not isa(rt, builtin.NoneType):
      # Ignore none types, these are simply omitted
      return_types.append(convert_fir_type_to_standard_if_needed(rt))

  fn_identifier=fn.sym_name
  if fn_identifier.data == "_QQmain":
    fn_identifier="main"

  new_fn_type=builtin.FunctionType.from_lists(fn_in_arg_types, return_types)

  new_func=func.FuncOp(fn_identifier, new_fn_type, body, fn.sym_visibility, arg_attrs=fn.arg_attrs, res_attrs=fn.res_attrs)
  return new_func

def convert_fir_type_to_standard_if_needed(fir_type):
  if isa(fir_type, fir.ReferenceType):
    return llvm.LLVMPointerType.opaque()
  else:
    return fir_type

def translate_stmt(program_state: ProgramState, ctx: SSAValueCtx, op: Operation):
  ops = try_translate_stmt(program_state, ctx, op)
  if ops is not None:
    return ops
  return []
  #raise Exception(f"Could not translate `{op}' as a definition or statement")

def try_translate_stmt(program_state: ProgramState, ctx: SSAValueCtx, op: Operation):
  if isa(op, hlfir.DeclareOp):
    return translate_declare(program_state, ctx, op)
  elif isa(op, fir.DoLoop):
    return translate_do_loop(program_state, ctx, op)
  elif isa(op, fir.IterateWhile):
    return translate_iterate_while(program_state, ctx, op)
  elif isa(op, fir.Alloca):
    # Will only process this if it is an internal flag,
    # otherwise pick up as part of the declareop
    return translate_alloca(program_state, ctx, op)
  elif isa(op, arith.Constant):
    return []
  elif isa(op, fir.HasValue):
    return translate_expr(program_state, ctx, op.resval)
  elif isa(op, fir.Load):
    return []
  elif (isa(op, arith.Addi) or isa(op, arith.Subi) or isa(op, arith.Muli) or isa(op, arith.DivUI) or isa(op, arith.DivSI) or
      isa(op, arith.FloorDivSI) or isa(op, arith.CeilDivSI) or isa(op, arith.CeilDivUI) or isa(op, arith.RemUI) or
      isa(op, arith.RemSI) or isa(op, arith.MinUI) or isa(op, arith.MaxUI) or isa(op, arith.MinSI) or isa(op, arith.MaxSI) or
      isa(op, arith.AndI) or isa(op, arith.OrI) or isa(op, arith.XOrI) or isa(op, arith.ShLI) or isa(op, arith.ShRUI) or
      isa(op, arith.ShRSI) or isa(op, arith.AddUIExtended)):
    return []
  elif (isa(op, arith.Addf) or isa(op, arith.Subf) or isa(op, arith.Mulf) or isa(op, arith.Divf) or isa(op, arith.Maximumf) or
      isa(op, arith.Maxnumf) or isa(op, arith.Minimumf) or isa(op, arith.Minnumf)):
    return []
  elif isa(op, func.Return):
    return translate_return(program_state, ctx, op)
  elif isa(op, hlfir.AssignOp):
    return translate_assign(program_state, ctx, op)
  elif isa(op, fir.Store):
    # Used internally by some ops still, e.g. to store loop bounds per iteration
    return translate_store(program_state, ctx, op)
  elif isa(op, fir.Result):
    return translate_result(program_state, ctx, op)
  elif isa(op, fir.Call):
    return translate_call(program_state, ctx, op)
  elif isa(op, fir.Freemem):
    return translate_freemem(program_state, ctx, op)
  elif isa(op, fir.If):
    return translate_conditional(program_state, ctx, op)
  elif isa(op, cf.Branch):
    return translate_branch(program_state, ctx, op)
  elif isa(op, cf.ConditionalBranch):
    return translate_conditional_branch(program_state, ctx, op)
  elif isa(op, fir.Unreachable):
    return [llvm.UnreachableOp()]
  else:
    return None

def translate_expr(program_state: ProgramState, ctx: SSAValueCtx, ssa_value: SSAValue):
  if isa(ssa_value, BlockArgument):
    return []
  else:
    ops = try_translate_expr(program_state, ctx, ssa_value.owner)
    if ops is not None:
      return ops

    raise Exception(f"Could not translate `{ssa_value.owner}' as an expression")

def try_translate_expr(program_state: ProgramState, ctx: SSAValueCtx, op: Operation):
  if isa(op, arith.Constant):
    return translate_constant(program_state, ctx, op)
  elif (isa(op, arith.Addi) or isa(op, arith.Subi) or isa(op, arith.Muli) or isa(op, arith.DivUI) or isa(op, arith.DivSI) or
      isa(op, arith.FloorDivSI) or isa(op, arith.CeilDivSI) or isa(op, arith.CeilDivUI) or isa(op, arith.RemUI) or
      isa(op, arith.RemSI) or isa(op, arith.MinUI) or isa(op, arith.MaxUI) or isa(op, arith.MinSI) or isa(op, arith.MaxSI) or
      isa(op, arith.AndI) or isa(op, arith.OrI) or isa(op, arith.XOrI) or isa(op, arith.ShLI) or isa(op, arith.ShRUI) or
      isa(op, arith.ShRSI) or isa(op, arith.AddUIExtended)):
    return translate_integer_binary_arithmetic(program_state, ctx, op)
  elif (isa(op, arith.Addf) or isa(op, arith.Subf) or isa(op, arith.Mulf) or isa(op, arith.Divf) or isa(op, arith.Maximumf) or
      isa(op, arith.Maxnumf) or isa(op, arith.Minimumf) or isa(op, arith.Minnumf)):
    return translate_float_binary_arithmetic(program_state, ctx, op)
  elif isa(op, arith.Negf):
    return translate_float_unary_arithmetic(program_state, ctx, op)
  elif isa(op, fir.Load):
    return translate_load(program_state, ctx, op)
  elif isa(op, fir.Convert):
    return translate_convert(program_state, ctx, op)
  elif isa(op, fir.DoLoop):
    # Do loop can be either an expression or statement
    return translate_do_loop(program_state, ctx, op)
  elif isa(op, fir.IterateWhile):
    return translate_iterate_while(program_state, ctx, op)
  elif isa(op, hlfir.DeclareOp):
    return translate_declare(program_state, ctx, op)
  elif isa(op, arith.Cmpi) or isa(op, arith.Cmpf):
    return translate_cmp(program_state, ctx, op)
  elif isa(op, fir.Call):
    return translate_call(program_state, ctx, op)
  elif isa(op, fir.StringLit):
    return translate_string_literal(program_state, ctx, op)
  elif isa(op, fir.AddressOf):
    return translate_address_of(program_state, ctx, op)
  elif isa(op, hlfir.NoReassocOp) or isa(op, fir.NoReassoc):
    return translate_reassoc(program_state, ctx, op)
  elif isa(op, fir.ZeroBits):
    return translate_zerobits(program_state, ctx, op)
  elif isa(op, fir.BoxAddr):
    # Ignore box address, just process argument and link to results of that
    expr_list=translate_expr(program_state, ctx, op.val)
    ctx[op.results[0]]=ctx[op.val]
    return expr_list
  elif isa(op, arith.Select):
    return translate_select(program_state, ctx, op)
  elif isa(op, fir.Embox) or isa(op, fir.Emboxchar):
    expr_ops=translate_expr(program_state, ctx, op.memref)
    ctx[op.results[0]]=ctx[op.memref.owner.results[0]]
    return expr_ops
  elif isa(op, hlfir.AssociateOp):
    expr_ops=translate_expr(program_state, ctx, op.source)
    ctx[op.results[0]]=ctx[op.source.owner.results[0]]
    return expr_ops
  elif isa(op, hlfir.AsExprOp):
    expr_ops=translate_expr(program_state, ctx, op.var)
    ctx[op.results[0]]=ctx[op.var]
    return expr_ops
  elif isa(op, fir.Rebox):
    expr_ops=translate_expr(program_state, ctx, op.box)
    ctx[op.results[0]]=ctx[op.box]
    return expr_ops
  elif isa(op, hlfir.DotProductOp):
    return translate_dotproduct(program_state, ctx, op)
  elif isa(op, hlfir.CopyInOp):
    return translate_copyin(program_state, ctx, op)
  elif isa(op, fir.Absent):
    return translate_absent(program_state, ctx, op)
  else:
    for math_op in math.Math.operations:
      # Check to see if this is a math operation
      if isa(op, math_op):
        return translate_math_operation(program_state, ctx, op)
    return None

def translate_absent(program_state: ProgramState, ctx: SSAValueCtx, op: fir.Absent):
  if ctx.contains(op.results[0]): return []

  null_ptr=llvm.ZeroOp(llvm.LLVMPointerType.opaque())
  ctx[op.results[0]]=null_ptr.results[0]
  return [null_ptr]

def translate_copyin(program_state: ProgramState, ctx: SSAValueCtx, op: hlfir.CopyInOp):
  if ctx.contains(op.results[0]): return []

  expr_ops=translate_expr(program_state, ctx, op.var)
  ctx[op.results[0]]=ctx[op.var]
  return expr_ops

def translate_dotproduct(program_state: ProgramState, ctx: SSAValueCtx, op: hlfir.DotProductOp):
  if ctx.contains(op.results[0]): return []
  lhs_ops_list=translate_expr(program_state, ctx, op.lhs)
  rhs_ops_list=translate_expr(program_state, ctx, op.rhs)

  if isa(ctx[op.lhs].type.element_type, memref.MemRefType):
    load_op, lhs_load_ssa=generate_dereference_memref(ctx[op.lhs])
    lhs_ops_list.append(load_op)
  else:
    lhs_load_ssa=ctx[op.lhs]

  if isa(ctx[op.rhs].type.element_type, memref.MemRefType):
    load_op, rhs_load_ssa=generate_dereference_memref(ctx[op.rhs])
    rhs_ops_list.append(load_op)
  else:
    rhs_load_ssa=ctx[op.rhs]

  output_memref_op=memref.Alloca.get(op.results[0].type, shape=[])
  dot_op=linalg.DotOp((lhs_load_ssa, rhs_load_ssa), [output_memref_op])
  extract_op=memref.Load.get(output_memref_op, [])

  ctx[op.results[0]]=extract_op.results[0]

  return lhs_ops_list+rhs_ops_list+[output_memref_op, dot_op, extract_op]

def translate_zerobits(program_state: ProgramState, ctx: SSAValueCtx, op: fir.ZeroBits):
  # This often appears in global regions for array declaration, if so then we need to
  # handle differently as can not use a memref in LLVM global operations
  result_type=op.results[0].type
  if isa(result_type, fir.SequenceType):
    base_type=result_type.type
    array_sizes=[]
    for d in result_type.shape.data:
      assert isa(d, builtin.IntegerAttr)
      array_sizes.append(d.value.data)
  else:
    base_type=result_type
    array_sizes=[1]

  if program_state.isInGlobal():
    # Need to allocate as LLVM compatible operation
    total_size=reduce((lambda x, y: x * y), array_sizes)
    llvm_array_type=llvm.LLVMArrayType.from_size_and_type(total_size, base_type)
    zero_op=llvm.ZeroOp(llvm_array_type)
    ctx[op.results[0]]=zero_op.results[0]
    return [zero_op]
  else:
    memref_alloc=memref.Alloc.get(base_type, shape=array_sizes)
    ctx[op.results[0]]=memref_alloc.results[0]
    return [memref_alloc]

def translate_select(program_state: ProgramState, ctx: SSAValueCtx, op: arith.Select):
  if ctx.contains(op.results[0]): return []

  cond_ops_list=translate_expr(program_state, ctx, op.cond)
  lhs_ops_list=translate_expr(program_state, ctx, op.lhs)
  rhs_ops_list=translate_expr(program_state, ctx, op.rhs)

  select_op=arith.Select(ctx[op.cond], ctx[op.lhs], ctx[op.rhs])

  ctx[op.results[0]]=select_op.results[0]

  return cond_ops_list+lhs_ops_list+rhs_ops_list+[select_op]

def translate_reassoc(program_state: ProgramState, ctx: SSAValueCtx, op: fir.NoReassoc | hlfir.NoReassocOp):
  if ctx.contains(op.results[0]): return []
  if isa(op, fir.NoReassoc):
    expr_list=translate_expr(program_state, ctx, op.val)
  elif isa(op, hlfir.NoReassocOp):
    expr_list=translate_expr(program_state, ctx, op.var)

  ctx[op.results[0]]=ctx[op.var]
  return expr_list

def translate_address_of(program_state: ProgramState, ctx: SSAValueCtx, op: fir.AddressOf):
  if ctx.contains(op.results[0]): return []

  assert isa(op.results[0].type, fir.ReferenceType)
  global_lookup=llvm.AddressOfOp(op.symbol, llvm.LLVMPointerType.opaque())

  ctx[op.results[0]]=global_lookup.results[0]
  return [global_lookup]

def translate_string_literal(program_state: ProgramState, ctx: SSAValueCtx, op: fir.StringLit):
  str_type=llvm.LLVMArrayType.from_size_and_type(op.size.value.data, builtin.IntegerType(8))
  str_global_op=llvm.GlobalOp(str_type, "temporary_identifier", "internal", 0, True, value=op.value, unnamed_addr=0)
  return [str_global_op]

def translate_cmp(program_state: ProgramState, ctx: SSAValueCtx, op: arith.Cmpi | arith.Cmpf):
  if ctx.contains(op.results[0]): return []

  lhs_expr_ops=translate_expr(program_state, ctx, op.lhs)
  rhs_expr_ops=translate_expr(program_state, ctx, op.rhs)

  if isa(op, arith.Cmpi):
    comparison_op=arith.Cmpi(ctx[op.lhs], ctx[op.rhs], op.predicate.value.data)
  elif isa(op, arith.Cmpf):
    comparison_op=arith.Cmpf(ctx[op.lhs], ctx[op.rhs], op.predicate.value.data)
  else:
    assert False

  ctx[op.results[0]]=comparison_op.results[0]
  return lhs_expr_ops+rhs_expr_ops+[comparison_op]


def translate_convert(program_state: ProgramState, ctx: SSAValueCtx, op: fir.Convert):
  if ctx.contains(op.results[0]): return []
  value_ops=translate_expr(program_state, ctx, op.value)
  in_type=op.value.type
  out_type=op.results[0].type
  new_conv=None
  if isa(in_type, builtin.Float32Type) and isa(out_type,
    builtin.Float64Type) or isa(in_type, builtin.Float16Type) and isa(out_type,
    builtin.Float64Type) or isa(in_type, builtin.Float16Type) and isa(out_type,
    builtin.Float32Type):
    new_conv=arith.ExtFOp(ctx[op.value], out_type)
    ctx[op.results[0]]=new_conv.results[0]

  if isa(in_type, builtin.Float64Type) and isa(out_type,
    builtin.Float32Type) or isa(in_type, builtin.Float64Type) and isa(out_type,
    builtin.Float16Type) or isa(in_type, builtin.Float32Type) and isa(out_type,
    builtin.Float16Type):
    new_conv=arith.TruncFOp(ctx[op.value], out_type)
    ctx[op.results[0]]=new_conv.results[0]

  if ((isa(in_type, builtin.IndexType) and isa(out_type, builtin.IntegerType)) or
        (isa(in_type, builtin.IntegerType) and isa(out_type, builtin.IndexType))):
    new_conv=arith.IndexCastOp(ctx[op.value], out_type)
    ctx[op.results[0]]=new_conv.results[0]

  if isa(in_type, builtin.IntegerType) and isa(out_type, builtin.AnyFloat):
    new_conv=arith.SIToFPOp(ctx[op.value], out_type)
    ctx[op.results[0]]=new_conv.results[0]

  if isa(in_type, builtin.AnyFloat) and isa(out_type, builtin.IntegerType):
    new_conv=arith.FPToSIOp(ctx[op.value], out_type)
    ctx[op.results[0]]=new_conv.results[0]

  if new_conv is not None: new_conv=[new_conv]

  if isa(in_type, builtin.IntegerType) and isa(out_type, builtin.IntegerType):
    in_width=in_type.width.data
    out_width=out_type.width.data
    if in_width < out_width:
      new_conv=arith.ExtUIOp(ctx[op.value], out_type)
      ctx[op.results[0]]=new_conv.results[0]
      new_conv=[new_conv]
    elif in_width > out_width:
      new_conv=arith.TruncIOp(ctx[op.value], out_type)
      ctx[op.results[0]]=new_conv.results[0]
      new_conv=[new_conv]
    else:
      # They are the same, ignore and use the input directly
      new_conv=[]
      ctx[op.results[0]]=ctx[op.value]

  if (isa(in_type, fir.ReferenceType) and isa(out_type, fir.ReferenceType) or
        (isa(in_type, fir.BoxType) and isa(out_type, fir.BoxType))):
    if isa(out_type.type, builtin.IntegerType) and out_type.type.width.data == 8:
      # Converting to an LLVM pointer
      # The element type is an LLVM array, we hard code this to be size 1 here which is OK as it just needs to
      # grab the starting pointer to this
      get_element_ptr=llvm.GEPOp(ctx[op.value], [0,0], result_type=llvm.LLVMPointerType.opaque(),
                        pointee_type=llvm.LLVMArrayType.from_size_and_type(1, builtin.IntegerType(8)))
      ctx[op.results[0]]=get_element_ptr.results[0]
      new_conv=[get_element_ptr]
    elif isa(out_type.type, fir.SequenceType) and isa(in_type.type, fir.SequenceType):
      # Converting between two shapes in the array
      assert out_type.type.type == in_type.type.type
      shape_size=[]
      for s in out_type.type.shape.data:
        if isa(s, fir.DeferredAttr):
          shape_size.append(-1)
        else:
          shape_size.append(s.value.data)
      # Reverse shape_size to get it from Fortran allocation to C/MLIR allocation
      shape_size.reverse()
      target_type=memref.MemRefType(convert_fir_type_to_standard(out_type.type.type), shape_size)
      cast_op=memref.Cast.get(ctx[op.value], target_type)

      ctx[op.results[0]]=cast_op.results[0]
      new_conv=[cast_op]
    elif isa(out_type.type, fir.BoxType) and isa(in_type.type, fir.BoxType):
      new_conv=[]
      ctx[op.results[0]]=ctx[op.value]

  if isa(in_type, fir.HeapType) and isa(out_type, fir.ReferenceType):
    # When passing arrays to subroutines will box_addr to a heaptype, then convert
    # to a reference type. Both these contain arrays, therefore set this to
    # short circuit to the type of the arg (effectively this is a pass through)
    assert isa(in_type.type, fir.SequenceType)
    assert isa(out_type.type, fir.SequenceType)
    # Assert that what we will forward to is in-fact a memref type
    assert isa(ctx[op.value].type, builtin.MemRefType)
    ctx[op.results[0]]=ctx[op.value]
    new_conv=[]

  if isa(in_type, fir.LogicalType):
    assert out_type == builtin.i1
    ctx[op.results[0]]=ctx[op.value]
    new_conv=[]

  if in_type == builtin.i1:
    assert isa(out_type, fir.LogicalType)
    ctx[op.results[0]]=ctx[op.value]
    new_conv=[]

  if new_conv is None:
    raise Exception(f"Could not convert between `{in_type}' and `{out_type}`")
  return value_ops+new_conv

def handle_conditional_true_or_false_region(program_state: ProgramState, ctx: SSAValueCtx, region: Region):
  arg_types=[]
  for arg in region.blocks[0].args:
    arg_types.append(arg.type)

  new_block = Block(arg_types=arg_types)

  for fir_arg, std_arg in zip(region.blocks[0].args, new_block.args):
    ctx[fir_arg]=std_arg

  region_body_ops=[]
  for single_op in region.blocks[0].ops:
    region_body_ops+=translate_stmt(program_state, ctx, single_op)

  assert isa(region_body_ops[-1], scf.Yield)
  new_block.add_ops(region_body_ops)

  return new_block

def check_if_condition_is_end_fn_allocatable_automatic_free(condition_op: arith.Cmpi | arith.Cmpf):
  if isa(condition_op, arith.Cmpi):
    if isa(condition_op.lhs.owner, fir.Convert):
      return (isa(condition_op.lhs.owner.value.type, fir.HeapType) and
                isa(condition_op.lhs.owner.results[0].type, builtin.IntegerType))
  return False

def translate_conditional(program_state: ProgramState, ctx: SSAValueCtx, op: fir.If):
  # Each function automatically deallocates scope local allocatable arrays at the end,
  # check to see if that is the purpose of this conditional. If so then just ignore it
  is_final_auto_free=check_if_condition_is_end_fn_allocatable_automatic_free(op.condition.owner)
  if is_final_auto_free: return []

  conditional_expr_ops=translate_expr(program_state, ctx, op.condition)

  true_block=handle_conditional_true_or_false_region(program_state, ctx, op.regions[0])
  false_block=handle_conditional_true_or_false_region(program_state, ctx, op.regions[1])

  scf_if=scf.If(ctx[op.condition], [], [true_block], [false_block])

  return conditional_expr_ops+[scf_if]

def translate_branch(program_state: ProgramState, ctx: SSAValueCtx, op: cf.Branch):
  current_fn_identifier=program_state.getCurrentFnState().fn_identifier
  target_block_index=program_state.function_definitions[current_fn_identifier].blocks.index(op.successor)

  ops_list=[]
  block_ssas=[]
  for arg in op.arguments:
    ops_list+=translate_expr(program_state, ctx, arg)
    block_ssas.append(ctx[arg])
  relative_branch=ftn_relative_cf.Branch(current_fn_identifier, target_block_index, *block_ssas)
  ops_list.append(relative_branch)
  return ops_list

def translate_conditional_branch(program_state: ProgramState, ctx: SSAValueCtx, op: cf.ConditionalBranch):
  current_fn_identifier=program_state.getCurrentFnState().fn_identifier
  then_block_index=program_state.function_definitions[current_fn_identifier].blocks.index(op.then_block)
  else_block_index=program_state.function_definitions[current_fn_identifier].blocks.index(op.else_block)

  ops_list=[]
  ops_list+=translate_expr(program_state, ctx, op.cond)

  then_block_ssas=[]
  else_block_ssas=[]
  for arg in op.then_arguments:
    ops_list+=translate_expr(program_state, ctx, arg)
    then_block_ssas.append(ctx[arg])
  for arg in op.else_arguments:
    ops_list+=translate_expr(program_state, ctx, arg)
    else_block_ssas.append(ctx[arg])

  relative_cond_branch=ftn_relative_cf.ConditionalBranch(current_fn_identifier, ctx[op.cond], then_block_index, then_block_ssas, else_block_index, else_block_ssas)
  ops_list.append(relative_cond_branch)

  return ops_list

def generate_dereference_memref(memref_ssa):
  load_op=memref.Load.get(memref_ssa, [])
  return load_op, load_op.results[0]

def translate_load(program_state: ProgramState, ctx: SSAValueCtx, op: fir.Load):
  if ctx.contains(op.results[0]): return []

  # If this is a block argument, then it might be a scalar if it's in only. Therefore
  # check to see if this is a block argument and whether type is not memref, if so
  # just link directly. Otherwise it must be a memref
  if isa(ctx[op.memref], BlockArgument) and not isa(ctx[op.memref].type, memref.MemRefType):
    ctx[op.results[0]]=ctx[op.memref]
    return []
  elif isa(op.memref.owner, hlfir.DeclareOp):
    # Scalar value
    if isa(ctx[op.memref].type, memref.MemRefType):
      # This is held in a memref, it is a variable in the user's code,
      # therefore load it up
      assert isa(op.memref.owner.results[0].type, fir.ReferenceType)
      # Check if this is an entire array we are loading
      if isa(op.memref.owner.results[0].type.type, fir.BoxType):
        # If its an entire array, link to the memref, as that will be used directly
        ctx[op.results[0]]=ctx[op.memref]
        return []
      else:
        # Otherwise assume it is a scalar

        load_op=memref.Load.get(ctx[op.memref], [])
        ctx[op.results[0]]=load_op.results[0]
        return [load_op]
    elif isa(ctx[op.memref].type, llvm.LLVMPointerType):
      # This is referenced by an LLVM pointer, it is because it has been loaded by an addressof
      # operation, most likely because that loads in a global. Regardless, we issue and LLVM
      # load operation to load the value
      assert isa(op.memref.owner.results[0].type, fir.ReferenceType)
      # As LLVM pointer types are opaque, we need to grab the element type from
      # the declaration fir.reference type
      load_op=llvm.LoadOp(ctx[op.memref], op.memref.owner.results[0].type.type)
      ctx[op.results[0]]=load_op.results[0]
      return [load_op]
    else:
      assert False
  elif isa(op.memref.owner, hlfir.DesignateOp):
    # Array value
    assert op.memref.owner.indices is not None
    assert isa(op.memref.owner.memref.owner, hlfir.DeclareOp) or isa(op.memref.owner.memref.owner, fir.Load)
    ops_list, indexes_ssa=array_access_components(program_state, ctx, op.memref.owner)

    if isa(op.memref.owner.memref.owner, hlfir.DeclareOp):
      src_ssa=op.memref.owner.memref
    elif isa(op.memref.owner.memref.owner, fir.Load):
      src_ssa=op.memref.owner.memref.owner.memref
    else:
      assert False

    assert isa(ctx[src_ssa].type, memref.MemRefType)
    if isa(ctx[src_ssa].type.element_type, memref.MemRefType):
      load_op, load_ssa=generate_dereference_memref(ctx[src_ssa])
      ops_list.append(load_op)
    else:
      load_ssa=ctx[src_ssa]

    # Reverse the indicies as Fortran and C/MLIR are opposite in terms of
    # the order of the contiguous dimension (F is least, whereas C/MLIR is highest)
    indexes_ssa_reversed=indexes_ssa.copy()
    indexes_ssa_reversed.reverse()
    load_op=memref.Load.get(load_ssa, indexes_ssa_reversed)
    ops_list.append(load_op)
    ctx[op.results[0]]=load_op.results[0]
    return ops_list
  elif isa(op.memref.owner, fir.Alloca):
    # This is used for loading an internal variable
    assert isa(op.memref.owner.results[0].type, fir.ReferenceType)

    load_op=memref.Load.get(ctx[op.memref], [])
    ctx[op.results[0]]=load_op.results[0]
    return [load_op]
  else:
    assert False

def translate_freemem(program_state: ProgramState, ctx: SSAValueCtx, op: fir.Freemem):
  assert isa(op.heapref.owner, fir.BoxAddr)
  assert isa(op.heapref.owner.val.owner, fir.Load)
  memref_ssa=op.heapref.owner.val.owner.memref

  ops_list=[]
  assert isa(ctx[memref_ssa].type, memref.MemRefType)
  if isa(ctx[memref_ssa].type.element_type, memref.MemRefType):
    load_op, load_ssa=generate_dereference_memref(ctx[memref_ssa])
    ops_list.append(load_op)
  else:
    load_ssa=ctx[memref_ssa]

  ops_list.append(memref.Dealloc.get(load_ssa))
  return ops_list

def translate_result(program_state: ProgramState, ctx: SSAValueCtx, op: fir.Result):
  ops_list=[]
  ssa_list=[]
  for operand in op.operands:
    expr_ops=translate_expr(program_state, ctx, operand)
    ops_list+=expr_ops
    ssa_list.append(ctx[operand])
  yield_op=scf.Yield(*ssa_list)
  return ops_list+[yield_op]

def remove_array_size_convert(op):
  # This is for array size, they are often converted from
  # integer to index, so work back to find the original integer
  if isa(op, fir.Convert):
    assert isa(op.results[0].type, builtin.IndexType)
    assert isa(op.value.type, builtin.IntegerType)
    return op.value.owner
  return op

def create_index_constant(val: int):
  return arith.Constant.create(properties={"value": builtin.IntegerAttr.from_index_int_value(val)},
                                               result_types=[builtin.IndexType()])

def generate_var_dim_size_load(ctx: SSAValueCtx, op: fir.Load):
  # Generates operations to load the size of a dimension from a variable
  # and ensure that it is typed as an index
  var_ssa=ctx[op.memref]
  ops_list=[]
  if isa(var_ssa.type, builtin.MemRefType):
    # If this is a memref then we need to load it to
    # retrieve the index value
    load_op=memref.Load.get(var_ssa, [])
    ops_list+=[load_op]
    var_ssa=load_op.results[0]
  if not isa(var_ssa.type, builtin.IndexType):
    assert isa(var_ssa.type, builtin.IntegerType)
    convert_op=arith.IndexCastOp(var_ssa, builtin.IndexType())
    ops_list.append(convert_op)
    return convert_op.results[0], ops_list
  else:
    return var_ssa, ops_list

def handle_array_size_lu_bound(program_state: ProgramState, ctx: SSAValueCtx, bound_op: Operation, ssa):
  # Handles extracting the literal size of a lower or upper array size bound
  # or the corresponding ssa and ops if it is driven by a variable
  bound_val=load_ssa=load_ops=None
  bound_op=remove_array_size_convert(bound_op)
  if isa(bound_op, arith.Constant):
    bound_val=bound_op.value.value.data
  elif isa(bound_op, fir.Load):
    load_ssa, load_ops=generate_var_dim_size_load(ctx, bound_op)
  else:
    # Otherwise this is a general expression, therefore translate that and
    # handle it generally
    load_ops=translate_expr(program_state, ctx, ssa)
    load_ssa=load_ops[-1].results[0]

  return bound_val, load_ssa, load_ops


def translate_store(program_state: ProgramState, ctx: SSAValueCtx, op: fir.Store):
  # This is used for internal program components, such as loop indexes and storing
  # allocated memory to an allocatable
  if isa(op.value.owner, fir.Embox):
    # This is a allocating memory for an allocatable array
    if isa(op.value.owner.memref.owner, fir.Allocmem):
      assert isa(op.value.owner.memref.owner.results[0].type, fir.HeapType)
      assert isa(op.value.owner.memref.owner.results[0].type.type, fir.SequenceType)
      base_type=op.value.owner.memref.owner.results[0].type.type.type
      for dim_shape in op.value.owner.memref.owner.results[0].type.type.shape.data:
        assert isa(dim_shape, fir.DeferredAttr)

      assert len(op.value.owner.shape) == 1
      default_start_idx_mode=isa(op.value.owner.shape[0].owner, fir.Shape)

      dim_sizes=[]
      dim_starts=[]
      dim_ends=[]
      dim_ssas=[]
      ops_list=[]
      for shape in op.value.owner.memref.owner.shape:
        assert isa(shape.owner, arith.Select)
        # Flang adds a guard to ensure that non-negative size is used, hence select
        # the actual provided size is the lhs
        size_op=remove_array_size_convert(shape.owner.lhs.owner)
        if isa(size_op, arith.Constant):
          # Default sized
          dim_sizes.append(size_op.value.value.data)
          const_op=create_index_constant(size_op.value.value.data)
          ops_list.append(const_op)
          dim_ssas.append(const_op.results[0])
        elif isa(size_op, arith.Addi):
          # Start dim is offset, this does the substract, then add one
          # so we need to work back to calculate
          assert isa(size_op.rhs.owner, arith.Constant)
          assert size_op.rhs.owner.value.value.data == 1
          assert isa(size_op.lhs.owner, arith.Subi)

          upper_bound_val, upper_load_ssa, upper_load_ops=handle_array_size_lu_bound(program_state, ctx, size_op.lhs.owner.lhs.owner, size_op.lhs.owner.lhs)
          if upper_bound_val is not None:
            assert upper_load_ssa is None
            assert upper_load_ops is None
            dim_ends.append(upper_bound_val)
          else:
            assert upper_load_ssa is not None
            assert upper_load_ops is not None
            ops_list+=upper_load_ops
            dim_ends.append(upper_load_ssa)
            ctx[upper_load_ssa]=upper_load_ssa

          lower_bound_val, lower_load_ssa, lower_load_ops=handle_array_size_lu_bound(program_state, ctx, size_op.lhs.owner.rhs.owner, size_op.lhs.owner.rhs)
          if lower_bound_val is not None:
            assert lower_load_ssa is None
            assert lower_load_ops is None
            dim_starts.append(lower_bound_val)
          else:
            assert lower_load_ssa is not None
            assert lower_load_ops is not None
            ops_list+=lower_load_ops
            dim_starts.append(lower_load_ssa)
            ctx[lower_load_ssa]=lower_load_ssa

          if lower_bound_val is not None and upper_bound_val is not None:
            # Constant based on literal dimension size, we know the value so put in directly
            dim_sizes.append((upper_bound_val-lower_bound_val)+1)
            const_op=create_index_constant((upper_bound_val-lower_bound_val)+1)
            ops_list.append(const_op)
            dim_ssas.append(const_op.results[0])
          else:
            if upper_load_ssa is None:
              upper_const_op=create_index_constant(upper_bound_val)
              ops_list.append(upper_const_op)
              upper_load_ssa=upper_const_op.results[0]
            if lower_load_ssa is None:
              lower_const_op=create_index_constant(lower_bound_val)
              ops_list.append(lower_const_op)
              lower_load_ssa=lower_const_op.results[0]

            one_const_op=create_index_constant(1)
            sub_op=arith.Subi(upper_load_ssa, lower_load_ssa)
            add_op=arith.Addi(sub_op, one_const_op)
            ops_list+=[one_const_op, sub_op, add_op]
            dim_ssas.append(add_op.results[0])
            dim_sizes.append(add_op.results[0])

        elif isa(size_op, fir.Load):
          # Sized based off a variable rather than constant, therefore need
          # to load this and convert to an index if it is an integer
          load_ssa, load_ops=generate_var_dim_size_load(ctx, size_op)
          ops_list+=load_ops
          dim_ssas.append(load_ssa)
          dim_sizes.append(load_ssa)
        else:
          assert False

      if default_start_idx_mode:
        assert len(dim_starts) == 0
        dim_starts=[1]*len(dim_sizes)
        dim_ends=dim_sizes

      assert len(dim_sizes) == len(dim_starts) == len(dim_ends)

      # Now create memref, passing -1 as shape will make this deferred size
      # Reverse the indicies as Fortran and C/MLIR are opposite in terms of
      # the order of the contiguous dimension (F is least, whereas C/MLIR is highest)
      dim_ssa_reversed=dim_ssas.copy()
      dim_ssa_reversed.reverse()
      memref_allocation_op=memref_alloca_op=memref.Alloc.get(base_type, shape=[-1]*len(dim_ssas), dynamic_sizes=dim_ssa_reversed)
      ops_list.append(memref_allocation_op)

      store_op=memref.Store.get(memref_allocation_op.results[0], ctx[op.memref.owner.results[0]], [])
      ops_list.append(store_op)
      #ctx[op.memref.owner.results[0]]=memref_allocation_op.results[0]
      #ctx[op.memref.owner.results[1]]=memref_allocation_op.results[0]

      fn_name=program_state.getCurrentFnState().fn_name
      array_name=op.memref.owner.uniq_name.data
      # Store information about the array - the size, and lower and upper bounds as we need this when accessing elements
      program_state.getCurrentFnState().array_info[array_name]=ArrayDescription(array_name, dim_sizes, dim_starts, dim_ends)

      return ops_list
    elif isa(op.value.owner.memref.owner, fir.ZeroBits):
      pass
    else:
      assert False

  else:
    expr_ops=translate_expr(program_state, ctx, op.value)

    if isa(op.memref.owner, hlfir.DeclareOp):
      storage_op=memref.Store.get(ctx[op.value], ctx[op.memref], [])
    elif isa(op.memref.owner, fir.Alloca):
      assert ctx[op.memref] is not None
      storage_op=memref.Store.get(ctx[op.value], ctx[op.memref], [])
    else:
      assert False
    return expr_ops+[storage_op]

def array_access_components(program_state: ProgramState, ctx: SSAValueCtx, op:hlfir.DesignateOp):
  # This will generate the required operations and SSA for index accesses to an array, whether
  # this is storage or loading in the wider context. It will offset depending upon the logical
  # start index to the physical start index of 0
  if isa(op.memref.owner, hlfir.DeclareOp):
    array_name=op.memref.owner.uniq_name.data
  elif isa(op.memref.owner, fir.Load):
    assert isa(op.memref.owner.memref.owner, hlfir.DeclareOp)
    array_name=op.memref.owner.memref.owner.uniq_name.data
  else:
    assert False

  ops_list=[]
  indexes_ssa=[]
  for idx, index in enumerate(op.indices):
    ops=translate_expr(program_state, ctx, index)
    ops_list+=ops
    if not isa(ctx[index].type, builtin.IndexType):
      assert isa(ctx[index].type, builtin.IntegerType)
      convert_op=arith.IndexCastOp(ctx[index], builtin.IndexType())
      ops_list.append(convert_op)
      index_ssa=convert_op.results[0]
    else:
      index_ssa=ctx[index]
    dim_start=program_state.getCurrentFnState().array_info[array_name].dim_starts[idx]
    if isa(dim_start, int):
      assert dim_start >= 0
      if dim_start > 0:
        # If zero start then we are good, otherwise need to zero index this
        offset_const=arith.Constant.create(properties={"value": builtin.IntegerAttr.from_index_int_value(dim_start)},
                                             result_types=[builtin.IndexType()])
        subtract_op=arith.Subi(index_ssa, offset_const)
        ops_list+=[offset_const, subtract_op]
        indexes_ssa.append(subtract_op.results[0])
      else:
        indexes_ssa.append(index_ssa)
    elif isa(dim_start, OpResult):
      # This is not a constant literal in the code, therefore use the variable that drives this
      # which was generated previously, so just link to this
      assert ctx[dim_start] is not None
      subtract_op=arith.Subi(index_ssa, ctx[dim_start])
      ops_list.append(subtract_op)
      indexes_ssa.append(subtract_op.results[0])
    else:
      assert False
  return ops_list, indexes_ssa

def translate_assign(program_state: ProgramState, ctx: SSAValueCtx, op: hlfir.AssignOp):
  expr_lhs_ops=translate_expr(program_state, ctx, op.lhs)
  if isa(op.rhs.owner, hlfir.DeclareOp):
    expr_rhs_ops=translate_expr(program_state, ctx, op.rhs)
    # Scalar value or assign entire array to another
    if isa(ctx[op.rhs].type, memref.MemRefType):
      if (isa(op.rhs.owner.results[0].type, fir.BoxType) or (isa(op.rhs.owner.results[0].type, fir.ReferenceType)
              and isa(op.rhs.owner.results[0].type.type, fir.BoxType))):
        # This is an array, we must be assigning one array to another, the type will
        # be fir.ref<fir.box<fir.heap<fir.array<..>>>> or fir.box<fir.array<...>>
        if isa(op.rhs.owner.results[0].type, fir.ReferenceType):
          # Of the form fir.ref<fir.box<fir.heap<fir.array<..>>>>
          assert isa(op.rhs.owner.results[0].type.type.type, fir.HeapType)
          assert isa(op.rhs.owner.results[0].type.type.type.type, fir.SequenceType)
          assert isa(op.lhs.owner, fir.Load)
          assert isa(op.lhs.owner.results[0].type, fir.BoxType)
          assert isa(op.lhs.owner.results[0].type.type, fir.HeapType)
          assert isa(op.lhs.owner.results[0].type.type.type, fir.SequenceType)
          lhs_array_op=op.rhs.owner.results[0].type.type.type.type
          rhs_array_op=op.lhs.owner.results[0].type.type.type
        else:
          # Of the form fir.box<fir.array<...>>
          assert isa(op.rhs.owner.results[0].type.type, fir.SequenceType)
          assert isa(op.lhs.owner.results[0].type.type, fir.SequenceType)
          lhs_array_op=op.rhs.owner.results[0].type.type
          rhs_array_op=op.rhs.owner.results[0].type.type

        # Check number of dimensions is the same
        lhs_dims=lhs_array_op.shape
        rhs_dims=rhs_array_op.shape
        assert len(lhs_dims) == len(rhs_dims)

        # Check the base type is the same
        assert lhs_array_op.type == rhs_array_op.type
        # We don't check the array sizes are the same, probably should but might need to be dynamic
        if isa(ctx[op.lhs].type.element_type, memref.MemRefType):
          load_op, lhs_load_ssa=generate_dereference_memref(ctx[op.lhs])
          expr_lhs_ops.append(load_op)
        else:
          lhs_load_ssa=ctx[op.lhs]

        if isa(ctx[op.rhs].type.element_type, memref.MemRefType):
          load_op, rhs_load_ssa=generate_dereference_memref(ctx[op.rhs])
          expr_rhs_ops.append(load_op)
        else:
          rhs_load_ssa=ctx[op.rhs]

        copy_op=memref.CopyOp(lhs_load_ssa, rhs_load_ssa)
        return expr_lhs_ops+expr_rhs_ops+[copy_op]
      else:
        assert isa(op.rhs.owner.results[0].type, fir.ReferenceType)

        storage_op=memref.Store.get(ctx[op.lhs], ctx[op.rhs], [])
        return expr_lhs_ops+expr_rhs_ops+[storage_op]
    elif isa(ctx[op.rhs].type, llvm.LLVMPointerType):
      storage_op=llvm.StoreOp(ctx[op.lhs], ctx[op.rhs])
      return expr_lhs_ops+expr_rhs_ops+[storage_op]
    else:
      assert False
  elif isa(op.rhs.owner, hlfir.DesignateOp):
    # Array value
    assert op.rhs.owner.indices is not None
    ops_list, indexes_ssa=array_access_components(program_state, ctx, op.rhs.owner)
    if isa(op.rhs.owner.memref.owner, hlfir.DeclareOp):
      memref_reference=op.rhs.owner.memref
    elif isa(op.rhs.owner.memref.owner, fir.Load):
      memref_reference=op.rhs.owner.memref.owner.memref
    else:
      assert False

    assert isa(ctx[memref_reference].type, memref.MemRefType)
    if isa(ctx[memref_reference].type.element_type, memref.MemRefType):
      load_op, load_ssa=generate_dereference_memref(ctx[memref_reference])
      ops_list.append(load_op)
    else:
      load_ssa=ctx[memref_reference]
    # Reverse the indicies as Fortran and C/MLIR are opposite in terms of
    # the order of the contiguous dimension (F is least, whereas C/MLIR is highest)
    indexes_ssa_reversed=indexes_ssa.copy()
    indexes_ssa_reversed.reverse()
    storage_op=memref.Store.get(ctx[op.lhs], load_ssa, indexes_ssa_reversed)
    ops_list.append(storage_op)
    return expr_lhs_ops+ops_list
  else:
    assert False

def translate_constant(program_state: ProgramState, ctx: SSAValueCtx, op: arith.Constant):
  if ctx.contains(op.results[0]): return []
  new_const=arith.Constant(op.value, op.results[0].type)
  ctx[op.results[0]]=new_const.results[0]
  return [new_const]

def handle_call_argument(program_state: ProgramState, ctx: SSAValueCtx, fn_name: str, arg: Operand, arg_index: int):
  if isa(arg.owner, hlfir.AssociateOp):
    # This is a scalar that we are passing by value
    assert arg.owner.uniq_name.data=="adapt.valuebyref"
    arg_defn=program_state.function_definitions[fn_name].args[arg_index]
    # For now we just work with scalars here, could pass arrays by literal too
    assert arg_defn.is_scalar
    if arg_defn.is_scalar and arg_defn.intent == ArgIntent.IN:
      # This is a scalar with intent in, therefore just pass the constant
      ops_list=translate_expr(program_state, ctx, arg.owner.source)
      ctx[arg]=ctx[arg.owner.source]
      return ops_list, False
    else:
      # Otherwise we need to pack the constant into a memref and pass this
      assert isa(arg_defn.arg_type, fir.ReferenceType)
      ops_list=translate_expr(program_state, ctx, arg.owner.source)
      memref_alloca_op=memref.Alloca.get(convert_fir_type_to_standard(arg_defn.arg_type.type), shape=[])

      storage_op=memref.Store.get(ctx[arg.owner.source], memref_alloca_op.results[0], [])
      ctx[arg]=memref_alloca_op.results[0]
      return ops_list+[memref_alloca_op, storage_op], True
  else:
    # Here passing a variable (array or scalar variable). This is a little confusing, as we
    # allow the translate_expr to handle it, but if the function accepts an integer due to
    # scalar and intent(in), then we need to load the memref.
    ops_list=translate_expr(program_state, ctx, arg)
    if not program_state.function_definitions[fn_name].is_definition_only:
      arg_defn=program_state.function_definitions[fn_name].args[arg_index]
      if arg_defn.is_scalar and arg_defn.intent == ArgIntent.IN and isa(ctx[arg].type, memref.MemRefType):
        # The function will accept a constant, but we are currently passing a memref
        # therefore need to load the value and pass this

        load_op=memref.Load.get(ctx[arg], [])

        # arg is already in our ctx from above, so remove it and add in the load as
        # we want to reference that instead
        del ctx[arg]
        ctx[arg]=load_op.results[0]
        ops_list+=[load_op]
      elif not arg_defn.is_scalar and not arg_defn.is_allocatable and isa(ctx[arg].type, memref.MemRefType) and isa(ctx[arg].type.element_type, memref.MemRefType):
        load_op=memref.Load.get(ctx[arg], [])
        del ctx[arg]
        ctx[arg]=load_op.results[0]
        ops_list+=[load_op]
    return ops_list, False

def handle_movealloc_intrinsic_call(program_state: ProgramState, ctx: SSAValueCtx, op: fir.Call):
  src_ssa=op.args[1]
  dst_ssa=op.args[0]

  src_list=translate_expr(program_state, ctx, src_ssa)
  dst_list=translate_expr(program_state, ctx, dst_ssa)

  assert isa(ctx[src_ssa].type, memref.MemRefType)
  assert isa(ctx[dst_ssa].type, memref.MemRefType)
  load_op=memref.Load.get(ctx[src_ssa], [])
  store_op=memref.Store.get(load_op.results[0], ctx[dst_ssa], [])
  return [load_op, store_op]

def translate_call(program_state: ProgramState, ctx: SSAValueCtx, op: fir.Call):
  if len(op.results) > 0 and ctx.contains(op.results[0]): return []

  fn_name=clean_func_name(op.callee.string_value())

  if fn_name in FortranIntrinsicsHandleExplicitly.keys():
    return FortranIntrinsicsHandleExplicitly[fn_name](program_state, ctx, op)

  # Create a new context scope, as we might overwrite some SSAs with packing
  call_ctx=SSAValueCtx(ctx)

  arg_ops=[]
  are_temps_allocated=False
  for idx, arg in enumerate(op.args):
    # This is more complex, as constants are passed directly or packed in a temporary
    specific_arg_ops, temporary_allocated=handle_call_argument(program_state, call_ctx, fn_name, arg, idx)
    if temporary_allocated: are_temps_allocated=True
    arg_ops+=specific_arg_ops

  arg_ssa=[]
  for arg in op.args:
    arg_ssa.append(call_ctx[arg])

  return_types=[]
  return_ssas=[]
  for ret in op.results:
    # Ignore none types, these are just omitted
    if not isa(ret.type, builtin.NoneType):
      return_types.append(convert_fir_type_to_standard_if_needed(ret.type))
      return_ssas.append(ret)

  call_op=func.Call(op.callee, arg_ssa, return_types)

  if program_state.function_definitions[fn_name].is_definition_only and not are_temps_allocated:
    for idx, ret_ssa in enumerate(return_ssas):
      ctx[ret_ssa]=call_op.results[idx]
    return arg_ops+[call_op]
  else:
    alloc_scope_return=memref.AllocaScopeReturnOp.build(operands=[call_op.results])
    alloca_scope_op=memref.AllocaScopeOp.build(regions=[Region([Block(arg_ops+[call_op, alloc_scope_return])])], result_types=[return_types])

    for idx, ret_ssa in enumerate(return_ssas):
      ctx[ret_ssa]=alloca_scope_op.results[idx]

    return [alloca_scope_op]

def translate_iterate_while(program_state: ProgramState, ctx: SSAValueCtx, op: fir.IterateWhile):
  # FIR's iterate while is like the do loop as it has a numeric counter but it also has an i1
  # flag to drive whether to continue (flag=true) or exit (flag=false). We map this to scf.While
  # however it's a bit of a different operation, so we need to more manual things here to achieve this
  if ctx.contains(op.results[1]):
    for fir_result in op.results[1:]:
      assert ctx.contains(fir_result)
    return []

  lower_bound_ops=translate_expr(program_state, ctx, op.lowerBound)
  upper_bound_ops=translate_expr(program_state, ctx, op.upperBound)
  step_ops=translate_expr(program_state, ctx, op.step)
  iterate_in_ops=translate_expr(program_state, ctx, op.iterateIn)
  initarg_ops=translate_expr(program_state, ctx, op.initArgs)

  zero_const=create_index_constant(0)
  # Will be true if smaller than zero, this is needed because if counting backwards
  # then check it is larger or equal to the upper bound, otherwise it's smaller or equals
  step_op_lt_zero=arith.Cmpi(ctx[op.step], zero_const, 2)
  step_zero_check_ops=[zero_const, step_op_lt_zero]

  assert len(op.regions)==1
  assert len(op.regions[0].blocks)==1

  arg_types=[builtin.IndexType(), builtin.i1, ctx[op.initArgs].type]

  before_block = Block(arg_types=arg_types)

  # Build the index check, the one to use depends on whether the step is positive or not
  true_cmp_op=arith.Cmpi(before_block.args[0], ctx[op.upperBound], 5)
  true_cmp_block=[true_cmp_op, scf.Yield(true_cmp_op)]
  false_cmp_op=arith.Cmpi(before_block.args[0], ctx[op.upperBound], 3)
  false_cmp_block=[false_cmp_op, scf.Yield(false_cmp_op)]
  index_comparison=scf.If(step_op_lt_zero, builtin.i1, true_cmp_block, false_cmp_block)

  # True if both are true, false otherwise (either the counter or bool can quit out of loop)
  or_comparison=arith.AndI(index_comparison, before_block.args[1])
  condition_op=scf.Condition(or_comparison, *before_block.args)

  before_block.add_ops([index_comparison, or_comparison, condition_op])

  after_block = Block(arg_types=arg_types)

  for fir_arg, std_arg in zip(op.regions[0].blocks[0].args, after_block.args):
    ctx[fir_arg]=std_arg

  loop_body_ops=[]
  for loop_op in op.regions[0].blocks[0].ops:
    loop_body_ops+=translate_stmt(program_state, ctx, loop_op)

  # This updates the loop counter
  update_loop_idx=arith.Addi(after_block.args[0], ctx[op.step])

  # Now we grab out the loop update to return, the true or false flag, and the initarg
  yield_op=loop_body_ops[-1]
  assert isa(yield_op, scf.Yield)
  ssa_args=[update_loop_idx.results[0], yield_op.arguments[0], after_block.args[2]]

  # Rebuilt yield with these new SSAs
  new_yieldop=scf.Yield(*ssa_args)
  del loop_body_ops[-1]
  # Add new yield and the update loop counter to the block
  after_block.add_ops(loop_body_ops+[update_loop_idx, new_yieldop])

  while_return_types=[builtin.IndexType(), builtin.i1, ctx[op.initArgs]]

  scf_while_loop=scf.While([ctx[op.lowerBound], ctx[op.iterateIn], ctx[op.initArgs]], arg_types, [before_block], [after_block])

  # It is correct to have them this way round, in fir it's i1, index whereas here
  # we have index, i1, index (and we ignore the last index)
  ctx[op.results[0]]=scf_while_loop.results[1]
  ctx[op.results[1]]=scf_while_loop.results[0]

  return lower_bound_ops+upper_bound_ops+step_ops+step_zero_check_ops+iterate_in_ops+initarg_ops+[scf_while_loop]

def get_loop_arg_val_if_known(ssa):
  # Grabs out the value corresponding the to loops input SSA
  # if this can be found statically, otherwise return None
  ssa_base=ssa
  if isa(ssa_base.owner, arith.IndexCastOp):
    ssa_base=ssa_base.owner.input

  if isa(ssa_base.owner, arith.Constant):
    assert isa(ssa_base.type, builtin.IndexType) or isa(ssa_base.type, builtin.IntegerType)
    val=ssa_base.owner.value.value.data
    return val
  else:
    return None

def determine_loop_step_direction(step_ssa):
  # Determines the loop step direction
  step_val=get_loop_arg_val_if_known(step_ssa)
  if step_val is not None:
    if step_val > 0:
      return LoopStepDirection.INCREMENT
    else:
      return LoopStepDirection.DECREMENT
  else:
    return LoopStepDirection.UNKNOWN

def generate_index_inversion_at_start_of_loop(index_ssa, lower_ssa, upper_ssa, target_type):
  # This is the index inversion required at the start of the loop if working backwards
  inversion_ops=[]
  reduce_idx_from_start=arith.Subi(index_ssa, lower_ssa)
  invert_idx=arith.Subi(upper_ssa, reduce_idx_from_start)
  inversion_ops+=[reduce_idx_from_start, invert_idx]
  if isa(target_type, builtin.IntegerType):
    index_cast=arith.IndexCastOp(invert_idx.results[0], target_type)
    inversion_ops.append(index_cast)
  return inversion_ops

def generate_convert_step_to_absolute(step_ssa):
  # Generates the MLIR to convert an index ssa to it's absolute (positive) form
  assert isa(step_ssa.type, builtin.IndexType)
  cast_int=arith.IndexCastOp(step_ssa, builtin.i64)
  step_absolute=math.AbsIOp(cast_int.results[0])
  cast_abs=arith.IndexCastOp(step_absolute.results[0], builtin.IndexType())
  return [cast_int, step_absolute, cast_abs], cast_abs.results[0]

def translate_do_loop(program_state: ProgramState, ctx: SSAValueCtx, op: fir.DoLoop):
  if ctx.contains(op.results[1]):
    for fir_result in op.results[1:]:
      assert ctx.contains(fir_result)
    return []

  lower_bound_ops=translate_expr(program_state, ctx, op.lowerBound)
  upper_bound_ops=translate_expr(program_state, ctx, op.upperBound)
  step_ops=translate_expr(program_state, ctx, op.step)
  initarg_ops=translate_expr(program_state, ctx, op.initArgs)

  lower_bound=ctx[op.lowerBound]
  upper_bound=ctx[op.upperBound]

  assert len(op.regions)==1
  assert len(op.regions[0].blocks)==1

  arg_types=[]
  for arg in op.regions[0].blocks[0].args:
    arg_types.append(arg.type)

  new_block = Block(arg_types=arg_types)

  for fir_arg, std_arg in zip(op.regions[0].blocks[0].args, new_block.args):
    ctx[fir_arg]=std_arg

  # Determine the step direction (increment, decrement or unknown)
  step_direction=determine_loop_step_direction(ctx[op.step])

  loop_body_ops=[]
  if step_direction == LoopStepDirection.DECREMENT:
    # If this is stepping down, then we assume the loop is do high, low, step
    # as this follows Fortran semantics, for instance do 10,1,-1 would count
    # down from 10, whereas do 1,10,-1 would not execute any iterations
    # scf.for always counts up, therefore we assume that it is high to low
    # and these need swapped around
    t=upper_bound
    upper_bound=lower_bound
    lower_bound=t

    # The loop counter is incrementing, we need to invert this based on the upper bound
    # to get the value as if the loop was actually counting downwards. Note that there are
    # two values each iteration, the actual loop index driven by scf.for (which is an index)
    # and an i32 integer which is the index we are updating from one iteration to the next
    # it is the later that is written to the loop variable. This integer value tracks
    # the scf index (at the end of the loop it is incremented based on the index)
    loop_body_ops+=generate_index_inversion_at_start_of_loop(new_block.args[0], lower_bound, upper_bound, op.regions[0].blocks[0].args[1].type)
    del ctx[op.regions[0].blocks[0].args[1]]
    ctx[op.regions[0].blocks[0].args[1]]=loop_body_ops[-1].results[0]

    # The step must be positive
    step_abs_ops,step_abs_ssa=generate_convert_step_to_absolute(ctx[op.step])
    step_ops+=step_abs_ops
    del ctx[op.step]
    ctx[op.step]=step_abs_ssa
  elif step_direction == LoopStepDirection.UNKNOWN:
    # We don't know if the step is positive or not, as this is from an input
    # variable. The same as the above, but support for decrement is driven
    # by conditionals
    loop_one_const=create_index_constant(1)
    loop_idx_cmp=arith.Cmpi(ctx[op.step], loop_one_const, 2) # checking if step is less than 1

    reduction_ops=generate_index_inversion_at_start_of_loop(new_block.args[0], lower_bound, upper_bound, op.regions[0].blocks[0].args[1].type)
    # We are going to wrap this in a conditional, therefore yield the result of the index inversion
    reduction_ops.append(scf.Yield(reduction_ops[-1]))
    # Wrap in a conditional, if so then invert the index, otherwise just send the index through,
    # see above explanation for more details on this step
    scf_if=scf.If(loop_idx_cmp.results[0], [ctx[op.regions[0].blocks[0].args[1]].type], reduction_ops, [scf.Yield(ctx[op.regions[0].blocks[0].args[1]])])
    loop_body_ops+=[loop_one_const, loop_idx_cmp, scf_if]
    del ctx[op.regions[0].blocks[0].args[1]]
    ctx[op.regions[0].blocks[0].args[1]]=scf_if.results[0]

    # This does the swapping between the lower and upper bounds, as above if we count down
    # then indexes will be high to low, i.e. do high, low, step but these need swapped
    # for the scf loop. This is driven by the conditional on the step
    outer_const=create_index_constant(1)
    outer_idx_cmp=arith.Cmpi(ctx[op.step], outer_const, 2)
    nscf_if=scf.If(outer_idx_cmp.results[0], [upper_bound.type, lower_bound.type], [scf.Yield(upper_bound, lower_bound)], [scf.Yield(lower_bound, upper_bound)])
    lower_bound=nscf_if.results[0]
    upper_bound=nscf_if.results[1]
    initarg_ops+=[outer_const, outer_idx_cmp, nscf_if]

    # Regardless of whether the step is positive or not then we convert
    # it to positive, as if it is already then it doesn't change anything
    # and it's not worth the conditional check
    step_abs_ops,step_abs_ssa=generate_convert_step_to_absolute(ctx[op.step])
    step_ops+=step_abs_ops
    del ctx[op.step]
    ctx[op.step]=step_abs_ssa

  for loop_op in op.regions[0].blocks[0].ops:
    loop_body_ops+=translate_stmt(program_state, ctx, loop_op)

  # We need to add one to the upper index, as scf.for is not inclusive on the
  # top bound, whereas fir for loops are
  one_val_op=create_index_constant(1)
  add_op=arith.Addi(upper_bound, one_val_op)
  initarg_ops+=[one_val_op, add_op]
  upper_bound=add_op.results[0]

  # The fir result has both the index and iterargs, whereas the yield has only
  # the iterargs. Therefore need to rebuild the yield with the first argument (the index)
  # removed from it
  yield_op=loop_body_ops[-1]
  assert isa(yield_op, scf.Yield)
  new_yieldop=scf.Yield(*yield_op.arguments[1:])
  del loop_body_ops[-1]
  loop_body_ops.append(new_yieldop)

  new_block.add_ops(loop_body_ops)

  scf_for_loop=scf.For(lower_bound, upper_bound, ctx[op.step], [ctx[op.initArgs]], new_block)

  for index, scf_result in enumerate(scf_for_loop.results):
    ctx[op.results[index+1]]=scf_result

  return lower_bound_ops+upper_bound_ops+step_ops+initarg_ops+[scf_for_loop]

def translate_return(program_state: ProgramState, ctx: SSAValueCtx, op: func.Return):
  ssa_to_return=[]
  args_ops=[]
  for arg in op.arguments:
    args_ops+=translate_expr(program_state, ctx, arg)
    ssa_to_return.append(ctx[arg])
  new_return=func.Return(*ssa_to_return)
  return args_ops+[new_return]

def translate_alloca(program_state: ProgramState, ctx: SSAValueCtx, op: fir.Alloca):
  if ctx.contains(op.results[0]): return []

  # If any use of the result is the declareop, then ignore this as
  # we will handle it elsewhere - otherwise this is used internally
  for use in op.results[0].uses:
    if isa(use.operation, hlfir.DeclareOp): return[]

  memref_alloca_op=memref.Alloca.get(convert_fir_type_to_standard(op.in_type), shape=[])
  ctx[op.results[0]]=memref_alloca_op.results[0]
  return [memref_alloca_op]

def translate_declare(program_state: ProgramState, ctx: SSAValueCtx, op: hlfir.DeclareOp):
  # If already seen then simply ignore
  if ctx.contains(op.results[0]): return []

  if isa(op.results[0].type, fir.ReferenceType) and isa(op.results[0].type.type, fir.BoxType):
    # This is an allocatable array, we will handle this on the allocation
    assert isa(op.results[0].type.type.type, fir.HeapType)
    assert isa(op.results[0].type.type.type.type, fir.SequenceType)
    num_dims=len(op.results[0].type.type.type.type.shape)
    alloc_memref_container=memref.Alloca.get(memref.MemRefType(op.results[0].type.type.type.type.type, shape=num_dims*[-1]), shape=[])
    ctx[op.results[0]]=alloc_memref_container.results[0]
    ctx[op.results[1]]=alloc_memref_container.results[0]
    return [alloc_memref_container]

  if len(op.results[0].uses) == 0 and len(op.results[1].uses) == 0:
    # Some declare ops are never actually used in the code, Flang seems to generate
    # the declare for global arrays regardless in some functions and therefore
    # we ignore them if the declare doesn't have any uses
    return []

  # Passing an allocatable with annonymous dimension e.g. memref<?,i32> it doesn't know the size

  if op.shape is None and not isa(op.results[0].type, fir.BoxType):
    # Ensure it doesn't have a shape, and there isn't a boxtype (which caries the shape)
    # this means that it is a scalar (TODO: could also check the inner type)
    return define_scalar_var(program_state, ctx, op)
  else:
    if op.shape is not None:
      # There is a shape we can use in determining the size
      if isa(op.shape.owner, fir.Shape):
        dim_sizes, dim_starts, dim_ends=gather_static_shape_dims_from_shape(op.shape.owner)
      elif isa(op.shape.owner, fir.ShapeShift):
        dim_sizes, dim_starts, dim_ends=gather_static_shape_dims_from_shapeshift(op.shape.owner)
      else:
        assert False

      static_size=dims_has_static_size(dim_sizes)
      if static_size:
        return define_stack_array_var(program_state, ctx, op, dim_sizes, dim_starts, dim_ends)
      elif isa(op.memref, BlockArgument) and isa(op.results[1].type, fir.ReferenceType):
        # This is an array passed into a function
        shape_expr_list=[]
        for ds in dim_starts:
          if not isa(ds, int) and ds is not None:
            shape_expr_list+=translate_expr(program_state, ctx, ds)
        ctx[op.results[0]]=ctx[op.memref]
        ctx[op.results[1]]=ctx[op.memref]
        array_name=op.uniq_name.data
        # Store information about the array - the size, and lower and upper bounds as we need this when accessing elements
        program_state.getCurrentFnState().array_info[array_name]=ArrayDescription(array_name, dim_sizes, dim_starts, dim_ends)
        return shape_expr_list
      else:
        assert False
    else:
      # There is no shape, we need to grab this from the memref using operations
      assert isa(op.memref, BlockArgument)
      assert isa(op.results[0].type, fir.BoxType) and isa(op.results[0].type.type, fir.SequenceType)
      num_dims=len(op.results[0].type.type.shape)

      one_op=create_index_constant(1)
      ops_list=[one_op]
      size_ssas=[]
      end_ssas=[]
      for dim in range(num_dims):
        dim_idx_op=create_index_constant(dim)
        get_dim_op=memref.Dim.from_source_and_index(ctx[op.memref], dim_idx_op)
        add_arith_op=arith.Addi(get_dim_op.results[0], one_op.results[0])
        ops_list+=[dim_idx_op, get_dim_op, add_arith_op]
        size_ssas.append(get_dim_op.results[0])
        end_ssas.append(add_arith_op.results[0])
      array_name=op.uniq_name.data
      ctx[op.results[0]]=ctx[op.memref]
      ctx[op.results[1]]=ctx[op.memref]
      program_state.getCurrentFnState().array_info[array_name]=ArrayDescription(array_name, size_ssas, [1]*len(size_ssas), end_ssas)
      return ops_list


def dims_has_static_size(dims):
  for dim in dims:
    if not isa(dim, int): return False
  return True

def gather_static_shape_dims_from_shape(shape_op: fir.Shape):
  # fir.Shape is for default, 1 indexed arrays
  dim_sizes=[]
  dim_starts=[]
  assert shape_op.extents is not None
  for extent in shape_op.extents:
    if isa(extent.owner, arith.Constant):
      assert isa(extent.owner.result.type, builtin.IndexType)
      dim_sizes.append(extent.owner.value.value.data)
    else:
      dim_sizes.append(None)
  dim_starts=[1]*len(dim_sizes)
  return dim_sizes, dim_starts, dim_sizes

def gather_static_shape_dims_from_shapeshift(shape_op: fir.ShapeShift):
  # fir.ShapeShift is for arrays indexed on a value other than 1
  dim_sizes=[]
  dim_starts=[]
  dim_ends=[]
  assert shape_op.pairs is not None
  # Now iterate in pairs of low, high e.g. (low, high), (low, high) etc
  paired_vals=list(zip(shape_op.pairs[::2], shape_op.pairs[1::2]))
  for low_arg, high_arg in paired_vals:
    if isa(low_arg.owner, arith.Constant):
      assert isa(low_arg.owner.result.type, builtin.IndexType)
      dim_starts.append(low_arg.owner.value.value.data)
    else:
      dim_starts.append(low_arg)

    if isa(high_arg.owner, arith.Constant):
      assert isa(high_arg.owner.result.type, builtin.IndexType)
      dim_sizes.append(high_arg.owner.value.value.data)
    else:
      dim_sizes.append(high_arg)

    if dim_starts[-1] is int and dim_sizes[-1] is int:
      dim_ends.append((dim_sizes[-1]+dim_starts[-1])-1)
    else:
      dim_ends.append(None)

  return dim_sizes, dim_starts, dim_ends

def define_stack_array_var(program_state: ProgramState, ctx: SSAValueCtx,
      op: hlfir.DeclareOp, dim_sizes: list, dim_starts: list, dim_ends: list):
  if ctx.contains(op.results[0]):
    assert ctx.contains(op.results[1])
    return []

  # It might be one of tree things - allocated from a global, an alloca in fir
  # for stack local variable, or an array function argument that is statically sized
  assert (isa(op.memref.owner, fir.AddressOf) or isa(op.memref.owner, fir.Alloca) or
          (op.memref.owner, Block))

  if isa(op.memref.owner, Block):
    # It is an array fn argument that is statically sized
    fir_array_type=op.memref.type
  else:
    # Global or fn stack local variab;e
    fir_array_type=op.memref.owner.results[0].type

  assert isa(fir_array_type, fir.ReferenceType)
  fir_array_type=fir_array_type.type

  assert isa(fir_array_type, fir.SequenceType)
  # Ensure collected dimensions and the addressof type dimensions are consistent
  for type_size, dim_size in zip(fir_array_type.shape, dim_sizes):
    assert isa(type_size.type, builtin.IntegerType)
    assert type_size.value.data == dim_size

  array_name=op.uniq_name.data
  # Store information about the array - the size, and lower and upper bounds as we need this when accessing elements
  program_state.getCurrentFnState().array_info[array_name]=ArrayDescription(array_name, dim_sizes, dim_starts, dim_ends)

  if isa(op.memref.owner, Block):
    # This is a statically sized array passed to the function
    # so just point to this
    ctx[op.results[0]] = ctx[op.memref]
    ctx[op.results[1]] = ctx[op.memref]
    return []
  elif isa(op.memref.owner, fir.AddressOf):
    # This is looking up a global array, we need to construct the memref from this
    addr_lookup=llvm.AddressOfOp(op.memref.owner.symbol, llvm.LLVMPointerType.opaque())
    ops_list, ssa=generate_memref_from_llvm_ptr(addr_lookup.results[0], dim_sizes, fir_array_type.type)
    ctx[op.results[0]] = ssa
    ctx[op.results[1]] = ssa
    return [addr_lookup]+ops_list
  elif isa(op.memref.owner, fir.Alloca):
    # Issue an allocation on the stack
    # Reverse the indicies as Fortran and C/MLIR are opposite in terms of
    # the order of the contiguous dimension (F is least, whereas C/MLIR is highest)
    dim_sizes_reversed=dim_sizes.copy()
    dim_sizes_reversed.reverse()
    memref_alloca_op=memref.Alloca.get(convert_fir_type_to_standard(fir_array_type.type), shape=dim_sizes_reversed)
    ctx[op.results[0]] = memref_alloca_op.results[0]
    ctx[op.results[1]] = memref_alloca_op.results[0]
    return [memref_alloca_op]
  else:
    assert False

def generate_memref_from_llvm_ptr(llvm_ptr_in_ssa, dim_sizes, target_type):
  # Builds a memref from an LLVM pointer. This is required if we are working with
  # global arrays, as they are llvm.array, and the pointer is grabbed from that and
  # then the memref constructed
  ptr_type=llvm.LLVMPointerType.opaque()

  offsets=[1]
  if len(dim_sizes) > 1:
    for d in dim_sizes[:-1]:
      offsets.append(d * offsets[-1])

  offsets.reverse()

  # Reverse the indicies as Fortran and C/MLIR are opposite in terms of
  # the order of the contiguous dimension (F is least, whereas C/MLIR is highest)
  dim_sizes=dim_sizes.copy()
  dim_sizes.reverse()

  array_type=llvm.LLVMArrayType.from_size_and_type(builtin.IntAttr(len(dim_sizes)), builtin.i64)
  struct_type=llvm.LLVMStructType.from_type_list([ptr_type, ptr_type, builtin.i64, array_type, array_type])

  undef_memref_struct_op=llvm.UndefOp.create(result_types=[struct_type])
  insert_alloc_ptr_op=llvm.InsertValueOp.create(properties={"position":  builtin.DenseArrayBase.from_list(builtin.i64, [0])},
    operands=[undef_memref_struct_op.results[0], llvm_ptr_in_ssa], result_types=[struct_type])
  insert_aligned_ptr_op=llvm.InsertValueOp.create(properties={"position":  builtin.DenseArrayBase.from_list(builtin.i64, [1])},
    operands=[insert_alloc_ptr_op.results[0], llvm_ptr_in_ssa], result_types=[struct_type])

  offset_op=arith.Constant.from_int_and_width(0, 64)
  insert_offset_op=llvm.InsertValueOp.create(properties={"position":  builtin.DenseArrayBase.from_list(builtin.i64, [2])},
    operands=[insert_aligned_ptr_op.results[0], offset_op.results[0]], result_types=[struct_type])

  ops_to_add=[undef_memref_struct_op, insert_alloc_ptr_op, insert_aligned_ptr_op, offset_op, insert_offset_op]

  for idx, dim in enumerate(dim_sizes):
    size_op=arith.Constant.from_int_and_width(dim, 64)
    insert_size_op=llvm.InsertValueOp.create(properties={"position":  builtin.DenseArrayBase.from_list(builtin.i64, [3, idx])},
      operands=[ops_to_add[-1].results[0], size_op.results[0]], result_types=[struct_type])

    # One for dimension stride
    stride_op=arith.Constant.from_int_and_width(offsets[idx], 64)
    insert_stride_op=llvm.InsertValueOp.create(properties={"position":  builtin.DenseArrayBase.from_list(builtin.i64, [4, idx])},
      operands=[insert_size_op.results[0], stride_op.results[0]], result_types=[struct_type])

    ops_to_add+=[size_op, insert_size_op, stride_op, insert_stride_op]

  target_memref_type=memref.MemRefType(convert_fir_type_to_standard(target_type), dim_sizes)

  unrealised_conv_cast_op=builtin.UnrealizedConversionCastOp.create(operands=[insert_stride_op.results[0]], result_types=[target_memref_type])
  ops_to_add.append(unrealised_conv_cast_op)
  return ops_to_add, unrealised_conv_cast_op.results[0]

def define_scalar_var(program_state: ProgramState, ctx: SSAValueCtx, op: hlfir.DeclareOp):
  if ctx.contains(op.results[0]):
    assert ctx.contains(op.results[1])
    return []
  if isa(op.memref, OpResult):
    allocation_op=op.memref.owner
    if isa(allocation_op, fir.Alloca):
      assert isa(allocation_op.results[0].type, fir.ReferenceType)
      assert allocation_op.results[0].type.type == allocation_op.in_type
      memref_alloca_op=memref.Alloca.get(convert_fir_type_to_standard(allocation_op.in_type), shape=[])
      ctx[op.results[0]] = memref_alloca_op.results[0]
      ctx[op.results[1]] = memref_alloca_op.results[0]
      return [memref_alloca_op]
    elif isa(allocation_op, fir.AddressOf):
      expr_ops=translate_expr(program_state, ctx, op.memref)
      ctx[op.results[0]] = ctx[allocation_op.results[0]]
      ctx[op.results[1]] = ctx[allocation_op.results[0]]
      return expr_ops
    elif isa(allocation_op, fir.Unboxchar):
      expr_ops=translate_expr(program_state, ctx, allocation_op.boxchar)
      ctx[op.results[0]] = ctx[allocation_op.results[0]]
      ctx[op.results[1]] = ctx[allocation_op.results[0]]
      return expr_ops
    else:
      assert False
  elif isa(op.memref, BlockArgument):
    ctx[op.results[0]] = ctx[op.memref]
    ctx[op.results[1]] = ctx[op.memref]
    return []

def translate_float_unary_arithmetic(program_state: ProgramState, ctx: SSAValueCtx, op: Operation):
  if ctx.contains(op.results[0]): return []
  operand_ops=translate_expr(program_state, ctx, op.operand)
  operand_ssa=ctx[op.operand]
  fast_math_attr=op.fastmath
  result_type=op.results[0].type
  unary_arith_op=None

  if isa(op, arith.Negf):
    unary_arith_op=arith.Negf(operand_ssa, fast_math_attr)
  else:
    raise Exception(f"Could not translate `{op}' as a unary float operation")

  assert unary_arith_op is not None
  ctx[op.results[0]]=unary_arith_op.results[0]
  return operand_ops+[unary_arith_op]

def translate_float_binary_arithmetic(program_state: ProgramState, ctx: SSAValueCtx, op: Operation):
  if ctx.contains(op.results[0]): return []
  lhs_ops=translate_expr(program_state, ctx, op.lhs)
  rhs_ops=translate_expr(program_state, ctx, op.rhs)
  lhs_ssa=ctx[op.lhs]
  rhs_ssa=ctx[op.rhs]
  fast_math_attr=op.fastmath
  result_type=op.results[0].type
  bin_arith_op=None

  if isa(op, arith.Addf):
    bin_arith_op=arith.Addf(lhs_ssa, rhs_ssa, fast_math_attr, result_type)
  elif isa(op, arith.Subf):
    bin_arith_op=arith.Subf(lhs_ssa, rhs_ssa, fast_math_attr, result_type)
  elif isa(op, arith.Mulf):
    bin_arith_op=arith.Mulf(lhs_ssa, rhs_ssa, fast_math_attr, result_type)
  elif isa(op, arith.Divf):
    bin_arith_op=arith.Divf(lhs_ssa, rhs_ssa, fast_math_attr, result_type)
  elif isa(op, arith.Maximumf):
    bin_arith_op=arith.Maximumf(lhs_ssa, rhs_ssa, fast_math_attr, result_type)
  elif isa(op, arith.Maxnumf):
    bin_arith_op=arith.Maxnumf(lhs_ssa, rhs_ssa, fast_math_attr, result_type)
  elif isa(op, arith.Minimumf):
    bin_arith_op=arith.Minimumf(lhs_ssa, rhs_ssa, fast_math_attr, result_type)
  elif isa(op, arith.Minnumf):
    bin_arith_op=arith.Minnumf(lhs_ssa, rhs_ssa, fast_math_attr, result_type)
  else:
    raise Exception(f"Could not translate `{op}' as a binary float operation")

  assert bin_arith_op is not None
  ctx[op.results[0]]=bin_arith_op.results[0]
  return lhs_ops+rhs_ops+[bin_arith_op]

def translate_integer_binary_arithmetic(program_state: ProgramState, ctx: SSAValueCtx, op: Operation):
  if ctx.contains(op.results[0]): return []
  lhs_ops=translate_expr(program_state, ctx, op.lhs)
  rhs_ops=translate_expr(program_state, ctx, op.rhs)
  lhs_ssa=ctx[op.lhs]
  rhs_ssa=ctx[op.rhs]
  bin_arith_op=None

  if isa(op, arith.Addi):
    bin_arith_op=arith.Addi(lhs_ssa, rhs_ssa)
  elif isa(op, arith.Subi):
    bin_arith_op=arith.Subi(lhs_ssa, rhs_ssa)
  elif isa(op, arith.Muli):
    bin_arith_op=arith.Muli(lhs_ssa, rhs_ssa)
  elif isa(op, arith.DivUI):
    bin_arith_op=arith.DivUI(lhs_ssa, rhs_ssa)
  elif isa(op, arith.DivSI):
    bin_arith_op=arith.DivSI(lhs_ssa, rhs_ssa)
  elif isa(op, arith.FloorDivSI):
    bin_arith_op=arith.FloorDivSI(lhs_ssa, rhs_ssa)
  elif isa(op, arith.CeilDivSI):
    bin_arith_op=arith.CeilDivSI(lhs_ssa, rhs_ssa)
  elif isa(op, arith.CeilDivUI):
    bin_arith_op=arith.CeilDivUI(lhs_ssa, rhs_ssa)
  elif isa(op, arith.RemUI):
    bin_arith_op=arith.RemUI(lhs_ssa, rhs_ssa)
  elif isa(op, arith.RemSI):
    bin_arith_op=arith.RemSI(lhs_ssa, rhs_ssa)
  elif isa(op, arith.MinUI):
    bin_arith_op=arith.MinUI(lhs_ssa, rhs_ssa)
  elif isa(op, arith.MaxUI):
    bin_arith_op=arith.MaxUI(lhs_ssa, rhs_ssa)
  elif isa(op, arith.MinSI):
    bin_arith_op=arith.MinSI(lhs_ssa, rhs_ssa)
  elif isa(op, arith.MaxSI):
    bin_arith_op=arith.MaxSI(lhs_ssa, rhs_ssa)
  elif isa(op, arith.AndI):
    bin_arith_op=arith.AndI(lhs_ssa, rhs_ssa)
  elif isa(op, arith.OrI):
    bin_arith_op=arith.OrI(lhs_ssa, rhs_ssa)
  elif isa(op, arith.XOrI):
    bin_arith_op=arith.XOrI(lhs_ssa, rhs_ssa)
  elif isa(op, arith.ShLI):
    bin_arith_op=arith.ShLI(lhs_ssa, rhs_ssa)
  elif isa(op, arith.ShRUI):
    bin_arith_op=arith.ShRUI(lhs_ssa, rhs_ssa)
  elif isa(op, arith.ShRSI):
    bin_arith_op=arith.ShRSI(lhs_ssa, rhs_ssa)
  elif isa(op, arith.AddUIExtended):
    bin_arith_op=arith.AddUIExtended(lhs_ssa, rhs_ssa)
  else:
    raise Exception(f"Could not translate `{op}' as a binary integer operation")

  assert bin_arith_op is not None
  ctx[op.results[0]]=bin_arith_op.results[0]
  return lhs_ops+rhs_ops+[bin_arith_op]

def translate_math_operation(program_state: ProgramState, ctx: SSAValueCtx, op: Operation):
  if ctx.contains(op.results[0]): return []

  expr_ops=[]
  if isa(op, math.FmaOp):
    expr_ops+=translate_expr(program_state, ctx, op.a)
    expr_ops+=translate_expr(program_state, ctx, op.b)
    expr_ops+=translate_expr(program_state, ctx, op.c)
  else:
    if hasattr(op, 'operand'):
      expr_ops+=translate_expr(program_state, ctx, op.operand)
    if hasattr(op, 'lhs'):
      expr_ops+=translate_expr(program_state, ctx, op.lhs)
    if hasattr(op, 'rhs'):
      expr_ops+=translate_expr(program_state, ctx, op.rhs)

  math_op=None
  if isa(op, math.AbsFOp):
    math_op=op.__class__(ctx[op.operand], op.fastmath)
  elif isa(op, math.AbsIOp):
    math_op=op.__class__(ctx[op.operand])
  elif isa(op, math.Atan2Op):
    math_op=op.__class__(ctx[op.lhs], ctx[op.rhs], op.fastmath)
  elif isa(op, math.AtanOp):
    math_op=op.__class__(ctx[op.operand], op.fastmath)
  elif isa(op, math.CbrtOp):
    math_op=op.__class__(ctx[op.operand], op.fastmath)
  elif isa(op, math.CeilOp):
    math_op=op.__class__(ctx[op.operand], op.fastmath)
  elif isa(op, math.CopySignOp):
    math_op=op.__class__(ctx[op.lhs], ctx[op.rhs], op.fastmath)
  elif isa(op, math.CosOp):
    math_op=op.__class__(ctx[op.operand], op.fastmath)
  elif isa(op, math.CountLeadingZerosOp):
    math_op=op.__class__(ctx[op.operand])
  elif isa(op, math.CountTrailingZerosOp):
    math_op=op.__class__(ctx[op.operand])
  elif isa(op, math.CtPopOp):
    math_op=op.__class__(ctx[op.operand])
  elif isa(op, math.ErfOp):
    math_op=op.__class__(ctx[op.operand], op.fastmath)
  elif isa(op, math.Exp2Op):
    math_op=op.__class__(ctx[op.operand], op.fastmath)
  elif isa(op, math.ExpM1Op):
    math_op=op.__class__(ctx[op.operand], op.fastmath)
  elif isa(op, math.ExpOp):
    math_op=op.__class__(ctx[op.operand], op.fastmath)
  elif isa(op, math.FPowIOp):
    math_op=op.__class__(ctx[op.lhs], ctx[op.rhs], op.fastmath)
  elif isa(op, math.FloorOp):
    math_op=op.__class__(ctx[op.operand], op.fastmath)
  elif isa(op, math.FmaOp):
    math_op=op.__class__(ctx[op.operand])
  elif isa(op, math.IPowIOp):
    math_op=op.__class__(ctx[op.lhs], ctx[op.rhs])
  elif isa(op, math.Log10Op):
    math_op=op.__class__(ctx[op.operand], op.fastmath)
  elif isa(op, math.Log1pOp):
    math_op=op.__class__(ctx[op.operand], op.fastmath)
  elif isa(op, math.Log2Op):
    math_op=op.__class__(ctx[op.operand], op.fastmath)
  elif isa(op, math.LogOp):
    math_op=op.__class__(ctx[op.operand], op.fastmath)
  elif isa(op, math.PowFOp):
    math_op=op.__class__(ctx[op.lhs], ctx[op.rhs], op.fastmath)
  elif isa(op, math.RoundEvenOp):
    math_op=op.__class__(ctx[op.operand], op.fastmath)
  elif isa(op, math.RoundOp):
    math_op=op.__class__(ctx[op.operand], op.fastmath)
  elif isa(op, math.RsqrtOp):
    math_op=op.__class__(ctx[op.operand], op.fastmath)
  elif isa(op, math.SinOp):
    math_op=op.__class__(ctx[op.operand], op.fastmath)
  elif isa(op, math.SqrtOp):
    math_op=op.__class__(ctx[op.operand], op.fastmath)
  elif isa(op, math.TanOp):
    math_op=op.__class__(ctx[op.operand], op.fastmath)
  elif isa(op, math.TanhOp):
    math_op=op.__class__(ctx[op.operand], op.fastmath)
  elif isa(op, math.TruncOp):
    math_op=op.__class__(ctx[op.operand], op.fastmath)
  else:
    raise Exception(f"Could not translate `{op}' as a math operation")

  assert math_op is not None
  ctx[op.results[0]]=math_op.results[0]
  return expr_ops+[math_op]

class RewriteRelativeBranch(RewritePattern):
  def __init__(self, functions):
    self.functions=functions

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: ftn_relative_cf.Branch, rewriter: PatternRewriter, /):
    containing_fn_name=op.function_name.data
    if containing_fn_name == "_QQmain": containing_fn_name="main"
    assert containing_fn_name in self.functions.keys()
    assert len(self.functions[containing_fn_name].regions[0].blocks) > op.successor.value.data
    cf_branch_op=cf.Branch(self.functions[containing_fn_name].regions[0].blocks[op.successor.value.data], *op.arguments)
    rewriter.replace_matched_op(cf_branch_op)

class RewriteRelativeConditionalBranch(RewritePattern):
  def __init__(self, functions):
    self.functions=functions

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: ftn_relative_cf.ConditionalBranch, rewriter: PatternRewriter, /):
    containing_fn_name=op.function_name.data
    if containing_fn_name == "_QQmain": containing_fn_name="main"
    assert containing_fn_name in self.functions.keys()
    assert len(self.functions[containing_fn_name].regions[0].blocks) > op.then_block.value.data
    assert len(self.functions[containing_fn_name].regions[0].blocks) > op.else_block.value.data
    cf_cbranch_op=cf.ConditionalBranch(op.cond, self.functions[containing_fn_name].regions[0].blocks[op.then_block.value.data],
                      op.then_arguments, self.functions[containing_fn_name].regions[0].blocks[op.else_block.value.data], op.else_arguments)
    rewriter.replace_matched_op(cf_cbranch_op)

@dataclass(frozen=True)
class RewriteFIRToStandard(ModulePass):
  """
  This is the entry point for the transformation pass which will then apply the rewriter
  """
  name = 'rewrite-fir-to-standard'

  def apply(self, ctx: MLContext, input_module: builtin.ModuleOp):
    program_state=ProgramState()
    fn_visitor=GatherFunctionInformation(program_state)
    fn_visitor.traverse(input_module)
    global_visitor=GatherFIRGlobals(program_state)
    global_visitor.traverse(input_module)
    res_module = translate_program(program_state, input_module)
    res_module.regions[0].move_blocks(input_module.regions[0])

    # Clean out module attributes to remove dlti and fir specific ones
    attr_list=list(input_module.attributes)
    for attr in attr_list:
      if attr.startswith("dlti.") or attr.startswith("fir."):
        del input_module.attributes[attr]

    fn_gatherer=GatherFunctions()
    fn_gatherer.traverse(input_module)


    walker = PatternRewriteWalker(GreedyRewritePatternApplier([
              RewriteRelativeBranch(fn_gatherer.functions),
              RewriteRelativeConditionalBranch(fn_gatherer.functions),
    ]), apply_recursively=False)
    walker.rewrite_module(input_module)

FortranIntrinsicsHandleExplicitly={"_FortranAMoveAlloc" : handle_movealloc_intrinsic_call}

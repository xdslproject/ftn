from abc import ABC
from enum import Enum
import itertools
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
from xdsl.dialects import builtin, func, llvm, arith, memref, scf

ArgIntent = Enum('ArgIntent', ['IN', 'OUT', 'INOUT', 'UNKNOWN'])

class ArrayDescription:
  def __init__(self, name, dim_sizes, dim_starts, dim_ends):
    self.name=name
    self.dim_sizes=dim_sizes
    self.dim_starts=dim_starts
    self.dim_ends=dim_ends

class ComponentState:
  def __init__(self, fn_name=None):
    self.fn_name=fn_name
    self.array_info={}

class ArgumentDefinition:
  def __init__(self, name, is_scalar, arg_type, intent, ):
    self.name=name
    self.is_scalar=is_scalar
    self.intent=intent
    self.arg_type=arg_type

class FunctionDefinition:
  def __init__(self, name, return_type, is_definition_only):
    self.name=name
    self.return_type=return_type
    self.args=[]
    self.is_definition_only=is_definition_only

  def add_arg_def(self, arg_def):
    self.args.append(arg_def)

class ProgramState:
  def __init__(self):
    self.function_definitions={}
    self.global_state=ComponentState()
    self.function_state=None

  def addFunctionDefinition(self, name, fn_def):
    assert name not in self.function_definitions.keys()
    self.function_definitions[name]=fn_def

  def enterFunction(self, fn_name):
    assert self.function_state is None
    self.function_state=ComponentState(fn_name)

  def getCurrentFnState(self):
    assert self.function_state is not None
    return self.function_state

  def leaveFunction(self):
    self.function_state=None

class GatherFunctionInformation(Visitor):
  def __init__(self, program_state):
    self.program_state=program_state

  def get_declare_from_arg_uses(self, arg_uses):
    for use in arg_uses:
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

  def traverse_func_op(self, func_op: func.FuncOp):
    fn_name=func_op.sym_name.data
    if "_QP" in fn_name:
      fn_name=fn_name.split("_QP")[1]
    return_type=None
    if len(func_op.function_type.outputs.data) > 0:
      return_type=func_op.function_type.outputs.data[0]
    fn_def=FunctionDefinition(fn_name, return_type, len(func_op.body.blocks) == 0)
    if len(func_op.body.blocks) != 0:
      # This has concrete implementation (e.g. is not a function definition)
      assert len(func_op.body.blocks) == 1
      for block_arg in func_op.body.blocks[0].args:
        declare_op=self.get_declare_from_arg_uses(block_arg.uses)
        assert declare_op is not None
        is_scalar=declare_op.shape is None
        arg_type=declare_op.results[0].type
        arg_name=declare_op.uniq_name.data
        assert fn_name+"E" in arg_name
        arg_name=arg_name.split(fn_name+"E")[1]
        arg_intent=self.map_ftn_attrs_to_intent(declare_op.fortran_attrs)
        arg_def=ArgumentDefinition(arg_name, is_scalar, arg_type, arg_intent)
        fn_def.add_arg_def(arg_def)
    self.program_state.addFunctionDefinition(fn_name, fn_def)

@dataclass
class SSAValueCtx:
    """
    Context that relates identifiers from the AST to SSA values used in the flat representation.
    """
    dictionary: Dict[str, SSAValue] = field(default_factory=dict)
    parent_scope = None

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

    def contains(self, identifier):
        return identifier in self.dictionary

def translate_program(program_state: ProgramState, input_module: builtin.ModuleOp) -> builtin.ModuleOp:
    # create an empty global context
    global_ctx = SSAValueCtx()
    body = Region()
    block = Block()
    for fn in input_module.ops:
      if isa(fn, func.FuncOp):
        fn_op=translate_function(program_state, global_ctx, fn)
        block.add_op(fn_op)
      elif isa(fn, fir.Global):
        pass
      else:
        assert False
    body.add_block(block)
    return builtin.ModuleOp(body)

def translate_function(program_state: ProgramState, ctx: SSAValueCtx, fn: func.FuncOp):
  fn_name=fn.sym_name.data
  if "_QP" in fn_name:
    fn_name=fn_name.split("_QP")[1]

  body = Region()
  for block in fn.body.blocks:
    arg_types=[]
    for idx, arg in enumerate(fn.args):
      fir_type=arg.type
      if (program_state.function_definitions[fn_name].args[idx].is_scalar and
          program_state.function_definitions[fn_name].args[idx].intent == ArgIntent.IN):
        # This is a scalar in, therefore it's just the constant type (don't encode as a memref)
        if isa(fir_type, fir.ReferenceType):
          arg_types.append(fir_type.type)
        else:
          arg_types.append(arg.type)
      else:
        if isa(fir_type, fir.ReferenceType):
          mrt=memref.MemRefType(fir_type.type, [1], builtin.NoneAttr(), builtin.NoneAttr())
          arg_types.append(mrt)
        else:
          arg_types.append(arg.type)

    new_block = Block(arg_types=arg_types)

    for fir_arg, std_arg in zip(block.args, new_block.args):
      ctx[fir_arg]=std_arg

    program_state.enterFunction(fn_name)
    ops_list=[]
    for op in block.ops:
      ops_list+=translate_stmt(program_state, ctx, op)
    program_state.leaveFunction()

    new_block.add_ops(ops_list)
    body.add_block(new_block)

  fn_name=fn.sym_name

  if fn_name.data == "_QQmain":
    fn_name="main"

  new_fn_type=builtin.FunctionType.from_lists(arg_types, [])

  new_func=func.FuncOp(fn_name, new_fn_type, body, fn.sym_visibility, arg_attrs=fn.arg_attrs, res_attrs=fn.res_attrs)
  return new_func

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
  elif isa(op, fir.Alloca):
    # Ignore this, as will handle with the declaration
    return []
  elif isa(op, arith.Constant):
    return []
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
  elif isa(op, fir.Load):
    return translate_load(program_state, ctx, op)
  elif isa(op, fir.Convert):
    return translate_convert(program_state, ctx, op)
  elif isa(op, fir.DoLoop):
    # Do loop can be either an expression or statement
    return translate_do_loop(program_state, ctx, op)
  elif isa(op, hlfir.DeclareOp):
    return translate_declare(program_state, ctx, op)
  elif isa(op, arith.Cmpi) or isa(op, arith.Cmpf):
    return translate_cmp(program_state, ctx, op)
  else:
    return None

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

  assert new_conv is not None
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
    assert isa(ctx[op.memref].type, memref.MemRefType)
    zero_val=arith.Constant.create(properties={"value": builtin.IntegerAttr.from_index_int_value(0)},
                                             result_types=[builtin.IndexType()])
    load_op=memref.Load.get(ctx[op.memref], [zero_val])
    ctx[op.results[0]]=load_op.results[0]
    return [zero_val, load_op]
  elif isa(op.memref.owner, hlfir.DesignateOp):
    # Array value
    assert op.memref.owner.indices is not None
    assert isa(op.memref.owner.memref.owner, hlfir.DeclareOp)
    ops_list, indexes_ssa=array_access_components(program_state, ctx, op.memref.owner)

    load_op=memref.Load.get(ctx[op.memref.owner.memref], indexes_ssa)
    ops_list.append(load_op)
    ctx[op.results[0]]=load_op.results[0]
    return ops_list
  else:
    assert False

def translate_freemem(program_state: ProgramState, ctx: SSAValueCtx, op: fir.Freemem):
  assert isa(op.heapref.owner, fir.BoxAddr)
  assert isa(op.heapref.owner.val.owner, fir.Load)
  memref_ssa=op.heapref.owner.val.owner.memref
  dealloc_op=memref.Dealloc.get(ctx[memref_ssa])
  return [dealloc_op]

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
  zero_index_op=create_index_constant(0)
  load_op=memref.Load.get(var_ssa, [zero_index_op])
  ops_list=[zero_index_op, load_op]
  if not isa(load_op.results[0].type, builtin.IndexType):
    assert isa(load_op.results[0].type, builtin.IntegerType)
    convert_op=arith.IndexCastOp(load_op.results[0], builtin.IndexType())
    ops_list.append(convert_op)
    return convert_op.results[0], ops_list
  else:
    return load_op.results[0], ops_list

def handle_array_size_lu_bound(ctx: SSAValueCtx, bound_op: Operation):
  # Handles extracting the literal size of a lower or upper array size bound
  # or the corresponding ssa and ops if it is driven by a variable
  bound_val=load_ssa=load_ops=None
  bound_op=remove_array_size_convert(bound_op)
  if isa(bound_op, arith.Constant):
    bound_val=bound_op.value.value.data
  elif isa(bound_op, fir.Load):
    load_ssa, load_ops=generate_var_dim_size_load(ctx, bound_op)
  else:
    assert False

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

      default_start_idx_mode=isa(op.value.owner.shape.owner, fir.Shape)

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

          upper_bound_val, upper_load_ssa, upper_load_ops=handle_array_size_lu_bound(ctx, size_op.lhs.owner.lhs.owner)
          if upper_bound_val is not None:
            assert upper_load_ssa is None
            assert upper_load_ops is None
            dim_starts.append(upper_bound_val)
          else:
            assert upper_load_ssa is not None
            assert upper_load_ops is not None
            ops_list+=upper_load_ops
            dim_starts.append(upper_load_ssa)

          lower_bound_val, lower_load_ssa, lower_load_ops=handle_array_size_lu_bound(ctx, size_op.lhs.owner.rhs.owner)
          if lower_bound_val is not None:
            assert lower_load_ssa is None
            assert lower_load_ops is None
            dim_ends.append(lower_bound_val)
          else:
            assert lower_load_ssa is not None
            assert lower_load_ops is not None
            ops_list+=lower_load_ops
            dim_ends.append(lower_load_ssa)

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
      memref_allocation_op=memref_alloca_op=memref.Alloc.get(base_type, shape=[-1]*len(dim_ssas), dynamic_sizes=dim_ssas)
      ops_list.append(memref_allocation_op)
      ctx[op.memref.owner.results[0]]=memref_allocation_op.results[0]
      ctx[op.memref.owner.results[1]]=memref_allocation_op.results[0]

      fn_name=program_state.getCurrentFnState().fn_name
      array_name=op.memref.owner.uniq_name.data
      assert fn_name+"E" in array_name
      array_name=array_name.split(fn_name+"E")[1]
      # Store information about the array - the size, and lower and upper bounds as we need this when accessing elements
      program_state.getCurrentFnState().array_info[array_name]=ArrayDescription(array_name, dim_sizes, dim_starts, dim_ends)

      return ops_list
    elif isa(op.value.owner.memref.owner, fir.ZeroBits):
      pass
    else:
      assert False

  else:
    expr_ops=translate_expr(program_state, ctx, op.value)
    zero_val=arith.Constant.create(properties={"value": builtin.IntegerAttr.from_index_int_value(0)},
                                               result_types=[builtin.IndexType()])
    assert isa(op.memref.owner, hlfir.DeclareOp)
    storage_op=memref.Store.get(ctx[op.value], ctx[op.memref], [zero_val])
    return expr_ops+[zero_val, storage_op]

def array_access_components(program_state: ProgramState, ctx: SSAValueCtx, op:hlfir.DesignateOp):
  # This will generate the required operations and SSA for index accesses to an array, whether
  # this is storage or loading in the wider context. It will offset depending upon the logical
  # start index to the physical start index of 0
  fn_name=program_state.getCurrentFnState().fn_name
  if isa(op.memref.owner, hlfir.DeclareOp):
    array_name=op.memref.owner.uniq_name.data
  elif isa(op.memref.owner, fir.Load):
    assert isa(op.memref.owner.memref.owner, hlfir.DeclareOp)
    array_name=op.memref.owner.memref.owner.uniq_name.data
  else:
    assert False

  assert fn_name+"E" in array_name
  array_name=array_name.split(fn_name+"E")[1]

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
      subtract_op=arith.Subi(index_ssa, dim_start)
      ops_list.append(subtract_op)
      indexes_ssa.append(subtract_op.results[0])
    else:
      assert False
  return ops_list, indexes_ssa

def translate_assign(program_state: ProgramState, ctx: SSAValueCtx, op: hlfir.AssignOp):
  expr_ops=translate_expr(program_state, ctx, op.lhs)
  if isa(op.rhs.owner, hlfir.DeclareOp):
    # Scalar value
    zero_val=arith.Constant.create(properties={"value": builtin.IntegerAttr.from_index_int_value(0)},
                                             result_types=[builtin.IndexType()])
    storage_op=memref.Store.get(ctx[op.lhs], ctx[op.rhs], [zero_val])
    return expr_ops+[zero_val, storage_op]
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

    storage_op=memref.Store.get(ctx[op.lhs], ctx[memref_reference], indexes_ssa)
    ops_list.append(storage_op)
    return expr_ops+ops_list
  else:
    assert False

def translate_constant(program_state: ProgramState, ctx: SSAValueCtx, op: arith.Constant):
  if ctx.contains(op.results[0]): return []
  new_const=arith.Constant(op.value, op.results[0].type)
  ctx[op.results[0]]=new_const.results[0]
  return [new_const]

def handle_call_argument(program_state: ProgramState, ctx: SSAValueCtx, fn_name: str, arg: Operand, arg_index: int):
  if isa(arg.owner, hlfir.AssociateOp):
    assert arg.owner.uniq_name.data=="adapt.valuebyref"
    arg_defn=program_state.function_definitions[fn_name].args[arg_index]
    # For now we just work with scalars here, could pass arrays by literal too
    assert arg_defn.is_scalar
    if arg_defn.is_scalar and arg_defn.intent == ArgIntent.IN:
      # This is a scalar with intent in, therefore just pass the constant
      ops_list=translate_expr(program_state, ctx, arg.owner.source)
      ctx[arg]=ctx[arg.owner.source]
      return ops_list
    else:
      # Otherwise we need to pack the constant into a memref and pass this
      assert isa(arg_defn.arg_type, fir.ReferenceType)
      ops_list=translate_expr(program_state, ctx, arg.owner.source)
      memref_alloca_op=memref.Alloca.get(arg_defn.arg_type.type)
      zero_val=arith.Constant.create(properties={"value": builtin.IntegerAttr.from_index_int_value(0)},
                                             result_types=[builtin.IndexType()])
      storage_op=memref.Store.get(ctx[arg.owner.source], memref_alloca_op.results[0], [zero_val])
      ctx[arg]=memref_alloca_op.results[0]
      return ops_list+[memref_alloca_op, zero_val, storage_op]
  else:
    return translate_expr(program_state, ctx, arg)

def translate_call(program_state: ProgramState, ctx: SSAValueCtx, op: fir.Call):
  fn_name=op.callee.string_value()
  if "_QP" in fn_name:
    fn_name=fn_name.split("_QP")[1]

  arg_ops=[]
  for idx, arg in enumerate(op.args):
    # This is more complex, as constants are passed directly or packed in a temporary
    arg_ops+=handle_call_argument(program_state, ctx, fn_name, arg, idx)

  arg_ssa=[]
  for arg in op.args:
    arg_ssa.append(ctx[arg])
  call_op=func.Call(op.callee, arg_ssa, [])
  return arg_ops+[call_op]

def translate_do_loop(program_state: ProgramState, ctx: SSAValueCtx, op: fir.DoLoop):
  if ctx.contains(op.results[1]):
    for fir_result in op.results[1:]:
      assert ctx.contains(fir_result)
    return []

  lower_bound_ops=translate_expr(program_state, ctx, op.lowerBound)
  upper_bound_ops=translate_expr(program_state, ctx, op.upperBound)
  step_ops=translate_expr(program_state, ctx, op.step)
  initarg_ops=translate_expr(program_state, ctx, op.initArgs)

  assert len(op.regions)==1
  assert len(op.regions[0].blocks)==1

  arg_types=[]
  for arg in op.regions[0].blocks[0].args:
    arg_types.append(arg.type)

  new_block = Block(arg_types=arg_types)

  for fir_arg, std_arg in zip(op.regions[0].blocks[0].args, new_block.args):
    ctx[fir_arg]=std_arg

  loop_body_ops=[]
  for loop_op in op.regions[0].blocks[0].ops:
    loop_body_ops+=translate_stmt(program_state, ctx, loop_op)

  # The fir result has both the index and iterargs, whereas the yield has only
  # the iterargs. Therefore need to rebuild the yield with the first argument (the index)
  # removed from it
  yield_op=loop_body_ops[-1]
  assert isa(yield_op, scf.Yield)
  new_yieldop=scf.Yield(*yield_op.arguments[1:])
  del loop_body_ops[-1]
  loop_body_ops.append(new_yieldop)

  new_block.add_ops(loop_body_ops)

  scf_for_loop=scf.For(ctx[op.lowerBound], ctx[op.upperBound], ctx[op.step], [ctx[op.initArgs]], new_block)

  for index, scf_result in enumerate(scf_for_loop.results):
    ctx[op.results[index+1]]=scf_result

  return lower_bound_ops+upper_bound_ops+step_ops+initarg_ops+[scf_for_loop]

def translate_return(program_state: ProgramState, ctx: SSAValueCtx, op: func.Return):
  ssa_to_return=[]
  for arg in op.arguments:
    ssa_to_return.append(ctx[arg])
  new_return=func.Return(*ssa_to_return)
  return [new_return]

def translate_declare(program_state: ProgramState, ctx: SSAValueCtx, op: hlfir.DeclareOp):
  if isa(op.results[0].type, fir.ReferenceType) and isa(op.results[0].type.type, fir.BoxType):
    # This is an allocatable array, we will handle this on the allocation
    return []

  if op.shape is None:
    return define_scalar_var(program_state, ctx, op)
  else:
    if isa(op.shape.owner, fir.Shape):
      dim_sizes, dim_starts, dim_ends=gather_static_shape_dims_from_shape(op.shape.owner)
    elif isa(op.shape.owner, fir.ShapeShift):
      dim_sizes, dim_starts, dim_ends=gather_static_shape_dims_from_shapeshift(op.shape.owner)
    else:
      assert False
    static_size=dims_has_static_size(dim_sizes)
    if static_size:
      return define_stack_array_var(program_state, ctx, op, dim_sizes, dim_starts, dim_ends)
    else:
      assert False

def dims_has_static_size(dims):
  for dim in dims:
    if dim is None: return False
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
  for low_arg, high_arg in itertools.pairwise(shape_op.pairs):
    if isa(low_arg.owner, arith.Constant):
      assert isa(low_arg.owner.result.type, builtin.IndexType)
      dim_starts.append(low_arg.owner.value.value.data)
    else:
      dim_starts.append(None)

    if isa(high_arg.owner, arith.Constant):
      assert isa(high_arg.owner.result.type, builtin.IndexType)
      dim_sizes.append(high_arg.owner.value.value.data)
    else:
      dim_sizes.append(None)

    if dim_starts[-1] is not None and dim_sizes[-1] is not None:
      dim_ends.append((dim_sizes[-1]+dim_starts[-1])-1)
    else:
      dim_ends.append(None)

  return dim_sizes, dim_starts, dim_sizes

def define_stack_array_var(program_state: ProgramState, ctx: SSAValueCtx,
      op: hlfir.DeclareOp, dim_sizes: list, dim_starts: list, dim_ends: list):
  if ctx.contains(op.results[0]):
    assert ctx.contains(op.results[1])
    return []
  # It might either allocate from a global or an alloca in fir, but both can be
  # handled the same
  assert isa(op.memref.owner, fir.AddressOf) or isa(op.memref.owner, fir.Alloca)
  assert isa(op.memref.owner.results[0].type, fir.ReferenceType)
  fir_array_type=op.memref.owner.results[0].type.type
  assert isa(fir_array_type, fir.SequenceType)
  # Ensure collected dimensions and the addressof type dimensions are consistent
  for type_size, dim_size in zip(fir_array_type.shape, dim_sizes):
    assert isa(type_size.type, builtin.IntegerType)
    assert type_size.value.data == dim_size

  fn_name=program_state.getCurrentFnState().fn_name
  array_name=op.uniq_name.data
  assert fn_name+"E" in array_name
  array_name=array_name.split(fn_name+"E")[1]
  # Store information about the array - the size, and lower and upper bounds as we need this when accessing elements
  program_state.getCurrentFnState().array_info[array_name]=ArrayDescription(array_name, dim_sizes, dim_starts, dim_ends)

  memref_alloca_op=memref.Alloca.get(fir_array_type.type, shape=dim_sizes)
  ctx[op.results[0]] = memref_alloca_op.results[0]
  ctx[op.results[1]] = memref_alloca_op.results[0]
  return [memref_alloca_op]

def define_scalar_var(program_state: ProgramState, ctx: SSAValueCtx, op: hlfir.DeclareOp):
  if ctx.contains(op.results[0]):
    assert ctx.contains(op.results[1])
    return []
  if isa(op.memref, OpResult):
    allocation_op=op.memref.owner
    assert isa(allocation_op, fir.Alloca)
    assert isa(allocation_op.results[0].type, fir.ReferenceType)
    assert allocation_op.results[0].type.type == allocation_op.in_type
    memref_alloca_op=memref.Alloca.get(allocation_op.in_type)
    ctx[op.results[0]] = memref_alloca_op.results[0]
    ctx[op.results[1]] = memref_alloca_op.results[0]
    return [memref_alloca_op]
  elif isa(op.memref, BlockArgument):
    ctx[op.results[0]] = ctx[op.memref]
    ctx[op.results[1]] = ctx[op.memref]
    return []

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
    res_module = translate_program(program_state, input_module)
    res_module.regions[0].move_blocks(input_module.regions[0])

from abc import ABC
from typing import TypeVar, cast
from dataclasses import dataclass
from xdsl.dialects.experimental import fir, hlfir
from dataclasses import dataclass, field
from typing import Dict, Optional
from xdsl.ir import SSAValue
from xdsl.utils.hints import isa
from xdsl.ir import Operation, SSAValue, OpResult, Attribute, MLContext, Block, Region

from xdsl.pattern_rewriter import (RewritePattern, PatternRewriter,
                                   op_type_rewrite_pattern,
                                   PatternRewriteWalker,
                                   GreedyRewritePatternApplier)
from xdsl.passes import ModulePass
from xdsl.dialects import builtin, func, llvm, arith, memref

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

def translate_program(input_module: builtin.ModuleOp) -> builtin.ModuleOp:
    # create an empty global context
    global_ctx = SSAValueCtx()
    body = Region()
    block = Block()
    for fn in input_module.ops:
      if isa(fn, func.FuncOp):
        fn_op=translate_function(global_ctx, fn)
        block.add_op(fn_op)
      elif isa(fn, fir.Global):
        pass
      else:
        assert False
    body.add_block(block)
    return builtin.ModuleOp(body)

def translate_function(ctx: SSAValueCtx, fn: func.FuncOp):
  body = Region()
  for block in fn.body.blocks:
    arg_types=[]
    for arg in fn.args:
      arg_types.append(arg.type)

    new_block = Block(arg_types=arg_types)
    ops_list=[]
    for op in block.ops:
      ops_list+=translate_op(ctx, op)

    new_block.add_ops(ops_list)
    body.add_block(new_block)

  new_func=func.FuncOp(fn.sym_name, fn.function_type, body, fn.sym_visibility, arg_attrs=fn.arg_attrs, res_attrs=fn.res_attrs)
  return new_func

def translate_op(ctx: SSAValueCtx, op: Operation):
  ops = try_translate_stmt(ctx, op)
  if ops is not None:
    return ops
  return []
  #raise Exception(f"Could not translate `{op}' as a definition or statement")

def try_translate_stmt(ctx: SSAValueCtx, op: Operation):
  if isa(op, hlfir.DeclareOp):
    return translate_declare(ctx, op)
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
    return translate_return(ctx, op)
  elif isa(op, hlfir.AssignOp):
    return translate_assign(ctx, op)
  else:
    return None

def translate_expr(ctx: SSAValueCtx, op: Operation):
  ops = try_translate_expr(ctx, op)
  if ops is not None:
    return ops
  raise Exception(f"Could not translate `{op}' as an expression")

def try_translate_expr(ctx: SSAValueCtx, op: Operation):
  if isa(op, arith.Constant):
    return translate_constant(ctx, op)
  elif (isa(op, arith.Addi) or isa(op, arith.Subi) or isa(op, arith.Muli) or isa(op, arith.DivUI) or isa(op, arith.DivSI) or
      isa(op, arith.FloorDivSI) or isa(op, arith.CeilDivSI) or isa(op, arith.CeilDivUI) or isa(op, arith.RemUI) or
      isa(op, arith.RemSI) or isa(op, arith.MinUI) or isa(op, arith.MaxUI) or isa(op, arith.MinSI) or isa(op, arith.MaxSI) or
      isa(op, arith.AndI) or isa(op, arith.OrI) or isa(op, arith.XOrI) or isa(op, arith.ShLI) or isa(op, arith.ShRUI) or
      isa(op, arith.ShRSI) or isa(op, arith.AddUIExtended)):
    return translate_integer_binary_arithmetic(ctx, op)
  elif (isa(op, arith.Addf) or isa(op, arith.Subf) or isa(op, arith.Mulf) or isa(op, arith.Divf) or isa(op, arith.Maximumf) or
      isa(op, arith.Maxnumf) or isa(op, arith.Minimumf) or isa(op, arith.Minnumf)):
    return translate_float_binary_arithmetic(ctx, op)
  elif isa(op, fir.Load):
    return translate_load(ctx, op)
  elif isa(op, fir.Convert):
    return translate_convert(ctx, op)
  else:
    return None

def translate_convert(ctx: SSAValueCtx, op: fir.Convert):
  value_ops=translate_expr(ctx, op.value.owner)
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

  if isa(in_type, builtin.IndexType) and isa(out_type, builtin.IntegerType):
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


def translate_load(ctx: SSAValueCtx, op: fir.Load):
  zero_val=arith.Constant.create(properties={"value": builtin.IntegerAttr.from_index_int_value(0)},
                                           result_types=[builtin.IndexType()])
  load_op=memref.Load.get(ctx[op.memref], [zero_val])
  ctx[op.results[0]]=load_op.results[0]
  return [zero_val, load_op]

def translate_assign(ctx: SSAValueCtx, op: hlfir.AssignOp):
  expr_ops=translate_expr(ctx, op.lhs.owner)
  if isa(op.rhs.owner, hlfir.DeclareOp):
    zero_val=arith.Constant.create(properties={"value": builtin.IntegerAttr.from_index_int_value(0)},
                                             result_types=[builtin.IndexType()])
    storage_op=memref.Store.get(ctx[op.lhs], ctx[op.rhs], [zero_val])
    return expr_ops+[zero_val, storage_op]
  elif isa(op.rhs.owner, hlfir.DesignateOp):
    assert op.rhs.owner.indices is not None
    ops_list=[]
    indexes_ssa=[]
    for index in op.rhs.owner.indices:
      ops=translate_expr(ctx, index.owner)
      ops_list+=ops
      if not isa(ctx[index].type, builtin.IndexType):
        assert isa(ctx[index].type, builtin.IntegerType)
        convert_op=arith.IndexCastOp(ctx[index], builtin.IndexType())
        ops_list.append(convert_op)
        indexes_ssa.append(convert_op.results[0])
      else:
        indexes_ssa.append(ctx[index])
    storage_op=memref.Store.get(ctx[op.lhs], ctx[op.rhs.owner.memref], indexes_ssa)
    ops_list.append(storage_op)
    return expr_ops+ops_list
  else:
    assert False

def translate_constant(ctx: SSAValueCtx, op: arith.Constant):
  new_const=arith.Constant(op.value, op.results[0].type)
  ctx[op.results[0]]=new_const.results[0]
  return [new_const]

def translate_return(ctx: SSAValueCtx, op: func.Return):
  ssa_to_return=[]
  for arg in op.arguments:
    ssa_to_return.append(ctx[arg])
  new_return=func.Return(*ssa_to_return)
  return [new_return]

def translate_declare(ctx: SSAValueCtx, op: hlfir.DeclareOp):
  if op.shape is None:
    return define_scalar_var(ctx, op)
  else:
    dims=gather_static_shape_dims(op.shape.owner)
    static_size=dims_has_static_size(dims)
    if static_size:
      return define_stack_array_var(ctx, op, dims)

def dims_has_static_size(dims):
  for dim in dims:
    if dim is None: return False
  return True

def gather_static_shape_dims(shape_op: fir.Shape):
  dims=[]
  assert shape_op.extents is not None
  for extent in shape_op.extents:
    if isa(extent.owner, arith.Constant):
      assert isa(extent.owner.result.type, builtin.IndexType)
      dims.append(extent.owner.value.value.data)
    else:
      dims.append(None)
  return dims

def define_stack_array_var(ctx: SSAValueCtx, op: hlfir.DeclareOp, static_dims):
  assert isa(op.memref.owner, fir.AddressOf)
  assert isa(op.memref.owner.results[0].type, fir.ReferenceType)
  fir_array_type=op.memref.owner.results[0].type.type
  assert isa(fir_array_type, fir.SequenceType)
  # Ensure collected dimensions and the addressof type dimensions are consistent
  for type_size, dim_size in zip(fir_array_type.shape, static_dims):
    assert isa(type_size.type, builtin.IntegerType)
    assert type_size.value.data == dim_size

  memref_alloca_op=memref.Alloca.get(fir_array_type.type, shape=static_dims)
  ctx[op.results[0]] = memref_alloca_op.results[0]
  ctx[op.results[1]] = memref_alloca_op.results[0]
  return [memref_alloca_op]

def define_scalar_var(ctx: SSAValueCtx, op: hlfir.DeclareOp):
  allocation_op=op.memref.owner
  assert isa(allocation_op, fir.Alloca)
  assert isa(allocation_op.results[0].type, fir.ReferenceType)
  assert allocation_op.results[0].type.type == allocation_op.in_type
  memref_alloca_op=memref.Alloca.get(allocation_op.in_type)
  ctx[op.results[0]] = memref_alloca_op.results[0]
  ctx[op.results[1]] = memref_alloca_op.results[0]
  return [memref_alloca_op]

def translate_float_binary_arithmetic(ctx: SSAValueCtx, op: Operation):
  lhs_ops=translate_expr(ctx, op.lhs.owner)
  rhs_ops=translate_expr(ctx, op.rhs.owner)
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

def translate_integer_binary_arithmetic(ctx: SSAValueCtx, op: Operation):
  lhs_ops=translate_expr(ctx, op.lhs.owner)
  rhs_ops=translate_expr(ctx, op.rhs.owner)
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
    res_module = translate_program(input_module)
    res_module.regions[0].move_blocks(input_module.regions[0])

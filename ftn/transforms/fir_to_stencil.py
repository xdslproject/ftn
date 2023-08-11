from abc import ABC
from typing import TypeVar, cast
from enum import Enum
from dataclasses import dataclass
import itertools
from xdsl.utils.hints import isa
from xdsl.dialects.memref import MemRefType
from xdsl.ir import Operation, SSAValue, OpResult, Attribute, MLContext, Block, Region

from xdsl.pattern_rewriter import (RewritePattern, PatternRewriter,
                                   op_type_rewrite_pattern,
                                   PatternRewriteWalker,
                                   GreedyRewritePatternApplier)
from xdsl.passes import ModulePass
from xdsl.dialects import builtin, func, llvm, arith
from xdsl.dialects.experimental import fir
from xdsl.dialects import stencil
from util.visitor import Visitor

class GetArrayAccessValue(Visitor):
  # Walks the index expression of an array (i.e. (a-1-1) and builds up expression
  # tree based on this, supports sub, add, var_name and constants
  class SideType(Enum):
    CONSTANT = 1
    VARIABLE = 2
    EXPRESSION = 3

  class ArithOp(Enum):
    SUB = 1
    ADD = 2

  class ExpressionDescription:
    def __init__(self, lhs, lhs_t, rhs=None, rhs_t=None, arith_operation=None):
      self.arith_operation=arith_operation
      self.lhs=lhs
      self.lhs_type=lhs_t
      self.rhs=rhs
      self.rhs_type=rhs_t

  def __init__(self):
    self.expression=None
    self.expression_t=None

  def get_offset(self):
    if self.expression_t == GetArrayAccessValue.SideType.EXPRESSION:
      return GetArrayAccessValue.calc_offset(self.expression)
    elif self.expression_t == GetArrayAccessValue.SideType.VARIABLE:
      return 0
    else:
      return expression.lhs

  def calc_offset(expression):
    if expression.lhs_type == GetArrayAccessValue.SideType.EXPRESSION:
      lhs=GetArrayAccessValue.calc_offset(expression.lhs)
    elif expression.lhs_type == GetArrayAccessValue.SideType.VARIABLE:
      lhs=0
    else:
      lhs=expression.lhs

    if expression.rhs_type is not None and expression.rhs is not None:
      if expression.rhs_type == GetArrayAccessValue.SideType.EXPRESSION:
        rhs=GetArrayAccessValue.calc_offset(expression.rhs)
      elif expression.rhs_type == GetArrayAccessValue.SideType.VARIABLE:
        rhs=0
      else:
        rhs=expression.rhs
    else:
      rhs=0

    if expression.arith_operation is not None:
      if expression.arith_operation == GetArrayAccessValue.ArithOp.SUB:
        return lhs-rhs
      elif expression.arith_operation == GetArrayAccessValue.ArithOp.ADD:
        return lhs+rhs
      else:
        assert False
    else:
      return lhs

  def get_var_names(self):
    # Gets variable names that have been referenced in the expression
    var_names=[]
    GetArrayAccessValue.search_var_names(var_names, self.expression, self.expression_t)
    return var_names

  def search_var_names(var_names, expr, expr_t):
    # Recursive procedure to search all nested expressions to locate variable names
    if expr_t == GetArrayAccessValue.SideType.VARIABLE:
      var_names.append(expr)
    elif expr_t == GetArrayAccessValue.SideType.EXPRESSION:
      GetArrayAccessValue.search_var_names(var_names, expr.lhs, expr.lhs_type)
      GetArrayAccessValue.search_var_names(var_names, expr.rhs, expr.rhs_type)

  def traverse_constant(self, constant_op: arith.Constant):
    # Grabs out constant value
    assert constant_op.value.typ == builtin.i32 or constant_op.value.typ == builtin.i64
    self.expression=constant_op.value.value.data
    self.expression_t=GetArrayAccessValue.SideType.CONSTANT

  def handle_arith_op(self, arith_op, arith_op_type):
    self.traverse(arith_op.lhs.owner)
    lhs_v=self.expression
    lhs_t=self.expression_t
    self.traverse(arith_op.rhs.owner)
    e=GetArrayAccessValue.ExpressionDescription(lhs_v, lhs_t, self.expression, self.expression_t, arith_op_type)
    self.expression=e
    self.expression_t=GetArrayAccessValue.SideType.EXPRESSION

  def traverse_subi(self, subi_op: arith.Subi):
    self.handle_arith_op(subi_op, GetArrayAccessValue.ArithOp.SUB)

  def traverse_addi(self, addi_op: arith.Addi):
    self.handle_arith_op(addi_op, GetArrayAccessValue.ArithOp.ADD)

  def traverse_convert(self, convert_op: fir.Convert):
    # We ignore converts apart from visiting the children
    self.traverse(convert_op.value.owner)

  def traverse_load(self, load_op: fir.Load):
    # Signifies a variable
    assert isa(load_op.memref.op, fir.Alloca)
    self.expression=load_op.memref.op.bindc_name.data
    self.expression_t=GetArrayAccessValue.SideType.VARIABLE

class LoopDescription():
  # Describes a loop, with its lower and upper bound along with the iterator
  # variable name and the loop operation itself
  def __init__(self, lb, ub, name, loop_op):
    self.lb=lb
    self.ub=ub
    self.name=name
    self.loop_op=loop_op

class GatherLoops(Visitor):
  def __init__(self):
    self.loop_description={}

  def get_constant(token):
    if isa(token, fir.Convert):
      return GatherLoops.get_constant(token.value.owner)
    elif isa(token, arith.Constant):
      assert token.value.typ == builtin.i32
      return token.value.value.data
    else:
      assert False

  def traverse_do_loop(self, doloop_op:fir.DoLoop):
    # Traverse a loop to grab out the description of it
    store_op=list(doloop_op.regions[0].block.args[1].uses)[0].operation
    assert isa(store_op, fir.Store)
    assert isa(store_op.memref.op, fir.Alloca)
    lb=GatherLoops.get_constant(doloop_op.lowerBound.owner)
    ub=GatherLoops.get_constant(doloop_op.upperBound.owner)
    loop_name=store_op.memref.op.bindc_name.data
    self.loop_description[loop_name]=LoopDescription(lb, ub, loop_name, doloop_op)
    for op in doloop_op.regions[0].block.ops:
      self.traverse(op)

class LocateReadUniqueDataNames(Visitor):
  # Finds all unique variables read on the RHS of an expression, this is so we know
  # what arrays are accessed in a stencil computation
  def __init__(self):
    self.read_vars={}

  def traverse_coordinate_of(self, coordinateof_op:fir.CoordinateOf):
    usage_op=list(coordinateof_op.results[0].uses)[0].operation
    if isa(usage_op, fir.Load):
      if isa(coordinateof_op.ref.owner, fir.Alloca):
       # This is stack allocated
       array_name=coordinateof_op.ref.owner.bindc_name.data
       if array_name not in self.read_vars.keys():
         self.read_vars[array_name]=coordinateof_op.ref.owner
      else:
        assert False

class GetStoreCalculationContributedOperations(Visitor):
  # For a calculation this will gather up all the operations that contribute
  # to it, for instance the arithmetic operations, LHS and RHS variables
  # and constants, conversions etc
  def __init__(self):
    self.ops=[]

  def handle_binary_op(self, bin_op):
    self.ops.append(bin_op)
    self.traverse(bin_op.lhs.owner)
    self.traverse(bin_op.rhs.owner)

  def traverse_subi(self, subi_op:arith.Subi):
    self.handle_binary_op(subi_op)

  def traverse_addi(self, addi_op:arith.Addi):
    self.handle_binary_op(addi_op)

  def traverse_addf(self, addf_op:arith.Addf):
    self.handle_binary_op(addf_op)

  def traverse_mulf(self, mulf_op:arith.Mulf):
    self.handle_binary_op(mulf_op)

  def traverse_subf(self, subf_op:arith.Subf):
    self.handle_binary_op(subf_op)

  def traverse_no_reassoc(self, noreassoc_op:fir.NoReassoc):
    self.ops.append(noreassoc_op)
    self.traverse(noreassoc_op.val.owner)

  def traverse_convert(self, convert_op:fir.Convert):
    self.ops.append(convert_op)
    self.traverse(convert_op.value.owner)

  def traverse_load(self, load_op:fir.Load):
    self.ops.append(load_op)
    self.traverse(load_op.memref.owner)

  def traverse_alloca(self, alloca_op:fir.Alloca):
    self.ops.append(alloca_op)

  def traverse_constant(self, constant_op:arith.Constant):
    self.ops.append(constant_op)

  def traverse_coordinate_of(self, coordinateof_op:fir.CoordinateOf):
    self.ops.append(coordinateof_op)
    self.traverse(coordinateof_op.ref.owner)
    for index_res in coordinateof_op.coor:
      self.traverse(index_res.owner)

class DetachCoordOperations(Visitor):
  # Will detach operations that provide arguments to an operation, this allows
  # us to detach the operation itself and then other operations which contribute to it
  # but now are worthless
  def handle_binary_op(self, bin_op):
    bin_op.detach()
    self.traverse(bin_op.lhs.owner)
    self.traverse(bin_op.rhs.owner)

  def traverse_subi(self, subi_op:arith.Subi):
    self.handle_binary_op(subi_op)

  def traverse_addi(self, addi_op:arith.Addi):
    self.handle_binary_op(addi_op)

  def traverse_convert(self, convert_op:fir.Convert):
    convert_op.detach()
    self.traverse(convert_op.value.owner)

  def traverse_load(self, load_op:fir.Load):
    load_op.detach()

  def traverse_constant(self, constant_op:arith.Constant):
    constant_op.detach()

  def traverse_coordinate_of(self, coordinateof_op:fir.CoordinateOf):
    coordinateof_op.detach()
    for index_res in coordinateof_op.coor:
      self.traverse(index_res.owner)

class LocateStoreToStencilOperationsAndGenerate(Visitor):
  # Will check if a store is keyed by loops and will be used for stencil, i.e.
  # a(i,j) where i and j are loop variables then this is true, else false
  def __init__(self, loop_description):
    self.loop_description=loop_description
    self.stencil_generations=[]

  def check_if_store_keyed_by_loops(self, store_op):
    if isa(store_op.memref.owner, fir.CoordinateOf):
      all_var_names, var_to_loops=self.getLoopsThatIndexStore(store_op)
      for k,v in var_to_loops.items():
        if v == None: return False
      return True
    else:
      return False

  def generate_stencil_var_load(alloca_op):
    # Generates the stencil load operations and result SSA for an array based on its FIR allocation
    el_type=alloca_op.in_type.type
    lb=stencil.IndexAttr.get(*([-1]*len(alloca_op.in_type.shape)))
    ub=[]
    field_bounds=[]
    for dim in alloca_op.in_type.shape:
      # TODO: handle if deferred (allocatable)
      dim_size=dim.value.data
      field_bounds.append((-1, dim_size-1))
      ub.append(dim_size-1)
    external_load_op=stencil.ExternalLoadOp.get(alloca_op.results[0], stencil.FieldType(field_bounds, el_type))
    cast_op=stencil.CastOp.get(external_load_op.results[0], stencil.StencilBoundsAttr(field_bounds), external_load_op.results[0].typ)
    load_op=stencil.LoadOp.get(cast_op.results[0], lb, ub)

    return [external_load_op, cast_op, load_op], load_op.results[0]

  def getLoopsThatIndexStore(self, store_op):
    # Get the loops that will index a store operations, i.e. if I have
    # a(i,j) it will return descriptors to the loops that have iterator
    # i and j, or if no loops drive these then None for those variables in the
    # map.
    assert isa(store_op.memref.owner, fir.CoordinateOf)
    # Check that this is indexed on the loop bounds
    indexed_loops={}
    all_var_names=[]
    coord_op=store_op.memref.owner
    for index_res in coord_op.coor:
      gaav=GetArrayAccessValue()
      gaav.traverse(index_res.owner)
      var_names=gaav.get_var_names()
      for var_name in var_names:
        all_var_names.append(var_name)
        if var_name in self.loop_description.keys():
          indexed_loops[var_name]=self.loop_description[var_name]
        else:
          indexed_loops[var_name]=None
    return all_var_names, indexed_loops

  def build_ops_for_stencil(self, contributed_ops, indexed_read_var_names, block_args):
    # Builds the operations needed inside a stencil.apply operation. This will extract these
    # from the FIR loop, replace array accesses by stencil.access and delete unnescesary operations
    ops=[]
    for op in contributed_ops.ops:
      if not isa(op, fir.Alloca) and op.parent is not None:
        op.detach()
        if isa(op, fir.Load):
          if isa(op.memref.owner, fir.CoordinateOf):
            alloca=op.memref.owner.ref.owner
            array_name=alloca.bindc_name.data
            block_idx=indexed_read_var_names[array_name]

            offsets=[]
            for index_res in op.memref.owner.coor:
              gaav=GetArrayAccessValue()
              gaav.traverse(index_res.owner)
              # Plus one here as FIR adds a minus one to zero index loops
              offsets.append(gaav.get_offset()+1)

            access_op=stencil.AccessOp.get(block_args[block_idx], offsets, None)
            op.results[0].replace_by(access_op.results[0])
            ops.append(access_op)
            DetachCoordOperations().traverse(op.memref.owner)
        else:
          ops.append(op)
    ops.reverse()
    # Add in the stencil return at the end of the block
    ops.append(stencil.ReturnOp.get([ops[-1]]))
    return ops

  def generate_stencil_var_store(store_op, stencil_apply_op):
    # Generates the stencil operations for the stencil.store to store the results of the
    # stencil.apply operation
    alloca_op=store_op.memref.owner.ref.owner
    assert isa(alloca_op, fir.Alloca)
    el_type=alloca_op.in_type.type
    lb=stencil.IndexAttr.get(*([-1]*len(alloca_op.in_type.shape)))
    store_lb_indexes=stencil.IndexAttr.get(*([0] * len(alloca_op.in_type.shape)))
    ub=[]
    store_ub_indexes=[]
    field_bounds=[]
    for dim in alloca_op.in_type.shape:
      # TODO: handle if deferred (allocatable)
      dim_size=dim.value.data
      field_bounds.append((-1, dim_size-1))
      ub.append(dim_size-1)
      store_ub_indexes.append(dim_size-2)
    external_load_op=stencil.ExternalLoadOp.get(alloca_op.results[0], stencil.FieldType(field_bounds, el_type))
    cast_op=stencil.CastOp.get(external_load_op.results[0], stencil.StencilBoundsAttr(field_bounds), external_load_op.results[0].typ)
    store_op=stencil.StoreOp.get(stencil_apply_op.results[0], cast_op.results[0], stencil.IndexAttr.get(*store_lb_indexes), stencil.IndexAttr.get(*store_ub_indexes))
    external_store_op=stencil.ExternalStoreOp.create(operands=[external_load_op.results[0], alloca_op.results[0]])

    return [external_load_op, cast_op, store_op, external_store_op]

  def traverse_store(self, store_op:fir.Store):
    # This drives our transformation, where we look at each store and if it can be transformed into a stencil
    # then we move to that format
    move_to_stencil=self.check_if_store_keyed_by_loops(store_op)
    if move_to_stencil:
      contributed_ops=GetStoreCalculationContributedOperations()
      # Now get all the operations that contribute to the RHS (value stored)
      contributed_ops.traverse(store_op.value.owner)
      # Locate all unique variable names read on the RHS
      read_data_discover=LocateReadUniqueDataNames()
      for op in contributed_ops.ops:
        read_data_discover.traverse(op)
      block_ops=[]
      block_types=[]
      indexed_read_var_names={}
      stencil_read_ops=[]
      # Now create stencil load ops and also grab the load SSA result
      for idx, (read_var, alloc_a) in enumerate(read_data_discover.read_vars.items()):
        ops, ssa=LocateStoreToStencilOperationsAndGenerate.generate_stencil_var_load(alloc_a)
        block_ops.append(ssa)
        block_types.append(ssa.typ)
        indexed_read_var_names[read_var]=idx
        stencil_read_ops+=ops

      # Get the loops that drive this store and store the ub for use by the stencil.apply
      all_var_names, stencil_loops=self.getLoopsThatIndexStore(store_op)
      loop_bounds=[]
      for vn in all_var_names:
        loop_bounds.append(stencil_loops[vn].ub)

      # Build the block, based on the read data types and then build the operations
      block = Block(arg_types=block_types)
      ops=self.build_ops_for_stencil(contributed_ops, indexed_read_var_names, block.args)
      block.add_ops(ops)

      # Create the stencil temporary type that results from the stencil.apply
      field_bounds=[]
      for dim in loop_bounds:
        field_bounds.append((0, dim-2))
      stencil_temptypes=[stencil.TempType(field_bounds, store_op.value.typ)]

      # Create the stencil apply itself
      apply_op=stencil.ApplyOp.get(block_ops, block, stencil_temptypes)

      # Now create the stencil.store to store the result of stencil.apply
      store_ops=LocateStoreToStencilOperationsAndGenerate.generate_stencil_var_store(store_op, apply_op)
      DetachCoordOperations().traverse(store_op.memref.owner)
      store_op.detach()

      self.stencil_generations.append((stencil_loops, stencil_read_ops + [apply_op] + store_ops))

class FindTopLevelApplicableStencilLoop(Visitor):
  # This is the first loop that drives the stencil, therefore the stencil should be added just before this
  def __init__(self, stencil_loops):
    self.stencil_loops=stencil_loops
    self.top_loop=None

  def traverse_do_loop(self, doloop_op:fir.DoLoop):
    if self.top_loop is not None: return
    # Traverse a loop to grab out the description of it
    store_op=list(doloop_op.regions[0].block.args[1].uses)[0].operation
    assert isa(store_op, fir.Store)
    assert isa(store_op.memref.op, fir.Alloca)
    loop_name=store_op.memref.op.bindc_name.data
    if loop_name in self.stencil_loops:
      if self.top_loop is None:
        self.top_loop=doloop_op
        return
    for op in doloop_op.regions[0].block.ops:
      self.traverse(op)

@dataclass
class FIRToStencil(ModulePass):
  """
  This is the entry point for the transformation pass which will then apply the rewriter
  """
  name = 'fir-to-stencil'

  def apply(self, ctx: MLContext, module: builtin.ModuleOp):
    loop_gather=GatherLoops()
    loop_gather.traverse(module)

    stencil_generator=LocateStoreToStencilOperationsAndGenerate(loop_gather.loop_description)
    stencil_generator.traverse(module)
    for stencil_generation in stencil_generator.stencil_generations:
      find_top_level_group=FindTopLevelApplicableStencilLoop(stencil_generation[0].keys())
      find_top_level_group.traverse(module)
      tl=find_top_level_group.top_loop
      tl.parent.insert_ops_before(stencil_generation[1], tl)

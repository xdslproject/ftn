from abc import ABC
from typing import TypeVar, cast
from dataclasses import dataclass
from enum import Enum
import itertools
from xdsl.utils.hints import isa
from xdsl.dialects import memref, scf, omp
import tenstorrent.dialects.data_movement as data_movement
import tenstorrent.dialects.host as host
import tenstorrent.dialects.circular_buffer as circular_buffer
import tenstorrent.dialects.compute as compute
from xdsl.ir import Operation, SSAValue, OpResult, Attribute, MLContext, Block, Region, BlockArgument

from xdsl.pattern_rewriter import (RewritePattern, PatternRewriter,
                                   op_type_rewrite_pattern,
                                   PatternRewriteWalker,
                                   GreedyRewritePatternApplier)
from xdsl.passes import ModulePass
from xdsl.dialects import builtin, func, llvm, arith
from ftn.util.visitor import Visitor

uint32 = builtin.IntegerType(32, signedness=builtin.Signedness.UNSIGNED)
uint64 = builtin.IntegerType(64, signedness=builtin.Signedness.UNSIGNED)

class LoopDescription():
  # Describes a loop, with its lower and upper bound along with the iterator
  # variable name and the loop operation itself
  def __init__(self, lb, ub, step, iterator_ssa, loop_op):
    self.lb=lb
    self.ub=ub
    self.step=step
    self.iterator_ssa=iterator_ssa
    self.loop_op=loop_op

class GatherLoops(Visitor):
  def __init__(self):
    self.loop_description={}

  def get_constant(token):
    if isa(token, arith.Constant):
      assert token.value.type == builtin.i32
      return token.value.value.data
    else:
      assert False

  def traverse_s_i_m_d_loop_op(self, simdloop: omp.SIMDLoopOp):
    lb=GatherLoops.get_constant(simdloop.lowerBound[0].owner)
    ub=GatherLoops.get_constant(simdloop.upperBound[0].owner)
    step=GatherLoops.get_constant(simdloop.step[0].owner)

    first_op=simdloop.body.block.ops.first
    assert isa(first_op, memref.Store)
    iterator_ssa=first_op.memref
    self.loop_description[iterator_ssa]=LoopDescription(lb, ub, step, iterator_ssa, simdloop)
    for op in simdloop.body.block.ops:
      self.traverse(op)

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
    MUL = 3

  class ExpressionDescription:
    def __init__(self, lhs, lhs_t, rhs=None, rhs_t=None, arith_operation=None):
      self.arith_operation=arith_operation
      self.lhs=lhs
      self.lhs_type=lhs_t
      self.rhs=rhs
      self.rhs_type=rhs_t

  def get_var_ssa(self):
    # Gets variable ssa that have been referenced in the expression
    var_ssas=[]
    GetArrayAccessValue.search_var_names(var_ssas, self.expression, self.expression_t)
    return var_ssas

  def search_var_names(var_ssas, expr, expr_t):
    # Recursive procedure to search all nested expressions to locate variable names
    if expr_t == GetArrayAccessValue.SideType.VARIABLE:
      var_ssas.append(expr)
    elif expr_t == GetArrayAccessValue.SideType.EXPRESSION:
      GetArrayAccessValue.search_var_names(var_ssas, expr.lhs, expr.lhs_type)
      GetArrayAccessValue.search_var_names(var_ssas, expr.rhs, expr.rhs_type)

  def traverse_constant(self, constant_op: arith.Constant):
    # Grabs out constant value
    assert constant_op.value.type == builtin.i32 or constant_op.value.type == builtin.i64 or isa(constant_op.value.type, builtin.IndexType)
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

  def traverse_muli(self, addi_op: arith.Muli):
    self.handle_arith_op(addi_op, GetArrayAccessValue.ArithOp.MUL)

  def traverse_ext_u_i_op(self, extui_op: arith.ExtUIOp):
    self.traverse(extui_op.input.owner)

  def traverse_ext_s_i_op(self, extsi_op: arith.ExtSIOp):
    self.traverse(extsi_op.input.owner)

  def traverse_index_cast_op(self, convert_op: arith.IndexCastOp):
    # We ignore converts apart from visiting the child
    self.traverse(convert_op.input.owner)

  def traverse_load(self, load_op: memref.Load):
    # Signifies a variable
    self.expression=load_op.memref
    self.expression_t=GetArrayAccessValue.SideType.VARIABLE

class GatherDeviceFunctionArgs(Visitor):
  def __init__(self):
    self.args=[]

  def traverse_func_op(self, func_op: func.FuncOp):
    for arg in func_op.args:
      self.args.append(arg)

class GatherMemoryPassedToDevice(Visitor):
  def __init__(self):
    self.memory_types=[]
    self.references=[]

  def traverse_target_op(self, target_op: omp.TargetOp):
    for idx, map_var in enumerate(target_op.map_vars):
      assert isa(map_var.type, builtin.MemRefType)
      self.memory_types.append(map_var.type.element_type)
      self.references.append(target_op.region.block.args[idx])

class IntermediateCBDescriptors():
  def __init__(self, start_index):
    self.intermediate_cb_idx=start_index
    self.dependent_in_cb_indexes={}

  def increment_cb(self, dependent_in_cb_index):
    self.dependent_in_cb_indexes[self.intermediate_cb_idx]=dependent_in_cb_index
    self.intermediate_cb_idx+=1
    return self.intermediate_cb_idx-1

class GetStoreCalculationContributedOperations(Visitor):
  # For a calculation this will gather up all the operations that contribute
  # to it, for instance the arithmetic operations, LHS and RHS variables
  # and constants, conversions etc

  class ContributedOperation():
    def walk(self, to_match):
      if isa(self, to_match):
        return [self]
      return []

    def label_intermediates(self, intermediate_cb_descriptors):
      pass

  class ArithmeticOperation(ContributedOperation):
    class ArithOpTypes(Enum):
      SUB = 1
      ADD = 2
      MUL = 3
      DIV = 4

    def __init__(self, lhs, rhs, arith_type, op_data_type):
      self.lhs=lhs
      self.rhs=rhs
      self.arith_type=arith_type
      self.op_data_type=op_data_type
      self.cb_idx=None
      self.is_intermediate=False
      self.generated_compute_already=False

    def get_cb_input_idx(self):
      lhs_idxes=self.lhs.get_cb_input_idx()
      rhs_idxes=self.rhs.get_cb_input_idx()

      return lhs_idxes+rhs_idxes

    def label_intermediates(self, intermediate_cb_descriptors):
      self.is_intermediate=self.cb_idx is None
      if self.cb_idx is None:
        in_idxes=self.get_cb_input_idx()
        assert len(in_idxes) > 0
        # We build up all the input CBs that are dependencies here, and store the
        # last one. This will be used to determine the size of the CB when it
        # is created on the host
        self.cb_idx=intermediate_cb_descriptors.increment_cb(in_idxes[-1])

      self.lhs.label_intermediates(intermediate_cb_descriptors)
      self.rhs.label_intermediates(intermediate_cb_descriptors)

    def walk(self, to_match):
      matched_ops=[]
      if isa(self, to_match):
        matched_ops.append(self)
      matched_ops+=self.lhs.walk(to_match)
      matched_ops+=self.rhs.walk(to_match)
      return matched_ops

    def generateCleanUp(self):
      if self.is_intermediate:
        # If this is outputting an intermediate CB then also need to clean that up too
        one_c=arith.Constant.from_int_and_width(1, 32)
        cb_id_op=arith.Constant.from_int_and_width(self.cb_idx, 32)
        pop_front_op=circular_buffer.CBPopFront(cb_id_op, one_c)

        return [one_c, cb_id_op, pop_front_op]
      else:
        return []

    def generate(self):
      assert self.cb_idx is not None
      lhs_ops, lhs_cb_id=self.lhs.generate()
      rhs_ops, rhs_cb_id=self.rhs.generate()

      cb0_i_op=arith.Constant.from_int_and_width(lhs_cb_id, 32)
      cb0_op=builtin.UnrealizedConversionCastOp.get([cb0_i_op], [uint32])

      cb1_i_op=arith.Constant.from_int_and_width(rhs_cb_id, 32)
      cb1_op=builtin.UnrealizedConversionCastOp.get([cb1_i_op], [uint32])

      cb_out_i_op=arith.Constant.from_int_and_width(self.cb_idx, 32)
      cb_out_op=builtin.UnrealizedConversionCastOp.get([cb_out_i_op], [uint32])
      false_op=arith.Constant.from_int_and_width(0, 1)
      binary_init_common_op=compute.BinaryOpInitCommon(cb0_op, cb1_op, cb_out_op)
      add_init_op=compute.AddInit(cb0_op, cb1_op, false_op)

      zero_i_op=arith.Constant.from_int_and_width(0, 32)
      zero_op=builtin.UnrealizedConversionCastOp.get([zero_i_op], [uint32])

      dst_i_op=arith.Constant.from_int_and_width(0, 32)
      dst_op=builtin.UnrealizedConversionCastOp.get([dst_i_op], [uint32])

      acquire_regs_op=compute.RegsAcquire()

      if (self.arith_type == GetStoreCalculationContributedOperations.ArithmeticOperation.ArithOpTypes.SUB):
        arith_tile_op=compute.Sub(cb0_op, cb1_op, zero_op, zero_op, dst_op)
      elif (self.arith_type == GetStoreCalculationContributedOperations.ArithmeticOperation.ArithOpTypes.ADD):
        arith_tile_op=compute.Add(cb0_op, cb1_op, zero_op, zero_op, dst_op)
      elif (self.arith_type == GetStoreCalculationContributedOperations.ArithmeticOperation.ArithOpTypes.MUL):
        arith_tile_op=compute.Mul(cb0_op, cb1_op, zero_op, zero_op, dst_op)
      elif (self.arith_type == GetStoreCalculationContributedOperations.ArithmeticOperation.ArithOpTypes.DIV):
        # No divide provided
        assert False
      else:
        assert False

      commit_regs_op=compute.RegsCommit()

      lhs_clean_up_ops=self.lhs.generateCleanUp()
      rhs_clean_up_ops=self.rhs.generateCleanUp()

      one_op=arith.Constant.from_int_and_width(1, 32)

      reserve_cb_out_op=circular_buffer.CBReserveBack(cb_out_i_op, one_op)

      assert self.cb_idx is not None
      wait_regs_op=compute.RegsWait()
      pack_tile_op=compute.PackTile(builtin.IntegerAttr.from_index_int_value(0), zero_op, cb_out_op, zero_op)
      release_regs_op=compute.RegsRelease()

      pushback_cb_out_op=circular_buffer.CBPushBack(cb_out_i_op, one_op)

      self.generated_compute_already=True

      compute_ops=lhs_ops+rhs_ops+[cb0_i_op, cb0_op, cb1_i_op, cb1_op, cb_out_i_op, cb_out_op, false_op, binary_init_common_op, add_init_op,
                zero_i_op, zero_op, dst_i_op, dst_op, acquire_regs_op, arith_tile_op, commit_regs_op] + lhs_clean_up_ops+rhs_clean_up_ops + [
                one_op, reserve_cb_out_op, wait_regs_op, pack_tile_op, release_regs_op, pushback_cb_out_op]

      if self.is_intermediate:
        # If this is an intermediate output then wait front as an input
        # to the next maths operation
        compute_ops.append(circular_buffer.CBWaitFront(cb_out_i_op, one_op))

      return compute_ops, self.cb_idx


  class ConstantOperation(ContributedOperation):
    def __init__(self, value, op_data_type):
      self.value=value
      self.op_data_type=op_data_type

    def generate(self):
      # Currently don't support this as no easy way of writing a constant to a tile
      assert False

    def generateCleanUp(self):
      assert False

  class VariableOperation(ContributedOperation):
    def __init__(self, var_ssa, var_type):
      self.var_ssa=var_ssa
      self.var_type=var_type
      self.cb_idx=None

    def get_cb_input_idx(self):
      assert self.cb_idx is not None
      return [self.cb_idx]

    def generate(self):
      assert self.cb_idx is not None
      one_c=arith.Constant.from_int_and_width(1, 32)
      cb_id_op=arith.Constant.from_int_and_width(self.cb_idx, 32)
      cb_waitfront_op=circular_buffer.CBWaitFront(cb_id_op.results[0], one_c)

      return [one_c, cb_id_op, cb_waitfront_op], self.cb_idx

    def generateCleanUp(self):
      assert self.cb_idx is not None
      one_c=arith.Constant.from_int_and_width(1, 32)
      cb_id_op=arith.Constant.from_int_and_width(self.cb_idx, 32)
      pop_front_op=circular_buffer.CBPopFront(cb_id_op, one_c)

      return [one_c, cb_id_op, pop_front_op]

  def __init__(self):
    self.tree=None

  def handle_binary_op(self, bin_op):
    lhs=self.traverse(bin_op.lhs.owner)
    rhs=self.traverse(bin_op.rhs.owner)
    # These are returned by the visitor, returns a list due to potential of multiple visits
    assert len(lhs) == 1
    assert len(rhs) == 1
    if isa(bin_op, arith.Subi) or isa(bin_op, arith.Subf):
      arith_type=GetStoreCalculationContributedOperations.ArithmeticOperation.ArithOpTypes.SUB
    elif isa(bin_op, arith.Muli) or isa(bin_op, arith.Mulf):
      arith_type=GetStoreCalculationContributedOperations.ArithmeticOperation.ArithOpTypes.MUL
    elif isa(bin_op, arith.Addi) or isa(bin_op, arith.Addf):
      arith_type=GetStoreCalculationContributedOperations.ArithmeticOperation.ArithOpTypes.ADD
    elif isa(bin_op, arith.DivUI) or isa(bin_op, arith.DivSI) or isa(bin_op, arith.Divf):
      arith_type=GetStoreCalculationContributedOperations.ArithmeticOperation.ArithOpTypes.DIV
    else:
      assert False
    arith_data_type = bin_op.results[0].type
    return GetStoreCalculationContributedOperations.ArithmeticOperation(lhs[0], rhs[0], arith_type, arith_data_type)

  def traverse_subi(self, subi_op:arith.Subi):
    return self.handle_binary_op(subi_op)

  def traverse_muli(self, muli_op:arith.Muli):
    return self.handle_binary_op(muli_op)

  def traverse_addi(self, addi_op:arith.Addi):
    return self.handle_binary_op(addi_op)

  def traverse_addf(self, addf_op:arith.Addf):
    return self.handle_binary_op(addf_op)

  def traverse_mulf(self, mulf_op:arith.Mulf):
    return self.handle_binary_op(mulf_op)

  def traverse_subf(self, subf_op:arith.Subf):
    return self.handle_binary_op(subf_op)

  def traverse_divf(self, divf_op:arith.Divf):
    return self.handle_binary_op(divf_op)

  def traverse_div_u_i(self, divui_op:arith.DivUI):
    return self.handle_binary_op(divui_op)

  def traverse_div_s_i(self, divsi_op:arith.DivSI):
    return self.handle_binary_op(divsi_op)

  def traverse_load(self, load_op:memref.Load):
    return GetStoreCalculationContributedOperations.VariableOperation(load_op.memref, load_op.results[0].type)

  def traverse_constant(self, constant_op:arith.Constant):
    return GetStoreCalculationContributedOperations.ConstantOperation(constant_op.value, constant_op.results[0].type)

  def traverse_store(self, store_op:memref.Store):
    ret_vals=self.traverse(store_op.value.owner)
    assert len(ret_vals) == 1
    self.tree=ret_vals[0]

class GatherComputeLoop(Visitor):
  def __init__(self):
    self.for_op=None
    self.simd_loop=None

  def traverse_for(self, for_op: scf.For):
    self.for_op=for_op

  def traverse_s_i_m_d_loop_op(self, simdloop: omp.SIMDLoopOp):
    self.simd_loop=simdloop

class BuildApplicableOpDependencyTrees(Visitor):
  def __init__(self, loop_description):
    self.loop_description=loop_description
    self.dependency_trees=[]

  def get_all_input_var_ssas(self, set_cb_idx=True):
    input_var_ssas=[]
    cb_idx=0
    for dependency_tree in self.dependency_trees:
      input_vars=dependency_tree[1].walk(GetStoreCalculationContributedOperations.VariableOperation)
      for input_var in input_vars:
        if set_cb_idx: input_var.cb_idx=cb_idx
        input_var_ssas.append((input_var.var_ssa, cb_idx))
        cb_idx+=1
    return input_var_ssas

  def get_all_output_var_ssas(self):
    return [d[0].memref for d in self.dependency_trees]

  def check_if_store_indexed_by_loops(self, store_op):
    if len(store_op.indices) > 0:
      all_var_names, var_to_loops=self.getLoopsThatIndexStore(store_op)
      for k,v in var_to_loops.items():
        if v == None: return False
      return True
    else:
      return False

  def getLoopsThatIndexStore(self, store_op):
    # Get the loops that will index a store operations, i.e. if I have
    # a(i,j) it will return descriptors to the loops that have iterator
    # i and j, or if no loops drive these then None for those variables in the
    # map.
    # Check that this is indexed on the loop bounds
    indexed_loops={}
    all_var_ssas=[]

    for index in store_op.indices:
      gaav=GetArrayAccessValue()
      gaav.traverse(index.owner)
      get_var_ssas=gaav.get_var_ssa()
      for var_ssa in get_var_ssas:
        all_var_ssas.append(var_ssa)
        if var_ssa in self.loop_description.keys():
          indexed_loops[var_ssa]=self.loop_description[var_ssa]
        else:
          indexed_loops[var_ssa]=None
    return all_var_ssas, indexed_loops

  def traverse_store(self, store_op:memref.Store):
    # First figure out if we care about this store, i.e. it is indexed by the loops
    # that we looking to simd
    vectorise=self.check_if_store_indexed_by_loops(store_op)
    if vectorise:
      # Now determine the dependency tree of operations that we are going to convert
      # into vectorised TT operations
      contributed_ops=GetStoreCalculationContributedOperations()
      # Now get all the operations that contribute to the RHS (value stored)
      contributed_ops.traverse(store_op)
      self.dependency_trees.append((store_op, contributed_ops.tree))

@dataclass(frozen=True)
class ConvertToTT(ModulePass):
  """
  This is the entry point for the transformation pass which will then apply the rewriter
  """
  name = 'convert-to-tt'

  default_simd_len: int = 512

  def generate_data_in(self, module, memory_type, references, cb_idxs, new_block, generate_writeback=False):
    write_back_ops=[]
    push_back_ops=[]
    one_c=arith.Constant.from_int_and_width(1, 32)
    read_ops=[one_c]
    for idx, (t, ref, cb_idx) in enumerate(zip(memory_type, references, cb_idxs)):
      is_deref_memref=isa(t, builtin.MemRefType)
      if is_deref_memref:
        element_type=t.element_type
      else:
        element_type=t
      dm_op=data_movement.DMGetNocAddrFromBankId(builtin.IntegerAttr.from_int_and_width(1, 1), new_block.args[len(memory_type)+idx], new_block.args[idx])
      dm_op.results[0].name_hint = f"src{idx}_dram_noc_addr"

      const_op=arith.Constant.from_int_and_width(cb_idx, 32)
      cb_reserve_op=circular_buffer.CBReserveBack(const_op.results[0], one_c)
      cb_op=circular_buffer.CBGetWritePointer(const_op.results[0])
      cb_op.results[0].name_hint = "l1_write_addr_in"+str(idx)

      assert element_type.width.data % 8 == 0
      data_type_byte_width=arith.Constant.from_int_and_width(int(element_type.width.data / 8), 32)
      dt_width_conversion_op=builtin.UnrealizedConversionCastOp.get([data_type_byte_width.results[0]], [uint32])
      mem_size_bytes_op=arith.Muli(dt_width_conversion_op, new_block.args[(len(memory_type)*2)+idx])
      read_op=data_movement.DMNocAsyncRead(dm_op.results[0], cb_op.results[0], mem_size_bytes_op)

      target_memref=builtin.MemRefType(element_type, [-1])
      conversion_op=builtin.UnrealizedConversionCastOp.get([cb_op.results[0]], [target_memref])
      conversion_op.results[0].name_hint = f"src{idx}_data"

      read_ops+=[dm_op, const_op, cb_reserve_op, cb_op, data_type_byte_width, dt_width_conversion_op, mem_size_bytes_op, read_op, conversion_op]

      for use in ref.uses:
        if isa(use.operation, memref.Load):
          # Currently just doing this for passing array/data (e.g. memref of memref, ignoring constants but probably want to handle that too)
          if is_deref_memref:
            use.operation.detach()
            use.operation.results[0].replace_by(conversion_op.results[0])

      push_back_ops.append(circular_buffer.CBPushBack(const_op.results[0], one_c))

      if generate_writeback:
        write_op=data_movement.DMNocAsyncWrite(cb_op.results[0], dm_op.results[0], mem_size_bytes_op)
        write_back_ops.append(write_op)

    read_ops.append(data_movement.DMNocAsyncReadBarrier())

    return read_ops, push_back_ops, write_back_ops

  def generate_device_rv_in(self, module, arith_ops_generator, loop_description):
    # For each memref first passed in is memory addresses, then bank ids, and then memory sizes (number elements)
    input_var_ssas=arith_ops_generator.get_all_input_var_ssas()
    memory_type=[ssa[0].type for ssa in input_var_ssas]
    cb_idxs=[ssa[1] for ssa in input_var_ssas]

    arg_types=[uint32]*len(memory_type)*3
    new_block = Block(arg_types=arg_types)

    read_ops, push_back_ops, write_back_ops=self.generate_data_in(module, memory_type, [ssa[0] for ssa in input_var_ssas], cb_idxs, new_block)

    # Just handle unnested loops for now
    assert len(list(loop_description.values())) == 1
    loop_descriptor=list(loop_description.values())[0]

    outer_loop_ops=self.generate_simd_loop(loop_descriptor, read_ops+push_back_ops)

    new_block.add_ops(outer_loop_ops)
    new_block.add_op(func.Return())

    body = Region()
    body.add_block(new_block)

    func_op = func.FuncOp(
      "kernel_main",
      builtin.FunctionType.from_lists(arg_types, []),
      body
    )

    new_module=builtin.ModuleOp([func_op], {"kernel_type": builtin.StringAttr("data_in")})

    return new_module, (memory_type, cb_idxs, [ssa[0] for ssa in input_var_ssas])

  def generate_device_rv_out(self, module, arith_ops_generator, loop_description):
     # For each memref first passed in is memory addresses, then bank ids, and then memory sizes (number elements)
    output_var_ssas=arith_ops_generator.get_all_output_var_ssas()
    memory_type=[ssa.type for ssa in output_var_ssas]
    cb_idxs=[dt[1].cb_idx for dt in arith_ops_generator.dependency_trees]

    arg_types=[uint32]*len(memory_type)*3
    new_block = Block(arg_types=arg_types)

    pop_front_ops=[]

    one_c=arith.Constant.from_int_and_width(1, 32)
    rv_out_ops=[one_c]
    for idx, t in enumerate(memory_type):
      is_deref_memref=isa(t, builtin.MemRefType)
      if is_deref_memref:
        element_type=t.element_type
      else:
        element_type=t
      dm_op=data_movement.DMGetNocAddrFromBankId(builtin.IntegerAttr.from_int_and_width(1, 1), new_block.args[len(memory_type)+idx], new_block.args[idx])
      dm_op.results[0].name_hint = f"src{idx}_dram_noc_addr"

      const_op=arith.Constant.from_int_and_width(cb_idxs[idx], 32)
      cb_waitfront_op=circular_buffer.CBWaitFront(const_op.results[0], one_c)
      cb_op=circular_buffer.CBGetReadPointer(const_op.results[0])
      cb_op.results[0].name_hint = "l1_read_addr_in"+str(idx)

      assert element_type.width.data % 8 == 0
      data_type_byte_width=arith.Constant.from_int_and_width(int(element_type.width.data / 8), 32)
      dt_width_conversion_op=builtin.UnrealizedConversionCastOp.get([data_type_byte_width.results[0]], [uint32])
      mem_size_bytes_op=arith.Muli(dt_width_conversion_op, new_block.args[(len(memory_type)*2)+idx])
      read_op=data_movement.DMNocAsyncWrite(cb_op.results[0], dm_op.results[0], mem_size_bytes_op)

      rv_out_ops+=[dm_op, const_op, cb_waitfront_op, cb_op, data_type_byte_width, dt_width_conversion_op, mem_size_bytes_op, read_op]

      pop_front_ops.append(circular_buffer.CBPopFront(const_op.results[0], one_c))

    rv_out_ops.append(data_movement.DMNocAsyncWriteBarrier())
    rv_out_ops+=pop_front_ops

    # Just handle unnested loops for now
    assert len(list(loop_description.values())) == 1
    loop_descriptor=list(loop_description.values())[0]

    outer_loop_ops=self.generate_simd_loop(loop_descriptor, rv_out_ops)

    new_block.add_ops(outer_loop_ops)
    new_block.add_op(func.Return())

    body = Region()
    body.add_block(new_block)

    func_op = func.FuncOp(
      "kernel_main",
      builtin.FunctionType.from_lists(arg_types, []),
      body
    )

    new_module=builtin.ModuleOp([func_op], {"kernel_type": builtin.StringAttr("data_out")})

    return new_module, (memory_type, cb_idxs, output_var_ssas)

  def generate_simd_loop(self, loop_descriptor, contained_ops):
    # For the moment assume we know this and it is a constant, should extend
    # to be an SSA value too
    if loop_descriptor.loop_op.simdlen is not None:
      simd_len=loop_descriptor.loop_op.simdlen.value.data
    else:
      simd_len=self.default_simd_len

    # To do, are assuming lower bound is 1 here, should be more flexible
    upper_bound=int(loop_descriptor.ub / simd_len)
    if upper_bound * simd_len < loop_descriptor.ub: upper_bound+=1

    ub_const=arith.Constant.create(properties={"value": builtin.IntegerAttr.from_index_int_value(upper_bound)},
                                               result_types=[builtin.i32])
    lb_const=arith.Constant.create(properties={"value": builtin.IntegerAttr.from_index_int_value(loop_descriptor.lb)},
                                               result_types=[builtin.i32])
    step_const=arith.Constant.create(properties={"value": builtin.IntegerAttr.from_index_int_value(loop_descriptor.step)},
                                               result_types=[builtin.i32])

    for_block=Block(arg_types=[builtin.i32])
    for_block.add_ops(contained_ops)
    for_block.add_op(scf.Yield())

    scf_for_loop=scf.For(lb_const, ub_const, step_const, [], for_block)

    return [ub_const, lb_const, step_const, scf_for_loop]

  def generate_device_compute(self, module, arith_ops_generator, simd_loop, loop_description):
    assert simd_loop is not None

    simd_loop.detach()
    f_lb_conv=simd_loop.lowerBound[0].owner.detach()
    f_ub_add=simd_loop.upperBound[0].owner.detach()
    f_step=simd_loop.step[0].owner.detach()

    compute_ops=[]
    out_cb_ids=[]
    out_cb_idx=16
    intermediate_cb_descriptors=IntermediateCBDescriptors(24)
    for dependency in arith_ops_generator.dependency_trees:
      # This sets the output cb index on the first node in the
      # dependency tree, as that will be an output
      dependency[1].cb_idx=out_cb_idx
      out_cb_idx+=1
      dependency[1].label_intermediates(intermediate_cb_descriptors)
      compute_ops_dep, out_cb_id=dependency[1].generate()
      compute_ops+=compute_ops_dep

    # Just handle unnested loops for now
    assert len(list(loop_description.values())) == 1
    loop_descriptor=list(loop_description.values())[0]

    outer_loop_ops=self.generate_simd_loop(loop_descriptor, compute_ops)

    new_block=Block()
    new_block.add_ops(outer_loop_ops)
    new_block.add_op(func.Return())

    body = Region()
    body.add_block(new_block)

    func_op = func.FuncOp(
      "kernel_main",
      builtin.FunctionType.from_lists([], []),
      body
    )

    new_module=builtin.ModuleOp([func_op], {"kernel_type": builtin.StringAttr("compute")})

    # Output the new module IR and also the number of intermediate CBs needed
    # as these must be created
    return new_module, intermediate_cb_descriptors


  def generate_device_all_on_rv(self, module, memory_type, references, for_op):
    assert for_op is not None
    # For each memref first passed in is memory addresses, then bank ids, and then memory sizes (number elements)
    arg_types=[uint32]*len(memory_type)*3
    new_block = Block(arg_types=arg_types)

    read_ops, push_back_ops, write_back_ops=self.generate_data_in(module, memory_type, references, range(len(memory_type)), new_block, True)

    new_block.add_ops(read_ops)

    for_op.detach()
    f_lb_conv=for_op.lb.owner
    f_lb_val=f_lb_conv.input.owner

    f_lb_conv.detach()
    f_lb_val.detach()

    f_ub_add=for_op.ub.owner
    f_ub_add_rhs=for_op.ub.owner.rhs.owner
    f_ub_add_lhs=for_op.ub.owner.lhs.owner
    f_ub_add_lhs_val=for_op.ub.owner.lhs.owner.input.owner

    f_ub_add.detach()
    f_ub_add_rhs.detach()
    f_ub_add_lhs.detach()
    f_ub_add_lhs_val.detach()

    f_step=for_op.step.owner
    f_step.detach()

    one_c_op=arith.Constant.from_int_and_width(1, 32)
    arith_add=arith.Addi(one_c_op, f_ub_add_lhs_val)

    new_block.add_ops([f_lb_val, f_ub_add_lhs_val, one_c_op, arith_add])

    b=for_op.body
    for_op.detach_region(b)

    ops_block=b.block
    b.detach_block(ops_block)

    memory = memref.Alloc([], [], builtin.MemRefType(builtin.i32, []))
    memory.results[0].name_hint = "i"
    new_block.add_op(memory)

    ops_list=list(ops_block.ops)

    idx_last_store=0
    for idx, opp in enumerate(ops_list):
      opp.detach()
      if isa(opp, memref.Store):
        idx_last_store=idx

    ops_list[0].memref.replace_by(memory.results[0])

    if idx_last_store > 0:
      ops_list=ops_list[:idx_last_store+1]

    pblock = Block(arg_types=[builtin.i32])
    ops_list[0].value.replace_by(pblock.args[0])
    pblock.add_ops(ops_list)
    pblock.add_op(scf.Yield())

    for_loop = scf.For(
            f_lb_val.results[0],
            arith_add.results[0],
            one_c_op.results[0],
            [],
            pblock
        )


    new_block.add_op(for_loop)
    new_block.add_ops(write_back_ops)
    new_block.add_op(data_movement.DMNocAsyncWriteBarrier())
    # This is a bit of a strange way of doing it, but cleans up. We reserve
    # for each CB and use its write pointer in memory, here clean them up
    new_block.add_ops(push_back_ops)
    new_block.add_op(func.Return())

    body = Region()
    body.add_block(new_block)

    func_op = func.FuncOp(
      "kernel_main",
      builtin.FunctionType.from_lists(arg_types, []),
      body
    )

    new_module=builtin.ModuleOp([func_op], {"kernel_type": builtin.StringAttr("data_in")})

    return new_module

  def create_dram_buffer(self, element_type, size_arg_idx, dram_idx):
    scalar_type=element_type
    if isa(scalar_type, builtin.MemRefType): scalar_type=scalar_type.element_type
    assert scalar_type.width.data % 8 == 0

    buffer_create_ops=[]
    data_type_byte_width=arith.Constant.from_int_and_width(int(scalar_type.width.data / 8), 32)
    if isa(element_type, builtin.MemRefType):
      # This is an array, use the size provided to the function
      new_conv=arith.IndexCastOp(size_arg_idx, builtin.i32)
      size_ssa=arith.Muli(data_type_byte_width, new_conv)
      buffer_create_ops+=[data_type_byte_width, new_conv, size_ssa]
    else:
      # This is a scalar, number of elements is therefore one
      size_ssa=data_type_byte_width
      buffer_create_ops+=[data_type_byte_width]

    dram_config_op=host.TTCreateDRAMConfig(size_ssa, size_ssa)
    dram_config_op.results[0].name_hint = "dram_config_"+str(dram_idx)

    create_buffer_op=host.TTCreateBuffer(dram_config_op.results[0])
    create_buffer_op.results[0].name_hint = f"src{dram_idx}_dram_buffer"

    return buffer_create_ops+[dram_config_op, create_buffer_op], create_buffer_op.results[0], size_ssa

  def create_cb_on_host(self, program_op, core_op, cb_idx, size_ssa):
    one_c_op=arith.Constant.from_int_and_width(1, 32)
    idx_c_op=arith.Constant.from_int_and_width(cb_idx, 32)
    cb_config_op=host.TTCreateCBConfig(one_c_op.results[0], size_ssa.results[0], idx_c_op.results[0], "int") # TODO fix type!
    cb_config_op.results[0].name_hint = f"cb_{cb_idx}_config"

    create_cb_op=host.TTCreateCircularBuffer(program_op.results[0], core_op.results[0], cb_config_op.results[0])
    create_cb_op.results[0].name_hint = f"cb_{cb_idx}"
    return [one_c_op, idx_c_op, cb_config_op, create_cb_op]

  def create_data_kernel_on_host(self, program_op, core_op, source_file, name, rv_location, noc_id, dram_buffers, zero_c_op):
    ops=[]
    kernel_create_op=host.TTCreateKernel(program_op.results[0], core_op.results[0], source_file, rv_location, noc_id)
    kernel_create_op.results[0].name_hint = name
    ops.append(kernel_create_op)

    # Set arguments for kernel
    num_mem_accesses=len(dram_buffers)
    rt_args=[None]*num_mem_accesses*2
    for idx, buff in enumerate(dram_buffers):
      mem_access=host.TTGetMemoryAddress(buff)
      ops.append(mem_access)
      rt_args[idx]=mem_access.results[0]
      rt_args[idx+num_mem_accesses]=zero_c_op.results[0]

    ops.append(host.TTSetRuntimeArgs(program_op.results[0], kernel_create_op.results[0], core_op.results[0], *rt_args))
    return ops

  def create_compute_kernel_on_host(self, program_op, core_op, source_file, name, math_fidelity, fp32_dest_acc_en, math_approx_mode):
    ops=[]
    kernel_create_op=host.TTCreateComputeKernel(program_op.results[0], core_op.results[0], source_file, math_fidelity, fp32_dest_acc_en, math_approx_mode)
    kernel_create_op.results[0].name_hint = name
    ops.append(kernel_create_op)

    # Currently don't set any runtime args for the compute kernel, will likely need to set problem size (or number of iterations anyway)
    return ops

  def generate_host(self, module, input_descs, output_descs, intermediate_descs, device_args, data_in_core_only):
    host_ops=[]

    fn_in_types=[]
    for arg in device_args:
      fn_in_types.append(arg.type)

    new_block=Block(arg_types=fn_in_types)

    zero_c_op=arith.Constant.from_int_and_width(0, 32)
    core_op=host.TTHostCore(zero_c_op.results[0], zero_c_op.results[0])
    core_op.results[0].name_hint = "core"

    device_op=host.TTCreateDevice(zero_c_op.results[0])
    device_op.results[0].name_hint = "device"

    command_queue_op=host.TTGetCommandQueue(device_op.results[0])
    command_queue_op.results[0].name_hint = "command_queue"

    program_op=host.TTCreateProgram()
    program_op.results[0].name_hint = "program"

    one_c_op=arith.Constant.from_int_and_width(1, 32)

    host_ops+=[zero_c_op, core_op, device_op, command_queue_op,
                program_op, one_c_op]

    buffer_create_ops=[]

    device_buffers={}
    data_sizes_for_cb={}

    # Create DRAM buffers for the inputs
    for input_type, cb_idx, input_idx in zip(input_descs[0], input_descs[1], input_descs[2]):
      arg_ssa=new_block.args[input_idx]
      assert isa(arg_ssa.type, builtin.MemRefType)

      buf_ops, buf_ssa, size_ssa=self.create_dram_buffer(arg_ssa.type.element_type, new_block.args[input_idx+2], input_idx)
      host_ops+=buf_ops
      device_buffers[input_idx]=buf_ssa
      data_sizes_for_cb[cb_idx]=size_ssa

    # Create DRAM buffers for the outputs
    for output_type, cb_idx, output_idx in zip(output_descs[0], output_descs[1], output_descs[2]):
      if output_idx not in device_buffers.keys():
        # Only create the buffer if it's not already created as an input
        arg_ssa=new_block.args[output_idx]
        assert isa(arg_ssa.type, builtin.MemRefType)

        buf_ops, buf_ssa, size_ssa=self.create_dram_buffer(arg_ssa.type.element_type, new_block.args[output_idx+2], output_idx)
        host_ops+=buf_ops
        device_buffers[output_idx]=buf_ssa
        data_sizes_for_cb[cb_idx]=size_ssa

    # Now create CBs for connecting the data reader core to the compute core
    for input_type, cb_idx in zip(input_descs[0], input_descs[1]):
      host_ops+=self.create_cb_on_host(program_op, core_op, cb_idx, data_sizes_for_cb[cb_idx])

    # Now create CBs for connecting the compute core to the data writer
    for output_type, cb_idx in zip(output_descs[0], output_descs[1]):
      host_ops+=self.create_cb_on_host(program_op, core_op, cb_idx, data_sizes_for_cb[cb_idx])

    # Now create CBs for intermediate CBs, these are a little different as we already gathered
    # the dependent CB which was input to the maths operation and we use that to determine
    # the data size to use
    for key,value in intermediate_descs.items():
      host_ops+=self.create_cb_on_host(program_op, core_op, key, data_sizes_for_cb[value])

    # Issue writes into the DRAM buffers from the memory passed in
    false_decl = arith.Constant(builtin.IntegerAttr.from_int_and_width(0, 1))
    host_ops.append(false_decl)
    for input_idx in input_descs[2]:
      host_ops.append(host.TTEnqueueWriteBuffer(command_queue_op.results[0], device_buffers[input_idx], new_block.args[input_idx], false_decl))

    host_ops+=self.create_data_kernel_on_host(program_op, core_op, "reader_kernel.cpp", "reader_kernel",
                                host.RISCVCoreFlagsAttr([host.RISCVCoreFlags.DATAMOVEMENT_0]),
                                0, [device_buffers[in_idx] for in_idx in input_descs[2]], zero_c_op)

    if not data_in_core_only:
      host_ops+=self.create_data_kernel_on_host(program_op, core_op, "writer_kernel.cpp", "writer_kernel",
                                host.RISCVCoreFlagsAttr([host.RISCVCoreFlags.DATAMOVEMENT_1]),
                                1, [device_buffers[out_idx] for out_idx in output_descs[2]], zero_c_op)

      host_ops+=self.create_compute_kernel_on_host(program_op, core_op, "compute_kernel.cpp", "compute_kernel",
                                host.MathFidelityFlagsAttr([host.MathFidelityFlags.HIFI4]), builtin.IntegerAttr(0, builtin.i1), builtin.IntegerAttr(0, builtin.i1))

    host_ops.append(host.TTEnqueueProgram(command_queue_op.results[0], program_op.results[0], false_decl))

    # Issue reads from the DRAM buffers into host memory
    data_writes=[]
    for output_idx in output_descs[2]:
      host_ops.append(host.TTEnqueueReadBuffer(command_queue_op.results[0], device_buffers[output_idx], new_block.args[input_idx], false_decl))

    finish_op=host.TTFinish(command_queue_op.results[0])

    close_device=host.TTCloseDevice(device_op.results[0])

    ret_op=func.Return(zero_c_op.results[0])

    host_ops+=[finish_op, close_device, ret_op]

    new_block.add_ops(host_ops)

    body = Region()
    body.add_block(new_block)

    func_op = func.FuncOp(
      "host",
      builtin.FunctionType.from_lists(fn_in_types, [builtin.i32]),
      body
    )

    new_module=builtin.ModuleOp([func_op], {"kernel_type": builtin.StringAttr("host")})

    return new_module

  def walk_to_get_inout_fn_arg_indexes(self, ssa_vals):
    var_fn_indexes=[]
    for a in ssa_vals:
      assert isa(a.owner.memref, BlockArgument)
      assert isa(a.owner.memref.owner, Block)
      omp_target_op=a.owner.memref.owner.parent.parent
      assert isa(omp_target_op, omp.TargetOp)

      tgt_in_var=omp_target_op.map_vars[a.owner.memref.index].owner
      assert isa(tgt_in_var, omp.MapInfoOp)
      fn_var=tgt_in_var.var_ptr

      assert len(fn_var) == 1
      assert isa(fn_var[0], BlockArgument)
      var_fn_indexes.append(fn_var[0].index)
    return var_fn_indexes

  def apply(self, ctx: MLContext, module: builtin.ModuleOp):
    memref_visitor=GatherMemoryPassedToDevice()
    memref_visitor.traverse(module)

    for_visitor=GatherComputeLoop()
    for_visitor.traverse(module)

    device_func_visitor=GatherDeviceFunctionArgs()
    device_func_visitor.traverse(module)

    if for_visitor.for_op is not None:
      # A plain for loop, just put this on the RV data in core
      device_in_kernel_module=self.generate_device_all_on_rv(module, memref_visitor.memory_types, memref_visitor.references, for_visitor.for_op)
      device_out_kernel_module=None
      device_compute_kernel_module=None

      # From the device function arguments we care about the memrefs (not the indexes which are start bounds and size)
      # Therefore search through and extract these
      data_access_idx=[]
      for idx, arg in enumerate(device_func_visitor.args):
        if isa(arg.type, builtin.MemRefType):
          data_access_idx.append(idx)

      assert len(memref_visitor.memory_types) == len(data_access_idx)
      in_out_desc=[memref_visitor.memory_types, list(range(len(memref_visitor.memory_types))), data_access_idx]

      host_func_op=self.generate_host(module, in_out_desc, in_out_desc, {}, device_func_visitor.args, True)
    elif for_visitor.simd_loop is not None:
      # This is a simd loop, therefore need to split across the Tensix core
      loop_gather=GatherLoops()
      loop_gather.traverse(module)
      arith_ops_generator=BuildApplicableOpDependencyTrees(loop_gather.loop_description)
      arith_ops_generator.traverse(module)

      device_in_kernel_module, input_desc=self.generate_device_rv_in(module, arith_ops_generator, loop_gather.loop_description)
      device_compute_kernel_module, intermediate_cb_descriptors=self.generate_device_compute(module, arith_ops_generator, for_visitor.simd_loop, loop_gather.loop_description)
      device_out_kernel_module, output_desc=self.generate_device_rv_out(module, arith_ops_generator, loop_gather.loop_description)

      input_var_fn_indexes=self.walk_to_get_inout_fn_arg_indexes(input_desc[2])
      output_var_fn_indexes=self.walk_to_get_inout_fn_arg_indexes(output_desc[2])

      in_pass=[input_desc[0], input_desc[1], input_var_fn_indexes]
      out_pass=[output_desc[0], output_desc[1], output_var_fn_indexes]
      host_func_op=self.generate_host(module, in_pass, out_pass, intermediate_cb_descriptors.dependent_in_cb_indexes, device_func_visitor.args, False)
    else:
      assert False

    for mod in module.regions[0].block.ops:
      assert isa(mod, builtin.ModuleOp)
      mod.detach()

    containing_mod=builtin.ModuleOp([])
    module.regions[0].move_blocks(containing_mod.regions[0])

    block = Block()
    block.add_op(device_in_kernel_module)
    if device_out_kernel_module is not None:
      block.add_op(device_out_kernel_module)
    if device_compute_kernel_module is not None:
      block.add_op(device_compute_kernel_module)
    block.add_op(host_func_op)
    module.regions[0].add_block(block)

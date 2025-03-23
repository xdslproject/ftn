from abc import ABC
from typing import TypeVar, cast
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


@dataclass
class _StencilExtractorRewriteBase(RewritePattern, ABC):
  bridgedFunctions: dict[str, Operation]
  bridge_id: int
  actioned: bool

  def __init__(self):
    self.bridgedFunctions={}
    self.bridge_id=0
    self.actioned=False

  def get_nested_type(self, in_type, search_type):
    if isinstance(in_type, search_type): return in_type
    return self.get_nested_type(in_type.type, search_type)

  @staticmethod
  def wrap_stencil_in_func(name, input_types, operations):
    """
    Wraps the extracted stencil operations for
    """

    input_types_translated=[]
    for typ in input_types:
      if isinstance(typ, fir.LLVMPointerType):
        input_types_translated.append(llvm.LLVMPointerType.typed(typ.type))
      else:
        input_types_translated.append(typ)

    block = Block(arg_types=input_types_translated)
    block.add_ops(operations)
    block.add_op(func.Return())
    body=Region()
    body.add_block(block)

    return func.FuncOp.from_region(name, input_types_translated, [], body)

class ExtractStencilOps(_StencilExtractorRewriteBase):

    def has_nested_type(in_type, search_type):
      if isinstance(in_type, search_type): return True
      if getattr(in_type, "type", None) is None: return False
      return ExtractStencilOps.has_nested_type(in_type.type, search_type)

    def getOperationsUntilExternalLoadOpFromStencilArg(arg):
      if isinstance(arg.op, stencil.LoadOp) or isinstance(arg.op,
          stencil.CastOp) or isinstance(arg.op, stencil.ExternalLoadOp):
        yield from ExtractStencilOps.getOperationsUntilExternalLoadOpFromStencilArg(arg.op.field)

      if isinstance(arg.op, stencil.LoadOp) or isinstance(arg.op,
          stencil.CastOp) or isinstance(arg.op, stencil.ExternalLoadOp) or isinstance(arg.op,
          builtin.UnrealizedConversionCastOp):
        yield arg.op

    """
    This will extract the stencil operations and replace them with a function call, converting the
    input array into fir.LLVMPointer type. Note currently this is very limited in terms of assuming
    one input only and a statically allocated array - it will get more complex as we add support
    for other things around this.

    Note that the logic here is fairly simple, we assume all the stencil operations are between the external
    load and the external store. Hence we grab all of those as a chunk and move them out
    """

    def find_ExternalLoad(ops):
      for op in ops:
        if isinstance(op, stencil.ExternalLoadOp): return op
      return None

    def stencilApplyNeedsArgsRewriting(stencil_apply):
      for arg in stencil_apply.args:
        if not isinstance(arg.typ, stencil.TempType):
          return True
      return False

    def bridge_fir_deferred_array(self, external_load_op, ptr_type):
      # This is deferred type, but just check the type is what we expect
      # Should be !fir.ref<<!fir.heap<!fir.array<?x?x?x!f64>>>
      # With any number of array dimensions
      # In this case need to load and debox it, that gives us a heap
      # we can then convert to an llvm pointer
      data_type=external_load_op.field.owner.inputs[0].typ
      assert isinstance(data_type, fir.ReferenceType)
      #assert ExtractStencilOps.has_nested_type(data_type, fir.BoxType)
      assert ExtractStencilOps.has_nested_type(data_type, fir.HeapType)

      #box_type=self.get_nested_type(data_type, fir.BoxType)
      heap_type=self.get_nested_type(data_type, fir.HeapType)

      load_op=fir.Load.create(operands=[external_load_op.field.owner.inputs[0]], result_types=[heap_type])
      #box_addr_op=fir.BoxAddr.create(operands=[load_op.results[0]], result_types=[heap_type])
      convert_op=fir.Convert.create(operands=[load_op.results[0]], result_types=[ptr_type])
      return [load_op, convert_op], convert_op

    def getFuncOpContainer(op):
      if op is None: return None
      if isinstance(op, func.FuncOp): return op
      parent_op=op.parent.parent.parent
      return ExtractStencilOps.getFuncOpContainer(parent_op)

    @op_type_rewrite_pattern
    def match_and_rewrite(self, apply_stencil_op: stencil.ApplyOp, rewriter: PatternRewriter, /):
      parent_func=ExtractStencilOps.getFuncOpContainer(apply_stencil_op)
      assert parent_func is not None and isinstance(parent_func, func.FuncOp)
      if "_InternalBridgeStencil_" in parent_func.sym_name.data: return

      input_args_to_ops={}
      stencil_ops=[]
      for input_arg in apply_stencil_op.args:
        ops=list(ExtractStencilOps.getOperationsUntilExternalLoadOpFromStencilArg(input_arg))
        input_args_to_ops[input_arg]=ops
        stencil_ops+=ops

      output_args_to_ops={}
      for output_arg in apply_stencil_op.res:
        for use in output_arg.uses:
          if isinstance(use.operation, stencil.StoreOp):
            ops=list(ExtractStencilOps.getOperationsUntilExternalLoadOpFromStencilArg(use.operation.field))
          else:
            assert False
        output_args_to_ops[output_arg]=ops
        stencil_ops+=ops

      stencil_ops.append(apply_stencil_op)

      for output_arg in apply_stencil_op.res:
        for arg_use in output_arg.uses:
          stencil_ops.append(arg_use.operation)
          # We are searching for the external store now
          ops=list(ExtractStencilOps.getOperationsUntilExternalLoadOpFromStencilArg(arg_use.operation.field))
          external_load_op=ExtractStencilOps.find_ExternalLoad(ops)
          for u in external_load_op.results[0].uses:
            if isinstance(u.operation, stencil.ExternalStoreOp):
              stencil_ops.append(u.operation)

      arg_ops=[]
      op_types=[]
      ops_to_add=[]
      for key, value in itertools.chain(input_args_to_ops.items(), output_args_to_ops.items()):
          external_load_op=ExtractStencilOps.find_ExternalLoad(value)
          if external_load_op is not None:
            nt=self.get_nested_type(external_load_op.field.typ, fir.SequenceType)
            ptr_type=fir.LLVMPointerType([nt.type])
            op_types.append(ptr_type)
            if isinstance(external_load_op.field.owner, builtin.UnrealizedConversionCastOp):
              # This is a deferred type array, so bridge it
              ops, arg=self.bridge_fir_deferred_array(external_load_op, ptr_type)
              ops_to_add+=ops
              arg_ops.append(arg)
            else:
              # Plain array type, easier to bridge
              convert_op=fir.Convert.create(operands=[external_load_op.field], result_types=[ptr_type])
              arg_ops.append(convert_op)
              ops_to_add+=[convert_op]
          else:
            # Is a scalar variable, pass this in directly
            arg_ops.append(key.owner)
            op_types.append(key.typ)

      parent=apply_stencil_op.parent

      function_name="_InternalBridgeStencil_"+str(self.bridge_id)
      self.bridge_id+=1

      call_stencil=fir.Call.create(attributes={"callee": builtin.SymbolRefAttr(function_name)}, operands=[el.results[0] for el in arg_ops], result_types=[])
      parent.insert_ops_before(ops_to_add+[call_stencil], apply_stencil_op)

      for op in stencil_ops:
        parent=op.parent_block()
        assert parent is not None
        parent.detach_op(op)

      stencil_bridge_fn=_StencilExtractorRewriteBase.wrap_stencil_in_func(function_name, op_types, stencil_ops)
      # Now if there is a scalar we need to rewrite the stencil op to refer to this
      # in the arguments (e.g. the argument passed to the extracted function)
      if ExtractStencilOps.stencilApplyNeedsArgsRewriting(apply_stencil_op):
        new_args=[]
        for index, arg in enumerate(apply_stencil_op.args):
          if isinstance(arg.typ, stencil.TempType):
            new_args.append(arg)
          else:
            new_args.append(stencil_bridge_fn.args[index])
        block=apply_stencil_op.region.block
        apply_stencil_op.region.detach_block(0)
        result_types=[]
        for res in apply_stencil_op.res:
          result_types.append(res.typ)
        new_stencil=stencil.ApplyOp.get(new_args, block, result_types)
        rewriter.replace_matched_op(new_stencil)

      self.bridgedFunctions[function_name]=stencil_bridge_fn
      self.actioned=True

class AddExternalFuncDefs(RewritePattern):
    """
    Will add in function signatures of externally bridged functions so that the Fortran code
    can call this
    """
    bridge_functions = list[str]

    def __init__(self, bridge_functions):
      self.bridge_functions=bridge_functions

    @op_type_rewrite_pattern
    def match_and_rewrite(self, module: builtin.ModuleOp,
                          rewriter: PatternRewriter, /):
        # collect all func calls to MPI functions
        funcs_to_emit: dict[str, tuple[list[Attribute],
                                       list[Attribute]]] = dict()

        def walker(op: Operation):
            if not isinstance(op, fir.Call):
                return
            if op.callee.string_value() not in self.bridge_functions:
                return
            funcs_to_emit[op.callee.string_value()] = (
                [arg.typ for arg in op.args],
                [res.typ for res in op.results],
            )

        for o in module.walk():
            walker(o)

        # for each func found, add a FuncOp to the top of the module.
        for name, types in funcs_to_emit.items():
            arg, res = types
            rewriter.insert_op_at_end(func.FuncOp.external(name, arg, res),
                                      module.body.blocks[0])

class ConnectExternalLoadToFunctionInput(RewritePattern):

  def count_instances_preceeding(ops, to_index, typ):
    instances=0
    for idx, op in enumerate(ops):
      if idx >= to_index: break
      if isinstance(op, typ): instances+=1
    return instances

  def get_nested_type(in_type, search_type):
    if isinstance(in_type, search_type): return in_type
    return ConnectExternalLoadToFunctionInput.get_nested_type(in_type.type, search_type)

  def get_array_dimension_size(array_type, dim):
    # We need to access shape of array type in reverse order, as indexing
    # is different between C and Fortran
    shape=array_type.shape.data
    access_dim=len(shape) - dim - 1
    return shape[access_dim].value.data

  def get_c_style_array_shape(array_type):
    shape=array_type.shape.data
    c_style=[]
    for i in range(len(shape)):
      c_style.insert(0, builtin.IntegerAttr.from_index_int_value(shape[i].value.data))
    return c_style

  def get_parent_arg(self, ptr_arg_num, func_op):
    current_index=-1
    for arg in func_op.args:
      if isinstance(arg.typ, llvm.LLVMPointerType):
        current_index+=1
        if current_index==ptr_arg_num: return arg
    assert False

  """
  Connects external load at the start of the bridged function with the
  input argument (note we will need to load this into a memref here)
  """
  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: stencil.ExternalLoadOp, rewriter: PatternRewriter, /):
    # If this already accepts a memref then don't need to wrap pointer
    if isinstance(op.field.typ, MemRefType): return

    array_type=ConnectExternalLoadToFunctionInput.get_nested_type(op.field.typ, fir.SequenceType)
    number_dims=len(array_type.shape.data)

    op_list=[sop for sop in op.parent.ops]
    idx = None
    for index, sop in enumerate(op_list):
      if sop == op:
        idx=index
        break

    assert idx is not None

    num_prev_external_loads=ConnectExternalLoadToFunctionInput.count_instances_preceeding(op.parent.ops, idx-1, stencil.ExternalLoadOp)

    ptr_type=op.field.typ
    # If this is not already an LLVM pointer then extract out the scalar type and type to
    # be an LLVM pointer to this
    if not isinstance(ptr_type, llvm.LLVMPointerType):
      nt=ConnectExternalLoadToFunctionInput.get_nested_type(ptr_type, fir.SequenceType)
      ptr_type=llvm.LLVMPointerType.typed(nt.type)

    array_typ=llvm.LLVMArrayType.from_size_and_type(builtin.IntAttr(number_dims), builtin.i64)
    struct_type=llvm.LLVMStructType.from_type_list([ptr_type, ptr_type, builtin.i64, array_typ, array_typ])

    func_arg=self.get_parent_arg(num_prev_external_loads, op.parent)

    undef_memref_struct=llvm.LLVMMLIRUndef.create(result_types=[struct_type])
    insert_alloc_ptr_op=llvm.LLVMInsertValue.create(attributes={"position":  builtin.DenseArrayBase.from_list(builtin.i64, [0])},
      operands=[undef_memref_struct.results[0], func_arg], result_types=[struct_type])
    insert_aligned_ptr_op=llvm.LLVMInsertValue.create(attributes={"position":  builtin.DenseArrayBase.from_list(builtin.i64, [1])},
      operands=[insert_alloc_ptr_op.results[0], func_arg], result_types=[struct_type])

    offset_op=arith.Constant.from_int_and_width(0, 64)
    insert_offset_op=llvm.LLVMInsertValue.create(attributes={"position":  builtin.DenseArrayBase.from_list(builtin.i64, [2])},
      operands=[insert_aligned_ptr_op.results[0], offset_op.results[0]], result_types=[struct_type])

    ops_to_add=[undef_memref_struct, insert_alloc_ptr_op, insert_aligned_ptr_op, offset_op, insert_offset_op]

    for dim in range(number_dims):
      dim_size=ConnectExternalLoadToFunctionInput.get_array_dimension_size(array_type, dim)
      size_op=arith.Constant.from_int_and_width(dim_size, 64)
      insert_size_op=llvm.LLVMInsertValue.create(attributes={"position":  builtin.DenseArrayBase.from_list(builtin.i64, [3, dim])},
        operands=[ops_to_add[-1].results[0], size_op.results[0]], result_types=[struct_type])

      # One for dimension stride
      stride_op=arith.Constant.from_int_and_width(1, 64)
      insert_stride_op=llvm.LLVMInsertValue.create(attributes={"position":  builtin.DenseArrayBase.from_list(builtin.i64, [4, dim])},
        operands=[insert_size_op.results[0], stride_op.results[0]], result_types=[struct_type])

      ops_to_add+=[size_op, insert_size_op, stride_op, insert_stride_op]

    #if isinstance(ptr_type, llvm.LLVMPointerType):
    target_memref_type=MemRefType.from_element_type_and_shape(ptr_type.type, ConnectExternalLoadToFunctionInput.get_c_style_array_shape(array_type))

    unrealised_conv_cast_op=builtin.UnrealizedConversionCastOp.create(operands=[insert_stride_op.results[0]], result_types=[target_memref_type])
    ops_to_add.append(unrealised_conv_cast_op)

    if idx > 0 and isinstance(op_list[idx-1], builtin.UnrealizedConversionCastOp):
      # Remove the unrealized conversion cast as it is no longer needed
      op_list[idx-1].detach()
      del op_list[idx-1]
      idx-=1

    block=op.parent
    if idx > 0:
      block.insert_ops_before(ops_to_add, op_list[idx])
    else:
      block.insert_ops_before(ops_to_add, block.first_op)

    op.operands=[unrealised_conv_cast_op.results[0]]

class ConnectExternalStoreToFunctionInput(RewritePattern):
  """
  Connects the external store at the end of the bridged function with the
  input argument (note we will need to load this into a memref and then an llvm pointer)
  """
  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: stencil.ExternalStoreOp, rewriter: PatternRewriter, /):
    # Look up the external load and then external store to that operand
    op.operands=[op.temp, op.temp.op.field]

@dataclass
class ExtractStencil(ModulePass):
  """
  This is the entry point for the transformation pass which will then apply the rewriter
  """
  name = 'extract-stencil'

  def apply(self, ctx: MLContext, module: builtin.ModuleOp):
    # First extract the stencil aspects and replace with a function call
    extractStencil=ExtractStencilOps()
    walker1 = PatternRewriteWalker(GreedyRewritePatternApplier([
              extractStencil,
    ]),
                                   apply_recursively=False)
    walker1.rewrite_module(module)

    while extractStencil.actioned:
      extractStencil.actioned=False
      walker1.rewrite_module(module)

    # Now add in external function signature for new bridged functions
    bridged_fn_names=[v for v in extractStencil.bridgedFunctions.keys()]
    walker2 = PatternRewriteWalker(AddExternalFuncDefs(bridged_fn_names))
    walker2.rewrite_module(module)

    # Create a new module with all the stencil functions as part of it
    bridged_functions=[v for v in extractStencil.bridgedFunctions.values()]
    new_module=builtin.ModuleOp(bridged_functions)

    # Rewrite the stencil functions to hook up the function/block arguments
    walker3 = PatternRewriteWalker(GreedyRewritePatternApplier([
              ConnectExternalLoadToFunctionInput(),
              ConnectExternalStoreToFunctionInput()
    ]), apply_recursively=False)
    walker3.rewrite_module(new_module)

    # Now we want to have two modules packaged together
    containing_mod=builtin.ModuleOp([])
    module.regions[0].move_blocks(containing_mod.regions[0])

    block = Block()
    block.add_ops([new_module, containing_mod])
    module.regions[0].add_block(block)

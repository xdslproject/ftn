from abc import ABC
from typing import TypeVar, cast
from dataclasses import dataclass
import itertools
from xdsl.utils.hints import isa
from xdsl.dialects import memref, scf, omp
import tenstorrent.dialects.data_movement as data_movement
import tenstorrent.dialects.host as host
import tenstorrent.dialects.circular_buffer as circular_buffer
import tenstorrent.dialects.compute as compute
from xdsl.ir import Operation, SSAValue, OpResult, Attribute, MLContext, Block, Region

from xdsl.pattern_rewriter import (RewritePattern, PatternRewriter,
                                   op_type_rewrite_pattern,
                                   PatternRewriteWalker,
                                   GreedyRewritePatternApplier)
from xdsl.passes import ModulePass
from xdsl.dialects import builtin, func, llvm, arith
from util.visitor import Visitor

uint32 = builtin.IntegerType(32, signedness=builtin.Signedness.UNSIGNED)
uint64 = builtin.IntegerType(64, signedness=builtin.Signedness.UNSIGNED)

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

class GatherComputeLoop(Visitor):
  def __init__(self):
    self.for_op=None
    self.simd_loop=None

  def traverse_for(self, for_op: scf.For):
    self.for_op=for_op

  def traverse_s_i_m_d_loop_op(self, simdloop: omp.SIMDLoopOp):
    self.simd_loop=simdloop

@dataclass(frozen=True)
class ConvertToTT(ModulePass):
  """
  This is the entry point for the transformation pass which will then apply the rewriter
  """
  name = 'convert-to-tt'

  def generate_data_in(self, module, memory_type, references, new_block, generate_writeback=False):
    write_back_ops=[]
    push_back_ops=[]
    one_c=arith.Constant.from_int_and_width(1, 32)
    new_block.add_op(one_c)
    for idx, t in enumerate(memory_type):
      is_deref_memref=isa(t, builtin.MemRefType)
      if is_deref_memref:
        element_type=t.element_type
      else:
        element_type=t
      dm_op=data_movement.DMGetNocAddrFromBankId(builtin.IntegerAttr.from_int_and_width(1, 1), new_block.args[len(memory_type)+idx], new_block.args[idx])
      dm_op.results[0].name_hint = f"src{idx}_dram_noc_addr"

      const_op=arith.Constant.from_int_and_width(idx, 32)
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

      new_block.add_ops([dm_op, const_op, cb_reserve_op, cb_op, data_type_byte_width, dt_width_conversion_op, mem_size_bytes_op, read_op, conversion_op])

      for use in references[idx].uses:
        if isa(use.operation, memref.Load):
          # Currently just doing this for passing array/data (e.g. memref of memref, ignoring constants but probably want to handle that too)
          if is_deref_memref:
            use.operation.detach()
            use.operation.results[0].replace_by(conversion_op.results[0])

      push_back_ops.append(circular_buffer.CBPushBack(const_op.results[0], one_c))

      if generate_writeback:
        write_op=data_movement.DMNocAsyncWrite(cb_op.results[0], dm_op.results[0], mem_size_bytes_op)
        write_back_ops.append(write_op)

    new_block.add_op(data_movement.DMNocAsyncReadBarrier())

    return push_back_ops, write_back_ops

  def generate_device_rv_in(self, module, memory_type, references):
    # For each memref first passed in is memory addresses, then bank ids, and then memory sizes (number elements)
    arg_types=[uint32]*len(memory_type)*3
    new_block = Block(arg_types=arg_types)

    push_back_ops, write_back_ops=self.generate_data_in(module, memory_type, references, new_block)

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

  def generate_device_rv_out(self, module, memory_type, references):
     # For each memref first passed in is memory addresses, then bank ids, and then memory sizes (number elements)
    arg_types=[uint32]*len(memory_type)*3
    new_block = Block(arg_types=arg_types)

    pop_front_ops=[]

    one_c=arith.Constant.from_int_and_width(1, 32)
    new_block.add_op(one_c)
    for idx, t in enumerate(memory_type):
      is_deref_memref=isa(t, builtin.MemRefType)
      if is_deref_memref:
        element_type=t.element_type
      else:
        element_type=t
      dm_op=data_movement.DMGetNocAddrFromBankId(builtin.IntegerAttr.from_int_and_width(1, 1), new_block.args[len(memory_type)+idx], new_block.args[idx])
      dm_op.results[0].name_hint = f"src{idx}_dram_noc_addr"

      const_op=arith.Constant.from_int_and_width(idx, 32)
      cb_waitfront_op=circular_buffer.CBWaitFront(const_op.results[0], one_c)
      cb_op=circular_buffer.CBGetReadPointer(const_op.results[0])
      cb_op.results[0].name_hint = "l1_read_addr_in"+str(idx)

      assert element_type.width.data % 8 == 0
      data_type_byte_width=arith.Constant.from_int_and_width(int(element_type.width.data / 8), 32)
      dt_width_conversion_op=builtin.UnrealizedConversionCastOp.get([data_type_byte_width.results[0]], [uint32])
      mem_size_bytes_op=arith.Muli(dt_width_conversion_op, new_block.args[(len(memory_type)*2)+idx])
      read_op=data_movement.DMNocAsyncWrite(cb_op.results[0], dm_op.results[0], mem_size_bytes_op)

      new_block.add_ops([dm_op, const_op, cb_waitfront_op, cb_op, data_type_byte_width, dt_width_conversion_op, mem_size_bytes_op, read_op])

      pop_front_ops.append(circular_buffer.CBPopFront(const_op.results[0], one_c))

    new_block.add_op(data_movement.DMNocAsyncWriteBarrier())
    new_block.add_ops(pop_front_ops)

    new_block.add_op(func.Return())

    body = Region()
    body.add_block(new_block)

    func_op = func.FuncOp(
      "kernel_main",
      builtin.FunctionType.from_lists(arg_types, []),
      body
    )

    new_module=builtin.ModuleOp([func_op], {"kernel_type": builtin.StringAttr("data_out")})

    return new_module


  def generate_device_compute(self, module, memory_type, references, simd_loop):
    assert simd_loop is not None

    simd_loop.detach()
    f_lb_conv=simd_loop.lowerBound[0].owner.detach()
    f_ub_add=simd_loop.upperBound[0].owner.detach()
    f_step=simd_loop.step[0].owner.detach()

    one_c=arith.Constant.from_int_and_width(1, 32)

    compute_ops=[one_c]
    for idx, t in enumerate(memory_type):
      const_op=arith.Constant.from_int_and_width(idx, 32)
      cb_waitfront_op=circular_buffer.CBWaitFront(const_op.results[0], one_c)
      compute_ops+=[const_op, cb_waitfront_op]

    cb1_i_op=arith.Constant.from_int_and_width(0, 32)
    cb1_op=builtin.UnrealizedConversionCastOp.get([cb1_i_op], [uint32])
    cb2_i_op=arith.Constant.from_int_and_width(1, 32)
    cb2_op=builtin.UnrealizedConversionCastOp.get([cb2_i_op], [uint32])
    const_16_i_op=arith.Constant.from_int_and_width(16, 32)
    const_16_op=builtin.UnrealizedConversionCastOp.get([const_16_i_op], [uint32])
    false_op=arith.Constant.from_int_and_width(0, 1)

    binary_init_common_op=compute.BinaryOpInitCommon(cb1_op, cb2_op, const_16_op)
    add_init_op=compute.AddInit(cb1_op, cb2_op, false_op)

    zero_i_op=arith.Constant.from_int_and_width(0, 32)
    zero_op=builtin.UnrealizedConversionCastOp.get([zero_i_op], [uint32])
    dst_i_op=arith.Constant.from_int_and_width(0, 32)
    dst_op=builtin.UnrealizedConversionCastOp.get([dst_i_op], [uint32])
    acquire_regs_op=compute.RegsAcquire()
    add_tile_op=compute.Add(cb1_op, cb2_op, zero_op, zero_op, dst_op)
    commit_regs_op=compute.RegsCommit()

    cb_out_i_op=arith.Constant.from_int_and_width(2, 32)
    cb_out_op=builtin.UnrealizedConversionCastOp.get([cb_out_i_op], [uint32])
    wait_regs_op=compute.RegsWait()
    pack_tile_op=compute.PackTile(builtin.IntegerAttr.from_index_int_value(0), zero_op, cb_out_op, zero_op)
    release_regs_op=compute.RegsRelease()

    compute_ops+=[cb1_i_op, cb1_op, cb2_i_op, cb2_op, const_16_i_op, const_16_op, false_op, binary_init_common_op,
                    add_init_op, zero_i_op, zero_op, dst_i_op, dst_op, acquire_regs_op,
                    add_tile_op, commit_regs_op, cb_out_i_op, cb_out_op, wait_regs_op, pack_tile_op, release_regs_op]

    for idx, t in enumerate(memory_type):
      const_op=arith.Constant.from_int_and_width(idx, 32)
      cb_popfront_op=circular_buffer.CBPopFront(const_op.results[0], one_c)
      compute_ops+=[const_op, cb_popfront_op]

    new_block=Block()
    new_block.add_ops(compute_ops)
    new_block.add_op(func.Return())

    body = Region()
    body.add_block(new_block)

    func_op = func.FuncOp(
      "kernel_main",
      builtin.FunctionType.from_lists([], []),
      body
    )

    new_module=builtin.ModuleOp([func_op], {"kernel_type": builtin.StringAttr("compute")})

    return new_module


  def generate_device_all_on_rv(self, module, memory_type, references, for_op):
    assert for_op is not None
    # For each memref first passed in is memory addresses, then bank ids, and then memory sizes (number elements)
    arg_types=[uint32]*len(memory_type)*3
    new_block = Block(arg_types=arg_types)

    push_back_ops, write_back_ops=self.generate_data_in(module, memory_type, references, new_block, True)

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

  def generate_host(self, module, memory_type, device_args):
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

    host_buffers=[]
    device_buffers=[]
    current_arg_idx=0
    for idx, mem in enumerate(memory_type):
      arg_ssa=new_block.args[current_arg_idx]
      assert isa(arg_ssa.type, builtin.MemRefType)

      element_type=arg_ssa.type.element_type
      if isa(element_type, builtin.MemRefType): element_type=element_type.element_type

      assert element_type.width.data % 8 == 0
      data_type_byte_width=arith.Constant.from_int_and_width(int(element_type.width.data / 8), 32)

      if isa(arg_ssa.type.element_type, builtin.MemRefType):
        # This is an array, use the size provided to the function
        new_conv=arith.IndexCastOp(new_block.args[current_arg_idx+2], builtin.i32)
        size_ssa=arith.Muli(data_type_byte_width, new_conv)
        current_arg_idx+=3
        buffer_create_ops+=[data_type_byte_width, new_conv, size_ssa]
      else:
        # This is a scalar, number of elements is therefore one
        size_ssa=data_type_byte_width
        current_arg_idx+=1
        buffer_create_ops+=[data_type_byte_width]

      dram_config_op=host.TTCreateDRAMConfig(size_ssa, size_ssa)
      dram_config_op.results[0].name_hint = "dram_config_"+str(idx)

      create_buffer_op=host.TTCreateBuffer(dram_config_op.results[0])
      create_buffer_op.results[0].name_hint = f"src{idx}_dram_buffer"

      device_buffers.append(create_buffer_op.results[0])

      idx_c_op=arith.Constant.from_int_and_width(idx, 32)
      cb_config_op=host.TTCreateCBConfig(one_c_op.results[0], size_ssa.results[0], idx_c_op.results[0], "int")
      cb_config_op.results[0].name_hint = f"cb_{idx}_config"

      create_cb_op=host.TTCreateCircularBuffer(program_op.results[0], core_op.results[0], cb_config_op.results[0])
      create_cb_op.results[0].name_hint = f"cb_{idx}"

      host_buffers.append(arg_ssa)

      buffer_create_ops+=[dram_config_op, create_buffer_op, idx_c_op, cb_config_op]

    host_ops+=buffer_create_ops

    false_decl = arith.Constant(builtin.IntegerAttr.from_int_and_width(0, 1))
    host_ops.append(false_decl)

    data_reads=[]
    for idx, mem in enumerate(memory_type):
      enqueueWrite=host.TTEnqueueWriteBuffer(command_queue_op.results[0], device_buffers[idx], host_buffers[idx], false_decl)
      data_reads+=[enqueueWrite]

    host_ops+=data_reads

    kernel_create_op=host.TTCreateKernel(program_op.results[0], core_op.results[0], "ftn_kernel.cpp", host.RISCVCoreFlagsAttr([host.RISCVCoreFlags.DATAMOVEMENT_0]), 0)
    kernel_create_op.results[0].name_hint = "kernel"

    host_ops.append(kernel_create_op)

    mem_access=[]
    rt_args=[None]*len(memory_type)*2
    for idx, mem in enumerate(memory_type):
      mem_access.append(host.TTGetMemoryAddress(device_buffers[idx]))
      rt_args[idx]=mem_access[-1].results[0]
      rt_args[idx+len(memory_type)]=zero_c_op.results[0]

    host_ops+=mem_access

    setRTArgs=host.TTSetRuntimeArgs(program_op.results[0], kernel_create_op.results[0], core_op.results[0], *rt_args)

    enqueue_prog=host.TTEnqueueProgram(command_queue_op.results[0], program_op.results[0], false_decl)

    data_writes=[]
    for idx, mem in enumerate(memory_type):
      enqueueRead=host.TTEnqueueReadBuffer(command_queue_op.results[0], device_buffers[idx], host_buffers[idx], false_decl)
      data_writes+=[enqueueRead]

    finish_op=host.TTFinish(command_queue_op.results[0])

    close_device=host.TTCloseDevice(device_op.results[0])

    ret_op=func.Return(zero_c_op.results[0])

    host_ops+=[setRTArgs, enqueue_prog, *data_writes, finish_op, close_device, ret_op]

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
    elif for_visitor.simd_loop is not None:
      # This is a simd loop, therefore need to split across the Tensix core
      device_in_kernel_module=self.generate_device_rv_in(module, memref_visitor.memory_types, memref_visitor.references)
      device_out_kernel_module=self.generate_device_rv_out(module, memref_visitor.memory_types, memref_visitor.references)
      device_compute_kernel_module=self.generate_device_compute(module, memref_visitor.memory_types, memref_visitor.references, for_visitor.simd_loop)
    else:
      assert False

    host_func_op=self.generate_host(module, memref_visitor.memory_types, device_func_visitor.args)

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

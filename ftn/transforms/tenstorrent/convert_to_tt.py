from abc import ABC
from typing import TypeVar, cast
from dataclasses import dataclass
import itertools
from xdsl.utils.hints import isa
from xdsl.dialects import memref, scf, omp
import tenstorrent.dialects.data_movement as data_movement
import tenstorrent.dialects.host as host
import tenstorrent.dialects.circular_buffer as circular_buffer
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

class GatherMemoryTypes(Visitor):
  def __init__(self):
    self.memory_types=[]
    self.references=[]

  def traverse_alloca(self, alloca_op: memref.Alloca):
    self.memory_types.append(alloca_op.results[0].type.element_type.element_type)
    assert len(alloca_op.results[0].uses) == 1
    assert isa(list(alloca_op.results[0].uses)[0].operation, memref.Load)
    self.references.append(list(alloca_op.results[0].uses)[0].operation.results[0])

class GatherFor(Visitor):
  def __init__(self):
    self.for_op=None

  def traverse_for(self, for_op: scf.For):
    self.for_op=for_op

@dataclass(frozen=True)
class ConvertToTT(ModulePass):
  """
  This is the entry point for the transformation pass which will then apply the rewriter
  """
  name = 'convert-to-tt'

  def generate_device(self, module, memory_type, references, for_op):
    assert for_op is not None

    arg_types=[uint32]*len(memory_type)*2

    new_block = Block(arg_types=arg_types)

    ops=[]
    ssa_res=[]
    write_back_ops=[]
    for idx, t in enumerate(memory_type):
      dm_op=data_movement.DMGetNocAddrFromBankId(builtin.IntegerAttr.from_int_and_width(1, 1), new_block.args[len(memory_type)+idx], new_block.args[idx])
      dm_op.results[0].name_hint = f"src{idx}_dram_noc_addr"

      const_op=arith.Constant.from_int_and_width(idx, 32)
      cb_op=circular_buffer.CBGetWritePointer(const_op.results[0])
      cb_op.results[0].name_hint = "l1_write_addr_in"+str(idx)

      size_op=arith.Constant.from_int_and_width(400, 32)
      conversion_size_op=builtin.UnrealizedConversionCastOp.get([size_op.results[0]], [uint32])
      read_op=data_movement.DMNocAsyncRead(dm_op.results[0], cb_op.results[0], conversion_size_op.results[0])

      target_memref=builtin.MemRefType(t, [100])
      conversion_op=builtin.UnrealizedConversionCastOp.get([cb_op.results[0]], [target_memref])
      conversion_op.results[0].name_hint = f"src{idx}_data"

      new_block.add_ops([dm_op, const_op, cb_op, size_op, conversion_size_op, read_op, conversion_op])
      ssa_res.append(conversion_op.results[0])
      references[idx].replace_by(conversion_op.results[0])

      write_op=data_movement.DMNocAsyncWrite(cb_op.results[0], dm_op.results[0], conversion_size_op.results[0])
      write_back_ops.append(write_op)

    new_block.add_op(data_movement.DMNocAsyncReadBarrier())

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

  def generate_host(self, module, memory_type):
    host_ops=[]

    zero_c_op=arith.Constant.from_int_and_width(0, 32)
    core_op=host.TTHostCore(zero_c_op.results[0], zero_c_op.results[0])
    core_op.results[0].name_hint = "core"

    device_op=host.TTCreateDevice(zero_c_op.results[0])
    device_op.results[0].name_hint = "device"

    command_queue_op=host.TTGetCommandQueue(device_op.results[0])
    command_queue_op.results[0].name_hint = "command_queue"

    program_op=host.TTCreateProgram()
    program_op.results[0].name_hint = "program"

    size_c_op=arith.Constant.from_int_and_width(400, 32)
    dram_config_op=host.TTCreateDRAMConfig(size_c_op, size_c_op)
    dram_config_op.results[0].name_hint = "dram_config"

    one_c_op=arith.Constant.from_int_and_width(1, 32)

    host_ops+=[zero_c_op, core_op, device_op, command_queue_op,
                program_op, size_c_op, dram_config_op, one_c_op]

    buffer_create_ops=[]

    host_buffers=[]
    device_buffers=[]
    for idx, mem in enumerate(memory_type):
      create_buffer_op=host.TTCreateBuffer(dram_config_op.results[0])
      create_buffer_op.results[0].name_hint = f"src{idx}_dram_buffer"

      device_buffers.append(create_buffer_op.results[0])

      idx_c_op=arith.Constant.from_int_and_width(idx, 32)
      cb_config_op=host.TTCreateCBConfig(one_c_op.results[0], size_c_op.results[0], idx_c_op.results[0], "int")
      cb_config_op.results[0].name_hint = f"cb_{idx}_config"

      create_cb_op=host.TTCreateCircularBuffer(program_op.results[0], core_op.results[0], cb_config_op.results[0])
      create_cb_op.results[0].name_hint = f"cb_{idx}"

      data_buffer=memref.Alloc.get(mem, shape=[400])
      data_buffer.results[0].name_hint = "host_src"+str(idx)

      host_buffers.append(data_buffer)

      buffer_create_ops+=[create_buffer_op, idx_c_op, cb_config_op, data_buffer]

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

    body = Region()
    body.add_block(Block(host_ops))

    func_op = func.FuncOp(
      "main",
      builtin.FunctionType.from_lists([], [builtin.i32]),
      body
    )

    new_module=builtin.ModuleOp([func_op], {"kernel_type": builtin.StringAttr("host")})

    return new_module


  def apply(self, ctx: MLContext, module: builtin.ModuleOp):
    memref_visitor=GatherMemoryTypes()
    memref_visitor.traverse(module)

    for_visitor=GatherFor()
    for_visitor.traverse(module)

    target_func_op=self.generate_device(module, memref_visitor.memory_types, memref_visitor.references, for_visitor.for_op)

    host_func_op=self.generate_host(module, memref_visitor.memory_types)

    for mod in module.regions[0].block.ops:
      assert isa(mod, builtin.ModuleOp)
      mod.detach()

    containing_mod=builtin.ModuleOp([])
    module.regions[0].move_blocks(containing_mod.regions[0])

    block = Block()
    block.add_ops([target_func_op, host_func_op])
    module.regions[0].add_block(block)

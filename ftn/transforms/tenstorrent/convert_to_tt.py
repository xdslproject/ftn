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
      dm_op_r=data_movement.DMGetNocAddrFromBankId(builtin.IntegerAttr.from_int_and_width(1, 1), new_block.args[len(memory_type)+idx], new_block.args[idx])
      dm_op_alloc = memref.Alloc([], [], builtin.MemRefType(uint64, []))
      dm_store = memref.Store.get(dm_op_r, dm_op_alloc.results[0], [])            
      dm_op_alloc.results[0].name_hint = f"src{idx}_dram_noc_addr"
      
      const_op=arith.Constant.from_int_and_width(idx, 32)
      cb_op_r=circular_buffer.CBGetWritePointer(const_op.results[0])
      cb_op_alloc = memref.Alloc([], [], builtin.MemRefType(uint32, []))
      cb_store = memref.Store.get(cb_op_r, cb_op_alloc.results[0], [])            
      cb_op_alloc.results[0].name_hint = "l1_write_addr_in"+str(idx)
      
      size_op=arith.Constant.from_int_and_width(400, uint32)
      load_dm=memref.Load.get(dm_op_alloc.results[0], [])
      load_cb=memref.Load.get(cb_op_alloc.results[0], [])
      read_op=data_movement.DMNocAsyncRead(load_dm.results[0], load_cb.results[0], size_op.results[0])   
         
      target_memref=builtin.MemRefType(t, [100])
      conversion_op=builtin.UnrealizedConversionCastOp.get([load_cb.results[0]], [target_memref])
      conversion_op.results[0].name_hint = f"src{idx}_data"
      
      new_block.add_ops([dm_op_r, dm_op_alloc, dm_store, const_op, cb_op_r, cb_op_alloc, cb_store, size_op, load_dm, load_cb, read_op, conversion_op])
      ssa_res.append(conversion_op.results[0])
      references[idx].replace_by(conversion_op.results[0])
      
      load_dm_w=memref.Load.get(dm_op_alloc.results[0], [])
      load_cb_r=memref.Load.get(cb_op_alloc.results[0], [])
      write_op=data_movement.DMNocAsyncWrite(load_cb_r.results[0], load_dm_w.results[0], size_op.results[0])
      write_back_ops+=[load_dm_w, load_cb_r, write_op]
      
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
    core_op_r=host.TTHostCore(zero_c_op.results[0], zero_c_op.results[0])
    core_op_alloc = memref.Alloc([], [], builtin.MemRefType(core_op_r.results[0].type, []))
    core_store = memref.Store.get(core_op_r, core_op_alloc.results[0], [])
    core_op_alloc.results[0].name_hint = "core"
    
    device_op_r=host.TTCreateDevice(zero_c_op.results[0])
    device_op_alloc = memref.Alloc([], [], builtin.MemRefType(device_op_r.results[0].type, []))
    device_store = memref.Store.get(device_op_r, device_op_alloc.results[0], [])
    device_op_alloc.results[0].name_hint = "device"
    
    device_load=memref.Load.get(device_op_alloc.results[0], [])
    command_queue_op_r=host.TTGetCommandQueue(device_load.results[0])
    command_queue_alloc = memref.Alloc([], [], builtin.MemRefType(command_queue_op_r.results[0].type, []))
    command_queue_store = memref.Store.get(command_queue_op_r, command_queue_alloc.results[0], [])
    command_queue_alloc.results[0].name_hint = "command_queue"        
    
    program_op_r=host.TTCreateProgram()
    program_op_alloc = memref.Alloc([], [], builtin.MemRefType(program_op_r.results[0].type, []))
    program_op_store = memref.Store.get(program_op_r, program_op_alloc.results[0], [])
    program_op_alloc.results[0].name_hint = "program"
    program_load=memref.Load.get(program_op_alloc.results[0], [])
    
    size_c_op=arith.Constant.from_int_and_width(400, 32)
    dram_config_op_r=host.TTCreateDRAMConfig(size_c_op, size_c_op)
    dram_config_op_alloc = memref.Alloc([], [], builtin.MemRefType(dram_config_op_r.results[0].type, []))
    dram_config_op_store = memref.Store.get(dram_config_op_r, dram_config_op_alloc.results[0], [])
    dram_config_op_alloc.results[0].name_hint = "dram_config"
    dram_config_load=memref.Load.get(dram_config_op_alloc.results[0], [])
        
    one_c_op=arith.Constant.from_int_and_width(1, 32)
    
    host_ops+=[zero_c_op, core_op_r, core_op_alloc, core_store, device_op_r, device_op_alloc, device_store, device_load, command_queue_op_r, 
                command_queue_alloc, command_queue_store, program_op_r, program_op_alloc, program_op_store, program_load, size_c_op, 
                dram_config_op_r, dram_config_op_alloc, dram_config_op_store, dram_config_load, one_c_op]
    
    buffer_create_ops=[]    
    
    host_buffers=[]
    device_buffers=[]
    for idx, mem in enumerate(memory_type):
      create_buffer_op_r=host.TTCreateBuffer(dram_config_load.results[0])    
      create_buffer_alloc = memref.Alloc([], [], builtin.MemRefType(create_buffer_op_r.results[0].type, []))
      create_buffer_store = memref.Store.get(create_buffer_op_r, create_buffer_alloc.results[0], []) 
      create_buffer_alloc.results[0].name_hint = f"src{idx}_dram_buffer"  
      buffer_load=memref.Load.get(create_buffer_alloc.results[0], [])
      
      device_buffers.append(buffer_load.results[0])    
      
      idx_c_op=arith.Constant.from_int_and_width(idx, 32)
      cb_config_r=host.TTCreateCBConfig(one_c_op.results[0], size_c_op.results[0], idx_c_op.results[0], "int")
      cb_config_alloc = memref.Alloc([], [], builtin.MemRefType(cb_config_r.results[0].type, []))
      cb_config_store = memref.Store.get(cb_config_r, cb_config_alloc.results[0], [])       
      cb_config_alloc.results[0].name_hint = f"cb_{idx}_config"
      cb_config_load=memref.Load.get(cb_config_alloc.results[0], [])      
      
      core_load=memref.Load.get(core_op_alloc.results[0], [])
      create_cb=host.TTCreateCircularBuffer(program_load.results[0], core_load.results[0], cb_config_load.results[0])
      create_cb.results[0].name_hint = f"cb_{idx}"
      
      data_buffer=memref.Alloc.get(mem, shape=[400])
      data_buffer.results[0].name_hint = "host_src"+str(idx)
      
      host_buffers.append(data_buffer)
      
      buffer_create_ops+=[create_buffer_op_r, create_buffer_alloc, create_buffer_store, buffer_load, idx_c_op, core_load, 
                          cb_config_r, cb_config_alloc, cb_config_store, cb_config_load, data_buffer]   
      
    host_ops+=buffer_create_ops
      
    false_decl = arith.Constant(builtin.IntegerAttr.from_int_and_width(0, 1))
    host_ops.append(false_decl)
    
    data_reads=[]
    for idx, mem in enumerate(memory_type):
      cmd_queue_load=memref.Load.get(command_queue_alloc.results[0], [])
      enqueueWrite=host.TTEnqueueWriteBuffer(cmd_queue_load.results[0], device_buffers[idx], host_buffers[idx], false_decl)
      data_reads+=[cmd_queue_load, enqueueWrite]
      
    host_ops+=data_reads
      
    core_load=memref.Load.get(core_op_alloc.results[0], [])
    kernelCreate=host.TTCreateKernel(program_load.results[0], core_load.results[0], "ftn_kernel.cpp", host.RISCVCoreFlagsAttr([host.RISCVCoreFlags.DATAMOVEMENT_0]), 0)
    kernel_alloc = memref.Alloc([], [], builtin.MemRefType(kernelCreate.results[0].type, []))
    kernel_store = memref.Store.get(kernelCreate, kernel_alloc.results[0], [])       
    kernel_alloc.results[0].name_hint = "kernel"
    kernel_load=memref.Load.get(kernel_alloc.results[0], [])      
    
    host_ops+=[core_load, kernelCreate, kernel_alloc, kernel_store, kernel_load]
    
    mem_access=[]
    rt_args=[None]*len(memory_type)*2
    for idx, mem in enumerate(memory_type):
      mem_access.append(host.TTGetMemoryAddress(device_buffers[idx]))
      rt_args[idx]=mem_access[-1].results[0]
      rt_args[idx+len(memory_type)]=zero_c_op.results[0]
      
    host_ops+=mem_access
      
    core_load=memref.Load.get(core_op_alloc.results[0], [])
    setRTArgs=host.TTSetRuntimeArgs(program_load.results[0], kernel_load.results[0], core_load.results[0], *rt_args)
    
    cmd_queue_load=memref.Load.get(command_queue_alloc.results[0], [])
    enqueue_prog=host.TTEnqueueProgram(cmd_queue_load.results[0], program_load.results[0], false_decl)
    
    data_writes=[]
    for idx, mem in enumerate(memory_type):      
      enqueueRead=host.TTEnqueueReadBuffer(cmd_queue_load.results[0], device_buffers[idx], host_buffers[idx], false_decl)
      data_writes+=[enqueueRead]
    
    finish_op=host.TTFinish(cmd_queue_load.results[0])
    
    device_load=memref.Load.get(device_op_alloc.results[0], [])
    close_device=host.TTCloseDevice(device_load.results[0])
    
    ret_op=func.Return(zero_c_op.results[0])
    
    host_ops+=[core_load, setRTArgs, cmd_queue_load, enqueue_prog, *data_writes, device_load, finish_op, close_device, ret_op]
    
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

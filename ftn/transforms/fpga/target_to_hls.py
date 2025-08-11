from dataclasses import dataclass, field
from socket import if_indextoname
from xdsl.utils.hints import isa
from xdsl.dialects import memref, scf, omp
from xdsl.context import Context
from xdsl.ir import SSAValue, Block

from xdsl.pattern_rewriter import (RewritePattern, PatternRewriter,
                                   op_type_rewrite_pattern,
                                   PatternRewriteWalker,
                                   GreedyRewritePatternApplier)
from xdsl.passes import ModulePass
from xdsl.dialects import builtin, func, arith, math
from xdsl.rewriter import InsertPoint
from xdsl.dialects.experimental.hls import PragmaPipelineOp, PragmaUnrollOp
from ftn.transforms.to_core.components.control_flow import generate_index_inversion_at_start_of_loop

class DerefMemrefs:
    @staticmethod
    def deref_scalar_memops(scalar_ssa: SSAValue, rewriter: PatternRewriter):
        for use in scalar_ssa.uses:
            if isinstance(use.operation, memref.LoadOp):
                load_op = use.operation
                # NOTE: the operand of the load operation is not a memref anymore, we have dereferenced it, 
                # so we forward it.
                load_op.res.replace_by(load_op.memref)
                rewriter.erase_op(use.operation)
            elif isinstance(use.operation, memref.StoreOp):
                rewriter.erase_op(use.operation)

    @staticmethod
    def deref_memref_memops(memref_ssa: SSAValue, rewriter: PatternRewriter):
        for use in memref_ssa.uses:
            if isinstance(use.operation, memref.LoadOp):
                # The first load was used to load the pointer to the array. The index to retrieve an element from the array is applied 
                # in the next load. Since we have dereferenced the first pointer, we need to end up with a single load that accesses
                # the array directly.
                ptr_load_op = use.operation

                # FIXME: this is assuming each dereferencing load only has one use
                for ptr_use in ptr_load_op.res.uses:
                    if isinstance(ptr_use.operation, memref.LoadOp):
                        array_load_op = ptr_use.operation
                        array_idx = array_load_op.indices

                        new_load_op = memref.LoadOp.get(memref_ssa, array_idx)
                        array_load_op.res.replace_by(ptr_load_op.res)
                        rewriter.replace_op(ptr_load_op, new_load_op)
                        rewriter.erase_op(array_load_op)

                    elif isinstance(ptr_use.operation, memref.StoreOp):
                        array_store_op = ptr_use.operation
                        array_idx = array_store_op.indices
                        new_store_op = memref.StoreOp.get(array_store_op.value, memref_ssa, array_idx)
                        rewriter.insert_op(new_store_op, InsertPoint.before(ptr_load_op))
                        rewriter.erase_op(array_store_op)
                        rewriter.erase_op(ptr_load_op)

    @staticmethod
    def deref_args(func_op: func.FuncOp, rewriter : PatternRewriter):
        """Dereference the arguments of a function operation."""
        new_input_types = []

        for arg in func_op.body.block.args:
            if isa(arg.type, builtin.MemRefType):
                deref_type = arg.type.element_type
                func_op.replace_argument_type(arg, deref_type, rewriter)
                new_input_types.append(deref_type)

class RemoveOps:
    @staticmethod
    def transform_omp_loop_nest_into_scf_for(loop_nest_op : omp.LoopNestOp, rewriter: PatternRewriter):
        flat_loop_body = rewriter.move_region_contents_to_new_regions(loop_nest_op.body)
        rewriter.replace_op(flat_loop_body.block.last_op, scf.YieldOp())
        flat_lb = arith.IndexCastOp(loop_nest_op.lowerBound[0], builtin.IndexType())
        flat_ub = arith.IndexCastOp(loop_nest_op.upperBound[0], builtin.IndexType())
        flat_step = arith.IndexCastOp(loop_nest_op.step[0], builtin.IndexType())

        rewriter.insert_op(flat_lb, InsertPoint.after(loop_nest_op.lowerBound[0].owner))
        rewriter.insert_op(flat_ub, InsertPoint.after(loop_nest_op.upperBound[0].owner))
        rewriter.insert_op(flat_step, InsertPoint.after(loop_nest_op.step[0].owner))
        rewriter.replace_value_with_new_type(flat_loop_body.block.args[0], builtin.IndexType())

        # TODO: convert between i32 and index where appropriate, since omp.loop_nest operates with i32 and 
        # scf.for with index.
        for arg_use in flat_loop_body.block.args[0].uses:
            if isinstance(arg_use.operation, memref.StoreOp):
                store_op = arg_use.operation
                
                ## Replace the store ops first
                if isinstance(store_op.memref.type.element_type, builtin.IntegerType):
                    index_to_i32 = arith.IndexCastOp(store_op.value, builtin.i32)
                    store_op.value.replace_by_if(index_to_i32.result, lambda use: isinstance(use.operation, memref.StoreOp))
                    rewriter.insert_op(index_to_i32, InsertPoint.before(store_op))

                else:
                    alloca_op = store_op.memref.owner
                    assert isinstance(alloca_op, memref.AllocaOp)
                    index_alloca = memref.AllocaOp.get(builtin.IndexType(), shape=alloca_op.memref.type.shape)
                    rewriter.replace_op(alloca_op, index_alloca)

                    idx_memref = store_op.memref
                    for idx_memref_use in idx_memref.uses:
                        if isinstance(idx_memref_use.operation, memref.LoadOp):
                            load_op = idx_memref_use.operation
                            index_load = memref.LoadOp.get(load_op.memref, load_op.indices)
                            rewriter.replace_op(load_op, index_load)

                            # Original type of the block arg of the loop nest op
                            cast_ind_var = arith.IndexCastOp(index_load.res, builtin.i32)
                            rewriter.insert_op(cast_ind_var, InsertPoint.after(index_load))
                            index_load.res.replace_by_if(cast_ind_var.result, lambda use: use.operation != cast_ind_var)



        flat_loop = scf.ForOp(flat_lb, flat_ub, flat_step, (), flat_loop_body)
        #flat_loop = scf.ForOp(loop_nest_op.lowerBound[0], omp_loop_op.upperBound[0], omp_loop_op.step[0], (), flat_loop_body)
        rewriter.replace_op(loop_nest_op, flat_loop)

        loop_body_ops = []
        if loop_nest_op.step[0].owner.value.value.data < 0:
            # If the step is negative, we need to reverse the bounds
            flat_lb, flat_ub = flat_ub, flat_lb
            loop_body_ops = generate_index_inversion_at_start_of_loop(
                flat_loop_body.block.args[0],
                flat_lb,
                flat_ub,
                flat_loop_body.block.args[0].type
            )
            rewriter.insert_op(loop_body_ops, InsertPoint.at_start(flat_loop.body.block))
            new_step = loop_body_ops[-1].result
            flat_loop_body.block.args[0].replace_by_if(new_step, lambda use: flat_loop_body.block.get_operation_index(use.operation) 
                                                       > flat_loop_body.block.get_operation_index(new_step.owner))
            
            abs_step = math.AbsIOp(flat_loop.step)
            rewriter.insert_op(abs_step, InsertPoint.before(flat_loop))
            #flat_loop.step.replace_by(abs_step.result)
            flat_loop.step = abs_step.result
        return flat_loop


    @staticmethod
    def remove_omp_parallel(parallel_op: omp.ParallelOp, rewriter: PatternRewriter):
        ws_loop = None
        for op in parallel_op.walk():
            if isinstance(op, omp.WsLoopOp):
                ws_loop = op
                break

        assert ws_loop
        print(ws_loop.body.block)
        omp_loop_op = ws_loop.body.block.first_op
        ws_loop_block = ws_loop.body.block
        ws_loop.body.detach_block(ws_loop_block)
        rewriter.inline_block(ws_loop_block, InsertPoint.before(ws_loop))
        rewriter.erase_op(ws_loop)

        if isinstance(omp_loop_op, omp.LoopNestOp):
            flat_loop = RemoveOps.transform_omp_loop_nest_into_scf_for(omp_loop_op, rewriter)
            one = arith.ConstantOp.from_int_and_width(1, 32)
            pragma_pipeline = PragmaPipelineOp(one)
            rewriter.insert_op([one, pragma_pipeline], InsertPoint.at_start(flat_loop.body.block))

        parallel_block = parallel_op.region.block
        parallel_op.region.detach_block(parallel_block)
        rewriter.erase_op(parallel_block.last_op)
        rewriter.inline_block(parallel_block, InsertPoint.before(parallel_op))
        rewriter.erase_op(parallel_op)

    @staticmethod
    def remove_omp_simd(simd_op : omp.SimdOp, rewriter : PatternRewriter):
        for priv_var in simd_op.private_vars:
            arg_idx = simd_op.operands.index(priv_var)
            simd_op.body.block.args[arg_idx].replace_by(priv_var)

        omp_loop_op = simd_op.body.block.first_op

        if isinstance(omp_loop_op, omp.LoopNestOp):
            flat_loop = RemoveOps.transform_omp_loop_nest_into_scf_for(omp_loop_op, rewriter)
            simd_factor = simd_op.simdlen.value.data
            ssa_simd_factor = arith.ConstantOp.from_int_and_width(simd_factor, 32)
            pragma_unroll = PragmaUnrollOp(ssa_simd_factor)
            rewriter.insert_op([ssa_simd_factor, pragma_unroll], InsertPoint.at_start(flat_loop.body.block))
        else:
            flat_loop = simd_op.body.block.first_op

        assert isinstance(flat_loop, scf.ForOp)
        flat_loop.detach()
        rewriter.insert_op(flat_loop, InsertPoint.before(simd_op))
        #rewriter.erase_matched_op() #FIXME: this does not work
        rewriter.erase_op(simd_op)


    @staticmethod
    def remove_remaining_omp_ops(target_func: func.FuncOp, rewriter: PatternRewriter):
        """Remove any remaining OpenMP operations in the target function."""
        for op in target_func.walk():
            if isinstance(op, omp.MapInfoOp):
                rewriter.erase_op(op)

        for op in target_func.walk():
            if isinstance(op, omp.MapBoundsOp):
                rewriter.erase_op(op)

    @staticmethod
    def forward_map_info(map_info: omp.MapInfoOp, rewriter: PatternRewriter):
        map_info.omp_ptr.replace_by(map_info.var_ptr)


@dataclass
class TargetFuncToHLS(RewritePattern):
    target_funcs : list[func.FuncOp] = field(default_factory=list)

    @op_type_rewrite_pattern
    def match_and_rewrite(self, module: builtin.ModuleOp, rewriter: PatternRewriter, /):
        if "target" not in module.attributes:
            return

        target_name = module.attributes["target"].data
        target_func = [op for op in module.walk() if isinstance(op, func.FuncOp) and op.sym_name.data == target_name][0]
        self.target_funcs.append(target_func)

        for map_info in target_func.walk():
            if not isinstance(map_info, omp.MapInfoOp):
                continue

            RemoveOps.forward_map_info(map_info, rewriter)

        omp_parallel_loops = []
        for op in target_func.walk():
            if isinstance(op, omp.ParallelOp):
                omp_parallel_loops.append(op)

        for omp_parallel in omp_parallel_loops:
            RemoveOps.remove_omp_parallel(omp_parallel, rewriter)

        omp_simd = None
        for op in target_func.walk():
            if isinstance(op, omp.SimdOp):
                omp_simd = op
                break

        if omp_simd:
            RemoveOps.remove_omp_simd(omp_simd, rewriter)

        RemoveOps.remove_remaining_omp_ops(target_func, rewriter)

@dataclass
class RemoveMemorySpaces(RewritePattern):
    """
    This pattern removes the memory space from memref types in the target function.
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, func_op: func.FuncOp, rewriter: PatternRewriter, /):
        if func_op.sym_visibility:
            return

        for arg in func_op.body.block.args:
            if isa(arg.type, builtin.MemRefType):
                new_type = builtin.MemRefType(arg.type.element_type, arg.type.shape)
                func_op.replace_argument_type(arg, new_type, rewriter)

        for op in func_op.walk():
            if isinstance(op, memref.AllocaOp):
                #new_type = builtin.MemRefType(op.memref.type.element_type, op.memref.type.shape)
                rewriter.replace_op(op, memref.AllocaOp.get(op.memref.type.element_type, shape=op.memref.type.shape))


@dataclass(frozen=True)
class TargetToHLSPass(ModulePass):
  """
  This is the entry point for the transformation pass which will then apply the rewriter
  """
  name = 'target-to-hls'

  generate : str = "hls"

  def apply(self, ctx: Context, module: builtin.ModuleOp):
    target_funcs : list[func.FuncOp] = []
    if self.generate == "device":
        walker = PatternRewriteWalker(GreedyRewritePatternApplier([
            TargetFuncToHLS(target_funcs),
            RemoveMemorySpaces(),
        ]), apply_recursively=False, walk_reverse=True)
    else:
        walker = PatternRewriteWalker(GreedyRewritePatternApplier([
            TargetFuncToHLS(target_funcs),
        ]), apply_recursively=False, walk_reverse=True)

    walker.rewrite_module(module)

    if self.generate == "device":
        # Keep only the top level module to contain the function
        for target_func in target_funcs:
            target_func.detach()

        module_block = module.body.block
        module.body.detach_block(module.body.block)
        module_block.erase()
        module.body.add_block(Block(target_funcs))
        module.attributes = {}
    elif self.generate == "host":
        host_module = None
        for op in module.body.walk():
            if isinstance(op, builtin.ModuleOp) and "target" not in op.attributes:
                host_module = op
                break

        module_block = module.body.block
        host_module_block = host_module.body.block
        host_module.body.detach_block(host_module.body.block)
        module.body.detach_block(module_block)
        module_block.erase()
        module.body.add_block(host_module_block)
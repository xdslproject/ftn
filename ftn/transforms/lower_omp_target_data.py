from xdsl.context import Context
from xdsl.utils.hints import isa
from xdsl.ir import BlockArgument
from enum import Enum
from xdsl.builder import Builder
from xdsl.rewriter import BlockInsertPoint

from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter,
    op_type_rewrite_pattern,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
)
from xdsl.passes import ModulePass

from ftn.dialects import device
from xdsl.dialects import builtin, arith, memref, scf, omp


class DataEnvironmentDirection(Enum):
    ENTER = 1
    EXIT = 2
    BOTH = 3


class DataMovementGenerator:
    def generate_for_mapped_vars(mapped_vars, rewriter, data_env_direction):
        """
        Generates operations for handling a list of mapped variables
        """
        preamble_ops = []
        postamble_ops = []

        wait_to_ssas_list = []
        wait_from_ssas_list = []
        alloc_ssas = []
        for input_ssa in mapped_vars:
            corresponding_mapinfo = input_ssa.owner

            alloc_ssa, wait_to_ssas, wait_from_ssas, alloc_ops, top_ops, bottom_ops = (
                DataMovementGenerator.generate_for_mapped_var(
                    corresponding_mapinfo, data_env_direction
                )
            )

            alloc_ssas.append(alloc_ssa)
            rewriter.replace_op(corresponding_mapinfo, alloc_ops, [alloc_ssa])

            preamble_ops += top_ops
            postamble_ops += bottom_ops

            if wait_to_ssas is not None:
                wait_to_ssas_list.append(wait_to_ssas)
            if wait_from_ssas is not None:
                wait_from_ssas_list.append(wait_from_ssas)

        # Generate DMA waits as required for data movement to and from
        if len(wait_to_ssas_list) > 0:
            preamble_ops += DataMovementGenerator.generate_dma_waits_for_tags(
                wait_to_ssas_list
            )

        if len(wait_from_ssas_list) > 0:
            postamble_ops += DataMovementGenerator.generate_dma_waits_for_tags(
                wait_from_ssas_list
            )

        return alloc_ssas, preamble_ops, postamble_ops

    def generate_for_mapped_var(mapinfo_op, data_env_direction):
        """
        Generates the operations for a mapped variable based upon some direction, the
        direction is needed to understand whether this is entering a data environment,
        exiting an environment, or both (this both case happens when there is a region
        such as in target data).
        """
        top_ops = []
        bottom_ops = []
        wait_to_ssas = None
        wait_from_ssas = None
        assert isa(mapinfo_op, omp.MapInfoOp)
        map_info_type = omp.OpenMPOffloadMappingFlags(mapinfo_op.map_type.value.data)

        var_type = mapinfo_op.var_type
        if not isa(var_type, builtin.MemRefType):
            # If this is a scalar then package as a memref
            var_type = builtin.MemRefType(var_type, [])

        alloc_memref_ssa, alloc_ops = DataMovementGenerator.generate_allocate_or_lookup(
            var_type, mapinfo_op.var_name, 1
        )

        if (
            data_env_direction == DataEnvironmentDirection.ENTER
            or data_env_direction == DataEnvironmentDirection.BOTH
        ):
            # If this is the entry to a data environment then handle the to
            if omp.OpenMPOffloadMappingFlags.TO in map_info_type:
                if omp.OpenMPOffloadMappingFlags.IMPLICIT in map_info_type:
                    # Implicit mapping it will defer to an outer environment so
                    # may or may not be needed to copy to the device
                    wait_to_ssas, to_ops_list = (
                        DataMovementGenerator.generate_conditional_copy_to_device(
                            mapinfo_op.var_name, 1, mapinfo_op.var_ptr, alloc_memref_ssa
                        )
                    )
                else:
                    wait_to_ssas, to_ops_list = (
                        DataMovementGenerator.generate_copy_to_device(
                            mapinfo_op.var_name, 1, mapinfo_op.var_ptr, alloc_memref_ssa
                        )
                    )
                top_ops += to_ops_list

            # Lastly, mark this variable as acquired by the current data environment
            top_ops.append(device.DataAcquire(mapinfo_op.var_name, 1))

        if (
            data_env_direction == DataEnvironmentDirection.EXIT
            or data_env_direction == DataEnvironmentDirection.BOTH
        ):
            # If this is the exit a data environment then handle the to

            # First, release this variable from the current data environment
            bottom_ops.append(device.DataRelease(mapinfo_op.var_name, 1))

            if omp.OpenMPOffloadMappingFlags.FROM in map_info_type:
                if omp.OpenMPOffloadMappingFlags.IMPLICIT in map_info_type:
                    # Implicit mapping it will defer to an outer environment so
                    # may or may not be needed to copy from the device
                    wait_from_ssas, from_ops_list = (
                        DataMovementGenerator.generate_conditional_copy_from_device(
                            mapinfo_op.var_name, 1, var_type, mapinfo_op.var_ptr
                        )
                    )
                else:
                    wait_from_ssas, from_ops_list = (
                        DataMovementGenerator.generate_copy_from_device(
                            mapinfo_op.var_name, 1, var_type, mapinfo_op.var_ptr
                        )
                    )
                bottom_ops += from_ops_list

        return (
            alloc_memref_ssa,
            wait_to_ssas,
            wait_from_ssas,
            alloc_ops,
            top_ops,
            bottom_ops,
        )

    def generate_allocate_or_lookup(var_type, var_name, memory_space):
        """
        Generates allocation of a variable on the device, or variable look up,
        depending on whether it exists. If it exists then do a look up,
        if not then an allocation
        """

        @Builder.implicit_region([])
        def true_region(args: tuple[BlockArgument, ...]) -> None:
            res = device.LookUpOp(var_name, memory_space, var_type)
            scf.YieldOp(res)

        @Builder.implicit_region([])
        def false_region(args: tuple[BlockArgument, ...]) -> None:
            res = DataMovementGenerator.generate_allocate_on_device(
                var_type, var_name, memory_space
            )
            scf.YieldOp(res)

        return DataMovementGenerator.generate_conditional_on_data_exists(
            var_name,
            memory_space,
            [
                builtin.MemRefType(
                    var_type.element_type,
                    var_type.shape,
                    memory_space=builtin.IntegerAttr.from_int_and_width(
                        memory_space, builtin.i32
                    ),
                )
            ],
            true_region,
            false_region,
        )

    def generate_conditional_on_data_exists(
        var_name,
        memory_space,
        conditional_return_type,
        true_region,
        false_region,
        is_not_conditional=False,
    ):
        """
        Generates a conditional check with a true and false region based upon whether
        a variable exists on the device or not.
        """
        data_exists_op = device.DataCheckExists(var_name, memory_space)

        ops = [data_exists_op]
        if is_not_conditional:
            const_op = arith.ConstantOp.from_int_and_width(1, 1)
            ex_io_op = arith.XOrIOp(const_op, data_exists_op, builtin.i1)
            ops += [const_op, ex_io_op]
            condition_ssa = ex_io_op
        else:
            condition_ssa = data_exists_op

        cond = scf.IfOp(
            condition_ssa, conditional_return_type, true_region, false_region
        )
        ops.append(cond)

        if len(conditional_return_type) == 1:
            return cond.results[0], ops
        elif len(conditional_return_type) == 2:
            return (cond.results[0], cond.results[1]), ops
        else:
            assert False

    def generate_allocate_on_device(var_type, var_name, memory_space):
        """
        Generates allocation of data on the device.
        """
        if isa(var_type, builtin.MemRefType):
            # We have a memref here, for now assume all static
            return device.AllocOp(
                var_name,
                var_type.element_type,
                builtin.IntegerAttr.from_int_and_width(memory_space, builtin.i32),
                var_type.shape,
            )
        else:
            # Element, therefore need to construct memref from bounds
            return None

    def generate_dma_data_copy(var_name, memory_space, src, dest):
        """
        Generates the DMA data copy between two memrefs each in different memory
        spaces. It returns a pair, the tag and number of elements SSAs as a tuple
        (needed for the DMA wait) and the operations as a list
        """
        tag_memref = memref.AllocaOp.get(builtin.i32, shape=[])
        number_elements = device.DataNumElements(var_name, memory_space)
        zero_idx = arith.ConstantOp(builtin.IntegerAttr.from_index_int_value(0))
        dma_start_copy_op = memref.DmaStartOp.get(
            src,
            [zero_idx],
            dest,
            [zero_idx],
            number_elements,
            tag_memref.results[0],
            [],
        )
        return (tag_memref.results[0], number_elements.results[0]), [
            tag_memref,
            number_elements,
            zero_idx,
            dma_start_copy_op,
        ]

    def generate_conditional_copy_to_device(var_name, memory_space, src, dest):
        """
        Conditional copy to device based on whether the data exists, if so
        it will issue the copy and otherwise not. This is called when we have an
        implicit variable, which is overridden if an outer data environment
        contains this variable.
        """

        @Builder.implicit_region([])
        def true_region(args: tuple[BlockArgument, ...]) -> None:
            res, ops = DataMovementGenerator.generate_copy_to_device(
                var_name, memory_space, src, dest
            )
            scf.YieldOp(*res)

        return DataMovementGenerator.generate_conditional_on_data_exists(
            var_name,
            memory_space,
            [builtin.MemRefType(builtin.i32, []), builtin.IndexType()],
            true_region,
            None,
            True,
        )

    def generate_copy_to_device(var_name, memory_space, src, dest):
        """
        Generate copy to device operations
        """
        return DataMovementGenerator.generate_dma_data_copy(
            var_name, memory_space, src, dest
        )

    def generate_conditional_copy_from_device(var_name, memory_space, var_type, dest):
        """
        Conditional copy from device based on whether the data exists, if so
        it will issue the copy and otherwise not. This is called when we have an
        implicit variable, which is overridden if an outer data environment
        contains this variable.
        """

        @Builder.implicit_region([])
        def true_region(args: tuple[BlockArgument, ...]) -> None:
            res, ops = DataMovementGenerator.generate_copy_from_device(
                var_name, memory_space, var_type, dest
            )
            scf.YieldOp(*res)

        return DataMovementGenerator.generate_conditional_on_data_exists(
            var_name,
            memory_space,
            [builtin.MemRefType(builtin.i32, []), builtin.IndexType()],
            true_region,
            None,
            True,
        )

    def generate_copy_from_device(var_name, memory_space, var_type, dest):
        """
        Generates the copy from device by looking up memory to get a handle
        to it, and then issuing the copy generation.
        """
        device_memref = device.LookUpOp(
            var_name,
            builtin.IntegerAttr.from_int_and_width(memory_space, builtin.i32),
            var_type,
        )
        tag_ssa, ops_list = DataMovementGenerator.generate_dma_data_copy(
            var_name, memory_space, device_memref, dest
        )
        return tag_ssa, [device_memref] + ops_list

    def generate_dma_waits_for_tags(wait_ssas_list):
        """
        Generates the DMA wait operations based upon the provided wait list, each
        entry in the wait list is a tuple (wait tag, number elements).
        """
        ops_list = []
        for tag, num_els in wait_ssas_list:
            wait_op = memref.DmaWaitOp.get(tag, [], num_els)
            ops_list.append(wait_op)
        return ops_list


class LowerTargetEnterData(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: omp.TargetEnterDataOp, rewriter: PatternRewriter):
        alloc_ssas, preamble_ops, postamble_ops = (
            DataMovementGenerator.generate_for_mapped_vars(
                op.mapped_vars, rewriter, DataEnvironmentDirection.ENTER
            )
        )
        rewriter.replace_matched_op(preamble_ops + postamble_ops, [])


class LowerTargetExitData(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: omp.TargetExitDataOp, rewriter: PatternRewriter):
        alloc_ssas, preamble_ops, postamble_ops = (
            DataMovementGenerator.generate_for_mapped_vars(
                op.mapped_vars, rewriter, DataEnvironmentDirection.EXIT
            )
        )
        rewriter.replace_matched_op(preamble_ops + postamble_ops, [])


class LowerTargetOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: omp.TargetOp, rewriter: PatternRewriter):
        alloc_ssas, preamble_ops, postamble_ops = (
            DataMovementGenerator.generate_for_mapped_vars(
                op.map_vars, rewriter, DataEnvironmentDirection.BOTH
            )
        )

        rewriter.insert_op_before_matched_op(preamble_ops)
        rewriter.insert_op_after_matched_op(postamble_ops)

        # Now rebuild the target operation with the memrefs passed in as
        # has_device_addr_vars , we no longer have map vars
        target_op = omp.TargetOp.build(
            operands=[
                op.allocate_vars,
                op.allocator_vars,
                op.depend_vars,
                op.device,
                alloc_ssas,
                op.host_eval_vars,
                op.if_expr,
                op.in_reduction_vars,
                op.is_device_ptr_vars,
                [],
                op.private_vars,
                op.thread_limit,
            ],
            regions=[rewriter.move_region_contents_to_new_regions(op.region)],
            properties=op.properties.copy(),
        )
        rewriter.replace_matched_op(target_op)


class LowerTargetDataOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: omp.TargetDataOp, rewriter: PatternRewriter):
        alloc_ssas, preamble_ops, postamble_ops = (
            DataMovementGenerator.generate_for_mapped_vars(
                op.mapped_vars, rewriter, DataEnvironmentDirection.BOTH
            )
        )

        rewriter.insert_op_before_matched_op(preamble_ops)

        block_terminator = op.region.block.last_op
        assert isa(block_terminator, omp.TerminatorOp)
        rewriter.erase_op(block_terminator)

        rewriter.inline_block_before_matched_op(op.region.block)
        rewriter.insert_op_after_matched_op(postamble_ops)

        rewriter.erase_op(op)


class LowerOmpTargetDataPass(ModulePass):
    name = "lower-omp-target-data"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerTargetEnterData(),
                    LowerTargetExitData(),
                    LowerTargetDataOp(),
                    LowerTargetOp(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)

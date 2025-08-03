from dataclasses import dataclass
from enum import Enum

from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import arith, builtin, memref, omp, scf
from xdsl.ir import BlockArgument
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa

from ftn.dialects import device


class DataEnvironmentDirection(Enum):
    ENTER = 1
    EXIT = 2
    BOTH = 3


class DataMovementGenerator:
    def collect_mapped_vars_by_stack_and_heap(mapped_vars):
        sorted_mapped_vars = []
        ignore_mapped_vars = []
        for mapped_var in mapped_vars:
            if mapped_var in ignore_mapped_vars:
                continue

            if mapped_var.owner.var_name is not None and mapped_var.owner.var_name.data:
                # This has a name, therefore the overarching descriptor
                if (
                    mapped_var.owner.members is not None
                    and len(mapped_var.owner.members) > 0
                ):
                    # Allocatable array as this is an op to the mapinfo
                    # For now we just handle one map_info as an argument
                    assert len(mapped_var.owner.members) == 1
                    sorted_mapped_vars.append(
                        (mapped_var.owner, mapped_var.owner.members[0].owner)
                    )
                    ignore_mapped_vars.append(mapped_var)
                else:
                    # On its own, so just add this alone (scalar or static array)
                    sorted_mapped_vars.append((mapped_var.owner, None))
        return sorted_mapped_vars

    def generate_for_mapped_vars(
        mapped_vars, rewriter, data_env_direction, device_mem_space
    ):
        """
        Generates operations for handling a list of mapped variables
        """
        preamble_ops = []
        postamble_ops = []

        wait_to_ssas_list = []
        wait_from_ssas_list = []
        alloc_ssas = []

        sorted_mapped_vars = (
            DataMovementGenerator.collect_mapped_vars_by_stack_and_heap(mapped_vars)
        )

        for input_mapped_var in sorted_mapped_vars:
            alloc_ssa, wait_to_ssas, wait_from_ssas, alloc_ops, top_ops, bottom_ops = (
                DataMovementGenerator.generate_for_mapped_var(
                    input_mapped_var, data_env_direction, device_mem_space
                )
            )

            alloc_ssas.append(alloc_ssa)
            rewriter.replace_op(input_mapped_var[0], alloc_ops, [alloc_ssa])

            if input_mapped_var[1] is not None:
                rewriter.erase_op(input_mapped_var[1], False)

            if input_mapped_var[0].bounds is not None:
                for bound in input_mapped_var[0].bounds:
                    rewriter.erase_op(bound.owner, False)

            if (
                input_mapped_var[1] is not None
                and input_mapped_var[1].bounds is not None
            ):
                for bound in input_mapped_var[1].bounds:
                    rewriter.erase_op(bound.owner, False)

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

    def gather_sizes_from_mapinfo_bounds(mapinfo_bounds_ops):
        size_ssas = []
        for bound in mapinfo_bounds_ops:
            assert bound.owner.extent is not None
            size_ssas.append(bound.owner.extent)
        size_ssas.reverse()
        return size_ssas

    def extract_mapinfo_description(map_infos):
        description = {}
        assert len(map_infos) == 2

        assert isa(map_infos[0], omp.MapInfoOp)
        if map_infos[1] is None:
            # Statically allocated
            description["map_type"] = map_infos[0].map_type
            description["bounds"] = map_infos[0].bounds
            description["var_type"] = map_infos[0].var_type
            description["var_name"] = map_infos[0].var_name
            description["var_ptr"] = map_infos[0].var_ptr
        else:
            # Heap allocated
            assert isa(map_infos[1], omp.MapInfoOp)
            description["map_type"] = map_infos[1].map_type
            description["bounds"] = map_infos[1].bounds
            description["var_type"] = map_infos[0].var_type
            description["var_name"] = map_infos[0].var_name
            description["var_ptr"] = map_infos[1].var_ptr
        return description

    def generate_for_mapped_var(
        mapinfos_description, data_env_direction, device_mem_space
    ):
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

        mapped_var_description = DataMovementGenerator.extract_mapinfo_description(
            mapinfos_description
        )

        map_info_type = omp.OpenMPOffloadMappingFlags(
            mapped_var_description["map_type"].value.data
        )

        if mapped_var_description["bounds"] is not None:
            size_ssas = DataMovementGenerator.gather_sizes_from_mapinfo_bounds(
                mapped_var_description["bounds"]
            )
        else:
            size_ssas = []

        var_type = mapped_var_description["var_type"]
        if not isa(var_type, builtin.MemRefType):
            # If this is a scalar then package as a memref
            var_type = builtin.MemRefType(var_type, [])

        alloc_memref_ssa, alloc_ops = DataMovementGenerator.generate_allocate_or_lookup(
            var_type, mapped_var_description["var_name"], device_mem_space, size_ssas
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
                            mapped_var_description["var_name"],
                            device_mem_space,
                            mapped_var_description["var_ptr"],
                            alloc_memref_ssa,
                        )
                    )
                else:
                    wait_to_ssas, to_ops_list = (
                        DataMovementGenerator.generate_copy_to_device(
                            mapped_var_description["var_name"],
                            device_mem_space,
                            mapped_var_description["var_ptr"],
                            alloc_memref_ssa,
                        )
                    )
                top_ops += to_ops_list

            # Lastly, mark this variable as acquired by the current data environment
            top_ops.append(
                device.DataAcquire(mapped_var_description["var_name"], device_mem_space)
            )

        if (
            data_env_direction == DataEnvironmentDirection.EXIT
            or data_env_direction == DataEnvironmentDirection.BOTH
        ):
            # If this is the exit a data environment then handle the to

            # First, release this variable from the current data environment
            bottom_ops.append(
                device.DataRelease(mapped_var_description["var_name"], device_mem_space)
            )

            if omp.OpenMPOffloadMappingFlags.FROM in map_info_type:
                if omp.OpenMPOffloadMappingFlags.IMPLICIT in map_info_type:
                    # Implicit mapping it will defer to an outer environment so
                    # may or may not be needed to copy from the device
                    wait_from_ssas, from_ops_list = (
                        DataMovementGenerator.generate_conditional_copy_from_device(
                            mapped_var_description["var_name"],
                            device_mem_space,
                            var_type,
                            mapped_var_description["var_ptr"],
                        )
                    )
                else:
                    wait_from_ssas, from_ops_list = (
                        DataMovementGenerator.generate_copy_from_device(
                            mapped_var_description["var_name"],
                            device_mem_space,
                            var_type,
                            mapped_var_description["var_ptr"],
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

    def generate_allocate_or_lookup(var_type, var_name, memory_space, size_ssas):
        """
        Generates allocation of a variable on the device, or variable look up,
        depending on whether it exists. If it exists then do a look up,
        if not then an allocation
        """

        @Builder.implicit_region([])
        def true_region(args: tuple[BlockArgument, ...]) -> None:
            res = device.LookUpOp(var_name, memory_space, var_type)
            scf.YieldOp(res.results[0])

        @Builder.implicit_region([])
        def false_region(args: tuple[BlockArgument, ...]) -> None:
            res = DataMovementGenerator.generate_allocate_on_device(
                var_type, var_name, memory_space, size_ssas
            )
            scf.YieldOp(res.results[0])

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
            raise Exception(
                "Too many return types for conditional return type, only 1 or 2 allowed"
            )

    def generate_allocate_on_device(var_type, var_name, memory_space, size_ssas):
        """
        Generates allocation of data on the device.
        """
        dynamic_ssas = []
        for idx, shape in enumerate(var_type.shape):
            if shape.data == -1:
                dynamic_ssas.append(size_ssas[idx])

        return device.AllocOp(
            var_name,
            var_type.element_type,
            builtin.IntegerAttr.from_int_and_width(memory_space, builtin.i32),
            var_type.shape,
            dynamic_ssas,
        )

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
            [zero_idx] * len(src.type.shape),
            dest,
            [zero_idx] * len(src.type.shape),
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
            var_name, memory_space, device_memref.results[0], dest
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


@dataclass(frozen=True)
class LowerTargetEnterData(RewritePattern):
    device_mem_space: int | None = None

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: omp.TargetEnterDataOp, rewriter: PatternRewriter):
        alloc_ssas, preamble_ops, postamble_ops = (
            DataMovementGenerator.generate_for_mapped_vars(
                op.mapped_vars,
                rewriter,
                DataEnvironmentDirection.ENTER,
                self.device_mem_space,
            )
        )
        assert self.device_mem_space is not None

        rewriter.replace_matched_op(preamble_ops + postamble_ops, [])


@dataclass(frozen=True)
class LowerTargetExitData(RewritePattern):
    device_mem_space: int | None = None

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: omp.TargetExitDataOp, rewriter: PatternRewriter):
        alloc_ssas, preamble_ops, postamble_ops = (
            DataMovementGenerator.generate_for_mapped_vars(
                op.mapped_vars,
                rewriter,
                DataEnvironmentDirection.EXIT,
                self.device_mem_space,
            )
        )
        assert self.device_mem_space is not None

        rewriter.replace_matched_op(preamble_ops + postamble_ops, [])


@dataclass(frozen=True)
class LowerTargetOp(RewritePattern):
    device_mem_space: int | None = None

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: omp.TargetOp, rewriter: PatternRewriter):
        alloc_ssas, preamble_ops, postamble_ops = (
            DataMovementGenerator.generate_for_mapped_vars(
                op.map_vars,
                rewriter,
                DataEnvironmentDirection.BOTH,
                self.device_mem_space,
            )
        )

        assert self.device_mem_space is not None

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


@dataclass(frozen=True)
class LowerTargetDataOp(RewritePattern):
    device_mem_space: int | None = None

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: omp.TargetDataOp, rewriter: PatternRewriter):
        alloc_ssas, preamble_ops, postamble_ops = (
            DataMovementGenerator.generate_for_mapped_vars(
                op.mapped_vars,
                rewriter,
                DataEnvironmentDirection.BOTH,
                self.device_mem_space,
            )
        )

        assert self.device_mem_space is not None

        rewriter.insert_op_before_matched_op(preamble_ops)

        block_terminator = op.region.block.last_op
        assert isa(block_terminator, omp.TerminatorOp)
        rewriter.erase_op(block_terminator)

        rewriter.inline_block_before_matched_op(op.region.block)
        rewriter.insert_op_after_matched_op(postamble_ops)

        rewriter.erase_op(op)


@dataclass(frozen=True)
class LowerOmpTargetDataPass(ModulePass):
    name = "lower-omp-target-data"

    memory_order: str = "HBM,DDR"

    def get_device_mem_space_name(self, accel_config):
        for mem in self.memory_order.split(","):
            mem_type = device.MemoryKindAttr(device.MemoryKind[mem])
            for mem_config in accel_config["memory"]:
                if mem_config.value["kind"] == mem_type:
                    return mem_config.key.data
        return None

    def get_mem_space_from_target_system_spec(self, target, configuration):
        accel_config = configuration[target]

        memspace_name = self.get_device_mem_space_name(accel_config)
        assert memspace_name is not None

        for el in configuration["memory_spaces"]:
            if el.value.data == memspace_name:
                return int(el.key.data)
        return None

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        assert "omp.target_triples" in op.attributes
        assert "dlti.target_system_spec" in op.attributes

        assert len(op.attributes["omp.target_triples"]) == 1
        target = op.attributes["omp.target_triples"].data[0]

        device_mem_space = self.get_mem_space_from_target_system_spec(
            target.data, op.attributes["dlti.target_system_spec"]
        )
        assert device_mem_space is not None

        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerTargetEnterData(device_mem_space=device_mem_space),
                    LowerTargetExitData(device_mem_space=device_mem_space),
                    LowerTargetDataOp(device_mem_space=device_mem_space),
                    LowerTargetOp(device_mem_space=device_mem_space),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)

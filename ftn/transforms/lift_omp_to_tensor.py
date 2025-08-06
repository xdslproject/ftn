from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum

from xdsl.context import Context
from xdsl.dialects import arith, bufferization, builtin, memref, omp, tensor, tosa
from xdsl.ir import Block, Region, SSAValue
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
from ftn.util.visitor import Visitor


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
            self.arith_operation = arith_operation
            self.lhs = lhs
            self.lhs_type = lhs_t
            self.rhs = rhs
            self.rhs_type = rhs_t

    def get_var_ssa(self):
        # Gets variable ssa that have been referenced in the expression
        var_ssas = []
        GetArrayAccessValue.search_var_names(
            var_ssas, self.expression, self.expression_t
        )
        return var_ssas

    def search_var_names(var_ssas, expr, expr_t):
        # Recursive procedure to search all nested expressions to locate variable names
        if expr_t == GetArrayAccessValue.SideType.VARIABLE:
            var_ssas.append(expr)
        elif expr_t == GetArrayAccessValue.SideType.EXPRESSION:
            GetArrayAccessValue.search_var_names(var_ssas, expr.lhs, expr.lhs_type)
            GetArrayAccessValue.search_var_names(var_ssas, expr.rhs, expr.rhs_type)

    def traverse_constant_op(self, constant_op: arith.ConstantOp):
        # Grabs out constant value
        assert (
            constant_op.value.type == builtin.i32
            or constant_op.value.type == builtin.i64
            or isa(constant_op.value.type, builtin.IndexType)
        )
        self.expression = constant_op.value.value.data
        self.expression_t = GetArrayAccessValue.SideType.CONSTANT

    def handle_arith_op(self, arith_op, arith_op_type):
        self.traverse(arith_op.lhs.owner)
        lhs_v = self.expression
        lhs_t = self.expression_t
        self.traverse(arith_op.rhs.owner)
        e = GetArrayAccessValue.ExpressionDescription(
            lhs_v, lhs_t, self.expression, self.expression_t, arith_op_type
        )
        self.expression = e
        self.expression_t = GetArrayAccessValue.SideType.EXPRESSION

    def traverse_subi_op(self, subi_op: arith.SubiOp):
        self.handle_arith_op(subi_op, GetArrayAccessValue.ArithOp.SUB)

    def traverse_addi_op(self, addi_op: arith.AddiOp):
        self.handle_arith_op(addi_op, GetArrayAccessValue.ArithOp.ADD)

    def traverse_muli_op(self, addi_op: arith.MuliOp):
        self.handle_arith_op(addi_op, GetArrayAccessValue.ArithOp.MUL)

    def traverse_ext_u_i_op(self, extui_op: arith.ExtUIOp):
        self.traverse(extui_op.input.owner)

    def traverse_ext_s_i_op(self, extsi_op: arith.ExtSIOp):
        self.traverse(extsi_op.input.owner)

    def traverse_index_cast_op(self, convert_op: arith.IndexCastOp):
        # We ignore converts apart from visiting the child
        self.traverse(convert_op.input.owner)

    def traverse_load_op(self, load_op: memref.LoadOp):
        # Signifies a variable
        self.expression = load_op.memref
        self.expression_t = GetArrayAccessValue.SideType.VARIABLE


class GetStoreCalculationContributedOperations(Visitor):
    # For a calculation this will gather up all the operations that contribute
    # to it, for instance the arithmetic operations, LHS and RHS variables
    # and constants, conversions etc

    class ContributedOperation(ABC):
        # Base class that operations (forming the dependency tree) implement
        def walk(self, to_match):
            if isa(self, to_match):
                return [self]
            return []

        def is_constant(self, ssa):
            return not isa(ssa.type, builtin.TensorType)

        def splat_constant_to_tensor(self, ssa, shape):
            return tensor.SplatOp(ssa, [], builtin.TensorType(ssa.type, shape))

        @abstractmethod
        def generate(self):
            raise NotImplementedError()

    class ArithmeticOperation(ContributedOperation):
        class ArithOpTypes(Enum):
            SUB = 1
            ADD = 2
            MUL = 3
            DIV = 4

        def __init__(self, lhs, rhs, arith_type, op_data_type):
            self.lhs = lhs
            self.rhs = rhs
            self.arith_type = arith_type
            self.op_data_type = op_data_type

        def walk(self, to_match):
            matched_ops = super().walk(to_match)
            matched_ops += self.lhs.walk(to_match)
            matched_ops += self.rhs.walk(to_match)
            return matched_ops

        def generate(self):
            lhs_ops, lhs_ssa = self.lhs.generate()
            rhs_ops, rhs_ssa = self.rhs.generate()

            if self.is_constant(lhs_ssa):
                assert not self.is_constant(rhs_ssa)
                splat = self.splat_constant_to_tensor(lhs_ssa, rhs_ssa.type.get_shape())
                lhs_ops.append(splat)
                lhs_ssa = splat.results[0]

            if self.is_constant(rhs_ssa):
                assert not self.is_constant(lhs_ssa)
                splat = self.splat_constant_to_tensor(rhs_ssa, lhs_ssa.type.get_shape())
                rhs_ops.append(splat)
                rhs_ssa = splat.results[0]

            if (
                self.arith_type
                == GetStoreCalculationContributedOperations.ArithmeticOperation.ArithOpTypes.SUB
            ):
                tensor_arith_op = tosa.SubOp.build(
                    operands=[lhs_ssa, rhs_ssa], result_types=[lhs_ssa.type]
                )
            elif (
                self.arith_type
                == GetStoreCalculationContributedOperations.ArithmeticOperation.ArithOpTypes.ADD
            ):
                tensor_arith_op = tosa.AddOp.build(
                    operands=[lhs_ssa, rhs_ssa], result_types=[lhs_ssa.type]
                )
            elif (
                self.arith_type
                == GetStoreCalculationContributedOperations.ArithmeticOperation.ArithOpTypes.MUL
            ):
                tensor_arith_op = tosa.MulOp.build(
                    operands=[lhs_ssa, rhs_ssa], result_types=[lhs_ssa.type]
                )
            elif (
                self.arith_type
                == GetStoreCalculationContributedOperations.ArithmeticOperation.ArithOpTypes.DIV
            ):
                # No divide provided
                raise Exception("Divide not supported")
            else:
                raise Exception("Unknown operation in tensorisation")

            return lhs_ops + rhs_ops + [tensor_arith_op], tensor_arith_op.results[0]

    class ConstantOperation(ContributedOperation):
        def __init__(self, value, op_data_type):
            self.value = value
            self.op_data_type = op_data_type

        def generate(self):
            constant_op = arith.ConstantOp(self.value, self.op_data_type)
            return [constant_op], constant_op.results[0]

    class VariableLoadOperation(ContributedOperation, ABC):
        # Variable loads can be intermediate or device mapped
        # this is the base class for these
        def __init__(self, var_ssa, var_type):
            self.var_ssa = var_ssa
            self.var_type = var_type

    class IntermediateVariableLoadOperation(VariableLoadOperation):
        # An intermediate variable, the dependency tree corresponding
        # to the memref that is being loaded is stored and used here
        def __init__(self, var_ssa, var_type, load_memref_dependency_tree):
            self.load_memref_dependency_tree = load_memref_dependency_tree
            super().__init__(var_ssa, var_type)

        def walk(self, to_match):
            matched_ops = super().walk(to_match)
            matched_ops += self.load_memref_dependency_tree.walk(to_match)

            return matched_ops

        def generate(self):
            ops, ssa = self.load_memref_dependency_tree.generate()
            return ops, ssa

    class MappedVariableLoadOperation(VariableLoadOperation):
        # A device mapped variable being loaded, the tensor_ssa is
        # set here, which is the tensor block argument that is loaded
        def __init__(self, var_ssa, var_type):
            self.tensor_ssa = None
            super().__init__(var_ssa, var_type)

        def set_tensor(self, tensor_ssa):
            self.tensor_ssa = tensor_ssa

        def generate(self):
            assert self.tensor_ssa is not None
            return [], self.tensor_ssa

    class VariableStoreOperation(ContributedOperation):
        # A variable store, this will either be a store on an intermediate
        # or device mapped variable
        def __init__(self, source_op):
            self.source_op = source_op

        def walk(self, to_match):
            matched_ops = super().walk(to_match)
            matched_ops += self.source_op.walk(to_match)

            return matched_ops

        def generate(self):
            ops, ssa = self.source_op.generate()
            return ops, ssa

    def __init__(self, private_memrefs, root_store_op):
        self.private_memrefs = private_memrefs
        self.root_store_op = root_store_op
        self.tree_root = None

    def handle_binary_op(self, bin_op):
        lhs = self.traverse(bin_op.lhs.owner)
        rhs = self.traverse(bin_op.rhs.owner)
        # These are returned by the visitor, returns a list due to potential of multiple visits
        assert len(lhs) == 1
        assert len(rhs) == 1
        if isa(bin_op, arith.SubiOp) or isa(bin_op, arith.SubfOp):
            arith_type = GetStoreCalculationContributedOperations.ArithmeticOperation.ArithOpTypes.SUB
        elif isa(bin_op, arith.MuliOp) or isa(bin_op, arith.MulfOp):
            arith_type = GetStoreCalculationContributedOperations.ArithmeticOperation.ArithOpTypes.MUL
        elif isa(bin_op, arith.AddiOp) or isa(bin_op, arith.AddfOp):
            arith_type = GetStoreCalculationContributedOperations.ArithmeticOperation.ArithOpTypes.ADD
        elif (
            isa(bin_op, arith.DivUIOp)
            or isa(bin_op, arith.DivSIOp)
            or isa(bin_op, arith.DivfOp)
        ):
            arith_type = GetStoreCalculationContributedOperations.ArithmeticOperation.ArithOpTypes.DIV
        else:
            raise Exception("Unrecognised binary operation, can not handle this")
        arith_data_type = bin_op.results[0].type
        return GetStoreCalculationContributedOperations.ArithmeticOperation(
            lhs[0], rhs[0], arith_type, arith_data_type
        )

    def traverse_subi_op(self, subi_op: arith.SubiOp):
        return self.handle_binary_op(subi_op)

    def traverse_muli_op(self, muli_op: arith.MuliOp):
        return self.handle_binary_op(muli_op)

    def traverse_addi_op(self, addi_op: arith.AddiOp):
        return self.handle_binary_op(addi_op)

    def traverse_addf_op(self, addf_op: arith.AddfOp):
        return self.handle_binary_op(addf_op)

    def traverse_mulf_op(self, mulf_op: arith.MulfOp):
        return self.handle_binary_op(mulf_op)

    def traverse_subf_op(self, subf_op: arith.SubfOp):
        return self.handle_binary_op(subf_op)

    def traverse_divf_op(self, divf_op: arith.DivfOp):
        return self.handle_binary_op(divf_op)

    def traverse_div_u_i_op(self, divui_op: arith.DivUIOp):
        return self.handle_binary_op(divui_op)

    def traverse_div_s_i_op(self, divsi_op: arith.DivSIOp):
        return self.handle_binary_op(divsi_op)

    def find_memref_store_on_memref(op, memref_ssa):
        # This works backwards from the operation, inspecting each memref.store
        # and checking whether it is a store on the provided memref SSA. It will
        # therefore return the closes previous store on this memref, or
        # none if there is none found
        while op is not None:
            if isa(op.prev_op, memref.StoreOp) and op.prev_op.memref == memref_ssa:
                return op.prev_op
            else:
                op = op.prev_op
        return None

    def traverse_load_op(self, load_op: memref.LoadOp):
        # This load can either be device mapped or an intermediate, we have the private
        # memrefs, so check whether the memref being loaded is in this list
        if load_op.memref in self.private_memrefs:
            # If it is a private memref then we need to find the closest store on that memref,
            # as the value being stored is the dependency tree that we want to use here (basically
            # the store and load are removed, and the value being stored is used as the result
            # of the load).
            store_op = (
                GetStoreCalculationContributedOperations.find_memref_store_on_memref(
                    load_op, load_op.memref
                )
            )
            assert store_op is not None
            ret_vals = self.traverse(store_op)
            assert len(ret_vals) == 1
            return GetStoreCalculationContributedOperations.IntermediateVariableLoadOperation(
                load_op.memref, load_op.results[0].type, ret_vals[0]
            )
        else:
            # A device mapped variable
            return GetStoreCalculationContributedOperations.MappedVariableLoadOperation(
                load_op.memref, load_op.results[0].type
            )

    def traverse_constant_op(self, constant_op: arith.ConstantOp):
        return GetStoreCalculationContributedOperations.ConstantOperation(
            constant_op.value, constant_op.results[0].type
        )

    def traverse_store_op(self, store_op: memref.StoreOp):
        ret_vals = self.traverse(store_op.value.owner)
        assert len(ret_vals) == 1
        # There will be a single store op that is the root of the tree, which is
        # where we started from. Check if this is it
        if store_op == self.root_store_op:
            # This is the root, the result of this store is written to a device
            # mapped variable, therefore store this as the tree starting point
            self.tree_root = (
                GetStoreCalculationContributedOperations.VariableStoreOperation(
                    ret_vals[0]
                )
            )
        else:
            # An intermediate value
            return GetStoreCalculationContributedOperations.VariableStoreOperation(
                ret_vals[0]
            )


class BuildApplicableOpDependencyTrees(Visitor):
    # Builds up dependency trees of abstract operations based upon stores to device
    # mapped data (outputs)
    def __init__(self, private_memrefs, device_mapped_data):
        self.private_memrefs = private_memrefs
        self.device_mapped_data = device_mapped_data
        self.dependency_trees = []

    def get_all_input_var_ssas(self):
        # Retrieves a set (unique list) of mapped input variable SSAs
        input_var_ssas = []
        for dependency_tree in self.dependency_trees:
            input_vars = dependency_tree[1].walk(
                GetStoreCalculationContributedOperations.MappedVariableLoadOperation
            )
            input_var_ssas += [iv.var_ssa for iv in input_vars]
        return set(input_var_ssas)

    def apply_tensor_ssa_mapping_to_inputs(self, memref_to_tensor_map):
        # For each device mapped input variable this sets the SSA of the tensor
        # that corresponds to the origional memref. This tensor is a block argument
        for dependency_tree in self.dependency_trees:
            input_vars = dependency_tree[1].walk(
                GetStoreCalculationContributedOperations.MappedVariableLoadOperation
            )
            for input_var in input_vars:
                # Ensure that the memref has a corresponding tenssor in the map
                assert input_var.var_ssa in memref_to_tensor_map
                input_var.set_tensor(memref_to_tensor_map[input_var.var_ssa])

    def get_all_output_var_ssas(self):
        # Examines all the dependency trees and extracts the output
        # memref SSAs from all of these (this is the first value
        # in the pair).
        return [d[0].memref for d in self.dependency_trees]

    def traverse_store_op(self, store_op: memref.StoreOp):
        # First figure out if this store is writing to a memref which is
        # device mapped, only process this if so (intermediates are
        # handled elsewhere)
        tensorise = store_op.memref in self.device_mapped_data
        if tensorise:
            # Now determine the dependency tree of operations that we are going to convert
            # into tensorised operations
            contributed_ops = GetStoreCalculationContributedOperations(
                self.private_memrefs, store_op
            )
            # Now get all the operations that contribute to the RHS (the value stored)
            contributed_ops.traverse(store_op)
            # Store a pair, the memref store operation and the root of the dependency tree,
            # the memref store operation is useful as we can look at these to determine
            # all the outputs
            self.dependency_trees.append((store_op, contributed_ops.tree_root))


class LiftOMPToTensors(RewritePattern, ABC):
    def find_parent_op(cls, op):
        # From an operation will walk backwards through the IR
        # to find an operation of a specific type
        if isa(op, cls):
            return op
        elif op is None:
            return None
        else:
            return LiftSIMDOp.find_parent_op(cls, op.parent)

    def find_child_op(cls, search_op):
        # From an operation walks forward through it's regions and
        # blocks to find a child operation
        if isa(search_op, cls):
            return search_op
        else:
            for region in search_op.regions:
                for block in region.blocks:
                    for op in block.ops:
                        c = LiftOMPToTensors.find_child_op(cls, op)
                        if c is not None:
                            return c
            return None

    def get_constant(token):
        # If an operation is an i32 constant it will return this
        if isa(token, arith.ConstantOp) and token.value.type == builtin.i32:
            return token.value.value.data
        else:
            return None

    def lift_op(
        self,
        private_memrefs: Sequence[SSAValue],
        op: omp.SimdOp | omp.WsLoopOp,
        rewriter: PatternRewriter,
    ):
        # First find the OpenMP loopnest operation, this is where the transformation will focus on
        loop_nest_op = LiftOMPToTensors.find_child_op(omp.LoopNestOp, op)
        assert loop_nest_op is not None

        # Also locate the kernel create, this gives us the input and output memref SSAs
        kernel_create_op = LiftOMPToTensors.find_parent_op(device.KernelCreate, op)
        assert kernel_create_op is not None

        # Work through the loop counts provided to the loop nest operation to build
        # the bounds of the tensors that will be operated on here, this supports
        # multi-dimension loops by having multi-dimension tensors
        tensor_sizes = []
        for lower, upper, step in zip(
            loop_nest_op.lowerBound, loop_nest_op.upperBound, loop_nest_op.step
        ):
            lower_const = LiftOMPToTensors.get_constant(lower.owner)
            upper_const = LiftOMPToTensors.get_constant(upper.owner)
            step_const = LiftOMPToTensors.get_constant(step.owner)
            if (
                lower_const is not None
                and upper_const is not None
                and step_const is not None
            ):
                # If this specific loop's bounds are known, then set the size statically
                tensor_sizes.append(int((upper_const - (lower_const - 1)) / step_const))
            else:
                # Otherwise this dimension size is dynamic
                tensor_sizes.append(-1)

        # Create dependency tree walker, this is passed the private (intermediate)
        # memrefs, and SSA of the device mapped data
        dependence_tree_generator = BuildApplicableOpDependencyTrees(
            private_memrefs, kernel_create_op.body.block.args
        )
        dependence_tree_generator.traverse(loop_nest_op)

        # Get all input var SSAs from the dependency tree and then
        # use these to build the block of input tensor types
        input_vars = dependence_tree_generator.get_all_input_var_ssas()

        arg_types = []
        for input_var in input_vars:
            assert len(input_var.type.shape) == len(tensor_sizes)
            arg_types.append(
                builtin.TensorType(input_var.type.element_type, tensor_sizes)
            )

        new_block = Block(arg_types=arg_types)

        # Now we have the block arguments, set the specific block argument SSA
        # for each mapped input variable that corresponds to it
        memref_to_tensor = {}
        for tensor_arg, memref_arg in zip(new_block.args, input_vars):
            memref_to_tensor[memref_arg] = tensor_arg
        dependence_tree_generator.apply_tensor_ssa_mapping_to_inputs(memref_to_tensor)

        # Now go through and generate IR in tensor notation for each dependence
        # tree, also for outputs store the mapping between tensor and the mapped
        # memref that this corresponds to
        ops_list = []
        ssa_out_to_memref = {}
        for tree in dependence_tree_generator.dependency_trees:
            ops, ssa = tree[1].generate()
            ssa_out_to_memref[ssa] = tree[0].memref
            ops_list += ops

        # We need to yield the tensors from this block, therefore create this
        # from all the output tensors that have been generated
        out_ssa = list(ssa_out_to_memref.keys())
        ops_list.append(device.TensorYieldOp(*out_ssa))
        new_block.add_ops(ops_list)

        # Build the tensor compute operation, with result types of tensors
        out_types = [ssa.type for ssa in out_ssa]
        compute_op = device.TensorComputeOp(
            input_vars,
            loop_nest_op.lowerBound,
            loop_nest_op.upperBound,
            loop_nest_op.step,
            out_types,
            Region(new_block),
        )

        rewriter.insert_op_before_matched_op(compute_op)

        # The compute tensor operation produces tensors as output, these need
        # to be materialize to the device mapped memrefs
        materialisation_ops = []
        for idx, ssa in enumerate(out_ssa):
            tensor_type = compute_op.results[0].type
            memref_type = ssa_out_to_memref[ssa].type
            assert tensor_type.element_type == memref_type.element_type
            assert len(tensor_type.shape) == len(memref_type.shape)

            materialisation_ssa = compute_op.results[0]
            if tensor_type.shape != memref_type.shape:
                # We need to cast the tensor's shape to the memref shape, so
                # that it can be materialised. This is quite common as we
                # try and make tensors static size (from the loop bounds) whereas
                # it is common for dimensions in the memref to be dynamic
                tensor_cast = tensor.CastOp(
                    compute_op.results[0],
                    builtin.TensorType(tensor_type.element_type, memref_type.shape),
                )
                materialisation_ssa = tensor_cast.results[0]
                materialisation_ops.append(tensor_cast)
            # Now create the materialization operation to materialize the tensor
            # into the appropriate device mapped output memref
            mat = bufferization.MaterializeInDestinationOp(
                operands=[materialisation_ssa, ssa_out_to_memref[ssa]],
                result_types=[[]],
            )
            materialisation_ops.append(mat)

        rewriter.insert_op_after_matched_op(materialisation_ops)

        # Lastly we erase the operation that this was matched on (either the worksharing
        # loop or the simd loop)
        rewriter.erase_op(op)


class LiftSIMDOp(LiftOMPToTensors):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: omp.SimdOp, rewriter: PatternRewriter):
        # TODO: extract private memref from the block
        self.lift_op([], op, rewriter)


class LiftWsLoopOp(LiftOMPToTensors):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: omp.WsLoopOp, rewriter: PatternRewriter):
        # The WS loop places private variables as allocations under the parallel
        # region, rather than using the parallel operand. Therefore extract these
        parallel_op = LiftOMPToTensors.find_parent_op(omp.ParallelOp, op)
        private_memrefs = []
        for par_op in parallel_op.region.block.ops:
            if isa(par_op, memref.AllocOp) or isa(par_op, memref.AllocaOp):
                private_memrefs.append(par_op.results[0])
        self.lift_op(private_memrefs, op, rewriter)


class RemoveParallelOp(RewritePattern):
    # This removes an enclosing OpenMP parallel operation if there is a child
    # tensor compute operation
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: omp.ParallelOp, rewriter: PatternRewriter):
        tensor_compute_op = LiftOMPToTensors.find_child_op(device.TensorComputeOp, op)
        if tensor_compute_op is not None:
            # Remove the parallel region terminator
            rewriter.erase_op(op.region.block.ops.last)
            # Now move this up
            rewriter.inline_block_before_matched_op(op.region.block)
            rewriter.erase_op(op)


class LiftOmpToTensorPass(ModulePass):
    name = "lift-omp-to-tensor"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LiftSIMDOp(),
                    LiftWsLoopOp(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)

        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    RemoveParallelOp(),
                ]
            ),
        ).rewrite_module(op)

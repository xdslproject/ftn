from abc import ABC
from enum import Enum

from xdsl.context import Context
from xdsl.dialects import arith, bufferization, builtin, memref, omp, tensor, tosa
from xdsl.ir import Block, Region
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

    class ContributedOperation:
        def walk(self, to_match):
            if isa(self, to_match):
                return [self]
            return []

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
            matched_ops = []
            if isa(self, to_match):
                matched_ops.append(self)
            matched_ops += self.lhs.walk(to_match)
            matched_ops += self.rhs.walk(to_match)
            return matched_ops

        def generate(self):
            lhs_ops, lhs_ssa = self.lhs.generate()
            rhs_ops, rhs_ssa = self.rhs.generate()

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

    class VariableLoadOperation(ContributedOperation):
        def __init__(self, var_ssa, var_type):
            self.var_ssa = var_ssa
            self.var_type = var_type
            self.tensor_ssa = None

        def set_tensor(self, tensor_ssa):
            self.tensor_ssa = tensor_ssa

        def generate(self):
            assert self.tensor_ssa is not None
            return [], self.tensor_ssa

    class VariableStoreOperation(ContributedOperation):
        def __init__(self, tensor_op):
            self.tensor_op = tensor_op

        def walk(self, to_match):
            return self.tensor_op.walk(to_match)

        def generate(self):
            ops, ssa = self.tensor_op.generate()

            return ops, ssa

    def __init__(self):
        self.tree = None

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

    def traverse_load_op(self, load_op: memref.LoadOp):
        return GetStoreCalculationContributedOperations.VariableLoadOperation(
            load_op.memref, load_op.results[0].type
        )

    def traverse_constant_op(self, constant_op: arith.ConstantOp):
        return GetStoreCalculationContributedOperations.ConstantOperation(
            constant_op.value, constant_op.results[0].type
        )

    def traverse_store_op(self, store_op: memref.StoreOp):
        ret_vals = self.traverse(store_op.value.owner)
        assert len(ret_vals) == 1
        self.tree = GetStoreCalculationContributedOperations.VariableStoreOperation(
            ret_vals[0]
        )


class BuildApplicableOpDependencyTrees(Visitor):
    def __init__(self, device_mapped_data):
        self.device_mapped_data = device_mapped_data
        self.dependency_trees = []

    def get_all_input_var_ssas(self):
        input_var_ssas = []
        for dependency_tree in self.dependency_trees:
            input_vars = dependency_tree[1].walk(
                GetStoreCalculationContributedOperations.VariableLoadOperation
            )
            for input_var in input_vars:
                if input_var.var_ssa not in input_var_ssas:
                    input_var_ssas.append(input_var.var_ssa)
        return input_var_ssas

    def apply_tensor_ssa_mapping_to_inputs(self, mapping):
        for dependency_tree in self.dependency_trees:
            input_vars = dependency_tree[1].walk(
                GetStoreCalculationContributedOperations.VariableLoadOperation
            )
            for input_var in input_vars:
                assert input_var.var_ssa in mapping
                input_var.set_tensor(mapping[input_var.var_ssa])

    def get_all_output_var_ssas(self):
        return [d[0].memref for d in self.dependency_trees]

    def traverse_store_op(self, store_op: memref.StoreOp):
        # First figure out if we care about this store, i.e. it is indexed by the loops
        # that we looking to simd
        tensorise = store_op.memref in self.device_mapped_data
        if tensorise:
            # Now determine the dependency tree of operations that we are going to convert
            # into tensorise operations
            contributed_ops = GetStoreCalculationContributedOperations()
            # Now get all the operations that contribute to the RHS (value stored)
            contributed_ops.traverse(store_op)
            self.dependency_trees.append((store_op, contributed_ops.tree))


class LiftOMPToTensors(RewritePattern, ABC):
    def find_parent_op(cls, op):
        if isa(op, cls):
            return op
        elif op is None:
            return None
        else:
            return LiftSIMDOp.find_parent_op(cls, op.parent)

    def find_child_op(cls, search_op):
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
        if isa(token, arith.ConstantOp):
            assert token.value.type == builtin.i32
            return token.value.value.data
        else:
            return None

    def lift_op(self, op: omp.SimdOp | omp.WsLoopOp, rewriter: PatternRewriter):
        loop_nest_op = LiftOMPToTensors.find_child_op(omp.LoopNestOp, op)
        assert loop_nest_op is not None

        kernel_create_op = LiftOMPToTensors.find_parent_op(device.KernelCreate, op)
        assert kernel_create_op is not None

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
                tensor_sizes.append(int((upper_const - (lower_const - 1)) / step_const))
            else:
                tensor_sizes.append(-1)

        arith_ops_generator = BuildApplicableOpDependencyTrees(
            kernel_create_op.body.block.args
        )
        arith_ops_generator.traverse(loop_nest_op)

        input_vars = arith_ops_generator.get_all_input_var_ssas()

        arg_types = []
        for input_var in input_vars:
            assert len(input_var.type.shape) == len(tensor_sizes)
            arg_types.append(
                builtin.TensorType(input_var.type.element_type, tensor_sizes)
            )

        new_block = Block(arg_types=arg_types)

        memref_to_tensor = {}
        for tensor_arg, memref_arg in zip(new_block.args, input_vars):
            memref_to_tensor[memref_arg] = tensor_arg

        arith_ops_generator.apply_tensor_ssa_mapping_to_inputs(memref_to_tensor)

        ops_list = []
        ssa_out_to_memref = {}
        for tree in arith_ops_generator.dependency_trees:
            ops, ssa = tree[1].generate()
            ssa_out_to_memref[ssa] = tree[0].memref
            ops_list += ops

        out_ssa = list(ssa_out_to_memref.keys())
        ops_list.append(device.TensorYieldOp(*out_ssa))
        new_block.add_ops(ops_list)

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

        materialisation_ops = []
        for idx, ssa in enumerate(out_ssa):
            tensor_type = compute_op.results[0].type
            memref_type = ssa_out_to_memref[ssa].type
            assert tensor_type.element_type == memref_type.element_type
            assert len(tensor_type.shape) == len(memref_type.shape)

            materialisation_ssa = compute_op.results[0]
            if tensor_type.shape != memref_type.shape:
                # Need to cast the tensor to the memref shape, this happens when
                # we have set a static tensor size due to discovering this, but
                # the memref is dynamically allocated
                tensor_cast = tensor.CastOp(
                    compute_op.results[0],
                    builtin.TensorType(tensor_type.element_type, memref_type.shape),
                )
                materialisation_ssa = tensor_cast.results[0]
                materialisation_ops.append(tensor_cast)
            mat = bufferization.MaterializeInDestinationOp(
                operands=[materialisation_ssa, ssa_out_to_memref[ssa]],
                result_types=[[]],
            )
            materialisation_ops.append(mat)

        rewriter.insert_op_after_matched_op(materialisation_ops)

        rewriter.erase_op(op)


class LiftSIMDOp(LiftOMPToTensors):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: omp.SimdOp, rewriter: PatternRewriter):
        self.lift_op(op, rewriter)


class LiftWsLoopOp(LiftOMPToTensors):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: omp.WsLoopOp, rewriter: PatternRewriter):
        self.lift_op(op, rewriter)


class RemoveParallelOp(RewritePattern):
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

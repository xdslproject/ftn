from abc import ABC
from enum import Enum
import itertools
import copy
from functools import reduce
from typing import TypeVar, cast
from dataclasses import dataclass
from xdsl.dialects.experimental import fir, hlfir
from dataclasses import dataclass, field
from typing import Dict, Optional
from xdsl.ir import SSAValue, BlockArgument
from xdsl.irdl import Operand
from xdsl.utils.hints import isa
from ftn.util.visitor import Visitor
from xdsl.context import Context
from xdsl.ir import Operation, SSAValue, OpResult, Attribute, Block, Region

from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter,
    op_type_rewrite_pattern,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
)
from xdsl.passes import ModulePass
from xdsl.dialects import builtin, func, llvm, arith, memref, scf, cf, linalg, omp, math
from ftn.dialects import ftn_relative_cf

from ftn.transforms.to_core.misc.fortran_code_description import (
    ProgramState,
    GlobalFIRComponent,
    FunctionDefinition,
    ArgumentDefinition,
    ComponentState,
    ArrayDescription,
    ArgIntent,
)
from ftn.transforms.to_core.misc.ssa_context import SSAValueCtx

from ftn.transforms.to_core.utils import clean_func_name
from ftn.transforms.to_core.components.intrinsics import (
    FortranIntrinsicsHandleExplicitly,
)
import ftn.transforms.to_core.components.functions as ftn_functions
import ftn.transforms.to_core.components.openmp as ftn_openmp

from ftn.transforms.to_core.expressions import translate_expr
from ftn.transforms.to_core.statements import translate_stmt


class GatherFIRGlobals(Visitor):
    def __init__(self, program_state):
        self.program_state = program_state

    def traverse_global(self, global_op: fir.GlobalOp):
        if global_op.constant is not None:
            gfir = GlobalFIRComponent(
                global_op.sym_name.data, global_op.type, global_op
            )
            self.program_state.fir_global_constants[global_op.sym_name.data] = gfir


class GatherFunctionInformation(Visitor):
    def __init__(self, program_state):
        self.program_state = program_state

    def get_declare_from_arg_uses(self, arg_uses):
        for use in arg_uses:
            # It can unbox characters into a declare, if so then just follow it through
            if isa(use.operation, fir.UnboxcharOp):
                ub_dec = self.get_declare_from_arg_uses(use.operation.results[0].uses)
                if ub_dec is not None:
                    return ub_dec
            if isa(use.operation, hlfir.DeclareOp):
                return use.operation
        return None

    def map_ftn_attrs_to_intent(self, ftn_attrs):
        if ftn_attrs is not None:
            for attr in ftn_attrs.data:
                if attr == fir.FortranVariableFlags.INTENT_IN:
                    return ArgIntent.IN
                elif attr == fir.FortranVariableFlags.INTENT_INOUT:
                    return ArgIntent.INOUT
                elif attr == fir.FortranVariableFlags.INTENT_OUT:
                    return ArgIntent.OUT
        return ArgIntent.UNKNOWN

    def check_if_has_allocatable_attr(self, op: hlfir.DeclareOp):
        if "fortran_attrs" in op.properties.keys():
            attrs = op.properties["fortran_attrs"]
            assert isa(attrs, fir.FortranVariableFlagsAttr)
            for attr in attrs.data:
                if attr == fir.FortranVariableFlags.ALLOCATABLE:
                    return True
        return False

    def get_base_type(self, t):
        if isa(t, fir.ReferenceType):
            return self.get_base_type(t.type)
        elif isa(t, fir.BoxType):
            return self.get_base_type(t.type)
        elif isa(t, fir.HeapType):
            return self.get_base_type(t.type)
        return t

    def traverse_func_op(self, func_op: func.FuncOp):
        fn_name = clean_func_name(func_op.sym_name.data)
        return_type = None
        if len(func_op.function_type.outputs.data) > 0:
            return_type = func_op.function_type.outputs.data[0]
        fn_def = FunctionDefinition(
            fn_name,
            return_type,
            len(func_op.body.blocks) == 0,
            list(func_op.body.blocks),
        )
        if len(func_op.body.blocks) != 0:
            # This has concrete implementation (e.g. is not a function definition)
            assert len(func_op.body.blocks) >= 1
            # Even if the body has more than one block, we only care about the first block as that is
            # the entry point from the function, so it has the function arguments in it
            for block_arg in func_op.body.blocks[0].args:
                declare_op = self.get_declare_from_arg_uses(block_arg.uses)
                if declare_op is not None:
                    arg_type = declare_op.results[0].type
                    base_type = self.get_base_type(arg_type)
                    is_scalar = declare_op.shape is None and not isa(
                        base_type, fir.SequenceType
                    )
                    arg_name = declare_op.uniq_name.data
                    is_allocatable = self.check_if_has_allocatable_attr(declare_op)
                    # This is a bit strange, in a module we have modulenamePprocname, however
                    # flang then uses modulenameFprocname for array literal string names
                    # assert fn_name.replace("P", "F")+"E" in arg_name
                    # arg_name=arg_name.split(fn_name.replace("P", "F")+"E")[1]
                    arg_intent = self.map_ftn_attrs_to_intent(declare_op.fortran_attrs)
                    arg_def = ArgumentDefinition(
                        arg_name, is_scalar, arg_type, arg_intent, is_allocatable
                    )
                else:
                    # This is an internal variable passed by Flang, for now we assume is a scalar
                    arg_type = block_arg.type
                    is_scalar = True
                    arg_name = builtin.StringAttr("")
                    is_allocatable = False
                    arg_intent = ArgIntent.INOUT
                    arg_def = ArgumentDefinition(
                        arg_name, is_scalar, arg_type, arg_intent, is_allocatable
                    )
                fn_def.add_arg_def(arg_def)
        else:
            # This is a definition, we will grab the argument types from the operation instead of the block
            # can have less information than a declared function (with a body) as it's simply calling
            # into a block box
            for input_arg in func_op.function_type.inputs.data:
                is_scalar = True
                arg_name = builtin.StringAttr("")
                is_allocatable = False
                arg_intent = ArgIntent.INOUT
                arg_def = ArgumentDefinition(
                    arg_name, is_scalar, input_arg, arg_intent, is_allocatable
                )
                fn_def.add_arg_def(arg_def)
        self.program_state.addFunctionDefinition(fn_name, fn_def)


class GatherFunctions(Visitor):
    def __init__(self):
        self.functions = {}

    def traverse_func_op(self, func_op: func.FuncOp):
        fn_name = func_op.sym_name.data
        if isa(fn_name, builtin.StringAttr):
            fn_name = fn_name.data
        fn_name = clean_func_name(fn_name)
        self.functions[fn_name] = func_op


def translate_program(
    program_state: ProgramState, input_module: builtin.ModuleOp
) -> builtin.ModuleOp:
    # create an empty global context
    global_ctx = SSAValueCtx()
    body = Region()
    block = Block()
    for fn in input_module.ops:
        if isa(fn, func.FuncOp):
            fn_op = ftn_functions.translate_function(program_state, global_ctx, fn)
            if fn_op is not None:
                block.add_op(fn_op)
        elif isa(fn, fir.GlobalOp):
            global_op = translate_global(program_state, global_ctx, fn)
            if global_op is not None:
                block.add_op(global_op)
        elif isa(fn, omp.PrivateClauseOp):
            private_op = ftn_openmp.translate_private(program_state, global_ctx, fn)
            if private_op is not None:
                block.add_op(private_op)
        elif isa(fn, omp.DeclareReductionOp):
            declare_reduction_op = ftn_openmp.translate_declarereduction(
                program_state, global_ctx, fn
            )
            if declare_reduction_op is not None:
                block.add_op(declare_reduction_op)
        else:
            assert False
    body.add_block(block)
    return builtin.ModuleOp(body)


def translate_global(program_state, global_ctx, global_op: fir.GlobalOp):
    if global_op.sym_name.data == "_QQEnvironmentDefaults":
        return None
    assert len(global_op.regions) == 1
    assert len(global_op.regions[0].blocks) == 1

    ops_list = []
    program_state.enterGlobal()
    for op in global_op.regions[0].blocks[0].ops:
        ops_list += translate_stmt(program_state, global_ctx, op)

    program_state.leaveGlobal()

    if isa(global_op.type, fir.CharacterType):
        assert len(ops_list) == 1
        rebuilt_global = llvm.GlobalOp(
            ops_list[0].global_type,
            global_op.sym_name,
            ops_list[0].linkage,
            ops_list[0].addr_space.value.data,
            global_op.constant,
            value=ops_list[0].value,
            unnamed_addr=ops_list[0].unnamed_addr.value.data,
        )
        return rebuilt_global
    elif (
        isa(global_op.type, builtin.IntegerType)
        or isa(global_op.type, builtin.AnyFloat)
        or isa(global_op.type, fir.SequenceType)
        or isa(global_op.type, fir.LogicalType)
    ):
        assert len(ops_list) == 1
        global_contained_op = ops_list[0]
        if isa(global_contained_op, memref.GlobalOp):
            # If this is a memref global operation then simply return that and are done
            return global_contained_op
        else:
            # Otherwise need to package in llvm.GlobalOp
            assert isa(global_contained_op, arith.ConstantOp) or isa(
                global_contained_op, llvm.ZeroOp
            )
            return_op = llvm.ReturnOp.build(operands=[global_contained_op.results[0]])

            return llvm.GlobalOp(
                global_contained_op.results[0].type,
                global_op.sym_name,
                "internal",
                constant=global_op.constant,
                body=Region([Block([global_contained_op, return_op])]),
            )
    elif (
        isa(global_op.type, fir.BoxType)
        and isa(global_op.type.type, fir.HeapType)
        and isa(global_op.type.type.type, fir.SequenceType)
    ):
        # This represents a container of an allocatable array, Flang will generate these when we have
        # an allocatable passed between procedures. However we handle this differently by a memref of a memref
        # which is scoped, and hence ignore this here to avoid a global
        pass
    elif (
        isa(global_op.type, fir.BoxType)
        and isa(global_op.type.type, fir.PointerType)
        and isa(global_op.type.type.type, fir.SequenceType)
    ):
        # This represents an initial value for a pointer, such as NULL(), for now we ignore these
        pass
    else:
        raise Exception(f"Could not translate global region of type `{global_op.type}'")


class RewriteRelativeBranch(RewritePattern):
    def __init__(self, functions):
        self.functions = functions

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: ftn_relative_cf.BranchOp, rewriter: PatternRewriter, /
    ):
        containing_fn_name = op.function_name.data
        assert containing_fn_name in self.functions.keys()
        assert (
            len(self.functions[containing_fn_name].regions[0].blocks)
            > op.successor.value.data
        )
        cf_branch_op = cf.BranchOp(
            self.functions[containing_fn_name]
            .regions[0]
            .blocks[op.successor.value.data],
            *op.arguments,
        )
        rewriter.replace_matched_op(cf_branch_op)


class RewriteRelativeConditionalBranch(RewritePattern):
    def __init__(self, functions):
        self.functions = functions

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: ftn_relative_cf.ConditionalBranchOp, rewriter: PatternRewriter, /
    ):
        containing_fn_name = op.function_name.data
        assert containing_fn_name in self.functions.keys()
        assert (
            len(self.functions[containing_fn_name].regions[0].blocks)
            > op.then_block.value.data
        )
        assert (
            len(self.functions[containing_fn_name].regions[0].blocks)
            > op.else_block.value.data
        )
        cf_cbranch_op = cf.ConditionalBranchOp(
            op.cond,
            self.functions[containing_fn_name]
            .regions[0]
            .blocks[op.then_block.value.data],
            op.then_arguments,
            self.functions[containing_fn_name]
            .regions[0]
            .blocks[op.else_block.value.data],
            op.else_arguments,
        )
        rewriter.replace_matched_op(cf_cbranch_op)


@dataclass(frozen=True)
class RewriteFIRToCore(ModulePass):
    """
    This is the entry point for the transformation pass which will then apply the rewriter
    """

    name = "rewrite-fir-to-core"

    def apply(self, ctx: Context, input_module: builtin.ModuleOp):
        program_state = ProgramState()
        fn_visitor = GatherFunctionInformation(program_state)
        fn_visitor.traverse(input_module)
        global_visitor = GatherFIRGlobals(program_state)
        global_visitor.traverse(input_module)
        res_module = translate_program(program_state, input_module)
        # Detach the Fortran block first
        input_module.body.detach_block(input_module.body.block)
        # Move blocks from newly created module to the input module region
        res_module.regions[0].move_blocks(input_module.regions[0])

        # Clean out module attributes to remove dlti and fir specific ones
        attr_list = list(input_module.attributes)
        for attr in attr_list:
            if attr.startswith("dlti.") or attr.startswith("fir."):
                del input_module.attributes[attr]

        fn_gatherer = GatherFunctions()
        fn_gatherer.traverse(input_module)

        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    RewriteRelativeBranch(fn_gatherer.functions),
                    RewriteRelativeConditionalBranch(fn_gatherer.functions),
                ]
            ),
            apply_recursively=False,
        )
        walker.rewrite_module(input_module)

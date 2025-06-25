from __future__ import annotations

from collections.abc import Sequence

from xdsl.dialects.builtin import IntegerType, IntegerAttr, StringAttr
from xdsl.ir import Block, Dialect, Operation, SSAValue
from xdsl.utils.hints import isa
from xdsl.irdl import (
    AnyAttr,
    AttrSizedOperandSegments,
    IRDLOperation,
    Operand,
    Successor,
    VarOperand,
    irdl_op_definition,
    operand_def,
    prop_def,
    traits_def,
    successor_def,
    var_operand_def,
)
from xdsl.traits import IsTerminator


@irdl_op_definition
class Branch(IRDLOperation):
    name = "ftn_relative_cf.br"

    arguments: VarOperand = var_operand_def(AnyAttr())
    successor: IntegerAttr = prop_def(IntegerAttr)
    function_name: StringAttr = prop_def(StringAttr)

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        function_name: str | StringAttr,
        dest: int | IntegerAttr,
        *ops: Operation | SSAValue,
    ):
        if isa(dest, int):
            dest = IntegerAttr.from_int_and_width(dest, 32)

        if isa(function_name, str):
            function_name = StringAttr(function_name)

        super().__init__(
            operands=[[op for op in ops]],
            properties={"function_name": function_name, "successor": dest},
        )


@irdl_op_definition
class ConditionalBranch(IRDLOperation):
    name = "ftn_relative_cf.cond_br"

    cond: Operand = operand_def(IntegerType(1))
    then_arguments: VarOperand = var_operand_def(AnyAttr())
    else_arguments: VarOperand = var_operand_def(AnyAttr())

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    then_block: IntegerAttr = prop_def(IntegerAttr)
    else_block: IntegerAttr = prop_def(IntegerAttr)
    function_name: StringAttr = prop_def(StringAttr)

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        function_name: str | StringAttr,
        cond: Operation | SSAValue,
        then_block: int | IntegerAttr,
        then_ops: Sequence[Operation | SSAValue],
        else_block: int | IntegerAttr,
        else_ops: Sequence[Operation | SSAValue],
    ):
        if isa(then_block, int):
            then_block = IntegerAttr.from_int_and_width(then_block, 32)

        if isa(else_block, int):
            else_block = IntegerAttr.from_int_and_width(else_block, 32)

        if isa(function_name, str):
            function_name = StringAttr(function_name)

        super().__init__(
            operands=[cond, then_ops, else_ops],
            properties={
                "function_name": function_name,
                "then_block": then_block,
                "else_block": else_block,
            },
        )


Ftn_relative_cf = Dialect(
    "ftn_relative_cf",
    [
        Branch,
        ConditionalBranch,
    ],
    [],
)

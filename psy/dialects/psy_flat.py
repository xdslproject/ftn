from __future__ import annotations
from typing import Union
from io import StringIO

from dataclasses import dataclass
from xdsl.dialects.builtin import StringAttr, IntegerAttr
from xdsl.irdl import irdl_op_definition, irdl_attr_definition, ParameterDef, OperandDef, ResultDef, OptResultDef, SingleBlockRegionDef, AttributeDef, VarOperandDef, AnyAttr, AnyOf, builder
from xdsl.ir import ParametrizedAttribute, Operation, MLContext, Data, SSAValue

from psy.dialects.psy_ast import FloatAttr
from psy.dialects.psy_type import int_type, str_type, bool_type, none_type, char_type
from xdsl.diagnostic import Diagnostic


def error(op: Operation, msg: str):
    diag = Diagnostic()
    diag.add_message(op, msg)
    diag.raise_exception(f"{op.name} operation does not verify", op)
    
@irdl_attr_definition
class FloatAttr(Data):
    name = 'psy.ir.float'
    data: float

    @staticmethod
    def parse(parser: Parser) -> Data:
        val = parser.parse_while(lambda char: char != '>')
        return FloatAttr(float(val))  # type: ignore        

    def print(self, printer: Printer) -> None:
        if self.data:
            printer.print_string(str(self.data))

    @staticmethod
    @builder
    def from_float(val: float) -> FloatAttr:
        return FloatAttr(float)   

@irdl_op_definition
class FileContainer(Operation):
    name = "psy.ir.filecontainer"
    
    containers = SingleBlockRegionDef()

@irdl_op_definition
class Container(Operation):
    name = "psy.ir.container"

    container_name = AttributeDef(StringAttr)
    routines = SingleBlockRegionDef()      

@irdl_op_definition
class Routine(Operation):
    name = "psy.ir.routine"

    routine_name = AttributeDef(StringAttr)
    return_type = AttributeDef(AnyAttr())
    #local_var_declarations = SingleBlockRegionDef()
    #params = SingleBlockRegionDef()
    routine_body = SingleBlockRegionDef()


@irdl_op_definition
class VarDef(Operation):
    name = "psy.ir.var_def"

    var_name = AttributeDef(StringAttr)
    type = ResultDef(AnyAttr())

# Statements

@irdl_op_definition
class If(Operation):
    name = "psy.ir.if"

    cond = OperandDef(bool_type)
    then = SingleBlockRegionDef()
    orelse = SingleBlockRegionDef()
    
@irdl_op_definition
class Do(Operation):
    name = "psy.ir.do"

    iter_ = OperandDef(AnyAttr())
    start = OperandDef(AnyAttr())
    stop = OperandDef(AnyAttr())
    step = OperandDef(AnyAttr())
    body = SingleBlockRegionDef()    

@irdl_op_definition
class Assign(Operation):
    name = "psy.ir.assign"

    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())

# Expressions

@irdl_op_definition
class Literal(Operation):
    name = "psy.ir.literal"

    value = AttributeDef(AnyAttr())
    result = ResultDef(AnyAttr())

    @staticmethod
    def get(value: Union[None, bool, int, str, float],
            verify_op: bool = True) -> Literal:
        if type(value) is int:
            attr = IntegerAttr.from_int_and_width(value, 32)
            ty = int_type
        elif type(value) is float:
            attr = FloatAttr.from_float(value)
            ty = float_type
        elif type(value) is str:
            attr = StringAttr.from_str(value)
            ty = str_type
        else:
            raise Exception(f"Unknown literal of type {type(value)}")

        res = Literal.build(operands=[],
                            attributes={"value": attr},
                            result_types=[ty])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

@irdl_op_definition
class BinaryExpr(Operation):
    name = "psy.ir.binary_expr"

    op = AttributeDef(StringAttr)
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())


@irdl_op_definition
class EffectfulBinaryExpr(Operation):
    name = "psy.ir.effectful_binary_expr"

    op = AttributeDef(StringAttr)
    lhs = SingleBlockRegionDef()
    rhs = SingleBlockRegionDef()
    result = ResultDef(AnyAttr())

@irdl_op_definition
class CallExpr(Operation):
    name = "psy.ir.call_expr"

    args = VarOperandDef(AnyAttr())
    func_name = AttributeDef(AnyAttr())
    result = OptResultDef(AnyAttr())

@dataclass
class PsyFlat:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_attr(FloatAttr)
        
        self.ctx.register_op(FileContainer)
        self.ctx.register_op(Container)
        self.ctx.register_op(Routine)
        self.ctx.register_op(VarDef)
        self.ctx.register_op(If)
        self.ctx.register_op(Do)
        self.ctx.register_op(Assign)
        self.ctx.register_op(Literal)
        self.ctx.register_op(BinaryExpr)
        self.ctx.register_op(EffectfulBinaryExpr)
        self.ctx.register_op(CallExpr)

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Type, Union

from xdsl.dialects.builtin import IntegerAttr, StringAttr, IntegerType, Float32Type, i32, f32, ArrayAttr
from xdsl.ir import Data, MLContext, Operation, ParametrizedAttribute
from xdsl.irdl import (AnyOf, AttributeDef, SingleBlockRegionDef, builder, AnyAttr, ResultDef, OperandDef,
                       irdl_attr_definition, irdl_op_definition)
from xdsl.parser import Parser
from xdsl.printer import Printer

@irdl_attr_definition
class BoolAttr(Data):
    name = "bool"
    data: bool

    @staticmethod
    def parse(parser: Parser) -> BoolAttr:
        data = parser.parse_int_literal()
        return IntAttr(data)

    def print(self, printer: Printer) -> None:
        printer.print_string(f'{self.data}')

    @staticmethod
    @builder
    def from_int(data: bool) -> BoolAttr:
        return BoolAttr(data)

@irdl_op_definition
class FileContainer(Operation):
    name = "psy.dag.filecontainer"

    programs = SingleBlockRegionDef()

    @staticmethod
    def get(programs: List[Operation],
            verify_op: bool = True) -> FileContainer:
      res = FileContainer.build(regions=[programs])
      if verify_op:
        res.verify(verify_nested_ops=False)
      return res

    def verify_(self) -> None:
      pass
    
@irdl_op_definition
class Container(Operation):
    name = "psy.dag.container"

    container_name = AttributeDef(StringAttr)
    routines = SingleBlockRegionDef()    

    @staticmethod
    def get(container_name: str,
            routines: List[Operation],
            verify_op: bool = True) -> Container:
      res = Container.build(attributes={"container_name": container_name}, regions=[routines])
      if verify_op:
        res.verify(verify_nested_ops=False)
      return res

    def verify_(self) -> None:
      pass
                
@irdl_op_definition
class Routine(Operation):
    name = "psy.dag.routine"

    routine_name = AttributeDef(StringAttr)
    params = SingleBlockRegionDef()
    return_type = AttributeDef(StringAttr)
    required_module_uses=AttributeDef(ArrayAttr)
    program_entry_point=AttributeDef(BoolAttr)
    
    local_var_declarations = SingleBlockRegionDef()
    routine_body = SingleBlockRegionDef()

    @staticmethod
    def get(routine_name: Union[str, StringAttr],
            return_type: str,
            params: List[Operation],            
            local_var_declarations: List[Operation],
            routine_body: List[Operation],
            program_entry_point : bool = True,
            verify_op: bool = True) -> Routine:
        res = Routine.build(attributes={"routine_name": routine_name, "return_type": return_type, "required_module_uses": [], "program_entry_point": program_entry_point},
                            regions=[params, local_var_declarations, routine_body])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

    def verify_(self) -> None:
      pass
    
@irdl_attr_definition
class FloatAttr(Data):
    name = 'psy.dag.float'
    data: float

    @staticmethod
    def parse(parser: Parser) -> Data:
        val = parser.parse_while(lambda char: char != '>')        
        return FloatAttr(str(val))        

    def print(self, printer: Printer) -> None:
        printer.print_string(f'{self.data}')

    @staticmethod
    @builder
    def from_float(val: float) -> FloatAttr:
        return FloatAttr(val) 
    
@irdl_op_definition
class ExprName(Operation):
    name = "psy.dag.id_expr"

    id = AttributeDef(StringAttr)
    var= OperandDef(AnyAttr())

    @staticmethod
    def get(name: Union[str, StringAttr], v, verify_op: bool = True) -> ExprName:
        res = ExprName.build(attributes={"id": name, "var": v})
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res    
            
@irdl_op_definition
class VarDef(Operation):
    name = "psy.dag.var_def"

    type = AttributeDef(AnyAttr())
    result = ResultDef(AnyAttr())
    var_name = AttributeDef(StringAttr)
    
@irdl_op_definition
class Assign(Operation):
    name = "psy.dag.assign"

    lhs = SingleBlockRegionDef()
    rhs = SingleBlockRegionDef()

    @staticmethod
    def get(lhs: Operation,
            rhs: Operation,
            verify_op: bool = True) -> Assign:
        res = Assign.build(regions=[[lhs], [rhs]])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

    def verify_(self) -> None:
      pass
                
@irdl_op_definition
class Literal(Operation):
    name = "psy.dag.literal"

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
class If(Operation):
    name = "psy.dag.if"

    cond = SingleBlockRegionDef()
    then = SingleBlockRegionDef()
    orelse = SingleBlockRegionDef()

    @staticmethod
    def get(cond: Operation,
            then: List[Operation],
            orelse: List[Operation],
            verify_op: bool = True) -> If:
        res = If.build(regions=[[cond], then, orelse])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

    def verify_(self) -> None:
      pass
    
@irdl_op_definition
class Do(Operation):
    name = "psy.dag.do"

    iter_name = AttributeDef(StringAttr)
    start = SingleBlockRegionDef()
    stop = SingleBlockRegionDef()
    step = SingleBlockRegionDef()
    body = SingleBlockRegionDef()

    @staticmethod
    def get(iter_name: Union[str, StringAttr],
            start: Operation,
            stop: Operation,
            step: Operation,
            body: List[Operation],
            verify_op: bool = True) -> If:
        res = Do.build(attributes={"iter_name": iter_name}, regions=[[start], [stop], [step], body])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

    def verify_(self) -> None:
      pass
      
@irdl_op_definition
class BinaryExpr(Operation):
    name = "psy.dag.binary_expr"

    op = AttributeDef(StringAttr)
    lhs = SingleBlockRegionDef()
    rhs = SingleBlockRegionDef()

    @staticmethod
    def get_valid_ops() -> List[str]:
        return ["+", "-", "*", "/", "%", "pow", "is", "&&", "||", ">", "<", "==", "!=", ">=", "<=", "copysign"]

    @staticmethod
    def get(op: str,
            lhs: Operation,
            rhs: Operation,
            verify_op: bool = True) -> BinaryExpr:
        res = BinaryExpr.build(attributes={"op": op}, regions=[[lhs], [rhs]])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

    def verify_(self) -> None:
      pass
            
@irdl_op_definition
class CallExpr(Operation):
    name = "psy.dag.call_expr"

    func = AttributeDef(StringAttr)
    isstatement = AttributeDef(BoolAttr)
    args = SingleBlockRegionDef()

    @staticmethod
    def get(func: str,
            args: List[Operation], isstatement,
            verify_op: bool = True) -> CallExpr:        
        res = CallExpr.build(regions=[args], attributes={"func": func, "isstatement": isstatement})
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

    def verify_(self) -> None:
      pass
                
@dataclass
class PsyDAG:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_attr(FloatAttr)
        
        self.ctx.register_op(FileContainer)
        self.ctx.register_op(Container)
        self.ctx.register_op(Routine)
        self.ctx.register_op(VarDef)        
        self.ctx.register_op(Assign)
        self.ctx.register_op(If)
        self.ctx.register_op(Do)
        self.ctx.register_op(Literal)
        self.ctx.register_op(ExprName)
        self.ctx.register_op(BinaryExpr)        
        self.ctx.register_op(CallExpr)

    @staticmethod
    def get_type(annotation: str) -> Operation:
        return TypeName.get(annotation)

    @staticmethod
    def get_statement_op_types() -> List[Type[Operation]]:
        statements: List[Type[Operation]] = [
            Assign, If, Do
        ]
        return statements + PsycloneAST.get_expression_op_types()

    @staticmethod
    def get_expression_op_types() -> List[Type[Operation]]:
        return [
            BinaryExpr, CallExpr, Literal, ExprName
        ]

    @staticmethod
    def get_type_op_types() -> List[Type[Operation]]:
        return []

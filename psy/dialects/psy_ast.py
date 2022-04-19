from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Type, Union

from xdsl.dialects.builtin import IntegerAttr, StringAttr, IntegerType, Float32Type, i32, f32
from xdsl.ir import Data, MLContext, Operation, ParametrizedAttribute
from xdsl.irdl import (AnyOf, AttributeDef, SingleBlockRegionDef, builder,
                       irdl_attr_definition, irdl_op_definition)
from xdsl.parser import Parser
from xdsl.printer import Printer

@irdl_op_definition
class FileContainer(Operation):
    name = "psy.ast.filecontainer"

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
    name = "psy.ast.container"

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
    name = "psy.ast.routine"

    routine_name = AttributeDef(StringAttr)
    params = SingleBlockRegionDef()
    return_type = AttributeDef(StringAttr)
    
    local_var_declarations = SingleBlockRegionDef()
    routine_body = SingleBlockRegionDef()

    @staticmethod
    def get(routine_name: Union[str, StringAttr],
            return_type: str,
            params: List[Operation],            
            local_var_declarations: List[Operation],
            routine_body: List[Operation],
            verify_op: bool = True) -> Routine:
        res = Routine.build(attributes={"routine_name": routine_name, "return_type": return_type},
                            regions=[params, local_var_declarations, routine_body])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

    def verify_(self) -> None:
      pass
    
@irdl_attr_definition
class FloatAttr(Data):
    name = 'psy.ast.float'
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
    name = "psy.ast.id_expr"

    id = AttributeDef(StringAttr)

    @staticmethod
    def get(name: Union[str, StringAttr], verify_op: bool = True) -> ExprName:
        res = ExprName.build(attributes={"id": name})
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res
            
@irdl_op_definition
class VarDef(Operation):
    name = "psy.ast.var_def"
    
    TYPE_MAP_TO_PSY = {"int": i32,
                       "float": f32}

    type = AttributeDef(AnyOf([IntegerType, Float32Type]))
    var_name = AttributeDef(StringAttr)

    @staticmethod
    def get(typed_var: str,
            var_name: str,
            verify_op: bool = True) -> VarDef:        
        res = VarDef.build(attributes={"var_name": var_name, "type": VarDef.TYPE_MAP_TO_PSY[typed_var]})
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

    def verify_(self) -> None:
      pass
            
@irdl_op_definition
class Assign(Operation):
    name = "psy.ast.assign"

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
    name = "psy.ast.literal"

    value = AttributeDef(AnyOf([StringAttr, IntegerAttr, FloatAttr]))

    @staticmethod
    def get(value: Union[None, bool, int, str, float],
            verify_op: bool = True) -> Literal:        
        if type(value) is int:
            attr = IntegerAttr.from_int_and_width(value, 32)
        elif type(value) is float:
            attr = FloatAttr.from_float(value)
        elif type(value) is str:
            attr = StringAttr.from_str(value)
        else:
            raise Exception(f"Unknown literal of type {type(value)}")
        res = Literal.create(attributes={"value": attr})
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res
      
@irdl_op_definition
class If(Operation):
    name = "psy.ast.if"

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
    name = "psy.ast.do"

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
    name = "psy.ast.binary_expr"

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
    name = "psy.ast.call_expr"

    func = AttributeDef(StringAttr)
    args = SingleBlockRegionDef()

    @staticmethod
    def get(func: str,
            args: List[Operation],
            verify_op: bool = True) -> CallExpr:
        res = CallExpr.build(regions=[args], attributes={"func": func})
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

    def verify_(self) -> None:
      pass
                
@dataclass
class PsyAST:
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

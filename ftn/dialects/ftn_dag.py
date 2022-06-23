from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Type, Union

from xdsl.dialects.builtin import IntegerAttr, StringAttr, IntegerType, Float32Type, i32, f32, ArrayAttr, BoolAttr
from xdsl.ir import Data, MLContext, Operation, ParametrizedAttribute
from xdsl.irdl import (AnyOf, AttributeDef, SingleBlockRegionDef, builder, AnyAttr, ResultDef, OperandDef,
                       irdl_attr_definition, irdl_op_definition, ParameterDef)
from xdsl.parser import Parser
from xdsl.printer import Printer

@irdl_attr_definition
class AnonymousAttr(ParametrizedAttribute):
    name = "anonymous"

@irdl_op_definition
class FileContainer(Operation):
    name = "ftn.dag.filecontainer"

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
    name = "ftn.dag.container"

    container_name = AttributeDef(StringAttr)
    imports = SingleBlockRegionDef()
    routines = SingleBlockRegionDef()
    default_visibility = AttributeDef(StringAttr)
    public_routines = AttributeDef(ArrayAttr)
    private_routines = AttributeDef(ArrayAttr)

    @staticmethod
    def get(container_name: str,
            default_visibility: str,
            public_routines: List[str],
            private_routines: List[str],
            imports: List[Operation],
            routines: List[Operation],
            verify_op: bool = True) -> Container:
      res = Container.build(attributes={"container_name": container_name, "default_visibility": default_visibility, 
                                        "public_routines": public_routines, "private_routines": private_routines}, regions=[imports, routines])
      if verify_op:
        res.verify(verify_nested_ops=False)
      return res

    def verify_(self) -> None:
      pass
    
@irdl_op_definition
class Import(Operation):
    name = "ftn.dag.import"
    
    import_name=AttributeDef(StringAttr)
    specific_procedures=AttributeDef(ArrayAttr)
    
    @staticmethod
    def get(import_name: str,
            specific_procedures: List[str],
            verify_op: bool = True) -> Container:
      res = Import.build(attributes={"import_name": import_name, "specific_procedures": specific_procedures})
      if verify_op:
        res.verify(verify_nested_ops=False)
      return res

    def verify_(self) -> None:
      pass  
                
@irdl_op_definition
class Routine(Operation):
    name = "ftn.dag.routine"

    routine_name = AttributeDef(StringAttr)
    args = AttributeDef(ArrayAttr)
    return_type = AttributeDef(StringAttr)
    program_entry_point=AttributeDef(BoolAttr)
    
    imports = SingleBlockRegionDef()
    local_var_declarations = SingleBlockRegionDef()
    routine_body = SingleBlockRegionDef()

    @staticmethod
    def get(routine_name: Union[str, StringAttr],
            return_type: str,
            args: List[Operation],
            imports: List[Operation],
            local_var_declarations: List[Operation],
            routine_body: List[Operation],
            program_entry_point : bool = False,
            verify_op: bool = True) -> Routine:
        res = Routine.build(attributes={"routine_name": routine_name, "return_type": return_type, "program_entry_point": program_entry_point, "args": args},
                            regions=[imports, local_var_declarations, routine_body])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

    def verify_(self) -> None:
      pass
    
@irdl_attr_definition
class FloatAttr(Data):
    name = 'ftn.dag.float'
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
class MemberAccess(Operation):
    name = "ftn.dag.member_access_expr"    

    var = AttributeDef(AnyAttr())
    fields = AttributeDef(ArrayAttr)
    
    @staticmethod
    def get(var, fields, verify_op: bool = True) -> ExprName:
        res = MemberAccess.build(attributes={"var": var, "fields": fields})
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res
      
@irdl_op_definition
class ArrayAccess(Operation):
    name="ftn.dag.array_access_expr"
    
    var = SingleBlockRegionDef()
    accessors = SingleBlockRegionDef()
    
    @staticmethod
    def get(var: List[Operation], 
            accessors: List[Operation], 
            verify_op: bool = True) -> ExprName:
        res = ArrayAccess.build(regions=[var, accessors])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res      
    
@irdl_op_definition
class ExprName(Operation):
    name = "ftn.dag.id_expr"

    id = AttributeDef(StringAttr)
    var= AttributeDef(AnyAttr())

    @staticmethod
    def get(name: Union[str, StringAttr], v, verify_op: bool = True) -> ExprName:
        res = ExprName.build(attributes={"id": name, "var": v})
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res
      
@irdl_attr_definition
class Token(ParametrizedAttribute):
    name = "ftn.dag.token"

    var_name = ParameterDef(StringAttr)
    type = ParameterDef(AnyAttr())
            
@irdl_op_definition
class VarDef(Operation):
    name = "ftn.dag.var_def"

    var= AttributeDef(AnyAttr())
    result = ResultDef(AnyAttr())
    is_proc_argument = AttributeDef(BoolAttr)
    is_constant = AttributeDef(BoolAttr)
    intent = AttributeDef(StringAttr)
    
@irdl_op_definition
class Assign(Operation):
    name = "ftn.dag.assign"

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
    name = "ftn.dag.literal"

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
    name = "ftn.dag.if"

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
    name = "ftn.dag.do"

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
    name = "ftn.dag.binary_expr"

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
    name = "ftn.dag.call_expr"

    func = AttributeDef(StringAttr)
    isstatement = AttributeDef(BoolAttr)
    args = SingleBlockRegionDef()
    bound_function_instance=SingleBlockRegionDef()
    bound_variables=AttributeDef(ArrayAttr)

    @staticmethod
    def get(func: str,
            args: List[Operation], isstatement,
            bound_variables=[],
            bound_function_instance=[],
            verify_op: bool = True) -> CallExpr:        
        res = CallExpr.build(regions=[args, bound_function_instance], attributes={"func": func, "isstatement": isstatement,
                                                                                  "bound_variables": bound_variables})
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

    def verify_(self) -> None:
      pass
                
@dataclass
class ftnDAG:
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
        return statements + FtnDAG.get_expression_op_types()

    @staticmethod
    def get_expression_op_types() -> List[Type[Operation]]:
        return [
            BinaryExpr, CallExpr, Literal, ExprName
        ]

    @staticmethod
    def get_type_op_types() -> List[Type[Operation]]:
        return []

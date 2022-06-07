from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Type, Union

from xdsl.dialects.builtin import IntegerAttr, StringAttr, IntegerType, Float32Type, i32, f32, ArrayAttr, ArrayOfConstraint, AnyAttr, BoolAttr
from xdsl.ir import Data, MLContext, Operation, ParametrizedAttribute
from xdsl.irdl import (AnyOf, AttributeDef, SingleBlockRegionDef, builder, ParameterDef,
                       irdl_attr_definition, irdl_op_definition)
from xdsl.parser import Parser
from xdsl.printer import Printer

@irdl_attr_definition
class DerivedType(ParametrizedAttribute):
    name = "derivedtype"
    
    type = ParameterDef(StringAttr)
    
    @staticmethod
    @builder
    def from_str(type: str) -> DerivedType:
        return DerivedType([StringAttr.from_str(type)])

    @staticmethod
    @builder
    def from_string_attr(data: StringAttr) -> DerivedType:
        return DerivedType([data])
      
@irdl_attr_definition
class AnonymousAttr(ParametrizedAttribute):
    name = "anonymous"
      
# Ideally would use vector, but need unknown dimension types (and ranges too!)
@irdl_attr_definition
class ArrayType(ParametrizedAttribute):
    name = "arraytype"

    shape = ParameterDef(ArrayAttr)
    element_type = ParameterDef(AnyOf([IntegerType, Float32Type, DerivedType]))

    def get_num_dims(self) -> int:
        return len(self.parameters[0].data)

    def get_shape(self) -> List[int]:
        return [i.parameters[0].data for i in self.shape.data]

    @staticmethod
    @builder
    def from_type_and_list(
            referenced_type: Attribute,
            shape: List[Union[int, IntegerAttr, AnonymousAttr]] = None) -> ArrayType:
        if shape is None:
            shape = [1]
        if (isinstance(referenced_type, str) and referenced_type in VarDef.TYPE_MAP_TO_PSY.keys()):
          type=VarDef.TYPE_MAP_TO_PSY[referenced_type]
        else:
          type=referenced_type
        return ArrayType([
            ArrayAttr.from_list([(IntegerAttr.build(d) if isinstance(d, int) else d) for d in shape]),
            type]
        )

    @staticmethod
    @builder
    def from_params(
        referenced_type: Attribute,
        shape: ArrayAttr) -> ArrayType:
        return ArrayType([shape, referenced_type])     
    

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
    imports = SingleBlockRegionDef()
    routines = SingleBlockRegionDef()

    @staticmethod
    def get(container_name: str,
            imports: List[Operation],
            routines: List[Operation],
            verify_op: bool = True) -> Container:
      res = Container.build(attributes={"container_name": container_name}, regions=[imports, routines])
      if verify_op:
        res.verify(verify_nested_ops=False)
      return res

    def verify_(self) -> None:
      pass
    
@irdl_op_definition
class Import(Operation):
    name = "psy.ast.import"
    
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
    name = "psy.ast.routine"

    routine_name = AttributeDef(StringAttr)
    imports = SingleBlockRegionDef()
    args = AttributeDef(ArrayAttr)
    return_type = AttributeDef(StringAttr)
    
    local_var_declarations = SingleBlockRegionDef()
    routine_body = SingleBlockRegionDef()

    @staticmethod
    def get(routine_name: Union[str, StringAttr],
            return_type: str,
            imports: List[Operation],
            args: List[Operation],            
            local_var_declarations: List[Operation],
            routine_body: List[Operation],
            verify_op: bool = True) -> Routine:
        res = Routine.build(attributes={"routine_name": routine_name, "return_type": return_type, "args": args},
                            regions=[imports, local_var_declarations, routine_body])
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
class ArrayAccess(Operation):
    name="psy.ast.array_access_expr"
    
    var_name = AttributeDef(StringAttr)
    accessors = SingleBlockRegionDef()
    
    @staticmethod
    def get(var_name: StringAttr, 
            accessors: List[Operation], 
            verify_op: bool = True) -> ExprName:
        res = ArrayAccess.build(attributes={"var_name": var_name}, regions=[accessors])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

    
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
class MemberAccess(Operation):
    name = "psy.ast.member_access_expr"    

    id = AttributeDef(StringAttr)
    member = SingleBlockRegionDef()
    
    @staticmethod
    def get(name: Union[str, StringAttr], member: Union[str, StringAttr], verify_op: bool = True) -> ExprName:
        res = MemberAccess.build(attributes={"id": name}, regions=[[member]])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res
            
@irdl_op_definition
class VarDef(Operation):
    name = "psy.ast.var_def"
    
    TYPE_MAP_TO_PSY = {"int": i32,
                       "float": f32}

    type = AttributeDef(AnyOf([IntegerType, Float32Type, DerivedType, ArrayType]))
    var_name = AttributeDef(StringAttr)    
    is_proc_argument = AttributeDef(BoolAttr)
    is_constant = AttributeDef(BoolAttr)
    intent = AttributeDef(StringAttr)

    @staticmethod
    def get(typed_var: str,
            var_name: str,
            is_proc_argument: bool = False,
            is_constant: bool = False,
            intent: str = "",
            verify_op: bool = True) -> VarDef:    
        #TODO: This is a bit nasty how we feed in both string and IR nodes, with arrays will be hard to fix though?    
        if (isinstance(typed_var, str) and typed_var in VarDef.TYPE_MAP_TO_PSY.keys()):
          type=VarDef.TYPE_MAP_TO_PSY[typed_var]
        else:
          type=typed_var
        res = VarDef.build(attributes={"var_name": var_name, "type": type, "is_proc_argument": is_proc_argument, "is_constant": is_constant, "intent": intent})
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
        self.ctx.register_attr(BoolAttr)
        self.ctx.register_attr(AnonymousAttr)        
        self.ctx.register_attr(DerivedType)
        self.ctx.register_attr(ArrayType)
        
        self.ctx.register_op(FileContainer)
        self.ctx.register_op(Container)
        self.ctx.register_op(Routine)
        self.ctx.register_op(Import)
        self.ctx.register_op(VarDef)        
        self.ctx.register_op(Assign)
        self.ctx.register_op(If)
        self.ctx.register_op(Do)
        self.ctx.register_op(Literal)
        self.ctx.register_op(ExprName)
        self.ctx.register_op(ArrayAccess)
        self.ctx.register_op(MemberAccess)
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
            BinaryExpr, CallExpr, Literal, ExprName, MemberAccess, ArrayAccess
        ]

    @staticmethod
    def get_type_op_types() -> List[Type[Operation]]:
        return []

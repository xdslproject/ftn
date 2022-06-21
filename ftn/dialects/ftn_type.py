from __future__ import annotations

from xdsl.dialects.builtin import StringAttr, ArrayAttr, IntAttr
from xdsl.ir import ParametrizedAttribute
from xdsl.irdl import AnyAttr, ParameterDef, irdl_attr_definition, AnyOf


@irdl_attr_definition
class NamedType(ParametrizedAttribute):
    name = "ftn.ir.named_type"

    type_name = ParameterDef(StringAttr)
    kind = ParameterDef(StringAttr)
    precision = ParameterDef(IntAttr)
    
@irdl_attr_definition
class DerivedType(ParametrizedAttribute):
    name = "derivedtype"
    
    type = ParameterDef(StringAttr)
    
@irdl_attr_definition
class ArrayType(ParametrizedAttribute):
    name = "arraytype"
    
    shape = ParameterDef(ArrayAttr)
    element_type = ParameterDef(AnyOf([NamedType, DerivedType]))
    
int_type=NamedType([StringAttr("integer"), StringAttr(""), IntAttr(4)])
float_type=NamedType([StringAttr("real"), StringAttr(""), IntAttr(4)])


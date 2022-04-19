from __future__ import annotations

from xdsl.dialects.builtin import StringAttr
from xdsl.ir import ParametrizedAttribute
from xdsl.irdl import AnyAttr, ParameterDef, irdl_attr_definition


@irdl_attr_definition
class NamedType(ParametrizedAttribute):
    name = "psy.ir.named_type"

    type_name = ParameterDef(StringAttr)


int_type = NamedType([StringAttr.from_str("int")])
float_type = NamedType([StringAttr.from_str("float")])
bool_type = NamedType([StringAttr.from_str("bool")])
str_type = NamedType([StringAttr.from_str("str")])
char_type = NamedType([StringAttr.from_str("char")])
none_type = NamedType([StringAttr.from_str("<None>")])
empty_type = NamedType([StringAttr.from_str("<Empty>")])
object_type = NamedType([StringAttr.from_str("object")])

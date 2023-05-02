from dataclasses import dataclass
from typing import TypeAlias, List, cast, Type, Sequence, Optional
from xdsl.ir import MLContext, ParametrizedAttribute, TypeAttribute, Dialect
from xdsl.irdl import (Operand, OpResult, AnyAttr, ParameterDef, AnyOf, Annotated, OptOpResult, VarOperand, ParameterDef, OptOperand,
                       VarRegion, irdl_op_definition, irdl_attr_definition, OpAttr, OptOpAttr, VarOpResult, Attribute, AttrSizedOperandSegments,
                       IRDLOperation)
from xdsl.dialects.builtin import (StringAttr, IntegerType, Float16Type, Float32Type, Float64Type, ArrayAttr, UnitAttr, IntAttr,
                                    DenseIntOrFPElementsAttr, AnyIntegerAttr, IntegerAttr, IndexType, SymbolRefAttr, TupleType)
from xdsl.printer import Printer

@irdl_attr_definition
class ReferenceType(ParametrizedAttribute, TypeAttribute):
      name = "fir.ref"
      type: ParameterDef[AnyAttr()]

      def print_parameters(self, printer: Printer) -> None:
        # We need this to pretty print a tuple and its members if
        # this is referencing one, otherwise just let the type
        # handle its own printing
        printer.print("<")
        if isinstance(self.type, TupleType):
          printer.print("tuple<")
          for idx, t in enumerate(self.type.types.data):
            if idx > 0: printer.print(", ")
            printer.print(t)
          printer.print(">")
        else:
          printer.print(self.type)
        printer.print(">")

@irdl_attr_definition
class DeferredAttr(ParametrizedAttribute, TypeAttribute):
  name = "fir.deferred"

  def print_parameters(self, printer: Printer) -> None:
      printer.print_string("?")

@irdl_attr_definition
class LLVMPointerType(ParametrizedAttribute, TypeAttribute):
  name = "fir.llvm_ptr"

  type: ParameterDef[AnyOf([IntegerType, Float16Type, Float32Type, Float64Type])]

@irdl_attr_definition
class NoneType(ParametrizedAttribute, TypeAttribute):
  name = "fir.none"

@irdl_attr_definition
class ArrayType(ParametrizedAttribute, TypeAttribute):
    name = "fir.array"
    shape: ParameterDef[ArrayAttr[AnyIntegerAttr | DeferredAttr]]
    type: ParameterDef[AnyOf([IntegerType, Float16Type, Float32Type, Float64Type, ReferenceType])]
    type2: ParameterDef[AnyOf([IntegerType, Float16Type, Float32Type, Float64Type, ReferenceType, NoneType])]

    @staticmethod
    def from_type_and_list(
        referenced_type: ParameterDef,
        shape: Optional[List[int | IntegerAttr[IndexType] | DeferredAttr]] = None):
        if shape is None:
            shape = [1]
        return ArrayType([
            ArrayAttr(
                [(IntegerAttr[IntegerType].from_params(d, 32) if isinstance(d, int) else d) for d in shape]),
            referenced_type, NoneType()
        ])

    @staticmethod
    def from_two_type(type: ParameterDef, type2: ParameterDef):
      return ArrayType([
            ArrayAttr([]),
            type, type2
        ])

    def print_parameters(self, printer: Printer) -> None:
      printer.print("<")
      if isinstance(self.type2, NoneType):
        for s in self.shape.data:
          if isinstance(s, DeferredAttr):
            printer.print_string("?")
          else:
            printer.print_string(f"{s.value.data}")
          printer.print_string("x")
        printer.print(self.type)
      else:
        printer.print_string("0xtuple<")
        printer.print(self.type)
        printer.print_string(", ")
        printer.print(self.type2)
        printer.print_string(">")
      printer.print(">")

    def hasDeferredShape(self):
      for s in self.shape.data:
        if isinstance(s, DeferredAttr): return True
      return False

    def getNumberDims(self):
      return len(self.shape.data)

@irdl_attr_definition
class CharType(ParametrizedAttribute, TypeAttribute):
  name = "fir.char"

  from_index: ParameterDef[IntAttr | DeferredAttr]
  to_index: ParameterDef[IntAttr | DeferredAttr]

  def print_parameters(self, printer: Printer) -> None:
      printer.print("<")
      if isinstance(self.from_index, DeferredAttr):
        printer.print_string("?")
      else:
        printer.print_string(f"{self.from_index.data}")

      printer.print_string(",")

      if isinstance(self.to_index, DeferredAttr):
        printer.print_string("?")
      else:
        printer.print_string(f"{self.to_index.data}")
      printer.print(">")


@irdl_attr_definition
class ShapeType(ParametrizedAttribute, TypeAttribute):
  name = "fir.shape"

  indexes: ParameterDef[IntAttr]

  def print_parameters(self, printer: Printer) -> None:
      printer.print("<")
      printer.print_string(f"{self.indexes.data}")
      printer.print(">")

@irdl_attr_definition
class HeapType(ParametrizedAttribute, TypeAttribute):
  name = "fir.heap"

  type: ParameterDef[ArrayType]

@irdl_attr_definition
class BoxType(ParametrizedAttribute, TypeAttribute):
  name = "fir.box"

  type: ParameterDef[HeapType | ArrayType]

@irdl_attr_definition
class BoxCharType(ParametrizedAttribute, TypeAttribute):
  name = "fir.boxchar"

  kind: ParameterDef[IntAttr]

  def print_parameters(self, printer: Printer) -> None:
      printer.print("<")
      printer.print_string(f"{self.kind.data}")
      printer.print(">")

@irdl_op_definition
class Absent(IRDLOperation):
     name =  "fir.absent"
     intype: Annotated[OpResult,AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Addc(IRDLOperation):
     name =  "fir.addc"
     lhs : Annotated[Operand, AnyAttr()]
     rhs : Annotated[Operand, AnyAttr()]
     result: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class AddressOf(IRDLOperation):
     name =  "fir.address_of"
     symbol: OpAttr[SymbolRefAttr]
     resTy: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Allocmem(IRDLOperation):
     name =  "fir.allocmem"
     in_type: OpAttr[AnyAttr()]
     uniq_name: OptOpAttr[StringAttr]
     typeparams : Annotated[VarOperand, AnyAttr()]
     shape: Annotated[VarOperand, AnyAttr()]

     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion

     irdl_options = [AttrSizedOperandSegments()]


@irdl_op_definition
class Alloca(IRDLOperation):
     name =  "fir.alloca"
     in_type: OpAttr[AnyAttr()]
     uniq_name: OptOpAttr[StringAttr]
     bindc_name: OptOpAttr[StringAttr]
     typeparams : Annotated[VarOperand, AnyAttr()]
     shape : Annotated[VarOperand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion
     valuebyref: OptOpAttr[UnitAttr]

     irdl_options = [AttrSizedOperandSegments()]


@irdl_op_definition
class ArrayAccess(IRDLOperation):
     name =  "fir.array_access"
     sequence: Annotated[Operand, AnyAttr()]
     indices: Annotated[Operand, AnyAttr()]
     typeparams: Annotated[Operand, AnyAttr()]
     element: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class ArrayAmend(IRDLOperation):
     name =  "fir.array_amend"
     sequence: Annotated[Operand, AnyAttr()]
     memref: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class ArrayCoor(IRDLOperation):
     name =  "fir.array_coor"
     memref : Annotated[Operand, AnyAttr()]
     shape : Annotated[Operand, AnyAttr()]
     slice : Annotated[Operand, AnyAttr()]
     indices : Annotated[Operand, AnyAttr()]
     typeparams : Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs : VarRegion


@irdl_op_definition
class ArrayFetch(IRDLOperation):
     name =  "fir.array_fetch"
     sequence: Annotated[Operand, AnyAttr()]
     indices: Annotated[Operand, AnyAttr()]
     typeparams: Annotated[Operand, AnyAttr()]
     element: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class ArrayLoad(IRDLOperation):
     name =  "fir.array_load"
     memref: Annotated[Operand, AnyAttr()]
     shape: Annotated[Operand, AnyAttr()]
     slice: Annotated[Operand, AnyAttr()]
     typeparams: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class ArrayMergeStore(IRDLOperation):
     name =  "fir.array_merge_store"
     original: Annotated[Operand, AnyAttr()]
     sequence: Annotated[Operand, AnyAttr()]
     memref: Annotated[Operand, AnyAttr()]
     slice: Annotated[Operand, AnyAttr()]
     typeparams: Annotated[Operand, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class ArrayModify(IRDLOperation):
     name =  "fir.array_modify"
     sequence: Annotated[Operand, AnyAttr()]
     indices: Annotated[Operand, AnyAttr()]
     typeparams: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     result_1: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class ArrayUpdate(IRDLOperation):
     name =  "fir.array_update"
     sequence: Annotated[Operand, AnyAttr()]
     merge: Annotated[Operand, AnyAttr()]
     indices: Annotated[Operand, AnyAttr()]
     typeparams: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class BoxAddr(IRDLOperation):
     name =  "fir.box_addr"
     val: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class BoxcharLen(IRDLOperation):
     name =  "fir.boxchar_len"
     val: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class BoxDims(IRDLOperation):
     name =  "fir.box_dims"
     val: Annotated[Operand, AnyAttr()]
     dim: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     result_1: Annotated[OpResult, AnyAttr()]
     result_2: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class BoxElesize(IRDLOperation):
     name =  "fir.box_elesize"
     val: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class BoxIsalloc(IRDLOperation):
     name =  "fir.box_isalloc"
     val: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class BoxIsarray(IRDLOperation):
     name =  "fir.box_isarray"
     val: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class BoxIsptr(IRDLOperation):
     name =  "fir.box_isptr"
     val: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class BoxprocHost(IRDLOperation):
     name =  "fir.boxproc_host"
     val: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class BoxRank(IRDLOperation):
     name =  "fir.box_rank"
     val: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class BoxTdesc(IRDLOperation):
     name =  "fir.box_tdesc"
     val: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Call(IRDLOperation):
     name =  "fir.call"
     callee: OpAttr[AnyAttr()]
     result_0: Annotated[OptOpResult, AnyAttr()]
     args: Annotated[VarOperand, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class CharConvert(IRDLOperation):
     name =  "fir.char_convert"
     _from: Annotated[Operand, AnyAttr()]
     count: Annotated[Operand, AnyAttr()]
     to: Annotated[Operand, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Cmpc(IRDLOperation):
     name =  "fir.cmpc"
     lhs: Annotated[Operand, AnyAttr()]
     rhs: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Constc(IRDLOperation):
     name =  "fir.constc"
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Convert(IRDLOperation):
     name =  "fir.convert"
     value: Annotated[Operand, AnyAttr()]
     res: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class CoordinateOf(IRDLOperation):
     name =  "fir.coordinate_of"
     baseType: OpAttr[AnyAttr()]
     ref: Annotated[Operand, AnyAttr()]
     coor: Annotated[VarOperand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class DtEntry(IRDLOperation):
     name =  "fir.dt_entry"
     regs: VarRegion


@irdl_op_definition
class Dispatch(IRDLOperation):
     name =  "fir.dispatch"
     object: Annotated[Operand, AnyAttr()]
     args: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class DispatchTable(IRDLOperation):
     name =  "fir.dispatch_table"
     regs: VarRegion


@irdl_op_definition
class Divc(IRDLOperation):
     name =  "fir.divc"
     lhs: Annotated[Operand, AnyAttr()]
     rhs: Annotated[Operand, AnyAttr()]
     result: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class DoLoop(IRDLOperation):
     name =  "fir.do_loop"
     lowerBound: Annotated[Operand, AnyAttr()]
     upperBound: Annotated[Operand, AnyAttr()]
     step: Annotated[Operand, AnyAttr()]
     finalValue: OptOpAttr[Attribute]
     initArgs: Annotated[Operand, AnyAttr()]
     _results: Annotated[VarOpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Emboxchar(IRDLOperation):
     name =  "fir.emboxchar"
     memref: Annotated[Operand, AnyAttr()]
     len: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Embox(IRDLOperation):
     name =  "fir.embox"
     memref: Annotated[Operand, AnyAttr()]
     shape: Annotated[Operand, AnyAttr()]
     slice: Annotated[VarOperand, AnyAttr()]
     typeparams: Annotated[VarOperand, AnyAttr()]
     sourceBox: Annotated[VarOperand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion

     irdl_options = [AttrSizedOperandSegments()]


@irdl_op_definition
class Emboxproc(IRDLOperation):
     name =  "fir.emboxproc"
     func: Annotated[Operand, AnyAttr()]
     host: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class ExtractValue(IRDLOperation):
     name =  "fir.extract_value"
     adt: Annotated[Operand, AnyAttr()]
     res: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class FieldIndex(IRDLOperation):
     name =  "fir.field_index"
     typeparams: Annotated[Operand, AnyAttr()]
     res: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class End(IRDLOperation):
     name =  "fir.end"
     regs: VarRegion


@irdl_op_definition
class Freemem(IRDLOperation):
     name =  "fir.freemem"
     heapref: Annotated[Operand, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Gentypedesc(IRDLOperation):
     name =  "fir.gentypedesc"
     res: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class GlobalLen(IRDLOperation):
     name =  "fir.global_len"
     regs: VarRegion


@irdl_op_definition
class Global(IRDLOperation):
     name =  "fir.global"
     regs : VarRegion
     sym_name: OpAttr[StringAttr]
     symref: OpAttr[SymbolRefAttr]
     type: OpAttr[AnyOf([IntegerType, Float16Type, Float32Type, Float64Type, ArrayType, BoxType, CharType, ReferenceType])]
     linkName: OptOpAttr[StringAttr]
     constant: OptOpAttr[UnitAttr]


@irdl_op_definition
class HasValue(IRDLOperation):
     name =  "fir.has_value"
     resval: Annotated[Operand, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class If(IRDLOperation):
     name =  "fir.if"
     condition: Annotated[Operand, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class InsertOnRange(IRDLOperation):
     name =  "fir.insert_on_range"
     seq: Annotated[Operand, AnyAttr()]
     val: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class InsertValue(IRDLOperation):
     name =  "fir.insert_value"
     adt: Annotated[Operand, AnyAttr()]
     val: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class IsPresent(IRDLOperation):
     name =  "fir.is_present"
     val: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class IterateWhile(IRDLOperation):
     name =  "fir.iterate_while"
     lowerBound: Annotated[Operand, AnyAttr()]
     upperBound: Annotated[Operand, AnyAttr()]
     step: Annotated[Operand, AnyAttr()]
     iterateIn: Annotated[Operand, AnyAttr()]
     initArgs: Annotated[Operand, AnyAttr()]
     _results: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class LenParamIndex(IRDLOperation):
     name =  "fir.len_param_index"
     typeparams: Annotated[Operand, AnyAttr()]
     res: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Load(IRDLOperation):
     name =  "fir.load"
     memref: Annotated[Operand, AnyAttr()]
     res: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Mulc(IRDLOperation):
     name =  "fir.mulc"
     lhs: Annotated[Operand, AnyAttr()]
     rhs: Annotated[Operand, AnyAttr()]
     result: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Negc(IRDLOperation):
     name =  "fir.negc"
     operand: Annotated[Operand, AnyAttr()]
     result: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class NoReassoc(IRDLOperation):
     name =  "fir.no_reassoc"
     val: Annotated[Operand, AnyAttr()]
     res: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Rebox(IRDLOperation):
     name =  "fir.rebox"
     box: Annotated[Operand, AnyAttr()]
     shape: Annotated[Operand, AnyAttr()]
     slice: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Result(IRDLOperation):
  name =  "fir.result"
  regs : VarRegion
  _results: Annotated[OptOperand, AnyAttr()]


@irdl_op_definition
class SaveResult(IRDLOperation):
     name =  "fir.save_result"
     value: Annotated[Operand, AnyAttr()]
     memref: Annotated[Operand, AnyAttr()]
     shape: Annotated[Operand, AnyAttr()]
     typeparams: Annotated[Operand, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class SelectCase(IRDLOperation):
     name =  "fir.select_case"
     selector: Annotated[Operand, AnyAttr()]
     compareArgs: Annotated[Operand, AnyAttr()]
     targetArgs: Annotated[Operand, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Select(IRDLOperation):
     name =  "fir.select"
     selector: Annotated[Operand, AnyAttr()]
     compareArgs: Annotated[Operand, AnyAttr()]
     targetArgs: Annotated[Operand, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class SelectRank(IRDLOperation):
     name =  "fir.select_rank"
     selector: Annotated[Operand, AnyAttr()]
     compareArgs: Annotated[Operand, AnyAttr()]
     targetArgs: Annotated[Operand, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class SelectType(IRDLOperation):
     name =  "fir.select_type"
     selector: Annotated[Operand, AnyAttr()]
     compareArgs: Annotated[Operand, AnyAttr()]
     targetArgs: Annotated[Operand, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Shape(IRDLOperation):
     name =  "fir.shape"
     extents: Annotated[VarOperand,AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     #regs : VarRegion


@irdl_op_definition
class ShapeShift(IRDLOperation):
     name =  "fir.shape_shift"
     pairs: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Shift(IRDLOperation):
     name =  "fir.shift"
     origins: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Slice(IRDLOperation):
     name =  "fir.slice"
     triples: Annotated[Operand, AnyAttr()]
     fields: Annotated[Operand, AnyAttr()]
     substr: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Store(IRDLOperation):
     name =  "fir.store"
     value: Annotated[Operand, AnyAttr()]
     memref: Annotated[Operand, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class StringLit(IRDLOperation):
     name =  "fir.string_lit"
     size: OpAttr[IntegerAttr]
     value: OpAttr[StringAttr]
     result_0: Annotated[OpResult, AnyAttr()]


@irdl_op_definition
class Subc(IRDLOperation):
     name =  "fir.subc"
     lhs: Annotated[Operand, AnyAttr()]
     rhs: Annotated[Operand, AnyAttr()]
     result: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Unboxchar(IRDLOperation):
     name =  "fir.unboxchar"
     boxchar: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     result_1: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Unboxproc(IRDLOperation):
     name =  "fir.unboxproc"
     boxproc: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     refTuple: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Undefined(IRDLOperation):
     name =  "fir.undefined"
     intype: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Unreachable(IRDLOperation):
     name =  "fir.unreachable"
     regs: VarRegion


@irdl_op_definition
class ZeroBits(IRDLOperation):
     name =  "fir.zero_bits"
     intype: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


FIR= Dialect([
    Absent,
    Addc,
    AddressOf,
    Allocmem,
    Alloca,
    ArrayAccess,
    ArrayAmend,
    ArrayCoor,
    ArrayFetch,
    ArrayLoad,
    ArrayMergeStore,
    ArrayModify,
    ArrayUpdate,
    BoxAddr,
    BoxcharLen,
    BoxDims,
    BoxElesize,
    BoxIsalloc,
    BoxIsarray,
    BoxIsptr,
    BoxprocHost,
    BoxRank,
    BoxTdesc,
    Call,
    CharConvert,
    Cmpc,
    Constc,
    Convert,
    CoordinateOf,
    DtEntry,
    Dispatch,
    DispatchTable,
    Divc,
    DoLoop,
    Emboxchar,
    Embox,
    Emboxproc,
    ExtractValue,
    FieldIndex,
    End,
    Freemem,
    Gentypedesc,
    GlobalLen,
    Global,
    HasValue,
    If,
    InsertOnRange,
    InsertValue,
    IsPresent,
    IterateWhile,
    LenParamIndex,
    Load,
    Mulc,
    Negc,
    NoReassoc,
    Rebox,
    Result,
    SaveResult,
    SelectCase,
    Select,
    SelectRank,
    SelectType,
    Shape,
    ShapeShift,
    Shift,
    Slice,
    Store,
    StringLit,
    Subc,
    Unboxchar,
    Unboxproc,
    Undefined,
    Unreachable,
    ZeroBits,
], [
    ReferenceType,
    DeferredAttr,
    LLVMPointerType,
    NoneType,
    ArrayType,
    CharType,
    ShapeType,
    HeapType,
    BoxType,
    BoxCharType,
])

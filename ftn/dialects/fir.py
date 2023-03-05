from dataclasses import dataclass
from typing import TypeAlias, List, cast, Type, Sequence, Optional
from xdsl.ir import Operation, MLContext, ParametrizedAttribute, MLIRType
from xdsl.irdl import (Operand, OpResult, AnyAttr, ParameterDef, AnyOf, Annotated, OptOpResult, VarOperand, ParameterDef, OptOperand,
                       VarRegion, irdl_op_definition, irdl_attr_definition, OpAttr, OptOpAttr, VarOpResult, Attribute, AttrSizedOperandSegments)
from xdsl.dialects.builtin import (StringAttr, IntegerType, Float16Type, Float32Type, Float64Type, ArrayAttr, UnitAttr, IntAttr,
                                    DenseIntOrFPElementsAttr, AnyIntegerAttr, IntegerAttr, IndexType, SymbolRefAttr)
from xdsl.printer import Printer

@dataclass
class Fir:
     ctx: MLContext

     def __post_init__(self):
         self.ctx.register_op(Absent)
         self.ctx.register_op(Addc)
         self.ctx.register_op(AddressOf)
         self.ctx.register_op(Allocmem)
         self.ctx.register_op(Alloca)
         self.ctx.register_op(ArrayAccess)
         self.ctx.register_op(ArrayAmend)
         self.ctx.register_op(ArrayCoor)
         self.ctx.register_op(ArrayFetch)
         self.ctx.register_op(ArrayLoad)
         self.ctx.register_op(ArrayMergeStore)
         self.ctx.register_op(ArrayModify)
         self.ctx.register_op(ArrayUpdate)
         self.ctx.register_op(BoxAddr)
         self.ctx.register_op(BoxcharLen)
         self.ctx.register_op(BoxDims)
         self.ctx.register_op(BoxElesize)
         self.ctx.register_op(BoxIsalloc)
         self.ctx.register_op(BoxIsarray)
         self.ctx.register_op(BoxIsptr)
         self.ctx.register_op(BoxprocHost)
         self.ctx.register_op(BoxRank)
         self.ctx.register_op(BoxTdesc)
         self.ctx.register_op(Call)
         self.ctx.register_op(CharConvert)
         self.ctx.register_op(Cmpc)
         self.ctx.register_op(Constc)
         self.ctx.register_op(Convert)
         self.ctx.register_op(CoordinateOf)
         self.ctx.register_op(DtEntry)
         self.ctx.register_op(Dispatch)
         self.ctx.register_op(DispatchTable)
         self.ctx.register_op(Divc)
         self.ctx.register_op(DoLoop)
         self.ctx.register_op(Emboxchar)
         self.ctx.register_op(Embox)
         self.ctx.register_op(Emboxproc)
         self.ctx.register_op(ExtractValue)
         self.ctx.register_op(FieldIndex)
         self.ctx.register_op(End)
         self.ctx.register_op(Freemem)
         self.ctx.register_op(Gentypedesc)
         self.ctx.register_op(GlobalLen)
         self.ctx.register_op(Global)
         self.ctx.register_op(HasValue)
         self.ctx.register_op(If)
         self.ctx.register_op(InsertOnRange)
         self.ctx.register_op(InsertValue)
         self.ctx.register_op(IsPresent)
         self.ctx.register_op(IterateWhile)
         self.ctx.register_op(LenParamIndex)
         self.ctx.register_op(Load)
         self.ctx.register_op(Mulc)
         self.ctx.register_op(Negc)
         self.ctx.register_op(NoReassoc)
         self.ctx.register_op(Rebox)
         self.ctx.register_op(Result)
         self.ctx.register_op(SaveResult)
         self.ctx.register_op(SelectCase)
         self.ctx.register_op(Select)
         self.ctx.register_op(SelectRank)
         self.ctx.register_op(SelectType)
         self.ctx.register_op(Shape)
         self.ctx.register_op(ShapeShift)
         self.ctx.register_op(Shift)
         self.ctx.register_op(Slice)
         self.ctx.register_op(Store)
         self.ctx.register_op(StringLit)
         self.ctx.register_op(Subc)
         self.ctx.register_op(Unboxchar)
         self.ctx.register_op(Unboxproc)
         self.ctx.register_op(Undefined)
         self.ctx.register_op(Unreachable)
         self.ctx.register_op(ZeroBits)

@irdl_attr_definition
class DeferredAttr(ParametrizedAttribute, MLIRType):
  name = "fir.deferred"

  @staticmethod
  def print_parameter(data: str, printer: Printer) -> None:
      printer.print_string("?")

@irdl_attr_definition
class LLVMPointerType(ParametrizedAttribute, MLIRType):
  name = "fir.llvm_ptr"

  type: ParameterDef[AnyOf([IntegerType, Float16Type, Float32Type, Float64Type])]


@irdl_attr_definition
class ArrayType(ParametrizedAttribute, MLIRType):
    name = "fir.array"
    shape: ParameterDef[ArrayAttr[AnyIntegerAttr | DeferredAttr]]
    type: ParameterDef[AnyOf([IntegerType, Float16Type, Float32Type, Float64Type])]

    @staticmethod
    def from_type_and_list(
        referenced_type: ParameterDef,
        shape: Optional[List[int | IntegerAttr[IndexType] | DeferredAttr]] = None):
        if shape is None:
            shape = [1]
        return ArrayType([
            ArrayAttr.from_list(
                [(IntegerAttr[IntegerType].from_params(d, 32) if isinstance(d, int) else d) for d in shape]),
            referenced_type
        ])

@irdl_attr_definition
class CharType(ParametrizedAttribute, MLIRType):
  name = "fir.char"

  from_index: ParameterDef[IntAttr]
  to_index: ParameterDef[IntAttr]

  def print_parameters(self, printer: Printer) -> None:
      printer.print("<")
      printer.print_string(f"{self.from_index.data},{self.to_index.data}")
      printer.print(">")


@irdl_attr_definition
class ShapeType(ParametrizedAttribute, MLIRType):
  name = "fir.shape"

  indexes: ParameterDef[IntAttr]

@irdl_attr_definition
class HeapType(ParametrizedAttribute, MLIRType):
  name = "fir.heap"

  type: ParameterDef[ArrayType]

@irdl_attr_definition
class BoxType(ParametrizedAttribute, MLIRType):
  name = "fir.box"

  type: ParameterDef[HeapType]

@irdl_attr_definition
class ReferenceType(ParametrizedAttribute, MLIRType):
      name = "fir.ref"
      type: ParameterDef[AnyOf([IntegerType, Float16Type, Float32Type, Float64Type, ArrayType, BoxType, CharType])]

@irdl_op_definition
class Absent(Operation):
     name =  "fir.absent"
     intype: Annotated[OpResult,AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Addc(Operation):
     name =  "fir.addc"
     lhs : Annotated[Operand, AnyAttr()]
     rhs : Annotated[Operand, AnyAttr()]
     result: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class AddressOf(Operation):
     name =  "fir.address_of"
     symbol: OpAttr[SymbolRefAttr]
     resTy: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Allocmem(Operation):
     name =  "fir.allocmem"
     in_type: OpAttr[AnyAttr()]
     shape: Annotated[VarOperand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Alloca(Operation):
     name =  "fir.alloca"
     in_type: Annotated[VarOperand, AnyAttr()]
     uniq_name: OptOpAttr[StringAttr]
     bindc_name: OptOpAttr[StringAttr]
     #operand_segment_sizes: OpAttr[ArrayAttr]
     # needs boolean of pinned
     #typeparams : Annotated[Operand, AnyAttr())
     #shape : Annotated[Operand, AnyAttr())
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion
     valuebyref: OptOpAttr[UnitAttr]

     #irdl_options = [AttrSizedOperandSegments()]


@irdl_op_definition
class ArrayAccess(Operation):
     name =  "fir.array_access"
     sequence: Annotated[Operand, AnyAttr()]
     indices: Annotated[Operand, AnyAttr()]
     typeparams: Annotated[Operand, AnyAttr()]
     element: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class ArrayAmend(Operation):
     name =  "fir.array_amend"
     sequence: Annotated[Operand, AnyAttr()]
     memref: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class ArrayCoor(Operation):
     name =  "fir.array_coor"
     memref : Annotated[Operand, AnyAttr()]
     shape : Annotated[Operand, AnyAttr()]
     slice : Annotated[Operand, AnyAttr()]
     indices : Annotated[Operand, AnyAttr()]
     typeparams : Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs : VarRegion


@irdl_op_definition
class ArrayFetch(Operation):
     name =  "fir.array_fetch"
     sequence: Annotated[Operand, AnyAttr()]
     indices: Annotated[Operand, AnyAttr()]
     typeparams: Annotated[Operand, AnyAttr()]
     element: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class ArrayLoad(Operation):
     name =  "fir.array_load"
     memref: Annotated[Operand, AnyAttr()]
     shape: Annotated[Operand, AnyAttr()]
     slice: Annotated[Operand, AnyAttr()]
     typeparams: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class ArrayMergeStore(Operation):
     name =  "fir.array_merge_store"
     original: Annotated[Operand, AnyAttr()]
     sequence: Annotated[Operand, AnyAttr()]
     memref: Annotated[Operand, AnyAttr()]
     slice: Annotated[Operand, AnyAttr()]
     typeparams: Annotated[Operand, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class ArrayModify(Operation):
     name =  "fir.array_modify"
     sequence: Annotated[Operand, AnyAttr()]
     indices: Annotated[Operand, AnyAttr()]
     typeparams: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     result_1: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class ArrayUpdate(Operation):
     name =  "fir.array_update"
     sequence: Annotated[Operand, AnyAttr()]
     merge: Annotated[Operand, AnyAttr()]
     indices: Annotated[Operand, AnyAttr()]
     typeparams: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class BoxAddr(Operation):
     name =  "fir.box_addr"
     val: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class BoxcharLen(Operation):
     name =  "fir.boxchar_len"
     val: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class BoxDims(Operation):
     name =  "fir.box_dims"
     val: Annotated[Operand, AnyAttr()]
     dim: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     result_1: Annotated[OpResult, AnyAttr()]
     result_2: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class BoxElesize(Operation):
     name =  "fir.box_elesize"
     val: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class BoxIsalloc(Operation):
     name =  "fir.box_isalloc"
     val: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class BoxIsarray(Operation):
     name =  "fir.box_isarray"
     val: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class BoxIsptr(Operation):
     name =  "fir.box_isptr"
     val: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class BoxprocHost(Operation):
     name =  "fir.boxproc_host"
     val: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class BoxRank(Operation):
     name =  "fir.box_rank"
     val: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class BoxTdesc(Operation):
     name =  "fir.box_tdesc"
     val: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Call(Operation):
     name =  "fir.call"
     callee: OpAttr[AnyAttr()]
     result_0: Annotated[OptOpResult, AnyAttr()]
     args: Annotated[VarOperand, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class CharConvert(Operation):
     name =  "fir.char_convert"
     _from: Annotated[Operand, AnyAttr()]
     count: Annotated[Operand, AnyAttr()]
     to: Annotated[Operand, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Cmpc(Operation):
     name =  "fir.cmpc"
     lhs: Annotated[Operand, AnyAttr()]
     rhs: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Constc(Operation):
     name =  "fir.constc"
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Convert(Operation):
     name =  "fir.convert"
     value: Annotated[Operand, AnyAttr()]
     res: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class CoordinateOf(Operation):
     name =  "fir.coordinate_of"
     baseType: OpAttr[AnyAttr()]
     ref: Annotated[Operand, AnyAttr()]
     coor: Annotated[VarOperand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class DtEntry(Operation):
     name =  "fir.dt_entry"
     regs: VarRegion


@irdl_op_definition
class Dispatch(Operation):
     name =  "fir.dispatch"
     object: Annotated[Operand, AnyAttr()]
     args: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class DispatchTable(Operation):
     name =  "fir.dispatch_table"
     regs: VarRegion


@irdl_op_definition
class Divc(Operation):
     name =  "fir.divc"
     lhs: Annotated[Operand, AnyAttr()]
     rhs: Annotated[Operand, AnyAttr()]
     result: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class DoLoop(Operation):
     name =  "fir.do_loop"
     lowerBound: Annotated[Operand, AnyAttr()]
     upperBound: Annotated[Operand, AnyAttr()]
     step: Annotated[Operand, AnyAttr()]
     finalValue: OptOpAttr[Attribute]
     initArgs: Annotated[Operand, AnyAttr()]
     _results: Annotated[VarOpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Emboxchar(Operation):
     name =  "fir.emboxchar"
     memref: Annotated[Operand, AnyAttr()]
     len: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Embox(Operation):
     name =  "fir.embox"
     memref: Annotated[Operand, AnyAttr()]
     shape: Annotated[Operand, AnyAttr()]
     #slice = OptOperandDef(AnyAttr())
     #typeparams = OptOperandDef(AnyAttr())
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Emboxproc(Operation):
     name =  "fir.emboxproc"
     func: Annotated[Operand, AnyAttr()]
     host: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class ExtractValue(Operation):
     name =  "fir.extract_value"
     adt: Annotated[Operand, AnyAttr()]
     res: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class FieldIndex(Operation):
     name =  "fir.field_index"
     typeparams: Annotated[Operand, AnyAttr()]
     res: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class End(Operation):
     name =  "fir.end"
     regs: VarRegion


@irdl_op_definition
class Freemem(Operation):
     name =  "fir.freemem"
     heapref: Annotated[Operand, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Gentypedesc(Operation):
     name =  "fir.gentypedesc"
     res: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class GlobalLen(Operation):
     name =  "fir.global_len"
     regs: VarRegion


@irdl_op_definition
class Global(Operation):
     name =  "fir.global"
     regs : VarRegion
     sym_name: OpAttr[StringAttr]
     symref: OpAttr[SymbolRefAttr]
     type: OpAttr[AnyOf([IntegerType, Float16Type, Float32Type, Float64Type, ArrayType, BoxType, CharType])]
     linkName: OpAttr[StringAttr]


@irdl_op_definition
class HasValue(Operation):
     name =  "fir.has_value"
     resval: Annotated[Operand, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class If(Operation):
     name =  "fir.if"
     condition: Annotated[Operand, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class InsertOnRange(Operation):
     name =  "fir.insert_on_range"
     seq: Annotated[Operand, AnyAttr()]
     val: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class InsertValue(Operation):
     name =  "fir.insert_value"
     adt: Annotated[Operand, AnyAttr()]
     val: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class IsPresent(Operation):
     name =  "fir.is_present"
     val: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class IterateWhile(Operation):
     name =  "fir.iterate_while"
     lowerBound: Annotated[Operand, AnyAttr()]
     upperBound: Annotated[Operand, AnyAttr()]
     step: Annotated[Operand, AnyAttr()]
     iterateIn: Annotated[Operand, AnyAttr()]
     initArgs: Annotated[Operand, AnyAttr()]
     _results: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class LenParamIndex(Operation):
     name =  "fir.len_param_index"
     typeparams: Annotated[Operand, AnyAttr()]
     res: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Load(Operation):
     name =  "fir.load"
     memref: Annotated[Operand, AnyAttr()]
     res: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Mulc(Operation):
     name =  "fir.mulc"
     lhs: Annotated[Operand, AnyAttr()]
     rhs: Annotated[Operand, AnyAttr()]
     result: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Negc(Operation):
     name =  "fir.negc"
     operand: Annotated[Operand, AnyAttr()]
     result: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class NoReassoc(Operation):
     name =  "fir.no_reassoc"
     val: Annotated[Operand, AnyAttr()]
     res: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Rebox(Operation):
     name =  "fir.rebox"
     box: Annotated[Operand, AnyAttr()]
     shape: Annotated[Operand, AnyAttr()]
     slice: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Result(Operation):
  name =  "fir.result"
  regs : VarRegion
  _results: Annotated[OptOperand, AnyAttr()]


@irdl_op_definition
class SaveResult(Operation):
     name =  "fir.save_result"
     value: Annotated[Operand, AnyAttr()]
     memref: Annotated[Operand, AnyAttr()]
     shape: Annotated[Operand, AnyAttr()]
     typeparams: Annotated[Operand, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class SelectCase(Operation):
     name =  "fir.select_case"
     selector: Annotated[Operand, AnyAttr()]
     compareArgs: Annotated[Operand, AnyAttr()]
     targetArgs: Annotated[Operand, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Select(Operation):
     name =  "fir.select"
     selector: Annotated[Operand, AnyAttr()]
     compareArgs: Annotated[Operand, AnyAttr()]
     targetArgs: Annotated[Operand, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class SelectRank(Operation):
     name =  "fir.select_rank"
     selector: Annotated[Operand, AnyAttr()]
     compareArgs: Annotated[Operand, AnyAttr()]
     targetArgs: Annotated[Operand, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class SelectType(Operation):
     name =  "fir.select_type"
     selector: Annotated[Operand, AnyAttr()]
     compareArgs: Annotated[Operand, AnyAttr()]
     targetArgs: Annotated[Operand, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Shape(Operation):
     name =  "fir.shape"
     extents: Annotated[VarOperand,AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     #regs : VarRegion


@irdl_op_definition
class ShapeShift(Operation):
     name =  "fir.shape_shift"
     pairs: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Shift(Operation):
     name =  "fir.shift"
     origins: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Slice(Operation):
     name =  "fir.slice"
     triples: Annotated[Operand, AnyAttr()]
     fields: Annotated[Operand, AnyAttr()]
     substr: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Store(Operation):
     name =  "fir.store"
     value: Annotated[Operand, AnyAttr()]
     memref: Annotated[Operand, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class StringLit(Operation):
     name =  "fir.string_lit"
     size: OpAttr[IntegerAttr]
     value: OpAttr[StringAttr]
     result_0: Annotated[OpResult, AnyAttr()]


@irdl_op_definition
class Subc(Operation):
     name =  "fir.subc"
     lhs: Annotated[Operand, AnyAttr()]
     rhs: Annotated[Operand, AnyAttr()]
     result: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Unboxchar(Operation):
     name =  "fir.unboxchar"
     boxchar: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     result_1: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Unboxproc(Operation):
     name =  "fir.unboxproc"
     boxproc: Annotated[Operand, AnyAttr()]
     result_0: Annotated[OpResult, AnyAttr()]
     refTuple: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Undefined(Operation):
     name =  "fir.undefined"
     intype: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


@irdl_op_definition
class Unreachable(Operation):
     name =  "fir.unreachable"
     regs: VarRegion


@irdl_op_definition
class ZeroBits(Operation):
     name =  "fir.zero_bits"
     intype: Annotated[OpResult, AnyAttr()]
     regs: VarRegion


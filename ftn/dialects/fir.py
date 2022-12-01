from dataclasses import dataclass
from typing import TypeAlias, List, cast, Type, Sequence, Optional
from xdsl.ir import Operation, MLContext, ParametrizedAttribute, MLIRType
from xdsl.irdl import (OperandDef, ResultDef, AnyAttr, AttributeDef, ParameterDef, AnyOf, OptOperandDef, VarResultDef, VarOperandDef, VarOperandDef,
                       VarRegionDef, irdl_op_definition, irdl_attr_definition, OptResultDef, OptAttributeDef, builder)
from xdsl.dialects.builtin import (StringAttr, IntegerType, Float16Type, Float32Type, Float64Type, ArrayAttr, UnitAttr,
                                    DenseIntOrFPElementsAttr, AnyIntegerAttr, IntegerAttr, IndexType, FlatSymbolRefAttr)


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
class ArrayType(ParametrizedAttribute, MLIRType):
    name = "fir.array"
    shape: ParameterDef[ArrayAttr[AnyIntegerAttr]]
    type: ParameterDef[AnyOf([IntegerType, Float16Type, Float32Type, Float64Type])]

    @staticmethod
    @builder
    def from_type_and_list(
        referenced_type: ParameterDef,
        shape: Optional[List[int | IntegerAttr[IndexType]]] = None):
        if shape is None:
            shape = [1]
        return ArrayType([
            ArrayAttr.from_list(
                [IntegerAttr[IntegerType].build(d) for d in shape]),
            referenced_type
        ])

@irdl_attr_definition
class ReferenceType(ParametrizedAttribute, MLIRType):
      name = "fir.ref"
      type : ParameterDef[AnyOf([IntegerType, Float16Type, Float32Type, Float64Type, ArrayType])]

@irdl_op_definition
class Absent(Operation):
     name =  "fir.absent"
     intype = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class Addc(Operation):
     name =  "fir.addc"
     lhs = OperandDef(AnyAttr())
     rhs = OperandDef(AnyAttr())
     result = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class AddressOf(Operation):
     name =  "fir.address_of"
     symbol: ParameterDef[FlatSymbolRefAttr]
     resTy = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class Allocmem(Operation):
     name =  "fir.allocmem"
     typeparams = OperandDef(AnyAttr())
     shape = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class Alloca(Operation):
     name =  "fir.alloca"
     in_type = AttributeDef(AnyAttr())
     uniq_name = OptAttributeDef(StringAttr)
     bindc_name = OptAttributeDef(StringAttr)
     #operand_segment_sizes = AttributeDef(ArrayAttr)
     # needs boolean of pinned
     #typeparams = OperandDef(AnyAttr())
     #shape = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()
     valuebyref = OptAttributeDef(UnitAttr)


@irdl_op_definition
class ArrayAccess(Operation):
     name =  "fir.array_access"
     sequence = OperandDef(AnyAttr())
     indices = OperandDef(AnyAttr())
     typeparams = OperandDef(AnyAttr())
     element = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class ArrayAmend(Operation):
     name =  "fir.array_amend"
     sequence = OperandDef(AnyAttr())
     memref = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class ArrayCoor(Operation):
     name =  "fir.array_coor"
     memref = OperandDef(AnyAttr())
     shape = OperandDef(AnyAttr())
     slice = OperandDef(AnyAttr())
     indices = OperandDef(AnyAttr())
     typeparams = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class ArrayFetch(Operation):
     name =  "fir.array_fetch"
     sequence = OperandDef(AnyAttr())
     indices = OperandDef(AnyAttr())
     typeparams = OperandDef(AnyAttr())
     element = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class ArrayLoad(Operation):
     name =  "fir.array_load"
     memref = OperandDef(AnyAttr())
     shape = OperandDef(AnyAttr())
     slice = OperandDef(AnyAttr())
     typeparams = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class ArrayMergeStore(Operation):
     name =  "fir.array_merge_store"
     original = OperandDef(AnyAttr())
     sequence = OperandDef(AnyAttr())
     memref = OperandDef(AnyAttr())
     slice = OperandDef(AnyAttr())
     typeparams = OperandDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class ArrayModify(Operation):
     name =  "fir.array_modify"
     sequence = OperandDef(AnyAttr())
     indices = OperandDef(AnyAttr())
     typeparams = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     result_1 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class ArrayUpdate(Operation):
     name =  "fir.array_update"
     sequence = OperandDef(AnyAttr())
     merge = OperandDef(AnyAttr())
     indices = OperandDef(AnyAttr())
     typeparams = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class BoxAddr(Operation):
     name =  "fir.box_addr"
     val = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class BoxcharLen(Operation):
     name =  "fir.boxchar_len"
     val = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class BoxDims(Operation):
     name =  "fir.box_dims"
     val = OperandDef(AnyAttr())
     dim = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     result_1 = ResultDef(AnyAttr())
     result_2 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class BoxElesize(Operation):
     name =  "fir.box_elesize"
     val = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class BoxIsalloc(Operation):
     name =  "fir.box_isalloc"
     val = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class BoxIsarray(Operation):
     name =  "fir.box_isarray"
     val = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class BoxIsptr(Operation):
     name =  "fir.box_isptr"
     val = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class BoxprocHost(Operation):
     name =  "fir.boxproc_host"
     val = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class BoxRank(Operation):
     name =  "fir.box_rank"
     val = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class BoxTdesc(Operation):
     name =  "fir.box_tdesc"
     val = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class Call(Operation):
     name =  "fir.call"
     callee = AttributeDef(AnyAttr())
     result_0 = OptResultDef(AnyAttr())
     args = VarOperandDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class CharConvert(Operation):
     name =  "fir.char_convert"
     _from = OperandDef(AnyAttr())
     count = OperandDef(AnyAttr())
     to = OperandDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class Cmpc(Operation):
     name =  "fir.cmpc"
     lhs = OperandDef(AnyAttr())
     rhs = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class Constc(Operation):
     name =  "fir.constc"
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class Convert(Operation):
     name =  "fir.convert"
     value = OperandDef(AnyAttr())
     res = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class CoordinateOf(Operation):
     name =  "fir.coordinate_of"
     baseType=AttributeDef(AnyAttr())
     ref = OperandDef(AnyAttr())
     coor = VarOperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class DtEntry(Operation):
     name =  "fir.dt_entry"
     regs = VarRegionDef()


@irdl_op_definition
class Dispatch(Operation):
     name =  "fir.dispatch"
     object = OperandDef(AnyAttr())
     args = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class DispatchTable(Operation):
     name =  "fir.dispatch_table"
     regs = VarRegionDef()


@irdl_op_definition
class Divc(Operation):
     name =  "fir.divc"
     lhs = OperandDef(AnyAttr())
     rhs = OperandDef(AnyAttr())
     result = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class DoLoop(Operation):
     name =  "fir.do_loop"
     lowerBound = OperandDef(AnyAttr())
     upperBound = OperandDef(AnyAttr())
     step = OperandDef(AnyAttr())
     finalValue=OptAttributeDef(AnyAttr())
     initArgs = OperandDef(AnyAttr())
     _results = VarResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class Emboxchar(Operation):
     name =  "fir.emboxchar"
     memref = OperandDef(AnyAttr())
     len = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class Embox(Operation):
     name =  "fir.embox"
     memref = OperandDef(AnyAttr())
     shape = OperandDef(AnyAttr())
     slice = OperandDef(AnyAttr())
     typeparams = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class Emboxproc(Operation):
     name =  "fir.emboxproc"
     func = OperandDef(AnyAttr())
     host = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class ExtractValue(Operation):
     name =  "fir.extract_value"
     adt = OperandDef(AnyAttr())
     res = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class FieldIndex(Operation):
     name =  "fir.field_index"
     typeparams = OperandDef(AnyAttr())
     res = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class End(Operation):
     name =  "fir.end"
     regs = VarRegionDef()


@irdl_op_definition
class Freemem(Operation):
     name =  "fir.freemem"
     heapref = OperandDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class Gentypedesc(Operation):
     name =  "fir.gentypedesc"
     res = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class GlobalLen(Operation):
     name =  "fir.global_len"
     regs = VarRegionDef()


@irdl_op_definition
class Global(Operation):
     name =  "fir.global"
     regs = VarRegionDef()
     sym_name = AttributeDef(StringAttr)
     symref = AttributeDef(FlatSymbolRefAttr)
     type : ParameterDef[AnyOf([IntegerType, Float16Type, Float32Type, Float64Type, ArrayType])]
     linkName = AttributeDef(StringAttr)


@irdl_op_definition
class HasValue(Operation):
     name =  "fir.has_value"
     resval = OperandDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class If(Operation):
     name =  "fir.if"
     condition = OperandDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class InsertOnRange(Operation):
     name =  "fir.insert_on_range"
     seq = OperandDef(AnyAttr())
     val = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class InsertValue(Operation):
     name =  "fir.insert_value"
     adt = OperandDef(AnyAttr())
     val = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class IsPresent(Operation):
     name =  "fir.is_present"
     val = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class IterateWhile(Operation):
     name =  "fir.iterate_while"
     lowerBound = OperandDef(AnyAttr())
     upperBound = OperandDef(AnyAttr())
     step = OperandDef(AnyAttr())
     iterateIn = OperandDef(AnyAttr())
     initArgs = OperandDef(AnyAttr())
     _results = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class LenParamIndex(Operation):
     name =  "fir.len_param_index"
     typeparams = OperandDef(AnyAttr())
     res = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class Load(Operation):
     name =  "fir.load"
     memref = OperandDef(AnyAttr())
     res = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class Mulc(Operation):
     name =  "fir.mulc"
     lhs = OperandDef(AnyAttr())
     rhs = OperandDef(AnyAttr())
     result = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class Negc(Operation):
     name =  "fir.negc"
     operand = OperandDef(AnyAttr())
     result = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class NoReassoc(Operation):
     name =  "fir.no_reassoc"
     val = OperandDef(AnyAttr())
     res = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class Rebox(Operation):
     name =  "fir.rebox"
     box = OperandDef(AnyAttr())
     shape = OperandDef(AnyAttr())
     slice = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class Result(Operation):
  name =  "fir.result"
  regs = VarRegionDef()
  _results = OptOperandDef(AnyAttr())


@irdl_op_definition
class SaveResult(Operation):
     name =  "fir.save_result"
     value = OperandDef(AnyAttr())
     memref = OperandDef(AnyAttr())
     shape = OperandDef(AnyAttr())
     typeparams = OperandDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class SelectCase(Operation):
     name =  "fir.select_case"
     selector = OperandDef(AnyAttr())
     compareArgs = OperandDef(AnyAttr())
     targetArgs = OperandDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class Select(Operation):
     name =  "fir.select"
     selector = OperandDef(AnyAttr())
     compareArgs = OperandDef(AnyAttr())
     targetArgs = OperandDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class SelectRank(Operation):
     name =  "fir.select_rank"
     selector = OperandDef(AnyAttr())
     compareArgs = OperandDef(AnyAttr())
     targetArgs = OperandDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class SelectType(Operation):
     name =  "fir.select_type"
     selector = OperandDef(AnyAttr())
     compareArgs = OperandDef(AnyAttr())
     targetArgs = OperandDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class Shape(Operation):
     name =  "fir.shape"
     extents = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class ShapeShift(Operation):
     name =  "fir.shape_shift"
     pairs = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class Shift(Operation):
     name =  "fir.shift"
     origins = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class Slice(Operation):
     name =  "fir.slice"
     triples = OperandDef(AnyAttr())
     fields = OperandDef(AnyAttr())
     substr = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class Store(Operation):
     name =  "fir.store"
     value = OperandDef(AnyAttr())
     memref = OperandDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class StringLit(Operation):
     name =  "fir.string_lit"
     result_0 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class Subc(Operation):
     name =  "fir.subc"
     lhs = OperandDef(AnyAttr())
     rhs = OperandDef(AnyAttr())
     result = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class Unboxchar(Operation):
     name =  "fir.unboxchar"
     boxchar = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     result_1 = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class Unboxproc(Operation):
     name =  "fir.unboxproc"
     boxproc = OperandDef(AnyAttr())
     result_0 = ResultDef(AnyAttr())
     refTuple = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class Undefined(Operation):
     name =  "fir.undefined"
     intype = ResultDef(AnyAttr())
     regs = VarRegionDef()


@irdl_op_definition
class Unreachable(Operation):
     name =  "fir.unreachable"
     regs = VarRegionDef()


@irdl_op_definition
class ZeroBits(Operation):
     name =  "fir.zero_bits"
     intype = ResultDef(AnyAttr())
     regs = VarRegionDef()


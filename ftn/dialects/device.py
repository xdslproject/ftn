from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import auto

from xdsl.dialects.builtin import (
    IndexType,
    IntAttr,
    IntegerAttr,
    MemRefType,
    StringAttr,
    i1,
    i32,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    EnumAttribute,
    Operation,
    SpacedOpaqueSyntaxAttribute,
    SSAValue,
    StrEnum,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    prop_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.traits import (
    MemoryAllocEffect,
)
from xdsl.utils.hints import isa


class MemoryKind(StrEnum):
    HBM = auto()
    DDR = auto()
    SRAM = auto()
    BRAM = auto()
    URAM = auto()


class ArchitectureKind(StrEnum):
    FPGA = auto()
    MANYCORE = auto()


class IntegrationKind(StrEnum):
    PCIe = auto()
    EMBEDDED = auto()


@irdl_attr_definition
class MemoryKindAttr(EnumAttribute[MemoryKind], SpacedOpaqueSyntaxAttribute):
    name = "device.memorykind"


@irdl_attr_definition
class ArchitectureKindAttr(
    EnumAttribute[ArchitectureKind], SpacedOpaqueSyntaxAttribute
):
    name = "device.architecturekind"


@irdl_attr_definition
class IntegrationKindAttr(EnumAttribute[IntegrationKind], SpacedOpaqueSyntaxAttribute):
    name = "device.integrationkind"


@irdl_op_definition
class AllocOp(IRDLOperation):
    name = "device.alloc"

    dynamic_sizes = var_operand_def(IndexType)

    memref = result_def(MemRefType)

    memory_name = prop_def(StringAttr, prop_name="name")
    memory_space = prop_def(IntegerAttr)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    traits = traits_def(MemoryAllocEffect())

    def __init__(
        self,
        memory_name: StringAttr,
        element_type: Attribute,
        memory_space: IntegerAttr | int,
        shape: Iterable[int | IntAttr] | None = None,
        dynamic_sizes: Sequence[SSAValue | Operation] | None = None,
    ):
        if dynamic_sizes is None:
            dynamic_sizes = []

        if isa(memory_space, int):
            memory_space = IntegerAttr.from_int_and_width(memory_space, i32)

        super().__init__(
            operands=([dynamic_sizes]),
            result_types=[MemRefType(element_type, shape, memory_space=memory_space)],
            properties={"name": memory_name, "memory_space": memory_space},
        )


@irdl_op_definition
class LookUpOp(IRDLOperation):
    name = "device.lookup"

    memory_name = prop_def(StringAttr, prop_name="name")
    memory_space = prop_def(IntegerAttr)

    memref = result_def(MemRefType)

    def __init__(
        self,
        memory_name: StringAttr,
        memory_space: IntegerAttr | int,
        result_type: Attribute,
    ):
        if isa(memory_space, int):
            memory_space = IntegerAttr.from_int_and_width(memory_space, i32)
        result_memref_type = MemRefType(
            result_type.element_type, result_type.shape, memory_space=memory_space
        )
        super().__init__(
            result_types=[result_memref_type],
            properties={"name": memory_name, "memory_space": memory_space},
        )


@irdl_op_definition
class DataCheckExists(IRDLOperation):
    name = "device.data_check_exists"

    memory_name = prop_def(StringAttr, prop_name="name")
    memory_space = prop_def(IntegerAttr)

    res = result_def(i1)

    def __init__(
        self,
        memory_name: StringAttr,
        memory_space: IntegerAttr | int,
    ):
        if isa(memory_space, int):
            memory_space = IntegerAttr.from_int_and_width(memory_space, i32)
        super().__init__(
            result_types=[i1],
            properties={"name": memory_name, "memory_space": memory_space},
        )


class DataStatusUpdate(IRDLOperation):
    memory_name = prop_def(StringAttr, prop_name="name")
    memory_space = prop_def(IntegerAttr)

    def __init__(
        self,
        memory_name: StringAttr,
        memory_space: IntegerAttr | int,
    ):
        if isa(memory_space, int):
            memory_space = IntegerAttr.from_int_and_width(memory_space, i32)
        super().__init__(
            properties={"name": memory_name, "memory_space": memory_space},
        )


@irdl_op_definition
class DataAcquire(DataStatusUpdate):
    name = "device.data_acquire"


@irdl_op_definition
class DataRelease(DataStatusUpdate):
    name = "device.data_release"


@irdl_op_definition
class DataNumElements(IRDLOperation):
    name = "device.data_num_elements"

    memory_name = prop_def(StringAttr, prop_name="name")
    memory_space = prop_def(IntegerAttr)

    res = result_def(IndexType)

    def __init__(
        self,
        memory_name: StringAttr,
        memory_space: IntegerAttr | int,
    ):
        if isa(memory_space, int):
            memory_space = IntegerAttr.from_int_and_width(memory_space, i32)
        super().__init__(
            result_types=[IndexType()],
            properties={"name": memory_name, "memory_space": memory_space},
        )


Device = Dialect(
    "device",
    [
        AllocOp,
        DataAcquire,
        DataCheckExists,
        DataNumElements,
        DataRelease,
        LookUpOp,
    ],
    [
        ArchitectureKindAttr,
        IntegrationKindAttr,
        MemoryKindAttr,
    ],
)

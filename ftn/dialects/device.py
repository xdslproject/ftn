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
    ParametrizedAttribute,
    Region,
    SpacedOpaqueSyntaxAttribute,
    SSAValue,
    StrEnum,
    TypeAttribute,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.traits import (
    IsTerminator,
    MemoryAllocEffect,
    Pure,
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


@irdl_attr_definition
class KernelHandle(ParametrizedAttribute, TypeAttribute):
    name = "device.kernelhandle"


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


@irdl_op_definition
class KernelCreate(IRDLOperation):
    name = "device.kernel_create"

    static_args = var_operand_def()
    runtime_args = var_operand_def()
    mapped_data = var_operand_def()

    target = prop_def(StringAttr)

    body = region_def("single_block")

    res = result_def(KernelHandle)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def __init__(
        self,
        target: StringAttr | str,
        region: Region,
        static_args: Sequence[SSAValue | Operation] | None = None,
        runtime_args: Sequence[SSAValue | Operation] | None = None,
        mapped_data: Sequence[SSAValue | Operation] | None = None,
    ):
        if static_args is None:
            static_args = []

        if runtime_args is None:
            runtime_args = []

        if mapped_data is None:
            mapped_data = []

        super().__init__(
            operands=([static_args, runtime_args, mapped_data]),
            result_types=[KernelHandle()],
            properties={"target": target},
            regions=[region],
        )


@irdl_op_definition
class KernelLaunch(IRDLOperation):
    name = "device.kernel_launch"

    handle = operand_def()

    def __init__(
        self,
        handle: SSAValue | Operation,
    ):
        super().__init__(
            operands=([handle]),
        )


@irdl_op_definition
class KernelWait(IRDLOperation):
    name = "device.kernel_wait"

    handle = operand_def()

    def __init__(
        self,
        handle: SSAValue | Operation,
    ):
        super().__init__(
            operands=([handle]),
        )


@irdl_op_definition
class KernelTerminatorOp(IRDLOperation):
    name = "device.kernel_terminator"

    traits = traits_def(IsTerminator(), Pure())


Device = Dialect(
    "device",
    [
        AllocOp,
        DataAcquire,
        DataCheckExists,
        DataNumElements,
        DataRelease,
        LookUpOp,
        KernelCreate,
        KernelLaunch,
        KernelTerminatorOp,
        KernelWait,
    ],
    [
        ArchitectureKindAttr,
        IntegrationKindAttr,
        MemoryKindAttr,
        KernelHandle,
    ],
)

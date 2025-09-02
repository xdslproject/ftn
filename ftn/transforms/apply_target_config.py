from dataclasses import dataclass
from abc import ABC, abstractmethod

from xdsl.context import Context
from xdsl.dialects import builtin, dlti
from xdsl.passes import ModulePass

from ftn.dialects import device


class TargetConfiguration(ABC):
    @classmethod
    @abstractmethod
    def get(cls) -> dlti.TargetDeviceSpecAttr: ...

    @classmethod
    @abstractmethod
    def _memory_subsystem(cls) -> dlti.MapAttr: ...

    @classmethod
    @abstractmethod
    def _compute_subsystem(cls) -> dlti.MapAttr: ...


class TenstorrentConfiguration(TargetConfiguration):
    @classmethod
    def get(cls):
        return dlti.TargetDeviceSpecAttr(
            {
                "memory": cls._memory_subsystem(),
                "compute": cls._compute_subsystem(),
            }
        )

    @classmethod
    def _memory_subsystem(cls):
        config = {
            "DRAM": {
                "kind": device.MemoryKindAttr(device.MemoryKind.DDR),
                "size": "16GB",
            },
        }
        return dlti.MapAttr(config)

    @classmethod
    def _compute_subsystem(cls):
        config = {
            "architecture_type": device.ArchitectureKindAttr(
                device.ArchitectureKind.MANYCORE
            ),
            "integration": device.IntegrationKindAttr(device.IntegrationKind.PCIe),
            "num_cores": 128,
            "core_config": {
                "vector_unit": {
                    "max_element_width": 1024,
                    "simd_lanes": 32,
                    "lane_width": 32,
                    "data_types": "fp32,int32,bf16,fp16,int16,int8",
                },
                "matrix_unit": {
                    "max_element_width": 1024,
                    "element_width": 19,
                    "data_types": "fp16,bf16,int8",
                },
                "local_memory": {
                    "kind": device.MemoryKindAttr(device.MemoryKind.SRAM),
                    "size": "1.5MB",
                },
            },
        }
        return dlti.MapAttr(config)


class U280Configuration(TargetConfiguration):
    @classmethod
    def get(cls):
        return dlti.TargetDeviceSpecAttr(
            {
                "memory": cls._memory_subsystem(),
                "compute": cls._compute_subsystem(),
            }
        )

    @classmethod
    def _memory_subsystem(cls):
        config = {
            "DRAM": {
                "kind": device.MemoryKindAttr(device.MemoryKind.DDR),
                "size": "16GB",
            },
        }
        for i in range(32):
            config["HBM" + str(i)] = {
                "kind": device.MemoryKindAttr(device.MemoryKind.HBM),
                "size": "256MB",
            }
        return dlti.MapAttr(config)

    @classmethod
    def _compute_subsystem(cls):
        config = {
            "architecture_type": device.ArchitectureKindAttr(
                device.ArchitectureKind.FPGA
            ),
            "integration": device.IntegrationKindAttr(device.IntegrationKind.PCIe),
        }
        return dlti.MapAttr(config)


class PhoenixConfiguration:
    def get():
        return dlti.TargetDeviceSpecAttr(
            {
                "memory": PhoenixConfiguration._memory_subsystem(),
                "compute": PhoenixConfiguration._compute_subsystem(),
            }
        )

    def _memory_subsystem():
        config = {}
        for i in range(5):
            config["TILE_" + str(i)] = {
                "kind": device.MemoryKindAttr(device.MemoryKind.SRAM),
                "size": "512KB",
            }
        return dlti.MapAttr(config)

    def _compute_subsystem():
        config = {
            "architecture_type": device.ArchitectureKindAttr(
                device.ArchitectureKind.MANYCORE
            ),
            "integration": device.IntegrationKindAttr(device.IntegrationKind.EMBEDDED),
            "num_cores": 20,
            "core_config": {
                "vector_unit": {
                    "bit_width": 512,
                    "data_types": "fp32,fp16int8,int16,int32,uint8",
                },
                "local_memory": {
                    "kind": device.MemoryKindAttr(device.MemoryKind.SRAM),
                    "size": "64KB",
                },
            },
        }
        return dlti.MapAttr(config)


SYSTEM_CONFIGURATIONS = {
    "tenstorrent": TenstorrentConfiguration,
    "u280": U280Configuration,
    "phoenix": PhoenixConfiguration,
}


@dataclass(frozen=True)
class ApplyTargetConfig(ModulePass):
    name = "apply-target"

    target: str = "tenstorrent"

    def generate_system_config(self, accelerator_name, accelerator_config):
        mem_config = accelerator_config["memory"]
        accel_memories = []
        for entry in mem_config.entries:
            accel_memories.append(entry.key.data)
        memory_spaces_config = {"0": "HOST_DRAM"}
        for idx, am in enumerate(accel_memories):
            memory_spaces_config[str(idx + 1)] = am
        return dlti.TargetSystemSpecAttr(
            {
                accelerator_name: accelerator_config,
                "memory_spaces": memory_spaces_config,
            }
        )

    def _get_config(self) -> dlti.TargetDeviceSpecAttr:
        """
        Get the device spec for the current `self.taregt`

        If overriding this function, make sure to *not* specify `name` field again
        """
        if config := SYSTEM_CONFIGURATIONS.get(self.target):
            return config.get()
        raise ValueError(f"No such target configuration {self.target}")

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        op.attributes["omp.target_triples"] = builtin.ArrayAttr(
            [builtin.StringAttr(self.target)]
        )

        config = self._get_config()

        op.attributes["dlti.target_system_spec"] = self.generate_system_config(
            self.target, config
        )

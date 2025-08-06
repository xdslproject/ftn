from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, dlti
from xdsl.passes import ModulePass

from ftn.dialects import device


class TenstorrentConfiguration:
    def get():
        return dlti.TargetDeviceSpecAttr(
            {
                "memory": TenstorrentConfiguration._memory_subsystem(),
                "compute": TenstorrentConfiguration._compute_subsystem(),
            }
        )

    def _memory_subsystem():
        config = {
            "DRAM": {
                "kind": device.MemoryKindAttr(device.MemoryKind.DDR),
                "size": "16GB",
            },
        }
        return dlti.MapAttr(config)

    def _compute_subsystem():
        config = {
            "architecture_type": device.ArchitectureKindAttr(
                device.ArchitectureKind.MANYCORE
            ),
            "integration": device.IntegrationKindAttr(device.IntegrationKind.PCIe),
            "num_cores": 128,
            "core_config": {
                "vector_length": 256,
                "local_memory": {
                    "kind": device.MemoryKindAttr(device.MemoryKind.SRAM),
                    "size": "1.5MB",
                },
            },
        }
        return dlti.MapAttr(config)


class U280Configuration:
    def get():
        return dlti.TargetDeviceSpecAttr(
            {
                "memory": U280Configuration._memory_subsystem(),
                "compute": U280Configuration._compute_subsystem(),
            }
        )

    def _memory_subsystem():
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

    def _compute_subsystem():
        config = {
            "architecture_type": device.ArchitectureKindAttr(
                device.ArchitectureKind.FPGA
            ),
            "integration": device.IntegrationKindAttr(device.IntegrationKind.PCIe),
        }
        return dlti.MapAttr(config)


SYSTEM_CONFIGURATIONS = {
    "tenstorrent": TenstorrentConfiguration,
    "u280": U280Configuration,
}


@dataclass(frozen=True)
class ApplyTargetConfig(ModulePass):
    name = "apply-target"

    target: str = "tenstorrent"

    def get_dlti_item(config, name):
        for e in config.entries.data:
            if e.key == builtin.StringAttr(name):
                return e.value

    def generate_system_config(self, accelerator_name, accelerator_config):
        mem_config = ApplyTargetConfig.get_dlti_item(accelerator_config, "memory")
        accel_memories = []
        for entry in mem_config.entries.data:
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

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        op.attributes["omp.target_triples"] = builtin.ArrayAttr(
            [builtin.StringAttr(self.target)]
        )

        assert self.target in SYSTEM_CONFIGURATIONS.keys()

        config = SYSTEM_CONFIGURATIONS[self.target].get()

        op.attributes["dlti.target_system_spec"] = self.generate_system_config(
            self.target, config
        )

import argparse
import os, glob

from xdsl.dialects.builtin import ModuleOp

from ftn.transforms.rewrite_fir_to_core import RewriteFIRToCore
from ftn.transforms.merge_memref_deref import MergeMemRefDeref
#from ftn.transforms.extract_target import ExtractTarget
#from ftn.transforms.isolate_target import IsolateTarget
#from psy.extract_stencil import ExtractStencil
#from ftn.transforms.tenstorrent.convert_to_tt import ConvertToTT
#from psy.infer_gpu_data_transfer import InferGPUDataTransfer

from pathlib import Path

from typing import Callable, Dict, List

from xdsl.xdsl_opt_main import xDSLOptMain
from ftn.dialects import dlti, ftn_relative_cf


class FtnOptMain(xDSLOptMain):

    def register_all_passes(self):
        super().register_all_passes()
        self.register_pass("rewrite-fir-to-core", lambda: RewriteFIRToCore)
        self.register_pass("merge-memref-deref", lambda: MergeMemRefDeref)
        #self.register_pass("extract-target", lambda: ExtractTarget)
        #self.register_pass("isolate-target", lambda: IsolateTarget)
        #self.register_pass("convert-to-tt", lambda: ConvertToTT)


    def register_all_targets(self):
        super().register_all_targets()

    def setup_pipeline(self):
      super().setup_pipeline()

    def register_all_arguments(self, arg_parser: argparse.ArgumentParser):
      super().register_all_arguments(arg_parser)
      arg_parser.add_argument(
            "--output-module-files",
            default=False,
            action="store_true",
            help="Outputs the generated module files on a module by module basis",
        )

    def register_all_dialects(self):
        super().register_all_dialects()
        self.ctx.load_dialect(dlti.DLTI)
        self.ctx.load_dialect(ftn_relative_cf.Ftn_relative_cf)

    @staticmethod
    def get_passes_as_dict(
    ) -> Dict[str, Callable[[ModuleOp], None]]:
        """Add all passes that can be called by psy-opt in a dictionary."""

        pass_dictionary = {}

        passes = FtnOptMain.passes_native

        for pass_function in passes:
            pass_dictionary[pass_function.__name__.replace(
                "_", "-")] = pass_function

        return pass_dictionary

    def get_passes_as_list(native=False, integrated=False) -> List[str]:
        """Add all passes that can be called by psy-opt in a dictionary."""

        pass_list = []

        passes = FtnOptMain.passes_native

        for pass_function in passes:
            pass_list.append(pass_function.__name__.replace("_", "-"))

        return pass_list

    def register_all_frontends(self):
        super().register_all_frontends()

def _output_modules_to_file_for_target(module, target, psy_main):
  psy_main.args.target=target
  i=0
  # This will generate output for every sub module that is part of the
  # top level module
  for op in module.regions[0].blocks[0].ops:
    if isinstance(op, ModuleOp):
      module_contents=psy_main.output_resulting_program(op)
      f = open("generated/module_"+str(i)+"."+target, "w")
      f.write(module_contents)
      f.close()
      i+=1

def _empty_generate_dir():
  if not os.path.isdir("generated"):
      Path("generated").mkdir(parents=True, exist_ok=True)

  files = glob.glob('generated/*')
  for f in files:
    os.remove(f)

def main():
    ftn_main = FtnOptMain()

    try:
        ftn_main.run()
        if ftn_main.args.output_module_files:
          chunks, file_extension = ftn_main.prepare_input()
          assert len(chunks) == 1
          module=ftn_main.parse_chunk(chunks[0], file_extension)
          ftn_main.apply_passes(module)
          contents = ftn_main.output_resulting_program(module)
          _empty_generate_dir()
          _output_modules_to_file_for_target(module, ftn_main.args.target, ftn_main)
    except SyntaxError as e:
        print(e.get_message())
        exit(0)
    except Exception as e:
        print("Error: %s" % str(e))
        exit(0)

if __name__ == "__main__":
    main()

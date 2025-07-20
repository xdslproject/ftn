#!/usr/bin/env python3.10
import argparse
import os
import shutil
from typing import IO


def initialise_argument_parser():
    parser = argparse.ArgumentParser(description="xDSL Fortran compiler flow")
    parser.add_argument("source file", help="Filename of source Fortran code")
    parser.add_argument(
        "-o",
        "--out",
        nargs="?",
        default=None,
        help="Filename of output file (executable or MLIR)",
    )
    parser.add_argument("-omp", "--openmp", action="store_true", help="Enable OpenMP")
    parser.add_argument("-fomp", "--fopenmp", action="store_true", help="Enable OpenMP")
    parser.add_argument(
        "-D",
        "--define-macro",
        action="append",
        dest="flang_pp_macros",
        default=[],
        help="Define preprocessor macros for Flang",
    )
    parser.add_argument(
        "-I",
        "--include-directory",
        action="append",
        dest="flang_includes",
        default=[],
        help="Include directory into path for modules and header files",
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Run OpenMP accelerator offloading flow, overrides stages argument if present",
    )
    parser.add_argument(
        "-t",
        "--tempdir",
        default="tmp",
        help="Specify temporary compilation directory (default is 'tmp')",
    )
    parser.add_argument(
        "--stages",
        default="flang,pre,ftn,post,mlir,clang",
        help="Specify which stages will run (a combination of: flang, pre, ftn, post, mlir, clang) in comma separated list without spaces",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Enable verbose mode (default is 1)",
    )
    return parser


def build_options_db_from_args(args):
    # We build an options database from the arguments, this is simply a dictionary but
    # we do it this way so that some options can turn on lots of other options,
    # an example is fopenmp and openmp being aliases for each other
    options_db = args.__dict__

    # Handle fopenmp and openmp aliases
    if options_db["fopenmp"]:
        options_db["openmp"] = True
    del options_db["fopenmp"]

    stages_to_run = options_db["stages"].split(",")

    # Set the stages that will run, by default all stages run
    options_db["run_flang_stage"] = "flang" in stages_to_run
    options_db["run_preprocess_stage"] = "pre" in stages_to_run
    options_db["run_lower_to_core_stage"] = "ftn" in stages_to_run
    options_db["run_postprocess_stage"] = "post" in stages_to_run
    options_db["run_mlir_to_llvmir_stage"] = "mlir" in stages_to_run
    options_db["run_build_executable_stage"] = "clang" in stages_to_run

    # Option specific variations to the options database, where
    # a specific option will enable or disable others
    if options_db["offload"]:
        options_db["openmp"] = True
        options_db["run_postprocess_stage"] = False
        options_db["run_mlir_to_llvmir_stage"] = False
        options_db["run_build_executable_stage"] = False

    return options_db


def display_verbose_start_message(options_db):
    print("Running compilation pipeline with configuration")
    print("-----------------------------------------------")
    print(f"OpenMP enabled: {options_db['openmp']}")
    print(f"Offload enabled: {options_db['offload']}")
    print("")
    print(
        f"Stage 'Flang': {'Enabled' if options_db['run_flang_stage'] else 'Disabled'}"
    )
    print(
        f"Stage 'Preprocess': {'Enabled' if options_db['run_preprocess_stage'] else 'Disabled'}"
    )
    print(
        f"Stage 'Lower to core dialects': {'Enabled' if options_db['run_lower_to_core_stage'] else 'Disabled'}"
    )
    print(
        f"Stage 'Postprocess for MLIR': {'Enabled' if options_db['run_postprocess_stage'] else 'Disabled'}"
    )
    print(
        f"Stage 'Convert MLIR to LLVM-IR': {'Enabled' if options_db['run_mlir_to_llvmir_stage'] else 'Disabled'}"
    )
    print(
        f"Stage 'Build executable': {'Enabled' if options_db['run_build_executable_stage'] else 'Disabled'}"
    )
    print("------------------------------------------------")


def remove_file_if_exists(dir_prefix, *filenames):
    for filename in filenames:
        fn = os.path.join(dir_prefix, filename)
        if os.path.exists(fn):
            os.remove(fn)


def validate_source_filename(src_fn):
    components = src_fn.split(".")
    if len(components) != 2:
        raise Exception(
            "Source filename must have a filetype, '.x', where x is 'F90', 'f90' or 'f'"
        )
    if components[-1] != "F90" and components[-1] != "f90" and components[-1] != "f":
        raise Exception("Source filename must end in 'F90', 'f90' or 'f'")


def print_verbose_message(options_db, *messages):
    assert len(messages) > 0

    verbose_level = options_db["verbose"]
    if verbose_level == 1:
        print(messages[0])
    elif verbose_level == 2:
        if len(messages) > 1:
            print(messages[1])
        else:
            print(messages[0])


def post_stage_check(check_fn, verbose_level, executable=False):
    if not os.path.exists(check_fn):
        exit(-1)
    if verbose_level >= 1:
        print(
            f"  -> Completed, {'executable' if executable else 'results'} in '{check_fn}'"
        )


def generate_flang_optional_args(options_db):
    optional_args = ""
    if options_db["openmp"]:
        optional_args += "-fopenmp "
    for flang_pp_macro in options_db["flang_pp_macros"]:
        optional_args += f"-D{flang_pp_macro} "
    for flang_include in options_db["flang_includes"]:
        optional_args += f"-I{flang_include} "
    return optional_args


def generate_clang_optional_args(options_db):
    optional_args = ""
    if options_db["openmp"]:
        optional_args += "-fopenmp "
    return optional_args


def run_ftnopt_passes(
    output_tmp_dir,
    input_fn,
    output_fn,
    passes,
    options_db,
    verbose_msg_preamble="Executing ftn-opt transformation passes",
):
    assert len(passes) > 0

    input_f = os.path.join(output_tmp_dir, input_fn)
    output_f = os.path.join(output_tmp_dir, output_fn)

    passes_list = ",".join(passes)

    ftn_args = f"{input_f} -p {passes_list} -o {output_f}"

    print_verbose_message(
        options_db,
        verbose_msg_preamble,
        f"{verbose_msg_preamble} with arguments '{ftn_args}'",
    )

    os.system(f"ftn-opt {ftn_args}")
    post_stage_check(output_f, options_db["verbose"])


def run_flang(source_filename, output_tmp_dir, output_filename, options_db):
    output_mlir_fn = os.path.join(output_tmp_dir, output_filename)
    flang_args = f"-fc1 -module-dir {output_tmp_dir} -emit-hlfir -mmlir -mlir-print-op-generic {generate_flang_optional_args(options_db)} {source_filename} -o {output_mlir_fn}"

    print_verbose_message(
        options_db, "Running Flang", f"Running Flang with arguments '{flang_args}'"
    )

    os.system(f"flang {flang_args}")
    post_stage_check(output_mlir_fn, options_db["verbose"])


def run_preprocess_flang_to_xdsl(output_tmp_dir, input_fn, output_fn, options_db):
    input_f = os.path.join(output_tmp_dir, input_fn)
    output_f = os.path.join(output_tmp_dir, output_fn)

    print_verbose_message(
        options_db,
        "Preprocessing to xDSL compatible form",
        f"Preprocessing '{input_f}' to xDSL compatible form stored in '{output_f}'",
    )

    os.system(f"preprocess_mlir_for_xdsl {input_f} {output_f}")
    post_stage_check(output_f, options_db["verbose"])


def lower_fir_to_core_dialects(output_tmp_dir, input_fn, output_fn, options_db):
    transformation_passes = ["rewrite-fir-to-core", "merge-memref-deref"]
    run_ftnopt_passes(
        output_tmp_dir,
        input_fn,
        output_fn,
        transformation_passes,
        options_db,
        "Lowering to core dialects",
    )


def run_postprocess_core_mlir(output_tmp_dir, input_fn, output_fn, options_db):
    input_f = os.path.join(output_tmp_dir, input_fn)
    output_f = os.path.join(output_tmp_dir, output_fn)

    print_verbose_message(
        options_db,
        "Postprocessing xDSL output to MLIR compatible form",
        f"Postprocessing xDSL output '{input_f}' to MLIR compatible form stored in '{output_f}'",
    )

    os.system(f"postprocess_xdsl_for_mlir {input_f} {output_f}")
    post_stage_check(output_f, options_db["verbose"])


def run_mlir_pipeline_to_llvm_ir(output_tmp_dir, input_fn, output_fn, options_db):
    input_f = os.path.join(output_tmp_dir, input_fn)
    output_f = os.path.join(output_tmp_dir, output_fn)

    mlir_pipeline_pass = '--pass-pipeline="builtin.module(canonicalize, cse, loop-invariant-code-motion, convert-linalg-to-loops, convert-scf-to-cf, convert-cf-to-llvm{index-bitwidth=64}, fold-memref-alias-ops,lower-affine,finalize-memref-to-llvm, convert-arith-to-llvm{index-bitwidth=64}, convert-func-to-llvm, math-uplift-to-fma, convert-math-to-llvm, fold-memref-alias-ops,lower-affine,finalize-memref-to-llvm, reconcile-unrealized-casts)"'
    mlir_args = f"{mlir_pipeline_pass} {input_f}"

    print_verbose_message(
        options_db,
        "Generating LLVM-IR",
        f"Running MLIR pipeline pass '{mlir_pipeline_pass}' and then 'mlir-translate --mlir-to-llvmir' on the output to generate LLVM-IR",
    )

    os.system(f"mlir-opt {mlir_args} | mlir-translate --mlir-to-llvmir -o {output_f}")
    post_stage_check(output_f, options_db["verbose"])


def build_executable(output_tmp_dir, input_fn, executable_fn, options_db):
    input_f = os.path.join(output_tmp_dir, input_fn)

    clang_args = f"-O3 -o {executable_fn} {input_f} {generate_clang_optional_args(options_db)} -lFortranRuntime -lFortranDecimal -lm -lgcc"

    print_verbose_message(
        options_db,
        "Building executable",
        f"Building executable by executing clang with arguments '{clang_args}'",
    )

    os.system(f"clang {clang_args}")
    post_stage_check(executable_fn, options_db["verbose"], executable=True)


def main():
    parser = initialise_argument_parser()
    args = parser.parse_args()
    options_db = build_options_db_from_args(args)
    if options_db["verbose"] == 2:
        display_verbose_start_message(options_db)

    src_fn = options_db["source file"]
    validate_source_filename(src_fn)
    source_fn_no_ext = src_fn.split(".")[-2]

    if options_db["out"] is not None:
        out_file = options_db["out"]
    else:
        out_file = source_fn_no_ext
        if options_db["offload"]:
            out_file = source_fn_no_ext + "_offload.mlir"

    tmp_dir = options_db["tempdir"]
    os.makedirs(tmp_dir, exist_ok=True)
    remove_file_if_exists(
        tmp_dir,
        source_fn_no_ext + ".mlir",
        source_fn_no_ext + "_pre.mlir",
        source_fn_no_ext + "_res.mlir",
        source_fn_no_ext + "_post.mlir",
        source_fn_no_ext + "_res.bc",
    )

    if options_db["run_flang_stage"]:
        run_flang(src_fn, tmp_dir, source_fn_no_ext + ".mlir", options_db)
    if options_db["run_preprocess_stage"]:
        run_preprocess_flang_to_xdsl(
            tmp_dir,
            source_fn_no_ext + ".mlir",
            source_fn_no_ext + "_pre.mlir",
            options_db,
        )
    if options_db["run_lower_to_core_stage"]:
        lower_fir_to_core_dialects(
            tmp_dir,
            source_fn_no_ext + "_pre.mlir",
            source_fn_no_ext + "_res.mlir",
            options_db,
        )
    if options_db["run_postprocess_stage"]:
        run_postprocess_core_mlir(
            tmp_dir,
            source_fn_no_ext + "_res.mlir",
            source_fn_no_ext + "_post.mlir",
            options_db,
        )
    if options_db["run_mlir_to_llvmir_stage"]:
        run_mlir_pipeline_to_llvm_ir(
            tmp_dir,
            source_fn_no_ext + "_post.mlir",
            source_fn_no_ext + "_res.bc",
            options_db,
        )
    if options_db["run_build_executable_stage"]:
        build_executable(tmp_dir, source_fn_no_ext + "_res.bc", out_file, options_db)

    if options_db["offload"]:
        # If this is the offload flow then copy the result in the temporary directory to the specified output
        shutil.copy(os.path.join(tmp_dir, source_fn_no_ext + "_res.mlir"), out_file)
        print_verbose_message(options_db, f"Offload MLIR in '{out_file}'")


if __name__ == "__main__":
    main()

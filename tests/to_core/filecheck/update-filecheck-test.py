#!/bin/python3

import sys
import os

if not "XFTN_PATH" in os.environ:
    print(
        "Error: 'XFTN_PATH' environment must be set to the root of the 'ftn' repository"
    )
    exit(-1)

filecheck_dir = os.path.join(os.environ["XFTN_PATH"], "tests/to_core/filecheck")

assert len(sys.argv) == 3

test_name = sys.argv[1]
ftn_src_code = sys.argv[2]

assert test_name.endswith(".test")
assert ftn_src_code.endswith(".F90")

if not os.path.exists(test_name):
    raise Exception(f"Test file '{test_name}' does not exist")

if not os.path.exists(ftn_src_code):
    raise Exception(f"Fortran source file '{ftn_src_code}' does not exist")

in_file = open(test_name, "r")
command_preamble = in_file.readline()
in_file.close()

xftn_command_with_fn = (
    command_preamble.split("|")[0]
    .replace("// RUN: ", "")
    .replace("-t %S/tmp", "")
    .replace("--stdout", "-o tmp_out.mlir")
    .strip()
)
xftn_command_with_fn = xftn_command_with_fn.split(" ")

command_to_run = f"xftn {ftn_src_code} {' '.join(xftn_command_with_fn[2:])}"

print(f"Running: {command_to_run}")
os.system(command_to_run)

assert os.path.exists("tmp_out.mlir")
post_processing_cmd = f"python3 {os.path.join(filecheck_dir, 'generate-filecheck-format.py')} tmp_out.mlir"
print(f"Running: {post_processing_cmd}")
os.system(post_processing_cmd)

assert os.path.exists("tmp_out.mlir_test")

processed_out_mlir = ""

with open("tmp_out.mlir_test") as file:
    for line in file:
        processed_out_mlir += line

with open(test_name, "w") as file:
    file.write(command_preamble)
    file.write(processed_out_mlir)

os.remove("tmp_out.mlir")
os.remove("tmp_out.mlir_test")

print(f"Update complete: {test_name} regenerated based on {ftn_src_code}")

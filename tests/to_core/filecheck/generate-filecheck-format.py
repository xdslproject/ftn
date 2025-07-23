#!/bin/python3

import sys
import re

in_file = open(sys.argv[1], "r")
out_file = open(sys.argv[1] + "_test", "w")

lines = in_file.readlines()

output_lines = ""

global_fn_lengths = []

for idx, line in enumerate(lines):
    needs_literal = "[[" in line or "]]" in line
    if idx == 0:
        output_lines += f"//CHECK{'{LITERAL}' if needs_literal else ''}:       "
    else:
        if len(line.strip()) == 0:
            output_lines += "//CHECK-EMPTY:  "
        else:
            output_lines += f"//CHECK-NEXT{'{LITERAL}' if needs_literal else ''}:  "
    if "llvm.mlir.addressof" in line and "global_name = @_" in line:
        components = line.split("global_name = ")
        line = (
            components[0] + "global_name = @{{.*}}}>" + components[1].split("}>", 1)[1]
        )

    if (
        "llvm.mlir.global" in line
        and ".F90" in line
        and re.search("!llvm.array<[0-9]+ x i8>", line)
    ):
        # Wildcard the source filename as this makes it much easier to generate the MLIR for comparison
        components = line.split("!llvm.array<")
        line = (
            components[0] + "!llvm.array<{{[0-9]+}} x" + components[1].split("x", 1)[1]
        )
        components = line.split('value = "')
        global_fn_lengths.append(len(components[1].split('"', 1)[0]))
        line = components[0] + 'value = "{{.*}}"' + components[1].split('"', 1)[1]
        components = line.split('sym_name = "')
        line = components[0] + 'sym_name = "_{{.*}}"' + components[1].split('"', 1)[1]
    output_lines += line

# Phase two replaces constant string lengths that match global strings that
# we have wildcarded and these lengths are then wildcarded
phase_two_out_lines = ""
for line in iter(output_lines.splitlines()):
    if "arith.constant" in line and ": index" in line:
        components = line.split("arith.constant")
        num = int(components[1].split(":")[0].strip())
        if num in global_fn_lengths:
            line = components[0] + "arith.constant {{.*}} : index"

    phase_two_out_lines += line + "\n"

out_file.write(phase_two_out_lines)

in_file.close()
out_file.close()

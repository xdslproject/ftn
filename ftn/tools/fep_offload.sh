#!/bin/sh

filename_no_ext="${1%.*}"

rm -f $filename_no_ext".mlir" $filename_no_ext"_pre.mlir" $filename_no_ext"_res.mlir" $filename_no_ext"_post.mlir" $filename_no_ext"_res.bc" $filename_no_ext
echo "Running Flang"
flang-new -fc1 -emit-hlfir -mmlir -mlir-print-op-generic -fopenmp  $1

preprocess_mlir_for_xdsl $filename_no_ext.mlir $filename_no_ext"_pre".mlir
ftn-opt $filename_no_ext"_pre.mlir" -p rewrite-fir-to-standard,merge-memref-deref -o $filename_no_ext"_res.mlir"
ftn-opt $filename_no_ext"_res.mlir" -p extract-target -o $filename_no_ext"_tgt.mlir"
ftn-opt $filename_no_ext"_tgt.mlir" -p isolate-target -o $filename_no_ext"_isol.mlir"

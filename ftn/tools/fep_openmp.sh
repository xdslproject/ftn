#!/bin/sh

mkdir -p tmp

filename_no_ext="${1%.*}"

rm -f $filename_no_ext".mlir" "tmp/"$filename_no_ext"_pre.mlir" "tmp/"$filename_no_ext"_res.mlir"
echo "Running Flang"
flang -fc1 -emit-hlfir -mmlir -mlir-print-op-generic -module-dir tmp -fopenmp  $1 -o "tmp"/$filename_no_ext.mlir

preprocess_mlir_for_xdsl "tmp/"$filename_no_ext.mlir "tmp/"$filename_no_ext"_pre".mlir
ftn-opt "tmp/"$filename_no_ext"_pre.mlir" -p rewrite-fir-to-core,merge-memref-deref -o "tmp/"$filename_no_ext"_res.mlir"

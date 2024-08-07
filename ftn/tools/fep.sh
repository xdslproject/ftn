#!/bin/sh

filename_no_ext="${1%.*}"

rm -f $filename_no_ext".mlir" $filename_no_ext"_res.mlir" $filename_no_ext"_res.bc" $filename_no_ext
echo "Running Flang"
flang-new -fc1 -emit-hlfir -mmlir -mlir-print-op-generic $1
if [ -f $filename_no_ext".mlir" ]; then
	echo "  -> Completed, results in '$filename_no_ext.mlir'"
	echo "Preprocessing to xDSL compatible form"
	preprocess_mlir_for_xdsl $filename_no_ext.mlir $filename_no_ext"_pp".mlir
	if [ -f $filename_no_ext"_pp.mlir" ]; then
	echo "  -> Completed, results in '"$filename_no_ext"_pp.mlir'"
		echo "Lowering to standard dialects"
		ftn-opt $filename_no_ext"_pp.mlir" -p rewrite-fir-to-standard -o $filename_no_ext"_res.mlir"
		if [ -f $filename_no_ext"_res.mlir" ]; then
			echo "  -> Completed, results in '"$filename_no_ext"_res.mlir'"
			echo "Generating LLVM-IR"
			mlir-opt --pass-pipeline="builtin.module(loop-invariant-code-motion, convert-scf-to-cf, convert-cf-to-llvm{index-bitwidth=64}, fold-memref-alias-ops,lower-affine,finalize-memref-to-llvm, convert-arith-to-llvm{index-bitwidth=64}, convert-func-to-llvm, math-uplift-to-fma, convert-math-to-llvm, fold-memref-alias-ops,lower-affine,finalize-memref-to-llvm, reconcile-unrealized-casts)" $filename_no_ext"_res.mlir" | mlir-translate --mlir-to-llvmir -o $filename_no_ext"_res.bc"
			if [ -f $filename_no_ext"_res.bc" ]; then
				echo "  -> Completed, results in '"$filename_no_ext"_res.bc'"
				echo "Building executable"
				flang-new -O3 -o $filename_no_ext $filename_no_ext"_res.bc"
				if [ -f $filename_no_ext ]; then
					echo "  -> Completed, executable in '$filename_no_ext'"
				fi
			fi
		fi
	fi
fi

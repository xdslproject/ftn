#!/bin/bash

ODIR=fpga
CLKPERIOD=10
TARGET_BOARD=xc7vx690t-ffg1930-3
KERNELNAME=$1
LIBDIR=/opt/soda-opt/lib/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source $SCRIPT_DIR/config.sh
shopt -s expand_aliases

soda-opt ex1_offload.mlir --lower-affine --canonicalize --lower-all-to-llvm=use-bare-ptr-memref-call-conv | \
soda-mlir-translate --mlir-to-llvmir --opaque-pointers=0 -o model.ll

opt-16 model.ll \
  -S \
  -enable-new-pm=0 \
  -load "${LIBDIR}/VhlsLLVMRewriter.so" \
  -mem2arr -strip-debug \
  -instcombine \
  -xlnname \
  -xlnanno -xlntop $KERNELNAME \
  -xlntbgen -xlntbdummynames="$KERNELNAME.dummy.c" \
  -xlntbtclnames="$KERNELNAME.run.tcl" \
  -xlnllvm="$KERNELNAME.opt.ll"  \
  -clock-period-ns=$CLKPERIOD -target=$TARGET_BOARD \
  > $KERNELNAME.opt.ll

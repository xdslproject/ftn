from abc import ABC
from enum import Enum
import itertools
import copy
from functools import reduce
from typing import TypeVar, cast
from dataclasses import dataclass
from xdsl.dialects.experimental import fir, hlfir
from dataclasses import dataclass, field
from typing import Dict, Optional
from xdsl.ir import SSAValue, BlockArgument
from xdsl.irdl import Operand
from xdsl.utils.hints import isa
from ftn.util.visitor import Visitor
from xdsl.context import Context
from xdsl.ir import Operation, SSAValue, OpResult, Attribute, Block, Region

from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter,
    op_type_rewrite_pattern,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
)
from xdsl.passes import ModulePass
from xdsl.dialects import builtin, func, llvm, arith, memref, scf, cf, linalg, omp, math


def convert_fir_global_linkname_to_memref_visibility(linkname):
    if linkname == "external":
        return "public"
    elif linkname == "internal":
        return "private"
    else:
        assert False


def create_index_constant(val: int):
    return arith.ConstantOp.create(
        properties={"value": builtin.IntegerAttr.from_index_int_value(val)},
        result_types=[builtin.IndexType()],
    )


def generate_extract_ptr_from_memref(memref_ssa):
    assert isa(memref_ssa.type, builtin.MemRefType)
    extract_ptr_as_idx_op = memref.ExtractAlignedPointerAsIndexOp.get(memref_ssa)
    i64_idx_op = arith.IndexCastOp(extract_ptr_as_idx_op.results[0], builtin.i64)
    ptr_op = llvm.IntToPtrOp(i64_idx_op.results[0])
    return [extract_ptr_as_idx_op, i64_idx_op, ptr_op], ptr_op.results[0]


def generate_dereference_memref(memref_ssa):
    load_op = memref.LoadOp.get(memref_ssa, [])
    return load_op, load_op.results[0]


def clean_func_name(func_name: str):
    if "_QP" in func_name:
        return func_name.split("_QP")[1]
    elif "_QM" in func_name:
        return func_name.split("_QM")[1]
    else:
        return func_name


def check_if_has_type(match_type, type_to_search):
    if isa(type_to_search, match_type):
        return True
    elif isa(type_to_search, memref.MemRefType):
        return check_if_has_type(match_type, type_to_search.element_type)
    return False


def remove_tuple_type_from_memref(src_type):
    # This is a hack as Flang generates a fir.ref with tuple, however
    # this is not allowed in a memref. Therefore we find the most
    # significant member of the tuple and build memref from that
    if isa(src_type, builtin.TupleType):
        for ty in src_type.types:
            if isa(ty, memref.MemRefType):
                return ty
        return src_type.types[0]
    elif isa(src_type, memref.MemRefType):
        base_type = remove_tuple_type_from_memref(src_type.element_type)
        return memref.MemRefType(
            base_type, src_type.shape, builtin.NoneAttr(), builtin.NoneAttr()
        )
    else:
        return src_type

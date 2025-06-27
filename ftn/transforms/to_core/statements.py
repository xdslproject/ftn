from xdsl.dialects.experimental import fir, hlfir
from xdsl.utils.hints import isa
from xdsl.ir import Operation
from xdsl.dialects import func, llvm, arith, cf

from ftn.transforms.to_core.misc.fortran_code_description import ProgramState
from ftn.transforms.to_core.misc.ssa_context import SSAValueCtx

import ftn.transforms.to_core.components.memory as ftn_memory
import ftn.transforms.to_core.components.load_store as ftn_load_store
import ftn.transforms.to_core.components.functions as ftn_functions
import ftn.transforms.to_core.components.control_flow as ftn_ctrl_flow
import ftn.transforms.to_core.expressions as expressions


def translate_stmt(program_state: ProgramState, ctx: SSAValueCtx, op: Operation):
    ops = try_translate_stmt(program_state, ctx, op)
    if ops is not None:
        return ops
    return []
    # raise Exception(f"Could not translate `{op}' as a definition or statement")


def try_translate_stmt(program_state: ProgramState, ctx: SSAValueCtx, op: Operation):
    if isa(op, hlfir.DeclareOp):
        return ftn_memory.translate_declare(program_state, ctx, op)
    elif isa(op, fir.DoLoopOp):
        return ftn_ctrl_flow.translate_do_loop(program_state, ctx, op)
    elif isa(op, fir.IterateWhileOp):
        return ftn_ctrl_flow.translate_iterate_while(program_state, ctx, op)
    elif isa(op, fir.AllocaOp):
        # Will only process this if it is an internal flag,
        # otherwise pick up as part of the declareop
        return ftn_memory.translate_alloca(program_state, ctx, op)
    elif isa(op, arith.ConstantOp):
        return []
    elif isa(op, fir.HasValueOp):
        return expressions.translate_expr(program_state, ctx, op.resval)
    elif isa(op, fir.LoadOp):
        return []
    elif (
        isa(op, arith.AddiOp)
        or isa(op, arith.SubiOp)
        or isa(op, arith.MuliOp)
        or isa(op, arith.DivUIOp)
        or isa(op, arith.DivSIOp)
        or isa(op, arith.FloorDivSIOp)
        or isa(op, arith.CeilDivSIOp)
        or isa(op, arith.CeilDivUIOp)
        or isa(op, arith.RemUIOp)
        or isa(op, arith.RemSIOp)
        or isa(op, arith.MinUIOp)
        or isa(op, arith.MaxUIOp)
        or isa(op, arith.MinSIOp)
        or isa(op, arith.MaxSIOp)
        or isa(op, arith.AndIOp)
        or isa(op, arith.OrIOp)
        or isa(op, arith.XOrIOp)
        or isa(op, arith.ShLIOp)
        or isa(op, arith.ShRUIOp)
        or isa(op, arith.ShRSIOp)
        or isa(op, arith.AddUIExtendedOp)
    ):
        return []
    elif (
        isa(op, arith.AddfOp)
        or isa(op, arith.SubfOp)
        or isa(op, arith.MulfOp)
        or isa(op, arith.DivfOp)
        or isa(op, arith.MaximumfOp)
        or isa(op, arith.MaxnumfOp)
        or isa(op, arith.MinimumfOp)
        or isa(op, arith.MinnumfOp)
    ):
        return []
    elif isa(op, func.ReturnOp):
        return ftn_ctrl_flow.translate_return(program_state, ctx, op)
    elif isa(op, hlfir.AssignOp):
        return ftn_load_store.translate_assign(program_state, ctx, op)
    elif isa(op, fir.StoreOp):
        # Used internally by some ops still, e.g. to store loop bounds per iteration
        return ftn_load_store.translate_store(program_state, ctx, op)
    elif isa(op, fir.ResultOp):
        return ftn_ctrl_flow.translate_result(program_state, ctx, op)
    elif isa(op, fir.CallOp):
        return ftn_functions.translate_call(program_state, ctx, op)
    elif isa(op, fir.FreememOp):
        return ftn_memory.translate_freemem(program_state, ctx, op)
    elif isa(op, fir.IfOp):
        return ftn_ctrl_flow.translate_conditional(program_state, ctx, op)
    elif isa(op, cf.BranchOp):
        return ftn_ctrl_flow.translate_branch(program_state, ctx, op)
    # elif isa(op, omp.TargetOp):
    #  return translate_omp_target(program_state, ctx, op)
    # elif isa(op, omp.TerminatorOp):
    #  return [omp.TerminatorOp.create()]
    # elif isa(op, omp.SIMDLoopOp):
    #  return translate_omp_simdloop(program_state, ctx, op)
    # elif isa(op, omp.TeamsOp):
    #  return translate_omp_team(program_state, ctx, op)
    # elif isa(op, omp.ParallelOp):
    #  return translate_omp_parallel(program_state, ctx, op)
    # elif isa(op, omp.YieldOp):
    #  return [omp.YieldOp.create()]
    elif isa(op, cf.ConditionalBranchOp):
        return ftn_ctrl_flow.translate_conditional_branch(program_state, ctx, op)
    elif isa(op, fir.UnreachableOp):
        return [llvm.UnreachableOp()]
    else:
        return None

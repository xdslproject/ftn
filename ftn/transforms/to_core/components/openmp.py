from xdsl.ir import Block, Region
from xdsl.utils.hints import isa
from xdsl.dialects import builtin, omp

from ftn.transforms.to_core.misc.fortran_code_description import ProgramState
from ftn.transforms.to_core.misc.ssa_context import SSAValueCtx

from ftn.transforms.to_core.utils import (
    create_index_constant,
    generate_extract_ptr_from_memref,
)

import ftn.transforms.to_core.expressions as expressions
import ftn.transforms.to_core.statements as statements


def handle_var_operand_field(program_state, ctx, var_operands):
    arg_types = []
    vars_ops = []
    vars_ssa = []

    if len(var_operands) > 0:
        for operand in var_operands:
            vars_ops += expressions.translate_expr(program_state, ctx, operand)
            vars_ssa.append(ctx[operand])
            arg_types.append(ctx[operand].type)

    return vars_ops, vars_ssa, arg_types


def handle_opt_operand_field(program_state, ctx, opt_operand):
    arg_types = []
    vars_ops = []
    vars_ssa = []

    if opt_operand is not None:
        vars_ops += expressions.translate_expr(program_state, ctx, opt_operand)
        vars_ssa = ctx[opt_operand]
        arg_types.append(ctx[opt_operand].type)

    return vars_ops, vars_ssa, arg_types


def duplicate_op_properties(op):
    new_props = {}
    for key, value in op.properties.items():
        if key != "operandSegmentSizes":
            new_props[key] = value
    return new_props


def create_block_and_properties_for_op(
    program_state: ProgramState,
    ctx: SSAValueCtx,
    op,
    arg_types,
    region,
    hoist_loop_indexes,
):
    block = region.blocks[0]

    new_block = Block(arg_types=arg_types)

    for fir_arg, std_arg in zip(block.args, new_block.args):
        ctx[fir_arg] = std_arg

    region_body_ops = []
    top_level_ops = []
    if hoist_loop_indexes:
        assert len(block.ops) == 1
        region_body_ops = statements.translate_stmt(program_state, ctx, block.ops.first)
        if len(region_body_ops) > 1:
            # The loopnest op will pull down the lower, upper and step constants into this region,
            # however they need to sit above simd as otherwise verification will fail
            # therefore extract out all but the last operation (the loop nest)
            top_level_ops = region_body_ops[0:-1]
            region_body_ops = [region_body_ops[-1]]

        assert len(region_body_ops) == 1
        assert (
            isa(region_body_ops[0], omp.LoopNestOp)
            or isa(region_body_ops[0], omp.WsLoopOp)
            or isa(region_body_ops[0], omp.SIMDOp)
            or isa(region_body_ops[0], omp.DistributeOp)
        )
    else:
        for single_op in block.ops:
            region_body_ops += statements.translate_stmt(program_state, ctx, single_op)

    new_block.add_ops(region_body_ops)

    new_props = duplicate_op_properties(op)

    return new_block, top_level_ops, new_props


def generate_op_region(program_state: ProgramState, ctx: SSAValueCtx, region: Region):
    if len(region.blocks) > 0:
        arg_types = []
        for arg in region.blocks[0].args:
            arg_types.append(arg.type)

        new_block = Block(arg_types=arg_types)

        for fir_arg, std_arg in zip(region.blocks[0].args, new_block.args):
            ctx[fir_arg] = std_arg

        region_ops = []
        for single_op in region.blocks[0].ops:
            region_ops += statements.translate_stmt(program_state, ctx, single_op)

        new_block.add_ops(region_ops)
        region_blocks = [new_block]
    else:
        region_blocks = []
    return region_blocks


def translate_declarereduction(
    program_state: ProgramState, ctx: SSAValueCtx, op: omp.DeclareReductionOp
):
    alloc_region_blocks = generate_op_region(program_state, ctx, op.alloc_region)
    init_region_blocks = generate_op_region(program_state, ctx, op.init_region)
    combiner_region_blocks = generate_op_region(program_state, ctx, op.combiner_region)
    atomic_region_blocks = generate_op_region(program_state, ctx, op.atomic_region)
    cleanup_region_blocks = generate_op_region(program_state, ctx, op.cleanup_region)

    return omp.DeclareReductionOp.build(
        regions=[
            Region(alloc_region_blocks),
            Region(init_region_blocks),
            Region(combiner_region_blocks),
            Region(atomic_region_blocks),
            Region(cleanup_region_blocks),
        ],
        properties={
            "sym_name": op.sym_name,
            "type": op.var_type,
        },
    )


def translate_private(program_state: ProgramState, ctx: SSAValueCtx, op: omp.PrivateClauseOp):
    if len(op.alloc_region.blocks) > 0:
        alloc_region_ops = []
        for single_op in op.alloc_region.blocks[0].ops:
            alloc_region_ops += statements.translate_stmt(program_state, ctx, single_op)
        alloc_region_blocks = [Block(alloc_region_ops)]
    else:
        alloc_region_blocks = []

    if len(op.copy_region.blocks) > 0:
        copy_region_ops = []
        for single_op in op.copy_region.blocks[0].ops:
            copy_region_ops += statements.translate_stmt(program_state, ctx, single_op)
        copy_region_blocks = [Block(copy_region_ops)]
    else:
        copy_region_blocks = []

    if len(op.dealloc_region.blocks) > 0:
        dealloc_region_ops = []
        for single_op in op.dealloc_region.blocks[0].ops:
            dealloc_region_ops += statements.translate_stmt(
                program_state, ctx, single_op
            )
        dealloc_region_blocks = [Block(dealloc_region_ops)]
    else:
        dealloc_region_blocks = []

    return omp.PrivateOp.build(
        regions=[
            Region(alloc_region_blocks),
            Region(copy_region_blocks),
            Region(dealloc_region_blocks),
        ],
        properties={
            "sym_name": op.sym_name,
            "type": op.var_type,
            "data_sharing_type": op.data_sharing_type,
        },
    )


def translate_omp_mapinfo(
    program_state: ProgramState, ctx: SSAValueCtx, op: omp.MapInfoOp
):
    if ctx.contains(op.results[0]):
        return []

    var_ptr_ops = expressions.translate_expr(program_state, ctx, op.var_ptr)
    var_ptr_ssa = ctx[op.var_ptr]
    var_ptr_type = ctx[op.var_ptr].type

    var_ptr_ptr_ops = []
    var_ptr_ptr_ssa = []
    if op.var_ptr_ptr is not None:
        var_ptr_ptr_ops = expressions.translate_expr(program_state, ctx, op.var_ptr_ptr)
        var_ptr_ptr_ssa = [ctx[op.var_ptr_ptr]]

    members_ops = []
    members_ssa = []
    for arg in op.members:
        members_ops += expressions.translate_expr(program_state, ctx, arg)
        members_ssa.append(ctx[arg])

    bounds_ops = []
    bounds_ssa = []
    for arg in op.bounds:
        bounds_ops += expressions.translate_expr(program_state, ctx, arg)
        bounds_ssa.append(bounds_ops[-1].results[0])

    new_props = duplicate_op_properties(op)

    mapinfo_op = omp.MapInfoOp.build(
        operands=[var_ptr_ssa, var_ptr_ptr_ssa, members_ssa, bounds_ssa],
        properties=new_props,
        result_types=[var_ptr_type],
    )

    ctx[op.results[0]] = mapinfo_op.results[0]

    return var_ptr_ops + var_ptr_ptr_ops + members_ops + bounds_ops + [mapinfo_op]


def translate_omp_bounds(
    program_state: ProgramState, ctx: SSAValueCtx, op: omp.MapBoundsOp
):
    if ctx.contains(op.results[0]):
        return []

    lower_ops = expressions.translate_expr(program_state, ctx, op.lower_bound)
    lower_ssa = ctx[op.lower_bound]

    upper_ops = expressions.translate_expr(program_state, ctx, op.upper_bound)
    upper_ssa = ctx[op.upper_bound]

    extent_ops = expressions.translate_expr(program_state, ctx, op.extent)
    extent_ssa = ctx[op.extent]

    stride_ops = expressions.translate_expr(program_state, ctx, op.stride)
    stride_ssa = ctx[op.stride]

    start_ops = expressions.translate_expr(program_state, ctx, op.start_idx)
    start_ssa = ctx[op.start_idx]

    new_props = duplicate_op_properties(op)

    bounds_op = omp.MapBoundsOp.build(
        operands=[lower_ssa, upper_ssa, extent_ssa, stride_ssa, start_ssa],
        properties=new_props,
        result_types=[omp.MapBoundsType()],
    )

    ctx[op.results[0]] = bounds_op.results[0]

    return lower_ops + upper_ops + extent_ops + stride_ops + start_ops + [bounds_op]


def translate_omp_wsloop(
    program_state: ProgramState, ctx: SSAValueCtx, op: omp.WsLoopOp
):
    arg_types = []

    allocate_vars_ops, allocate_vars_ssa, allocate_vars_types = (
        handle_var_operand_field(program_state, ctx, op.allocate_vars)
    )
    arg_types += allocate_vars_types

    allocator_vars_ops, allocator_vars_ssa, allocator_vars_types = (
        handle_var_operand_field(program_state, ctx, op.allocator_vars)
    )
    arg_types += allocator_vars_types

    linear_vars_ops, linear_vars_ssa, linear_vars_types = handle_var_operand_field(
        program_state, ctx, op.linear_vars
    )
    arg_types += linear_vars_types

    linear_step_vars_ops, linear_step_vars_ssa, linear_step_vars_types = (
        handle_var_operand_field(program_state, ctx, op.linear_step_vars)
    )
    arg_types += linear_step_vars_types

    private_vars_ops, private_vars_ssa, private_vars_types = handle_var_operand_field(
        program_state, ctx, op.private_vars
    )
    arg_types += private_vars_types

    reduction_vars_ops, reduction_vars_ssa, reduction_vars_types = (
        handle_var_operand_field(program_state, ctx, op.reduction_vars)
    )

    # OpenMP can not handle memref in wsloop arguments, which are used for reduction
    # args. Therefore convert these into LLVM pointers which are passed in, these
    # can then be loaded and worked with similarly to memref
    new_reduction_ssas = reduction_vars_ssa
    for idx, reduction_var_ssa in enumerate(reduction_vars_ssa):
        if isa(reduction_var_ssa.type, builtin.MemRefType):
            extract_ops, extract_ssa = generate_extract_ptr_from_memref(
                reduction_var_ssa
            )
            reduction_vars_ops += extract_ops
            new_reduction_ssas[idx] = extract_ssa
            reduction_vars_types[idx] = extract_ssa.type

    reduction_vars_ssa = new_reduction_ssas
    arg_types += reduction_vars_types

    schedule_chunk_ops, schedule_chunk_ssa, schedule_chunk_types = (
        handle_opt_operand_field(program_state, ctx, op.schedule_chunk)
    )
    arg_types += schedule_chunk_types

    new_block, top_level_ops, new_props = create_block_and_properties_for_op(
        program_state, ctx, op, arg_types, op.body, True
    )

    omp_wsloop = omp.WsLoopOp.build(
        operands=[
            allocate_vars_ssa,
            allocator_vars_ssa,
            linear_vars_ssa,
            linear_step_vars_ssa,
            private_vars_ssa,
            reduction_vars_ssa,
            schedule_chunk_ssa,
        ],
        regions=[Region([new_block])],
        properties=new_props,
    )

    return (
        allocate_vars_ops
        + allocator_vars_ops
        + linear_vars_ops
        + linear_step_vars_ops
        + private_vars_ops
        + reduction_vars_ops
        + schedule_chunk_ops
        + top_level_ops
        + [omp_wsloop]
    )


def translate_omp_parallel(
    program_state: ProgramState, ctx: SSAValueCtx, op: omp.ParallelOp
):
    arg_types = []

    allocate_vars_ops, allocate_vars_ssa, allocate_vars_types = (
        handle_var_operand_field(program_state, ctx, op.allocate_vars)
    )
    arg_types += allocate_vars_types

    allocators_vars_ops, allocators_vars_ssa, allocators_vars_types = (
        handle_var_operand_field(program_state, ctx, op.allocators_vars)
    )
    arg_types += allocators_vars_types

    if_expr_ops, if_expr_ssa, if_expr_types = handle_opt_operand_field(
        program_state, ctx, op.if_expr
    )
    arg_types += if_expr_types

    num_threads_ops, num_threads_ssa, num_threads_types = handle_opt_operand_field(
        program_state, ctx, op.num_threads
    )
    arg_types += num_threads_types

    private_vars_ops, private_vars_ssa, private_vars_types = handle_var_operand_field(
        program_state, ctx, op.private_vars
    )
    arg_types += private_vars_types

    reduction_vars_ops, reduction_vars_ssa, reduction_vars_types = (
        handle_var_operand_field(program_state, ctx, op.reduction_vars)
    )
    arg_types += reduction_vars_types

    new_block, __, new_props = create_block_and_properties_for_op(
        program_state, ctx, op, arg_types, op.region, False
    )

    omp_parallel = omp.ParallelOp.build(
        operands=[
            allocate_vars_ssa,
            allocators_vars_ssa,
            if_expr_ssa,
            num_threads_ssa,
            private_vars_ssa,
            reduction_vars_ssa,
        ],
        regions=[Region([new_block])],
        properties=new_props,
    )

    return (
        allocate_vars_ops
        + allocators_vars_ops
        + if_expr_ops
        + num_threads_ops
        + private_vars_ops
        + reduction_vars_ops
        + [omp_parallel]
    )


def translate_omp_loopnest(
    program_state: ProgramState, ctx: SSAValueCtx, op: omp.LoopNestOp
):
    arg_types = []

    assert len(op.lowerBound) == len(op.upperBound) == len(op.step)

    operands_ssa = [[], [], []]
    ops_list = []
    for lower, upper, step in zip(op.lowerBound, op.upperBound, op.step):
        lb_ops = expressions.translate_expr(program_state, ctx, lower)
        ub_ops = expressions.translate_expr(program_state, ctx, upper)
        step_ops = expressions.translate_expr(program_state, ctx, step)
        operands_ssa[0].append(ctx[lower])
        operands_ssa[1].append(ctx[upper])
        operands_ssa[2].append(ctx[step])

        ops_list += lb_ops + ub_ops + step_ops

    new_block, __, new_props = create_block_and_properties_for_op(
        program_state, ctx, op, [builtin.i32] * len(op.lowerBound), op.body, False
    )

    loopnest_op = omp.LoopNestOp.build(
        operands=operands_ssa,
        regions=[Region([new_block])],
        properties=new_props,
    )

    return ops_list + [loopnest_op]


def translate_omp_teams(program_state: ProgramState, ctx: SSAValueCtx, op: omp.TeamsOp):
    arg_types = []

    allocate_vars_ops, allocate_vars_ssa, allocate_vars_types = (
        handle_var_operand_field(program_state, ctx, op.allocate_vars)
    )
    arg_types += allocate_vars_types

    allocator_vars_ops, allocator_vars_ssa, allocator_vars_types = (
        handle_var_operand_field(program_state, ctx, op.allocator_vars)
    )
    arg_types += allocator_vars_types

    if_expr_ops, if_expr_ssa, if_expr_types = handle_opt_operand_field(
        program_state, ctx, op.if_expr
    )
    arg_types += if_expr_types

    num_teams_lower_ops, num_teams_lower_ssa, num_teams_lower_types = (
        handle_opt_operand_field(program_state, ctx, op.num_teams_lower)
    )
    arg_types += num_teams_lower_types

    num_teams_upper_ops, num_teams_upper_ssa, num_teams_upper_types = (
        handle_opt_operand_field(program_state, ctx, op.num_teams_upper)
    )
    arg_types += num_teams_upper_types

    private_vars_ops, private_vars_ssa, private_vars_types = handle_var_operand_field(
        program_state, ctx, op.private_vars
    )
    arg_types += private_vars_types

    reduction_vars_ops, reduction_vars_ssa, reduction_vars_types = (
        handle_var_operand_field(program_state, ctx, op.reduction_vars)
    )
    arg_types += reduction_vars_types

    thread_limit_ops, thread_limit_ssa, thread_limit_types = handle_opt_operand_field(
        program_state, ctx, op.thread_limit
    )
    arg_types += thread_limit_types

    new_block, __, new_props = create_block_and_properties_for_op(
        program_state, ctx, op, arg_types, op.body, False
    )

    teams_op = omp.TeamsOp.build(
        operands=[
            allocate_vars_ssa,
            allocator_vars_ssa,
            if_expr_ssa,
            num_teams_lower_ssa,
            num_teams_upper_ssa,
            private_vars_ssa,
            reduction_vars_ssa,
            thread_limit_ssa,
        ],
        regions=[Region([new_block])],
        properties=new_props,
    )
    return (
        allocate_vars_ops
        + allocator_vars_ops
        + if_expr_ops
        + num_teams_lower_ops
        + num_teams_upper_ops
        + private_vars_ops
        + reduction_vars_ops
        + thread_limit_ops
        + [teams_op]
    )


def translate_omp_distribute(
    program_state: ProgramState, ctx: SSAValueCtx, op: omp.DistributeOp
):
    arg_types = []

    allocate_vars_ops, allocate_vars_ssa, allocate_vars_types = (
        handle_var_operand_field(program_state, ctx, op.allocate_vars)
    )
    arg_types += allocate_vars_types

    allocator_vars_ops, allocator_vars_ssa, allocator_vars_types = (
        handle_var_operand_field(program_state, ctx, op.allocator_vars)
    )
    arg_types += allocator_vars_types

    (
        dist_schedule_chunk_size_ops,
        dist_schedule_chunk_size_ssa,
        dist_schedule_chunk_size_types,
    ) = handle_opt_operand_field(program_state, ctx, op.dist_schedule_chunk_size)
    arg_types += dist_schedule_chunk_size_types

    private_vars_ops, private_vars_ssa, private_vars_types = handle_var_operand_field(
        program_state, ctx, op.private_vars
    )
    arg_types += private_vars_types

    new_block, top_level_ops, new_props = create_block_and_properties_for_op(
        program_state, ctx, op, arg_types, op.body, True
    )

    distribute_op = omp.DistributeOp.build(
        operands=[
            allocate_vars_ssa,
            allocator_vars_ssa,
            dist_schedule_chunk_size_ssa,
            private_vars_types,
        ],
        regions=[Region([new_block])],
        properties=new_props,
    )

    return (
        allocate_vars_ops
        + allocator_vars_ops
        + dist_schedule_chunk_size_ops
        + private_vars_ops
        + top_level_ops
        + [distribute_op]
    )


def translate_omp_simd(program_state: ProgramState, ctx: SSAValueCtx, op: omp.SimdOp):
    arg_types = []

    aligned_vars_ops, aligned_vars_ssa, aligned_var_types = handle_var_operand_field(
        program_state, ctx, op.aligned_vars
    )
    arg_types += aligned_var_types

    if_expr_ops, if_expr_ssa, if_expr_types = handle_opt_operand_field(
        program_state, ctx, op.if_expr
    )
    arg_types += if_expr_types

    linear_vars_ops, linear_vars_ssa, linear_vars_types = handle_var_operand_field(
        program_state, ctx, op.linear_vars
    )
    arg_types += linear_vars_types

    linear_step_vars_ops, linear_step_vars_ssa, linear_step_vars_types = (
        handle_var_operand_field(program_state, ctx, op.linear_step_vars)
    )
    arg_types += linear_step_vars_types

    nontemporal_vars_ops, nontemporal_vars_ssa, nontemporal_vars_types = (
        handle_opt_operand_field(program_state, ctx, op.nontemporal_vars)
    )
    arg_types += nontemporal_vars_types

    private_vars_ops, private_vars_ssa, private_vars_types = handle_opt_operand_field(
        program_state, ctx, op.private_vars
    )
    arg_types += private_vars_types

    reduction_vars_ops, reduction_vars_ssa, reduction_vars_types = (
        handle_opt_operand_field(program_state, ctx, op.reduction_vars)
    )
    arg_types += reduction_vars_types

    new_block, top_level_ops, new_props = create_block_and_properties_for_op(
        program_state, ctx, op, arg_types, op.body, True
    )

    simd_op = omp.SIMDOp.build(
        operands=[
            aligned_vars_ssa,
            if_expr_ssa,
            linear_vars_ssa,
            linear_step_vars_ssa,
            nontemporal_vars_ssa,
            private_vars_ssa,
            reduction_vars_ssa,
        ],
        regions=[Region([new_block])],
        properties=new_props,
    )

    return (
        aligned_vars_ops
        + if_expr_ops
        + linear_vars_ops
        + linear_step_vars_ops
        + nontemporal_vars_ops
        + private_vars_ops
        + reduction_vars_ops
        + top_level_ops
        + [simd_op]
    )


def translate_omp_target(
    program_state: ProgramState, ctx: SSAValueCtx, op: omp.TargetOp
):
    arg_types = []

    allocate_vars_ops, allocate_vars_ssa, allocate_vars_types = (
        handle_var_operand_field(program_state, ctx, op.allocate_vars)
    )
    arg_types += allocate_vars_types

    allocator_vars_ops, allocator_vars_ssa, allocator_vars_types = (
        handle_var_operand_field(program_state, ctx, op.allocator_vars)
    )
    arg_types += allocator_vars_types

    depend_vars_ops, depend_vars_ssa, depend_vars_types = handle_var_operand_field(
        program_state, ctx, op.depend_vars
    )
    arg_types += depend_vars_types

    device_ops, device_ssa, device_types = handle_opt_operand_field(
        program_state, ctx, op.device
    )
    arg_types += device_types

    has_device_addr_vars_ops, has_device_addr_vars_ssa, has_device_addr_vars_types = (
        handle_var_operand_field(program_state, ctx, op.has_device_addr_vars)
    )
    arg_types += has_device_addr_vars_types

    host_eval_vars_ops, host_eval_vars_ssa, host_eval_vars_types = (
        handle_var_operand_field(program_state, ctx, op.host_eval_vars)
    )
    arg_types += host_eval_vars_types

    if_expr_vars_ops, if_expr_vars_ssa, if_expr_vars_types = handle_opt_operand_field(
        program_state, ctx, op.if_expr
    )
    arg_types += if_expr_vars_types

    in_reduction_vars_ops, in_reduction_vars_ssa, in_reduction_vars_types = (
        handle_var_operand_field(program_state, ctx, op.in_reduction_vars)
    )
    arg_types += in_reduction_vars_types

    is_device_ptr_vars_ops, is_device_ptr_vars_ssa, is_device_ptr_vars_types = (
        handle_var_operand_field(program_state, ctx, op.is_device_ptr_vars)
    )
    arg_types += is_device_ptr_vars_types

    map_vars_ops, map_vars_ssa, map_vars_types = handle_var_operand_field(
        program_state, ctx, op.map_vars
    )
    arg_types += map_vars_types

    private_vars_ops, private_vars_ssa, private_vars_types = handle_var_operand_field(
        program_state, ctx, op.private_vars
    )
    arg_types += private_vars_types

    thread_limit_ops, thread_limit_ssa, thread_limit_types = handle_opt_operand_field(
        program_state, ctx, op.thread_limit
    )
    arg_types += thread_limit_types

    new_block, __, new_props = create_block_and_properties_for_op(
        program_state, ctx, op, arg_types, op.region, False
    )

    target_op = omp.TargetOp.build(
        operands=[
            allocate_vars_ssa,
            allocator_vars_ssa,
            depend_vars_ssa,
            device_ssa,
            has_device_addr_vars_ssa,
            host_eval_vars_ssa,
            if_expr_vars_ssa,
            in_reduction_vars_ssa,
            is_device_ptr_vars_ssa,
            map_vars_ssa,
            private_vars_ssa,
            thread_limit_ssa,
        ],
        regions=[Region([new_block])],
        properties=new_props,
    )

    return (
        allocate_vars_ops
        + allocator_vars_ops
        + depend_vars_ops
        + device_ops
        + has_device_addr_vars_ops
        + host_eval_vars_ops
        + if_expr_vars_ops
        + in_reduction_vars_ops
        + is_device_ptr_vars_ops
        + map_vars_ops
        + private_vars_ops
        + [target_op]
    )


def translate_omp_target_data(
    program_state: ProgramState, ctx: SSAValueCtx, op: omp.TargetDataOp
):
    arg_types = []

    device_ops, device_ssa, device_types = handle_opt_operand_field(
        program_state, ctx, op.device
    )
    arg_types += device_types

    if_expr_ops, if_expr_ssa, if_expr_types = handle_opt_operand_field(
        program_state, ctx, op.if_expr
    )
    arg_types += if_expr_types

    map_vars_ops, map_vars_ssa, map_vars_types = handle_var_operand_field(
        program_state, ctx, op.map_vars
    )
    arg_types += map_vars_types

    use_device_addr_vars_ops, use_device_addr_vars_ssa, use_device_addr_vars_types = (
        handle_var_operand_field(program_state, ctx, op.use_device_addr_vars)
    )
    arg_types += use_device_addr_vars_types

    use_device_ptr_vars_ops, use_device_ptr_vars_ssa, use_device_ptr_vars_types = (
        handle_var_operand_field(program_state, ctx, op.use_device_ptr_vars)
    )
    arg_types += use_device_ptr_vars_types

    new_block, __, new_props = create_block_and_properties_for_op(
        program_state, ctx, op, arg_types, op.region, False
    )

    target_data_op = omp.TargetDataOp.build(
        operands=[
            device_ssa,
            if_expr_ssa,
            map_vars_ssa,
            use_device_addr_vars_ssa,
            use_device_ptr_vars_ssa,
        ],
        regions=[Region([new_block])],
        properties=new_props,
    )

    return (
        device_ops
        + if_expr_ops
        + map_vars_ops
        + use_device_addr_vars_ops
        + use_device_ptr_vars_ops
        + [target_data_op]
    )


def translate_omp_target_task_based_data_movement(
    program_state: ProgramState,
    ctx: SSAValueCtx,
    op: omp.TargetEnterDataOp | omp.TargetExitDataOp | omp.TargetUpdateOp,
):
    arg_types = []

    depend_vars_ops, depend_vars_ssa, depend_vars_types = handle_var_operand_field(
        program_state, ctx, op.depend_vars
    )
    arg_types += depend_vars_types

    device_ops, device_ssa, device_types = handle_opt_operand_field(
        program_state, ctx, op.device
    )
    arg_types += device_types

    if_expr_ops, if_expr_ssa, if_expr_types = handle_opt_operand_field(
        program_state, ctx, op.if_expr
    )
    arg_types += if_expr_types

    map_vars_ops, map_vars_ssa, map_vars_types = handle_var_operand_field(
        program_state, ctx, op.map_vars
    )
    arg_types += map_vars_types

    new_props = duplicate_op_properties(op)

    target_update_op = type(op).build(
        operands=[
            depend_vars_ssa,
            device_ssa,
            if_expr_ssa,
            map_vars_ssa,
        ],
        properties=new_props,
    )

    return (
        depend_vars_ops + device_ops + if_expr_ops + map_vars_ops + [target_update_op]
    )


def translate_omp_target_enter_data(
    program_state: ProgramState, ctx: SSAValueCtx, op: omp.TargetEnterDataOp
):
    return translate_omp_target_task_based_data_movement(program_state, ctx, op)


def translate_omp_target_exit_data(
    program_state: ProgramState, ctx: SSAValueCtx, op: omp.TargetExitDataOp
):
    return translate_omp_target_task_based_data_movement(program_state, ctx, op)


def translate_omp_target_update(
    program_state: ProgramState, ctx: SSAValueCtx, op: omp.TargetUpdateOp
):
    return translate_omp_target_task_based_data_movement(program_state, ctx, op)


def translate_omp_yield(program_state: ProgramState, ctx: SSAValueCtx, op: omp.YieldOp):
    ops_list = []
    ssa_list = []
    for operand in op.arguments:
        expr_ops = expressions.translate_expr(program_state, ctx, operand)
        ops_list += expr_ops
        ssa_list.append(ctx[operand])
    return ops_list + [omp.YieldOp(*ssa_list)]

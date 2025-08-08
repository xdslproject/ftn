from __future__ import annotations

from functools import singledispatchmethod

from xdsl.dialects import arith, builtin, memref, func, llvm, scf, affine
from xdsl.dialects.experimental import hls
from xdsl.ir import Operation, SSAValue, BlockArgument
from xdsl.utils.base_printer import BasePrinter
from ftn.dialects import device

RESULT = 0
HLS_STREAM = 1
CONSTANT = 2
HLS_READ = 3
HLS_WRITE = 4
ARGUMENT = 5
LOOP_IDX = 6

class HostPrinter(BasePrinter):
    name_dict: dict[SSAValue, str] = dict()
    count = 0 


    res_counter = 0
    res_names = []
    hls_stream_counter = 0
    hls_stream_names = []

    counters = {RESULT: 0, HLS_STREAM: 0, CONSTANT: 0, HLS_READ: 0, LOOP_IDX: 0}
    names = {RESULT: [], HLS_STREAM: [], CONSTANT: [], HLS_READ: [], LOOP_IDX: []}
    is_hitTile_by_name = set()
    new_hitTile_types = set()

    n_streams_read = {} # how many streams each function has read. map df function name -> n
    n_streams_written = {} # how many streams each function has written. map df function name -> n

    nesting_level = 0
    indent = 0

    trip_vars_per_nesting_level = {0: 'i', 1: 'j', 2: 'k', 3: 'l'}

    ssa_name : dict[SSAValue, str] = dict()
    name_counter = 0

    def gen_name(self, v: SSAValue) -> str:
        # If the value is returned in a yield then parent operation generates a value for it in the outer scope
        # In that case a variable has already been generated for it (e.g. scf.if). This does not apply when the
        # arguments of the block are yielded.
        #for use in v.uses:
        #    if isinstance(use.operation, scf.YieldOp) and not isinstance(v.owner, Block) and isinstance(use.operation.parent_op(), scf.IfOp):
        #        results_parent = use.operation.parent_op().results
        #        return self.ssa_name[results_parent[use.index]]
        if isinstance(v, BlockArgument):
            parent_op = v.block.parent_op()
            assert parent_op is not None
            if "_name" in parent_op.attributes:
                # If the block argument has a name hint, use it
                name_dict = parent_op.attributes["_name"].data
                name = name_dict[f"arg:{v.index}"].data
                self.ssa_name[v] = name

                return name

        name = f"v{self.name_counter}"
        self.ssa_name[v] = name
        self.name_counter += 1

        return name

    def get_name(self, v: SSAValue):
        return self.ssa_name[v]


    def wgsl_name(self, v: SSAValue):
        if v not in self.name_dict:
            if v.name_hint is not None:
                self.name_dict[v] = f"v{v.name_hint}"
            else:
                self.name_dict[v] = f"v{self.count}"
                self.count += 1
        return self.name_dict[v]


    @singledispatchmethod
    def print(self, op: Operation) -> None:
        raise NotImplementedError(
            f"Printing of '{op.name}' to OpenCL is not implemented yet."
        )

    @print.register
    def _(self, op: builtin.ModuleOp):
        self.print_string("#include <stdlib.h>\n")
        self.print_string("#include <stdio.h>\n")

        for o in op.body.ops:
            self.print(o)

    def convert_result_type(result_type):
        c_result_type = ""

        if isinstance(result_type,  builtin.IntegerType):
            if result_type.width.data == 1:
                c_result_type = "bool"
            elif result_type.width.data == 32:
                c_result_type = "int"
            elif result_type.width.data == 64:
                c_result_type = "long"
        elif isinstance(result_type,  builtin.IndexType):
            c_result_type = "int"
        elif isinstance(result_type, hls.HLSStreamType):
            c_result_type = "stream"
        elif isinstance(result_type, builtin.Float32Type):
            c_result_type = "float"
        elif isinstance(result_type, builtin.Float64Type):
            c_result_type = "double"
        elif isinstance(result_type, llvm.LLVMPointerType):
            c_result_type = "__global struct packaged_double * restrict"
        elif isinstance(result_type, memref.MemRefType):
            #memref_type = HostPrinter.convert_result_type(result_type.element_type)
            ##c_result_type = f"__global {memref_type} * const restrict"
            #c_result_type = f"{memref_type}"
            # FIXME: this is only for device side buffers
            c_result_type = "cl_mem"

        return c_result_type


    @print.register
    def _(self, while_op: scf.WhileOp):
        for arg_idx, arg in enumerate(while_op.after_region.blocks[0].args):
            self.gen_name(arg)
            # The argument name must be the same in both regions
            self.ssa_name[while_op.before_region.blocks[0].args[arg_idx]] = self.ssa_name[arg]
            arg_type = HostPrinter.convert_result_type(arg.type)
            value = self.get_name(while_op.arguments[arg_idx])

            self.print_string(self.indent * "\t" + f"{arg_type} {self.ssa_name[arg]} = {value};\n")

        # Declare cyclic variables

        self.print_string("while (true) {\n")

        # Print do region
        for op in while_op.after_region.ops:
            self.print(op)

        # Exit condition - extracted from the before region. Note it needs to be negated in the
        # conditional, since the condition states when we can stay in the loop
        while_cond = None
        for op in while_op.before_region.ops:
            if isinstance(op, scf.ConditionOp):
                while_cond = op.condition
                break
            self.print(op)

        assert while_cond
        self.print_string(f"if(!{self.get_name(while_cond)}) {{ break; }}\n")


        self.print_string("}\n")

    @print.register
    def _(self, op: arith.SignlessIntegerBinaryOperation | arith.FloatingPointLikeBinaryOperation):
        const_name = self.gen_name(op.result)

        op_sign = ""
        if isinstance(op, arith.AddiOp):
            op_sign = "+"
        elif isinstance(op, arith.AddfOp):
            op_sign = "+"
        elif isinstance(op, arith.SubiOp):
            op_sign = "-"
        elif isinstance(op, arith.SubfOp):
            op_sign = "-"
        elif isinstance(op, arith.MuliOp):
            op_sign = "*"
        elif isinstance(op, arith.MulfOp):
            op_sign = "*"
        elif isinstance(op, arith.DivSIOp):
            op_sign = "/"
        elif isinstance(op, arith.DivfOp):
            op_sign = "/"
        elif isinstance(op, arith.AndIOp):
            op_sign = "&"
        elif isinstance(op, arith.OrIOp):
            op_sign = "|"

        lhs = None
        rhs = None

        lhs = self.get_name(op.lhs)
        rhs = self.get_name(op.rhs)

        result_type = HostPrinter.convert_result_type(op.result.type)
        self.print_string(self.indent * "\t" + f"{result_type} {const_name} = {lhs} {op_sign} {rhs};\n")

    def propagate_name_to_use(name, use):
            if "func_arg_name" not in use.operation.attributes:
                use.operation.attributes["func_arg_name"] = builtin.ArrayAttr([builtin.ArrayAttr([builtin.StringAttr(name), builtin.IntegerAttr(use.index, builtin.i32)])])
            else:
                use.operation.attributes["func_arg_name"] = builtin.ArrayAttr(use.operation.attributes["func_arg_name"].data + \
                        (builtin.ArrayAttr([builtin.StringAttr(name), builtin.IntegerAttr(use.index, builtin.i32)]),)) # concatenation of tuples

    @print.register
    def _(self, func_op: func.FuncOp):
        #if func_op.sym_visibility and func_op.sym_visibility.data == "private":
        if func_op.sym_visibility:
            # Do not print private functions
            return
        self.print_string(f"void {func_op.sym_name.data}(")

        arg_names_lst = []
        for arg_idx,arg in enumerate(func_op.body.block.args):
            arg_type = HostPrinter.convert_result_type(arg.type)
            arg_name = self.gen_name(arg)
            if isinstance(arg.type, builtin.MemRefType):
                # TODO: this is very hacky. Refactor
                # The read functions have a 1D array as first argument and a HitTile_float as second argument
                arg_shape = "".join(list(map(lambda dim: f"[{dim.data}]", arg.type.shape.data)))
                if "read_into_sr" in func_op.sym_name.data:
                    if arg_idx == 0:
                        arg_names_lst.append(f"{arg_type} {arg_name} {arg_shape}")
                    else:
                        arg_names_lst.append(f"HitTile_{arg_type} {arg_name}")
                        self.is_hitTile_by_name.add(arg_name)
                        if arg_type not in self.new_hitTile_types:
                            self.new_hitTile_types.add(arg_type)
                else:
                    #arg_names_lst.append(f"{arg_type} {arg_name} {arg_shape}")
                    arg_names_lst.append(f"HitTile_{arg_type} {arg_name}")
                    self.is_hitTile_by_name.add(arg_name)
                    if arg_type not in self.new_hitTile_types:
                        self.new_hitTile_types.add(arg_type)
            else:
                arg_names_lst.append(f"{arg_type} {arg_name}")

        self.print_string(", ".join(arg_names_lst) + ") {\n")

        for op in func_op.body.ops:
            self.print(op)

        self.print_string("}\n") 

    @print.register
    def _(self, call_op: func.CallOp):
        for res in call_op.results:
            res_name = self.gen_name(res)
            res_type = HostPrinter.convert_result_type(res.type)
            self.print_string(self.indent * "\t" + f"{res_type} {res_name};\n")
            self.print_string(self.indent * "\t" + f"{res_name} = ")
        self.print_string(call_op.callee.root_reference.data + "(")
        arg_names_lst = []
        for arg in call_op.arguments:
            arg_name = self.get_name(arg)
            arg_names_lst.append(arg_name)

        self.print_string(", ".join(arg_names_lst) + ");\n")

    @print.register
    def _(self, op: func.ReturnOp):
        self.print_string("return;\n")


    @print.register
    def _(self, yield_op: scf.YieldOp):
        #pass
        #if not isinstance(yield_op.parent_op(), scf.WhileOp):
        #if not isinstance(yield_op.parent_op(), scf.IfOp):
        #    return
        if isinstance(yield_op.parent_op(), scf.IfOp):
            if_op : scf.IfOp = yield_op.parent_op()
            for arg_idx, arg in enumerate(yield_op.arguments):
        #        cyclic_var = self.get_name(yield_op.parent_op().results[arg_idx])
                res_var = self.get_name(if_op.results[arg_idx])

        #        if isinstance(arg.owner, scf.IfOp):
        #            arg = arg.owner.results[arg.index]

                value = self.get_name(arg)

                self.print_string(self.indent * "\t" + f"{res_var} = {value};\n")

        if isinstance(yield_op.parent_op(), scf.WhileOp):
            while_op : scf.WhileOp = yield_op.parent_op()
            for arg_idx, arg in enumerate(yield_op.arguments): 
                cyclic_var = self.get_name(while_op.after_region.blocks[0].args[arg_idx])

                if isinstance(arg.owner, scf.IfOp):
                    arg = arg.owner.results[arg.index]

                value = self.get_name(arg)

                self.print_string(self.indent * "\t" + f"{cyclic_var} = {value};\n")

    @print.register
    def _(self, op: scf.ParallelOp):
        START_IDX = 0
        END_IDX = 1
        STEP_IDX = 2

        trip_var = self.trip_vars_per_nesting_level[self.nesting_level]
        start_val = list(filter(lambda t: t.data[1].value.data == START_IDX, op.attributes["func_arg_name"].data))[0].data[0].data
        end_val = list(filter(lambda t: t.data[1].value.data == END_IDX, op.attributes["func_arg_name"].data))[0].data[0].data
        step_val = list(filter(lambda t: t.data[1].value.data == STEP_IDX, op.attributes["func_arg_name"].data))[0].data[0].data

        self.print(self.indent * "\t" + f"for(int {trip_var} = {start_val}; {trip_var} < {end_val}; {trip_var} += {step_val}) {{\n")

        self.propagate_arg(op.body.blocks[0].args[0], LOOP_IDX)
        self.indent += 1
        self.nesting_level += 1
        for _op in op.body.ops:
            self.print(_op)

        self.indent -= 1
        self.print(self.indent * "\t" + "}\n")
        self.nesting_level -= 1


    @print.register
    def _(self, for_op: affine.ForOp):
        index_name = self.gen_name(for_op.body.block.args[0])

        start_val = for_op.lowerBoundMap.get_affine_map().results[0]
        end_val = for_op.upperBoundMap.get_affine_map().results[0]
        step_val = for_op.step.value.data

        self.print_string(f"for(int {index_name} = {start_val}; {index_name} < {end_val}; {index_name} += {step_val}) {{\n")
        
        for op in for_op.body.ops:
            self.print(op)

    @print.register
    def _(self, op: affine.StoreOp):
        value_name = self.get_name(op.value)
        memref_name = self.get_name(op.memref)
        indices_names = []
        for index in op.indices:
            indices_names.append(self.get_name(index))

        if memref_name in self.is_hitTile_by_name:
            # If the memref is a HitTile, we use the hit function
            # to store the value in the tile.
            indices_list = ",".join(indices_names)
            self.print_string(f"hit({memref_name}, {indices_list}) = {value_name};\n")
        else:
            # Otherwise, we use the standard array notation
            indices_array_form = "".join([f"[{ind_name}]" for ind_name in indices_names])
            self.print_string(f"{memref_name}{indices_array_form} = {value_name};\n")

    @print.register
    def _(self, op: memref.StoreOp):
        value_name = self.get_name(op.value)
        memref_name = self.get_name(op.memref)
        indices_names = []
        for index in op.indices:
            indices_names.append(self.get_name(index))

        indices_array_form = "".join([f"[{ind_name}]" for ind_name in indices_names])
        self.print_string(f"{memref_name}{indices_array_form} = {value_name};\n")


    @print.register
    def _(self, op: affine.YieldOp):
        self.print_string("}\n")


    def _(self, op: scf.ForOp):
        START_IDX = 0
        END_IDX = 1
        STEP_IDX = 2

        trip_var = self.trip_vars_per_nesting_level[self.nesting_level]
        start_val = list(filter(lambda t: t.data[1].value.data == START_IDX, op.attributes["func_arg_name"].data))[0].data[0].data
        end_val = list(filter(lambda t: t.data[1].value.data == END_IDX, op.attributes["func_arg_name"].data))[0].data[0].data
        step_val = list(filter(lambda t: t.data[1].value.data == STEP_IDX, op.attributes["func_arg_name"].data))[0].data[0].data

        self.print(self.indent * "\t" + f"for(int {trip_var} = {start_val}; {trip_var} < {end_val}; {trip_var} += {step_val}) {{\n")

        self.propagate_arg(op.body.blocks[0].args[0], LOOP_IDX)
        self.nesting_level += 1
        self.indent += 1
        for _op in op.body.ops:
            self.print(_op)

        self.indent -= 1
        self.print(self.indent * "\t" + "}\n")
        self.nesting_level -= 1

    def propagate_arg(self, var, var_type, arg_idx=None): # var_type = RESULT; HLS_STREAM
        if var_type == ARGUMENT:
            var_name = f"arg{arg_idx}"
        elif var_type == LOOP_IDX:
            var_name = self.trip_vars_per_nesting_level[self.nesting_level]
        else:
            var_name = self.names[var_type][self.counters[var_type]]
        for use in var.uses:
            if "func_arg_name" not in use.operation.attributes:
                use.operation.attributes["func_arg_name"] = builtin.ArrayAttr([builtin.ArrayAttr([builtin.StringAttr(var_name), builtin.IntegerAttr(use.index, builtin.i32)])])
            else:
                use.operation.attributes["func_arg_name"] = builtin.ArrayAttr(use.operation.attributes["func_arg_name"].data + \
                        (builtin.ArrayAttr([builtin.StringAttr(var_name), builtin.IntegerAttr(use.index, builtin.i32)]),)) # concatenation of tuples

        if var_type != ARGUMENT:
            self.counters[var_type] += 1

    @print.register
    def _(self, op: arith.ConstantOp):
        const_name = self.gen_name(op.result)

        value = ""
        constant_type = ""
        if isinstance(op.result.type, builtin.IntegerType):
            if op.result.type.width.data == 1:
                constant_type = "bool"
                if op.value.value.data == 1:
                    value = "true"
                else:
                    value = "false"
            else:
                constant_type = "int"
        elif isinstance(op.result.type, builtin.IndexType):
            constant_type = "int"
        elif isinstance(op.result.type, builtin.Float32Type):
            constant_type = "float"
        elif isinstance(op.result.type, builtin.Float64Type):
            constant_type = "double"

        self.print_string(f"{constant_type} {const_name} = ")

        if value:
            self.print_string(f"{value};\n")
        else:
            self.print_string(f"{op.value.value.data};\n")

    @print.register
    def _(self, op: arith.IndexCastOp):
        src = self.get_name(op.input)
        casted = self.gen_name(op.result)
        result_type = HostPrinter.convert_result_type(op.result.type)

        self.print_string(self.indent * "\t" + f"{result_type} {casted} = (int) {src};\n")

    @print.register
    def _(self, op: BlockArgument):
        pass


    @print.register
    def _(self, op: affine.LoadOp):
        memref_load_res = self.gen_name(op.result)
        memref_load_operand = self.get_name(op.memref)
        res_type = HostPrinter.convert_result_type(op.result.type)
        indices_names = []
        for index in op.indices:
            indices_names.append(self.get_name(index))


        if memref_load_operand in self.is_hitTile_by_name:
            indices_list = ",".join(indices_names) 
            self.print_string(f"{res_type} {memref_load_res} = hit({memref_load_operand}, {indices_list});\n")
        else:
            indices_array_form = "".join([f"[{ind_name}]" for ind_name in indices_names]) 
            self.print_string(f"{res_type} {memref_load_res} = {memref_load_operand}{indices_array_form};\n")

    @print.register
    def _(self, op: memref.LoadOp):
        self.names[RESULT].append(f"res{self.counters[RESULT]}")
        memref_load_res = self.names[RESULT][self.counters[RESULT]]
        memref_load_operand = op.attributes["func_arg_name"].data[0].data[0].data

        # TODO: there might be more than one dimension in the load and this should be chosen
        # based on the index stored
        load_idx = op.attributes["func_arg_name"].data[1].data[0].data

        self.propagate_arg(op.res, RESULT)
    
        result_type = ""
        if isinstance(op.res.type, builtin.Float32Type):
            result_type = "float"
        elif isinstance(op.res.type, builtin.Float64Type):
            result_type = "double"

        self.print(self.indent * "\t" + f"{result_type} {memref_load_res} = {memref_load_operand}[{load_idx}];\n")

    @print.register
    def _(self, op: memref.AllocaOp | memref.AllocOp):
        alloca_res = self.gen_name(op.memref)

        elem_type = op.memref.type.element_type
        alloca_type = HostPrinter.convert_result_type(elem_type)

        array_shape = " * ".join(list(map(lambda dim: f"{dim.data}", op.memref.type.shape.data)))
        if array_shape == "":
            array_shape = "1"

        self.print_string(self.indent * "\t" + f"{alloca_type}* {alloca_res} = malloc({array_shape} * sizeof({alloca_type}));\n")

    @print.register
    def _(self, op: llvm.AllocaOp):
        self.names[RESULT].append(f"res{self.counters[RESULT]}")
        alloca_res = self.names[RESULT][self.counters[RESULT]]
        alloca_operand = op.attributes["func_arg_name"].data[0].data[0].data

        elem_type = op.res.type.type
        alloca_type = HostPrinter.convert_result_type(elem_type)
        
        self.propagate_arg(op.res, RESULT)

        self.print(self.indent * "\t" + f"{alloca_type} * {alloca_res} = malloc({op.size.owner.value.value.data} * sizeof({alloca_type}));\n")

    @print.register
    def _(self, op: llvm.LoadOp):
        self.names[RESULT].append(f"res{self.counters[RESULT]}")
        load_res = self.names[RESULT][self.counters[RESULT]]
        load_operand = op.attributes["func_arg_name"].data[0].data[0].data

        elem_type = op.ptr.type.type
        load_type = HostPrinter.convert_result_type(elem_type)
        
        self.propagate_arg(op.dereferenced_value, RESULT)

        self.print(self.indent * "\t" + f"{load_type} {load_res} = *{load_operand};\n")

    @print.register
    def _(self, op: arith.ComparisonOperation):
        res_name = self.gen_name(op.result)
        lhs = self.get_name(op.lhs)
        rhs = self.get_name(op.rhs)

        int_predicate = op.predicate.value.data
        sign = None
        if int_predicate in {0, 6}:
            sign = "=="
        elif int_predicate in {1, 7}:
            sign = "!="
        elif int_predicate in {2, 6}:
            sign = "<"
        elif int_predicate in {3, 7}:
            sign = "<="
        elif int_predicate in {4, 8}:
            sign = ">"
        elif int_predicate in {5, 9}:
            sign = ">="

        self.print_string(self.indent * "\t" + f"bool {res_name} = {lhs} {sign} {rhs};\n")

    @print.register
    def _(self, op: scf.IfOp):
        cond_name = self.get_name(op.cond)

        for res in op.results:
            res_name = self.gen_name(res)
            res_type = HostPrinter.convert_result_type(res.type)
            self.print_string(self.indent * "\t" + f"{res_type} {res_name};\n")

        self.print_string(f"if({cond_name})\n")

        if op.true_region.blocks:
            self.print_string("{\n")
            for true_op in op.true_region.blocks[0].ops:
                self.print(true_op)
            self.print_string("}\n")

        if op.false_region.blocks:
            self.print_string("else {\n")
            for false_op in op.false_region.blocks[0].ops:
                self.print(false_op)
            self.print_string("}\n")

    @print.register
    def _(self, op: device.DataCheckExists):
        bool_check = self.gen_name(op.res)
        self.print_string(f"bool {bool_check} = data_check_exists(\"{op.memory_name.data}\", {op.memory_space.value.data});\n")

    @print.register
    def _(self, op: device.LookUpOp):
        memref_name = self.gen_name(op.memref)
        self.print_string(f"{HostPrinter.convert_result_type(op.memref.type)} {memref_name} = lookup(\"{op.memory_name.data}\", {op.memory_space.value.data});\n")

    @print.register
    def _(self, op: device.AllocOp):
        memref_name = self.gen_name(op.memref)
        shape = ", ".join([str(dim.data) for dim in op.memref.type.shape.data])
        self.print_string(f"{HostPrinter.convert_result_type(op.memref.type)} {memref_name} = alloc(\"{op.memory_name.data}\", {shape}, {op.memory_space.value.data});\n")

    @print.register
    def _(self, op: device.DataAcquire):
        memref_name = op.memory_name.data
        self.print_string(f"data_acquire(\"{memref_name}\", {op.memory_space.value.data});\n")


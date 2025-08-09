from __future__ import annotations

from ast import arg
from functools import singledispatchmethod

from xdsl.dialects import arith, builtin, memref, func, llvm, scf, affine
from xdsl.dialects.experimental import hls
from xdsl.ir import Operation, SSAValue, BlockArgument
from xdsl.utils.base_printer import BasePrinter
from ftn.dialects import device
from math import prod

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
        self.print_string("#include <iostream>\n")
        self.print_string("#include <unordered_map>\n")
        self.print_string("#include <string>\n")
        self.print_string("#include <CL/cl2.hpp>\n")
        self.print_string(
            "#ifndef DEBUG\n"
            "#include <CL/cl_ext_xilinx.h>\n"
            "#endif\n"
        )

        self.print_string("cl::Context context;\n")
        self.print_string("cl::CommandQueue queue;\n")
        self.print_string("cl::CommandQueue compute_queue;\n")
        self.print_string("cl::Program program;\n")
        self.print_string("cl_int err;\n")

        self.print_string("std::unordered_map<std::string, cl::Buffer> bufferMap;")
        self.print_string("""
        cl_ulong getBufferSize(const cl::Buffer &buffer) {
            cl_ulong size = 0;
            // Query the buffer info with CL_MEM_SIZE
            cl_int err = buffer.getInfo(CL_MEM_SIZE, &size);
            if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to get buffer size");
            }
            return size;
        }
        """)

        self.print_string("""
        void initOpenCL() {
            // 1. Get platforms
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);
            if (platforms.empty()) throw std::runtime_error("No OpenCL platforms found.");

            // Pick the first platform
            cl::Platform platform = platforms[0];
            std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << "\\n";

            // 2. Get devices (prefer GPU if available)
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
            if (devices.empty()) {
            platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
            if (devices.empty())
                throw std::runtime_error("No OpenCL devices found.");
            }

            cl::Device device = devices[0];
            std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << "\\n";

            // 3. Create context with device
            context = cl::Context(device);

            // 4. Create command queues
            queue = cl::CommandQueue(context, device, 0, &err);
            if (err != CL_SUCCESS) throw std::runtime_error("Failed to create command queue");

            compute_queue = cl::CommandQueue(context, device, 0, &err);
            if (err != CL_SUCCESS) throw std::runtime_error("Failed to create compute queue");

            // 5. Build a simple program (replace with your actual kernel source)
            const char* kernel_source = R"CLC(
            __kernel void tt_device() {
                // example kernel - does nothing
            }
            )CLC";

            cl::Program::Sources sources;
            sources.push_back({kernel_source, strlen(kernel_source)});

            program = cl::Program(context, sources);
            err = program.build({device});
            if (err != CL_SUCCESS) {
            std::string build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            std::cerr << "Build error:\\n" << build_log << "\\n";
            throw std::runtime_error("Failed to build program");
            }
        }
        """)

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
        elif isinstance(result_type, builtin.Float32Type):
            c_result_type = "float"
        elif isinstance(result_type, builtin.Float64Type):
            c_result_type = "double"
        elif isinstance(result_type, memref.MemRefType):
            # FIXME: this is only for device side buffers
            c_result_type = "cl::Buffer"

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
        elif isinstance(op, arith.XOrIOp):
            op_sign = "^"

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
        if func_op.sym_visibility:
            # Do not print private functions
            return

        # FIXME: this is a wrapper main function needed by flang that should be removed for 
        # the C compiler. This is hack, the function should be removed at the MLIR level
        if func_op.sym_name.data == "main":
            return
        # FIXME: Generating the main function here for now. Note that the name of the function called
        # might be mangled differently. This is a temporary hack.
        if func_op.sym_name.data == "_QQmain":
            self.print_string("""
            int main() {
                try {
                    initOpenCL();
                    _QMex1_testPcalc();
                } catch (const std::exception &ex) {
                    std::cerr << "Error: " << ex.what() << "\\n";
                    return 1;
                }
                return 0;
            }
            """)
            return
        else:
            self.print_string(f"void {func_op.sym_name.data}(")

        arg_names_lst = []
        for arg_idx,arg in enumerate(func_op.body.block.args):
            arg_type = HostPrinter.convert_result_type(arg.type)
            arg_name = self.gen_name(arg)
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
        if isinstance(yield_op.parent_op(), scf.IfOp):
            if_op : scf.IfOp = yield_op.parent_op()
            for arg_idx, arg in enumerate(yield_op.arguments):
                res_var = self.get_name(if_op.results[arg_idx])
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
        memref_load_operand = self.get_name(op.memref)

        # TODO: there might be more than one dimension in the load and this should be chosen
        # based on the index stored
        load_idx = self.get_name(op.indices[0])

        result_type = ""
        if isinstance(op.res.type, builtin.Float32Type):
            result_type = "float"
        elif isinstance(op.res.type, builtin.Float64Type):
            result_type = "double"

        self.print_string(self.indent * "\t" + f"{result_type} {memref_load_res} = {memref_load_operand}[{load_idx}];\n")

    @print.register
    def _(self, op: memref.AllocaOp | memref.AllocOp):
        alloca_res = self.gen_name(op.memref)

        for use in op.memref.uses:
            if isinstance(use.operation, memref.DmaStartOp) and use.operation.tag == op.memref:
                self.print_string(f"cl::Event {alloca_res};\n")
                return

        elem_type = op.memref.type.element_type
        alloca_type = HostPrinter.convert_result_type(elem_type)

        array_shape = " * ".join(list(map(lambda dim: f"{dim.data}", op.memref.type.shape.data)))
        if array_shape == "":
            array_shape = "1"

        self.print_string(self.indent * "\t" + f"{alloca_type}* {alloca_res} = ({alloca_type}*)malloc({array_shape} * sizeof({alloca_type}));\n")

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

        # FIXME: this is a hack to get the type of an event right. In the future, we should
        # tag event memrefs with an attribute. Similarly with the buffer size
        for res_idx,res in enumerate(op.results):
            res_name = self.gen_name(res)
            if len(op.results) == 2:
                if res_idx == 0:
                    res_type = "cl::Event"
                else:
                    res_type = "cl_ulong"
            else:
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
        self.print_string(f"bool {bool_check} = bufferMap.find(\"{op.memory_name.data}\") != bufferMap.end();\n")

    @print.register
    def _(self, op: device.LookUpOp):
        memref_name = self.gen_name(op.memref)
        self.print_string(f"cl::Buffer {memref_name} = bufferMap.at(\"{op.memory_name.data}\");\n")

    @print.register
    def _(self, op: device.AllocOp):
        memref_name = self.gen_name(op.memref)

        size = prod([dim.data for dim in op.memref.type.shape.data])

        self.print_string(f"cl::Buffer {memref_name}(context, CL_MEM_READ_WRITE, {size});\n")
        self.print_string(f"bufferMap.emplace(\"{op.memory_name.data}\", {memref_name});\n")

    @print.register
    def _(self, op: device.DataAcquire):
        memref_name = op.memory_name.data
        return
        self.print_string(f"data_acquire(\"{memref_name}\", {op.memory_space.value.data});\n")


    @print.register
    def _(self, op: device.DataNumElements):
        num_elements_name = self.gen_name(op.res)
        self.print_string(f"cl_ulong {num_elements_name} = getBufferSize(bufferMap.at(\"{op.memory_name.data}\"));\n")

    @print.register
    def _(self, op: device.KernelCreate):
        kernel = self.gen_name(op.res)
        kernel_name = op.device_function.root_reference.data

        self.print_string(f"cl::Kernel {kernel}(program, \"{kernel_name}\", &err);\n")


    @print.register
    def _(self, op: device.KernelLaunch):
        kernel = self.get_name(op.handle)
        kernel_create : device.KernelCreate = op.handle.owner

        arg_names = []
        for arg in kernel_create.mapped_data:
            arg_names.append(self.get_name(arg))

        self.print_string(
            f"#ifdef DEBUG\n"
            f"compute_queue.enqueueNDRangeKernel({kernel}, cl::NullRange, cl::NDRange(1), cl::NullRange);\n"
            f"#else\n"
            f"compute_queue.enqueueTask({kernel}, nullptr, nullptr);\n"
            f"#endif\n"
        )

    @print.register
    def _(self, op: device.KernelWait):
        self.print_string("compute_queue.finish();\n")

    @print.register
    def _(self, op: device.DataRelease):
        memref_name = op.memory_name.data
        return
        self.print_string(f"data_release(\"{memref_name}\", {op.memory_space.value.data});\n")

    @print.register
    def _(self, op: memref.DmaStartOp):
        src_name = self.get_name(op.src)
        dst_name = self.get_name(op.dest)
        n_elems = self.get_name(op.num_elements)
        event = self.get_name(op.tag)

        if op.src.type.memory_space != builtin.NoneAttr() and op.src.type.memory_space.value.data == 2:
            # Device to host
            self.print_string(f"queue.enqueueReadBuffer({src_name}, CL_TRUE, 0, {n_elems}, {dst_name}, nullptr, &{event});\n")
        else:
            # Host to device
            self.print_string(f"queue.enqueueWriteBuffer({dst_name}, CL_TRUE, 0, {n_elems}, {src_name}, nullptr, &{event});\n")

    @print.register
    def _(self, op: memref.DmaWaitOp):
        event = self.get_name(op.tag)

        self.print_string(f"{event}.wait();\n")

    @print.register
    def _(self, scope_op: memref.AllocaScopeOp):
        for op in scope_op.scope.ops:
            self.print(op)
    
    @print.register
    def _(self, op: memref.AllocaScopeReturnOp):
        return
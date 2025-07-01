from enum import Enum

ArgIntent = Enum("ArgIntent", ["IN", "OUT", "INOUT", "UNKNOWN"])


class ArrayDescription:
    def __init__(self, name, dim_sizes, dim_starts, dim_ends):
        self.name = name
        self.dim_sizes = dim_sizes
        self.dim_starts = dim_starts
        self.dim_ends = dim_ends


class ComponentState:
    def __init__(self, fn_name=None, module_name=None, fn_identifier=None):
        self.fn_name = fn_name
        self.module_name = module_name
        self.fn_identifier = fn_identifier
        self.array_info = {}
        self.pointers = []


class ArgumentDefinition:
    def __init__(self, name, is_scalar, arg_type, intent, is_allocatable):
        self.name = name
        self.is_scalar = is_scalar
        self.intent = intent
        self.arg_type = arg_type
        self.is_allocatable = is_allocatable


class FunctionDefinition:
    def __init__(self, name, return_type, is_definition_only, blocks):
        self.name = name
        self.return_type = return_type
        self.args = []
        self.is_definition_only = is_definition_only
        self.blocks = blocks

    def add_arg_def(self, arg_def):
        self.args.append(arg_def)


class GlobalFIRComponent:
    def __init__(self, sym_name, type, fir_mlir):
        self.sym_name = sym_name
        self.type = type
        self.fir_mlir = fir_mlir
        self.standard_mlir = None


class ProgramState:
    def __init__(self):
        self.function_definitions = {}
        self.global_state = ComponentState()
        self.function_state = None
        self.fir_global_constants = {}
        self.is_in_global = False

    def enterGlobal(self):
        assert self.function_state is None
        self.is_in_global = True

    def leaveGlobal(self):
        assert self.function_state is None
        self.is_in_global = False

    def isInGlobal(self):
        return self.is_in_global

    def addFunctionDefinition(self, name, fn_def):
        assert name not in self.function_definitions.keys()
        self.function_definitions[name] = fn_def

    def enterFunction(self, fn_name, function_identifier, module_name=None):
        assert self.function_state is None
        self.function_state = ComponentState(fn_name, module_name, function_identifier)

    def getCurrentFnState(self):
        assert self.function_state is not None
        return self.function_state

    def leaveFunction(self):
        self.function_state = None

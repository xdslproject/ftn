from typing import Callable, List, Optional, Type, Union
import re
from xdsl.ir import Operation


def camel_to_snake(name):
    pattern = re.compile(r"(?<!^)(?=[A-Z])")
    return pattern.sub("_", name).lower()


def get_method(instance: object, method: str) -> Optional[Callable]:
    if not hasattr(instance, method):
        return None
    else:
        f = getattr(instance, method)
        if callable(f):
            return f
        else:
            return None


class Visitor:
    def traverse(self, operation: Operation):
        class_name = camel_to_snake(type(operation).__name__)

        traverse = get_method(self, f"traverse_{class_name}")
        if traverse:
            return [traverse(operation)]
        else:
            ret_ops = []
            for r in operation.regions:
                for b in r.blocks:
                    for op in b.ops:
                        ret_ops.append(self.traverse(op))
            return ret_ops

        visit = get_method(self, f"visit_{class_name}")
        if visit:
            return [visit(operation)]

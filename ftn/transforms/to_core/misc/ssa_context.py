from dataclasses import dataclass, field
from xdsl.ir import SSAValue
from typing import Dict, Optional


@dataclass
class SSAValueCtx:
    """
    Context that relates identifiers from the AST to SSA values used in the flat representation.
    """

    dictionary: Dict[str, SSAValue] = field(default_factory=dict)
    parent_scope = None

    def __init__(self, parent_scope=None):
        self.parent_scope = parent_scope
        self.dictionary = {}

    def __getitem__(self, identifier: str) -> Optional[SSAValue]:
        """Check if the given identifier is in the current scope, or a parent scope"""
        ssa_value = self.dictionary.get(identifier, None)
        if ssa_value:
            return ssa_value
        elif self.parent_scope:
            return self.parent_scope[identifier]
        else:
            return None

    def __delitem__(self, identifier: str):
        if identifier in self.dictionary:
            del self.dictionary[identifier]

    def __setitem__(self, identifier: str, ssa_value: SSAValue):
        """Relate the given identifier and SSA value in the current scope"""
        if ssa_value is None:
            raise Exception()
        if identifier in self.dictionary:
            raise Exception()
        else:
            self.dictionary[identifier] = ssa_value

    def contains(self, identifier):
        if identifier in self.dictionary:
            return True
        if self.parent_scope is not None:
            return self.parent_scope.contains(identifier)
        else:
            return False

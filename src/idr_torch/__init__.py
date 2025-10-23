import sys

from .interface import Interface

# Replace this module with the Interface instance
# This allows dynamic routing to the appropriate API at runtime
# Type hints are provided in __init__.pyi for static type checkers
sys.modules[__name__] = Interface()  # type: ignore[assignment]

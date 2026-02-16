"""Test configuration — make component modules importable as src.{name}."""

import importlib
import sys
import types
from pathlib import Path

_project_root = Path(__file__).parent.parent
_impls = _project_root / ".pact" / "implementations"

# Create a virtual 'src' package that lazily loads from implementations
if "src" not in sys.modules:
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = []
    src_pkg.__package__ = "src"
    sys.modules["src"] = src_pkg

# Add each component's src/ directory to the 'src' package path
if _impls.exists():
    for component_dir in sorted(_impls.iterdir()):
        src_dir = component_dir / "src"
        if src_dir.is_dir():
            src_str = str(src_dir)
            if src_str not in sys.modules["src"].__path__:
                sys.modules["src"].__path__.append(src_str)

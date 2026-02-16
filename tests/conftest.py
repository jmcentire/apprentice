"""Test configuration — make component modules importable as src.{name}."""

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

_project_root = Path(__file__).parent.parent
_impls = _project_root / ".pact" / "implementations"
_comps = _project_root / ".pact" / "compositions"

# Create a virtual 'src' package that lazily loads from implementations
if "src" not in sys.modules:
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = []
    src_pkg.__package__ = "src"
    sys.modules["src"] = src_pkg

# Add each component's src/ directory to the 'src' package path
# Also add to sys.path so bare imports in composition glue modules work
for _base in (_impls, _comps):
    if _base.exists():
        for component_dir in sorted(_base.iterdir()):
            src_dir = component_dir / "src"
            if src_dir.is_dir():
                src_str = str(src_dir)
                if src_str not in sys.modules["src"].__path__:
                    sys.modules["src"].__path__.append(src_str)
                if src_str not in sys.path:
                    sys.path.append(src_str)


@pytest.fixture(autouse=True)
def _wire_root_components(request):
    """If the test uses mock_component_overrides, patch _components in root composition."""
    if "mock_component_overrides" in request.fixturenames:
        overrides = request.getfixturevalue("mock_component_overrides")
        with patch("src.root.composition._components", overrides):
            yield
    else:
        yield

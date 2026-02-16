"""Re-export CLI models from the canonical location.

This shim ensures that `from src.cli_models import ...` (used by composition
modules) resolves to the *same* Pydantic classes that `apprentice.cli.cli`
uses internally.  Without this, Pydantic strict-mode rejects cross-module
instances because the class objects differ.
"""

from apprentice.cli.cli_models import *  # noqa: F401,F403

"""Make ``modular`` importable without installing the Isaac Lab package."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture(scope="session")
def schema():
    """The repository's schema.py, loaded by path. It needs only the stdlib."""
    spec = importlib.util.spec_from_file_location("_jose_schema", ROOT / "schema.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_jose_schema"] = module
    spec.loader.exec_module(module)
    return module

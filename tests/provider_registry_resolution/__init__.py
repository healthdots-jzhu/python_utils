import io
import importlib.util
from contextlib import redirect_stdout
from pathlib import Path


def load_module():
    """Import the main pipeline script as a Python module.

    redirect_stdout suppresses any residual print output so test runs stay
    clean.  The returned module object exposes all public functions and
    module-level constants (registry_a/b/c, FEATURE_COLS, …).
    """
    root = Path(__file__).resolve().parents[2]
    module_path = root / "provider_identify_resolution_poc.py"
    spec = importlib.util.spec_from_file_location("provider_identify_resolution_poc", module_path)
    module = importlib.util.module_from_spec(spec)
    with redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module

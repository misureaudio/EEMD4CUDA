import importlib
import sys

import pytest

HARD_DEPS = ["soundfile", "numpy", "scipy", "matplotlib", "pyloudnorm", "tqdm"]


@pytest.mark.parametrize("mod", HARD_DEPS)
def test_hard_dependency_imports(mod):
    importlib.import_module(mod)


def test_emd_par_importable():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    p = str(root / "eemd4cuda")
    if p not in sys.path:
        sys.path.insert(0, p)
    import emd_par  # noqa: F401

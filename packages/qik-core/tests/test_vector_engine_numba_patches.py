"""Guard the Numba inline IR copy workaround and its compiled behavior."""

import copy
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numba
import pytest
from numba.core import inline_closurecall

# Importing the kernel directly must install the workaround too.
from qik.vector_engine import _numba_patches
from qik.vector_engine.kernel import run_columns  # noqa: F401


def test_patch_installed_and_idempotent():
    assert numba.__version__ == "0.66.0"
    patched = inline_closurecall.InlineWorker.inline_ir
    assert getattr(patched, "_qik_copy_ir_patch", False)
    _numba_patches.apply()
    assert inline_closurecall.InlineWorker.inline_ir is patched


def test_importing_the_kernel_submodule_patches_a_fresh_interpreter():
    """A first import of ``qik.vector_engine.kernel`` alone must install the workaround.

    The package ``__init__`` is the only install site, so this runs in a new interpreter:
    inside this session other test modules may have imported the package first.
    """
    from qik import vector_engine

    sources = Path(vector_engine.__file__).resolve().parents[2]
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(sources), os.environ.get("PYTHONPATH", "")]),
    )
    probe = (
        "import sys\n"
        "from qik.vector_engine.kernel import run_columns\n"
        "from numba.core import inline_closurecall\n"
        "patched = getattr(inline_closurecall.InlineWorker.inline_ir, '_qik_copy_ir_patch', False)\n"
        "sys.exit(0 if patched else 3)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_source_mismatch_refuses_patch(monkeypatch):
    def unrelated(*args, **kwargs):
        return None

    monkeypatch.setattr(inline_closurecall.InlineWorker, "inline_ir", unrelated)
    monkeypatch.setattr(_numba_patches, "_warned", False)
    with pytest.warns(RuntimeWarning, match="Numba.*source"):
        _numba_patches.apply()
    assert inline_closurecall.InlineWorker.inline_ir is unrelated


def test_version_mismatch_refuses_patch(monkeypatch):
    def unrelated(*args, **kwargs):
        return None

    monkeypatch.setattr(inline_closurecall.InlineWorker, "inline_ir", unrelated)
    monkeypatch.setattr(numba, "__version__", "0.66.1")
    monkeypatch.setattr(_numba_patches, "_warned", False)
    with pytest.warns(RuntimeWarning, match="expected Numba 0.66.0"):
        _numba_patches.apply()
    assert inline_closurecall.InlineWorker.inline_ir is unrelated


@pytest.mark.skipif(numba.config.DISABLE_JIT, reason="requires Numba compilation")
def test_inlined_multiblock_callee_uses_shared_memo_and_matches_python(monkeypatch):
    copied_blocks = []
    original_deepcopy = copy.deepcopy

    def observe(value, *args, **kwargs):
        if (
            isinstance(value, dict)
            and len(value) > 1
            and all(isinstance(block, numba.core.ir.Block) for block in value.values())
        ):
            copied_blocks.append(value)
        return original_deepcopy(value, *args, **kwargs)

    # Patch the name the installed ``inline_ir`` resolves at call time. It reads module globals
    # from ``numba.core.inline_closurecall``, so seeing this shim also proves the patched
    # function runs against that module rather than a private copy of its namespace.
    monkeypatch.setattr(inline_closurecall, "copy", SimpleNamespace(deepcopy=observe))

    @numba.njit(inline="always")
    def callee(x):
        if x < 0:
            return x * 2
        if x == 0:
            return 17
        return x + 3

    @numba.njit
    def caller(x):
        return callee(x) + callee(x - 1)

    for x in (-4, -1, 0, 1, 5):
        assert caller(x) == callee.py_func(x) + callee.py_func(x - 1)
    assert copied_blocks

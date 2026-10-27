"""Temporary, strictly versioned workarounds for Numba."""

from __future__ import annotations

import hashlib
import inspect
import textwrap
import warnings
from typing import Any

import numba
from numba.core import inline_closurecall

_VERSION = "0.66.0"
_INLINE_IR_SHA256 = "26fd2b7600c9d9295c925c2dd4d498929ef6ae43c09bf285298255527da5362b"
_OLD_COPY = """\
        kernel_copy.blocks = {}
        for block_label, block in the_ir.blocks.items():
            new_block = copy.deepcopy(the_ir.blocks[block_label])
            kernel_copy.blocks[block_label] = new_block
"""
_NEW_COPY = "        kernel_copy.blocks = copy.deepcopy(the_ir.blocks)\n"
_warned = False


def _warn(reason: str) -> None:
    global _warned
    if not _warned:
        warnings.warn(
            f"Numba copy_ir workaround not installed: {reason}", RuntimeWarning, stacklevel=2
        )
        _warned = True


def apply() -> None:
    """Install only over the exact upstream implementation we have verified.

    Safe to call repeatedly: the installed function carries a marker, so a second import, a
    reload or a test that calls this directly is a no-op. A version or source mismatch leaves
    upstream Numba untouched, warns once and returns — the engine still compiles, just slowly.
    """
    original = inline_closurecall.InlineWorker.inline_ir
    if getattr(original, "_qik_copy_ir_patch", False):
        return
    if numba.__version__ != _VERSION:
        _warn(f"expected Numba {_VERSION}, found {numba.__version__}")
        return
    try:
        source = inspect.getsource(original)
    except (OSError, TypeError):
        _warn("inline_ir source is unavailable")
        return
    if hashlib.sha256(source.encode()).hexdigest() != _INLINE_IR_SHA256:
        _warn("inline_ir source does not match Numba 0.66.0")
        return
    source = textwrap.dedent(source)
    if source.count(_OLD_COPY) != 1:
        _warn("inline_ir source does not contain the expected copy loop")
        return

    # Each block otherwise deep-copies the entire scope, quadratic in blocks × vars.
    # Drop this workaround when upstream fixes the shared-memo copy.
    namespace: dict[str, Any] = {}
    exec(
        compile(source.replace(_OLD_COPY, _NEW_COPY), "<qik numba inline_ir patch>", "exec"),
        vars(inline_closurecall),
        namespace,
    )
    patched = namespace["inline_ir"]
    patched._qik_copy_ir_patch = True
    inline_closurecall.InlineWorker.inline_ir = patched

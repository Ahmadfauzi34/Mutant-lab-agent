"""Canonical compatibility-shim loader for modular V2.31.

The real implementation now lives in ``agen_lab`` subsystem modules.  This
loader only binds the historical module name ``agen_kognitif_v2_28`` to the
thin file in ``core/`` so trusted-local pickle/checkpoint lookup and legacy
imports remain compatible.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

CANONICAL_MODULE_NAME = "agen_kognitif_v2_28"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = PROJECT_ROOT / "core" / "agen_kognitif_v2_28.py"


def _load_kernel() -> ModuleType:
    existing = sys.modules.get(CANONICAL_MODULE_NAME)
    if existing is not None:
        existing_path = Path(getattr(existing, "__file__", "")).resolve()
        if existing_path != KERNEL_PATH.resolve():
            raise ImportError(
                "Canonical module name agen_kognitif_v2_28 is already bound "
                f"to a different path: {existing_path}"
            )
        return existing

    spec = importlib.util.spec_from_file_location(
        CANONICAL_MODULE_NAME,
        KERNEL_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load canonical compatibility shim from {KERNEL_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[CANONICAL_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


core = _load_kernel()
CORE_PATH = KERNEL_PATH
CORE_VERSION = core.CORE_VERSION

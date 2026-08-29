"""Lab Agen V2.40 modular public API — structural-pattern cognition baseline.

All cognitive implementation is physically modular.  ``_kernel`` now loads
only the thin historical compatibility shim so ``import agen_lab as core`` and
trusted-local V2.28/V2.40 checkpoints keep the legacy public surface.
"""
from __future__ import annotations

from ._kernel import core as _core, CORE_PATH, CORE_VERSION
from .agent import IntegratedCognitiveAgent

# Compatibility surface: re-export every public symbol from the frozen kernel.
# This is deliberate during M1 so regression callers do not have to migrate in
# the same change that introduces module boundaries.
for _name in dir(_core):
    if not _name.startswith("_"):
        globals().setdefault(_name, getattr(_core, _name))


def __getattr__(name):
    return getattr(_core, name)


__all__ = sorted(
    name
    for name in globals()
    if not name.startswith("_")
)

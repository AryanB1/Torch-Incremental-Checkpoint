"""
Delta Engine — detects which model parameters changed above a threshold.

Tries to import the compiled C++ extension (delta_engine_cpp) for the
fast OpenMP path.  Falls back transparently to pure Python when the
extension is not available (e.g. during development before setup.py has
been run).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_CPP_AVAILABLE = False
_cpp_ext = None

try:
    import delta_engine_cpp as _cpp_ext
    _CPP_AVAILABLE = True
except ImportError:
    pass


@dataclass
class DeltaResult:
    """Output of DeltaEngine.compute_dirty()."""
    dirty_names: list[str]
    deltas: dict[str, torch.Tensor]
    dirty_ratio: float
    per_layer_norms: dict[str, float]

    @property
    def num_dirty(self) -> int:
        return len(self.dirty_names)


# ---------------------------------------------------------------------------
# Pure-Python fallback
# ---------------------------------------------------------------------------

def _py_relative_l2(current: torch.Tensor, base: torch.Tensor) -> float:
    c = current.detach().to(torch.float32).cpu()
    b = base.detach().to(torch.float32).cpu()
    diff_norm = float((c - b).norm())
    base_norm = float(b.norm())
    return diff_norm / (base_norm + 1e-8)


def _py_compute_dirty(
    names: list[str],
    current_tensors: list[torch.Tensor],
    base_tensors: list[torch.Tensor],
    threshold: float,
) -> tuple[list[str], list[torch.Tensor], list[float]]:
    dirty_names: list[str] = []
    deltas: list[torch.Tensor] = []
    norms: list[float] = []

    for name, cur, bas in zip(names, current_tensors, base_tensors):
        norm = _py_relative_l2(cur, bas)
        if norm > threshold:
            dirty_names.append(name)
            delta = (cur.to(torch.float32).cpu() -
                     bas.to(torch.float32).cpu()).contiguous()
            deltas.append(delta)
            norms.append(norm)

    return dirty_names, deltas, norms

class DeltaEngine:
    """
    Computes the dirty set between two model state dicts.

    Parameters
    ----------
    threshold : float
        Minimum relative L2 norm ||current - base|| / ||base|| for a
        parameter to be considered "dirty" and included in the delta.
    use_cpp : bool | None
        If True, force the C++ path (raises if unavailable).
        If False, force the pure-Python path.
        If None (default), use C++ if available.
    """

    def __init__(
        self,
        threshold: float = 1e-4,
        use_cpp: Optional[bool] = None,
    ):
        self.threshold = threshold

        if use_cpp is True and not _CPP_AVAILABLE:
            raise ImportError(
                "C++ delta engine not available. "
                "Run `pip install -e .` to build the extension."
            )
        if use_cpp is None:
            self._use_cpp = _CPP_AVAILABLE
        else:
            self._use_cpp = use_cpp

        if not self._use_cpp and use_cpp is None and not _CPP_AVAILABLE:
            warnings.warn(
                "delta_engine_cpp not found — falling back to pure Python. "
                "Run `pip install -e .` to build the C++ extension for best "
                "performance.",
                ImportWarning,
                stacklevel=2,
            )

    @property
    def backend(self) -> str:
        return "cpp" if self._use_cpp else "python"

    def compute_dirty(
        self,
        current: dict[str, torch.Tensor],
        base: dict[str, torch.Tensor],
    ) -> DeltaResult:
        """
        Identify parameters that changed above the threshold.

        Parameters
        ----------
        current : dict[str, Tensor]  — current model state dict
        base    : dict[str, Tensor]  — baseline (last-checkpointed) state dict

        Returns
        -------
        DeltaResult
        """
        # Use intersection of keys (handles models that grew/shrank layers)
        current_keys = set(current.keys())
        base_keys = set(base.keys())
        names = sorted(current_keys & base_keys)

        added = current_keys - base_keys
        removed = base_keys - current_keys
        if added:
            logger.info("New parameters not in base (will be treated as dirty): %s", added)
        if removed:
            logger.info("Parameters removed since base: %s", removed)

        if not names:
            return DeltaResult(
                dirty_names=[],
                deltas={},
                dirty_ratio=0.0,
                per_layer_norms={},
            )

        current_tensors = [current[n] for n in names]
        base_tensors    = [base[n]    for n in names]

        # Validate matching shapes
        for name, cur, bas in zip(names, current_tensors, base_tensors):
            if cur.shape != bas.shape:
                raise ValueError(
                    f"Shape mismatch for parameter {name!r}: "
                    f"current={cur.shape}, base={bas.shape}"
                )

        if self._use_cpp and _cpp_ext is not None:
            dirty_names, delta_list, dirty_norms = _cpp_ext.compute_dirty_tensors(
                names, current_tensors, base_tensors, self.threshold
            )
            per_layer_norms: dict[str, float] = dict(
                zip(dirty_names, dirty_norms)
            )
        else:
            dirty_names, delta_list, dirty_norms = _py_compute_dirty(
                names, current_tensors, base_tensors, self.threshold
            )
            per_layer_norms = dict(zip(dirty_names, dirty_norms))

        deltas: dict[str, torch.Tensor] = dict(zip(dirty_names, delta_list))
        dirty_ratio = len(dirty_names) / max(len(names), 1)

        return DeltaResult(
            dirty_names=dirty_names,
            deltas=deltas,
            dirty_ratio=dirty_ratio,
            per_layer_norms=per_layer_norms,
        )

    def compute_all_norms(
        self,
        current: dict[str, torch.Tensor],
        base: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """
        Compute relative L2 norms for ALL parameters (pure Python).

        Useful for diagnostics / visualisation.
        """
        result: dict[str, float] = {}
        for name in sorted(set(current.keys()) & set(base.keys())):
            result[name] = _py_relative_l2(current[name], base[name])
        return result

"""
In-process metrics collector for the checkpoint engine.

Tracks per-save statistics (blocking time, dirty ratio, storage) and
exposes aggregated summaries.  Does not depend on any external metrics
backend — callers can read the data and forward to W&B, MLflow, etc.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SaveRecord:
    step: int
    wall_time: float        # Unix timestamp of the save call
    blocking_ms: float      # Time the training loop was blocked
    dirty_ratio: float      # Fraction of params written
    storage_bytes: int      # Total store size after this save


class CheckpointMetrics:
    """Lightweight in-memory metrics store."""

    def __init__(self) -> None:
        self._records: list[SaveRecord] = []

    def record_save(
        self,
        step: int,
        blocking_ms: float,
        dirty_ratio: float,
        storage_bytes: int,
    ) -> None:
        self._records.append(
            SaveRecord(
                step=step,
                wall_time=time.time(),
                blocking_ms=blocking_ms,
                dirty_ratio=dirty_ratio,
                storage_bytes=storage_bytes,
            )
        )

    @property
    def records(self) -> list[SaveRecord]:
        return list(self._records)

    def summary(self) -> dict:
        if not self._records:
            return {}

        blocking_times = [r.blocking_ms for r in self._records]
        dirty_ratios   = [r.dirty_ratio  for r in self._records]

        return {
            "num_saves": len(self._records),
            "total_storage_bytes": self._records[-1].storage_bytes,
            "avg_blocking_ms": statistics.mean(blocking_times),
            "p50_blocking_ms": statistics.median(blocking_times),
            "p95_blocking_ms": _percentile(blocking_times, 95),
            "max_blocking_ms": max(blocking_times),
            "avg_dirty_ratio": statistics.mean(dirty_ratios),
            "min_dirty_ratio": min(dirty_ratios),
            "max_dirty_ratio": max(dirty_ratios),
        }

    def reset(self) -> None:
        self._records.clear()

    def __repr__(self) -> str:
        s = self.summary()
        if not s:
            return "CheckpointMetrics(empty)"
        return (
            f"CheckpointMetrics("
            f"saves={s['num_saves']}, "
            f"avg_blocking={s['avg_blocking_ms']:.1f}ms, "
            f"avg_dirty={s['avg_dirty_ratio']:.2%}"
            f")"
        )


def _percentile(data: list[float], p: int) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    frac = k - lo
    return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])

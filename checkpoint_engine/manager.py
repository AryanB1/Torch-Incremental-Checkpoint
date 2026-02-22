"""
CheckpointManager — top-level API for incremental model checkpointing.

Usage
-----
    manager = CheckpointManager(
        save_dir="./checkpoints",
        model=model,
        keep_last_n=5,
        keep_best_n=3,
        dirty_threshold=1e-4,
        async_write=True,
    )

    for step in range(1000):
        train_one_step(model)
        if step % 100 == 0:
            manager.save(step, metrics={"loss": loss.item()})

    manager.restore(step=500)  # load a specific checkpoint into model

    manager.close()            # flush async writes and shut down
"""

from __future__ import annotations

import time
import warnings
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from .async_writer import AsyncWriter
from .delta import DeltaEngine, DeltaResult
from .lifecycle import LifecycleManager
from .manifest import CheckpointVersion, Manifest
from .metrics import CheckpointMetrics
from .store import ContentAddressedStore


class CheckpointManager:
    """
    Incremental checkpoint manager.

    Parameters
    ----------
    save_dir        : str | Path — directory to store blobs and manifest
    model           : nn.Module  — the model to checkpoint
    keep_last_n     : int        — always retain last N checkpoints
    keep_best_n     : int        — retain best N by metric
    dirty_threshold : float      — relative L2 threshold for dirty detection
    async_write     : bool       — write blobs on background thread
    metric_key      : str        — metric to optimise for keep_best_n
    lower_is_better : bool       — True for loss, False for accuracy
    """

    def __init__(
        self,
        save_dir: str | Path,
        model: nn.Module,
        keep_last_n: int = 5,
        keep_best_n: int = 3,
        dirty_threshold: float = 1e-4,
        async_write: bool = True,
        metric_key: str = "loss",
        lower_is_better: bool = True,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.dirty_threshold = dirty_threshold
        self.async_write = async_write

        self._store = ContentAddressedStore(self.save_dir / "blobs")
        self._manifest = Manifest(self.save_dir)
        self._delta_engine = DeltaEngine(threshold=dirty_threshold)
        self._lifecycle = LifecycleManager(
            manifest=self._manifest,
            store=self._store,
            keep_last_n=keep_last_n,
            keep_best_n=keep_best_n,
            metric_key=metric_key,
            lower_is_better=lower_is_better,
        )
        self._writer = AsyncWriter(
            store=self._store,
            manifest=self._manifest,
            max_pending=2,
        )

        self._base_state: Optional[dict[str, torch.Tensor]] = None
        self._base_hashes: dict[str, str] = {}

        self._pending_futures: list[Future] = []
        self.metrics = CheckpointMetrics()

        self._maybe_load_base_from_manifest()

    def _maybe_load_base_from_manifest(self) -> None:
        """
        On restart, reconstruct the base snapshot from the stored manifest
        so that subsequent saves correctly compute deltas.
        """
        base_version = self._manifest.base_version()
        if base_version is None:
            return
        try:
            self._base_state = self._reconstruct_state_dict(base_version)
            self._base_hashes = dict(base_version.param_hashes)
        except Exception as exc:
            warnings.warn(
                f"Could not restore base snapshot from manifest: {exc}. "
                "Next save() will write a full checkpoint.",
                stacklevel=2,
            )

    def _current_state(self) -> dict[str, torch.Tensor]:
        """Return a CPU float32 copy of the model's current state dict."""
        return {
            k: v.detach().cpu().to(torch.float32).clone()
            for k, v in self.model.state_dict().items()
        }

    def _save_base_checkpoint(
        self,
        step: int,
        metrics: dict[str, float],
    ) -> Future:
        """Write the full model as the base checkpoint (step 0 / first save)."""
        state = self._current_state()
        self._base_state = state

        # Treat every parameter as dirty for a full save
        dirty_tensors = dict(state)
        # We don't have previous hashes yet
        fut = self._writer.enqueue(
            step=step,
            dirty_tensors=dirty_tensors,
            full_param_hashes={},   # writer will fill these from store
            metrics=metrics,
            dirty_ratio=1.0,
            is_full=True,
        )
        # After the future resolves we can read the hashes back
        def _capture_hashes(f: Future) -> None:
            try:
                version: CheckpointVersion = f.result()
                self._base_hashes = dict(version.param_hashes)
            except Exception:
                pass
        fut.add_done_callback(_capture_hashes)
        return fut

    def _reconstruct_state_dict(
        self, version: CheckpointVersion
    ) -> dict[str, torch.Tensor]:
        """
        Reconstruct full state dict from version hashes.
        Each version stores the hash for every parameter, enabling direct fetch.
        """
        state: dict[str, torch.Tensor] = {}
        for name, hexdigest in version.param_hashes.items():
            state[name] = self._store.get(hexdigest)
        return state

    def save(
        self,
        step: int,
        metrics: Optional[dict[str, float]] = None,
        force_full: bool = False,
    ) -> Future:
        """
        Save a checkpoint for the current model state.

        Parameters
        ----------
        step        : int              — training step / epoch number
        metrics     : dict[str, float] — e.g. {"loss": 2.3, "acc": 0.87}
        force_full  : bool             — ignore delta engine, save everything

        Returns
        -------
        concurrent.futures.Future that resolves to CheckpointVersion.
        If async_write=False the future is already resolved on return.
        """
        t0 = time.perf_counter()
        metrics = metrics or {}

        if self._base_state is None or force_full:
            fut = self._save_base_checkpoint(step, metrics)
            if not self.async_write:
                fut.result()
            self._pending_futures.append(fut)
            blocking_ms = (time.perf_counter() - t0) * 1000
            self.metrics.record_save(
                step=step,
                blocking_ms=blocking_ms,
                dirty_ratio=1.0,
                storage_bytes=self._store.total_bytes(),
            )
            return fut
        current_state = self._current_state()
        delta_result: DeltaResult = self._delta_engine.compute_dirty(
            current=current_state,
            base=self._base_state,
        )

        full_hashes = dict(self._base_hashes)
        dirty_as_current = {
            name: current_state[name] for name in delta_result.dirty_names
        }

        fut = self._writer.enqueue(
            step=step,
            dirty_tensors=dirty_as_current,
            full_param_hashes=full_hashes,
            metrics=metrics,
            dirty_ratio=delta_result.dirty_ratio,
            is_full=False,
        )

        self._base_state = current_state

        def _update_base_hashes(f: Future) -> None:
            try:
                version: CheckpointVersion = f.result()
                self._base_hashes = dict(version.param_hashes)
            except Exception:
                pass
        fut.add_done_callback(_update_base_hashes)

        if not self.async_write:
            fut.result()

        self._pending_futures.append(fut)

        try:
            self._lifecycle.run_gc()
        except Exception as exc:
            warnings.warn(f"Lifecycle GC failed: {exc}", stacklevel=2)

        blocking_ms = (time.perf_counter() - t0) * 1000
        self.metrics.record_save(
            step=step,
            blocking_ms=blocking_ms,
            dirty_ratio=delta_result.dirty_ratio,
            storage_bytes=self._store.total_bytes(),
        )
        return fut

    def restore(self, step: Optional[int] = None) -> None:
        """
        Restore model weights from a checkpoint.

        Parameters
        ----------
        step : int | None
            Step to restore. If None, restores the latest checkpoint.
        """
        # Flush any pending async writes first
        self.wait_all()

        if step is None:
            version = self._manifest.latest_version()
            if version is None:
                raise RuntimeError("No checkpoints found in manifest.")
        else:
            version = self._manifest.get_version(step)
            if version is None:
                available = self._manifest.steps()
                raise ValueError(
                    f"Step {step} not found in manifest. "
                    f"Available steps: {available}"
                )

        state_dict = self._reconstruct_state_dict(version)

        model_sd = self.model.state_dict()
        converted: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            if name in model_sd:
                converted[name] = tensor.to(
                    dtype=model_sd[name].dtype,
                    device=model_sd[name].device,
                )
            else:
                converted[name] = tensor

        self.model.load_state_dict(converted, strict=False)

        self._base_state = {k: v.detach().cpu().to(torch.float32).clone()
                            for k, v in converted.items()}
        self._base_hashes = dict(version.param_hashes)

    def wait_all(self, timeout: Optional[float] = None) -> None:
        """Block until all pending async writes have completed."""
        self._writer.wait_all(timeout=timeout)
        for fut in self._pending_futures:
            if fut.done() and fut.exception() is not None:
                raise RuntimeError(
                    f"Async write failed: {fut.exception()}"
                ) from fut.exception()
        self._pending_futures.clear()

    def close(self) -> None:
        """Flush and shut down the background writer."""
        self._writer.shutdown(wait=True)

    def __enter__(self) -> "CheckpointManager":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def list_checkpoints(self) -> list[dict]:
        """Return a list of all checkpoint metadata dicts."""
        return [v.to_dict() for v in self._manifest.versions]

    def storage_stats(self) -> dict:
        """Return storage statistics."""
        return {
            "total_blobs": len(self._store.all_hashes()),
            "total_bytes": self._store.total_bytes(),
            "num_versions": len(self._manifest),
            "lifecycle_summary": self._lifecycle.summary(),
        }

    def __repr__(self) -> str:
        return (
            f"CheckpointManager("
            f"save_dir={self.save_dir!r}, "
            f"versions={len(self._manifest)}, "
            f"backend={self._delta_engine.backend!r}"
            f")"
        )

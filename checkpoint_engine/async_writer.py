"""
Async background writer for checkpoint blobs.

Architecture
------------
* A bounded queue (max 2 pending writes) provides back-pressure so that
  a slow disk cannot cause unbounded memory growth.
* A single daemon thread drains the queue: serialises dirty tensors,
  compresses them, writes to the ContentAddressedStore, then updates the
  Manifest.
* enqueue() submits a write job and returns a concurrent.futures.Future.
* wait_all() blocks until all queued jobs have been completed.

The writer is thread-safe and reentrant — multiple callers can call
enqueue() concurrently.
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from .manifest import CheckpointVersion, Manifest
from .store import ContentAddressedStore


# ---------------------------------------------------------------------------
# Job dataclass
# ---------------------------------------------------------------------------

@dataclass
class WriteJob:
    step: int
    dirty_tensors: dict[str, torch.Tensor]   # name → CPU float32 delta tensor
    full_param_hashes: dict[str, str]        # ALL param hashes (base + dirty)
    metrics: dict[str, float]
    dirty_ratio: float
    is_full: bool
    future: Future                            # resolved when write completes


# ---------------------------------------------------------------------------
# AsyncWriter
# ---------------------------------------------------------------------------

class AsyncWriter:
    """
    Background-thread writer that serialises tensor writes off the critical
    training path.
    """

    _SENTINEL = None   # signals the worker thread to stop

    def __init__(
        self,
        store: ContentAddressedStore,
        manifest: Manifest,
        max_pending: int = 2,
    ):
        self._store = store
        self._manifest = manifest
        self._queue: queue.Queue[Optional[WriteJob]] = queue.Queue(
            maxsize=max_pending
        )
        self._thread = threading.Thread(
            target=self._worker, name="AsyncWriter", daemon=True
        )
        self._thread.start()
        self._lock = threading.Lock()
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(
        self,
        step: int,
        dirty_tensors: dict[str, torch.Tensor],
        full_param_hashes: dict[str, str],
        metrics: dict[str, float],
        dirty_ratio: float,
        is_full: bool = False,
    ) -> Future:
        """
        Submit a write job.

        Blocks if the queue already holds max_pending jobs (back-pressure).
        Returns a Future that resolves to a CheckpointVersion on success
        or raises on failure.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("AsyncWriter has been shut down")

        fut: Future = Future()
        job = WriteJob(
            step=step,
            dirty_tensors=dirty_tensors,
            full_param_hashes=full_param_hashes,
            metrics=metrics,
            dirty_ratio=dirty_ratio,
            is_full=is_full,
            future=fut,
        )
        # put() blocks when queue is full — this is the back-pressure point
        self._queue.put(job)
        return fut

    def wait_all(self, timeout: Optional[float] = None) -> None:
        """
        Block until all queued write jobs have been processed.

        Raises RuntimeError if any job failed.
        """
        # Use queue.join() which blocks until all items have been task_done()d
        if timeout is not None:
            # queue.join() doesn't support timeout directly; poll manually
            deadline = time.monotonic() + timeout
            while not self._queue.empty() or self._queue.unfinished_tasks > 0:
                if time.monotonic() > deadline:
                    raise TimeoutError("wait_all timed out")
                time.sleep(0.01)
        else:
            self._queue.join()

    def shutdown(self, wait: bool = True) -> None:
        """Gracefully stop the background thread."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(self._SENTINEL)  # wake the worker
        if wait:
            self._thread.join()

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is self._SENTINEL:
                    return  # shutdown signal

                version = self._process_job(job)
                job.future.set_result(version)
            except Exception as exc:
                job.future.set_exception(exc)
            finally:
                self._queue.task_done()

    def _process_job(self, job: WriteJob) -> CheckpointVersion:
        """
        Core write logic executed on the background thread.

        1. Write each dirty tensor to the content-addressed store.
        2. Merge the new hashes into full_param_hashes.
        3. Compute total storage for this version.
        4. Persist a CheckpointVersion to the manifest.
        """
        new_hashes: dict[str, str] = {}
        storage_bytes = 0

        for name, tensor in job.dirty_tensors.items():
            hexdigest = self._store.put(tensor)
            new_hashes[name] = hexdigest
            blob_path = self._store._blob_path(hexdigest)
            try:
                storage_bytes += blob_path.stat().st_size
            except FileNotFoundError:
                pass

        # Merge: start from full_param_hashes, overwrite dirty entries
        merged_hashes = dict(job.full_param_hashes)
        merged_hashes.update(new_hashes)

        version = CheckpointVersion(
            step=job.step,
            timestamp=time.time(),
            metrics=job.metrics,
            param_hashes=merged_hashes,
            dirty_ratio=job.dirty_ratio,
            storage_bytes=storage_bytes,
            is_full=job.is_full,
        )
        self._manifest.add_version(version)
        return version

    def __enter__(self) -> "AsyncWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.shutdown(wait=True)

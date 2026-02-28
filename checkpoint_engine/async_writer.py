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

import logging
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from .manifest import CheckpointVersion, Manifest
from .store import ContentAddressedStore

logger = logging.getLogger(__name__)


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
        self._all_done = threading.Condition()
        self._pending_count = 0
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
        with self._all_done:
            self._pending_count += 1
        # put() blocks when queue is full — this is the back-pressure point
        self._queue.put(job)
        return fut

    def wait_all(self, timeout: Optional[float] = None) -> None:
        """
        Block until all queued write jobs have been processed.

        Uses a condition variable for efficient waiting instead of polling.
        Raises TimeoutError if timeout is exceeded.
        """
        with self._all_done:
            if not self._all_done.wait_for(
                lambda: self._pending_count == 0,
                timeout=timeout,
            ):
                raise TimeoutError("wait_all timed out")

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
                logger.error("Async write failed for step %s: %s", job.step, exc)
                job.future.set_exception(exc)
            finally:
                self._queue.task_done()
                with self._all_done:
                    self._pending_count -= 1
                    self._all_done.notify_all()

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

        names = list(job.dirty_tensors.keys())
        tensors = list(job.dirty_tensors.values())

        if names:
            hex_digests = self._store.put_batch(tensors)
            for name, hexdigest in zip(names, hex_digests):
                new_hashes[name] = hexdigest
                try:
                    storage_bytes += self._store.blob_size(hexdigest)
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

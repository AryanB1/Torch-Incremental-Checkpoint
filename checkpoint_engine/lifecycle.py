"""
Lifecycle Manager — garbage collection of old checkpoint versions.

Retention policy
----------------
* Keep the last N steps unconditionally (keep_last_n).
* Keep the best N steps by a nominated metric (keep_best_n), e.g. lowest
  validation loss.
* The first (base) checkpoint is always kept because all incremental
  checkpoints are relative to it.
* Any blob not referenced by a surviving version is orphaned and eligible
  for deletion.
"""

from __future__ import annotations

from typing import Optional

from .manifest import CheckpointVersion, Manifest
from .store import ContentAddressedStore


class LifecycleManager:
    """
    Computes and applies retention policies for checkpoint versions.

    Parameters
    ----------
    manifest       : Manifest
    store          : ContentAddressedStore
    keep_last_n    : int   — always retain the N most-recent steps
    keep_best_n    : int   — retain the N steps with the best metric value
    metric_key     : str   — which metrics dict key to optimise
    lower_is_better: bool  — True for loss, False for accuracy
    """

    def __init__(
        self,
        manifest: Manifest,
        store: ContentAddressedStore,
        keep_last_n: int = 5,
        keep_best_n: int = 3,
        metric_key: str = "loss",
        lower_is_better: bool = True,
    ):
        self.manifest = manifest
        self.store = store
        self.keep_last_n = keep_last_n
        self.keep_best_n = keep_best_n
        self.metric_key = metric_key
        self.lower_is_better = lower_is_better

    # ------------------------------------------------------------------
    # Policy computation
    # ------------------------------------------------------------------

    def get_versions_to_keep(self) -> set[int]:
        """
        Return the set of step numbers that must be retained.

        A step is kept if it satisfies ANY of:
          - It is the base (oldest) checkpoint
          - It is among the last keep_last_n steps
          - It is among the best keep_best_n steps by metric
        """
        versions = self.manifest.versions
        if not versions:
            return set()

        keep: set[int] = set()

        # Always keep the base
        keep.add(versions[0].step)

        # Keep last N
        for v in versions[-self.keep_last_n:]:
            keep.add(v.step)

        # Keep best N by metric
        scored = [
            v for v in versions
            if self.metric_key in v.metrics
        ]
        if scored:
            scored_sorted = sorted(
                scored,
                key=lambda v: v.metrics[self.metric_key],
                reverse=not self.lower_is_better,
            )
            for v in scored_sorted[:self.keep_best_n]:
                keep.add(v.step)

        return keep

    def get_versions_to_delete(self) -> list[int]:
        """
        Return step numbers that are safe to delete.

        Safe = not in keep set.
        """
        keep = self.get_versions_to_keep()
        all_steps = self.manifest.steps()
        return [s for s in all_steps if s not in keep]

    def get_orphaned_blobs(self, retained_steps: Optional[set[int]] = None) -> list[str]:
        """
        Return blob hashes that are no longer referenced by any retained version.

        Parameters
        ----------
        retained_steps : set[int] | None
            If provided, treat only those steps as retained; otherwise
            uses get_versions_to_keep().
        """
        if retained_steps is None:
            retained_steps = self.get_versions_to_keep()

        # Collect hashes referenced by retained versions
        referenced: set[str] = set()
        for v in self.manifest.versions:
            if v.step in retained_steps:
                referenced.update(v.param_hashes.values())

        # Compare against everything in the store
        all_hashes = set(self.store.all_hashes())
        return list(all_hashes - referenced)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_gc(self, dry_run: bool = False) -> dict[str, list]:
        """
        Execute garbage collection.

        1. Determine versions to delete.
        2. Remove those versions from the manifest.
        3. Find orphaned blobs.
        4. Delete orphaned blobs from the store.

        Parameters
        ----------
        dry_run : bool
            If True, compute what would be deleted but make no changes.

        Returns
        -------
        dict with keys:
          "deleted_steps"  : list[int]  — steps removed from manifest
          "deleted_blobs"  : list[str]  — blob hashes deleted from store
          "freed_bytes"    : int        — approximate bytes reclaimed
        """
        steps_to_delete = self.get_versions_to_delete()
        retained_steps = self.get_versions_to_keep()

        # Compute orphaned blobs *before* removing versions so we can
        # accurately compute freed_bytes
        # First simulate manifest without the to-delete steps
        orphaned = self.get_orphaned_blobs(retained_steps)

        freed_bytes = 0
        deleted_blobs: list[str] = []
        deleted_steps: list[int] = []

        if not dry_run:
            # Remove manifest entries
            for step in steps_to_delete:
                if self.manifest.remove_version(step):
                    deleted_steps.append(step)

            # Delete orphaned blobs
            for hexdigest in orphaned:
                try:
                    size = self.store._blob_path(hexdigest).stat().st_size
                except FileNotFoundError:
                    size = 0
                if self.store.delete(hexdigest):
                    deleted_blobs.append(hexdigest)
                    freed_bytes += size
        else:
            deleted_steps = list(steps_to_delete)
            deleted_blobs = orphaned
            for hexdigest in orphaned:
                try:
                    freed_bytes += self.store._blob_path(hexdigest).stat().st_size
                except FileNotFoundError:
                    pass

        return {
            "deleted_steps": deleted_steps,
            "deleted_blobs": deleted_blobs,
            "freed_bytes": freed_bytes,
        }

    def summary(self) -> dict:
        """Return a human-readable summary of the current lifecycle state."""
        keep = self.get_versions_to_keep()
        delete = self.get_versions_to_delete()
        orphans = self.get_orphaned_blobs(keep)
        return {
            "total_versions": len(self.manifest),
            "versions_to_keep": sorted(keep),
            "versions_to_delete": sorted(delete),
            "orphaned_blobs": len(orphans),
        }

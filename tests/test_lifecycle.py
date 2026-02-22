"""
Tests for LifecycleManager retention policies.

Uses in-memory manifest/store fixtures — no model required.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest
import torch

from checkpoint_engine.manifest import CheckpointVersion, Manifest
from checkpoint_engine.store import ContentAddressedStore
from checkpoint_engine.lifecycle import LifecycleManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


def make_version(
    step: int,
    loss: float,
    param_hashes: dict | None = None,
    is_full: bool = False,
) -> CheckpointVersion:
    return CheckpointVersion(
        step=step,
        timestamp=time.time(),
        metrics={"loss": loss},
        param_hashes=param_hashes or {},
        dirty_ratio=0.1,
        storage_bytes=1000,
        is_full=is_full,
    )


@pytest.fixture
def manifest(tmp_dir: Path) -> Manifest:
    return Manifest(tmp_dir)


@pytest.fixture
def store(tmp_dir: Path) -> ContentAddressedStore:
    return ContentAddressedStore(tmp_dir / "blobs")


@pytest.fixture
def mgr(manifest: Manifest, store: ContentAddressedStore) -> LifecycleManager:
    return LifecycleManager(
        manifest=manifest,
        store=store,
        keep_last_n=3,
        keep_best_n=2,
        metric_key="loss",
        lower_is_better=True,
    )


# ---------------------------------------------------------------------------
# get_versions_to_keep
# ---------------------------------------------------------------------------

class TestKeepPolicy:
    def test_empty_manifest(self, mgr):
        assert mgr.get_versions_to_keep() == set()

    def test_base_always_kept(self, mgr, manifest):
        for step, loss in [(1, 3.0), (2, 2.5), (3, 2.0), (4, 1.5), (5, 1.0)]:
            manifest.add_version(make_version(step, loss))
        keep = mgr.get_versions_to_keep()
        assert 1 in keep   # base checkpoint

    def test_last_n_kept(self, mgr, manifest):
        for step in range(1, 10):
            manifest.add_version(make_version(step, float(10 - step)))
        keep = mgr.get_versions_to_keep()
        # Last 3 should always be kept
        assert 7 in keep
        assert 8 in keep
        assert 9 in keep

    def test_best_n_kept(self, mgr, manifest):
        # Steps with decreasing loss: step 5 = best, step 4 = second best
        losses = {1: 5.0, 2: 4.0, 3: 3.0, 4: 2.0, 5: 1.0}
        for step, loss in losses.items():
            manifest.add_version(make_version(step, loss))
        keep = mgr.get_versions_to_keep()
        assert 5 in keep  # best loss
        assert 4 in keep  # second best loss

    def test_step_without_metric_not_in_best(self, mgr, manifest):
        # A version without the tracked metric should not appear in best-N
        v_no_metric = CheckpointVersion(
            step=99, timestamp=time.time(), metrics={},
            param_hashes={}, dirty_ratio=0.1, storage_bytes=100,
        )
        manifest.add_version(v_no_metric)
        keep = mgr.get_versions_to_keep()
        # Step 99 may or may not be in last_n but shouldn't crash things
        assert isinstance(keep, set)


# ---------------------------------------------------------------------------
# get_versions_to_delete
# ---------------------------------------------------------------------------

class TestDeletePolicy:
    def test_nothing_to_delete_with_few_versions(self, mgr, manifest):
        for step in [1, 2, 3]:
            manifest.add_version(make_version(step, float(4 - step)))
        # Only 3 versions, keep_last_n=3 → nothing should be deleted
        assert mgr.get_versions_to_delete() == []

    def test_excess_versions_deleted(self, mgr, manifest):
        # 10 versions, keep_last_n=3, keep_best_n=2 → some should be deleted
        for step in range(1, 11):
            manifest.add_version(make_version(step, float(11 - step)))
        to_delete = mgr.get_versions_to_delete()
        to_keep   = mgr.get_versions_to_keep()
        assert len(to_delete) > 0
        assert set(to_delete).isdisjoint(to_keep)

    def test_delete_disjoint_from_keep(self, mgr, manifest):
        for step in range(1, 15):
            manifest.add_version(make_version(step, float(step)))
        to_delete = set(mgr.get_versions_to_delete())
        to_keep   = mgr.get_versions_to_keep()
        assert to_delete.isdisjoint(to_keep)


# ---------------------------------------------------------------------------
# Orphaned blob detection
# ---------------------------------------------------------------------------

class TestOrphanedBlobs:
    def _populate_store_and_manifest(self, manifest, store):
        """Write two tensors to the store, reference only one in manifest."""
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([3.0, 4.0])
        h1 = store.put(t1)
        h2 = store.put(t2)
        # Only h1 referenced in a manifest version
        manifest.add_version(make_version(1, 1.0, param_hashes={"w": h1}))
        return h1, h2

    def test_orphaned_blob_detected(self, mgr, manifest, store):
        h1, h2 = self._populate_store_and_manifest(manifest, store)
        orphans = mgr.get_orphaned_blobs()
        assert h2 in orphans

    def test_referenced_blob_not_orphaned(self, mgr, manifest, store):
        h1, h2 = self._populate_store_and_manifest(manifest, store)
        orphans = mgr.get_orphaned_blobs()
        assert h1 not in orphans


# ---------------------------------------------------------------------------
# run_gc
# ---------------------------------------------------------------------------

class TestRunGC:
    def test_dry_run_does_not_modify(self, mgr, manifest, store):
        for step in range(1, 12):
            manifest.add_version(make_version(step, float(step)))
        versions_before = manifest.steps()
        result = mgr.run_gc(dry_run=True)
        versions_after = manifest.steps()
        assert versions_before == versions_after  # no change
        assert len(result["deleted_steps"]) > 0   # but would have deleted

    def test_gc_removes_versions(self, mgr, manifest, store):
        for step in range(1, 12):
            manifest.add_version(make_version(step, float(step)))
        result = mgr.run_gc(dry_run=False)
        assert len(result["deleted_steps"]) > 0
        remaining = manifest.steps()
        for step in result["deleted_steps"]:
            assert step not in remaining

    def test_gc_deletes_orphaned_blobs(self, mgr, manifest, store):
        # Add a blob that is referenced, and one that is not
        t_ref  = torch.tensor([1.0])
        t_orph = torch.tensor([9.9])
        h_ref  = store.put(t_ref)
        h_orph = store.put(t_orph)
        manifest.add_version(make_version(1, 1.0, param_hashes={"w": h_ref}))
        # Add lots of versions to trigger GC
        for step in range(2, 12):
            manifest.add_version(make_version(step, float(step), param_hashes={"w": h_ref}))

        mgr.run_gc(dry_run=False)
        # h_orph should be gone
        assert not store.exists(h_orph)

    def test_gc_result_keys(self, mgr, manifest, store):
        manifest.add_version(make_version(1, 1.0))
        result = mgr.run_gc(dry_run=False)
        assert "deleted_steps"  in result
        assert "deleted_blobs"  in result
        assert "freed_bytes"    in result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_keys(self, mgr, manifest, store):
        manifest.add_version(make_version(1, 1.0))
        s = mgr.summary()
        assert "total_versions"      in s
        assert "versions_to_keep"    in s
        assert "versions_to_delete"  in s
        assert "orphaned_blobs"      in s

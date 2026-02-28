"""
Tests for ContentAddressedStore.

Covers: write, read, deduplication, delete, all_hashes, total_bytes, gc.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from checkpoint_engine.store import ContentAddressedStore


@pytest.fixture
def store(tmp_path: Path) -> ContentAddressedStore:
    return ContentAddressedStore(tmp_path / "blobs")


# ---------------------------------------------------------------------------
# put / get roundtrip
# ---------------------------------------------------------------------------

class TestPutGet:
    def test_scalar_tensor(self, store):
        t = torch.tensor(3.14)
        h = store.put(t)
        assert isinstance(h, str) and len(h) == 64
        out = store.get(h)
        assert torch.allclose(out.float(), t.float(), atol=1e-5)

    def test_2d_tensor(self, store):
        t = torch.randn(64, 128)
        h = store.put(t)
        out = store.get(h)
        assert out.shape == t.shape
        assert torch.allclose(out, t, atol=1e-5)

    def test_bfloat16_tensor(self, store):
        t = torch.randn(32, 32, dtype=torch.bfloat16)
        h = store.put(t)
        out = store.get(h)
        # bfloat16 is preserved through serialization
        assert out.dtype == torch.bfloat16
        assert out.shape == t.shape

    def test_large_tensor(self, store):
        t = torch.randn(512, 512)
        h = store.put(t)
        out = store.get(h)
        assert torch.allclose(out, t, atol=1e-5)

    def test_get_nonexistent_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.get("a" * 64)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDedup:
    def test_identical_tensors_same_hash(self, store):
        t = torch.ones(10)
        h1 = store.put(t)
        h2 = store.put(t)
        assert h1 == h2

    def test_blob_written_once(self, store):
        t = torch.ones(10)
        store.put(t)
        before = len(store.all_hashes())
        store.put(t)  # second put should be a no-op
        after = len(store.all_hashes())
        assert before == after == 1

    def test_different_tensors_different_hash(self, store):
        t1 = torch.ones(10)
        t2 = torch.zeros(10)
        h1 = store.put(t1)
        h2 = store.put(t2)
        assert h1 != h2
        assert len(store.all_hashes()) == 2


# ---------------------------------------------------------------------------
# exists / delete
# ---------------------------------------------------------------------------

class TestExistsDelete:
    def test_exists_true_after_put(self, store):
        h = store.put(torch.randn(5))
        assert store.exists(h)

    def test_exists_false_before_put(self, store):
        assert not store.exists("b" * 64)

    def test_delete_returns_true_when_present(self, store):
        h = store.put(torch.randn(5))
        assert store.delete(h) is True

    def test_delete_removes_blob(self, store):
        h = store.put(torch.randn(5))
        store.delete(h)
        assert not store.exists(h)
        assert len(store.all_hashes()) == 0

    def test_delete_returns_false_when_absent(self, store):
        assert store.delete("c" * 64) is False


# ---------------------------------------------------------------------------
# all_hashes / total_bytes
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_all_hashes_empty_initially(self, store):
        assert store.all_hashes() == []

    def test_all_hashes_accumulates(self, store):
        hashes = []
        for i in range(5):
            h = store.put(torch.tensor(float(i)))
            hashes.append(h)
        stored = set(store.all_hashes())
        assert set(hashes) == stored

    def test_total_bytes_grows_with_puts(self, store):
        assert store.total_bytes() == 0
        store.put(torch.randn(100))
        assert store.total_bytes() > 0


# ---------------------------------------------------------------------------
# Subdirectory layout (git-like)
# ---------------------------------------------------------------------------

class TestLayout:
    def test_subdir_structure(self, store, tmp_path):
        h = store.put(torch.tensor(1.0))
        subdir = tmp_path / "blobs" / h[:2]
        blob   = subdir / h[2:]
        assert subdir.is_dir()
        assert blob.is_file()


# ---------------------------------------------------------------------------
# Integrity verification
# ---------------------------------------------------------------------------

class TestIntegrity:
    def test_get_with_verify_passes_valid_blob(self, store):
        t = torch.randn(16)
        h = store.put(t)
        out = store.get(h, verify=True)
        assert torch.allclose(out, t, atol=1e-5)

    def test_get_detects_corrupted_blob(self, store, tmp_path):
        t = torch.randn(16)
        h = store.put(t)
        # Corrupt the blob on disk
        blob_path = store._blob_path(h)
        data = blob_path.read_bytes()
        corrupted = bytes([b ^ 0xFF for b in data[:8]]) + data[8:]
        blob_path.write_bytes(corrupted)
        with pytest.raises(ValueError, match="integrity check failed"):
            store.get(h, verify=True)

    def test_get_without_verify_skips_check(self, store):
        t = torch.randn(16)
        h = store.put(t)
        out = store.get(h, verify=False)
        assert torch.allclose(out, t, atol=1e-5)


# ---------------------------------------------------------------------------
# blob_size
# ---------------------------------------------------------------------------

class TestBlobSize:
    def test_blob_size_returns_positive(self, store):
        h = store.put(torch.randn(100))
        assert store.blob_size(h) > 0

    def test_blob_size_nonexistent_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.blob_size("d" * 64)

    def test_blob_size_matches_total_bytes(self, store):
        hashes = [store.put(torch.tensor(float(i))) for i in range(3)]
        total = sum(store.blob_size(h) for h in hashes)
        assert total == store.total_bytes()


# ---------------------------------------------------------------------------
# put_batch
# ---------------------------------------------------------------------------

class TestPutBatch:
    def test_batch_returns_correct_count(self, store):
        tensors = [torch.randn(64, 64) for _ in range(5)]
        hashes = store.put_batch(tensors)
        assert len(hashes) == 5
        assert all(len(h) == 64 for h in hashes)

    def test_batch_blobs_readable_by_get(self, store):
        """Blobs written by put_batch must be deserializable by get()."""
        tensors = [torch.randn(16, 16) for _ in range(3)]
        hashes = store.put_batch(tensors)
        for orig, h in zip(tensors, hashes):
            recovered = store.get(h)
            assert torch.allclose(recovered, orig, atol=1e-5)

    def test_batch_dedup(self, store):
        """Identical tensors in a batch should produce the same hash."""
        t = torch.ones(10)
        hashes = store.put_batch([t, t, t])
        assert hashes[0] == hashes[1] == hashes[2]

    def test_batch_empty(self, store):
        assert store.put_batch([]) == []

    def test_batch_bfloat16(self, store):
        """bfloat16 tensors should be handled correctly."""
        t = torch.randn(8, 8, dtype=torch.bfloat16)
        hashes = store.put_batch([t])
        out = store.get(hashes[0])
        assert out.dtype == torch.bfloat16

    def test_batch_large_tensors(self, store):
        """Batch of larger tensors should work correctly."""
        tensors = [torch.randn(256, 256) for _ in range(4)]
        hashes = store.put_batch(tensors)
        assert len(set(hashes)) == 4  # all different
        for orig, h in zip(tensors, hashes):
            recovered = store.get(h)
            assert torch.allclose(recovered, orig, atol=1e-5)

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
        # put() casts to float32 internally; recovered tensor is float32
        assert out.dtype == torch.float32
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

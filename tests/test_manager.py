"""
Tests for CheckpointManager: save/restore roundtrip on a small toy model.

Uses a 2-layer MLP — no Llama required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from checkpoint_engine import CheckpointManager


# ---------------------------------------------------------------------------
# Toy model
# ---------------------------------------------------------------------------

class ToyMLP(nn.Module):
    def __init__(self, in_features: int = 16, hidden: int = 32, out: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture
def model() -> ToyMLP:
    torch.manual_seed(42)
    return ToyMLP()


@pytest.fixture
def save_dir(tmp_path: Path) -> Path:
    return tmp_path / "checkpoints"


# ---------------------------------------------------------------------------
# Basic save / restore
# ---------------------------------------------------------------------------

class TestSaveRestore:
    def test_save_creates_manifest(self, model, save_dir):
        with CheckpointManager(str(save_dir), model, async_write=False) as mgr:
            mgr.save(step=1, metrics={"loss": 1.0})
        manifest_path = save_dir / "manifest.json"
        assert manifest_path.exists()

    def test_save_creates_blobs(self, model, save_dir):
        with CheckpointManager(str(save_dir), model, async_write=False) as mgr:
            mgr.save(step=1)
        blobs = list((save_dir / "blobs").rglob("*"))
        assert len(blobs) > 0

    def test_restore_recovers_weights(self, model, save_dir):
        # Capture original weights
        original_sd = {k: v.clone() for k, v in model.state_dict().items()}

        with CheckpointManager(str(save_dir), model, async_write=False) as mgr:
            mgr.save(step=1)
            # Perturb model
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * 10.0)
            # Restore
            mgr.restore(step=1)

        for name, original in original_sd.items():
            restored = model.state_dict()[name].float()
            assert torch.allclose(restored, original.float(), atol=1e-4), (
                f"Parameter {name} not restored correctly"
            )

    def test_restore_latest(self, model, save_dir):
        original_sd = {k: v.clone() for k, v in model.state_dict().items()}

        with CheckpointManager(str(save_dir), model, async_write=False) as mgr:
            mgr.save(step=10)
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * 100.0)
            mgr.restore()  # latest

        for name, original in original_sd.items():
            restored = model.state_dict()[name].float()
            assert torch.allclose(restored, original.float(), atol=1e-4)

    def test_restore_invalid_step_raises(self, model, save_dir):
        with CheckpointManager(str(save_dir), model, async_write=False) as mgr:
            mgr.save(step=1)
            with pytest.raises(ValueError, match="Step 999 not found"):
                mgr.restore(step=999)

    def test_restore_no_checkpoints_raises(self, model, save_dir):
        with CheckpointManager(str(save_dir), model, async_write=False) as mgr:
            with pytest.raises(RuntimeError, match="No checkpoints"):
                mgr.restore()


# ---------------------------------------------------------------------------
# Incremental correctness (multiple saves)
# ---------------------------------------------------------------------------

class TestIncrementalSaves:
    def test_multi_step_roundtrip(self, model, save_dir):
        snapshots: dict[int, dict[str, torch.Tensor]] = {}

        with CheckpointManager(
            str(save_dir), model, keep_last_n=10, keep_best_n=10,
            dirty_threshold=1e-5, async_write=False
        ) as mgr:
            for step in [1, 2, 3, 4, 5]:
                snapshots[step] = {
                    k: v.clone() for k, v in model.state_dict().items()
                }
                mgr.save(step=step)
                # Perturb for next step
                with torch.no_grad():
                    for p in model.parameters():
                        p.add_(torch.randn_like(p) * 0.01)

            # Restore each step and compare
            for step, expected_sd in snapshots.items():
                mgr.restore(step=step)
                for name, expected in expected_sd.items():
                    actual = model.state_dict()[name].float()
                    assert torch.allclose(actual, expected.float(), atol=1e-4), (
                        f"Step {step}, param {name}: mismatch"
                    )

    def test_dirty_ratio_decreases_without_updates(self, model, save_dir):
        """When model weights don't change, dirty ratio should be ~0 after step 1."""
        ratios: list[float] = []

        with CheckpointManager(
            str(save_dir), model, dirty_threshold=1e-4, async_write=False
        ) as mgr:
            for step in range(1, 4):
                mgr.save(step=step)
                r = mgr.metrics.records[-1].dirty_ratio if mgr.metrics.records else 1.0
                ratios.append(r)

        # Step 1 is a full save (ratio = 1.0), subsequent steps with no
        # weight changes should have ratio ≈ 0
        assert ratios[0] == pytest.approx(1.0)
        assert ratios[1] == pytest.approx(0.0, abs=0.05)
        assert ratios[2] == pytest.approx(0.0, abs=0.05)


# ---------------------------------------------------------------------------
# Async write
# ---------------------------------------------------------------------------

class TestAsyncWrite:
    def test_async_save_restore(self, model, save_dir):
        original_sd = {k: v.clone() for k, v in model.state_dict().items()}

        with CheckpointManager(str(save_dir), model, async_write=True) as mgr:
            mgr.save(step=1)
            mgr.wait_all()
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * 5.0)
            mgr.restore(step=1)

        for name, original in original_sd.items():
            restored = model.state_dict()[name].float()
            assert torch.allclose(restored, original.float(), atol=1e-4)


# ---------------------------------------------------------------------------
# list_checkpoints / storage_stats
# ---------------------------------------------------------------------------

class TestInfo:
    def test_list_checkpoints(self, model, save_dir):
        with CheckpointManager(str(save_dir), model, async_write=False) as mgr:
            mgr.save(step=10, metrics={"loss": 2.0})
            mgr.save(step=20, metrics={"loss": 1.5})
            ckpts = mgr.list_checkpoints()
        assert len(ckpts) >= 1
        steps = [c["step"] for c in ckpts]
        assert 10 in steps

    def test_storage_stats_keys(self, model, save_dir):
        with CheckpointManager(str(save_dir), model, async_write=False) as mgr:
            mgr.save(step=1)
            stats = mgr.storage_stats()
        assert "total_bytes" in stats
        assert "num_versions" in stats
        assert stats["total_bytes"] > 0

"""
Tests for DeltaEngine (pure-Python path).

Uses known tensors so that dirty detection is deterministic.
"""

from __future__ import annotations

import pytest
import torch

from checkpoint_engine.delta import DeltaEngine, DeltaResult


# Always force pure-Python so tests don't depend on C++ build
@pytest.fixture
def engine() -> DeltaEngine:
    return DeltaEngine(threshold=1e-4, use_cpp=False)


def make_state(seed: int = 0) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {
        "layer.0.weight": torch.randn(32, 32),
        "layer.0.bias":   torch.randn(32),
        "layer.1.weight": torch.randn(16, 32),
        "layer.1.bias":   torch.randn(16),
    }


# ---------------------------------------------------------------------------
# Basic detection
# ---------------------------------------------------------------------------

class TestDirtyDetection:
    def test_no_change_is_clean(self, engine):
        base = make_state(0)
        current = {k: v.clone() for k, v in base.items()}
        result = engine.compute_dirty(current, base)
        assert result.dirty_names == []
        assert result.dirty_ratio == pytest.approx(0.0)

    def test_large_change_is_dirty(self, engine):
        base = make_state(0)
        current = {k: v.clone() for k, v in base.items()}
        # Add large perturbation to one layer
        current["layer.0.weight"] = current["layer.0.weight"] + 10.0
        result = engine.compute_dirty(current, base)
        assert "layer.0.weight" in result.dirty_names

    def test_small_change_is_clean(self, engine):
        base = make_state(0)
        current = {k: v.clone() for k, v in base.items()}
        # Perturbation smaller than threshold
        current["layer.0.weight"] = current["layer.0.weight"] + 1e-8
        result = engine.compute_dirty(current, base)
        assert "layer.0.weight" not in result.dirty_names

    def test_multiple_dirty_layers(self, engine):
        base = make_state(0)
        current = {k: v.clone() for k, v in base.items()}
        current["layer.0.weight"] += 5.0
        current["layer.1.bias"]   += 5.0
        result = engine.compute_dirty(current, base)
        assert "layer.0.weight" in result.dirty_names
        assert "layer.1.bias"   in result.dirty_names

    def test_dirty_ratio_correct(self, engine):
        base = make_state(0)
        current = {k: v.clone() for k, v in base.items()}
        current["layer.0.weight"] += 5.0  # 1 out of 4 tensors dirty
        result = engine.compute_dirty(current, base)
        assert result.dirty_ratio == pytest.approx(1 / 4)


# ---------------------------------------------------------------------------
# Delta correctness
# ---------------------------------------------------------------------------

class TestDeltaValues:
    def test_delta_is_difference(self, engine):
        base = make_state(0)
        current = {k: v.clone() for k, v in base.items()}
        noise = torch.ones(32, 32) * 2.0
        current["layer.0.weight"] = base["layer.0.weight"] + noise

        result = engine.compute_dirty(current, base)
        delta = result.deltas["layer.0.weight"].float()
        expected = noise.float()
        assert torch.allclose(delta, expected, atol=1e-4)

    def test_deltas_only_for_dirty(self, engine):
        base = make_state(0)
        current = {k: v.clone() for k, v in base.items()}
        current["layer.0.weight"] += 5.0

        result = engine.compute_dirty(current, base)
        assert set(result.deltas.keys()) == set(result.dirty_names)
        assert "layer.1.weight" not in result.deltas


# ---------------------------------------------------------------------------
# Per-layer norms
# ---------------------------------------------------------------------------

class TestPerLayerNorms:
    def test_norms_positive(self, engine):
        base = make_state(0)
        current = {k: v.clone() for k, v in base.items()}
        current["layer.0.weight"] += 1.0
        result = engine.compute_dirty(current, base)
        for norm in result.per_layer_norms.values():
            assert norm > 0

    def test_norm_ordering(self, engine):
        base = make_state(0)
        current = {k: v.clone() for k, v in base.items()}
        current["layer.0.weight"] += 10.0
        current["layer.1.weight"] += 0.5
        result = engine.compute_dirty(current, base)
        if "layer.0.weight" in result.per_layer_norms and \
           "layer.1.weight" in result.per_layer_norms:
            assert (result.per_layer_norms["layer.0.weight"] >
                    result.per_layer_norms["layer.1.weight"])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_state_dicts(self, engine):
        result = engine.compute_dirty({}, {})
        assert result.dirty_names == []
        assert result.dirty_ratio == pytest.approx(0.0)

    def test_disjoint_keys_uses_intersection(self, engine):
        base    = {"a": torch.zeros(5)}
        current = {"b": torch.zeros(5)}   # no common keys
        result = engine.compute_dirty(current, base)
        assert result.dirty_names == []

    def test_threshold_boundary(self):
        engine_strict = DeltaEngine(threshold=0.0, use_cpp=False)
        base = {"w": torch.ones(10)}
        current = {"w": torch.ones(10) + 1e-9}   # tiny change
        result = engine_strict.compute_dirty(current, base)
        # With threshold=0, even tiny changes count
        assert "w" in result.dirty_names

    def test_use_cpp_false_explicit(self):
        e = DeltaEngine(threshold=1e-4, use_cpp=False)
        assert e.backend == "python"

    def test_use_cpp_true_raises_if_unavailable(self):
        from checkpoint_engine.delta import _CPP_AVAILABLE
        if not _CPP_AVAILABLE:
            with pytest.raises(ImportError):
                DeltaEngine(threshold=1e-4, use_cpp=True)

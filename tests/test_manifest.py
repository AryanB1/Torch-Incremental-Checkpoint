"""
Tests for Manifest — schema validation, corruption handling, persistence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from checkpoint_engine.manifest import CheckpointVersion, Manifest


def _make_version(step: int) -> CheckpointVersion:
    return CheckpointVersion(
        step=step,
        timestamp=1000.0 + step,
        metrics={"loss": 1.0 / step},
        param_hashes={"w": "a" * 64},
        dirty_ratio=0.5,
        storage_bytes=100,
        is_full=(step == 1),
    )


class TestSchemaValidation:
    def test_loads_valid_schema_v1(self, tmp_path: Path):
        manifest_path = tmp_path / "manifest.json"
        payload = {
            "schema_version": 1,
            "versions": [
                {
                    "step": 1,
                    "timestamp": 1000.0,
                    "metrics": {"loss": 1.0},
                    "param_hashes": {"w": "a" * 64},
                    "dirty_ratio": 1.0,
                    "storage_bytes": 100,
                    "is_full": True,
                }
            ],
        }
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        m = Manifest(tmp_path)
        assert len(m) == 1
        assert m.versions[0].step == 1

    def test_rejects_unsupported_schema_version(self, tmp_path: Path):
        manifest_path = tmp_path / "manifest.json"
        payload = {
            "schema_version": 999,
            "versions": [],
        }
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        # Should warn and start empty, not crash
        m = Manifest(tmp_path)
        assert len(m) == 0

    def test_corrupted_json_creates_backup(self, tmp_path: Path):
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("{{invalid json!!", encoding="utf-8")
        m = Manifest(tmp_path)
        assert len(m) == 0
        backup = manifest_path.with_suffix(".json.bak")
        assert backup.exists()
        assert backup.read_text(encoding="utf-8") == "{{invalid json!!"


class TestPersistence:
    def test_add_version_persists(self, tmp_path: Path):
        m = Manifest(tmp_path)
        m.add_version(_make_version(1))
        # Reload from disk
        m2 = Manifest(tmp_path)
        assert len(m2) == 1
        assert m2.versions[0].step == 1

    def test_remove_version_persists(self, tmp_path: Path):
        m = Manifest(tmp_path)
        m.add_version(_make_version(1))
        m.add_version(_make_version(2))
        m.remove_version(1)
        m2 = Manifest(tmp_path)
        assert len(m2) == 1
        assert m2.versions[0].step == 2

    def test_all_referenced_hashes(self, tmp_path: Path):
        m = Manifest(tmp_path)
        v1 = CheckpointVersion(
            step=1, timestamp=1.0, metrics={},
            param_hashes={"a": "hash1", "b": "hash2"},
            dirty_ratio=1.0, storage_bytes=0, is_full=True,
        )
        v2 = CheckpointVersion(
            step=2, timestamp=2.0, metrics={},
            param_hashes={"a": "hash1", "b": "hash3"},
            dirty_ratio=0.5, storage_bytes=0,
        )
        m.add_version(v1)
        m.add_version(v2)
        assert m.all_referenced_hashes() == {"hash1", "hash2", "hash3"}

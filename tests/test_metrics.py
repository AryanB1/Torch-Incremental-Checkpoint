"""
Tests for CheckpointMetrics — export, max records cap, percentile validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from checkpoint_engine.metrics import CheckpointMetrics, _percentile


class TestMetricsExport:
    def test_export_json_creates_file(self, tmp_path: Path):
        m = CheckpointMetrics()
        m.record_save(step=1, blocking_ms=10.0, dirty_ratio=1.0, storage_bytes=100)
        m.record_save(step=2, blocking_ms=5.0, dirty_ratio=0.25, storage_bytes=200)

        out_path = tmp_path / "metrics.json"
        m.export_json(out_path)

        assert out_path.exists()
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert "summary" in data
        assert "records" in data
        assert len(data["records"]) == 2
        assert data["records"][0]["step"] == 1

    def test_export_empty_metrics(self, tmp_path: Path):
        m = CheckpointMetrics()
        out_path = tmp_path / "empty.json"
        m.export_json(out_path)
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["summary"] == {}
        assert data["records"] == []


class TestMaxRecordsCap:
    def test_records_capped_at_max(self):
        m = CheckpointMetrics(max_records=5)
        for i in range(10):
            m.record_save(step=i, blocking_ms=1.0, dirty_ratio=0.1, storage_bytes=i * 100)
        assert len(m.records) == 5
        # Should keep the most recent 5
        assert m.records[0].step == 5
        assert m.records[-1].step == 9

    def test_default_cap_is_large(self):
        m = CheckpointMetrics()
        assert m._max_records == 100_000


class TestPercentile:
    def test_basic_percentile(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(data, 50) == pytest.approx(3.0)
        assert _percentile(data, 0) == pytest.approx(1.0)
        assert _percentile(data, 100) == pytest.approx(5.0)

    def test_clamps_invalid_values(self):
        data = [1.0, 2.0, 3.0]
        assert _percentile(data, -10) == pytest.approx(1.0)
        assert _percentile(data, 200) == pytest.approx(3.0)

    def test_empty_data(self):
        assert _percentile([], 50) == 0.0

    def test_single_element(self):
        assert _percentile([42.0], 95) == pytest.approx(42.0)


class TestSummary:
    def test_summary_keys(self):
        m = CheckpointMetrics()
        m.record_save(step=1, blocking_ms=10.0, dirty_ratio=1.0, storage_bytes=100)
        s = m.summary()
        assert "num_saves" in s
        assert "avg_blocking_ms" in s
        assert "p95_blocking_ms" in s
        assert "avg_dirty_ratio" in s

    def test_empty_summary(self):
        m = CheckpointMetrics()
        assert m.summary() == {}

    def test_reset_clears(self):
        m = CheckpointMetrics()
        m.record_save(step=1, blocking_ms=10.0, dirty_ratio=1.0, storage_bytes=100)
        m.reset()
        assert len(m.records) == 0
        assert m.summary() == {}

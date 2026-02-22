"""
Checkpoint manifest: tracks every saved version with its metadata.

The manifest is a single JSON file stored at <save_dir>/manifest.json.
Writes are atomic (temp file + os.replace) so a crash mid-write never
corrupts the existing manifest.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


@dataclass
class CheckpointVersion:
    """Metadata for a single checkpoint step."""

    step: int
    timestamp: float                        # Unix epoch seconds
    metrics: dict[str, float]              # e.g. {"loss": 2.3, "perplexity": 10.1}
    param_hashes: dict[str, str]           # param_name → SHA-256 hex digest
    dirty_ratio: float                     # fraction of params that were dirty
    storage_bytes: int                     # compressed bytes for this version
    is_full: bool = False                  # True for the first (base) checkpoint

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CheckpointVersion":
        return cls(
            step=d["step"],
            timestamp=d["timestamp"],
            metrics=d.get("metrics", {}),
            param_hashes=d["param_hashes"],
            dirty_ratio=d.get("dirty_ratio", 1.0),
            storage_bytes=d.get("storage_bytes", 0),
            is_full=d.get("is_full", False),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Manifest:
    """
    Persisted list of CheckpointVersions.

    Versions are stored in step-ascending order. The manifest is loaded
    eagerly on construction and flushed explicitly via save().
    """

    _FILENAME = "manifest.json"

    def __init__(self, save_dir: str | Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.save_dir / self._FILENAME
        self._versions: list[CheckpointVersion] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._versions = [
                CheckpointVersion.from_dict(v) for v in raw.get("versions", [])
            ]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            import warnings
            warnings.warn(
                f"Manifest at {self._path} could not be parsed ({exc}). "
                "Starting with empty manifest.",
                stacklevel=2,
            )
            self._versions = []

    def _save(self) -> None:
        """Atomically write the manifest to disk."""
        payload = {
            "schema_version": 1,
            "versions": [v.to_dict() for v in self._versions],
        }
        serialised = json.dumps(payload, indent=2, ensure_ascii=False)

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self.save_dir, prefix=".manifest_tmp_", suffix=".json"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                fh.write(serialised)
            os.replace(tmp_path, self._path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @property
    def versions(self) -> list[CheckpointVersion]:
        return list(self._versions)

    def add_version(self, version: CheckpointVersion) -> None:
        """Add or replace a version for the given step and persist."""
        self._versions = [v for v in self._versions if v.step != version.step]
        self._versions.append(version)
        self._versions.sort(key=lambda v: v.step)
        self._save()

    def get_version(self, step: int) -> Optional[CheckpointVersion]:
        """Return the version for a specific step, or None."""
        for v in self._versions:
            if v.step == step:
                return v
        return None

    def latest_version(self) -> Optional[CheckpointVersion]:
        """Return the most recently saved version."""
        return self._versions[-1] if self._versions else None

    def base_version(self) -> Optional[CheckpointVersion]:
        """Return the oldest (base) version — the one all deltas build on."""
        return self._versions[0] if self._versions else None

    def remove_version(self, step: int) -> bool:
        """Remove a version by step number. Returns True if it existed."""
        before = len(self._versions)
        self._versions = [v for v in self._versions if v.step != step]
        if len(self._versions) < before:
            self._save()
            return True
        return False

    def all_referenced_hashes(self) -> set[str]:
        """Return the union of all param hashes across all versions."""
        hashes: set[str] = set()
        for v in self._versions:
            hashes.update(v.param_hashes.values())
        return hashes

    def steps(self) -> list[int]:
        return [v.step for v in self._versions]

    def __len__(self) -> int:
        return len(self._versions)

    def __repr__(self) -> str:
        return (
            f"Manifest(path={self._path!r}, versions={len(self._versions)}, "
            f"steps={self.steps()})"
        )

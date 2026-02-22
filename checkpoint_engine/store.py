"""
Content-addressed blob store for checkpoint tensors.

Layout mirrors git's object store:
    <root>/<first-2-hex-chars>/<remaining-62-hex-chars>

Each blob is a zstandard-compressed, torch.save()-serialised tensor.
Deduplication is automatic: if the SHA-256 hash already exists the write
is silently skipped.
"""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from pathlib import Path
from typing import Optional

import torch
import zstandard as zstd


_ZSTD_LEVEL = 3
_ZSTD_THREADS = 0


class ContentAddressedStore:
    """Immutable blob store addressed by SHA-256 hash of compressed content."""

    def __init__(self, root: str | Path, compression_level: int = _ZSTD_LEVEL):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._compressor = zstd.ZstdCompressor(
            level=compression_level,
            threads=_ZSTD_THREADS,
        )
        self._decompressor = zstd.ZstdDecompressor()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _blob_path(self, hexdigest: str) -> Path:
        """Return the filesystem path for a given SHA-256 hex digest."""
        return self.root / hexdigest[:2] / hexdigest[2:]

    def _serialize_tensor(self, tensor: torch.Tensor) -> bytes:
        buf = io.BytesIO()
        torch.save(tensor.cpu().contiguous(), buf)
        return buf.getvalue()

    def _deserialize_tensor(self, data: bytes) -> torch.Tensor:
        buf = io.BytesIO(data)
        return torch.load(buf, weights_only=True)

    def _compress(self, raw: bytes) -> bytes:
        return self._compressor.compress(raw)

    def _decompress(self, compressed: bytes) -> bytes:
        return self._decompressor.decompress(compressed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def put(self, tensor: torch.Tensor) -> str:
        """
        Serialize, compress, and store a tensor.

        Returns the SHA-256 hex digest (content address).
        If a blob with that hash already exists, the write is skipped
        (deduplication) and the hash is still returned.
        """
        raw = self._serialize_tensor(tensor)
        compressed = self._compress(raw)
        hexdigest = hashlib.sha256(compressed).hexdigest()

        blob_path = self._blob_path(hexdigest)
        if blob_path.exists():
            return hexdigest

        blob_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=blob_path.parent, prefix=".tmp_"
        )
        try:
            with os.fdopen(tmp_fd, "wb") as fh:
                fh.write(compressed)
            os.replace(tmp_path, blob_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        return hexdigest

    def get(self, hexdigest: str) -> torch.Tensor:
        """
        Retrieve and deserialise a tensor by its content address.

        Raises FileNotFoundError if the blob does not exist.
        """
        blob_path = self._blob_path(hexdigest)
        if not blob_path.exists():
            raise FileNotFoundError(
                f"Blob {hexdigest!r} not found in store at {self.root}"
            )
        compressed = blob_path.read_bytes()
        raw = self._decompress(compressed)
        return self._deserialize_tensor(raw)

    def exists(self, hexdigest: str) -> bool:
        """Return True if a blob with this hash is present in the store."""
        return self._blob_path(hexdigest).exists()

    def delete(self, hexdigest: str) -> bool:
        """
        Delete a blob by hash.

        Returns True if the blob existed and was deleted, False otherwise.
        """
        blob_path = self._blob_path(hexdigest)
        try:
            blob_path.unlink()
            try:
                blob_path.parent.rmdir()
            except OSError:
                pass
            return True
        except FileNotFoundError:
            return False

    def all_hashes(self) -> list[str]:
        """Return all blob hashes currently in the store."""
        hashes: list[str] = []
        for subdir in sorted(self.root.iterdir()):
            if not subdir.is_dir() or len(subdir.name) != 2:
                continue
            for blob in sorted(subdir.iterdir()):
                if blob.is_file():
                    hashes.append(subdir.name + blob.name)
        return hashes

    def total_bytes(self) -> int:
        """Return total compressed bytes stored."""
        total = 0
        for hexdigest in self.all_hashes():
            try:
                total += self._blob_path(hexdigest).stat().st_size
            except FileNotFoundError:
                pass
        return total

    def __repr__(self) -> str:
        n = len(self.all_hashes())
        return (
            f"ContentAddressedStore(root={self.root!r}, blobs={n}, "
            f"total_bytes={self.total_bytes():,})"
        )

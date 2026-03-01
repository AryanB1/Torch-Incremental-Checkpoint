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

# Detect C++ batch blob preparation capability
_BATCH_CPP_AVAILABLE = False
try:
    import delta_engine_cpp as _cpp_ext
    if getattr(_cpp_ext, "HAS_BLOB_PREPARE", False):
        _BATCH_CPP_AVAILABLE = True
except ImportError:
    _cpp_ext = None


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

    def put_batch(self, tensors: list[torch.Tensor]) -> list[str]:
        """
        Prepare and store a batch of tensors, using C++ parallel blob
        preparation when available.

        Falls back to sequential put() if the C++ extension is unavailable.

        Returns a list of SHA-256 hex digests in the same order as the input.
        """
        if not _BATCH_CPP_AVAILABLE or not tensors:
            return [self.put(t) for t in tensors]

        # Ensure tensors are CPU + contiguous for C++
        prepared = [t.cpu().contiguous() for t in tensors]

        # C++ parallel: serialize + compress + hash
        results = _cpp_ext.batch_prepare_blobs(prepared, _ZSTD_LEVEL)

        # Disk writes (sequential — inherently serial for a single disk)
        hex_digests: list[str] = []
        for sha_hex, compressed in results:
            blob_path = self._blob_path(sha_hex)
            if not blob_path.exists():
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
            hex_digests.append(sha_hex)

        return hex_digests

    def get(self, hexdigest: str, verify: bool = True) -> torch.Tensor:
        """
        Retrieve and deserialise a tensor by its content address.

        Parameters
        ----------
        hexdigest : str  — SHA-256 hex digest of the blob
        verify    : bool — if True, verify hash matches on read (detects corruption)

        Raises FileNotFoundError if the blob does not exist.
        Raises ValueError if hash verification fails (data corruption).
        """
        blob_path = self._blob_path(hexdigest)
        if not blob_path.exists():
            raise FileNotFoundError(
                f"Blob {hexdigest!r} not found in store at {self.root}"
            )
        compressed = blob_path.read_bytes()
        if verify:
            actual_hash = hashlib.sha256(compressed).hexdigest()
            if actual_hash != hexdigest:
                raise ValueError(
                    f"Blob integrity check failed for {hexdigest!r}: "
                    f"expected {hexdigest}, got {actual_hash}. "
                    f"The blob at {blob_path} may be corrupted."
                )
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

    def blob_size(self, hexdigest: str) -> int:
        """Return the compressed size in bytes of a single blob."""
        blob_path = self._blob_path(hexdigest)
        try:
            return blob_path.stat().st_size
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Blob {hexdigest!r} not found in store at {self.root}"
            )

    def all_hashes(self) -> list[str]:
        """Return all blob hashes currently in the store."""
        hashes: list[str] = []
        if not self.root.exists():
            return hashes
        for subdir in sorted(self.root.iterdir()):
            if not subdir.is_dir() or len(subdir.name) != 2:
                continue
            for blob in sorted(subdir.iterdir()):
                if blob.is_file() and not blob.name.startswith(".tmp_"):
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

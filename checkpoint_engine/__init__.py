"""
checkpoint_engine — Incremental Checkpoint Engine for PyTorch.

Public API
----------
    from checkpoint_engine import CheckpointManager

    manager = CheckpointManager(
        save_dir="./ckpts",
        model=model,
        keep_last_n=5,
        keep_best_n=3,
        dirty_threshold=1e-4,
        async_write=True,
    )
    manager.save(step=100, metrics={"loss": 2.3})
    manager.restore(step=100)
    manager.close()
"""

from .manager import CheckpointManager
from .delta import DeltaEngine, DeltaResult
from .store import ContentAddressedStore
from .manifest import Manifest, CheckpointVersion
from .lifecycle import LifecycleManager
from .async_writer import AsyncWriter
from .metrics import CheckpointMetrics

__all__ = [
    "CheckpointManager",
    "DeltaEngine",
    "DeltaResult",
    "ContentAddressedStore",
    "Manifest",
    "CheckpointVersion",
    "LifecycleManager",
    "AsyncWriter",
    "CheckpointMetrics",
]

__version__ = "0.1.0"

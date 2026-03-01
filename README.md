# PyTorch Incremental Checkpoint Engine

A drop-in replacement for `torch.save()` that implements delta-based incremental checkpointing. Instead of saving the full model state every checkpoint, it detects which parameters changed above a threshold and only writes those deltas — skipping untouched parameters entirely.

The hot path (dirty detection across all layers) is parallelised with a C++ / OpenMP extension, achieving a **19x speedup** over the pure-Python fallback on a Llama 3.2 1B model.

| Backend | Avg time | Median time | Speedup |
|---|---|---|---|
| Pure Python | 5,304 ms | 5,324 ms | 1.0x |
| C++ / OpenMP (4 threads) | 279 ms | 209 ms | **19.0x** |

> Llama 3.2 1B, 1.5B parameters, 147 tensors, Apple M4

![Benchmark Results](benchmark_results.png)

## Why

Large-model training is routinely disrupted by checkpoint overhead. Meta's Llama 3 training postmortem (2024) reported 419 hardware failures over 54 days, with roughly **2.1% of total GPU-hours spent on checkpointing** — each full save blocking all GPUs for ~20 seconds. ByteDance's ByteCheckpoint (NSDI 2025) showed that decoupling the serialisation path and using incremental delta writes reduces this overhead by up to 90%.

This project implements the same idea at library level: delta detection + content-addressed compressed storage + async background writes.

## Architecture

```
Training loop
     │
     │  save(step, metrics)
     ▼
┌────────────────────────────────────────────────────┐
│               CheckpointManager                    │
│                                                    │
│  ┌─────────────┐    ┌──────────────────────────┐  │
│  │ DeltaEngine │    │      AsyncWriter          │  │
│  │             │    │  ┌────────────────────┐   │  │
│  │  C++ ext    │───▶│  │  background thread  │   │  │
│  │  (OpenMP)   │    │  │  bounded queue (2) │   │  │
│  │  or pure Py │    │  └────────┬───────────┘   │  │
│  └─────────────┘    └───────────┼───────────────┘  │
│                                 │                   │
│  ┌──────────────┐   ┌───────────▼───────────────┐  │
│  │LifecycleManager│  │  ContentAddressedStore    │  │
│  │ keep_last_n  │   │  SHA-256 + zstandard      │  │
│  │ keep_best_n  │   │  git-object layout        │  │
│  └──────────────┘   └───────────────────────────┘  │
│                                                    │
│              Manifest (JSON, atomic writes)        │
└────────────────────────────────────────────────────┘
```

- **Delta detection** computes relative L2 norms `‖current − base‖ / ‖base‖` across all layers. The C++ extension parallelises this with OpenMP.
- **Content-addressed storage** mirrors git's object store — blobs are keyed by SHA-256 of their compressed content, so identical tensors are stored once.
- **Async writes** run on a daemon thread with a bounded queue (max 2 pending). The training loop only blocks for delta computation and CPU-copy, not disk I/O.
- **Lifecycle GC** runs after every save, pruning versions outside the `keep_last_n` / `keep_best_n` windows and deleting orphaned blobs.

## Installation

```bash
# Python >= 3.10, PyTorch >= 2.1
pip install torch

# OpenMP (macOS only)
brew install libomp

# Install (builds C++ extension automatically)
cd pytorch-incremental-checkpoint
pip install -e ".[cli,bench,dev]"

# Verify
python -c "import delta_engine_cpp; print(delta_engine_cpp.omp_thread_count())"
```

## Usage

```python
from checkpoint_engine import CheckpointManager

manager = CheckpointManager(
    save_dir="./checkpoints",
    model=model,
    keep_last_n=5,
    keep_best_n=3,
    dirty_threshold=1e-4,
    async_write=True,
)

for step in range(1000):
    loss = train_step(model)
    if step % 100 == 0:
        manager.save(step, metrics={"loss": loss.item()})

manager.restore(step=500)
manager.close()
```

Or with a context manager:

```python
with CheckpointManager("./ckpts", model) as mgr:
    for step in range(1000):
        train_step(model)
        if step % 100 == 0:
            mgr.save(step, metrics={"loss": compute_loss()})
```

### CLI

```bash
ckpt list ./checkpoints                # list all checkpoints
ckpt info ./checkpoints 500            # detailed info for step 500
ckpt stats ./checkpoints               # storage statistics
ckpt gc ./checkpoints                  # garbage collection (dry run)
ckpt gc ./checkpoints --no-dry-run --keep-last 5 --keep-best 3
```

## Benchmarks

```bash
# Requires: transformers, matplotlib
python benchmarks/bench_delta.py --repeats 10
# Falls back to a synthetic model if Llama 3.2 1B is unavailable
```

Results on Apple M4, Llama 3.2 1B (bfloat16), 30% of tensors perturbed:

```
[Pure Python] (10 repeats)
  Run 1: 6480.5 ms    dirty=44/147    (29.9%)
  Run 2: 5777.4 ms    dirty=44/147    (29.9%)
  Run 3: 4261.5 ms    dirty=44/147    (29.9%)
  Run 4: 6744.4 ms    dirty=44/147    (29.9%)
  Run 5: 4138.9 ms    dirty=44/147    (29.9%)
  Run 6: 5969.7 ms    dirty=44/147    (29.9%)
  Run 7: 5132.3 ms    dirty=44/147    (29.9%)
  Run 8: 5515.8 ms    dirty=44/147    (29.9%)
  Run 9: 4612.0 ms    dirty=44/147    (29.9%)
  Run 10: 4412.1 ms    dirty=44/147    (29.9%)
  avg=5304.5ms    median=5324.1ms

[C++ / OpenMP] (10 repeats, 4 threads)
  Run 1: 647.0 ms    dirty=44/147    (29.9%)
  Run 2: 462.3 ms    dirty=44/147    (29.9%)
  Run 3: 220.5 ms    dirty=44/147    (29.9%)
  Run 4: 208.3 ms    dirty=44/147    (29.9%)
  Run 5: 207.7 ms    dirty=44/147    (29.9%)
  Run 6: 209.6 ms    dirty=44/147    (29.9%)
  Run 7: 206.6 ms    dirty=44/147    (29.9%)
  Run 8: 208.0 ms    dirty=44/147    (29.9%)
  Run 9: 209.3 ms    dirty=44/147    (29.9%)
  Run 10: 211.8 ms    dirty=44/147    (29.9%)
  avg=279.1ms    median=209.4ms

Speedup (C++ vs Python): 19.01x
```

## Tests

```bash
pytest tests/ -v   # 87 tests
```

## Project Structure

```
pytorch-incremental-checkpoint/
├── csrc/
│   ├── delta_engine.h/.cpp      OpenMP parallel dirty detection
│   ├── blob_prepare.h/.cpp      C++ parallel serialize + compress + hash
│   └── bindings.cpp             pybind11 module
├── checkpoint_engine/
│   ├── manager.py               CheckpointManager (top-level API)
│   ├── delta.py                 DeltaEngine (C++ / Python fallback)
│   ├── store.py                 ContentAddressedStore (SHA-256 + zstd)
│   ├── async_writer.py          Background thread writer
│   ├── manifest.py              Manifest (atomic JSON)
│   ├── lifecycle.py             Retention policy & GC
│   └── metrics.py               In-process metrics
├── cli/ckpt.py                  Click CLI
├── benchmarks/bench_delta.py    C++ vs Python benchmark
├── tests/                       87 pytest tests
├── setup.py                     C++ extension build
└── pyproject.toml
```

## References

- Meta AI, *"Llama 3 Herd of Models"*, arXiv 2407.21783 (2024) — Section 3.3: 419 hardware failures, 2.1% checkpoint overhead.
- Jiarui Fang et al., *"ByteCheckpoint: A Unified Checkpointing System for Large Foundation Model Development"*, NSDI 2025.
- PyTorch `torch.utils.cpp_extension` docs.
- [zstandard Python bindings](https://python-zstandard.readthedocs.io/)

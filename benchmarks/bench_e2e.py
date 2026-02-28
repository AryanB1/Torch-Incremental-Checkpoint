"""
bench_e2e.py — end-to-end blob preparation benchmark.

Compares three blob-writing strategies on dirty tensors extracted from a
Llama 3.2 1B state dict (or a synthetic fallback):

  1. Naive torch.save() of the full state dict
  2. Sequential store.put() (Python serialize + compress + hash)
  3. Batch store.put_batch() (C++ parallel serialize + compress + hash)

Usage
-----
    python benchmarks/bench_e2e.py
    python benchmarks/bench_e2e.py --repeats 5
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

# Ensure package root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from checkpoint_engine.delta import DeltaEngine, _CPP_AVAILABLE
from checkpoint_engine.store import ContentAddressedStore, _BATCH_CPP_AVAILABLE


# ---------------------------------------------------------------------------
# Model loading (reused from bench_delta)
# ---------------------------------------------------------------------------

def load_state_dict() -> dict[str, torch.Tensor]:
    """Load a large state dict for benchmarking."""
    try:
        from transformers import AutoModelForCausalLM
        print("Loading Llama 3.2 1B state dict...")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        sd = {k: v.cpu() for k, v in model.state_dict().items()}
        del model
        print(f"  {len(sd)} tensors, "
              f"{sum(v.numel() for v in sd.values()):,} parameters")
        return sd
    except Exception as exc:
        print(f"[warn] Could not load Llama: {exc}")
        print("[warn] Using synthetic 50M-param state dict.")
        return _synthetic_state_dict()


def _synthetic_state_dict(
    hidden: int = 2048, layers: int = 24
) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for i in range(layers):
        sd[f"layer.{i}.weight"] = torch.randn(hidden, hidden, dtype=torch.bfloat16)
        sd[f"layer.{i}.bias"]   = torch.randn(hidden,         dtype=torch.bfloat16)
    sd["head.weight"] = torch.randn(hidden, 32000, dtype=torch.bfloat16)
    n = sum(v.numel() for v in sd.values())
    print(f"  Synthetic: {len(sd)} tensors, {n:,} parameters")
    return sd


def make_perturbed(base: dict[str, torch.Tensor], fraction: float = 0.3,
                   noise: float = 0.01) -> dict[str, torch.Tensor]:
    """Return a copy with `fraction` of tensors randomly perturbed."""
    import random
    keys = list(base.keys())
    dirty_keys = set(random.sample(keys, int(len(keys) * fraction)))
    current = {}
    for k, v in base.items():
        if k in dirty_keys:
            current[k] = v + torch.randn_like(v) * noise
        else:
            current[k] = v.clone()
    return current


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def bench_naive_save(state_dict: dict[str, torch.Tensor], path: Path) -> float:
    """Full torch.save(). Returns wall-clock ms."""
    t0 = time.perf_counter()
    torch.save(state_dict, path)
    return (time.perf_counter() - t0) * 1000


def bench_sequential_put(store: ContentAddressedStore,
                         tensors: list[torch.Tensor]) -> float:
    """Sequential store.put() for each tensor. Returns wall-clock ms."""
    t0 = time.perf_counter()
    for t in tensors:
        store.put(t)
    return (time.perf_counter() - t0) * 1000


def bench_batch_put(store: ContentAddressedStore,
                    tensors: list[torch.Tensor]) -> float:
    """Batch store.put_batch(). Returns wall-clock ms."""
    t0 = time.perf_counter()
    store.put_batch(tensors)
    return (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(repeats: int = 3, threshold: float = 1e-4) -> None:
    print(f"\n{'='*60}")
    print("End-to-End Blob Preparation Benchmark")
    print(f"{'='*60}")
    print(f"C++ delta engine:      {_CPP_AVAILABLE}")
    print(f"C++ batch blob prep:   {_BATCH_CPP_AVAILABLE}")

    if _CPP_AVAILABLE:
        import delta_engine_cpp
        n_omp = delta_engine_cpp.omp_thread_count()
        print(f"OpenMP threads:        {n_omp}")
        has_blob = getattr(delta_engine_cpp, "HAS_BLOB_PREPARE", False)
        print(f"HAS_BLOB_PREPARE:      {has_blob}")

    base = load_state_dict()
    current = make_perturbed(base, fraction=0.3)

    # Extract dirty tensors using delta engine
    print("\nComputing dirty set...")
    engine = DeltaEngine(threshold=threshold, use_cpp=_CPP_AVAILABLE)
    result = engine.compute_dirty(current, base)
    dirty_tensors = list(result.deltas.values())
    total_bytes = sum(t.numel() * t.element_size() for t in dirty_tensors)
    print(f"  {result.num_dirty} dirty tensors, "
          f"{total_bytes / 1e6:.1f} MB (uncompressed f32)")

    # --- Benchmark 1: Naive torch.save() ---
    print(f"\n[1] Naive torch.save() ({repeats} repeats)")
    naive_dir = Path(tempfile.mkdtemp(prefix="bench_naive_"))
    naive_times: list[float] = []
    for i in range(repeats):
        path = naive_dir / f"ckpt_{i}.pt"
        t = bench_naive_save(current, path)
        naive_times.append(t)
        print(f"  Run {i+1}: {t:.1f} ms")
    print(f"  avg={statistics.mean(naive_times):.1f}ms  "
          f"median={statistics.median(naive_times):.1f}ms")

    # --- Benchmark 2: Sequential put() ---
    print(f"\n[2] Sequential store.put() ({repeats} repeats)")
    seq_times: list[float] = []
    for i in range(repeats):
        seq_dir = Path(tempfile.mkdtemp(prefix="bench_seq_"))
        store = ContentAddressedStore(seq_dir / "blobs")
        t = bench_sequential_put(store, dirty_tensors)
        seq_times.append(t)
        print(f"  Run {i+1}: {t:.1f} ms  ({len(dirty_tensors)} tensors)")
        shutil.rmtree(seq_dir, ignore_errors=True)
    print(f"  avg={statistics.mean(seq_times):.1f}ms  "
          f"median={statistics.median(seq_times):.1f}ms")

    # --- Benchmark 3: Batch put_batch() ---
    if _BATCH_CPP_AVAILABLE:
        print(f"\n[3] C++ batch store.put_batch() ({repeats} repeats)")
        batch_times: list[float] = []
        for i in range(repeats):
            batch_dir = Path(tempfile.mkdtemp(prefix="bench_batch_"))
            store = ContentAddressedStore(batch_dir / "blobs")
            t = bench_batch_put(store, dirty_tensors)
            batch_times.append(t)
            print(f"  Run {i+1}: {t:.1f} ms  ({len(dirty_tensors)} tensors)")
            shutil.rmtree(batch_dir, ignore_errors=True)
        print(f"  avg={statistics.mean(batch_times):.1f}ms  "
              f"median={statistics.median(batch_times):.1f}ms")
    else:
        print("\n[3] C++ batch blob preparation not available — skipped.")
        batch_times = seq_times  # fallback for summary

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    naive_avg  = statistics.mean(naive_times)
    seq_avg    = statistics.mean(seq_times)
    batch_avg  = statistics.mean(batch_times)

    print(f"Naive torch.save():         {naive_avg:>8.1f} ms")
    print(f"Sequential put() (dirty):   {seq_avg:>8.1f} ms")
    if _BATCH_CPP_AVAILABLE:
        print(f"C++ batch put_batch():      {batch_avg:>8.1f} ms")
        print(f"\nSpeedup (batch vs seq):     {seq_avg / max(batch_avg, 0.001):.2f}x")
        print(f"Speedup (batch vs naive):   {naive_avg / max(batch_avg, 0.001):.2f}x")
    print(f"Speedup (seq vs naive):     {naive_avg / max(seq_avg, 0.001):.2f}x")
    print(f"{'='*60}")

    # Cleanup
    shutil.rmtree(naive_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end blob preparation benchmark"
    )
    parser.add_argument("--repeats",   type=int,   default=3,
                        help="Number of timed repetitions (default: 3)")
    parser.add_argument("--threshold", type=float, default=1e-4,
                        help="Dirty detection threshold (default: 1e-4)")
    args = parser.parse_args()

    run_benchmark(repeats=args.repeats, threshold=args.threshold)

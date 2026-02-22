"""
bench_delta.py — microbenchmark: C++ vs pure-Python delta engine.

Measures the time to compute the dirty set on a Llama 3.2 1B state dict
(or a large synthetic proxy if Llama is unavailable).

Usage
-----
    python benchmarks/bench_delta.py
    python benchmarks/bench_delta.py --repeats 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure package root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn

from checkpoint_engine.delta import DeltaEngine, _CPP_AVAILABLE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_state_dict(device: torch.device) -> dict[str, torch.Tensor]:
    """
    Returns a large state dict for benchmarking.
    Tries Llama 3.2 1B first, falls back to a synthetic 1B-parameter model.
    """
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
    """Generate a large state dict without downloading any model."""
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


def timed(fn, *args, **kwargs) -> tuple[float, object]:
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return time.perf_counter() - t0, result


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(repeats: int = 5, threshold: float = 1e-4) -> None:
    print(f"\n{'='*60}")
    print("Delta Engine Microbenchmark")
    print(f"{'='*60}")
    print(f"C++ extension available: {_CPP_AVAILABLE}")

    device = get_device()
    base = load_state_dict(device)
    current = make_perturbed(base, fraction=0.3)

    # Pre-warm
    py_engine  = DeltaEngine(threshold=threshold, use_cpp=False)
    py_engine.compute_dirty(current, base)

    # --- Pure Python ---
    print(f"\n[Pure Python] ({repeats} repeats)")
    py_times: list[float] = []
    for i in range(repeats):
        t, result = timed(py_engine.compute_dirty, current, base)
        py_times.append(t * 1000)
        print(f"  Run {i+1}: {t*1000:.1f} ms  "
              f"  dirty={result.num_dirty}/{len(base)}  "
              f"  ({result.dirty_ratio:.1%})")

    import statistics
    print(f"  avg={statistics.mean(py_times):.1f}ms  "
          f"  median={statistics.median(py_times):.1f}ms")

    # --- C++ ---
    if _CPP_AVAILABLE:
        import delta_engine_cpp
        n_omp = delta_engine_cpp.omp_thread_count()
        print(f"\n[C++ / OpenMP] ({repeats} repeats, {n_omp} threads)")
        cpp_engine = DeltaEngine(threshold=threshold, use_cpp=True)
        # Pre-warm
        cpp_engine.compute_dirty(current, base)

        cpp_times: list[float] = []
        for i in range(repeats):
            t, result = timed(cpp_engine.compute_dirty, current, base)
            cpp_times.append(t * 1000)
            print(f"  Run {i+1}: {t*1000:.1f} ms  "
                  f"  dirty={result.num_dirty}/{len(base)}  "
                  f"  ({result.dirty_ratio:.1%})")

        print(f"  avg={statistics.mean(cpp_times):.1f}ms  "
              f"  median={statistics.median(cpp_times):.1f}ms")

        speedup = statistics.mean(py_times) / max(statistics.mean(cpp_times), 0.001)
        print(f"\nSpeedup (C++ vs Python): {speedup:.2f}x")
    else:
        print("\n[C++] extension not built — run `pip install -e .` to enable.")

    print(f"\n{'='*60}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Microbenchmark: C++ vs pure-Python delta engine"
    )
    parser.add_argument("--repeats",   type=int,   default=5,
                        help="Number of timed repetitions per backend (default: 5)")
    parser.add_argument("--threshold", type=float, default=1e-4,
                        help="Dirty detection threshold (default: 1e-4)")
    args = parser.parse_args()

    run_benchmark(repeats=args.repeats, threshold=args.threshold)

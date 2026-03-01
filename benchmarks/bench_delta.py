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
import statistics
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


def _plot_results(
    py_times: list[float],
    cpp_times: list[float],
    speedup: float,
    output_png: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Delta Engine — C++ (OpenMP) vs Pure Python", fontsize=14)

    # 1. Per-run timing comparison
    ax = axes[0]
    runs = list(range(1, len(py_times) + 1))
    ax.plot(runs, py_times, "o-", label="Pure Python", color="tomato")
    ax.plot(runs, cpp_times, "s-", label="C++ / OpenMP", color="steelblue")
    ax.set_xlabel("Run")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Per-Run Dirty Detection Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(top=max(py_times) * 1.15)

    # 2. Average comparison bar chart
    ax = axes[1]
    py_avg = statistics.mean(py_times)
    cpp_avg = statistics.mean(cpp_times)
    categories = ["Pure Python", "C++ / OpenMP"]
    avgs = [py_avg, cpp_avg]
    colors = ["tomato", "steelblue"]
    bars = ax.bar(categories, avgs, color=colors, alpha=0.85)
    ax.set_ylabel("Avg Time (ms)")
    ax.set_title(f"Average Dirty Detection Time ({speedup:.1f}x speedup)")
    for bar, avg in zip(bars, avgs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(avgs) * 0.02,
            f"{avg:.1f} ms",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax.set_ylim(top=max(avgs) * 1.15)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_benchmark(repeats: int = 5, threshold: float = 1e-4, output_png: str = "benchmark_results.png") -> None:
    print(f"\n{'='*60}")
    print("Delta Engine Microbenchmark")
    print(f"{'='*60}")
    print(f"C++ extension available: {_CPP_AVAILABLE}")

    device = get_device()
    base = load_state_dict(device)
    current = make_perturbed(base, fraction=0.3)

    py_engine = DeltaEngine(threshold=threshold, use_cpp=False)
    py_engine.compute_dirty(current, base)  # pre-warm

    print(f"\n[Pure Python] ({repeats} repeats)")
    py_times: list[float] = []
    for i in range(repeats):
        t, result = timed(py_engine.compute_dirty, current, base)
        py_times.append(t * 1000)
        print(f"  Run {i+1}: {t*1000:.1f} ms  "
              f"  dirty={result.num_dirty}/{len(base)}  "
              f"  ({result.dirty_ratio:.1%})")

    print(f"  avg={statistics.mean(py_times):.1f}ms  "
          f"  median={statistics.median(py_times):.1f}ms")

    if _CPP_AVAILABLE:
        import delta_engine_cpp
        n_omp = delta_engine_cpp.omp_thread_count()
        print(f"\n[C++ / OpenMP] ({repeats} repeats, {n_omp} threads)")
        cpp_engine = DeltaEngine(threshold=threshold, use_cpp=True)
        cpp_engine.compute_dirty(current, base)  # pre-warm

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

        _plot_results(py_times, cpp_times, speedup, output_png)
        print(f"Chart saved to: {output_png}")
    else:
        print("\n[C++] extension not built — run `pip install -e .` to enable.")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Microbenchmark: C++ vs pure-Python delta engine"
    )
    parser.add_argument("--repeats",   type=int,   default=5,
                        help="Number of timed repetitions per backend (default: 5)")
    parser.add_argument("--threshold", type=float, default=1e-4,
                        help="Dirty detection threshold (default: 1e-4)")
    parser.add_argument("--output",    type=str,   default="benchmark_results.png",
                        help="Output path for the matplotlib chart (default: benchmark_results.png)")
    args = parser.parse_args()

    run_benchmark(repeats=args.repeats, threshold=args.threshold, output_png=args.output)

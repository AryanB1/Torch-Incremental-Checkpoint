"""
bench_synthetic.py — end-to-end benchmark comparing naive torch.save()
vs. CheckpointManager on a Llama 3.2 1B model with synthetic weight updates.

Usage
-----
    python benchmarks/bench_synthetic.py [--steps 1000] [--ckpt-every 100]
    python benchmarks/bench_synthetic.py --help

Outputs
-------
* Console table with per-checkpoint timing and dirty ratios
* ./benchmark_results.png — matplotlib comparison chart
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Ensure package root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")   # headless backend
import matplotlib.pyplot as plt

from checkpoint_engine import CheckpointManager

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_llama_1b(device: torch.device) -> nn.Module:
    """
    Load Llama 3.2 1B in bfloat16.
    Falls back to a small synthetic model if transformers is not installed
    or the model is not cached locally.
    """
    try:
        from transformers import AutoModelForCausalLM
        print("Loading Llama 3.2 1B from HuggingFace (bfloat16)...")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        model = model.to(device)
        print(f"  Loaded {sum(p.numel() for p in model.parameters()):,} parameters "
              f"on {device}")
        return model
    except Exception as exc:
        print(f"[warn] Could not load Llama 3.2 1B: {exc}")
        print("[warn] Falling back to synthetic 50M-parameter model for benchmarking.")
        return _make_synthetic_model(device)


def _make_synthetic_model(device: torch.device, hidden: int = 2048, layers: int = 12) -> nn.Module:
    """50M-parameter MLP proxy for quick local testing."""
    class SyntheticModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
                for _ in range(layers)
            ])
            self.head = nn.Linear(hidden, hidden)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return self.head(x)

    model = SyntheticModel().to(torch.bfloat16).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"  Synthetic model: {n:,} parameters on {device}")
    return model


# ---------------------------------------------------------------------------
# Simulate training step
# ---------------------------------------------------------------------------

@torch.no_grad()
def simulate_weight_update(model: nn.Module, noise_scale: float = 0.001) -> None:
    """Add small Gaussian noise to each parameter — simulates a gradient step."""
    for param in model.parameters():
        param.add_(torch.randn_like(param) * noise_scale)


# ---------------------------------------------------------------------------
# Naive baseline: torch.save() full state dict
# ---------------------------------------------------------------------------

def naive_save(model: nn.Module, path: Path) -> float:
    """Save full state dict. Returns wall-clock seconds blocked."""
    t0 = time.perf_counter()
    torch.save(model.state_dict(), path)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    steps: int = 1000,
    ckpt_every: int = 100,
    noise_scale: float = 0.001,
    output_png: str = "benchmark_results.png",
) -> None:
    device = get_device()
    print(f"\nDevice: {device}")
    model = load_llama_1b(device)

    naive_dir = Path(tempfile.mkdtemp(prefix="ckpt_naive_"))
    incr_dir  = Path(tempfile.mkdtemp(prefix="ckpt_incr_"))

    # Per-checkpoint measurements
    naive_times:   list[float] = []
    incr_times:    list[float] = []
    dirty_ratios:  list[float] = []
    steps_logged:  list[int]   = []

    print(f"\nRunning {steps} steps, checkpointing every {ckpt_every} steps...\n")
    print(f"{'Step':>6}  {'NaiveTime':>12}  {'IncrTime':>12}  {'DirtyRatio':>12}")
    print("-" * 50)

    with CheckpointManager(
        save_dir=str(incr_dir),
        model=model,
        keep_last_n=5,
        keep_best_n=3,
        dirty_threshold=1e-4,
        async_write=True,
    ) as mgr:
        for step in range(1, steps + 1):
            simulate_weight_update(model, noise_scale=noise_scale)

            if step % ckpt_every == 0:
                # --- Naive save ---
                naive_path = naive_dir / f"step_{step:06d}.pt"
                naive_t = naive_save(model, naive_path)

                # --- Incremental save ---
                t0 = time.perf_counter()
                fut = mgr.save(step, metrics={"loss": 1.0 / step, "step": step})
                incr_blocking_t = time.perf_counter() - t0

                naive_times.append(naive_t * 1000)       # ms
                incr_times.append(incr_blocking_t * 1000)
                dirty_ratio = mgr.metrics.records[-1].dirty_ratio if mgr.metrics.records else 1.0
                dirty_ratios.append(dirty_ratio)
                steps_logged.append(step)

                print(
                    f"{step:>6}  {naive_t*1000:>11.1f}ms  "
                    f"{incr_blocking_t*1000:>11.1f}ms  "
                    f"{dirty_ratio:>11.1%}"
                )

        # Flush async writes before measuring storage
        mgr.wait_all()

    # --- Storage comparison ---
    naive_bytes = sum(
        f.stat().st_size for f in naive_dir.glob("*.pt") if f.is_file()
    )
    incr_bytes = sum(
        f.stat().st_size for f in incr_dir.rglob("*") if f.is_file()
    )

    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"Naive  total storage : {naive_bytes / 1e6:>10.1f} MB")
    print(f"Incr   total storage : {incr_bytes  / 1e6:>10.1f} MB")
    if naive_bytes > 0:
        print(f"Storage reduction    : {(1 - incr_bytes/naive_bytes):>10.1%}")
    import statistics
    if naive_times:
        print(f"Naive  avg block time: {statistics.mean(naive_times):>10.1f} ms")
        print(f"Incr   avg block time: {statistics.mean(incr_times):>10.1f} ms")
        print(f"Speedup (blocking)   : {statistics.mean(naive_times)/max(statistics.mean(incr_times),0.001):>10.1f}x")

    # --- Plot ---
    _plot_results(
        steps=steps_logged,
        naive_times=naive_times,
        incr_times=incr_times,
        dirty_ratios=dirty_ratios,
        naive_bytes=naive_bytes,
        incr_bytes=incr_bytes,
        output_png=output_png,
    )
    print(f"\nChart saved to: {output_png}")

    # Cleanup
    shutil.rmtree(naive_dir, ignore_errors=True)
    shutil.rmtree(incr_dir,  ignore_errors=True)


def _plot_results(
    steps: list[int],
    naive_times: list[float],
    incr_times: list[float],
    dirty_ratios: list[float],
    naive_bytes: int,
    incr_bytes: int,
    output_png: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Incremental Checkpoint Engine — Benchmark Results", fontsize=14)

    # 1. Blocking time per checkpoint
    ax = axes[0]
    ax.plot(steps, naive_times, "o-", label="torch.save (naive)", color="tomato")
    ax.plot(steps, incr_times,  "s-", label="CheckpointManager", color="steelblue")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Blocking Time (ms)")
    ax.set_title("Blocking Time per Checkpoint")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Dirty ratio over time
    ax = axes[1]
    ax.bar(steps, [r * 100 for r in dirty_ratios], color="steelblue", alpha=0.8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Dirty Parameters (%)")
    ax.set_title("Dirty Ratio per Checkpoint")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Cumulative storage
    ax = axes[2]
    categories = ["torch.save\n(naive)", "CheckpointManager\n(incremental)"]
    sizes_mb = [naive_bytes / 1e6, incr_bytes / 1e6]
    colors = ["tomato", "steelblue"]
    bars = ax.bar(categories, sizes_mb, color=colors, alpha=0.85)
    ax.set_ylabel("Total Storage (MB)")
    ax.set_title("Total Storage Used")
    for bar, size in zip(bars, sizes_mb):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(sizes_mb) * 0.01,
            f"{size:.0f} MB",
            ha="center", va="bottom", fontsize=10
        )
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark incremental vs naive checkpointing on Llama 3.2 1B"
    )
    parser.add_argument("--steps",     type=int,   default=1000,
                        help="Total simulated training steps (default: 1000)")
    parser.add_argument("--ckpt-every", type=int,  default=100,
                        help="Checkpoint interval in steps (default: 100)")
    parser.add_argument("--noise-scale", type=float, default=0.001,
                        help="Gaussian noise scale for weight updates (default: 0.001)")
    parser.add_argument("--output",    type=str,   default="benchmark_results.png",
                        help="Output path for the matplotlib chart")
    args = parser.parse_args()

    run_benchmark(
        steps=args.steps,
        ckpt_every=args.ckpt_every,
        noise_scale=args.noise_scale,
        output_png=args.output,
    )

"""
setup.py — builds the C++ delta_engine_cpp extension.

Apple Silicon (M-series) note
------------------------------
Homebrew installs libomp to /opt/homebrew/opt/libomp (arm64) or
/usr/local/opt/libomp (x86_64).  We auto-detect the correct path.

To build:
    pip install -e .

To build in-place (for development without installing):
    python setup.py build_ext --inplace
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup, find_packages

# Import torch AFTER setuptools so we can call CppExtension
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension


# ---------------------------------------------------------------------------
# Detect OpenMP on Apple Silicon / Intel Mac / Linux
# ---------------------------------------------------------------------------

def detect_openmp() -> tuple[list[str], list[str], list[str]]:
    """
    Returns (include_dirs, library_dirs, extra_compile_args) for OpenMP.
    """
    platform = sys.platform

    if platform == "darwin":
        # Try Homebrew libomp (arm64 first, then x86)
        brew_paths = [
            "/opt/homebrew/opt/libomp",   # Apple Silicon
            "/usr/local/opt/libomp",       # Intel Mac
        ]
        for brew_path in brew_paths:
            if Path(brew_path).exists():
                inc = f"{brew_path}/include"
                lib = f"{brew_path}/lib"
                print(f"[setup] Found Homebrew libomp at {brew_path}")
                return (
                    [inc],
                    [lib],
                    ["-Xpreprocessor", "-fopenmp", "-O3"],
                )
        # libomp not found — build without OpenMP (still works, just serial)
        print("[setup] WARNING: libomp not found. Building without OpenMP.")
        print("[setup]          Install with: brew install libomp")
        return ([], [], ["-O3"])

    elif platform.startswith("linux"):
        return ([], [], ["-fopenmp", "-O3"])

    else:
        # Windows / unknown — conservative fallback
        return ([], [], ["/O2"])


def detect_openmp_link_args() -> list[str]:
    """Extra linker args for OpenMP on the current platform."""
    if sys.platform == "darwin":
        brew_paths = [
            "/opt/homebrew/opt/libomp",
            "/usr/local/opt/libomp",
        ]
        for brew_path in brew_paths:
            lib = f"{brew_path}/lib"
            if Path(brew_path).exists():
                return [f"-L{lib}", "-lomp"]
        return []
    elif sys.platform.startswith("linux"):
        return ["-fopenmp"]
    return []


# ---------------------------------------------------------------------------
# Extension definition
# ---------------------------------------------------------------------------

include_dirs, library_dirs, extra_compile_args = detect_openmp()
extra_link_args = detect_openmp_link_args()

# PyTorch include dirs are added automatically by CppExtension; we add them
# explicitly here for clarity / IDE support.
torch_inc = torch.utils.cpp_extension.include_paths()

delta_ext = CppExtension(
    name="delta_engine_cpp",
    sources=[
        "csrc/delta_engine.cpp",
        "csrc/bindings.cpp",
    ],
    include_dirs=["csrc"] + include_dirs + torch_inc,
    library_dirs=library_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)


# ---------------------------------------------------------------------------
# setup()
# ---------------------------------------------------------------------------

setup(
    name="pytorch-incremental-checkpoint",
    version="0.1.0",
    author="Aryan",
    description=(
        "Incremental checkpoint engine for PyTorch: "
        "delta-based saves with async writes and content-addressed storage."
    ),
    long_description=Path("README.md").read_text(encoding="utf-8")
        if Path("README.md").exists() else "",
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "benchmarks*", "csrc*"]),
    ext_modules=[delta_ext],
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch>=2.1",
        "zstandard>=0.21",
        "pybind11>=2.11",
    ],
    extras_require={
        "cli": ["click>=8.1", "rich>=13.0"],
        "bench": ["transformers>=4.40", "matplotlib>=3.8"],
        "dev":  ["pytest>=7.4"],
    },
    entry_points={
        "console_scripts": [
            "ckpt=cli.ckpt:cli",
        ],
    },
)

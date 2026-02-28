#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <string>
#include <tuple>

namespace py = pybind11;

// Each PreparedBlob holds the SHA-256 hex digest and the compressed bytes.
struct PreparedBlob {
    std::string sha256_hex;   // 64-char lowercase hex digest
    std::string compressed;   // zstd-compressed torch pickle bytes
};

// Prepare blobs in parallel using OpenMP.
// Input: vector of CPU contiguous tensors
// Output: vector of PreparedBlob (same order as input)
std::vector<PreparedBlob> batch_prepare_blobs(
    const std::vector<torch::Tensor>& tensors,
    int zstd_level = 3
);

// Pybind-friendly wrapper returning list[tuple[str, bytes]]
std::vector<std::tuple<std::string, py::bytes>> batch_prepare_blobs_py(
    const std::vector<torch::Tensor>& tensors,
    int zstd_level = 3
);

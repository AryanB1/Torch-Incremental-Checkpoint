#pragma once

#include <torch/extension.h>
#include <vector>
#include <string>
#include <tuple>

// Result type returned by compute_dirty_tensors
struct DirtyResult {
    std::vector<std::string> dirty_names;
    std::vector<torch::Tensor> deltas;
    std::vector<double> norms;
};

// Compute dirty tensors between current and base state dicts.
// Returns names, delta tensors, and per-tensor relative L2 norms for
// parameters whose relative norm exceeds the given threshold.
DirtyResult compute_dirty_tensors(
    const std::vector<std::string>& names,
    const std::vector<torch::Tensor>& current,
    const std::vector<torch::Tensor>& base,
    double threshold
);

// Pybind-friendly variant returning a tuple
std::tuple<
    std::vector<std::string>,
    std::vector<torch::Tensor>,
    std::vector<double>
>
compute_dirty_tensors_py(
    const std::vector<std::string>& names,
    const std::vector<torch::Tensor>& current,
    const std::vector<torch::Tensor>& base,
    double threshold
);

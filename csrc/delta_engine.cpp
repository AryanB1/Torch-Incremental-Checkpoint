#include "delta_engine.h"

#include <torch/extension.h>
#include <ATen/Parallel.h>

#include <vector>
#include <string>
#include <tuple>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Compute relative L2 norm: ||current - base|| / max(||base||, eps)
// Uses float32 arithmetic regardless of input dtype to avoid overflow with
// bfloat16 and to keep the hot path fast on CPU.
static double relative_l2(const torch::Tensor& current,
                           const torch::Tensor& base) {
    // Promote to float32 on CPU for stable arithmetic
    auto c = current.detach().to(torch::kFloat32).cpu();
    auto b = base.detach().to(torch::kFloat32).cpu();

    auto diff   = c - b;
    double diff_norm = diff.norm().item<double>();
    double base_norm = b.norm().item<double>();

    constexpr double eps = 1e-8;
    return diff_norm / (base_norm + eps);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

DirtyResult compute_dirty_tensors(
    const std::vector<std::string>& names,
    const std::vector<torch::Tensor>& current,
    const std::vector<torch::Tensor>& base,
    double threshold)
{
    if (names.size() != current.size() || names.size() != base.size()) {
        throw std::invalid_argument(
            "names, current, and base must have the same length");
    }

    const int n = static_cast<int>(names.size());

    // Per-tensor relative norms (computed in parallel)
    std::vector<double> norms(n, 0.0);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int i = 0; i < n; ++i) {
        norms[i] = relative_l2(current[i], base[i]);
    }

    // Collect dirty tensors (serial — avoids vector reallocation races)
    DirtyResult result;
    for (int i = 0; i < n; ++i) {
        if (norms[i] > threshold) {
            result.dirty_names.push_back(names[i]);
            // Store delta as float32 CPU tensor for storage efficiency
            auto delta = (current[i].to(torch::kFloat32).cpu() -
                          base[i].to(torch::kFloat32).cpu());
            result.deltas.push_back(delta.contiguous());
            result.norms.push_back(norms[i]);
        }
    }

    return result;
}

std::tuple<
    std::vector<std::string>,
    std::vector<torch::Tensor>,
    std::vector<double>
>
compute_dirty_tensors_py(
    const std::vector<std::string>& names,
    const std::vector<torch::Tensor>& current,
    const std::vector<torch::Tensor>& base,
    double threshold)
{
    auto result = compute_dirty_tensors(names, current, base, threshold);
    return {result.dirty_names, result.deltas, result.norms};
}

#include "delta_engine.h"

#include <torch/extension.h>
#include <ATen/Parallel.h>

#include <vector>
#include <string>
#include <tuple>
#include <cstring>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Fast byte-level equality check.  Returns true when the raw storage of
// `a` and `b` is bit-identical (same dtype, same shape, same bytes).
// This lets us skip the expensive f32-conversion + norm path for the ~70 %
// of tensors that are completely unchanged.
static bool tensors_bitwise_equal(const torch::Tensor& a,
                                   const torch::Tensor& b) {
    if (a.dtype() != b.dtype())         return false;
    if (a.sizes() != b.sizes())         return false;

    auto a_c = a.contiguous();
    auto b_c = b.contiguous();

    const auto nbytes = static_cast<size_t>(a_c.nbytes());
    return std::memcmp(a_c.data_ptr(), b_c.data_ptr(), nbytes) == 0;
}

// Compute relative L2 norm: ||current - base|| / max(||base||, eps)
// Uses float32 arithmetic regardless of input dtype to avoid overflow with
// bfloat16 and to keep the hot path fast on CPU.
static double relative_l2(const torch::Tensor& current,
                           const torch::Tensor& base) {
    // Promote to float32 on CPU for stable arithmetic
    auto c_f32 = current.detach().to(torch::kFloat32).cpu();
    auto b_f32 = base.detach().to(torch::kFloat32).cpu();

    auto diff   = c_f32 - b_f32;
    double diff_norm = diff.norm().item<double>();
    double base_norm = b_f32.norm().item<double>();

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

    // Phase 1: fast bitwise check to classify each tensor as definitely-clean
    // (bitwise identical) or possibly-dirty (needs norm computation).
    // memcmp is a pure memory scan with no allocation — much cheaper than
    // converting to f32 and computing norms.
    std::vector<bool> needs_norm(n, false);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int i = 0; i < n; ++i) {
        needs_norm[i] = !tensors_bitwise_equal(current[i], base[i]);
    }

    // Collect indices of tensors that need norm computation.
    std::vector<int> changed_idx;
    changed_idx.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (needs_norm[i]) changed_idx.push_back(i);
    }

    const int m = static_cast<int>(changed_idx.size());

    // Phase 2: fused norm + delta for changed tensors only (serial).
    // Let PyTorch use its own internal threading for per-tensor ops — these
    // are memory-bandwidth bound and ATen's intra-op parallelism is optimal.
    // Convert to f32 once and compute both norm and delta in the same pass.
    DirtyResult result;
    for (int j = 0; j < m; ++j) {
        int i = changed_idx[j];
        auto c_f32 = current[i].detach().to(torch::kFloat32).cpu();
        auto b_f32 = base[i].detach().to(torch::kFloat32).cpu();

        auto diff = c_f32 - b_f32;
        double diff_norm = diff.norm().item<double>();
        double base_norm = b_f32.norm().item<double>();

        constexpr double eps = 1e-8;
        double norm = diff_norm / (base_norm + eps);

        if (norm > threshold) {
            result.dirty_names.push_back(names[i]);
            result.deltas.push_back(diff.contiguous());
            result.norms.push_back(norm);
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

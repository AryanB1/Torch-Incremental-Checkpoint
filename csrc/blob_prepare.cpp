#include "blob_prepare.h"

#include <torch/extension.h>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <stdexcept>

#include <ATen/Parallel.h>
#include <zstd.h>
#include <openssl/sha.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Internal helpers (all thread-safe — no shared mutable state)
// ---------------------------------------------------------------------------

// SHA-256 of a byte buffer → 64-char lowercase hex string
static std::string sha256_hex(const std::string& data) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(data.data()),
           data.size(), hash);

    std::ostringstream ss;
    ss << std::hex << std::setfill('0');
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        ss << std::setw(2) << static_cast<int>(hash[i]);
    }
    return ss.str();
}

// Serialize a tensor to bytes using torch pickle (same format as torch.save)
static std::string pickle_tensor(const torch::Tensor& tensor) {
    auto data = torch::pickle_save(tensor);
    return std::string(data.begin(), data.end());
}

// Compress bytes with zstd at the given level
static std::string zstd_compress(const std::string& input, int level) {
    size_t bound = ZSTD_compressBound(input.size());
    std::string output(bound, '\0');

    size_t compressed_size = ZSTD_compress(
        output.data(), bound,
        input.data(), input.size(),
        level
    );

    if (ZSTD_isError(compressed_size)) {
        throw std::runtime_error(
            std::string("ZSTD compression failed: ") +
            ZSTD_getErrorName(compressed_size)
        );
    }

    output.resize(compressed_size);
    return output;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::vector<PreparedBlob> batch_prepare_blobs(
    const std::vector<torch::Tensor>& tensors,
    int zstd_level)
{
    const int n = static_cast<int>(tensors.size());
    std::vector<PreparedBlob> results(n);

    // Validate inputs
    for (int i = 0; i < n; ++i) {
        TORCH_CHECK(tensors[i].device().is_cpu(),
                    "batch_prepare_blobs: tensor ", i, " must be on CPU");
    }

    // Disable ATen's internal parallelism inside our OpenMP region to avoid
    // thread oversubscription.
    int old_threads = at::get_num_threads();
    at::set_num_threads(1);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int i = 0; i < n; ++i) {
        // Ensure contiguous before serialization
        auto t = tensors[i].contiguous();

        // Step 1: serialize via torch pickle
        std::string serialized = pickle_tensor(t);

        // Step 2: compress with zstd
        std::string compressed = zstd_compress(serialized, zstd_level);

        // Step 3: SHA-256 of the compressed bytes
        std::string hex = sha256_hex(compressed);

        results[i].sha256_hex = std::move(hex);
        results[i].compressed = std::move(compressed);
    }

    at::set_num_threads(old_threads);
    return results;
}

std::vector<std::tuple<std::string, py::bytes>> batch_prepare_blobs_py(
    const std::vector<torch::Tensor>& tensors,
    int zstd_level)
{
    // Release GIL during the parallel computation
    std::vector<PreparedBlob> blobs;
    {
        py::gil_scoped_release release;
        blobs = batch_prepare_blobs(tensors, zstd_level);
    }

    // Convert to Python types (requires GIL for py::bytes construction)
    std::vector<std::tuple<std::string, py::bytes>> out;
    out.reserve(blobs.size());
    for (auto& blob : blobs) {
        out.emplace_back(
            std::move(blob.sha256_hex),
            py::bytes(blob.compressed)
        );
    }
    return out;
}

#include "delta_engine.h"
#include <torch/extension.h>
#include <pybind11/stl.h>   // for automatic std::vector / std::string conversion

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Incremental Checkpoint Engine — C++ delta computation via OpenMP";

    m.def(
        "compute_dirty_tensors",
        &compute_dirty_tensors_py,
        R"doc(
Compute which tensors changed above a relative-L2 threshold.

Args:
    names     : list[str]    — parameter names, same order as current/base
    current   : list[Tensor] — current model parameters
    base      : list[Tensor] — baseline (last-checkpointed) parameters
    threshold : float        — minimum relative L2 norm to mark dirty

Returns:
    Tuple of (dirty_names, delta_tensors, per_tensor_norms)
    where delta_tensors[i] = current[i] - base[i] cast to float32 on CPU.
)doc",
        py::arg("names"),
        py::arg("current"),
        py::arg("base"),
        py::arg("threshold") = 1e-4
    );

    // Expose the number of OpenMP threads being used (useful for debugging)
    m.def("omp_thread_count", []() -> int {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }, "Return the number of OpenMP threads available (1 if built without OpenMP).");
}

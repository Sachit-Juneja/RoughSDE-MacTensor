#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include "core/matrix.h"

namespace py = pybind11;

// Forward declarations
Matrix generate_fbm(float H, size_t N, float T);
Matrix lead_lag_transform(const Matrix& X);
Matrix compute_signature(const Matrix& path, size_t depth);
Matrix euler_maruyama_path(
    const Matrix& X0,
    float T,
    const Matrix& W_increments,
    const std::function<Matrix(float, const Matrix&)>& drift,
    const std::function<Matrix(float, const Matrix&)>& diffusion
);

// Numpy <-> MacTensor Converters
Matrix numpy_to_matrix(py::array_t<float> input) {
    py::buffer_info buf = input.request();
    if (buf.ndim != 1 && buf.ndim != 2) {
        throw std::runtime_error("Number of dimensions must be 1 or 2");
    }
    
    size_t rows = buf.shape[0];
    size_t cols = (buf.ndim == 2) ? buf.shape[1] : 1;
    
    Matrix m(rows, cols);
    float* ptr = static_cast<float*>(buf.ptr);
    
    // Numpy is typically row-major (C-contiguous)
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            m(i, j) = ptr[i * cols + j];
        }
    }
    return m;
}

py::array_t<float> matrix_to_numpy(const Matrix& m) {
    auto result = py::array_t<float>({m.rows, m.cols});
    py::buffer_info buf = result.request();
    float* ptr = static_cast<float*>(buf.ptr);
    
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            ptr[i * m.cols + j] = m(i, j);
        }
    }
    return result;
}

// Wrapper for euler_maruyama
py::array_t<float> euler_maruyama_wrapper(
    py::array_t<float> py_X0, 
    float T, 
    py::array_t<float> py_W_increments,
    py::function py_drift, 
    py::function py_diffusion
) {
    Matrix X0 = numpy_to_matrix(py_X0);
    Matrix W_inc = numpy_to_matrix(py_W_increments);
    
    auto drift = [&py_drift](float t, const Matrix& X) -> Matrix {
        py::array_t<float> py_X = matrix_to_numpy(X);
        py::object py_res = py_drift(t, py_X);
        return numpy_to_matrix(py::cast<py::array_t<float>>(py_res));
    };
    
    auto diffusion = [&py_diffusion](float t, const Matrix& X) -> Matrix {
        py::array_t<float> py_X = matrix_to_numpy(X);
        py::object py_res = py_diffusion(t, py_X);
        return numpy_to_matrix(py::cast<py::array_t<float>>(py_res));
    };
    
    Matrix path = euler_maruyama_path(X0, T, W_inc, drift, diffusion);
    return matrix_to_numpy(path);
}

// Module Definition
PYBIND11_MODULE(rough_sde, m) {
    m.doc() = "Rough Neural SDE plugin using MacTensor";

    m.def("generate_fbm", [](float H, size_t N, float T = 1.0f) {
        return matrix_to_numpy(generate_fbm(H, N, T));
    }, py::arg("H"), py::arg("N"), py::arg("T") = 1.0f, "Generate Fractional Brownian Motion");

    m.def("lead_lag_transform", [](py::array_t<float> py_X) {
        return matrix_to_numpy(lead_lag_transform(numpy_to_matrix(py_X)));
    }, py::arg("X"), "Apply Lead-Lag transformation to a 1D time series");

    m.def("compute_signature", [](py::array_t<float> py_path, size_t depth) {
        return matrix_to_numpy(compute_signature(numpy_to_matrix(py_path), depth));
    }, py::arg("path"), py::arg("depth"), "Compute truncated signature of a discrete path");

    m.def("euler_maruyama_path", &euler_maruyama_wrapper, 
          py::arg("X0"), py::arg("T"), py::arg("W_increments"), py::arg("drift"), py::arg("diffusion"),
          "Simulate an SDE path using Euler-Maruyama");
}

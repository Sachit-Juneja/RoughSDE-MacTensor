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

std::tuple<Matrix, Matrix, Matrix> euler_maruyama_adjoint_path(
    const Matrix& Z_seq,
    const Matrix& grad_output_seq,
    float T,
    const Matrix& W_increments,
    const std::function<std::pair<Matrix, Matrix>(float, const Matrix&, const Matrix&, float)>& vjp_drift,
    const std::function<std::pair<Matrix, Matrix>(float, const Matrix&, const Matrix&, const Matrix&)>& vjp_diffusion
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

// Wrapper for euler_maruyama adjoint
py::tuple euler_maruyama_adjoint_wrapper(
    py::array_t<float> py_Z_seq,
    py::array_t<float> py_grad_output_seq,
    float T,
    py::array_t<float> py_W_increments,
    py::function py_vjp_drift,
    py::function py_vjp_diffusion
) {
    Matrix Z_seq = numpy_to_matrix(py_Z_seq);
    Matrix grad_output_seq = numpy_to_matrix(py_grad_output_seq);
    Matrix W_inc = numpy_to_matrix(py_W_increments);
    
    auto vjp_drift = [&py_vjp_drift](float t, const Matrix& Z, const Matrix& a, float dt) -> std::pair<Matrix, Matrix> {
        py::tuple res = py_vjp_drift(t, matrix_to_numpy(Z), matrix_to_numpy(a), dt);
        return {numpy_to_matrix(res[0].cast<py::array_t<float>>()), 
                numpy_to_matrix(res[1].cast<py::array_t<float>>())};
    };
    
    auto vjp_diffusion = [&py_vjp_diffusion](float t, const Matrix& Z, const Matrix& a, const Matrix& dW) -> std::pair<Matrix, Matrix> {
        py::tuple res = py_vjp_diffusion(t, matrix_to_numpy(Z), matrix_to_numpy(a), matrix_to_numpy(dW));
        return {numpy_to_matrix(res[0].cast<py::array_t<float>>()), 
                numpy_to_matrix(res[1].cast<py::array_t<float>>())};
    };
    
    auto result = euler_maruyama_adjoint_path(Z_seq, grad_output_seq, T, W_inc, vjp_drift, vjp_diffusion);
    
    return py::make_tuple(
        matrix_to_numpy(std::get<0>(result)),
        matrix_to_numpy(std::get<1>(result)),
        matrix_to_numpy(std::get<2>(result))
    );
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

    m.def("euler_maruyama_adjoint_path", &euler_maruyama_adjoint_wrapper,
          py::arg("Z_seq"), py::arg("grad_output_seq"), py::arg("T"), py::arg("W_increments"), 
          py::arg("vjp_drift"), py::arg("vjp_diffusion"),
          "Adjoint backward solver for the Euler-Maruyama method");
}

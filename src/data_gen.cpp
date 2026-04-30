#include "core/matrix.h"
#include <cmath>
#include <stdexcept>

// Generate Fractional Brownian Motion (fBM) path using Cholesky Decomposition
// H: Hurst parameter (0 < H < 1). Rough paths have H < 0.5.
// N: Number of discrete steps
// T: Total time
// Returns: A (N+1) x 2 MacTensor Matrix where Column 0 is Time and Column 1 is X_t
Matrix generate_fbm(float H, size_t N, float T = 1.0f) {
    if (H <= 0.0f || H >= 1.0f) {
        throw std::invalid_argument("Hurst parameter must be in (0, 1)");
    }
    if (N == 0) {
        return Matrix(0, 2);
    }
    
    float dt = T / N;
    
    // Construct the auto-covariance matrix for t_1 ... t_N
    Matrix Sigma(N, N);
    
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float ti = (i + 1) * dt;
            float tj = (j + 1) * dt;
            
            float val = 0.5f * (std::pow(ti, 2.0f * H) + std::pow(tj, 2.0f * H) - std::pow(std::abs(ti - tj), 2.0f * H));
            
            // Add jitter to the diagonal to ensure the matrix is strictly positive definite 
            // for the Cholesky decomposition (mitigates floating point inaccuracies).
            if (i == j) {
                val += 1e-6f; 
            }
            Sigma(i, j) = val;
        }
    }
    
    // L * L^T = Sigma
    Matrix L = Sigma.cholesky();
    
    // Z ~ N(0, I)
    Matrix Z = Matrix::random(N, 1);
    
    // X = L * Z
    Matrix X = L.matmul(Z);
    
    // Construct the 2D path (Time, Value) starting at (0, 0)
    Matrix path(N + 1, 2);
    
    path(0, 0) = 0.0f; // t_0
    path(0, 1) = 0.0f; // X_0
    
    for (size_t i = 0; i < N; ++i) {
        path(i + 1, 0) = (i + 1) * dt;
        path(i + 1, 1) = X(i, 0);
    }
    
    return path;
}

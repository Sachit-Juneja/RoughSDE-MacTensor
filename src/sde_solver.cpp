#include "core/matrix.h"
#include <functional>
#include <stdexcept>

// Perform a single step of the Euler-Maruyama method
// X: Current state (size D x 1)
// t: Current time
// dt: Time step size
// dW: Noise increment for the step (size M x 1)
// drift: Function evaluating mu(t, X_t) returning (D x 1) Matrix
// diffusion: Function evaluating sigma(t, X_t) returning (D x M) Matrix
// Returns: X_{t + dt}
Matrix euler_maruyama_step(
    const Matrix& X, 
    float t, 
    float dt, 
    const Matrix& dW,
    const std::function<Matrix(float, const Matrix&)>& drift,
    const std::function<Matrix(float, const Matrix&)>& diffusion
) {
    if (X.cols != 1) {
        throw std::invalid_argument("State X must be a column vector (D x 1)");
    }
    
    // Evaluate drift: mu(t, X_t) * dt
    Matrix mu = drift(t, X);
    Matrix drift_term = mu * dt; 
    
    // Evaluate diffusion: sigma(t, X_t) * dW
    Matrix sigma = diffusion(t, X);
    Matrix diffusion_term = sigma.matmul(dW);
    
    // X_{t+dt} = X_t + mu(t, X_t)dt + sigma(t, X_t)dW
    Matrix X_next = X + drift_term + diffusion_term;
    
    return X_next;
}

// Generate an entire path using Euler-Maruyama
// X0: Initial state (size D x 1)
// T: Total time
// W: Noise path increments (N x M) where N is number of steps
// drift and diffusion functions
// Returns: (N+1) x D Matrix containing the state trajectory
Matrix euler_maruyama_path(
    const Matrix& X0,
    float T,
    const Matrix& W_increments,
    const std::function<Matrix(float, const Matrix&)>& drift,
    const std::function<Matrix(float, const Matrix&)>& diffusion
) {
    size_t N = W_increments.rows;
    size_t M = W_increments.cols;
    size_t D = X0.rows;
    
    if (X0.cols != 1) {
        throw std::invalid_argument("Initial state X0 must be a column vector (D x 1)");
    }
    
    float dt = T / N;
    
    // Trajectory matrix: N+1 rows, D columns
    Matrix path(N + 1, D);
    
    // Set initial state
    for(size_t i = 0; i < D; ++i) {
        path(0, i) = X0(i, 0);
    }
    
    Matrix X_current = X0.clone();
    
    for (size_t i = 0; i < N; ++i) {
        float t = i * dt;
        
        // Extract dW for this step as an M x 1 column vector
        Matrix dW = W_increments.row(i).transpose();
        
        // Take a step
        X_current = euler_maruyama_step(X_current, t, dt, dW, drift, diffusion);
        
        // Store in path
        for(size_t j = 0; j < D; ++j) {
            path(i + 1, j) = X_current(j, 0);
        }
    }
    
    return path;
}

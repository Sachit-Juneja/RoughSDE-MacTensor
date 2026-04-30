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

#include <utility>

// Adjoint backward solver for the Euler-Maruyama method
// Z_seq: Forward trajectory of states (N+1 x D)
// grad_output_seq: Gradients of the loss w.r.t the trajectory (N+1 x D)
// T: Total time
// W_increments: The same noise increments used in the forward pass (N x M)
// vjp_drift: Callback to compute VJPs of the drift step (mu * dt)
// vjp_diffusion: Callback to compute VJPs of the diffusion step (sigma * dW)
// Returns: Tuple containing (a_0, a_theta_mu, a_theta_sigma)
std::tuple<Matrix, Matrix, Matrix> euler_maruyama_adjoint_path(
    const Matrix& Z_seq,
    const Matrix& grad_output_seq,
    float T,
    const Matrix& W_increments,
    const std::function<std::pair<Matrix, Matrix>(float, const Matrix&, const Matrix&, float)>& vjp_drift,
    const std::function<std::pair<Matrix, Matrix>(float, const Matrix&, const Matrix&, const Matrix&)>& vjp_diffusion
) {
    size_t N = W_increments.rows;
    size_t D = Z_seq.cols;
    float dt = T / N;
    
    if (Z_seq.rows != N + 1 || grad_output_seq.rows != N + 1) {
        throw std::invalid_argument("Z_seq and grad_output_seq must have N+1 rows");
    }
    
    // Initialize adjoint state with the gradient at the final time step T
    Matrix a_t = grad_output_seq.row(N).transpose();
    
    // We do a dummy call to the VJP functions at t=T to get the parameter vector sizes
    Matrix Z_dummy = Z_seq.row(N).transpose();
    Matrix dW_dummy = W_increments.row(0).transpose(); 
    
    auto drift_vjp_dummy = vjp_drift(T, Z_dummy, a_t, dt);
    auto diff_vjp_dummy = vjp_diffusion(T, Z_dummy, a_t, dW_dummy);
    
    size_t P_mu = drift_vjp_dummy.second.rows;
    size_t P_sigma = diff_vjp_dummy.second.rows;
    
    Matrix a_theta_mu(P_mu, 1);
    Matrix a_theta_sigma(P_sigma, 1);
    
    // Iterate backward
    for (int i = (int)N - 1; i >= 0; --i) {
        float t = i * dt;
        Matrix Z_t = Z_seq.row(i).transpose();
        Matrix dW = W_increments.row(i).transpose();
        
        // Get VJPs for this step
        auto drift_vjps = vjp_drift(t, Z_t, a_t, dt);
        auto diff_vjps = vjp_diffusion(t, Z_t, a_t, dW);
        
        // Update adjoint state: a_{t} = a_{t+dt} + VJP_Z
        a_t = a_t + drift_vjps.first + diff_vjps.first;
        
        // Inject intermediate gradients if the loss depends on the full path
        Matrix current_grad = grad_output_seq.row(i).transpose();
        a_t = a_t + current_grad;
        
        // Accumulate parameter gradients
        a_theta_mu = a_theta_mu + drift_vjps.second;
        a_theta_sigma = a_theta_sigma + diff_vjps.second;
    }
    
    return std::make_tuple(a_t, a_theta_mu, a_theta_sigma);
}

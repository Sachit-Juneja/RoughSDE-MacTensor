#include "core/matrix.h"
#include <vector>
#include <cmath>
#include <stdexcept>

// Applies the Lead-Lag transformation to a 1D series of values
// Input: X of size (N+1) x 1
// Output: (2N+1) x 2 Matrix where Column 0 is Lead, Column 1 is Lag
Matrix lead_lag_transform(const Matrix& X) {
    if (X.cols != 1) {
        throw std::invalid_argument("Lead-Lag input must be a 1D time series (N+1 x 1)");
    }
    
    size_t N = X.rows - 1;
    Matrix LL(2 * N + 1, 2);
    
    // Initial point (X_0, X_0)
    LL(0, 0) = X(0, 0); // Lead
    LL(0, 1) = X(0, 0); // Lag
    
    for (size_t i = 0; i < N; ++i) {
        // Step 1: Lead jumps, Lag stays
        LL(2 * i + 1, 0) = X(i + 1, 0);
        LL(2 * i + 1, 1) = X(i, 0);
        
        // Step 2: Lag jumps, Lead stays (both at X_{i+1})
        LL(2 * i + 2, 0) = X(i + 1, 0);
        LL(2 * i + 2, 1) = X(i + 1, 0);
    }
    
    return LL;
}

// Compute the truncated signature of a path up to depth M
// Input: Path of size K x d (K points in R^d)
// Output: 1D Matrix (vector) of size (d^{M+1} - 1) / (d - 1) containing the signature terms.
Matrix compute_signature(const Matrix& path, size_t depth) {
    size_t K = path.rows;
    size_t d = path.cols;
    
    if (K < 2) {
        throw std::invalid_argument("Path must have at least 2 points to compute a signature");
    }
    
    // Total size of signature S up to depth M: 1 + d + d^2 + ... + d^M
    size_t sig_size = 0;
    std::vector<size_t> level_sizes(depth + 1);
    level_sizes[0] = 1;
    sig_size += 1;
    
    for (size_t k = 1; k <= depth; ++k) {
        level_sizes[k] = level_sizes[k - 1] * d;
        sig_size += level_sizes[k];
    }
    
    // We will use std::vector to hold the current signature accumulation
    std::vector<float> S(sig_size, 0.0f);
    S[0] = 1.0f; // 0th order term is always 1
    
    // Iterate over each segment of the path
    for (size_t i = 0; i < K - 1; ++i) {
        // Segment displacement vector Delta
        std::vector<float> delta(d);
        for (size_t j = 0; j < d; ++j) {
            delta[j] = path(i + 1, j) - path(i, j);
        }
        
        // Compute signature of the segment: exp(Delta) = 1 + Delta + 1/2 Delta^2 + ...
        std::vector<float> seg_S(sig_size, 0.0f);
        seg_S[0] = 1.0f;
        
        size_t current_offset = 1;
        std::vector<float> prev_tensor(1, 1.0f); // 0th order is 1
        
        float factorial = 1.0f;
        for (size_t k = 1; k <= depth; ++k) {
            factorial *= static_cast<float>(k);
            std::vector<float> next_tensor(level_sizes[k]);
            
            // Tensor product: prev_tensor (size d^{k-1}) \otimes Delta (size d)
            for (size_t u = 0; u < level_sizes[k - 1]; ++u) {
                for (size_t v = 0; v < d; ++v) {
                    float val = prev_tensor[u] * delta[v];
                    next_tensor[u * d + v] = val;
                    seg_S[current_offset + u * d + v] = val / factorial;
                }
            }
            
            prev_tensor = next_tensor;
            current_offset += level_sizes[k];
        }
        
        // Use Chen's relation to merge seg_S into S: S_new = S \otimes seg_S
        std::vector<float> S_new(sig_size, 0.0f);
        
        for (size_t k = 0; k <= depth; ++k) {
            // C_k = \sum_{j=0}^k A_j \otimes B_{k-j}
            size_t C_offset = 0;
            for(size_t l=0; l<k; ++l) C_offset += level_sizes[l];
            
            for (size_t j = 0; j <= k; ++j) {
                size_t A_offset = 0;
                for(size_t l=0; l<j; ++l) A_offset += level_sizes[l];
                
                size_t B_offset = 0;
                for(size_t l=0; l<(k-j); ++l) B_offset += level_sizes[l];
                
                size_t A_size = level_sizes[j];
                size_t B_size = level_sizes[k - j];
                
                // Tensor product S^(j) \otimes seg_S^(k-j)
                for (size_t u = 0; u < A_size; ++u) {
                    for (size_t v = 0; v < B_size; ++v) {
                        S_new[C_offset + u * B_size + v] += S[A_offset + u] * seg_S[B_offset + v];
                    }
                }
            }
        }
        
        S = std::move(S_new);
    }
    
    // Convert back to a MacTensor Matrix (Column vector)
    Matrix S_mat(sig_size, 1);
    for (size_t i = 0; i < sig_size; ++i) {
        S_mat(i, 0) = S[i];
    }
    
    return S_mat;
}

// Compute the exact analytical gradient of the signature w.r.t the path
Matrix compute_signature_backward(const Matrix& path, const Matrix& grad_output, size_t depth) {
    size_t K = path.rows;
    size_t d = path.cols;
    
    if (K < 2) {
        throw std::invalid_argument("Path must have at least 2 points to compute a signature backward");
    }
    
    size_t sig_size = 0;
    std::vector<size_t> level_sizes(depth + 1);
    level_sizes[0] = 1;
    sig_size += 1;
    for (size_t k = 1; k <= depth; ++k) {
        level_sizes[k] = level_sizes[k - 1] * d;
        sig_size += level_sizes[k];
    }
    
    // 1. Forward Pass with caching
    std::vector<std::vector<float>> S_history(K);
    std::vector<std::vector<float>> seg_S_history(K - 1);
    
    S_history[0] = std::vector<float>(sig_size, 0.0f);
    S_history[0][0] = 1.0f;
    
    for (size_t i = 0; i < K - 1; ++i) {
        std::vector<float> delta(d);
        for (size_t j = 0; j < d; ++j) {
            delta[j] = path(i + 1, j) - path(i, j);
        }
        
        std::vector<float> seg_S(sig_size, 0.0f);
        seg_S[0] = 1.0f;
        size_t current_offset = 1;
        std::vector<float> prev_tensor(1, 1.0f);
        float factorial = 1.0f;
        for (size_t k = 1; k <= depth; ++k) {
            factorial *= static_cast<float>(k);
            std::vector<float> next_tensor(level_sizes[k]);
            for (size_t u = 0; u < level_sizes[k - 1]; ++u) {
                for (size_t v = 0; v < d; ++v) {
                    float val = prev_tensor[u] * delta[v];
                    next_tensor[u * d + v] = val;
                    seg_S[current_offset + u * d + v] = val / factorial;
                }
            }
            prev_tensor = next_tensor;
            current_offset += level_sizes[k];
        }
        seg_S_history[i] = seg_S;
        
        std::vector<float> S_new(sig_size, 0.0f);
        for (size_t k = 0; k <= depth; ++k) {
            size_t C_offset = 0;
            for(size_t l=0; l<k; ++l) C_offset += level_sizes[l];
            for (size_t j = 0; j <= k; ++j) {
                size_t A_offset = 0;
                for(size_t l=0; l<j; ++l) A_offset += level_sizes[l];
                size_t B_offset = 0;
                for(size_t l=0; l<(k-j); ++l) B_offset += level_sizes[l];
                size_t A_size = level_sizes[j];
                size_t B_size = level_sizes[k - j];
                for (size_t u = 0; u < A_size; ++u) {
                    for (size_t v = 0; v < B_size; ++v) {
                        S_new[C_offset + u * B_size + v] += S_history[i][A_offset + u] * seg_S[B_offset + v];
                    }
                }
            }
        }
        S_history[i + 1] = S_new;
    }
    
    // 2. Backward Pass
    Matrix grad_path(K, d);
    std::vector<float> grad_S(sig_size, 0.0f);
    for(size_t i=0; i<sig_size; ++i) grad_S[i] = grad_output(i, 0);
    
    for (int i = (int)(K - 2); i >= 0; --i) {
        std::vector<float> grad_S_prev(sig_size, 0.0f);
        std::vector<float> grad_seg_S(sig_size, 0.0f);
        
        // Backprop through Chen's Identity: S_new = S_prev \otimes seg_S
        for (size_t k = 0; k <= depth; ++k) {
            size_t C_offset = 0;
            for(size_t l=0; l<k; ++l) C_offset += level_sizes[l];
            for (size_t j = 0; j <= k; ++j) {
                size_t A_offset = 0;
                for(size_t l=0; l<j; ++l) A_offset += level_sizes[l];
                size_t B_offset = 0;
                for(size_t l=0; l<(k-j); ++l) B_offset += level_sizes[l];
                size_t A_size = level_sizes[j];
                size_t B_size = level_sizes[k - j];
                
                for (size_t u = 0; u < A_size; ++u) {
                    for (size_t v = 0; v < B_size; ++v) {
                        grad_S_prev[A_offset + u] += grad_S[C_offset + u * B_size + v] * seg_S_history[i][B_offset + v];
                        grad_seg_S[B_offset + v] += grad_S[C_offset + u * B_size + v] * S_history[i][A_offset + u];
                    }
                }
            }
        }
        
        // Backprop through Exponential of Delta
        std::vector<float> grad_delta(d, 0.0f);
        std::vector<float> delta(d);
        for (size_t j = 0; j < d; ++j) delta[j] = path(i + 1, j) - path(i, j);
        
        std::vector<float> prev_tensor(1, 1.0f);
        std::vector<float> grad_prev_tensor(1, 0.0f);
        size_t current_offset = 1;
        
        // We need to backpropagate through the iterative tensor product
        // We will store forward pass intermediate prev_tensors
        std::vector<std::vector<float>> prev_tensor_history(depth);
        prev_tensor_history[0] = prev_tensor;
        
        for (size_t k = 1; k < depth; ++k) {
            std::vector<float> next_tensor(level_sizes[k]);
            for (size_t u = 0; u < level_sizes[k - 1]; ++u) {
                for (size_t v = 0; v < d; ++v) {
                    next_tensor[u * d + v] = prev_tensor[u] * delta[v];
                }
            }
            prev_tensor = next_tensor;
            prev_tensor_history[k] = prev_tensor;
        }
        
        // Now go backward through the levels
        std::vector<float> grad_next_tensor;
        for (int k = depth; k >= 1; --k) {
            float factorial = 1.0f;
            for(int f=1; f<=k; ++f) factorial *= static_cast<float>(f);
            
            size_t C_offset = 0;
            for(int l=0; l<=k; ++l) C_offset += level_sizes[l];
            C_offset -= level_sizes[k]; // offset to current level
            
            grad_next_tensor.assign(level_sizes[k], 0.0f);
            for (size_t idx = 0; idx < level_sizes[k]; ++idx) {
                grad_next_tensor[idx] += grad_seg_S[C_offset + idx] / factorial;
            }
            // Add gradients coming from higher level next_tensor (if any)
            if (k < (int)depth) {
                for (size_t idx = 0; idx < level_sizes[k]; ++idx) {
                    grad_next_tensor[idx] += grad_prev_tensor[idx];
                }
            }
            
            grad_prev_tensor.assign(level_sizes[k - 1], 0.0f);
            for (size_t u = 0; u < level_sizes[k - 1]; ++u) {
                for (size_t v = 0; v < d; ++v) {
                    grad_prev_tensor[u] += grad_next_tensor[u * d + v] * delta[v];
                    grad_delta[v] += grad_next_tensor[u * d + v] * prev_tensor_history[k - 1][u];
                }
            }
        }
        
        // Update path gradients
        for (size_t j = 0; j < d; ++j) {
            grad_path(i + 1, j) += grad_delta[j];
            grad_path(i, j) -= grad_delta[j];
        }
        
        grad_S = std::move(grad_S_prev);
    }
    
    return grad_path;
}

// Applies backward pass for Lead-Lag
Matrix lead_lag_transform_backward(const Matrix& path, const Matrix& grad_LL) {
    size_t N = path.rows - 1;
    Matrix grad_X(N + 1, 1);
    
    // Initial point
    grad_X(0, 0) += grad_LL(0, 0);
    grad_X(0, 0) += grad_LL(0, 1);
    
    for (size_t i = 0; i < N; ++i) {
        // Step 1
        grad_X(i + 1, 0) += grad_LL(2 * i + 1, 0);
        grad_X(i, 0) += grad_LL(2 * i + 1, 1);
        
        // Step 2
        grad_X(i + 1, 0) += grad_LL(2 * i + 2, 0);
        grad_X(i + 1, 0) += grad_LL(2 * i + 2, 1);
    }
    
    return grad_X;
}


import sys
import os
import torch
import numpy as np
import torch.optim as optim

# Add the build directory to path to import rough_sde
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build')))

try:
    import rough_sde
except ImportError:
    print("Failed to import rough_sde. Have you compiled the CMake project?")
    print("Run: mkdir build && cd build && cmake .. && make")
    sys.exit(1)

from neural_sde import DriftNet, DiffusionNet

# Hyperparameters
HURST = 0.1
STEPS = 50
T = 1.0
DEPTH = 3
EPOCHS = 100
LR = 1e-3

STATE_DIM = 1
NOISE_DIM = 1

def main():
    print(f"Generating true rough path with Hurst = {HURST}...")
    # 1. Generate real rough data (fBM) using C++ MacTensor
    # Returns (STEPS+1) x 2 array [Time, Value]
    true_path = rough_sde.generate_fbm(HURST, STEPS, T)
    
    # 2. Extract values and apply Lead-Lag
    true_values = true_path[:, 1].reshape(-1, 1)
    true_ll = rough_sde.lead_lag_transform(true_values)
    
    # 3. Compute true signature
    true_sig = rough_sde.compute_signature(true_ll, DEPTH)
    true_sig_tensor = torch.tensor(true_sig, dtype=torch.float32)
    
    print("True signature computed. Shape:", true_sig.shape)
    
    # 4. Initialize Neural SDE
    drift = DriftNet(STATE_DIM)
    diffusion = DiffusionNet(STATE_DIM, NOISE_DIM)
    
    # Optimizer
    optimizer = optim.Adam(list(drift.parameters()) + list(diffusion.parameters()), lr=LR)
    
    # Custom training loop
    # NOTE: Since the C++ MacTensor backend does not track PyTorch Autograd,
    # backpropagating exactly through `euler_maruyama_path` requires either:
    # A) The Continuous Adjoint Sensitivity Method (Neural SDEs)
    # B) Finite Differences over parameters
    # For demonstration of the orchestration, we use a proxy loss or assume
    # finite differences if gradients are needed.
    # Here we show the orchestration logic for the Signature Kernel Loss.
    
    print("Starting training loop...")
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # We need to simulate the path.
        # Generate standard Brownian increments for the solver
        dt = T / STEPS
        dW = np.random.normal(0, np.sqrt(dt), size=(STEPS, NOISE_DIM)).astype(np.float32)
        X0 = np.zeros((STATE_DIM, 1), dtype=np.float32)
        
        # Simulate using C++ Solver. We wrap the PyTorch networks to detach during C++ eval
        def drift_wrapper(t, x):
            with torch.no_grad():
                return drift(t, x)
                
        def diffusion_wrapper(t, x):
            with torch.no_grad():
                return diffusion(t, x)
                
        # Simulated path (STEPS+1) x STATE_DIM
        sim_path = rough_sde.euler_maruyama_path(X0, T, dW, drift_wrapper, diffusion_wrapper)
        
        # Apply Lead-Lag and compute Signature in C++
        sim_ll = rough_sde.lead_lag_transform(sim_path)
        sim_sig = rough_sde.compute_signature(sim_ll, DEPTH)
        sim_sig_tensor = torch.tensor(sim_sig, dtype=torch.float32, requires_grad=True)
        
        # Signature Kernel Loss (MMD)
        # Inner product distance ||S(X) - S(Y)||^2
        loss = torch.sum((sim_sig_tensor - true_sig_tensor) ** 2)
        
        # To actually update weights without the adjoint method, we'd apply finite diff or REINFORCE.
        # Calling backward on the tensor directly won't flow to Drift/Diffusion automatically
        # unless integrated with Adjoint equations.
        # loss.backward() 
        # optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | MMD Loss: {loss.item():.6f}")

    print("Training loop complete!")

if __name__ == "__main__":
    main()

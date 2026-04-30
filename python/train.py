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

from neural_sde import DriftNet, DiffusionNet, EulerMaruyamaSDE

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
        
        # Use Custom Autograd SDE Solver
        params = list(drift.parameters()) + list(diffusion.parameters())
        sim_path_tensor = EulerMaruyamaSDE.apply(torch.tensor(X0, dtype=torch.float32, requires_grad=True), T, torch.tensor(dW, dtype=torch.float32), drift, diffusion, *params)
        
        # We need the signature for the loss
        sim_path_np = sim_path_tensor.detach().numpy()
        sim_ll = rough_sde.lead_lag_transform(sim_path_np)
        sim_sig = rough_sde.compute_signature(sim_ll, DEPTH)
        
        # Create a proxy tensor for the signature so we can flow gradients backward.
        # Note: The true way to flow gradients back through compute_signature requires differentiating
        # the signature computation. For demonstration, we assume we want to backpropagate
        # directly into the path using some path-level loss, e.g. MSE against a target path.
        # Since `compute_signature` is purely in C++, we cannot autograd through it natively.
        # We'll compute the gradient of the MMD loss w.r.t the path manually or assume path MSE.
        
        # Proxy Path Loss for Orchestration Test
        # loss = ||sim_path - true_path||^2
        target_path_tensor = torch.tensor(true_path[:, 1:], dtype=torch.float32)
        loss = torch.sum((sim_path_tensor - target_path_tensor) ** 2)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Path Loss: {loss.item():.6f}")

    print("Training loop complete!")

if __name__ == "__main__":
    main()

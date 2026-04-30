import sys
import os
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

# Add the build directory to path to import rough_sde
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build')))

try:
    import rough_sde
except ImportError:
    print("Failed to import rough_sde. Have you compiled the CMake project?")
    sys.exit(1)

from neural_sde import DriftNet, DiffusionNet, EulerMaruyamaSDE

# Hyperparameters
HURST = 0.1
STEPS = 50
T = 1.0
DEPTH = 3
EPOCHS = 500
LR = 0.005

STATE_DIM = 1
NOISE_DIM = 1

def main():
    print(f"Generating true rough path with Hurst = {HURST}...")
    # 1. Generate real rough data (fBM)
    true_path = rough_sde.generate_fbm(HURST, STEPS, T)
    true_values = true_path[:, 1].reshape(-1, 1)
    
    # 2. Compute true signature
    true_ll = rough_sde.lead_lag_transform(true_values)
    true_sig = rough_sde.compute_signature(true_ll, DEPTH)
    
    print(f"True signature computed. Shape: {true_sig.shape}")
    
    # 3. Initialize Neural SDE
    drift = DriftNet(STATE_DIM)
    diffusion = DiffusionNet(STATE_DIM, NOISE_DIM)
    optimizer = optim.Adam(list(drift.parameters()) + list(diffusion.parameters()), lr=LR)
    
    # Save Untrained Path for Visualization
    dW_test = np.random.normal(0, np.sqrt(T/STEPS), size=(STEPS, NOISE_DIM)).astype(np.float32)
    X0_test = np.zeros((STATE_DIM, 1), dtype=np.float32)
    params_test = list(drift.parameters()) + list(diffusion.parameters())
    
    untrained_path_tensor = EulerMaruyamaSDE.apply(
        torch.tensor(X0_test, dtype=torch.float32), T, 
        torch.tensor(dW_test, dtype=torch.float32), 
        drift, diffusion, *params_test
    )
    untrained_path = untrained_path_tensor.detach().numpy()

    loss_history = []
    
    print("Starting training loop...")
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # Forward Pass
        dt = T / STEPS
        dW = np.random.normal(0, np.sqrt(dt), size=(STEPS, NOISE_DIM)).astype(np.float32)
        X0 = np.zeros((STATE_DIM, 1), dtype=np.float32)
        
        params = list(drift.parameters()) + list(diffusion.parameters())
        sim_path_tensor = EulerMaruyamaSDE.apply(
            torch.tensor(X0, dtype=torch.float32, requires_grad=True), T, 
            torch.tensor(dW, dtype=torch.float32), 
            drift, diffusion, *params
        )
        
        sim_path_np = sim_path_tensor.detach().numpy()
        sim_ll = rough_sde.lead_lag_transform(sim_path_np)
        sim_sig = rough_sde.compute_signature(sim_ll, DEPTH)
        
        # Signature Kernel Loss (MMD)
        loss_val = np.sum((sim_sig - true_sig) ** 2)
        loss_history.append(loss_val)
        
        # To flow gradients natively through our custom C++ Adjoint Solver, 
        # we need `grad_output`: the gradient of the MMD Loss w.r.t the simulated path.
        # We calculate the exact analytical gradient of the Signature Loss.
        # L = ||S_sim - S_true||^2  =>  dL/dS_sim = 2 * (S_sim - S_true)
        g_sig = 2.0 * (sim_sig - true_sig)
        
        # Exact Backward Pass through Signature
        grad_ll = rough_sde.compute_signature_backward(sim_ll, g_sig, DEPTH)
        
        # Exact Backward Pass through Lead-Lag transformation
        grad_output = rough_sde.lead_lag_transform_backward(sim_path_np, grad_ll)
        
        # Trigger custom Adjoint C++ backward pass
        sim_path_tensor.backward(torch.tensor(grad_output, dtype=torch.float32))
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:03d} | Signature MMD Loss: {loss_val:.6f}")

    print("Training complete! Generating academic visualizations...")
    
    # Save Trained Path for Visualization
    trained_path_tensor = EulerMaruyamaSDE.apply(
        torch.tensor(X0_test, dtype=torch.float32), T, 
        torch.tensor(dW_test, dtype=torch.float32), 
        drift, diffusion, *params_test
    )
    trained_path = trained_path_tensor.detach().numpy()
    
    # Plotting
    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        plt.style.use('ggplot')
        
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), dpi=300)
    t_axis = true_path[:, 0]
    
    # Plot 1: Untrained
    axs[0].plot(t_axis, true_values, label='True fBM ($H=0.1$)', color='black', linewidth=1.5)
    axs[0].plot(t_axis, untrained_path, label='Untrained Neural SDE', color='red', linestyle='--', alpha=0.8)
    axs[0].set_title("Untrained vs True Path")
    axs[0].set_xlabel("Time ($t$)")
    axs[0].set_ylabel("State ($X_t$)")
    axs[0].legend()
    axs[0].grid(True, alpha=0.4)
    
    # Plot 2: Trained
    axs[1].plot(t_axis, true_values, label='True fBM ($H=0.1$)', color='black', linewidth=1.5)
    axs[1].plot(t_axis, trained_path, label='Trained Neural SDE', color='blue', linestyle='-', alpha=0.8)
    axs[1].set_title("Trained vs True Path")
    axs[1].set_xlabel("Time ($t$)")
    axs[1].legend()
    axs[1].grid(True, alpha=0.4)
    
    # Plot 3: Convergence
    # Add a small epsilon to avoid log(0) if loss becomes perfectly 0
    axs[2].plot(range(EPOCHS), np.log10(np.array(loss_history) + 1e-12), color='green', linewidth=2)
    axs[2].set_title("Signature Kernel Convergence")
    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("$\log_{10}$(MMD Loss)")
    axs[2].grid(True, alpha=0.4)
    
    plt.tight_layout()
    output_file = os.path.join(os.path.dirname(__file__), "..", "rough_volatility_results.pdf")
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Saved publication-quality plot to: {output_file}")

if __name__ == "__main__":
    main()

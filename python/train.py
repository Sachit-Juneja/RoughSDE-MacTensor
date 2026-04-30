import sys
import os
import torch
import numpy as np
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add the build directory to path to import rough_sde
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build')))

try:
    import rough_sde
except ImportError:
    print("Failed to import rough_sde. Have you compiled the CMake project?")
    sys.exit(1)

from neural_sde import DriftNet, DiffusionNet, EulerMaruyamaSDE

# ──────────────────────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────────────────────
HURST = 0.1
STEPS = 50
T = 1.0
DEPTH = 3
EPOCHS = 800
LR = 3e-3
GRAD_CLIP = 2.0

STATE_DIM = 1
NOISE_DIM = 1

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    print(f"=== Rough Neural SDE Training ===")
    print(f"Hurst={HURST}, Steps={STEPS}, Depth={DEPTH}, Epochs={EPOCHS}")
    
    # ──────────────────────────────────────────────────────────
    # 1. Generate ground-truth rough volatility path
    # ──────────────────────────────────────────────────────────
    true_path = rough_sde.generate_fbm(HURST, STEPS, T)
    true_values = true_path[:, 1].reshape(-1, 1).astype(np.float32)
    
    true_ll = rough_sde.lead_lag_transform(true_values)
    true_sig = rough_sde.compute_signature(true_ll, DEPTH)
    print(f"True signature computed. Shape: {true_sig.shape}")
    
    # ──────────────────────────────────────────────────────────
    # 2. Initialize Neural SDE
    # ──────────────────────────────────────────────────────────
    drift = DriftNet(STATE_DIM)
    diffusion = DiffusionNet(STATE_DIM, NOISE_DIM)
    all_params = list(drift.parameters()) + list(diffusion.parameters())
    optimizer = optim.Adam(all_params, lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    
    # ──────────────────────────────────────────────────────────
    # 3. Pre-generate fixed noise for stable training
    #    Using a fixed set of noise realizations ensures the loss
    #    landscape is deterministic, enabling stable convergence.
    # ──────────────────────────────────────────────────────────
    dt = T / STEPS
    NUM_FIXED_NOISE = 8
    fixed_noises = [
        np.random.normal(0, np.sqrt(dt), size=(STEPS, NOISE_DIM)).astype(np.float32)
        for _ in range(NUM_FIXED_NOISE)
    ]
    X0 = np.zeros((STATE_DIM, 1), dtype=np.float32)
    
    # Use noise #0 as the test/visualization noise
    dW_test = fixed_noises[0]
    
    # Capture untrained path
    untrained_path_tensor = EulerMaruyamaSDE.apply(
        torch.tensor(X0, dtype=torch.float32), T, 
        torch.tensor(dW_test, dtype=torch.float32), 
        drift, diffusion, *all_params
    )
    untrained_path = untrained_path_tensor.detach().numpy()
    
    # ──────────────────────────────────────────────────────────
    # 4. Training Loop
    # ──────────────────────────────────────────────────────────
    loss_history = []
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # Cycle through fixed noise realizations
        dW = fixed_noises[epoch % NUM_FIXED_NOISE]
        
        params = list(drift.parameters()) + list(diffusion.parameters())
        sim_path_tensor = EulerMaruyamaSDE.apply(
            torch.tensor(X0, dtype=torch.float32, requires_grad=True), T, 
            torch.tensor(dW, dtype=torch.float32), 
            drift, diffusion, *params
        )
        
        sim_path_np = sim_path_tensor.detach().numpy()
        sim_ll = rough_sde.lead_lag_transform(sim_path_np)
        sim_sig = rough_sde.compute_signature(sim_ll, DEPTH)
        
        # Signature Kernel MMD Loss
        loss_val = float(np.sum((sim_sig - true_sig) ** 2))
        loss_history.append(loss_val)
        
        # Exact analytical gradient: dL/dS = 2(S_sim - S_true)
        g_sig = (2.0 * (sim_sig - true_sig)).astype(np.float32)
        
        # Exact backward through Signature (C++)
        grad_ll = rough_sde.compute_signature_backward(sim_ll, g_sig, DEPTH)
        
        # Exact backward through Lead-Lag (C++)
        grad_output = rough_sde.lead_lag_transform_backward(sim_path_np, grad_ll)
        
        # Trigger C++ Adjoint backward pass
        sim_path_tensor.backward(torch.tensor(grad_output, dtype=torch.float32))
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP)
        
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1:04d}/{EPOCHS} | MMD Loss: {loss_val:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    print(f"\nTraining complete. Final loss: {loss_history[-1]:.4f}")
    print("Generating publication-quality visualizations...")
    
    # ──────────────────────────────────────────────────────────
    # 5. Capture trained path (same noise as untrained)
    # ──────────────────────────────────────────────────────────
    params_eval = list(drift.parameters()) + list(diffusion.parameters())
    trained_path_tensor = EulerMaruyamaSDE.apply(
        torch.tensor(X0, dtype=torch.float32), T, 
        torch.tensor(dW_test, dtype=torch.float32), 
        drift, diffusion, *params_eval
    )
    trained_path = trained_path_tensor.detach().numpy()
    
    # ──────────────────────────────────────────────────────────
    # 6. Publication-Quality Visualization (NeurIPS / ICML Style)
    # ──────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 8,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.4,
        'lines.linewidth': 1.5,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'text.usetex': False,
    })
    
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 3.8))
    t_axis = true_path[:, 0]
    
    # Color Palette (carefully chosen for academic readability)
    c_true = '#1a1a2e'      # Deep navy
    c_untrained = '#e63946'  # Crimson
    c_trained = '#457b9d'    # Steel blue
    c_loss = '#2a9d8f'       # Teal
    
    # Panel (a): Before Training
    axs[0].plot(t_axis, true_values, label=r'True fBM ($H=0.1$)', color=c_true, linewidth=1.8)
    axs[0].plot(t_axis, untrained_path, label='Untrained Neural SDE', color=c_untrained, linestyle='--', linewidth=1.2, alpha=0.85)
    axs[0].set_title("(a) Before Training", fontweight='bold')
    axs[0].set_xlabel(r"Time $t$")
    axs[0].set_ylabel(r"State $X_t$")
    axs[0].legend(loc='best', framealpha=0.9, edgecolor='none')
    axs[0].grid(True, alpha=0.3, linestyle=':')
    
    # Panel (b): After Training
    axs[1].plot(t_axis, true_values, label=r'True fBM ($H=0.1$)', color=c_true, linewidth=1.8)
    axs[1].plot(t_axis, trained_path, label='Trained Neural SDE', color=c_trained, linewidth=1.4, alpha=0.9)
    axs[1].set_title("(b) After Training", fontweight='bold')
    axs[1].set_xlabel(r"Time $t$")
    axs[1].legend(loc='best', framealpha=0.9, edgecolor='none')
    axs[1].grid(True, alpha=0.3, linestyle=':')
    
    # Panel (c): Loss Convergence
    log_loss = np.log10(np.array(loss_history) + 1e-12)
    window = 30
    if len(log_loss) >= window:
        kernel = np.ones(window) / window
        smooth_loss = np.convolve(log_loss, kernel, mode='valid')
        x_smooth = np.arange(window - 1, len(log_loss))
        axs[2].plot(range(len(log_loss)), log_loss, color=c_loss, alpha=0.15, linewidth=0.6)
        axs[2].plot(x_smooth, smooth_loss, color=c_loss, linewidth=2.0, label='Smoothed')
    else:
        axs[2].plot(range(len(log_loss)), log_loss, color=c_loss, linewidth=2.0)
    axs[2].set_title("(c) Convergence", fontweight='bold')
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel(r"$\log_{10}$ MMD Loss")
    axs[2].legend(loc='upper right', framealpha=0.9, edgecolor='none')
    axs[2].grid(True, alpha=0.3, linestyle=':')
    
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout(w_pad=2.5)
    output_file = os.path.join(os.path.dirname(__file__), "..", "rough_volatility_results.pdf")
    plt.savefig(output_file, bbox_inches='tight', format='pdf')
    print(f"Saved: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ablation.py — Hurst Parameter Ablation Study

Trains the MacTensor Neural SDE across three volatility regimes:
  H = 0.1  (Highly rough / Anti-persistent)
  H = 0.3  (Moderately rough)
  H = 0.5  (Standard Brownian Motion / Diffusive)

Output: ablation_results.pdf
"""

import sys, os, gc
import torch
import numpy as np
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build')))
try:
    import rough_sde
except ImportError:
    print("Failed to import rough_sde. Build the CMake project first.")
    sys.exit(1)

from neural_sde import DriftNet, DiffusionNet, EulerMaruyamaSDE

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
HURST_VALUES = [0.1, 0.3, 0.5]
STEPS = 50
T = 1.0
DEPTH = 3
EPOCHS = 500
LR = 3e-3
GRAD_CLIP = 2.0
NUM_FIXED_NOISE = 8

STATE_DIM = 1
NOISE_DIM = 1


def train_single_hurst(H, seed=0):
    """Train a fresh Neural SDE for one Hurst parameter. Returns (true_values, t_axis, untrained_path, trained_path, loss_history)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*50}")
    print(f"  Training H = {H}")
    print(f"{'='*50}")

    # Ground truth
    true_path = rough_sde.generate_fbm(H, STEPS, T)
    t_axis = true_path[:, 0]
    true_values = true_path[:, 1].reshape(-1, 1).astype(np.float32)
    true_ll = rough_sde.lead_lag_transform(true_values)
    true_sig = rough_sde.compute_signature(true_ll, DEPTH)

    # Fresh networks
    drift = DriftNet(STATE_DIM)
    diffusion = DiffusionNet(STATE_DIM, NOISE_DIM)
    all_params = list(drift.parameters()) + list(diffusion.parameters())
    optimizer = optim.Adam(all_params, lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # Fixed noise realizations
    dt = T / STEPS
    fixed_noises = [
        np.random.normal(0, np.sqrt(dt), size=(STEPS, NOISE_DIM)).astype(np.float32)
        for _ in range(NUM_FIXED_NOISE)
    ]
    X0 = np.zeros((STATE_DIM, 1), dtype=np.float32)
    dW_test = fixed_noises[0]

    # Untrained snapshot
    untrained_tensor = EulerMaruyamaSDE.apply(
        torch.tensor(X0, dtype=torch.float32), T,
        torch.tensor(dW_test, dtype=torch.float32),
        drift, diffusion, *all_params
    )
    untrained_path = untrained_tensor.detach().numpy()

    # Training loop
    loss_history = []
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
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

        loss_val = float(np.sum((sim_sig - true_sig) ** 2))
        loss_history.append(loss_val)

        g_sig = (2.0 * (sim_sig - true_sig)).astype(np.float32)
        grad_ll = rough_sde.compute_signature_backward(sim_ll, g_sig, DEPTH)
        grad_output = rough_sde.lead_lag_transform_backward(sim_path_np, grad_ll)
        sim_path_tensor.backward(torch.tensor(grad_output, dtype=torch.float32))

        torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1:04d}/{EPOCHS} | MMD Loss: {loss_val:.4f}")

    # Trained snapshot (same noise as untrained)
    params_eval = list(drift.parameters()) + list(diffusion.parameters())
    trained_tensor = EulerMaruyamaSDE.apply(
        torch.tensor(X0, dtype=torch.float32), T,
        torch.tensor(dW_test, dtype=torch.float32),
        drift, diffusion, *params_eval
    )
    trained_path = trained_tensor.detach().numpy()

    print(f"  Final Loss: {loss_history[-1]:.4f}")
    gc.collect()

    return true_values, t_axis, untrained_path, trained_path, loss_history


def main():
    results = {}
    for H in HURST_VALUES:
        results[H] = train_single_hurst(H, seed=int(H * 1000))

    # ──────────────────────────────────────────────────────────
    # NeurIPS-Grade Visualization
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

    c_true = '#1a1a2e'
    c_trained = '#457b9d'
    labels = ['(a)', '(b)', '(c)']
    regime_names = ['Highly Rough', 'Moderately Rough', 'Standard BM']

    fig, axs = plt.subplots(1, 3, figsize=(14.5, 3.8))

    for idx, H in enumerate(HURST_VALUES):
        true_vals, t_axis, _, trained_path, loss_hist = results[H]

        axs[idx].plot(t_axis, true_vals, color=c_true, linewidth=1.8,
                      label=f'True fBM ($H={H}$)')
        axs[idx].plot(t_axis, trained_path, color=c_trained, linewidth=1.3, alpha=0.9,
                      label='Trained Neural SDE')
        axs[idx].set_title(f"{labels[idx]} $H={H}$ — {regime_names[idx]}", fontweight='bold')
        axs[idx].set_xlabel(r"Time $t$")
        if idx == 0:
            axs[idx].set_ylabel(r"State $X_t$")
        axs[idx].legend(loc='best', framealpha=0.9, edgecolor='none')
        axs[idx].grid(True, alpha=0.3, linestyle=':')
        axs[idx].spines['top'].set_visible(False)
        axs[idx].spines['right'].set_visible(False)

    plt.tight_layout(w_pad=2.5)
    output_file = os.path.join(os.path.dirname(__file__), "..", "ablation_results.pdf")
    plt.savefig(output_file, bbox_inches='tight', format='pdf')
    print(f"\nSaved: {os.path.abspath(output_file)}")

    # Summary table
    print("\n" + "=" * 50)
    print(f"{'H':>6} | {'Init Loss':>12} | {'Final Loss':>12} | {'Reduction':>10}")
    print("-" * 50)
    for H in HURST_VALUES:
        _, _, _, _, hist = results[H]
        init_l, final_l = hist[0], hist[-1]
        reduction = (1 - final_l / init_l) * 100 if init_l > 0 else 0
        print(f"{H:>6.1f} | {init_l:>12.2f} | {final_l:>12.2f} | {reduction:>8.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
benchmark.py — Systems Benchmarking: O(1) Adjoint vs O(N) Autograd

Compares peak memory and wall-clock time of:
  1. Naive PyTorch Baseline  — unrolled Euler-Maruyama with full Autograd graph (O(N) memory)
  2. MacTensor C++ Adjoint   — custom adjoint backward solver with VJP callbacks (O(1) graph memory)

Output: memory_benchmark_results.pdf
"""

import sys, os, gc, time, tracemalloc
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Import MacTensor C++ extension ──────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build')))
try:
    import rough_sde
except ImportError:
    print("Failed to import rough_sde. Build the CMake project first.")
    sys.exit(1)

from neural_sde import DriftNet, DiffusionNet, EulerMaruyamaSDE

# ════════════════════════════════════════════════════════════════════
# Naive PyTorch Baseline: Unrolled Euler-Maruyama with full Autograd
# ════════════════════════════════════════════════════════════════════

class NaiveDriftNet(nn.Module):
    """Identical architecture to DriftNet — pure PyTorch, graph-tracked."""
    def __init__(self, state_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, t, x):
        # t: scalar tensor, x: (D,) tensor
        inp = torch.cat([t.unsqueeze(0) if t.dim() == 0 else t, x])
        return self.net(inp)


class NaiveDiffusionNet(nn.Module):
    """Identical architecture to DiffusionNet — pure PyTorch, graph-tracked."""
    def __init__(self, state_dim, noise_dim, hidden_dim=32):
        super().__init__()
        self.state_dim = state_dim
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(1 + state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim * noise_dim)
        )

    def forward(self, t, x):
        inp = torch.cat([t.unsqueeze(0) if t.dim() == 0 else t, x])
        return self.net(inp).reshape(self.state_dim, self.noise_dim)


def naive_euler_maruyama(y0, T, W_increments, drift_net, diffusion_net):
    """
    Standard unrolled Euler-Maruyama with every step retained in the Autograd graph.
    Memory cost: O(N) because PyTorch stores all intermediate activations.
    """
    N = W_increments.shape[0]
    dt = T / N
    D = y0.shape[0]

    path = [y0]
    x = y0
    for i in range(N):
        t = torch.tensor(i * dt, dtype=torch.float32)
        dW = W_increments[i]

        mu = drift_net(t, x)          # (D,)
        sigma = diffusion_net(t, x)   # (D, M)
        x = x + mu * dt + sigma @ dW # all kept in graph
        path.append(x)

    return torch.stack(path, dim=0)  # (N+1, D)


# ════════════════════════════════════════════════════════════════════
# Benchmarking Harness
# ════════════════════════════════════════════════════════════════════

STATE_DIM = 1
NOISE_DIM = 1
T = 1.0
DEPTH = 3
HURST = 0.1

N_STEPS_LIST = [10, 50, 100, 250, 500, 1000]


def sync_weights(src_drift, src_diff, dst_drift, dst_diff):
    """Copy weights from source nets to destination nets so both use identical parameters."""
    dst_drift.load_state_dict(src_drift.state_dict())
    dst_diff.load_state_dict(src_diff.state_dict())


def benchmark_naive_pytorch(N, drift_net, diff_net, true_sig, dW_tensor):
    """Run forward + backward for the naive PyTorch baseline. Returns (peak_mem_MB, elapsed_s)."""
    gc.collect()
    tracemalloc.start()

    y0 = torch.zeros(STATE_DIM, dtype=torch.float32, requires_grad=True)

    t0 = time.perf_counter()

    # Forward
    path = naive_euler_maruyama(y0, T, dW_tensor, drift_net, diff_net)  # (N+1, D)

    # Compute signature-based loss in PyTorch-compatible manner
    # We detach the path to numpy, compute sig + analytical gradient in C++,
    # but for the naive baseline we need the graph intact.
    # So we use a simple path-level proxy loss that preserves the full graph:
    # L = sum(path^2) — this is sufficient to measure memory scaling.
    loss = torch.sum(path ** 2)

    # Backward
    loss.backward()

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return peak / (1024 * 1024), elapsed  # MB, seconds


def benchmark_mactensor_adjoint(N, drift_net, diff_net, true_sig, dW_np):
    """Run forward + backward for our MacTensor C++ Adjoint solver. Returns (peak_mem_MB, elapsed_s)."""
    gc.collect()
    tracemalloc.start()

    X0 = np.zeros((STATE_DIM, 1), dtype=np.float32)
    params = list(drift_net.parameters()) + list(diff_net.parameters())

    t0 = time.perf_counter()

    # Forward via custom autograd (C++ Euler-Maruyama)
    sim_path_tensor = EulerMaruyamaSDE.apply(
        torch.tensor(X0, dtype=torch.float32, requires_grad=True), T,
        torch.tensor(dW_np, dtype=torch.float32),
        drift_net, diff_net, *params
    )

    # Compute signature-based loss with exact analytical backward
    sim_path_np = sim_path_tensor.detach().numpy()
    sim_ll = rough_sde.lead_lag_transform(sim_path_np)
    sim_sig = rough_sde.compute_signature(sim_ll, DEPTH)

    loss_val = float(np.sum((sim_sig - true_sig) ** 2))

    g_sig = (2.0 * (sim_sig - true_sig)).astype(np.float32)
    grad_ll = rough_sde.compute_signature_backward(sim_ll, g_sig, DEPTH)
    grad_output = rough_sde.lead_lag_transform_backward(sim_path_np, grad_ll)

    # Backward via custom C++ Adjoint solver
    sim_path_tensor.backward(torch.tensor(grad_output, dtype=torch.float32))

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return peak / (1024 * 1024), elapsed  # MB, seconds


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    print("=" * 60)
    print("  SYSTEMS BENCHMARK: O(1) Adjoint vs O(N) Autograd")
    print("=" * 60)

    # Generate a target signature (we use the same one for all N)
    true_path = rough_sde.generate_fbm(HURST, max(N_STEPS_LIST), T)
    true_values = true_path[:, 1].reshape(-1, 1).astype(np.float32)
    true_ll = rough_sde.lead_lag_transform(true_values)
    true_sig = rough_sde.compute_signature(true_ll, DEPTH)

    # Results storage
    naive_mem, naive_time = [], []
    adjoint_mem, adjoint_time = [], []

    # ── Warmup pass to eliminate JIT / import overhead ──
    print("\nWarmup pass...")
    _wd = NaiveDriftNet(STATE_DIM)
    _wdiff = NaiveDiffusionNet(STATE_DIM, NOISE_DIM)
    _wdW = torch.randn(10, NOISE_DIM)
    _wy0 = torch.zeros(STATE_DIM, requires_grad=True)
    _wp = naive_euler_maruyama(_wy0, T, _wdW, _wd, _wdiff)
    torch.sum(_wp ** 2).backward()
    del _wd, _wdiff, _wdW, _wy0, _wp

    _ad = DriftNet(STATE_DIM); _adiff = DiffusionNet(STATE_DIM, NOISE_DIM)
    _adW = np.random.normal(0, 0.1, size=(10, NOISE_DIM)).astype(np.float32)
    _aX0 = np.zeros((STATE_DIM, 1), dtype=np.float32)
    _ap = list(_ad.parameters()) + list(_adiff.parameters())
    _at = EulerMaruyamaSDE.apply(torch.tensor(_aX0, dtype=torch.float32, requires_grad=True), T, torch.tensor(_adW, dtype=torch.float32), _ad, _adiff, *_ap)
    _at.backward(torch.ones_like(_at))
    del _ad, _adiff, _adW, _aX0, _ap, _at
    gc.collect()
    print("Warmup complete.\n")

    for N in N_STEPS_LIST:
        print(f"── N = {N} steps ──")
        dt = T / N

        # Create networks with identical weights
        naive_drift = NaiveDriftNet(STATE_DIM)
        naive_diff = NaiveDiffusionNet(STATE_DIM, NOISE_DIM)
        adj_drift = DriftNet(STATE_DIM)
        adj_diff = DiffusionNet(STATE_DIM, NOISE_DIM)
        sync_weights(naive_drift, naive_diff, adj_drift, adj_diff)

        # Generate noise
        dW_np = np.random.normal(0, np.sqrt(dt), size=(N, NOISE_DIM)).astype(np.float32)
        dW_tensor = torch.tensor(dW_np, dtype=torch.float32)

        # Subsample the true signature for this N
        sub_path = rough_sde.generate_fbm(HURST, N, T)
        sub_values = sub_path[:, 1].reshape(-1, 1).astype(np.float32)
        sub_ll = rough_sde.lead_lag_transform(sub_values)
        sub_sig = rough_sde.compute_signature(sub_ll, DEPTH)

        # ── Naive PyTorch Baseline ──
        mem, t_elapsed = benchmark_naive_pytorch(N, naive_drift, naive_diff, sub_sig, dW_tensor)
        naive_mem.append(mem)
        naive_time.append(t_elapsed)
        print(f"  PyTorch Autograd  | Peak Mem: {mem:8.2f} MB | Time: {t_elapsed:.4f} s")

        # ── MacTensor C++ Adjoint ──
        mem, t_elapsed = benchmark_mactensor_adjoint(N, adj_drift, adj_diff, sub_sig, dW_np)
        adjoint_mem.append(mem)
        adjoint_time.append(t_elapsed)
        print(f"  MacTensor Adjoint | Peak Mem: {mem:8.2f} MB | Time: {t_elapsed:.4f} s")

    # ════════════════════════════════════════════════════════════
    # NeurIPS-Grade Visualization
    # ════════════════════════════════════════════════════════════
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.4,
        'lines.linewidth': 2.0,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'text.usetex': False,
    })

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))

    c_pytorch = '#e63946'    # Crimson
    c_adjoint = '#457b9d'    # Steel blue
    marker_kw = dict(markersize=5, markeredgewidth=0.8, markeredgecolor='white')

    # ── Panel (a): Memory Scaling ──
    axs[0].plot(N_STEPS_LIST, naive_mem, 'o-', color=c_pytorch,
                label=r'PyTorch Autograd $\mathcal{O}(N)$', **marker_kw)
    axs[0].plot(N_STEPS_LIST, adjoint_mem, 's-', color=c_adjoint,
                label=r'MacTensor Adjoint $\mathcal{O}(1)$', **marker_kw)
    axs[0].set_xlabel("Time Steps $N$")
    axs[0].set_ylabel("Peak Memory (MB)")
    axs[0].set_title("(a) Memory Scaling", fontweight='bold')
    axs[0].legend(loc='upper left', framealpha=0.9, edgecolor='none')
    axs[0].grid(True, alpha=0.3, linestyle=':')
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')

    # ── Panel (b): Runtime Scaling ──
    axs[1].plot(N_STEPS_LIST, naive_time, 'o-', color=c_pytorch,
                label=r'PyTorch Autograd', **marker_kw)
    axs[1].plot(N_STEPS_LIST, adjoint_time, 's-', color=c_adjoint,
                label=r'MacTensor Adjoint', **marker_kw)
    axs[1].set_xlabel("Time Steps $N$")
    axs[1].set_ylabel("Wall-Clock Time (s)")
    axs[1].set_title("(b) Runtime Scaling", fontweight='bold')
    axs[1].legend(loc='upper left', framealpha=0.9, edgecolor='none')
    axs[1].grid(True, alpha=0.3, linestyle=':')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(w_pad=3.0)
    output_file = os.path.join(os.path.dirname(__file__), "..", "memory_benchmark_results.pdf")
    plt.savefig(output_file, bbox_inches='tight', format='pdf')
    print(f"\nSaved: {os.path.abspath(output_file)}")

    # ── Summary Table ──
    print("\n" + "=" * 60)
    print(f"{'N':>6} | {'PyTorch Mem':>12} | {'Adjoint Mem':>12} | {'Ratio':>8}")
    print("-" * 60)
    for i, N in enumerate(N_STEPS_LIST):
        ratio = naive_mem[i] / adjoint_mem[i] if adjoint_mem[i] > 0 else float('inf')
        print(f"{N:>6} | {naive_mem[i]:>10.2f} MB | {adjoint_mem[i]:>10.2f} MB | {ratio:>7.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()

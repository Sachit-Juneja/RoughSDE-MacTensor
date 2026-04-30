import sys
import os
import torch
import torch.nn as nn
import numpy as np

# Ensure rough_sde is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build')))
import rough_sde

class EulerMaruyamaSDE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y0, T, W_increments, drift_net, diffusion_net, *params):
        # We need to flatten params to satisfy autograd
        ctx.drift_net = drift_net
        ctx.diffusion_net = diffusion_net
        ctx.T = T
        ctx.W_increments = W_increments
        
        y0_np = y0.detach().cpu().numpy()
        W_inc_np = W_increments.detach().cpu().numpy()
        
        def drift_wrapper(t, x):
            with torch.no_grad():
                # Allow taking float t and numpy x
                t_t = torch.tensor([t], dtype=torch.float32)
                x_t = torch.tensor(x, dtype=torch.float32).flatten()
                inp = torch.cat([t_t, x_t])
                out = drift_net.net(inp)
                return out.numpy().reshape(-1, 1)
                
        def diffusion_wrapper(t, x):
            with torch.no_grad():
                t_t = torch.tensor([t], dtype=torch.float32)
                x_t = torch.tensor(x, dtype=torch.float32).flatten()
                inp = torch.cat([t_t, x_t])
                out = diffusion_net.net(inp)
                return out.numpy().reshape(diffusion_net.state_dim, diffusion_net.noise_dim)
        
        # Run C++ Euler-Maruyama forward solver
        Z_seq = rough_sde.euler_maruyama_path(y0_np, T, W_inc_np, drift_wrapper, diffusion_wrapper)
        
        Z_tensor = torch.tensor(Z_seq, dtype=torch.float32)
        ctx.save_for_backward(Z_tensor)
        
        return Z_tensor

    @staticmethod
    def backward(ctx, grad_output):
        Z_seq, = ctx.saved_tensors
        T = ctx.T
        W_inc_np = ctx.W_increments.detach().cpu().numpy()
        
        # grad_output is (N+1, D)
        grad_output_np = grad_output.detach().cpu().numpy()
        
        drift_net = ctx.drift_net
        diffusion_net = ctx.diffusion_net
        
        def vjp_drift(t, z, a, dt):
            # We want VJP of mu(t, Z) * dt
            # a is adjoint state (D, 1)
            t_t = torch.tensor([t], dtype=torch.float32)
            z_t = torch.tensor(z, dtype=torch.float32, requires_grad=True).flatten()
            a_t = torch.tensor(a, dtype=torch.float32).flatten()
            
            with torch.enable_grad():
                inp = torch.cat([t_t, z_t])
                mu = drift_net.net(inp)
                mu_step = mu * dt
                
                # Compute VJP using autograd.grad
                params = list(drift_net.parameters())
                grads = torch.autograd.grad(
                    outputs=mu_step, 
                    inputs=[z_t] + params, 
                    grad_outputs=a_t,
                    retain_graph=False
                )
                
            vjp_z = grads[0].detach().numpy().reshape(-1, 1)
            vjp_theta = np.concatenate([g.detach().flatten().numpy() for g in grads[1:]]).reshape(-1, 1)
            
            return vjp_z, vjp_theta
            
        def vjp_diffusion(t, z, a, dW):
            t_t = torch.tensor([t], dtype=torch.float32)
            z_t = torch.tensor(z, dtype=torch.float32, requires_grad=True).flatten()
            a_t = torch.tensor(a, dtype=torch.float32).flatten()
            dW_t = torch.tensor(dW, dtype=torch.float32).flatten()
            
            with torch.enable_grad():
                inp = torch.cat([t_t, z_t])
                sigma = diffusion_net.net(inp).reshape(diffusion_net.state_dim, diffusion_net.noise_dim)
                # sigma * dW
                diff_step = torch.matmul(sigma, dW_t)
                
                params = list(diffusion_net.parameters())
                grads = torch.autograd.grad(
                    outputs=diff_step,
                    inputs=[z_t] + params,
                    grad_outputs=a_t,
                    retain_graph=False
                )
                
            vjp_z = grads[0].detach().numpy().reshape(-1, 1)
            vjp_theta = np.concatenate([g.detach().flatten().numpy() for g in grads[1:]]).reshape(-1, 1)
            
            return vjp_z, vjp_theta

        # Run C++ Adjoint Solver
        Z_seq_np = Z_seq.detach().cpu().numpy()
        a_0, a_theta_mu, a_theta_sigma = rough_sde.euler_maruyama_adjoint_path(
            Z_seq_np, grad_output_np, T, W_inc_np, vjp_drift, vjp_diffusion
        )
        
        # Map parameters back to their shapes
        grad_y0 = torch.tensor(a_0, dtype=torch.float32)
        grad_drift_params = []
        offset = 0
        for p in drift_net.parameters():
            numel = p.numel()
            g = torch.tensor(a_theta_mu[offset:offset+numel], dtype=torch.float32).view(p.shape)
            grad_drift_params.append(g)
            offset += numel
            
        grad_diff_params = []
        offset = 0
        for p in diffusion_net.parameters():
            numel = p.numel()
            g = torch.tensor(a_theta_sigma[offset:offset+numel], dtype=torch.float32).view(p.shape)
            grad_diff_params.append(g)
            offset += numel
            
        # Return gradients matching the forward arguments:
        # (y0, T, W_increments, drift_net, diffusion_net, *params)
        return (grad_y0, None, None, None, None, *grad_drift_params, *grad_diff_params)


class DriftNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=32):
        super().__init__()
        self.state_dim = state_dim
        self.net = nn.Sequential(
            nn.Linear(1 + state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, t, x):
        pass # Evaluated via custom autograd instead

class DiffusionNet(nn.Module):
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
        pass # Evaluated via custom autograd instead

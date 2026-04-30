import torch
import torch.nn as nn
import numpy as np

class DriftNet(nn.Module):
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
        # Allow both tensor and float/numpy inputs
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], dtype=torch.float32)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).flatten()
            
        inp = torch.cat([t, x])
        out = self.net(inp)
        
        # When called from C++, we need to return a numpy array
        if not torch.is_grad_enabled():
            return out.detach().numpy().reshape(-1, 1)
        return out.reshape(-1, 1)

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
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], dtype=torch.float32)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).flatten()
            
        inp = torch.cat([t, x])
        out = self.net(inp)
        
        if not torch.is_grad_enabled():
            return out.detach().numpy().reshape(self.state_dim, self.noise_dim)
        return out.reshape(self.state_dim, self.noise_dim)

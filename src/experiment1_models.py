# experiment1_models.py
import torch
import torch.nn as nn

class H_ev_Model(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.A = nn.Parameter(torch.tensor(1.0, device=device))
        self.B = nn.Parameter(torch.tensor(1.0, device=device))
        self.C = nn.Parameter(torch.tensor(0.1, device=device))
        self.D = nn.Parameter(torch.tensor(1.0, device=device))

    def forward(self, N, M):
        return self.A * torch.log(M) + self.B * (1 - torch.exp(-self.C * (N / M**self.D)))

class Sigma_ev_Model(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.A = nn.Parameter(torch.tensor(0.1, device=device))
        self.B = nn.Parameter(torch.tensor(0.1, device=device))
        self.C = nn.Parameter(torch.tensor(1.0, device=device))
        self.D = nn.Parameter(torch.tensor(0.1, device=device))
        self.E = nn.Parameter(torch.tensor(0.01, device=device))

    def forward(self, N, M):
        return self.A / (1 + self.B * (N / M**self.C)) + self.D * torch.exp(-self.E * M)

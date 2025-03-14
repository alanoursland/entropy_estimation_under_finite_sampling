# experiment1_models.py
import torch
import torch.nn as nn

class H_ev_Model1(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0, device=device))
        self.b = nn.Parameter(torch.tensor(0.0, device=device))

    def forward(self, N, M):
        return torch.log(M) * (1 - torch.exp(-self.a * (N / M))) + self.b

class H_ev_Model2(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0, device=device))
        self.b = nn.Parameter(torch.tensor(0.0, device=device))
        self.c = nn.Parameter(torch.tensor(0.0, device=device))

    def forward(self, N, M):
        return torch.log(M) * (1 - torch.exp(-self.a * (N / M))) + self.c * torch.log(N) + self.b

class Sigma_ev_Model1(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.1, device=device))
        self.b = nn.Parameter(torch.tensor(0.5, device=device))
        self.c = nn.Parameter(torch.tensor(0.01, device=device))

    def forward(self, N, M):
        return self.a * (N / M)**(-self.b) + self.c

class Sigma_ev_Model2(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.1, device=device))
        self.b = nn.Parameter(torch.tensor(0.1, device=device))
        self.c = nn.Parameter(torch.tensor(1.0, device=device))
        self.d = nn.Parameter(torch.tensor(0.1, device=device))
        self.e = nn.Parameter(torch.tensor(0.01, device=device))

    def forward(self, N, M):
        n_over_m = N/M
        return (self.a + self.b * n_over_m) / (self.c + self.d * n_over_m + self.e * n_over_m**2)

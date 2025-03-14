#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Histogram-based entropy estimation utilities.
This module provides core functions for entropy estimation using histograms with PyTorch GPU acceleration.
"""

import torch
import math
import os
from typing import Optional, Tuple, Dict, List, Union


def torch_histogram(data: torch.Tensor, bins: int, min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
    """
    Calculates a histogram of the input data using PyTorch.

    Args:
        data: A 1D PyTorch tensor containing the data.
        bins: The number of bins.
        min_val: The minimum value of the range.
        max_val: The maximum value of the range.

    Returns:
        A 1D PyTorch tensor containing the histogram counts.
    """
    bin_width = (max_val - min_val) / bins
    indices = torch.floor((data - min_val) / bin_width).long()
    # Clamp indices to be within the valid range [0, bins-1]
    indices = torch.clamp(indices, 0, bins - 1)
    counts = torch.bincount(indices, minlength=bins).float()
    return counts


def calculate_entropy(counts: torch.Tensor, N: int, eps: float = 1e-12) -> torch.Tensor:
    """
    Calculates the entropy from histogram counts.

    Args:
        counts: A 1D PyTorch tensor containing the histogram counts.
        N: The total number of samples.
        eps: Small constant to avoid log(0).

    Returns:
        The entropy value.
    """
    probs = counts / N
    # Avoid log(0) by replacing zeros with a small value
    probs = torch.where(probs == 0, torch.tensor(eps, device=probs.device), probs)
    return -torch.sum(probs * torch.log(probs))


def generate_uniform_samples(N: int, R: int = 1, device: str = None) -> torch.Tensor:
    """
    Generate uniform random samples on [0, 1].

    Args:
        N: Number of samples per dataset.
        R: Number of datasets (repetitions).
        device: Device to use ('cuda' or 'cpu'). If None, use CUDA if available.

    Returns:
        A torch.Tensor of shape (R, N) containing uniform random samples.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.rand((R, N), device=device)


def chao_shen_estimator(counts: torch.Tensor, N: int, eps: float = 1e-12) -> torch.Tensor:
    """
    Implements the Chao-Shen entropy estimator.

    Args:
        counts: A 1D PyTorch tensor containing histogram counts.
        N: The total number of samples.
        eps: Small constant to avoid log(0).

    Returns:
        The Chao-Shen corrected entropy value.
    """
    probs = counts / N
    singles = (counts == 1).sum().item()
    C = 1 - (singles / N)
    corrected_probs = probs * C
    corrected_probs = torch.where(corrected_probs == 0, torch.tensor(eps, device=probs.device), corrected_probs)
    return -torch.sum(corrected_probs * torch.log(corrected_probs))


def bootstrap_entropy_std(data: torch.Tensor, bins: int, bootstrap_samples: int = 100,
                         min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Estimate the standard deviation of entropy using bootstrap resampling.

    Args:
        data: Input data tensor.
        bins: Number of bins for histogram.
        bootstrap_samples: Number of bootstrap samples.
        min_val: Minimum value for histogram range.
        max_val: Maximum value for histogram range.

    Returns:
        Bootstrap standard deviation of entropy.
    """
    N = len(data)
    device = data.device
    entropies = []
    for _ in range(bootstrap_samples):
        indices = torch.randint(0, N, (N,), device=device)
        resampled_data = data[indices]
        counts = torch_histogram(resampled_data, bins=bins, min_val=min_val, max_val=max_val)
        H = calculate_entropy(counts, N)
        entropies.append(H) #Append tensor
    
    return torch.std(torch.stack(entropies)) #Stack and calculate std


def rice_rule(N: int) -> int:
    """
    Implements the Rice Rule for bin count selection.
    
    Args:
        N: Number of samples.
        
    Returns:
        Recommended number of bins.
    """
    return int(2 * (N ** 0.5))


def freedman_diaconis_rule(N: int, IQR: float = 0.5) -> int:
    """
    Implements the Freedman-Diaconis Rule for bin count selection.
    For uniform [0,1], IQR = 0.5. Adjust for other distributions.
    
    Args:
        N: Number of samples.
        IQR: Interquartile range of the data.
        
    Returns:
        Recommended number of bins.
    """
    # For range [0,1], max-min = 1
    bin_width = 2 * IQR / (N ** (1/3))
    return max(1, int(1 / bin_width))


def occupancy_based_bins(N: int, C: float = 5, max_bins: int = 1024) -> int:
    """
    Select number of bins based on ensuring minimum average bin occupancy.
    
    Args:
        N: Number of samples.
        C: Minimum average samples per bin.
        max_bins: Maximum number of bins to allow.
        
    Returns:
        Recommended number of bins.
    """
    return min(max_bins, int(N / C))
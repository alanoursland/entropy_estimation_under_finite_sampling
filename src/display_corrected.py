import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def compute_entropy_variance(p_j, H):
    return np.sum(p_j * (np.log(p_j) ** 2)) - H ** 2

def max_entropy(M):
    return np.log(M)  # Maximum entropy for M bins (uniform distribution)

def plot_entropy_vs_NM_ratio(M, data_dir='results/entropy_data', std_multiplier=1, device='cuda'):
    """
    Plots log(N/M) vs Entropy_max minus entropy curves on two side-by-side plots.

    Args:
        M: Fixed bin count (number of categories).
        data_dir: Directory where the results CSV is stored.
        std_multiplier: Multiple of entropy standard deviation to add.
        device: 'cuda' or 'cpu'.
    """
    # Load entropy data
    csv_file = os.path.join(data_dir, 'analysis_results.csv')
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Could not find analysis_results.csv in {data_dir}.")
    df = pd.read_csv(csv_file)
    df = df[df.iloc[:, 1] == M]  # Filter for fixed M value
    
    if df.empty:
        raise ValueError(f"No data found for M = {M}")
    
    # Extract values
    Ns = df.iloc[:, 0].values  # N values
    H_obs = df.iloc[:, 2].values  # Observed entropy values
    
    # Compute Miller-Madow correction
    H_mm = H_obs + (M - 1) / (2 * Ns)
    
    # Compute entropy variance and standard deviation
    entropy_variance = np.array([(1 / N) * compute_entropy_variance(np.ones(M) / M, H) for N, H in zip(Ns, H_obs)])
    entropy_stddev = np.sqrt(entropy_variance)
    
    # Compute max entropy
    H_max = max_entropy(M)
    
    # Compute differences from max entropy
    H_obs_diff = (H_max - H_obs)
    H_mm_diff = (H_max - H_mm)
    H_stddev_diff = (H_max - (H_obs + std_multiplier * entropy_stddev))
    
    # Compute N/M ratio
    NM_ratio = Ns / M
    
    # Plot side-by-side
    plt.figure(figsize=(12, 5))
    
    # Linear Y-axis plot
    plt.subplot(1, 2, 1)
    plt.plot(NM_ratio, H_obs_diff, marker='o', linestyle='--', label='Max Entropy - Observed', alpha=0.7)
    plt.plot(NM_ratio, H_mm_diff, marker='o', linestyle='-', label='Max Entropy - Miller-Madow Correction')
    plt.plot(NM_ratio, H_stddev_diff, marker='o', linestyle='-', label=f'Max Entropy - (Observed + {std_multiplier}σ)')
    plt.xscale('log')
    plt.xlabel('N / M (Expected Items per Bin)')
    plt.ylabel('Max Entropy - Entropy')
    plt.title(f'Log(N/M) vs Entropy Difference (Linear Y) for M={M}')
    plt.legend()
    plt.grid()
    
    # Log Y-axis plot
    plt.subplot(1, 2, 2)
    plt.plot(NM_ratio, abs(H_obs_diff), marker='o', linestyle='--', label='Max Entropy - Observed', alpha=0.7)
    plt.plot(NM_ratio, abs(H_mm_diff), marker='o', linestyle='-', label='Max Entropy - Miller-Madow Correction')
    plt.plot(NM_ratio, abs(H_stddev_diff), marker='o', linestyle='-', label=f'Max Entropy - (Observed + {std_multiplier}σ)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('N / M (Expected Items per Bin)')
    plt.ylabel('Max Entropy - Entropy')
    plt.title(f'Log(N/M) vs Entropy Difference (Log Y) for M={M}')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()

# Example usage
import sys
if __name__ == '__main__':
    M = 20  # Example fixed bin count, adjust as needed
    
    # Default value
    std_multiplier = 1
    
    # Check if command line argument is provided
    if len(sys.argv) > 1:
        try:
            # Try to convert the first argument to a float
            M = int(sys.argv[1])
        except ValueError:
            print(f"Error: '{sys.argv[1]}' is not a valid number. Using default M = 20.")
    
    plot_entropy_vs_NM_ratio(M, std_multiplier=std_multiplier)

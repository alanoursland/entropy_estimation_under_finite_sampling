# experiment1_analysis_chunked.py
import torch
import os
import pandas as pd
import numpy as np

def analyze_data_chunked(data_dir='results/entropy_data', device='cuda'):
    """
    Loads data from chunk files, calculates H_ev and sigma_ev, and saves to CSV.

    Args:
        data_dir: Directory where the data chunks are stored.
        device: 'cuda' or 'cpu'.
    """

    results = []
    chunk_files = [f for f in os.listdir(data_dir) if f.startswith('chunk_') and f.endswith('.pt')]

    for chunk_file in chunk_files:
        print(f"Loading {chunk_file}")
        chunk_data = torch.load(os.path.join(data_dir, chunk_file))
        for (N, M), H_obs_values in chunk_data.items():
            H_obs_values = H_obs_values.to(device)
            H_ev = torch.mean(H_obs_values)
            sigma_ev = torch.std(H_obs_values)
            results.append((N, M, H_ev.item(), sigma_ev.item()))

    df = pd.DataFrame(results, columns=['N', 'M', 'H_ev', 'sigma_ev'])
    csv_file = os.path.join(data_dir, 'analysis_results.csv')
    df.to_csv(csv_file, index=False)
    print(f"Analysis results saved to {csv_file}")

    return df

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = analyze_data_chunked(device=device)
    print(df)
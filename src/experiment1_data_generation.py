# experiment1_data_generation_chunked.py
import torch
import os
import math
import hist
import numpy as np

def generate_nm_values():
    Ns = []
    Ms = []

    # Logarithmic spacing for broad coverage (N)
    for n in [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]:
        Ns.append(n)

    # Logarithmic spacing for broad coverage (M)
    for m in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        Ms.append(m)

    # Linear spacing around the sweet spot (N/M = 10 to 100)
    target_ratios = [2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 250, 300]  # Ratios to target
    for n in [50, 100, 200, 500, 1000, 2000, 5000, 10000]:  # Base N values
        for ratio in target_ratios:
            m = round(n / ratio)
            if m >= 2 and m <= 2048: # Keep within reasonable bounds.
                Ns.append(n)
                Ms.append(m)

    # Ensure unique combinations
    nm_combinations = set(zip(Ns, Ms))
    Ns = []
    Ms = []
    for n, m in nm_combinations:
        Ns.append(n)
        Ms.append(m)
    return sorted(list(set(Ns))), sorted(list(set(Ms))) # Remove duplicates


def generate_and_store_data_chunked(Ns, Ms, R, chunk_size, data_dir='results/entropy_data', device='cuda'):
    """
    Generates data for Experiment 1, calculates histograms, and stores the
    observed entropies in chunks.

    Args:
        Ns: List of sample sizes (N).
        Ms: List of bin counts (M).
        R: Number of repetitions for each (N, M) pair.
        chunk_size: Number of (N, M) pairs to process before saving to disk.
        data_dir: Directory to store the data.
        device: 'cuda' or 'cpu'.
    """

    os.makedirs(data_dir, exist_ok=True)
    chunk_num = 0
    data_chunk = {}

    for i, N in enumerate(Ns):
        for j, M in enumerate(Ms):
            print(f"Generating data for N={N}, M={M}")
            H_obs_values = []
            for _ in range(R):
                data = hist.generate_uniform_samples(N, R=1, device=device).squeeze()
                counts = hist.torch_histogram(data, bins=M)
                H_obs = hist.calculate_entropy(counts, N)
                H_obs_values.append(H_obs)

            H_obs_values = torch.stack(H_obs_values)
            data_chunk[(N, M)] = H_obs_values

            # Check if it's time to save a chunk
            if len(data_chunk) >= chunk_size:
                chunk_file = os.path.join(data_dir, f'chunk_{chunk_num}.pt')
                torch.save(data_chunk, chunk_file)
                print(f"  Chunk {chunk_num} saved to {chunk_file}")
                data_chunk = {}  # Reset the chunk
                chunk_num += 1

    # Save any remaining data
    if data_chunk:
        chunk_file = os.path.join(data_dir, f'chunk_{chunk_num}.pt')
        torch.save(data_chunk, chunk_file)
        print(f"  Final chunk {chunk_num} saved to {chunk_file}")


if __name__ == '__main__':
    torch.manual_seed(42)  # for reproducibility

    Ns, Ms = generate_nm_values()
    R = 10000  # Number of repetitions
    chunk_size = 100  # Number of (N, M) pairs per chunk
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    generate_and_store_data_chunked(Ns, Ms, R, chunk_size, device=device)
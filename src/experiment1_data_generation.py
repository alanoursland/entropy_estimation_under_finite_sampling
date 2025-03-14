# experiment1_data_generation.py
import torch
import os
import math
import hist  # Import the functions from hist.py


def generate_and_store_data(Ns, Ms, R, data_dir='results/entropy_data', device='cuda'):
    """
    Generates data for Experiment 1, calculates histograms, and stores the
    observed entropies.

    Args:
        Ns: List of sample sizes (N).
        Ms: List of bin counts (M).
        R: Number of repetitions for each (N, M) pair.
        data_dir: Directory to store the data.
        device: 'cuda' or 'cpu'.
    """

    os.makedirs(data_dir, exist_ok=True)

    all_data = {}

    for N in Ns:
        for M in Ms:
            print(f"Generating data for N={N}, M={M}")
            H_obs_values = []
            for _ in range(R):
                data = hist.generate_uniform_samples(N, R=1, device=device).squeeze()  # (1, N) -> (N,)
                counts = hist.torch_histogram(data, bins=M)
                H_obs = hist.calculate_entropy(counts, N)
                H_obs_values.append(H_obs)

            H_obs_values = torch.stack(H_obs_values)
            all_data[(N, M)] = H_obs_values

    torch.save(all_data, os.path.join(data_dir, 'all_data.pt'))
    print(f"Data saved to {os.path.join(data_dir, 'all_data.pt')}")


if __name__ == '__main__':
    torch.manual_seed(42)  # for reproducibility

    Ns = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    Ms = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    R = 1000  # Number of repetitions

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    generate_and_store_data(Ns, Ms, R, device=device)
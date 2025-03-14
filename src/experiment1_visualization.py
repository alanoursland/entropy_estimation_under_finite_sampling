# experiment1_visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch

def visualize_results(data_dir='results/entropy_data'):
    """
    Loads the analysis results from a CSV file and generates plots.

    Args:
        data_dir: Directory where the data is stored.
    """

    csv_file = os.path.join(data_dir, 'analysis_results.csv')
    df = pd.read_csv(csv_file)

    # Convert DataFrame to PyTorch tensors (for consistency with later code)
    results = torch.tensor(df.values)
    Ns = torch.unique(results[:, 0]).cpu().numpy()  # Get unique N values
    Ms = torch.unique(results[:, 1]).cpu().numpy()  # Get unique M values

    # --- H_ev Plots ---
    plt.figure(figsize=(12, 6))

    # H_ev vs. N for different M
    plt.subplot(1, 2, 1)
    for M in Ms:
        subset = df[df['M'] == M]
        plt.plot(subset['N'], subset['H_ev'], label=f'M={int(M)}')
    plt.xscale('log')
    plt.xlabel('N (Sample Size)')
    plt.ylabel('H_ev (Expected Entropy)')
    plt.title('H_ev vs. N for different M')
    plt.legend()
    plt.grid(True)

    # H_ev vs. M for different N
    plt.subplot(1, 2, 2)
    for N in Ns:
        subset = df[df['N'] == N]
        plt.plot(subset['M'], subset['H_ev'], label=f'N={int(N)}')
    plt.xscale('log')
    plt.xlabel('M (Number of Bins)')
    plt.ylabel('H_ev (Expected Entropy)')
    plt.title('H_ev vs. M for different N')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'H_ev_plots.png'))
    plt.show()


    # H_ev / H_max vs. N/M
    plt.figure(figsize=(8, 6))
    plt.plot(results[:,0].cpu()/results[:,1].cpu(), results[:,2].cpu()/torch.log(results[:,1]).cpu(), 'o')
    plt.xscale('log')
    plt.xlabel('N/M')
    plt.ylabel('H_ev / H_max')
    plt.title('H_ev / H_max vs. N/M')
    plt.grid(True)
    plt.savefig(os.path.join(data_dir, 'H_ev_over_Hmax.png'))
    plt.show()


    # --- sigma_ev Plots ---
    plt.figure(figsize=(12, 6))

    # sigma_ev vs. N for different M
    plt.subplot(1, 2, 1)
    for M in Ms:
        subset = df[df['M'] == M]
        plt.plot(subset['N'], subset['sigma_ev'], label=f'M={int(M)}')
    plt.xscale('log')
    plt.xlabel('N (Sample Size)')
    plt.ylabel('sigma_ev (Standard Deviation of Entropy)')
    plt.title('sigma_ev vs. N for different M')
    plt.legend()
    plt.grid(True)

    # sigma_ev vs. M for different N
    plt.subplot(1, 2, 2)
    for N in Ns:
        subset = df[df['N'] == N]
        plt.plot(subset['M'], subset['sigma_ev'], label=f'N={int(N)}')
    plt.xscale('log')
    plt.xlabel('M (Number of Bins)')
    plt.ylabel('sigma_ev (Standard Deviation of Entropy)')
    plt.title('sigma_ev vs. M for different N')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'sigma_ev_plots.png'))
    plt.show()


    # sigma_ev vs. N/M
    plt.figure(figsize=(8, 6))
    plt.plot(results[:,0].cpu()/results[:,1].cpu(), results[:,3].cpu(), 'o')
    plt.xscale('log')
    plt.xlabel('N/M')
    plt.ylabel('sigma_ev')
    plt.title('sigma_ev vs. N/M')
    plt.grid(True)
    plt.savefig(os.path.join(data_dir, 'sigma_ev_vs_NM.png'))
    plt.show()


    # sigma_ev / H_ev vs. N/M
    plt.figure(figsize=(8, 6))
    plt.plot(results[:,0].cpu()/results[:,1].cpu(), results[:,3].cpu()/results[:,2].cpu(), 'o')
    plt.xscale('log')
    plt.xlabel('N/M')
    plt.ylabel('sigma_ev / H_ev')
    plt.title('sigma_ev / H_ev vs. N/M')
    plt.grid(True)
    plt.savefig(os.path.join(data_dir, 'sigma_ev_over_H_ev.png'))
    plt.show()


if __name__ == '__main__':
    visualize_results()
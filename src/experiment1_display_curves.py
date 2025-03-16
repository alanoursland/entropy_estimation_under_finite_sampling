# experiment1_display_curves.py
import torch
import torch.nn as nn
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# from experiment1_models import H_ev_Model1, H_ev_Model2, Sigma_ev_Model1, Sigma_ev_Model2
from experiment1_models import H_ev_Model, Sigma_ev_Model

# Assuming hist.py is in the same directory (or in your PYTHONPATH)
# If not, you'll need to adjust the import statement.

def display_curves(data_dir='results/entropy_data', device='cuda'):
    """
    Loads the trained models and data, and generates the result plots.

    Args:
        data_dir: Directory where the models and data are stored.
        device: 'cuda' or 'cpu'.
    """

    # --- 1. Load Data ---
    csv_file = os.path.join(data_dir, 'analysis_results.csv')
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Could not find analysis_results.csv in {data_dir}.  Make sure to run experiment1_analysis.py first.")
    df = pd.read_csv(csv_file)
    results = torch.tensor(df.values, device=device, dtype=torch.float32)
    Ns = torch.unique(results[:, 0]).cpu().numpy()
    Ms = torch.unique(results[:, 1]).cpu().numpy()

    # --- 2. Define Model Classes (from experiment1_models.py) ---

    # --- 3. Load Models ---
    # h_ev_model1 = H_ev_Model1(device)
    # h_ev_model2 = H_ev_Model2(device)
    # sigma_ev_model1 = Sigma_ev_Model1(device)
    # sigma_ev_model2 = Sigma_ev_Model2(device)
    h_ev_model = H_ev_Model(device)
    sigma_ev_model = Sigma_ev_Model(device)

    try:
        # h_ev_model1.load_state_dict(torch.load(os.path.join(data_dir, 'model_h_ev_1.pt')))
        # h_ev_model2.load_state_dict(torch.load(os.path.join(data_dir, 'model_h_ev_2.pt')))
        # sigma_ev_model1.load_state_dict(torch.load(os.path.join(data_dir, 'model_sigma_ev_1.pt')))
        # sigma_ev_model2.load_state_dict(torch.load(os.path.join(data_dir, 'model_sigma_ev_2.pt')))
        h_ev_model.load_state_dict(torch.load(os.path.join(data_dir, 'model_h_ev.pt')))
        sigma_ev_model.load_state_dict(torch.load(os.path.join(data_dir, 'model_sigma_ev.pt')))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find model .pt files in {data_dir}. Make sure to run experiment1_curve_fitting.py first.") from e


    # --- 4. Visualization ---

    def plot_results(model, results, model_name, Ns, Ms):
      """Plots actual vs. predicted values and residuals."""

      N_values = results[:, 0].float().cpu()
      M_values = results[:, 1].float().cpu()
      NM_values = (N_values / M_values).cpu().numpy()

      if "H_ev" in model_name:
          actual_values = results[:, 2].float().cpu().numpy()
          predicted_values = model(results[:, 0].float(), results[:, 1].float()).detach().cpu().numpy()
          ylabel = 'H_ev'
      elif "sigma" in model_name:
          actual_values = results[:, 3].float().cpu().numpy()
          predicted_values = model(results[:, 0].float(), results[:, 1].float()).detach().cpu().numpy()
          ylabel = 'sigma_ev'
      else:
          raise ValueError("Invalid model_name")

      residuals = actual_values - predicted_values

      # --- Actual vs. Predicted Plot ---
      plt.figure(figsize=(12, 6))
      plt.subplot(1, 2, 1)

      # Group by M ranges for better visualization
      M_ranges = np.quantile(Ms, [0, 0.25, 0.5, 0.75, 1])
      colors = ['r', 'g', 'b', 'k']
      markers = ['o', 's', '^', 'D']

      for i in range(len(M_ranges) - 1):
          mask = (M_values >= M_ranges[i]) & (M_values < M_ranges[i + 1])
          plt.scatter(NM_values[mask], actual_values[mask], c=colors[i], marker=markers[i],
                      label=f'{M_ranges[i]:.0f} ≤ M < {M_ranges[i+1]:.0f}', alpha=0.5)
      plt.scatter(NM_values, predicted_values, c='cyan', marker='x', label='Predicted', alpha=0.5) #Plot all predicted

      plt.xscale('log')
      plt.xlabel('N / M')
      plt.ylabel(ylabel)
      plt.title(f'{model_name}: Actual vs. Predicted')
      plt.legend()
      plt.grid(True)

      # --- Residual Plot ---
      plt.subplot(1, 2, 2)
      for i in range(len(M_ranges) - 1):
          mask = (M_values >= M_ranges[i]) & (M_values < M_ranges[i+1])
          plt.scatter(NM_values[mask], residuals[mask], c=colors[i], marker=markers[i], alpha=0.5)

      plt.xscale('log')
      plt.xlabel('N / M')
      plt.ylabel('Residuals')
      plt.title(f'{model_name}: Residuals')
      plt.axhline(0, color='k', linestyle='--')
      plt.grid(True)

      plt.tight_layout()
      plt.savefig(os.path.join(data_dir, f'prediction_{model_name}.png')) #Save the figure
      plt.show()


    # Call the plotting function for each model
    # plot_results(h_ev_model1, results, "H_ev1", Ns, Ms)
    # plot_results(h_ev_model2, results, "H_ev2", Ns, Ms)
    # plot_results(sigma_ev_model1, results, "sigma_ev1", Ns, Ms)
    # plot_results(sigma_ev_model2, results, "sigma_ev2", Ns, Ms)
    plot_results(h_ev_model, results, "H_ev", Ns, Ms)
    plot_results(sigma_ev_model, results, "sigma_ev", Ns, Ms)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    display_curves(device=device)
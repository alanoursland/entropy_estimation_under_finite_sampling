# experiment1_curve_fitting.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

from putils import Timer
from experiment1_models import H_ev_Model1, H_ev_Model2, Sigma_ev_Model1, Sigma_ev_Model2

def curve_fitting(data_dir='results/entropy_data', device='cuda'):
    """
    Performs curve fitting for H_ev(N, M) and sigma_ev(N, M).

    Args:
        data_dir: Directory where the data is stored.
        device: 'cuda' or 'cpu'.
    """

    # --- 1. Load Data ---
    csv_file = os.path.join(data_dir, 'analysis_results.csv')
    df = pd.read_csv(csv_file)
    print(f"Dataset size: {len(df)}")
    results = torch.tensor(df.values, device=device, dtype=torch.float32)
    Ns = torch.unique(results[:, 0]).cpu().numpy()
    Ms = torch.unique(results[:, 1]).cpu().numpy()

    # --- 2. Define Model Classes (from experiment1_models.py) ---

    # --- 3. Training Loop ---

    def train_model(model, results, optimizer, loss_fn, epochs=10000, model_name=""):
        timer = Timer()

        # Extract N, M, H_ev, and sigma_ev values into separate tensors *before* the loop
        N_values = results[:, 0].float()
        M_values = results[:, 1].float()

        if "H_ev" in model_name:
            targets = results[:, 2].float()
        elif "sigma" in model_name:
            targets = results[:, 3].float()

        losses = []  # Store losses for plotting
        for epoch in range(epochs):
            optimizer.zero_grad()  # Zero the gradients
            # Forward pass:  Calculate predictions for *all* data points
            predictions = model(N_values, M_values)

            # Compute the loss over the *entire* dataset
            loss = loss_fn(predictions, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track loss (convert to Python float for consistency)
            losses.append(loss.item())

            if (epoch + 1) % 1000 == 0:
                print(f'Model: {model_name} Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f} ({timer.tick():.2f})')
        print()
        return losses

    # Instantiate models, optimizers, and loss function
    h_ev_model1 = H_ev_Model1(device)
    h_ev_model2 = H_ev_Model2(device)
    sigma_ev_model1 = Sigma_ev_Model1(device)
    sigma_ev_model2 = Sigma_ev_Model2(device)

    optimizer_H1 = optim.Adam(h_ev_model1.parameters(), lr=0.01)
    optimizer_H2 = optim.Adam(h_ev_model2.parameters(), lr=0.01)
    optimizer_sigma1 = optim.Adam(sigma_ev_model1.parameters(), lr=0.01)
    optimizer_sigma2 = optim.Adam(sigma_ev_model2.parameters(), lr=0.01)

    loss_fn = nn.MSELoss()

    # Train models and store losses
    losses = {}
    losses["H_ev1"] = train_model(h_ev_model1, results, optimizer_H1, loss_fn, model_name = "H_ev1")
    losses["H_ev2"] = train_model(h_ev_model2, results, optimizer_H2, loss_fn, model_name = "H_ev2")
    losses["sigma_ev1"] = train_model(sigma_ev_model1, results, optimizer_sigma1, loss_fn, model_name="sigma_ev1")
    losses["sigma_ev2"] = train_model(sigma_ev_model2, results, optimizer_sigma2, loss_fn, model_name="sigma_ev2")

    # --- 4. Model Evaluation ---
    def calculate_r_squared(model, results, model_name):
        """
        Calculates the R-squared value using PyTorch.

        Args:
            model: The trained PyTorch model.
            results: A PyTorch tensor containing the (N, M, H_ev, sigma_ev) data.
            model_name: A string indicating the model type ("H_ev" or "sigma_ev").

        Returns:
            The R-squared value as a Python float.
        """
        device = next(model.parameters()).device  # Get the device the model is on

        # Extract N, M, and target values into separate tensors *before* the loop
        N_values = results[:, 0].float()
        M_values = results[:, 1].float()

        if "H_ev" in model_name:
            y_true = results[:, 2].float()
        elif "sigma" in model_name:
            y_true = results[:, 3].float()
        else:
            raise ValueError("Invalid model_name. Must be 'H_ev' or 'sigma_ev'.")

        # Calculate predictions for *all* data points in a single forward pass
        y_pred = model(N_values, M_values)

        # Calculate R-squared using PyTorch operations
        ss_res = torch.sum((y_true - y_pred) ** 2)  # Residual sum of squares
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)  # Total sum of squares
        r2 = 1 - (ss_res / ss_tot)

        return r2.item()  # Return as a Python float

    print(f"R-squared (H_ev_Model1): {calculate_r_squared(h_ev_model1, results, 'H_ev1'):.4f}")
    print(f"R-squared (H_ev_Model2): {calculate_r_squared(h_ev_model2, results, 'H_ev2'):.4f}")
    print(f"R-squared (Sigma_ev_Model1): {calculate_r_squared(sigma_ev_model1, results, 'sigma_ev1'):.4f}")
    print(f"R-squared (Sigma_ev_Model2): {calculate_r_squared(sigma_ev_model2, results, 'sigma_ev2'):.4f}")

     # --- 5. Visualize ---
    # def plot_results(model, results, model_name, Ns, Ms):
    #     """Plots actual vs. predicted values and residuals."""

    #     N_values = results[:, 0].float().cpu()
    #     M_values = results[:, 1].float().cpu()
    #     NM_values = (N_values / M_values).cpu().numpy()

    #     if "H_ev" in model_name:
    #         actual_values = results[:, 2].float().cpu().numpy()
    #         predicted_values = model(results[:, 0].float(), results[:, 1].float()).detach().cpu().numpy()
    #         ylabel = 'H_ev'
    #     elif "sigma" in model_name:
    #         actual_values = results[:, 3].float().cpu().numpy()
    #         predicted_values = model(results[:, 0].float(), results[:, 1].float()).detach().cpu().numpy()
    #         ylabel = 'sigma_ev'
    #     else:
    #         raise ValueError("Invalid model_name")

    #     residuals = actual_values - predicted_values

    #     # --- Actual vs. Predicted Plot ---
    #     plt.figure(figsize=(12, 6))
    #     plt.subplot(1, 2, 1)  # One row, two columns, first plot (actual vs. predicted)

    #     # Group by M ranges for better visualization
    #     M_ranges = np.quantile(Ms, [0, 0.25, 0.5, 0.75, 1])  # Example quantiles
    #     colors = ['r', 'g', 'b', 'k']
    #     markers = ['o', 's', '^', 'D']

    #     for i in range(len(M_ranges) - 1):
    #         mask = (M_values >= M_ranges[i]) & (M_values < M_ranges[i + 1])
    #         plt.scatter(NM_values[mask], actual_values[mask], c=colors[i], marker=markers[i],
    #                     label=f'{M_ranges[i]:.0f} ≤ M < {M_ranges[i+1]:.0f}', alpha=0.5)
    #     plt.scatter(NM_values, predicted_values, c='cyan', marker='x', label='Predicted', alpha=0.5)

    #     plt.xscale('log')
    #     plt.xlabel('N / M')
    #     plt.ylabel(ylabel)
    #     plt.title(f'{model_name}: Actual vs. Predicted')
    #     plt.legend()
    #     plt.grid(True)

    #     # --- Residual Plot ---
    #     plt.subplot(1, 2, 2) # One row, two columns, second plot (residuals)
    #     for i in range(len(M_ranges) - 1):
    #         mask = (M_values >= M_ranges[i]) & (M_values < M_ranges[i+1])
    #         plt.scatter(NM_values[mask], residuals[mask], c=colors[i], marker=markers[i], alpha=0.5)

    #     plt.xscale('log')
    #     plt.xlabel('N / M')
    #     plt.ylabel('Residuals')
    #     plt.title(f'{model_name}: Residuals')
    #     plt.axhline(0, color='k', linestyle='--')  # Add a horizontal line at zero
    #     plt.grid(True)

    #     plt.tight_layout()
    #     plt.show()

    # # Call the plotting function for each model
    # plot_results(h_ev_model1, results, "H_ev1", Ns, Ms)
    # plot_results(h_ev_model2, results, "H_ev2", Ns, Ms)
    # plot_results(sigma_ev_model1, results, "sigma_ev1", Ns, Ms)
    # plot_results(sigma_ev_model2, results, "sigma_ev2", Ns, Ms)



    # --- 6. Save Models ---
    torch.save(h_ev_model1.state_dict(), os.path.join(data_dir, 'model_h_ev_1.pt'))
    torch.save(h_ev_model2.state_dict(), os.path.join(data_dir, 'model_h_ev_2.pt'))
    torch.save(sigma_ev_model1.state_dict(), os.path.join(data_dir, 'model_sigma_ev_1.pt'))
    torch.save(sigma_ev_model2.state_dict(), os.path.join(data_dir, 'model_sigma_ev_2.pt'))
    print(f"Models saved to {data_dir}")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    curve_fitting(device=device)
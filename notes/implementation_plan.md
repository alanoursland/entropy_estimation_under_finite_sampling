# Implementation Plan: Entropy Estimation Experiments (PyTorch GPU)

This document outlines the implementation plan for the entropy estimation experiments, leveraging PyTorch for GPU acceleration.

## General Notes

*   **PyTorch Tensors:** All data and calculations will use PyTorch tensors on the GPU.
*   **Vectorized Operations:**  We will utilize PyTorch's vectorized operations to avoid explicit loops.
*   **GPU Utilization:**  `torch.cuda.is_available()` will be used to check for and utilize the GPU. `torch.cuda.synchronize()` will be used for accurate timing.
*   **Data Generation:**  `torch.rand()` will generate uniform random numbers on the GPU.
*   **Histogram Calculation:** A custom, vectorized PyTorch histogram function will be used (implementation provided below).
*   **Function Fitting:** `torch.optim` and custom loss functions (with automatic differentiation) will be used for curve fitting.
*   **Reproducibility:** `torch.manual_seed` will be used.

## PyTorch Histogram Function

```python
import torch

def torch_histogram(data, bins, min_val=0.0, max_val=1.0):
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
```

## Experiment 1: Deriving `H_ev(N, M)` and `σ_ev(N, M)`

**Goal:** Empirically derive functions for the expected entropy, `H_ev(N, M)`, and its standard deviation, `σ_ev(N, M)`, as functions of sample size (`N`) and bin count (`M`), when sampling from a uniform distribution on [0, 1]. These functions will be used in subsequent experiments for bias correction, adaptive binning, and model comparison.

**Hypothesis:**

*   `H_ev(N, M)` will be less than `H_max = log(M)` and will increase with `N` for a fixed `M`, approaching `log(M)` asymptotically.
*   `σ_ev(N, M)` will decrease with `N` and increase with `M`.
*   There will be a strong relationship between `σ_ev`, `N`, and `M`, likely involving the ratio `N/M`.

**Procedure:**

1.  **Setup (Define Parameters):**

    ```python
    import torch
    import math
    import os

    torch.manual_seed(42)  # for reproducibility

    Ns = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]  # Log-spaced N
    Ms = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]  # Log-spaced M
    R = 1000  # Number of repetitions for each (N, M) pair

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create a directory to store the data
    data_dir = 'entropy_data'
    os.makedirs(data_dir, exist_ok=True)
    ```

2.  **Data Generation and Histogram Calculation (GPU Accelerated):**

    ```python
    # Pre-allocate a dictionary to store H_obs values
    all_data = {}

    for N in Ns:
        for M in Ms:
            H_obs_values = []
            for _ in range(R):
                data = torch.rand(N, device=device)  # Generate data directly on the GPU
                counts = torch_histogram(data, bins=M)  # Use custom PyTorch histogram
                probs = counts / N
                # Avoid log(0)
                probs = torch.where(probs == 0, torch.tensor(1e-12, device=device), probs)
                H_obs = -torch.sum(probs * torch.log(probs))
                H_obs_values.append(H_obs) # Append

            H_obs_values = torch.stack(H_obs_values) # Convert list of tensors to tensor.
            all_data[(N, M)] = H_obs_values # Store

    # Save the data
    torch.save(all_data, os.path.join(data_dir, 'all_data.pt'))

    ```

3.  **Data Loading and Preparation:**

    ```python
    import torch
    import os

    data_dir = 'entropy_data'
    loaded_data = torch.load(os.path.join(data_dir, 'all_data.pt'))

    results = []
    # Calculate H_ev and sigma_ev from the loaded data:
    for (N, M), H_obs_values in loaded_data.items():
        H_ev = torch.mean(H_obs_values)
        sigma_ev = torch.std(H_obs_values)
        results.append((N, M, H_ev.item(), sigma_ev.item()))

    results = torch.tensor(results, device='cuda' if torch.cuda.is_available() else 'cpu')
    ```

4.  **Curve Fitting (Using PyTorch):**

    *   Define parametric model classes for `H_ev(N, M)` and `σ_ev(N, M)` using `torch.nn.Module`.  Start with simple, interpretable functions based on theoretical expectations (see previous discussion for candidate functions).

        ```python
        class H_ev_Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Parameter(torch.tensor(1.0, device=device))
                self.b = torch.nn.Parameter(torch.tensor(0.1, device=device))
                self.c = torch.nn.Parameter(torch.tensor(0.01, device=device))

            def forward(self, N, M):
                # Example function - adjust as needed!
                return torch.log(M) * (1 - torch.exp(-self.a * (N / M))) + self.b*torch.log(N) + self.c

        class Sigma_ev_Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Parameter(torch.tensor(1.0, device=device))
                self.b = torch.nn.Parameter(torch.tensor(0.1, device=device))

            def forward(self, N, M):
                return self.a / (torch.sqrt(N) + self.b*M)

        h_ev_model = H_ev_Model().to(device)
        sigma_ev_model = Sigma_ev_Model().to(device)

        optimizer_H = torch.optim.Adam(h_ev_model.parameters(), lr=0.01)
        optimizer_sigma = torch.optim.Adam(sigma_ev_model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()

        # Training loops (adjust epochs as needed)
        epochs = 10000
        for epoch in range(epochs):
          for N, M, H_ev, _ in results:
              N_tensor = torch.tensor(N, device=device, dtype=torch.float32)
              M_tensor = torch.tensor(M, device=device, dtype=torch.float32)
              H_ev_tensor = torch.tensor(H_ev, device=device, dtype=torch.float32)

              optimizer_H.zero_grad()
              H_ev_pred = h_ev_model(N_tensor, M_tensor)
              loss = loss_fn(H_ev_pred, H_ev_tensor)
              loss.backward()
              optimizer_H.step()

        for epoch in range(epochs):
            for N, M, _, sigma_ev in results:
                N_tensor = torch.tensor(N, device=device, dtype=torch.float32)
                M_tensor = torch.tensor(M, device=device, dtype=torch.float32)
                sigma_ev_tensor = torch.tensor(sigma_ev, device=device, dtype=torch.float32)

                optimizer_sigma.zero_grad()
                sigma_ev_pred = sigma_ev_model(N_tensor, M_tensor)
                loss = loss_fn(sigma_ev_pred, sigma_ev_tensor)
                loss.backward()
                optimizer_sigma.step()
        ```

    *   Use an appropriate optimizer (e.g., `torch.optim.Adam`) and loss function (e.g., `torch.nn.MSELoss`).
    *   Train the models using the `results` tensor (which contains the `N`, `M`, `H_ev`, and `σ_ev` values).

5.  **Visualization:**

    *   Use `matplotlib` to create the following plots (move tensors to CPU with `.cpu()` before plotting):
        *   `H_ev(N, M)` vs. `N` for different `M` values (both empirical data and fitted function).
        *   `H_ev(N, M)` vs. `M` for different `N` values (both empirical data and fitted function).
        *   `H_ev(N, M) / H_max` vs. `N/M` (both empirical data and fitted function).
        *   `σ_ev(N, M)` vs. `N` for different `M` values (both empirical data and fitted function).
        *   `σ_ev(N, M)` vs. `M` for different `N` values (both empirical data and fitted function).
        *   `σ_ev(N, M)` vs. `N/M` (both empirical data and fitted function).
        *   `σ_ev(N, M) / H_ev(N, M)` vs. `N/M` (both empirical data and fitted function).
        *   Residual plots (predicted - actual) for both `H_ev` and `σ_ev`.

6.  **Evaluation of Fit:**

    *   Calculate R-squared values for both fitted functions.
    *   Analyze the residual plots for any systematic patterns.
    *   Visually inspect the plots to assess the goodness of fit.

**Metrics:**

*   Goodness of fit (R-squared, residual analysis) for the `H_ev(N, M)` and `σ_ev(N, M)` functions.
*   Visual confirmation of expected trends in the plots.

**Failure Criteria:**

*   Poor fit of the chosen functions to the empirical data (low R-squared, systematic patterns in residuals).
*   Inability to find a clear and consistent relationship between `σ_ev`, `N`, and `M`.
*   Fitted functions that violate theoretical constraints (e.g., `H_ev > log(M)`).

**Deliverables:**

*   Fitted functions for `H_ev(N, M)` and `σ_ev(N, M)` (including the PyTorch model definitions and the learned parameter values).
*   Plots visualizing the empirical data, the fitted functions, and the residuals.
*   R-squared values and other goodness-of-fit statistics.
*   Analysis of the relationships between `H_ev`, `σ_ev`, `N`, `M`, and `N/M`, based on the fitted functions and the plots.
*   Justification for the chosen functional forms.

This revised Experiment 1 plan includes data storage and retrieval, uses PyTorch for all calculations, and provides a detailed procedure for deriving and evaluating the `H_ev(N, M)` and `σ_ev(N, M)` functions. It also includes clear deliverables and failure criteria. The use of a dictionary to store the results before saving them to disk is a significant improvement for organization and flexibility.

## Experiment 2: "Sweet Spot" Analysis and `M = f(N)` Derivation

1.  **Use Fitted Functions:** Use the fitted `H_ev(N, M)` and `σ_ev(N, M)` models from Experiment 1.

2.  **Define Criterion:** Choose a criterion (e.g., minimize `σ_ev / H_ev`).

3.  **Derive `M = f(N)`:**

    ```python
    def find_best_M(N, criterion_fn, M_range, h_ev_model, sigma_ev_model):
      best_M = None
      best_criterion_val = float('inf')

      for M in M_range:
        M_tensor = torch.tensor(M, device=device, dtype=torch.float32)
        N_tensor = torch.tensor(N, device=device, dtype=torch.float32)
        H_ev = h_ev_model(N_tensor, M_tensor)
        sigma_ev = sigma_ev_model(N_tensor, M_tensor)
        criterion_val = criterion_fn(H_ev, sigma_ev)

        if criterion_val < best_criterion_val:
          best_criterion_val = criterion_val
          best_M = M

      return best_M

    # Example criterion: minimize relative standard deviation
    def criterion(H_ev, sigma_ev):
      return sigma_ev / H_ev

    M_range = torch.arange(2, 1025, device=device)  # Example M range
    best_Ms = []
    for N in Ns:
      best_M = find_best_M(N, criterion, M_range, h_ev_model, sigma_ev_model)
      best_Ms.append((N, best_M))
      print(f"N={N}, Best M={best_M}") #Output to console

    best_Ms = torch.tensor(best_Ms, device='cpu') #move back to CPU for curve fitting.

    ```

    ```python
    # Example function for M = f(N)
    class M_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.tensor(1.0, device=device))
            self.b = torch.nn.Parameter(torch.tensor(0.1, device=device))

        def forward(self, N):
            return torch.round(self.a * torch.sqrt(N) + self.b).int() # M must be an integer

    m_model = M_Model().to(device)
    optimizer_M = torch.optim.Adam(m_model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss() #Use Mean Squared Error

    epochs = 10000
    for epoch in range(epochs):
        for N, M in best_Ms:
            N_tensor = torch.tensor(N, device=device, dtype=torch.float32)
            M_tensor = torch.tensor(M, device=device, dtype=torch.float32)
            optimizer_M.zero_grad()
            M_pred = m_model(N_tensor)
            loss = loss_fn(M_pred.float(), M_tensor) #Use Mean Squared Error
            loss.backward()
            optimizer_M.step()
    ```

4.  **Define Exclusion Zones:**

    *   Define `τ_N` by analyzing `H_ev` and `σ_ev` stability.
    *   Define `C` for `N/M > C` based on `σ_ev` behavior.

5.  **Visualization:** Plot `M = f(N)` and the exclusion zones.

## Experiment 3: Evaluation of Bin Selection Methods

1.  **Implement Methods:**

    ```python
    def rice_rule(N):
      return int(2 * (N**0.5))

    def fd_rule(N):
      # For uniform [0,1], IQR = 0.5, max-min = 1
      return int(2 * (N**(1/3)))

    # Placeholder for Bayesian Blocks - Needs a full implementation
    def bayesian_blocks(data):
      # ... (Implementation) ...  (This will be more complex)
      return M  # Return the optimal number of bins

    def occupancy(N, C=5, M_max=1024):
        return int(min(M_max, N/C))

    ```

2.  **Data Generation:** Generate uniform datasets for a range of `N` values.

3.  **Binning and Entropy Calculation:** For each dataset and each method, calculate `M` and then `H_obs`.

4.  **Bias and Variance Calculation:** Calculate bias (`H_obs - H_ev(N, M)`) and variance of `H_obs` using the fitted `H_ev` and `σ_ev` from Experiment 1.

5.  **Comparison:** Use plots and summary statistics.

## Experiment 4: Bias Correction and Selection Method Comparison

1.  **Implement Methods:**

    ```python
    def chao_shen_correction(counts, N):
        M = len(counts)
        singles = (counts == 1).sum()
        return (M - singles) / N #Simplified because we assume uniform

    def bootstrap_correction(data, M, R=100):
      entropies = []
      for _ in range(R):
        resampled_data = data[torch.randint(0, len(data), (len(data),), device=device)]
        counts = torch_histogram(resampled_data, bins=M)
        probs = counts / len(data)
        probs = torch.where(probs == 0, torch.tensor(1e-12, device=device), probs)
        H = -torch.sum(probs * torch.log(probs))
        entropies.append(H)
      return torch.std(torch.stack(entropies))

    ```

2.  **Data Generation:** Generate uniform datasets.

3.  **Apply Corrections and Calculate Metrics:** Apply each correction, calculate `H_effective = H_obs + k * σ_ev(N, M)`, and `H_sel = (H_obs + k * σ_ev(N,M)) / H_max`.

4.  **Comparison:** Evaluate bias reduction, variance, and computational cost. Vary `k` and analyze its impact.

This implementation plan provides a detailed, step-by-step guide for conducting the experiments using PyTorch and GPU acceleration. It includes code snippets for the key components, making it directly actionable. The use of vectorized operations and custom PyTorch functions ensures efficiency. The plan also incorporates the refinements and clarifications discussed previously, such as the focus on the uniform distribution, the "sweet spot" analysis, and the comparison of different bin selection and bias correction methods.

## Experiment 5: Evaluation of Model Selection Criteria with Splitting

This experiment evaluates the performance of different model selection criteria in the context of the model *splitting* scenario. It builds upon the previous experiments, particularly the derived functions for `H_ev(N, M)`, `σ_ev(N, M)`, and the bin selection strategy `M = f(N)`.  The key difference from previous experiments is that we will now simulate the model splitting process and evaluate how well the selection criteria identify beneficial splits.

### 1. Goal

The primary goal is to determine which model selection criterion most effectively identifies beneficial model splits.  A "beneficial split" is one where the weighted average entropy of the two child models is significantly *higher* than the entropy of the parent model. We'll evaluate:

*   **`H_obs` (Bias-Corrected):** Using the bias-corrected observed entropy as the selection criterion.
*   **`H_effective = H_obs + k * σ_ev`:**  Using the observed entropy plus a multiple of the estimated standard deviation.
*   **`H_sel = (H_obs + k * σ_ev) / H_max`:**  Using the normalized version of `H_effective`.

We'll also analyze the impact of the weighting factor, `k`, on the performance of `H_effective` and `H_sel`.

### 2. Hypothesis

We hypothesize that `H_effective` and `H_sel` will outperform `H_obs` (even with bias correction) because they explicitly account for the uncertainty in the entropy estimates. We also hypothesize that there will be an optimal range for the value of `k`.

### 3. Procedure

1.  **Setup:**
    *   Choose a range of initial sample sizes, `N_initial`.
    *   Set a maximum number of splits, `S_max` (or a maximum tree depth).
    *   Choose a bias correction method (from Experiment 4).  Let's assume Chao-Shen for this description.
    *   Choose a range of `k` values for `H_effective` and `H_sel` (e.g., `k = 0, 0.5, 1, 1.5, 2`).
    *   Set a significance level, `α`, for statistical testing (e.g., `α = 0.05`).

2.  **Simulation Loop (for each `N_initial`):**

    *   **Initialization:**
        *   Generate a dataset of size `N_initial` from the uniform distribution on [0, 1].
        *   Create an initial model, `K_0`, with this dataset.
        *   Calculate `M_0 = f(N_initial)` using the "sweet spot" function from Experiment 2.
        *   Calculate `H_obs,0` (bias-corrected) for `K_0` using `M_0` bins.

    *   **Splitting Iterations (for `s` in 1 to `S_max`):**
        *   **Identify Candidate Model:** Find the leaf node (model) in the current tree with the lowest value according to each selection criterion (`H_obs`, `H_effective`, `H_sel`). We will evaluate each separately. Let's denote the candidate model as `K_i`.
        *   **Split the Model:**
            *   Randomly split the data associated with `K_i` into two roughly equal subsets (e.g., 50/50 split).
            *   Create two new child models, `K_i,0` and `K_i,1`, with the corresponding data subsets.
            *   Calculate `M_i,0 = f(N_i,0)` and `M_i,1 = f(N_i,1)`.
            *   Calculate `H_obs,i,0` and `H_obs,i,1` (bias-corrected) for the child models.
        *  **Calculate Weighted Average Entropy:**
            *   `H_avg = (N_i,0 * H_obs,i,0 + N_i,1 * H_obs,i,1) / (N_i,0 + N_i,1)`
        *   **Statistical Test:**
            *   Perform a statistical test (e.g., a t-test or a bootstrap test) to determine if `H_avg` is significantly greater than `H_obs,i` (the parent model's entropy). We will use bootstrapping.
        *   **Accept/Reject Split:**
            *   If the test indicates a significant improvement (p-value < `α`), accept the split:  Remove `K_i` from the tree and add `K_i,0` and `K_i,1` as its children.
            *   Otherwise, reject the split.
    * **Track "True" KLD**
        * Calculate a very good approximation of the "True" KLD.

3.  **Evaluation:**

    *   After `S_max` splits (or when no more splits are possible), compare the performance of the different selection criteria:
       * Calculate the "True" KLD using the final set of leaf nodes.
        *   Compare the "True" KLD achieved by each selection criterion. Lower "True" KLD is better.
        * Calculate how many splits were made.
        * Calculate how long each method took.
    *   Repeat the entire simulation multiple times (for each `N_initial`) to obtain statistically robust results.

### 4. Metrics

*   **True KLD:**  The *actual* KLD between the final model's distribution (represented by the leaf nodes) and the uniform distribution, calculated with a very large number of bins (to approximate the true continuous distribution).  This serves as our "ground truth" for comparison.
*   **Number of Splits:** The total number of splits performed.
*   **Computational Time:** The time taken for the entire splitting process.
*   **Accuracy of Split Decisions:** We will track "True Positive", "False Positive", "True Negative" and "False Negative" rates.

### 5. Failure Criteria

*   A selection criterion consistently leads to higher "True" KLD values than other criteria.
*   A selection criterion leads to significantly more splits without a corresponding improvement in "True" KLD.
*   The computational time for a selection criterion is prohibitively high.
*   No suitable `k` value can be found that provides a good balance between accepting beneficial splits and rejecting detrimental ones.

### 6. Deliverables

*   Plots comparing the "True" KLD achieved by different selection criteria (`H_obs`, `H_effective`, `H_sel`) as a function of `N_initial` and `k`.
*   Plots showing the number of splits performed by each criterion.
*   Analysis of the computational time for each criterion.
*   Recommendations for the best selection criterion and the optimal range for `k`.
*   Analysis of "True Positive", "False Positive", "True Negative" and "False Negative"

### 7. Implementation Notes (PyTorch)

*   **Tree Structure:** Implement a simple tree structure to represent the models.  Each node in the tree should store:
    *   The data subset associated with the model.
    *   The number of bins, `M`.
    *   The calculated entropy (`H_obs`, potentially other metrics).
    *   Pointers to its parent and child nodes (if any).
*   **Statistical Test:** Implement a bootstrap test for comparing the weighted average entropy of the child models to the parent model's entropy. This involves resampling the data and recalculating the entropies multiple times.
*   **Vectorization:** Continue to use PyTorch's vectorized operations for efficiency.

This experiment plan provides a concrete way to evaluate the different model selection criteria in the context of the model splitting behavior.  It emphasizes comparing the criteria based on a "ground truth" KLD, which is crucial for determining their effectiveness in identifying truly beneficial splits. The use of statistical tests and the tracking of "True KLD" will provide a robust evaluation framework.
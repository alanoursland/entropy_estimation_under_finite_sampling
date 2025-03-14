# Experiment Plan: Entropy Estimation with Uniform Distribution

This document outlines the experimental plan for developing and evaluating entropy estimation techniques, focusing on the case of a uniform distribution on [0, 1] as a baseline for maximum entropy.

## 1. Goals

The primary goals of these experiments are:

1.  **Empirically derive functions for `H_ev(N, M)` and `σ_ev(N, M)`:** Determine how the expected entropy and its standard deviation depend on the sample size (`N`) and the number of bins (`M`) when sampling from a uniform distribution on [0, 1].
2.  **Define a "sweet spot" for bin selection:** Develop a data-driven rule, `M = f(N)`, for choosing the number of bins that balances bias and variance in entropy estimation.
3.  **Evaluate and compare different bin selection methods:** Compare our `M = f(N)` rule with established methods like the Rice Rule and Freedman-Diaconis Rule, and potentially Bayesian Blocks.
4.  **Evaluate and compare different bias correction methods:** Compare the Chao-Shen estimator, bootstrap-based correction, and our proposed `H_effective = H_obs + k * σ_ev` method.
5.  **Evaluate and compare different selection methods:** Compare using `H_obs`, `H_effective`, and `H_sel = (H_obs + k * σ_ev) / H_max` as selection methods.
6. **Determine exclusion zones:** Create a function that determines areas of the (N, M) plane to avoid.

## 2. Experiment 1: Deriving `H_ev(N, M)` and `σ_ev(N, M)`

*   **Hypothesis:**  `H_ev(N, M)` will be less than `H_max = log(M)` and will increase with `N` for a fixed `M`, approaching `log(M)` asymptotically. `σ_ev(N, M)` will decrease with `N` and increase with `M`.  There will be a strong relationship between `σ_ev` and `N/M`.

*   **Procedure:**
    1.  **Parameter Selection:** Choose a wide, logarithmically spaced range of `N` and `M` values.  Examples:
        *   `N`: 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, ...
        *   `M`: 2, 4, 8, 16, 32, 64, 128, 256, ... (Ensure coverage of `N/M` < 1, `N/M` ≈ 1, and `N/M` > 1)
    2.  **Data Generation:** For each `(N, M)` pair, generate a large number (e.g., `R = 1000`) of datasets. Each dataset consists of `N` samples drawn independently from a uniform distribution on [0, 1].
    3.  **Histogram Calculation:** For each dataset, create a histogram with `M` bins of equal width (0 to 1/M, 1/M to 2/M, etc.) and calculate the observed entropy, `H_obs`.
    4.  **Statistics Calculation:** For each `(N, M)` pair, calculate:
        *   `H_ev(N, M) = mean(H_obs)` (average `H_obs` over the `R` datasets)
        *   `σ_ev(N, M) = std(H_obs)` (standard deviation of `H_obs` over the `R` datasets)
    5.  **Curve Fitting:** Fit functions to the `H_ev(N, M)` and `σ_ev(N, M)` data.  Try different functional forms, prioritizing simplicity and interpretability. Consider functions that incorporate `N/M`.
    6.  **Visualization:** Create plots to visualize the data and the fitted functions (see detailed list in previous response).

*   **Metrics:**
    *   Goodness of fit (R-squared, residual analysis) for the curve fitting.
    *   Visual inspection of plots to confirm expected trends.

*   **Failure Criteria:**
    *   Poor fit of the chosen functions to the empirical data.
    *   Inability to find a clear relationship between `σ_ev`, `N`, and `M`.

*   **Deliverables:**
    *   Fitted functions for `H_ev(N, M)` and `σ_ev(N, M)`.
    *   Plots and goodness-of-fit statistics.
    *   Analysis of the relationships.

## 3. Experiment 2: "Sweet Spot" Analysis and `M = f(N)` Derivation

*   **Hypothesis:** There exists a "sweet spot" for `M`, given `N`, that minimizes variance or achieves a desirable balance between bias and variance. This sweet spot can be approximated by a function `M = f(N)`.

*   **Procedure:**
    1.  **Use Results from Experiment 1:**  Use the fitted functions `H_ev(N, M)` and `σ_ev(N, M)`.
    2.  **Define "Sweet Spot" Criterion:** Choose a criterion for the sweet spot.  Examples:
        *   Minimize `σ_ev(N, M)` for a given `N`.
        *   Minimize `σ_ev(N, M) / H_ev(N, M)` (relative standard deviation).
        *   Find `M` where `H_ev(N, M) / H_max` reaches a certain threshold (e.g., 0.99).
    3.  **Derive `M = f(N)`:**  For a range of `N` values, find the `M` that satisfies the chosen criterion.  Fit a function to this `(N, M)` data.
    4. **Define exclusion zones** Use the derived functions to find values for τ_N and a function for C in N/M > C.
    5.  **Visualization:** Plot `M = f(N)` along with the data from Experiment 1 to visually assess the "sweet spot".

*   **Metrics:**
    *   How well the chosen criterion captures the desired balance between bias and variance.
    *   Goodness of fit of the `M = f(N)` function.

*   **Failure Criteria:**
    *   The chosen criterion doesn't lead to a well-defined "sweet spot."
    *   The `M = f(N)` function is overly complex or doesn't fit the data well.

*   **Deliverables:**
    *   A clearly defined "sweet spot" criterion.
    *   A function `M = f(N)` representing the sweet spot.
    *   Justification and visualization.

## 4. Experiment 3: Evaluation of Bin Selection Methods

*   **Hypothesis:** The `M = f(N)` method will outperform the Rice Rule and Freedman-Diaconis Rule in terms of bias and variance for the uniform distribution.

*   **Procedure:**
    1.  **Methods:**
        *   `M = f(N)` (from Experiment 2)
        *   Rice Rule: `M = 2 * sqrt(N)`
        *   Freedman-Diaconis Rule: `M = 1 / (IQR * N^(-1/3)) = 2 * N^(1/3)` (since IQR = 0.5 for uniform [0, 1])
        *   Bayesian Blocks (implementation required)
         *   Occupancy: Select \( M_k = \min \left( M_k^{\max}, \frac{N_k}{C} \right) \)
    2.  **Data Generation:** Generate datasets from the uniform distribution for a range of `N` values.
    3.  **Binning and Entropy Calculation:** For each dataset and each bin selection method, calculate `H_obs`.
    4.  **Bias and Variance Calculation:**  Calculate the bias (`H_obs - H_ev(N, M)`) and variance of `H_obs` for each method and each `N`.
    5.  **Comparison:** Compare the methods based on bias, variance, and computational cost.

*   **Metrics:**
    *   Bias (average difference between `H_obs` and `H_ev`).
    *   Variance of `H_obs`.
    *   Computational time.

*   **Failure Criteria:**
    *   `M = f(N)` performs significantly worse than other methods in terms of bias or variance.

*   **Deliverables:**
    *   Quantitative comparison of the bin selection methods.
    *   Recommendations.

## 5. Experiment 4: Bias Correction and Selection Method Comparison

*   **Hypothesis:**  `H_effective` will perform comparably to or better than other bias correction methods, and `H_sel` will be a better selection method than either `H_obs` or `H_effective`.

*   **Procedure:**
    1.  **Implement:**
        *   Chao-Shen estimator.
        *   Bootstrap-based bias correction.
        *   `H_effective = H_obs + k * σ_ev(N, M)` (using `σ_ev` from Experiment 1)
         *    `H_sel = (H_obs + k * σ_ev(N,M)) / H_max`
    2.  **Data Generation:** Generate datasets from the uniform distribution for a range of `N` values.
    3.  **Bias Correction:** Apply each bias correction method to the `H_obs` values obtained using the chosen bin selection method (from Experiment 3).
    4.  **Comparison:** Compare the methods based on:
        *   Bias reduction (how close the corrected entropy is to `H_ev`).
        *   Variance of the corrected entropy estimates.
        *   Computational cost.
    5.  **Vary `k`:** Experiment with different values of `k` in `H_effective` and `H_sel`.

*   **Metrics:**
    *   Bias reduction.
    *   Variance.
    *   Computational time.

*   **Failure Criteria:**
    *   `H_effective` or `H_sel` consistently performs worse than other methods.
    *   No suitable value of `k` can be found.

*   **Deliverables:**
    *   Quantitative comparison of the bias correction methods.
    *   Recommendations for the best method and for choosing `k`.

## 6. Iteration

The results of each experiment will inform the subsequent steps. For example, if the "sweet spot" analysis reveals a different relationship between `N` and `M` than initially hypothesized, the curve fitting and subsequent experiments will be adjusted accordingly.

This detailed experiment plan provides a clear roadmap for systematically investigating entropy estimation with a uniform distribution baseline. It focuses on empirical validation and comparison of different strategies, leading to a well-justified and robust methodology.
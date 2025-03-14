# Entropy Plots Analysis and Equation Modeling

This document summarizes the analysis of the plots generated from Experiment 1, which investigated the behavior of expected entropy (\( H_{ev} \)) and its standard deviation (\( σ_{ev} \)) for a uniform distribution on [0, 1] as a function of sample size (\( N \)) and bin count (\( M \)).  The goal is to derive empirical functions for \( H_{ev}(N, M) \) and \( σ_{ev}(N, M) \) to be used in subsequent experiments.

## Plot 1: \( H_{ev} \) vs. \( N \) and \( H_{ev} \) vs. \( M \)

**Description:**

This figure contains two plots:

*   **Left:** \( H_{ev} \) vs. \( N \) (log scale) for different values of \( M \).
*   **Right:** \( H_{ev} \) vs. \( M \) (log scale) for different values of \( N \).

**Observations:**

*   **Left Plot (\( H_{ev} \) vs. \( N \)):**
    *   \( H_{ev} \) approaches \( log(M) \) asymptotically as \( N \) increases, confirming theoretical expectations.
    *   Larger \( M \) values require larger \( N \) to converge to \( log(M) \).
    *   The curves exhibit a sigmoidal shape, suggesting an exponential approach to the asymptote.
    *   For very small N, H is also very small.

*   **Right Plot (\( H_{ev} \) vs. \( M \)):**
    *   For larger \( N \), the relationship between \( H_{ev} \) and \( M \) is approximately linear on the log-log scale, hinting at a power-law relationship, but we know \( log(M) \) is involved.
    *   For smaller \( N \), curves flatten out at higher \( M \) due to bin sparsity.
    * Appears to be linear when M < N.

**Implications:**

*   The \( N/M \) ratio is a crucial factor influencing \( H_{ev} \).
*   A candidate function for \( H_{ev}(N, M) \) should include a term like \( 1 - exp(-a * (N/M)) \) to capture the sigmoidal behavior and asymptotic approach to \( log(M) \).
*   The function may need an additional term like \( b*log(N) \).

## Plot 2: \( H_{ev} / H_{max} \) vs. \( N/M \)

**Description:**

This plot shows the normalized expected entropy (\( H_{ev} / H_{max} \), where \( H_{max} = log(M) \)) as a function of \( N/M \) (log scale).

**Observations:**

*   **Data Collapse:** The data points collapse onto a single, sigmoidal curve. This *strongly* confirms that \( N/M \) is the primary controlling factor for the normalized expected entropy.
*   **Sigmoidal Shape:**  The S-curve reinforces the idea of an exponential approach to the maximum entropy.
*   **Transition Region:**  The steepest rise occurs around \( N/M ≈ 1 \).

**Implications:**

*   Very strong empirical support for using \( N/M \) as the key variable.
*   Further strengthens the case for a candidate function of the form \( H_{ev}(N, M) = log(M) * (1 - exp(-a * (N/M))) + ... \).
*   Simplifies the problem, suggesting that \( H_{ev} / H_{max} \) can be modeled as a function of \( N/M \) alone:  \( H_{ev} / H_{max} = f(N/M) \).

## Plot 3: \( σ_{ev} \) vs. \( N \) and \( σ_{ev} \) vs. \( M \)

**Description:**

This figure contains two plots:

*   **Left:** \( σ_{ev} \) vs. \( N \) (log scale) for different values of \( M \).
*   **Right:** \( σ_{ev} \) vs. \( M \) (log scale) for different values of \( N \).

**Observations:**

*   **Left Plot (σ_{ev} vs. N):**
    *   \( σ_{ev} \) decreases with increasing \( N \), as expected (Central Limit Theorem).
    *   The curves are roughly linear on the log-log scale, suggesting a power-law relationship: \( σ_{ev} ≈ a * N^(-b) \). The slope appears close to -0.5.
    *   Larger \( M \) values generally have higher \( σ_{ev} \).
    *   "Cross-hatching" is observed: lines for different \( M \) values cross each other.

*   **Right Plot (σ_{ev} vs. M):**
    *   Non-monotonic relationship: \( σ_{ev} \) initially increases with \( M \), reaches a peak, and then decreases.
    *   The peak shifts to higher \( M \) values as \( N \) increases.

**Implications:**

*   The cross-hatching and non-monotonic behavior in the \( σ_{ev} \) vs. \( M \) plot indicate a complex relationship.
*   \( N/M \) is again a crucial factor, but likely not the *only* factor influencing \( σ_{ev} \).
*   A simple power law in \( N \) will not fully capture the behavior. We need a more complex function for \( σ_{ev}(N, M) \).

## Plot 4: \( σ_{ev} \) vs. \( N/M \)

**Description:**

This plot shows the standard deviation of the entropy estimate (\( σ_{ev} \)) as a function of \( N/M \) (log scale).

**Observations:**

*   **Partial Collapse:** Data points cluster around a general trend, but with more scatter than the \( H_{ev} / H_{max} \) plot.  \( N/M \) is a major factor, but not the sole determinant of \( σ_{ev} \).
*   **Decreasing Trend:**  \( σ_{ev} \) decreases rapidly as \( N/M \) increases, then levels off.
* **Two Regimes:** Suggests there may be two different regimes in the data.

**Implications:**

*   Confirms the importance of \( N/M \), but also indicates that other factors (possibly \( N \) and \( M \) individually) play a role.
*   Suggests candidate functions like a power law with an offset, an exponential decay, a combination of the two, or possibly a piecewise function.

## Plot 5: \( σ_{ev} / H_{ev} \) vs. \( N/M \)

**Description:**

This plot shows the relative standard deviation of the entropy estimate (\( σ_{ev} / H_{ev} \)) as a function of \( N/M \) (log scale).

**Observations:**

*   **Clear Minimum:**  The plot exhibits a clear minimum, indicating the region of greatest reliability (lowest relative uncertainty).
*   **"Sweet Spot" Range:** The minimum lies within a relatively flat region between approximately \( N/M = 10 \) and \( N/M = 100 \).
*   **Rapid Decrease:** \( σ_{ev} / H_{ev} \) decreases rapidly for \( N/M < 10 \).
*   **Slight Increase (Potentially):**  A possible slight increase in \( σ_{ev} / H_{ev} \) for very large \( N/M \) values (beyond 100).

**Implications:**

*   Provides strong empirical support for an occupancy-based binning rule.
*   Defines a clear "sweet spot" range for \( N/M \):  **10 to 100**.
*   Suggests a simple and effective binning strategy: \( M = round(N / C) \), where \( C \) is a constant between 10 and 100.

## Summary and Proposed Equations

Based on the analysis of all five plots, we propose the following:

**1. "Sweet Spot" for Binning:**

*   Target \( N/M \) to be within the range of **10 to 100**. This minimizes the relative standard deviation of the entropy estimate.
*   This justifies a simple adaptive binning rule: \( M = round(N / C) \), where \( C \) is a constant between 10 and 100 (e.g., \( C = 20 \)).

**2. Candidate Function for \( H_{ev}(N, M) \):**

\[
H_{ev}(N, M) = log(M) * (1 - exp(-a * (N / M))) + b
\]
or
\[
H_{ev}(N, M) = log(M) * (1 - exp(-a * (N / M))) + c*log(N) + b
\]

*   \( log(M) \): Theoretical maximum entropy.
*   \( (1 - exp(-a * (N/M))) \):  Captures the sigmoidal shape and dependence on \( N/M \).
*    \( b \): Offset.
*   \( a \):  Parameter controlling the rate of convergence.
* \( c*log(N) \): Additional term to improve fit.

**3. Candidate Functions for \( σ_{ev}(N, M) \) (Options, increasing in complexity):**

*   **Option 1 (Power Law with Offset):**

    \[
    sigma_{ev}(N, M) = a * (N/M)^(-b) + c
    \]

*   **Option 2 (Exponential Decay with Offset):**

    \[
    sigma_{ev}(N, M) = a * exp(-b * (N/M)) + c
    \]

*   **Option 3 (Combined Power Law and Exponential):**

    \[
    sigma_{ev}(N, M) = a * (N/M)^(-b) + c * exp(-d * (N/M))
    \]
*   **Option 4 (Rational Function):**
    \[
    sigma_{ev}(N,M) = (a + b*(N/M)) / (c + d*(N/M) + e*(N/M)**2)
    \]
*   **Option 5 (Piecewise Function):**

    if N/M < threshold:
    \[
        sigma_{ev}(N, M) = f1(N, M)
    \]
    e.g., a power law
    else:
    \[
        sigma_{ev}(N, M) = f2(N, M)
    \]
    e.g., a slower power law or a constant    
* **Option 6 (Adding a term for M)**
    \[
    sigma_{ev}(N, M) = a * (N/M)^{-b} * M^c + d
    \]
    or
    \[
    sigma_{ev}(N, M) = (a * (N/M)^{-b} + c) * (1 + d*M)
    \]

**Next Steps:**

1.  **Implement Curve Fitting:** Use PyTorch to fit the candidate functions to the empirical data.
2.  **Evaluate Fit:** Assess the goodness of fit using R-squared, residual plots, and visual inspection.
3.  **Refine Functions:** If necessary, try more complex functional forms or piecewise functions.
4. **Define Exclusion Zones**: Create a definition for values of N and M that are not acceptable.

This document provides a comprehensive analysis of the plots generated in Experiment 1, leading to concrete recommendations for the functional forms of \( H_{ev}(N, M) \) and \( σ_{ev}(N, M) \), a clear definition of the "sweet spot" for bin selection, and a roadmap for the next steps in the project.

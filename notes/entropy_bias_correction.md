# Entropy Bias Correction

## 1. Introduction
Entropy estimation from finite samples often suffers from bias, particularly when the ratio of samples to bins is low. Histograms provide a simple and interpretable way to approximate probability distributions, making them a widely used method for entropy estimation. In this context, they are particularly useful when comparing entropy to a known reference distribution, such as a uniform or Gaussian distribution. By using histograms, we can ensure that probability estimates align with predefined bin structures, simplifying comparisons across models and reducing computational complexity.

This document examines the sources of bias in entropy estimation and outlines correction methods to improve accuracy and comparability across models.
Entropy estimation from finite samples often suffers from bias, particularly when the ratio of samples to bins is low. This document examines the sources of bias in entropy estimation and outlines correction methods to improve accuracy and comparability across models.

## 2. Sources of Bias in Entropy Estimation
For a given model \( M_k \) with \( N_k \) samples distributed into \( M_k \) histogram bins, the empirical entropy estimate is computed as:
\[
H_k = - \sum_{i=1}^{M_k} p_{i,k} \log p_{i,k},
\]
where \( p_{i,k} = \frac{n_{i,k}}{N_k} \) represents the empirical probability of bin \( i \). However, when \( N_k \) is small relative to \( M_k \), many bins remain empty, causing entropy to be underestimated.

The expected number of occupied bins, \( S_k \), follows:
\[
S_k \approx M_k \left( 1 - e^{-N_k / M_k} \right),
\]
indicating that for low \( N_k / M_k \), a significant number of bins remain empty, leading to entropy underestimation.

## 3. Justification for Histogram-Based Entropy Estimation
Although alternative entropy estimators exist (e.g., kernel density estimation, nearest-neighbor methods), histograms are particularly well-suited for this problem due to the structure of the reference distribution \( Q(x) \) used in Kullback-Leibler Divergence (KLD) estimation. Specifically:
- The histogram binning approach enables **percentile-based bin definitions**, simplifying entropy calculations by ensuring each bin has equal probability mass in \( Q(x) \).
- For **comparisons against a uniform or Gaussian reference distribution**, histograms provide a **direct partitioning of probability space**, making the computation of KLD straightforward.
- In the PCA-transformed and whitened space, histograms remain a practical choice because the transformed data aligns with a **fixed grid representation of probability mass**, ensuring stable entropy estimation.
- Computationally, histograms are **efficient** and **scalable**, avoiding the overhead of kernel-based density estimation while maintaining interpretability.

Given these advantages, histograms are the preferred choice for entropy estimation in this framework, particularly when analyzing how data deviates from a known reference distribution.

## 4. Bias Correction Methods
Several methods can adjust entropy estimates to mitigate this bias.

### 4.1. Expected Occupancy Correction
A simple correction accounts for the number of occupied bins:
\[
H_k^* = H_k + \frac{M_k - S_k}{N_k}.
\]
This correction adjusts for the missing entropy contribution from unoccupied bins.

### 4.2. Jackknife Bias Correction
The **jackknife estimator** provides an alternative correction by systematically removing samples and recalculating entropy to estimate the bias:
\[
H_k^* = H_k + \frac{1}{2N_k}.
\]
This correction is useful for small-sample cases.

### 4.3. Chao-Shen Correction
The **Chao-Shen estimator** uses an effective probability adjustment to account for missing bins:
\[
H_k^* = - \sum_{i=1}^{M_k} \hat{p}_{i,k} \log \hat{p}_{i,k},
\]
where \( \hat{p}_{i,k} \) is a smoothed probability estimate using a Bayesian prior.

### 4.4. Bootstrap-Based Correction
A **bootstrap approach** resamples the dataset multiple times to estimate an expected entropy distribution and apply a bias adjustment:
\[
H_k^* = H_k + \lambda \cdot \sigma_H,
\]
where \( \sigma_H \) is the bootstrap-estimated standard deviation of entropy.

## 5. Choosing a Correction Method
The best correction method depends on the sample size:
- **For small \( N_k / M_k \)**: Use **Chao-Shen or Jackknife corrections**.
- **For moderate \( N_k / M_k \)**: Use **expected occupancy correction**.
- **For large \( N_k \)**: Minimal correction is needed as entropy estimation approaches the true value.

## 6. Conclusion
Bias in entropy estimation is a significant issue in finite sample analysis, particularly when histograms contain many empty bins. Applying appropriate correction methods ensures more accurate entropy estimates and enables fair model comparisons.


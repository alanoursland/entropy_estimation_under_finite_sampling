# Variance Analysis of Entropy Estimation

## 1. Introduction
Entropy estimation from finite samples not only suffers from bias but also exhibits variance due to stochastic fluctuations in the observed probability distribution. This variance affects the reliability of entropy comparisons across models. This document explores the sources of variance in entropy estimation, provides analytical approximations, and proposes methods for variance reduction and correction.

## 2. Sources of Variance in Entropy Estimation
For a model \( M_k \) with \( N_k \) samples distributed across \( M_k \) histogram bins, the empirical entropy estimate is:
\[
H_k = - \sum_{i=1}^{M_k} p_{i,k} \log p_{i,k},
\]
where \( p_{i,k} = \frac{n_{i,k}}{N_k} \) is the empirical probability of bin \( i \). Due to finite sampling, \( p_{i,k} \) deviates from the true underlying probability, introducing variability in \( H_k \).

## 3. Analytical Approximation of Variance
The variance of the entropy estimator can be approximated as:
\[
\sigma_{H,k}^2 \approx \frac{1}{N_k} \left( \sum_{i=1}^{M_k} p_{i,k} (\log p_{i,k})^2 - H_k^2 \right).
\]
This expression captures the impact of sample size and binning on the spread of entropy estimates.

Alternatively, for large \( N_k \), a central limit approximation suggests:
\[
\sigma_{H,k}^2 \approx \frac{1}{M_k N_k} \sum_{i=1}^{M_k} (1 - p_{i,k}) (\log p_{i,k})^2.
\]
This formulation highlights that variance decreases with increasing sample size but can be amplified when many bins have low occupancy.

## 4. Variance Reduction Methods
### 4.1. Adaptive Binning
Reducing the number of bins \( M_k \) based on available sample size can lower variance while preserving resolution. Criteria include:
- Setting \( M_k \) such that \( N_k / M_k > C \) (e.g., \( C = 5 \) ensures at least 5 samples per bin on average).
- Using **Bayesian Blocks** or **Freedman-Diaconis bin selection** for adaptive binning.

### 4.2. Bootstrap Estimation
Bootstrap resampling can provide a direct empirical estimate of entropy variance:
1. Resample the dataset with replacement.
2. Compute entropy for each resample.
3. Compute the variance of the entropy estimates across resamples.

This method is computationally expensive but provides nonparametric confidence intervals.

### 4.3. Bayesian Estimators
Bayesian entropy estimation methods incorporate prior information to regularize variance. The **Dirichlet prior approach** adjusts bin probabilities as:
\[
\hat{p}_{i,k} = \frac{n_{i,k} + \alpha}{N_k + M_k \alpha},
\]
where \( \alpha \) is a smoothing parameter.

This reduces variance for low-sample cases by preventing extreme probability estimates.

### 4.4. Confidence-Weighted Adjustments
Using estimated variance, we can define confidence-adjusted entropy estimates:
\[
H_k^* = H_k + \lambda \cdot \sigma_{H,k},
\]
where \( \lambda \) is a tuning parameter (e.g., \( \lambda = 1 \) for one standard deviation adjustment).

## 5. Conclusion
Entropy estimation variance arises from finite sampling and binning effects. Analytical approximations, bootstrap resampling, and Bayesian regularization provide strategies to quantify and reduce this variance, ensuring more reliable entropy-based model comparisons.


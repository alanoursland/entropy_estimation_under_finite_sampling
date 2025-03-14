# Kullback-Leibler Divergence Estimation

## 1. Introduction
Kullback-Leibler Divergence (KLD) measures the difference between a probability distribution \( P(x) \) and a reference distribution \( Q(x) \). In this work, we estimate KLD using entropy calculations, with \( Q(x) \) defined as a **reference Gaussian distribution**. The challenge is ensuring stable and accurate KLD estimation when using histograms, particularly after applying PCA and whitening transformations.

## 2. Definition of KLD
The Kullback-Leibler Divergence between two distributions \( P(x) \) and \( Q(x) \) is defined as:
\[
D_{KL}(P || Q) = \sum_{i} p_i \log \frac{p_i}{q_i},
\]
where:
- \( p_i \) are the empirical probabilities from the data distribution \( P(x) \).
- \( q_i \) are the probabilities under the reference distribution \( Q(x) \).

We can approximate \( Q(x) \) using a histogram. If we use percentiles of the \( Q(x) \) Cumulative Density Function, we get a uniform histogram where all bins have an equal probability \( q_i = 1/M \). This method ensures that each bin in \( Q(x) \) has equal probability, making entropy estimation computationally simpler and removing the need for explicit bin-wise probability normalization.Using histograms for comparison simplifies the equation to:
\[
D_{KL}(P || Q) = H_Q - H_P.
\]
\( H_Q \) is constant for a fixed \( Q(x) \). If we are comparing multiple \( P(x) \), we and drop \( H_Q \) without changing the comparison results. This results in a standard entropy calculation. 
\[
D_{KL}(P || Q) = -H_P.
\]
Maximizing entropy directly corresponds to minimizing divergence.

## 3. Reference Distribution and Histogram-Based Estimation
In this framework, \( Q(x) \) is a **standard normal distribution** \( N(0,1) \), and the data is transformed before estimation:
1. Compute the **mean and covariance** of the high-dimensional data.
2. Apply **PCA** to extract principal components.
3. **Whiten** the data so that it matches \( N(0,1) \) in the transformed space.
4. Compute entropy on the **whitened** histogram representation.

Since the reference \( Q(x) \) is standard normal, we define bins based on percentiles:
- Each bin represents an equal-probability partition of \( Q(x) \).
- This ensures that \( q_i = 1/M \) for all bins, eliminating the need to explicitly calculate \( H_Q \).
- The result is a direct entropy-based estimation:
  \[
  D_{KL}(P || Q) \approx -H_P.
  \]
  where \( H_P \) is the entropy of the whitened data.

## 4. Advantages of Histogram-Based KLD Estimation
### 4.1. Computational Simplicity
- Since \( Q(x) \) has equal-probability bins, computing KLD **only requires entropy estimation** without explicit \( q_i \) calculations.
- PCA and whitening make the entropy estimation **more stable** by transforming the data into a space where the reference is simple.

### 4.2. Robustness to Dimensionality
- Direct KLD estimation in high dimensions is unstable due to sparsity.
- **By using PCA and whitening**, we reduce the problem to **one-dimensional entropy estimation per principal component**, improving robustness.

## 5. Considerations and Limitations
### 5.1. Bin Selection Effects
- The choice of \( M \) affects entropy estimation and, consequently, KLD.
- Adaptive binning methods should be used to maintain stability.

### 5.2. Effect of PCA Transformation
- PCA captures variance but may discard important distributional details.
- Whitening assumes a Gaussian structure, which may not always hold.

### 5.3. High-Dimensional Dependencies
- Entropy estimation per principal component assumes **independent projections**.
- Some structure may be lost when reducing dimensions.

## 6. Conclusion
Using entropy-based estimation of KLD provides a simple and effective way to compare data distributions against a Gaussian reference. The combination of PCA, whitening, and histogram-based entropy estimation ensures computational efficiency while maintaining stability in high-dimensional spaces.


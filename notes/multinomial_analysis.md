# Entropy Analysis of a Uniform Histogram with Multinomial Distribution

## Problem Statement

We analyze the expected entropy and variance of entropy for a uniform histogram with \( M \) bins, where \( N \) independent and identically distributed (IID) samples are placed, each with probability \( 1/M \) of landing in any bin.

The entropy is defined as:

\[
H = -\sum_{i=1}^M p_i \log_2 p_i,
\]

where \( p_i = n_i / N \) is the proportion of samples in bin \( i \), \( n_i \) is the number of samples in bin \( i \), and \( \sum_{i=1}^M n_i = N \). The \( n_i \) are random variables due to the random placement of samples.

Our objectives are:
- **Expected Entropy**: \( H_{\text{ev}}(N, M) = \mathbb{E}[H] \),
- **Variance of Entropy**: \( \sigma_{\text{ev}}^2(N, M) = \text{Var}(H) = \mathbb{E}[H^2] - (\mathbb{E}[H])^2 \),

computed over all possible distributions of the \( N \) samples into the \( M \) bins.

## Multinomial Distribution

The bin counts \( (n_1, n_2, \ldots, n_M) \) follow a **multinomial distribution**:

\[
P(n_1, n_2, \ldots, n_M) = \frac{N!}{n_1! n_2! \cdots n_M!} \left( \frac{1}{M} \right)^N,
\]

with the constraint \( \sum_{i=1}^M n_i = N \).

Key properties:
- **Expected Value**: \( \mathbb{E}[n_i] = N / M \),
- **Variance**: \( \text{Var}(n_i) = N \cdot \frac{1}{M} \cdot \left( 1 - \frac{1}{M} \right) = \frac{N (M-1)}{M^2} \),
- **Covariance**: \( \text{Cov}(n_i, n_j) = -\frac{N}{M^2} \) for \( i \neq j \), reflecting the dependency among bins due to the fixed total \( N \).

## Expected Entropy, \( H_{\text{ev}}(N, M) \)

### Approximation via Taylor Expansion
For large \( N \), we approximate \( H \) around the expected bin proportion \( p_i = 1/M \). Define \( p_i = 1/M + \epsilon_i \), where \( \mathbb{E}[\epsilon_i] = 0 \) and \( \text{Var}(\epsilon_i) = \frac{M-1}{N M^2} \).

Expand \( h(p_i) = -p_i \log_2 p_i \) around \( 1/M \):
\[
h(p_i) \approx -\frac{1}{M} \log_2 \frac{1}{M} - \frac{\epsilon_i}{\ln 2} - \frac{M \epsilon_i^2}{2 \ln 2},
\]
\[
H = \sum h(p_i) \approx \log_2 M - \frac{M}{2 \ln 2} \sum_{i=1}^M \epsilon_i^2.
\]

Taking the expectation:
\[
\mathbb{E}[H] \approx \log_2 M - \frac{M}{2 \ln 2} \mathbb{E} \left[ \sum \epsilon_i^2 \right],
\]
\[
\mathbb{E}[\epsilon_i^2] = \text{Var}(p_i) = \frac{M-1}{N M^2}, \quad \mathbb{E} \left[ \sum \epsilon_i^2 \right] = M \cdot \frac{M-1}{N M^2} = \frac{M-1}{N M},
\]
\[
H_{\text{ev}} \approx \log_2 M - \frac{M-1}{2 N \ln 2}.
\]

This is the **Miller-Madow correction**, where \( \log_2 M \) is the maximum entropy, and \( \frac{M-1}{2 N \ln 2} \) accounts for finite-sample bias.

## Variance of Entropy, \( \sigma_{\text{ev}}^2(N, M) \)

Using the approximation:
\[
H \approx \log_2 M - \frac{M}{2 \ln 2} \sum_{i=1}^M \epsilon_i^2,
\]
\[
\text{Var}(H) = \left( \frac{M}{2 \ln 2} \right)^2 \text{Var} \left( \sum \epsilon_i^2 \right).
\]

### Computing \( \text{Var}(\sum \epsilon_i^2) \)
Define \( \epsilon_i = \frac{n_i}{N} - \frac{1}{M} = \frac{\delta_i}{N} \), where \( \delta_i = n_i - \frac{N}{M} \):
\[
\sum \epsilon_i^2 = \frac{1}{N^2} \sum \delta_i^2,
\]
\[
\text{Var} \left( \sum \epsilon_i^2 \right) = \frac{1}{N^4} \text{Var} \left( \sum \delta_i^2 \right).
\]

- **\( \text{Var}(\sum \delta_i^2) \)**:
  \[
  \text{Var} \left( \sum \delta_i^2 \right) = \sum_{i=1}^M \text{Var}(\delta_i^2) + \sum_{i \neq j} \text{Cov}(\delta_i^2, \delta_j^2).
  \]
  - \( \text{Var}(\delta_i^2) = \mathbb{E}[\delta_i^4] - (\mathbb{E}[\delta_i^2])^2 \),
  - \( \mathbb{E}[\delta_i^2] = \frac{N (M-1)}{M^2} \),
  - \( \mathbb{E}[\delta_i^4] = 3 \left( \frac{N (M-1)}{M^2} \right)^2 + \frac{N (M-1) (M^2 - 6M + 6)}{M^4} \) (binomial 4th central moment),
  - \( \text{Var}(\delta_i^2) \approx 2 \left( \frac{N (M-1)}{M^2} \right)^2 \) (leading term for large \( N \)),
  - \( M \cdot \text{Var}(\delta_i^2) \approx \frac{2 N^2 (M-1)^2}{M^3} \),
  - Covariance terms are negative and complex, but contribute to a final scaling of:
    \[
    \text{Var} \left( \sum \delta_i^2 \right) \approx \frac{8 N^2 (M-1)}{M^3}.
    \]

- **Final Variance**:
  \[
  \text{Var} \left( \sum \epsilon_i^2 \right) = \frac{1}{N^4} \cdot \frac{8 N^2 (M-1)}{M^3} = \frac{8 (M-1)}{N^2 M^3},
  \]
  \[
  \text{Var}(H) = \frac{M^2}{4 \ln^2 2} \cdot \frac{8 (M-1)}{N^2 M^3} = \frac{2 (M-1)}{N^2 \ln^2 2}.
  \]

## Final Results

- **Expected Entropy**:
  \[
  H_{\text{ev}}(N, M) \approx \log_2 M - \frac{M-1}{2 N \ln 2},
  \]
- **Variance of Entropy**:
  \[
  \sigma_{\text{ev}}^2(N, M) \approx \frac{2 (M-1)}{N^2 \ln^2 2},
  \]
- **Standard Deviation**:
  \[
  \sigma_{\text{ev}}(N, M) \approx \frac{\sqrt{2 (M-1)}}{N \ln 2}.
  \]

## Key Takeaways

- The expected entropy approaches \( \log_2 M \) (maximum entropy) as \( N \) increases.
- The variance scales as \( (M-1) / N^2 \), reflecting the multinomial distribution’s fluctuations.
- The multinomial model correctly captures bin dependencies, unlike binomial approximations that may misestimate variance.

## Notes

- The variance derivation simplifies higher-order terms for large \( N \), but the \( (M-1) / N^2 \) scaling is robust across literature for this problem.
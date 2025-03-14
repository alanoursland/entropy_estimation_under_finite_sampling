# Theoretical Bounds on Entropy Estimation

## 1. Introduction
Entropy estimation from finite samples introduces both **bias and variance**, particularly when the number of bins \( M_k \) is large relative to the sample size \( N_k \). This document derives an **upper bound on entropy underestimation** and analyzes the **asymptotic behavior of bias and variance** as \( N_k \to \infty \) and \( M_k \to \infty \).

## 2. Upper Bound on Entropy Underestimation

### 2.1. Problem Formulation
Given a distribution discretized into \( M_k \) histogram bins, the true entropy is:
\[
H = -\sum_{i=1}^{M_k} p_i \log p_i.
\]
The empirical entropy estimator is:
\[
H_k = -\sum_{i=1}^{M_k} \hat{p}_i \log \hat{p}_i,
\]
where \( \hat{p}_i = n_i / N_k \) represents the empirical probability of bin \( i \), with \( n_i \) being the observed count in that bin.

For small \( N_k / M_k \), many bins remain empty, leading to entropy underestimation. The expected number of **occupied bins** follows:
\[
S_k = M_k \left( 1 - e^{-N_k / M_k} \right).
\]

### 2.2. Derivation of the Upper Bound
The probability of a given bin being empty is:
\[
P(\text{empty bin}) = e^{-N_k / M_k}.
\]
Since missing bins contribute zero entropy, the missing entropy contribution is upper-bounded by:
\[
\Delta H \leq P(\text{empty bin}) \cdot M_k \log M_k = e^{-N_k/M_k} M_k \log M_k.
\]
Thus, the entropy underestimation is bounded by:
\[
H - \mathbb{E}[H_k] \leq e^{-N_k / M_k} M_k \log M_k.
\]
This result shows that **entropy underestimation is severe when \( N_k / M_k \) is small**, but the bias decays exponentially as \( N_k \) increases.

---

## 3. Asymptotic Analysis of Bias and Variance

### 3.1. Bias Behavior

#### **As \( N_k \to \infty \) (Fixed \( M_k \))**
- The probability of each bin remaining empty vanishes \( (e^{-N_k/M_k} \to 0) \).
- The entropy estimate converges to the true entropy:
  \[
  \lim_{N_k \to \infty} \mathbb{E}[H_k] = H.
  \]
- **Bias disappears asymptotically.**

#### **As \( M_k \to \infty \) (Fixed \( N_k \))**
- The number of occupied bins saturates at \( S_k \approx M_k(1 - e^{-N_k/M_k}) \).
- If \( M_k \gg N_k \), most bins remain empty, leading to **severe entropy underestimation**.
- The bias scales as:
  \[
  H - \mathbb{E}[H_k] \approx e^{-N_k/M_k} M_k \log M_k.
  \]
- **Over-binning increases bias unless \( N_k \) grows proportionally.**

---

### 3.2. Variance Behavior
The variance of entropy estimation is given by:
\[
\text{Var}(H_k) \approx \frac{1}{N_k} \sum_{i=1}^{M_k} p_i (\log p_i)^2 - H_k^2.
\]

#### **As \( N_k \to \infty \) (Fixed \( M_k \))**
- The empirical probabilities \( \hat{p}_i \) converge to their true values.
- Variance decreases as:
  \[
  \text{Var}(H_k) = O(1/N_k).
  \]
- **Entropy estimation stabilizes with more samples.**

#### **As \( M_k \to \infty \) (Fixed \( N_k \))**
- Many bins remain empty, increasing sampling noise.
- Variance **increases** as:
  \[
  \text{Var}(H_k) \approx O(M_k / N_k).
  \]
- **Over-binning worsens variance unless \( N_k \) scales accordingly.**

---

## 4. Conclusion
- **Entropy underestimation is upper-bounded by**:
  \[
  H - \mathbb{E}[H_k] \leq e^{-N_k / M_k} M_k \log M_k.
  \]
  The bound **decays exponentially** as \( N_k \) increases.
- **Asymptotic results**:
  - As \( N_k \to \infty \), **bias vanishes**, and variance scales as \( O(1/N_k) \).
  - As \( M_k \to \infty \), **bias grows as** \( O(e^{-N_k/M_k} M_k \log M_k) \) and variance as \( O(M_k / N_k) \).
- **Implication**: Choosing **too many bins without sufficient samples leads to both high bias and high variance**, reinforcing the need for **adaptive binning strategies**.


# A Tutorial on the Expected Entropy and Variance of a Uniform Histogram: Derivations, Misconceptions, and Applications

## Abstract

Entropy serves as a fundamental measure of uncertainty within the context of histograms. Estimating this measure accurately becomes a significant challenge when dealing with a limited number of samples. This paper addresses the problem of entropy estimation under such finite sampling conditions, emphasizing the importance of the Miller-Madow correction for mitigating bias and the derivation of the variance formula to quantify the uncertainty of the estimate. Common misunderstandings prevalent in the literature surrounding entropy variance estimation are highlighted and clarified. The primary objectives of this paper are to provide a clear and accessible derivation of the expected entropy and its variance for uniform histograms, to rectify errors frequently encountered in the estimation of entropy variance, to present a comprehensive survey of existing research and alternative entropy estimation techniques, and to explore the contemporary applications of this theory, particularly within the realms of machine learning and artificial intelligence, including discussions of experimental validations where available. This tutorial is specifically aimed at undergraduate students and new researchers entering the field, providing them with a solid foundation in this critical area.

## 1. Introduction

### 1.1 Motivation

Entropy stands as a cornerstone concept across a multitude of scientific and engineering disciplines, providing a quantitative measure of randomness and uncertainty. From the microscopic world of statistical mechanics to the vastness of cosmological studies, entropy helps to characterize the inherent unpredictability of systems. In the realm of data analysis, histograms serve as practical and widely employed estimators of underlying probability distributions. By dividing the data into bins and counting the occurrences within each, histograms offer a visual and numerical representation of the likelihood of different outcomes. Consequently, the entropy of a histogram provides a measure of the uncertainty associated with the distribution of the data.

Accurate estimation of entropy is of critical importance in diverse fields such as machine learning, physics, and bioinformatics. In machine learning, entropy plays a vital role in various algorithms, including decision trees where it guides the splitting of nodes to maximize information gain. It is also fundamental in evaluating the performance of models and quantifying the uncertainty of predictions. In physics, entropy is a key concept in thermodynamics and statistical mechanics, characterizing the number of microscopic configurations that correspond to a given macroscopic state. Its estimation is crucial for understanding the behavior of physical systems. In bioinformatics, entropy measures the diversity of biological sequences, such as DNA or protein sequences, providing insights into evolutionary processes and functional characteristics. The estimation of Shannon entropy has found applications in measuring genetic diversity, quantifying neural activity, and detecting network anomalies, highlighting its broad relevance across scientific domains.

Despite the fundamental importance of entropy, its reliable estimation poses a significant challenge, particularly when the number of available data samples is limited. This issue of finite sampling introduces bias and uncertainty into the entropy estimates, necessitating careful consideration and appropriate correction methods.

### 1.2 Problem Statement

The Shannon entropy (H) for a discrete probability distribution with M possible outcomes, where p_i is the probability of the i-th outcome, is mathematically defined as:

\[
H = -\sum_{i=1}^{M} p_i \log_2 p_i
\]

When dealing with empirical data, histograms are often used to estimate these probabilities. Given N samples, the probability of the i-th bin is estimated as \( p_i = \frac{n_i}{N} \), where \( n_i \) is the number of samples falling into the i-th bin.

It is crucial to recognize that when entropy is estimated from a finite sample, the bin counts \( n_i \) are subject to random fluctuations. This inherent randomness in the observed data directly leads to the estimated entropy H becoming a random variable. The value of the estimated entropy will vary depending on the specific set of samples observed. Consequently, the naive estimator of entropy, based directly on the observed frequencies, is known to be negatively biased, meaning that on average, it underestimates the true entropy of the underlying distribution. Furthermore, this estimator has a variance that depends on the sample size.

This paper specifically focuses on the case of uniform histograms, where the true underlying probabilities of all M bins are equal, i.e., \( p_i = 1/M \) for all i. Under these conditions, the paper investigates the specific effects that arise due to finite-sample sizes on the estimation of entropy, particularly its expected value and variance.

### 1.3 Paper Overview

This paper is structured to provide a comprehensive understanding of the expected entropy and variance of a uniform histogram under finite sampling. Following this introduction, Section 2 will lay the necessary background on the multinomial distribution, which governs the bin counts in a uniform histogram, and will further elaborate on the concept of histogram entropy and the challenges associated with its estimation. Section 3 will present a rigorous derivation of the expected value of the entropy estimator, leading to the well-known Miller-Madow correction. Section 4 will then focus on the derivation of the variance of the entropy estimator, highlighting common mistakes and misconceptions often found in the literature. Section 5 will provide a survey of various entropy estimation techniques, including both classical bias correction methods and more advanced approaches. Section 6 will explore modern applications of entropy estimation, particularly in machine learning, goodness-of-fit testing, and Kullback-Leibler divergence estimation. Section 7 will outline potential numerical experiments and simulations that can be used to validate the theoretical results and illustrate their practical implications. Section 8 will offer practical recommendations for using the Miller-Madow correction and choosing appropriate estimators for entropy and its variance. Finally, Section 9 will conclude the paper with a summary of the key findings and a discussion of potential future research directions.

The key contributions of this paper include:

- A detailed and rigorous derivation of both the expected value \( \mathbb{E}[H] \) and the variance \( \text{Var}(H) \) of the entropy estimator for uniform histograms under finite sampling.
- A thorough discussion of common mistakes and misconceptions prevalent in the literature, particularly concerning the calculation and interpretation of the variance of entropy.
- A comprehensive survey of various entropy estimation techniques, encompassing both classical and advanced methodologies aimed at addressing the bias inherent in finite-sample entropy estimates.
- An exploration of contemporary applications of these concepts in machine learning, goodness-of-fit testing, and Kullback-Leibler divergence estimation, demonstrating the practical relevance of the theoretical findings.

## 2. Background on the Multinomial Distribution and Histogram Entropy

### 2.1 The Multinomial Distribution

When we consider sampling N independent and identically distributed (i.i.d.) items from a discrete uniform distribution with M bins (or categories), the number of items \( n_i \) that fall into each bin i (where i = 1, 2, ..., M) follows a multinomial distribution. This distribution is denoted as:

\[
(n_1, n_2, ..., n_M) \sim \text{Multinomial}(N, 1/M)
\]

Here, N represents the total number of trials (samples), and \( 1/M \) is the probability of success (falling into a specific bin) for each trial, assuming a uniform distribution across the M bins. The multinomial distribution describes the probability of observing a particular combination of counts \( (n_1, n_2, ..., n_M) \) such that:

\[
\sum_{i=1}^M n_i = N
\]

For a uniform distribution, the expected number of samples in each bin is the same and can be calculated as:

\[
\mathbb{E}[n_i] = N \times (1/M) = N/M
\]

This indicates that, on average, we expect each of the M bins to contain \( N/M \) samples.

The variance of the number of counts in a single bin i is given by:

\[
\text{Var}(n_i) = N \times (1/M) \times (1 - 1/M) = N \frac{M-1}{M^2}
\]

This variance quantifies the expected spread or dispersion of the counts around the mean value \( N/M \) for any given bin.

A crucial aspect of the multinomial distribution, especially relevant for our analysis of entropy variance, is the covariance between the counts of different bins. For any two distinct bins i and j (i ≠ j), the covariance is:

\[
\text{Cov}(n_i, n_j) = -N \frac{1}{M} \frac{1}{M} = -N/M^2
\]

The negative sign of this covariance is significant. It indicates that if the number of samples in one bin is higher than its expected value, the number of samples in other bins is likely to be lower than their expected values, due to the constraint that the total number of samples (N) is fixed. This negative dependence between the bin counts is a key factor that needs to be considered when calculating the variance of the entropy estimator.

### 2.2 Histogram Entropy

Given a histogram with M bins, where the estimated probability of the i-th bin is \( p_i = n_i / N \), the entropy of this histogram is calculated using the Shannon entropy formula:

\[
H = -\sum_{i=1}^M p_i \log_2 p_i = -\sum_{i=1}^M \frac{n_i}{N} \log_2 \left( \frac{n_i}{N} \right)
\]

Since the sample counts \( n_i \) are random variables following a multinomial distribution, the estimated probabilities \( p_i \) are also random variables. Consequently, the entropy H, which is a function of these random probabilities, is itself a random variable. Different samples from the underlying uniform distribution will result in different sets of bin counts \( n_i \), leading to different values of the estimated entropy H.

Due to the finite nature of the sampling process, the histogram-based entropy estimation is generally biased. This means that the expected value of the estimated entropy, \( \mathbb{E}[H] \), is not equal to the true entropy of the uniform distribution, which is \( \log_2 M \). The bias arises because the empirical distribution obtained from a finite sample is rarely perfectly uniform, even if the underlying distribution is. Statistical fluctuations cause some bins to have slightly higher counts and others slightly lower, which tends to make the empirical distribution appear less uniform than it actually is. As a result, the entropy calculated from the empirical distribution tends to underestimate the true entropy. This inherent bias necessitates the use of correction techniques, such as the Miller-Madow correction, to obtain more accurate estimates of the true entropy.

Understanding the properties of the multinomial distribution and how it affects the estimated entropy is crucial for developing and applying these corrections effectively.

## 3. Expected Entropy, \( \mathbb{E}[H] \)

### 3.1 Exact Expression

The expected entropy \( \mathbb{E}[H] \) can be formally expressed as a sum over all possible outcomes of the multinomial distribution, where each outcome is a specific set of bin counts \( (n_1, n_2, ..., n_M) \) such that:

\[
\sum_{i=1}^M n_i = N
\]

The probability of each such outcome is given by the multinomial probability mass function:

\[
P(n_1, n_2, ..., n_M) = \frac{N!}{n_1! n_2! ... n_M!} \left( \frac{1}{M} \right)^{N}
\]

The expected entropy is then the sum of the entropy calculated for each outcome, weighted by the probability of that outcome:

\[
\mathbb{E}[H] = \sum_{n_1 + ... + n_M = N} P(n_1, ..., n_M) \left( -\sum_{i=1}^M \frac{n_i}{N} \log_2 \left( \frac{n_i}{N} \right) \right)
\]

However, directly computing this exact expression is practically infeasible for most realistic scenarios. The number of possible multinomial outcomes grows rapidly with both the number of samples \( N \) and the number of bins \( M \), leading to a combinatorial explosion. For instance, even with moderately sized \( N \) and \( M \), the number of terms in this summation can be astronomically large, making direct calculation computationally intractable. This infeasibility motivates the use of approximation techniques to estimate the expected entropy.

### 3.2 Taylor Expansion Derivation (Miller-Madow Correction)

To overcome the computational challenges associated with the exact expression for expected entropy, we can employ a Taylor series expansion to derive an approximation. Let's consider small fluctuations \( \epsilon_i \) around the true probability \( 1/M \) such that the estimated probability for the i-th bin is \( p_i = 1/M + \epsilon_i \). Since \( \sum_{i=1}^M p_i = 1 \) and \( \sum_{i=1}^M (1/M) = 1 \), it follows that \( \sum_{i=1}^M \epsilon_i = 0 \). The entropy can then be expressed as:

\[
H = -\sum_{i=1}^M (1/M + \epsilon_i) \log_2 (1/M + \epsilon_i)
\]

We can expand the term \( \log_2 (1/M + \epsilon_i) \) using a Taylor series around \( 1/M \). Recall that \( \log_2 x = \frac{\ln x}{\ln 2} \). Let \( x = 1/M + \epsilon_i \). Then \( \ln x = \ln(1/M (1 + M\epsilon_i)) = -\ln M + \ln(1 + M\epsilon_i) \). For small \( M\epsilon_i \), we can use the Taylor expansion \( \ln(1+u) = u - u^2/2 + u^3/3 - ... \). Thus:

\[
\ln(1 + M\epsilon_i) \approx M\epsilon_i - (M\epsilon_i)^2 / 2
\]

So:

\[
\log_2 (1/M + \epsilon_i) \approx \frac{1}{\ln 2} (-\ln M + M\epsilon_i - (M\epsilon_i)^2 / 2) = \log_2 M + \frac{M\epsilon_i}{\ln 2} - \frac{M^2\epsilon_i^2}{2\ln 2}
\]

Now, we can substitute this back into the entropy formula:

\[
H \approx -\sum_{i=1}^M (1/M + \epsilon_i) \left( \log_2 M + \frac{M\epsilon_i}{\ln 2} - \frac{M^2\epsilon_i^2}{2\ln 2} \right)
\]

Expanding this sum, we get:

\[
H \approx -\sum_{i=1}^M \left( \frac{1}{M} \log_2 M + \epsilon_i \log_2 M + \frac{\epsilon_i}{\ln 2} + \epsilon_i^2 \frac{M}{2\ln 2} - \frac{M\epsilon_i^2}{2\ln 2} \right)
\]

Since \( \sum_{i=1}^M \epsilon_i = 0 \), the terms involving \( \epsilon_i \) vanish. We are left with:

\[
H \approx -\log_2 M - \frac{M-1}{2\ln 2} \sum_{i=1}^M \epsilon_i^2
\]

Now, we need to find the expected value of \( \sum_{i=1}^M \epsilon_i^2 \). Recall that \( p_i = n_i / N = 1/M + \epsilon_i \), so \( \epsilon_i = (n_i - N/M) / N = (n_i - N/M) / N \). Thus, \( \epsilon_i^2 = (n_i - N/M)^2 / N^2 \). The expected value of \( (n_i - N/M)^2 \) is the variance of \( n_i \), which we know is \( N(M-1)/M^2 \). Therefore:

\[
\mathbb{E}[\epsilon_i^2] = \frac{1}{N^2} \mathbb{E}[(n_i - N/M)^2] = \frac{1}{N^2} \frac{N(M-1)}{M^2} = \frac{M-1}{NM^2}
\]

The expected value of the sum is then:

\[
\mathbb{E}\left[ \sum_{i=1}^M \epsilon_i^2 \right] = M \frac{M-1}{NM^2} = \frac{M-1}{NM}
\]

Finally, taking the expectation of the approximate entropy, we arrive at the Miller-Madow correction:

\[
\mathbb{E}[H] \approx \log_2 M - \frac{M-1}{2N \ln 2}
\]

The term \( \frac{M-1}{2N \ln 2} \) represents the correction to the true entropy \( \log_2 M \) due to the finite sample size. This correction is positive, indicating that the expected value of the estimated entropy is slightly lower than the true entropy. The magnitude of the correction decreases as the number of samples \( N \) increases, and it increases with the number of bins \( M \). For large \( N \), the correction becomes negligible, and the expected entropy approaches the true entropy.


### The Miller-Madow Estimator

The Miller-Madow estimator, which attempts to correct for the bias, is often given as \( H_{\text{MM}} = H_{\text{naive}} + \frac{M - 1}{2N \ln 2} \), where \( H_{\text{naive}} \) is the uncorrected empirical entropy.

## 4. Variance of Entropy, \( \text{Var}(H) \)

### 4.1 Why Variance Matters

Understanding the variance of the entropy estimator is crucial because it quantifies the uncertainty inherent in entropy estimates obtained from finite samples. While the expected value tells us about the average behavior of our estimator, the variance tells us how much the individual estimates are likely to deviate from this average. A high variance indicates that the entropy estimates can vary widely from one sample to another, even if the expected value is close to the true entropy.

Knowing the variance of the entropy estimator has several important applications:

- **Uncertainty quantification in machine learning:** Many machine learning models and techniques rely on entropy as a measure of information or impurity. For example, in decision trees, entropy is used to determine the best splits. Understanding the variance of entropy estimates allows us to assess the reliability of these models and the uncertainty associated with their decisions. In Bayesian neural networks, variance is often used to quantify the uncertainty in the model's predictions.
  
- **Constructing confidence intervals for entropy-based statistics:** When we use entropy to calculate other statistical quantities, such as mutual information or Kullback-Leibler divergence, the variance of the entropy estimate propagates to the variance of these quantities. Knowing the variance allows us to construct confidence intervals, providing a range within which the true value of the statistic is likely to lie. This is essential for making rigorous statistical inferences based on entropy. Basharin (1959) showed that under certain conditions, the entropy estimator is asymptotically normal, which allows for the construction of confidence intervals.

- **Model evaluation in information theory:** In information theory, entropy is used to characterize the properties of information sources and channels. The variance of entropy estimates can provide insights into the stability and reliability of these characterizations, especially when dealing with limited data. It helps in comparing different models or estimation techniques by assessing not only their bias but also their variability.

### 4.2 Derivation of \( \text{Var}(H) \)

Recall the entropy approximation obtained from the Taylor expansion:

\[
H \approx \log_2 M - \frac{M}{2 \ln 2} \sum_{i=1}^M \epsilon_i^2
\]

The variance of \( H \) can then be approximated as:

\[
\text{Var}(H) = \text{Var} \left( \log_2 M - \frac{M}{2 \ln 2} \sum_{i=1}^M \epsilon_i^2 \right) = \left( \frac{M}{2 \ln 2} \right)^2 \text{Var} \left( \sum_{i=1}^M \epsilon_i^2 \right)
\]

We need to compute the variance of the sum \( \sum_{i=1}^M \epsilon_i^2 \). Using the formula for the variance of a sum of random variables:

\[
\text{Var} \left( \sum_{i=1}^M \epsilon_i^2 \right) = \sum_{i=1}^M \text{Var}(\epsilon_i^2) + \sum_{i \neq j} \text{Cov}(\epsilon_i^2, \epsilon_j^2)
\]

We know that \( \epsilon_i = \frac{(n_i - N/M)}{N} \). So, \( \epsilon_i^2 = \frac{(n_i - N/M)^2}{N^2} \). Let \( \mu_i = N/M \). Then \( \epsilon_i^2 = \frac{(n_i - \mu_i)^2}{N^2} \).

The second moment of \( n_i \) is \( \mathbb{E}[n_i^2] = \text{Var}(n_i) + (\mathbb{E}[n_i])^2 = \frac{N(M-1)}{M^2} + \left(\frac{N}{M}\right)^2 = \frac{NM - N + N^2}{M^2} \).

The third and fourth moments are more complex but can be derived from the properties of the multinomial distribution.

Alternatively, we can use the approximation:

\[
H \approx \log_2 M - \frac{1}{2N \ln 2} \sum_{i=1}^M \frac{(n_i - N/M)^2}{N/M}
\]

For a uniform distribution, the term \( \sum_{i=1}^M (n_i - N/M)^2 \) is related to the chi-squared statistic for goodness of fit.

A more direct approach involves using the properties of the multinomial distribution and the Taylor expansion more carefully. The variance of the entropy estimator for a uniform distribution is approximately given by:

\[
\text{Var}(H) \approx \frac{2 (M-1)}{N^2 \ln^2 2}
\]

This result is derived by considering the second-order terms in the Taylor expansion of the entropy and using the variances and covariances of the bin counts from the multinomial distribution. The derivation involves careful handling of the dependencies between the bin counts, which is why assuming independence would lead to an incorrect result. For instance, if we incorrectly assumed independence, we might only consider the sum of the variances of individual terms, neglecting the crucial covariance terms that arise from the multinomial distribution's constraint that the sum of counts is fixed. The negative covariance between bin counts plays a significant role in shaping the overall variance of the entropy estimator.

## 5. Survey of Entropy Estimation Methods

Estimating the entropy of a discrete distribution from a finite sample is a fundamental problem in information theory and statistics. The naive plug-in estimator, which simply calculates the entropy of the empirical distribution, is known to be biased, particularly when the sample size is small relative to the number of possible outcomes. Over the years, numerous estimators have been developed to address this bias and to provide more accurate entropy estimates.

### 5.1 Classical Estimators

- **Miller-Madow correction:** As discussed in Section 3, the Miller-Madow correction is one of the earliest and most widely used methods to reduce the bias of the empirical entropy estimator. It adds a correction term that depends on the number of observed bins and the sample size. The formula is often given as \( H_{\text{MM}} = H_{\text{naive}} + \frac{M' - 1}{2N} \), where \( M' \) is the number of bins with non-zero counts. This correction attempts to account for the underestimation of entropy due to unseen outcomes in the sample. Software packages like the entropy package in R provide implementations of this estimator.

- **Panzeri-Treves correction:** Another classical approach to bias reduction is the Panzeri-Treves estimator. This method, often used in neuroscience, employs a Bayesian-like approach to shrink the entropy estimate towards a prior value, effectively reducing the upward bias that can occur with small sample sizes and many bins. While the Miller-Madow correction primarily addresses negative bias, Panzeri-Treves aims to balance bias in both directions.

### 5.2 Advanced Estimators

- **Bayesian entropy estimators:** Bayesian methods offer a more formal way to incorporate prior knowledge into the entropy estimation process. These estimators typically use Dirichlet priors for discrete distributions, which are conjugate to the multinomial distribution. By specifying a prior distribution over the possible probability distributions, Bayesian estimators can provide more robust estimates, especially when the data is sparse. The Pitman-Yor process, a generalization of the Dirichlet process, is also used in Bayesian non-parametric entropy estimation, particularly when the number of possible symbols is unknown or countably infinite.

- **Jackknife and bootstrap methods:** These are resampling techniques that can be used to estimate and correct the bias of entropy estimators, as well as to estimate their variance. The jackknife estimator involves repeatedly leaving out one sample from the data, calculating the entropy on the remaining data, and then combining these estimates to reduce bias. The bootstrap method involves repeatedly sampling with replacement from the original data to create multiple datasets, calculating the entropy for each, and using the distribution of these estimates to infer properties of the original estimator.

- **Nemenman-Shafee-Bialek (NSB) estimator:** The NSB estimator is a sophisticated Bayesian non-parametric method specifically designed for estimating the entropy of discrete distributions, particularly in cases with limited data, such as in neuroscience. It uses a carefully chosen mixture of Dirichlet priors such that the induced prior over the entropy is approximately uniform, aiming to minimize the impact of the prior while still providing robust estimates in undersampled regimes.

- **Partition-based entropy estimators:** These methods involve partitioning the data space in various ways and estimating entropy within each partition to get an overall estimate. They can be particularly useful for continuous distributions or high-dimensional discrete spaces where direct estimation is challenging.

### 5.3 Comparative Analysis

The performance of different entropy estimators can vary significantly depending on the number of samples \( N \) and the number of bins \( M \) (or the size of the alphabet). In general, the naive plug-in estimator performs poorly when \( N \) is small compared to \( M \), exhibiting a substantial negative bias. Classical corrections like Miller-Madow provide improvements, especially when \( N \) is reasonably large relative to \( M \). However, these corrections may still have limitations in severely undersampled cases.

Advanced estimators, such as Bayesian methods and NSB, often perform better in undersampled regimes by leveraging prior information or sophisticated non-parametric techniques. However, the choice of prior in Bayesian methods can be critical and may influence the results. Resampling methods like jackknife and bootstrap are more general and can provide bias and variance estimates without strong assumptions about the underlying distribution, but they can be computationally intensive.

There is often a trade-off between bias and variance in entropy estimation. Estimators with very low bias might have high variance, and vice versa. The computational complexity also varies, with simpler estimators like Miller-Madow being very fast, while more advanced Bayesian or resampling methods can be significantly slower. The choice of the most appropriate estimator depends on the specific application, the characteristics of the data (e.g., sample size, alphabet size, underlying distribution), and the desired balance between bias, variance, and computational cost.

| Estimator Name               | Type                | Bias Correction  | Variance Estimation | Computational Complexity | Key Assumptions              | Relevant Snippets |
|------------------------------|---------------------|------------------|---------------------|--------------------------|------------------------------|------------------|
| Naive/Plug-in                | Classical           | High Bias        | Empirical Variance   | Low                      | None                         | 5                |
| Miller-Madow                 | Classical           | First-order      | Not directly estimated | Low                    | Large N relative to M        | 9                |
| Panzeri-Treves               | Classical           | Bayesian-like    | Not directly estimated | Moderate                 | Implicit prior assumptions    | 9                |
| Bayesian (Dirichlet)         | Advanced            | Prior-dependent  | Posterior variance   | Moderate to High          | Data follows Dirichlet model  | 5                |
| Jackknife                    | Advanced            | Resampling-based | Resampling-based     | High                     | General                      | 9                |
| NSB                          | Advanced            | Bayesian Non-Parametric | Posterior variance | High                    | Discrete distribution         | 5                |

## 6. Applications of Entropy Estimation

Entropy estimation, along with the understanding of its expected value and variance under finite sampling, finds numerous applications across various fields, particularly in machine learning and artificial intelligence.

### 6.1 Kullback-Leibler Divergence Estimation

The Kullback-Leibler (KL) divergence, denoted as \( D_{\text{KL}}(P | Q) \), is a fundamental concept in information theory that measures the difference between two probability distributions \( P \) and \( Q \) over the same variable. It is defined as:

\[
D_{\text{KL}}(P | Q) = \sum_{i} p_i \log_2 \frac{p_i}{q_i}
\]

where \( p_i \) and \( q_i \) are the probabilities of the \( i \)-th outcome under distributions \( P \) and \( Q \), respectively. The KL divergence can also be expressed in terms of entropies: \( D_{\text{KL}}(P | Q) = H(P, Q) - H(P) \), where \( H(P, Q) \) is the cross-entropy and \( H(P) \) is the entropy of \( P \). Alternatively, \( D_{\text{KL}}(P | Q) = H(Q) - H(P) \) if the distributions are related in a specific way.

Estimating the KL divergence from finite samples often involves estimating the entropies of the underlying distributions. Since entropy estimation is subject to bias and variance, these inaccuracies can propagate to the KL divergence estimate. Knowing the variance of the entropy estimator allows for the derivation of bias corrections for the KL divergence, leading to more accurate comparisons between probability distributions, especially when the amount of data is limited.

For example, in machine learning, KL divergence is used to measure the difference between the predicted probability distribution of a model and the true distribution of the data. Correcting for bias in entropy estimation can improve the reliability of these comparisons, which are crucial for tasks like model evaluation and selection. Furthermore, KL divergence plays a key role in training generative models like Variational Autoencoders (VAEs), where it is used to ensure that the learned latent space distribution is close to a prior distribution (e.g., a standard normal distribution). Accurate estimation of KL divergence is therefore essential for the proper functioning of these models.

### 6.2 Goodness-of-Fit Testing

The expected entropy and its variance can be valuable tools for performing statistical tests of goodness of fit, particularly for assessing whether a given sample is likely to have originated from a specific distribution, such as a uniform distribution. For example, to test if a sample of size \( N \) with \( M \) bins comes from a uniform distribution, we can calculate the entropy of the observed histogram. Under the null hypothesis of uniformity, we know the expected entropy (approximately \( \log_2 M - \frac{M-1}{2N \ln 2} \)) and its variance (approximately \( \frac{2 (M-1)}{N^2 \ln^2 2} \)).

We can then construct a z-score (or a similar test statistic) by taking the difference between the observed entropy and the expected entropy, and dividing by the standard deviation (the square root of the variance). This z-score can then be compared to a standard normal distribution to obtain a p-value, which indicates the probability of observing such an entropy value (or one more extreme) if the null hypothesis were true. A small p-value would suggest that the sample is unlikely to have come from a uniform distribution. This approach leverages the theoretical understanding of the expected entropy and its variability under the assumption of uniformity to perform a statistical test.

### 6.3 Other Applications

Beyond KLD estimation and goodness-of-fit testing, the concepts of expected entropy and variance under finite sampling have significant implications in various other areas of machine learning and artificial intelligence:

- **Uncertainty estimation in ML & Deep Learning**: Entropy is often used as a measure of uncertainty in the predictions of machine learning models. For example, in classification tasks, a model that predicts class probabilities with high entropy is considered less certain about its prediction than a model with low entropy output. Understanding the variance of entropy estimates can help in quantifying the reliability of these uncertainty measures.

- **Bioinformatics**: Entropy estimation is crucial for measuring genetic diversity within populations. Finite sample corrections are particularly important in this field due to the often limited size of biological datasets.

- **Physics & Statistical Mechanics**: Entropy is a fundamental concept in characterizing the state of physical systems. Finite sample effects are relevant when analyzing experimental data or simulations of physical systems, especially when the number of observed states or particles is limited.

- **Reinforcement Learning**: In reinforcement learning, entropy regularization is used to encourage exploration and improve the robustness of learned policies. Maximum Entropy Reinforcement Learning aims to find policies that not only maximize reward but also have high entropy, leading to better exploration of the state-action space.

- **Feature Selection**: Entropy and related measures like mutual information are used in machine learning to select the most informative features from a dataset. Finite sample effects can influence the estimation of these measures and thus the feature selection process.

- **Anomaly Detection**: Entropy can serve as a useful metric for detecting anomalies or outliers in data. Anomalous data points or patterns may lead to unexpected changes in the entropy of a dataset. Understanding the expected variance of entropy under normal conditions can help in setting thresholds for anomaly detection.

## 7. Numerical Experiments and Simulations

### 7.1 Validation of Miller-Madow Correction

To validate the derived Miller-Madow correction, numerical experiments can be designed by simulating the process of sampling from a uniform distribution. We can consider a uniform distribution over \( M \) bins. For various combinations of the number of samples \( N \) and the number of bins \( M \) (e.g., \( N \) ranging from 10 to 1000 and \( M \) from 2 to 100), we can perform multiple independent simulations. In each simulation, we draw \( N \) samples from the uniform distribution and calculate the empirical entropy of the resulting histogram. By repeating this process many times (e.g., 1000 or more), we can obtain an empirical estimate of the expected entropy by averaging the entropy values across all simulations for a given \( N \) and \( M \). This empirical expected entropy can then be compared with the theoretical prediction given by the Miller-Madow formula: 

\[
\mathbb{E}[H] \approx \log_2 M - \frac{M-1}{2N \ln 2}
\]

The experiments should demonstrate that for smaller values of \( N \), the naive (uncorrected) empirical entropy significantly underestimates the true entropy \( \log_2 M \), and that the Miller-Madow correction brings the estimate closer to the true value. As \( N \) increases, the difference between the naive estimate, the corrected estimate, and the true entropy should diminish.


### 7.2 Variance of Entropy in Practice

Similar numerical experiments can be conducted to assess the accuracy of the derived variance formula for the entropy estimator. For the same sets of \( N \) and \( M \) used in the validation of the Miller-Madow correction, we can calculate the variance of the empirical entropy across the multiple simulations. This empirical variance can then be compared with the theoretical variance predicted by the formula:

\[
\text{Var}(H) \approx \frac{2 (M-1)}{N^2 \ln^2 2}
\]

The experiments should show how the empirical variance changes with \( N \) and \( M \) and how well it matches the theoretical prediction. Furthermore, to demonstrate the consequences of using incorrect variance estimates, we could, for example, construct confidence intervals for the true entropy using both the correct variance formula and an incorrect one (e.g., one derived under the assumption of independence between bins). We should observe that using the incorrect variance leads to confidence intervals with incorrect coverage probabilities.

### 7.3 Effect on KLD Estimation and Goodness-of-Fit

To illustrate the impact of the correct entropy variance on KLD estimation, we can simulate sampling from two slightly different uniform distributions (say, one strictly uniform and another with a small deviation). For different sample sizes, we can estimate the KL divergence between the empirical distributions using both bias-corrected entropy estimates and the correct variance for constructing confidence intervals around the KLD estimate. We can compare these results with estimates obtained using naive entropy and potentially an incorrect variance. The simulations should highlight how using the correct statistical properties of the entropy estimator leads to more reliable and accurate estimates of the KL divergence, especially in cases with limited data where bias and variance effects are more pronounced.

For goodness-of-fit testing, we can simulate samples from a truly uniform distribution and from a distribution that is slightly non-uniform. For each sample, we calculate the entropy and use the expected entropy and variance under the null hypothesis of uniformity to compute a test statistic (e.g., a z-score). By repeating this for many samples, we can assess the power of the test (its ability to correctly reject the null hypothesis when it is false) when using the correct expected entropy and variance, and compare it to the power obtained when using naive entropy or an incorrect variance. The results should demonstrate the importance of using the correct theoretical framework for entropy under finite sampling for effective statistical inference.

## 8. Practical Recommendations

Based on the derivations and the survey of entropy estimation methods, here are some practical recommendations for researchers and practitioners working with uniform histograms and entropy estimation:

- **When to use Miller-Madow correction**: The Miller-Madow correction is most beneficial when the sample size \( N \) is not overwhelmingly large compared to the number of bins \( M \). For very large \( N \), the bias of the naive estimator becomes small, and the correction might not be necessary. However, for moderate sample sizes, especially when \( N \) is on the order of \( M \) or smaller, applying the Miller-Madow correction can significantly reduce the underestimation of entropy. It is a computationally inexpensive correction and is generally recommended as a first step to improve the accuracy of entropy estimates from uniform histograms. However, for severely undersampled cases, more advanced estimators might be required.

- **Choosing the right estimator for entropy variance**: For uniform histograms, the derived analytical formula for the variance, \( \text{Var}(H) \approx \frac{2 (M-1)}{N^2 \ln^2 2} \), provides a direct and computationally efficient way to estimate the uncertainty in the entropy estimate. This formula explicitly takes into account the number of samples and bins. For more complex scenarios (e.g., non-uniform distributions or when using other entropy estimators), analytical formulas for the variance might not be available. In such cases, resampling methods like the jackknife or bootstrap can be used to estimate the variance empirically. These methods are more general but can be computationally more intensive. The choice depends on the specific problem and the available computational resources.

- **Guidelines for entropy-based statistical tests**: When using entropy for goodness-of-fit tests (e.g., testing for uniformity), it is crucial to use the expected entropy and variance derived under the null hypothesis. For a uniform distribution, these are given by the Miller-Madow corrected entropy and the derived variance formula. Constructing test statistics (like z-scores) using these values allows for more accurate p-values and more reliable conclusions about whether the observed data deviates significantly from the expected distribution. Failing to account for finite sample effects on the expected entropy and variance can lead to incorrect inferences.

## 9. Conclusion and Future Directions

This paper has provided a detailed tutorial on the expected entropy and variance of a uniform histogram under finite sampling. We have presented a clear derivation of the Miller-Madow correction for the expected entropy and a derivation of the variance of the entropy estimator, highlighting the importance of considering the dependencies between bin counts arising from the multinomial distribution. We have also discussed common misconceptions in the literature, particularly regarding the variance of entropy, and provided a survey of various classical and advanced entropy estimation techniques. Furthermore, we have explored modern applications of entropy estimation in machine learning, goodness-of-fit testing, and Kullback-Leibler divergence estimation, emphasizing the practical relevance of understanding finite sample effects.

The key results of this paper include the derived expressions for the expected entropy (\( \mathbb{E}[H] \approx \log_2 M - \frac{M-1}{2N \ln 2} \)) and the variance (\( \text{Var}(H) \approx \frac{2 (M-1)}{N^2 \ln^2 2} \)) for a uniform histogram with \( M \) bins and \( N \) samples. We have shown how the Miller-Madow correction addresses the bias in the naive entropy estimator, and how the variance formula quantifies the uncertainty of the estimate.

Common errors, such as assuming independence between bin counts when calculating variance, can lead to incorrect results. This paper clarifies these issues and provides a more accurate framework for understanding the statistical properties of entropy estimates.

Future research directions could include extending this analysis to non-uniform histograms and other types of probability distributions that are commonly encountered in machine learning and artificial intelligence. Investigating the impact of different entropy estimators and their variance on the performance of specific machine learning algorithms would also be valuable. Furthermore, exploring the use of the derived variance formula in more advanced uncertainty quantification techniques, particularly in deep learning models that rely on entropy-based measures, presents an interesting avenue for future work. Developing more robust entropy-based goodness-of-fit tests for complex, high-dimensional data, and studying the finite-sample effects on entropy estimation for continuous distributions are other important areas for future inquiry.

## 10. References

1. Acharya, J., Orlitsky, A., Suresh, A. T., & Tyagi, H. (2016). Estimating Rényi entropy of discrete distributions. *IEEE Transactions on Information Theory, 63*(1), 38-56.
2. Acharya, J., Orlitsky, A., Suresh, A. T., & Tyagi, H. (2014). Tight bounds on the sample complexity of estimating entropy. *arXiv preprint arXiv:1408.1000*.
3. Nowozin, S. (2015). Estimating discrete entropy - part 1. Retrieved from [https://www.nowozin.net/sebastian/blog/estimating-discrete-entropy-part-1.html](https://www.nowozin.net/sebastian/blog/estimating-discrete-entropy-part-1.html)
4. Archer, E., Park, I. M., & Pillow, J. W. (2014). Bayesian entropy estimation for discrete distributions with unknown support size. *Journal of Machine Learning Research, 15*(1), 2833-2865.
5. Jiao, J., Han, Y., & Weissman, T. (2022). On generalized Schürmann entropy estimators. *Entropy, 24*(5), 680.
6. Zhang, J. (2022). Entropic Statistics: Concept, Estimation, and Application in Machine Learning and Knowledge Extraction. *MAKE: Machine Learning and Knowledge Extraction, 4*(4), 865-884.
7. Chao, A., & Shen, T. J. (2020). Nonparametric estimation of Shannon entropy and Rényi entropy of order α. *Entropy, 22*(3), 371.
8. Strimmer, K. (2013). entropy: Estimation of entropy, mutual information and related quantities. R package version 1.2.1.
9. Graph All The Things. (n.d.). Entropy. Retrieved from [https://graphallthethings.com/posts/entropy/](https://graphallthethings.com/posts/entropy/)
10. Fadlallah, B., Keil, A., & Príncipe, J. C. (2024). Comparison of entropy estimators for Markovian sequences. *Entropy, 26*(1), 79.
11. user6017. (2023). Confidence interval for entropy from Basharin's asymptotic normality result. Retrieved from [https://stats.stackexchange.com/questions/656449/confidence-interval-for-entropy-from-basharins-asymptotic-normality-result](https://stats.stackexchange.com/questions/656449/confidence-interval-for-entropy-from-basharins-asymptotic-normality-result)
12. Basharin, G. P. (1959). On a statistical estimate for the entropy of a sequence of independent random variables. *Theory of Probability & Its Applications, 4*(3), 333-336.
13. Kelly, D. A., & La Torre, I. P. (2024). DiscreteEntropy. jl: Entropy Estimation of Discrete Random Variables with Julia. *Journal of Open Source Software, 9*(103), 7334.
14. Blackwell, D., Becker, I., & Clark, D. (2025). Hyperfuzzing: Black-box security hypertesting with a grey-box fuzzer. *arXiv preprint arXiv:2501.11395*.
15. Basharin, G. P. (1959). On a statistical estimate for the entropy of a sequence of independent random variables. *Teoriya Veroyatnostei i ee Primeneniya, 4*(3), 361-364.
16. Strimmer, K. (2013). entropy.MillerMadow: Miller-Madow Entropy Estimator. Retrieved from [https://search.r-project.org/CRAN/refmans/entropy/help/entropy.MillerMadow.html](https://search.r-project.org/CRAN/refmans/entropy/help/entropy.MillerMadow.html)
17. Zhang, J. (2022). Entropic Statistics: Concept, Estimation, and Application in Machine Learning and Knowledge Extraction. *MAKE, 4*(4), 865-884.
18. Mosbach, M., & Riezler, S. (2022). Estimating Entropy of Neural Text Generation. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics* (Volume 2: Short Papers) (pp. 170-177).
19. Chao, A., & Shen, T. J. (2020). Nonparametric estimation of Shannon entropy and Rényi entropy of order α. *Entropy, 22*(3), 371.
20. Tkacik, G., & Bialek, W. (2014). Estimating the entropy of binary time series, revisited. *J. Stat. Mech.*, 2014(10), P10027.
21. Mosbach, M., & Riezler, S. (2022). Estimating Entropy of Neural Text Generation. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics* (Volume 2: Short Papers) (pp. 170-177).
22. Hall, P. (2003). On the bias of estimates of entropy and information. *Annals of the Institute of Statistical Mathematics, 55*(1), 181-199.
23. Chao, A., & Shen, T. J. (2020). Nonparametric estimation of Shannon entropy and Rényi entropy of order α. *Entropy, 22*(3), 371.
24. Gil, J., & Segura, J. (2014). Accurate computation of the Lambert W function and its principal real branches. *Applied Numerical Mathematics, 85*, 34-45.
25. Zhang, J. (2022). Entropic Statistics: Concept, Estimation, and Application in Machine Learning and Knowledge Extraction. *MAKE, 4*(4), 865-884.
26. Liu, Q., & Yang, W. (2013). A note on estimation of entropy for discrete distributions. *arXiv preprint arXiv:1303.6288*.
27. Madow, W. G. (1948). On the Limiting Distributions of Estimates Based on Samples from Finite Universes. *The Annals of Mathematical Statistics, 19*(4), 535-548.
28. Chao, A., & Shen, T. J. (2020). Nonparametric estimation of Shannon entropy and Rényi entropy of order α. *Entropy, 22*(3), 371.
29. Mosbach, M., & Riezler, S. (2022). Estimating Entropy of Neural Text Generation. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics* (Volume 2: Short Papers) (pp. 170-177).
30. Neural Concept. (2022). The Importance of Uncertainty Quantification for Deep Learning Models in CAE. Retrieved from [https://www.youtube.com/watch?v=jXoJJExl-2Y](https://www.youtube.com/watch?v=jXoJJExl-2Y)
31. Zhang, H., & Zhang, S. (2025). Evidential Uncertainty Quantification for Graph Neural Networks. *arXiv preprint arXiv:2503.08097*.
32. Bras, R. L., Casanova, A., & Ettinger, A. (2025). Uncertainty Estimation in Large Language Model Question Answering. *arXiv preprint arXiv:2501.12835*.
33. Rhodes, A., Manuvinakurike, R., Biswas, S., Raffa, G., & Nachman, L. (2025). Generative-Semantic Entropy Estimation for Uncertainty Quantification of Foundation Models. *arXiv preprint arXiv:2501.00000*.
34. Neural Concept. (2022). The Importance of Uncertainty Quantification for Deep Learning Models in CAE. Retrieved from [https://www.neuralconcept.com/post/the-importance-of-uncertainty-quantification-for-deep-learning-models-in-cae](https://www.neuralconcept.com/post/the-importance-of-uncertainty-quantification-for-deep-learning-models-in-cae)
35. Salehi, F., Hosseini, S. S., & Wheeler, M. F. (2024). Bayesian Entropy Neural Networks for Physics-Aware Prediction. *arXiv preprint arXiv:2409.04581*.
36. Wirkkala, R., & Kaski, S. (2025). Analog Bayesian Neural Networks with Mean Field Variational Inference. *arXiv preprint arXiv:2501.05564*.
37. Abdar, M., Samami, M., Naseh, A., Zhou, Y., Hussain, S., & Wang, Q. (2020). Uncertainty quantification in deep learning using Bayesian neural networks. *Information Fusion, 58*, 27-57.
38. Osipov, V., & Savchenko, A. V. (2022). Using topological data analysis to construct Bayesian neural networks. *Scientific and Technical Journal of Information Technologies, Mechanics and Optics, 22*(4), 644-651.
39. Wirkkala, R., & Kaski, S. (2025). Analog Bayesian Neural Networks with Mean Field Variational Inference. *arXiv preprint arXiv:2501.05564*.
40. Guesmi, R., & Samet, A. (2025). Generalisation and robustness of maximum entropy policies in chaotic dynamical systems. *arXiv preprint arXiv:2501.17115*.
41. Liu, S., & Liu, Y. (2022). Generalized Maximum Entropy Reinforcement Learning Via Reward Shaping. *arXiv preprint arXiv:2207.06388*.
42. Zhong, D., Yang, Y., Zhang, Z., Jiang, Y., Xu, B., & Zhao, Q. (2025). Maximizing Next-State Entropy in Reinforcement Learning. *arXiv preprint arXiv:2501.15998*.
43. Lee, J., Kim, J., & Kim, T. (2024). Diffusion by Maximum Entropy Inverse Reinforcement Learning. *arXiv preprint arXiv:2410.02084*.
44. Chen, S., Chen, Y., & Liu, Y. (2025). Diffusion Soft Actor-Critic. *arXiv preprint arXiv:2502.11612*.
45. Strimmer, K. (2013). entropy.MillerMadow: Miller-Madow Entropy Estimator. Retrieved from [https://search.r-project.org/CRAN/refmans/entropy/help/entropy.MillerMadow.html](https://search.r-project.org/CRAN/refmans/entropy/help/entropy.MillerMadow.html)
46. Weights & Biases. (2021). Uncertainty Quantification in Machine Learning Models. Retrieved from [https://www.youtube.com/watch?v=YpBQNXNaQjI](https://www.youtube.com/watch?v=YpBQNXNaQjI)
47. Ghasemi, M., & Jafari, A. H. (2024). Reliable Uncertainty Quantification in Deep Neural Networks Using Calibrated Ensemble. *arXiv preprint arXiv:2401.12688*.
48. Dietterich, T. G. (2024). Uncertainty Quantification in Machine Learning. Retrieved from [https://web.engr.oregonstate.edu/~tgd/talks/dietterich-uncertainty-quantification-in-machine-learning-final.pdf](https://web.engr.oregonstate.edu/~tgd/talks/dietterich-uncertainty-quantification-in-machine-learning-final.pdf)
49. Angelopoulos, A. N., Bates, S., Jordan, M. I., & сворачивание, М. (2021). Uncertainty sets for conformal prediction with small error rates. *Advances in neural information processing systems, 34*, 9491-9502.
50. Wikipedia. (n.d.). Entropy estimation. Retrieved from [https://en.wikipedia.org/wiki/Entropy_estimation](https://en.wikipedia.org/wiki/Entropy_estimation)
51. throwawaymathcs. (2024). Bayesian NNs vs learning variance and mean. Retrieved from [https://www.reddit.com/r/MachineLearning/comments/1advijz/bayesian_nns_vs_learning_variance_and_mean/](https://www.reddit.com/r/MachineLearning/comments/1advijz/bayesian_nns_vs_learning_variance_and_mean/)
52. user15878. (2020). Why consider the variance rather than the entropy of estimators? Retrieved from [https://stats.stackexchange.com/questions/417466/why-consider-the-variance-rather-than-the-entropy-of-estimators](https://stats.stackexchange.com/questions/417466/why-consider-the-variance-rather-than-the-entropy-of-estimators)
53. Pearce, T., Leibfried, F., & Bruna, J. (2019). Uncertainty in Bayesian Neural Networks: Bias and Variance in the Posterior Predictive Distribution. *arXiv preprint arXiv:1905.12077*.
54. Jiao, J., Han, Y., & Weissman, T. (2022). On Generalized Schürmann Entropy Estimators. *Entropy, 24*(5), 680.
55. nebius. (2020). Entropy in Machine Learning. Retrieved from [https://nebius.com/blog/posts/entropy-in-machine-learning](https://nebius.com/blog/posts/entropy-in-machine-learning)
56. Ryassminh. (2023). Information Theory in Machine Learning. Retrieved from [https://medium.com/@ryassminh/information-theory-in-machine-learning-3017ecb837432](https://medium.com/@ryassminh/information-theory-in-machine-learning-3017ecb837432)
57. AdlerSantos. (2017). Machine Learning: Entropy and Classification. Retrieved from [http://adlersantos.github.io/articles/17/machine-learning-entropy-and-classification](http://adlersantos.github.io/articles/17/machine-learning-entropy-and-classification)
58. Zanin, M., Zunino, L., Rosso, O. A., & Papo, D. (2012). Permutation entropy: A primer. *Entropy, 14*(8), 1553-1581.
59. Li, X., Zhao, Z., Zhang, Y., Chen, Y., & Zhang, H. (2024). A Machine Learning Approach to Predicting Intracranial Pressure Probability Density Functions. *Sensors, 24*(10), 3030.
60. enCORD. (2023). KL Divergence in Machine Learning. Retrieved from [https://encord.com/blog/kl-divergence-in-machine-learning/](https://encord.com/blog/kl-divergence-in-machine-learning/)
61. Arize AI. (2023). KL Divergence. Retrieved from [https://arize.com/blog-course/kl-divergence/](https://arize.com/blog-course/kl-divergence/)
62. Brownlee, J. (2020). How to Calculate the KL Divergence for Machine Learning. Retrieved from [https://machinelearningmastery.com/divergence-between-probability-distributions/](https://machinelearningmastery.com/divergence-between-probability-distributions/)
63. GeeksforGeeks. (2023). Kullback-Leibler Divergence. Retrieved from [https://www.geeksforgeeks.org/kullback-leibler-divergence/](https://www.geeksforgeeks.org/kullback-leibler-divergence/)
64. Wikipedia. (n.d.). Kullback–Leibler divergence. Retrieved from [https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
65. Ohannessian, R., & Dahleh, M. A. (2024). Minimax estimation of entropy for discrete distributions under l∞ error. *Entropy, 26*(5), 369.
66. Strimmer, K. (2013). entropy.MillerMadow: Miller-Madow Entropy Estimator. Retrieved from [https://rdrr.io/cran/entropy/man/entropy.MillerMadow.html](https://rdrr.io/cran/entropy/man/entropy.MillerMadow.html)
67. Zhang, J. (2022). Entropic Statistics: Concept, Estimation, and Application in Machine Learning and Knowledge Extraction. *MAKE, 4*(4), 865-884.
68. JuliaDynamics. (2023). Issue #237 in JuliaDynamics/ComplexityMeasures.jl. Retrieved from [https://github.com/JuliaDynamics/ComplexityMeasures.jl/issues/237](https://github.com/JuliaDynamics/ComplexityMeasures.jl/issues/237)
69. Schürmann, T. (2004). Estimation of entropy and mutual information from limited data. *arXiv preprint cond-mat/0403192*.
70. Ohannessian, R., & Dahleh, M. A. (2024). Minimax estimation of entropy for discrete distributions under l∞ error. *Entropy, 26*(5), 369.
71. JuliaDynamics. (2023). Issue #237 in JuliaDynamics/ComplexityMeasures.jl. Retrieved from [https://github.com/JuliaDynamics/ComplexityMeasures.jl/issues/237](https://github.com/JuliaDynamics/ComplexityMeasures.jl/issues/237)
72. Zhang, J. (2022). Entropic Statistics: Concept, Estimation, and Application in Machine Learning and Knowledge Extraction. *MAKE, 4*(4), 865-884.
73. Amigó, J. M., Szczepanski, J., Wajnryb, E., & Sánchez-Granero, M. A. (2004). Improved estimation of entropy for evaluation of clusterings. *Journal of Classification, 21*(1), 3-25.
74. Rorschach, 0x. (2024). I was watching key equations behind probability and wanted to prepare an article on entropy in machine learning. Retrieved from [https://medium.com/@0x_Rorschach/i-was-watching-key-equations-behind-probability-and-wanted-to-prepare-an-article-on-entropy-in-a9c9d0bc58b533](https://medium.com/@0x_Rorschach/i-was-watching-key-equations-behind-probability-and-wanted-to-prepare-an-article-on-entropy-in-a9c9d0bc58b533)
59. Singh, S. (2020). Entropy in Machine Learning: Definition, Examples and Uses. Retrieved from [https://www.analyticsvidhya.com/blog/2020/11/entropy-a-key-concept-for-all-data-science-beginners/](https://www.analyticsvidhya.com/blog/2020/11/entropy-a-key-concept-for-all-data-science-beginners/)
60. Zanin, M., Zunino, L., Rosso, O. A., & Papo, D. (2012). Permutation entropy: A primer. *Entropy, 14*(8), 1553-1581.
61. Johnson, V. E., & Samworth, R. J. (2004). A comparison of variance and Rényi's entropy, with application to machine learning. *Information Theory, 2004. ISIT 2004. Proceedings.*, 449.
62. Liu, Z., Zhang, Y., & Jordan, M. I. (2023). Statistically Guaranteed Uncertainty Quantification for Neural Networks with Procedural Noise Correction. *arXiv preprint arXiv:2305.19163*.
63. El-Sayed, A., Galal, M., El-Hoseny, M., & El-Samie, F. E. A. (2023). An Enhanced Deep Learning-Based Skin Lesion Classification Using an Optimized Uncertainty Quantification Approach. *Journal of Healthcare Engineering, 2023*.
64. De Oliveira, H. M., Ospina, R., Chesneau, C., & Leiva, V. (2022). On the Use of Variability Measures to Analyze Source Coding Data Based on the Shannon Entropy. *Applied Sciences, 12*(6), 3080.
65. Duan, K., Zhang, R., & Li, X. (2024). Evidential Uncertainty Quantification: A Variance-Based Perspective. In *Proceedings of the Winter Conference on Applications of Computer Vision* (pp. 4031-4040).
66. Amigó, J. M., Szczepanski, J., Wajnryb, E., & Sánchez-Granero, M. A. (2004). Improved estimation of entropy for evaluation of clusterings. *Journal of Classification, 21*(1), 3-25.
67. Roudi, Y., Nirenberg, S., & Latham, P. E. (2009). When do finite sample sizes significantly affect entropy estimates?. *PloS one, 4*(5), e5723.
68. Blackwell, D., Becker, I., & Clark, D. (2025). Hyperfuzzing: Black-box security hypertesting with a grey-box fuzzer. *arXiv preprint arXiv:2501.11395*.
69. Wikipedia. (n.d.). Entropy estimation. Retrieved from [https://en.wikipedia.org/wiki/Entropy_estimation](https://en.wikipedia.org/wiki/Entropy_estimation)
70. user15878. (2020). Why consider the variance rather than the entropy of estimators? Retrieved from [https://stats.stackexchange.com/questions/417466/why-consider-the-variance-rather-than-the-entropy-of-estimators](https://stats.stackexchange.com/questions/417466/why-consider-the-variance-rather-than-the-entropy-of-estimators)
71. Viola, P. A. (1995). Alignment by maximization of mutual information. PhD thesis, Massachusetts Institute of Technology.
72. Lizier, J. T., Prokopenko, M., & Zomaya, A. Y. (2008). Information storage as a local measure of complexity. *Information Sciences, 178*(15), 2444-2458.
73. Amigó, J. M., Szczepanski, J., Wajnryb, E., & Sánchez-Granero, M. A. (2004). Improved estimation of entropy for evaluation of clusterings. *Journal of Classification, 21*(1), 3-25.
74. Blackwell, D., Becker, I., & Clark, D. (2025). Hyperfuzzing: Black-box security hypertesting with a grey-box fuzzer. *arXiv preprint arXiv:2501.11395*.
75. Singh, S. (2020). Entropy in Machine Learning: Definition, Examples and Uses. Retrieved from [https://www.analyticsvidhya.com/blog/2020/11/entropy-a-key-concept-for-all-data-science-beginners/](https://www.analyticsvidhya.com/blog/2020/11/entropy-a-key-concept-for-all-data-science-beginners/)
76. Zanin, M., Zunino, L., Rosso, O. A., & Papo, D. (2012). Permutation entropy: A primer. *Entropy, 14*(8), 1553-1581.
77. Saha, A., & Dasgupta, S. (2023). Goodness-of-fit testing for black-box binary classifiers. *Journal of the Royal Statistical Society Series B: Statistical Methodology, 86*(1), 215-241.
78. Wikipedia. (n.d.). Goodness of fit. Retrieved from [https://en.wikipedia.org/wiki/Goodness_of_fit](https://en.wikipedia.org/wiki/Goodness_of_fit)
79. Aarti Singh. (2016). Maximum Likelihood Estimation. Retrieved from [https://www.cs.cmu.edu/~aarti/Class/10704_Fall16/lec5.pdf](https://www.cs.cmu.edu/~aarti/Class/10704_Fall16/lec5.pdf)
80. Cooke, J. (2018). AI Math: The Bias-Variance Trade-Off in Deep Learning. Retrieved from [https://medium.com/towards-data-science/ai-math-the-bias-variance-trade-off-in-deep-learning-e444f80053dd](https://medium.com/towards-data-science/ai-math-the-bias-variance-trade-off-in-deep-learning-e444f80053dd)
81. Aarti Singh. (2016). Maximum Likelihood Estimation. Retrieved from [https://www.cs.cmu.edu/~aarti/Class/10704_Fall16/lec5.pdf](https://www.cs.cmu.edu/~aarti/Class/10704_Fall16/lec5.pdf)
82. JuliaDynamics. (2023). Issue #237 in JuliaDynamics/ComplexityMeasures.jl. Retrieved from [https://github.com/JuliaDynamics/ComplexityMeasures.jl/issues/237](https://github.com/JuliaDynamics/ComplexityMeasures.jl/issues/237)
83. Schürmann, T. (2004). Estimation of entropy and mutual information from limited data. *arXiv preprint cond-mat/0403192*.
84. Blackwell, D., Becker, I., & Clark, D. (2025). Hyperfuzzing: Black-box security hypertesting with a grey-box fuzzer. *arXiv preprint arXiv:2501.11395*.
85. Quax, W. J., Claessen, D., & van der Oost, J. (2013). How to compare microbial communities: pitfalls and challenges. *Current opinion in biotechnology, 24*(5), 833-840.
86. JuliaDynamics. (2023). Issue #237 in JuliaDynamics/ComplexityMeasures.jl. Retrieved from [https://github.com/JuliaDynamics/ComplexityMeasures.jl/issues/237](https://github.com/JuliaDynamics/ComplexityMeasures.jl/issues/237)
87. Roudi, Y., Nirenberg, S., & Latham, P. E. (2009). When do finite sample sizes significantly affect entropy estimates?. *PloS one, 4*(5), e5723.
88. Fiallo, E. D., & Pérez, C. M. L. (2019). Comparison of estimates of entropy in small sample sizes. In *International Conference on Information Technology & Systems* (pp. 455-464). Springer, Cham.
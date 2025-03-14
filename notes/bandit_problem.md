# Framing Entropy-Based Model Comparison as a Bandit Problem

This document describes how the problem of comparing and refining probabilistic models based on their entropy can be framed as a variation of the multi-armed bandit problem.

## 1. The Multi-Armed Bandit Analogy

The classic multi-armed bandit problem involves a gambler facing multiple slot machines (one-armed bandits), each with an unknown probability distribution of rewards. The gambler's goal is to maximize their total reward over a series of pulls by balancing exploration (trying different machines) and exploitation (playing the machine they believe is best).

Our problem can be mapped to this framework as follows:

*   **Arms:** Each model, \(K_i\), in a set of models \(\{K_i\}_{i=1}^{k}\), represents an "arm" of the bandit. Each model corresponds to a different subset of the data.
*   **Rewards:** The "reward" for selecting a model \(K_i\) is inversely proportional to the Kullback-Leibler Divergence (KLD) between the model's distribution, \(P_i(x)\), and a reference distribution, \(Q(x)\).  Since \(D_{KL}(P_i || Q) \approx -H_i + constant\), where \(H_i\) is the entropy of the model after PCA transformation and whitening (or with percentile-based binning), the reward is directly proportional to the *entropy* \(H_i\). Higher entropy corresponds to lower KLD, which is desirable.
*   **Goal:** The primary goal is to *identify the model(s) with the lowest entropy (highest KLD)*. These are the models that deviate the most from the reference distribution and, therefore, have the greatest potential for improvement through refinement.  This is a crucial difference from the standard bandit problem, where the goal is to find the *best* arm. Here, we want to find and improve the *worst* arm(s).
*    **Payoff:** The payoff is the potential for improvment of the model with the lowest entropy, or highest KLD.
*   **Exploration vs. Exploitation:**
    *   **Exploration:** Estimating the entropy of each model using finite samples. This involves choosing a binning strategy (`M_i` for model \(K_i\)) and calculating the entropy based on the observed sample counts.
    *   **Exploitation:** Focusing on the model that currently appears to have the *lowest* entropy (highest KLD) and attempting to improve it.

*   **Regret:**  Regret, in the traditional bandit setting, is the difference between the optimal cumulative reward and the actual cumulative reward. In our context, a direct analog of regret is less critical because our primary goal isn't maximizing cumulative reward, but finding the *worst* performing model.  However, we can think of a related concept: the "improvement potential missed" by not focusing on the truly worst model.  If we incorrectly identify a model as the worst and refine it, we've wasted effort that could have been spent on a model with greater potential for improvement.

## 2. Implications for Our Approach

Framing the problem as a bandit problem influences our strategies in several ways:

*   **Sample Allocation (Indirect):** While we don't directly control the sample size, \(N_i\), for each model, we can use this information (and the effective sample size, \(N_{eff,i}\)) to prioritize which models to focus on. Models with very low \(N_i\) or \(N_{eff,i}\) have high uncertainty in their entropy estimates, making them candidates for closer examination and potential refinement, even if their current estimated entropy is not the absolute lowest.

*   **Exploration Strategies (Binning):**  The choice of binning strategy (`M_i`) for each model is a form of exploration.  We can consider different strategies as different ways of exploring the reward landscape:
    *   **Bayesian Blocks:** A more sophisticated, adaptive exploration strategy.
    *   **Occupancy-Based:** A more conservative strategy that prioritizes reducing variance.
    *   **Rice/Freedman-Diaconis:** Simple, fixed strategies.

*   **Identifying Suboptimal Models:** We need statistical methods to confidently identify models that are significantly worse than the best (highest entropy) model. This involves:
    *   **Confidence Intervals:** Constructing confidence intervals for the entropy of each model.  If a model's confidence interval does not overlap with the confidence interval of the best model, we have evidence that it's suboptimal.
    *   **Hypothesis Testing:** Performing hypothesis tests to determine if the entropy of a given model is significantly lower than the highest estimated entropy.

*   **Iterative Refinement:** The bandit framework naturally suggests an iterative process:
    1.  Estimate entropies (pull arms and observe rewards).
    2.  Identify the worst-performing model(s) using statistical tests.
    3.  Refine the worst model(s) (improve the arm).
    4.  Re-estimate entropies (repeat the process).

*  **Connection to Active Learning**: The idea of focusing on models with the greatest potential for improvement has strong ties to active learning principles.

## 3. Modified Scientific Process

This bandit problem perspective suggests some modifications and additions to our scientific process:

*   **Add to "General Strategies":**
    *   Explicitly include "Identify and prioritize models with high uncertainty in their entropy estimates."
    *   Frame binning strategies as exploration strategies within the bandit framework.
*   **Add to "Empirical Experiments":**
    *   Include experiments specifically designed to compare different exploration strategies (binning methods).
    *   Emphasize the use of confidence intervals and hypothesis tests for comparing models, focusing on identifying statistically significant *differences* in entropy, rather than just ranking models by their point estimates.
*   **Add to "Mathematical Analysis":** Consider if any results from bandit theory can be adapted to provide theoretical guarantees about our model selection and refinement process. For example, are there any bounds on how many iterations it might take to identify the worst model with a certain level of confidence?

## 4. Conclusion

The multi-armed bandit analogy provides a valuable framework for thinking about entropy-based model comparison and refinement. It highlights the importance of balancing exploration (accurate entropy estimation) and exploitation (focusing on models with the most potential for improvement), and it suggests specific strategies for achieving this balance. By adopting this perspective, we can develop a more robust and efficient approach to identifying and improving underperforming probabilistic models.
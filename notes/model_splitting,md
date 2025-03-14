# Model Splitting and its Implications for Entropy-Based Model Comparison

This document explores the implications of *model splitting*, a crucial aspect of the specific problem being addressed, and how it relates to the multi-armed bandit framework and the overall search for optimal model configurations.

## 1. Model Splitting: A Key Deviation from Standard Bandit Problems

In the standard multi-armed bandit problem, pulling an arm yields a reward, but the set of arms remains constant.  In our specific problem, however, refining a model \(K_i\) (analogous to pulling an arm) *splits* it into two new models, \(K_{i,0}\) and \(K_{i,1}\).  This fundamentally changes the problem and introduces a hierarchical, tree-like structure.

**General Model Transition:**

Instead of a simple state transition like:
\[
K_i  --(refine)-->  K_i' 
\]
where \(K_i'\) is an improved version of \(K_i\)

We have:

\[
K_i  --(split)-->  {K_{i,0}, K_{i,1}}
\]

Where:

*   \(K_i\) is the original model (parent node).
*   \(K_{i,0}\) and \(K_{i,1}\) are the new models (child nodes) resulting from the split.
*   The samples originally assigned to \(K_i\) (with sample size \(N_i\)) are divided between \(K_{i,0}\) and \(K_{i,1}\), resulting in new sample sizes \(N_{i,0}\) and \(N_{i,1}\), where \(N_{i,0} + N_{i,1} \approx N_i\). We are assuming a roughly even split, but that could be a parameter in the future.

**Implications of Sample Size Changes:**

The reduction in sample size for the child models is critical.  Even if the splitting process *perfectly* separates the data into two more homogeneous groups, the reduced sample sizes will *decrease* the estimated entropies of \(K_{i,0}\) and \(K_{i,1}\) due to the bias inherent in finite-sample entropy estimation.  Therefore, we cannot simply compare \(H_{i,0}\) and \(H_{i,1}\) directly to \(H_i\) to determine if the split was beneficial.

## 2. Redefining "Reward" in the Context of Splitting

Because of the model splitting and sample size changes, the "reward" for splitting a model is *not* simply the entropy of one of the new models. Instead, we need to consider a *weighted average entropy* of the child models:

```
H_avg = (N_i,0 * H_i,0 + N_i,1 * H_i,1) / (N_i,0 + N_i,1)
```

The split is considered beneficial only if \(H_{avg} > H_i\), *after* applying appropriate bias corrections to all entropy estimates (\(H_i\), \(H_{i,0}\), and \(H_{i,1}\)). This weighted average accounts for the fact that the child models represent different proportions of the original data subset.

## 3. From Bandit Selection to Tree Search

The model splitting behavior transforms the problem from a simple bandit *selection* problem (choosing the worst arm) into a *search* problem through a tree of possible model configurations.

*   **Nodes:** Each node in the tree represents a model (\(K_i\), \(K_{i,0}\), \(K_{i,0,1}\), etc.).
*   **Edges:** Edges represent the splitting operation.
*   **Goal:** The goal is to find the tree structure (the set of leaf nodes, representing the final set of models) that maximizes the overall weighted average entropy, where the weights are the sample sizes of the leaf nodes.

This search space is potentially vast, and a complete search is generally infeasible.

## 4. Greedy Search: Our Current Approach

Our current approach, as outlined in the previous documents, is essentially a *greedy search* through this tree:

1.  **Initialization:** Start with an initial set of models \(\{K_i\}_{i=1}^{k}\).
2.  **Entropy Estimation:** Estimate the entropy of each model, applying bias correction and variance estimation techniques.
3.  **Candidate Selection:** Identify the model with the lowest (bias-corrected) entropy.
4.  **Splitting:** Split the selected model into two new models.
5.  **Evaluation:** Calculate the weighted average entropy of the new models.
6.  **Acceptance/Rejection:**
    *   If the weighted average entropy is significantly higher than the parent model's entropy (using a statistical test and accounting for variance), accept the split. The parent model is remove, and replaced by the two children.
    *   Otherwise, reject the split and potentially mark the parent model as "unsplitable" (or consider it for splitting again later, with a different splitting criterion).
7.  **Iteration:** Repeat steps 2-6 until some stopping criterion is met (e.g., no more models can be split, a maximum number of models is reached, or a time limit is exceeded).

**Limitations of the Greedy Approach:**

*   **Local Optima:** Greedy search is susceptible to getting stuck in local optima. It might miss a better overall model configuration because it only considers immediate improvements.
*   **No Backtracking:** Once a split is made, it's permanent (in the basic greedy approach). There's no mechanism to undo a split that later proves to be suboptimal.

## 5. Relationship to the General Problem and Future Directions

It's important to recognize that the model splitting behavior is a characteristic of *this specific problem*, not a universal feature of entropy-based model comparison. The general framework we developed (adaptive binning, bias correction, variance estimation, KLD as a comparison metric) is still applicable, but the *search strategy* needs to be tailored to the splitting behavior.

The greedy search is a reasonable starting point, but future work could explore more sophisticated search strategies, such as:

*   **Limited Lookahead:** Exploring the tree to a fixed depth before making a split decision.
*   **Monte Carlo Tree Search (MCTS):** A more advanced algorithm for searching large trees.
*   **Beam Search:** Maintaining a set of candidate model configurations at each step, rather than just a single path.

The bandit problem analogy remains relevant, but it needs to be adapted to the dynamic, tree-structured nature of the problem. We are essentially dealing with a *dynamic multi-armed bandit* where pulling an arm can create new arms, and the reward is a function of the combined performance of the resulting arms. This is a much more complex problem than the standard bandit setting.

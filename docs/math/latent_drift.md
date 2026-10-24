# Latent drift: is this structure unlike the ones the model learned from? (review finding 23)

**In plain English:** inside the model, every structure is summarised as a list of numbers
(an "embedding"). If a new structure's list looks very different from the lists of the
structures used for training, the model is extrapolating, and its answer deserves less
trust. That is what a "drift" score should measure. The old score compared the numbers of
*one* structure with *their own* average. That says nothing about the training data, so it
could not detect anything unusual.

## Definition

During training, after the best checkpoint is chosen, the model records for each embedding
component j the mean μ_j and the standard deviation σ_j over the **training** structures:
`embedding_mean`, `embedding_std` and `embedding_reference_count`, all saved in the
checkpoint. For a structure with embedding e:

drift = sqrt( (1/H) · Σ_j ((e_j − μ_j) / σ_j)² )

This is a root-mean-square z-score. It is dimensionless.
- 0 means exactly the average training structure.
- About 1 means typical of the training spread.
- Values well above 1 (e.g. > 3) mean unlike the training structures in many components.

σ_j is floored at 1e-6 so a constant component cannot divide by zero.

*Source:* standardised (z-score) distance from a reference distribution. It is the
diagonal-covariance special case of the Mahalanobis distance, as used for
out-of-distribution detection (Mahalanobis 1936; Lee et al. 2018, NeurIPS, "A simple
unified framework for detecting out-of-distribution samples").

## Limits

- It uses a diagonal covariance, so correlated components are not accounted for. It is a
  screening number, not a probability.
- A model without a training reference (e.g. the untrained development model) reports
  `null`, not a number.

## Tests (`tests/test_latent_drift.py`)

- e = μ gives 0; e = μ + 3σ gives 3; a hand-computed partial offset matches.
- The untrained engine returns `None`. The old code returned a non-zero number.
- With a reference set from known embeddings, `explain` returns exactly the formula.
- `train()` saves a reference (count > 0, finite mean, positive std).

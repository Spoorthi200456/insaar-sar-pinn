# Train / validation / test split by structure group (review finding 24)

**In plain English:** to know whether a model works, you must test it on structures it
never saw while learning. Before this fix, training used every structure, and the
"best" model was the one with the lowest *training* error. That rewards memorising,
not understanding. Now the structures are divided once, with a fixed random seed, into:

- **training** (about 70 %): the model learns from these;
- **validation** (about 15 %): used only to pick which training epoch to keep;
- **test** (about 15 %): looked at exactly once, at the end, to report how well the chosen
  model does on structures it has never seen.

## Why split by connected group, not by single structure

Structures within the connection radius are linked in the model. Their water levels
interact through lateral flow (`docs/math/graph_stencil.md`), and the graph-attention layers
read neighbours' data. If structure A is in the test set but its neighbour B is in training,
A's satellite record still enters B's training computation. The test would then not be
truly unseen. So the unit of splitting is a **connected component** of the structure graph:
a group of structures linked directly or through other structures. A whole group goes into
one split, so no graph edge crosses splits.

*Source:* "group" or spatially blocked cross-validation for spatially autocorrelated data
(Roberts et al. 2017, *Ecography* 40, "Cross-validation strategies for data with temporal,
spatial, hierarchical, or phylogenetic structure").

## Procedure (`src/data/splits.py::split_structures`)

1. Build the fixed structure graph from all records. Find its connected components.
2. Shuffle the components with `numpy.random.default_rng(seed)`. The seed is fixed, 0 by
   default.
3. Give each component in turn to the split whose target size (fraction × number of
   structures) is furthest from met. Each split with a positive target first gets one
   component, so none is empty when there are at least three components.
4. Optional `stratify` (used only in synthetic experiments, where working/failing labels
   are known): do steps 2–3 within each label, so every split contains both kinds.

Because components are indivisible, the sizes match the fractions only approximately when
some components have more than one structure.

## What training does with it (`src/training/train.py::train`)

- Only training structures are batched and reach the optimiser.
- Design-feature normalisation (mean, standard deviation) is fitted on the **training
  split only** and reused, unchanged, for validation and test. It is also saved in the
  checkpoint.
- After every epoch, the validation loss is computed. The checkpoint with the **lowest
  validation loss** is kept (`selected_on = "validation"`).
- After training, that checkpoint is evaluated **once** on the test split (`test_loss`).
- The split (structure ids), its seed, the normalisation statistics, and the train,
  validation and test losses are stored in the checkpoint.
- With fewer than three independent groups nothing can be held out. Training then uses all
  structures, logs a warning (`training.no_validation_split`), and records
  `selected_on = "training"`.

## PROVISIONAL

Fractions 70/15/15 and seed 0. With very few structure groups the validation and test sets
are small, and their losses are noisy.

## Tests (`tests/test_splits.py`)

- Splits are disjoint, cover every structure, are reproducible with a fixed seed, and
  change with the seed.
- No graph edge crosses splits; the two 1.5 km clusters each stay together.
- 40 isolated structures split exactly 28/6/6.
- A stratified split puts both classes in every split.
- Only training structures reach the data loader; the checkpoint records a disjoint split,
  `val_loss`, `selected_on = "validation"` and `test_loss`.
- Validation and test use the training split's normalisation statistics.

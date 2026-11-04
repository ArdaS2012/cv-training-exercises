# Make overfitting visible: coding reference

[Quick start](README.md) · [Pair task](GROUP_TASK.md)

## Data and model (supplied)

- Images: sklearn digits, 8×8 grayscale, intensities 0..1. Tensor layout `(N,1,8,8)` (batch, channel, row, column). Targets: int64 class IDs 0..9.
- Training rows: `day2.small_train_rows()` returns 50 row indices (the first 5 training rows of each digit). Validation: 60 rows. Reserved test: 60 rows. No row is in two splits.
- Model: `day2.TinyCNN(dropout=p)`: conv 3×3 (1→8) · ReLU · max-pool 2 · conv 3×3 (8→12) · ReLU · max-pool 2 · flatten (48) · dropout p · linear (48→10). 1,446 parameters, **random initialization** (seeded).
- Optimizer: full-batch Adam, 150 epochs (one update per epoch). Baseline A: rate 0.01, λ 0, dropout 0.

## TODO 1 — `cross_entropy(logits, targets)`

- `logits`: finite float CPU tensor `(B,C)`, row i = image i, column c = score $s_{ic}$; B ≥ 1, C ≥ 2. May require gradients.
- `targets`: int64 tensor `(B,)`, values `0..C-1`, aligned with rows. Not one-hot.
- Return a scalar tensor (shape `()`) connected to the autograd graph: $\mathrm{CE}=\frac1B\sum_i[\log\sum_c e^{s_{ic}}-s_{i,y_i}]$.
- Do not modify inputs. Do not call `.item()`, `.detach()` or convert to NumPy.

Useful: `torch.logsumexp(x, dim=1)` → `(B,)` stable log-normalizers. `torch.arange(B)` → `(B,)` row indices; `a[rows, cols]` with two equal-length index tensors picks one entry per row. `.mean()` → scalar.

## TODO 2 — `objective(logits, targets, weights, penalty)`

- `logits`, `targets`: as TODO 1.
- `weights`: nonempty Python list of float parameter tensors (all conv/linear `weight` tensors; biases excluded). Shapes differ per tensor.
- `penalty`: Python float λ, 0 ≤ λ ≤ 0.1.
- Return scalar tensor $L=\mathrm{CE}+\lambda\sum_W\sum_j W_j^2$. No 1/2 factor, no averaging over parameters. Use your `cross_entropy`.

Useful: `w.square().sum()` → scalar per tensor; Python `sum(...)` over tensors returns a tensor.

## TODO 3 — `best_epoch(validation_losses)`

- `validation_losses`: nonempty Python list of finite floats ≥ 0; element e = validation data CE after epoch e (0-based).
- Return Python `int` in `0..len-1`: the index of the smallest value, first index on ties. `[0.9,0.5,0.5,0.7]` → `1`.

Useful: `min(range(n), key=...)`, or `int(np.argmin(list))`.

## Supplied experiment

- `day2.candidate_config(factor, value)`: `factor` in `'learning_rate'|'penalty'|'dropout'`; returns the baseline dict with that one key replaced.
- `day2.compare_runs(config, objective, seed)`: trains A (baseline) and B (`config`) with your objective, same seed/rows/epochs. Returns `(reports, checkpoints)` or `None` if the objective is unfinished. Each report: `config` dict, `curves` dict with `train_loss`, `validation_loss`, `validation_accuracy` (150 floats each, plain data CE / accuracy), `final_validation_loss`, `final_validation_accuracy`, `epochs`, `seed`, `train_rows`, `validation_rows`.
- `day2.select_checkpoint(reports, best_epoch)`: calls your function per run; returns `{'per_run': [{'run','epoch','validation_loss','validation_accuracy'}, ...], 'selected_run': int, 'selected_epoch': int}`; lower validation loss wins, first run on ties.
- `day2.evaluate_reserved(model)`: returns `test_n`, `accuracy`, `image_ids`, `protocol`. Use once, after selection.

The plot uses a log y-axis; dots mark each run's best validation epoch. One small dataset and three seeds do not establish a universal best setting.

[PyTorch 2.8 logsumexp](https://docs.pytorch.org/docs/2.8/generated/torch.logsumexp.html) · [cross_entropy](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.cross_entropy.html) · [Dropout](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Dropout.html) · [Adam](https://docs.pytorch.org/docs/2.8/generated/torch.optim.Adam.html) · [autograd](https://docs.pytorch.org/docs/2.8/autograd.html) · [Python min](https://docs.python.org/3.12/library/functions.html#min)

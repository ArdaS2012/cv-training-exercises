# Train a Tiny Neural Classifier

[Shared installation and help](../../README.md) · [Exercise index](../README.md)

## Start

From the student workspace: `uv run --locked python start.py --block 5`.

## Exercise + breakout

Follow the [hands-on investigation (around 30 minutes)](GROUP_TASK.md) after guided setup. It keeps the prior-slide takeaways beside the mission, code steps, controlled comparison and shared result. Use the full room for tracing, experimentation and explaining a decision. Keep the technical help below for the bounded TODOs.

## Help

If results are waiting, complete the indicated TODO; independent inputs remain usable. A shape error means checking rows versus columns before changing the supplied pipeline. Reloading resets unsaved controls; use the findings download. Port conflict: add `--port 2721`.

This practice belongs to CV Day 1 Block 5. It needs no private trainer guide, account, model download or GPU.

## Current teaching handoff

CV Day 1, Block 5: actual slide **64 — Exercise: test learning evidence** launches the combined exercise + breakout. Return to plenary at the agenda’s room end. The notebook and group task are self-contained; private trainer files are not required.

## TODO input and output reference

### TODO 1 — activation(z)

`np` is NumPy, already imported. `z` is a finite floating NumPy array of pre-activation values; signs may be positive, zero or negative. It has no probability or pixel-intensity constraint. The supplied model passes arrays `(n, 24)`, one row per image and one column per hidden unit; the function should preserve **any input array shape**.

ReLU is defined elementwise: negative inputs become zero, and nonnegative inputs retain their value. Return `activated`, a finite numeric NumPy array with exactly `z.shape`. For a shape-only example, an input `(2, 3)` must still return `(2, 3)`; do not reduce over a batch or choose a predicted class. Zero remains zero.

### TODO 2 — mean_loss(probabilities, labels)

| Function input | Definition |
|---|---|
| `probabilities` | Float NumPy array `(n, c)`, with `n>0` samples and `c` classes. Entry `[i, j]` is the model probability that sample `i` belongs to class ID `j`. Values are in `[0, 1]`, and each row sums to 1 (up to rounding). These are already softmax probabilities, not logits. The digit task has `c=10`; do not hard-code that column count. |
| `labels` | Integer NumPy array `(n,)`. Entry `i` is the true class ID for probability row `i`, from `0` through `c-1`. It is neither a one-hot matrix nor a predicted class. Rows are aligned and must stay in order. |

The supplied training batch has `n=180` and classes 0–9; the function should also work on smaller nonempty batches. For example, with a shape `(2, 3)` matrix and `labels=[2, 0]`, the true class for the first row is class 2 and for the second row class 0. This establishes the input meaning, not implementation syntax.

Cross-entropy here means the mean negative **natural** log of the true-class probability over samples. For numerical safety, probabilities below `1e-12` count as `1e-12` when computing this loss; zero probabilities must therefore give a finite result. Return `loss`, one finite nonnegative numeric scalar in natural-log units, rather than one value per row. The supplied code guarantees nonempty, valid inputs; no new validation, softmax, normalization, gradient or data split is required. The prepared derivatives assume these ReLU/cross-entropy definitions. `None` is only the unfinished state.

#### API reference

NumPy 2.2: [maximum](https://numpy.org/doc/2.2/reference/generated/numpy.maximum.html) compares elements with broadcasting; [integer-array indexing](https://numpy.org/doc/2.2/user/basics.indexing.html#integer-array-indexing) pairs row/column index arrays elementwise; [arange](https://numpy.org/doc/2.2/reference/generated/numpy.arange.html) supplies integer ranges with an excluded stop; [clip](https://numpy.org/doc/2.2/reference/generated/numpy.clip.html) bounds values; [log](https://numpy.org/doc/2.2/reference/generated/numpy.log.html) is the natural logarithm; [mean](https://numpy.org/doc/2.2/reference/generated/numpy.mean.html) reduces an array to its average when no axis is specified.

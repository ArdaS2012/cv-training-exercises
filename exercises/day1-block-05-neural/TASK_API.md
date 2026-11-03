# Tiny digit reader — task reference

[Start and challenge](README.md). This separate reference gives the full definitions beside the notebook’s concise contracts. All arrays are CPU NumPy arrays; no GPU, downloads or invalid/empty training inputs are required.

## TODO 1 — Choose training and validation rows

    `prepare_data()` has **no arguments**. Supplied `data.classification_split()` returns `(x, y, ids, tr, va, te)`:

    - `x`: finite float NumPy array `(300,64)`, one flattened 8×8 grayscale image per row, row-major pixel order, intensities already divided by 16 into `[0,1]`. No further normalization is needed.
    - `y`: integer `(300,)` true digit IDs `0..9`; `ids`: integer `(300,)` original source IDs, aligned with `x` and `y`.
    - `tr`, `va`, `te`: disjoint integer row-index arrays of lengths 180, 60, 60. They are fixed, stratified by digit and reproducible. An ID is not a row index.

    Return a dictionary with **exactly** `train_x`, `train_y`, `valid_x`, `valid_y`, `valid_ids`: select the corresponding rows using `tr` and `va`. Their shapes are `(180,64)`, `(180,)`, `(60,64)`, `(60,)`, `(60,)`. Preserve row alignment and inputs; never use `te` to select examples. The supplied loader is the only place creating splits. No empty batches occur.

    **Hint:** `array[index_array]` selects those rows and preserves their order. Generic example: selecting names at positions `[3,1]` returns position 3 before position 1. Before continuing, explain why fitting on validation rows would invalidate this comparison.

## TODO 2 — Initialize a 64 → 24 → 10 model

    `initialize()` has **no arguments**. Return `params`, a dictionary of finite float NumPy arrays with exactly these keys:

    | Key | Shape | Role |
    |---|---|---|
    | `w1` | `(64,24)` | input-to-hidden weights |
    | `b1` | `(24,)` | one bias per hidden unit |
    | `w2` | `(24,10)` | hidden-to-class weights |
    | `b2` | `(10,)` | one bias per digit class |

    A seeded `rng` is supplied inside the function. Draw `w1` **then** `w2` from a normal distribution of mean 0, standard deviation 0.15. Initialize both biases to zero. Recreate fresh arrays each call: each experiment must start with the same parameters. Do not initialize every weight to zero: equal hidden units would learn the same thing.

    **Toolbox:** `rng.normal(loc, scale, size)` returns random floats of the requested tuple shape; `np.zeros(shape)` returns a zero array (an integer shape makes a vector). Parameters have no physical units. The fixed architecture has no extra layers or batch dimension in its weights. **Trace:** `(n,64)` times which matrix gives `(n,24)`?

## TODO 3 — Build the forward pass

    `forward(x, params)` receives finite float image rows `x` of shape `(n,64)`, `n>0`, values `[0,1]`, and the four-array parameter dictionary from TODO 2. Return the **tuple** `(z, h, probabilities)` in that order, not a dictionary:

    $$z=XW_1+b_1,\quad h=\max(0,z),\quad s=hW_2+b_2,\quad p=\mathrm{softmax}(s).$$

    `z` is the pre-activation `(n,24)`, `h` the ReLU output `(n,24)`, `s` the raw class scores `(n,10)` and `probabilities` the finite output `(n,10)` in `[0,1]`, each row summing to 1. Columns correspond to digit IDs 0 through 9. Keep the original inputs unchanged. ReLU preserves positives and sets negatives to zero; softmax compares scores within each image.

    **Toolbox:** `a @ b` is matrix multiplication; adding a bias vector broadcasts across rows. `np.maximum(a,b)` selects the elementwise maximum, broadcasting scalar arguments. Supplied `softmax(logits)` takes any finite `(n,c)` score array with `n,c>0` and returns same-shape normalized probabilities. It subtracts each row maximum before exponentiation to avoid overflow; it does not mutate scores. No derivative or softmax implementation is required.

    **Hint:** follow the four mathematical stages and track the shape after each. Trace one negative pre-activation through ReLU. The same function must work for 180 training rows, 60 validation rows or one image.

## TODO 4 — Score the true class

    `mean_loss(probabilities, labels)` receives finite floating probabilities `(n,c)` in `[0,1]` with rows summing to 1 and integer true-class IDs `labels` of shape `(n,)` in `0..c-1`. `n>0`, `c>=1`; rows align. Return one finite nonnegative Python/NumPy numeric scalar, leaving inputs unchanged. Do not apply softmax again or score the maximum predicted probability.

    $$L=-\frac{1}{n}\sum_{i=0}^{n-1}\ln(\max(p_{i,y_i},10^{-12})).$$

    `n` is batch size; `p` is the probability matrix; `y_i` is the true class of row `i`. The natural logarithm gives natural-log units. A target probability 0.25 gives loss about 1.386; target certainty gives zero. Clipping at `1e-12` makes zero probabilities finite. This formula must match the supplied derivatives.

    **Toolbox:** `np.arange(n)` makes indices `0..n-1`; `a[rows, columns]` with equal-length index arrays selects paired entries (one per row). `np.clip(a, low, high)` bounds values without mutating `a`; `np.log(a)` takes elementwise natural logs; `np.mean(a)` returns the mean scalar. **Hint:** identify each row's true-class column, score it, then average. For schema only: labels `[2,0]` refer to column 2 of row 0 and column 0 of row 1.

## TODO 5 — Apply a gradient update

    `update(params, grads, rate)` receives the parameter dictionary from TODO 2 and a gradient dictionary with **exactly the same keys and shapes**. Each gradient is a finite float array of derivatives of **mean training cross-entropy**, with respect to that parameter. `rate` is a positive finite scalar, selected here from `0.05, 0.2, 0.5, 2.0`. Return a **new dictionary and new arrays**; do not change either input.

    $$\theta_{new}=\theta_{old}-\eta\nabla_\theta L.$$

    `theta` is each weight/bias array; `eta` is `rate`. Subtracting moves against increasing loss; a large step can overshoot. Illustrative scalar trace: value 0.4, derivative 0.2, rate 0.1 gives 0.38. A zero gradient leaves its parameter unchanged.

    **Hint:** iterate over dictionary keys (or write four explicit entries) and apply the same rule to each. `for key in mapping` visits keys; `mapping[key]` retrieves its array; ordinary array arithmetic creates a new array. Do not recompute gradients inside this function.

## TODO 6 — Connect the training loop

    `fit(dataset, params, steps, rate)` receives the exact dictionaries from TODOs 1/2, integer update count `steps>=0` (slider: 0–160 by 20), and positive scalar `rate` as in TODO 5. Return `(trained_params, history)`. `history` is a list of `steps+1` finite loss floats, including the baseline **before any update**. Rebind local `params` using your nonmutating update function; keep caller inputs unchanged.

    The loop and history storage are supplied. Replace the four `None` values in the marked region with the forward-pass, loss, backward and update connections; keep the other loop lines. Use **training rows only**, including for gradients. `forward` returns a three-item cache; its last item is probabilities. `mean_loss` needs those probabilities and aligned true labels.

    **Supplied derivative API:** `backward(x, labels, params, cache)` accepts nonempty finite training inputs `(n,64)`, integer labels `(n,)` in `0..9`, the current parameter dictionary, and the `(z,h,probabilities)` tuple from `forward` on **those same inputs and current parameters**. It returns `grads` with keys `w1,b1,w2,b2`, each matching its parameter's float shape. It calculates derivatives for ReLU and mean softmax cross-entropy and never mutates inputs or updates parameters. Use it before changing parameters. There is no need to derive backpropagation here.

    **Hint:** the supplied `if iteration == steps` break keeps 0 steps a genuine control and avoids an extra update. `history.append(float(value))` stores a scalar. Explain the difference between computing a gradient and applying one. Do not call the old all-in-one `train` helper.

## TODO 7 — Evaluate without learning from validation

    `evaluate(dataset, params)` receives the dataset dictionary from TODO 1 and trained parameter dictionary from TODO 6. Use only `valid_x` and `valid_y` here. Return `(predictions, accuracy)`:

    - `predictions`: integer NumPy vector `(60,)`, one predicted digit ID `0..9` per validation row, same row order as `valid_ids`.
    - `accuracy`: one finite scalar fraction in `[0,1]`: number correct divided by number of validation images. It is not a percentage or the training loss.

    Run your forward pass on validation rows, choose each row's highest-probability class and compare to its true label. Do not calculate gradients or update parameters; inputs must remain unchanged. General operation supports any nonempty validation batch. For tied probabilities, choose the first class index.

    **Toolbox:** `np.argmax(a, axis=...)` returns indices of largest entries along that axis; axis 0 indexes rows, axis 1 indexes columns. Without `axis`, it flattens the array. `predictions == labels` returns aligned Boolean correctness values; `np.mean` averages them as 0/1. **Trace:** why does an `(n,10)` probability matrix become `(n,)` predictions?

    ### Your connected experiment

    Press **Train on CPU**. The supplied display controller calls your functions in order and labels the first unfinished stage. Every press rebuilds the fixed split and seeded parameters. Compare `(steps,rate)=(0,0.5),(80,0.5),(80,2.0)`. The plots show training loss at every step and validation accuracy **before and after training only**, not a per-step validation curve. Keep image row 0 fixed and inspect its pixels and source ID. Changing probabilities need not change the predicted digit.

    Inspect the computed settings, loss, validation accuracy and held-out prediction. Trial downloads are optional; no written prediction/observation log or participant report is required. The trainer presents the solution and its limits, including different handwriting or cameras. Choosing a rate using validation is model selection; it does not make validation an untouched test. The 60 test rows remain unused.

    ### Official NumPy 2.2 API references

    [Row indexing](https://numpy.org/doc/2.2/user/basics.indexing.html#integer-array-indexing) · [normal](https://numpy.org/doc/2.2/reference/random/generated/numpy.random.Generator.normal.html) · [zeros](https://numpy.org/doc/2.2/reference/generated/numpy.zeros.html) · [matmul](https://numpy.org/doc/2.2/reference/generated/numpy.matmul.html) · [maximum](https://numpy.org/doc/2.2/reference/generated/numpy.maximum.html) · [arange](https://numpy.org/doc/2.2/reference/generated/numpy.arange.html) · [clip](https://numpy.org/doc/2.2/reference/generated/numpy.clip.html) · [log](https://numpy.org/doc/2.2/reference/generated/numpy.log.html) · [mean](https://numpy.org/doc/2.2/reference/generated/numpy.mean.html) · [argmax](https://numpy.org/doc/2.2/reference/generated/numpy.argmax.html).
## Supplied display helpers

`digit_examples()` takes no arguments and returns a Matplotlib figure containing three bundled 8×8 input digits and their true labels. It performs no inference or fitting.

`pipeline_report(prepare_data, initialize, forward, mean_loss, fit, evaluate, *, steps, rate)` receives your six callable functions with the exact contracts above, integer `steps>=0`, and positive scalar `rate`. `fit` uses your `update` through its closure. The controller calls these functions only after **Train on CPU** is pressed, starting from the fixed split and seeded initialization each time. It returns `(report, message)`: `report=None` with a readable stage message while unfinished, otherwise a dictionary with exactly:

- `loss`: list of `steps+1` finite training-loss floats; `validation_accuracy`: two floats in [0,1], before/after training.
- `predictions`, `targets`, `validation_ids`: aligned lists of 60 integers: predicted digits, true digits 0–9, and source IDs.
- `train_count=180`, `validation_count=60`, `test_count=60`: integer row counts; `test_used=False`.
- `steps`: integer update count; `rate`: positive scalar learning rate.

`pipeline_figure(report, image_row=0)` accepts that completed report and integer validation-row position 0–59 (not a source ID). It returns a Matplotlib figure with training losses, before/after validation accuracy and the selected image. It never trains. Loss is measured every step; validation accuracy only at the endpoints.

The steps slider offers 0–160 by 20; rate choices are 0.05, 0.2, 0.5 and 2.0. The image slider selects validation rows 0–59. Controls are dimensionless except integer row/update counts. **Save run (.json)** appears after a completed run and downloads that run’s settings, selected image row and report; save a separate file for each comparison. There is no text evidence log; discuss your explanation with your partner. Python saving does not retain slider settings.

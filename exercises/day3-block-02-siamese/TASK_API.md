# Task definitions, inputs and API reference

## TODO definitions and API reference

### First understand a pair and its embedding

A **pair** contains a left image and a right image. An **encoder** converts an image into a vector of learned features, called an **embedding**. Both branches here use the same encoder weights. The eight coordinates describe learned features; they are not eight pixels or digit labels. Distance is computed between the two vectors belonging to the **same pair**.

NumPy is already imported as `np`. An array with shape `(P, D)` has `P` rows (pairs) and `D` columns (feature coordinates). **Reducing an axis** means combining its entries into a shorter result: one result per pair must retain the row axis and combine coordinates within each row.

### TODO 1 — distance for each pair: `distance(a, b)`

| Argument | Type, meaning and guarantees |
| --- | --- |
| `a` | Finite floating-point NumPy array `(P, D)` on CPU: row `i` is the left image's embedding for pair `i`. Current data has `(P, D) = (24, 8)`. |
| `b` | Finite floating-point NumPy array of the identical shape: row `i` is the right image's embedding for that same pair. Columns have the same feature meaning as in `a`. |

The model supplies nonempty batches and positive feature dimensions; do not hard-code `24` or `8`. No labels or images need loading in this function. Coordinates are dimensionless. The current encoder makes vectors unit length, so current-data Euclidean distances are in `[0, 2]` up to floating-point rounding; the distance definition also works for other finite vectors used in a manual check.

For pair `i`, with feature index `j`:

$$d_i=\sqrt{\sum_{j=1}^{D}(a_{ij}-b_{ij})^2}.$$

`a_ij` and `b_ij` are corresponding coordinates. The sum combines coordinates **inside one pair**; it does not compare every left image with every right image. The square root returns Euclidean distance rather than squared distance.

**Return.** A floating-point NumPy array `(P,)`: one finite nonnegative scalar per pair, preserving row order. `(P,)` is a one-dimensional array, not `(P, 1)`, an all-pairs `(P, P)` matrix or a single batch-wide number. Equal vectors have distance zero. Keep both inputs unchanged.

**Available operation.** `np.linalg.norm(x, ord=None, axis=None, keepdims=False)` returns a norm. For a vector or one selected axis with default `ord`, it computes the Euclidean norm. `axis` chooses which axis to reduce; keeping dimensions is optional and is off by default. Calling it without an axis on a matrix reduces the whole matrix, which differs from one result per row. NumPy arithmetic between same-shaped arrays combines matching entries and returns an array.

### TODO 2 — loss for each pair: `contrastive(d, y, margin)`

A **loss** is a penalty used during learning. Here you calculate its value for already-computed embeddings to understand the objective; changing this function or the UI threshold does not retrain the supplied checkpoint.

| Argument | Type, meaning and guarantees |
| --- | --- |
| `d` | Finite nonnegative NumPy float array `(P,)`: the distances from TODO 1, in original pair order. |
| `y` | Finite NumPy binary numeric array `(P,)`, aligned with `d`: the live data use `float64`, with **`1.0` meaning similar and `0.0` meaning different**. Integer or Boolean 0/1 arrays in small manual examples have the same meaning. These are pair targets, not digit class IDs. Some references use the opposite convention; follow this one. |
| `margin` | Positive finite scalar float `m`, currently `1.0`, in the same dimensionless distance units as `d`. This is the dissimilar-pair separation margin, not the UI decision threshold. |

For each pair, use this exact definition:

$$L_i=y_i d_i^2+(1-y_i)\max(0,m-d_i)^2.$$

A similar pair uses the distance penalty; a dissimilar pair uses only the shortfall from the margin. `max(0, …)` means no penalty once the required separation is reached. Squaring makes a larger shortfall more costly. There is **no additional factor of one half** in this exercise's definition.

**Return.** A floating-point NumPy array `(P,)`, with one finite nonnegative loss per input pair in the same order. Keep `d` and `y` unchanged. Do not sum or average; the report computes the mean separately. At `d == margin`, a dissimilar pair has zero loss; above the margin it remains zero. A similar pair at zero distance has zero loss. The runner supplies nonempty, aligned valid inputs; validation of malformed batches is outside the TODO.

**Available operations.** `np.maximum(x1, x2)` returns the elementwise larger value from its two inputs. A scalar such as zero broadcasts to every position of an array; **broadcasting** means using that scalar at each position without manually building a same-length array. `np.maximum` does not reduce an axis. Normal arithmetic and powers work elementwise on NumPy arrays; Python `^` denotes bitwise XOR, not squaring. These operations return new results unless an output argument or in-place operator is explicitly used.

### Run, inspect and explain

Replace `distances = None` and `losses = None` within the marked regions; keep the signatures and existing return lines. Run the cell with **Shift+Enter**. While either function returns `None`, the report waits. Its finite-value/nonnegative/shape checks establish structure only; independently trace a small similar and dissimilar case using the equations above.

The supplied `day3.pair_report(distance, contrastive, threshold=0.7)` calls both functions on the fixed pairs, using margin `1.0`. It returns `None` for an unfinished function, otherwise `(report, figure)`. A threshold is a nonnegative distance cutoff; the UI selects `[0, 2]`. The report accepts a match when `distance <= threshold`, including equality. It loads a fixed encoder and does not update weights.

Dotted field paths below describe entries in the displayed JSON. They are Python dictionary keys, not object attributes. `report.pairs` is an ordered list of records with `pair` (zero-based row ID), `left_digit`/`right_digit` (image labels), `similar_target` (Boolean reference), `distance`, `threshold`, `match` (Boolean decision), `correct` (decision agrees with target) and `loss`. The chart shows the first eight image pairs; the table contains all 24. A **false match** accepts a reference-different pair; a **missed match** rejects a reference-similar pair. `mean_loss` averages your per-pair losses; it is unchanged by a threshold-only edit. Increasing the threshold can accept more pairs but cannot change distances or losses. Distinguish the cost of each error before recommending a threshold, and reserve fresh pairs for validation/testing beyond these inspected examples.

**Official API references (NumPy 2.2):** [vector norms and axis](https://numpy.org/doc/2.2/reference/generated/numpy.linalg.norm.html), [elementwise maximum](https://numpy.org/doc/2.2/reference/generated/numpy.maximum.html), [broadcasting rules](https://numpy.org/doc/2.2/user/basics.broadcasting.html). The notebook's label convention and local report contract are defined above.

The same definitions appear directly above the editable functions in the notebook.

# Task input and output contracts

## Function inputs and API reference

### TODO 1: `distance(a, b)`

`a` and `b` are finite floating-point NumPy arrays of identical shape `(P, 8)`, currently `(24, 8)`. `P` is the number of image pairs, not the number of pixels. Row `i` in `a` is the left image embedding for pair `i`; row `i` in `b` is that **same pair's** right embedding. Axis 1 contains eight learned feature coordinates. A coordinate is not a digit label. The supplied encoder makes each vector unit length, so its coordinates are dimensionless and pair distances lie in `[0, 2]` up to numerical rounding.

Return a NumPy floating-point array of shape `(P,)`: one nonnegative Euclidean distance per corresponding pair, in input row order. Do not return an all-pairs `P×P` matrix, squared distances, or one scalar for the entire batch. Keep both arrays unchanged. The Euclidean-distance definition appears immediately above the TODO.

### TODO 2: `contrastive(d, y, margin)`

| Input | Definition and guarantees |
|---|---|
| `d` | Finite nonnegative NumPy float array shape `(P,)`: the distance for each pair, in the same order as the labels. |
| `y` | NumPy array shape `(P,)` of binary integer labels: **1 means similar**, **0 means different**. This is the opposite of some libraries' conventions; use this notebook's definition. |
| `margin` | One positive finite float `m`, currently `1.0`, in the same distance units as `d`. It is the training separation margin, not the UI match threshold. |

Return a NumPy float array of shape `(P,)`: the per-pair losses from the exact equation printed above, with **no extra factor of one half**. Do not average the losses; the supplied report computes the mean separately. The result is finite and nonnegative; at the margin a dissimilar pair has zero loss. Keep `y` aligned with the original pair rows.

**API vocabulary:** NumPy axis `0` selects the pair/batch dimension; axis `1` selects embedding coordinates. `np.linalg.norm` accepts an `axis` to choose the reduction dimension. `np.maximum(a, b)` takes an elementwise maximum (it does not reduce an axis); scalar arithmetic broadcasts over an array. This vocabulary defines operations, not an implementation sequence.

**Official references:** [NumPy norm](https://numpy.org/doc/2.2/reference/generated/numpy.linalg.norm.html), [NumPy maximum](https://numpy.org/doc/2.2/reference/generated/numpy.maximum.html), [broadcasting](https://numpy.org/doc/2.2/user/basics.broadcasting.html).

The same contracts are displayed beside the TODOs in the notebook. API references define available operations; selecting and combining them remains your task.

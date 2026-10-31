# Reduce and Reconstruct Image Data

[Shared installation and help](../../README.md) · [Exercise index](../README.md)

## Start

From the student workspace: `uv run --locked python start.py --block 2`.

## Exercise + breakout

Follow the [hands-on investigation (around 30 minutes)](GROUP_TASK.md) after guided setup. It keeps the prior-slide takeaways beside the mission, code steps, controlled comparison and shared result. Use the full room for tracing, experimentation and explaining a decision. Keep the technical help below for the bounded TODOs.

## Help

If results are waiting, complete the indicated TODO; independent inputs remain usable. A shape error means checking rows versus columns before changing the supplied pipeline. Reloading resets unsaved controls; use the findings download. Port conflict: add `--port 2721`.

This practice belongs to CV Day 1 Block 2. It needs no private trainer guide, account, model download or GPU.

## Current teaching handoff

CV Day 1, Block 2: actual slide **31 — Exercise: defend a projection** launches the combined exercise + breakout. Return to plenary at the agenda’s room end. The notebook and group task are self-contained; private trainer files are not required.

## TODO input and output reference

### TODO 1 — PCA inputs and return contract

All numeric inputs below are finite NumPy arrays supplied by the preceding setup cell. `np` is NumPy, imported for the notebook. Shapes use `(rows, columns)`.

| Available input | Definition |
|---|---|
| `centered` | Float array `(300, 64)`. Each row is one 8×8 digit image, flattened row by row; column `j` is one fixed pixel position. The mean of each feature column has already been subtracted. Values are intensity differences, so negatives are expected. |
| `mean` | Float array `(64,)`, the 300-image mean intensity at each pixel; original intensities were normalized to `[0, 1]`. Its feature order matches `centered`. |
| `vt` | Float array `(64, 64)` from the supplied `np.linalg.svd(centered, full_matrices=False)`. Each **row** is a unit feature-space direction, ordered by decreasing singular value. All 64 directions are provided; do not refit SVD. |
| `components` | Marimo slider object. `.value` is the selected integer `k`, from 1 to 64: how many leading principal directions to retain. Use the setting as a number, rather than the widget object. |

PCA represents each centered sample by coordinates along the retained orthogonal directions. The retained basis consists of the leading `k` directions as **columns**. Reconstruction returns from those coordinates to the original 64-feature intensity space, including the supplied mean. This describes the required representation, rather than Python expressions for it.

Return all three named arrays: `basis`, shape `(64, k)`; `projected`, shape `(300, k)`; `reconstructed`, shape `(300, 64)`. Row `i` must always refer to the same image. For a shape-only example with 5 images, 6 features and `k=2`, these outputs would have shapes `(6, 2)`, `(5, 2)`, `(5, 6)`. At `k=64`, reconstruction should differ from the input only by floating-point roundoff. At smaller `k`, approximate reconstructed intensities can leave `[0, 1]`; do not clip them for this comparison. None of these arrays has a class-label column. Leave `None` only while unfinished.

#### API reference

NumPy 2.2: [SVD return conventions](https://numpy.org/doc/2.2/reference/generated/numpy.linalg.svd.html), [slicing/indexing](https://numpy.org/doc/2.2/user/basics.indexing.html) (`a[start:stop]` excludes `stop`), [transpose](https://numpy.org/doc/2.2/reference/generated/numpy.ndarray.T.html) (swap array axes), and [matrix multiplication](https://numpy.org/doc/2.2/reference/generated/numpy.matmul.html) (`@`: `(a,b)` with `(b,c)` gives `(a,c)`). The nonlinear optimizer views below are supplied and are not TODOs.

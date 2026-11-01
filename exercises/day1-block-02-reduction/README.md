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

### TODO 1 — retain directions, project images and reconstruct them

**Your job:** replace `basis`, `projected` and `reconstructed` with three NumPy array expressions. The cell receives `centered`, `components`, `mean` and `vt`. Data loading, centering and SVD are already done. Keep those inputs unchanged and do not fit another decomposition.

### Understand the operation before coding

An 8×8 image has 64 pixel-intensity features. **PCA** finds perpendicular directions in that 64-dimensional feature space, ordered by how much variation they capture. A component is a direction; a projected coordinate tells how far an image lies along it. Retaining `k` directions compresses every image to `k` coordinates. Reconstructing maps those coordinates back to the 64 pixel positions, then restores the mean image.

The prepared **SVD**, $X_c=U\,\mathrm{diag}(s)\,V^T$, supplies those directions. Here $X_c$ is the centered image matrix; columns of $U$ describe sample coordinates, entries of $s$ measure direction strength, and rows of $V^T$ are feature directions. Only `vt`, the array representing $V^T$, is needed for your TODO. The leading `k` directions must form the **columns** of your retained basis $B_k$.

$$Z=X_c B_k,\qquad \widehat X=Z B_k^T+\mu$$

$Z$ contains projected coordinates, $\widehat X$ the reconstructed intensities and $\mu$ the mean image added to **every row**. $B_k^T$ reverses the projection's axis change; at smaller `k` it cannot recover discarded information. These are mathematical definitions, with the NumPy expressions left to you.

### Inputs you can use

All supplied numeric arrays are finite `float64` NumPy arrays. Shapes are `(rows, columns)`; a one-dimensional shape such as `(64,)` is a vector.

| Input | Meaning, layout and valid values |
|---|---|
| `centered` | Shape `(300, 64)`: one image per row, one fixed pixel position per column. Pixels are flattened row by row. Each column's mean has already been subtracted. Values are normalized intensity differences and may be negative. |
| `mean` | Shape `(64,)`: original mean intensity at each pixel, in the same pixel order. Original image intensities are in `[0, 1]`. |
| `vt` | Shape `(64, 64)`: rows are orthonormal feature directions in decreasing singular-value order. It comes from `np.linalg.svd(centered, full_matrices=False)`, whose three returns are `u`, `singular`, `vt`. A row of `vt` is a direction, not an image. |
| `components` | Marimo slider **object**. `.value: int` is `k`, from 1 through 64. Read the current value; do not use the widget object in arithmetic or fix `k` to one trial. |

### NumPy toolbox

`np` is imported in the notebook. For a two-dimensional array, `a[row_selection, column_selection]` selects axes in that order; `:` retains an axis, and a slice's stop is excluded. Transposition `a.T` swaps rows and columns. `@` performs matrix multiplication: `(a, b) @ (b, c)` yields `(a, c)`. The inner dimensions must agree. `*` is elementwise multiplication and does not perform this projection. Adding a length-`d` vector to an `(n, d)` array broadcasts the vector across all `n` rows. These operations return arrays; do not overwrite the provided inputs.

### Required results and checks

| Output variable | Required shape | Meaning |
|---|---|---|
| `basis` | `(64, k)` | The retained feature directions as columns, preserving their supplied order. |
| `projected` | `(300, k)` | Each centered image's coordinates along the retained directions. |
| `reconstructed` | `(300, 64)` | Approximate original intensities, with the mean restored. |

For a shape-only example with 5 images, 6 features and `k=2`, the shapes are `(6, 2)`, `(5, 2)`, `(5, 6)`. Row `i` must continue to mean image `i`; none of these outputs contains class labels. All outputs must be finite numeric arrays; `None` means unfinished. The supplied data are nonempty and `k` is valid, so no new input validation is needed.

At `k=64`, reconstruction should match the original images up to floating-point roundoff. At smaller `k`, reconstructed values may leave `[0, 1]`; **do not clip the arrays** used for measurement. The displayed grayscale range is fixed for viewing. If the image remains too dark, check the missing mean; if shapes fail, say aloud what each axis represents before changing the expression.

### Read the supplied comparison outputs

**MSE** is $\frac{1}{nd}\sum_{i,j}(X_{ij}-\widehat X_{ij})^2$: average squared reconstruction error over `n=300` images and `d=64` pixels, in normalized-intensity units squared. **Retained variance** is $\sum_{j=1}^{k}s_j^2/\sum_{j=1}^{64}s_j^2$, a unitless fraction. `singular` supplies the `s` values. Lower MSE and higher retained variance still do not guarantee that your chosen visible digit detail survives.

The supplied `display.nonlinear_views(x, neighbors=15)` takes the original finite `(n, d)` image matrix and an integer neighborhood setting, and returns `{'t-SNE': array, 'UMAP': array}`, each float array `(n, 2)` in the same sample order. It runs CPU optimizers with seed 42; no labels enter fitting. The button invokes it using `neighbors.value` (5, 15 or 30). That one setting is **t-SNE perplexity** (effective neighborhood size, positive and smaller than `n`) versus **UMAP n_neighbors** (graph-neighbor count, at least 2 and smaller than `n`). They are different definitions, not equivalent methods. Two-dimensional coordinates have arbitrary orientation/scale; digit colors are supplied labels for interpretation only. Views can run independently of the PCA TODO. Changing `k` does not make either nonlinear view a PCA reconstruction.

Official NumPy **2.2** references: [SVD](https://numpy.org/doc/2.2/reference/generated/numpy.linalg.svd.html), [slicing/indexing](https://numpy.org/doc/2.2/user/basics.indexing.html), [transpose](https://numpy.org/doc/2.2/reference/generated/numpy.ndarray.T.html), [matrix multiplication](https://numpy.org/doc/2.2/reference/generated/numpy.matmul.html), [broadcasting](https://numpy.org/doc/2.2/user/basics.broadcasting.html). Supplied view interfaces: [scikit-learn 1.7 t-SNE](https://scikit-learn.org/1.7/modules/generated/sklearn.manifold.TSNE.html), [UMAP parameters](https://umap-learn.readthedocs.io/en/0.5dev/parameters.html).

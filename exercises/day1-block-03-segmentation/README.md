# Segment a Small Image

[Shared installation and help](../../README.md) · [Exercise index](../README.md)

## Start

From the student workspace: `uv run --locked python start.py --block 3`.

## Exercise + breakout

Follow the [hands-on investigation (around 30 minutes)](GROUP_TASK.md) after guided setup. It keeps the prior-slide takeaways beside the mission, code steps, controlled comparison and shared result. Use the full room for tracing, experimentation and explaining a decision. Keep the technical help below for the bounded TODOs.

## Help

If results are waiting, complete the indicated TODO; independent inputs remain usable. A shape error means checking rows versus columns before changing the supplied pipeline. Reloading resets unsaved controls; use the findings download. Port conflict: add `--port 2721`.

This practice belongs to CV Day 1 Block 3. It needs no private trainer guide, account, model download or GPU.

## Current teaching handoff

CV Day 1, Block 3: actual slide **39 — Exercise: test a pixel partition** launches the combined exercise + breakout. Return to plenary at the agenda’s room end. The notebook and group task are self-contained; private trainer files are not required.

## TODO input and output reference

### TODO 1 — define which pixels should be considered similar

**Your job:** create `spatial_features` by joining the supplied RGB values with weighted position features. The cell receives `color`, `np`, `position` and `spatial_weight`. Image creation, coordinate normalization, color-only clustering and K-Means fitting are supplied.

### What is one feature row?

This exercise clusters **pixels**, rather than whole images. A pixel's RGB values describe appearance; its position says where it is. K-Means sees only a numeric row and Euclidean distance. Adding position can prefer nearby pixels even when two distant regions share a color. It does not teach the algorithm a name such as “car”.

| Input | Exact type, shape and meaning |
|---|---|
| `color` | Finite `float64` NumPy array `(1536, 3)`. Each row is one pixel. Columns are **red, green, blue**, in that order; intensities are in `[0, 1]`. |
| `position` | Finite `float64` NumPy array `(1536, 2)` for the same pixels in the same order. Columns are normalized **row, column**: `row/(H-1)` and `column/(W-1)`, each in `[0, 1]`. Origin is top left; rows increase downward and columns rightward. |
| `spatial_weight` | Marimo slider object; `.value` is the current nonnegative numeric weight `w`, 0–2 in steps of 0.1. It is dimensionless, not a count or an image coordinate. |
| `np` | NumPy, already imported. No additional imports are required. |

The supplied authored scene has height `H=32` and width `W=48`, hence `H*W=1536` pixels. Both arrays use **C / row-major order**: left to right along a row, then the next row. Flat index `i = row*W + column` identifies the same pixel in both inputs. For example, index 49 is coordinate `(1, 1)`, with normalized position `(1/31, 1/47)`. Preserve that correspondence. The scene is an illustration with a lighting gradient, not a labelled photograph.

### Feature definition and effect of the weight

For pixel `i`, the required feature vector is $f_i=(R_i,G_i,B_i,w r_i,w c_i)$, where `R/G/B` are normalized intensities, `r/c` normalized row/column, and `w` the current setting. This gives squared distance

$$\|f_i-f_j\|^2=\|\mathrm{RGB}_i-\mathrm{RGB}_j\|^2+w^2\|(r_i,c_i)-(r_j,c_j)\|^2.$$

A zero weight removes position's contribution. A weight of one keeps its normalized scale; doubling the weight multiplies its **squared-distance** contribution by four. This changes the grouping rule, not the image or its pixel order. Repeated colors/duplicate feature rows are valid inputs.

### NumPy toolbox and required result

Multiplying a numeric array by a scalar scales each element and returns an array with the same shape. `np.column_stack(tup)` accepts a tuple/list of arrays with equal row counts and joins their **columns**; joining `(n, a)` and `(n, b)` gives `(n, a+b)`. It does not create a new batch axis. Choose how to combine those operations yourself.

Return `spatial_features`: a finite floating array `(1536, 5)` whose columns are red, green, blue, weighted normalized row, weighted normalized column, in exactly that order. Keep all rows aligned; do not mutate `color` or `position`, scale RGB with the position weight, fit a model here or replace `None` with labels. `None` is only the unfinished state. Empty/malformed arrays are not supplied; no validation or coordinate-generation code is needed.

After Shift+Enter, the prepared cell fits color-only and color-plus-position K-Means, with current `cluster_count.value` (2–7), seed 42 and 10 initializations. Completing TODO 1 does not yet restore a 2D mask: TODO 2 handles that. The notebook displays these definitions beside its editable cell.

Official NumPy **2.2** interfaces: [column_stack](https://numpy.org/doc/2.2/reference/generated/numpy.column_stack.html), [scalar broadcasting](https://numpy.org/doc/2.2/user/basics.broadcasting.html).

### TODO 2 — restore the label at each image coordinate

**Your job:** turn a one-dimensional label vector into a height-by-width label image, while preserving its pixel order. The cell receives `rgb` and `spatial_labels`. It should also remain safe while TODO 1 is unfinished.

| Input | Exact contract |
|---|---|
| `rgb` | Finite floating NumPy array `(H, W, 3)`, here `(32, 48, 3)`. Axes mean **image row, image column, RGB channel**; intensities are in `[0, 1]`. This array supplies image dimensions; do not change its pixels. |
| `spatial_labels` | `None` until TODO 1 supplies features; otherwise an integer NumPy array `(H*W,)`, here `(1536,)`. Entry `i` is the fitted group ID of feature row `i`, from `0` through `K-1`, where `K` is the supplied cluster-count setting. |

### What is a mask here?

A **label mask** stores one integer ID per pixel. It has no RGB channel dimension. The output must satisfy $M[r,c]=\ell[rW+c]$, with `M` the mask, `ell` (`spatial_labels`) the flat labels, `r` the row and `c` the column. The width `W` determines where one row ends. Use dimensions from the provided image, rather than hard-coding 32 and 48.

Shape-only example: a 2×3 image has six labels. The first three labels belong to its top row, the next three to its bottom row. Reshaping should preserve those six entries, rather than refit K-Means, sort IDs or turn labels into colors.

### Supplied API and required behavior

`array.shape` is a tuple of axis sizes; its first two entries are height and width for this `rgb` layout. `array.reshape(shape, order='C')` returns an array with the requested dimensions, preserving C/row-major element order. The product of the new dimensions must equal the original element count. It can share storage with the input, so do not edit the returned labels in place. `None` has neither array shape nor a reshape method; handle the waiting state before performing array operations.

Return `spatial_mask=None` while `spatial_labels is None`. Once labels exist, return an integer array `(H, W)` with exactly those labels in their original pixel positions. Preserve input arrays and do not add a third axis. The runner guarantees positive image dimensions and matching label count; no malformed-input checks are required. A mask assigning every pixel the same group is structurally valid even when it is unhelpful for your purpose.

Check that the displayed mask aligns with the original scene. Probe coordinates `(18, 12)` and `(10, 36)` mean **row, column**, not `x, y`. Within one fitted mask, equal IDs mean shared membership. IDs can change between fits, so compare “same group?” and boundaries across weights rather than ID values. A grouping does not establish semantic class or separate object instances: disconnected pixels may share one ID, and one object may be split by lighting.

Official NumPy **2.2** interfaces: [ndarray.shape](https://numpy.org/doc/2.2/reference/generated/numpy.ndarray.shape.html), [ndarray.reshape](https://numpy.org/doc/2.2/reference/generated/numpy.ndarray.reshape.html).

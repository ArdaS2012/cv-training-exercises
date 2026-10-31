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

### TODO 1 — pixel-feature inputs and return contract

`np` is NumPy, already imported. Array shapes use `(rows, columns)`.

| Available input | Definition |
|---|---|
| `color` | Float NumPy array `(1536, 3)`. Each row is one pixel; columns are red, green, blue intensities in that order, each in `[0, 1]`. |
| `position` | Float NumPy array `(1536, 2)`. For that same pixel, columns are normalized **row**, then **column**: `row/(H-1)` and `column/(W-1)`, in `[0, 1]`. Origin is top left. |
| `spatial_weight` | Marimo slider object; `.value` is a float from 0 to 2. It is the relative weight of position features compared with color, not a cluster count or a pixel coordinate. |

Both arrays describe the authored scene with height `H=32`, width `W=48`. Row order is C/row-major: left to right, then top to bottom. Flat row `i = row*W + column` refers to the same pixel in both arrays. For example, row 49 corresponds to image coordinate `(1, 1)`; positions there are `(1/31, 1/47)`. Do not sort or shuffle pixels.

Return `spatial_features`, a finite float array `(1536, 5)`. Its five columns describe red, green, blue, weighted normalized row and weighted normalized column, in that order. A weight of zero makes position contribute zero distance; one keeps its normalized scale. K-Means fitting and a color-only baseline are already supplied.

### TODO 2 — label-image inputs and return contract

| Available input | Definition |
|---|---|
| `rgb` | Float NumPy array `(H, W, 3)`, here `(32, 48, 3)`; dimensions are image rows, columns, RGB channels. `.shape` reports those sizes. |
| `spatial_labels` | `None` until TODO 1 is ready; afterwards an integer NumPy array `(H*W,)`, here `(1536,)`. Entry `i` is K-Means' cluster ID for pixel row `i` of the feature array. IDs range from 0 to `K-1`, where the supplied cluster-count slider sets `K`. |

Return `spatial_mask=None` while labels are unavailable; otherwise return an integer array `(H, W)`, with label `i` restored to its original pixel coordinate. A mask contains one ID per pixel, with **no color-channel axis**. For a shape-only example, labels for a 2×3 image are a six-entry vector whose first three entries belong to the top row. Equal IDs indicate common membership within one fitted mask; IDs do not name semantic objects or stay comparable across separate fits.

#### API reference

NumPy 2.2: [column_stack](https://numpy.org/doc/2.2/reference/generated/numpy.column_stack.html) accepts a sequence of equal-length/equal-row arrays and joins their columns; [reshape](https://numpy.org/doc/2.2/reference/generated/numpy.reshape.html) changes shape without changing the number of entries, with C/row-major ordering by default. [ndarray shape](https://numpy.org/doc/2.2/reference/generated/numpy.ndarray.shape.html) returns the dimension-size tuple. Choosing how these interfaces express each required output is your task.

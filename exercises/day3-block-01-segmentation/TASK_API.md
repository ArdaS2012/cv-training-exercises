# Task definitions, inputs and API reference

## TODO definitions and API reference

### First understand the three kinds of image data

An RGB image stores three color numbers per pixel. A **semantic label map** instead stores one class number per pixel: background, warm shape or cool shape. A **Boolean mask** answers one yes/no question at each pixel, such as “is this pixel in the selected class?” `True` includes that pixel; `False` leaves it out. An **instance map** identifies individual objects; its object IDs must not be interpreted as semantic classes.

The notebook supplies NumPy as `np`. An array's **shape** tells you how many entries lie along each axis. `(H, W)` means `H` rows and `W` columns, with `labels[y, x]` referring to row `y`, column `x`. These are arrays in CPU memory, not PyTorch tensors, files or widgets.

### TODO 1 — select one class: `select_class(labels, class_id)`

**Purpose.** Produce the pixel region whose overlap we want to measure, while keeping its location in the image.

| Argument | Type, meaning and guarantees |
| --- | --- |
| `labels` | NumPy integer array `(H, W)`, currently `(32, 48)`. Each number is one semantic class ID: `0` background, `1` warm shape, `2` cool shape. Both image dimensions are positive. |
| `class_id` | Python `int`, currently one of `0`, `1`, `2`, passed from the class control. Use this argument on each call; it is already the selected value, not a widget needing `.value`. |

**Return.** A NumPy array of `bool` dtype and the same `(H, W)` shape. A pixel is `True` exactly when its input class ID equals `class_id`. Preserve row/column order and leave `labels` unchanged. If the selected class is absent, every entry is `False`; an absent class is a valid result, not an error.

**Read a small format example.** An integer array with rows `[0, 1, 2]` and `[1, 0, 2]` is a two-row, three-column label map. The number `2` denotes a class, not a count of objects. Your result must still have two rows and three columns. Do not hard-code the current scene dimensions.

**Available operation.** `np.equal(x1, x2)` compares corresponding values and returns Boolean membership; when one input is a scalar it is compared with every array entry. Python's array/scalar equality operator has the same elementwise meaning. The operation leaves its inputs unchanged unless an explicit output buffer is provided. Choosing how to express membership is your task.

### TODO 2 — compare aligned regions: `overlap(a, b)`

**Purpose.** Quantify agreement between the selected predicted region and the same reference class. The reference is an authored annotation, separate from the model prediction.

| Argument | Type, meaning and guarantees |
| --- | --- |
| `a` | NumPy Boolean array `(H, W)`: your prediction mask from TODO 1. |
| `b` | NumPy Boolean array with exactly the same shape: the supplied reference mask for that same class and image. `a[y, x]` and `b[y, x]` always refer to the same pixel. |

No image intensities, probabilities, instance IDs or channel dimension enter this function. Do not resize or shift either input. The runner supplies aligned, nonzero-size grids; a mismatched shape or invalid dtype is outside the task. “Empty region” below means **zero `True` pixels**, not an array with zero rows.

Define `I` as the number of pixels included in **both** masks, `U` as the number included in **either** mask, `N_A` as the number included in `a`, and `N_B` as the number included in `b`. Then

$$\mathrm{IoU}=\frac{I}{U},\qquad \mathrm{Dice}=\frac{2I}{N_A+N_B}.$$

These are dimensionless fractions, not percentages. Intersection counts each shared pixel once. Union also counts each shared pixel once; simply adding the region sizes would count shared pixels twice.

**Return.** A Python dictionary with exactly the string keys `"iou"` and `"dice"`, each mapped to a finite scalar float in `[0, 1]`. Keep both input masks unchanged. Return two numbers, not two Boolean masks or a per-pixel score map.

**Boundary rules.** Both regions empty: return `1.0` for both scores by this notebook's explicit convention. Exactly one empty: both scores are `0.0`. Identical nonempty regions have both scores `1.0`; disjoint nonempty regions have both scores `0.0`. Apply the empty convention before dividing by a zero count.

**Available operations.** `np.logical_and(x1, x2)` and `np.logical_or(x1, x2)` take aligned Boolean arrays and return Boolean arrays of the same shape, for membership in both/either. `np.count_nonzero(x)` with no `axis` returns one integer count over the entire array; `True` counts as one. Each operation returns a value and leaves its arguments unchanged. Python's scalar `and`/`or` do not combine array pixels. `float(value)` converts one numeric scalar into a Python float.

### Run, inspect and explain

Replace `mask = None` and `metrics = None` only inside their marked regions; keep the function signatures and existing return lines. The waiting message remains until both functions return results. The report checks shape, Boolean membership dtype, required score keys and finite score range; those checks cannot prove your formula is correct. Independently hand-check a tiny mask pair and the empty cases.

The prepared `day3.segmentation_report(select_mask, overlap, index=0, class_id=1)` calls your first function on the predicted semantic map, then calls your second on that result and the reference region. `index` is the scene ID `0..3`; `class_id` is the selected semantic ID. It returns `None` while a TODO returns `None`, otherwise `(report, figure, scene_data)`; the notebook displays and exports these for you. It loads fixed local model weights but does not train them.

Dotted field paths below describe entries in the displayed JSON. They are Python dictionary keys, not object attributes. Read `report.metrics.iou` and `report.metrics.dice` as overlap for **only the selected class**. `predicted_instance_count` and `reference_instance_count` count foreground object IDs across the scene, not just that selected class. `instance_ids` includes `0` when background is present; count positive IDs as objects. Use the reference/prediction panels to locate an actual boundary or identity mistake. High class overlap cannot establish correct object count, and correct count cannot establish correct object boundaries.

**Official API references (NumPy 2.2):** [array comparison](https://numpy.org/doc/2.2/reference/generated/numpy.equal.html), [logical AND](https://numpy.org/doc/2.2/reference/generated/numpy.logical_and.html), [logical OR](https://numpy.org/doc/2.2/reference/generated/numpy.logical_or.html), [counting included entries](https://numpy.org/doc/2.2/reference/generated/numpy.count_nonzero.html). All essential local contracts are given here; the links provide additional API detail.

The same definitions appear directly above the editable functions in the notebook.

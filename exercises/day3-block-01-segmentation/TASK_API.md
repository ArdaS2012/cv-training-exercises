# Task input and output contracts

## Function inputs and API reference

### TODO 1: `select_class(labels, class_id)`

| Input | Definition and guarantees |
|---|---|
| `labels` | NumPy integer array of shape `(H, W)`, currently `(32, 48)`. Each entry is a **semantic class ID** for one pixel, not a probability/channel vector. Axis 0 is row (y), axis 1 column (x). The current IDs are `0` background, `1` warm shape, `2` cool shape. |
| `class_id` | Python integer from the semantic-class control: `0`, `1` or `2`. It names the region to inspect. |

Return a NumPy Boolean array with the **same shape and pixel alignment**; `True` means that pixel belongs to the requested class, `False` means it does not. If that class is absent, the mask is all `False`. Example input structure: `labels = [[0, 1, 2], [1, 0, 2]]` as an integer array, with `class_id = 1`. This is only a format example; do not assume a particular scene. Do not mutate `labels` or replace semantic IDs by instance IDs.

### TODO 2: `overlap(a, b)`

`a` and `b` are same-shaped NumPy **Boolean** arrays `(H, W)`, on the same image grid and for the same class. In the prepared call, `a` is your prediction mask and `b` the reference. `True` denotes included pixels; numeric label IDs and RGB values are not passed to this function.

Return a Python dictionary with exactly `iou` and `dice` as string keys, mapped to finite scalar floats in `[0, 1]`. Use the definitions in **Read labels, regions and identities separately** above. The vertical bars in those equations mean **number of included pixels**, not a vector norm. Both-empty masks return 1 for both metrics; exactly one empty mask returns 0 for both. Keep input masks unchanged and return scalars, not per-pixel maps or percentages.

**API vocabulary:** comparison of a NumPy array with a scalar is elementwise and returns a Boolean array. `np.logical_and` and `np.logical_or` combine Boolean membership at matching positions; `np.count_nonzero` counts nonzero/True entries. Python's scalar `and`/`or` do not perform these array operations. Choose your own implementation of the stated metric definitions.

**Official references:** [NumPy comparisons](https://numpy.org/doc/2.2/reference/generated/numpy.equal.html), [logical_and](https://numpy.org/doc/2.2/reference/generated/numpy.logical_and.html), [logical_or](https://numpy.org/doc/2.2/reference/generated/numpy.logical_or.html), [count_nonzero](https://numpy.org/doc/2.2/reference/generated/numpy.count_nonzero.html).

The same contracts are displayed beside the TODOs in the notebook. API references define available operations; selecting and combining them remains your task.

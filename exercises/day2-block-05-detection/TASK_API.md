# Task input and output contracts

## Function inputs and API reference

### TODO 1: `confidence_selection(detections, threshold)`

`detections` is a Python list (possibly empty). Each item is an ordinary dictionary with **exactly these adapter fields**:

| Field | Definition |
|---|---|
| `id` | Unique string identifying a candidate within one detector run, for example `"candidate-7"`. Preserve it. |
| `label` | Class-name string, for example `"person"`; this is not a numeric class ID. |
| `score` | Finite float in `[0, 1]`, the detector's confidence. It is not measured IoU or a guarantee of correctness. |
| `box` | List of four finite floats `[x_min, y_min, x_max, y_max]` in original-image pixels. |

`threshold` is a finite scalar float in `[0, 1]` from **Confidence cutoff**. Return a **list of the original record objects**, retaining their input order and fields, for the stated inclusive cutoff (`score` equal to the threshold is retained). Empty input or no retained candidate produces `[]`. Do not change scores, boxes or input records. Filtering confidence is separate from the supplied later class-aware non-maximum suppression (NMS), which removes overlapping duplicate predictions; do not implement NMS here.

### TODO 2: `box_iou(box_a, box_b)`

Each argument is a length-four Python list of finite numbers in the above **xyxy** order, referring to the same image. Coordinates use x rightwards and y downwards from the top-left. Both rectangles have positive width/height: `x_min < x_max` and `y_min < y_max`. The last two values are maximum corners, **not width and height**. Example structure: `[10.0, 20.0, 50.0, 70.0]`.

Return one finite scalar IoU in `[0, 1]`: intersection area divided by union area. Area uses continuous pixel-coordinate geometry; do not add 1 to widths/heights. Disjoint or merely touching boxes have zero intersection; identical valid boxes have IoU 1. No tensor, mask or dictionary is returned. The helper handles reference matching; preserve your function's two-box interface.

**API vocabulary:** dictionary `record["field"]` access, list indexing, and Python `min`/`max` on scalar coordinates are available. Their operation definitions are in the references; you choose how to express the overlap calculation.

**Official references:** [Python dictionaries](https://docs.python.org/3.12/library/stdtypes.html#mapping-types-dict), [Python min/max](https://docs.python.org/3.12/library/functions.html#max), [Torchvision box_iou coordinate/metric contract](https://docs.pytorch.org/vision/0.23/generated/torchvision.ops.box_iou.html).

The same contracts are displayed beside the TODOs in the notebook. API references define available operations; selecting and combining them remains your task.

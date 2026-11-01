# Task definitions and API help

## TODO guide — filter confidence and measure box overlap

A detector proposes **candidate detections**, each with an identity, class label,
confidence score and box. Your first function decides which candidates meet a
score threshold. Your second measures geometry between two boxes. The supplied
code performs model inference, later duplicate suppression, plotting and matching
to teaching reference boxes. You do not implement a detector or NMS.

### TODO 1: `confidence_selection(detections, threshold)`

| Argument | Type and meaning |
| --- | --- |
| `detections` | Python `list` of ordinary detection dictionaries, possibly empty. Order is the adapter's candidate order; it need not be sorted by score. |
| `threshold` | Finite Python float in `[0, 1]` from **Confidence cutoff**. It is an inclusive minimum score, not the NMS overlap cutoff. |

Every full adapter record contains these exact keys:

| Key | Value type, meaning and convention |
| --- | --- |
| `id` | Unique `str` within one detector run, for example `"candidate-7"`. It identifies the existing record; it is not a class ID. |
| `label` | Class-name `str`, for example `"person"`. Both adapters convert their model-specific numeric IDs to names. |
| `score` | Finite float in `[0, 1]`: the detector's confidence. It is neither measured IoU nor a correctness guarantee. |
| `box` | Length-four `list[float]`, `[x_min, y_min, x_max, y_max]`, in original-image pixel coordinates. The last two entries are the opposite corner, not width/height. |

Shape example only:
`{"id": "candidate-7", "label": "person", "score": 0.63, "box": [10.0, 20.0, 50.0, 70.0]}`.
This illustrates data layout, not a reference answer or a calibrated confidence.
Selection depends only on `score`; small hand-check records may contain only that
field. Preserve any other supplied fields without interpreting them.

Return a **new list containing the original dictionary objects** whose score is
at least the threshold, in their original relative order. Preserve IDs, labels,
scores, boxes and all record fields; do not rebuild or edit retained records.
A score exactly equal to the cutoff is retained. Return `[]` for empty input or
when none qualify. Duplicate/overlapping candidates are still separate records
here. Do not sort, merge boxes, filter by class or apply the later NMS rule.
Missing/nonfinite scores and invalid thresholds are outside the supplied domain.

**Python toolbox:** dictionary lookup uses exact quoted keys; list positions
start at zero. A generic `items.append(value)` adds one item and returns `None`,
so its result is not a new list. `>=` includes equality. Basic loops, conditionals
and list construction need no extra import. Choose your own implementation.

### TODO 2: `box_iou(box_a, box_b)`

Each argument is a length-four Python list of finite numbers, in the **same xyxy
order** and same original-image pixel coordinate system. The origin is top-left,
x increases rightward and y downward. Coordinates may be fractional. The runner
supplies positive-area rectangles: `x_min < x_max` and `y_min < y_max`.
No normalization to `[0, 1]`, array axis conversion or image dimensions are needed.

**Intersection over Union (IoU)** compares shared area with area covered by either
box. With `A` and `B` denoting the two rectangles:

$$w_I=\max(0,\min(x_{A,max},x_{B,max})-\max(x_{A,min},x_{B,min})),$$
$$h_I=\max(0,\min(y_{A,max},y_{B,max})-\max(y_{A,min},y_{B,min})),$$
$$I=w_Ih_I,\qquad U=|A|+|B|-I,\qquad \mathrm{IoU}=I/U.$$

`w_I` and `h_I` are overlapping width/height in pixels; `I` is intersection area,
`U` union area, and `|A|`, `|B|` each box's width times height in pixels squared.
The later starting edge and earlier ending edge bound an overlap. Clipping a
negative span to zero expresses no overlap. Union subtracts the shared area once
because the two individual areas counted it twice. Valid positive-area boxes
guarantee a positive denominator.

Return one finite Python/NumPy scalar in `[0, 1]`, with no units. Merely touching
or disjoint boxes have IoU zero; identical valid boxes have IoU one. This is
continuous-coordinate geometry: do **not** add an inclusive-pixel `+1` to widths.
Keep both inputs unchanged. Degenerate/reversed boxes are outside the supplied
domain; no additional invalid-box policy is required. Python `min(a, b)` and
`max(a, b)` return a scalar smaller/larger argument; they do not interpret boxes.

The Torchvision reference below documents the same metric but accepts batches
of tensors and returns a matrix of pairwise IoUs. Your TODO instead keeps this
local **two Python boxes → one scalar** interface; do not return that matrix.
IoU alone does not compare class labels or establish that either box is correct.

### Before running and reading the evidence

Replace only `selected=None` and `overlap=None` in the marked regions. Keep the
function signatures/return lines and rerun with **Shift+Enter**. Check score
equality, no retained candidates, identical boxes, a partial overlap and a
separated/touching pair by hand. These checks test distinct boundary meanings.

Until **Run actual YOLO and SSD on CPU** is pressed, the notebook labels its mode
as a measured pretrained snapshot. Live inference uses cached model weights;
a missing cache reports a preparation error. Both paths expose decoded candidate
records after permissive native confidence `0.01` and NMS `0.95`, so they are not
raw logits or all possible boxes.

After your filter, supplied **class-aware non-maximum suppression (NMS)** can
remove a lower-scoring same-class box when overlap exceeds its cutoff. It does
not change the retained box coordinates. Hold this control fixed during the
confidence experiment. The report's `models` dictionary maps each detector name
to `input_candidates`, `above_cutoff`, `after_class_aware_nms` integer counts,
`kept` detection records and `reference_matches`. Each reference match has
`reference: str`, `class: str`, `best_iou: float`, `match: bool`, and
`detection: str | None`; a match requires the same class and IoU at least `0.5`. Matching is greedy in the authored reference order: each reference considers only same-class predictions not already matched to an earlier reference. Thus one prediction cannot satisfy two references; `best_iou` is the best overlap among the remaining eligible predictions, or zero if none remain.
These reference annotations are teaching examples, not an exhaustive benchmark.
Record the model/mode and inspect a concrete miss or duplicate: postprocessing
cannot invent a candidate the detector never produced.

**Version-matched official references:** [Python 3.12 lists and dictionaries](https://docs.python.org/3.12/library/stdtypes.html),
[Python 3.12 min/max](https://docs.python.org/3.12/library/functions.html#max),
[Torchvision 0.23 box_iou](https://docs.pytorch.org/vision/0.23/generated/torchvision.ops.box_iou.html),
[Torchvision 0.23 class-aware batched_nms](https://docs.pytorch.org/vision/0.23/generated/torchvision.ops.batched_nms.html).

The same guide is displayed immediately before the editable notebook cell. Complete only the marked TODO regions; the implementation remains your work.

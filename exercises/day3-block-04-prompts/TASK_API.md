# Task input and output contracts

## Function inputs and API reference

### TODO 1: `text_candidates()`

This function takes **no arguments**. Return a Python list containing 2–6 nonempty strings: the candidate image descriptions in the order you want the chart/report to use. The current image is selected outside this function. Use descriptions appropriate for that fixed image; do not load an image, construct tokens or call a model here. The helper's `text.candidates`, `text.cosines` and `text.scores` lists align by position. `cosines` are image/text cosine similarities; `scores` are a softmax over only this supplied candidate list, summing to 1. Their positions are not permanent global class IDs.

### TODO 2: `spatial_prompt()`

This function also takes **no arguments**. Return an ordinary Python dictionary using the following adapter fields; these are **not** the Hugging Face processor's raw keyword names:

| Key | Value / requirement |
|---|---|
| `points` | Optional nonempty list of points, each a length-two numeric list `[x, y]` in **original-image pixels**. |
| `labels` | List with one integer per point, in the same order: `1` asks to include content at that point, `0` asks to exclude it. Supply it explicitly when you use points. These are spatial-prompt labels, not semantic class IDs. |
| `box` | Optional length-four numeric list `[x_min, y_min, x_max, y_max]` constraining the region. Positive width/height are required; this is not `[x, y, width, height]`. |

Supply points with labels, a box, or both. At least one of `points` and `box` must be nonempty. Leave unused fields out. The supplied wrapper adds the batch nesting and preprocessing; your return must have **only the flat per-image structures described here**. A schema-only example of a point value is `[[20, 30]]`; it is not a recommended point for the target region.

The origin is the top-left: x increases rightwards and y downwards. For image width `W` and height `H`, points satisfy `0 <= x < W`, `0 <= y < H`; boxes satisfy `0 <= x_min < x_max <= W`, `0 <= y_min < y_max <= H`. **Bus:** `W=810`, `H=1080`; **football:** `W=1280`, `H=720`. Read coordinates on the original image even when the displayed preview is resized. Changing the image requires rechecking your geometry. The target-region task and baseline are stated above; your prompt choice remains your investigation.

**API vocabulary:** Python lists preserve item order; dictionaries associate named fields with values. The prepared `SamProcessor` converts original-image prompt coordinates into model coordinates and postprocesses masks back to the image. You do not call it in the TODO. Model-estimated mask quality is not a measured reference IoU or class probability.

**Official references:** [Python lists/dictionaries](https://docs.python.org/3.12/library/stdtypes.html#sequence-types-list-tuple-range), [Transformers 4.57.1 SAM prompt and processor interfaces](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/sam), [original CLIP interface](https://github.com/openai/CLIP#usage).

The same contracts are displayed beside the TODOs in the notebook. API references define available operations; selecting and combining them remains your task.

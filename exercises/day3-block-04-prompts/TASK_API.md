# Task definitions, inputs and API reference

## TODO definitions and API reference

### First choose the kind of output you want

**CLIP** compares the fixed image with each candidate description. A candidate is a possible text description, not a model answer or a permanent class ID. CLIP supplies image/text similarities and relative candidate scores; it does not return pixels belonging to the described object.

**SAM** returns a Boolean pixel mask using a spatial prompt: included points, excluded points and/or a bounding box. A positive point requests the object or region at that location; a negative point indicates content to leave out. These prompts do not name a semantic class. The models' weights remain fixed in both routes.

### TODO 1 — prepare descriptions: `text_candidates()`

**Inputs.** This function takes **no arguments**. The selected image and model are handled by the prepared runner. You choose the descriptions by editing this function; do not read a widget, load the image, call a model or build tokens inside it.

**Return.** A Python `list[str]` with **2–6 nonempty, non-whitespace strings**. Keep descriptions short enough for CLIP's 77-token context including its special tokens; words and tokens are not the same unit. The prepared tokenizer does not automatically shorten an overlong candidate. Use a list, not one comma-separated string or a dictionary.

Choose a few plausible competing descriptions with parallel wording: the same level of detail and similar phrasing, varying the content you want to test. Keep candidate order and all unedited wording fixed for a one-description comparison. Duplicates are accepted by the adapter but add repeated alternatives to the normalization; use distinct descriptions for an informative core comparison. The runner preserves your list order.

**How the scores are produced.** Let `s_j` be image/text cosine similarity for candidate `j`, and `alpha > 0` the fixed learned CLIP scale. The displayed relative score is

$$p_j=\frac{\exp(\alpha s_j)}{\sum_{k=1}^{K}\exp(\alpha s_k)},$$

where `K` is the number of candidates. The scores sum to one **over your list only**. Adding, removing or editing a candidate changes this comparison set and may change other scores too. A higher score is not a verified probability that the description is true.

### TODO 2 — prepare geometry: `spatial_prompt()`

**Inputs.** This function also takes **no arguments**. The runner supplies the selected image separately. Read the original-image coordinate rules below, then return your chosen geometry for that image.

**Return.** A new Python dictionary with the adapter fields below. At least one of `points` or `box` must be present and nonempty. You may use points with labels, a box, or both. Omit unused fields. These exact short keys belong to this exercise's adapter; they are **not** raw `SamProcessor` keyword arguments.

| Exact key | Value type and meaning |
| --- | --- |
| `points` | Optional nonempty nested Python list; coordinates are `int` or `float`. Each inner list has exactly two finite numbers `[x, y]`: one point in original-image pixels. Order is preserved. |
| `labels` | A `list[int]` of the same length/order as `points`: `1` includes content at that point, `0` excludes it. Supply this key explicitly when you supply points. Its numbers are prompt roles, not semantic class IDs. |
| `box` | Optional list of four finite numbers `[x_min, y_min, x_max, y_max]`. It gives opposite box corners with positive width and height, **not** `[x, y, width, height]`. |

**Coordinate definitions.** The image origin `(0, 0)` is the top-left. `x` increases to the right; `y` increases downward. An image array may be indexed `[y, x]`, but a prompt point uses the order **`[x, y]`**. Width is `W` and height is `H`, measured in original-image pixels:

- Every point satisfies `0 <= x < W` and `0 <= y < H`.
- Every box satisfies `0 <= x_min < x_max <= W` and `0 <= y_min < y_max <= H`.
- **Bus:** `W=810`, `H=1080`. **Football:** `W=1280`, `H=720`.

The displayed preview is resized; do not use its screen width as `W`. Recheck geometry if you switch images. A schema-only point value such as `[[20, 30]]` demonstrates nesting and coordinate order; it is not a recommended target point. A point list has one list per point, with **no extra image/batch brackets**. The wrapper supplies those internally. The bus investigation targets the person near `(280, 650)`; the fixed baseline's positive point lies on the bus instead. Choosing useful positive/exclusion points or a box is your experimental decision, not a guaranteed correct mask.

**Boundary and alignment rules.** A point on `x == W` or `y == H` is outside the image. A box may end at `W`/`H`, but its two corners must not be identical. Each label must stay aligned with its own point; reordering points without labels changes the prompt. An empty dictionary, labels without any geometry, or a mismatched point/label count is invalid. The adapter rejects invalid geometry with a warning; your TODO does not need to implement that validation. No strings, normalized coordinates, tensors or model objects belong in the returned geometry.

### Python tools and the supplied adapter

Python lists use `[...]` and preserve item order. Dictionaries use named string keys and associate each with a value, which may itself be a list or dictionary. Generic syntax unrelated to the task: `book = {"title": "Atlas", "pages": 120}`. Return the container itself, not its printed form or a JSON string. `append(item)` changes a list in place and returns `None`; assigning that return value would erase your list variable.

`day3.prompt_report(candidates, prompt, image_id='bus')` is prepared for you. `candidates` and `prompt` follow your two return contracts; `image_id` is `'bus'` or `'zidane'`. It returns `None` if either input is `None`, otherwise `(report, figure, mask)`. It runs actual text encoding and SAM prompt decoding against cached image encodings for the fixed model revisions. It does not train weights. The adapter converts your original-image coordinates to model coordinates and returns masks at original `(H, W)` resolution; do not preprocess coordinates yourself.

Dotted field paths below describe entries in the displayed JSON. They are Python dictionary keys, not object attributes. **Read the actual result.** `report.text.candidates`, `.cosines` and `.scores` are parallel lists of length `K`; position `j` refers to your description `j`. `report.text.scale` is `alpha` in the equation. `report.spatial.prompt` records the geometry used; `.model_mask_quality_scores` has one predicted quality estimate for each of SAM's candidate masks, and `.chosen` is the zero-based selected candidate index (largest estimate). The returned `mask` is a Boolean NumPy array `(H, W)`; `report.mask_pixels` counts its `True` pixels. These quality estimates are not a measured reference IoU, and selecting the largest estimate does not prove it outlines the intended person. Inspect the overlay and exported mask for included/excluded content.

### Run and compare one change

Replace `candidates = None` and `prompt = None` inside their marked regions, keeping the existing signatures and return lines. Run the cell, then press the Run button; the costly models only run on that button. Both TODOs must return data before the paired report can run. Preserve the fixed baseline and your own first export. For the text comparison hold geometry fixed; for the spatial comparison restore text and change just one point, label or box item. These supplied photographs and your inspected masks are development evidence, not an accuracy benchmark.

**Official references:** [Python 3.12 lists](https://docs.python.org/3.12/library/stdtypes.html#sequence-types-list-tuple-range), [Python dictionaries](https://docs.python.org/3.12/library/stdtypes.html#mapping-types-dict), [Transformers 4.57.1 SAM interfaces](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/sam), [Transformers 4.57.1 CLIP interfaces](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/clip). The exercise adapter's flat prompt schema is defined above and differs from the raw library's batched schema.

The same definitions appear directly above the editable functions in the notebook.

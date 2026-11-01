import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="Test Two Vision Prompt Routes")


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(next(p for p in Path(__file__).resolve().parents if (p/'cv_exercises').is_dir())))
    import marimo as mo
    import numpy as np
    import json
    from cv_exercises import day3,display

    return day3, display, json, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Prompt repair: words and points do different jobs

    **Around 30 minutes in pairs.** CLIP compares an image with text candidates; normalized scores depend on that set. SAM accepts spatial prompts and returns masks. Prompt edits change inference inputs, not weights.

    ### Investigate, do not just run

    1. Use the bus image throughout the core comparison. Predict why the supplied point-on-bus baseline misses the intended person near x280,y650, and which model can return that person’s pixels.

    2. Complete comparable text candidates and a valid original-pixel spatial prompt. Run both real models, inspect baseline scores and mask, and save the baseline before editing.

    3. Freeze the spatial prompt. Change ONE text description in otherwise identical parallel candidates, rerun and compare scores/ranking. Explain what changed in the denominator and what that cannot prove about semantic correctness.

    4. Swap roles. Restore the text candidates, change only ONE spatial prompt item (point, exclusion or box) and compare masks on the fixed image/revisions. Mark included/excluded pixels around the intended person and diagnose a failure.

    5. Recommend a prompt for the stated output, citing score and mask evidence separately. Propose one next controlled test; explain why a text label and a spatial mask are different outputs and neither comparison trains weights.

    **Evidence log:** run | text item changed | spatial item changed | score/rank evidence | mask inclusion/exclusion. Save a run before changing controls. Export both the current findings and your comparison log.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Test Two Vision Prompt Routes

    Investigation after guided setup. Run actual CLIP text inference and SAM prompt decoding with official pinned checkpoints and cached matching image encodings.

    Predict → complete short TODOs → change one setting → inspect → explain → download.

    Work in pairs and swap editor/observer. Edit only TODO regions; Shift+Enter runs cells. Ctrl+S / Command+S saves code. Download findings to retain control values and observations. The fresh starter waits politely; prepared loading/plotting/serving code needs no edits. No GPU or API key.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    prompt_image=mo.ui.dropdown(options={'Bus photograph':'bus','Football photograph':'zidane'},value='Bus photograph',label='Fixed image')
    run_prompt=mo.ui.run_button(label='Run both actual vision models')
    prediction=mo.ui.text_area(label='Predict: which prompt change will affect scores or masks?',full_width=True)
    mo.vstack([prediction,prompt_image,run_prompt])
    return prediction, prompt_image, run_prompt


@app.cell(hide_code=True)
def _(day3, mo, prompt_image):
    mo.image(src=str(day3.ASSETS/(prompt_image.value+'.jpg')),width=380)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Choose the interface before the prompt
    CLIP compares image and text vectors with cosine similarity $s=f(x)^Tg(t)/(\|f(x)\|\|g(t)\|)$. Reported candidate scores normalize over YOUR supplied candidate set. Changing one description can change both relative meaning and the normalization; the score is not a standalone probability that a description is true.

    SAM takes original-image spatial coordinates: a positive point requests included content, a negative point excludes content, and a box constrains a region. It returns pixels, not a text class. On this bus photo, inspect the intended person near x280,y650 before editing. Hold the other prompt route fixed when comparing; changing text and geometry together would leave the cause ambiguous. Keep images and model revisions fixed; neither prompt edit updates weights.
    """)
    return






@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Your two prompt inputs to the supplied vision models

    You prepare data for two different interfaces: a list of text descriptions for CLIP, and a spatial instruction for SAM. The real models, image loading, tokenization, preprocessing and mask decoding are already connected.

    Complete both functions below, run their cell with **Shift+Enter**, then press **Run both actual vision models**. On the bus image, the spatial target is the person around original coordinate x=280, y=650. Inspect the fixed point-on-bus baseline before choosing your prompt. Save a baseline, then change only one prompt item while the other route remains fixed. Model assets must be cached using the README's preflight before the investigation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
    """)
    return



@app.cell
def _(np):
    def text_candidates():
        # TODO 1: parallel label descriptions for this image comparison.
        candidates=None
        # END TODO 1
        return candidates
    def spatial_prompt():
        # TODO 2: original-image points + labels, or an xyxy box.
        prompt=None
        # END TODO 2
        return prompt

    return spatial_prompt, text_candidates


@app.cell(hide_code=True)
def _(
    day3,
    json,
    mo,
    prompt_image,
    run_prompt,
    spatial_prompt,
    text_candidates,
):
    prompt_result=None
    base_prompt_result=None
    _prompt_error=None
    if run_prompt.value and text_candidates() is not None and spatial_prompt() is not None:
        try:
            _image=prompt_image.value
            _baseline={'points':[[400,300]],'labels':[1]} if _image=='bus' else {'points':[[700,80]],'labels':[1]}
            base_prompt_result=day3.prompt_report(['bus','bicycle','person'] if _image=='bus' else ['person','football','bus'],_baseline,_image)
            prompt_result=day3.prompt_report(text_candidates(),spatial_prompt(),_image)
        except (ValueError,TypeError,KeyError,OSError) as _e:_prompt_error=str(_e)
    mo.callout(_prompt_error,kind='warn') if _prompt_error else mo.callout('Complete candidates/spatial prompt, then run. Models must be cached by preflight.',kind='info') if prompt_result is None else mo.vstack([mo.md('### Fixed baseline'),base_prompt_result[1],mo.md('### Your changed prompt'),prompt_result[1],mo.md('```json\n'+json.dumps(prompt_result[0],indent=2)+'\n```')])
    return base_prompt_result, prompt_result


@app.cell(hide_code=True)
def _(mo):
    experiment_log=mo.ui.text_area(value="run | text item changed | spatial item changed | score/rank evidence | mask inclusion/exclusion\nA | \nB | \nStress case | \nDecision + limitation | ",label='Comparison log — preserve run A before changing a control',full_width=True)
    mo.vstack([mo.md("### Keep your evidence while experimenting\nEach rerun replaces the current result. Copy measured values and name a visible feature or response field here before changing controls."),experiment_log])
    return (experiment_log,)


@app.cell(hide_code=True)
def _(display, experiment_log, mo):
    display.download(mo,experiment_log.value,{'columns':"run | text item changed | spatial item changed | score/rank evidence | mask inclusion/exclusion"},'day3-block4-comparison-log',label='Download comparison log (baseline + changed runs)')
    return


@app.cell(hide_code=True)
def _(mo):
    observations=mo.ui.text_area(label='Findings: controlled change, measured result, failure explanation and limitation',full_width=True)
    observations
    return (observations,)


@app.cell(hide_code=True)
def _(
    base_prompt_result,
    display,
    mo,
    observations,
    prediction,
    prompt_result,
):
    _downloads=[display.download(mo,observations.value,{'prediction':prediction.value,'baseline':None if base_prompt_result is None else base_prompt_result[0],'changed':None if prompt_result is None else prompt_result[0]},'vision-prompt-findings')]
    if prompt_result is not None:
        import io
        from PIL import Image
        for _name,_result in [('baseline',base_prompt_result),('changed',prompt_result)]:
            _mask=io.BytesIO();Image.fromarray(_result[2].astype('uint8')*255).save(_mask,format='PNG')
            _downloads.append(mo.download(_mask.getvalue(),filename=_name+'-prompt-mask.png',label='Download '+_name+' mask'))
    mo.vstack(_downloads)
    return


if __name__ == "__main__":
    app.run()

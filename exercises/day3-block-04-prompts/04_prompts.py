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
    mo.md("""
    ## TODO — candidates and spatial prompt
    `text_candidates()` returns2–6 comparable nonempty strings. Use parallel wording; the output scores normalize over this set only.
    `spatial_prompt()` returns a dictionary containing `points` as original-image `[x,y]` pixel coordinates and matching `labels` (1 positive,0 negative), or `box` as `[x0,y0,x1,y1]`. The bus image is810×1080; football image1280×720. A model input point is geometry, not a text class.
    A prepared baseline uses a point on the bus instead of the intended person region: inspect its failure before improving it. For the bus photo, your goal is the person near x280,y650; use the displayed photograph and add exclusion/box evidence. For football, inspect the supplied portrait regions. Hold image and weights fixed; change one prompt item per comparison.
    Export both baseline and your changed prompt, actual candidate scores and masks. This cell does not train weights. First online preflight downloads about1GB of official model files; cached offline inference uses the same local revisions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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

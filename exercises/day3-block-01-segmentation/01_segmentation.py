import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="Compare Semantic and Instance Masks")


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
    ## Mask detective: good overlap, wrong object count?

    **Around 30 minutes in pairs.** Semantic masks label classes; instance masks identify objects. U-Net skips preserve spatial detail. IoU/Dice compare aligned regions; class overlap alone cannot certify object count.

    ### Investigate, do not just run

    1. Inspect the separate-shapes scene and predict class overlap and instance count after objects touch. Name the class and reference you will compare.

    2. Complete class membership and IoU/Dice TODOs. Hand-check a tiny mask pair and the stated empty-mask convention before interpreting real learned outputs.

    3. Keep the selected class and models fixed; compare Separate shapes and Touching shapes. Record class IoU/Dice, reference versus predicted identities, and point to the boundary where an identity is lost or retained.

    4. Swap roles and inspect Touching ambiguity as a stress case. Explain whether the failure is wrong class membership, merged identity or boundary displacement. Save an overlay that supports your diagnosis; unchanged results also need evidence.

    5. Decide whether semantic or instance output is enough for counting individual shapes. Cite overlap AND identity evidence; explain why these authored scenes cannot establish performance on real clinical or industrial images.

    **Evidence log:** scene | selected class | IoU | Dice | reference/predicted identities | boundary evidence. Save a run before changing controls. Export both the current findings and your comparison log.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Compare Semantic and Instance Masks

    Investigation after guided setup. Run supplied pretrained tiny U-Nets on held-out authored scenes; compare semantic classes, learned instance identities and the retained Day1 cluster mask.

    Predict → complete short TODOs → change one setting → inspect → explain → download.

    Work in pairs and swap editor/observer. Edit only TODO regions; Shift+Enter runs cells. Ctrl+S / Command+S saves code. Download findings to retain control values and observations. The fresh starter waits politely; prepared loading/plotting/serving code needs no edits. No GPU or API key.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    scene_choice=mo.ui.dropdown(options={'Separate shapes':0,'Touching shapes':1,'Touching ambiguity':2,'Exact Day1 scene':3},value='Touching shapes',label='Scene')
    class_choice=mo.ui.dropdown(options={'Warm shape':1,'Cool shape':2,'Background':0},value='Warm shape',label='Semantic class')
    prediction=mo.ui.text_area(label='Predict: can foreground overlap be high while object count is wrong?',full_width=True)
    mo.vstack([prediction,mo.hstack([scene_choice,class_choice])])
    return class_choice, prediction, scene_choice


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Read labels, regions and identities separately
    A semantic output assigns one class per pixel. A class-selection mask is Boolean membership in that class. An instance map also supplies separate IDs for individual objects. The foreground union can remain the same even if identities merge.

    For aligned masks $A,B$, IoU is $|A\cap B|/|A\cup B|$ and Dice is $2|A\cap B|/(|A|+|B|)$. Counts refer to pixels in the SAME class or matched instance. Use the notebook's stated convention when both masks are empty. Read overlap beside the overlay and identity count: good foreground overlap can coexist with a counting failure. U-Net skips combine retained spatial detail with decoded features; this architecture alone does not guarantee instance IDs.
    """)
    return






@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Your two small steps in the supplied segmentation pipeline

    The models already produce label maps. Your work is to turn one semantic class into a Boolean region, then measure that region against the reference. You are not training a U-Net or writing an instance-separation algorithm.

    Complete the two functions below, run their cell with **Shift+Enter**, then use the scene and class controls to inspect the prepared overlays and report. The first task supplies the prediction mask needed by the second. Keep the selected class fixed when comparing scenes; export your first run before changing controls.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
    """)
    return



@app.cell
def _(np):
    def select_class(labels,class_id):
        # TODO 1: Boolean membership for this class.
        mask=None
        # END TODO 1
        return mask
    def overlap(a,b):
        # TODO 2: IoU and Dice with the stated empty-mask convention.
        metrics=None
        # END TODO 2
        return metrics

    return overlap, select_class


@app.cell(hide_code=True)
def _(class_choice, day3, json, mo, overlap, scene_choice, select_class):
    seg_result=None
    _seg_error=None
    try:seg_result=day3.segmentation_report(select_class,overlap,scene_choice.value,class_choice.value)
    except (ValueError,TypeError,KeyError,ZeroDivisionError) as _e:_seg_error=str(_e)
    mo.callout(_seg_error,kind='warn') if _seg_error else mo.callout('Complete class selection and overlap TODOs.',kind='info') if seg_result is None else mo.vstack([seg_result[1],mo.md('```json\n'+json.dumps(seg_result[0],indent=2)+'\n```')])
    return (seg_result,)


@app.cell(hide_code=True)
def _(mo):
    experiment_log=mo.ui.text_area(value="scene | selected class | IoU | Dice | reference/predicted identities | boundary evidence\nA | \nB | \nStress case | \nDecision + limitation | ",label='Comparison log — preserve run A before changing a control',full_width=True)
    mo.vstack([mo.md("### Keep your evidence while experimenting\nEach rerun replaces the current result. Copy measured values and name a visible feature or response field here before changing controls."),experiment_log])
    return (experiment_log,)


@app.cell(hide_code=True)
def _(display, experiment_log, mo):
    display.download(mo,experiment_log.value,{'columns':"scene | selected class | IoU | Dice | reference/predicted identities | boundary evidence"},'day3-block1-comparison-log',label='Download comparison log (baseline + changed runs)')
    return


@app.cell(hide_code=True)
def _(mo):
    observations=mo.ui.text_area(label='Findings: controlled change, measured result, failure explanation and limitation',full_width=True)
    observations
    return (observations,)


@app.cell(hide_code=True)
def _(display, mo, observations, prediction, seg_result):
    _values={'prediction':prediction.value,'report':None if seg_result is None else seg_result[0]}
    _downloads=[display.download(mo,observations.value,_values,'segmentation-findings')]
    if seg_result is not None:
        import io
        _plot=io.BytesIO();seg_result[1].savefig(_plot,format='png',dpi=150)
        _downloads.append(mo.download(_plot.getvalue(),filename='semantic-instance-overlays.png',label='Download overlays'))
    mo.vstack(_downloads)
    return


if __name__ == "__main__":
    app.run()

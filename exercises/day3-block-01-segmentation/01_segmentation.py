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

    **30 minutes in pairs.** Semantic masks label classes; instance masks identify objects. U-Net skips preserve spatial detail. IoU/Dice compare aligned regions; class overlap alone cannot certify object count.

    ### Investigate, do not just run

    1. Inspect the separate-shapes scene and predict class overlap and instance count after objects touch. Name the class and reference you will compare.

    2. Complete class membership and IoU/Dice TODOs. Hand-check a tiny mask pair and the stated empty-mask convention before interpreting real learned outputs.

    3. Keep the selected class and models fixed; compare Separate shapes and Touching shapes. Record class IoU/Dice, reference versus predicted identities, and point to the boundary where an identity is lost or retained.

    4. Swap roles and inspect Touching ambiguity as a stress case. Explain whether the failure is wrong class membership, merged identity or boundary displacement. Save an overlay that supports your diagnosis; unchanged results also need evidence.

    5. Decide whether semantic or instance output is enough for counting individual shapes. Cite overlap AND identity evidence; explain why these authored scenes cannot establish performance on real clinical or industrial images.

    **Evidence log:** scene | selected class | IoU | Dice | reference/predicted identities | boundary evidence. Save a run before changing controls. Export both the current findings and your comparison log.

    0–4 min: read the takeaways, inspect inputs and predict. 4–12: implement the bounded TODOs and inspect a real result. 12–22: compare controlled runs; swap editor/checker at minute 15. 22–27: diagnose a stress case and choose. 27–30: agree, save and export evidence.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Compare Semantic and Instance Masks

    30-minute investigation after guided setup. Run supplied pretrained tiny U-Nets on held-out authored scenes; compare semantic classes, learned instance identities and the retained Day1 cluster mask.

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
    mo.md("""
    ## TODO — select a class and calculate region overlap
    `select_class(labels,class_id)` receives an H×W integer semantic label grid. Return an H×W Boolean region for that class.
    `overlap(a,b)` receives same-shaped Boolean masks. Return a dictionary with dimensionless `iou` and `dice` values. Use intersection/union/region pixel counts. Convention: both empty gives1 for both; only one empty gives0. Do not treat arbitrary instance IDs as semantic classes.
    Conceptual hint: define the region first, then distinguish pixels in both masks from pixels in either.
    Inspect all three output meanings. Download semantic/instance overlays and record a touching-object failure; retain the exact Day1 scene as a separate domain comparison.
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

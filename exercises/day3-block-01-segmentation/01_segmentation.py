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

    return day3, display, json, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Exercise + breakout: your shared task

    **15 minutes in pairs after guided setup.** Assess segmentation by both pixel overlap and the task’s class/identity requirements.

    1. Predict whether high foreground overlap can coexist with a wrong object count. Complete both TODOs and check one small mask pair manually.

    2. Compare separate versus touching shapes with model and warm-class selection fixed. Record semantic/instance overlays, overlap and observed identity/count errors.

    3. Inspect the exact Day 1 scene and retained cluster mask. Explain which method supplies class meaning and which supplies object identities; use a concrete boundary or touching-object failure as evidence.

    **Bring back:** Overlap calculation, semantic/instance comparisons and a supported distinction between mask quality and instance correctness.

    Swap editor/checker halfway. Record in the observation fields, save code and download findings. The complete task and setup are in this block’s `GROUP_TASK.md`. Prerequisite TODOs discussed in plenary belong to setup.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Compare Semantic and Instance Masks

    15-minute group task after guided setup. Run supplied pretrained tiny U-Nets on held-out authored scenes; compare semantic classes, learned instance identities and the retained Day1 cluster mask.

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
    mo.md("""
    ## TODO — select a class and calculate region overlap
    `select_class(labels,class_id)` receives an H×W integer semantic label grid. Return an H×W Boolean region for that class.
    `overlap(a,b)` receives same-shaped Boolean masks. Return a dictionary with dimensionless `iou` and `dice` values. Use intersection/union/region pixel counts. Convention: both empty gives1 for both; only one empty gives0. Do not treat arbitrary instance IDs as semantic classes.
    Conceptual hint: define the region first, then distinguish pixels in both masks from pixels in either.
    Inspect all three output meanings. Download semantic/instance overlays and record a touching-object failure; retain the exact Day1 scene as a separate domain comparison.
    """)
    return


@app.cell
def _():
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

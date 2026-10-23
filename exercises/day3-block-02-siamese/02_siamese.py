import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="Compare Image Pairs")


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(next(p for p in Path(__file__).resolve().parents if (p/'cv_exercises').is_dir())))
    import marimo as mo
    import numpy as np
    import json
    from cv_exercises import day3,display

    return day3, display, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Exercise + breakout: your shared task

    **15 minutes in pairs after guided setup.** Connect embedding distance, pair-loss convention and decision-cost tradeoffs.

    1. Predict which error type will grow with a higher threshold. Complete both TODOs and check one similar and one dissimilar pair loss with the stated label convention.

    2. Keep pairs/encoder/embeddings fixed. Compare thresholds 0.5 and 0.9, counting false matches and missed matches; identify a changed decision or explain why none crossed the cutoff.

    3. Choose a threshold for a scenario where false matches are more costly. Justify it with observed errors and describe how you would validate it on new pairs without tuning on the test set.

    **Bring back:** Checked pair losses, two-threshold error table, one explained pair and a conditional threshold decision.

    Swap editor/checker halfway. Record in the observation fields, save code and download findings. The complete task and setup are in this block’s `GROUP_TASK.md`. Prerequisite TODOs discussed in plenary belong to setup.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Compare Image Pairs

    15-minute group task after guided setup. Use a supplied trained Siamese encoder, held-out pairs and margin convention y=1 for similar; no training from scratch.

    Predict → complete short TODOs → change one setting → inspect → explain → download.

    Work in pairs and swap editor/observer. Edit only TODO regions; Shift+Enter runs cells. Ctrl+S / Command+S saves code. Download findings to retain control values and observations. The fresh starter waits politely; prepared loading/plotting/serving code needs no edits. No GPU or API key.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    threshold=mo.ui.slider(0.,2.,step=.01,value=.7,label='Match threshold: distance ≤ threshold')
    prediction=mo.ui.text_area(label='Predict which error type grows when the threshold increases',full_width=True)
    mo.vstack([prediction,threshold])
    return prediction, threshold


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## TODO — distance and contrastive loss
    `distance(a,b)`: two P×8 embedding arrays, one row per pair. Return P Euclidean distances, reducing embedding coordinates only.
    `contrastive(d,y,margin)`: P distances, P labels (1 similar,0 different), positive margin. Return P losses using the formula from the teaching slides.
    Hint: similar and dissimilar pairs need different terms; dissimilar loss is zero when the margin is met.
    Change threshold while images/encodings stay fixed. Record a false match and a missed match at deliberately different thresholds, not a claim that both occur at the default.
    """)
    return


@app.cell
def _(np):
    def distance(a,b):
        # TODO 1: one Euclidean distance per pair.
        distances=None
        # END TODO 1
        return distances
    def contrastive(d,y,margin):
        # TODO 2: one loss per pair; y=1 means similar.
        losses=None
        # END TODO 2
        return losses

    return contrastive, distance


@app.cell(hide_code=True)
def _(contrastive, day3, distance, mo, threshold):
    pair_result=None
    _pair_error=None
    try:pair_result=day3.pair_report(distance,contrastive,threshold.value)
    except (ValueError,TypeError,KeyError) as _e:_pair_error=str(_e)
    mo.callout(_pair_error,kind='warn') if _pair_error else mo.callout('Complete distance/loss TODOs.',kind='info') if pair_result is None else mo.vstack([pair_result[1],mo.ui.table(pair_result[0]['pairs']),mo.md('False matches: '+str(pair_result[0]['false_matches'])+' · missed matches: '+str(pair_result[0]['missed_matches']))])
    return (pair_result,)


@app.cell(hide_code=True)
def _(mo):
    observations=mo.ui.text_area(label='Findings: controlled change, measured result, failure explanation and limitation',full_width=True)
    observations
    return (observations,)


@app.cell(hide_code=True)
def _(display, mo, observations, pair_result, prediction, threshold):
    display.download(mo,observations.value,{'prediction':prediction.value,'threshold':threshold.value,'report':None if pair_result is None else pair_result[0]},'pair-findings')
    return


if __name__ == "__main__":
    app.run()

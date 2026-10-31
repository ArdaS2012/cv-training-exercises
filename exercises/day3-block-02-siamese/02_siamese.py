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
    ## Match desk: which mistake costs more?

    **Around 30 minutes in pairs.** Both branches share encoder weights. Distance compares embeddings; a decision threshold does not retrain them. Here y=1 means similar. A wider acceptance threshold trades missed matches for false matches.

    ### Investigate, do not just run

    1. Look at a similar and dissimilar image pair; predict their distance ordering and which error increases when the cutoff grows. State the y=1 convention before coding.

    2. Complete per-pair Euclidean distance and contrastive-loss TODOs. Manually trace one similar and one dissimilar loss, including the margin case; inspect the actual pair images beside measured distances.

    3. Freeze pairs/encoder/embeddings and compare cutoffs 0.5 and 0.9. Record false matches and missed matches; identify the pair whose decision changes, or explain why no distance crossed the cutoff.

    4. Swap roles. Act as two clients: an access system that fears false matches, and a search tool that fears missed matches. Use the SAME error table to defend a different threshold for each; inspect one visual counterexample to your intuition.

    5. Choose one policy and explain its cost tradeoff with an actual pair. Specify how new validation pairs would set the threshold and untouched test pairs would estimate performance; do not tune repeatedly on these teaching pairs.

    **Evidence log:** cutoff | false matches | missed matches | changed pair | intended error cost. Save a run before changing controls. Export both the current findings and your comparison log.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Compare Image Pairs

    Investigation after guided setup. Use a supplied trained Siamese encoder, held-out pairs and margin convention y=1 for similar; no training from scratch.

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
    mo.md(r"""
    ### Read one image pair at a time
    The same learned encoder maps both images into eight-coordinate feature vectors. For each pair, $d=\|a-b\|_2$ is Euclidean distance: compare corresponding coordinates and reduce within that pair, not across the batch. Small distance means close representations under this encoder, not guaranteed semantic equivalence.

    Here $y=1$ denotes a similar pair. Contrastive loss is $L=yd^2+(1-y)\max(0,m-d)^2$, where $m>0$ is the separation margin. Similar pairs are penalized for distance; dissimilar pairs stop paying once separated beyond the margin. Decision cutoff and training margin are different settings. Accept a match when distance is at most the cutoff; increasing it accepts more pairs without changing embeddings.
    """)
    return






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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Function inputs and API reference

    ### TODO 1: `distance(a, b)`

    `a` and `b` are finite floating-point NumPy arrays of identical shape `(P, 8)`, currently `(24, 8)`. `P` is the number of image pairs, not the number of pixels. Row `i` in `a` is the left image embedding for pair `i`; row `i` in `b` is that **same pair's** right embedding. Axis 1 contains eight learned feature coordinates. A coordinate is not a digit label. The supplied encoder makes each vector unit length, so its coordinates are dimensionless and pair distances lie in `[0, 2]` up to numerical rounding.

    Return a NumPy floating-point array of shape `(P,)`: one nonnegative Euclidean distance per corresponding pair, in input row order. Do not return an all-pairs `P×P` matrix, squared distances, or one scalar for the entire batch. Keep both arrays unchanged. The Euclidean-distance definition appears immediately above the TODO.

    ### TODO 2: `contrastive(d, y, margin)`

    | Input | Definition and guarantees |
    |---|---|
    | `d` | Finite nonnegative NumPy float array shape `(P,)`: the distance for each pair, in the same order as the labels. |
    | `y` | NumPy array shape `(P,)` of binary integer labels: **1 means similar**, **0 means different**. This is the opposite of some libraries' conventions; use this notebook's definition. |
    | `margin` | One positive finite float `m`, currently `1.0`, in the same distance units as `d`. It is the training separation margin, not the UI match threshold. |

    Return a NumPy float array of shape `(P,)`: the per-pair losses from the exact equation printed above, with **no extra factor of one half**. Do not average the losses; the supplied report computes the mean separately. The result is finite and nonnegative; at the margin a dissimilar pair has zero loss. Keep `y` aligned with the original pair rows.

    **API vocabulary:** NumPy axis `0` selects the pair/batch dimension; axis `1` selects embedding coordinates. `np.linalg.norm` accepts an `axis` to choose the reduction dimension. `np.maximum(a, b)` takes an elementwise maximum (it does not reduce an axis); scalar arithmetic broadcasts over an array. This vocabulary defines operations, not an implementation sequence.

    **Official references:** [NumPy norm](https://numpy.org/doc/2.2/reference/generated/numpy.linalg.norm.html), [NumPy maximum](https://numpy.org/doc/2.2/reference/generated/numpy.maximum.html), [broadcasting](https://numpy.org/doc/2.2/user/basics.broadcasting.html).
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
    experiment_log=mo.ui.text_area(value="cutoff | false matches | missed matches | changed pair | intended error cost\nA | \nB | \nStress case | \nDecision + limitation | ",label='Comparison log — preserve run A before changing a control',full_width=True)
    mo.vstack([mo.md("### Keep your evidence while experimenting\nEach rerun replaces the current result. Copy measured values and name a visible feature or response field here before changing controls."),experiment_log])
    return (experiment_log,)


@app.cell(hide_code=True)
def _(display, experiment_log, mo):
    display.download(mo,experiment_log.value,{'columns':"cutoff | false matches | missed matches | changed pair | intended error cost"},'day3-block2-comparison-log',label='Download comparison log (baseline + changed runs)')
    return


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

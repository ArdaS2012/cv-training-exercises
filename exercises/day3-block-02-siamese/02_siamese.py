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
    mo.md(r"""
    ## Your two small steps in the supplied pair comparison

    The trained shared encoder and the image pairs are supplied. You calculate one distance and one loss for each pair. Neither TODO trains a model or chooses the match threshold.

    Complete the two functions below and run their cell with **Shift+Enter**. Then compare the same pairs at different thresholds. Keep the pair order, encoder and training margin fixed. Export the first error table before changing the threshold.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TODO definitions and API reference

    ### First understand a pair and its embedding

    A **pair** contains a left image and a right image. An **encoder** converts an image into a vector of learned features, called an **embedding**. Both branches here use the same encoder weights. The eight coordinates describe learned features; they are not eight pixels or digit labels. Distance is computed between the two vectors belonging to the **same pair**.

    NumPy is already imported as `np`. An array with shape `(P, D)` has `P` rows (pairs) and `D` columns (feature coordinates). **Reducing an axis** means combining its entries into a shorter result: one result per pair must retain the row axis and combine coordinates within each row.

    ### TODO 1 — distance for each pair: `distance(a, b)`

    | Argument | Type, meaning and guarantees |
    | --- | --- |
    | `a` | Finite floating-point NumPy array `(P, D)` on CPU: row `i` is the left image's embedding for pair `i`. Current data has `(P, D) = (24, 8)`. |
    | `b` | Finite floating-point NumPy array of the identical shape: row `i` is the right image's embedding for that same pair. Columns have the same feature meaning as in `a`. |

    The model supplies nonempty batches and positive feature dimensions; do not hard-code `24` or `8`. No labels or images need loading in this function. Coordinates are dimensionless. The current encoder makes vectors unit length, so current-data Euclidean distances are in `[0, 2]` up to floating-point rounding; the distance definition also works for other finite vectors used in a manual check.

    For pair `i`, with feature index `j`:

    $$d_i=\sqrt{\sum_{j=1}^{D}(a_{ij}-b_{ij})^2}.$$

    `a_ij` and `b_ij` are corresponding coordinates. The sum combines coordinates **inside one pair**; it does not compare every left image with every right image. The square root returns Euclidean distance rather than squared distance.

    **Return.** A floating-point NumPy array `(P,)`: one finite nonnegative scalar per pair, preserving row order. `(P,)` is a one-dimensional array, not `(P, 1)`, an all-pairs `(P, P)` matrix or a single batch-wide number. Equal vectors have distance zero. Keep both inputs unchanged.

    **Available operation.** `np.linalg.norm(x, ord=None, axis=None, keepdims=False)` returns a norm. For a vector or one selected axis with default `ord`, it computes the Euclidean norm. `axis` chooses which axis to reduce; keeping dimensions is optional and is off by default. Calling it without an axis on a matrix reduces the whole matrix, which differs from one result per row. NumPy arithmetic between same-shaped arrays combines matching entries and returns an array.

    ### TODO 2 — loss for each pair: `contrastive(d, y, margin)`

    A **loss** is a penalty used during learning. Here you calculate its value for already-computed embeddings to understand the objective; changing this function or the UI threshold does not retrain the supplied checkpoint.

    | Argument | Type, meaning and guarantees |
    | --- | --- |
    | `d` | Finite nonnegative NumPy float array `(P,)`: the distances from TODO 1, in original pair order. |
    | `y` | Finite NumPy binary numeric array `(P,)`, aligned with `d`: the live data use `float64`, with **`1.0` meaning similar and `0.0` meaning different**. Integer or Boolean 0/1 arrays in small manual examples have the same meaning. These are pair targets, not digit class IDs. Some references use the opposite convention; follow this one. |
    | `margin` | Positive finite scalar float `m`, currently `1.0`, in the same dimensionless distance units as `d`. This is the dissimilar-pair separation margin, not the UI decision threshold. |

    For each pair, use this exact definition:

    $$L_i=y_i d_i^2+(1-y_i)\max(0,m-d_i)^2.$$

    A similar pair uses the distance penalty; a dissimilar pair uses only the shortfall from the margin. `max(0, …)` means no penalty once the required separation is reached. Squaring makes a larger shortfall more costly. There is **no additional factor of one half** in this exercise's definition.

    **Return.** A floating-point NumPy array `(P,)`, with one finite nonnegative loss per input pair in the same order. Keep `d` and `y` unchanged. Do not sum or average; the report computes the mean separately. At `d == margin`, a dissimilar pair has zero loss; above the margin it remains zero. A similar pair at zero distance has zero loss. The runner supplies nonempty, aligned valid inputs; validation of malformed batches is outside the TODO.

    **Available operations.** `np.maximum(x1, x2)` returns the elementwise larger value from its two inputs. A scalar such as zero broadcasts to every position of an array; **broadcasting** means using that scalar at each position without manually building a same-length array. `np.maximum` does not reduce an axis. Normal arithmetic and powers work elementwise on NumPy arrays; Python `^` denotes bitwise XOR, not squaring. These operations return new results unless an output argument or in-place operator is explicitly used.

    ### Run, inspect and explain

    Replace `distances = None` and `losses = None` within the marked regions; keep the signatures and existing return lines. Run the cell with **Shift+Enter**. While either function returns `None`, the report waits. Its finite-value/nonnegative/shape checks establish structure only; independently trace a small similar and dissimilar case using the equations above.

    The supplied `day3.pair_report(distance, contrastive, threshold=0.7)` calls both functions on the fixed pairs, using margin `1.0`. It returns `None` for an unfinished function, otherwise `(report, figure)`. A threshold is a nonnegative distance cutoff; the UI selects `[0, 2]`. The report accepts a match when `distance <= threshold`, including equality. It loads a fixed encoder and does not update weights.

    Dotted field paths below describe entries in the displayed JSON. They are Python dictionary keys, not object attributes. `report.pairs` is an ordered list of records with `pair` (zero-based row ID), `left_digit`/`right_digit` (image labels), `similar_target` (Boolean reference), `distance`, `threshold`, `match` (Boolean decision), `correct` (decision agrees with target) and `loss`. The chart shows the first eight image pairs; the table contains all 24. A **false match** accepts a reference-different pair; a **missed match** rejects a reference-similar pair. `mean_loss` averages your per-pair losses; it is unchanged by a threshold-only edit. Increasing the threshold can accept more pairs but cannot change distances or losses. Distinguish the cost of each error before recommending a threshold, and reserve fresh pairs for validation/testing beyond these inspected examples.

    **Official API references (NumPy 2.2):** [vector norms and axis](https://numpy.org/doc/2.2/reference/generated/numpy.linalg.norm.html), [elementwise maximum](https://numpy.org/doc/2.2/reference/generated/numpy.maximum.html), [broadcasting rules](https://numpy.org/doc/2.2/user/basics.broadcasting.html). The notebook's label convention and local report contract are defined above.
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

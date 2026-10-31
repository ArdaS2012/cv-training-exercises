import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="Reduce and Reconstruct Image Data")


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p/'cv_exercises').is_dir())))
    import marimo as mo
    import numpy as np
    from cv_exercises import data, display

    return data, display, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Hands-on investigation: your shared mission

    **Around 30 minutes in pairs.** Choose an image compression budget: how few coordinates preserve the detail your user needs?

    ### Prior-slide takeaways to use now

    Slides 23–28: center first; PCA uses SVD directions and discards components. Error and retained variance answer different questions; 2D gaps do not prove classes.

    1. Choose one visible detail in input row 0 that must survive compression; predict the smallest acceptable rank. Complete the three PCA expressions, checking 300×64 → 300×k → 300×64 shapes. Explain why the mean must be restored.

    2. Use the identical rows at k=1, 8 and 32. Log retained variance, MSE and whether that detail survives; save each run. Swap roles and choose the smallest tested rank meeting your declared goal, even when its plot looks less attractive.

    3. Run the supplied nonlinear views at neighbor setting 5 and 30 while keeping rows/seed fixed. Track a recognizable labelled region and describe what changed. Challenge a class-separation claim with evidence; explain why perplexity and UMAP neighbor count are different controls.

    **Bring back:** Three measured rank rows, a detail-preservation decision and a fixed-row neighborhood comparison with an interpretation limit.

    Record each run in the evidence log before changing a control. Downloads include the current run plus your cumulative written log; save after each trial. See this block’s `GROUP_TASK.md`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Reduce and Reconstruct Image Data

    Investigation after guided setup. Use centered SVD to calculate PCA projections and reconstructions. Compare retained variance and image error at three k values. Run the supplied t-SNE and UMAP helpers on the same rows with fixed seeds; record two neighborhood-interpretation limitations.

    Inspect → predict → implement and trace → compare → challenge → explain and download evidence.

    Use the cell code control to open TODO cells. Edit only `# TODO` areas and run with Shift+Enter. Waiting messages are normal before completion. Save Python with Ctrl+S / Command+S; widget values need the findings download. Work in pairs and swap who edits.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    components=mo.ui.slider(1,64,value=8,label='Retained components k')
    neighbors=mo.ui.dropdown(options=[5,15,30],value=15,label='t-SNE perplexity / UMAP neighbors (different meanings)')
    run_views=mo.ui.run_button(label='Run nonlinear views (CPU)')
    prediction=mo.ui.text_area(label='Prediction: will more components reduce reconstruction error?',full_width=True)
    mo.hstack([components,neighbors,run_views],wrap=True)
    return components, neighbors, prediction, run_views


@app.cell(hide_code=True)
def _(data, np):
    x,y,sample_ids=data.images()
    mean=x.mean(axis=0)
    centered=x-mean
    u,singular,vt=np.linalg.svd(centered,full_matrices=False)
    return centered, mean, sample_ids, singular, vt, x, y


@app.cell(hide_code=True)
def _(mo, prediction):
    mo.vstack([prediction,mo.md("""## TODO 1 — PCA inputs and return contract

    All numeric inputs below are finite NumPy arrays supplied by the preceding setup cell. `np` is NumPy, imported for the notebook. Shapes use `(rows, columns)`.

    | Available input | Definition |
    |---|---|
    | `centered` | Float array `(300, 64)`. Each row is one 8×8 digit image, flattened row by row; column `j` is one fixed pixel position. The mean of each feature column has already been subtracted. Values are intensity differences, so negatives are expected. |
    | `mean` | Float array `(64,)`, the 300-image mean intensity at each pixel; original intensities were normalized to `[0, 1]`. Its feature order matches `centered`. |
    | `vt` | Float array `(64, 64)` from the supplied `np.linalg.svd(centered, full_matrices=False)`. Each **row** is a unit feature-space direction, ordered by decreasing singular value. All 64 directions are provided; do not refit SVD. |
    | `components` | Marimo slider object. `.value` is the selected integer `k`, from 1 to 64: how many leading principal directions to retain. Use the setting as a number, rather than the widget object. |

    PCA represents each centered sample by coordinates along the retained orthogonal directions. The retained basis consists of the leading `k` directions as **columns**. Reconstruction returns from those coordinates to the original 64-feature intensity space, including the supplied mean. This describes the required representation, rather than Python expressions for it.

    Return all three named arrays: `basis`, shape `(64, k)`; `projected`, shape `(300, k)`; `reconstructed`, shape `(300, 64)`. Row `i` must always refer to the same image. For a shape-only example with 5 images, 6 features and `k=2`, these outputs would have shapes `(6, 2)`, `(5, 2)`, `(5, 6)`. At `k=64`, reconstruction should differ from the input only by floating-point roundoff. At smaller `k`, approximate reconstructed intensities can leave `[0, 1]`; do not clip them for this comparison. None of these arrays has a class-label column. Leave `None` only while unfinished.

    ### API reference

    NumPy 2.2: [SVD return conventions](https://numpy.org/doc/2.2/reference/generated/numpy.linalg.svd.html), [slicing/indexing](https://numpy.org/doc/2.2/user/basics.indexing.html) (`a[start:stop]` excludes `stop`), [transpose](https://numpy.org/doc/2.2/reference/generated/numpy.ndarray.T.html) (swap array axes), and [matrix multiplication](https://numpy.org/doc/2.2/reference/generated/numpy.matmul.html) (`@`: `(a,b)` with `(b,c)` gives `(a,c)`). The nonlinear optimizer views below are supplied and are not TODOs.""")])
    return


@app.cell
def _(centered, components, mean, vt):
    # TODO 1: three small expressions for basis, projection, reconstruction.
    basis=None
    projected=None
    reconstructed=None
    # END TODO 1
    return basis, projected, reconstructed


@app.cell(hide_code=True)
def _(display, neighbors, run_views, x):
    nonlinear=None
    if run_views.value:
        nonlinear=display.nonlinear_views(x,neighbors.value)
    return (nonlinear,)


@app.cell(hide_code=True)
def _(
    basis,
    components,
    display,
    mo,
    nonlinear,
    np,
    projected,
    reconstructed,
    singular,
    x,
    y,
):
    _view = None
    if any(v is None for v in (basis,projected,reconstructed)):
        _view = mo.callout('Waiting for the three PCA expressions. Nonlinear views can run independently.',kind='info')
    elif np.asarray(projected).shape!=(len(x),components.value) or np.asarray(reconstructed).shape!=x.shape:
        _view = mo.callout('Projection must be 300×k; reconstruction must restore 300×64.',kind='warn')
    elif nonlinear is None:
        _view = mo.vstack([mo.md('PCA arrays ready. Press Run nonlinear views to compute both neighborhood layouts.'),display.reduction_plot(x,reconstructed,singular,components.value,{},y)])
    else:
        _view = mo.vstack([display.reduction_plot(x,reconstructed,singular,components.value,nonlinear,y),mo.md('These are real optimizer outputs, fixed seed 42. Digit colors are labels for interpretation only. Axes, gaps and island sizes do not certify semantic classes.')])
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    observations=mo.ui.text_area(value='Rank k | retained variance | MSE | chosen detail survives? | evidence in row 0\n1 | | | |\n8 | | | |\n32 | | | |\nAccepted detail/goal and smallest acceptable tested rank:\nNonlinear setting 5 versus 30: one located change per method:\nClaim the plots cannot establish and why:',label="Evidence log — keep every trial, mechanism and decision",full_width=True)
    observations
    return (observations,)


@app.cell(hide_code=True)
def _(
    components,
    display,
    mo,
    neighbors,
    nonlinear,
    np,
    observations,
    prediction,
    reconstructed,
    sample_ids,
    singular,
    x,
):
    measurements={'k':components.value,'neighbor_setting':neighbors.value,'seed':42,'sample_ids':sample_ids.tolist(),'prediction':prediction.value,'variance_fraction':float(np.sum(singular[:components.value]**2)/np.sum(singular**2))}
    if reconstructed is not None: measurements['mse']=float(np.mean((x-reconstructed)**2))
    if nonlinear is not None: measurements['views']={n:v.tolist() for n,v in nonlinear.items()}
    display.download(mo,observations.value,measurements,'reduction-findings')
    return


if __name__ == "__main__":
    app.run()

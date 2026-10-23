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

    **30 minutes in pairs.** Choose an image compression budget: how few coordinates preserve the detail your user needs?

    ### Prior-slide takeaways to use now

    Slides 23–28: center first; PCA uses SVD directions and discards components. Error and retained variance answer different questions; 2D gaps do not prove classes.

    1. Choose one visible detail in input row 0 that must survive compression; predict the smallest acceptable rank. Complete the three PCA expressions, checking 300×64 → 300×k → 300×64 shapes. Explain why the mean must be restored.

    2. Use the identical rows at k=1, 8 and 32. Log retained variance, MSE and whether that detail survives; save each run. Swap roles at minute 15 and choose the smallest tested rank meeting your declared goal, even when its plot looks less attractive.

    3. Run the supplied nonlinear views at neighbor setting 5 and 30 while keeping rows/seed fixed. Track a recognizable labelled region and describe what changed. Challenge a class-separation claim with evidence; explain why perplexity and UMAP neighbor count are different controls.

    **Bring back:** Three measured rank rows, a detail-preservation decision and a fixed-row neighborhood comparison with an interpretation limit.

    **Route:** 0–4 predict; 4–12 implement and trace; 12–22 compare (swap roles at 15); 22–27 challenge and decide; 27–30 explain, save and export. Record each run in the evidence log before changing a control. Downloads include the current run plus your cumulative written log; save after each trial. See this block’s `GROUP_TASK.md`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Reduce and Reconstruct Image Data

    30-minute investigation after guided setup. Use centered SVD to calculate PCA projections and reconstructions. Compare retained variance and image error at three k values. Run the supplied t-SNE and UMAP helpers on the same rows with fixed seeds; record two neighborhood-interpretation limitations.

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
    mo.vstack([prediction,mo.md("""## TODO: PCA through the supplied SVD
    The centered array is 300×64. vt contains right singular vectors as rows. Select k feature directions as columns (64×k), multiply centered rows by that basis, then project back and add the 64-feature mean. Return all three arrays. Use @ for matrix multiplication and .T to transpose. The optimizer helpers below are supplied, not a TODO.""")])
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

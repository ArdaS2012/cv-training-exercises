import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="Segment a Small Image")


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
    ## Exercise + breakout: your shared task

    **15 minutes in pairs after guided setup.** Explain how feature design changes pixel partitions and why a partition does not establish object identity.

    1. Predict whether the two distant red regions will share a group. Complete feature/reshape TODOs and compare the color-only mask with the spatial mask at weight 0.5, same cluster count and seed.

    2. Change only position weight to 1.0. Compare both spatial masks, identifying one boundary or disconnected region that changes and one region that does not.

    3. Decide whether either mask justifies an object-count or semantic-label claim. Use a lighting or same-color counterexample and retain the baseline for Day 3.

    **Bring back:** Color-only and spatial masks, fixed-settings comparison and a supported usefulness/failure explanation.

    Swap editor/checker halfway. Record in the observation fields, save code and download findings. The complete task and setup are in this block’s `GROUP_TASK.md`. Prerequisite TODOs discussed in plenary belong to setup.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Segment a Small Image

    15-minute group task after guided setup. Compare color-only clustering with color plus scaled position on the supplied authored RGB scene. Export two masks. Explain a useful partition, lighting sensitivity and why a cluster does not establish object meaning or object identity.

    Predict → complete the short TODOs → change one control → explain → download findings.

    Use the cell code control to open TODO cells. Edit only `# TODO` areas and run with Shift+Enter. Waiting messages are normal before completion. Save Python with Ctrl+S / Command+S; widget values need the findings download. Work in pairs and swap who edits.
    """)
    return


@app.cell(hide_code=True)
def _(data):
    rgb=data.scene()
    color,position=data.pixel_inputs(rgb)
    return color, rgb


@app.cell(hide_code=True)
def _(mo):
    cluster_count=mo.ui.slider(2,7,value=3,label='Clusters')
    spatial_weight=mo.ui.slider(0,2,step=0.1,value=0.5,label='Position weight')
    prediction=mo.ui.text_area(label='Prediction: will two distant red regions share a group?',full_width=True)
    mo.hstack([cluster_count,spatial_weight],wrap=True)
    return cluster_count, prediction, spatial_weight


@app.cell(hide_code=True)
def _(mo, prediction):
    mo.vstack([prediction,mo.md("""## TODO: pixels → feature rows → label image
    color is 1536×3 with values in 0..1; position is 1536×2 with normalized row/column coordinates. Build a 1536×5 array by joining color with weighted position using np.column_stack. Then reshape the fitted 1536 labels into the image height and width, without a color channel. Color-only labels and fitting plumbing are supplied.""")])
    return


@app.cell
def _():
    # TODO 1: combine color and weighted position as feature columns.
    spatial_features=None
    # END TODO 1
    return (spatial_features,)


@app.cell(hide_code=True)
def _(cluster_count, color, spatial_features):
    from sklearn.cluster import KMeans
    color_labels=KMeans(n_clusters=cluster_count.value,random_state=42,n_init=10).fit_predict(color)
    spatial_labels=None if spatial_features is None else KMeans(n_clusters=cluster_count.value,random_state=42,n_init=10).fit_predict(spatial_features)
    return (color_labels,)


@app.cell
def _():
    # TODO 2: restore H×W labels when spatial_labels is ready; otherwise None.
    spatial_mask=None
    # END TODO 2
    return (spatial_mask,)


@app.cell(hide_code=True)
def _(color_labels, display, mo, np, rgb, spatial_mask):
    _view = None
    masks={'Color only':color_labels.reshape(rgb.shape[:2])}
    if spatial_mask is None:
        _view = mo.callout('Waiting for feature and mask TODOs. Color-only grouping is supplied as a baseline.',kind='info')
    elif np.asarray(spatial_mask).shape!=rgb.shape[:2]:
        _view = mo.callout('A mask has height×width labels, without a color channel.',kind='warn')
    else:
        masks['Color + position']=spatial_mask
        _view = display.masks_plot(rgb,masks)
    _view
    return (masks,)


@app.cell(hide_code=True)
def _(mo):
    observations=mo.ui.text_area(label='Findings: one useful region, one failure, lighting effect; keep for Day 3',full_width=True)
    observations
    return (observations,)


@app.cell(hide_code=True)
def _(
    cluster_count,
    display,
    masks,
    mo,
    np,
    observations,
    prediction,
    spatial_weight,
):
    display.download(mo,observations.value,{'k':cluster_count.value,'position_weight':spatial_weight.value,'seed':42,'prediction':prediction.value,'masks':{n:np.asarray(m).tolist() for n,m in masks.items()}},'segmentation-findings')
    return


if __name__ == "__main__":
    app.run()

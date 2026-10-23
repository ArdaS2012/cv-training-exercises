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
    ## Hands-on investigation: your shared mission

    **30 minutes in pairs.** Build a region proposal for an image tool: can it separate distant red regions without inventing object labels?

    ### Prior-slide takeaways to use now

    Slides 36–38: preserve pixel↔row order to restore H×W. Color and position change similarity; a cluster is not an object label or instance.

    1. Locate the two red regions and predict whether they will share a group. Complete the feature and H×W reshape TODOs. Use K=3 and seed 42; explain what one feature row represents before running.

    2. Hold image, K and seed fixed; compare position weights 0, 0.5 and 1.0. Log whether probes (row18,col12) and (row10,col36) share a group, and locate a boundary that moves or remains fixed. Swap roles at minute 15. Compare membership, not the numeric value of a cluster ID.

    3. Choose a weight for a declared region-selection purpose. Use the lighting gradient or a split background as a counterexample to your own choice. State what annotation would be needed to call the regions objects; retain the masks for Day 3.

    **Bring back:** Three weight trials with fixed settings, a probe/boundary comparison, chosen mask and a concrete semantic/instance limitation.

    **Route:** 0–4 predict; 4–12 implement and trace; 12–22 compare (swap roles at 15); 22–27 challenge and decide; 27–30 explain, save and export. Record each run in the evidence log before changing a control. Downloads include the current run plus your cumulative written log; save after each trial. See this block’s `GROUP_TASK.md`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Segment a Small Image

    30-minute investigation after guided setup. Compare color-only clustering with color plus scaled position on the supplied authored RGB scene. Export two masks. Explain a useful partition, lighting sensitivity and why a cluster does not establish object meaning or object identity.

    Inspect → predict → implement and trace → compare → challenge → explain and download evidence.

    Use the cell code control to open TODO cells. Edit only `# TODO` areas and run with Shift+Enter. Waiting messages are normal before completion. Save Python with Ctrl+S / Command+S; widget values need the findings download. Work in pairs and swap who edits.
    """)
    return


@app.cell(hide_code=True)
def _(data):
    rgb=data.scene()
    color,position=data.pixel_inputs(rgb)
    return color, position, rgb


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
def _(color, np, position, spatial_weight):
    # TODO 1: combine color and weighted position as feature columns.
    spatial_features=None
    # END TODO 1
    return (spatial_features,)


@app.cell(hide_code=True)
def _(cluster_count, color, spatial_features):
    from sklearn.cluster import KMeans
    color_labels=KMeans(n_clusters=cluster_count.value,random_state=42,n_init=10).fit_predict(color)
    spatial_labels=None if spatial_features is None else KMeans(n_clusters=cluster_count.value,random_state=42,n_init=10).fit_predict(spatial_features)
    return color_labels, spatial_labels


@app.cell
def _(rgb, spatial_labels):
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
        _probes=[{'mask':_name,'red probe (18,12)':int(_mask[18,12]),'red probe (10,36)':int(_mask[10,36]),'same group?':bool(_mask[18,12]==_mask[10,36])} for _name,_mask in masks.items()]
        _view = mo.vstack([display.masks_plot(rgb,masks),mo.ui.table(_probes,selection=None),mo.md('Probe coordinates are (row, column). Equal IDs within one mask mean shared membership; ID values across fits do not name the same object. Log the probes plus one boundary before changing position weight.')])
    _view
    return (masks,)


@app.cell(hide_code=True)
def _(mo):
    observations=mo.ui.text_area(value='Position weight | K/seed | red probes share a group? | changed boundary | useful region / failure\n0 | 3/42 | | |\n0.5 | 3/42 | | |\n1.0 | 3/42 | | |\nRegion-selection goal and choice:\nLighting/background counterexample:\nWhat extra evidence would justify an object label/count?',label="Evidence log — keep every trial, mechanism and decision",full_width=True)
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

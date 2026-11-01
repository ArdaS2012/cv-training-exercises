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

    **Around 30 minutes in pairs.** Build a region proposal for an image tool: can it separate distant red regions without inventing object labels?

    ### Prior-slide takeaways to use now

    Slides 36–38: preserve pixel↔row order to restore H×W. Color and position change similarity; a cluster is not an object label or instance.

    1. Locate the two red regions and predict whether they will share a group. Complete the feature and H×W reshape TODOs. Use K=3 and seed 42; explain what one feature row represents before running.

    2. Hold image, K and seed fixed; compare position weights 0, 0.5 and 1.0. Log whether probes (row18,col12) and (row10,col36) share a group, and locate a boundary that moves or remains fixed. Swap roles. Compare membership, not the numeric value of a cluster ID.

    3. Choose a weight for a declared region-selection purpose. Use the lighting gradient or a split background as a counterexample to your own choice. State what annotation would be needed to call the regions objects; retain the masks for Day 3.

    **Bring back:** Three weight trials with fixed settings, a probe/boundary comparison, chosen mask and a concrete semantic/instance limitation.

    Record each run in the evidence log before changing a control. Downloads include the current run plus your cumulative written log; save after each trial. See this block’s `GROUP_TASK.md`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Segment a Small Image

    Investigation after guided setup. Compare color-only clustering with color plus scaled position on the supplied authored RGB scene. Export two masks. Explain a useful partition, lighting sensitivity and why a cluster does not establish object meaning or object identity.

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
    mo.vstack([prediction,mo.md(r"""
    ## TODO 1 — define which pixels should be considered similar

    **Your job:** create `spatial_features` by joining the supplied RGB values with weighted position features. The cell receives `color`, `np`, `position` and `spatial_weight`. Image creation, coordinate normalization, color-only clustering and K-Means fitting are supplied.

    ### What is one feature row?

    This exercise clusters **pixels**, rather than whole images. A pixel's RGB values describe appearance; its position says where it is. K-Means sees only a numeric row and Euclidean distance. Adding position can prefer nearby pixels even when two distant regions share a color. It does not teach the algorithm a name such as “car”.

    | Input | Exact type, shape and meaning |
    |---|---|
    | `color` | Finite `float64` NumPy array `(1536, 3)`. Each row is one pixel. Columns are **red, green, blue**, in that order; intensities are in `[0, 1]`. |
    | `position` | Finite `float64` NumPy array `(1536, 2)` for the same pixels in the same order. Columns are normalized **row, column**: `row/(H-1)` and `column/(W-1)`, each in `[0, 1]`. Origin is top left; rows increase downward and columns rightward. |
    | `spatial_weight` | Marimo slider object; `.value` is the current nonnegative numeric weight `w`, 0–2 in steps of 0.1. It is dimensionless, not a count or an image coordinate. |
    | `np` | NumPy, already imported. No additional imports are required. |

    The supplied authored scene has height `H=32` and width `W=48`, hence `H*W=1536` pixels. Both arrays use **C / row-major order**: left to right along a row, then the next row. Flat index `i = row*W + column` identifies the same pixel in both inputs. For example, index 49 is coordinate `(1, 1)`, with normalized position `(1/31, 1/47)`. Preserve that correspondence. The scene is an illustration with a lighting gradient, not a labelled photograph.

    ### Feature definition and effect of the weight

    For pixel `i`, the required feature vector is $f_i=(R_i,G_i,B_i,w r_i,w c_i)$, where `R/G/B` are normalized intensities, `r/c` normalized row/column, and `w` the current setting. This gives squared distance

    $$\|f_i-f_j\|^2=\|\mathrm{RGB}_i-\mathrm{RGB}_j\|^2+w^2\|(r_i,c_i)-(r_j,c_j)\|^2.$$

    A zero weight removes position's contribution. A weight of one keeps its normalized scale; doubling the weight multiplies its **squared-distance** contribution by four. This changes the grouping rule, not the image or its pixel order. Repeated colors/duplicate feature rows are valid inputs.

    ### NumPy toolbox and required result

    Multiplying a numeric array by a scalar scales each element and returns an array with the same shape. `np.column_stack(tup)` accepts a tuple/list of arrays with equal row counts and joins their **columns**; joining `(n, a)` and `(n, b)` gives `(n, a+b)`. It does not create a new batch axis. Choose how to combine those operations yourself.

    Return `spatial_features`: a finite floating array `(1536, 5)` whose columns are red, green, blue, weighted normalized row, weighted normalized column, in exactly that order. Keep all rows aligned; do not mutate `color` or `position`, scale RGB with the position weight, fit a model here or replace `None` with labels. `None` is only the unfinished state. Empty/malformed arrays are not supplied; no validation or coordinate-generation code is needed.

    After Shift+Enter, the prepared cell fits color-only and color-plus-position K-Means, with current `cluster_count.value` (2–7), seed 42 and 10 initializations. Completing TODO 1 does not yet restore a 2D mask: TODO 2 handles that. The folder's `README.md`, under **TODO input and output reference**, repeats these definitions.

    Official NumPy **2.2** interfaces: [column_stack](https://numpy.org/doc/2.2/reference/generated/numpy.column_stack.html), [scalar broadcasting](https://numpy.org/doc/2.2/user/basics.broadcasting.html).""")])
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TODO 2 — restore the label at each image coordinate

    **Your job:** turn a one-dimensional label vector into a height-by-width label image, while preserving its pixel order. The cell receives `rgb` and `spatial_labels`. It should also remain safe while TODO 1 is unfinished.

    | Input | Exact contract |
    |---|---|
    | `rgb` | Finite floating NumPy array `(H, W, 3)`, here `(32, 48, 3)`. Axes mean **image row, image column, RGB channel**; intensities are in `[0, 1]`. This array supplies image dimensions; do not change its pixels. |
    | `spatial_labels` | `None` until TODO 1 supplies features; otherwise an integer NumPy array `(H*W,)`, here `(1536,)`. Entry `i` is the fitted group ID of feature row `i`, from `0` through `K-1`, where `K` is the supplied cluster-count setting. |

    ### What is a mask here?

    A **label mask** stores one integer ID per pixel. It has no RGB channel dimension. The output must satisfy $M[r,c]=\ell[rW+c]$, with `M` the mask, `ell` (`spatial_labels`) the flat labels, `r` the row and `c` the column. The width `W` determines where one row ends. Use dimensions from the provided image, rather than hard-coding 32 and 48.

    Shape-only example: a 2×3 image has six labels. The first three labels belong to its top row, the next three to its bottom row. Reshaping should preserve those six entries, rather than refit K-Means, sort IDs or turn labels into colors.

    ### Supplied API and required behavior

    `array.shape` is a tuple of axis sizes; its first two entries are height and width for this `rgb` layout. `array.reshape(shape, order='C')` returns an array with the requested dimensions, preserving C/row-major element order. The product of the new dimensions must equal the original element count. It can share storage with the input, so do not edit the returned labels in place. `None` has neither array shape nor a reshape method; handle the waiting state before performing array operations.

    Return `spatial_mask=None` while `spatial_labels is None`. Once labels exist, return an integer array `(H, W)` with exactly those labels in their original pixel positions. Preserve input arrays and do not add a third axis. The runner guarantees positive image dimensions and matching label count; no malformed-input checks are required. A mask assigning every pixel the same group is structurally valid even when it is unhelpful for your purpose.

    Check that the displayed mask aligns with the original scene. Probe coordinates `(18, 12)` and `(10, 36)` mean **row, column**, not `x, y`. Within one fitted mask, equal IDs mean shared membership. IDs can change between fits, so compare “same group?” and boundaries across weights rather than ID values. A grouping does not establish semantic class or separate object instances: disconnected pixels may share one ID, and one object may be split by lighting.

    Official NumPy **2.2** interfaces: [ndarray.shape](https://numpy.org/doc/2.2/reference/generated/numpy.ndarray.shape.html), [ndarray.reshape](https://numpy.org/doc/2.2/reference/generated/numpy.ndarray.reshape.html).
    """)
    return


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

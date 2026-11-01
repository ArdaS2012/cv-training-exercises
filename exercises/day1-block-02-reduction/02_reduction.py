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
    mo.vstack([prediction,mo.md(r"""
    ## TODO 1 — retain directions, project images and reconstruct them

    **Your job:** replace `basis`, `projected` and `reconstructed` with three NumPy array expressions. The cell receives `centered`, `components`, `mean` and `vt`. Data loading, centering and SVD are already done. Keep those inputs unchanged and do not fit another decomposition.

    ### Understand the operation before coding

    An 8×8 image has 64 pixel-intensity features. **PCA** finds perpendicular directions in that 64-dimensional feature space, ordered by how much variation they capture. A component is a direction; a projected coordinate tells how far an image lies along it. Retaining `k` directions compresses every image to `k` coordinates. Reconstructing maps those coordinates back to the 64 pixel positions, then restores the mean image.

    The prepared **SVD**, $X_c=U\,\mathrm{diag}(s)\,V^T$, supplies those directions. Here $X_c$ is the centered image matrix; columns of $U$ describe sample coordinates, entries of $s$ measure direction strength, and rows of $V^T$ are feature directions. Only `vt`, the array representing $V^T$, is needed for your TODO. The leading `k` directions must form the **columns** of your retained basis $B_k$.

    $$Z=X_c B_k,\qquad \widehat X=Z B_k^T+\mu$$

    $Z$ contains projected coordinates, $\widehat X$ the reconstructed intensities and $\mu$ the mean image added to **every row**. $B_k^T$ reverses the projection's axis change; at smaller `k` it cannot recover discarded information. These are mathematical definitions, with the NumPy expressions left to you.

    ### Inputs you can use

    All supplied numeric arrays are finite `float64` NumPy arrays. Shapes are `(rows, columns)`; a one-dimensional shape such as `(64,)` is a vector.

    | Input | Meaning, layout and valid values |
    |---|---|
    | `centered` | Shape `(300, 64)`: one image per row, one fixed pixel position per column. Pixels are flattened row by row. Each column's mean has already been subtracted. Values are normalized intensity differences and may be negative. |
    | `mean` | Shape `(64,)`: original mean intensity at each pixel, in the same pixel order. Original image intensities are in `[0, 1]`. |
    | `vt` | Shape `(64, 64)`: rows are orthonormal feature directions in decreasing singular-value order. It comes from `np.linalg.svd(centered, full_matrices=False)`, whose three returns are `u`, `singular`, `vt`. A row of `vt` is a direction, not an image. |
    | `components` | Marimo slider **object**. `.value: int` is `k`, from 1 through 64. Read the current value; do not use the widget object in arithmetic or fix `k` to one trial. |

    ### NumPy toolbox

    `np` is imported in the notebook. For a two-dimensional array, `a[row_selection, column_selection]` selects axes in that order; `:` retains an axis, and a slice's stop is excluded. Transposition `a.T` swaps rows and columns. `@` performs matrix multiplication: `(a, b) @ (b, c)` yields `(a, c)`. The inner dimensions must agree. `*` is elementwise multiplication and does not perform this projection. Adding a length-`d` vector to an `(n, d)` array broadcasts the vector across all `n` rows. These operations return arrays; do not overwrite the provided inputs.

    ### Required results and checks

    | Output variable | Required shape | Meaning |
    |---|---|---|
    | `basis` | `(64, k)` | The retained feature directions as columns, preserving their supplied order. |
    | `projected` | `(300, k)` | Each centered image's coordinates along the retained directions. |
    | `reconstructed` | `(300, 64)` | Approximate original intensities, with the mean restored. |

    For a shape-only example with 5 images, 6 features and `k=2`, the shapes are `(6, 2)`, `(5, 2)`, `(5, 6)`. Row `i` must continue to mean image `i`; none of these outputs contains class labels. All outputs must be finite numeric arrays; `None` means unfinished. The supplied data are nonempty and `k` is valid, so no new input validation is needed.

    At `k=64`, reconstruction should match the original images up to floating-point roundoff. At smaller `k`, reconstructed values may leave `[0, 1]`; **do not clip the arrays** used for measurement. The displayed grayscale range is fixed for viewing. If the image remains too dark, check the missing mean; if shapes fail, say aloud what each axis represents before changing the expression.

    ### Read the supplied comparison outputs

    **MSE** is $\frac{1}{nd}\sum_{i,j}(X_{ij}-\widehat X_{ij})^2$: average squared reconstruction error over `n=300` images and `d=64` pixels, in normalized-intensity units squared. **Retained variance** is $\sum_{j=1}^{k}s_j^2/\sum_{j=1}^{64}s_j^2$, a unitless fraction. `singular` supplies the `s` values. Lower MSE and higher retained variance still do not guarantee that your chosen visible digit detail survives.

    The supplied `display.nonlinear_views(x, neighbors=15)` takes the original finite `(n, d)` image matrix and an integer neighborhood setting, and returns `{'t-SNE': array, 'UMAP': array}`, each float array `(n, 2)` in the same sample order. It runs CPU optimizers with seed 42; no labels enter fitting. The button invokes it using `neighbors.value` (5, 15 or 30). That one setting is **t-SNE perplexity** (effective neighborhood size, positive and smaller than `n`) versus **UMAP n_neighbors** (graph-neighbor count, at least 2 and smaller than `n`). They are different definitions, not equivalent methods. Two-dimensional coordinates have arbitrary orientation/scale; digit colors are supplied labels for interpretation only. Views can run independently of the PCA TODO. Changing `k` does not make either nonlinear view a PCA reconstruction.

    Official NumPy **2.2** references: [SVD](https://numpy.org/doc/2.2/reference/generated/numpy.linalg.svd.html), [slicing/indexing](https://numpy.org/doc/2.2/user/basics.indexing.html), [transpose](https://numpy.org/doc/2.2/reference/generated/numpy.ndarray.T.html), [matrix multiplication](https://numpy.org/doc/2.2/reference/generated/numpy.matmul.html), [broadcasting](https://numpy.org/doc/2.2/user/basics.broadcasting.html). Supplied view interfaces: [scikit-learn 1.7 t-SNE](https://scikit-learn.org/1.7/modules/generated/sklearn.manifold.TSNE.html), [UMAP parameters](https://umap-learn.readthedocs.io/en/0.5dev/parameters.html).""")])
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

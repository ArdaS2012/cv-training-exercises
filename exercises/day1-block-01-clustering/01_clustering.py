import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="Compare Three Clustering Methods")


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

    **Around 30 minutes in pairs.** Organize an unlabelled image collection: can your grouping rule survive a counterexample?

    ### Prior-slide takeaways to use now

    Slides 9–14: distance depends on scale; K-Means uses centroids, linkage uses merges, DBSCAN uses density. Cluster IDs are not class names.

    1. Predict which method will cut across the curved shapes. Use curved data, K=2, single linkage, standardization on and min_samples=5; complete the estimator and assignment TODOs. Trace one assignment or merge on the plots.

    2. Keep those rows/settings fixed. Compare DBSCAN eps=0.3 with 0.6. Log group count excluding noise, noise rows and one changed or unchanged membership; point to the affected area. Swap editor/checker.

    3. Challenge your choice: keep the chosen settings and switch to noisy data, then inspect image thumbnails. Does the rule still serve your goal? Defend a choice with the comparison, a counterexample and a limit; never rename IDs as object classes.

    **Bring back:** An eps comparison log with counts and settings, one traced assignment/merge, a counterexample and a qualified grouping recommendation.

    Record each run in the evidence log before changing a control. Downloads include the current run plus your cumulative written log; save after each trial. See this block’s `GROUP_TASK.md`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Compare Three Clustering Methods

    Investigation after guided setup. Compare K-Means, hierarchical clustering and DBSCAN on compact, curved and noisy geometries, then inspect grouped image thumbnails. Explain a centroid, a dendrogram cut and noise labels. Export one observation testing your room prediction.

    Inspect → predict → implement and trace → compare → challenge → explain and download evidence.

    Use the cell code control to open TODO cells. Edit only `# TODO` areas and run with Shift+Enter. Waiting messages are normal before completion. Save Python with Ctrl+S / Command+S; widget values need the findings download. Work in pairs and swap who edits.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    dataset=mo.ui.dropdown(options=['compact','curved','noisy','images'],value='compact',label='Dataset')
    k_clusters=mo.ui.slider(2,6,value=3,label='K / hierarchical cut')
    epsilon=mo.ui.slider(0.1,1.5,step=0.1,value=0.3,label='DBSCAN eps')
    min_points=mo.ui.slider(2,12,value=5,label='DBSCAN min_samples (includes self)')
    linkage_kind=mo.ui.dropdown(options=['single','complete','average','ward'],value='single',label='Linkage')
    scaling=mo.ui.checkbox(value=True,label='Standardize features')
    mo.hstack([dataset,k_clusters,epsilon,min_points,linkage_kind,scaling],wrap=True)
    return dataset, epsilon, k_clusters, linkage_kind, min_points, scaling


@app.cell(hide_code=True)
def _(mo):
    prediction=mo.ui.text_area(label='Prediction: which method suits this geometry, and why?',full_width=True)
    mo.vstack([mo.md(r"""
    ## TODO 1 — configure three different grouping rules

    **Your job:** replace the three `None` values in `methods` with configured estimator objects. This cell has **no function arguments**: use the current widgets named below. Do not fit data here. The fitting belongs to TODO 2.

    ### What are you building?

    A **sample** is one row to group; a **feature** is one numeric property of that row. A **cluster** is a group defined by a method's similarity rule. An **estimator** is a Python object that stores that rule and its settings. Calling a class constructor creates an estimator; it does not return group assignments.

    - **K-Means** alternates assigning rows to the nearest centroid and updating each centroid to its members' mean. It minimizes within-group squared distance. `n_clusters` is the requested number of centroids.
    - **Agglomerative clustering** starts with separate rows and merges groups. `n_clusters` sets where to cut the merge tree. `linkage` chooses how to compare groups: `single` uses the closest pair, `complete` the farthest pair, `average` the mean pair distance, and `ward` the increase in within-group squared error.
    - **DBSCAN** connects dense neighborhoods. `eps` is their radius; `min_samples` is the count needed for a core point, **including that point itself**. Border points can join a dense group; isolated rows receive noise ID `-1`. There is no requested cluster count.

    For Euclidean distance, $d(a,b)=\sqrt{\sum_j(a_j-b_j)^2}$: `a` and `b` are two feature rows, and `j` runs over feature columns. Distance uses the current feature scale. A larger `eps` includes at least as many neighbors, but it need not change the final grouping.

    ### Inputs and supplied APIs

    The three classes are imported in the editable cell. Controls are marimo **widget objects**; `.value` reads their current Python setting. Pass that setting, rather than the widget object or its quoted name.

    | Available input | Meaning and allowed setting in this notebook |
    |---|---|
    | `k_clusters` | Slider; `.value: int`, 2–6. Use its current count for both K-Means and the hierarchical cut. In general the count must be positive and cannot exceed the available sample count here. |
    | `linkage_kind` | Dropdown; `.value: str`, exactly `'single'`, `'complete'`, `'average'` or `'ward'`. |
    | `epsilon` | Slider; `.value: float`, 0.1–1.5, step 0.1. `eps` must be positive; units are those of the prepared feature distances. |
    | `min_points` | Slider; `.value: int`, 2–12. General `min_samples` must be a positive integer. |
    | `KMeans` | Constructor keywords used here: `n_clusters` (count), `random_state` (integer seed), `n_init` (how many starting configurations to try). Use seed `42` and `10` initializations. |
    | `AgglomerativeClustering` | Constructor keywords used here: `n_clusters` and `linkage`. Keep its default Euclidean metric; `ward` requires it. |
    | `DBSCAN` | Constructor keywords used here: `eps` and `min_samples`. Keep its default Euclidean metric. Neither DBSCAN nor the hierarchical estimator needs a random seed here. |

    Every constructor returns one estimator instance. Constructor keyword names are case sensitive. Leave all unmentioned options at their defaults. Installation, scaling, plotting and data loading are supplied.

    ### Required result and self-check

    `methods` must be a `dict` with exactly these **string keys**, in this display order: `'K-Means'`, `'Hierarchical'`, `'DBSCAN'`. Each value must be an instance of the corresponding class. It must not be the class itself, a label array, a string naming the class, or `None`. Preserve the dictionary and its return line; replace its unfinished values. Do not change widget values inside this cell.

    Before moving on, explain to your partner which settings require a count, a distance or a merge rule. If every placeholder is replaced but results still wait, TODO 2 is still needed. The same task reference is in this exercise folder's `README.md`, under **TODO input and output reference**.

    Official scikit-learn **1.7** references: [KMeans](https://scikit-learn.org/1.7/modules/generated/sklearn.cluster.KMeans.html), [AgglomerativeClustering](https://scikit-learn.org/1.7/modules/generated/sklearn.cluster.AgglomerativeClustering.html), [DBSCAN](https://scikit-learn.org/1.7/modules/generated/sklearn.cluster.DBSCAN.html)."""),prediction])
    return (prediction,)


@app.cell
def _():
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
    # TODO 1: replace three None values with configured estimators.
    methods={'K-Means':None,'Hierarchical':None,'DBSCAN':None}
    # END TODO 1
    return (methods,)


@app.cell(hide_code=True)
def _(data, dataset, scaling):
    from sklearn.preprocessing import StandardScaler
    if dataset.value=='images':
        raw,unused_labels,unused_ids=data.images()
    else:
        raw=data.geometry(dataset.value)
    x=StandardScaler().fit_transform(raw) if scaling.value else raw.copy()
    return raw, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TODO 2 — assign every prepared row with every method

    **Your job:** create `results`, containing one group-ID array per configured method. This cell has **no function arguments**. It can use `methods` from TODO 1 and `x` from the prepared data cell immediately above. Complete TODO 1 first.

    ### Inputs: what does one row mean?

    | Available input | Exact contract |
    |---|---|
    | `methods` | `dict[str, estimator]` with exactly `'K-Means'`, `'Hierarchical'`, `'DBSCAN'`. All values are configured instances from TODO 1. |
    | `x` | A finite `float64` NumPy array with shape `(n, d)`. `n` is sample count; `d` is feature count. Every method receives **the same whole matrix, in the same row order**. There is no target-label column. |

    For `compact` and `curved`, `x.shape` is `(150, 2)` and columns are authored 2D coordinates; `noisy` adds four rows, giving `(154, 2)`. For `images`, shape is `(300, 64)`: each row is a complete 8×8 grayscale digit image, flattened left to right and top to bottom. Thus row 12 means image 12, rather than pixel 12. Digit labels are deliberately not used for clustering.

    The **Standardize features** control prepares each feature as $(x_j-\mu_j)/\sigma_j$, using its mean and population standard deviation over these rows; a constant column becomes zero. These standardized values are dimensionless and may be negative. With scaling off, coordinates retain authored units and image intensities lie in `[0, 1]`. Do not scale again in this TODO, append labels, shuffle rows or alter `x`.

    ### Supplied operation and Python toolbox

    Each estimator supplies `fit_predict(X)`: `X` is the full `(n, d)` numeric matrix. It **fits the estimator in place** and returns a new integer NumPy array `(n,)`. Entry `i` is the group assigned to row `i`. It accepts no class targets here. Calling `fit(X)` alone returns the estimator, rather than the labels this task needs.

    To work with a dictionary, `mapping.items()` supplies `(key, value)` pairs; square brackets look up a value by its exact key. Either an explicit loop or separate assignments is reasonable. The task is to connect each existing estimator's result with its existing name; do not handwrite cluster IDs.

    ### Required result and interpretation

    `results` must be a `dict[str, np.ndarray]` with the same three keys and display order as `methods`. Each value has shape `(n,)`, integer IDs and no extra feature/channel axis. For a shape-only example, a `(4, 2)` input has four labels per method, never two. Preserve the original `methods` mapping and all input row/feature values; the documented fitting updates inside the estimator are expected.

    K-Means and hierarchical IDs are nonnegative group numbers. DBSCAN additionally uses `-1` for noise. An all-noise DBSCAN result is valid: its group count is zero, while its noise count is `n`. Empty data are not supplied. The starter uses `None` only for unfinished work.

    Check the table for one assigned ID per row and trace a point on a plot. Numeric group IDs and plot colors can permute between fits: compare which rows share a group, not whether ID `0` kept its name. Log **changed or unchanged** membership when comparing `eps=0.3` and `0.6`; an unchanged partition is useful evidence. For images, inspect the supplied fixed thumbnails before making a grouping claim. The dendrogram is a separately supplied merge visualization, truncated to 12 displayed groups; it does not show all leaves or add a fourth method.

    Official scikit-learn 1.7 interfaces: [KMeans.fit_predict](https://scikit-learn.org/1.7/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans.fit_predict), [AgglomerativeClustering.fit_predict](https://scikit-learn.org/1.7/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering.fit_predict), [DBSCAN.fit_predict](https://scikit-learn.org/1.7/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN.fit_predict).
    """)
    return


@app.cell
def _():
    # TODO 2: replace None with a dictionary of fit_predict(x) results.
    results=None
    # END TODO 2
    return (results,)


@app.cell(hide_code=True)
def _(dataset, display, linkage_kind, methods, mo, np, raw, results, x):
    _view = None
    if results is None or any(m is None for m in methods.values()):
        _view = mo.callout('Waiting for TODO 1 and TODO 2. Configure methods, then assign each input row.',kind='info')
    elif set(results)!=set(methods) or any(np.asarray(a).shape!=(len(x),) for a in results.values()):
        _view = mo.callout('Return one label per input row for each of the three named methods.',kind='warn')
    else:
        _counts=[{'method':_name,'groups (excluding noise)':len(set(np.asarray(_labels).tolist())-{-1}),'noise rows':int(np.sum(np.asarray(_labels)==-1)),'total rows':len(_labels)} for _name,_labels in results.items()]
        _summary=mo.ui.table(_counts,selection=None)
        if dataset.value=='images':
            _view = mo.vstack([_summary,display.thumbnails(raw,results['K-Means']),mo.md('Rows are images. The first 12 thumbnails are fixed IDs; labels were never used to fit clusters. Record what the thumbnails support before naming any group.')])
        else:
            _view = mo.vstack([_summary,display.clustering_plot(x,results),display.dendrogram_plot(x,linkage_kind.value),mo.md('Locate a changed group/noise point, then log count and settings before changing eps. Group colors/IDs can permute; compare membership.')])
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    findings=mo.ui.text_area(value='Trial | dataset | K/linkage/scale/min_samples | eps | groups excluding noise | noise rows | located change\nBaseline | curved | 2/single/on/5 | 0.3 | | |\nComparison | curved | unchanged | 0.6 | | |\nCounterexample | noisy | chosen settings | | | |\nDecision and evidence:\nMechanism and limitation:',label="Evidence log — keep every trial, mechanism and decision",full_width=True)
    mo.vstack([findings])
    return (findings,)


@app.cell(hide_code=True)
def _(
    dataset,
    display,
    epsilon,
    findings,
    k_clusters,
    linkage_kind,
    min_points,
    mo,
    np,
    prediction,
    results,
    scaling,
):
    settings={'dataset':dataset.value,'k':k_clusters.value,'eps':epsilon.value,'min_samples':min_points.value,'linkage':linkage_kind.value,'standardized':scaling.value,'seed':42,'prediction':prediction.value}
    if results is not None:
        settings['labels']={n:np.asarray(v).tolist() for n,v in results.items()}
    display.download(mo,findings.value,settings,'clustering-findings')
    return


if __name__ == "__main__":
    app.run()

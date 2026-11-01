# Compare Three Clustering Methods

[Shared installation and help](../../README.md) · [Exercise index](../README.md)

## Start

From the student workspace: `uv run --locked python start.py --block 1`.

## Exercise + breakout

Follow the [hands-on investigation (around 30 minutes)](GROUP_TASK.md) after guided setup. It keeps the prior-slide takeaways beside the mission, code steps, controlled comparison and shared result. Use the full room for tracing, experimentation and explaining a decision. Keep the technical help below for the bounded TODOs.

## Help

If results are waiting, complete the indicated TODO; independent inputs remain usable. A shape error means checking rows versus columns before changing the supplied pipeline. Reloading resets unsaved controls; use the findings download. Port conflict: add `--port 2721`.

This practice belongs to CV Day 1 Block 1. It needs no private trainer guide, account, model download or GPU.

## Current teaching handoff

CV Day 1, Block 1: actual slide **17 — Exercise: challenge a clustering choice** launches the combined exercise + breakout. Return to plenary at the agenda’s room end. The notebook and group task are self-contained; private trainer files are not required.

## TODO input and output reference

### TODO 1 — configure three different grouping rules

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

Before moving on, explain to your partner which settings require a count, a distance or a merge rule. If every placeholder is replaced but results still wait, TODO 2 is still needed. Read the same definitions beside the editable notebook cell.

Official scikit-learn **1.7** references: [KMeans](https://scikit-learn.org/1.7/modules/generated/sklearn.cluster.KMeans.html), [AgglomerativeClustering](https://scikit-learn.org/1.7/modules/generated/sklearn.cluster.AgglomerativeClustering.html), [DBSCAN](https://scikit-learn.org/1.7/modules/generated/sklearn.cluster.DBSCAN.html).

### TODO 2 — assign every prepared row with every method

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

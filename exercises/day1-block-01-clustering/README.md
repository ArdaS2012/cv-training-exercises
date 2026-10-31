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

### TODO 1 — estimator inputs and return contract

An **estimator** is a configured scikit-learn object; constructing it does not yet fit data. The three constructors are imported in the editable cell. Controls are marimo widgets: their current Python setting is in `.value`.

| Available input | Definition |
|---|---|
| `k_clusters` | Slider; `.value` is an integer from 2 to 6. `n_clusters` is the requested number of groups for K-Means and the hierarchical cut. |
| `linkage_kind` | Dropdown; `.value` is `'single'`, `'complete'`, `'average'` or `'ward'`. `linkage` selects the hierarchical merge-distance rule. |
| `epsilon` | Slider; `.value` is a float from 0.1 to 1.5. `eps` is DBSCAN's neighborhood radius in the current feature-distance units. |
| `min_points` | Slider; `.value` is an integer from 2 to 12. `min_samples` is DBSCAN's minimum neighborhood count, including the point itself. |
| `KMeans`, `AgglomerativeClustering`, `DBSCAN` | Imported classes. Their constructor keyword arguments configure behavior; an instance exposes `fit_predict(X)`. |

Return `methods`: a dictionary with exactly the string keys `'K-Means'`, `'Hierarchical'`, `'DBSCAN'`, each holding its configured estimator, not labels or class names. For repeatability, use `random_state=42` (random initialization seed) and `n_init=10` (number of initializations) for K-Means. The other two estimators have no seed parameter here. Keep their default Euclidean distance. Configure the controls indicated above; no fitting belongs to TODO 1.

### TODO 2 — data inputs and return contract

| Available input | Definition |
|---|---|
| `methods` | The TODO 1 dictionary, once all three placeholders have been replaced. Dictionary keys name methods; values are estimator objects. |
| `x` | Finite floating NumPy array, shape `(n, d)`: row = one sample, column = one feature. `compact`/`curved`: `(150, 2)` coordinate rows; `noisy`: `(154, 2)`; `images`: `(300, 64)`, one row per flattened 8×8 digit image. There is no label column. |

When **Standardize features** is on, supplied preprocessing centers/scales each feature; `eps` then uses standardized units. With it off, geometric coordinates retain their authored units and image intensities are in `[0, 1]`. All methods must receive these same current rows in the same order. For a shape-only example, a `(4, 2)` input represents four samples with two features each.

`fit_predict(X)` fits an estimator on the full numeric matrix and returns an integer array of shape `(n,)`, with entry `i` identifying the group assigned to input row `i`. No target `y` is needed. DBSCAN uses `-1` for noise; other group IDs are arbitrary nonnegative integers. Return `results`, a dictionary with the same three keys and one such label array per method. Preserve sample order. The starter's `None` values mean unfinished work, not a valid final label array.

#### API reference

Official scikit-learn 1.7 docs: [KMeans constructor and fit_predict](https://scikit-learn.org/1.7/modules/generated/sklearn.cluster.KMeans.html), [AgglomerativeClustering](https://scikit-learn.org/1.7/modules/generated/sklearn.cluster.AgglomerativeClustering.html), [DBSCAN](https://scikit-learn.org/1.7/modules/generated/sklearn.cluster.DBSCAN.html). These document the interfaces; choosing and combining the calls is your task.

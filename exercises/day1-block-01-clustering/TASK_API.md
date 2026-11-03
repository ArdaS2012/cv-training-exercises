# Clustering: coding reference

[Quick start](README.md) · [Pair task](GROUP_TASK.md)

## TODO 1 — `assign(points, centers)`

Assign each point to the nearest center using squared Euclidean distance.

- `points`: finite floating NumPy array `(n,d)`, one sample per row, feature columns; n,d ≥ 1.
- `centers`: finite floating NumPy array `(K,d)`, one center per row in ID order, K ≥ 1. Same feature order/units as points.
- Return integer NumPy array `(n,)`, IDs `0..K-1`. Ties choose the lowest center index. Do not mutate either input. Duplicate points/centers are valid; empty arrays and nonfinite inputs are outside the task.
- Squared distance: $D_{ik}=\sum_{j=0}^{d-1}(x_{ij}-\mu_{kj})^2$. Return the minimizing k for each i. Here i indexes point rows, k center rows, j feature columns. Distances are in squared feature units; a square root would not change the nearest center.

Supplied `np` is NumPy. Useful operations: array subtraction and squaring are elementwise; compatible axes broadcast (lengths match or one is 1). `np.sum(a, axis=...)` reduces the selected axis; `np.argmin(a, axis=...)` returns indices of first minima, removing that axis. An explicit loop is also valid.

## TODO 2 — `update(points, labels, centers)`

Move each center to its assigned points' arithmetic mean.

- `points`, `centers`: same contract as TODO 1.
- `labels`: integer NumPy array `(n,)`, aligned with points; values `0..K-1`.
- Return a **new** float array `(K,d)`, row k for center k. A cluster with no points retains its previous center. Preserve all inputs; do not round.
- Formula: $\mu'_k=\frac{1}{|C_k|}\sum_{i\in C_k}x_i$, where $C_k$ is the set of rows labelled k. The result has the same units as the features. Apply the keep-old rule when $|C_k|=0$.

`a.copy()` makes independent storage; `a[mask]` selects rows for a boolean `(n,)` mask. `np.mean(a, axis=0)` averages rows, returning one value per column; an empty mean is undefined, so handle empty membership first. `len(a)` gives the row count. `np` is supplied.

### Supplied experiment

Eight illustrative 2D points use arbitrary coordinate units, without scaling:
`[[0,0],[0,2],[2,0],[2,2],[6,0],[6,2],[8,0],[8,2]]`.
Starting center rows: `[[0,0],[2,2]]`. Point IDs are row indices.

The prepared loop alternates your functions for 1–6 updates. A scikit-learn KMeans reference uses the same points, starting centers and iteration limit, `n_init=1`, `tol=0`, `algorithm='lloyd'`, seed 42. It may stop early at convergence. Final assignments are recomputed against the last centers for both comparisons; the first-step panel retains the assignments that produced its center update. X marks new centers, + marks starting centers; arrows show the first movement.

The displayed squared-distance sum is $J=\sum_i\|x_i-\mu_{c_i}\|^2$. Lower J means a closer centroid fit on these rows, not better semantic classes. Compare 1 with 6 updates and inspect the maximum absolute center-coordinate difference (matching center row order). Small floating-point differences are normal. These points avoid empty clusters; our empty-cluster rule intentionally differs from sklearn's relocation behavior.

## TODO 3 — configure estimators

The cell imports `KMeans`, `AgglomerativeClustering`, `DBSCAN`. Constructors return estimator **instances**, not assignments. `epsilon` is a marimo slider; `epsilon.value` is the radius float .1–1.5 in steps of .1.

Return `methods`, a dictionary in this exact key order:

| Key | Class and keyword settings |
|---|---|
| `'K-Means'` | `KMeans`: `n_clusters=2`, `random_state=42`, `n_init=10` |
| `'Hierarchical'` | `AgglomerativeClustering`: `n_clusters=2`, `linkage='single'` |
| `'DBSCAN'` | `DBSCAN`: `eps=epsilon.value`, `min_samples=5` (includes self) |

Leave unmentioned options at defaults; do not fit here. K-Means uses nearest centroids; single linkage merges using the closest pair of points across groups; DBSCAN connects dense neighborhoods within eps and labels isolated rows as noise. This second experiment uses sklearn's normal multi-start K-Means, unlike the controlled manual-loop comparison above.

## TODO 4 — fit every method

Inputs are `methods` from TODO 3 and `x`, a finite standardized float64 NumPy array `(n,d)` with samples in rows and features in columns. Curved/compact: `(150,2)` authored coordinates; noisy: `(154,2)` including four added outliers. Images: `(300,64)`; each row is a whole 8×8 grayscale digit, flattened left to right, top to bottom. Class labels are not used.

Prepared `StandardScaler` subtracts each feature mean and divides by its population standard deviation; a constant feature becomes zero. Values are dimensionless and may be negative. Preserve x and its row order; do not scale again.

Each instance supplies `fit_predict(X)`: fits the estimator in place on the whole numeric `(n,d)` matrix and returns integer NumPy labels `(n,)`, one per input row. No target argument is needed. `fit(X)` alone returns an estimator. `mapping.items()` yields `(key,value)` pairs.

Return `results: dict[str,np.ndarray]` with the same three keys/order as `methods`. Each value is integer `(n,)`; K-Means/hierarchy IDs are nonnegative, DBSCAN additionally uses `-1` for noise. All-noise is valid. No empty data are supplied. Preserve the mapping; fitting mutations inside estimators are expected.

Compare curved points at eps .3 and .6, then noisy points. The supplied plots/counts show groups excluding noise and noise rows. Cluster IDs/colors may permute: compare memberships, not numeric names. An unchanged partition is valid. No observation text is required; the optional trial download saves settings and computed results automatically.

## Official APIs (installed NumPy 2.2 / scikit-learn 1.7)

[Broadcasting](https://numpy.org/doc/2.2/user/basics.broadcasting.html) · [sum](https://numpy.org/doc/2.2/reference/generated/numpy.sum.html) · [argmin](https://numpy.org/doc/2.2/reference/generated/numpy.argmin.html) · [indexing](https://numpy.org/doc/2.2/user/basics.indexing.html) · [mean](https://numpy.org/doc/2.2/reference/generated/numpy.mean.html) · [copy](https://numpy.org/doc/2.2/reference/generated/numpy.ndarray.copy.html)

[KMeans / fit_predict](https://scikit-learn.org/1.7/modules/generated/sklearn.cluster.KMeans.html) · [AgglomerativeClustering](https://scikit-learn.org/1.7/modules/generated/sklearn.cluster.AgglomerativeClustering.html) · [DBSCAN](https://scikit-learn.org/1.7/modules/generated/sklearn.cluster.DBSCAN.html) · [StandardScaler](https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.StandardScaler.html)

## Reading the image-clustering view

Choose **images** in Data. The input preview shows one example of each digit. The supplied dataset contains 300 whole 8×8 grayscale digit images (30 per digit). Each becomes 64 pixel features in left-to-right, top-to-bottom order, normalized to [0,1], then standardized per pixel for fitting. Known digit labels are used only for inspecting the output, never as model inputs.

After TODOs 3–4, select the **K-Means**, **Hierarchical**, or **DBSCAN** result tab. Each displayed row is one predicted group with its total membership, percentage, up to six examples spaced across its member list, and a mean image computed from all its members at original intensity scale. These examples are not necessarily nearest the centroid. Image # is the dataset row index. Cluster IDs are arbitrary, independent across methods, and do not name digits. Mean images summarize membership; they are not new samples or necessarily model centers.

The count chart shows how many of each known digit went into each displayed group. Read across a row to inspect a group's mixture, or down a digit column to see where its 30 images went. K=2 deliberately asks for two groups across ten digit types, not ten-class recognition. Compare membership and mixtures rather than equating cluster IDs between methods. A visually similar image can be assigned differently because all 64 standardized features contribute.

Change eps while viewing **DBSCAN**; this control does not change the other two algorithms. In 64 standardized dimensions the supplied small radii may yield only noise. Noise −1 means unassigned samples, not a discovered cluster. The noise row's mean averages those unassigned samples and is not a cluster center. At most eight group rows are shown (largest first, always retaining noise); any omitted groups are explicitly counted in the title. The summary table and optional trial download retain all memberships. The images stay unchanged; the grouping is the computed result.

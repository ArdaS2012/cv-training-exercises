# Projection and neighbourhood audit — coding reference

[Back to the challenge](README.md). Each editable cell includes a compact contract. This optional reference contains the full definitions.

## Supplied data

`data.images()` supplies `(x, y, sample_ids)`: finite float64 image array `x`
`(300,64)` with intensities in `[0,1]`; integer digit labels `y` `(300,)`, values
0–9; and integer source-row IDs `sample_ids` `(300,)`. Rows stay aligned.
Each image is 8×8, flattened row by row (C order): feature `r*8+c` is pixel row
`r`, column `c`, origin top left. Labels only color plots and never enter fitting.
There are no empty or malformed inputs in this exercise.

You implement the mean image `(64,)` and centered images `(300,64)`. Signed centered values are normalized-intensity differences. NumPy is imported as `np`; the subsequent SVD is supplied.
`np.linalg.svd(centered, full_matrices=False)` returns `u` `(300,64)`,
`singular` `(64,)` and `vt` `(64,64)`. The decomposition is
$X_c=U\operatorname{diag}(s)V^T$: rows of `vt` are orthonormal feature directions
in decreasing singular-value order. Singular values are nonnegative.

`components` is a marimo slider object. `components.value` is integer `k` in
1..64, not an array or the widget itself. Changing k updates the TODO results.
TODOs 1/5 define functions; TODOs 2–4 are guarded cell assignments. Keep the supplied names/returns and implement only marked regions. None means unfinished. Prepared guards wait until preceding arrays have valid shape.

## TODO 1 — `center_images(x)`

Input `x`: finite float NumPy array `(n,d)`, positive dimensions; rows are images, columns pixels. Return `(centered,mean)`: float arrays `(n,d)` and `(d,)`, with one mean across images per pixel and that mean removed from every row. Preserve x. With one image the centered image is zero. `np.mean(a,axis=...)` averages along that axis; subtraction broadcasts a `(d,)` vector across `(n,d)`.

## TODO 2 — basis

Input `vt`: float `(64,64)`, strongest SVD directions first in rows, or None until centering is ready; `components.value`: integer k in1..64. Return `basis`: float `(64,k)`, retaining the first k directions as columns, otherwise None. Preserve inputs/order. Slices exclude their stop; `.T` transposes a 2D array. A transpose may share storage; do not modify it in place.

## TODO 3 — projection

Inputs: centered float `(300,64)` and basis float `(64,k)`; supplied `basis_ok` indicates structural readiness. Return float `projected: (300,k)`, preserving image order/inputs, otherwise None. Formula $Z=X_cB$. `@` is matrix multiplication: inner dimensions match and are summed. Coordinates are not classes or probabilities.

## TODO 4 — reconstruction

Inputs: projected float `(300,k)`, basis float `(64,k)`, mean float `(64,)`; supplied `projected_ok` indicates readiness. Return float `reconstructed: (300,64)`, same row-major image/pixel order, otherwise None. Formula $\hat X=ZB^T+\mu$. Values may lie outside [0,1] at reduced rank: preserve them for measurement; only the grayscale display fixes its range. No input mutation.

## TODO 5 — `reconstruction_error(original,restored)`

Inputs: equal finite float NumPy arrays `(n,d)`, positive dimensions. Return one nonnegative Python float, averaging squared pixel differences over **both** axes. No mutation. Identical inputs give zero; units are intensity squared. `np.mean`, elementwise subtraction and squaring are available.

## Supplied experiment outputs

MSE $=\frac{1}{nd}\sum_{i,j}(X_{ij}-\widehat X_{ij})^2$ averages squared error
over n=300 images and d=64 pixels, in normalized-intensity units squared.
SVD benchmark variance $=\sum_{j=1}^k s_j^2/\sum_{j=1}^{64}s_j^2$ is unitless. It describes the supplied leading SVD directions,
not arbitrary learner output. Lower MSE/higher benchmark variance need not preserve your particular visible detail.
Row 0 is fixed for the original/reconstruction comparison.

`display.nonlinear_views(x, neighbors=15)` accepts a finite numeric matrix
`(n,d)` and integer neighborhood setting and returns a dictionary with exactly
`'t-SNE'` and `'UMAP'`, each a finite numeric array `(n,2)` in original row order.
It runs real CPU optimizers with seed 42, without labels or input mutation.
Here x is always the same `(300,64)` original data. `neighbors.value` is one of
5, 15, 30. For t-SNE it is perplexity (positive and <n), using PCA initialization
and 500 iterations. For UMAP it is n_neighbors (integer 2..n-1), with min_dist=.1.
The settings have different definitions; identical numbers do not make them
identical algorithms. Press the Run button again after changing the setting.
The first CPU/JIT run can be slower, so warm the views before rooms open.

2D coordinates have arbitrary orientation/scale. Digit labels 0–9 color the
plots for interpretation only; the outlined point is row 0. Gaps, island size
and separation do not prove real classes or classification accuracy. These views
run independently of the PCA TODOs and cannot reconstruct the images.

## TODO 6 — `choose_rank(singular_values, target)`

`singular_values`: finite nonnegative float NumPy array `(r,)`, r ≥ 1, descending, with positive sum of squares. `target`: Python float, 0 < target < 1; widgets offer .8, .9, .95, .99. Return the **smallest** Python integer rank in 1..r for which cumulative squared singular values divided by total squared singular values reach or exceed target. Preserve inputs. Zero trailing singular values are allowed; a single positive singular value needs rank 1. Array indices start at zero, ranks at one. No empty or all-zero spectrum is supplied.

`np.cumsum(a)` returns cumulative sums with the same 1D length. `np.sum(a)` returns the total. `np.flatnonzero(condition)` returns indices of true entries; `np.searchsorted(sorted_array, value, side='left')` returns the first insertion position before equal values. These are alternative tools, not a required call sequence. The displayed curve checks sufficiency and minimality; set the independent reconstruction slider to your chosen rank. This is a variance budget, not classification accuracy or a full file-size claim: storing the shared mean and basis costs space too.

## Neighbourhood setting: what is it?

A neighbour is another image whose **64-pixel vector** is close under Euclidean distance. It is not an adjacent pixel in the 8×8 grid. Both layouts try to represent local relationships in two dimensions. The control supplies the same number to two distinct parameters:

- UMAP `n_neighbors` controls the number of nearest neighbours used to approximate the manifold with a local graph. Lower values emphasize finer local structure; higher values incorporate broader structure. It is not the number of clusters, and is unrelated to PCA rank k.
- t-SNE `perplexity` sets the effective size of a probability neighbourhood. It equals $2^{H(P_i)}$, where $H(P_i)=-\sum_j p_{j|i}\log_2 p_{j|i}$ is entropy in bits and $p_{j|i}$ is image i's normalized similarity probability for another image j. The local Gaussian bandwidth is adapted to this target. It is not a hard list of exactly that many neighbours. Do not implement this optimizer.

Compare 5 → 30 **within** t-SNE, then within UMAP, with rows, normalization and seed fixed. Equal numbers across methods do not produce equivalent models. Larger values need not improve a view. Rotations, gaps or numbers of islands alone do not establish classes or original distances.

## TODO 7 — `nearest_neighbor_ids(points, q)`

`points`: finite float NumPy array `(n,d)`, n ≥ 2, d ≥ 1. Rows are images, columns feature coordinates; here d=64 for original normalized pixels and d=2 for optimizer coordinates, whose scale/orientation is arbitrary. `q`: Python integer in 1..n−1; the audit fixes q=5. Return integer NumPy array `(n,q)` containing **row positions**, not digit labels or `sample_ids`. Each row i lists the q closest **other** rows in increasing squared Euclidean distance $D_{ij}=\sum_{f=1}^d(A_{if}-A_{jf})^2$. Break exact ties by smaller row index. Do not mutate points. Exclude only self i; another row with identical coordinates is a valid zero-distance neighbour. q=n−1 returns every other row. No empty/invalid inputs are supplied.

Loops or broadcasting are allowed. `np.sum(a, axis=...)` sums the chosen feature axis; `np.argsort(a, axis=..., kind='stable')` returns sorting indices and preserves input order on exact ties. `np.fill_diagonal(a, value)` changes a 2D array **in place** and returns None; use it only on your own distance array. `np.inf` sorts after finite distances. New singleton dimensions can be inserted with `None` in indexing; NumPy broadcasting aligns trailing axes. Distances squared have units of squared pixel intensity in 64D, and squared arbitrary layout units in 2D. Only rankings are compared across spaces.

## TODO 8 — `neighbor_retention(original_ids, view_ids)`

Inputs: integer NumPy arrays `(n,q)`, n ≥ 2, 1 ≤ q < n. Aligned row i refers to the same image in both spaces. Each row contains q distinct row indices in 0..n−1, excluding i. The order can differ. Return finite float NumPy array `(n,)`, values in [0,1], preserving inputs:
$R_i=|N_q^{original}(i)\cap N_q^{view}(i)|/q$.
Sets contain neighbour row indices; | | counts elements. `np.intersect1d(a,b)` returns the sorted unique common values of two 1D arrays. Python sets and loops are alternatives. Equal lists in a different order give 1; disjoint lists give 0; q=1 gives 0 or 1. With q=n−1 every valid pair gives 1. Return per-image values; the prepared display computes their mean.

Audit q=5 is deliberately fixed and **separate** from the optimizer scale 5/15/30: otherwise both the representation and the measuring rule change at once. The image gallery compares the query and five neighbours in original space with the same query and five neighbours in one 2D view. A matching digit color can hide changed neighbour identities. The mean can hide a poorly preserved individual image. This simple overlap is not the rank-penalized trustworthiness metric, classifier accuracy or proof of semantic classes. It audits this data/metric only.

The audit runs independently of PCA TODOs. Each optimizer remains supplied. Optional trial downloads save k, target variance, selected rank, local scale, seed, sample IDs, MSE, benchmark variance, all nonlinear coordinates, audit q, inspected row and per-image retention arrays under exact method keys `t-SNE`/`UMAP`. Save each setting separately if desired. No written prediction/observation field or participant report is required.

## Official API references

- [NumPy 2.2 mean](https://numpy.org/doc/2.2/reference/generated/numpy.mean.html)
- [NumPy 2.2 SVD](https://numpy.org/doc/2.2/reference/generated/numpy.linalg.svd.html)
- [Slicing/indexing](https://numpy.org/doc/2.2/user/basics.indexing.html)
- [Transpose](https://numpy.org/doc/2.2/reference/generated/numpy.ndarray.T.html)
- [Matrix multiplication](https://numpy.org/doc/2.2/reference/generated/numpy.matmul.html)
- [Broadcasting](https://numpy.org/doc/2.2/user/basics.broadcasting.html)
- [scikit-learn 1.7 t-SNE](https://scikit-learn.org/1.7/modules/generated/sklearn.manifold.TSNE.html)
- [UMAP parameter meanings](https://umap-learn.readthedocs.io/en/latest/parameters.html) (installed 0.5.9.post2; core n_neighbors definition also checked against its local documentation)
- [NumPy cumulative sums](https://numpy.org/doc/2.2/reference/generated/numpy.cumsum.html)
- [Sorting indices and stable ties](https://numpy.org/doc/2.2/reference/generated/numpy.argsort.html)
- [Diagonal assignment](https://numpy.org/doc/2.2/reference/generated/numpy.fill_diagonal.html)
- [Intersection](https://numpy.org/doc/2.2/reference/generated/numpy.intersect1d.html)

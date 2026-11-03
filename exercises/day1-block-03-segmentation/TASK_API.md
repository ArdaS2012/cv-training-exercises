# Magic-wand API reference

## TODO 1 — features

`color`: finite float64 `(1536,3)`, RGB columns in [0,1]. `position`: finite float64 `(1536,2)`, normalized row/column in [0,1]. The image is H=32, W=48. Both arrays have C/row-major order: index `r*W+c` is pixel `(r,c)`. Position columns are `r/(H-1)`, `c/(W-1)`, top-left origin. `spatial_weight.value` is dimensionless w in [0,2].

Return `spatial_features`: float `(1536,5)`, columns R,G,B,w*r,w*c; preserve inputs/order. `None` means unfinished. No empty/malformed inputs. K-Means fitting is supplied, K=3, seed=42, n_init=10.

NumPy `np.column_stack(tup)` joins equal-row arrays by columns: `(n,a)` + `(n,b)` gives `(n,a+b)`. Scalar multiplication scales entries without changing shape. [NumPy 2.2](https://numpy.org/doc/2.2/reference/generated/numpy.column_stack.html).

## TODO 2 — mask

`rgb`: float `(H,W,3)`, row/column/RGB axes, values [0,1]. `spatial_labels`: None or integer `(H*W,)`, group IDs 0..2 in pixel order. Return None while labels wait, otherwise integer `(H,W)` with M[r,c]=labels[r*W+c]. Preserve inputs. Positive matching dimensions are guaranteed; a single-group mask is valid. `array.shape` gives axis sizes; `array.reshape(shape)` preserves C order and may share storage. [NumPy 2.2](https://numpy.org/doc/2.2/reference/generated/numpy.ndarray.reshape.html).

## TODO 3 — selection

`select_region(mask,row,col)`: mask is None or integer `(H,W)`; row/col are valid integer pixel coordinates (down/right from top left). Return None while waiting, otherwise bool `(H,W)` selecting every pixel sharing the seed's group, including disconnected regions. Preserve mask. A one-group mask selects everything.

`a[r,c]` reads one scalar; `a == value` returns elementwise booleans. [NumPy indexing](https://numpy.org/doc/2.2/user/basics.indexing.html).

Probe A=(18,12), B=(10,36). IDs may change between fits: compare membership. The supplied scene is illustrative; groups do not establish object identities. Each trial download contains current weight/seed, seed coordinates, mask and selection.

## TODO 4 — connected patch

`connected_region(mask,row,col)`: mask is None or a NumPy integer `(H,W)` array with H,W positive; cluster IDs are nonnegative. Seed row/column are valid integers. Return None for None input; otherwise return a new boolean `(H,W)` array selecting **exactly** the seed's same-ID 4-connected component. No mutation. Horizontal/vertical neighbors connect; diagonal contact alone does not. Handle the four image edges, one-pixel components, and a one-group mask (whole image). Empty arrays/invalid seeds are not supplied.

Flood fill maintains a frontier of pixel coordinates. Start with the seed, visit its valid same-ID neighbors and continue until no unvisited neighbors remain. Avoid revisiting pixels. `np.zeros(shape,dtype=bool)` allocates a false mask. `deque([item])` creates a queue; `.popleft()` removes/returns its oldest item, `.append(item)` adds a new item; a deque is false when empty. Each item here is a `(row,col)` integer pair. `np` and `deque` are supplied.

The display compares original RGB, every matching-ID pixel and only the connected seed patch, with pixel counts. Compare weights 0,.5,1 and seeds A/B with K=3 and seed42 fixed. Connectivity separates patches without recognizing semantic objects. Optional trial download includes `connected`, a boolean `(H,W)` nested list or null. No observation text or group report is required.

[Python 3.12 deque](https://docs.python.org/3.12/library/collections.html#collections.deque) · [NumPy 2.2 zeros](https://numpy.org/doc/2.2/reference/generated/numpy.zeros.html)

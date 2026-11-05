# Segmentation coding reference

The notebook defines all four TODO contracts beside the code (TODO 4 is an optional stretch). Arrays are NumPy CPU arrays; H is height and W width. Row index is y, column is x. Learned model class 0=background, 1=warm, 2=cool; instance IDs separately identify objects.

## Tools
`np.logical_and(a,b)` and `logical_or` return elementwise Boolean arrays. `array.sum()` counts true pixels. `np.zeros(shape,dtype=int)` creates a zero integer image. Tuple indexing uses `[row,column]`. `stack.append(item)` adds an item; `stack.pop()` removes/returns the last item. `range(n)` visits 0…n−1. Any equivalent traversal satisfying the documented seed order is valid. No arrays may be mutated except your new result.

The prepared comparison calls `scipy.ndimage.label(mask)` using its default 2D four-connectivity, returning `(integer_map, count)` with background 0. It is a reference after your implementation, not an API to call in TODO 3. Diagonal pixels remain separate; foreground that touches by an edge forms one region. This rule does not separate touching objects. IoU/Dice compare the selected class, while reference/learned instance counts cover all foreground.

`day3.segmentation(index)` loads one of four authored scenes, predicts with cached small semantic/center U-Nets and returns image `(H,W,3)` float, semantic/reference_semantic integer `(H,W)`, instance/reference_instance integer `(H,W)`, centers `(H,W)`, probabilities `(3,H,W)`, cluster `(H,W)` and name string. No training or class labels enter your connected-component function.

## Stretch · instance matching
`np.unique(array)` returns the sorted distinct values, including background 0. `pred==p` gives one object's Boolean mask. TODO 4 compares objects, not classes: each reference object may claim at most one predicted object, so a merged prediction leaves one reference unmatched (missed) and a split object leaves an extra prediction. The summary table applies your function with threshold .5 to the learned centre-based instances and reference instances of all four scenes, beside the foreground IoU from your TODO 2. This is the matching rule used by instance metrics such as [COCO mask AP](https://cocodataset.org/#detection-eval).

[NumPy logic](https://numpy.org/doc/2.2/reference/routines.logic.html) · [NumPy zeros](https://numpy.org/doc/2.2/reference/generated/numpy.zeros.html) · [SciPy 1.16 label](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.ndimage.label.html) · [Python lists](https://docs.python.org/3.12/tutorial/datastructures.html).

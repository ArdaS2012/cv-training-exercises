# Siamese coding reference

All four TODO contracts/formulas are adjacent to their code. P is pair count, D feature count; current pretrained encoder returns P=24,D=8 unit-normalized embeddings. y=1 means similar (some publications invert this convention). Margin is a loss parameter; cutoff is a decision parameter. No inputs are mutated and malformed/empty arrays are outside this task.

`np.sum(array,axis=...)` combines the selected axis; `np.sqrt` and powers work elementwise; `np.linalg.norm(a,axis=...)` is an alternative Euclidean reduction. `np.maximum(a,b)` returns elementwise maxima. NumPy comparisons create Boolean arrays; `&`/`~` combine/invert those arrays; Python `int(value)` converts a scalar count. Python `and` does not combine whole Boolean arrays.

`day3.pairs()` returns `(data,left,right)`: data contains a/b float32 image rows `(24,64)` in [0,1], labels_a/labels_b integer digit labels `(24,)`, and similar binary numeric targets `(24,)`. left/right are actual trained encoder outputs `(24,8)` in matching row order. `day3.pair_report(distance,contrastive,threshold)` supplies thumbnails and an independent threshold summary using your distance/loss functions. Its report includes pairs with pair index, left/right digit, similar_target, distance, threshold, match, correct and loss; false_matches, missed_matches, mean_loss, margin, model. It returns `(report,figure)` or None while unfinished.

The prepared sweep evaluates your error function on fixed cutoffs 0…2. It is development inspection, not fresh validation or retraining. The optional download records current computed values automatically.

[NumPy 2.2 norm](https://numpy.org/doc/2.2/reference/generated/numpy.linalg.norm.html) · [maximum](https://numpy.org/doc/2.2/reference/generated/numpy.maximum.html) · [sum](https://numpy.org/doc/2.2/reference/generated/numpy.sum.html) · [Boolean indexing](https://numpy.org/doc/2.2/user/basics.indexing.html).

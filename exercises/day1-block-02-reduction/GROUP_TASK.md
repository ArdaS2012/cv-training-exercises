# Build and audit a projection

30 minutes in pairs; swap who edits.

From slides 19–22: center before projection; reconstruct to see lost information; 2D gaps do not establish classes.

1. Implement centering, basis selection, projection, reconstruction and pixel MSE. Compare k=1,8,32,64 using the same digit and error heatmap.
2. Implement a minimum-rank selection rule. Compare variance targets 80% → 95%; inspect your chosen rank's reconstruction. Variance is not accuracy.
3. Implement nearest-neighbour ranking with self excluded, then per-image neighbour retention. Keep audit q=5 fixed; compare supplied t-SNE/UMAP local scales 5 → 30. Inspect the original and 2D neighbours of the same image, then another image.

“Local scale” supplies t-SNE perplexity (effective neighbourhood size) and UMAP n_neighbors (graph neighbour count). These differ; neither is a cluster count. Neighbours are similar image vectors, not adjacent pixels. Larger settings need not improve retention.

Keep working code and plots for the trainer’s solution walkthrough. No written prediction, observation log or group presentation required. Trial download is optional.

[Input/output and API reference](TASK_API.md).

# Build and audit a projection

`uv run --locked python start.py --block 2`

Implement the five PCA stages, then choose the smallest rank meeting a variance target. Compare k=1,8,32,64 and targets 80% → 95% using the same digit, error map and budget curve.

Implement nearest-neighbour ranking and per-image neighbour retention. Run supplied t-SNE/UMAP views at local scales 5 → 30, holding the audit at five neighbours. Inspect actual neighbouring digit images. The local-scale control means t-SNE perplexity (effective probability-neighbourhood size) and UMAP n_neighbors (graph neighbour count); it is not a cluster count. Similar digit colors can hide lost neighbours.

[Pair task](GROUP_TASK.md) · [Coding reference](TASK_API.md). No written prediction or observation required; the trainer walks through the completed solution.

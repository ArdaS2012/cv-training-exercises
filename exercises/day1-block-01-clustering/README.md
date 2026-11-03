# Build a K-Means step

`uv run --locked python start.py --block 1`

Implement nearest-center assignment and centroid updates. Compare 1–6 updates with scikit-learn using the same starting centers. Then fit three methods on curved/noisy points and vary DBSCAN eps .3 → .6.

No written prediction or observation required. Save your code; trial download is optional. The trainer walks through the solution afterward.

[Coding reference](TASK_API.md) · [Pair task](GROUP_TASK.md) · CV Day 1 Block 1, slides 17–18.

For image data, inspect the input digit strip, then compare method tabs with per-cluster examples, sizes, mean images and a digit-count chart. Cluster IDs are arbitrary; −1 is unassigned noise. [How to read the result](TASK_API.md#reading-the-image-clustering-view).

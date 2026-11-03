# Build a magic wand

`uv run --locked python start.py --block 3`

Build color/position features, restore the label image, then implement same-ID selection and a 4-connected flood fill. Compare all matching pixels with the connected seed patch at weights0,.5,1. Inspect both seeds; connectivity is not object recognition.

[Pair task](GROUP_TASK.md) · [Coding reference](TASK_API.md). No written prediction or observation required; the trainer walks through the completed solution.

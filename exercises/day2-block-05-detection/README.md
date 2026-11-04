# Build detector postprocessing

`uv run --locked python start.py --day 2 --block 5`

Implement confidence filtering, box IoU and greedy class-aware NMS. Apply to actual/snapshot YOLO and SSD candidates; vary one threshold and inspect a duplicate or missed reference. Stretch: sweep the confidence cutoff and plot precision against recall.

30 minutes in pairs. Keep code and computed plots for the trainer solution walkthrough; no written predictions/observations or group reports. [Coding/API reference](TASK_API.md) · [Pair task](GROUP_TASK.md).

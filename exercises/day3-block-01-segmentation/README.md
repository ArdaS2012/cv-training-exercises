# Compare Semantic and Instance Masks


## Exercise + breakout

Follow the [30-minute group task](GROUP_TASK.md) after guided setup.

CV Day 3 Block 1 · guided setup, then a 30-minute exercise + breakout.

Run supplied pretrained tiny U-Nets on held-out authored scenes; compare semantic classes, learned instance identities and the retained Day1 cluster mask.

[Shared cross-platform setup](../../README.md). After online preflight, launch:

```sh
uv run --locked python start.py --day 3 --block 1 --check
uv run --locked python start.py --day 3 --block 1
```

Complete only marked `# TODO` / `# END TODO` regions in [01_segmentation.py](01_segmentation.py). The supplied instructions specify inputs, output shapes, conventions and conceptual hints. Unfinished functions return None and show waiting feedback. Model training/from-scratch infrastructure is prepared; do not rewrite it during the slot.

1. Predict the change before coding.
2. Complete the bounded concepts; run with Shift+Enter.
3. Change one controlled input, record the fixed inputs/model and inspect a failure.
4. Save the notebook and download findings/assets. Share a measured observation, causal explanation and limitation.

Semantic class IDs describe categories; instance IDs describe separate objects within an image. Choose the same class or matched instance when calculating overlap. If setup fails, re-run preflight before class; ask for the complete student ZIP. Day3 Block4 requires official model downloads once; cached offline launch uses `--offline-assets`.


## Current teaching handoff

CV Day 3, Block 1: actual slide **15 — Exercise: overlap versus object identity** launches the combined exercise + breakout. Return to plenary at the agenda’s room end. The notebook and group task are self-contained; private trainer files are not required.

## Investigation focus

Mask detective: good overlap, wrong object count?. Semantic masks label classes; instance masks identify objects. U-Net skips preserve spatial detail. IoU/Dice compare aligned regions; class overlap alone cannot certify object count. Keep the task beside the output, preserve run A in the comparison log, and compare a controlled change before making your decision.

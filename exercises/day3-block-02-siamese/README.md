# Compare Image Pairs


## Exercise + breakout

Follow the [30-minute group task](GROUP_TASK.md) after guided setup.

CV Day 3 Block 2 · guided setup, then a 30-minute exercise + breakout.

Use a supplied trained Siamese encoder, held-out pairs and margin convention y=1 for similar; no training from scratch.

[Shared cross-platform setup](../../README.md). After online preflight, launch:

```sh
uv run --locked python start.py --day 3 --block 2 --check
uv run --locked python start.py --day 3 --block 2
```

Complete only marked `# TODO` / `# END TODO` regions in [02_siamese.py](02_siamese.py). The supplied instructions specify inputs, output shapes, conventions and conceptual hints. Unfinished functions return None and show waiting feedback. Model training/from-scratch infrastructure is prepared; do not rewrite it during the slot.

1. Predict the change before coding.
2. Complete the bounded concepts; run with Shift+Enter.
3. Change one controlled input, record the fixed inputs/model and inspect a failure.
4. Save the notebook and download findings/assets. Share a measured observation, causal explanation and limitation.

The checkpoint is a shared encoder on held-out pairs. Keep the pairs and embeddings fixed while changing a decision threshold. Similar-label conventions and unit-normalized vector ranges are stated in the notebook. If setup fails, re-run preflight before class; ask for the complete student ZIP. Day3 Block4 requires official model downloads once; cached offline launch uses `--offline-assets`.


## Current teaching handoff

CV Day 3, Block 2: actual slide **24 — Exercise: defend a match threshold** launches the combined exercise + breakout. Return to plenary at the agenda’s room end. The notebook and group task are self-contained; private trainer files are not required.

## Investigation focus

Match desk: which mistake costs more?. Both branches share encoder weights. Distance compares embeddings; a decision threshold does not retrain them. Here y=1 means similar. A wider acceptance threshold trades missed matches for false matches. Keep the task beside the output, preserve run A in the comparison log, and compare a controlled change before making your decision.

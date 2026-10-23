# Test Two Vision Prompt Routes


## Exercise + breakout

Follow the [30-minute group task](GROUP_TASK.md) after guided setup.

CV Day 3 Block 4 · guided setup, then a 30-minute exercise + breakout.

Run actual CLIP text inference and SAM prompt decoding with official pinned checkpoints and cached matching image encodings.

[Shared cross-platform setup](../../README.md). After online preflight, launch:

```sh
uv run --locked python start.py --day 3 --block 4 --check
uv run --locked python start.py --day 3 --block 4
```

Complete only marked `# TODO` / `# END TODO` regions in [04_prompts.py](04_prompts.py). The supplied instructions specify inputs, output shapes, conventions and conceptual hints. Unfinished functions return None and show waiting feedback. Model training/from-scratch infrastructure is prepared; do not rewrite it during the slot.

1. Predict the change before coding.
2. Complete the bounded concepts; run with Shift+Enter.
3. Change one controlled input, record the fixed inputs/model and inspect a failure.
4. Save the notebook and download findings/assets. Share a measured observation, causal explanation and limitation.

CLIP scores normalize over the chosen text candidates. SAM receives original-image geometry and does not assign a text class. Keep the image and model revisions fixed; compare one prompt change at a time. If setup fails, re-run preflight before class; ask for the complete student ZIP. Day3 Block4 requires official model downloads once; cached offline launch uses `--offline-assets`.


## Current teaching handoff

CV Day 3, Block 4: actual slide **48 — Exercise: isolate a vision prompt** launches the combined exercise + breakout. Return to plenary at the agenda’s room end. The notebook and group task are self-contained; private trainer files are not required.

## Investigation focus

Prompt repair: words and points do different jobs. CLIP compares an image with text candidates; normalized scores depend on that set. SAM accepts spatial prompts and returns masks. Prompt edits change inference inputs, not weights. Keep the task beside the output, preserve run A in the comparison log, and compare a controlled change before making your decision.

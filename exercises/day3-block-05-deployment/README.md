# Adapt Save Reload and Serve


## Exercise + breakout

Follow the [15-minute group task](GROUP_TASK.md) after guided setup.

CV Day 3 Block 5 · guided setup, then a 15-minute exercise + breakout.

Fit a small target head under an explicit brightness/contrast shift, export validation/package metadata and witness a real local request/response.

[Shared cross-platform setup](../../README.md). After online preflight, launch:

```sh
uv run --locked python start.py --day 3 --block 5 --check
uv run --locked python start.py --day 3 --block 5
```

Complete only marked `# TODO` / `# END TODO` regions in [05_deployment.py](05_deployment.py). The supplied instructions specify inputs, output shapes, conventions and conceptual hints. Unfinished functions return None and show waiting feedback. Model training/from-scratch infrastructure is prepared; do not rewrite it during the slot.

1. Predict the change before coding.
2. Complete the bounded concepts; run with Shift+Enter.
3. Change one controlled input, record the fixed inputs/model and inspect a failure.
4. Save the notebook and download findings/assets. Share a measured observation, causal explanation and limitation.

Use the declared target label order and unchanged preprocessing. Compare validation before/after, frozen parameter deltas, reloaded score vectors and the actual local HTTP response. If setup fails, re-run preflight before class; ask for the complete student ZIP. Day3 Block4 requires official model downloads once; cached offline launch uses `--offline-assets`.


## Current teaching handoff

CV Day 3, Block 5: actual slide **61 — Exercise: verify a served pipeline** launches the combined exercise + breakout. Return to plenary at the agenda’s room end. The notebook and group task are self-contained; private trainer files are not required.

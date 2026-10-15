# CV Day 2 Block 3 — Adapt a Pretrained Classifier

Fit a new head on frozen pretrained features, then compare a real selective last-block fine-tune and held-out predictions.

## Start

Use the [shared cross-platform setup](../../README.md). From the student repository:

```sh
uv run --locked python start.py --day 2 --block 3 --check
uv run --locked python start.py --day 2 --block 3
```

Open the supplied `03_transfer.py` editor on localhost:2720. Run online preflight before class; the cached route uses `uv run --offline --locked python start.py --day 2 --block 3`. Block 5 downloads actual detector weights into ignored cache during online preflight. No private trainer repository is needed.

## Exercise + breakout

Follow the [15-minute group task](GROUP_TASK.md) after guided setup. It defines the required comparison, evidence and shared result. Keep the technical help below for the bounded TODOs.

## Submit/share

Your completed notebook, findings JSON, plot and a brief explanation of the comparison. Do not claim a universal architecture ranking from small examples.

## Help

Check exact tensor shape, coordinate convention and normalization in the notebook. Read the feedback; it points to the concept without giving a copy-ready solution. Restart from the shared launcher if the editor stops. If a required model cannot load, run online preflight first; Block 5 also has an explicitly labelled measured-output fallback.

## Course mapping

CV Day 2, Block 3, guided setup plus a 15-minute exercise + breakout. The private trainer guide is not needed for the task. Model/data provenance is bundled under `cv_exercises/assets/day2/`.

## Current teaching handoff

CV Day 2, Block 3: actual slide **42 — Exercise: verify what fine-tunes** launches the combined exercise + breakout. Return to plenary at the agenda’s room end. The notebook and group task are self-contained; private trainer files are not required.

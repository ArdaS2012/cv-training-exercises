# CV Day 2 Block 4 — Inspect Compression and Denoising

Compare genuine autoencoder checkpoints at two bottleneck widths, fixed-noise reconstructions, per-pixel errors and lost detail.

## Start

Use the [shared cross-platform setup](../../README.md). From the student repository:

```sh
uv run --locked python start.py --day 2 --block 4 --check
uv run --locked python start.py --day 2 --block 4
```

Open the supplied `04_autoencoders.py` editor on localhost:2720. Run online preflight before class; the cached route uses `uv run --offline --locked python start.py --day 2 --block 4`. Block 5 downloads actual detector weights into ignored cache during online preflight. No private trainer repository is needed.

## Exercise + breakout

Follow the [30-minute group task](GROUP_TASK.md) after guided setup. It defines the required comparison, evidence and shared result. Keep the technical help below for the bounded TODOs.

## Submit/share

Your completed notebook, findings JSON, plot and a brief explanation of the comparison. Do not claim a universal architecture ranking from small examples.

## Help

Check exact tensor shape, coordinate convention and normalization in the notebook. Read the feedback; it points to the concept without giving a copy-ready solution. Restart from the shared launcher if the editor stops. If a required model cannot load, run online preflight first; Block 5 also has an explicitly labelled measured-output fallback.

## Course mapping

CV Day 2, Block 4, guided setup plus a 30-minute exercise + breakout. The private trainer guide is not needed for the task. Model/data provenance is bundled under `cv_exercises/assets/day2/`.

## Current teaching handoff

CV Day 2, Block 4: actual slide **50 — Exercise: assess reconstruction quality** launches the combined exercise + breakout. Return to plenary at the agenda’s room end. The notebook and group task are self-contained; private trainer files are not required.

## Investigation focus

Restoration lab: cleaner pixels or lost detail?. Encoder → latent code → decoder reconstructs pixels. MSE averages squared errors in matching units. Denoising uses a clean target. Low average error can hide a lost stroke; checkpoint comparisons are descriptive. Keep the task beside the output, preserve run A in the comparison log, and compare a controlled change before making your decision.

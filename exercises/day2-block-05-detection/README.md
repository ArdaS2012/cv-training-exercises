# CV Day 2 Block 5 — Compare Detector Outputs

Run two real pretrained CPU detectors on shared images; inspect confidence filtering, IoU, reference matches and a missed or duplicate object.

## Start

Use the [shared cross-platform setup](../../README.md). From the student repository:

```sh
uv run --locked python start.py --day 2 --block 5 --check
uv run --locked python start.py --day 2 --block 5
```

Open the supplied `05_detection.py` editor on localhost:2720. Run online preflight before class; the cached route uses `uv run --offline --locked python start.py --day 2 --block 5`. Block 5 downloads actual detector weights into ignored cache during online preflight. No private trainer repository is needed.

## Exercise + breakout

Follow the [30-minute group task](GROUP_TASK.md) after guided setup. It defines the required comparison, evidence and shared result. Keep the technical help below for the bounded TODOs.

## Submit/share

Your completed notebook, findings JSON, plot and a brief explanation of the comparison. Do not claim a universal architecture ranking from small examples.

## Help

Check exact tensor shape, coordinate convention and normalization in the notebook. Read the feedback; it points to the concept without giving a copy-ready solution. Restart from the shared launcher if the editor stops. If a required model cannot load, run online preflight first; Block 5 also has an explicitly labelled measured-output fallback.

## Course mapping

CV Day 2, Block 5, guided setup plus a 30-minute exercise + breakout. The private trainer guide is not needed for the task. Model/data provenance is bundled under `cv_exercises/assets/day2/`.

## Current teaching handoff

CV Day 2, Block 5: actual slide **64 — Exercise: audit detector filtering** launches the combined exercise + breakout. Return to plenary at the agenda’s room end. The notebook and group task are self-contained; private trainer files are not required.

## Investigation focus

Detector audit: find what the threshold cannot fix. Detectors return boxes, labels and scores. IoU measures geometry, not class correctness. Confidence filters predictions; NMS suppresses overlap. Filtering cannot create a missing box. Keep the task beside the output, preserve run A in the comparison log, and compare a controlled change before making your decision.

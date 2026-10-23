# Exercise: test learning evidence

**Exercise + breakout · CV Day 1 Block 5 · 30 minutes in pairs**

## Your mission

Audit a learning claim: did gradient updates improve this classifier, and what do the images actually show?

## Keep these prior-slide takeaways beside the task

Slides 57–63: ReLU supplies nonlinearity; cross-entropy scores the target probability. Updates use training rows; validation informs choices and the test stays untouched.

Why this helps: explain the mechanism before trusting a colorful output. Your partner must be able to reproduce the comparison from your settings and evidence.

## Before the room opens

Use [shared setup](../../README.md); launch checks and any warming happen before class. Start from the repository root with `uv run --locked python start.py --block 5`. Locate the supplied input, controls, TODOs and results; the conceptual implementation and investigation below belong to the room.

## Investigate together

1. Predict the learning curves and manually check one target-probability loss. Implement ReLU and mean cross-entropy; trace one negative activation and one target probability. Keep the supplied matching derivatives and fixed split unchanged.

2. At rate 0.5 compare 0 and 80 updates from the same initialization. Log final training loss, validation accuracy and the same held-out image ID/prediction; inspect its actual pixels. Swap roles at minute 15 and explain one changed or unchanged prediction.

3. Repeat 80 updates changing only rate to 2.0. Compare the curves and held-out image; accept deterioration as evidence. Recommend a next training experiment and name a population/condition your current validation cannot establish.

## Use the 30 minutes purposefully

- **0–4:** Inspect the input, choose the goal and write a prediction before running.
- **4–12:** Complete the small concept TODOs; trace one input through your code and fix shape/value mistakes together.
- **12–22:** Run the controlled comparison, record each result before changing controls, and swap editor/checker at minute 15.
- **22–27:** Test the counterexample/changed condition; explain the mechanism and decide.
- **27–30:** Both partners rehearse a 45-second evidence-based explanation; save code and download findings.

## Bring back

Three controlled run rows and curves, one inspected held-out image, an explained learning claim and a next-test limitation.

Use the notebook's evidence log; a final screenshot alone loses the comparison. Downloads contain current controls/measurements plus your cumulative written log, so save after each trial. Saving Python does not retain widget inputs. An unchanged or worse result is useful when you can explain it. If blocked, request a room visit through the trainer chat; inspect the supplied input/baseline together while help arrives.

## Explain it to another pair

“We predicted __ because __. We changed only __; the evidence was __. We choose __ for __, but cannot conclude __. Our next test would hold __ fixed and change __.”

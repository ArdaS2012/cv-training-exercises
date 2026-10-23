# Exercise: defend a projection

**Exercise + breakout · CV Day 1 Block 2 · 30 minutes in pairs**

## Your mission

Choose an image compression budget: how few coordinates preserve the detail your user needs?

## Keep these prior-slide takeaways beside the task

Slides 23–28: center first; PCA uses SVD directions and discards components. Error and retained variance answer different questions; 2D gaps do not prove classes.

Why this helps: explain the mechanism before trusting a colorful output. Your partner must be able to reproduce the comparison from your settings and evidence.

## Before the room opens

Use [shared setup](../../README.md); launch checks and any warming happen before class. Start from the repository root with `uv run --locked python start.py --block 2`. Locate the supplied input, controls, TODOs and results; the conceptual implementation and investigation below belong to the room.

## Investigate together

1. Choose one visible detail in input row 0 that must survive compression; predict the smallest acceptable rank. Complete the three PCA expressions, checking 300×64 → 300×k → 300×64 shapes. Explain why the mean must be restored.

2. Use the identical rows at k=1, 8 and 32. Log retained variance, MSE and whether that detail survives; save each run. Swap roles at minute 15 and choose the smallest tested rank meeting your declared goal, even when its plot looks less attractive.

3. Run the supplied nonlinear views at neighbor setting 5 and 30 while keeping rows/seed fixed. Track a recognizable labelled region and describe what changed. Challenge a class-separation claim with evidence; explain why perplexity and UMAP neighbor count are different controls.

## Use the 30 minutes purposefully

- **0–4:** Inspect the input, choose the goal and write a prediction before running.
- **4–12:** Complete the small concept TODOs; trace one input through your code and fix shape/value mistakes together.
- **12–22:** Run the controlled comparison, record each result before changing controls, and swap editor/checker at minute 15.
- **22–27:** Test the counterexample/changed condition; explain the mechanism and decide.
- **27–30:** Both partners rehearse a 45-second evidence-based explanation; save code and download findings.

## Bring back

Three measured rank rows, a detail-preservation decision and a fixed-row neighborhood comparison with an interpretation limit.

Use the notebook's evidence log; a final screenshot alone loses the comparison. Downloads contain current controls/measurements plus your cumulative written log, so save after each trial. Saving Python does not retain widget inputs. An unchanged or worse result is useful when you can explain it. If blocked, request a room visit through the trainer chat; inspect the supplied input/baseline together while help arrives.

## Explain it to another pair

“We predicted __ because __. We changed only __; the evidence was __. We choose __ for __, but cannot conclude __. Our next test would hold __ fixed and change __.”

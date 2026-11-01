# Exercise: challenge a clustering choice

**Exercise + breakout · CV Day 1 Block 1 · around 30 minutes in pairs**

## Your mission

Organize an unlabelled image collection: can your grouping rule survive a counterexample?

## Keep these prior-slide takeaways beside the task

Slides 9–14: distance depends on scale; K-Means uses centroids, linkage uses merges, DBSCAN uses density. Cluster IDs are not class names.

Why this helps: explain the mechanism before trusting a colorful output. Your partner must be able to reproduce the comparison from your settings and evidence.

## Before the room opens

Use [shared setup](../../README.md); launch checks and any warming happen before class. Start from the repository root with `uv run --locked python start.py --block 1`. Locate the supplied input, controls, TODOs and results; the conceptual implementation and investigation below belong to the room.

## Definitions before coding

The notebook displays the exact types, shapes, field meanings and return contracts beside every TODO. The same [input/output definitions and API references](README.md#todo-input-and-output-reference) are available here locally. Read those definitions before editing; you do not need to inspect private files or guess the data structure.

## Investigate together

1. Predict which method will cut across the curved shapes. Use curved data, K=2, single linkage, standardization on and min_samples=5; complete the estimator and assignment TODOs. Trace one assignment or merge on the plots.

2. Keep those rows/settings fixed. Compare DBSCAN eps=0.3 with 0.6. Log group count excluding noise, noise rows and one changed or unchanged membership; point to the affected area. Swap editor/checker.

3. Challenge your choice: keep the chosen settings and switch to noisy data, then inspect image thumbnails. Does the rule still serve your goal? Defend a choice with the comparison, a counterexample and a limit; never rename IDs as object classes.

## Bring back

An eps comparison log with counts and settings, one traced assignment/merge, a counterexample and a qualified grouping recommendation.

Use the notebook's evidence log; a final screenshot alone loses the comparison. Downloads contain current controls/measurements plus your cumulative written log, so save after each trial. Saving Python does not retain widget inputs. An unchanged or worse result is useful when you can explain it. If blocked, request a room visit through the trainer chat; inspect the supplied input/baseline together while help arrives.

## Explain it to another pair

“We predicted __ because __. We changed only __; the evidence was __. We choose __ for __, but cannot conclude __. Our next test would hold __ fixed and change __.”

# Exercise: test a pixel partition

**Exercise + breakout · CV Day 1 Block 3 · around 30 minutes in pairs**

## Your mission

Build a region proposal for an image tool: can it separate distant red regions without inventing object labels?

## Keep these prior-slide takeaways beside the task

Slides 36–38: preserve pixel↔row order to restore H×W. Color and position change similarity; a cluster is not an object label or instance.

Why this helps: explain the mechanism before trusting a colorful output. Your partner must be able to reproduce the comparison from your settings and evidence.

## Before the room opens

Use [shared setup](../../README.md); launch checks and any warming happen before class. Start from the repository root with `uv run --locked python start.py --block 3`. Locate the supplied input, controls, TODOs and results; the conceptual implementation and investigation below belong to the room.

## Definitions before coding

The notebook displays the exact types, shapes, field meanings and return contracts beside every TODO. The same [input/output definitions and API references](README.md#todo-input-and-output-reference) are available here locally. Read those definitions before editing; you do not need to inspect private files or guess the data structure.

## Investigate together

1. Locate the two red regions and predict whether they will share a group. Complete the feature and H×W reshape TODOs. Use K=3 and seed 42; explain what one feature row represents before running.

2. Hold image, K and seed fixed; compare position weights 0, 0.5 and 1.0. Log whether probes (row18,col12) and (row10,col36) share a group, and locate a boundary that moves or remains fixed. Swap roles. Compare membership, not the numeric value of a cluster ID.

3. Choose a weight for a declared region-selection purpose. Use the lighting gradient or a split background as a counterexample to your own choice. State what annotation would be needed to call the regions objects; retain the masks for Day 3.

## Bring back

Three weight trials with fixed settings, a probe/boundary comparison, chosen mask and a concrete semantic/instance limitation.

Use the notebook's evidence log; a final screenshot alone loses the comparison. Downloads contain current controls/measurements plus your cumulative written log, so save after each trial. Saving Python does not retain widget inputs. An unchanged or worse result is useful when you can explain it. If blocked, request a room visit through the trainer chat; inspect the supplied input/baseline together while help arrives.

## Explain it to another pair

“We predicted __ because __. We changed only __; the evidence was __. We choose __ for __, but cannot conclude __. Our next test would hold __ fixed and change __.”

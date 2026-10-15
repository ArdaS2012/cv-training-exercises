# Exercise: test a pixel partition

**Exercise + breakout · CV Day 1 Block 3 · 15 minutes in pairs**

Learning goal: Explain how feature design changes pixel partitions and why a partition does not establish object identity.

## Before the room opens

Open the authored RGB scene and show how flattened pixel rows map back to H×W coordinates. Confirm the feature/reshape interfaces and fixed seed; leave both small TODOs for rooms.

Use the [shared setup](../../README.md); model downloads and launch checks happen before the task. Start from the repository root with `uv run --locked python start.py --block 3`. Keep the starter open. Prerequisite TODOs discussed in plenary belong to setup; the task below is the room assignment.

## Your group task

1. Predict whether the two distant red regions will share a group. Complete feature/reshape TODOs and compare the color-only mask with the spatial mask at weight 0.5, same cluster count and seed.

2. Change only position weight to 1.0. Compare both spatial masks, identifying one boundary or disconnected region that changes and one region that does not.

3. Decide whether either mask justifies an object-count or semantic-label claim. Use a lighting or same-color counterexample and retain the baseline for Day 3.

## Bring back

Color-only and spatial masks, fixed-settings comparison and a supported usefulness/failure explanation.

Use the notebook’s prediction/observation fields. Save the Python source and download findings to retain browser inputs and measurements. Both partners should be able to explain the conclusion; swap editor/checker around halfway. If you finish early, state a competing explanation and a new controlled test that would distinguish it.

## Work within the timebox

2 minutes to predict and agree; 7 to implement and compare; 4 to diagnose and defend; 2 to record/export. Ask for help through the trainer’s chat/room-visit route if blocked. Return to plenary for review and the scheduled quiz. Honest unchanged or failed results count as evidence when they are explained; do not invent measurements.

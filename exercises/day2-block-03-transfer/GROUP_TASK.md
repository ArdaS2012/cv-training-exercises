# Exercise: verify what fine-tunes

**Exercise + breakout · CV Day 2 Block 3 · 15 minutes in pairs**

Learning goal: Verify adaptation behavior from changed weights and held-out outcomes rather than a label such as fine-tuning.

## Before the room opens

Preload the checkpoint/cached features and show target label order and group names. Locate the before/after parameter-delta report. Leave head width and selective-training booleans for rooms.

Use the [shared setup](../../README.md); model downloads and launch checks happen before the task. Start from the repository root with `uv run --locked python start.py --day 2 --block 3`. Keep the starter open. Prerequisite TODOs discussed in plenary belong to setup; the task below is the room assignment.

## Your group task

1. Predict which groups can change. Complete the target-head and selective-training TODOs, keeping the first convolution frozen; run the prepared head fit and brief last-block fine-tune.

2. Compare parameter counts/deltas and validation predictions across both routes. Identify evidence that a frozen group remained fixed and a selected group was eligible to learn.

3. Recommend a route for a small target dataset with a possible domain shift. State one limitation and what fresh validation evidence would change your recommendation.

## Bring back

Trainable-group/count/delta table, validation comparison and a conditional adaptation choice.

Use the notebook’s prediction/observation fields. Save the Python source and download findings to retain browser inputs and measurements. Both partners should be able to explain the conclusion; swap editor/checker around halfway. If you finish early, state a competing explanation and a new controlled test that would distinguish it.

## Work within the timebox

2 minutes to predict and agree; 7 to implement and compare; 4 to diagnose and defend; 2 to record/export. Ask for help through the trainer’s chat/room-visit route if blocked. Return to plenary for review and the scheduled quiz. Honest unchanged or failed results count as evidence when they are explained; do not invent measurements.

# Exercise: test learning evidence

**Exercise + breakout · CV Day 1 Block 5 · 15 minutes in pairs**

Learning goal: Separate evidence of fitting from evidence of generalization using a controlled training experiment.

## Before the room opens

Open the prepared two-layer classifier, fixed split and CPU loop. Identify activation/loss TODOs and the zero-update control. Explain that the supplied derivatives match ReLU and mean cross-entropy; leave those functions for rooms.

Use the [shared setup](../../README.md); model downloads and launch checks happen before the task. Start from the repository root with `uv run --locked python start.py --block 5`. Keep the starter open. Prerequisite TODOs discussed in plenary belong to setup; the task below is the room assignment.

## Your group task

1. Predict training loss and validation behavior. Complete the two TODOs and manually check one negative-log target loss.

2. At learning rate 0.5 compare 0 versus 80 updates with the same initialization/data. Record loss and a held-out prediction from both runs.

3. Repeat 80 updates changing only rate to 2.0. Accept the actual result even if it worsens. Defend which evidence shows learning and which further evidence would be needed for generalization.

## Bring back

Loss curve, zero-update/baseline/rate comparison, held-out prediction and a qualified learning conclusion.

Use the notebook’s prediction/observation fields. Save the Python source and download findings to retain browser inputs and measurements. Both partners should be able to explain the conclusion; swap editor/checker around halfway. If you finish early, state a competing explanation and a new controlled test that would distinguish it.

## Work within the timebox

2 minutes to predict and agree; 7 to implement and compare; 4 to diagnose and defend; 2 to record/export. Ask for help through the trainer’s chat/room-visit route if blocked. Return to plenary for review and the scheduled quiz. Honest unchanged or failed results count as evidence when they are explained; do not invent measurements.

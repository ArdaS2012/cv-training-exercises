# Exercise: test a training intervention

**Exercise + breakout · CV Day 2 Block 2 · 15 minutes in pairs**

Learning goal: Make a controlled improvement decision without leaking test evidence into selection.

## Before the room opens

Show the training/validation curves as a presenter-led worked example. Open the fixed checkpoint/splits and baseline rate 0.01, penalty 0.0; explain the declared validation metric. Leave configuration and selection TODOs for rooms.

Use the [shared setup](../../README.md); model downloads and launch checks happen before the task. Start from the repository root with `uv run --locked python start.py --day 2 --block 2`. Keep the starter open. Prerequisite TODOs discussed in plenary belong to setup; the task below is the room assignment.

## Your group task

1. Predict the effect of rate 0.03 or penalty 0.001; choose one factor only. Complete the configuration TODO and run the two supplied 24-epoch CPU trainings.

2. Complete validation selection. Compare curves and the declared validation losses, reporting a result that supports or contradicts your hypothesis and an alternative explanation.

3. Freeze the selected configuration, then use the reserved-test action once. Explain why using this test score for another tuning decision would invalidate its final-evaluation role.

## Bring back

Configuration/curve table, validation-based choice, one reserved-test result and an uncertainty statement.

Use the notebook’s prediction/observation fields. Save the Python source and download findings to retain browser inputs and measurements. Both partners should be able to explain the conclusion; swap editor/checker around halfway. If you finish early, state a competing explanation and a new controlled test that would distinguish it.

## Work within the timebox

2 minutes to predict and agree; 7 to implement and compare; 4 to diagnose and defend; 2 to record/export. Ask for help through the trainer’s chat/room-visit route if blocked. Return to plenary for review and the scheduled quiz. Honest unchanged or failed results count as evidence when they are explained; do not invent measurements.

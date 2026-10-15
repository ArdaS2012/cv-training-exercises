# Exercise: verify a served pipeline

**Exercise + breakout · CV Day 3 Block 5 · 15 minutes in pairs**

Learning goal: Verify an adapted model’s inference contract through reload and serving rather than merely saving weights.

## Before the room opens

Preload the small backbone/domain dataset and test the local serving wrapper. Show target labels, transform and revision metadata. Leave the adaptation-configuration TODO for rooms; saving/reloading/HTTP plumbing is supplied.

Use the [shared setup](../../README.md); model downloads and launch checks happen before the task. Start from the repository root with `uv run --locked python start.py --day 3 --block 5`. Keep the starter open. Prerequisite TODOs discussed in plenary belong to setup; the task below is the room assignment.

## Your group task

1. Predict which parameter subset changes. Complete the adaptation config with the declared target order and rate, then run adapt/save/reload/local request at rate 0.03.

2. Check before/after validation, frozen-group deltas, original versus reloaded score vectors and the actual HTTP response. Identify the evidence for identical preprocessing and label interpretation.

3. Specify a concrete failure if label order or preprocessing were omitted. Describe how the observed local evidence would detect it and what extra checks a hosted service would require.

## Bring back

Validation comparison, frozen-weight check, reproducible package and recorded local request/response with a deployment failure analysis.

Use the notebook’s prediction/observation fields. Save the Python source and download findings to retain browser inputs and measurements. Both partners should be able to explain the conclusion; swap editor/checker around halfway. If you finish early, state a competing explanation and a new controlled test that would distinguish it.

## Work within the timebox

2 minutes to predict and agree; 7 to implement and compare; 4 to diagnose and defend; 2 to record/export. Ask for help through the trainer’s chat/room-visit route if blocked. Return to plenary for review and the scheduled quiz. Honest unchanged or failed results count as evidence when they are explained; do not invent measurements.

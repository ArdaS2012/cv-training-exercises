# Exercise: overlap versus object identity

**Exercise + breakout · CV Day 3 Block 1 · 15 minutes in pairs**

Learning goal: Assess segmentation by both pixel overlap and the task’s class/identity requirements.

## Before the room opens

Preload learned-mask assets and locate semantic classes, instance IDs and reference masks. Explain same-class/matched-instance overlap and empty-mask convention. Leave class selection and IoU/Dice TODOs for rooms.

Use the [shared setup](../../README.md); model downloads and launch checks happen before the task. Start from the repository root with `uv run --locked python start.py --day 3 --block 1`. Keep the starter open. Prerequisite TODOs discussed in plenary belong to setup; the task below is the room assignment.

## Your group task

1. Predict whether high foreground overlap can coexist with a wrong object count. Complete both TODOs and check one small mask pair manually.

2. Compare separate versus touching shapes with model and warm-class selection fixed. Record semantic/instance overlays, overlap and observed identity/count errors.

3. Inspect the exact Day 1 scene and retained cluster mask. Explain which method supplies class meaning and which supplies object identities; use a concrete boundary or touching-object failure as evidence.

## Bring back

Overlap calculation, semantic/instance comparisons and a supported distinction between mask quality and instance correctness.

Use the notebook’s prediction/observation fields. Save the Python source and download findings to retain browser inputs and measurements. Both partners should be able to explain the conclusion; swap editor/checker around halfway. If you finish early, state a competing explanation and a new controlled test that would distinguish it.

## Work within the timebox

2 minutes to predict and agree; 7 to implement and compare; 4 to diagnose and defend; 2 to record/export. Ask for help through the trainer’s chat/room-visit route if blocked. Return to plenary for review and the scheduled quiz. Honest unchanged or failed results count as evidence when they are explained; do not invent measurements.

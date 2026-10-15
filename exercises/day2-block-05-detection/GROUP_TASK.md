# Exercise: audit detector filtering

**Exercise + breakout · CV Day 2 Block 5 · 15 minutes in pairs**

Learning goal: Distinguish model predictions, confidence selection and geometric overlap in an evidence-based detector comparison.

## Before the room opens

Preload both detector checkpoints, run one warm-up image and show reference xyxy boxes. Explain continuous-coordinate areas and separate confidence/NMS thresholds. Leave confidence-selection and IoU TODOs for rooms.

Use the [shared setup](../../README.md); model downloads and launch checks happen before the task. Start from the repository root with `uv run --locked python start.py --day 2 --block 5`. Keep the starter open. Prerequisite TODOs discussed in plenary belong to setup; the task below is the room assignment.

## Your group task

1. Predict a match/miss. Complete both TODOs, manually check IoU for one pair of supplied boxes and verify equality at the confidence cutoff.

2. Run YOLO and SSD on the bus image at confidence 0.5. Change only confidence to 0.7 with NMS fixed; compare retained boxes and one reference match.

3. Identify one miss, duplicate or incorrect box and propose a follow-up test. Explain why filtering cannot recover an object absent from raw predictions and why this photograph cannot rank detector families universally.

## Bring back

Annotated outputs, checked IoU/cutoff case, threshold comparison and one diagnosed detection limit.

Use the notebook’s prediction/observation fields. Save the Python source and download findings to retain browser inputs and measurements. Both partners should be able to explain the conclusion; swap editor/checker around halfway. If you finish early, state a competing explanation and a new controlled test that would distinguish it.

## Work within the timebox

2 minutes to predict and agree; 7 to implement and compare; 4 to diagnose and defend; 2 to record/export. Ask for help through the trainer’s chat/room-visit route if blocked. Return to plenary for review and the scheduled quiz. Honest unchanged or failed results count as evidence when they are explained; do not invent measurements.

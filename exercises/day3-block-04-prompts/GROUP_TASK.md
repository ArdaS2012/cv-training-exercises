# Exercise: isolate a vision prompt

**Exercise + breakout · CV Day 3 Block 4 · 15 minutes in pairs**

Learning goal: Match text versus geometric prompt design to the actual model’s input/output contract.

## Before the room opens

Preload official pinned checkpoints and matching cached image encodings. Warm both routes on the bus photo. Show pixel coordinates, the baseline point-on-bus failure and text-candidate scoring. Leave candidate/spatial-prompt TODOs for rooms.

Use the [shared setup](../../README.md); model downloads and launch checks happen before the task. Start from the repository root with `uv run --locked python start.py --day 3 --block 4`. Keep the starter open. Prerequisite TODOs discussed in plenary belong to setup; the task below is the room assignment.

## Your group task

1. Predict which route can isolate the person near x280,y650. Complete candidate and spatial-prompt TODOs for the bus image, using parallel text descriptions and valid original-image geometry.

2. Run both actual models and compare your prompt with the supplied baseline. Change one text description or one spatial prompt item only, with image and revisions fixed; preserve scores and masks.

3. Explain one failed or ambiguous result and propose the next single-variable test. Distinguish relative CLIP candidate scores from semantic correctness and SAM’s spatial mask from a text label.

## Bring back

Baseline/changed text scores and spatial masks, fixed-settings record and an interface-based failure explanation.

Use the notebook’s prediction/observation fields. Save the Python source and download findings to retain browser inputs and measurements. Both partners should be able to explain the conclusion; swap editor/checker around halfway. If you finish early, state a competing explanation and a new controlled test that would distinguish it.

## Work within the timebox

2 minutes to predict and agree; 7 to implement and compare; 4 to diagnose and defend; 2 to record/export. Ask for help through the trainer’s chat/room-visit route if blocked. Return to plenary for review and the scheduled quiz. Honest unchanged or failed results count as evidence when they are explained; do not invent measurements.

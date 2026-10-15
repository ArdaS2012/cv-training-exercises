# Exercise: assess reconstruction quality

**Exercise + breakout · CV Day 2 Block 4 · 15 minutes in pairs**

Learning goal: Assess reconstruction quantitatively and visually while distinguishing capacity from objective.

## Before the room opens

Preload the four genuine checkpoints and choose image row 0 with noise 0.25. Explain the clean reference, noisy input and objectives; leave reconstruction/error TODOs for rooms.

Use the [shared setup](../../README.md); model downloads and launch checks happen before the task. Start from the repository root with `uv run --locked python start.py --day 2 --block 4`. Keep the starter open. Prerequisite TODOs discussed in plenary belong to setup; the task below is the room assignment.

## Your group task

1. Predict detail retention. Complete the two TODOs and check the averaging convention for per-pixel mean squared error.

2. At identical image/noise/seed, compare latent widths 4 and 16. For one width compare clean-trained and denoising-trained outputs; record errors and point to one preserved or lost detail.

3. Defend a checkpoint choice for denoising, including one case where low pixel error would still be inadequate. Explain which comparisons change width versus training objective.

## Bring back

Fixed-input reconstruction/error comparison, marked lost detail and a qualified checkpoint decision.

Use the notebook’s prediction/observation fields. Save the Python source and download findings to retain browser inputs and measurements. Both partners should be able to explain the conclusion; swap editor/checker around halfway. If you finish early, state a competing explanation and a new controlled test that would distinguish it.

## Work within the timebox

2 minutes to predict and agree; 7 to implement and compare; 4 to diagnose and defend; 2 to record/export. Ask for help through the trainer’s chat/room-visit route if blocked. Return to plenary for review and the scheduled quiz. Honest unchanged or failed results count as evidence when they are explained; do not invent measurements.

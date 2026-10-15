# Exercise: defend a match threshold

**Exercise + breakout · CV Day 3 Block 2 · 15 minutes in pairs**

Learning goal: Connect embedding distance, pair-loss convention and decision-cost tradeoffs.

## Before the room opens

Preload the trained shared encoder and held-out pairs. Explain labels y=1 for similar, margin and distance range. Leave distance/contrastive-loss TODOs for rooms.

Use the [shared setup](../../README.md); model downloads and launch checks happen before the task. Start from the repository root with `uv run --locked python start.py --day 3 --block 2`. Keep the starter open. Prerequisite TODOs discussed in plenary belong to setup; the task below is the room assignment.

## Your group task

1. Predict which error type will grow with a higher threshold. Complete both TODOs and check one similar and one dissimilar pair loss with the stated label convention.

2. Keep pairs/encoder/embeddings fixed. Compare thresholds 0.5 and 0.9, counting false matches and missed matches; identify a changed decision or explain why none crossed the cutoff.

3. Choose a threshold for a scenario where false matches are more costly. Justify it with observed errors and describe how you would validate it on new pairs without tuning on the test set.

## Bring back

Checked pair losses, two-threshold error table, one explained pair and a conditional threshold decision.

Use the notebook’s prediction/observation fields. Save the Python source and download findings to retain browser inputs and measurements. Both partners should be able to explain the conclusion; swap editor/checker around halfway. If you finish early, state a competing explanation and a new controlled test that would distinguish it.

## Work within the timebox

2 minutes to predict and agree; 7 to implement and compare; 4 to diagnose and defend; 2 to record/export. Ask for help through the trainer’s chat/room-visit route if blocked. Return to plenary for review and the scheduled quiz. Honest unchanged or failed results count as evidence when they are explained; do not invent measurements.

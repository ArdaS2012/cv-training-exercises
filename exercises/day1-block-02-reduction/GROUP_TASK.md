# Exercise: defend a projection

**Exercise + breakout · CV Day 1 Block 2 · 15 minutes in pairs**

Learning goal: Choose compression from quantitative evidence and interpret nonlinear views within their limits.

## Before the room opens

Prewarm the supplied t-SNE/UMAP helpers in plenary and show both plots for the same rows. Explain centering and U/S/V shapes. Leave the three small PCA expressions for rooms; no optimizer setup is part of the room task.

Use the [shared setup](../../README.md); model downloads and launch checks happen before the task. Start from the repository root with `uv run --locked python start.py --block 2`. Keep the starter open. Prerequisite TODOs discussed in plenary belong to setup; the task below is the room assignment.

## Your group task

1. Predict how retained variance and reconstruction error change. Complete the basis/projection/reconstruction TODOs using the supplied SVD; check matrix shapes.

2. Compare k=1,8,32 on identical image rows. Choose the smallest of these k values you would accept for a declared detail-preservation goal and justify it with measured variance/error and one image.

3. Inspect the prewarmed t-SNE and UMAP views. Challenge the statement “widely separated islands prove different classes,” and name a claim each plot cannot support.

## Bring back

Three-k measurement table, one reconstruction comparison, justified k choice and two visualization limitations.

Use the notebook’s prediction/observation fields. Save the Python source and download findings to retain browser inputs and measurements. Both partners should be able to explain the conclusion; swap editor/checker around halfway. If you finish early, state a competing explanation and a new controlled test that would distinguish it.

## Work within the timebox

2 minutes to predict and agree; 7 to implement and compare; 4 to diagnose and defend; 2 to record/export. Ask for help through the trainer’s chat/room-visit route if blocked. Return to plenary for review and the scheduled quiz. Honest unchanged or failed results count as evidence when they are explained; do not invent measurements.

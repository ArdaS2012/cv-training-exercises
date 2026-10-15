# Exercise: trace a CNN change

**Exercise + breakout · CV Day 2 Block 1 · 15 minutes in pairs**

Learning goal: Trace a numerical convolution and distinguish architecture geometry from semantic interpretation.

## Before the room opens

Preload tiny-CNN and VGG assets and open fixed image row 0. Explain convolution versus ReLU/pooling and locate measured shapes/hooks. Leave filter-response and output-size expressions for rooms.

Use the [shared setup](../../README.md); model downloads and launch checks happen before the task. Start from the repository root with `uv run --locked python start.py --day 2 --block 1`. Keep the starter open. Prerequisite TODOs discussed in plenary belong to setup; the task below is the room assignment.

## Your group task

1. Predict a filter response and stride-1 output shape. Complete both TODOs and check one patch response by hand, including the bias.

2. Keep image, kernel and padding fixed; compare stride 1 and 2. Verify the predicted shape against the computed map and explain what spatial sampling changed.

3. Compare one tiny-CNN stage with the supplied pretrained VGG trace. Identify channel/spatial changes and explain why an activation is not proof of a named object.

## Bring back

Checked response, predicted/measured shape table, fixed-stride comparison and a feature-interpretation caveat.

Use the notebook’s prediction/observation fields. Save the Python source and download findings to retain browser inputs and measurements. Both partners should be able to explain the conclusion; swap editor/checker around halfway. If you finish early, state a competing explanation and a new controlled test that would distinguish it.

## Work within the timebox

2 minutes to predict and agree; 7 to implement and compare; 4 to diagnose and defend; 2 to record/export. Ask for help through the trainer’s chat/room-visit route if blocked. Return to plenary for review and the scheduled quiz. Honest unchanged or failed results count as evidence when they are explained; do not invent measurements.

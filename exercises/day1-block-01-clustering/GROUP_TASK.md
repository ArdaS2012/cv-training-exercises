# Exercise: challenge a clustering choice

**Exercise + breakout · CV Day 1 Block 1 · 15 minutes in pairs**

Learning goal: Use geometry, parameters and counterexamples to assess clustering rather than recognize an algorithm name.

## Before the room opens

Open the supplied datasets, scaling option and estimator interfaces. Locate centroids, dendrogram and noise labels in the prepared display. Use a presenter-led compact-versus-curved example to establish the question; leave the estimator/assignment TODOs for rooms.

Use the [shared setup](../../README.md); model downloads and launch checks happen before the task. Start from the repository root with `uv run --locked python start.py --block 1`. Keep the starter open. Prerequisite TODOs discussed in plenary belong to setup; the task below is the room assignment.

## Your group task

1. Choose curved data with standardization on. Predict one failure, complete the short estimator/assignment TODOs and inspect K-Means, hierarchical and DBSCAN outputs on the same points.

2. Keep all settings fixed except DBSCAN eps; compare 0.3 and 0.6. Record cluster membership/noise changes and use the dendrogram to explain one hierarchical merge.

3. Recommend a method for this geometry, giving numerical/visual evidence and a failure risk. Check the supplied image-thumbnail view and explain why cluster IDs cannot be renamed as object classes without evidence.

## Bring back

Three-method comparison, eps/noise comparison, one merge/centroid observation and an evidence-based recommendation.

Use the notebook’s prediction/observation fields. Save the Python source and download findings to retain browser inputs and measurements. Both partners should be able to explain the conclusion; swap editor/checker around halfway. If you finish early, state a competing explanation and a new controlled test that would distinguish it.

## Work within the timebox

2 minutes to predict and agree; 7 to implement and compare; 4 to diagnose and defend; 2 to record/export. Ask for help through the trainer’s chat/room-visit route if blocked. Return to plenary for review and the scheduled quiz. Honest unchanged or failed results count as evidence when they are explained; do not invent measurements.

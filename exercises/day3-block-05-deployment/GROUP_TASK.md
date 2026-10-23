# Deployment workshop: does the saved model still mean the same thing?

**Exercise + breakout · CV Day 3 Block 5 · 30 minutes in pairs**

Your job is to produce a defensible decision from the actual outputs, including a result that challenges your first guess. One partner edits; the other checks controls, arithmetic and visual evidence. Swap at minute 15.

## Keep these slide takeaways beside your experiment

Domain shift needs target validation. Head-only adaptation freezes the backbone. A package includes transforms, labels and revision. Hub storage differs from a running endpoint; reload and an actual request test the contract.

## Before the room opens

Preload the small backbone/domain dataset and test the local serving wrapper. Show target labels, transform and revision metadata. Leave the adaptation-configuration TODO for rooms; saving/reloading/HTTP plumbing is supplied.

Use the [shared setup](../../README.md); model downloads and launch checks happen before the task. Start from the repository root with `uv run --locked python start.py --day 3 --block 5`. Keep the starter open. Prerequisite TODOs discussed in plenary belong to setup; the task below is the room assignment.

## Your investigation

1. Predict which groups change during head-only adaptation and what must survive save/reload. Sketch pixels → transform → model → label before configuring the real pipeline.

2. Complete the bounded adaptation/label configuration. At rate 0.03 run adapt/save/reload/local HTTP; inspect validation before/after and frozen versus head deltas.

3. Inspect the package files and actual response. Match score indices to saved labels, compare original/reloaded vectors within tolerance, and trace the same raw 8×8 pixels through the declared normalization.

4. Swap roles. Change ONLY learning rate to 0.10, rerun with fixed data/seed/budget, and compare validation and changed-head evidence. Keep the first package/report before rerunning; a worse result is useful evidence.

5. Perform a label-interpretation fault drill using the actual response: on paper, swap the two displayed label names while leaving the vector fixed. Show which interpretation becomes wrong although numeric reload agreement still passes. Choose a package using validation, and explain the extra checks a hosted service would require.

## Evidence to bring back

Keep a short comparison log with columns **run/rate | validation before/after | frozen/head deltas | reload agreement | HTTP/label evidence**. Save run A before changing settings: the notebook’s current report is replaced when controls rerun. Download each comparison figure/report plus the experiment log. Save the Python source separately; browser inputs are not saved in it.

Bring one prediction, measured comparison, marked visual example or traced response, a decision and one limitation. Both partners must explain why the evidence supports it.

## Work within the timebox

0–4 min: read the takeaways, inspect inputs and predict. 4–12: implement the bounded TODOs and inspect a real result. 12–22: compare controlled runs; swap editor/checker at minute 15. 22–27: diagnose a stress case and choose. 27–30: agree, save and export evidence.

If blocked, ask through the trainer’s chat/room-visit route. Use the working partner’s device if needed. Honest unchanged or worse outputs count when explained; label a supplied measured fallback instead of claiming fresh inference.

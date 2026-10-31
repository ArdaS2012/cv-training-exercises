# Restoration lab: cleaner pixels or lost detail?

**Exercise + breakout · CV Day 2 Block 4 · around 30 minutes in pairs**

Your job is to produce a defensible decision from the actual outputs, including a result that challenges your first guess. One partner edits; the other checks controls, arithmetic and visual evidence. Swap roles during the investigation.

## Keep these slide takeaways beside your experiment

Encoder → latent code → decoder reconstructs pixels. MSE averages squared errors in matching units. Denoising uses a clean target. Low average error can hide a lost stroke; checkpoint comparisons are descriptive.

## Before the room opens

Preload the four genuine checkpoints and choose image row 0 with noise 0.25. Explain the clean reference, noisy input and objectives; leave reconstruction/error TODOs for rooms.

Use the [shared setup](../../README.md); model downloads and launch checks happen before the task. Start from the repository root with `uv run --locked python start.py --day 2 --block 4`. Keep the starter open. Prerequisite TODOs discussed in plenary belong to setup; the task below is the room assignment.

## Your investigation

1. At row 0 and noise 0.25, choose a visible stroke/edge you want to preserve. Predict what changing latent width 4→16 will do to that detail and clean-target error.

2. Complete encode/decode and pixel-MSE TODOs. Check the averaging convention with a tiny hand example; inspect clean input, noisy image, reconstruction and error map together.

3. Compare latent widths 4 and 16 with the SAME row/noise. Record MSE and annotate the chosen detail in each output; compare both clean-reconstruction and denoising checkpoints shown by the notebook.

4. Swap roles, freeze width, and change only noise (0.25→0.45). Inspect whether smoothing removes noise and useful detail together. Use a second digit only after saving this fixed-image comparison.

5. Choose a checkpoint for preserving the selected detail under noise. Defend with both measured error and visible evidence, including any disagreement. Explain why different training histories prevent a pure causal claim about latent width.

## Evidence to bring back

Keep a short comparison log with columns **row | noise | width/objective | clean-target MSE | preserved/lost detail**. Save run A before changing settings: the notebook’s current report is replaced when controls rerun. Download each comparison figure/report plus the experiment log. Save the Python source separately; browser inputs are not saved in it.

Bring one prediction, measured comparison, marked visual example or traced response, a decision and one limitation. Both partners must explain why the evidence supports it.

If blocked, ask through the trainer’s chat/room-visit route. Use the working partner’s device if needed. Honest unchanged or worse outputs count when explained; label a supplied measured fallback instead of claiming fresh inference.


## Inputs and API help

Read the [function input/output contracts and official API references](TASK_API.md), also displayed immediately beside the notebook TODOs. They define every argument, data field, shape and convention; no private trainer material or inferred field names are needed.

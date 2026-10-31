# Adaptation audit: prove which weights learned

**Exercise + breakout · CV Day 2 Block 3 · around 30 minutes in pairs**

Your job is to produce a defensible decision from the actual outputs, including a result that challenges your first guess. One partner edits; the other checks controls, arithmetic and visual evidence. Swap roles during the investigation.

## Keep these slide takeaways beside your experiment

The head width follows target labels. Frozen means no parameter update, not identical activations. Fine-tuning selects a trainable subset; more changed weights do not guarantee better validation.

## Before the room opens

Preload the checkpoint/cached features and show target label order and group names. Locate the before/after parameter-delta report. Leave head width and selective-training booleans for rooms.

Use the [shared setup](../../README.md); model downloads and launch checks happen before the task. Start from the repository root with `uv run --locked python start.py --day 2 --block 3`. Keep the starter open. Prerequisite TODOs discussed in plenary belong to setup; the task below is the room assignment.

## Your investigation

1. Sketch the target output shape and predict which parameter groups may change in head-only and last-block adaptation. Use the target label order shown in the notebook.

2. Complete the head and trainable-group TODOs, leaving the first convolution frozen. Run the supplied real head update and bounded last-block update. Inspect names, counts and actual parameter deltas.

3. Swap roles. Trace one frozen and one eligible group from configuration to measured delta, then compare held-out outcomes on the same split and record the extra fine-tuning budget. Explain the difference between permission to update and a measured nonzero change.

4. Inspect a target example whose prediction changed, or document that no displayed decision changed. Explain how scores/activations can change while the first convolution stays fixed. Distinguish that evidence from a claim that every eligible coefficient changed.

5. Recommend the simpler head-only baseline or selective route for this small target task. Cite a measured delta and validation result, then name the fresh domain-validation evidence that could reverse your decision.

## Evidence to bring back

Keep a short comparison log with columns **route | trainable groups/count | frozen delta | eligible delta | validation evidence**. Save run A before changing settings: the notebook’s current report is replaced when controls rerun. Download each comparison figure/report plus the experiment log. Save the Python source separately; browser inputs are not saved in it.

Bring one prediction, measured comparison, marked visual example or traced response, a decision and one limitation. Both partners must explain why the evidence supports it.

If blocked, ask through the trainer’s chat/room-visit route. Use the working partner’s device if needed. Honest unchanged or worse outputs count when explained; label a supplied measured fallback instead of claiming fresh inference.


## Inputs and API help

Read the [function input/output contracts and official API references](TASK_API.md), also displayed immediately beside the notebook TODOs. They define every argument, data field, shape and convention; no private trainer material or inferred field names are needed.

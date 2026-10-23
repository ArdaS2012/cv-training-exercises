# Prompt repair: words and points do different jobs

**Exercise + breakout · CV Day 3 Block 4 · 30 minutes in pairs**

Your job is to produce a defensible decision from the actual outputs, including a result that challenges your first guess. One partner edits; the other checks controls, arithmetic and visual evidence. Swap at minute 15.

## Keep these slide takeaways beside your experiment

CLIP compares an image with text candidates; normalized scores depend on that set. SAM accepts spatial prompts and returns masks. Prompt edits change inference inputs, not weights.

## Before the room opens

Preload official pinned checkpoints and matching cached image encodings. Warm both routes on the bus photo. Show pixel coordinates, the baseline point-on-bus failure and text-candidate scoring. Leave candidate/spatial-prompt TODOs for rooms.

Use the [shared setup](../../README.md); model downloads and launch checks happen before the task. Start from the repository root with `uv run --locked python start.py --day 3 --block 4`. Keep the starter open. Prerequisite TODOs discussed in plenary belong to setup; the task below is the room assignment.

## Your investigation

1. Use the bus image throughout the core comparison. Predict why the supplied point-on-bus baseline misses the intended person near x280,y650, and which model can return that person’s pixels.

2. Complete comparable text candidates and a valid original-pixel spatial prompt. Run both real models, inspect baseline scores and mask, and save the baseline before editing.

3. Freeze the spatial prompt. Change ONE text description in otherwise identical parallel candidates, rerun and compare scores/ranking. Explain what changed in the denominator and what that cannot prove about semantic correctness.

4. Swap roles. Restore the text candidates, change only ONE spatial prompt item (point, exclusion or box) and compare masks on the fixed image/revisions. Mark included/excluded pixels around the intended person and diagnose a failure.

5. Recommend a prompt for the stated output, citing score and mask evidence separately. Propose one next controlled test; explain why a text label and a spatial mask are different outputs and neither comparison trains weights.

## Evidence to bring back

Keep a short comparison log with columns **run | text item changed | spatial item changed | score/rank evidence | mask inclusion/exclusion**. Save run A before changing settings: the notebook’s current report is replaced when controls rerun. Download each comparison figure/report plus the experiment log. Save the Python source separately; browser inputs are not saved in it.

Bring one prediction, measured comparison, marked visual example or traced response, a decision and one limitation. Both partners must explain why the evidence supports it.

## Work within the timebox

0–4 min: read the takeaways, inspect inputs and predict. 4–12: implement the bounded TODOs and inspect a real result. 12–22: compare controlled runs; swap editor/checker at minute 15. 22–27: diagnose a stress case and choose. 27–30: agree, save and export evidence.

If blocked, ask through the trainer’s chat/room-visit route. Use the working partner’s device if needed. Honest unchanged or worse outputs count when explained; label a supplied measured fallback instead of claiming fresh inference.

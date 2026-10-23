# Detector audit: find what the threshold cannot fix

**Exercise + breakout · CV Day 2 Block 5 · 30 minutes in pairs**

Your job is to produce a defensible decision from the actual outputs, including a result that challenges your first guess. One partner edits; the other checks controls, arithmetic and visual evidence. Swap at minute 15.

## Keep these slide takeaways beside your experiment

Detectors return boxes, labels and scores. IoU measures geometry, not class correctness. Confidence filters predictions; NMS suppresses overlap. Filtering cannot create a missing box.

## Before the room opens

Preload both detector checkpoints, run one warm-up image and show reference xyxy boxes. Explain continuous-coordinate areas and separate confidence/NMS thresholds. Leave confidence-selection and IoU TODOs for rooms.

Use the [shared setup](../../README.md); model downloads and launch checks happen before the task. Start from the repository root with `uv run --locked python start.py --day 2 --block 5`. Keep the starter open. Prerequisite TODOs discussed in plenary belong to setup; the task below is the room assignment.

## Your investigation

1. On the bus photograph, mark one object you expect to survive confidence 0.5→0.7. Predict whether its geometry will change or only its retention.

2. Complete confidence selection and continuous-coordinate IoU TODOs. Hand-check one supplied box pair, a disjoint pair and equality at the score cutoff. Then run the actual CPU detectors.

3. Hold image, model outputs and NMS fixed; compare confidence 0.5 versus 0.7 for YOLO and SSD. Record boxes kept, one reference IoU and a concrete miss/duplicate; label live inference or measured fallback honestly.

4. Swap roles. Restore confidence 0.5, change ONLY NMS overlap cutoff, and inspect one overlapping pair. Explain the difference from your confidence experiment. If counts stay fixed, use the displayed scores/overlaps to explain why.

5. Recommend settings for retaining the chosen object without a duplicate. Identify a remaining bad box or absent object that filtering cannot repair. Use the second photograph only as a bounded generalization check, not a universal model ranking.

## Evidence to bring back

Keep a short comparison log with columns **image | detector/mode | confidence | NMS | kept boxes | reference IoU / visible error**. Save run A before changing settings: the notebook’s current report is replaced when controls rerun. Download each comparison figure/report plus the experiment log. Save the Python source separately; browser inputs are not saved in it.

Bring one prediction, measured comparison, marked visual example or traced response, a decision and one limitation. Both partners must explain why the evidence supports it.

## Work within the timebox

0–4 min: read the takeaways, inspect inputs and predict. 4–12: implement the bounded TODOs and inspect a real result. 12–22: compare controlled runs; swap editor/checker at minute 15. 22–27: diagnose a stress case and choose. 27–30: agree, save and export evidence.

If blocked, ask through the trainer’s chat/room-visit route. Use the working partner’s device if needed. Honest unchanged or worse outputs count when explained; label a supplied measured fallback instead of claiming fresh inference.

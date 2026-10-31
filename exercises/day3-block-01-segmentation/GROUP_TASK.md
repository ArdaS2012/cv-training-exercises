# Mask detective: good overlap, wrong object count?

**Exercise + breakout · CV Day 3 Block 1 · around 30 minutes in pairs**

Your job is to produce a defensible decision from the actual outputs, including a result that challenges your first guess. One partner edits; the other checks controls, arithmetic and visual evidence. Swap roles during the investigation.

## Keep these slide takeaways beside your experiment

Semantic masks label classes; instance masks identify objects. U-Net skips preserve spatial detail. IoU/Dice compare aligned regions; class overlap alone cannot certify object count.

## Before the room opens

Preload learned-mask assets and locate semantic classes, instance IDs and reference masks. Explain same-class/matched-instance overlap and empty-mask convention. Leave class selection and IoU/Dice TODOs for rooms.

Use the [shared setup](../../README.md); model downloads and launch checks happen before the task. Start from the repository root with `uv run --locked python start.py --day 3 --block 1`. Keep the starter open. Prerequisite TODOs discussed in plenary belong to setup; the task below is the room assignment.

## Your investigation

1. Inspect the separate-shapes scene and predict class overlap and instance count after objects touch. Name the class and reference you will compare.

2. Complete class membership and IoU/Dice TODOs. Hand-check a tiny mask pair and the stated empty-mask convention before interpreting real learned outputs.

3. Keep the selected class and models fixed; compare Separate shapes and Touching shapes. Record class IoU/Dice, reference versus predicted identities, and point to the boundary where an identity is lost or retained.

4. Swap roles and inspect Touching ambiguity as a stress case. Explain whether the failure is wrong class membership, merged identity or boundary displacement. Save an overlay that supports your diagnosis; unchanged results also need evidence.

5. Decide whether semantic or instance output is enough for counting individual shapes. Cite overlap AND identity evidence; explain why these authored scenes cannot establish performance on real clinical or industrial images.

## Evidence to bring back

Keep a short comparison log with columns **scene | selected class | IoU | Dice | reference/predicted identities | boundary evidence**. Save run A before changing settings: the notebook’s current report is replaced when controls rerun. Download each comparison figure/report plus the experiment log. Save the Python source separately; browser inputs are not saved in it.

Bring one prediction, measured comparison, marked visual example or traced response, a decision and one limitation. Both partners must explain why the evidence supports it.

If blocked, ask through the trainer’s chat/room-visit route. Use the working partner’s device if needed. Honest unchanged or worse outputs count when explained; label a supplied measured fallback instead of claiming fresh inference.


## Inputs and API help

Read the [function input/output contracts and official API references](TASK_API.md), also displayed immediately beside the notebook TODOs. They define every argument, data field, shape and convention; no private trainer material or inferred field names are needed.

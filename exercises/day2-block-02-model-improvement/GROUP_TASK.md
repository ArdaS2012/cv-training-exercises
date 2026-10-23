# Training clinic: can you defend an improvement?

**Exercise + breakout · CV Day 2 Block 2 · 30 minutes in pairs**

Your job is to produce a defensible decision from the actual outputs, including a result that challenges your first guess. One partner edits; the other checks controls, arithmetic and visual evidence. Swap at minute 15.

## Keep these slide takeaways beside your experiment

Parameters are learned; hyperparameters are chosen. Hold split, seed and budget fixed. Choose by validation loss; reserve test data until the decision is frozen. A lower training loss alone is insufficient.

## Before the room opens

Show the training/validation curves as a presenter-led worked example. Open the fixed checkpoint/splits and baseline rate 0.01, penalty 0.0; explain the declared validation metric. Leave configuration and selection TODOs for rooms.

Use the [shared setup](../../README.md); model downloads and launch checks happen before the task. Start from the repository root with `uv run --locked python start.py --day 2 --block 2`. Keep the starter open. Prerequisite TODOs discussed in plenary belong to setup; the task below is the room assignment.

## Your investigation

1. Inspect the baseline settings (rate 0.01, penalty 0). Predict the training/validation curve change for either a rate or penalty intervention; write which outcome would contradict your hypothesis.

2. Complete configuration and validation-selection TODOs. Run the supplied short CPU trainings with just that factor changed. Read both curves, not only the last number; record a point where their behavior differs.

3. Swap roles. Test a second value of the SAME factor, holding the other setting, split, seed and budget fixed. Save both comparisons before rerunning; choose using the declared validation loss only.

4. Look for a contrary result: a faster initial loss drop, better training fit or larger penalty that did not give the best validation outcome. Explain it using the measured short-run curves; if absent, inspect the labelled longer-run reference without mixing its budget into your live comparison.

5. Freeze your candidate choice and justification, then use the reserved-test action once. Report that estimate without retuning. Explain what small data, one seed and the short budget prevent you from claiming.

## Evidence to bring back

Keep a short comparison log with columns **run | changed factor/value | train-curve evidence | validation loss | decision before test**. Save run A before changing settings: the notebook’s current report is replaced when controls rerun. Download each comparison figure/report plus the experiment log. Save the Python source separately; browser inputs are not saved in it.

Bring one prediction, measured comparison, marked visual example or traced response, a decision and one limitation. Both partners must explain why the evidence supports it.

## Work within the timebox

0–4 min: read the takeaways, inspect inputs and predict. 4–12: implement the bounded TODOs and inspect a real result. 12–22: compare controlled runs; swap editor/checker at minute 15. 22–27: diagnose a stress case and choose. 27–30: agree, save and export evidence.

If blocked, ask through the trainer’s chat/room-visit route. Use the working partner’s device if needed. Honest unchanged or worse outputs count when explained; label a supplied measured fallback instead of claiming fresh inference.

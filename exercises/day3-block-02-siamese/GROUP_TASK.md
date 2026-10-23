# Match desk: which mistake costs more?

**Exercise + breakout · CV Day 3 Block 2 · 30 minutes in pairs**

Your job is to produce a defensible decision from the actual outputs, including a result that challenges your first guess. One partner edits; the other checks controls, arithmetic and visual evidence. Swap at minute 15.

## Keep these slide takeaways beside your experiment

Both branches share encoder weights. Distance compares embeddings; a decision threshold does not retrain them. Here y=1 means similar. A wider acceptance threshold trades missed matches for false matches.

## Before the room opens

Preload the trained shared encoder and held-out pairs. Explain labels y=1 for similar, margin and distance range. Leave distance/contrastive-loss TODOs for rooms.

Use the [shared setup](../../README.md); model downloads and launch checks happen before the task. Start from the repository root with `uv run --locked python start.py --day 3 --block 2`. Keep the starter open. Prerequisite TODOs discussed in plenary belong to setup; the task below is the room assignment.

## Your investigation

1. Look at a similar and dissimilar image pair; predict their distance ordering and which error increases when the cutoff grows. State the y=1 convention before coding.

2. Complete per-pair Euclidean distance and contrastive-loss TODOs. Manually trace one similar and one dissimilar loss, including the margin case; inspect the actual pair images beside measured distances.

3. Freeze pairs/encoder/embeddings and compare cutoffs 0.5 and 0.9. Record false matches and missed matches; identify the pair whose decision changes, or explain why no distance crossed the cutoff.

4. Swap roles. Act as two clients: an access system that fears false matches, and a search tool that fears missed matches. Use the SAME error table to defend a different threshold for each; inspect one visual counterexample to your intuition.

5. Choose one policy and explain its cost tradeoff with an actual pair. Specify how new validation pairs would set the threshold and untouched test pairs would estimate performance; do not tune repeatedly on these teaching pairs.

## Evidence to bring back

Keep a short comparison log with columns **cutoff | false matches | missed matches | changed pair | intended error cost**. Save run A before changing settings: the notebook’s current report is replaced when controls rerun. Download each comparison figure/report plus the experiment log. Save the Python source separately; browser inputs are not saved in it.

Bring one prediction, measured comparison, marked visual example or traced response, a decision and one limitation. Both partners must explain why the evidence supports it.

## Work within the timebox

0–4 min: read the takeaways, inspect inputs and predict. 4–12: implement the bounded TODOs and inspect a real result. 12–22: compare controlled runs; swap editor/checker at minute 15. 22–27: diagnose a stress case and choose. 27–30: agree, save and export evidence.

If blocked, ask through the trainer’s chat/room-visit route. Use the working partner’s device if needed. Honest unchanged or worse outputs count when explained; label a supplied measured fallback instead of claiming fresh inference.

# Convolution detective: where did the pixels go?

**Exercise + breakout · CV Day 2 Block 1 · around 30 minutes in pairs**

Your job is to produce a defensible decision from the actual outputs, including a result that challenges your first guess. One partner edits; the other checks controls, arithmetic and visual evidence. Swap roles during the investigation.

## Keep these slide takeaways beside your experiment

A response is a weighted sum plus bias. Stride changes sampling, not learned weights. ReLU clips negatives; pooling loses positions. Feature channels are not class labels.

## Before the room opens

Preload tiny-CNN and VGG assets and open fixed image row 0. Explain convolution versus ReLU/pooling and locate measured shapes/hooks. Leave filter-response and output-size expressions for rooms.

Use the [shared setup](../../README.md); model downloads and launch checks happen before the task. Start from the repository root with `uv run --locked python start.py --day 2 --block 1`. Keep the starter open. Prerequisite TODOs discussed in plenary belong to setup; the task below is the room assignment.

## Your investigation

1. Predict one patch response, including bias, and the stride-1/stride-2 shapes before running. Mark a thin stroke in fixed digit row 0 that could become harder to locate.

2. Complete the response and output-size TODOs. Check one patch by hand, then inspect the actual learned maps. Explain the first disagreement between your prediction and the measured trace.

3. Keep image, kernel and padding fixed. Run stride 1 then 2; record predicted/measured dimensions and point to a visible change in spatial detail. Swap editor/checker and explain which operation caused it.

4. Keep stride fixed and choose one different digit row. Test whether the same channel lights up in the same place. Compare the supplied VGG trace: name one channel/spatial change, without assigning a human object name to a channel.

5. Decide which sampling setting you would start with if preserving a small stroke mattered. Support it with one shape measurement and one marked visual feature; state why these two digits cannot establish classifier accuracy.

## Evidence to bring back

Keep a short comparison log with columns **row | stride | predicted H×W | measured H×W | map/stroke evidence**. Save run A before changing settings: the notebook’s current report is replaced when controls rerun. Download each comparison figure/report plus the experiment log. Save the Python source separately; browser inputs are not saved in it.

Bring one prediction, measured comparison, marked visual example or traced response, a decision and one limitation. Both partners must explain why the evidence supports it.

If blocked, ask through the trainer’s chat/room-visit route. Use the working partner’s device if needed. Honest unchanged or worse outputs count when explained; label a supplied measured fallback instead of claiming fresh inference.


## Inputs and API help

Read the [function input/output contracts and official API references](TASK_API.md), also displayed immediately beside the notebook TODOs. They define every argument, data field, shape and convention; no private trainer material or inferred field names are needed.

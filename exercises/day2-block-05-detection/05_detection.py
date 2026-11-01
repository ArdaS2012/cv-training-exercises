import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="Compare Detector Outputs")


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(next(p for p in Path(__file__).resolve().parents if (p/'cv_exercises').is_dir())))
    import marimo as mo
    from cv_exercises import day2,display

    return day2, display, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Detector audit: find what the threshold cannot fix

    **Around 30 minutes in pairs.** Detectors return boxes, labels and scores. IoU measures geometry, not class correctness. Confidence filters predictions; NMS suppresses overlap. Filtering cannot create a missing box.

    ### Investigate, do not just run

    1. On the bus photograph, mark one object you expect to survive confidence 0.5→0.7. Predict whether its geometry will change or only its retention.

    2. Complete confidence selection and continuous-coordinate IoU TODOs. Hand-check one supplied box pair, a disjoint pair and equality at the score cutoff. Then run the actual CPU detectors.

    3. Hold image, model outputs and NMS fixed; compare confidence 0.5 versus 0.7 for YOLO and SSD. Record boxes kept, one reference IoU and a concrete miss/duplicate; label live inference or measured fallback honestly.

    4. Swap roles. Restore confidence 0.5, change ONLY NMS overlap cutoff, and inspect one overlapping pair. Explain the difference from your confidence experiment. If counts stay fixed, use the displayed scores/overlaps to explain why.

    5. Recommend settings for retaining the chosen object without a duplicate. Identify a remaining bad box or absent object that filtering cannot repair. Use the second photograph only as a bounded generalization check, not a universal model ranking.

    **Evidence log:** image | detector/mode | confidence | NMS | kept boxes | reference IoU / visible error. Save a run before changing controls. Export both the current findings and your comparison log.
    """)
    return






@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Compare Detector Outputs

    Investigation after guided setup. Run two real pretrained CPU detectors on shared images; inspect confidence filtering, IoU, reference matches and a missed or duplicate object.

    Predict → complete the short TODOs → change one control → inspect → explain → download.

    Edit only `# TODO` regions; run with Shift+Enter. Save Python with Ctrl+S / Command+S. Download findings to retain current controls, measurements and observations. The untouched starter shows friendly waiting messages. Work in pairs and swap who edits. Prepared code and trained assets are supplied; no API key or GPU.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    image_id=mo.ui.dropdown(options=['bus','zidane'],value='bus',label='Shared photograph')
    cutoff=mo.ui.slider(0,1,step=.05,value=.5,label='Confidence cutoff (inclusive)')
    overlap_cutoff=mo.ui.slider(.1,.95,step=.05,value=.5,label='Class-aware NMS IoU cutoff')
    run_detectors=mo.ui.run_button(label='Run actual YOLO and SSD on CPU')
    prediction=mo.ui.text_area(label='Predict a match, miss or duplicate as thresholds change',full_width=True)
    mo.vstack([mo.hstack([image_id,cutoff,overlap_cutoff],wrap=True),prediction,run_detectors])
    return cutoff, image_id, overlap_cutoff, prediction, run_detectors


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Read a detection, not just its score
    Each record contains a class label, score and rectangle $[x_1,y_1,x_2,y_2]$ in original-image pixels. Continuous-coordinate area is $(x_2-x_1)(y_2-y_1)$; intersecting widths/heights are clipped at zero. IoU is intersection area divided by union area. It measures geometric overlap, independently of whether the class is right.

    Confidence selection keeps scores at least the cutoff. Class-aware NMS suppresses a lower-scoring same-class rectangle when its IoU with a retained rectangle exceeds the NMS cutoff. Test these controls separately. Neither edits a retained box's geometry nor creates a prediction absent from the decoded candidates. Dashed reference boxes are teaching annotations; the two photographs are not a benchmark.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Your implementation task

    Complete confidence selection and a two-box overlap metric. Keep original
    candidate records at an inclusive score cutoff; compute continuous-coordinate
    IoU for valid xyxy rectangles. The guide below defines every record field,
    coordinate, boundary case and output. YOLO/SSD inference, class-aware NMS,
    reference matching and plotting are supplied separately.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TODO guide — filter confidence and measure box overlap

    A detector proposes **candidate detections**, each with an identity, class label,
    confidence score and box. Your first function decides which candidates meet a
    score threshold. Your second measures geometry between two boxes. The supplied
    code performs model inference, later duplicate suppression, plotting and matching
    to teaching reference boxes. You do not implement a detector or NMS.

    ### TODO 1: `confidence_selection(detections, threshold)`

    | Argument | Type and meaning |
    | --- | --- |
    | `detections` | Python `list` of ordinary detection dictionaries, possibly empty. Order is the adapter's candidate order; it need not be sorted by score. |
    | `threshold` | Finite Python float in `[0, 1]` from **Confidence cutoff**. It is an inclusive minimum score, not the NMS overlap cutoff. |

    Every full adapter record contains these exact keys:

    | Key | Value type, meaning and convention |
    | --- | --- |
    | `id` | Unique `str` within one detector run, for example `"candidate-7"`. It identifies the existing record; it is not a class ID. |
    | `label` | Class-name `str`, for example `"person"`. Both adapters convert their model-specific numeric IDs to names. |
    | `score` | Finite float in `[0, 1]`: the detector's confidence. It is neither measured IoU nor a correctness guarantee. |
    | `box` | Length-four `list[float]`, `[x_min, y_min, x_max, y_max]`, in original-image pixel coordinates. The last two entries are the opposite corner, not width/height. |

    Shape example only:
    `{"id": "candidate-7", "label": "person", "score": 0.63, "box": [10.0, 20.0, 50.0, 70.0]}`.
    This illustrates data layout, not a reference answer or a calibrated confidence.
    Selection depends only on `score`; small hand-check records may contain only that
    field. Preserve any other supplied fields without interpreting them.

    Return a **new list containing the original dictionary objects** whose score is
    at least the threshold, in their original relative order. Preserve IDs, labels,
    scores, boxes and all record fields; do not rebuild or edit retained records.
    A score exactly equal to the cutoff is retained. Return `[]` for empty input or
    when none qualify. Duplicate/overlapping candidates are still separate records
    here. Do not sort, merge boxes, filter by class or apply the later NMS rule.
    Missing/nonfinite scores and invalid thresholds are outside the supplied domain.

    **Python toolbox:** dictionary lookup uses exact quoted keys; list positions
    start at zero. A generic `items.append(value)` adds one item and returns `None`,
    so its result is not a new list. `>=` includes equality. Basic loops, conditionals
    and list construction need no extra import. Choose your own implementation.

    ### TODO 2: `box_iou(box_a, box_b)`

    Each argument is a length-four Python list of finite numbers, in the **same xyxy
    order** and same original-image pixel coordinate system. The origin is top-left,
    x increases rightward and y downward. Coordinates may be fractional. The runner
    supplies positive-area rectangles: `x_min < x_max` and `y_min < y_max`.
    No normalization to `[0, 1]`, array axis conversion or image dimensions are needed.

    **Intersection over Union (IoU)** compares shared area with area covered by either
    box. With `A` and `B` denoting the two rectangles:

    $$w_I=\max(0,\min(x_{A,max},x_{B,max})-\max(x_{A,min},x_{B,min})),$$
    $$h_I=\max(0,\min(y_{A,max},y_{B,max})-\max(y_{A,min},y_{B,min})),$$
    $$I=w_Ih_I,\qquad U=|A|+|B|-I,\qquad \mathrm{IoU}=I/U.$$

    `w_I` and `h_I` are overlapping width/height in pixels; `I` is intersection area,
    `U` union area, and `|A|`, `|B|` each box's width times height in pixels squared.
    The later starting edge and earlier ending edge bound an overlap. Clipping a
    negative span to zero expresses no overlap. Union subtracts the shared area once
    because the two individual areas counted it twice. Valid positive-area boxes
    guarantee a positive denominator.

    Return one finite Python/NumPy scalar in `[0, 1]`, with no units. Merely touching
    or disjoint boxes have IoU zero; identical valid boxes have IoU one. This is
    continuous-coordinate geometry: do **not** add an inclusive-pixel `+1` to widths.
    Keep both inputs unchanged. Degenerate/reversed boxes are outside the supplied
    domain; no additional invalid-box policy is required. Python `min(a, b)` and
    `max(a, b)` return a scalar smaller/larger argument; they do not interpret boxes.

    The Torchvision reference below documents the same metric but accepts batches
    of tensors and returns a matrix of pairwise IoUs. Your TODO instead keeps this
    local **two Python boxes → one scalar** interface; do not return that matrix.
    IoU alone does not compare class labels or establish that either box is correct.

    ### Before running and reading the evidence

    Replace only `selected=None` and `overlap=None` in the marked regions. Keep the
    function signatures/return lines and rerun with **Shift+Enter**. Check score
    equality, no retained candidates, identical boxes, a partial overlap and a
    separated/touching pair by hand. These checks test distinct boundary meanings.

    Until **Run actual YOLO and SSD on CPU** is pressed, the notebook labels its mode
    as a measured pretrained snapshot. Live inference uses cached model weights;
    a missing cache reports a preparation error. Both paths expose decoded candidate
    records after permissive native confidence `0.01` and NMS `0.95`, so they are not
    raw logits or all possible boxes.

    After your filter, supplied **class-aware non-maximum suppression (NMS)** can
    remove a lower-scoring same-class box when overlap exceeds its cutoff. It does
    not change the retained box coordinates. Hold this control fixed during the
    confidence experiment. The report's `models` dictionary maps each detector name
    to `input_candidates`, `above_cutoff`, `after_class_aware_nms` integer counts,
    `kept` detection records and `reference_matches`. Each reference match has
    `reference: str`, `class: str`, `best_iou: float`, `match: bool`, and
    `detection: str | None`; a match requires the same class and IoU at least `0.5`. Matching is greedy in the authored reference order: each reference considers only same-class predictions not already matched to an earlier reference. Thus one prediction cannot satisfy two references; `best_iou` is the best overlap among the remaining eligible predictions, or zero if none remain.
    These reference annotations are teaching examples, not an exhaustive benchmark.
    Record the model/mode and inspect a concrete miss or duplicate: postprocessing
    cannot invent a candidate the detector never produced.

    **Version-matched official references:** [Python 3.12 lists and dictionaries](https://docs.python.org/3.12/library/stdtypes.html),
    [Python 3.12 min/max](https://docs.python.org/3.12/library/functions.html#max),
    [Torchvision 0.23 box_iou](https://docs.pytorch.org/vision/0.23/generated/torchvision.ops.box_iou.html),
    [Torchvision 0.23 class-aware batched_nms](https://docs.pytorch.org/vision/0.23/generated/torchvision.ops.batched_nms.html).
    """)
    return



@app.cell
def _():
    def confidence_selection(detections,threshold):
        # TODO 1: list of unchanged detection records with score >= threshold.
        selected=None
        # END TODO 1
        return selected
    def box_iou(box_a,box_b):
        # TODO 2: intersection over union for valid continuous xyxy rectangles.
        overlap=None
        # END TODO 2
        return overlap

    return box_iou, confidence_selection


@app.cell(hide_code=True)
def _(day2, image_id, mo, run_detectors):
    live_outputs=None
    inference_times=None
    _failure=None
    if run_detectors.value:
        try:live_outputs,inference_times=day2.detector_inference(image_id.value,offline=True)
        except (ValueError,OSError,RuntimeError) as _error:_failure=str(_error)
    mo.callout(_failure,kind='warn') if _failure else mo.md('Mode: actual CPU pretrained inference') if live_outputs is not None else mo.md('Mode: measured pretrained snapshot (explicit fallback; press Run for actual CPU inference).')
    return inference_times, live_outputs


@app.cell(hide_code=True)
def _(
    box_iou,
    confidence_selection,
    cutoff,
    day2,
    image_id,
    live_outputs,
    mo,
    overlap_cutoff,
):
    detection_result=None
    _message=None
    try:detection_result=day2.inspect_detections(confidence_selection,box_iou,image_id.value,cutoff.value,overlap_cutoff.value,live_outputs)
    except (ValueError,TypeError,IndexError,KeyError) as _error:_message=str(_error)
    mo.callout(_message,kind='warn') if _message else mo.callout('Complete confidence_selection and box_iou to inspect detections.',kind='info') if detection_result is None else mo.md('Inspect the annotated comparison; scores and overlap are separate evidence.')
    return (detection_result,)


@app.cell(hide_code=True)
def _(detection_result, mo):
    import io
    import matplotlib.pyplot as plt
    _view=None
    if detection_result is not None:
        _report,_figure=detection_result
        _buffer=io.BytesIO();_figure.savefig(_buffer,format='png',dpi=150,bbox_inches='tight')
        _view=mo.vstack([_figure,mo.download(_buffer.getvalue(),filename='comparison.png',label='Download figure'),mo.md('```json\n'+__import__('json').dumps(_report,indent=2)+'\n```')])
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    experiment_log=mo.ui.text_area(value="image | detector/mode | confidence | NMS | kept boxes | reference IoU / visible error\nA | \nB | \nStress case | \nDecision + limitation | ",label='Comparison log — preserve run A before changing a control',full_width=True)
    mo.vstack([mo.md("### Keep your evidence while experimenting\nEach rerun replaces the current result. Copy measured values and name a visible feature or response field here before changing controls."),experiment_log])
    return (experiment_log,)


@app.cell(hide_code=True)
def _(display, experiment_log, mo):
    display.download(mo,experiment_log.value,{'columns':"image | detector/mode | confidence | NMS | kept boxes | reference IoU / visible error"},'day2-block5-comparison-log',label='Download comparison log (baseline + changed runs)')
    return


@app.cell(hide_code=True)
def _(mo):
    observations=mo.ui.text_area(label='Findings: measured change, explanation and one limitation',full_width=True)
    observations
    return (observations,)


@app.cell(hide_code=True)
def _(
    cutoff,
    detection_result,
    display,
    image_id,
    inference_times,
    mo,
    observations,
    overlap_cutoff,
    prediction,
):
    display.download(mo,observations.value,{'prediction':prediction.value,'image':image_id.value,'confidence':cutoff.value,'nms_iou':overlap_cutoff.value,'inference_seconds':inference_times,'report':detection_result[0] if detection_result else None},'detector-findings')
    return


if __name__ == "__main__":
    app.run()

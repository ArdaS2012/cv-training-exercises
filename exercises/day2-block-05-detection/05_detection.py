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
    mo.md("""
    ## TODO — confidence selection and rectangle overlap
    `detections` is a list of records with `id`, `label`, `score`, `box`. Keep the original records whose score is at least the cutoff; return a list. `box_a` and `box_b` are valid continuous xyxy boxes in original-image pixel coordinates. Compute intersection width/height with nonnegative clipping, intersection area and union, then return IoU in [0,1]. There is no inclusive `+1` pixel convention.

    Actual pretrained CPU adapters are supplied for YOLO11n and SSDLite320. Run online `--day 2 --block 5 --check` before class to cache weights. Until live inference is requested, the notebook clearly uses measured pretrained outputs as a fallback. Both routes include native permissive NMS0.95 and confidence0.01 before the classroom filter; these are decoded candidates, not raw logits. Orange boxes are retained predictions; dashed green boxes are manual teaching references, not an exhaustive benchmark. Inspect one class/IoU match and a miss or duplicate. A high confidence alone cannot establish box correctness.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Function inputs and API reference

    ### TODO 1: `confidence_selection(detections, threshold)`

    `detections` is a Python list (possibly empty). Each item is an ordinary dictionary with **exactly these adapter fields**:

    | Field | Definition |
    |---|---|
    | `id` | Unique string identifying a candidate within one detector run, for example `"candidate-7"`. Preserve it. |
    | `label` | Class-name string, for example `"person"`; this is not a numeric class ID. |
    | `score` | Finite float in `[0, 1]`, the detector's confidence. It is not measured IoU or a guarantee of correctness. |
    | `box` | List of four finite floats `[x_min, y_min, x_max, y_max]` in original-image pixels. |

    `threshold` is a finite scalar float in `[0, 1]` from **Confidence cutoff**. Return a **list of the original record objects**, retaining their input order and fields, for the stated inclusive cutoff (`score` equal to the threshold is retained). Empty input or no retained candidate produces `[]`. Do not change scores, boxes or input records. Filtering confidence is separate from the supplied later class-aware non-maximum suppression (NMS), which removes overlapping duplicate predictions; do not implement NMS here.

    ### TODO 2: `box_iou(box_a, box_b)`

    Each argument is a length-four Python list of finite numbers in the above **xyxy** order, referring to the same image. Coordinates use x rightwards and y downwards from the top-left. Both rectangles have positive width/height: `x_min < x_max` and `y_min < y_max`. The last two values are maximum corners, **not width and height**. Example structure: `[10.0, 20.0, 50.0, 70.0]`.

    Return one finite scalar IoU in `[0, 1]`: intersection area divided by union area. Area uses continuous pixel-coordinate geometry; do not add 1 to widths/heights. Disjoint or merely touching boxes have zero intersection; identical valid boxes have IoU 1. No tensor, mask or dictionary is returned. The helper handles reference matching; preserve your function's two-box interface.

    **API vocabulary:** dictionary `record["field"]` access, list indexing, and Python `min`/`max` on scalar coordinates are available. Their operation definitions are in the references; you choose how to express the overlap calculation.

    **Official references:** [Python dictionaries](https://docs.python.org/3.12/library/stdtypes.html#mapping-types-dict), [Python min/max](https://docs.python.org/3.12/library/functions.html#max), [Torchvision box_iou coordinate/metric contract](https://docs.pytorch.org/vision/0.23/generated/torchvision.ops.box_iou.html).
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

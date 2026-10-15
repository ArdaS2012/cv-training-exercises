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
    ## Exercise + breakout: your shared task

    **15 minutes in pairs after guided setup.** Distinguish model predictions, confidence selection and geometric overlap in an evidence-based detector comparison.

    1. Predict a match/miss. Complete both TODOs, manually check IoU for one pair of supplied boxes and verify equality at the confidence cutoff.

    2. Run YOLO and SSD on the bus image at confidence 0.5. Change only confidence to 0.7 with NMS fixed; compare retained boxes and one reference match.

    3. Identify one miss, duplicate or incorrect box and propose a follow-up test. Explain why filtering cannot recover an object absent from raw predictions and why this photograph cannot rank detector families universally.

    **Bring back:** Annotated outputs, checked IoU/cutoff case, threshold comparison and one diagnosed detection limit.

    Swap editor/checker halfway. Record in the observation fields, save code and download findings. The complete task and setup are in this block’s `GROUP_TASK.md`. Prerequisite TODOs discussed in plenary belong to setup.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Compare Detector Outputs

    15-minute group task after guided setup. Run two real pretrained CPU detectors on shared images; inspect confidence filtering, IoU, reference matches and a missed or duplicate object.

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
    mo.md("""
    ## TODO — confidence selection and rectangle overlap
    `detections` is a list of records with `id`, `label`, `score`, `box`. Keep the original records whose score is at least the cutoff; return a list. `box_a` and `box_b` are valid continuous xyxy boxes in original-image pixel coordinates. Compute intersection width/height with nonnegative clipping, intersection area and union, then return IoU in [0,1]. There is no inclusive `+1` pixel convention.

    Actual pretrained CPU adapters are supplied for YOLO11n and SSDLite320. Run online `--day 2 --block 5 --check` before class to cache weights. Until live inference is requested, the notebook clearly uses measured pretrained outputs as a fallback. Both routes include native permissive NMS0.95 and confidence0.01 before the classroom filter; these are decoded candidates, not raw logits. Orange boxes are retained predictions; dashed green boxes are manual teaching references, not an exhaustive benchmark. Inspect one class/IoU match and a miss or duplicate. A high confidence alone cannot establish box correctness.
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

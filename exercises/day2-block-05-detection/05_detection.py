import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title='Build detector postprocessing')


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path
    root=next(p for p in Path(__file__).resolve().parents if (p/'cv_exercises').is_dir())
    sys.path.insert(0,str(root))
    import marimo as mo
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from cv_exercises import day2, display
    api=(root/'exercises/day2-block-05-detection/TASK_API.md').read_bytes()
    return api, day2, display, mo, np, plt, torch


@app.cell(hide_code=True)
def _(api, mo):
    mo.hstack([mo.md('# Build detector postprocessing'),mo.download(api,filename='TASK_API.md',label='API reference (.md)')],justify='space-between')
    return


@app.cell(hide_code=True)
def _(day2, mo):
    from PIL import Image
    mo.image(Image.open(day2.ASSETS/'bus.jpg'),width=650)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Build confidence filtering, rectangle overlap and class-aware NMS. Run applies them to actual cached YOLO/SSD outputs; the measured snapshot is explicitly labelled.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1 · Confidence filter
    `confidence_selection(detections,threshold)`: list of dicts with `id:str`, `label:str`, `score:float[0,1]`, `box:list[float]` in original-image pixel `[x1,y1,x2,y2]` order. threshold float [0,1]. Return unchanged records with score ≥ threshold in original order. [] is valid; preserve inputs.
    """)
    return


@app.cell
def _():
    def confidence_selection(detections,threshold):
        # TODO 1: retain source records meeting the inclusive cutoff.
        selected=None
        # END TODO 1
        return selected
    return (confidence_selection,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2 · Intersection over union
    `box_iou(a,b)`: positive-area continuous xyxy boxes, x2>x1/y2>y1. Return float intersection/(areaA+areaB−intersection) in [0,1]. Touching/disjoint boxes have zero overlap. No +1 pixel convention; preserve inputs.
    """)
    return


@app.cell
def _():
    def box_iou(a,b):
        # TODO 2: clipped intersection area over union area.
        overlap=None
        # END TODO 2
        return overlap
    return (box_iou,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3 · Greedy class-aware suppression
    `suppress(detections,threshold)`: same records, threshold float [0,1]. Visit descending score; ties keep original order. Keep a candidate unless an already-kept **same-label** box has IoU > threshold. Return unchanged records in visit order; [] stays []; equality at cutoff is retained. Use your box_iou; preserve inputs.
    """)
    return


@app.cell
def _(box_iou):
    def suppress(detections,threshold):
        # TODO 3: stable score order and same-class overlap suppression.
        kept=None
        # END TODO 3
        return kept
    return (suppress,)


@app.cell(hide_code=True)
def _(day2, mo):
    image_id=mo.ui.dropdown([e['id'] for e in day2.detector_snapshot()['images']],value='bus',label='Image')
    cutoff=mo.ui.slider(.1,.9,step=.1,value=.5,show_value=True,label='Confidence')
    overlap_cutoff=mo.ui.slider(.1,.9,step=.1,value=.5,show_value=True,label='NMS IoU')
    run_detectors=mo.ui.run_button(label='Run actual YOLO and SSD on CPU')
    mo.vstack([mo.hstack([image_id,cutoff,overlap_cutoff]),run_detectors])
    return cutoff, image_id, overlap_cutoff, run_detectors


@app.cell(hide_code=True)
def _(day2, image_id, mo, run_detectors):
    live_outputs=None
    inference_times=None
    _status=day2.detector_status();_note=''
    if run_detectors.value:
        if not all(_status.values()):
            _note=' **Live models are not cached on this computer**, so the snapshot stays active. Keep working with it: it holds the same models\' real outputs. To enable live runs later, run the online check once: `uv run --locked python start.py --day 2 --block 5 --check`.'
        else:
            try:live_outputs,inference_times=day2.detector_inference(image_id.value,offline=True)
            except Exception as _e:_note=f' Live run failed ({type(_e).__name__}: {_e}); the snapshot stays active.'
    _mode='Mode: actual CPU pretrained inference' if live_outputs is not None else 'Mode: measured pretrained snapshot (bundled real YOLO11n/SSDLite outputs).'
    mo.md(_mode+(f" Inference seconds: {', '.join(f'{k} {v:.2f}' for k,v in inference_times.items())}." if inference_times else '')+_note)
    return inference_times, live_outputs


@app.cell(hide_code=True)
def _(box_iou, confidence_selection, cutoff, day2, image_id, live_outputs, mo, overlap_cutoff, suppress):
    detection_result=None
    try:
        detection_result=day2.inspect_detections(confidence_selection,box_iou,image_id.value,cutoff.value,overlap_cutoff.value,live_outputs,suppress=suppress)
        if detection_result is None:mo.output.replace(mo.md('Waiting: complete filtering, IoU and NMS.'))
        else:
            _report,_figure=detection_result
            _counts=' · '.join(f"{k}: {v['input_candidates']} candidates → {v['above_cutoff']} filtered → {v['after_class_aware_nms']} NMS" for k,v in _report['models'].items())
            mo.output.replace(mo.vstack([_figure,mo.md(_counts),mo.md('Orange = your kept boxes; green dashed = teaching references.\n\n**Try:** (1) Confidence 0.5 → 0.2: which extra boxes appear, and are they duplicates or new objects? (2) NMS IoU 0.5 → 0.9: when do duplicates survive? (3) Find one reference that YOLO matches and SSD misses, or the reverse. What does that say about one image as evidence?')]))
    except (ValueError,TypeError,IndexError,KeyError) as _e:mo.output.replace(mo.md(str(_e)))
    return (detection_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4 · Stretch: precision and recall over confidence
    `precision_recall(report)`: one model's dict from the result above, keys `'reference_matches'` (list of dicts, one per teaching reference, each with `'match'`: bool, plus `'reference'`, `'class'`, `'best_iou'`, `'detection'`) and `'after_class_aware_nms'` (int, number of kept boxes, ≥ 0). Return a tuple of two Python floats `(precision, recall)`:

    $\text{precision}=\frac{TP}{\text{kept}}$, $\;\text{recall}=\frac{TP}{\text{references}}$, TP = number of references with `match == True` (IoU ≥ 0.5, same class, one detection per reference). Define precision = 1.0 when nothing is kept. The supplied sweep calls it for cutoffs 0.1…0.9 with your current NMS threshold.
    """)
    return


@app.cell
def _():
    def precision_recall(report):
        # TODO 4 (stretch): count true positives, then two ratios.
        values=None
        # END TODO 4
        return values
    return (precision_recall,)


@app.cell(hide_code=True)
def _(box_iou, confidence_selection, day2, image_id, live_outputs, mo, overlap_cutoff, precision_recall, suppress):
    pr_result=None
    try:
        pr_result=day2.pr_sweep(confidence_selection,box_iou,suppress,image_id.value,overlap_cutoff.value,precision_recall,live=live_outputs)
        if pr_result is None:_v=mo.md('Optional: complete TODO 1–3 and TODO 4 for the precision/recall sweep.')
        else:_v=mo.vstack([pr_result[1],mo.md("""
    **Try:** (1) Which direction does a higher cutoff move each curve? (2) Where would you put the cutoff for a safety application that must not miss a person? (3) Average precision (AP) summarizes this curve per class; mAP averages AP over classes and IoU thresholds. Two images are far too few for a benchmark.""")])
    except (ValueError,TypeError,KeyError,ZeroDivisionError) as _e:_v=mo.md(str(_e))
    _v
    return (pr_result,)


@app.cell(hide_code=True)
def _(detection_result, display, inference_times, mo):
    display.download(mo,'',{'report':detection_result[0] if detection_result else None,'inference_seconds':inference_times},'detection-trial',label='Save trial')
    return


if __name__ == "__main__":
    app.run()

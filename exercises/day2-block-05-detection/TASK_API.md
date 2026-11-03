# Detection coding/API reference

Complete record schema, shapes and boundary rules sit beside each TODO. All coordinates use continuous original-image pixels in xyxy order; width=x2-x1, height=y2-y1 (no +1). Scores are model-specific confidence, not IoU or calibrated cross-model probabilities. Input boxes have positive area; union cannot be zero. Return original record objects, preserving all fields and input list.

Python sorted(records,key=...) returns a new stable list; equal keys preserve original order. NMS compares only retained boxes of the same label, suppressing when IoU is strictly greater than cutoff. Empty inputs return []; different labels are independent. Cutoff0 removes any same-class positive overlap; cutoff1 removes none. IoU equality is retained.

`day2.detector_inference(image_id,offline=True)` uses cached YOLO11n/SSDLite320 models and returns `(outputs,times)`: outputs maps model names to detection lists; times maps names to float seconds. Missing weights report preparation failure. Until Run, explicit measured snapshots are used. Both paths expose decoded candidates after permissive native confidence .01/NMS .95; these are not raw logits/all possible boxes.

`day2.inspect_detections(filter_scores,iou,image_id,cutoff,nms_cutoff,live=None,suppress=your_function)` calls all three student functions and returns `(report,figure)` or None when unfinished. There is no reference-solution fallback for unfinished NMS. Report fields: image_id,confidence_cutoff,nms_iou_cutoff,match_iou_cutoff(.5),mode,limits,models. Each model has input_candidates/above_cutoff/after_class_aware_nms integer counts, kept:list[record], reference_matches:list[dict]. Each match has reference:str,class:str,best_iou:float[0,1],match:bool,detection:str|None. Matching is greedy in authored reference order, same-class, without reusing a prediction, IoU≥.5. Green dashed boxes are teaching references; orange boxes are your detections. References are not exhaustive; two images do not establish benchmark accuracy or model ranking.

Compare confidence .5→.7 with NMS fixed, then NMS .3→.7 at fixed confidence. Postprocessing cannot invent missing candidates.

[Python stable sorting](https://docs.python.org/3.12/howto/sorting.html), [Torchvision 0.23 box_iou](https://docs.pytorch.org/vision/0.23/generated/torchvision.ops.box_iou.html), [batched_nms](https://docs.pytorch.org/vision/0.23/generated/torchvision.ops.batched_nms.html).

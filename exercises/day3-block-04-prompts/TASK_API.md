# Vision-prompt coding reference

Three TODO contracts/formulas appear beside the code: cosine similarity, stable candidate softmax and normalized-to-original-pixel prompt conversion. The prepared Run path performs real CLIP text encoding and SAM mask decoding using the pinned local model cache and matching cached image embeddings. No model weights are updated. Loading and tokenization are supplied; the prepared comparison uses `day3.text_scores` to initialize/reuse the actual model.

## NumPy operations
`np.linalg.norm(a,axis=...)` reduces the selected axis; `texts @ image` is matrix-vector multiplication. `np.max` gives the largest scalar; `np.exp` is elementwise exponential. `array.tolist()` converts arrays to JSON-compatible nested Python lists. `list(labels)` copies a list. Broadcasting a `(2,)` vector over `(P,2)` applies one x/y scale per row. No zero feature vectors or malformed inputs enter these TODOs.

## Prepared model adapters
`day3.text_scores(candidates,image_id,online=False)` accepts 2–6 nonempty text strings and image ID `bus` or `zidane`; returns dictionary image:str, candidates:list[str], cosines:list[float], scores:list[float], scale:float, scope:str. Scores normalize across candidates only; an unchanged cosine can have a changed score when candidates change. The exercise compares your result to this actual API computation.

`day3.spatial_mask(prompt,image_id,online=False)` accepts `{'points': [[x,y],…], 'labels':[1/0,…]}` in original-image pixel coordinates; each x is in [0,W), y in [0,H). Our normalized interface chooses first/last pixel centers using W−1/H−1. The adapter also supports optional `box:[x0,y0,x1,y1]` in xyxy edge coordinates, but this exercise uses points only. It returns image, prompt, model_mask_quality_scores:list[float], chosen:int, masks:bool[M,H,W], mask:bool[H,W], scope:str. It runs the actual SAM decoder and chooses the largest predicted quality score; this is not measured accuracy or a class label.

Keep image, pinned revisions and text candidates fixed for spatial changes. The second photograph is an optional stress case after inspecting bus results. The displayed image is RGB H×W×3; plotting uses original coordinates. No written record is required; optional Save trial includes the measured scores/settings.

[NumPy norm](https://numpy.org/doc/2.2/reference/generated/numpy.linalg.norm.html) · [exp](https://numpy.org/doc/2.2/reference/generated/numpy.exp.html) · [broadcasting](https://numpy.org/doc/2.2/user/basics.broadcasting.html) · [Transformers 4.57.1 CLIP](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/clip) · [SAM](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/sam).

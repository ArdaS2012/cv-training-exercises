import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Build vision prompt scoring")


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(next(p for p in Path(__file__).resolve().parents if (p/'cv_exercises').is_dir())))
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import json
    import torch
    from cv_exercises import day3, display
    reference=(next(p for p in Path(__file__).resolve().parents if (p/'cv_exercises').is_dir())/'exercises/day3-block-04-prompts/TASK_API.md').read_bytes()
    return day3, display, json, mo, np, plt, reference, torch


@app.cell(hide_code=True)
def _(mo, reference):
    mo.hstack([mo.md('# Build vision prompt scoring'),mo.download(reference,filename='TASK_API.md',label='API reference (.md)')],justify='space-between',align='center')
    return


@app.cell(hide_code=True)
def _(mo):
    image_id=mo.ui.dropdown({'Bus':'bus','Football':'zidane'},value='Bus',label='Image')
    prompt_x=mo.ui.slider(0.,1.,step=.01,value=.44,show_value=True,label='Point x / width')
    prompt_y=mo.ui.slider(0.,1.,step=.01,value=.6,show_value=True,label='Point y / height')
    negative=mo.ui.checkbox(value=False,label='Add negative point')
    run=mo.ui.run_button(label='Run CLIP and SAM')
    mo.vstack([mo.hstack([image_id,negative]),mo.hstack([prompt_x,prompt_y]),run])
    return image_id, negative, prompt_x, prompt_y, run


@app.cell(hide_code=True)
def _(mo):
    description_style=mo.ui.dropdown(['bare labels','photo descriptions'],value='photo descriptions',label='Text candidates')
    description_style
    return (description_style,)


@app.cell(hide_code=True)
def _(day3, image_id, mo):
    mo.image(str(day3.ASSETS/(image_id.value+'.jpg')),width=420)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1 · Score image–text alignment
    `image`: finite nonzero float NumPy `(D,)`; `texts`: finite nonzero-row float NumPy `(N,D)`, N=2–6, D≥1. Return float NumPy `(N,)`: $s_i=(v^Tt_i)/(\|v\|_2\|t_i\|_2)$ in candidate order. Vectors are dimensionless; normalize each text row separately. Preserve inputs; zero vectors are outside this task.
    """)
    return


@app.cell
def _(np, torch):
    def cosine(image, texts):
        # TODO: calculate cosine similarity per candidate.
        result=None
        # END TODO
        return result
    return (cosine,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2 · Normalize candidate scores
    `scores`: finite float NumPy `(N,)` cosine scores; `scale`: finite positive float from the model. Return float NumPy `(N,)`: $p_i=e^{\ell_i-\max\ell}/\sum_j e^{\ell_j-\max\ell}$, where $\ell_i=scale\,s_i$. Use the stable shifted form; ties give equal scores. Sum is 1, preserve candidate order/inputs. These scores are relative to this candidate set, not independent truth probabilities.
    """)
    return


@app.cell
def _(np, torch):
    def probabilities(scores, scale):
        # TODO: implement stable softmax over the scaled scores.
        result=None
        # END TODO
        return result
    return (probabilities,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3 · Convert a spatial prompt
    `points`: float NumPy `(P,2)` normalized `(x,y)` in [0,1], P≥1; `labels`: Python list of P binary ints, 1=include/0=exclude. `size`: `(W,H)` Python integer tuple, W,H≥2. Return `{'points': list[list[float]], 'labels': list[int]}` in original-image pixels: multiply x by W−1 and y by H−1 (no rounding). Thus 0/1 select first/last pixel centers. Copy labels; preserve inputs. SAM takes geometry, not text classes.
    """)
    return


@app.cell
def _(np, torch):
    def pixel_prompt(points, labels, size):
        # TODO: map normalized xy points into original-image pixel centers.
        result=None
        # END TODO
        return result
    return (pixel_prompt,)


@app.cell(hide_code=True)
def _(cosine, day3, description_style, image_id, mo, negative, np, pixel_prompt, plt, probabilities, prompt_x, prompt_y, run, torch):
    from PIL import Image
    prompt_result=None
    _view=mo.md('Complete scoring and coordinate functions, then run. Model assets must be cached by preflight.')
    if run.value:
        _name=image_id.value
        _candidates=['a photo of a bus','a photo of a bicycle','a photo of a person'] if _name=='bus' else ['a photo of a person','a photo of a football','a photo of a bus']
        if description_style.value=='bare labels':_candidates=[_text.removeprefix('a photo of a ') for _text in _candidates]
        _image=Image.open(day3.ASSETS/(_name+'.jpg')).convert('RGB')
        _p=np.array([[prompt_x.value,prompt_y.value]]+([[.6,.3]] if negative.value else []))
        _labels=[1]+([0] if negative.value else [])
        _prompt=pixel_prompt(_p,_labels,_image.size)
        if _prompt is not None:
            _api=day3.text_scores(_candidates,_name)
            _tokens=day3._CLIP_PROCESSOR(text=_candidates,return_tensors='pt',padding=True)
            with torch.no_grad():
                _texts=day3._CLIP.get_text_features(**_tokens).cpu().numpy()
            _cached=torch.load(day3.ASSETS/'foundation-encodings.pt',weights_only=True)
            _vector=_cached['clip_'+_name][0].cpu().numpy()
            _cos=cosine(_vector,_texts)
            if _cos is not None:
                _scores=probabilities(_cos,_api['scale'])
                if _scores is not None:
                    _baseline={'points':[[400,300]],'labels':[1]} if _name=='bus' else {'points':[[700,80]],'labels':[1]}
                    _base=day3.spatial_mask(_baseline,_name)
                    _changed=day3.spatial_mask(_prompt,_name)
                    _fig,_axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
                    _rows=np.arange(len(_candidates));_axes[0].barh(_rows-.15,_api['scores'],height=.3,label='Transformers');_axes[0].barh(_rows+.15,_scores,height=.3,label='Yours');_axes[0].set_yticks(_rows,['bus','bicycle','person'] if _name=='bus' else ['person','football','bus']);_axes[0].set_xlim(0,1);_axes[0].legend();_axes[0].set_title('Same candidates')
                    for _ax,_mask,_title,_pr in [(_axes[1],_base['mask'],'Baseline geometry',_baseline),(_axes[2],_changed['mask'],'Your geometry',_prompt)]:
                        _ax.imshow(_image);_ax.imshow(np.ma.masked_where(~_mask,_mask),alpha=.5,cmap='spring');_ax.set_title(_title);_ax.axis('off')
                        for (_x,_y),_lab in zip(_pr['points'],_pr['labels']):_ax.plot(_x,_y,'+' if _lab else 'x',color='lime' if _lab else 'red',markersize=9)
                    prompt_result={'image':_name,'candidates':_candidates,'cosines':np.asarray(_cos).tolist(),'scores':np.asarray(_scores).tolist(),'reference_scores':_api['scores'],'max_score_difference':float(np.max(np.abs(_scores-_api['scores']))),'prompt':_prompt,'baseline_mask_pixels':int(_base['mask'].sum()),'changed_mask_pixels':int(_changed['mask'].sum())}
                    _view=mo.vstack([_fig,mo.md(f"**Max score difference:** {prompt_result['max_score_difference']:.3g} · baseline/changed mask pixels: {prompt_result['baseline_mask_pixels']} / {prompt_result['changed_mask_pixels']}"),mo.md('First compare bare/photo descriptions with geometry fixed. Restore the text; move one point, then add the negative point. The mask has no text class. Model quality scores are estimates; no reference mask is supplied.')])
    _view
    return (prompt_result,)


@app.cell(hide_code=True)
def _(display, mo, prompt_result):
    display.download(mo,'',{'report':prompt_result},'day3-prompts',label='Save trial')
    return


if __name__ == "__main__":
    app.run()

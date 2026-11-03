import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Overlap is not object identity")


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
    reference=(next(p for p in Path(__file__).resolve().parents if (p/'cv_exercises').is_dir())/'exercises/day3-block-01-segmentation/TASK_API.md').read_bytes()
    return day3, display, json, mo, np, plt, reference, torch


@app.cell(hide_code=True)
def _(mo, reference):
    mo.hstack([mo.md('# Overlap is not object identity'),mo.download(reference,filename='TASK_API.md',label='API reference (.md)')],justify='space-between',align='center')
    return


@app.cell(hide_code=True)
def _(mo):
    scene=mo.ui.dropdown({'Separate':0,'Touching':1,'Ambiguous':2,'Day 1 scene':3},value='Touching',label='Scene')
    class_id=mo.ui.dropdown({'Warm':1,'Cool':2,'Background':0},value='Warm',label='Class')
    mo.hstack([scene,class_id])
    return class_id, scene


@app.cell(hide_code=True)
def _(day3, mo, plt, scene):
    seg_data=day3.segmentation(scene.value)
    _fig,_ax=plt.subplots(figsize=(4,3),layout='constrained')
    _ax.imshow(seg_data['image']);_ax.axis('off');_ax.set_title('Input · '+seg_data['name'])
    _fig
    return (seg_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1 · Select a class
    Build a Boolean region from a semantic label image. `labels`: integer NumPy `(H,W)`, IDs 0/1/2; `class_id`: one of those Python ints. Return Boolean NumPy `(H,W)`, true exactly at that class. H,W≥1; absent class gives all false. Preserve inputs.
    """)
    return


@app.cell
def _(np, torch):
    def select_class(labels, class_id):
        # TODO: select one semantic class.
        result=None
        # END TODO
        return result
    return (select_class,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2 · Measure overlap
    For aligned Boolean NumPy masks `a,b: (H,W)`, return `{'iou': float, 'dice': float}` in [0,1]. $\mathrm{IoU}=|A\cap B|/|A\cup B|$; $\mathrm{Dice}=2|A\cap B|/(|A|+|B|)$. Counts are pixels; both empty gives 1 for both, only one empty gives 0. Preserve inputs.
    """)
    return


@app.cell
def _(np, torch):
    def overlap(a, b):
        # TODO: calculate IoU and Dice.
        result=None
        # END TODO
        return result
    return (overlap,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3 · Label connected regions
    Implement flood fill, without calling a labeling library. `mask`: Boolean NumPy `(H,W)`, H,W≥1. Return integer NumPy `(H,W)`: background 0, each four-connected foreground region a positive ID. Visit seeds row by row, left to right; number new regions 1,2,… . Neighbors share an edge, not just a corner. All false returns zeros; preserve mask. Compare your result with SciPy below.
    """)
    return


@app.cell
def _(np, torch):
    def components(mask):
        # TODO: flood fill four-connected foreground regions.
        result=None
        # END TODO
        return result
    return (components,)


@app.cell(hide_code=True)
def _(class_id, components, day3, mo, np, overlap, plt, scene, seg_data, select_class):
    from scipy import ndimage
    seg_result=None
    _data=seg_data
    _mask=select_class(_data['semantic'],class_id.value)
    _view=mo.md('Waiting: select class, overlap and connected regions.')
    if _mask is not None:
        if not isinstance(_mask,np.ndarray) or _mask.dtype!=bool or _mask.shape!=_data['semantic'].shape:
            _view=mo.md('Select class: return a Boolean H×W NumPy mask.')
        else:
            _metrics=overlap(_mask,_data['reference_semantic']==class_id.value)
            _regions=components(_mask)
            if _metrics is not None and _regions is not None:
                if not isinstance(_regions,np.ndarray) or _regions.shape!=_mask.shape or not np.issubdtype(_regions.dtype,np.integer):
                    _view=mo.md('Components: return an integer H×W map.')
                else:
                    _reference,_count=ndimage.label(_mask)
                    _fig,_axes=plt.subplots(2,3,figsize=(10,5),layout='constrained')
                    for _ax,(_title,_arr) in zip(_axes.flat,[('Input',_data['image']),('Semantic class',_mask),('Your connected regions',_regions),('SciPy four-connected',_reference),('Learned-center instances',_data['instance']),('Reference instances',_data['reference_instance'])]):
                        _ax.imshow(_arr,cmap='tab10' if _arr.ndim==2 else None);_ax.set_title(_title);_ax.axis('off')
                    seg_result={'scene':_data['name'],'class':class_id.value,'overlap':_metrics,'your_regions':int(_regions.max()),'scipy_regions':int(_count),'same_label_map':bool(np.array_equal(_regions,_reference)),'reference_instances':int(_data['reference_instance'].max())}
                    _view=mo.vstack([_fig,mo.md(f"**IoU / Dice:** {_metrics['iou']:.3f} / {_metrics['dice']:.3f} · **connected regions:** {seg_result['your_regions']} (SciPy {seg_result['scipy_regions']}) · **reference objects (all classes):** {seg_result['reference_instances']}"),mo.md('Compare separate → touching. Connectivity can merge two objects even when foreground overlap is high. Region IDs are not classes; the learned-center panel uses all foreground classes.')])
    _view
    return (seg_result,)


@app.cell(hide_code=True)
def _(display, mo, seg_result):
    display.download(mo,'',{'report':seg_result},'day3-segmentation',label='Save trial')
    return


if __name__ == "__main__":
    app.run()

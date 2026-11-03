import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Magic Wand")


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path
    repo = next(p for p in Path(__file__).resolve().parents if (p/'cv_exercises').is_dir())
    sys.path.insert(0, str(repo))
    import marimo as mo
    import numpy as np
    from cv_exercises import data, display, segmentation_ui as ui
    import matplotlib.pyplot as plt
    from collections import deque
    task_api = (repo/'exercises/day1-block-03-segmentation/TASK_API.md').read_bytes()
    return data, deque, display, mo, np, plt, task_api, ui


@app.cell(hide_code=True)
def _(mo, task_api):
    mo.hstack([mo.md('# Magic wand'), mo.download(task_api, filename='segmentation-api.md', label='API reference (.md)')], align='center')
    return


@app.cell(hide_code=True)
def _(data):
    rgb = data.scene()
    color, position = data.pixel_inputs(rgb)
    return color, position, rgb


@app.cell(hide_code=True)
def _(rgb, ui):
    ui.scene_plot(rgb)
    return


@app.cell(hide_code=True)
def _(mo):
    spatial_weight = mo.ui.slider(0, 2, step=.1, value=0, show_value=True, label='Position weight')
    probe = mo.ui.dropdown({'A': (18, 12), 'B': (10, 36)}, value='A', label='Select from')
    spatial_weight
    return probe, spatial_weight


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1 · Build pixel features
    Combine appearance with location: $x_{r,c}=[R,G,B,wr',wc']$, with normalized coordinates r',c' and position weight w.

    `color: float(1536,3)`: RGB in [0,1]; `position: float(1536,2)`: normalized (row,col) in [0,1], top-left origin. Both use row-major pixel order for H=32,W=48. `spatial_weight.value`: float w=0..2. Return float `spatial_features: (1536,5)`, preserving order/inputs. `np.column_stack` joins columns.
    """)
    return


@app.cell
def _(color, np, position, spatial_weight):
    # TODO 1: build [R,G,B,w*row,w*col].
    spatial_features=None
    # END TODO 1
    return (spatial_features,)


@app.cell(hide_code=True)
def _(color, mo, spatial_features, ui):
    spatial_labels, _status = ui.feature_labels(spatial_features, color)
    mo.md(_status)
    return (spatial_labels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2 · Restore image coordinates
    Map one label per pixel back to the image: $M_{r,c}=\ell_{rW+c}$.

    `rgb: float(H,W,3)`, `spatial_labels`: integer `(H*W,)` IDs 0..2, or None while fitting waits. Return integer `spatial_mask: (H,W)` in the same row-major order, otherwise None. Positive dimensions are guaranteed; preserve inputs. `.reshape(shape)` restores axes.
    """)
    return


@app.cell
def _(rgb, spatial_labels):
    # TODO 2: reshape ready labels; keep None while waiting.
    spatial_mask=None
    # END TODO 2
    return (spatial_mask,)


@app.cell(hide_code=True)
def _(mo, rgb, spatial_labels, spatial_mask, ui):
    ready_mask, _status = ui.mask_feedback(spatial_mask, spatial_labels, rgb)
    mo.md(_status)
    return (ready_mask,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3 · Select matching pixels
    Select every pixel with the seed's cluster ID, including disconnected patches.

    `select_region(mask,row,col)`: None or integer mask `(H,W)`; valid integer seed coordinates down/right from top left. Return None while waiting, otherwise bool `(H,W)`; True means equal seed ID. Preserve mask. One group selects the whole image; `a[r,c]` reads the seed value.
    """)
    return


@app.cell
def _():
    def select_region(mask,row,col):
        # TODO 3: select all pixels sharing the seed's group.
        selected=None
        # END TODO 3
        return selected
    return (select_region,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4 · Keep only the connected patch
    Implement a flood fill: start at the seed and visit matching neighbors up, down, left and right. Diagonal contact alone does not connect pixels.

    `connected_region(mask,row,col)`: same inputs as step 3. Return None while waiting, otherwise a NEW bool `(H,W)` containing exactly the seed's 4-connected same-ID component. Preserve mask; handle edges and one-pixel regions. `np` and `deque` are supplied; `deque([item])`, `.popleft()` and `.append(item)` manage a queue.
    """)
    return


@app.cell
def _(deque, np):
    def connected_region(mask,row,col):
        # TODO 4: visit only in-bounds, unvisited, same-ID neighbors.
        connected=None
        # END TODO 4
        return connected
    return (connected_region,)


@app.cell(hide_code=True)
def _(connected_region, mo, np, plt, probe, ready_mask, rgb, select_region):
    selection=connected=None
    _view=mo.md('Waiting: complete the selection functions.')
    if ready_mask is not None:
        selection=select_region(ready_mask,*probe.value)
        connected=connected_region(ready_mask,*probe.value)
        _valid=lambda a:isinstance(a,np.ndarray) and a.shape==rgb.shape[:2] and a.dtype==bool and a[probe.value]
        if _valid(selection) and _valid(connected):
            _fig,_axes=plt.subplots(1,3,figsize=(9,2.8),layout='constrained')
            for _ax,_image,_title in zip(_axes,[rgb,np.where(selection[...,None],rgb,rgb*.15),np.where(connected[...,None],rgb,rgb*.15)],['Original','All matching pixels','Connected patch']):
                _ax.imshow(_image);_ax.plot(probe.value[1],probe.value[0],'wo',markeredgecolor='black');_ax.axis('off');_ax.set_title(_title)
            _view=mo.vstack([_fig,mo.md(f'**Matching:** {int(selection.sum())} pixels · **connected:** {int(connected.sum())} pixels. A/B share ID: {bool(ready_mask[18,12]==ready_mask[10,36])}.')])
        elif selection is not None or connected is not None:
            _view=mo.md('Steps 3–4: return boolean (H,W) arrays including the seed, or None while waiting.')
    mo.vstack([probe,_view])
    return connected, selection


@app.cell(hide_code=True)
def _(connected, display, mo, probe, ready_mask, selection, spatial_weight):
    mo.vstack([mo.md('Compare weights **0 → 0.5 → 1** with K=3/seed=42 fixed. Inspect both seeds. Connectivity separates patches; it does not recognize objects.'),display.download(mo,'',{'position_weight':spatial_weight.value,'k':3,'seed':42,'probe_row_col':list(probe.value),'mask':None if ready_mask is None else ready_mask.tolist(),'selection':None if selection is None else selection.tolist(),'connected':None if connected is None else connected.tolist()},'segmentation-trial',label='Save trial')])
    return


if __name__ == "__main__":
    app.run()

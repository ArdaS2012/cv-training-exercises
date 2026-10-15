import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="Inspect a Small CNN and VGG")


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

    **15 minutes in pairs after guided setup.** Trace a numerical convolution and distinguish architecture geometry from semantic interpretation.

    1. Predict a filter response and stride-1 output shape. Complete both TODOs and check one patch response by hand, including the bias.

    2. Keep image, kernel and padding fixed; compare stride 1 and 2. Verify the predicted shape against the computed map and explain what spatial sampling changed.

    3. Compare one tiny-CNN stage with the supplied pretrained VGG trace. Identify channel/spatial changes and explain why an activation is not proof of a named object.

    **Bring back:** Checked response, predicted/measured shape table, fixed-stride comparison and a feature-interpretation caveat.

    Swap editor/checker halfway. Record in the observation fields, save code and download findings. The complete task and setup are in this block’s `GROUP_TASK.md`. Prerequisite TODOs discussed in plenary belong to setup.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Inspect a Small CNN and VGG

    15-minute group task after guided setup. Calculate a filter response and shape, inspect actual learned feature maps, and compare tiny-CNN versus VGG feature traces.

    Predict → complete the short TODOs → change one control → inspect → explain → download.

    Edit only `# TODO` regions; run with Shift+Enter. Save Python with Ctrl+S / Command+S. Download findings to retain current controls, measurements and observations. The untouched starter shows friendly waiting messages. Work in pairs and swap who edits. Prepared code and trained assets are supplied; no API key or GPU.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    image_index=mo.ui.slider(0,299,value=0,label='Fixed image row')
    stride=mo.ui.dropdown(options=[1,2],value=1,label='Test convolution stride')
    prediction=mo.ui.text_area(label='Predict the output shape and a ReLU/pooling change',full_width=True)
    mo.vstack([mo.hstack([image_index,stride],wrap=True),prediction])
    return image_index, prediction, stride


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## TODO 1 — one patch response; TODO 2 — spatial output count
    `patch` and `kernel` are matching 3×3 NumPy arrays; `bias` is a scalar. Return one weighted sum including bias. `height`, `kernel_size`, `padding`, `stride` are positive integer geometry settings (padding may be zero). Return the integer number of fully fitting windows for unit dilation; use floor and include the first window. These are small conceptual steps, not a whole training loop.

    After completion inspect the actual tiny-CNN maps, test stride 1/2, export the shape table, then compare the genuine pretrained VGG partial trace with its measured full trace. The digit is outside ImageNet semantics; a bright map is not a known object label.
    """)
    return


@app.cell
def _():
    import numpy as np
    def patch_response(patch,kernel,bias):
        # TODO 1: combine every aligned product and the bias into one scalar.
        response=None
        # END TODO 1
        return response
    def output_size(height,kernel_size,padding,stride):
        # TODO 2: count complete windows, including the first, with unit dilation.
        size=None
        # END TODO 2
        return size

    return output_size, patch_response


@app.cell(hide_code=True)
def _(day2, image_index, mo, output_size, patch_response, stride):
    cnn_result=None
    _message=None
    try:cnn_result=day2.cnn_inspect(patch_response,output_size,image_index.value,stride.value)
    except (ValueError,TypeError,IndexError) as _error:_message=str(_error)
    mo.callout(_message,kind='warn') if _message else mo.callout('Complete patch_response and output_size to inspect maps.',kind='info') if cnn_result is None else mo.md('Actual pretrained maps and shape traces are ready below.')
    return (cnn_result,)


@app.cell(hide_code=True)
def _(cnn_result, mo):
    import io
    import matplotlib.pyplot as plt
    _view=None
    if cnn_result is not None:
        _report,_figure=cnn_result
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
def _(cnn_result, display, image_index, mo, observations, prediction, stride):
    display.download(mo,observations.value,{'prediction':prediction.value,'row':image_index.value,'stride':stride.value,'report':cnn_result[0] if cnn_result else None},'cnn-vgg-findings')
    return


if __name__ == "__main__":
    app.run()

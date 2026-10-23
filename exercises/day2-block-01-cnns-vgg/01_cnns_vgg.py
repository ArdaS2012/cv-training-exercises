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
    ## Convolution detective: where did the pixels go?

    **30 minutes in pairs.** A response is a weighted sum plus bias. Stride changes sampling, not learned weights. ReLU clips negatives; pooling loses positions. Feature channels are not class labels.

    ### Investigate, do not just run

    1. Predict one patch response, including bias, and the stride-1/stride-2 shapes before running. Mark a thin stroke in fixed digit row 0 that could become harder to locate.

    2. Complete the response and output-size TODOs. Check one patch by hand, then inspect the actual learned maps. Explain the first disagreement between your prediction and the measured trace.

    3. Keep image, kernel and padding fixed. Run stride 1 then 2; record predicted/measured dimensions and point to a visible change in spatial detail. Swap editor/checker and explain which operation caused it.

    4. Keep stride fixed and choose one different digit row. Test whether the same channel lights up in the same place. Compare the supplied VGG trace: name one channel/spatial change, without assigning a human object name to a channel.

    5. Decide which sampling setting you would start with if preserving a small stroke mattered. Support it with one shape measurement and one marked visual feature; state why these two digits cannot establish classifier accuracy.

    **Evidence log:** row | stride | predicted H×W | measured H×W | map/stroke evidence. Save a run before changing controls. Export both the current findings and your comparison log.

    0–4 min: read the takeaways, inspect inputs and predict. 4–12: implement the bounded TODOs and inspect a real result. 12–22: compare controlled runs; swap editor/checker at minute 15. 22–27: diagnose a stress case and choose. 27–30: agree, save and export evidence.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Inspect a Small CNN and VGG

    30-minute investigation after guided setup. Calculate a filter response and shape, inspect actual learned feature maps, and compare tiny-CNN versus VGG feature traces.

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
    mo.md(r"""
    ### Read the maps like an investigator
    A convolution multiplies aligned image/kernel entries, adds them and adds one bias: $y=\sum_i x_i k_i+b$. Inputs have matching shape; the output is one scalar response, not a probability. For one spatial axis, $H'=\lfloor(H+2P-K)/S\rfloor+1$: $H$ is input size, $P$ zero padding per side, $K$ kernel size and $S$ stride. Floor keeps only complete windows. Width follows the same rule.

    ReLU keeps the shape and replaces negative responses with zero. Pooling summarizes local windows and can remove a narrow stroke. Compare the SAME channel and color scale before naming a change; bright means a larger response, not that a known object has been detected. Use the labelled isolated stride probe: it and the fixed conv1 panel share a color scale. The full pretrained network trace retains its original stride. Changing the probe changes sampling geometry; it does not retrain the tiny model or VGG.
    """)
    return


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
    experiment_log=mo.ui.text_area(value="row | stride | predicted H\u00d7W | measured H\u00d7W | map/stroke evidence\nA | \nB | \nStress case | \nDecision + limitation | ",label='Comparison log — preserve run A before changing a control',full_width=True)
    mo.vstack([mo.md("### Keep your evidence while experimenting\nEach rerun replaces the current result. Copy measured values and name a visible feature or response field here before changing controls."),experiment_log])
    return (experiment_log,)


@app.cell(hide_code=True)
def _(display, experiment_log, mo):
    display.download(mo,experiment_log.value,{'columns':"row | stride | predicted H\u00d7W | measured H\u00d7W | map/stroke evidence"},'day2-block1-comparison-log',label='Download comparison log (baseline + changed runs)')
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

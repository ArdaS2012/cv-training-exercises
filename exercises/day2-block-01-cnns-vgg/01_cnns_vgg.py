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

    **Around 30 minutes in pairs.** A response is a weighted sum plus bias. Stride changes sampling, not learned weights. ReLU clips negatives; pooling loses positions. Feature channels are not class labels.

    ### Investigate, do not just run

    1. Predict one patch response, including bias, and the stride-1/stride-2 shapes before running. Mark a thin stroke in fixed digit row 0 that could become harder to locate.

    2. Complete the response and output-size TODOs. Check one patch by hand, then inspect the actual learned maps. Explain the first disagreement between your prediction and the measured trace.

    3. Keep image, kernel and padding fixed. Run stride 1 then 2; record predicted/measured dimensions and point to a visible change in spatial detail. Swap editor/checker and explain which operation caused it.

    4. Keep stride fixed and choose one different digit row. Test whether the same channel lights up in the same place. Compare the supplied VGG trace: name one channel/spatial change, without assigning a human object name to a channel.

    5. Decide which sampling setting you would start with if preserving a small stroke mattered. Support it with one shape measurement and one marked visual feature; state why these two digits cannot establish classifier accuracy.

    **Evidence log:** row | stride | predicted H×W | measured H×W | map/stroke evidence. Save a run before changing controls. Export both the current findings and your comparison log.
    """)
    return






@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Inspect a Small CNN and VGG

    Investigation after guided setup. Calculate a filter response and shape, inspect actual learned feature maps, and compare tiny-CNN versus VGG feature traces.

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
    mo.md(r"""
    ## Your implementation task

    Complete the two small functions below. `patch_response` computes one learned
    filter's response for a supplied image window. `output_size` predicts the count
    of valid window positions on one spatial axis. The detailed guide below defines
    every argument, the formulas, available operations and how the runner reads your
    results. Images, trained weights, imports and plots are already prepared.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TODO guide — calculate a response, then count positions

    The prepared runner supplies images, learned weights and plots. Your two functions
    answer different questions: **what number does one filter produce at one position?**
    and **how many positions fit along an image axis?** Neither function trains a model.

    ### TODO 1: `patch_response(patch, kernel, bias)`

    A **patch** is a small rectangular window of image values. A **kernel** contains
    one learned weight for each position in that window. A **bias** is one extra offset
    for the whole response. Positive and negative weights can reinforce or cancel
    image values; the result is a raw activation, so a negative response is valid.

    | Argument | Type, structure and meaning |
    | --- | --- |
    | `patch` | Nonempty numeric NumPy array of shape `(R, C)`. Axis 0 is row/y; axis 1 is column/x. The live runner supplies a `(3, 3)` `float32` window from one grayscale digit, with intensities in `[0, 1]`. |
    | `kernel` | Numeric NumPy array with exactly the same shape. `kernel[r, c]` weights `patch[r, c]`. Entries are finite, may be negative and are not intensities. The live runner supplies the first learned filter for one input/output channel. |
    | `bias` | One finite numeric scalar, supplied as a Python float by the runner. It is added once, not once per pixel. |

    Use the defining operation

    $$y=\sum_{r=0}^{R-1}\sum_{c=0}^{C-1}x_{r,c}w_{r,c}+b.$$

    Here `x` denotes `patch`, `w` denotes `kernel`, `b` is `bias`, and `y` is the one
    response. Both sums together include every aligned coordinate. This is the
    cross-correlation convention used by PyTorch: do not reverse the kernel.
    Return one finite Python/NumPy numeric scalar; no row, column or channel axis
    remains. Keep both input arrays unchanged. Use their supplied size rather than
    assuming exactly nine entries: a small hand check may use another matching shape
    and arbitrary finite numbers. Mismatched shapes and empty arrays are outside the
    supplied task domain; no defensive validation is required.

    **NumPy toolbox:** `a * b` on matching arrays multiplies entries at the same
    coordinates; `a @ b` performs matrix multiplication and has a different meaning.
    `np.sum(a, axis=None)` adds every entry and returns a scalar. With `axis=0` or
    `axis=1`, it instead leaves values for the unreduced axis. `np` is already imported
    in the editable cell. Choose how to combine the documented operations yourself.

    ### TODO 2: `output_size(height, kernel_size, padding, stride)`

    This function counts output positions along **one spatial axis**. It receives
    plain Python integer settings, not an image, array shape tuple or widget object.

    | Argument | Valid domain and units |
    | --- | --- |
    | `height` | Positive integer: unpadded input length, in pixels. The live digit length is `8`. The same function can be used for width. |
    | `kernel_size` | Positive integer: window length on this axis, in pixels. The runner uses `3`. Dilation (spacing inside the kernel) is fixed at `1`. |
    | `padding` | Nonnegative integer: number of zero-filled pixels added on **each** side. The runner uses `1`, so the total added length is twice this value. |
    | `stride` | Positive integer: distance between consecutive window starts, in input pixels. The widget selects `1` or `2`; the definition also applies to other positive integer strides. |

    For these settings the padded input is guaranteed to fit at least one window.
    The defining count is

    $$H_{out}=\left\lfloor\frac{H+2P-K}{S}\right\rfloor+1.$$

    `H`, `P`, `K`, `S` stand for the four arguments in the table. The numerator is
    how far a window can move after its first valid placement. Dividing by stride
    counts moves; floor discards a final incomplete move, and the extra `1` counts
    the first position. Return one positive Python `int`, not a float, shape tuple or
    channel count. `//` is Python floor division for these nonnegative geometry
    counts. A stride need not divide the available distance exactly. No batch or
    channel size enters this spatial count.

    ### Before running and reading the evidence

    Replace only `response=None` and `size=None` inside their marked TODO regions;
    keep both signatures and return lines. Run the cell with **Shift+Enter**. `None`
    means unfinished and leaves the plots waiting. The prepared runner calls both
    functions with the current settings; a finite scalar/positive size alone does
    not establish that your arithmetic is correct. Hand-check aligned products and
    count complete windows before trusting the plot.

    In the JSON, `patch_response` is your scalar, `predicted_spatial_size` is your
    one-axis count, and `actual_stride_shape` is the measured shape in
    `[batch, channels, height, width]` order. `tiny_trace` and `vgg_partial_trace` contain
    `stage` strings and `shape` integer lists in that same tensor convention; class
    `scores` have `[batch, classes]` instead. Compare your spatial count with both
    last entries of the isolated stride shape. The full pretrained CNN trace keeps
    its original stride; only the labelled isolated probe uses the chosen stride.
    Inspect the same feature channel and shared color scale when discussing a lost
    stroke. Channel number is not a known object category, and VGG's digit preview
    is outside its original ImageNet image domain.

    **Version-matched official references:** [NumPy 2.2 sum](https://numpy.org/doc/2.2/reference/generated/numpy.sum.html),
    [NumPy array arithmetic](https://numpy.org/doc/2.2/user/quickstart.html#basic-operations),
    [Python 3.12 arithmetic](https://docs.python.org/3.12/reference/expressions.html#binary-arithmetic-operations),
    [PyTorch 2.8 Conv2d shape and cross-correlation](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Conv2d.html).
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

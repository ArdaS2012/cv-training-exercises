import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="Inspect Compression and Denoising")


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
    ## Restoration lab: cleaner pixels or lost detail?

    **Around 30 minutes in pairs.** Encoder → latent code → decoder reconstructs pixels. MSE averages squared errors in matching units. Denoising uses a clean target. Low average error can hide a lost stroke; checkpoint comparisons are descriptive.

    ### Investigate, do not just run

    1. At row 0 and noise 0.25, choose a visible stroke/edge you want to preserve. Predict what changing latent width 4→16 will do to that detail and clean-target error.

    2. Complete encode/decode and pixel-MSE TODOs. Check the averaging convention with a tiny hand example; inspect clean input, noisy image, reconstruction and error map together.

    3. Compare latent widths 4 and 16 with the SAME row/noise. Record MSE and annotate the chosen detail in each output; compare both clean-reconstruction and denoising checkpoints shown by the notebook.

    4. Swap roles, freeze width, and change only noise (0.25→0.45). Inspect whether smoothing removes noise and useful detail together. Use a second digit only after saving this fixed-image comparison.

    5. Choose a checkpoint for preserving the selected detail under noise. Defend with both measured error and visible evidence, including any disagreement. Explain why different training histories prevent a pure causal claim about latent width.

    **Evidence log:** row | noise | width/objective | clean-target MSE | preserved/lost detail. Save a run before changing controls. Export both the current findings and your comparison log.
    """)
    return






@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Inspect Compression and Denoising

    Investigation after guided setup. Compare genuine autoencoder checkpoints at two bottleneck widths, fixed-noise reconstructions, per-pixel errors and lost detail.

    Predict → complete the short TODOs → change one control → inspect → explain → download.

    Edit only `# TODO` regions; run with Shift+Enter. Save Python with Ctrl+S / Command+S. Download findings to retain current controls, measurements and observations. The untouched starter shows friendly waiting messages. Work in pairs and swap who edits. Prepared code and trained assets are supplied; no API key or GPU.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    image_row=mo.ui.slider(0,299,value=0,label='Same image row')
    latent=mo.ui.dropdown(options=[4,16],value=4,label='Latent coordinates')
    noise=mo.ui.slider(0,.6,step=.05,value=.25,label='Input noise standard deviation')
    prediction=mo.ui.text_area(label='Predict detail preservation and denoising error',full_width=True)
    mo.vstack([mo.hstack([image_row,latent,noise],wrap=True),prediction])
    return image_row, latent, noise, prediction


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Read the restoration panel
    The encoder makes code $z=f(x)$; the decoder reconstructs $\hat x=g(z)$. The latent width is the number of code coordinates, not the number of classes. Per-pixel MSE is $\frac1d\sum_{j=1}^d(x_j-\hat x_j)^2$, where $d$ is the number of compared pixels in the same intensity convention. Compare against the clean target even when the input is noisy.

    Averaging can hide a small lost stroke because many easy background pixels dominate the score. Locate your chosen detail before reading MSE; then use both sources of evidence. Changing noise with the same image and checkpoint isolates input difficulty. Width comparisons use separately trained checkpoints, so their histories prevent a pure width-only causal conclusion.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Your implementation task

    Connect the provided encoder and decoder for the noisy input, preserving the
    batch/pixel shape. Then measure average squared coordinate error between a clean
    reference and a reconstruction. The guide below distinguishes tensors from
    NumPy arrays, defines both module interfaces and explains exactly what is
    averaged. Loading, trained weights, noise generation and comparison plots are
    already supplied.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TODO guide — connect the autoencoder and measure reconstruction error

    An **encoder** maps the input pixels to a shorter vector, the **latent code**.
    A **decoder** maps that code back to estimated pixels, the **reconstruction**.
    The models already have trained weights. Your first function connects their
    provided interfaces; your second gives one average error against a clean target.
    Neither TODO trains a model or removes noise by a hand-written image filter.

    ### TODO 1: `reconstruct(model, noisy)`

    | Argument | Type, shape and meaning |
    | --- | --- |
    | `model` | Supplied pretrained PyTorch `nn.Module`, already loaded on CPU and set to evaluation mode. It has callable attributes `encoder` and `decoder`; they are not dictionary keys. The selected latent width `k` is `4` or `16`. |
    | `noisy` | CPU `torch.Tensor`, dtype `torch.float32`, shape `(1, 64)`: one normalized noisy grayscale digit. Axis 0 is the one-image batch; axis 1 is flattened pixel position. All values are finite and clipped to `[0, 1]`. |

    Flattening is **row-major**: the first eight values are the first row of the
    original `8 × 8` image, then the second row, and so on. There is no separate
    channel axis in this function. Preserve the one-image batch axis throughout.

    The supplied local model API is:

    | Callable | Accepted input | Returned value |
    | --- | --- | --- |
    | `model.encoder(values)` | CPU float32 tensor with shape `(1, 64)` in normalized intensity units. | CPU float32 tensor with shape `(1, k)`: latent features. These coordinates need not be in `[0, 1]` and are not class scores. |
    | `model.decoder(code)` | The encoder's CPU float32 tensor with shape `(1, k)`. | CPU float32 tensor with shape `(1, 64)`: reconstructed intensities in `[0, 1]`, due to the supplied final sigmoid layer. |

    A module is a callable object: generic syntax `layer(values)` runs a supplied
    layer's forward computation. Accessing an attribute without calling it gives the
    module object, not its output. The defining route is

    $$z=f(x_{noisy}),\qquad \hat x=g(z).$$

    `f` is the encoder, `g` the decoder, `z` the `k`-coordinate code, and `x_hat` the
    reconstruction. Return that reconstruction as a CPU tensor of shape `(1, 64)`.
    Do not return the code, the model object, a NumPy array or a reshaped `8 × 8`
    image. Keep the input and weights unchanged. The runner already disables gradient
    recording; device movement, fitting, `.backward()` and `.detach()` are not part of
    this TODO. Different latent widths are supplied through the model, so avoid
    hard-coding the code length.

    ### TODO 2: `pixel_mse(clean, reconstruction)`

    This function receives **NumPy arrays**, not the tensors passed to TODO 1. The
    runner performs that conversion for you after inference. Both arrays have the
    same nonempty shape and corresponding finite numeric entries. In the live run
    they are float32 arrays of shape `(1, 64)` in normalized intensity units `[0, 1]`;
    a small arithmetic check may use a one-dimensional or smaller matching array.

    `clean` is the original reference before corruption. `reconstruction` is the
    model's output for the noisy input. Compare each reconstruction against `clean`,
    not against the corrupted input. No element may be reordered or changed.

    The **mean squared error** (MSE) is

    $$\mathrm{MSE}=\frac{1}{d}\sum_{j=1}^{d}(x_j-\hat x_j)^2.$$

    `d` is the total number of compared entries (64 in one live image), `x_j` is a
    clean pixel and `x_hat_j` the reconstruction at the same position. Squaring makes
    positive/negative errors contribute positively; the mean divides by the number
    of entries. Return one finite nonnegative Python/NumPy numeric scalar, in
    **normalized intensity squared** units. Identical arrays have MSE zero. A
    coordinate error map retains positions; this function reduces it to one number.
    There are no missing pixels, empty arrays or shape mismatches in the supplied
    domain, so no masking or defensive input policy is required.

    **NumPy toolbox:** matching-array subtraction and `**` exponentiation act on
    corresponding entries. `np.mean(a, axis=None)` averages all entries and returns a
    scalar; an explicit axis can leave a vector. `np` is already imported in the
    editable cell. No tensor conversion or file loading belongs in this function.

    ### Before running and reading the evidence

    Replace only `output=None` and `error=None` in their marked regions, keeping
    function signatures/return lines. Rerun with **Shift+Enter**. The prepared runner
    calls your reconstruction for both `clean` and `denoise` checkpoints at the
    selected width, then calls your metric on each result. `None` leaves a waiting
    message. A shape/finite-value check is a guard, not proof that the right arrays
    were compared: verify the mean convention on your small hand example.

    The clean-trained checkpoint learned clean-input/clean-target reconstruction;
    the denoising checkpoint learned noisy-input/clean-target reconstruction with
    training noise standard deviation `0.25`. Both receive the current noisy image at
    inference. The same row and random seed keep the corruption controlled. Changing
    the **noise standard deviation** scales noise in normalized-intensity units;
    it does not update model weights. Width changes select separately trained models.

    In the report, `clean_target_mse` maps `clean` and `denoise` to your scalar errors;
    `noisy_input_mse` gives the corrupted-input error against the same clean target.
    `latent_width` counts code coordinates and `coordinate_ratio` is `64/k`, a
    coordinate-count ratio, not a compressed file size or bitrate. Inspect a chosen
    stroke in the images/error maps alongside the scalar: background pixels can hide
    a lost detail in an average. Distinct checkpoint histories limit causal claims
    about width alone.

    **Version-matched official references:** [PyTorch 2.8 Module calls](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Module.html),
    [PyTorch 2.8 Sigmoid](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Sigmoid.html),
    [NumPy 2.2 mean](https://numpy.org/doc/2.2/reference/generated/numpy.mean.html),
    [NumPy array arithmetic](https://numpy.org/doc/2.2/user/quickstart.html#basic-operations).
    """)
    return



@app.cell
def _():
    import numpy as np
    def reconstruct(model,noisy):
        # TODO 1: encode the noisy input, then decode the resulting code.
        output=None
        # END TODO 1
        return output
    def pixel_mse(clean,reconstruction):
        # TODO 2: mean squared error over all matching pixel coordinates.
        error=None
        # END TODO 2
        return error

    return pixel_mse, reconstruct


@app.cell(hide_code=True)
def _(day2, image_row, latent, mo, noise, pixel_mse, reconstruct):
    ae_result=None
    _message=None
    try:ae_result=day2.ae_compare(reconstruct,pixel_mse,image_row.value,latent.value,noise.value)
    except (ValueError,TypeError,IndexError) as _error:_message=str(_error)
    mo.callout(_message,kind='warn') if _message else mo.callout('Complete reconstruction and MSE to inspect outputs.',kind='info') if ae_result is None else mo.md('Same clean target and seeded noise; compare measured errors and details below.')
    return (ae_result,)


@app.cell(hide_code=True)
def _(ae_result, mo):
    import io
    import matplotlib.pyplot as plt
    _view=None
    if ae_result is not None:
        _report,_figure=ae_result
        _buffer=io.BytesIO();_figure.savefig(_buffer,format='png',dpi=150,bbox_inches='tight')
        _view=mo.vstack([_figure,mo.download(_buffer.getvalue(),filename='comparison.png',label='Download figure'),mo.md('```json\n'+__import__('json').dumps(_report,indent=2)+'\n```')])
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    experiment_log=mo.ui.text_area(value="row | noise | width/objective | clean-target MSE | preserved/lost detail\nA | \nB | \nStress case | \nDecision + limitation | ",label='Comparison log — preserve run A before changing a control',full_width=True)
    mo.vstack([mo.md("### Keep your evidence while experimenting\nEach rerun replaces the current result. Copy measured values and name a visible feature or response field here before changing controls."),experiment_log])
    return (experiment_log,)


@app.cell(hide_code=True)
def _(display, experiment_log, mo):
    display.download(mo,experiment_log.value,{'columns':"row | noise | width/objective | clean-target MSE | preserved/lost detail"},'day2-block4-comparison-log',label='Download comparison log (baseline + changed runs)')
    return


@app.cell(hide_code=True)
def _(mo):
    observations=mo.ui.text_area(label='Findings: measured change, explanation and one limitation',full_width=True)
    observations
    return (observations,)


@app.cell(hide_code=True)
def _(
    ae_result,
    display,
    image_row,
    latent,
    mo,
    noise,
    observations,
    prediction,
):
    display.download(mo,observations.value,{'prediction':prediction.value,'row':image_row.value,'latent':latent.value,'noise':noise.value,'report':ae_result[0] if ae_result else None},'autoencoder-findings')
    return


if __name__ == "__main__":
    app.run()

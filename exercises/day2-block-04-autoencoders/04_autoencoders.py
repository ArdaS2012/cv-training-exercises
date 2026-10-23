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

    **30 minutes in pairs.** Encoder → latent code → decoder reconstructs pixels. MSE averages squared errors in matching units. Denoising uses a clean target. Low average error can hide a lost stroke; checkpoint comparisons are descriptive.

    ### Investigate, do not just run

    1. At row 0 and noise 0.25, choose a visible stroke/edge you want to preserve. Predict what changing latent width 4→16 will do to that detail and clean-target error.

    2. Complete encode/decode and pixel-MSE TODOs. Check the averaging convention with a tiny hand example; inspect clean input, noisy image, reconstruction and error map together.

    3. Compare latent widths 4 and 16 with the SAME row/noise. Record MSE and annotate the chosen detail in each output; compare both clean-reconstruction and denoising checkpoints shown by the notebook.

    4. Swap roles, freeze width, and change only noise (0.25→0.45). Inspect whether smoothing removes noise and useful detail together. Use a second digit only after saving this fixed-image comparison.

    5. Choose a checkpoint for preserving the selected detail under noise. Defend with both measured error and visible evidence, including any disagreement. Explain why different training histories prevent a pure causal claim about latent width.

    **Evidence log:** row | noise | width/objective | clean-target MSE | preserved/lost detail. Save a run before changing controls. Export both the current findings and your comparison log.

    0–4 min: read the takeaways, inspect inputs and predict. 4–12: implement the bounded TODOs and inspect a real result. 12–22: compare controlled runs; swap editor/checker at minute 15. 22–27: diagnose a stress case and choose. 27–30: agree, save and export evidence.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Inspect Compression and Denoising

    30-minute investigation after guided setup. Compare genuine autoencoder checkpoints at two bottleneck widths, fixed-noise reconstructions, per-pixel errors and lost detail.

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
    mo.md("""
    ## TODO — reconstruction and per-pixel MSE
    `model.encoder` maps a 1×64 tensor to its latent code; `model.decoder` maps that code to a reconstructed 1×64 tensor. In `reconstruct`, connect the two supplied modules for the noisy input. In `pixel_mse`, compare matching NumPy arrays against the clean target and return their mean squared coordinate error. Keep `np` available in this cell.

    Compare latent widths 4/16 on the same image and the same seeded corruption. Four genuine pretrained checkpoints are supplied: clean and denoising objectives at both widths. Denoisers learned noisy-input/clean-target reconstruction with σ=0.25. Changing the inference noise does not retrain them. Export originals/reconstructions and note one lost detail or overly smoothed region. The latent coordinate ratio is not a file-compression claim.
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

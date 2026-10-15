import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="Adapt Save Reload and Serve")


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(next(p for p in Path(__file__).resolve().parents if (p/'cv_exercises').is_dir())))
    import marimo as mo
    import numpy as np
    import json
    from cv_exercises import day3,display

    return day3, display, json, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Exercise + breakout: your shared task

    **15 minutes in pairs after guided setup.** Verify an adapted model’s inference contract through reload and serving rather than merely saving weights.

    1. Predict which parameter subset changes. Complete the adaptation config with the declared target order and rate, then run adapt/save/reload/local request at rate 0.03.

    2. Check before/after validation, frozen-group deltas, original versus reloaded score vectors and the actual HTTP response. Identify the evidence for identical preprocessing and label interpretation.

    3. Specify a concrete failure if label order or preprocessing were omitted. Describe how the observed local evidence would detect it and what extra checks a hosted service would require.

    **Bring back:** Validation comparison, frozen-weight check, reproducible package and recorded local request/response with a deployment failure analysis.

    Swap editor/checker halfway. Record in the observation fields, save code and download findings. The complete task and setup are in this block’s `GROUP_TASK.md`. Prerequisite TODOs discussed in plenary belong to setup.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Adapt Save Reload and Serve

    15-minute group task after guided setup. Fit a small target head under an explicit brightness/contrast shift, export validation/package metadata and witness a real local request/response.

    Predict → complete short TODOs → change one setting → inspect → explain → download.

    Work in pairs and swap editor/observer. Edit only TODO regions; Shift+Enter runs cells. Ctrl+S / Command+S saves code. Download findings to retain control values and observations. The fresh starter waits politely; prepared loading/plotting/serving code needs no edits. No GPU or API key.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    adapt_rate=mo.ui.slider(.01,1.,step=.01,value=.03,label='Head learning rate')
    run_package=mo.ui.run_button(label='Adapt, save, reload and send local HTTP request')
    prediction=mo.ui.text_area(label='Predict which parameters change and what must survive reload',full_width=True)
    mo.vstack([prediction,adapt_rate,run_package])
    return adapt_rate, prediction, run_package


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## TODO — bounded adaptation and label metadata
    `adaptation_config(rate)` returns a dictionary with `trainable` (the head-only strategy), `learning_rate` (the passed finite rate) and `id2label` (string keys0/1 mapped to the stated even/odd target labels in order).
    The model outputs48 features and the new head has2 logits. Keep the pretrained convolutional backbone fixed. Prepared Adam training uses180 target rows;60 validation rows compare both source-like and brightness-shifted images;60 test rows stay unused.
    The prepared pipeline declares8×8 grayscale raw pixels in0..16, divides by16, runs the saved model and interprets the logits. Inspect those transforms and the source revision in the package. The local HTTP server accepts POST `/predict` with `pixels` as an8×8 JSON array, then closes after the witnessed request.
    Before export, check frozen deltas, changed head, validation before/after, label order, score agreement and HTTP response. This is local deployment evidence; no weights are published and no paid hosted endpoint is provisioned.
    """)
    return


@app.function
def adaptation_config(rate):
    # TODO: trainable subset, passed rate and stated target label order.
    config=None
    # END TODO
    return config


@app.cell(hide_code=True)
def _(adapt_rate, day3, json, mo, run_package):
    package_result=None
    _package_error=None
    if run_package.value:
        try:package_result=day3.adapt_package(adaptation_config(adapt_rate.value))
        except (ValueError,TypeError,KeyError) as _e:_package_error=str(_e)
    mo.callout(_package_error,kind='warn') if _package_error else mo.callout('Complete the configuration TODO, then run the prepared pipeline.',kind='info') if package_result is None else mo.md('```json\n'+json.dumps(package_result[0],indent=2)+'\n```')
    return (package_result,)


@app.cell(hide_code=True)
def _(mo):
    observations=mo.ui.text_area(label='Findings: controlled change, measured result, failure explanation and limitation',full_width=True)
    observations
    return (observations,)


@app.cell(hide_code=True)
def _(display, mo, observations, package_result, prediction):
    _downloads=[display.download(mo,observations.value,{'prediction':prediction.value,'report':None if package_result is None else package_result[0]},'adaptation-deployment-findings')]
    if package_result is not None:_downloads.append(mo.download(package_result[1],filename='reproducible-vision-pipeline.zip',label='Download saved model/processor/labels/revision package'))
    mo.vstack(_downloads)
    return


if __name__ == "__main__":
    app.run()

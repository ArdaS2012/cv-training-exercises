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
    ## Deployment workshop: does the saved model still mean the same thing?

    **30 minutes in pairs.** Domain shift needs target validation. Head-only adaptation freezes the backbone. A package includes transforms, labels and revision. Hub storage differs from a running endpoint; reload and an actual request test the contract.

    ### Investigate, do not just run

    1. Predict which groups change during head-only adaptation and what must survive save/reload. Sketch pixels → transform → model → label before configuring the real pipeline.

    2. Complete the bounded adaptation/label configuration. At rate 0.03 run adapt/save/reload/local HTTP; inspect validation before/after and frozen versus head deltas.

    3. Inspect the package files and actual response. Match score indices to saved labels, compare original/reloaded vectors within tolerance, and trace the same raw 8×8 pixels through the declared normalization.

    4. Swap roles. Change ONLY learning rate to 0.10, rerun with fixed data/seed/budget, and compare validation and changed-head evidence. Keep the first package/report before rerunning; a worse result is useful evidence.

    5. Perform a label-interpretation fault drill using the actual response: on paper, swap the two displayed label names while leaving the vector fixed. Show which interpretation becomes wrong although numeric reload agreement still passes. Choose a package using validation, and explain the extra checks a hosted service would require.

    **Evidence log:** run/rate | validation before/after | frozen/head deltas | reload agreement | HTTP/label evidence. Save a run before changing controls. Export both the current findings and your comparison log.

    0–4 min: read the takeaways, inspect inputs and predict. 4–12: implement the bounded TODOs and inspect a real result. 12–22: compare controlled runs; swap editor/checker at minute 15. 22–27: diagnose a stress case and choose. 27–30: agree, save and export evidence.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Adapt Save Reload and Serve

    30-minute investigation after guided setup. Fit a small target head under an explicit brightness/contrast shift, export validation/package metadata and witness a real local request/response.

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
    mo.md(r"""
    ### Treat the package as a prediction contract
    The actual route is raw pixels → saved preprocessing → adapted model → scores → saved label order. Here raw images are 8×8 grayscale values in 0..16; the declared transform divides by 16. The backbone has 48 features and the target head emits two even/odd logits. Head-only adaptation permits head updates while frozen groups have zero deltas.

    Compare target validation before/after fitting, then reload with the same pixels and metadata. Equal score vectors within tolerance test numeric reproduction. Label order is a separate semantic check: swapping label names can leave every number equal while changing what the response means. The real local HTTP request tests a running local contract; storing artifacts on the Hub and operating a hosted service are distinct steps.
    """)
    return


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
    experiment_log=mo.ui.text_area(value="run/rate | validation before/after | frozen/head deltas | reload agreement | HTTP/label evidence\nA | \nB | \nStress case | \nDecision + limitation | ",label='Comparison log — preserve run A before changing a control',full_width=True)
    mo.vstack([mo.md("### Keep your evidence while experimenting\nEach rerun replaces the current result. Copy measured values and name a visible feature or response field here before changing controls."),experiment_log])
    return (experiment_log,)


@app.cell(hide_code=True)
def _(display, experiment_log, mo):
    display.download(mo,experiment_log.value,{'columns':"run/rate | validation before/after | frozen/head deltas | reload agreement | HTTP/label evidence"},'day3-block5-comparison-log',label='Download comparison log (baseline + changed runs)')
    return


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

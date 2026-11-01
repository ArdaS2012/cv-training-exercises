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

    **Around 30 minutes in pairs.** Domain shift needs target validation. Head-only adaptation freezes the backbone. A package includes transforms, labels and revision. Hub storage differs from a running endpoint; reload and an actual request test the contract.

    ### Investigate, do not just run

    1. Predict which groups change during head-only adaptation and what must survive save/reload. Sketch pixels → transform → model → label before configuring the real pipeline.

    2. Complete the bounded adaptation/label configuration. At rate 0.03 run adapt/save/reload/local HTTP; inspect validation before/after and frozen versus head deltas.

    3. Inspect the package files and actual response. Match score indices to saved labels, compare original/reloaded vectors within tolerance, and trace the same raw 8×8 pixels through the declared normalization.

    4. Swap roles. Change ONLY learning rate to 0.10, rerun with fixed data/seed/budget, and compare validation and changed-head evidence. Keep the first package/report before rerunning; a worse result is useful evidence.

    5. Perform a label-interpretation fault drill using the actual response: on paper, swap the two displayed label names while leaving the vector fixed. Show which interpretation becomes wrong although numeric reload agreement still passes. Choose a package using validation, and explain the extra checks a hosted service would require.

    **Evidence log:** run/rate | validation before/after | frozen/head deltas | reload agreement | HTTP/label evidence. Save a run before changing controls. Export both the current findings and your comparison log.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Adapt Save Reload and Serve

    Investigation after guided setup. Fit a small target head under an explicit brightness/contrast shift, export validation/package metadata and witness a real local request/response.

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
    mo.md(r"""
    ## Your configuration step in the supplied deployment pipeline

    You return a small configuration describing which model group may learn, which learning rate to use, and how the output positions map to even/odd labels. Training, model files, save/reload and the local HTTP request are already supplied.

    Complete the function below, run its cell with **Shift+Enter**, then press **Adapt, save, reload and send local HTTP request**. Compare learning rates 0.03 and 0.10 with data, initialization and training budget fixed. Save the first report and package before rerunning; a later run replaces the current output.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TODO definitions and API reference

    ### First understand what the package must preserve

    A **backbone** produces reusable image features; a **head** converts those features into task scores. Here the pretrained convolutional backbone produces 48 features from an 8×8 grayscale image. The new head produces two **logits** (unnormalized scores) for the target **even/odd digit** task.

    **Head-only adaptation** freezes backbone parameters and permits updates only to the head. A frozen parameter keeps exactly its previous value. A trainable parameter is allowed to change; that does not imply every individual value must change. The **learning rate** controls the optimizer's update scale; it is not a number of epochs or an accuracy target. Validation measures the effect of a strategy; a larger learning rate does not guarantee an improvement.

    ### TODO 1 — configuration: `adaptation_config(rate)`

    **Input.** `rate` is one finite Python `float`, already read from the **Head learning rate** control. The UI offers `[0.01, 1.0]` in steps of `0.01`, starting at `0.03`; the underlying adapter accepts the wider interval `[0.001, 1.0]`. Use this argument on each call. It is not the slider object, a percentage, an epoch count or a hard-coded default. Valid finite inputs within that interval are supplied; writing input-validation code is outside this TODO.

    **Return.** A new ordinary Python dictionary with these required exact keys and types:

    | Exact key | Required value / interpretation |
    | --- | --- |
    | `trainable` | Python `str` naming the group to update: the adapter's literal `"head"`. It is not a Boolean and not a list of parameter names. |
    | `learning_rate` | The passed scalar `rate`, preserved as a float. A change of the slider must reach the prepared optimizer. |
    | `id2label` | Nested Python dictionary with **string** keys `"0"` and `"1"`; they map to the target-label strings `"even"` and `"odd"` respectively. These names are the task's fixed label definitions. |

    `id2label` means “output index to human-readable label.” Score position `0` means the label under key `"0"`; position `1` means the label under `"1"`. Dictionary insertion order does not define the model's class order; its keys do. Integer `0` and string `"0"` are different Python keys. Use strings because the saved JSON configuration and response interpretation use strings. No weights, images, model objects, optimizer objects or HTTP calls belong in your returned dictionary.

    **Python container vocabulary.** A dictionary associates a key with a value; a nested dictionary is simply another dictionary used as a value. Generic syntax unrelated to this task: `record = {"city": "Bonn", "counts": {"morning": 3}}`. Return the Python container, not a printed representation or a JSON string. The supplied `return config` already returns your variable; replace `config = None` only inside the marked region, keeping the function signature.

    ### What the supplied training runner does

    `day3.adapt_package(config, epochs=40)` takes your configuration dictionary and an optional positive integer update budget, normally `40`. Passing `None` returns `None` so the starter waits. A valid configuration returns `(report, package_bytes)`: a Python report dictionary and ZIP bytes for the download button. The notebook passes the selected rate to your function and then passes your result to this adapter. You do not call it inside your TODO.

    The adapter uses CPU PyTorch, fixed seed `123`, the supplied train/validation/test split, a new two-output head and the Adam optimizer. Only head parameters are optimized. The fixed target shift transforms normalized brightness `x` into `clip(0.45*x + 0.25, 0, 1)`. It fits 180 target-training rows, compares 60 validation rows, and leaves 60 test rows unused. Each run restarts from the same source checkpoint and seeded head; it does not continue training the previous run. The notebook's learning-rate experiment keeps the 40-step full-batch budget fixed.

    Let `theta_F` be the frozen backbone parameters and `theta_H` the head parameters. Head-only adaptation requires

    $$\Delta\theta_F=0,\qquad \theta'_H=\theta_H+\Delta\theta_H.$$

    `Delta` denotes the change from the run's initial weights; primes denote the final weights. `report.parameter_max_deltas` records the largest absolute element change in each named parameter tensor. All non-head entries should stay zero; nonzero head entries supply evidence that fitting occurred. This is separate from validation quality: changed weights alone do not show improved predictions.

    ### The saved preprocessing and actual request contract

    Raw image input is an **8-row list of 8 numeric grayscale values** each, in `[0, 16]`. Axis 0 is image row `y`; axis 1 is column `x`. There is one grayscale channel. The saved processor divides values by `16` and supplies the model's batch/channel dimensions. Do not normalize twice or send an RGB image to this pipeline.

    The temporary server receives **`POST /predict`**, with a JSON object whose required key is `"pixels"` and whose value is the raw nested image list. A successful HTTP 200 response contains:

    | Field | Type / meaning |
    | --- | --- |
    | `label` | `str`: saved label corresponding to the largest score; an exact tie selects the first score position. |
    | `scores` | Two finite floats, a softmax distribution in saved output-index order, summing to one up to rounding. They are not a guarantee of correctness. |
    | `revision` | `str`: SHA-256 fingerprint of the source checkpoint. This identifies the source; the adapted weights are additionally stored in `model.pt`. |

    For logits `z_0, z_1`, softmax scores are $p_j=e^{z_j}/(e^{z_0}+e^{z_1})$, for `j=0,1`. The model is reloaded with its saved preprocessing and label mapping before the request. Invalid image shape, nonfinite values or values outside `[0, 16]` yield HTTP 400 with an `"error"` string. The supplied runner sends one valid request and one invalid-shape request, then closes the temporary localhost server. You do not need to construct a request or keep a server running.

    ### Read the evidence before choosing a package

    Press **Adapt, save, reload and send local HTTP request** after running your function cell. `baseline_validation` and `adapted_validation` each contain `source_like` and `target_shifted` accuracy fractions on the same validation rows. The baseline is measured before target adaptation, including the newly initialized even/odd head. Do not treat it as the source model's original 10-class test accuracy.

    Dotted field paths below describe entries in the displayed JSON. They are Python dictionary keys, not object attributes. Check `reload_scores_max_difference` for numeric reproduction, `reload_agreement` for original/reloaded/HTTP response equality, `http_status` for the real successful request, and `invalid_request_status` for rejection of the invalid example. `request.json.pixels` is the raw submitted sample; `response` records the actual reply. The report also exposes `config` and `package_files` so you can inspect the saved label order, transform, source revision and files. The ZIP contains `model.pt`, `config.json` and `README.md`.

    Compare rates with all other controls fixed, keeping the first report/package before rerunning. Equal score vectors are necessary evidence of reproduction, but wrong label names could still change their meaning; trace score positions to saved labels as a separate check. This exercise establishes a local serving trace. It does not publish weights to the Hub or demonstrate hosted-service reliability, throughput or generalization.

    **Official references:** [Python 3.12 dictionaries](https://docs.python.org/3.12/library/stdtypes.html#mapping-types-dict), [Python JSON conversion](https://docs.python.org/3.12/library/json.html#json-to-py-table), [PyTorch 2.8 Module and parameters](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Module.html), [PyTorch 2.8 Adam](https://docs.pytorch.org/docs/2.8/generated/torch.optim.Adam.html). Local fitting, packaging and request behavior are defined above; implementing them is not part of this configuration TODO.
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

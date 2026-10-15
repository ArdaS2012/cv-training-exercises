import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="Adapt a Pretrained Classifier")


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

    **15 minutes in pairs after guided setup.** Verify adaptation behavior from changed weights and held-out outcomes rather than a label such as fine-tuning.

    1. Predict which groups can change. Complete the target-head and selective-training TODOs, keeping the first convolution frozen; run the prepared head fit and brief last-block fine-tune.

    2. Compare parameter counts/deltas and validation predictions across both routes. Identify evidence that a frozen group remained fixed and a selected group was eligible to learn.

    3. Recommend a route for a small target dataset with a possible domain shift. State one limitation and what fresh validation evidence would change your recommendation.

    **Bring back:** Trainable-group/count/delta table, validation comparison and a conditional adaptation choice.

    Swap editor/checker halfway. Record in the observation fields, save code and download findings. The complete task and setup are in this block’s `GROUP_TASK.md`. Prerequisite TODOs discussed in plenary belong to setup.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Adapt a Pretrained Classifier

    15-minute group task after guided setup. Fit a new head on frozen pretrained features, then compare a real selective last-block fine-tune and held-out predictions.

    Predict → complete the short TODOs → change one control → inspect → explain → download.

    Edit only `# TODO` regions; run with Shift+Enter. Save Python with Ctrl+S / Command+S. Download findings to retain current controls, measurements and observations. The untouched starter shows friendly waiting messages. Work in pairs and swap who edits. Prepared code and trained assets are supplied; no API key or GPU.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    run_adaptation=mo.ui.run_button(label='Fit head and briefly fine-tune last block')
    prediction=mo.ui.text_area(label='Predict which parameter groups change in each route',full_width=True)
    mo.vstack([prediction,run_adaptation])
    return prediction, run_adaptation


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## TODO — head width and selective training groups
    The source checkpoint predicts 10 digit classes. The target is even=0 versus odd=1. Return the target output width from `target_head_classes`. Return a dictionary of Boolean flags for `conv1`, `conv2`, `head` from `selective_trainable_groups`: keep the first convolution fixed and train the last convolution plus target head. The fully frozen-feature baseline is supplied for comparison.

    The new head fits genuine cached source-CNN features for 35 epochs. A separate copy then fine-tunes for 12 short epochs with a smaller backbone learning rate. Inspect trainable names, parameter counts, actual deltas and validation predictions. The target split and source pretraining IDs are disjoint; this related-domain example is not a universal transfer benchmark.
    """)
    return


@app.cell
def _():
    def target_head_classes():
        # TODO 1: output width for the stated target labels.
        classes=None
        # END TODO 1
        return classes
    def selective_trainable_groups():
        # TODO 2: explicit booleans for conv1, conv2 and head.
        groups=None
        # END TODO 2
        return groups

    return selective_trainable_groups, target_head_classes


@app.cell(hide_code=True)
def _(
    day2,
    mo,
    run_adaptation,
    selective_trainable_groups,
    target_head_classes,
):
    adaptation_report=None
    _message=None
    if run_adaptation.value:
        try:adaptation_report=day2.adapt(target_head_classes(),selective_trainable_groups())
        except (ValueError,TypeError,KeyError) as _error:_message=str(_error)
    mo.callout(_message,kind='warn') if _message else mo.callout('Complete head/classes and trainable groups, then fit.',kind='info') if adaptation_report is None else mo.vstack([mo.ui.table(adaptation_report['routes']),mo.md('```json\n'+__import__('json').dumps(adaptation_report,indent=2)+'\n```')])
    return (adaptation_report,)


@app.cell(hide_code=True)
def _(mo):
    observations=mo.ui.text_area(label='Findings: measured change, explanation and one limitation',full_width=True)
    observations
    return (observations,)


@app.cell(hide_code=True)
def _(adaptation_report, display, mo, observations, prediction):
    display.download(mo,observations.value,{'prediction':prediction.value,'report':adaptation_report},'transfer-findings')
    return


if __name__ == "__main__":
    app.run()

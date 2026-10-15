import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="Train a Tiny Neural Classifier")


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p/'cv_exercises').is_dir())))
    import marimo as mo
    import numpy as np
    from cv_exercises import data, display

    return display, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Exercise + breakout: your shared task

    **15 minutes in pairs after guided setup.** Separate evidence of fitting from evidence of generalization using a controlled training experiment.

    1. Predict training loss and validation behavior. Complete the two TODOs and manually check one negative-log target loss.

    2. At learning rate 0.5 compare 0 versus 80 updates with the same initialization/data. Record loss and a held-out prediction from both runs.

    3. Repeat 80 updates changing only rate to 2.0. Accept the actual result even if it worsens. Defend which evidence shows learning and which further evidence would be needed for generalization.

    **Bring back:** Loss curve, zero-update/baseline/rate comparison, held-out prediction and a qualified learning conclusion.

    Swap editor/checker halfway. Record in the observation fields, save code and download findings. The complete task and setup are in this block’s `GROUP_TASK.md`. Prerequisite TODOs discussed in plenary belong to setup.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Train a Tiny Neural Classifier

    15-minute group task after guided setup. Complete a nonlinear activation and classification loss in a supplied two-layer model. Compare loss before and after real CPU gradient updates. Inspect held-out validation predictions; explain why low training loss alone does not demonstrate generalization. The final test set stays untouched.

    Predict → complete the short TODOs → change one control → explain → download findings.

    Use the cell code control to open TODO cells. Edit only `# TODO` areas and run with Shift+Enter. Waiting messages are normal before completion. Save Python with Ctrl+S / Command+S; widget values need the findings download. Work in pairs and swap who edits.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    steps=mo.ui.slider(0,160,step=20,value=80,label='Gradient steps')
    learning_rate=mo.ui.dropdown(options=[0.05,0.2,0.5,2.0],value=0.5,label='Learning rate')
    run_training=mo.ui.run_button(label='Train on CPU')
    prediction=mo.ui.text_area(label='Prediction: what should happen to training loss and validation accuracy?',full_width=True)
    mo.hstack([steps,learning_rate,run_training],wrap=True)
    return learning_rate, prediction, run_training, steps


@app.cell(hide_code=True)
def _(mo, prediction):
    mo.vstack([prediction,mo.md("""## TODO: ReLU and mean target-class cross-entropy
    Write activation(z) for an array of any shape: return elementwise max of z and zero. Write mean_loss(probabilities, labels) for n×10 probabilities and n integer class IDs: select each row’s target probability, clip below at 1e-12, take negative natural log, then average. np.maximum, np.arange, np.clip, np.log and np.mean are available. The 64→24→10 model, softmax, fixed split and gradient plumbing are supplied; derivatives assume these exact definitions.""")])
    return


@app.cell
def _():
    def activation(z):
        # TODO 1: one elementwise activation expression.
        activated=None
        # END TODO 1
        return activated
    def mean_loss(probabilities, labels):
        # TODO 2: mean negative natural log of each target probability.
        loss=None
        # END TODO 2
        return loss

    return activation, mean_loss


@app.cell(hide_code=True)
def _(activation, learning_rate, mean_loss, mo, run_training, steps):
    _view = None
    from cv_exercises.neural import train
    report=None
    training_error=None
    if run_training.value:
        try: report=train(activation,mean_loss,steps.value,learning_rate.value)
        except (ValueError,TypeError,IndexError) as error: training_error=str(error)
    if training_error: _view = mo.callout(training_error,kind='warn')
    elif report is None: _view = mo.callout('Complete activation and loss, then press Train on CPU.',kind='info')
    _view
    return (report,)


@app.cell(hide_code=True)
def _(mo, report):
    _view = None
    import matplotlib.pyplot as plt
    if report is not None:
        figure,axes=plt.subplots(1,2,figsize=(10,3),layout='constrained')
        axes[0].plot(report['loss']); axes[0].set(title='Training cross-entropy',xlabel='gradient steps',ylabel='natural-log units')
        axes[1].plot(report['validation_accuracy']); axes[1].set(title='Validation accuracy',xlabel='gradient steps',ylabel='fraction correct',ylim=(0,1))
        rows=[{'image_id':i,'target':y,'prediction':p} for i,y,p in zip(report['validation_ids'],report['targets'],report['predictions'])]
        _view = mo.vstack([figure,mo.ui.table(rows),mo.md('180 train / 60 validation / 60 untouched test. Seed 42. This tiny local experiment is not a production-quality or CNN benchmark.')])
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    observations=mo.ui.text_area(label='Findings: loss before/after; held-out prediction; training versus inference; one generalization limit',full_width=True)
    observations
    return (observations,)


@app.cell(hide_code=True)
def _(display, learning_rate, mo, observations, prediction, report, steps):
    display.download(mo,observations.value,{'prediction':prediction.value,'steps':steps.value,'rate':learning_rate.value,'report':report},'neural-findings')
    return


if __name__ == "__main__":
    app.run()

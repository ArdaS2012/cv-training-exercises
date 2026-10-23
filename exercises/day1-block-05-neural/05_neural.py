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

    return data, display, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Hands-on investigation: your shared mission

    **30 minutes in pairs.** Audit a learning claim: did gradient updates improve this classifier, and what do the images actually show?

    ### Prior-slide takeaways to use now

    Slides 57–63: ReLU supplies nonlinearity; cross-entropy scores the target probability. Updates use training rows; validation informs choices and the test stays untouched.

    1. Predict the learning curves and manually check one target-probability loss. Implement ReLU and mean cross-entropy; trace one negative activation and one target probability. Keep the supplied matching derivatives and fixed split unchanged.

    2. At rate 0.5 compare 0 and 80 updates from the same initialization. Log final training loss, validation accuracy and the same held-out image ID/prediction; inspect its actual pixels. Swap roles at minute 15 and explain one changed or unchanged prediction.

    3. Repeat 80 updates changing only rate to 2.0. Compare the curves and held-out image; accept deterioration as evidence. Recommend a next training experiment and name a population/condition your current validation cannot establish.

    **Bring back:** Three controlled run rows and curves, one inspected held-out image, an explained learning claim and a next-test limitation.

    **Route:** 0–4 predict; 4–12 implement and trace; 12–22 compare (swap roles at 15); 22–27 challenge and decide; 27–30 explain, save and export. Record each run in the evidence log before changing a control. Downloads include the current run plus your cumulative written log; save after each trial. See this block’s `GROUP_TASK.md`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Train a Tiny Neural Classifier

    30-minute investigation after guided setup. Complete a nonlinear activation and classification loss in a supplied two-layer model. Compare loss before and after real CPU gradient updates. Inspect held-out validation predictions; explain why low training loss alone does not demonstrate generalization. The final test set stays untouched.

    Inspect → predict → implement and trace → compare → challenge → explain and download evidence.

    Use the cell code control to open TODO cells. Edit only `# TODO` areas and run with Shift+Enter. Waiting messages are normal before completion. Save Python with Ctrl+S / Command+S; widget values need the findings download. Work in pairs and swap who edits.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    steps=mo.ui.slider(0,160,step=20,value=80,label='Gradient steps')
    learning_rate=mo.ui.dropdown(options=[0.05,0.2,0.5,2.0],value=0.5,label='Learning rate')
    run_training=mo.ui.run_button(label='Train on CPU')
    example_index=mo.ui.slider(0,59,value=0,label='Validation image row — keep fixed across runs')
    prediction=mo.ui.text_area(label='Prediction: what should happen to training loss and validation accuracy?',full_width=True)
    mo.hstack([steps,learning_rate,run_training,example_index],wrap=True)
    return example_index, learning_rate, prediction, run_training, steps


@app.cell(hide_code=True)
def _(mo, prediction):
    mo.vstack([prediction,mo.md("""## TODO: ReLU and mean target-class cross-entropy
    Write activation(z) for an array of any shape: return elementwise max of z and zero. Write mean_loss(probabilities, labels) for n×10 probabilities and n integer class IDs: select each row’s target probability, clip below at 1e-12, take negative natural log, then average. np.maximum, np.arange, np.clip, np.log and np.mean are available. The 64→24→10 model, softmax, fixed split and gradient plumbing are supplied; derivatives assume these exact definitions.""")])
    return


@app.cell
def _(np):
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
def _(data, example_index, mo, np, report):
    _view = None
    import matplotlib.pyplot as plt
    if report is not None:
        figure,axes=plt.subplots(1,3,figsize=(12,3),layout='constrained')
        axes[0].plot(report['loss']); axes[0].set(title='Training cross-entropy',xlabel='gradient steps',ylabel='natural-log units')
        axes[1].plot(report['validation_accuracy']); axes[1].set(title='Validation accuracy',xlabel='gradient steps',ylabel='fraction correct',ylim=(0,1))
        _x,_y,_ids=data.images()
        _pick=example_index.value
        _image_id=report['validation_ids'][_pick]
        _source=int(np.flatnonzero(_ids==_image_id)[0])
        axes[2].imshow(_x[_source].reshape(8,8),cmap='gray',vmin=0,vmax=1)
        axes[2].set_title(f"Held-out ID {_image_id}\nPrediction {report['predictions'][_pick]} / target {report['targets'][_pick]}")
        axes[2].axis('off')
        rows=[{'image_id':i,'target':y,'prediction':p} for i,y,p in zip(report['validation_ids'],report['targets'],report['predictions'])]
        _view = mo.vstack([mo.md(f"**Current run:** {report['steps']} updates; rate {report['rate']}; final loss {report['loss'][-1]:.6f}; validation accuracy {report['validation_accuracy'][-1]:.3f}. Record these before rerunning."),figure,mo.ui.table(rows),mo.md('180 train / 60 validation / 60 untouched test. Seed 42. This tiny local experiment is not a production-quality or CNN benchmark.')])
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    observations=mo.ui.text_area(value='Updates | rate | final train loss | validation accuracy | fixed image ID | prediction/target\n0 | 0.5 | | | |\n80 | 0.5 | | | |\n80 | 2.0 | | | |\nOne visible image feature and changed/unchanged prediction:\nWhat supports the learning claim?\nNext controlled experiment and generalization limit:',label="Evidence log — keep every trial, mechanism and decision",full_width=True)
    observations
    return (observations,)


@app.cell(hide_code=True)
def _(display, example_index, learning_rate, mo, observations, prediction, report, steps):
    display.download(mo,observations.value,{'validation_row':example_index.value,'prediction':prediction.value,'steps':steps.value,'rate':learning_rate.value,'report':report},'neural-findings')
    return


if __name__ == "__main__":
    app.run()

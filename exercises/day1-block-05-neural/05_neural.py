import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Teach a Digit Reader")


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p/'cv_exercises').is_dir())))
    import marimo as mo
    import numpy as np
    from cv_exercises import data, display
    from cv_exercises.neural import softmax, backward, pipeline_report, pipeline_figure, digit_examples
    return Path, backward, data, digit_examples, display, mo, np, pipeline_figure, pipeline_report, softmax


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Teach a digit reader

    Build a small training pipeline. Compare learned predictions with a zero-update baseline. Work in pairs; swap who edits.
    """)
    return


@app.cell(hide_code=True)
def _(digit_examples):
    digit_examples()
    return


@app.cell(hide_code=True)
def _(Path, mo):
    _root = next(p for p in Path(__file__).resolve().parents if (p/'cv_exercises').is_dir())
    mo.download((_root/'exercises/day1-block-05-neural/TASK_API.md').read_bytes(), filename='neural-TASK_API.md', label='API reference (.md)')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 1 · Select data
    Keep learning and evaluation rows separate.

    `prepare_data()` has no arguments. Supplied `data.classification_split()` returns x float(300,64) row-major pixels [0,1], y/ids int(300,), and disjoint tr/va/te index arrays (180/60/60). Return dict: train_x float(180,64), train_y int(180,), valid_x float(60,64), valid_y/valid_ids int(60,), with aligned rows. Test rows stay unused.
    """)
    return


@app.cell
def _(data):
    def prepare_data():
        x, y, ids, tr, va, te = data.classification_split()
        # TODO 1: Select training and validation rows with their matching labels and IDs.
        dataset = None
        # END TODO 1
        return dataset
    return (prepare_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2 · Initialize
    Give each run the same initial model.

    `initialize()` has no arguments. Return dict of fresh float arrays: w1(64,24), b1(24,), w2(24,10), b2(10,). Supplied rng seed42; `rng.normal(mean,SD,shape)` creates weights with mean0, SD0.15; `np.zeros(shape)` creates biases. Preserve the given key order.
    """)
    return


@app.cell
def _(np):
    def initialize():
        rng = np.random.default_rng(42)
        # TODO 2: Create fresh weights and biases for both layers using the supplied seed.
        params = None
        # END TODO 2
        return params
    return (initialize,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3 · Forward pass
    Calculate hidden activations and digit probabilities.

    `forward(x,params)`: finite float x(n,64), n>0, pixels [0,1]; params from step2. Return float z/h(n,24), probabilities(n,10), in that order. z=XW1+b1; h=ReLU(z); p=softmax(hW2+b2). Supplied softmax(scores) returns row probabilities; `np.maximum` is elementwise. Preserve inputs.
    """)
    return


@app.cell
def _(np, softmax):
    def forward(x, params):
        # TODO 3: Transform each image through the hidden layer into digit probabilities.
        z = h = probabilities = None
        # END TODO 3
        return z, h, probabilities
    return (forward,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4 · Loss
    Measure how much probability the model gives each true digit.

    `mean_loss(probabilities,labels)`: float(n,c) row probabilities, int labels(n,) in0..c-1, n,c>0. Return a Python float: mean negative natural log of true-label probabilities, clipping below1e-12. `np.arange(n)`, `np.clip`, `np.log`, `np.mean` are available. Preserve inputs.
    """)
    return


@app.cell
def _(np):
    def mean_loss(probabilities, labels):
        # TODO 4: Average negative log probabilities for the true digits, clipping below 1e-12.
        loss = None
        # END TODO 4
        return loss
    return (mean_loss,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 5 · Gradient update
    Move each parameter against its loss gradient.

    `update(params,grads,rate)`: same float arrays/keys/shapes as step2 in both dictionaries; positive finite float rate. Return a NEW dictionary of NEW arrays using theta_new=theta-rate*gradient for every key; do not mutate inputs.
    """)
    return


@app.cell
def _():
    def update(params, grads, rate):
        # TODO 5: Apply the gradient descent rule to every weight and bias.
        updated = None
        # END TODO 5
        return updated
    return (update,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 6 · Training loop
    Connect prediction, loss and updates.

    `fit(dataset,params,steps,rate)`: dictionaries from steps1/2; steps integer≥0, rate positive finite float. Return (trained_params, history), with steps+1 Python float losses, initial loss first. Supplied `backward(x,labels,params,cache)` returns gradients with step2 keys/shapes; cache is the (z,h,probabilities) tuple. Record loss before each update; zero steps performs no update. Preserve dataset and starting params.
    """)
    return


@app.cell
def _(backward, forward, mean_loss, update):
    def fit(dataset, params, steps, rate):
        history = []
        for iteration in range(steps + 1):
            # TODO 6: Connect prediction, loss, gradients and updates inside the supplied loop.
            # Record the initial loss before learning so zero steps remains a valid baseline.
            cache = None
            loss = None
            if cache is None or loss is None: return None
            history.append(float(loss))
            if iteration == steps: break
            grads = None
            params = None
            # END TODO 6
            if params is None: return None
        return params, history
    return (fit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 7 · Validation
    Evaluate images the model did not train on.

    `evaluate(dataset,params)`: dictionaries from steps1/2; valid_x float(m,64), valid_y/valid_ids int(m,), aligned rows, m>0. Return (predictions,accuracy): integer NumPy array(m,), IDs0..9, and Python float fraction correct[0,1]. `np.argmax(axis=1)` uses the first tied class. No updates; preserve inputs.
    """)
    return


@app.cell
def _(forward, np):
    def evaluate(dataset, params):
        # TODO 7: Predict one digit per validation image and calculate the fraction correct.
        predictions = accuracy = None
        # END TODO 7
        return predictions, accuracy
    return (evaluate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Challenge
    Compare **0 vs 80 steps at rate 0.5**, then **80 at 2.0**. Keep the same validation image. Inspect loss and held-out predictions; the test split stays unused.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    steps = mo.ui.slider(0,160,step=20,value=80,show_value=True,label='Steps')
    learning_rate = mo.ui.dropdown(options=[0.05,0.2,0.5,2.0],value=0.5,label='Rate')
    run_training = mo.ui.run_button(label='Train on CPU')
    example_index = mo.ui.slider(0,59,value=0,show_value=True,label='Validation row')
    mo.hstack([steps,learning_rate,run_training,example_index],wrap=True)
    return example_index, learning_rate, run_training, steps


@app.cell(hide_code=True)
def _(evaluate, fit, forward, initialize, learning_rate, mean_loss, mo, pipeline_report, prepare_data, run_training, steps):
    report = None
    _message = 'Complete the code, then Train on CPU.'
    if run_training.value:
        report, _message = pipeline_report(prepare_data, initialize, forward, mean_loss, fit, evaluate, steps=steps.value, rate=learning_rate.value)
    mo.md(_message)
    return (report,)


@app.cell(hide_code=True)
def _(example_index, pipeline_figure, report):
    pipeline_figure(report,example_index.value) if report else None
    return


@app.cell(hide_code=True)
def _(display, example_index, mo, report):
    display.download(mo,'',{'validation_row':example_index.value,'steps':report['steps'],'rate':report['rate'],'report':report},'neural-findings',label='Save run (.json)') if report else None
    return


if __name__ == "__main__":
    app.run()

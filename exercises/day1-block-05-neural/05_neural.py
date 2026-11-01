import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="Build a Tiny Neural Pipeline")


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p/'cv_exercises').is_dir())))
    import marimo as mo
    import numpy as np
    from cv_exercises import data, display
    from cv_exercises.neural import softmax, backward

    return backward, data, display, mo, np, softmax




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Build a tiny image classifier end to end

    **Around 30 minutes in pairs.** Build the route **images → split → parameters → forward pass → loss → gradients → update → validation**. Then audit whether learning happened.

    **Use slides 56–63:** a layer multiplies inputs by weights and adds biases; ReLU adds nonlinearity; softmax produces class probabilities; cross-entropy scores the true class; gradient descent changes parameters. Training and validation use different images; the final test stays untouched.

    Seven short TODOs build one connected pipeline. The fixed data loader, stable softmax, derivatives, charts and export are supplied. Read the shape hints beside each task, implement a few lines, and trace one row with your partner. Swap editor/checker halfway.

    **Predict before running:** what changes between 0 and 80 updates? At rate 0.5 compare both from the same seed, then repeat 80 updates at rate 2.0. Keep validation image row 0 fixed. Record three runs, inspect the pixels, explain a changed or unchanged prediction and name a generalization limit.

    Edit only TODO regions; Shift+Enter runs a cell. Use **Train on CPU** after completing the stages. Save code with Ctrl+S / Command+S and download findings after each trial; saving Python does not preserve widget values. `None` means a stage is unfinished. All data are local; no download or GPU is needed. Help: this block's README and GROUP_TASK.md.
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
def _(prediction):
    prediction
    return


@app.cell(hide_code=True)
def _(data, mo):
    import matplotlib.pyplot as _plt
    _pixels, _labels, _ids = data.images()
    _fig, _axes = _plt.subplots(1, 3, figsize=(7, 2), layout='constrained')
    for _axis, _row in zip(_axes, [0, 30, 60]):
        _axis.imshow(_pixels[_row].reshape(8, 8), cmap='gray', vmin=0, vmax=1)
        _axis.set_title(f'ID {_ids[_row]} / label {_labels[_row]}')
        _axis.axis('off')
    mo.vstack([mo.md('### Inspect the input before building\nThese are examples from the supplied 300-image dataset, not evaluation results. Each 8×8 image becomes one row of 64 normalized pixel values. Flattening preserves row-major pixel order; labels name the true digit. The fixed split will select separate training and validation rows.'), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TODO 1 — Choose training and validation rows

    `prepare_data()` has **no arguments**. Supplied `data.classification_split()` returns `(x, y, ids, tr, va, te)`:

    - `x`: finite float NumPy array `(300,64)`, one flattened 8×8 grayscale image per row, row-major pixel order, intensities already divided by 16 into `[0,1]`. No further normalization is needed.
    - `y`: integer `(300,)` true digit IDs `0..9`; `ids`: integer `(300,)` original source IDs, aligned with `x` and `y`.
    - `tr`, `va`, `te`: disjoint integer row-index arrays of lengths 180, 60, 60. They are fixed, stratified by digit and reproducible. An ID is not a row index.

    Return a dictionary with **exactly** `train_x`, `train_y`, `valid_x`, `valid_y`, `valid_ids`: select the corresponding rows using `tr` and `va`. Their shapes are `(180,64)`, `(180,)`, `(60,64)`, `(60,)`, `(60,)`. Preserve row alignment and inputs; never use `te` to select examples. The supplied loader is the only place creating splits. No empty batches occur.

    **Hint:** `array[index_array]` selects those rows and preserves their order. Generic example: selecting names at positions `[3,1]` returns position 3 before position 1. Before continuing, explain why fitting on validation rows would invalidate this comparison.
    """)
    return


@app.cell
def _(data):
    def prepare_data():
        x, y, ids, tr, va, te = data.classification_split()
        # TODO 1: select aligned training/validation rows into the documented dictionary.
        dataset = None
        # END TODO 1
        return dataset

    return (prepare_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TODO 2 — Initialize a 64 → 24 → 10 model

    `initialize()` has **no arguments**. Return `params`, a dictionary of finite float NumPy arrays with exactly these keys:

    | Key | Shape | Role |
    |---|---|---|
    | `w1` | `(64,24)` | input-to-hidden weights |
    | `b1` | `(24,)` | one bias per hidden unit |
    | `w2` | `(24,10)` | hidden-to-class weights |
    | `b2` | `(10,)` | one bias per digit class |

    A seeded `rng` is supplied inside the function. Draw `w1` **then** `w2` from a normal distribution of mean 0, standard deviation 0.15. Initialize both biases to zero. Recreate fresh arrays each call: each experiment must start with the same parameters. Do not initialize every weight to zero: equal hidden units would learn the same thing.

    **Toolbox:** `rng.normal(loc, scale, size)` returns random floats of the requested tuple shape; `np.zeros(shape)` returns a zero array (an integer shape makes a vector). Parameters have no physical units. The fixed architecture has no extra layers or batch dimension in its weights. **Trace:** `(n,64)` times which matrix gives `(n,24)`?
    """)
    return


@app.cell
def _(np):
    def initialize():
        rng = np.random.default_rng(42)
        # TODO 2: build the four parameter arrays; draw w1 before w2.
        params = None
        # END TODO 2
        return params

    return (initialize,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TODO 3 — Build the forward pass

    `forward(x, params)` receives finite float image rows `x` of shape `(n,64)`, `n>0`, values `[0,1]`, and the four-array parameter dictionary from TODO 2. Return the **tuple** `(z, h, probabilities)` in that order, not a dictionary:

    $$z=XW_1+b_1,\quad h=\max(0,z),\quad s=hW_2+b_2,\quad p=\mathrm{softmax}(s).$$

    `z` is the pre-activation `(n,24)`, `h` the ReLU output `(n,24)`, `s` the raw class scores `(n,10)` and `probabilities` the finite output `(n,10)` in `[0,1]`, each row summing to 1. Columns correspond to digit IDs 0 through 9. Keep the original inputs unchanged. ReLU preserves positives and sets negatives to zero; softmax compares scores within each image.

    **Toolbox:** `a @ b` is matrix multiplication; adding a bias vector broadcasts across rows. `np.maximum(a,b)` selects the elementwise maximum, broadcasting scalar arguments. Supplied `softmax(logits)` takes any finite `(n,c)` score array with `n,c>0` and returns same-shape normalized probabilities. It subtracts each row maximum before exponentiation to avoid overflow; it does not mutate scores. No derivative or softmax implementation is required.

    **Hint:** follow the four mathematical stages and track the shape after each. Trace one negative pre-activation through ReLU. The same function must work for 180 training rows, 60 validation rows or one image.
    """)
    return


@app.cell
def _(np, softmax):
    def forward(x, params):
        # TODO 3: affine layer, ReLU, affine layer, supplied softmax.
        z = h = probabilities = None
        # END TODO 3
        return z, h, probabilities

    return (forward,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TODO 4 — Score the true class

    `mean_loss(probabilities, labels)` receives finite floating probabilities `(n,c)` in `[0,1]` with rows summing to 1 and integer true-class IDs `labels` of shape `(n,)` in `0..c-1`. `n>0`, `c>=1`; rows align. Return one finite nonnegative Python/NumPy numeric scalar, leaving inputs unchanged. Do not apply softmax again or score the maximum predicted probability.

    $$L=-\frac{1}{n}\sum_{i=0}^{n-1}\ln(\max(p_{i,y_i},10^{-12})).$$

    `n` is batch size; `p` is the probability matrix; `y_i` is the true class of row `i`. The natural logarithm gives natural-log units. A target probability 0.25 gives loss about 1.386; target certainty gives zero. Clipping at `1e-12` makes zero probabilities finite. This formula must match the supplied derivatives.

    **Toolbox:** `np.arange(n)` makes indices `0..n-1`; `a[rows, columns]` with equal-length index arrays selects paired entries (one per row). `np.clip(a, low, high)` bounds values without mutating `a`; `np.log(a)` takes elementwise natural logs; `np.mean(a)` returns the mean scalar. **Hint:** identify each row's true-class column, score it, then average. For schema only: labels `[2,0]` refer to column 2 of row 0 and column 0 of row 1.
    """)
    return


@app.cell
def _(np):
    def mean_loss(probabilities, labels):
        # TODO 4: mean negative natural log of each clipped true-class probability.
        loss = None
        # END TODO 4
        return loss

    return (mean_loss,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TODO 5 — Apply a gradient update

    `update(params, grads, rate)` receives the parameter dictionary from TODO 2 and a gradient dictionary with **exactly the same keys and shapes**. Each gradient is a finite float array of derivatives of **mean training cross-entropy**, with respect to that parameter. `rate` is a positive finite scalar, selected here from `0.05, 0.2, 0.5, 2.0`. Return a **new dictionary and new arrays**; do not change either input.

    $$\theta_{new}=\theta_{old}-\eta\nabla_\theta L.$$

    `theta` is each weight/bias array; `eta` is `rate`. Subtracting moves against increasing loss; a large step can overshoot. Illustrative scalar trace: value 0.4, derivative 0.2, rate 0.1 gives 0.38. A zero gradient leaves its parameter unchanged.

    **Hint:** iterate over dictionary keys (or write four explicit entries) and apply the same rule to each. `for key in mapping` visits keys; `mapping[key]` retrieves its array; ordinary array arithmetic creates a new array. Do not recompute gradients inside this function.
    """)
    return


@app.cell
def _():
    def update(params, grads, rate):
        # TODO 5: apply the descent rule to every weight and bias.
        updated = None
        # END TODO 5
        return updated

    return (update,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TODO 6 — Connect the training loop

    `fit(dataset, params, steps, rate)` receives the exact dictionaries from TODOs 1/2, integer update count `steps>=0` (slider: 0–160 by 20), and positive scalar `rate` as in TODO 5. Return `(trained_params, history)`. `history` is a list of `steps+1` finite loss floats, including the baseline **before any update**. Rebind local `params` using your nonmutating update function; keep caller inputs unchanged.

    The loop and history storage are supplied. Fill the forward-pass, loss, backward and update connections at their named placeholders. Use **training rows only**, including for gradients. `forward` returns a three-item cache; its last item is probabilities. `mean_loss` needs those probabilities and aligned true labels.

    **Supplied derivative API:** `backward(x, labels, params, cache)` accepts nonempty finite training inputs `(n,64)`, integer labels `(n,)` in `0..9`, the current parameter dictionary, and the `(z,h,probabilities)` tuple from `forward` on **those same inputs and current parameters**. It returns `grads` with keys `w1,b1,w2,b2`, each matching its parameter's float shape. It calculates derivatives for ReLU and mean softmax cross-entropy and never mutates inputs or updates parameters. Use it before changing parameters. There is no need to derive backpropagation here.

    **Hint:** the supplied `if iteration == steps` break keeps 0 steps a genuine control and avoids an extra update. `history.append(float(value))` stores a scalar. Explain the difference between computing a gradient and applying one. Do not call the old all-in-one `train` helper.
    """)
    return


@app.cell
def _(backward, forward, mean_loss, update):
    def fit(dataset, params, steps, rate):
        history = []
        for iteration in range(steps + 1):
            # TODO 6a: forward on training rows and score their true labels.
            cache = None
            loss = None
            # END TODO 6a
            if cache is None or loss is None:
                return None
            history.append(float(loss))
            if iteration == steps:
                break
            # TODO 6b: obtain gradients for this cache, then apply your update.
            grads = None
            params = None
            # END TODO 6b
            if params is None:
                return None
        return params, history

    return (fit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TODO 7 — Evaluate without learning from validation

    `evaluate(dataset, params)` receives the dataset dictionary from TODO 1 and trained parameter dictionary from TODO 6. Use only `valid_x` and `valid_y` here. Return `(predictions, accuracy)`:

    - `predictions`: integer NumPy vector `(60,)`, one predicted digit ID `0..9` per validation row, same row order as `valid_ids`.
    - `accuracy`: one finite scalar fraction in `[0,1]`: number correct divided by number of validation images. It is not a percentage or the training loss.

    Run your forward pass on validation rows, choose each row's highest-probability class and compare to its true label. Do not calculate gradients or update parameters; inputs must remain unchanged. General operation supports any nonempty validation batch. For tied probabilities, choose the first class index.

    **Toolbox:** `np.argmax(a, axis=...)` returns indices of largest entries along that axis; axis 0 indexes rows, axis 1 indexes columns. Without `axis`, it flattens the array. `predictions == labels` returns aligned Boolean correctness values; `np.mean` averages them as 0/1. **Trace:** why does an `(n,10)` probability matrix become `(n,)` predictions?

    ### Your connected experiment

    Press **Train on CPU**. The supplied display controller calls your functions in order and labels the first unfinished stage. Every press rebuilds the fixed split and seeded parameters. Compare `(steps,rate)=(0,0.5),(80,0.5),(80,2.0)`. The plots show training loss at every step and validation accuracy **before and after training only**, not a per-step validation curve. Keep image row 0 fixed and inspect its pixels and source ID. Changing probabilities need not change the predicted digit.

    Record settings, final loss, validation accuracy, image ID and prediction/target for every run. Name the evidence of fitting and one condition not tested (for example different handwriting/cameras). Choosing a rate using validation is model selection; it does not make validation an untouched test. The 60 test rows remain unused.

    ### Official NumPy 2.2 API references

    [Row indexing](https://numpy.org/doc/2.2/user/basics.indexing.html#integer-array-indexing) · [normal](https://numpy.org/doc/2.2/reference/random/generated/numpy.random.Generator.normal.html) · [zeros](https://numpy.org/doc/2.2/reference/generated/numpy.zeros.html) · [matmul](https://numpy.org/doc/2.2/reference/generated/numpy.matmul.html) · [maximum](https://numpy.org/doc/2.2/reference/generated/numpy.maximum.html) · [arange](https://numpy.org/doc/2.2/reference/generated/numpy.arange.html) · [clip](https://numpy.org/doc/2.2/reference/generated/numpy.clip.html) · [log](https://numpy.org/doc/2.2/reference/generated/numpy.log.html) · [mean](https://numpy.org/doc/2.2/reference/generated/numpy.mean.html) · [argmax](https://numpy.org/doc/2.2/reference/generated/numpy.argmax.html).
    """)
    return


@app.cell
def _(forward, np):
    def evaluate(dataset, params):
        # TODO 7: validation forward pass, one class per row, fraction correct.
        predictions = accuracy = None
        # END TODO 7
        return predictions, accuracy

    return (evaluate,)


@app.cell(hide_code=True)
def _(prepare_data, initialize, forward, mean_loss, update, fit, evaluate, learning_rate, mo, run_training, steps):
    report = None
    _message = 'Complete the seven stages, then press Train on CPU.'
    if run_training.value:
        try:
            _dataset = prepare_data()
            _params = initialize()
            if _dataset is None:
                raise ValueError('TODO 1: prepare the dataset dictionary.')
            if _params is None:
                raise ValueError('TODO 2: initialize parameters.')
            _cache = forward(_dataset['train_x'], _params)
            if any(_part is None for _part in _cache):
                raise ValueError('TODO 3: complete the forward pass.')
            if mean_loss(_cache[2], _dataset['train_y']) is None:
                raise ValueError('TODO 4: complete cross-entropy.')
            _outcome = fit(_dataset, _params, steps.value, learning_rate.value)
            if _outcome is None:
                raise ValueError('TODOs 5–6: connect gradients, updates and the loop.')
            _trained, _history = _outcome
            _before, _baseline = evaluate(_dataset, _params)
            _predictions, _accuracy = evaluate(_dataset, _trained)
            if _predictions is None or _accuracy is None:
                raise ValueError('TODO 7: evaluate validation predictions and accuracy.')
            report = dict(loss=_history, validation_accuracy=[float(_baseline), float(_accuracy)],
                          predictions=_predictions.tolist(), targets=_dataset['valid_y'].tolist(),
                          validation_ids=_dataset['valid_ids'].tolist(), train_count=180,
                          validation_count=60, test_count=60, test_used=False, steps=steps.value, rate=learning_rate.value)
        except (ValueError, TypeError, IndexError, KeyError, AttributeError) as _error:
            _message = str(_error)
    mo.callout('Pipeline connected — inspect and record the evidence.' if report else _message, kind='success' if report else 'info')
    return (report,)


@app.cell(hide_code=True)
def _(data, example_index, mo, np, report):
    _view = None
    import matplotlib.pyplot as plt
    if report is not None:
        figure,axes=plt.subplots(1,3,figsize=(12,3),layout='constrained')
        axes[0].plot(report['loss']); axes[0].set(title='Training cross-entropy',xlabel='gradient steps',ylabel='natural-log units')
        axes[1].bar(['Before', 'After'], report['validation_accuracy']); axes[1].set(title='Validation accuracy (endpoints)',ylabel='fraction correct',ylim=(0,1))
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

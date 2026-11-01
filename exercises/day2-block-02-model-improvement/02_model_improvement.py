import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="Compare Controlled Training Runs")


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
    ## Training clinic: can you defend an improvement?

    **Around 30 minutes in pairs.** Parameters are learned; hyperparameters are chosen. Hold split, seed and budget fixed. Choose by validation loss; reserve test data until the decision is frozen. A lower training loss alone is insufficient.

    ### Investigate, do not just run

    1. Inspect the baseline settings (rate 0.01, penalty 0). Predict the training/validation curve change for either a rate or penalty intervention; write which outcome would contradict your hypothesis.

    2. Complete configuration and validation-selection TODOs. Run the supplied short CPU trainings with just that factor changed. Read both curves, not only the last number; record a point where their behavior differs.

    3. Swap roles. Test a second value of the SAME factor, holding the other setting, split, seed and budget fixed. Save both comparisons before rerunning; choose using the declared validation loss only.

    4. Look for a contrary result: a faster initial loss drop, better training fit or larger penalty that did not give the best validation outcome. Explain it using the measured short-run curves; if absent, inspect the labelled longer-run reference without mixing its budget into your live comparison.

    5. Freeze your candidate choice and justification, then use the reserved-test action once. Report that estimate without retuning. Explain what small data, one seed and the short budget prevent you from claiming.

    **Evidence log:** run | changed factor/value | train-curve evidence | validation loss | decision before test. Save a run before changing controls. Export both the current findings and your comparison log.
    """)
    return






@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Compare Controlled Training Runs

    Investigation after guided setup. Run two real short CPU trainings, select by validation and evaluate the chosen configuration once on reserved examples.

    Predict → complete the short TODOs → change one control → inspect → explain → download.

    Edit only `# TODO` regions; run with Shift+Enter. Save Python with Ctrl+S / Command+S. Download findings to retain current controls, measurements and observations. The untouched starter shows friendly waiting messages. Work in pairs and swap who edits. Prepared code and trained assets are supplied; no API key or GPU.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    change=mo.ui.dropdown(options=['learning_rate','penalty'],value='learning_rate',label='Change one factor')
    rate=mo.ui.dropdown(options=[.001,.03,.08],value=.03,label='Candidate learning rate')
    penalty=mo.ui.dropdown(options=[.0001,.001,.01],value=.001,label='Candidate L2 penalty')
    run_compare=mo.ui.run_button(label='Run two short CPU trainings')
    prediction=mo.ui.text_area(label='Predict train/validation behavior before running',full_width=True)
    mo.vstack([mo.hstack([change,rate,penalty,run_compare],wrap=True),prediction])
    return change, penalty, prediction, rate, run_compare


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Read the curves before choosing
    A parameter is a learned weight. Learning rate controls the size of training updates; the L2 penalty adds $\lambda\|\theta\|_2^2$ to the training objective. Here $\lambda$ is nonnegative and $\theta$ is the weight vector. The baseline uses rate 0.01 and penalty 0. A fair experiment changes just one setting at the same split, seed and training budget.

    Read horizontal position as training progress and vertical position as the stated loss. Training loss measures fit to training data; validation data loss guides selection. A widening gap suggests a hypothesis, not its cause. Choose the smaller declared validation loss, freeze that choice, then inspect reserved test once. The longer-run measured reference has a different budget and must be described separately.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Your implementation task

    Complete a configuration function and a validation-choice function. First build
    a candidate that changes exactly one setting from the declared baseline. After
    the supplied training runs, choose their original list index by final validation
    loss. The guide below explains all fields, split roles and tie behavior. Keep the
    reserved-test action until after your decision and justification are fixed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TODO guide — make a fair candidate, then select on validation

    The supplied code trains the same small CNN twice. A **configuration** is a
    small dictionary of chosen settings, not learned weights. **Training data** drive
    weight updates; **validation data** guide your choice between candidates; the
    reserved **test data** estimate the chosen model only after that choice is fixed.
    Your functions construct the configuration and choose a report index. Training,
    plotting and reserved-test evaluation are already supplied.

    ### TODO 1: `candidate_config(changed_factor, rate_value, penalty_value)`

    The baseline A uses `learning_rate=0.01` and `penalty=0.0`. Candidate B must differ
    in exactly one of those settings. Learning rate scales a gradient update. The
    L2 penalty discourages large learned weights by adding

    $$L_{train}(\theta)=L_{data}(\theta)+\lambda\sum_{j\in W}\theta_j^2.$$

    `theta` denotes learned parameters, `W` indexes weight entries only (not biases),
    `lambda` is the nonnegative `penalty`, and `L_data` is mean training
    cross-entropy. A larger penalty is a stronger preference for small weights;
    it does not directly specify accuracy. The supplied code implements this
    explicit objective rather than an optimizer's separate weight-decay rule.

    | Argument | Type, values and meaning |
    | --- | --- |
    | `changed_factor` | Python `str`, exactly `"learning_rate"` or `"penalty"`: which setting to change. |
    | `rate_value` | Finite Python `float`: proposed new learning rate. Widget choices: `0.001`, `0.03`, `0.08`; the trainer accepts `[0.00001, 0.1]`. |
    | `penalty_value` | Finite Python `float`: proposed nonnegative L2 coefficient. Widget choices: `0.0001`, `0.001`, `0.01`; the trainer accepts `[0, 0.1]`. |

    The caller passes widget **values**, such as a string or number, not widget
    objects. Both proposed numbers arrive on every call. Use the proposal for the
    named factor, and keep the other factor at its baseline. An unselected widget's
    number must not change that other factor. Current widget proposals differ from
    the relevant baseline, so exactly one setting changes.

    Return a **new Python dictionary** with exactly the case-sensitive keys
    `learning_rate` and `penalty`, each holding a float. Do not return a model,
    optimizer, list or text representation of a dictionary. The function has no
    training side effects. Invalid factor names, nonfinite numbers and proposals
    outside the documented trainer domain are outside the runner's guarantees;
    you are not asked to add an input validator.

    ### TODO 2: `choose_validation(reports)`

    `reports` is a nonempty **list of two dictionaries**, preserving candidate order:
    index `0` is A (baseline), index `1` is B (changed). Each dictionary provides
    `validation_loss: float`, a finite final mean held-out data cross-entropy. A lower
    value is better. Your selection needs only this field; small hand-check records
    may omit the other reporting fields.

    The full live reports also contain:

    | Exact key | Full runner value and interpretation |
    | --- | --- |
    | `validation_accuracy` | Float in `[0, 1]`: fraction of validation labels predicted correctly. This is supporting evidence, not the declared selection criterion. |
    | `config` | Dictionary with float values for `learning_rate` and `penalty`. |
    | `curves` | Dictionary with exactly `train_loss`, `validation_loss`, `validation_accuracy`; each value is a list of 24 floats, in epoch order. Both loss curves exclude the regularization term. |
    | `epochs` | Integer `24`: one fixed training budget for both candidates. |
    | `seed` | Integer `42`: same random initialization/control setting for both runs. |

    A loss is not an error count. Cross-entropy averages `-log(p_correct)` across
    examples, so it also changes when the model becomes more/less confident without
    changing the winning label. This is why accuracy and loss can rank runs differently.
    Return the Python integer **position** `0` or `1` with the smaller final
    `validation_loss`. For an exact tie, return the earlier position `0`. Keep reports
    and their order unchanged. Return neither the loss itself nor a dictionary.
    Empty report lists, missing criterion fields and nonfinite losses are outside
    this task's supplied domain. No test measurement is available to this function.

    **Python toolbox:** lists are ordered and start at index zero; dictionaries
    access values using exact string keys. Unrelated syntax example:
    `items[0]["name"]` reads a field in the first record; it does not compare records.
    `min(values)` returns a smallest value; `min(values, key=...)` compares by a chosen
    property but returns an input item. Use your own comparison approach. No extra
    imports are required for basic Python control flow and dictionaries.

    ### Before running and reading the evidence

    Replace only `config=None` and `index=None` in the marked regions, keep the
    signatures/return lines, and rerun with **Shift+Enter**. Press **Run two short CPU
    trainings** after the configuration is ready. The supplied trainer uses the same
    checkpoint, fixed split (180 train / 60 validation / 60 reserved test), seed and
    24 full-batch training updates for each candidate. Compare the entire training
    and validation curves, then check which final validation value supports your
    selected index. The interface validates a usable index; you must verify that
    it follows the declared loss rule yourself, including a tie.

    Record/freeze the choice before pressing **Evaluate chosen model once on reserved
    test**. That button is a separate action, not an enforced one-use lock: do not
    retune after inspecting the result. Report `test_examples` and the measured
    correct fraction. A small same-split comparison establishes neither a universal
    hyperparameter rule nor performance on a new domain. The longer-run reference
    has a different budget and cannot be mixed into a controlled live A/B claim.

    **Version-matched official references:** [Python 3.12 dictionaries](https://docs.python.org/3.12/library/stdtypes.html#mapping-types-dict),
    [Python 3.12 min and its key/first-tie behavior](https://docs.python.org/3.12/library/functions.html#min),
    [PyTorch 2.8 cross-entropy](https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html).
    """)
    return



@app.cell
def _():
    def candidate_config(changed_factor,rate_value,penalty_value):
        # TODO 1: a two-key dictionary; change one factor from the stated baseline.
        config=None
        # END TODO 1
        return config
    def choose_validation(reports):
        # TODO 2: index of the smaller declared validation loss; no test access.
        index=None
        # END TODO 2
        return index

    return candidate_config, choose_validation


@app.cell(hide_code=True)
def _(candidate_config, change, day2, mo, penalty, rate, run_compare):
    comparison=None
    _error_message=None
    if run_compare.value:
        try:comparison=day2.compare_runs(candidate_config(change.value,rate.value,penalty.value))
        except (ValueError,TypeError,KeyError) as _error:_error_message=str(_error)
    mo.callout(_error_message,kind='warn') if _error_message else mo.callout('Complete candidate_config, then run the comparison.',kind='info') if comparison is None else mo.md('Two live CPU runs complete; choose on validation.')
    return (comparison,)


@app.cell(hide_code=True)
def _(comparison, day2, mo):
    _view=None
    if comparison is not None:
        _reports,_models=comparison
        _view=mo.vstack([day2.training_figure(_reports),mo.ui.table([{'candidate':i,**r['config'],'validation_loss':r['validation_loss'],'validation_accuracy':r['validation_accuracy']} for i,r in enumerate(_reports)])])
    _view
    return


@app.cell(hide_code=True)
def _(choose_validation, comparison, day2, mo):
    selected=None
    _message=None
    if comparison is not None:
        try:selected=day2.selection_index(choose_validation(comparison[0]),comparison[0])
        except (ValueError,TypeError,IndexError) as _error:_message=str(_error)
    mo.callout(_message,kind='warn') if _message else mo.md('Selection pending: complete choose_validation.') if selected is None else mo.md(f'Selected candidate {selected} by validation loss. Freeze this choice before test access.')
    return (selected,)


@app.cell(hide_code=True)
def _(mo):
    evaluate_test=mo.ui.run_button(label='Evaluate chosen model once on reserved test')
    evaluate_test
    return (evaluate_test,)


@app.cell(hide_code=True)
def _(comparison, day2, evaluate_test, mo, selected):
    test_report=None
    if evaluate_test.value and selected is not None and comparison is not None:test_report=day2.evaluate_reserved(comparison[1][selected])
    mo.md('Test has not been accessed in this comparison.') if test_report is None else mo.ui.table([{'test_examples':test_report['test_n'],'accuracy':test_report['accuracy']}])
    return (test_report,)


@app.cell(hide_code=True)
def _(day2, mo):
    import json
    _reference_path=day2.ASSETS/'longer-training-reference.json'
    _reference=json.loads(_reference_path.read_text()) if _reference_path.is_file() else None
    mo.vstack([mo.md('### Longer-run measured reference (separate preparation experiment)'), mo.md(_reference['label']), mo.ui.table(_reference['summary']), day2.training_figure(_reference['runs'], title='Measured 120-epoch preparation runs')]) if _reference else mo.md('Reference preflight required.')
    return


@app.cell(hide_code=True)
def _(mo):
    experiment_log=mo.ui.text_area(value="run | changed factor/value | train-curve evidence | validation loss | decision before test\nA | \nB | \nStress case | \nDecision + limitation | ",label='Comparison log — preserve run A before changing a control',full_width=True)
    mo.vstack([mo.md("### Keep your evidence while experimenting\nEach rerun replaces the current result. Copy measured values and name a visible feature or response field here before changing controls."),experiment_log])
    return (experiment_log,)


@app.cell(hide_code=True)
def _(display, experiment_log, mo):
    display.download(mo,experiment_log.value,{'columns':"run | changed factor/value | train-curve evidence | validation loss | decision before test"},'day2-block2-comparison-log',label='Download comparison log (baseline + changed runs)')
    return


@app.cell(hide_code=True)
def _(mo):
    observations=mo.ui.text_area(label='Findings: measured change, explanation and one limitation',full_width=True)
    observations
    return (observations,)


@app.cell(hide_code=True)
def _(
    change,
    comparison,
    display,
    mo,
    observations,
    penalty,
    prediction,
    rate,
    selected,
    test_report,
):
    display.download(mo,observations.value,{'prediction':prediction.value,'changed_factor':change.value,'rate':rate.value,'penalty':penalty.value,'runs':comparison[0] if comparison else None,'selected_index':selected,'test':test_report},'model-improvement-findings')
    return


if __name__ == "__main__":
    app.run()

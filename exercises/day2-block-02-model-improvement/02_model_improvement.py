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

    **30 minutes in pairs.** Parameters are learned; hyperparameters are chosen. Hold split, seed and budget fixed. Choose by validation loss; reserve test data until the decision is frozen. A lower training loss alone is insufficient.

    ### Investigate, do not just run

    1. Inspect the baseline settings (rate 0.01, penalty 0). Predict the training/validation curve change for either a rate or penalty intervention; write which outcome would contradict your hypothesis.

    2. Complete configuration and validation-selection TODOs. Run the supplied short CPU trainings with just that factor changed. Read both curves, not only the last number; record a point where their behavior differs.

    3. Swap roles. Test a second value of the SAME factor, holding the other setting, split, seed and budget fixed. Save both comparisons before rerunning; choose using the declared validation loss only.

    4. Look for a contrary result: a faster initial loss drop, better training fit or larger penalty that did not give the best validation outcome. Explain it using the measured short-run curves; if absent, inspect the labelled longer-run reference without mixing its budget into your live comparison.

    5. Freeze your candidate choice and justification, then use the reserved-test action once. Report that estimate without retuning. Explain what small data, one seed and the short budget prevent you from claiming.

    **Evidence log:** run | changed factor/value | train-curve evidence | validation loss | decision before test. Save a run before changing controls. Export both the current findings and your comparison log.

    0–4 min: read the takeaways, inspect inputs and predict. 4–12: implement the bounded TODOs and inspect a real result. 12–22: compare controlled runs; swap editor/checker at minute 15. 22–27: diagnose a stress case and choose. 27–30: agree, save and export evidence.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Compare Controlled Training Runs

    30-minute investigation after guided setup. Run two real short CPU trainings, select by validation and evaluate the chosen configuration once on reserved examples.

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
    mo.md("""
    ## TODO — configuration and validation selection
    Baseline: `learning_rate=0.01`, `penalty=0.0`. In `candidate_config`, return a dict with exactly those keys, changing only the selected factor to the given value. The supplied training loop uses the same pretrained start, split IDs, seed 42 and 24 epochs for both runs; penalty adds a declared squared norm of weight tensors, not generic AdamW decay.

    In `choose_validation`, return the index (0 or 1) with the smaller `validation_loss` in the two prepared report dictionaries. Do not select by training or test loss. After choosing, click the separate test action once and report its sample size/limits; do not tune to this result. Compare the live short curves with the labelled longer measured references below.
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

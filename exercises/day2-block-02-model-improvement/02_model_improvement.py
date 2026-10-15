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
    ## Exercise + breakout: your shared task

    **15 minutes in pairs after guided setup.** Make a controlled improvement decision without leaking test evidence into selection.

    1. Predict the effect of rate 0.03 or penalty 0.001; choose one factor only. Complete the configuration TODO and run the two supplied 24-epoch CPU trainings.

    2. Complete validation selection. Compare curves and the declared validation losses, reporting a result that supports or contradicts your hypothesis and an alternative explanation.

    3. Freeze the selected configuration, then use the reserved-test action once. Explain why using this test score for another tuning decision would invalidate its final-evaluation role.

    **Bring back:** Configuration/curve table, validation-based choice, one reserved-test result and an uncertainty statement.

    Swap editor/checker halfway. Record in the observation fields, save code and download findings. The complete task and setup are in this block’s `GROUP_TASK.md`. Prerequisite TODOs discussed in plenary belong to setup.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Compare Controlled Training Runs

    15-minute group task after guided setup. Run two real short CPU trainings, select by validation and evaluate the chosen configuration once on reserved examples.

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

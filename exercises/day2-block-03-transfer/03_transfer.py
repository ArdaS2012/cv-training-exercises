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
    ## Adaptation audit: prove which weights learned

    **30 minutes in pairs.** The head width follows target labels. Frozen means no parameter update, not identical activations. Fine-tuning selects a trainable subset; more changed weights do not guarantee better validation.

    ### Investigate, do not just run

    1. Sketch the target output shape and predict which parameter groups may change in head-only and last-block adaptation. Use the target label order shown in the notebook.

    2. Complete the head and trainable-group TODOs, leaving the first convolution frozen. Run the supplied real head update and bounded last-block update. Inspect names, counts and actual parameter deltas.

    3. Swap roles. Trace one frozen and one eligible group from configuration to measured delta, then compare held-out outcomes on the same split and record the extra fine-tuning budget. Explain the difference between permission to update and a measured nonzero change.

    4. Inspect a target example whose prediction changed, or document that no displayed decision changed. Explain how scores/activations can change while the first convolution stays fixed. Distinguish that evidence from a claim that every eligible coefficient changed.

    5. Recommend the simpler head-only baseline or selective route for this small target task. Cite a measured delta and validation result, then name the fresh domain-validation evidence that could reverse your decision.

    **Evidence log:** route | trainable groups/count | frozen delta | eligible delta | validation evidence. Save a run before changing controls. Export both the current findings and your comparison log.

    0–4 min: read the takeaways, inspect inputs and predict. 4–12: implement the bounded TODOs and inspect a real result. 12–22: compare controlled runs; swap editor/checker at minute 15. 22–27: diagnose a stress case and choose. 27–30: agree, save and export evidence.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Adapt a Pretrained Classifier

    30-minute investigation after guided setup. Fit a new head on frozen pretrained features, then compare a real selective last-block fine-tune and held-out predictions.

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
    mo.md(r"""
    ### Read the weight audit
    The backbone computes features $z=f_\theta(x)$; the new head computes scores $s=Wz+b$. If each image has $d$ features and the task has $c$ target classes, $W$ has shape $c\times d$, $b$ has length $c$ and the output has $c$ scores. Match their order to the target labels.

    Frozen means $\Delta\theta_F=0$ for the frozen subset $F$. New inputs can still produce different activations. Head-only fitting updates the new head; selective tuning also permits the chosen last block to change. Read eligibility from the group configuration and actual changes from before/after deltas. A nonzero delta proves an update, not an improvement; held-out target results answer the second question.
    """)
    return


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
    mo.callout(_message,kind='warn') if _message else mo.callout('Complete head/classes and trainable groups, then fit.',kind='info') if adaptation_report is None else mo.vstack([mo.ui.table(adaptation_report['routes']),day2.adaptation_figure(adaptation_report),mo.ui.table(adaptation_report['validation_comparison']),mo.md('```json\n'+__import__('json').dumps(adaptation_report,indent=2)+'\n```')])
    return (adaptation_report,)


@app.cell(hide_code=True)
def _(mo):
    experiment_log=mo.ui.text_area(value="route | trainable groups/count | frozen delta | eligible delta | validation evidence\nA | \nB | \nStress case | \nDecision + limitation | ",label='Comparison log — preserve run A before changing a control',full_width=True)
    mo.vstack([mo.md("### Keep your evidence while experimenting\nEach rerun replaces the current result. Copy measured values and name a visible feature or response field here before changing controls."),experiment_log])
    return (experiment_log,)


@app.cell(hide_code=True)
def _(display, experiment_log, mo):
    display.download(mo,experiment_log.value,{'columns':"route | trainable groups/count | frozen delta | eligible delta | validation evidence"},'day2-block3-comparison-log',label='Download comparison log (baseline + changed runs)')
    return


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

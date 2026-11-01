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

    **Around 30 minutes in pairs.** The head width follows target labels. Frozen means no parameter update, not identical activations. Fine-tuning selects a trainable subset; more changed weights do not guarantee better validation.

    ### Investigate, do not just run

    1. Sketch the target output shape and predict which parameter groups may change in head-only and last-block adaptation. Use the target label order shown in the notebook.

    2. Complete the head and trainable-group TODOs, leaving the first convolution frozen. Run the supplied real head update and bounded last-block update. Inspect names, counts and actual parameter deltas.

    3. Swap roles. Trace one frozen and one eligible group from configuration to measured delta, then compare held-out outcomes on the same split and record the extra fine-tuning budget. Explain the difference between permission to update and a measured nonzero change.

    4. Inspect a target example whose prediction changed, or document that no displayed decision changed. Explain how scores/activations can change while the first convolution stays fixed. Distinguish that evidence from a claim that every eligible coefficient changed.

    5. Recommend the simpler head-only baseline or selective route for this small target task. Cite a measured delta and validation result, then name the fresh domain-validation evidence that could reverse your decision.

    **Evidence log:** route | trainable groups/count | frozen delta | eligible delta | validation evidence. Save a run before changing controls. Export both the current findings and your comparison log.
    """)
    return






@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Adapt a Pretrained Classifier

    Investigation after guided setup. Fit a new head on frozen pretrained features, then compare a real selective last-block fine-tune and held-out predictions.

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
    mo.md(r"""
    ## Your implementation task

    Describe the target head and selective training route using two small return
    values: a class count and a Boolean group dictionary. The source task recognized
    ten digits; the target vocabulary here is even/odd. Keep the first convolution
    fixed and allow the last convolution plus new head to train. The guide below
    defines logits, group names, shapes and the measured evidence needed to audit
    that choice. The actual training and frozen-feature baseline are supplied.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TODO guide — describe the target head and the trainable subset

    A pretrained **backbone** converts pixels into features; a task **head** converts
    those features into one score per target class. **Head-only fitting** keeps the
    backbone fixed. **Selective fine-tuning** additionally permits a chosen backbone
    part to update. Here the training loops, new-layer construction, cached features
    and before/after measurement are supplied. Your functions return small
    configuration values; they do not perform training.

    ### TODO 1: `target_head_classes()`

    This function has **no arguments**. The source model recognized digit IDs `0` to
    `9`. The new target is parity, with the complete ordered label vocabulary:

    | Target label ID / score column | Meaning |
    | --- | --- |
    | `0` | Even digit |
    | `1` | Odd digit |

    Return one positive Python `int`: the number of scores the new head needs for
    this target vocabulary. A **logit** is a raw, unnormalized score. The supplied
    cross-entropy/argmax classification route expects one logit per listed class;
    it does not use the alternative single-logit binary-classification convention.
    The index of the highest score becomes the predicted label ID. Do not return
    the source-class count, a list of labels, an input-feature dimension or a layer.

    For a batch of `B` images, each with `d=48` backbone features, the head is

    $$S=ZW^T+b.$$

    `Z` has shape `(B, d)`; `W` has shape `(c, d)`; `b` has shape `(c,)` and is applied
    to every row; the returned scores `S` have shape `(B, c)`. Here `c` is the integer
    your function returns. Batch is the image axis; class columns follow the label
    table. Changing the target vocabulary changes `c`, not the backbone feature width.

    ### TODO 2: `selective_trainable_groups()`

    This function also has **no arguments**. It describes the requested experiment:
    **keep the first convolution fixed; allow the last convolution and target head
    to train**. Return a new ordinary Python dictionary with exactly these string keys:

    | Exact key | Supplied model group and parameters |
    | --- | --- |
    | `conv1` | First convolution, from 1 grayscale input channel to 8 output feature channels; contains `conv1.weight` and `conv1.bias`. |
    | `conv2` | Last convolution, from 8 input to 12 output feature channels; contains `conv2.weight` and `conv2.bias`. |
    | `head` | New linear target classifier, from 48 features to `c` logits; contains `head.weight` and `head.bias`. |

    Every value must be the Python Boolean `True` or `False`. `True` grants that
    group permission to update; `False` freezes all its parameters. Strings such as
    `"True"` or `"frozen"` and integers such as `1` are not this configuration type.
    Choose the flags that express the requested experiment. Do not change global
    models, call an optimizer or alter a weight tensor inside this function.

    The local adapter applies these flags by mapping each parameter's group prefix
    to its `requires_grad` setting, and includes enabled groups in the optimizer.
    The frozen-feature baseline is already supplied; your dictionary configures the
    separate selective route. It is not a table of measured changes.

    ### How to prove what changed

    For a parameter group `G`, freezing requires

    $$\Delta\theta_G=\theta_{G,after}-\theta_{G,before}=0.$$

    `theta_G` denotes all weight/bias entries in that group. The report measures the
    largest absolute entry change **per named parameter tensor**, in the tensor's
    own parameter units. A nonzero measurement proves at least one entry changed;
    it does not prove that every eligible entry changed or that validation improved.
    Different input images may produce different activations through frozen weights.

    The prepared `day2.adapt(head_classes, trainable_groups)` consumes your integer
    and dictionary; it returns the report shown below. Relevant fields are:

    | Report field | Type and how to read it |
    | --- | --- |
    | `trainable_names` | List of full parameter-name strings enabled in the selective route. Compare the prefixes with your flags. |
    | `fine_tune_max_absolute_delta` | Dictionary from each of the six full parameter-name strings in the table above to one nonnegative float. Deltas compare the selective model before/after its additional updates. |
    | `routes` | List of two records, each with `route: str`, `trainable_parameters: int`, `validation_loss: float` (finite and nonnegative, with no upper bound of 1) and `validation_accuracy: float` (fraction in `[0, 1]`). |
    | `validation_comparison` | List of records with `image_id: int`, `target: int`, `head_only: int`, `last_block: int` and `changed: bool`. Label values use the parity table; `changed` compares the two predictions. |
    | `budgets` | Dictionary with `head_epochs: int` and `fine_tune_epochs: int`, currently `35` and `12`. |

    These fields are read for interpretation; you do not construct them. The source
    pretraining and target-example IDs are disjoint. Both routes share the fitted
    head start; the selective route then receives an extra short training budget,
    with a smaller backbone rate. Therefore the measured validation difference does
    not isolate "unfreezing alone" at an equal total budget.

    ### Before running

    Replace `classes=None` and `groups=None` only inside their TODO regions. Keep the
    signatures and return lines; **Shift+Enter** reruns the cell. No imports are needed
    for integer/Boolean/dictionary values. Press **Fit head and briefly fine-tune last
    block** after both functions are ready. Confirm the frozen group has zero delta,
    an eligible group actually changed, and the class-column interpretation matches
    the target table. Use the held-out parity results to justify a conditional route
    choice; this small related-domain experiment does not establish transfer quality
    for medical, industrial or other unseen domains.

    **Version-matched official references:** [Python 3.12 dictionaries and Boolean values](https://docs.python.org/3.12/library/stdtypes.html),
    [PyTorch 2.8 Linear and its tensor shapes](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Linear.html),
    [PyTorch 2.8 Module / named_parameters](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Module.html).
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

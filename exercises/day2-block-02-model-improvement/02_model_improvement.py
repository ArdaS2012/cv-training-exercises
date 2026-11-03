import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title='Build a training objective')


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path
    root=next(p for p in Path(__file__).resolve().parents if (p/'cv_exercises').is_dir())
    sys.path.insert(0,str(root))
    import marimo as mo
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from cv_exercises import day2, display
    api=(root/'exercises/day2-block-02-model-improvement/TASK_API.md').read_bytes()
    return api, day2, display, mo, np, plt, torch


@app.cell(hide_code=True)
def _(api, mo):
    mo.hstack([mo.md('# Build a training objective'),mo.download(api,filename='TASK_API.md',label='API reference (.md)')],justify='space-between')
    return


@app.cell(hide_code=True)
def _(day2, plt):
    _x,_y,_ids,_tr,_va,_te=day2.target_data()
    _fig,_axes=plt.subplots(1,4,figsize=(7,2),layout='constrained')
    for _ax,_i in zip(_axes,_tr[:4]):
        _ax.imshow(_x[_i,0],cmap='gray',vmin=0,vmax=1);_ax.set_title(f'Train target {int(_y[_i])}');_ax.axis('off')
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Build the training objective, compare one changed setting, then select on validation. The reserved test stays separate; no written run log.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1 · Cross-entropy
    `cross_entropy(logits,targets)`: finite CPU floating torch tensor `(B,C)`, B≥1/C≥2, and int64 targets `(B,)` in `0..C-1`. Return one differentiable scalar tensor: mean $[\log\sum_c e^{s_{ic}}-s_{i,y_i}]$. Use stable `torch.logsumexp`; preserve inputs and gradients.
    """)
    return


@app.cell
def _(torch):
    def cross_entropy(logits,targets):
        # TODO 1: per-row log-normalizer minus target score, then mean.
        loss=None
        # END TODO 1
        return loss
    return (cross_entropy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2 · Regularized objective
    `objective(logits,targets,weights,penalty)`: logits/targets as above; nonempty list of floating parameter tensors on CPU; penalty float `0..0.1`. Return differentiable scalar $L=CE+\lambda\sum_W\sum_j W_j^2$. Biases are excluded by the supplied list; no 1/2 factor. Use your cross_entropy; preserve inputs.
    """)
    return


@app.cell
def _(cross_entropy):
    def objective(logits,targets,weights,penalty):
        # TODO 2: data loss plus squared-weight penalty.
        loss=None
        # END TODO 2
        return loss
    return (objective,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3 · Choose on validation
    `choose_validation(reports)`: nonempty list of dicts; each has finite `validation_loss: float≥0`, `validation_accuracy: float[0,1]`, `config` with learning_rate/penalty floats, `curves` with train_loss/validation_loss/validation_accuracy float lists, epochs/seed ints. Return Python int original index of smallest validation_loss; first index wins ties. Never use test results.
    """)
    return


@app.cell
def _():
    def choose_validation(reports):
        # TODO 3: original index minimizing final validation loss.
        index=None
        # END TODO 3
        return index
    return (choose_validation,)


@app.cell(hide_code=True)
def _(mo):
    change=mo.ui.dropdown(['learning_rate','penalty'],value='learning_rate',label='Change one factor')
    rate=mo.ui.dropdown([.001,.03,.08],value=.03,label='Candidate rate')
    penalty=mo.ui.dropdown([.0001,.001,.01],value=.001,label='Candidate L2')
    run_compare=mo.ui.run_button(label='Run two short CPU trainings')
    mo.vstack([mo.hstack([change,rate,penalty]),run_compare])
    return change, penalty, rate, run_compare


@app.cell(hide_code=True)
def _(mo):
    get_training, set_training=mo.state(None)
    return get_training, set_training


@app.cell(hide_code=True)
def _(change, cross_entropy, day2, mo, objective, penalty, rate, run_compare, set_training, torch):
    _probe=cross_entropy(torch.tensor([[0.,0.]],requires_grad=True),torch.tensor([0]))
    if _probe is None:
        mo.output.replace(mo.md('Waiting: complete cross-entropy and objective, then Run.'))
    elif run_compare.value:
        _cfg={'learning_rate':.01,'penalty':0.}
        _cfg[change.value]=rate.value if change.value=='learning_rate' else penalty.value
        try:
            _comparison=day2.compare_runs(_cfg,objective=objective)
            if _comparison is not None:set_training(_comparison)
            else:mo.output.replace(mo.md('Objective unfinished: return a differentiable scalar tensor.'))
        except (ValueError,TypeError,RuntimeError,IndexError) as _e:mo.output.replace(mo.md(str(_e)))
    return


@app.cell(hide_code=True)
def _(choose_validation, day2, get_training, mo):
    comparison=get_training()
    selected=None
    if comparison is not None:
        selected=choose_validation(comparison[0])
        if selected is not None:selected=day2.selection_index(selected,comparison[0])
        _rows=[{'candidate':i,**r['config'],'validation_loss':r['validation_loss'],'validation_accuracy':r['validation_accuracy']} for i,r in enumerate(comparison[0])]
        _table='| Candidate | Rate | L2 | Validation loss | Accuracy |'+chr(10)+'|---|---:|---:|---:|---:|'+chr(10)+chr(10).join(f"| {r['candidate']} | {r['learning_rate']} | {r['penalty']} | {r['validation_loss']:.4f} | {r['validation_accuracy']:.3f} |" for r in _rows)
        mo.output.replace(mo.vstack([day2.training_figure(comparison[0]),mo.md(_table),mo.md(f'**Selected candidate:** {selected}. Last completed runs; displayed config is authoritative. Freeze the choice before test access.')]))
    return comparison, selected


@app.cell(hide_code=True)
def _(mo):
    evaluate_test=mo.ui.run_button(label='Evaluate selected model on reserved test')
    evaluate_test
    return (evaluate_test,)


@app.cell(hide_code=True)
def _(comparison, day2, evaluate_test, mo, selected):
    test_report=None
    if evaluate_test.value and comparison is not None and selected is not None:
        test_report=day2.evaluate_reserved(comparison[1][selected])
        mo.output.replace(mo.md(f"Reserved test: {test_report['test_n']} examples; accuracy {test_report['accuracy']:.3f}. Do not retune."))
    return (test_report,)


@app.cell(hide_code=True)
def _(comparison, display, mo, selected, test_report):
    _factor=None if comparison is None else next(k for k in ('learning_rate','penalty') if comparison[0][0]['config'][k]!=comparison[0][1]['config'][k])
    display.download(mo,'',{'factor':_factor,'runs':comparison[0] if comparison else None,'selected':selected,'test':test_report},'training-trial',label='Save trial')
    return


if __name__ == "__main__":
    app.run()

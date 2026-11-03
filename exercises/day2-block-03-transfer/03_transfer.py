import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title='Implement a transfer audit')


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
    api=(root/'exercises/day2-block-03-transfer/TASK_API.md').read_bytes()
    return api, day2, display, mo, np, plt, torch


@app.cell(hide_code=True)
def _(api, mo):
    mo.hstack([mo.md('# Implement a transfer audit'),mo.download(api,filename='TASK_API.md',label='API reference (.md)')],justify='space-between')
    return


@app.cell(hide_code=True)
def _(day2, plt):
    _x,_y,*_=day2.target_data()
    _fig,_axes=plt.subplots(1,4,figsize=(7,2),layout='constrained')
    for _ax,_i in zip(_axes,[0,30,60,90]):
        _ax.imshow(_x[_i,0],cmap='gray',vmin=0,vmax=1);_ax.set_title(f"Target {int(_y[_i]%2)}: {'even' if int(_y[_i])%2==0 else 'odd'}");_ax.axis('off')
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Configure which pretrained tensors can learn, implement the new head calculation, then audit actual parameter changes. Source: ten digits; target score columns: 0 even, 1 odd.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1 · New head scores
    `head_scores(features,weight,bias)`: CPU float tensors `(B,48)`, `(2,48)`, `(2,)`. Return differentiable logits `(B,2)` using $S=ZW^T+b$; rows are images, columns even/odd. Preserve tensors; `torch` is supplied.
    """)
    return


@app.cell
def _():
    def head_scores(features,weight,bias):
        # TODO 1: one linear score per target class.
        scores=None
        # END TODO 1
        return scores
    return (head_scores,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2 · Freeze the first convolution
    `configure(model)`: supplied PyTorch TinyCNN with `conv1`, `conv2`, `head`; head maps 48 features to two logits. Set each parameter's `requires_grad` from its group prefix: conv1 frozen, conv2/head enabled. Return the **same** model; preserve all weights. `named_parameters()` yields `(name,Parameter)` pairs such as `conv1.weight`; use `requires_grad_(bool)`.
    """)
    return


@app.cell
def _():
    def configure(model):
        # TODO 2: set gradient eligibility by parameter-name prefix.
        configured=None
        # END TODO 2
        return configured
    return (configure,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3 · Measure parameter changes
    `parameter_deltas(before,after)`: dictionaries with identical six keys `conv1.weight`, `conv1.bias`, `conv2.weight`, `conv2.bias`, `head.weight`, `head.bias`; values are finite CPU float tensors of matching shape per key. Return dict with same keys, each value Python float $\max_j|\theta_{after,j}-\theta_{before,j}|$. Preserve both snapshots; no empty tensors.
    """)
    return


@app.cell
def _():
    def parameter_deltas(before,after):
        # TODO 3: maximum absolute change per named tensor.
        deltas=None
        # END TODO 3
        return deltas
    return (parameter_deltas,)


@app.cell(hide_code=True)
def _(mo):
    run_adaptation=mo.ui.run_button(label='Fit head and fine-tune last block')
    run_adaptation
    return (run_adaptation,)


@app.cell(hide_code=True)
def _(configure, day2, head_scores, mo, np, parameter_deltas, plt, run_adaptation, torch):
    adaptation_report=None
    _model=day2.tiny_model()
    _features=torch.tensor(np.load(day2.ASSETS/'cached-features.npz')['features'][:4])
    _head=torch.nn.Linear(48,2)
    _scores=head_scores(_features,_head.weight,_head.bias)
    if _scores is None:
        mo.output.replace(mo.md('Waiting: complete head scores, freezing and audit.'))
    elif not isinstance(_scores,torch.Tensor) or _scores.shape!=(4,2):
        mo.output.replace(mo.md('Head scores must be a tensor (B,2).'))
    else:
        _gap=float((_scores-_head(_features)).detach().abs().max())
        mo.output.replace(mo.md(f'**Head vs torch Linear max difference:** {_gap:.3g}. Press Run after completing freezing/audit.'))
        if run_adaptation.value:
            try:
                adaptation_report=day2.adapt(2,{'conv1':False,'conv2':True,'head':True},configure=configure,audit=parameter_deltas)
                if adaptation_report is not None:
                    _delta=adaptation_report['fine_tune_max_absolute_delta']
                    _fig,_ax=plt.subplots(figsize=(9,2.8),layout='constrained')
                    _ax.bar(list(_delta),list(_delta.values()));_ax.tick_params(axis='x',rotation=20);_ax.set(ylabel='max absolute weight change',title='Your before/after audit')
                    mo.output.append(mo.vstack([_fig,day2.adaptation_figure(adaptation_report),mo.md('| Route | Trainable parameters | Validation loss | Accuracy |\n|---|---:|---:|---:|\n'+'\n'.join(f"| {r['route']} | {r['trainable_parameters']} | {r['validation_loss']:.4f} | {r['validation_accuracy']:.3f} |" for r in adaptation_report['routes']))]))
                else:mo.output.append(mo.md('Configure/audit unfinished.'))
            except (ValueError,TypeError,RuntimeError,KeyError) as _e:mo.output.append(mo.md(str(_e)))
    return (adaptation_report,)


@app.cell(hide_code=True)
def _(adaptation_report, display, mo):
    display.download(mo,'',{'report':adaptation_report},'transfer-trial',label='Save trial')
    return


if __name__ == "__main__":
    app.run()

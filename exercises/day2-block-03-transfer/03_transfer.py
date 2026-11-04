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
    **Problem.** A TinyCNN was pretrained on 1,497 other digit images for the 10-class task. Reuse it for a new task: **even vs odd** (target column 0 = even, 1 = odd), with 180 labelled rows. Two routes get the **same budget**: same pretrained body, same seeded new head, 150 full-batch Adam steps at rate 0.01.

    - **Frozen features:** only the new head learns.
    - **Last-block fine-tune:** conv2 and the head learn; conv1 stays frozen.

    You write the head that both routes use, the freezing rule and the audit that proves what changed.

    **Slide takeaways:** replace the head to match the new label space · freezing = no gradient updates · fine-tuning more layers adds capacity and risk · transfer only helps when source and target are related.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1 · New head scores
    `head_scores(features,weight,bias)`: CPU float tensors `(B,48)`, `(2,48)`, `(2,)`. Return differentiable logits `(B,2)` using $S=ZW^\top+b$; rows are images, columns even/odd. Both routes train **through your function**, so keep the autograd graph (no `.detach()`); preserve inputs. Operators: `@` is matrix multiply, `.T` transposes, `+` broadcasts the bias over rows.
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
    run_adaptation=mo.ui.run_button(label='Train both routes (150 steps each, about 2 s)')
    run_adaptation
    return (run_adaptation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4 · Stretch: domain-shift probe
    `shift(images)`: float tensor `(N,1,8,8)`, intensities 0..1 (dark background, bright strokes), N ≥ 1. Return a new tensor of the same shape with **inverted** intensities $x' = 1-x$ (bright background, dark strokes). Do not modify the input. Neither route sees inverted images during training; the table then reports accuracy on the inverted validation images.
    """)
    return


@app.cell
def _():
    def shift(images):
        # TODO 4 (stretch): invert every intensity.
        shifted=None
        # END TODO 4
        return shifted
    return (shift,)


@app.cell(hide_code=True)
def _(configure, day2, head_scores, mo, np, parameter_deltas, run_adaptation, shift, torch):
    adaptation_report=None
    _features=torch.tensor(np.load(day2.ASSETS/'cached-features.npz')['features'][:4])
    torch.manual_seed(0);_head=torch.nn.Linear(48,2)
    _scores=head_scores(_features,_head.weight,_head.bias)
    if _scores is None:
        mo.output.replace(mo.md('Waiting: complete TODO 1–3 (head scores, freezing, audit).'))
    elif not isinstance(_scores,torch.Tensor) or _scores.shape!=(4,2):
        mo.output.replace(mo.md('Head scores must be a tensor (B,2).'))
    else:
        _gap=float((_scores-_head(_features)).detach().abs().max())
        mo.output.replace(mo.md(f'**Your head vs torch Linear, max difference:** {_gap:.3g}. Press Run after completing freezing and audit.'))
        if run_adaptation.value:
            try:
                _probe=torch.zeros(1,1,8,8)
                _use_shift=shift if shift(_probe) is not None else None
                adaptation_report=day2.adapt(head_scores,configure,parameter_deltas,shift=_use_shift)
                if adaptation_report is not None:
                    _cols='| Route | Trainable parameters | Update steps | Validation loss | Validation accuracy |'+(' Inverted accuracy |' if _use_shift else '')
                    _sep='|---|---:|---:|---:|---:|'+('---:|' if _use_shift else '')
                    _rows=[f"| {r['route']} | {r['trainable_parameters']} | {r['update_steps']} | {r['validation_loss']:.3f} | {r['validation_accuracy']:.3f} |"+(f" {r['shifted_validation_accuracy']:.3f} |" if _use_shift else '') for r in adaptation_report['routes']]
                    mo.output.append(mo.vstack([day2.delta_figure(adaptation_report['fine_tune_max_absolute_delta']),mo.md('\n'.join([_cols,_sep]+_rows)),mo.md(f"Validation images: {sum(r['changed'] for r in adaptation_report['validation_comparison'])} of 60 predictions differ between the routes; those come first."),day2.adaptation_figure(adaptation_report),mo.md("""
    **Try:** (1) Which tensors show 0? That is your freezing rule working. (2) The fine-tune route trains about 10× more parameters with the same 150 steps: what did that buy on validation? (3) Is 60 validation images enough to call a 3-point difference real? (4) Stretch: with inverted digits, chance level is 0.5. What does a result below 0.5 say about features that were learned on dark backgrounds?""")]))
                else:mo.output.append(mo.md('Configure or audit unfinished: return the model / the delta dictionary.'))
            except (ValueError,TypeError,RuntimeError,KeyError) as _e:mo.output.append(mo.md(str(_e)))
    return (adaptation_report,)


@app.cell(hide_code=True)
def _(adaptation_report, display, mo):
    display.download(mo,'',{'report':adaptation_report},'transfer-trial',label='Save trial')
    return


if __name__ == "__main__":
    app.run()

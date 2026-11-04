import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title='Make overfitting visible')


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
    mo.hstack([mo.md('# Make overfitting visible'),mo.download(api,filename='TASK_API.md',label='API reference (.md)')],justify='space-between')
    return


@app.cell(hide_code=True)
def _(day2, mo, plt):
    _x,_y,_ids,_tr,_va,_te=day2.target_data();_rows=day2.small_train_rows()
    _fig,_axes=plt.subplots(1,10,figsize=(10,1.4),layout='constrained')
    for _ax,_i in zip(_axes,_rows[::5]):
        _ax.imshow(_x[_i,0],cmap='gray',vmin=0,vmax=1);_ax.set_title(f'digit {int(_y[_i])}',fontsize=8);_ax.axis('off')
    mo.vstack([mo.md(f"""
    **Problem.** A randomly initialized TinyCNN (1,446 parameters) sees only **{len(_rows)} training images** (5 per digit), so it can memorize them. You write the loss, add an L2 penalty, implement early stopping, then compare one regularizer against the baseline. Validation ({len(_va)} images) chooses; the reserved test ({len(_te)} images) is opened once at the end.

    **Slide takeaways:** training loss alone cannot detect overfitting · validation selects, test reports · change one factor at a time · $L=\\mathrm{{CE}}+\\lambda\\sum W^2$.
    """),_fig,mo.md('One training image per digit. Before running: will the baseline validation loss keep falling for all 150 epochs? No written answer needed.')])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1 · Cross-entropy
    `cross_entropy(logits,targets)`: `logits` finite float CPU tensor `(B,C)`, rows = images, columns = class scores, B≥1, C≥2; `targets` int64 tensor `(B,)` with class IDs `0..C-1` (not one-hot). Return one differentiable scalar tensor (shape `()`):

    $\mathrm{CE}=\frac1B\sum_{i}\big[\log\sum_c e^{s_{ic}}-s_{i,y_i}\big]$

    $s_{ic}$ is the score of image i for class c and $y_i$ its target ID. `torch.logsumexp(logits,dim=1)` gives the stable log-normalizer, one value per row. Keep the autograd graph: no `.item()`, `.detach()` or NumPy. Do not modify inputs.
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
def _(cross_entropy, mo, torch):
    _s=torch.tensor([[2.,0.,-1.],[0.,0.,0.]],requires_grad=True);_t=torch.tensor([0,2])
    _mine=cross_entropy(_s,_t)
    if _mine is None:_v=mo.md('Waiting: complete TODO 1.')
    elif not isinstance(_mine,torch.Tensor) or _mine.ndim!=0:_v=mo.md('Return one scalar tensor (shape `()`), not a list or float.')
    else:
        _ref=torch.nn.functional.cross_entropy(_s,_t)
        _v=mo.md(f"Check: yours {float(_mine.detach()):.4f} · PyTorch {float(_ref.detach()):.4f} · gradient kept: {_mine.requires_grad}. Row 2 has equal scores, so its own loss is log 3 ≈ 1.0986.")
    _v
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2 · Regularized objective
    `objective(logits,targets,weights,penalty)`: logits/targets as in TODO 1; `weights` a nonempty Python list of float parameter tensors (conv and head **weights**; biases are already excluded); `penalty` float λ in `0..0.1`. Return the differentiable scalar

    $L=\mathrm{CE}+\lambda\sum_{W\in\text{weights}}\sum_j W_j^2$

    Use your `cross_entropy`. No 1/2 factor, no division by parameter count. The supplied trainer calls this every epoch for both runs; the plotted curves are plain CE, so a penalty does not inflate them directly.
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
    ### 3 · Early stopping
    `best_epoch(validation_losses)`: nonempty Python list of finite floats ≥ 0, one data cross-entropy per epoch; index 0 = after the first update. Return a Python `int`: the index of the smallest value; the **first** index wins ties. Schema example: `[0.9, 0.5, 0.5, 0.7]` → `1`.

    The supplied code applies your rule to each run, keeps that epoch's checkpoint, and picks the run with the lower validation loss at its chosen epoch. Test data are never used here.
    """)
    return


@app.cell
def _():
    def best_epoch(validation_losses):
        # TODO 3: index of the lowest validation loss (first on ties).
        index=None
        # END TODO 3
        return index
    return (best_epoch,)


@app.cell(hide_code=True)
def _(mo):
    factor=mo.ui.dropdown(['penalty','dropout','learning_rate'],value='penalty',label='Change one factor')
    penalty=mo.ui.dropdown(['0.001','0.003','0.01'],value='0.003',label='L2 λ')
    dropout=mo.ui.dropdown(['0.3','0.5'],value='0.5',label='Dropout p')
    rate=mo.ui.dropdown(['0.003','0.03'],value='0.003',label='Adam rate')
    seed=mo.ui.dropdown(['0','1','2'],value='0',label='Seed (same for A and B)')
    run_compare=mo.ui.run_button(label='Train A and B (about 2 s)')
    mo.vstack([mo.md('**Controlled comparison.** A = baseline (Adam rate 0.01, λ 0, dropout 0). B = baseline with one factor changed. Same seed, 50 rows, 150 epochs and optimizer. Only the dropdown of the selected factor is used.'),mo.hstack([factor,penalty,dropout,rate,seed]),run_compare])
    return dropout, factor, penalty, rate, run_compare, seed


@app.cell(hide_code=True)
def _(mo):
    get_training, set_training=mo.state(None)
    return get_training, set_training


@app.cell(hide_code=True)
def _(cross_entropy, day2, dropout, factor, mo, objective, penalty, rate, run_compare, seed, set_training, torch):
    _probe=cross_entropy(torch.tensor([[0.,0.]],requires_grad=True),torch.tensor([0]))
    if _probe is None:
        mo.output.replace(mo.md('Waiting: complete TODO 1 and TODO 2, then train.'))
    elif run_compare.value:
        _value={'penalty':penalty,'dropout':dropout,'learning_rate':rate}[factor.value].value
        try:
            _comparison=day2.compare_runs(day2.candidate_config(factor.value,_value),objective,seed=int(seed.value))
            if _comparison is not None:set_training(_comparison)
            else:mo.output.replace(mo.md('Objective unfinished: return a differentiable scalar tensor.'))
        except (ValueError,TypeError,RuntimeError,IndexError) as _e:mo.output.replace(mo.md(str(_e)))
    return


@app.cell(hide_code=True)
def _(best_epoch, day2, get_training, mo):
    comparison=get_training()
    selection=None
    if comparison is not None:
        _reports=comparison[0];_err=None
        try:selection=day2.select_checkpoint(_reports,best_epoch)
        except (ValueError,TypeError) as _e:_err=str(_e)
        _lines=['| Run | Changed | Final train CE | Final val CE | Best epoch | Val CE at best | Val acc at best |','|---|---|---:|---:|---:|---:|---:|']
        for _i,_r in enumerate(_reports):
            _changed=', '.join(f"{k}={_r['config'][k]:g}" for k in day2.B2_BASELINE if _r['config'][k]!=day2.B2_BASELINE[k]) or 'baseline'
            _base=f"| {'AB'[_i]} | {_changed} | {_r['curves']['train_loss'][-1]:.4f} | {_r['final_validation_loss']:.3f} |"
            _p=selection['per_run'][_i] if selection else None
            _lines.append(_base+(f" {_p['epoch']} | {_p['validation_loss']:.3f} | {_p['validation_accuracy']:.3f} |" if _p else ' – | – | – |'))
        if selection:_sel=f"**Selected:** run {'AB'[selection['selected_run']]}, epoch {selection['selected_epoch']} (validation only). Freeze this choice before opening the test."
        else:_sel=_err or 'Waiting: complete TODO 3 to mark best epochs and select a checkpoint.'
        mo.output.replace(mo.vstack([day2.training_figure(_reports,selection),mo.md('\n'.join(_lines)),mo.md(_sel),mo.md("""
    **Try:** (1) Where does A's validation loss turn upward while its train loss keeps falling? That gap is overfitting. (2) Change seed 0 → 1 → 2: does the pattern hold? (3) Compare λ 0.001 with 0.01, then dropout 0.5: which delays the turn, which raises the final train loss? (4) Rate 0.003: slower fitting is not the same as regularization.""")]))
    return comparison, selection


@app.cell(hide_code=True)
def _(mo):
    evaluate_test=mo.ui.run_button(label='Evaluate selected checkpoint on reserved test (once)')
    evaluate_test
    return (evaluate_test,)


@app.cell(hide_code=True)
def _(comparison, day2, evaluate_test, mo, selection):
    test_report=None
    if evaluate_test.value and comparison is not None and selection is not None:
        _model=day2.checkpoint_model(comparison[0],comparison[1],selection['selected_run'],selection['selected_epoch'])
        test_report=day2.evaluate_reserved(_model)
        mo.output.replace(mo.md(f"Reserved test: {test_report['test_n']} images; accuracy {test_report['accuracy']:.3f} for run {'AB'[selection['selected_run']]}, epoch {selection['selected_epoch']}. Report it; do not retune."))
    return (test_report,)


@app.cell(hide_code=True)
def _(comparison, display, mo, selection, test_report):
    _runs=None if comparison is None else comparison[0]
    display.download(mo,'',{'runs':_runs,'selection':selection,'test':test_report},'training-trial',label='Save trial')
    return


if __name__ == "__main__":
    app.run()

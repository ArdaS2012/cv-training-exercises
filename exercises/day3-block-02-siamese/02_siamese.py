import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Build a similarity decision")


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(next(p for p in Path(__file__).resolve().parents if (p/'cv_exercises').is_dir())))
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import json
    import torch
    from cv_exercises import day3, display
    reference=(next(p for p in Path(__file__).resolve().parents if (p/'cv_exercises').is_dir())/'exercises/day3-block-02-siamese/TASK_API.md').read_bytes()
    return day3, display, json, mo, np, plt, reference, torch


@app.cell(hide_code=True)
def _(mo, reference):
    mo.hstack([mo.md('# Build a similarity decision'),mo.download(reference,filename='TASK_API.md',label='API reference (.md)')],justify='space-between',align='center')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Build the pair distance, the contrastive loss and the threshold decision of a Siamese model, then follow one loss gradient in embedding space. Compare cutoffs .4 → .7 → 1.0 on the same 24 real digit pairs, first clean, then with pixel noise. Fast pairs continue with choosing a cutoff (TODO 5).

    **Slide takeaways:** one shared encoder $f_\theta$ for both images · $d=\lVert f_\theta(x_a)-f_\theta(x_b)\rVert_2$ · $L=y\,d^2+(1-y)\max(0,m-d)^2$ pulls similar pairs together and pushes different pairs to at least the margin · the cutoff is a decision rule chosen after training: lower → more missed matches, higher → more false matches.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    threshold=mo.ui.slider(0.,2.,step=.05,value=.7,show_value=True,label='Match cutoff')
    pair_set=mo.ui.dropdown({'Clean pairs':0.,'Noisy pairs (σ = .3)':.3},value='Noisy pairs (σ = .3)',label='Pair set')
    mo.hstack([pair_set,threshold])
    return pair_set, threshold


@app.cell(hide_code=True)
def _(day3, pair_set, plt):
    pair_data, left, right=day3.pairs(pair_set.value)
    _fig,_axes=plt.subplots(2,4,figsize=(7,3),layout='constrained')
    for _i in range(4):
        for _row,_key in enumerate(['a','b']):_axes[_row,_i].imshow(pair_data[_key][_i].reshape(8,8),cmap='gray');_axes[_row,_i].axis('off')
        _axes[0,_i].set_title('Pair '+str(_i))
    _fig
    return left, pair_data, right


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1 · Pair distance
    `a,b`: finite float NumPy `(P,D)` with P,D≥1; matching rows are the two embeddings of one pair. Return float NumPy `(P,)`, $d_i=\sqrt{\sum_j(a_{ij}-b_{ij})^2}$. Features are dimensionless; unit vectors in this dataset yield distances 0–2. Equal vectors give zero; preserve inputs.
    """)
    return


@app.cell
def _(np, torch):
    def distance(a, b):
        # TODO 1: one Euclidean distance per matching row.
        result=None
        # END TODO 1
        return result
    return (distance,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2 · Contrastive loss
    `d`: nonnegative float NumPy `(P,)`; aligned `y`: binary NumPy `(P,)`, 1=similar, 0=different; `margin`: positive float. Return float NumPy `(P,)`: $L_i=y_i d_i^2+(1-y_i)\max(0,m-d_i)^2$, m=margin. At/above margin, a different pair pays zero. No averaging or extra ½; preserve inputs.
    """)
    return


@app.cell
def _(np, torch):
    def contrastive(d, y, margin):
        # TODO 2: calculate a loss for each pair.
        result=None
        # END TODO 2
        return result
    return (contrastive,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3 · Count threshold errors
    Same `d,y` as above; `cutoff`: float ≥0. Accept a match when d≤cutoff, including equality. Return `{'false_matches': int, 'missed_matches': int}` using Python ints: accepted y=0 pairs and rejected y=1 pairs. Preserve all inputs. Compare the supplied threshold sweep and inspect image pairs.
    """)
    return


@app.cell
def _(np, torch):
    def errors(d, y, cutoff):
        # TODO 3: threshold the distances and count both error types.
        result=None
        # END TODO 3
        return result
    return (errors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4 · Follow the contrastive gradient
    `a,b: float[P,D]`, `d: float[P]` and binary `y: int/float[P]` are aligned embeddings, their Euclidean distances and targets from above. `margin`: positive float; `rate`: nonnegative float (runner .1). Return new float NumPy `(P,D)` left embeddings $a'_i=a_i-\eta g_i$, with $g_i=2y_i(a_i-b_i)-2(1-y_i)\max(0,m-d_i)(a_i-b_i)/d_i$. Set the whole gradient to zero at d=0; preserve inputs. This bounded embedding-space step is not encoder retraining; outputs need not stay unit normalized.
    """)
    return


@app.cell
def _(np, torch):
    def embedding_step(a, b, d, y, margin, rate):
        # TODO 4: apply one contrastive gradient step to the left embeddings.
        result=None
        # END TODO 4
        return result
    return (embedding_step,)


@app.cell(hide_code=True)
def _(contrastive, day3, distance, embedding_step, errors, left, mo, np, pair_data, pair_set, plt, right, threshold):
    pair_result=None
    _data,_a,_b=pair_data,left,right
    _dist=distance(_a,_b)
    _view=mo.md('Waiting: complete TODO 1–4 (distance, contrastive loss, threshold errors, embedding step).')
    if _dist is not None:
        _loss=contrastive(_dist,_data['similar'],1.)
        _counts=errors(_dist,_data['similar'],threshold.value)
        _moved=embedding_step(_a,_b,_dist,_data['similar'],1.,.1)
        if _loss is not None and _counts is not None and _moved is not None:
            _prepared=day3.pair_report(distance,contrastive,threshold.value,pair_set.value)
            _cuts=np.linspace(0,2,41)
            _sweep=[errors(_dist,_data['similar'],float(_t)) for _t in _cuts]
            _fig,_axes=plt.subplots(1,3,figsize=(11,2.8),layout='constrained')
            for _target,_color,_label in [(1,'#2563eb','similar'),(0,'#dc2626','different')]:
                _axes[0].hist(_dist[_data['similar']==_target],bins=np.linspace(0,2,16),alpha=.55,color=_color,label=_label)
            _axes[0].axvline(threshold.value,color='black');_axes[0].set(xlabel='Distance',ylabel='Pairs');_axes[0].legend()
            _axes[1].plot(_cuts,[_r['false_matches'] for _r in _sweep],label='False matches')
            _axes[1].plot(_cuts,[_r['missed_matches'] for _r in _sweep],label='Missed matches')
            _axes[1].axvline(threshold.value,color='black');_axes[1].set(xlabel='Cutoff',ylabel='Pairs');_axes[1].legend()
            _after=contrastive(distance(_moved,_b),_data['similar'],1.)
            _axes[2].bar(['Before','After step'],[float(np.mean(_loss)),float(np.mean(_after))]);_axes[2].set(ylabel='Mean pair loss',title='Embedding step, rate .1')
            pair_result={'pair_set':pair_set.value,'loss_before':float(np.mean(_loss)),'loss_after_embedding_step':float(np.mean(_after)),'cutoff':threshold.value,'errors':_counts,'mean_loss':float(np.mean(_loss)),'pairs':_prepared[0]['pairs']}
            _view=mo.vstack([_fig,_prepared[1],mo.md(f"**Errors at {threshold.value:.2f}:** {_counts} · mean loss {np.mean(_loss):.4f}"),mo.md('''Left: distance histogram for similar (blue) and different (red) pairs with your cutoff. Middle: your false/missed counts over all cutoffs. Right: mean pair loss before/after one embedding step (rate .1). The step moves embeddings directly, not the saved encoder; no test-accuracy claim.

**Try:** (1) Noisy pairs: move the cutoff .4 → .7 → 1.0. Which error type rises, which falls? Then switch to Clean pairs: why is every cutoff from .1 to 1.1 perfect? (2) Where do the two histograms overlap, and which pair thumbnails sit there? (3) Why does a different pair beyond the margin contribute zero loss? (4) Why did one embedding step lower the mean loss without changing any weight?''')])
    _view
    return (pair_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5 · Stretch: choose a cutoff
    `pick_cutoff(d, y, cutoffs)`: distances `d` float NumPy `(P,)` ≥ 0; aligned binary targets `y` NumPy `(P,)`, 1 = similar; `cutoffs`: nonempty float NumPy `(T,)` in ascending order. Accept a match when d ≤ cutoff. Return the Python `float` cutoff with the smallest total errors (false matches + missed matches); on ties return the first (smallest) such cutoff. Preserve inputs. Choosing a cutoff on these pairs makes them validation pairs: a fair error estimate needs fresh test pairs.
    """)
    return


@app.cell
def _(np, torch):
    def pick_cutoff(d, y, cutoffs):
        # TODO 5: choose the cutoff with the fewest total errors.
        result=None
        # END TODO 5
        return result
    return (pick_cutoff,)


@app.cell(hide_code=True)
def _(distance, errors, left, mo, np, pair_data, pick_cutoff, right):
    _d=distance(left,right);_cuts=np.linspace(0,2,41)
    _c=None if _d is None else pick_cutoff(_d,pair_data['similar'],_cuts)
    if _c is None or errors(_d,pair_data['similar'],.7) is None:
        _v=mo.md('Optional: complete TODO 1, 3 and 5 to choose a cutoff from the error counts.')
    else:
        _e=errors(_d,pair_data['similar'],_c)
        _v=mo.md(f"**Selected cutoff:** {_c:.2f} · errors there: {_e} · largest similar distance {float(_d[pair_data['similar']==1].max()):.3f} · smallest different distance {float(_d[pair_data['similar']==0].min()):.3f}.\n\n**Try:** is the selected cutoff unique, or does a whole interval give the same errors? Which cutoff would you deploy if a false match were ten times worse than a missed match?")
    _v
    return


@app.cell(hide_code=True)
def _(display, mo, pair_result):
    display.download(mo,'',{'report':pair_result},'day3-siamese',label='Save trial')
    return


if __name__ == "__main__":
    app.run()

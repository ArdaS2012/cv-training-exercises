import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Build and audit a projection")


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path
    root = next(p for p in Path(__file__).resolve().parents if (p/'cv_exercises').is_dir())
    sys.path.insert(0,str(root))
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from cv_exercises import data, display
    task_reference=(root/'exercises/day1-block-02-reduction/TASK_API.md').read_bytes()
    return data, display, mo, np, plt, task_reference


@app.cell(hide_code=True)
def _(mo, task_reference):
    mo.vstack([mo.hstack([mo.md('# Build and audit a projection'),mo.download(task_reference,filename='TASK_API.md',label='API reference (.md)')],align='center'),mo.md('Build a compressor, select its rank, then measure which image neighbours survive a 2D view. Keep code and plots; no written report.')])
    return


@app.cell(hide_code=True)
def _(data, mo, plt):
    x,y,sample_ids=data.images()
    _fig,_ax=plt.subplots(figsize=(2.5,2.2),layout='constrained')
    _ax.imshow(x[0].reshape(8,8),cmap='gray',vmin=0,vmax=1);_ax.axis('off');_ax.set_title('Original · row 0')
    components=mo.ui.slider(1,64,value=8,show_value=True,label='Retained directions k')
    mo.vstack([_fig,components])
    return components, sample_ids, x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1 · Center the images
    Remove the mean image before finding directions: $X_c=X-\mu$, where $\mu$ is the mean across image rows.

    `center_images(x)`: finite float `(n,d)`, n,d ≥ 1; rows are images, columns pixels. Return `(centered, mean)`, floats `(n,d)` and `(d,)`. Preserve x. Here x is `(300,64)`, 8×8 row-major pixels in [0,1]. `np` is supplied.
    """)
    return


@app.cell
def _(np):
    def center_images(x):
        # TODO 1: calculate one mean per pixel and subtract it from every image.
        centered=mean=None
        # END TODO 1
        return centered,mean
    return (center_images,)


@app.cell(hide_code=True)
def _(center_images, mo, np, x):
    centered,mean=center_images(x)
    centered_ok=(isinstance(centered,np.ndarray) and centered.shape==x.shape and np.issubdtype(centered.dtype,np.floating) and np.isfinite(centered).all() and isinstance(mean,np.ndarray) and mean.shape==(64,) and np.issubdtype(mean.dtype,np.floating) and np.isfinite(mean).all())
    singular=vt=None
    if centered_ok:
        _,singular,vt=np.linalg.svd(centered,full_matrices=False)
    mo.md('SVD directions ready.' if centered_ok else 'Waiting: return centered images (300,64) and a mean image (64,).')
    return centered, centered_ok, mean, singular, vt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2 · Keep the strongest directions
    Retain the first k supplied SVD directions as columns. They define the compression budget.

    `vt`: float `(64,64)`, strongest direction first in rows, or None while step 1 waits. `components.value`: integer k=1..64. Return float `basis: (64,k)`, preserving direction order and inputs; otherwise None. Use slicing and transpose `.T`.
    """)
    return


@app.cell
def _(components, vt):
    basis=None
    if vt is not None:
        # TODO 2: select k direction rows and arrange them as columns.
        basis=None
        # END TODO 2
    return (basis,)


@app.cell(hide_code=True)
def _(basis, components, np):
    basis_ok=isinstance(basis,np.ndarray) and basis.shape==(64,components.value) and np.issubdtype(basis.dtype,np.floating) and np.isfinite(basis).all()
    return (basis_ok,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3 · Compress
    Project centered images onto the retained directions: $Z=X_cB$, with image rows in $X_c$ and basis columns in $B$.

    `centered: float(300,64)`, `basis: float(64,k)`, supplied `basis_ok: bool`. Return float `projected: (300,k)`, same image order; leave None until ready. `@` is matrix multiplication. Preserve inputs.
    """)
    return


@app.cell
def _(basis, basis_ok, centered):
    projected=None
    if basis_ok:
        # TODO 3: calculate each image's coordinates in the retained basis.
        projected=None
        # END TODO 3
    return (projected,)


@app.cell(hide_code=True)
def _(components, np, projected):
    projected_ok=isinstance(projected,np.ndarray) and projected.shape==(300,components.value) and np.issubdtype(projected.dtype,np.floating) and np.isfinite(projected).all()
    return (projected_ok,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4 · Reconstruct
    Reverse the projection and restore image brightness: $\hat X=ZB^T+\mu$.

    `projected: float(300,k)`, `basis: float(64,k)`, `mean: float(64,)`, supplied `projected_ok: bool`. Return float `reconstructed: (300,64)`, same image/pixel order; leave None until ready. Use `@`, `.T`, `+`; preserve inputs, do not clip values.
    """)
    return


@app.cell
def _(basis, mean, projected, projected_ok):
    reconstructed=None
    if projected_ok:
        # TODO 4: reconstruct pixel rows and restore the mean image.
        reconstructed=None
        # END TODO 4
    return (reconstructed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5 · Measure the loss
    Compute mean squared pixel error: $MSE=\frac{1}{nd}\sum_{i,j}(X_{ij}-\hat X_{ij})^2$; i indexes images, j pixels.

    `reconstruction_error(original, restored)`: equal finite float arrays `(n,d)`, n,d ≥ 1. Return one nonnegative Python float, in intensity units squared. Preserve inputs. `np.mean`, subtraction and elementwise squaring are available.
    """)
    return


@app.cell
def _(np):
    def reconstruction_error(original,restored):
        # TODO 5: average squared differences over all images and pixels.
        error=None
        # END TODO 5
        return error
    return (reconstruction_error,)


@app.cell(hide_code=True)
def _(components, mo, np, plt, reconstructed, reconstruction_error, singular, x):
    reconstruction_ok=isinstance(reconstructed,np.ndarray) and reconstructed.shape==x.shape and np.issubdtype(reconstructed.dtype,np.floating) and np.isfinite(reconstructed).all()
    mse=pca_error_ok=variance_fraction=None
    _view=mo.md('Waiting: finish the five steps to reveal your reconstruction and error map.')
    if reconstruction_ok:
        mse=reconstruction_error(x,reconstructed)
        if mse is not None and isinstance(mse,(int,float,np.floating)) and np.isfinite(mse) and mse>=0:
            _expected=float(np.sum(singular[components.value:]**2)/x.size)
            pca_error_ok=bool(np.isclose(mse,_expected,rtol=1e-7,atol=1e-10))
            variance_fraction=float(np.sum(singular[:components.value]**2)/np.sum(singular**2))
            _fig,_axes=plt.subplots(1,3,figsize=(8,2.8),layout='constrained')
            for _ax,_image,_title in zip(_axes[:2],[x[0],reconstructed[0]],['Original',f'Your reconstruction · k={components.value}']):
                _ax.imshow(_image.reshape(8,8),cmap='gray',vmin=0,vmax=1);_ax.axis('off');_ax.set_title(_title)
            _map=_axes[2].imshow(np.abs(x[0]-reconstructed[0]).reshape(8,8),cmap='magma',vmin=0,vmax=1)
            _axes[2].axis('off');_axes[2].set_title('Absolute pixel error');_fig.colorbar(_map,ax=_axes[2],shrink=.65)
            _check='SVD residual check passed.' if pca_error_ok else 'Check mean, directions and error axes: SVD residual check failed.'
            _view=mo.vstack([_fig,mo.md(f'**Your MSE {mse:.6f}** · SVD benchmark {_expected:.6f} · retained variance {variance_fraction:.1%}. {_check}'),mo.md('Compare k **1 → 8 → 32**, then 64. Which details survive? Lower average error does not guarantee every detail.')])
        else:
            _view=mo.md('Step 5: return one finite nonnegative float error.')
    _view
    return mse, pca_error_ok, reconstruction_ok, variance_fraction


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6 · Choose the smallest compression budget
    Select the **smallest** rank whose retained variance reaches the target:
    $C_k=\sum_{j=1}^k s_j^2/\sum_j s_j^2$, with singular values $s_j$ in decreasing order.

    `choose_rank(singular_values, target)`: finite nonnegative float `(r,)`, r ≥ 1, positive total squared energy; target is a float with 0 < target < 1. Return a Python int in 1..r, not a zero-based index. Preserve inputs. `np.cumsum`, `np.sum`, `np.flatnonzero` or `np.searchsorted` are available; full API in the download.
    """)
    return


@app.cell
def _(np):
    def choose_rank(singular_values, target):
        # TODO 6: accumulate squared singular values and find the first sufficient rank.
        rank=None
        # END TODO 6
        return rank
    return (choose_rank,)


@app.cell(hide_code=True)
def _(mo):
    target_variance=mo.ui.dropdown(options=[0.8,0.9,0.95,0.99],value=0.9,label='Target retained variance')
    target_variance
    return (target_variance,)


@app.cell(hide_code=True)
def _(choose_rank, components, mo, np, plt, singular, target_variance):
    chosen_rank=None
    _view=mo.md('Waiting: complete centering and rank selection to show the budget curve.')
    if singular is not None:
        chosen_rank=choose_rank(singular,target_variance.value)
        if isinstance(chosen_rank,(int,np.integer)) and not isinstance(chosen_rank,bool) and 1<=chosen_rank<=len(singular):
            _curve=np.cumsum(singular**2)/np.sum(singular**2)
            _sufficient=_curve[chosen_rank-1]>=target_variance.value
            _minimal=chosen_rank==1 or _curve[chosen_rank-2]<target_variance.value
            _fig,_ax=plt.subplots(figsize=(7,2.8),layout='constrained')
            _ax.plot(np.arange(1,len(singular)+1),_curve)
            _ax.axhline(target_variance.value,color='gray',ls='--',label='Target')
            _ax.axvline(chosen_rank,color='tab:green',label='Your selected rank')
            _ax.axvline(components.value,color='tab:orange',ls=':',label='Reconstruction slider')
            _ax.set(xlabel='Retained directions k',ylabel='Variance fraction',ylim=(0,1.02));_ax.legend()
            _status='Target met with the smallest rank.' if _sufficient and _minimal else 'Check both conditions: target reached AND previous rank below target.'
            _view=mo.vstack([_fig,mo.md(f'Your budget: **{chosen_rank} coordinates/image**. {_status} Set the reconstruction slider to this rank and inspect the digit. Compare targets 80% → 95%. This counts scores only; a shared basis and mean also need storage.')])
        elif chosen_rank is not None:
            _view=mo.md('Step 6: return one integer rank in 1..64.')
    _view
    return (chosen_rank,)


@app.cell(hide_code=True)
def _(mo):
    _explanation=mo.md(r"""
    ### What is the neighbourhood setting?
    A neighbour is **another image with a similar pixel vector**, not a nearby pixel inside one image. In the original data, similarity uses Euclidean distance over 64 normalized pixels. The layout tries to keep local relationships nearby in 2D.

    - **UMAP `n_neighbors`:** the neighbour count used to construct its local graph. 5 emphasizes a smaller local scale; 30 includes broader relationships.
    - **t-SNE `perplexity`:** an *effective* neighbour count derived from similarity probabilities, not a hard cutoff at exactly 5 or 30 images.

    This shared control supplies the number to both different parameters. Keep rows, preprocessing and seed 42 fixed; compare 5 → 30 **within each method**. Larger is not automatically better. Neither setting is a cluster count or PCA rank.
    """)
    neighbors=mo.ui.dropdown(options=[5,15,30],value=5,label='Local scale: t-SNE perplexity / UMAP n_neighbors')
    run_views=mo.ui.run_button(label='Run 2D views')
    mo.vstack([_explanation,mo.md('Change the local scale, then press Run. Optimizers are supplied; the audit below is your code.'),mo.hstack([neighbors,run_views],justify='start',wrap=True)])
    return neighbors, run_views


@app.cell(hide_code=True)
def _(display, neighbors, run_views, x):
    nonlinear=display.nonlinear_views(x,neighbors.value) if run_views.value else None
    return (nonlinear,)


@app.cell(hide_code=True)
def _(mo, nonlinear, plt, y):
    _view=None
    if nonlinear is not None:
        _fig,_axes=plt.subplots(1,2,figsize=(7,3),layout='constrained')
        for _ax,(_name,_points) in zip(_axes,nonlinear.items()):
            _ax.scatter(_points[:,0],_points[:,1],c=y,cmap='tab10',s=10)
            _ax.scatter(*_points[0],s=100,facecolors='none',edgecolors='black');_ax.set_title(_name+' · ring = row 0')
        _view=mo.vstack([_fig,mo.md('Same rows, seed 42. Labels only color the plot; islands do not establish classes. Re-run after changing the setting.')])
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7 · Find each image's nearest neighbours
    Build a neighbour table from distances: $D_{ij}=\sum_f(A_{if}-A_{jf})^2$; i,j are image rows, f is a feature. Squared distance gives the same ranking as Euclidean distance.

    `nearest_neighbor_ids(points, q)`: finite float `(n,d)`, n ≥ 2, d ≥ 1; integer q in 1..n−1. Return integer `(n,q)` **row indices**, closest first; exclude the image itself, retain distinct duplicate images, and break equal-distance ties by smaller row index. Preserve inputs. Here n=300, d=64 or 2, q=5. Loops or broadcasting are fine. `np.sum`, `np.argsort(..., axis=..., kind='stable')`, `np.fill_diagonal` and `np.inf` are available.
    """)
    return


@app.cell
def _(np):
    def nearest_neighbor_ids(points, q):
        # TODO 7: calculate distances, exclude self, and return the nearest row indices.
        ids=None
        # END TODO 7
        return ids
    return (nearest_neighbor_ids,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 8 · Audit what the projection preserved
    Measure the fraction of original neighbours retained for **each** image:
    $R_i=|N_q^{original}(i)\cap N_q^{view}(i)|/q$. The bars mean set size; order inside a neighbour list does not matter. Keep audit q=5 fixed while changing the optimizer's scale 5 → 30.

    `neighbor_retention(original_ids, view_ids)`: integer arrays `(n,q)`, n ≥ 2, 1 ≤ q < n. Each aligned row contains q distinct other row indices in 0..n−1, no self; order may differ. Return finite float `(n,)` in [0,1], not a single overall score. Preserve inputs. `np.intersect1d(a,b)` returns common values of two 1D arrays; Python sets/loops also work. Identical lists in different order give 1; disjoint lists give 0.
    """)
    return


@app.cell
def _(np):
    def neighbor_retention(original_ids, view_ids):
        # TODO 8: count shared neighbour IDs per image and normalize by q.
        retained=None
        # END TODO 8
        return retained
    return (neighbor_retention,)


@app.cell(hide_code=True)
def _(mo):
    inspected_row=mo.ui.slider(0,299,value=0,show_value=True,label='Inspect image row')
    inspected_view=mo.ui.dropdown(options=['t-SNE','UMAP'],value='t-SNE',label='View to audit')
    mo.hstack([inspected_row,inspected_view],wrap=True)
    return inspected_row, inspected_view


@app.cell(hide_code=True)
def _(inspected_row, inspected_view, mo, nearest_neighbor_ids, neighbor_retention, nonlinear, np, plt, x, y):
    audit_scores={}
    _view=mo.md('Waiting: implement steps 7–8 and run the 2D views. This audit works independently of your PCA code.')
    if nonlinear is not None:
        _original=nearest_neighbor_ids(x,5)
        _valid=lambda a: isinstance(a,np.ndarray) and a.shape==(300,5) and np.issubdtype(a.dtype,np.integer) and ((a>=0)&(a<300)).all() and all(len(set(row))==5 and i not in row for i,row in enumerate(a))
        if _valid(_original):
            _view_ids={}
            for _name,_points in nonlinear.items():
                _ids=nearest_neighbor_ids(_points,5)
                if _valid(_ids):
                    _score=neighbor_retention(_original,_ids)
                    if isinstance(_score,np.ndarray) and _score.shape==(300,) and np.issubdtype(_score.dtype,np.floating) and np.isfinite(_score).all() and ((_score>=0)&(_score<=1)).all():
                        audit_scores[_name]=_score
                        _view_ids[_name]=_ids
            if len(audit_scores)==2:
                _row=inspected_row.value;_name=inspected_view.value
                _fig,_axes=plt.subplots(2,6,figsize=(9,3.5),layout='constrained')
                for _axs,_ids,_caption in zip(_axes,[_original[_row],_view_ids[_name][_row]],['Original 64D','2D '+_name]):
                    _compared=_view_ids[_name][_row] if _caption=='Original 64D' else _original[_row]
                    for _col,(_ax,_id) in enumerate(zip(_axs,[_row,*_ids])):
                        _ax.imshow(x[_id].reshape(8,8),cmap='gray',vmin=0,vmax=1);_ax.axis('off')
                        _tag='Query' if _col==0 else ('Retained' if _id in _compared else ('Lost' if _caption=='Original 64D' else 'New'))
                        _ax.set_title(f'{_tag} #{_id}\ndigit {y[_id]}',fontsize=9)
                    _axs[0].set_ylabel(_caption)
                _fig.suptitle(f'Row {_row}: original neighbours (top) vs {_name} neighbours (bottom)',fontsize=11)
                _summary=' · '.join(f'{n}: mean {s.mean():.1%}, row {_row}: {s[_row]:.0%}' for n,s in audit_scores.items())
                _view=mo.vstack([_fig,mo.md(f'**Your neighbour retention (q=5):** {_summary}'),mo.md('Compare scale 5 → 30 for the same row, then inspect another row. Look for a lost neighbour despite matching digit colors. This score audits local pixel-distance relationships; it is not accuracy, trustworthiness, or proof of classes. Save each trial to compare measurements.')])
            else:
                _view=mo.md('Steps 7–8: return valid neighbour tables and one float retention value per image.')
        elif _original is not None:
            _view=mo.md('Step 7: return integer (300,5), unique other row indices per row, with no self neighbour.')
    _view
    return (audit_scores,)


@app.cell(hide_code=True)
def _(audit_scores, chosen_rank, components, display, inspected_row, mo, mse, neighbors, nonlinear, sample_ids, target_variance, variance_fraction):
    _values={'k':components.value,'neighbor_setting':neighbors.value,'seed':42,'sample_ids':sample_ids.tolist(),'svd_benchmark_variance_fraction':variance_fraction,'mse':mse,'target_variance':target_variance.value,'chosen_rank':None if chosen_rank is None else int(chosen_rank),'audit_q':5,'inspected_row':inspected_row.value,'neighbor_retention':{n:s.tolist() for n,s in audit_scores.items()}}
    if nonlinear is not None:_values['views']={n:v.tolist() for n,v in nonlinear.items()}
    display.download(mo,'',_values,'reduction-trial',label='Save trial')
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Build a K-Means Step")


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p/'cv_exercises').is_dir())))
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from cv_exercises import data, display
    task_api_bytes=(next(p for p in Path(__file__).resolve().parents if (p/'cv_exercises').is_dir())/'exercises/day1-block-01-clustering/TASK_API.md').read_bytes()
    return data, display, mo, np, plt, task_api_bytes


@app.cell(hide_code=True)
def _(mo, task_api_bytes):
    mo.hstack([mo.md("# Build a K-Means step"),mo.download(task_api_bytes,filename="TASK_API.md",label="API reference (.md)")],justify="space-between",align="center")
    return


@app.cell(hide_code=True)
def _(mo, np, plt):
    points=np.array([[0.,0.],[0.,2.],[2.,0.],[2.,2.],[6.,0.],[6.,2.],[8.,0.],[8.,2.]])
    initial=np.array([[0.,0.],[2.,2.]])
    iterations=mo.ui.slider(1,6,value=1,show_value=True,label='K-Means updates')
    _fig,_ax=plt.subplots(figsize=(6,2),layout='constrained')
    _ax.scatter(points[:,0],points[:,1],color='#64748b')
    for _i,_p in enumerate(points):_ax.annotate(str(_i),_p,xytext=(3,4),textcoords='offset points')
    _ax.scatter(initial[:,0],initial[:,1],marker='X',s=120,c=['#3b82f6','#ef4444'],edgecolors='black')
    _ax.set(xlim=(-1,9),ylim=(-1,3),aspect='equal',title='Eight points · X = starting centers 0 and 1')
    mo.vstack([mo.md('Which way will the centers move? No written answer needed.'),_fig,iterations])
    return initial, iterations, points


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1 · Assign points
    Implement nearest-centroid assignment using squared Euclidean distance: $c_i=\arg\min_k\sum_j(x_{ij}-\mu_{kj})^2$.

    `assign(points, centers)`: finite float arrays `(n,d)` and `(K,d)`, n,d,K ≥ 1; rows are samples/centers, columns are features in the same units. Return integer array `(n,)`, IDs `0..K-1`; ties choose the first center. Preserve inputs. `np` is supplied.
    """)
    return


@app.cell
def _(np):
    def assign(points, centers):
        # TODO 1: calculate distances and choose one center per point.
        labels=None
        # END TODO 1
        return labels
    return (assign,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2 · Move the centers
    Each center becomes the mean of its assigned points: $\mu_k'=\frac{1}{|C_k|}\sum_{i\in C_k}x_i$; $C_k$ contains the rows assigned to k.

    `update(points, labels, centers)`: points/centers as above; `labels: int[n]`, IDs `0..K-1`, aligned with point rows. Return a new float array `(K,d)` in center-ID order. Keep an empty cluster's old center; preserve all inputs. `np` is supplied.
    """)
    return


@app.cell
def _(np):
    def update(points, labels, centers):
        # TODO 2: average each cluster; retain its old center if empty.
        moved=None
        # END TODO 2
        return moved
    return (update,)


@app.cell(hide_code=True)
def _(assign, initial, iterations, mo, np, plt, points, update):
    from sklearn.cluster import KMeans as ReferenceKMeans
    manual_result=None
    _view=mo.md('Waiting: complete assign and update. The loop and sklearn comparison are supplied.')
    def _labels_ok(value):
        return isinstance(value,np.ndarray) and value.shape==(len(points),) and np.issubdtype(value.dtype,np.integer) and np.all((value>=0)&(value<len(initial)))
    def _centers_ok(value):
        return isinstance(value,np.ndarray) and value.shape==initial.shape and np.issubdtype(value.dtype,np.floating) and np.isfinite(value).all()
    _first=assign(points,initial.copy())
    if _first is not None:
        if not _labels_ok(_first):
            _view=mo.md('Assign: return an integer (n,) array with IDs 0..K-1.')
        else:
            _moved=update(points,_first,initial.copy())
            if _moved is not None and not _centers_ok(_moved):
                _view=mo.md('Update: return a finite float (K,d) array, one row per center.')
            elif _moved is not None:
                _centers=initial.copy()
                _valid=True
                for _ in range(iterations.value):
                    _ids=assign(points,_centers)
                    if not _labels_ok(_ids):
                        _valid=False
                        break
                    _next=update(points,_ids,_centers)
                    if not _centers_ok(_next):
                        _valid=False
                        break
                    _centers=_next
                _final=assign(points,_centers) if _valid else None
                if not _labels_ok(_final):
                    _view=mo.md('Check assign/update on every loop step; use the documented shapes and IDs.')
                else:
                    _reference=ReferenceKMeans(n_clusters=2,init=initial.copy(),n_init=1,max_iter=iterations.value,tol=0.,algorithm='lloyd',random_state=42).fit(points)
                    _loss=float(np.sum((points-_centers[_final])**2))
                    _gap=float(np.max(np.abs(_centers-_reference.cluster_centers_)))
                    _fig,_axes=plt.subplots(1,3,figsize=(9,2.8),layout='constrained',sharex=True,sharey=True)
                    for _ax,_ids,_c,_title in zip(_axes,[_first,_final,_reference.labels_],[_moved,_centers,_reference.cluster_centers_],['First assignment + update','Your loop','scikit-learn']):
                        _ax.scatter(points[:,0],points[:,1],c=_ids,cmap='coolwarm',vmin=0,vmax=1,s=35)
                        _ax.scatter(_c[:,0],_c[:,1],c=[0,1],cmap='coolwarm',vmin=0,vmax=1,marker='X',edgecolors='black',s=120)
                        _ax.set(title=_title,xlim=(-1,9),ylim=(-1,3),aspect='equal')
                    for _a,_b in zip(initial,_moved):
                        _axes[0].annotate('',xy=_b,xytext=_a,arrowprops={'arrowstyle':'->','color':'black'})
                    _axes[0].scatter(initial[:,0],initial[:,1],marker='+',c='black',s=80)
                    manual_result={'updates':iterations.value,'initial':initial.tolist(),'centers':_centers.tolist(),'labels':_final.tolist(),'inertia':_loss,'sklearn_inertia':float(_reference.inertia_),'center_max_difference':_gap}
                    _view=mo.vstack([_fig,mo.md(f'**Squared-distance sum:** yours {_loss:.3f} · sklearn {_reference.inertia_:.3f} · **max center difference:** {_gap:.3g}'),mo.md('Compare 1 → 6 updates. X = updated center; + = start. The first panel keeps the assignments used for its update; the other panels reassign to the final centers. Same start, no restarts; rounding-sized differences are normal. An empty cluster uses our keep-old rule, which differs from sklearn.')])
    _view
    return (manual_result,)


@app.cell(hide_code=True)
def _(mo):
    dataset=mo.ui.dropdown(['curved','noisy','compact','images'],value='curved',label='Data')
    epsilon=mo.ui.slider(.1,1.5,step=.1,value=.3,show_value=True,label='DBSCAN eps')
    mo.hstack([dataset,epsilon],justify='start',wrap=True)
    return dataset, epsilon


@app.cell(hide_code=True)
def _(data, dataset):
    from sklearn.preprocessing import StandardScaler
    raw=data.images()[0] if dataset.value=='images' else data.geometry(dataset.value)
    x=StandardScaler().fit_transform(raw)
    return raw, x


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3 · Configure the three methods
    Now test whether centroid distance follows a curved shape. Create estimator instances without fitting.

    `methods`: keys `'K-Means'`, `'Hierarchical'`, `'DBSCAN'` in that order. K=2; single linkage; K-Means seed 42, 10 starts; DBSCAN eps=`epsilon.value` (float .1–1.5), min_samples=5 including self. Leave other options at defaults.
    """)
    return


@app.cell
def _(epsilon):
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
    # TODO 3: replace None with configured estimators.
    methods={
        'K-Means':None,
        'Hierarchical':None,
        'DBSCAN':None,
    }
    # END TODO 3
    return (methods,)


@app.cell(hide_code=True)
def _(data, dataset, display, mo, plt, raw, x):
    if dataset.value=='images':
        _view=mo.vstack([
            display.image_preview(raw,data.images()[1]),
            mo.md("**What is clustered?** Each 8×8 image becomes one row of 64 pixel values. All 300 rows are standardized per pixel, then clustered by pixel similarity. The digit labels below are for inspection only; the algorithms never receive them. **Image #** is its dataset row number, not a cluster. K=2 asks for two groups across ten digits, not digit recognition.")])
    else:
        _view,_ax=plt.subplots(figsize=(5,2),layout='constrained')
        _ax.scatter(x[:,0],x[:,1],s=12,color='#64748b')
        for _i in [0,75]:_ax.annotate(str(_i),x[_i],xytext=(4,4),textcoords='offset points')
        _ax.set_aspect('equal');_ax.set_title('Which method follows both curves?')
    _view
    return



@app.cell(hide_code=True)
def _(methods):
    methods_ready=(isinstance(methods,dict) and list(methods)==['K-Means','Hierarchical','DBSCAN']
        and all(not isinstance(_m,type) and callable(getattr(_m,'fit_predict',None)) for _m in methods.values()))
    return (methods_ready,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4 · Fit and challenge
    Fit every method on the same samples. Compare curved data at eps .3 → .6, then switch to noisy data. Which grouping fails?

    `x: float64[n,d]`: finite standardized samples (n≥150, d=2; images: n=300, d=64). `methods`: instances from TODO 3. Return `results: dict[str,int[n]]` with the same keys/order, one label per input row. `fit_predict(x)` fits in place and returns labels; DBSCAN noise is -1. Preserve x.
    """)
    return


@app.cell
def _(methods, methods_ready, x):
    results=None
    if methods_ready:
        # TODO 4: use each estimator's fit_predict(x).
        results=None
        # END TODO 4
    return (results,)


@app.cell(hide_code=True)
def _(data, dataset, display, mo, np, raw, results, x):
    results_ready=(isinstance(results,dict) and list(results)==['K-Means','Hierarchical','DBSCAN']
        and all(isinstance(_v,np.ndarray) and _v.shape==(len(x),) and np.issubdtype(_v.dtype,np.integer) for _v in results.values()))
    if results_ready:
        if dataset.value=='images':
            _view=mo.vstack([
                mo.md("### After fitting: images sorted into their predicted groups\nSelect a method below. **Cluster 0 is an arbitrary group name, not digit 0**; IDs are independent across methods. Each row shows the group size, up to six examples spread across its member list, and its mean image (an average, not a new sample or necessarily a fitted center). The count chart includes every member of the displayed groups.\n\n**Try:** compare which digits share a group in K-Means and Hierarchical. Then select DBSCAN and change eps. Eps affects DBSCAN only; this small radius in 64 standardized dimensions may leave all images as noise. Noise −1 means unassigned images, not one cluster. The images do not change—only their memberships do."),
                mo.ui.tabs({_name:display.image_cluster_gallery(raw,_ids,data.images()[1],_name) for _name,_ids in results.items()})])
        else:
            _view=display.clustering_plot(x,results)
    else:
        _view=mo.md('Waiting: configure three methods, then return one integer label per row.')
    _view
    return (results_ready,)


@app.cell(hide_code=True)
def _(mo, np, results, results_ready):
    if results_ready:
        _lines=['| Method | Groups | Noise |','|---|---:|---:|']
        for _name,_labels in results.items():
            _lines.append(f'| {_name} | {len(set(_labels)-{-1})} | {int(np.sum(_labels==-1))} |')
        mo.output.replace(mo.md('\n'.join(_lines)))
    return


@app.cell(hide_code=True)
def _(dataset, display, epsilon, manual_result, mo, results, results_ready):
    _values={'dataset':dataset.value,'eps':epsilon.value,'k':2,'linkage':'single','min_samples':5,'standardized':True,'seed':42,'manual_kmeans':manual_result}
    if results_ready:_values['labels']={_n:_v.tolist() for _n,_v in results.items()}
    display.download(mo,'',_values,'clustering-trial',label='Save trial')
    return


if __name__ == "__main__":
    app.run()

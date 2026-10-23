import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="Compare Three Clustering Methods")


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p/'cv_exercises').is_dir())))
    import marimo as mo
    import numpy as np
    from cv_exercises import data, display

    return data, display, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Hands-on investigation: your shared mission

    **30 minutes in pairs.** Organize an unlabelled image collection: can your grouping rule survive a counterexample?

    ### Prior-slide takeaways to use now

    Slides 9–14: distance depends on scale; K-Means uses centroids, linkage uses merges, DBSCAN uses density. Cluster IDs are not class names.

    1. Predict which method will cut across the curved shapes. Use curved data, K=2, single linkage, standardization on and min_samples=5; complete the estimator and assignment TODOs. Trace one assignment or merge on the plots.

    2. Keep those rows/settings fixed. Compare DBSCAN eps=0.3 with 0.6. Log group count excluding noise, noise rows and one membership change; point to the affected area. Swap editor/checker at minute 15.

    3. Challenge your choice: keep the chosen settings and switch to noisy data, then inspect image thumbnails. Does the rule still serve your goal? Defend a choice with the comparison, a counterexample and a limit; never rename IDs as object classes.

    **Bring back:** An eps comparison log with counts and settings, one traced assignment/merge, a counterexample and a qualified grouping recommendation.

    **Route:** 0–4 predict; 4–12 implement and trace; 12–22 compare (swap roles at 15); 22–27 challenge and decide; 27–30 explain, save and export. Record each run in the evidence log before changing a control. Downloads include the current run plus your cumulative written log; save after each trial. See this block’s `GROUP_TASK.md`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Compare Three Clustering Methods

    30-minute investigation after guided setup. Compare K-Means, hierarchical clustering and DBSCAN on compact, curved and noisy geometries, then inspect grouped image thumbnails. Explain a centroid, a dendrogram cut and noise labels. Export one observation testing your room prediction.

    Inspect → predict → implement and trace → compare → challenge → explain and download evidence.

    Use the cell code control to open TODO cells. Edit only `# TODO` areas and run with Shift+Enter. Waiting messages are normal before completion. Save Python with Ctrl+S / Command+S; widget values need the findings download. Work in pairs and swap who edits.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    dataset=mo.ui.dropdown(options=['compact','curved','noisy','images'],value='compact',label='Dataset')
    k_clusters=mo.ui.slider(2,6,value=3,label='K / hierarchical cut')
    epsilon=mo.ui.slider(0.1,1.5,step=0.1,value=0.3,label='DBSCAN eps')
    min_points=mo.ui.slider(2,12,value=5,label='DBSCAN min_samples (includes self)')
    linkage_kind=mo.ui.dropdown(options=['single','complete','average','ward'],value='single',label='Linkage')
    scaling=mo.ui.checkbox(value=True,label='Standardize features')
    mo.hstack([dataset,k_clusters,epsilon,min_points,linkage_kind,scaling],wrap=True)
    return dataset, epsilon, k_clusters, linkage_kind, min_points, scaling


@app.cell(hide_code=True)
def _(mo):
    prediction=mo.ui.text_area(label='Prediction: which method suits this geometry, and why?',full_width=True)
    mo.vstack([mo.md("""## TODO: configure three estimators and assign their input rows
    Use the supplied constructors KMeans, AgglomerativeClustering and DBSCAN. KMeans takes n_clusters, random_state=42 and n_init=10. AgglomerativeClustering takes n_clusters and linkage. DBSCAN takes eps and min_samples. Then use each estimator’s fit_predict method on x. Keep dictionary keys unchanged."""),prediction])
    return (prediction,)


@app.cell
def _():
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
    # TODO 1: replace three None values with configured estimators.
    methods={'K-Means':None,'Hierarchical':None,'DBSCAN':None}
    # END TODO 1
    return (methods,)


@app.cell(hide_code=True)
def _(data, dataset, scaling):
    from sklearn.preprocessing import StandardScaler
    if dataset.value=='images':
        raw,unused_labels,unused_ids=data.images()
    else:
        raw=data.geometry(dataset.value)
    x=StandardScaler().fit_transform(raw) if scaling.value else raw.copy()
    return raw, x


@app.cell
def _():
    # TODO 2: replace None with a dictionary of fit_predict(x) results.
    results=None
    # END TODO 2
    return (results,)


@app.cell(hide_code=True)
def _(dataset, display, linkage_kind, methods, mo, np, raw, results, x):
    _view = None
    if results is None or any(m is None for m in methods.values()):
        _view = mo.callout('Waiting for TODO 1 and TODO 2. Configure methods, then assign each input row.',kind='info')
    elif set(results)!=set(methods) or any(np.asarray(a).shape!=(len(x),) for a in results.values()):
        _view = mo.callout('Return one label per input row for each of the three named methods.',kind='warn')
    else:
        _counts=[{'method':_name,'groups (excluding noise)':len(set(np.asarray(_labels).tolist())-{-1}),'noise rows':int(np.sum(np.asarray(_labels)==-1)),'total rows':len(_labels)} for _name,_labels in results.items()]
        _summary=mo.ui.table(_counts,selection=None)
        if dataset.value=='images':
            _view = mo.vstack([_summary,display.thumbnails(raw,results['K-Means']),mo.md('Rows are images. The first 12 thumbnails are fixed IDs; labels were never used to fit clusters. Record what the thumbnails support before naming any group.')])
        else:
            _view = mo.vstack([_summary,display.clustering_plot(x,results),display.dendrogram_plot(x,linkage_kind.value),mo.md('Locate a changed group/noise point, then log count and settings before changing eps. Group colors/IDs can permute; compare membership.')])
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    findings=mo.ui.text_area(value='Trial | dataset | K/linkage/scale/min_samples | eps | groups excluding noise | noise rows | located change\nBaseline | curved | 2/single/on/5 | 0.3 | | |\nComparison | curved | unchanged | 0.6 | | |\nCounterexample | noisy | chosen settings | | | |\nDecision and evidence:\nMechanism and limitation:',label="Evidence log — keep every trial, mechanism and decision",full_width=True)
    mo.vstack([findings])
    return (findings,)


@app.cell(hide_code=True)
def _(
    dataset,
    display,
    epsilon,
    findings,
    k_clusters,
    linkage_kind,
    min_points,
    mo,
    np,
    prediction,
    results,
    scaling,
):
    settings={'dataset':dataset.value,'k':k_clusters.value,'eps':epsilon.value,'min_samples':min_points.value,'linkage':linkage_kind.value,'standardized':scaling.value,'seed':42,'prediction':prediction.value}
    if results is not None:
        settings['labels']={n:np.asarray(v).tolist() for n,v in results.items()}
    display.download(mo,findings.value,settings,'clustering-findings')
    return


if __name__ == "__main__":
    app.run()

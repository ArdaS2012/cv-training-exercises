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
    ## Exercise + breakout: your shared task

    **15 minutes in pairs after guided setup.** Use geometry, parameters and counterexamples to assess clustering rather than recognize an algorithm name.

    1. Choose curved data with standardization on. Predict one failure, complete the short estimator/assignment TODOs and inspect K-Means, hierarchical and DBSCAN outputs on the same points.

    2. Keep all settings fixed except DBSCAN eps; compare 0.3 and 0.6. Record cluster membership/noise changes and use the dendrogram to explain one hierarchical merge.

    3. Recommend a method for this geometry, giving numerical/visual evidence and a failure risk. Check the supplied image-thumbnail view and explain why cluster IDs cannot be renamed as object classes without evidence.

    **Bring back:** Three-method comparison, eps/noise comparison, one merge/centroid observation and an evidence-based recommendation.

    Swap editor/checker halfway. Record in the observation fields, save code and download findings. The complete task and setup are in this block’s `GROUP_TASK.md`. Prerequisite TODOs discussed in plenary belong to setup.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Compare Three Clustering Methods

    15-minute group task after guided setup. Compare K-Means, hierarchical clustering and DBSCAN on compact, curved and noisy geometries, then inspect grouped image thumbnails. Explain a centroid, a dendrogram cut and noise labels. Export one observation testing your room prediction.

    Predict → complete the short TODOs → change one control → explain → download findings.

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
        if dataset.value=='images':
            _view = mo.vstack([display.thumbnails(raw,results['K-Means']),mo.md('Rows are images. The first 12 thumbnails are fixed IDs; labels were never used to fit clusters.')])
        else:
            _view = mo.vstack([display.clustering_plot(x,results),display.dendrogram_plot(x,linkage_kind.value)])
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    findings=mo.ui.text_area(label='Findings: centroids / merge heights / noise; one failed prediction and scaling consequence',full_width=True)
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

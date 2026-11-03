"""Prepared plotting and findings export; no student TODO solutions."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE
from umap import UMAP


def clustering_plot(x, results):
    fig, axs = plt.subplots(1,3,figsize=(11,3),layout='constrained')
    for ax,(name,labels) in zip(axs,results.items()):
        ax.scatter(x[:,0],x[:,1],c=labels,cmap='tab10',s=18)
        ax.set(title=name,xlabel='feature 1',ylabel='feature 2')
        ax.set_aspect('equal')
        noise = labels == -1
        ax.scatter(x[noise,0],x[noise,1],c='black',marker='x',s=45)
    return fig


def dendrogram_plot(x, method):
    fig, ax = plt.subplots(figsize=(9,3),layout='constrained')
    dendrogram(linkage(x,method=method),truncate_mode='lastp',p=12,ax=ax)
    ax.set(title=f'{method} linkage — truncated to 12 groups',ylabel='linkage dissimilarity')
    return fig


def thumbnails(x, labels):
    fig, axs = plt.subplots(2,6,figsize=(9,3),layout='constrained')
    for ax,i in zip(axs.flat,range(12)):
        ax.imshow(x[i].reshape(8,8),cmap='gray',vmin=0,vmax=1)
        ax.set_title(f'row {i} · group {labels[i]}'); ax.axis('off')
    return fig


def image_preview(x, digits):
    """One supplied example per known digit; labels are for inspection only."""
    fig, axs = plt.subplots(1, 10, figsize=(10, 1.8), layout='constrained')
    for digit, ax in enumerate(axs):
        i = np.flatnonzero(digits == digit)[0]
        ax.imshow(x[i].reshape(8, 8), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Digit {digit}\nimage #{i}', fontsize=9)
        ax.axis('off')
    fig.suptitle('Input: 300 images, 30 per digit — no clusters yet', fontsize=12)
    return fig


def image_cluster_gallery(x, labels, digits, method):
    """Inspect fitted memberships; true digit labels never enter the fit.

    x: normalized float[n,64]; labels: int[n] (noise=-1); digits: int[n],
    inspection labels 0..9. Shows at most eight groups, largest first,
    always retaining noise. Examples are evenly spaced in source-row order,
    not nearest-center representatives. Means use original intensities.
    """
    groups, counts = np.unique(labels, return_counts=True)
    ranked = sorted(zip(groups, counts), key=lambda item: (-item[1], item[0]))
    shown = [int(g) for g, _ in ranked if g != -1][:7 if -1 in groups else 8]
    if -1 in groups:
        shown.append(-1)
    omitted = len(groups) - len(shown)
    n = len(shown)
    fig = plt.figure(figsize=(10, 1.25*n + 3.4), layout='constrained')
    grid = fig.add_gridspec(n + 1, 9, height_ratios=[1]*n + [2.5])
    for row, group in enumerate(shown):
        ids = np.flatnonzero(labels == group)
        name = 'Noise (−1)' if group == -1 else f'Cluster {group}'
        color = '#555555' if group == -1 else plt.get_cmap('tab10')(shown.index(group) % 10)
        ax = fig.add_subplot(grid[row, :2])
        ax.text(0, .6, f'{name}\n{len(ids)} images ({len(ids)/len(x):.0%})',
                color=color, fontsize=11, va='center', weight='bold')
        ax.axis('off')
        examples = ids[np.linspace(0, len(ids)-1, min(6, len(ids)), dtype=int)]
        for col in range(6):
            ax = fig.add_subplot(grid[row, col+2])
            if col < len(examples):
                i = examples[col]
                ax.imshow(x[i].reshape(8, 8), cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'#{i} · digit {digits[i]}', fontsize=8)
            ax.axis('off')
        ax = fig.add_subplot(grid[row, 8])
        ax.imshow(x[ids].mean(axis=0).reshape(8, 8), cmap='gray', vmin=0, vmax=1)
        ax.set_title('Mean image', fontsize=8)
        ax.axis('off')
    ax = fig.add_subplot(grid[n, :])
    composition = np.array([[np.sum((labels == g) & (digits == d)) for d in range(10)] for g in shown])
    ax.imshow(composition, cmap='Blues', aspect='auto', vmin=0, vmax=max(1, int(composition.max())))
    for r in range(n):
        for d in range(10):
            value = composition[r, d]
            ax.text(d, r, str(value), ha='center', va='center', fontsize=9,
                    color='white' if value > composition.max()*.55 else 'black')
    ax.set_xticks(range(10), [str(d) for d in range(10)])
    ax.set_yticks(range(n), ['Noise (−1)' if g == -1 else f'Cluster {g}' for g in shown])
    ax.set(xlabel='Known digit — used only to inspect the result',
           title='Where did all images go? Each cell counts images of this digit in this group.')
    suffix = f' · {omitted} smaller groups omitted' if omitted else ''
    fig.suptitle(f'{method}: {len(groups) - int(-1 in groups)} clusters · {np.sum(labels == -1)} noise images{suffix}\n'
                 'Rows show clusters or unassigned noise; sample images stay unchanged.', fontsize=12)
    return fig


def nonlinear_views(x, neighbors=15):
    """Live CPU optimizers; same input rows, fixed seed; labels never fitted."""
    return {
        't-SNE': TSNE(n_components=2,perplexity=neighbors,random_state=42,max_iter=500,init='pca').fit_transform(x),
        'UMAP': UMAP(n_components=2,n_neighbors=neighbors,min_dist=0.1,random_state=42,n_jobs=1).fit_transform(x),
    }


def reduction_plot(x, reconstructed, singular, k, views, y):
    fig, axs = plt.subplots(2,3,figsize=(10,6),layout='constrained')
    axs[0,0].imshow(x[0].reshape(8,8),cmap='gray',vmin=0,vmax=1); axs[0,0].set_title('Input row 0')
    axs[0,1].imshow(reconstructed[0].reshape(8,8),cmap='gray',vmin=0,vmax=1); axs[0,1].set_title(f'Rank {k} reconstruction')
    ratios=singular**2/(singular**2).sum()
    axs[0,2].plot(np.arange(1,len(ratios)+1),np.cumsum(ratios)); axs[0,2].axvline(k,color='red')
    axs[0,2].set(title='Cumulative variance',xlabel='components',ylabel='fraction',ylim=(0,1.02))
    for ax,(name,z) in zip(axs[1,:2],views.items()):
        ax.scatter(z[:,0],z[:,1],c=y,cmap='tab10',s=8); ax.set_title(name+' · labels for inspection only')
    axs[1,2].axis('off'); axs[1,2].text(0,0.5,f'MSE = {np.mean((x-reconstructed)**2):.6f}\nIntensity units squared\nSame rows in both views')
    return fig


def masks_plot(rgb, masks):
    count = len(masks) + 1
    fig, axs = plt.subplots(1, count, figsize=(3.4 * count, 3), layout='constrained', squeeze=False)
    axs = axs[0]
    axs[0].imshow(rgb); axs[0].set_title('Authored scene')
    for ax,(name,mask) in zip(axs[1:],masks.items()): ax.imshow(mask,cmap='tab10',interpolation='nearest'); ax.set_title(name)
    for ax in axs: ax.axis('off')
    return fig


def download(mo, observations, values, name, label='Download findings (current controls + observations)'):
    payload=json.dumps({'observations':observations,'settings_and_measurements':values},indent=2)
    return mo.download(payload.encode(),filename=name+'.json',label=label)

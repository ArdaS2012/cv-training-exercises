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
    fig, axs = plt.subplots(1,3,figsize=(10,3),layout='constrained')
    axs[0].imshow(rgb); axs[0].set_title('Authored scene')
    for ax,(name,mask) in zip(axs[1:],masks.items()): ax.imshow(mask,cmap='tab10',interpolation='nearest'); ax.set_title(name)
    for ax in axs: ax.axis('off')
    return fig


def download(mo, observations, values, name, label='Download findings (current controls + observations)'):
    payload=json.dumps({'observations':observations,'settings_and_measurements':values},indent=2)
    return mo.download(payload.encode(),filename=name+'.json',label=label)

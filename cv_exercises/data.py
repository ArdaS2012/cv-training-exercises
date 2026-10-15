"""Prepared local data: sklearn digits and authored geometric examples."""
import numpy as np
from sklearn.datasets import load_digits, make_blobs, make_moons
from sklearn.model_selection import train_test_split


def images():
    """300 fixed 8x8 image rows; intensities normalized from 0..16 to 0..1."""
    data = load_digits()
    indices = np.concatenate([np.flatnonzero(data.target == c)[:30] for c in range(10)])
    return data.data[indices].astype(float) / 16, data.target[indices].copy(), indices


def classification_split():
    x, y, ids = images()
    train, rest = train_test_split(np.arange(len(y)), train_size=0.6, stratify=y, random_state=42)
    valid, test = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=43)
    return x, y, ids, train, valid, test


def geometry(name):
    if name == 'compact':
        x, _ = make_blobs(n_samples=150, centers=[[-2,-1],[2,1],[0,4]], cluster_std=0.4, random_state=42)
    else:
        x, _ = make_moons(n_samples=150, noise=0.05, random_state=42)
        if name == 'noisy':
            x = np.vstack([x, [[-1.5,1.6],[2.6,1.8],[0,-1.2],[3,-1]]])
    return x


def scene():
    """Authored RGB scene, not a real photograph; reproducible lighting gradient."""
    h, w = 32, 48
    yy, xx = np.mgrid[:h,:w]
    rgb = np.empty((h,w,3), dtype=float)
    rgb[:] = [0.20,0.45,0.75]
    rgb[yy > 20] = [0.20,0.60,0.25]
    rgb[(xx>7)&(xx<20)&(yy>11)&(yy<25)] = [0.85,0.18,0.15]
    rgb[(xx>31)&(xx<42)&(yy>4)&(yy<16)] = [0.85,0.18,0.15]
    rgb *= (0.5+0.5*xx/(w-1))[...,None]
    return rgb


def pixel_inputs(rgb):
    h, w, _ = rgb.shape
    yy, xx = np.mgrid[:h,:w]
    color = rgb.reshape(-1,3)
    position = np.column_stack([yy.ravel()/(h-1), xx.ravel()/(w-1)])
    return color, position

"""Supplied display/feedback for the pixel exercise; no student implementations."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def scene_plot(rgb):
    fig, ax = plt.subplots(figsize=(5, 2.5), layout='constrained')
    ax.imshow(rgb)
    for row, col, name in [(18, 12, 'A'), (10, 36, 'B')]:
        ax.plot(col, row, 'wo', markeredgecolor='black')
        ax.text(col+2, row, name, color='white', weight='bold')
    ax.axis('off')
    return fig


def feature_labels(features, color):
    if features is None:
        return None, 'Complete TODO 1.'
    values = np.asarray(features)
    if values.shape != (len(color), 5):
        return None, f'TODO 1: expected ({len(color)}, 5), got {values.shape}.'
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        return None, 'TODO 1: use finite numbers.'
    if not np.array_equal(values[:, :3], color):
        return None, 'TODO 1: preserve RGB in the first three columns.'
    return KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(values), 'Feature shape ready.'


def mask_feedback(mask, labels, image):
    if mask is None:
        return None, 'Complete TODO 2.'
    values = np.asarray(mask)
    if values.shape != image.shape[:2] or not np.issubdtype(values.dtype, np.integer):
        return None, f'TODO 2: expected integer {image.shape[:2]}.'
    if labels is None:
        return None, 'Complete TODO 1 first.'
    if not np.array_equal(values.ravel(), labels):
        return None, 'TODO 2: preserve pixel order.'
    return values, 'Mask ready.'


def selection_view(mo, rgb, mask, selected, point):
    if selected is None:
        return mo.md('Complete TODO 3.')
    values = np.asarray(selected)
    if values.shape != rgb.shape[:2] or values.dtype != bool:
        return mo.md('TODO 3: expected boolean (32, 48).')
    if not values[point]:
        return mo.md('TODO 3: include the seed pixel.')
    fig, ax = plt.subplots(figsize=(5, 2.5), layout='constrained')
    ax.imshow(np.where(values[..., None], rgb, rgb * .15))
    ax.axis('off')
    return mo.vstack([fig, mo.md(f'{int(values.sum())} pixels selected. A/B together: {bool(mask[18,12] == mask[10,36])}.')])

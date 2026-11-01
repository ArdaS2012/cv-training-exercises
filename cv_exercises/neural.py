"""Supplied numerical tools; students build the surrounding training pipeline."""
import numpy as np

# Small numerical tools for the end-to-end student pipeline.
def softmax(logits):
    """Finite (n,c) scores -> same-shape row probabilities; n,c > 0."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponents = np.exp(shifted)
    return exponents / exponents.sum(axis=1, keepdims=True)


def backward(x, labels, params, cache):
    """Mean cross-entropy/ReLU gradients, without updating parameters."""
    z, h, probabilities = cache
    dlogits = probabilities.copy()
    dlogits[np.arange(len(labels)), labels] -= 1
    dlogits /= len(labels)
    dz = (dlogits @ params['w2'].T) * (z > 0)
    return {'w1': x.T @ dz, 'b1': dz.sum(axis=0),
            'w2': h.T @ dlogits, 'b2': dlogits.sum(axis=0)}

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


def digit_examples():
    """Return a small figure of three supplied 8x8 input digits (not predictions)."""
    import matplotlib.pyplot as plt
    from . import data
    pixels, labels, _ = data.images()
    figure, axes = plt.subplots(1, 3, figsize=(6, 1.6), layout='constrained')
    for axis, row in zip(axes, [0, 30, 60]):
        axis.imshow(pixels[row].reshape(8, 8), cmap='gray', vmin=0, vmax=1)
        axis.set_title(f'True digit {labels[row]}')
        axis.axis('off')
    return figure


def pipeline_report(prepare_data, initialize, forward, mean_loss, fit, evaluate, *, steps, rate):
    """Run student callbacks; return (report, status). No TODO is implemented here.

    Every callback uses the student TASK_API contract. Training is performed by
    the student's fit function, never by this display/controller helper.
    """
    stage = 1
    try:
        dataset = prepare_data()
        if dataset is None:
            return None, 'Step 1: select the training and validation rows.'
        stage = 2
        params = initialize()
        if params is None:
            return None, 'Step 2: create the four parameter arrays.'
        stage = 3
        cache = forward(dataset['train_x'], params)
        if cache is None or any(part is None for part in cache):
            return None, 'Step 3: return pre-activations, activations and probabilities.'
        stage = 4
        if mean_loss(cache[2], dataset['train_y']) is None:
            return None, 'Step 4: return one mean loss.'
        stage = 6
        outcome = fit(dataset, params, steps, rate)
        if outcome is None:
            return None, 'Steps 5–6: connect gradients, updates and the loop.'
        trained, history = outcome
        if trained is None or history is None:
            return None, 'Steps 5–6: return parameters and loss history.'
        stage = 7
        _, baseline = evaluate(dataset, params)
        predictions, accuracy = evaluate(dataset, trained)
        if predictions is None or accuracy is None:
            return None, 'Step 7: return validation predictions and accuracy.'
        report = dict(loss=history, validation_accuracy=[float(baseline), float(accuracy)],
                      predictions=predictions.tolist(), targets=dataset['valid_y'].tolist(),
                      validation_ids=dataset['valid_ids'].tolist(), train_count=180,
                      validation_count=60, test_count=60, test_used=False, steps=steps, rate=rate)
        return report, f"Loss {history[-1]:.4f} · validation accuracy {accuracy:.1%} · test untouched."
    except (ValueError, TypeError, IndexError, KeyError, AttributeError) as error:
        return None, f'Step {stage}: check its inputs and return shape. {error}'


def pipeline_figure(report, image_row=0):
    """Plot one completed report and selected validation row; return a figure.

    report is the exact pipeline_report dictionary; image_row is an integer
    validation-row position 0..59, not a source ID. No training occurs here.
    """
    import matplotlib.pyplot as plt
    from . import data
    figure, axes = plt.subplots(1, 3, figsize=(10, 2.6), layout='constrained')
    axes[0].plot(report['loss'])
    axes[0].set(title='Training loss', xlabel='Updates', ylabel='Cross-entropy')
    axes[1].bar(['Before', 'After'], report['validation_accuracy'])
    axes[1].set(title='Validation accuracy', ylim=(0, 1))
    pixels, _, ids = data.images()
    source_id = report['validation_ids'][image_row]
    source_row = int(np.flatnonzero(ids == source_id)[0])
    axes[2].imshow(pixels[source_row].reshape(8, 8), cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f"ID {source_id}: predicted {report['predictions'][image_row]}\nTrue digit {report['targets'][image_row]}")
    axes[2].axis('off')
    return figure

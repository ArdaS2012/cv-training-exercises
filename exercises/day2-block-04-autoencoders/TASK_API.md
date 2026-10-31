# Task input and output contracts

## Function inputs and API reference

### TODO 1: `reconstruct(model, noisy)`

| Input | Definition and guarantees |
|---|---|
| `model` | Supplied pretrained PyTorch autoencoder (`nn.Module`) in evaluation mode. Its callable `encoder` accepts shape `(1, 64)` and emits shape `(1, k)`; its callable `decoder` accepts `(1, k)` and emits `(1, 64)`. The selected latent width `k` is `4` or `16`. These are attributes, not dictionary keys. |
| `noisy` | CPU `torch.Tensor`, floating-point shape `(1, 64)`: one noisy 8×8 grayscale image flattened in row-major order. Axis 0 is batch (one image), axis 1 is pixel position. Values have been clipped to `[0, 1]`. |

Return a CPU `torch.Tensor` with shape `(1, 64)`, containing the model's reconstructed normalized intensities. Preserve its batch axis. The prepared runner already disables gradient recording and loads weights; no fitting, image loading or tensor conversion is required. A latent code is the compact `k`-coordinate representation; it is not a class label.

**Calling a supplied module:** a module is a callable object. In generic notation, `layer(values)` invokes its forward computation and returns a tensor; a method such as `.parameters()` has a different purpose. The task is to connect the documented model interfaces; no solution expression is supplied.

### TODO 2: `pixel_mse(clean, reconstruction)`

Both arguments are finite floating-point **NumPy arrays**, shape `(1, 64)`, with corresponding pixels in the same row-major order and normalized intensity units `[0, 1]`. `clean` is the reference before corruption; `reconstruction` is one model output, not the noisy input. Return one finite nonnegative numeric scalar: mean squared pixel difference across **all** entries. Its units are normalized intensity squared. The mean, rather than a sum, makes this a per-pixel measure. Identical arrays have error zero; do not change either input.

**API vocabulary:** array subtraction and exponentiation are coordinatewise for matching arrays. `np.mean(a)` reduces all entries when `axis` is omitted; an explicit axis reduces only that dimension. `np` is already imported in the editable cell. The displayed error maps retain pixel positions, while this function returns a scalar summary.

**Official references:** [PyTorch 2.8 Module calls](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Module.html), [NumPy mean](https://numpy.org/doc/2.2/reference/generated/numpy.mean.html), [NumPy array arithmetic](https://numpy.org/doc/2.2/user/quickstart.html#basic-operations).

The same contracts are displayed beside the TODOs in the notebook. API references define available operations; selecting and combining them remains your task.

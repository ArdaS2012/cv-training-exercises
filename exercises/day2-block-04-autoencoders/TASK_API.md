# Task definitions and API help

## TODO guide — connect the autoencoder and measure reconstruction error

An **encoder** maps the input pixels to a shorter vector, the **latent code**.
A **decoder** maps that code back to estimated pixels, the **reconstruction**.
The models already have trained weights. Your first function connects their
provided interfaces; your second gives one average error against a clean target.
Neither TODO trains a model or removes noise by a hand-written image filter.

### TODO 1: `reconstruct(model, noisy)`

| Argument | Type, shape and meaning |
| --- | --- |
| `model` | Supplied pretrained PyTorch `nn.Module`, already loaded on CPU and set to evaluation mode. It has callable attributes `encoder` and `decoder`; they are not dictionary keys. The selected latent width `k` is `4` or `16`. |
| `noisy` | CPU `torch.Tensor`, dtype `torch.float32`, shape `(1, 64)`: one normalized noisy grayscale digit. Axis 0 is the one-image batch; axis 1 is flattened pixel position. All values are finite and clipped to `[0, 1]`. |

Flattening is **row-major**: the first eight values are the first row of the
original `8 × 8` image, then the second row, and so on. There is no separate
channel axis in this function. Preserve the one-image batch axis throughout.

The supplied local model API is:

| Callable | Accepted input | Returned value |
| --- | --- | --- |
| `model.encoder(values)` | CPU float32 tensor with shape `(1, 64)` in normalized intensity units. | CPU float32 tensor with shape `(1, k)`: latent features. These coordinates need not be in `[0, 1]` and are not class scores. |
| `model.decoder(code)` | The encoder's CPU float32 tensor with shape `(1, k)`. | CPU float32 tensor with shape `(1, 64)`: reconstructed intensities in `[0, 1]`, due to the supplied final sigmoid layer. |

A module is a callable object: generic syntax `layer(values)` runs a supplied
layer's forward computation. Accessing an attribute without calling it gives the
module object, not its output. The defining route is

$$z=f(x_{noisy}),\qquad \hat x=g(z).$$

`f` is the encoder, `g` the decoder, `z` the `k`-coordinate code, and `x_hat` the
reconstruction. Return that reconstruction as a CPU tensor of shape `(1, 64)`.
Do not return the code, the model object, a NumPy array or a reshaped `8 × 8`
image. Keep the input and weights unchanged. The runner already disables gradient
recording; device movement, fitting, `.backward()` and `.detach()` are not part of
this TODO. Different latent widths are supplied through the model, so avoid
hard-coding the code length.

### TODO 2: `pixel_mse(clean, reconstruction)`

This function receives **NumPy arrays**, not the tensors passed to TODO 1. The
runner performs that conversion for you after inference. Both arrays have the
same nonempty shape and corresponding finite numeric entries. In the live run
they are float32 arrays of shape `(1, 64)` in normalized intensity units `[0, 1]`;
a small arithmetic check may use a one-dimensional or smaller matching array.

`clean` is the original reference before corruption. `reconstruction` is the
model's output for the noisy input. Compare each reconstruction against `clean`,
not against the corrupted input. No element may be reordered or changed.

The **mean squared error** (MSE) is

$$\mathrm{MSE}=\frac{1}{d}\sum_{j=1}^{d}(x_j-\hat x_j)^2.$$

`d` is the total number of compared entries (64 in one live image), `x_j` is a
clean pixel and `x_hat_j` the reconstruction at the same position. Squaring makes
positive/negative errors contribute positively; the mean divides by the number
of entries. Return one finite nonnegative Python/NumPy numeric scalar, in
**normalized intensity squared** units. Identical arrays have MSE zero. A
coordinate error map retains positions; this function reduces it to one number.
There are no missing pixels, empty arrays or shape mismatches in the supplied
domain, so no masking or defensive input policy is required.

**NumPy toolbox:** matching-array subtraction and `**` exponentiation act on
corresponding entries. `np.mean(a, axis=None)` averages all entries and returns a
scalar; an explicit axis can leave a vector. `np` is already imported in the
editable cell. No tensor conversion or file loading belongs in this function.

### Before running and reading the evidence

Replace only `output=None` and `error=None` in their marked regions, keeping
function signatures/return lines. Rerun with **Shift+Enter**. The prepared runner
calls your reconstruction for both `clean` and `denoise` checkpoints at the
selected width, then calls your metric on each result. `None` leaves a waiting
message. A shape/finite-value check is a guard, not proof that the right arrays
were compared: verify the mean convention on your small hand example.

The clean-trained checkpoint learned clean-input/clean-target reconstruction;
the denoising checkpoint learned noisy-input/clean-target reconstruction with
training noise standard deviation `0.25`. Both receive the current noisy image at
inference. The same row and random seed keep the corruption controlled. Changing
the **noise standard deviation** scales noise in normalized-intensity units;
it does not update model weights. Width changes select separately trained models.

In the report, `clean_target_mse` maps `clean` and `denoise` to your scalar errors;
`noisy_input_mse` gives the corrupted-input error against the same clean target.
`latent_width` counts code coordinates and `coordinate_ratio` is `64/k`, a
coordinate-count ratio, not a compressed file size or bitrate. Inspect a chosen
stroke in the images/error maps alongside the scalar: background pixels can hide
a lost detail in an average. Distinct checkpoint histories limit causal claims
about width alone.

**Version-matched official references:** [PyTorch 2.8 Module calls](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Module.html),
[PyTorch 2.8 Sigmoid](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Sigmoid.html),
[NumPy 2.2 mean](https://numpy.org/doc/2.2/reference/generated/numpy.mean.html),
[NumPy array arithmetic](https://numpy.org/doc/2.2/user/quickstart.html#basic-operations).

The same guide is displayed immediately before the editable notebook cell. Complete only the marked TODO regions; the implementation remains your work.

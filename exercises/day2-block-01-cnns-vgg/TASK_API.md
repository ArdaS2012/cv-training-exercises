# CNN coding/API reference

The notebook contracts define TODO 1–4 (core) and TODO 5–6 (stretch). Arrays are NumPy, row/y then column/x; digit intensities are float32 in [0,1], learned kernel entries/bias may be negative. Convolution uses PyTorch cross-correlation (no kernel flip), unit dilation and zero padding. Its reference includes one grayscale input/output channel; this teaching implementation does not handle batches/channels. Returned arrays may use float64; inputs remain unchanged. Pooling consumes ReLU responses, discards incomplete windows, and has no padding.

`np.pad(a,p)` adds p zeros on all four sides of a 2D array; a.shape returns axis lengths. `np.empty((h,w),dtype=float)` allocates an output to fill. `np.sum`/`np.max` reduce entries. Basic slicing uses excluded stop indices. `np.maximum(a,0)` applies ReLU elementwise. Functions receive the same feature units as the images/maps, not file paths or widgets.

The supplied reference runs `torch.nn.functional.conv2d` and `max_pool2d` on NCHW tensors; manual/reference differences compare matching shapes. Plot input, convolution, reference and pooled response; inspect stride 1 versus 2 on the same image. Prepared `day2.cnn_inspect(response,output_size,index,stride)` takes your TODO 1 `patch_response` and your TODO 2 `output_size(n,k,p,s)` callables; it returns `(report, matplotlib.figure.Figure)` or None until unfinished callbacks are complete. Report contains image_id, target, prediction, patch_response, predicted_spatial_size, actual_stride_shape in NCHW, tiny_trace, vgg_partial_trace and vgg_full_trace lists of stage/shape records, plus a note. Full CNN/VGG maps use supplied pretrained checkpoints; only the isolated stride probe changes stride. VGG digit previews are out of its ImageNet domain; feature channels are not class labels.

Official installed-version APIs: [NumPy 2.2 pad](https://numpy.org/doc/2.2/reference/generated/numpy.pad.html), [sum](https://numpy.org/doc/2.2/reference/generated/numpy.sum.html), [indexing](https://numpy.org/doc/2.2/user/basics.indexing.html), [PyTorch 2.8 conv2d](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.conv2d.html), [max_pool2d](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.max_pool2d.html).

## TODO 2 — `output_size(n,k,p,s)`
Python ints, n ≥ 1, k ≥ 1, p ≥ 0, s ≥ 1, n+2p ≥ k. Return Python int $\lfloor (n+2p-k)/s\rfloor+1$, the number of window positions along one axis. `//` is floor division for nonnegative ints. `output_size(5,3,0,2)` → 2.

## TODO 5 (stretch) — `receptive_field(layers)`
`layers`: nonempty list of dicts in forward order, keys `'kernel'` (int ≥ 1) and `'stride'` (int ≥ 1); a 2×2 max-pool is `{'kernel':2,'stride':2}`. Return Python int r (input pixels along one axis that can affect one output). Start r=1, jump j=1; per layer r ← r + (k−1)·j, then j ← j·s. Padding and channels do not change r.

## TODO 6 (stretch) — `conv_parameters(k,c_in,c_out)`
Python ints ≥ 1. Return Python int $k^2 c_{in} c_{out}+c_{out}$ (weights plus one bias per output channel) for a square `Conv2d`. The notebook compares your counts with PyTorch `numel()`.

Receptive-field background: [Araujo et al., Distill 2019](https://distill.pub/2019/computing-receptive-fields/). VGG: [Simonyan & Zisserman 2014](https://arxiv.org/abs/1409.1556).

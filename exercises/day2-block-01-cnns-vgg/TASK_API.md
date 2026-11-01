# Task definitions and API help

## TODO guide — calculate a response, then count positions

The prepared runner supplies images, learned weights and plots. Your two functions
answer different questions: **what number does one filter produce at one position?**
and **how many positions fit along an image axis?** Neither function trains a model.

### TODO 1: `patch_response(patch, kernel, bias)`

A **patch** is a small rectangular window of image values. A **kernel** contains
one learned weight for each position in that window. A **bias** is one extra offset
for the whole response. Positive and negative weights can reinforce or cancel
image values; the result is a raw activation, so a negative response is valid.

| Argument | Type, structure and meaning |
| --- | --- |
| `patch` | Nonempty numeric NumPy array of shape `(R, C)`. Axis 0 is row/y; axis 1 is column/x. The live runner supplies a `(3, 3)` `float32` window from one grayscale digit, with intensities in `[0, 1]`. |
| `kernel` | Numeric NumPy array with exactly the same shape. `kernel[r, c]` weights `patch[r, c]`. Entries are finite, may be negative and are not intensities. The live runner supplies the first learned filter for one input/output channel. |
| `bias` | One finite numeric scalar, supplied as a Python float by the runner. It is added once, not once per pixel. |

Use the defining operation

$$y=\sum_{r=0}^{R-1}\sum_{c=0}^{C-1}x_{r,c}w_{r,c}+b.$$

Here `x` denotes `patch`, `w` denotes `kernel`, `b` is `bias`, and `y` is the one
response. Both sums together include every aligned coordinate. This is the
cross-correlation convention used by PyTorch: do not reverse the kernel.
Return one finite Python/NumPy numeric scalar; no row, column or channel axis
remains. Keep both input arrays unchanged. Use their supplied size rather than
assuming exactly nine entries: a small hand check may use another matching shape
and arbitrary finite numbers. Mismatched shapes and empty arrays are outside the
supplied task domain; no defensive validation is required.

**NumPy toolbox:** `a * b` on matching arrays multiplies entries at the same
coordinates; `a @ b` performs matrix multiplication and has a different meaning.
`np.sum(a, axis=None)` adds every entry and returns a scalar. With `axis=0` or
`axis=1`, it instead leaves values for the unreduced axis. `np` is already imported
in the editable cell. Choose how to combine the documented operations yourself.

### TODO 2: `output_size(height, kernel_size, padding, stride)`

This function counts output positions along **one spatial axis**. It receives
plain Python integer settings, not an image, array shape tuple or widget object.

| Argument | Valid domain and units |
| --- | --- |
| `height` | Positive integer: unpadded input length, in pixels. The live digit length is `8`. The same function can be used for width. |
| `kernel_size` | Positive integer: window length on this axis, in pixels. The runner uses `3`. Dilation (spacing inside the kernel) is fixed at `1`. |
| `padding` | Nonnegative integer: number of zero-filled pixels added on **each** side. The runner uses `1`, so the total added length is twice this value. |
| `stride` | Positive integer: distance between consecutive window starts, in input pixels. The widget selects `1` or `2`; the definition also applies to other positive integer strides. |

For these settings the padded input is guaranteed to fit at least one window.
The defining count is

$$H_{out}=\left\lfloor\frac{H+2P-K}{S}\right\rfloor+1.$$

`H`, `P`, `K`, `S` stand for the four arguments in the table. The numerator is
how far a window can move after its first valid placement. Dividing by stride
counts moves; floor discards a final incomplete move, and the extra `1` counts
the first position. Return one positive Python `int`, not a float, shape tuple or
channel count. `//` is Python floor division for these nonnegative geometry
counts. A stride need not divide the available distance exactly. No batch or
channel size enters this spatial count.

### Before running and reading the evidence

Replace only `response=None` and `size=None` inside their marked TODO regions;
keep both signatures and return lines. Run the cell with **Shift+Enter**. `None`
means unfinished and leaves the plots waiting. The prepared runner calls both
functions with the current settings; a finite scalar/positive size alone does
not establish that your arithmetic is correct. Hand-check aligned products and
count complete windows before trusting the plot.

In the JSON, `patch_response` is your scalar, `predicted_spatial_size` is your
one-axis count, and `actual_stride_shape` is the measured shape in
`[batch, channels, height, width]` order. `tiny_trace` and `vgg_partial_trace` contain
`stage` strings and `shape` integer lists in that same tensor convention; class
`scores` have `[batch, classes]` instead. Compare your spatial count with both
last entries of the isolated stride shape. The full pretrained CNN trace keeps
its original stride; only the labelled isolated probe uses the chosen stride.
Inspect the same feature channel and shared color scale when discussing a lost
stroke. Channel number is not a known object category, and VGG's digit preview
is outside its original ImageNet image domain.

**Version-matched official references:** [NumPy 2.2 sum](https://numpy.org/doc/2.2/reference/generated/numpy.sum.html),
[NumPy array arithmetic](https://numpy.org/doc/2.2/user/quickstart.html#basic-operations),
[Python 3.12 arithmetic](https://docs.python.org/3.12/reference/expressions.html#binary-arithmetic-operations),
[PyTorch 2.8 Conv2d shape and cross-correlation](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Conv2d.html).

The same guide is displayed immediately before the editable notebook cell. Complete only the marked TODO regions; the implementation remains your work.

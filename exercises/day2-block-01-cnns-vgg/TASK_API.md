# Task input and output contracts

## Function inputs and API reference

### TODO 1: `patch_response(patch, kernel, bias)`

| Input | Definition and guarantees |
|---|---|
| `patch` | NumPy floating-point array, shape `(3, 3)`: one grayscale image window. Axis 0 is image row (y), axis 1 column (x); intensities are finite numbers in `[0, 1]`. |
| `kernel` | NumPy floating-point array of the same shape: learned weights for one input/output channel. Weights may be negative; `kernel[r, c]` aligns with `patch[r, c]`. This PyTorch operation uses cross-correlation: the provided kernel is not spatially reversed. |
| `bias` | One finite Python float, added once to the response. It is not a pixel or an array. |

Return one finite numeric scalar, with no remaining row/channel axes. The response is an activation before ReLU, so negative values are valid. Inputs are already loaded; keep them unchanged. The defining weighted-sum equation is in **Read the maps like an investigator** above.

### TODO 2: `output_size(height, kernel_size, padding, stride)`

| Input | Definition and guarantees |
|---|---|
| `height` | Positive integer: original input length along one spatial axis, in pixels. Current call: `8`. |
| `kernel_size` | Positive integer: window length on that same axis, in pixels. Current call: `3`; dilation is fixed at 1. |
| `padding` | Nonnegative integer: zero-filled pixels added **on each side** of that axis. Current call: `1`. |
| `stride` | Positive integer: distance in input pixels between consecutive window starts. Current controls choose `1` or `2`. |

Return one Python integer: output length on that axis. The supplied geometry always permits at least one complete window. Height and width follow the same definition independently. No batch size or channel count is included in this scalar. The equation and meaning of floor are given above.

**API vocabulary:** NumPy `*` between matching arrays multiplies aligned entries; `np.sum(a)` returns a sum over all entries if no axis is specified. Python `//` is floor division for these nonnegative geometry counts. These operation definitions do not specify how to assemble your functions.

**Official references:** [NumPy sum](https://numpy.org/doc/2.2/reference/generated/numpy.sum.html), [Python arithmetic](https://docs.python.org/3.12/reference/expressions.html#binary-arithmetic-operations), [PyTorch 2.8 Conv2d: shape and cross-correlation conventions](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Conv2d.html).

The same contracts are displayed beside the TODOs in the notebook. API references define available operations; selecting and combining them remains your task.

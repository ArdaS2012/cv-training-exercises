# Task input and output contracts

## Function inputs and API reference

### TODO 1: `target_head_classes()`

This function has **no arguments**. Use the target-task definition supplied here: labels are integer `0` for even digits and `1` for odd digits. The old source task had ten labels, but the replacement head must represent the target label set. Return one positive Python integer: the number of output logits needed for that target. A logit is an unnormalized score for one output class; its column index is the label ID. Do not return a list of names, an input-feature count or a layer object.

### TODO 2: `selective_trainable_groups()`

This function also has **no arguments**. Return a new Python dictionary with exactly three string keys. The meanings are fixed by the supplied model:

| Key | Model group / parameter meaning |
|---|---|
| `conv1` | First convolution, mapping one grayscale channel to eight feature channels. |
| `conv2` | Last convolution, mapping eight feature channels to twelve. |
| `head` | New target linear classifier: 48 input features and one output logit per target label. |

Each value must be a Python Boolean (`True` or `False`), not a string such as `"frozen"`. `True` permits weight/bias updates in that group; `False` keeps its parameters fixed. The requested experiment is stated in the TODO above: freeze the first convolution and adapt the last convolution and target head. Do not construct an optimizer, call training, or modify a global model in this configuration function.

The prepared report's `trainable_names` is a list of full names such as `conv1.weight`, `conv1.bias`, `conv2.weight`, `conv2.bias`, `head.weight`, `head.bias`; `fine_tune_max_absolute_delta` maps those names to the largest absolute coordinate change. These are measured parameter changes, not gradients or percentages. A zero frozen-group delta is expected; a nonzero trainable-group delta confirms an actual update for that run.

**API vocabulary:** a Python dictionary maps keys to values. In PyTorch, `nn.Module` is a model/layer object; `named_parameters()` yields parameter names and tensors, while a parameter's `requires_grad` flag controls gradient tracking. The supplied adapter applies your Boolean group flags; you need not manipulate these APIs in the TODO.

**Official references:** [Python dictionaries and booleans](https://docs.python.org/3.12/library/stdtypes.html#mapping-types-dict), [PyTorch 2.8 Module and named_parameters](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Module.html).

The same contracts are displayed beside the TODOs in the notebook. API references define available operations; selecting and combining them remains your task.

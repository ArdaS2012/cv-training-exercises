# Task definitions and API help

## TODO guide — describe the target head and the trainable subset

A pretrained **backbone** converts pixels into features; a task **head** converts
those features into one score per target class. **Head-only fitting** keeps the
backbone fixed. **Selective fine-tuning** additionally permits a chosen backbone
part to update. Here the training loops, new-layer construction, cached features
and before/after measurement are supplied. Your functions return small
configuration values; they do not perform training.

### TODO 1: `target_head_classes()`

This function has **no arguments**. The source model recognized digit IDs `0` to
`9`. The new target is parity, with the complete ordered label vocabulary:

| Target label ID / score column | Meaning |
| --- | --- |
| `0` | Even digit |
| `1` | Odd digit |

Return one positive Python `int`: the number of scores the new head needs for
this target vocabulary. A **logit** is a raw, unnormalized score. The supplied
cross-entropy/argmax classification route expects one logit per listed class;
it does not use the alternative single-logit binary-classification convention.
The index of the highest score becomes the predicted label ID. Do not return
the source-class count, a list of labels, an input-feature dimension or a layer.

For a batch of `B` images, each with `d=48` backbone features, the head is

$$S=ZW^T+b.$$

`Z` has shape `(B, d)`; `W` has shape `(c, d)`; `b` has shape `(c,)` and is applied
to every row; the returned scores `S` have shape `(B, c)`. Here `c` is the integer
your function returns. Batch is the image axis; class columns follow the label
table. Changing the target vocabulary changes `c`, not the backbone feature width.

### TODO 2: `selective_trainable_groups()`

This function also has **no arguments**. It describes the requested experiment:
**keep the first convolution fixed; allow the last convolution and target head
to train**. Return a new ordinary Python dictionary with exactly these string keys:

| Exact key | Supplied model group and parameters |
| --- | --- |
| `conv1` | First convolution, from 1 grayscale input channel to 8 output feature channels; contains `conv1.weight` and `conv1.bias`. |
| `conv2` | Last convolution, from 8 input to 12 output feature channels; contains `conv2.weight` and `conv2.bias`. |
| `head` | New linear target classifier, from 48 features to `c` logits; contains `head.weight` and `head.bias`. |

Every value must be the Python Boolean `True` or `False`. `True` grants that
group permission to update; `False` freezes all its parameters. Strings such as
`"True"` or `"frozen"` and integers such as `1` are not this configuration type.
Choose the flags that express the requested experiment. Do not change global
models, call an optimizer or alter a weight tensor inside this function.

The local adapter applies these flags by mapping each parameter's group prefix
to its `requires_grad` setting, and includes enabled groups in the optimizer.
The frozen-feature baseline is already supplied; your dictionary configures the
separate selective route. It is not a table of measured changes.

### How to prove what changed

For a parameter group `G`, freezing requires

$$\Delta\theta_G=\theta_{G,after}-\theta_{G,before}=0.$$

`theta_G` denotes all weight/bias entries in that group. The report measures the
largest absolute entry change **per named parameter tensor**, in the tensor's
own parameter units. A nonzero measurement proves at least one entry changed;
it does not prove that every eligible entry changed or that validation improved.
Different input images may produce different activations through frozen weights.

The prepared `day2.adapt(head_classes, trainable_groups)` consumes your integer
and dictionary; it returns the report shown below. Relevant fields are:

| Report field | Type and how to read it |
| --- | --- |
| `trainable_names` | List of full parameter-name strings enabled in the selective route. Compare the prefixes with your flags. |
| `fine_tune_max_absolute_delta` | Dictionary from each of the six full parameter-name strings in the table above to one nonnegative float. Deltas compare the selective model before/after its additional updates. |
| `routes` | List of two records, each with `route: str`, `trainable_parameters: int`, `validation_loss: float` (finite and nonnegative, with no upper bound of 1) and `validation_accuracy: float` (fraction in `[0, 1]`). |
| `validation_comparison` | List of records with `image_id: int`, `target: int`, `head_only: int`, `last_block: int` and `changed: bool`. Label values use the parity table; `changed` compares the two predictions. |
| `budgets` | Dictionary with `head_epochs: int` and `fine_tune_epochs: int`, currently `35` and `12`. |

These fields are read for interpretation; you do not construct them. The source
pretraining and target-example IDs are disjoint. Both routes share the fitted
head start; the selective route then receives an extra short training budget,
with a smaller backbone rate. Therefore the measured validation difference does
not isolate "unfreezing alone" at an equal total budget.

### Before running

Replace `classes=None` and `groups=None` only inside their TODO regions. Keep the
signatures and return lines; **Shift+Enter** reruns the cell. No imports are needed
for integer/Boolean/dictionary values. Press **Fit head and briefly fine-tune last
block** after both functions are ready. Confirm the frozen group has zero delta,
an eligible group actually changed, and the class-column interpretation matches
the target table. Use the held-out parity results to justify a conditional route
choice; this small related-domain experiment does not establish transfer quality
for medical, industrial or other unseen domains.

**Version-matched official references:** [Python 3.12 dictionaries and Boolean values](https://docs.python.org/3.12/library/stdtypes.html),
[PyTorch 2.8 Linear and its tensor shapes](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Linear.html),
[PyTorch 2.8 Module / named_parameters](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Module.html).

The same guide is displayed immediately before the editable notebook cell. Complete only the marked TODO regions; the implementation remains your work.

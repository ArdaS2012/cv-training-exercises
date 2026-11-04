# Transfer implementation / API reference

TODO contracts are beside the code. `features` is float32 CPU B×48, not images; weight has two output rows, bias two entries. Matrix multiplication `@`, tensor `.T` and broadcasting retain autograd. Compare your head with `torch.nn.Linear(48,2)`. Source labels were digits0–9; target IDs are 0 even and 1 odd, one logit each.

`model.named_parameters()` yields `(str,torch.nn.Parameter)` pairs; exact six names: conv1.weight (8,1,3,3), conv1.bias (8,), conv2.weight (12,8,3,3), conv2.bias (12,), head.weight (2,48), head.bias (2,). `parameter.requires_grad_(enabled)` mutates gradient eligibility in place and returns the parameter. Return the same supplied model. No weight values are edited by this function. `name.split('.')[0]` gives the owning group.

Snapshots passed to parameter_deltas have these same six names/shapes and independent detached CPU tensor storage. Return all six maximum-absolute differences as Python floats; `abs()`/`max()` reduce magnitudes, `float(scalar_tensor)` reads a scalar. Preserve snapshots. Zero delta means no entry changed; nonzero means at least one changed.

`day2.adapt(head_scores,configure,audit,shift=None)` runs two routes with an **equal update budget**: both start from the same pretrained body and the same seeded new head (seed 42) and take 150 full-batch Adam steps on the 180 training rows. Frozen route: only the head learns (rate 0.01) on cached frozen features (identical to `model.features(x)`). Fine-tune route: your `configure` decides trainable tensors; the head uses rate 0.01, pretrained conv2 a 10× smaller rate 0.001 (common practice so pretrained weights move less). **Both routes compute logits with your `head_scores`.** Your audit receives detached before/after snapshots of the fine-tune route. Any callback returning None leaves the output waiting; there is no reference fallback.

Returned report: `routes` list of dicts (`route`:str, `trainable_parameters`:int, `update_steps`:int, `validation_loss`:float ≥ 0, `validation_accuracy`:float in [0,1], and `shifted_validation_accuracy` when a shift is supplied), `fine_tune_max_absolute_delta` (your six-name dictionary), `trainable_names`, `validation_comparison` records (`image_id`:int, `target`/`head_only`/`last_block`: int 0/1, `changed`: bool), `budget` (`optimizer`, `rate`, `pretrained_rate`, `update_steps_per_route`, `train_rows`), `validation_predictions_by_route`, `validation_targets`, `validation_ids`, `source_target_disjoint`, task/seed metadata.

**TODO 4 (stretch) — `shift(images)`:** float tensor `(N,1,8,8)` in [0,1]; return a new tensor with every intensity inverted, $x'=1-x$, same shape. The routes are not retrained; the table adds accuracy on inverted validation images. Chance level for even/odd is 0.5.

`day2.delta_figure(deltas)` draws your audit on a log scale; exact zeros appear as gray "0 (frozen)" bars.

[PyTorch 2.8 Linear](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Linear.html), [Module.named_parameters](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Module.html#torch.nn.Module.named_parameters), [requires_grad_](https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.requires_grad_.html), [transfer tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

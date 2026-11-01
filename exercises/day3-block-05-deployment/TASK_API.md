# Task definitions, inputs and API reference

## TODO definitions and API reference

### First understand what the package must preserve

A **backbone** produces reusable image features; a **head** converts those features into task scores. Here the pretrained convolutional backbone produces 48 features from an 8×8 grayscale image. The new head produces two **logits** (unnormalized scores) for the target **even/odd digit** task.

**Head-only adaptation** freezes backbone parameters and permits updates only to the head. A frozen parameter keeps exactly its previous value. A trainable parameter is allowed to change; that does not imply every individual value must change. The **learning rate** controls the optimizer's update scale; it is not a number of epochs or an accuracy target. Validation measures the effect of a strategy; a larger learning rate does not guarantee an improvement.

### TODO 1 — configuration: `adaptation_config(rate)`

**Input.** `rate` is one finite Python `float`, already read from the **Head learning rate** control. The UI offers `[0.01, 1.0]` in steps of `0.01`, starting at `0.03`; the underlying adapter accepts the wider interval `[0.001, 1.0]`. Use this argument on each call. It is not the slider object, a percentage, an epoch count or a hard-coded default. Valid finite inputs within that interval are supplied; writing input-validation code is outside this TODO.

**Return.** A new ordinary Python dictionary with these required exact keys and types:

| Exact key | Required value / interpretation |
| --- | --- |
| `trainable` | Python `str` naming the group to update: the adapter's literal `"head"`. It is not a Boolean and not a list of parameter names. |
| `learning_rate` | The passed scalar `rate`, preserved as a float. A change of the slider must reach the prepared optimizer. |
| `id2label` | Nested Python dictionary with **string** keys `"0"` and `"1"`; they map to the target-label strings `"even"` and `"odd"` respectively. These names are the task's fixed label definitions. |

`id2label` means “output index to human-readable label.” Score position `0` means the label under key `"0"`; position `1` means the label under `"1"`. Dictionary insertion order does not define the model's class order; its keys do. Integer `0` and string `"0"` are different Python keys. Use strings because the saved JSON configuration and response interpretation use strings. No weights, images, model objects, optimizer objects or HTTP calls belong in your returned dictionary.

**Python container vocabulary.** A dictionary associates a key with a value; a nested dictionary is simply another dictionary used as a value. Generic syntax unrelated to this task: `record = {"city": "Bonn", "counts": {"morning": 3}}`. Return the Python container, not a printed representation or a JSON string. The supplied `return config` already returns your variable; replace `config = None` only inside the marked region, keeping the function signature.

### What the supplied training runner does

`day3.adapt_package(config, epochs=40)` takes your configuration dictionary and an optional positive integer update budget, normally `40`. Passing `None` returns `None` so the starter waits. A valid configuration returns `(report, package_bytes)`: a Python report dictionary and ZIP bytes for the download button. The notebook passes the selected rate to your function and then passes your result to this adapter. You do not call it inside your TODO.

The adapter uses CPU PyTorch, fixed seed `123`, the supplied train/validation/test split, a new two-output head and the Adam optimizer. Only head parameters are optimized. The fixed target shift transforms normalized brightness `x` into `clip(0.45*x + 0.25, 0, 1)`. It fits 180 target-training rows, compares 60 validation rows, and leaves 60 test rows unused. Each run restarts from the same source checkpoint and seeded head; it does not continue training the previous run. The notebook's learning-rate experiment keeps the 40-step full-batch budget fixed.

Let `theta_F` be the frozen backbone parameters and `theta_H` the head parameters. Head-only adaptation requires

$$\Delta\theta_F=0,\qquad \theta'_H=\theta_H+\Delta\theta_H.$$

`Delta` denotes the change from the run's initial weights; primes denote the final weights. `report.parameter_max_deltas` records the largest absolute element change in each named parameter tensor. All non-head entries should stay zero; nonzero head entries supply evidence that fitting occurred. This is separate from validation quality: changed weights alone do not show improved predictions.

### The saved preprocessing and actual request contract

Raw image input is an **8-row list of 8 numeric grayscale values** each, in `[0, 16]`. Axis 0 is image row `y`; axis 1 is column `x`. There is one grayscale channel. The saved processor divides values by `16` and supplies the model's batch/channel dimensions. Do not normalize twice or send an RGB image to this pipeline.

The temporary server receives **`POST /predict`**, with a JSON object whose required key is `"pixels"` and whose value is the raw nested image list. A successful HTTP 200 response contains:

| Field | Type / meaning |
| --- | --- |
| `label` | `str`: saved label corresponding to the largest score; an exact tie selects the first score position. |
| `scores` | Two finite floats, a softmax distribution in saved output-index order, summing to one up to rounding. They are not a guarantee of correctness. |
| `revision` | `str`: SHA-256 fingerprint of the source checkpoint. This identifies the source; the adapted weights are additionally stored in `model.pt`. |

For logits `z_0, z_1`, softmax scores are $p_j=e^{z_j}/(e^{z_0}+e^{z_1})$, for `j=0,1`. The model is reloaded with its saved preprocessing and label mapping before the request. Invalid image shape, nonfinite values or values outside `[0, 16]` yield HTTP 400 with an `"error"` string. The supplied runner sends one valid request and one invalid-shape request, then closes the temporary localhost server. You do not need to construct a request or keep a server running.

### Read the evidence before choosing a package

Press **Adapt, save, reload and send local HTTP request** after running your function cell. `baseline_validation` and `adapted_validation` each contain `source_like` and `target_shifted` accuracy fractions on the same validation rows. The baseline is measured before target adaptation, including the newly initialized even/odd head. Do not treat it as the source model's original 10-class test accuracy.

Dotted field paths below describe entries in the displayed JSON. They are Python dictionary keys, not object attributes. Check `reload_scores_max_difference` for numeric reproduction, `reload_agreement` for original/reloaded/HTTP response equality, `http_status` for the real successful request, and `invalid_request_status` for rejection of the invalid example. `request.json.pixels` is the raw submitted sample; `response` records the actual reply. The report also exposes `config` and `package_files` so you can inspect the saved label order, transform, source revision and files. The ZIP contains `model.pt`, `config.json` and `README.md`.

Compare rates with all other controls fixed, keeping the first report/package before rerunning. Equal score vectors are necessary evidence of reproduction, but wrong label names could still change their meaning; trace score positions to saved labels as a separate check. This exercise establishes a local serving trace. It does not publish weights to the Hub or demonstrate hosted-service reliability, throughput or generalization.

**Official references:** [Python 3.12 dictionaries](https://docs.python.org/3.12/library/stdtypes.html#mapping-types-dict), [Python JSON conversion](https://docs.python.org/3.12/library/json.html#json-to-py-table), [PyTorch 2.8 Module and parameters](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Module.html), [PyTorch 2.8 Adam](https://docs.pytorch.org/docs/2.8/generated/torch.optim.Adam.html). Local fitting, packaging and request behavior are defined above; implementing them is not part of this configuration TODO.

The same definitions appear directly above the editable functions in the notebook.

# Task input and output contracts

## Function inputs and API reference

### TODO: `adaptation_config(rate)`

`rate` is one finite Python float from **Head learning rate**, currently selectable in `[0.01, 1.0]` with step `0.01` and initial value `0.03`. It is passed to the prepared Adam optimizer as the learning rate; it is neither a percentage nor a number of epochs. The lower-level adapter accepts `[0.001, 1.0]`. Preserve the passed value in your returned configuration.

Return a new Python dictionary with these required string keys and value types:

| Key | Required contract / meaning |
|---|---|
| `trainable` | String naming the requested trainable group. The adapter accepts the literal `"head"` for this head-only experiment. It is not a list or a Boolean flag map. |
| `learning_rate` | Float: the supplied `rate`, without replacing it by a hard-coded UI default. |
| `id2label` | Nested dictionary mapping **string** class-index keys `"0"` and `"1"` to the target-label strings `"even"` and `"odd"` respectively. The index keys define what each output-score position means; dictionary insertion order is irrelevant. Integer keys are not the saved JSON contract. |

The supplied model converts one 8×8 grayscale image into 48 features and two logits (unnormalized class scores). The pipeline converts logits into a two-item softmax `scores` list. The score at position 0 denotes the label stored under `"0"`; the score at position 1 denotes the label under `"1"`. Do not include model weights, tensors, processor objects or HTTP requests in your configuration function. Configuration is your TODO; fitting, save/reload and the request are prepared.

**Inspect the supplied request/response:** `POST /predict` accepts a JSON object with `pixels`: an 8-row list, each row containing 8 finite numeric grayscale values in `[0, 16]`, in row-major image order. The saved processor divides by 16; pass **raw** pixels rather than already-normalized values. A successful response has `label` (string), `scores` (two floats in saved class order) and `revision` (source-checkpoint hash string). Invalid shape/value input returns HTTP 400 with an `error` string. The temporary localhost server closes after the prepared request checks; it is not a persistent hosted service.

**API vocabulary:** dictionaries contain named fields, including nested dictionaries; JSON represents object keys as strings. `nn.Module` is a model/layer object; the prepared runner applies trainable flags and Adam settings. You need only return the declared configuration.

**Official references:** [Python dictionaries](https://docs.python.org/3.12/library/stdtypes.html#mapping-types-dict), [Python JSON conversion](https://docs.python.org/3.12/library/json.html#json-to-py-table), [PyTorch 2.8 Module](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Module.html).

The same contracts are displayed beside the TODOs in the notebook. API references define available operations; selecting and combining them remains your task.

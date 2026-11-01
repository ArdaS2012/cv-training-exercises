# Task definitions and API help

## TODO guide — make a fair candidate, then select on validation

The supplied code trains the same small CNN twice. A **configuration** is a
small dictionary of chosen settings, not learned weights. **Training data** drive
weight updates; **validation data** guide your choice between candidates; the
reserved **test data** estimate the chosen model only after that choice is fixed.
Your functions construct the configuration and choose a report index. Training,
plotting and reserved-test evaluation are already supplied.

### TODO 1: `candidate_config(changed_factor, rate_value, penalty_value)`

The baseline A uses `learning_rate=0.01` and `penalty=0.0`. Candidate B must differ
in exactly one of those settings. Learning rate scales a gradient update. The
L2 penalty discourages large learned weights by adding

$$L_{train}(\theta)=L_{data}(\theta)+\lambda\sum_{j\in W}\theta_j^2.$$

`theta` denotes learned parameters, `W` indexes weight entries only (not biases),
`lambda` is the nonnegative `penalty`, and `L_data` is mean training
cross-entropy. A larger penalty is a stronger preference for small weights;
it does not directly specify accuracy. The supplied code implements this
explicit objective rather than an optimizer's separate weight-decay rule.

| Argument | Type, values and meaning |
| --- | --- |
| `changed_factor` | Python `str`, exactly `"learning_rate"` or `"penalty"`: which setting to change. |
| `rate_value` | Finite Python `float`: proposed new learning rate. Widget choices: `0.001`, `0.03`, `0.08`; the trainer accepts `[0.00001, 0.1]`. |
| `penalty_value` | Finite Python `float`: proposed nonnegative L2 coefficient. Widget choices: `0.0001`, `0.001`, `0.01`; the trainer accepts `[0, 0.1]`. |

The caller passes widget **values**, such as a string or number, not widget
objects. Both proposed numbers arrive on every call. Use the proposal for the
named factor, and keep the other factor at its baseline. An unselected widget's
number must not change that other factor. Current widget proposals differ from
the relevant baseline, so exactly one setting changes.

Return a **new Python dictionary** with exactly the case-sensitive keys
`learning_rate` and `penalty`, each holding a float. Do not return a model,
optimizer, list or text representation of a dictionary. The function has no
training side effects. Invalid factor names, nonfinite numbers and proposals
outside the documented trainer domain are outside the runner's guarantees;
you are not asked to add an input validator.

### TODO 2: `choose_validation(reports)`

`reports` is a nonempty **list of two dictionaries**, preserving candidate order:
index `0` is A (baseline), index `1` is B (changed). Each dictionary provides
`validation_loss: float`, a finite final mean held-out data cross-entropy. A lower
value is better. Your selection needs only this field; small hand-check records
may omit the other reporting fields.

The full live reports also contain:

| Exact key | Full runner value and interpretation |
| --- | --- |
| `validation_accuracy` | Float in `[0, 1]`: fraction of validation labels predicted correctly. This is supporting evidence, not the declared selection criterion. |
| `config` | Dictionary with float values for `learning_rate` and `penalty`. |
| `curves` | Dictionary with exactly `train_loss`, `validation_loss`, `validation_accuracy`; each value is a list of 24 floats, in epoch order. Both loss curves exclude the regularization term. |
| `epochs` | Integer `24`: one fixed training budget for both candidates. |
| `seed` | Integer `42`: same random initialization/control setting for both runs. |

A loss is not an error count. Cross-entropy averages `-log(p_correct)` across
examples, so it also changes when the model becomes more/less confident without
changing the winning label. This is why accuracy and loss can rank runs differently.
Return the Python integer **position** `0` or `1` with the smaller final
`validation_loss`. For an exact tie, return the earlier position `0`. Keep reports
and their order unchanged. Return neither the loss itself nor a dictionary.
Empty report lists, missing criterion fields and nonfinite losses are outside
this task's supplied domain. No test measurement is available to this function.

**Python toolbox:** lists are ordered and start at index zero; dictionaries
access values using exact string keys. Unrelated syntax example:
`items[0]["name"]` reads a field in the first record; it does not compare records.
`min(values)` returns a smallest value; `min(values, key=...)` compares by a chosen
property but returns an input item. Use your own comparison approach. No extra
imports are required for basic Python control flow and dictionaries.

### Before running and reading the evidence

Replace only `config=None` and `index=None` in the marked regions, keep the
signatures/return lines, and rerun with **Shift+Enter**. Press **Run two short CPU
trainings** after the configuration is ready. The supplied trainer uses the same
checkpoint, fixed split (180 train / 60 validation / 60 reserved test), seed and
24 full-batch training updates for each candidate. Compare the entire training
and validation curves, then check which final validation value supports your
selected index. The interface validates a usable index; you must verify that
it follows the declared loss rule yourself, including a tie.

Record/freeze the choice before pressing **Evaluate chosen model once on reserved
test**. That button is a separate action, not an enforced one-use lock: do not
retune after inspecting the result. Report `test_examples` and the measured
correct fraction. A small same-split comparison establishes neither a universal
hyperparameter rule nor performance on a new domain. The longer-run reference
has a different budget and cannot be mixed into a controlled live A/B claim.

**Version-matched official references:** [Python 3.12 dictionaries](https://docs.python.org/3.12/library/stdtypes.html#mapping-types-dict),
[Python 3.12 min and its key/first-tie behavior](https://docs.python.org/3.12/library/functions.html#min),
[PyTorch 2.8 cross-entropy](https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html).

The same guide is displayed immediately before the editable notebook cell. Complete only the marked TODO regions; the implementation remains your work.

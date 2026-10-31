# Task input and output contracts

## Function inputs and API reference

### TODO 1: `candidate_config(changed_factor, rate_value, penalty_value)`

| Input | Definition and guarantees |
|---|---|
| `changed_factor` | String, exactly `"learning_rate"` or `"penalty"`, from **Change one factor**. This names the setting selected for intervention. |
| `rate_value` | Finite float proposed by **Candidate learning rate**: `0.001`, `0.03` or `0.08`. The prepared trainer accepts `[0.00001, 0.1]`. Learning rate scales a training update; it is not accuracy. |
| `penalty_value` | Finite nonnegative float proposed by **Candidate L2 penalty**: `0.0001`, `0.001` or `0.01`; trainer range `[0, 0.1]`. It weights the sum of squared weight entries in the training objective; biases are excluded. |

Return a new Python dictionary with exactly the string keys `learning_rate` and `penalty`, each mapped to a float. The baseline values are `0.01` and `0.0` respectively; the experiment changes the named setting only. Both proposed values are passed even though just one setting is selected. No tensor, optimizer or model object is required.

### TODO 2: `choose_validation(reports)`

`reports` is a **list of two dictionaries**, in candidate order: index `0` is baseline A; index `1` is changed candidate B. It is not a dictionary keyed by candidate name. Each record has the following fields:

| Field | Meaning/type |
|---|---|
| `validation_loss` | Finite float: final mean held-out cross-entropy; smaller is better. This is the selection criterion. |
| `validation_accuracy` | Float in `[0, 1]`: final correct fraction, available for interpretation. |
| `config` | Dictionary with `learning_rate` and `penalty` float values. |
| `curves` | Dictionary whose `train_loss`, `validation_loss`, `validation_accuracy` keys each hold a list of 24 floats, in epoch order. The loss curves report **data loss**, excluding the training penalty. |
| `epochs`, `seed` | Integers, currently `24` and `42`. |

Return a Python integer `0` or `1`, referring to the original list position. If the losses tie exactly, select index `0` so the choice is reproducible. Do not modify the report or use a test result. For example, `reports[0]["validation_loss"]` names the baseline's final validation loss; this is a field-access example, not a selection implementation.

**API vocabulary:** `record["key"]` accesses a named dictionary field; list indices start at zero. Python `min` supports a comparison key; NumPy `argmin` reports a position rather than a minimum value. Choose your own implementation.

**Official references:** [Python dictionaries](https://docs.python.org/3.12/library/stdtypes.html#mapping-types-dict), [Python min](https://docs.python.org/3.12/library/functions.html#min), [NumPy argmin](https://numpy.org/doc/2.2/reference/generated/numpy.argmin.html).

The same contracts are displayed beside the TODOs in the notebook. API references define available operations; selecting and combining them remains your task.

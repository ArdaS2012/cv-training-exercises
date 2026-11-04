# Make overfitting visible

`uv run --locked python start.py --day 2 --block 2`

A randomly initialized TinyCNN trains on only 50 digit images. Implement stable cross-entropy, an L2-regularized objective and early stopping. Then run a controlled comparison: baseline against one changed factor (L2, dropout or learning rate), same seed and data. Select a checkpoint on validation and open the reserved test once.

40 minutes in pairs; swap who edits. Keep code and plots for the trainer solution walkthrough; no written predictions or reports. [Coding/API reference](TASK_API.md) · [Pair task](GROUP_TASK.md).

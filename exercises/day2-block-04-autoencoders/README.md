# Build a latent-space experiment

`uv run --locked python start.py --day 2 --block 4`

Connect encoder/decoder and implement pixel MSE. Implement PCA reconstruction via SVD and compare it with the autoencoders at code width 4 and 16 (link to Day 1). Interpolate encoded coordinates and compare clean/denoising reconstructions. Stretch: refine only the latent code by gradient steps and compare noisy-fit with clean-target error.

40 minutes in pairs; swap who edits. Keep code and computed plots for the trainer solution walkthrough; no written predictions or reports. [Coding/API reference](TASK_API.md) · [Pair task](GROUP_TASK.md).

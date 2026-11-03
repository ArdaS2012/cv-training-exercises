# Build a latent-space experiment

`uv run --locked python start.py --day 2 --block 4`

Connect encoder/decoder, implement pixel MSE and interpolate encoded coordinates. Compare clean/denoising reconstructions, width 4/16 and latent blends. Implement a bounded gradient loop that refines only the latent code against noisy input; inspect noisy-fit versus clean-target error.

30 minutes in pairs. Keep code and computed plots for the trainer solution walkthrough; no written predictions/observations or group reports. [Coding/API reference](TASK_API.md) · [Pair task](GROUP_TASK.md).

# Computer Vision — student practices (Days 1, 2 and 3)

## Exercise + breakout

Every supplied training exercise runs as a 30-minute pair investigation in breakout rooms after guided setup. Each block folder has a `GROUP_TASK.md` with the comparison, shared output and timebox; the same task is visible at the top of its notebook. Keep the preceding slide takeaways beside the task. Predict, implement a small step, explore a controlled change and use the measured evidence to make a decision. Swap editor/checker at minute 15 and bring both the finding and a limitation back to plenary. Blocks with no notebook exercise have no separate breakout task.

Day 1 contains four prepared marimo notebooks: clustering, dimensionality reduction, pixel segmentation and a tiny neural classifier. You need programming familiarity, but no GPU, API key or paid service. Day 1 needs no model downloads. Day 2 adds five practices: CNN/VGG inspection, controlled model improvement, transfer/fine-tuning, autoencoders and detection. Small pretrained teaching assets are bundled. The detection practice uses genuine YOLO11n and SSDLite320 pretrained models; its online preflight downloads their official weights into the local ignored cache. No GPU is required.

[Choose a practice](exercises/README.md). Each scheduled exercise includes a 30-minute pair investigation in breakout rooms, after guided preparation. Swap editor/checker roles halfway. Complete the short TODOs, compare one controlled change, then bring a prediction, measured result and limitation to plenary. The same task stays available in the notebook and its `GROUP_TASK.md`.

## Get the workspace

Clone the independent student repository, then open the folder:

```sh
git clone https://github.com/ArdaS2012/cv-training-exercises.git
cd cv-training-exercises
```

Without Git, choose **Code → Download ZIP** on the repository page and extract it.
Your trainer may instead supply a day-specific student ZIP. Students do not need
access to the private training repository.

## One shared environment

Install [uv from its official instructions](https://docs.astral.sh/uv/getting-started/installation/) for your operating system, then reopen the terminal. In the extracted workspace run:

```sh
uv sync --locked
uv run --locked python start.py --block 1 --check
uv run --locked python start.py --block 1
```

uv manages Python 3.12 and the locked environment; do not create a second virtual environment or install a separate Python. The first sync needs package-registry access. On Windows, `start.cmd --block 2` forwards to the same launcher. On macOS/Linux, `sh ./start.sh --block 2` does the same. Paths with spaces are handled by the wrappers.

Available selectors are **1**, **2**, **3**, **5**. Blocks 0 and 4 have no coding practice in the agenda. The launcher defaults to Block 1 and binds only to `127.0.0.1`. If the browser does not open, enter `http://127.0.0.1:2720` on the same computer. A trainer’s localhost address does not open on your computer.

## Day 2 setup and selectors

After the shared sync, use `uv run --locked python start.py --day 2 --block 1 --check`, then the same command without `--check`. Day 2 selectors are **1, 2, 3, 4, 5**. Windows: `start.cmd --day 2 --block 4`; macOS/Linux: `sh ./start.sh --day 2 --block 4`. The default remains Day 1.

Before class, run each Day 2 selector with `--check`, including Block 5 online to cache both detector weights. Then verify `uv run --offline --locked python start.py --day 2 --block 5 --offline-assets --check`. Add `--offline-assets` to the corresponding offline launch. The bundled detector outputs are clearly labelled preparation snapshots; the live action runs both models on CPU. Cached assets are documented in [model and data provenance](cv_exercises/assets/day2/README.md).

## Day 3 setup and selectors

Day 3 selectors are **1,2,4,5**: deep masks, Siamese pairs, vision prompts, adaptation/deployment. Block3 uses presenter-led diagrams, with no notebook or breakout room. After shared sync: `uv run --locked python start.py --day 3 --block 1 --check`, then omit `--check` to open. Windows: `start.cmd --day 3 --block 2`; macOS/Linux: `sh ./start.sh --day 3 --block 2`. Use `--port 2724` if another day is running.

Preflight all four selectors online. Block 4 needs approximately 1 GB of official CLIP/SAM weights on the first check; its progress completes before the notebook is opened. Then check `uv run --offline --locked python start.py --day 3 --block 4 --offline-assets --check`; add `--offline-assets` to its offline launch. All other Day 3 model assets are bundled. [Day 3 model/data provenance](cv_exercises/assets/day3/README.md). The four starters contain short concept TODOs with waiting states; support code performs genuine CPU computation, plotting and local serving. Save edited code and separately download observations, mask images and the reloadable inference ZIP. No Hub credentials or paid deployment is required.

## Work, save and stop

Open the TODO cell’s code control, edit the `None` placeholders and run with **Shift+Enter**. Waiting messages are expected in a fresh starter. The supplied controls and independent tasks remain usable. Save code with **Ctrl+S** (macOS: **Command+S**). Download findings to retain predictions, control settings, plots’ data and written observations; browser widget values are not stored by saving Python. Reloading restores saved code and default controls.

Leave the terminal open while working. Save and stop the current editor with **Ctrl+C** before selecting another day/block. Restart with the same selector. Use `--port 2721` if another notebook already uses 2720. Replace only the indicated TODOs before changing supplied plumbing.

## Pre-class and offline checks

Run online `uv sync --locked`, then each selector’s `--check` before class. After successful sync, `uv run --offline --locked python start.py --block 2 --check` and the corresponding launch use the cached environment. SVD and optimizer computations run locally. The first UMAP compilation may take several seconds; press the supplied nonlinear run control and wait for real output.

The pinned PyTorch environment is not guaranteed on Intel Macs; confirm compatibility with the trainer before class. This workspace was executed on Linux. A Windows/macOS/Linux CI workflow is supplied, but remote CI and managed classroom machines have not been witnessed. If a managed proxy blocks installation, use your organization’s approved proxy/certificate setup; do not disable TLS verification. Request trainer help before class.

## Troubleshooting

- `uv` not found: install it and reopen the terminal so PATH updates apply.
- Package download failure: confirm network access and organization proxy/certificates, then repeat `uv sync --locked`.
- Port in use: add `--port 2721` to the command, then use that port in the browser.
- Results remain waiting: complete the correct TODO cell, run it, and press the nonlinear/training run control where present.
- Shape errors: compare image rows (300×64) with pixel rows (1536×3 or 1536×5); restore the correct axes.
- Repeated nonlinear run feels slow: hold input and setting fixed and run once; plots use a small local dataset, not a GPU.

## Dataset and scope

The digit examples use `sklearn.datasets.load_digits`: 8×8 grayscale digit images from the UCI Optical Recognition of Handwritten Digits source, bundled with scikit-learn. We use the first 30 samples per digit class and normalize original 0..16 intensities by 16. [Dataset attribution and description](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html). The segmentation scene and geometric examples are authored teaching data. No performance figure is a claim about arbitrary photographs or a production model.

The neural experiment reserves disjoint 180 training / 60 validation / 60 test rows. It displays validation predictions; the test split stays unused. In clustering and projection, labels are for interpretation only and do not fit the unsupervised methods.

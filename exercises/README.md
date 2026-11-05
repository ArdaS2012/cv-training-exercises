# CV practices

## Exercise + breakout

Allow around 30 minutes per pair investigation after setup. Implement the marked steps, compare a controlled change and inspect the computed plots. Swap who edits; save your code. The trainer walks through the completed solution afterward. No written predictions, observation logs or group reports are required.

[Shared setup](../README.md)

Each TODO has a brief task and input/output contract. Download `TASK_API.md`
for API details when needed. Setup, model loading and visualizations are supplied.
Keep function names and return structures; edit the marked regions and run with
**Shift+Enter**. Waiting messages are normal before completion. Optional trial
downloads save settings and measurements automatically.

## Day 1

Four visual coding challenges. Start with the picture and first TODO; open the API details when needed. AI coding help is allowed—be ready to explain and change your code.

| Block | Challenge | You build |
|---|---|---|
| 1 | [Build a K-Means step](day1-block-01-clustering/README.md) | Nearest-center assignment, mean updates and a library/failure comparison |
| 2 | [Build and audit a projection](day1-block-02-reduction/README.md) | Centering, projection, reconstruction and error measurement |
| 3 | [Build a magic wand](day1-block-03-segmentation/README.md) | Pixel features, label image and connected-region selection |
| 5 | [Teach a tiny network](day1-block-05-neural/README.md) | Seven short stages from image rows to predictions |

## Day 2

Use `--day 2` with the shared launcher. Follow the README in each owning block;
all definitions needed for the student task are included in that block and notebook.

Blocks 1, 3 and 5 have **around 30 minutes in pairs**; Blocks 2 and 4 have **around 40 minutes**. Guided handoff and plenary review are separate; follow the trainer’s announced return time and the matching `GROUP_TASK.md`.

| Block | Practice | You implement |
|---|---|---|
| 1 | [CNNs and VGG](day2-block-01-cnns-vgg/README.md) | Filter response, output size, spatial convolution and pooling; stretch: receptive field, parameter count |
| 2 | [Model improvement](day2-block-02-model-improvement/README.md) | Cross-entropy, L2 objective and early stopping; controlled L2/dropout/rate comparison on 50 training images |
| 3 | [Transfer and fine-tuning](day2-block-03-transfer/README.md) | Task head, trainable layers and parameter-change audit under an equal budget; stretch: domain-shift probe |
| 4 | [Autoencoders](day2-block-04-autoencoders/README.md) | Reconstruction, error, PCA baseline via SVD and interpolation; stretch: latent-code updates |
| 5 | [Detection](day2-block-05-detection/README.md) | Confidence filtering, IoU and class-aware NMS; stretch: precision/recall sweep |

## Day 3 — advanced vision

Each pair investigation has **around 30 minutes**, after its guided handoff. The task and preceding-slide takeaways stay together in the notebook and matching `GROUP_TASK.md`.

| Block | Practice | You implement |
|---|---|---|
| 1 | [Deep segmentation](day3-block-01-segmentation/README.md) | Class masks, overlap, connected components; stretch instance matching |
| 2 | [Siamese pairs](day3-block-02-siamese/README.md) | Distances, contrastive loss/update, threshold errors on clean/noisy pairs; stretch cutoff choice |
| 4 | [Vision prompts](day3-block-04-prompts/README.md) | Cosine scores, candidate softmax, spatial coordinates; stretch box prompt |
| 5 | [Adaptation and deployment](day3-block-05-deployment/README.md) | Adaptation config, preprocessing, response, reload checks; stretch batching |

Block 3 has a presenter-led architecture comparison and no exercise or breakout rooms. Use the shared root setup; selectors `--day 3 --block 1|2|4|5`.

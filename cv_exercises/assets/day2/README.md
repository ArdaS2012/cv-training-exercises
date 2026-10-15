# CV Day 2 — data and pretrained asset provenance

These are real trained checkpoints, measured feature traces and actual detector outputs. They support bounded CPU teaching experiments, not a production benchmark. [asset-manifest.json](asset-manifest.json) gives SHA-256 and byte counts for bundled files; [training-provenance.json](training-provenance.json) records seed, splits, source/target separation, features and measured reconstruction error.

## Digits, tiny CNN and autoencoders

Data: scikit-learn's bundled [8×8 digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html), from UCI Optical Recognition of Handwritten Digits. Intensities are divided by 16, resulting in 0..1 pixels. The target comprises the first 30 rows per digit class (300 rows), with fixed 180/60/60 train/validation/test splits. All 1,497 remaining rows form the disjoint source task. Labels identify the ten digits for the source task and even/odd for target adaptation.

The supplied tiny CNN uses two learned 3×3 padded convolutions, ReLU/max pooling, 48 flattened features and a ten-class head. It was trained on source rows for 100 Adam epochs at seed 42. `cached-features.npz` stores the genuine frozen extractor output, shape 300×48, with exact target row IDs. The transfer practice fits a new two-class head on training rows and then updates only the last feature block plus head; it measures actual parameter deltas. Cached features cannot stand in for changed extractor outputs after unfreezing.

Four dense autoencoders have architecture 64→32→k→32→64, ReLU hidden stages and sigmoid reconstruction, k=4 or16. Each was trained on source rows for 220 epochs, seed42. Denoising checkpoints used independently sampled Gaussian noise σ=0.25 with clipping to 0..1 and clean targets. Clean checkpoints used clean input/target pairs. Validation errors are measured, not invented. At another noise level the evaluation is a shift from the corruption used in training. Latent width counts coordinates, not a compressed file size.

`longer-training-reference.json` contains three measured 120-epoch target CNN runs, exact configurations and complete train/validation curves; no test examples were used. It is labelled separately from the live 24-epoch classroom comparison. Public supplied training/model plumbing is in `cv_exercises/day2.py`; completed student steps and preparation scripts remain ignored and local.

## VGG inspection

The genuine torchvision VGG16 ImageNet1K_V1 checkpoint (`vgg16-397923af.pth`) supplied the first two feature blocks in `vgg16-first-two-blocks.pt`. A measured full feature-stack shape trace is in `vgg16-full-shape-trace.json`; the large full checkpoint is not bundled. The practice repeats grayscale into three channels, resizes to32×32 and applies the declared ImageNet normalization. Input dimensions differ from the recommended ImageNet classification evaluation; no pretrained classification accuracy or semantic feature claim is made for digits. VGG16 has13 convolutional and3 dense weight layers. [Official VGG16 model/weights](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.vgg16.html), [original VGG paper](https://arxiv.org/abs/1409.1556). Torchvision code is [BSD-3-Clause](https://github.com/pytorch/vision/blob/v0.23.0/LICENSE); dataset/model-weight terms are separate from code licensing.

## Detection

`bus.jpg` and `zidane.jpg` are official Ultralytics sample photographs from [ultralytics/assets](https://github.com/ultralytics/assets/tree/main/im). Their original dimensions are810×1080 and1280×720. Use these supplied examples for this class; image availability is not a blanket license claim for arbitrary third-party photographs.

Actual CPU inference used **ultralytics8.3.236**, **torch2.8.0**, **torchvision0.23.0**, with exact locked package versions in `uv.lock`:

| Model | Weights and preprocessing | Source/terms |
|---|---|---|
| YOLO11n | Official `yolo11n.pt`, COCO classes; library letterbox at imgsz640; confidence0.01, native NMS IoU0.95, max100. | [Official YOLO11 docs](https://docs.ultralytics.com/models/yolo11/), [repository/license AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/v8.3.236/LICENSE) |
| SSDLite320 MobileNetV3 Large | `SSDLite320_MobileNet_V3_Large_Weights.COCO_V1`, `ssdlite320_mobilenet_v3_large_coco-a79551df.pth`; RGB float0..1 then internal normalization/320 resize; score0.01, NMS0.95, max100. | [Official torchvision model/weights](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html), BSD code license above |

`detections.json` records candidates after native decoding/permissive suppression, mapped back to original continuous xyxy pixel coordinates. It is not raw network logits. The browser and starter fallback use these visibly labelled snapshots; the notebook live action executes both actual models, not a fixture. Native postprocessing is followed by the teaching class-aware greedy NMS. Scores are model-specific and not assumed calibrated or comparable across models.

Reference boxes are manually authored coarse visible extents, explicitly marked per box. They were not generated by the detector and are not exhaustive COCO ground truth. Reference matching requires class equality and IoU≥0.50 with one-to-one assignments. Miss/duplicate examples support reasoning; they cannot establish mAP or overall model rankings. No model training, cloud inference, GPU or API key is needed for detection.

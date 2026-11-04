# Autoencoder coding/API reference

All five function contracts sit beside their TODOs (TODO 1–4 core, TODO 5 stretch). Models are prepared PyTorch CPU Autoencoders. Encoder: Linear64→32, ReLU, Linear32→k; decoder: Lineark→32, ReLU, Linear32→64, Sigmoid. Calling model.encoder(tensor)/decoder(tensor) returns a tensor; do not convert to NumPy inside the model path or train here. Images flatten row-major to64 intensities. Evaluation mode/no-grad are supplied around calls; functions preserve model/input values. NumPy MSE consumes arrays only, over all coordinates, with nonempty matching shapes (one image `(1,64)` or the validation set `(60,64)`); units intensity squared.

`day2.ae_inputs(row,k,noise)` takes row int0..299,k4/16, noise std float0..0.5 and returns clean/noisy tensors1×64, dictionary models with exact keys clean/denoise, and integer image_id. Corruption uses seed42 and clamps to[0,1]. The clean/denoise checkpoints have distinct training objectives/histories; denoising was trained at noise std.25. Changing current noise does not retrain. `day2.ae_compare(reconstruct,pixel_mse,row,k,noise)` calls your functions and returns `(report,figure)` or None until unfinished. Report fields: image_id:int,latent_width:int,coordinate_ratio:float64/k,noise_std:float,seed:int42,clean_target_mse:dict with clean/denoise nonnegative floats,noisy_input_mse:float,units/note:str. Coordinate count is not codec bitrate.

Latent interpolation consumes clean images from two selected rows and the clean-trained model. It encodes both, interpolates same-index latent coordinates, then decodes. Alpha0/1 must equal their **reconstructions**, not exact originals; both endpoints are valid even when rows coincide. Pixel interpolation is supplied separately as a visible comparison. Intermediate images are illustrative, not proof of semantic disentanglement. Compare alpha0/.5/1, width4/16 and one noise change without changing all controls together.

[PyTorch 2.8 Module](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Module.html), [Linear](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Linear.html), [Sigmoid](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Sigmoid.html), [NumPy mean](https://numpy.org/doc/2.2/reference/generated/numpy.mean.html).

## TODO 3 — `pca_reconstruct(train, test, k)`

- `train`: float64 NumPy `(n,64)`, n = 180 target training digits, row-major pixels in [0,1]. `test`: float64 `(m,64)`, same pixel order. `k`: int, 1 ≤ k ≤ min(n,64).
- Steps: mean $\mu$ over train rows `(64,)`; `np.linalg.svd(train-mu, full_matrices=False)` → `U (n,r)`, `S (r,)`, `Vt (r,64)` with r = min(n,64), rows of `Vt` sorted by decreasing singular value; basis $V_k$ = `Vt[:k]` `(k,64)`; return $(\text{test}-\mu)V_k^\top V_k+\mu$, shape `(m,64)`.
- Center the test rows with the **train** mean. Do not clip, do not modify inputs.
- `day2.pca_vs_ae(pca_reconstruct, pixel_mse, row)` evaluates k = 1, 2, 4, 8, 16, 32 on the 60 validation rows and returns `(report, figure)` or None: `report['pca_validation_mse']` dict k → float, `report['ae_validation_mse']` dict with keys `clean-4`, `denoise-4`, `clean-16`, `denoise-16`. The image panel shows `row` if it is a validation row, otherwise the first validation row.
- PCA here is fit on 180 target rows; the autoencoders were trained on 1,497 disjoint digits. [NumPy 2.2 svd](https://numpy.org/doc/2.2/reference/generated/numpy.linalg.svd.html).

## TODO 5 (stretch) — code refinement API

`refine_code(model,noisy,steps,rate)` optimizes only the initial encoded coordinates, against the noisy input, with the fixed pretrained decoder. Inputs/return are defined beside the TODO. `tensor.detach()` returns a view disconnected from the old autograd graph; do not mutate that storage. `.requires_grad_(True)` enables derivatives on the new code. `torch.autograd.grad(loss,inputs)` returns a tuple of gradients for the requested input tensors; with one code tensor, element0 has its shape. It does not accumulate `.grad` on model parameters. Keep a fresh detached code after every update to bound graph history. Decode the final code and detach the returned reconstruction. The runner uses rate 5 and steps 0/20/80. A zero-update result must equal the original encoded reconstruction, not the noisy input. Do not call loss.backward(), change model flags/weights, or optimize against the clean target.

Plots compare noisy input, initial reconstruction and refined reconstruction; separate measured MSEs against noisy and clean images reveal their different objectives. Measured (k=4, noise 0.25, four rows): noisy-fit and clean-target MSE both fall with more steps, because the encoder's one-shot code is not optimal (amortization gap) and the fixed decoder can only produce digit-like images. Code fitting at inference is not weight training.

[PyTorch2.8 autograd.grad](https://docs.pytorch.org/docs/2.8/generated/torch.autograd.grad.html), [detach](https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.detach.html).

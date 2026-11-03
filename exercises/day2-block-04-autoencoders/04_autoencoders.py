import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title='Build a latent-space experiment')


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path
    root=next(p for p in Path(__file__).resolve().parents if (p/'cv_exercises').is_dir())
    sys.path.insert(0,str(root))
    import marimo as mo
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from cv_exercises import day2, display
    api=(root/'exercises/day2-block-04-autoencoders/TASK_API.md').read_bytes()
    return api, day2, display, mo, np, plt, torch


@app.cell(hide_code=True)
def _(api, mo):
    mo.hstack([mo.md('# Build a latent-space experiment'),mo.download(api,filename='TASK_API.md',label='API reference (.md)')],justify='space-between')
    return


@app.cell(hide_code=True)
def _(day2, plt):
    _clean,_noisy,_,_=day2.ae_inputs(0,4,.25)
    _fig,_axes=plt.subplots(1,2,figsize=(5,2.5),layout='constrained')
    for _ax,_im,_name in zip(_axes,[_clean,_noisy],['Clean target','Seeded noisy input']):
        _ax.imshow(_im.reshape(8,8),cmap='gray',vmin=0,vmax=1);_ax.set_title(_name);_ax.axis('off')
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Connect the pretrained encoder/decoder, compute reconstruction error, then interpolate in latent space. Compare width 4/16 and the same seeded corruption; no written log.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1 · Reconstruct
    `reconstruct(model,noisy)`: supplied CPU torch Autoencoder; noisy float32 tensor `(B,64)` in [0,1], row-major 8×8 digits. `model.encoder` maps to `(B,k)`, `model.decoder` maps back to `(B,64)`, k=4 or16. Return decoded tensor `(B,64)`; preserve input/model. Model calls retain tensor type; evaluation/no-grad is supplied.
    """)
    return


@app.cell
def _():
    def reconstruct(model,noisy):
        # TODO 1: encode, then decode.
        output=None
        # END TODO 1
        return output
    return (reconstruct,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2 · Pixel error
    `pixel_mse(clean,reconstruction)`: matching nonempty finite NumPy float arrays, live shape `(1,64)`, intensities [0,1]. Return Python float mean over **all** entries, $MSE=\frac1N\sum_j(x_j-\hat x_j)^2$, in squared intensity units. Preserve arrays; `np` supplied.
    """)
    return


@app.cell
def _(np):
    def pixel_mse(clean,reconstruction):
        # TODO 2: mean squared difference over every pixel.
        error=None
        # END TODO 2
        return error
    return (pixel_mse,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3 · Interpolate the code
    `latent_mix(model,first,second,alpha)`: supplied model; matching float32 CPU tensors `(B,64)` in [0,1]; alpha float [0,1]. Encode both, mix $z=(1-\alpha)z_A+\alpha z_B$, decode and return `(B,64)`. At 0/1 recover the corresponding reconstruction, not the raw input. Preserve inputs/weights.
    """)
    return


@app.cell
def _():
    def latent_mix(model,first,second,alpha):
        # TODO 3: interpolate encoded coordinates, then decode.
        mixed=None
        # END TODO 3
        return mixed
    return (latent_mix,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4 · Refine the code, keep the decoder fixed
    `refine_code(model,noisy,steps,rate)`: same CPU model; noisy float32 `(B,64)` in [0,1]; steps int0..80, rate float .01..1. Start at the encoded noisy input. For each step minimize noisy-input MSE with $z\leftarrow z-\eta\nabla_z MSE(decoder(z),noisy)$. Return detached decoded tensor `(B,64)`; preserve model weights, parameter gradients and input. Zero steps returns the initial reconstruction. Use `torch.autograd.grad(loss,z)`; only the latent code changes.
    """)
    return


@app.cell
def _(torch):
    def refine_code(model,noisy,steps,rate):
        # TODO 4: initialize a differentiable code, update only it, then decode.
        refined=None
        # END TODO 4
        return refined
    return (refine_code,)


@app.cell(hide_code=True)
def _(mo):
    image_row=mo.ui.slider(0,299,value=0,show_value=True,label='First image row')
    second_row=mo.ui.slider(0,299,value=30,show_value=True,label='Second image row')
    latent=mo.ui.dropdown([4,16],value=4,label='Latent width')
    noise=mo.ui.slider(0,.5,step=.05,value=.25,show_value=True,label='Noise std')
    code_steps=mo.ui.dropdown([0,20,80],value=20,label='Latent code updates')
    alpha=mo.ui.slider(0,1,step=.1,value=.5,show_value=True,label='Latent mix alpha')
    mo.vstack([mo.hstack([image_row,second_row,latent]),mo.hstack([noise,alpha,code_steps])])
    return alpha, code_steps, image_row, latent, noise, second_row


@app.cell(hide_code=True)
def _(alpha, day2, image_row, latent, latent_mix, mo, noise, pixel_mse, plt, reconstruct, second_row, torch):
    ae_result=None
    mix_report=None
    try:
        ae_result=day2.ae_compare(reconstruct,pixel_mse,image_row.value,latent.value,noise.value)
        if ae_result is None:mo.output.replace(mo.md('Waiting: complete reconstruction and MSE.'))
        else:
            mo.output.replace(ae_result[1])
            _first,_,_models,_=day2.ae_inputs(image_row.value,latent.value,0)
            _second,_,_,_=day2.ae_inputs(second_row.value,latent.value,0)
            with torch.no_grad():_mixed=latent_mix(_models['clean'],_first,_second,alpha.value)
            if _mixed is None:mo.output.append(mo.md('Waiting: implement latent interpolation.'))
            elif not isinstance(_mixed,torch.Tensor) or _mixed.shape!=_first.shape:mo.output.append(mo.md('Latent mix must return a tensor (B,64).'))
            else:
                _pixel=(1-alpha.value)*_first+alpha.value*_second
                _fig,_axes=plt.subplots(1,4,figsize=(10,2.5),layout='constrained')
                for _ax,_im,_name in zip(_axes,[_first,_second,_pixel,_mixed],['First','Second','Pixel mix','Your latent mix']):
                    _ax.imshow(_im.numpy().reshape(8,8),cmap='gray',vmin=0,vmax=1);_ax.set_title(_name);_ax.axis('off')
                mix_report={'alpha':alpha.value,'pixel_vs_latent_mse':pixel_mse(_pixel.numpy(),_mixed.numpy())}
                mo.output.append(mo.vstack([_fig,mo.md(f"Pixel/latent mix MSE: {mix_report['pixel_vs_latent_mse']:.4f} · clean-trained target MSE: {ae_result[0]['clean_target_mse']['clean']:.4f} · denoising target MSE: {ae_result[0]['clean_target_mse']['denoise']:.4f}")]))
    except (ValueError,TypeError,RuntimeError,IndexError) as _e:mo.output.replace(mo.md(str(_e)))
    return ae_result, mix_report


@app.cell(hide_code=True)
def _(code_steps, day2, image_row, latent, mo, noise, pixel_mse, plt, reconstruct, refine_code):
    refinement_report=None
    _clean,_noisy,_models,_=day2.ae_inputs(image_row.value,latent.value,noise.value)
    _refined=refine_code(_models['clean'],_noisy,code_steps.value,.1)
    if _refined is None:
        mo.output.replace(mo.md('Waiting: implement latent-code refinement. Decoder weights stay fixed.'))
    else:
        _initial=reconstruct(_models['clean'],_noisy)
        if _initial is not None:
            _initial=_initial.detach()
            _fig,_axes=plt.subplots(1,3,figsize=(8,2.5),layout='constrained')
            for _ax,_im,_name in zip(_axes,[_noisy,_initial,_refined],['Noisy input','Encoded reconstruction','Your refined code']):
                _ax.imshow(_im.numpy().reshape(8,8),cmap='gray',vmin=0,vmax=1);_ax.set_title(_name);_ax.axis('off')
            refinement_report={'steps':code_steps.value,'initial_clean_mse':pixel_mse(_clean.numpy(),_initial.numpy()),'refined_clean_mse':pixel_mse(_clean.numpy(),_refined.numpy()),'initial_noisy_mse':pixel_mse(_noisy.numpy(),_initial.numpy()),'refined_noisy_mse':pixel_mse(_noisy.numpy(),_refined.numpy())}
            if all(v is not None for v in refinement_report.values()):
                mo.output.replace(mo.vstack([_fig,mo.md(f"Noisy-fit MSE: {refinement_report['initial_noisy_mse']:.5f} → {refinement_report['refined_noisy_mse']:.5f} · clean-target MSE: {refinement_report['initial_clean_mse']:.5f} → {refinement_report['refined_clean_mse']:.5f}. Only the code changes; fitting noise is not proof of better reconstruction.")]))
    return (refinement_report,)


@app.cell(hide_code=True)
def _(ae_result, display, mix_report, mo, refinement_report):
    display.download(mo,'',{'reconstruction':ae_result[0] if ae_result else None,'interpolation':mix_report,'refinement':refinement_report},'autoencoder-trial',label='Save trial')
    return


if __name__ == "__main__":
    app.run()

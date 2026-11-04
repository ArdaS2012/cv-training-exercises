import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title='Build a convolution')


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
    api=(root/'exercises/day2-block-01-cnns-vgg/TASK_API.md').read_bytes()
    return api, day2, display, mo, np, plt, torch


@app.cell(hide_code=True)
def _(api, mo):
    mo.hstack([mo.md('# Build a convolution'),mo.download(api,filename='TASK_API.md',label='API reference (.md)')],justify='space-between')
    return


@app.cell(hide_code=True)
def _(day2, mo, plt):
    _x,*_=day2.target_data();_model=day2.tiny_model()
    _fig,_axes=plt.subplots(1,2,figsize=(6,2.5),layout='constrained')
    _axes[0].imshow(_x[0,0],cmap='gray',vmin=0,vmax=1);_axes[0].set_title('Supplied digit row 0')
    _axes[1].imshow(_model.conv1.weight[0,0].detach(),cmap='coolwarm');_axes[1].set_title('Learned 3×3 kernel')
    for _ax in _axes:_ax.axis('off')
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Build a spatial filter, its output-size rule and pooling, then inspect real pretrained CNN/VGG maps. Compare stride 1 → 2 on the same digit. Fast pairs continue with receptive field and parameter counts (TODO 5–6).

    **Slide takeaways:** one kernel = one template applied everywhere (weight sharing) · $n_{out}=\lfloor (n+2p-k)/s\rfloor+1$ · ReLU then max-pool keeps strong local evidence · stacked 3×3 layers grow the receptive field with fewer weights than one large kernel.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1 · One filter response
    `patch_response(patch,kernel,bias)`: finite NumPy float arrays `(r,c)` of identical nonempty shape, plus a scalar bias. Return one float, $y=\sum_{r,c}x_{rc}w_{rc}+b$. Use aligned products, no kernel reversal; preserve arrays. `np` is supplied.
    """)
    return


@app.cell
def _(np):
    def patch_response(patch,kernel,bias):
        # TODO 1: sum aligned products and add bias once.
        response=None
        # END TODO 1
        return response
    return (patch_response,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2 · Output size
    `output_size(n,k,p,s)`: Python ints: input length n ≥ 1, kernel length k ≥ 1, zero padding p ≥ 0 per side, stride s ≥ 1, with n+2p ≥ k. Return Python `int` $\lfloor (n+2p-k)/s\rfloor+1$ for one spatial axis. Example schema: `output_size(5,3,0,2)` → 2.
    """)
    return


@app.cell
def _():
    def output_size(n,k,p,s):
        # TODO 2: number of valid window positions along one axis.
        size=None
        # END TODO 2
        return size
    return (output_size,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3 · Slide the filter
    `convolve(image,kernel,bias,stride,padding)`: float arrays `(H,W)` / `(Kh,Kw)`, scalar bias, positive integer stride and nonnegative integer zero padding per side. Return float array of shape `(output_size(H,Kh,P,S), output_size(W,Kw,P,S))`; window (i,j) starts at padded row i·S, column j·S. At least one window fits. Apply `patch_response` at every valid window; preserve inputs. `np.pad` and `np` are supplied.
    """)
    return


@app.cell
def _(np, output_size, patch_response):
    def convolve(image,kernel,bias,stride,padding):
        # TODO 3: pad, visit valid windows and store each response.
        output=None
        # END TODO 3
        return output
    return (convolve,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4 · Pool local responses
    `pool2(values)`: finite nonnegative NumPy float array `(H,W)`, H,W ≥ 2. Return float `(H//2,W//2)` using maxima of nonoverlapping 2×2 windows; discard incomplete bottom/right edges. Preserve input. This follows supplied ReLU `maximum(response,0)`.
    """)
    return


@app.cell
def _(np):
    def pool2(values):
        # TODO 4: maximum per complete 2×2 window.
        pooled=None
        # END TODO 4
        return pooled
    return (pool2,)


@app.cell(hide_code=True)
def _(mo):
    image_index=mo.ui.slider(0,299,value=0,show_value=True,label='Fixed digit row')
    stride=mo.ui.dropdown([1,2],value=1,label='Stride')
    mo.hstack([image_index,stride])
    return image_index, stride


@app.cell(hide_code=True)
def _(convolve, day2, image_index, mo, np, patch_response, plt, pool2, stride, torch):
    cnn_result=None
    manual_report=None
    _x,*_=day2.target_data()
    _image=_x[image_index.value,0].numpy()
    _model=day2.tiny_model()
    _kernel=_model.conv1.weight[0,0].detach().numpy()
    _bias=float(_model.conv1.bias[0].detach())
    _response=convolve(_image,_kernel,_bias,stride.value,1)
    if _response is None:
        mo.output.replace(mo.md('Waiting: complete TODO 1–4 (response, output size, convolution, pooling).'))
    elif not isinstance(_response,np.ndarray) or _response.ndim!=2 or not np.isfinite(_response).all():
        mo.output.replace(mo.md('Convolve must return a finite 2D NumPy array.'))
    else:
        _pooled=pool2(np.maximum(_response,0))
        if _pooled is None:
            mo.output.replace(mo.md('Waiting: complete pooling.'))
        else:
            _reference=torch.nn.functional.conv2d(_x[image_index.value:image_index.value+1],_model.conv1.weight[:1],_model.conv1.bias[:1],stride=stride.value,padding=1).detach()[0,0].numpy()
            _poolref=torch.nn.functional.max_pool2d(torch.tensor(np.maximum(_response,0))[None,None],2)[0,0].numpy()
            _fig,_axes=plt.subplots(1,4,figsize=(10,2.8),layout='constrained')
            for _ax,_im,_title in zip(_axes,[_image,_response,_reference,_pooled],['Input','Your convolution','PyTorch reference','Your ReLU + pool']):
                _ax.imshow(_im,cmap='viridis');_ax.set_title(_title,fontsize=10);_ax.axis('off')
            manual_report={'shape':list(_response.shape),'conv_max_error':float(np.max(np.abs(_response-_reference))) if _response.shape==_reference.shape else None,'pool_max_error':float(np.max(np.abs(_pooled-_poolref))) if np.shape(_pooled)==np.shape(_poolref) else None}
            mo.output.replace(mo.vstack([_fig,mo.md(f'**Reference differences:** {manual_report}')]))
            cnn_result=day2.cnn_inspect(patch_response,output_size,image_index.value,stride.value)
            if cnn_result is not None:
                _r=cnn_result[0]
                mo.output.append(mo.vstack([mo.md(f"**Pretrained trace.** Your output_size predicts {_r['predicted_spatial_size']} per axis; PyTorch stride-{stride.value} map is {_r['actual_stride_shape'][-2:]}. TinyCNN predicts digit {_r['prediction']} (target {_r['target']}). Row 1–3: TinyCNN stages (channel 0); last panels: isolated stride probe and VGG-16 block 2."),cnn_result[1],mo.md("""
    **Try:** (1) Stride 1 → 2: how many positions does the probe keep, and which thin strokes vanish? (2) Compare conv1 with relu1: which values disappear? (3) Pool halves height and width: what is kept, what is lost? (4) Move the digit slider: does the same kernel respond to the same stroke direction everywhere?""")]))
    return cnn_result, manual_report


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5 · Stretch: receptive field
    `receptive_field(layers)`: nonempty Python list of dicts, in forward order, each `{'kernel': int ≥ 1, 'stride': int ≥ 1}` (pooling counts as a layer with its window as kernel). Return Python `int` r: how many input pixels along one axis can influence one output value. Recurrence with jump j (input pixels between neighbouring outputs): start $r=1, j=1$; per layer $r \leftarrow r+(k-1)\,j$, then $j \leftarrow j\,s$. Padding does not change r. Example schema: `[{'kernel':3,'stride':1}]` → 3.
    """)
    return


@app.cell
def _():
    def receptive_field(layers):
        # TODO 5: apply the r/j recurrence layer by layer.
        r=None
        # END TODO 5
        return r
    return (receptive_field,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6 · Stretch: parameter count
    `conv_parameters(k,c_in,c_out)`: Python ints ≥ 1: square kernel size k, input channels, output channels. Return Python `int` = weights + biases of one `Conv2d(c_in,c_out,k)`: $k^2 c_{in} c_{out} + c_{out}$.
    """)
    return


@app.cell
def _():
    def conv_parameters(k,c_in,c_out):
        # TODO 6: weights plus one bias per output channel.
        count=None
        # END TODO 6
        return count
    return (conv_parameters,)


@app.cell(hide_code=True)
def _(conv_parameters, day2, mo, receptive_field):
    _tiny=[{'kernel':3,'stride':1},{'kernel':2,'stride':2},{'kernel':3,'stride':1},{'kernel':2,'stride':2}]
    _r=receptive_field(_tiny);_c=conv_parameters(3,1,8)
    if _r is None or _c is None:
        _v=mo.md('Optional: complete TODO 5–6 to compare receptive fields and parameter counts.')
    else:
        _m=day2.tiny_model()
        _two=receptive_field([{'kernel':3,'stride':1}]*2);_five=receptive_field([{'kernel':5,'stride':1}])
        _vgg=receptive_field(([{'kernel':3,'stride':1}]*2+[{'kernel':2,'stride':2}])*2)
        _rows=[('TinyCNN conv1 (your count vs PyTorch)',f"{_c} vs {_m.conv1.weight.numel()+_m.conv1.bias.numel()}"),
               ('TinyCNN conv2 (your count vs PyTorch)',f"{conv_parameters(3,8,12)} vs {_m.conv2.weight.numel()+_m.conv2.bias.numel()}"),
               ('TinyCNN receptive field after pool2',f"{_r} px (input is 8 px wide)"),
               ('VGG-16 receptive field after block 2',f"{_vgg} px"),
               ('Two 3×3 vs one 5×5 receptive field',f"{_two} vs {_five} px"),
               ('Two 3×3 vs one 5×5 parameters, 64→64 channels',f"{2*conv_parameters(3,64,64):,} vs {conv_parameters(5,64,64):,}")]
        _v=mo.md('| Quantity | Value |\n|---|---:|\n'+'\n'.join(f'| {a} | {b} |' for a,b in _rows)+'\n\n**Try:** why did VGG prefer stacks of 3×3 kernels? Same receptive field, fewer weights, one more ReLU in between.')
    _v
    return


@app.cell(hide_code=True)
def _(cnn_result, display, image_index, manual_report, mo, stride):
    display.download(mo,'',{'row':image_index.value,'stride':stride.value,'manual':manual_report,'pretrained_trace':cnn_result[0] if cnn_result else None},'cnn-trial',label='Save trial')
    return


if __name__ == "__main__":
    app.run()

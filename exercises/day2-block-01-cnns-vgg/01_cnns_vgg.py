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
    Build a spatial filter and pooling operation, then inspect real pretrained CNN/VGG maps. Compare stride 1 → 2 on the same digit; no written answer needed.
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
    ### 2 · Slide the filter
    `convolve(image,kernel,bias,stride,padding)`: float arrays `(H,W)` / `(Kh,Kw)`, scalar bias, positive integer stride and nonnegative integer zero padding per side. Return float array `(floor((H+2P-Kh)/S)+1, floor((W+2P-Kw)/S)+1)`. At least one window fits. Apply `patch_response` at every valid window; preserve inputs. `np.pad` and `np` are supplied.
    """)
    return


@app.cell
def _(np, patch_response):
    def convolve(image,kernel,bias,stride,padding):
        # TODO 2: pad, visit valid windows and store each response.
        output=None
        # END TODO 2
        return output
    return (convolve,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3 · Pool local responses
    `pool2(values)`: finite nonnegative NumPy float array `(H,W)`, H,W ≥ 2. Return float `(H//2,W//2)` using maxima of nonoverlapping 2×2 windows; discard incomplete bottom/right edges. Preserve input. This follows supplied ReLU `maximum(response,0)`.
    """)
    return


@app.cell
def _(np):
    def pool2(values):
        # TODO 3: maximum per complete 2×2 window.
        pooled=None
        # END TODO 3
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
        mo.output.replace(mo.md('Waiting: complete the three filter/pooling steps.'))
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
            def _size(h,k,p,s):return (h+2*p-k)//s+1
            cnn_result=day2.cnn_inspect(patch_response,_size,image_index.value,stride.value)
            if cnn_result is not None:mo.output.append(cnn_result[1])
    return cnn_result, manual_report


@app.cell(hide_code=True)
def _(cnn_result, display, image_index, manual_report, mo, stride):
    display.download(mo,'',{'row':image_index.value,'stride':stride.value,'manual':manual_report,'pretrained_trace':cnn_result[0] if cnn_result else None},'cnn-trial',label='Save trial')
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Build the serving contract")


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(next(p for p in Path(__file__).resolve().parents if (p/'cv_exercises').is_dir())))
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import json
    import torch
    from cv_exercises import day3, display
    reference=(next(p for p in Path(__file__).resolve().parents if (p/'cv_exercises').is_dir())/'exercises/day3-block-05-deployment/TASK_API.md').read_bytes()
    return day3, display, json, mo, np, plt, reference, torch


@app.cell(hide_code=True)
def _(mo, reference):
    mo.hstack([mo.md('# Build the serving contract'),mo.download(reference,filename='TASK_API.md',label='API reference (.md)')],justify='space-between',align='center')
    return


@app.cell(hide_code=True)
def _(mo):
    rate=mo.ui.slider(.01,.2,step=.01,value=.03,show_value=True,label='Head learning rate')
    run=mo.ui.run_button(label='Adapt, reload and test local HTTP')
    mo.hstack([rate,run])
    return rate, run


@app.cell(hide_code=True)
def _(day3, mo, plt):
    _x,_y,_ids,_tr,_va,_te=day3.day2.target_data()
    _fig,_ax=plt.subplots(figsize=(3,2.5),layout='constrained')
    _ax.imshow(_x[_va[0],0],cmap='gray',vmin=0,vmax=1);_ax.axis('off');_ax.set_title('Representative validation image')
    mo.vstack([_fig,mo.md('Input: 8×8 grayscale pixels → saved scale → two even/odd logits → saved labels. Training and request plumbing are supplied.')])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1 · Configure adaptation
    `rate`: finite Python float .001–1. Return a new dict `{'trainable': 'head', 'learning_rate': rate, 'id2label': {'0':'even','1':'odd'}}`. This is the required output-index meaning, not dictionary insertion order. The supplied API fits 40 head updates with fixed seed/splits; the backbone stays frozen. Preserve rate.
    """)
    return


@app.cell
def _(np, torch):
    def adaptation_config(rate):
        # TODO: configure head-only tuning and label metadata.
        result=None
        # END TODO
        return result
    return (adaptation_config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2 · Prepare real request pixels
    `pixels`: finite NumPy `(8,8)` values in [0,16], rows y then columns x. `scale`: positive float, saved processor scale 16. Return CPU torch float32 `(1,1,8,8)`: batch, grayscale channel, height, width; divide raw pixels by scale exactly once. Preserve pixels. This function will run inside the actual HTTP serving path.
    """)
    return


@app.cell
def _(np, torch):
    def preprocess(pixels, scale):
        # TODO: convert raw pixels to the model tensor.
        result=None
        # END TODO
        return result
    return (preprocess,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3 · Build the HTTP response
    `logits`: finite CPU torch float tensor `(1,2)`; `id2label`: dict with string keys '0'/'1' and label strings; `revision`: checkpoint fingerprint string. Return JSON-compatible `{'label': str, 'scores': list[float], 'revision': str}`. Softmax over class axis; first maximum wins ties. Use saved label lookup, preserve inputs. This function produces the actual HTTP reply.
    """)
    return


@app.cell
def _(np, torch):
    def make_response(logits, id2label, revision):
        # TODO: normalize logits and attach the saved label and revision.
        result=None
        # END TODO
        return result
    return (make_response,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4 · Verify the serving contract
    `before,after`: response dicts from TODO 3; `atol`: nonnegative float. Return Python bool: same label and revision, and two score vectors agree elementwise with absolute tolerance atol and **zero relative tolerance**. Preserve inputs. Equal numbers with swapped labels must fail; this check consumes the real HTTP response and a reloaded metadata-fault case.
    """)
    return


@app.cell
def _(np, torch):
    def same_response(before, after, atol):
        # TODO: compare score values and response meaning.
        result=None
        # END TODO
        return result
    return (same_response,)


@app.cell(hide_code=True)
def _(adaptation_config, day3, make_response, mo, np, plt, preprocess, rate, run, same_response, torch):
    import io
    import json as _json
    import tempfile
    import zipfile
    import threading
    from pathlib import Path as _Path
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.request import Request, urlopen
    package_result=None
    _view=mo.md('Complete the four pipeline functions, then run. The HTTP handler uses your preprocessing and response functions.')
    _config=adaptation_config(rate.value)
    _probe=preprocess(np.zeros((8,8)),16.)
    _reply=make_response(torch.zeros((1,2)),{'0':'even','1':'odd'},'probe')
    _ready=_config is not None and _probe is not None and _reply is not None
    if _ready:_ready=same_response(_reply,_reply,1e-6) is not None
    if run.value and _ready:
        _report,_zip=day3.adapt_package(_config)
        _folder=_Path(tempfile.mkdtemp(prefix='cv-student-serving-'))
        with zipfile.ZipFile(io.BytesIO(_zip)) as _archive:_archive.extractall(_folder)
        _loaded=day3.DomainPipeline.reload(_folder)
        _cfg=_loaded.config
        def _predict(pixels,labels=None):
            _raw=np.asarray(pixels,dtype=float)
            if _raw.shape!=(8,8) or not np.isfinite(_raw).all() or _raw.min()<0 or _raw.max()>16:raise ValueError('Expected raw finite 8×8 pixels in [0,16].')
            _tensor=preprocess(_raw,float(_cfg['processor']['scale']))
            if not isinstance(_tensor,torch.Tensor) or tuple(_tensor.shape)!=(1,1,8,8) or _tensor.dtype!=torch.float32:raise ValueError('Preprocess must return CPU float32 (1,1,8,8).')
            with torch.no_grad():_logits=_loaded.model(_tensor)
            return make_response(_logits,_cfg['id2label'] if labels is None else labels,_cfg['source_revision'])
        class _Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                try:
                    if self.path!='/predict':raise ValueError('Unknown route')
                    _n=int(self.headers.get('Content-Length',0))
                    if _n>20000:raise ValueError('Payload too large')
                    _response=_predict(_json.loads(self.rfile.read(_n))['pixels']);_status=200
                except (ValueError,TypeError,KeyError) as _exc:_response={'error':str(_exc)};_status=400
                _body=_json.dumps(_response).encode();self.send_response(_status);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(_body)
        _server=ThreadingHTTPServer(('127.0.0.1',0),_Handler)
        _thread=threading.Thread(target=_server.serve_forever,daemon=True);_thread.start()
        _sample=_report['request']['json']['pixels']
        try:
            _request=Request(f'http://127.0.0.1:{_server.server_port}/predict',data=_json.dumps({'pixels':_sample}).encode(),headers={'Content-Type':'application/json'})
            with urlopen(_request,timeout=10) as _response:_http=_json.load(_response);_status=_response.status
        finally:_server.shutdown();_server.server_close();_thread.join(timeout=2)
        _reference=_loaded.predict(_sample)
        _swapped=_predict(_sample,{'0':'odd','1':'even'})
        _checks={'numeric_and_semantic_match':same_response(_reference,_http,1e-6),'swapped_labels_match':same_response(_http,_swapped,1e-6),'http_status':_status}
        _fig,_axes=plt.subplots(1,3,figsize=(10,2.8),layout='constrained')
        _axes[0].imshow(_sample,cmap='gray',vmin=0,vmax=16);_axes[0].set_title('Actual raw request');_axes[0].axis('off')
        _positions=np.arange(2)
        _axes[1].bar(_positions-.18,[_report['baseline_validation']['source_like'],_report['baseline_validation']['target_shifted']],width=.36,label='Before')
        _axes[1].bar(_positions+.18,[_report['adapted_validation']['source_like'],_report['adapted_validation']['target_shifted']],width=.36,label='After')
        _axes[1].set_xticks(_positions,['source-like','shifted']);_axes[1].set(ylim=(0,1),title='Validation accuracy');_axes[1].legend(fontsize=8)
        _axes[2].bar(['even','odd'],_http['scores']);_axes[2].set(ylim=(0,1),title='Actual HTTP scores')
        package_result={'training':_report,'http_response':_http,'metadata_fault_response':_swapped,'checks':_checks}
        _view=mo.vstack([_fig,mo.md(str(_checks)),mo.md(f"**HTTP label:** {_http['label']} · **same scores, swapped metadata label:** {_swapped['label']}"),mo.md('Compare rates .03 → .10 with the same 40-update budget and validation rows. Check frozen/head deltas in the optional trial. Correct reload and HTTP behavior do not establish hosted reliability or generalization.'),mo.download(_zip,filename='reproducible-vision-pipeline.zip',label='Download model package')])
    _view
    return (package_result,)


@app.cell(hide_code=True)
def _(display, mo, package_result):
    display.download(mo,'',{'report':package_result},'day3-deployment',label='Save trial')
    return


if __name__ == "__main__":
    app.run()

"""Prepared Day 2 CPU plumbing. Student concept implementations are supplied by notebooks."""
from pathlib import Path
import copy, json, time
import numpy as np
import torch
from torch import nn
from cv_exercises.data import classification_split
ROOT=Path(__file__).resolve().parents[1]
ASSETS=Path(__file__).resolve().parent/'assets/day2'
CACHE=ROOT/'.cache/day2'
torch.set_num_threads(2)

class TinyCNN(nn.Module):
    def __init__(self,classes=10,dropout=0.):
        super().__init__()
        self.conv1=nn.Conv2d(1,8,3,padding=1)
        self.conv2=nn.Conv2d(8,12,3,padding=1)
        self.drop=nn.Dropout(dropout)
        self.head=nn.Linear(48,classes)
    def features(self,x):
        x=torch.max_pool2d(torch.relu(self.conv1(x)),2)
        x=torch.max_pool2d(torch.relu(self.conv2(x)),2)
        return x.flatten(1)
    def forward(self,x):return self.head(self.drop(self.features(x)))

class Autoencoder(nn.Module):
    def __init__(self,k):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,k))
        self.decoder=nn.Sequential(nn.Linear(k,32),nn.ReLU(),nn.Linear(32,64),nn.Sigmoid())
    def forward(self,x):return self.decoder(self.encoder(x))

def target_data():
    x,y,ids,tr,va,te=classification_split()
    return torch.tensor(x,dtype=torch.float32).reshape(-1,1,8,8),torch.tensor(y),ids,tr,va,te

def tiny_model():
    model=TinyCNN()
    model.load_state_dict(torch.load(ASSETS/'tiny-cnn.pt',map_location='cpu',weights_only=True))
    return model.eval()

def cnn_inspect(response,output_size,index=0,stride=1):
    import matplotlib.pyplot as plt
    model=tiny_model();x,y,ids,*_=target_data()
    image=x[index:index+1];patch=image[0,0,:3,:3].numpy();kernel=model.conv1.weight[0,0].detach().numpy()
    value=response(patch,kernel,float(model.conv1.bias[0].detach()))
    predicted=output_size(8,3,1,stride)
    if value is None or predicted is None:return None
    if not np.isscalar(value) or not np.isfinite(value):raise ValueError('Response must be one finite scalar.')
    if predicted!=int(predicted) or predicted<1:raise ValueError('Output size must be a positive integer.')
    actual=nn.functional.conv2d(image,model.conv1.weight,model.conv1.bias,stride=stride,padding=1)
    with torch.no_grad():
        c1=model.conv1(image);r1=torch.relu(c1);p1=torch.max_pool2d(r1,2);c2=model.conv2(p1);r2=torch.relu(c2);p2=torch.max_pool2d(r2,2);scores=model.head(p2.flatten(1))
    stages={'input':image,'conv1':c1,'relu1':r1,'pool1':p1,'conv2':c2,'relu2':r2,'pool2':p2,'scores':scores}
    trace=[{'stage':k,'shape':list(v.shape)} for k,v in stages.items()]
    # VGG feature modules constructed without the unused 500MB dense classifier.
    from torchvision.models.vgg import make_layers,cfgs
    vgg=nn.Sequential(*list(make_layers(cfgs['D']).children())[:10]).eval()
    vgg.load_state_dict(torch.load(ASSETS/'vgg16-first-two-blocks.pt',map_location='cpu',weights_only=True))
    rgb=nn.functional.interpolate(image.repeat(1,3,1,1),size=(32,32),mode='bilinear',align_corners=False)
    rgb=(rgb-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
    vtrace=[]
    with torch.no_grad():
        for layer in vgg:
            rgb=layer(rgb);vtrace.append({'stage':type(layer).__name__,'shape':list(rgb.shape)})
    # Keep the pretrained network trace fixed; show the isolated stride probe explicitly.
    fig,axes=plt.subplots(3,3,figsize=(11,8),layout='constrained')
    labels={'input':'Input digit','conv1':'conv1 3×3 (ch 0)','relu1':'ReLU 1','pool1':'max-pool 1','conv2':'conv2 3×3 (ch 0)','relu2':'ReLU 2','pool2':'max-pool 2'}
    panels=[(labels[k],v) for k,v in list(stages.items())[:7]]+[(f'Isolated conv1 stride={stride}',actual.detach()),('VGG-16 block 2 (ch 0, 32→16 px)',rgb)]
    probe_min=float(c1[0,0].min());probe_max=float(c1[0,0].max())
    for ax,(name,v) in zip(axes.flat,panels):
        scale={'vmin':probe_min,'vmax':probe_max} if name.startswith('conv1') or name.startswith('Isolated') else {}
        ax.imshow(v[0,0].numpy(),cmap='viridis',**scale);ax.set_title(f"{name}\n{tuple(v.shape[-2:])} · {v.shape[1]} ch",fontsize=9);ax.axis('off')
    report={'image_id':int(ids[index]),'target':int(y[index]),'prediction':int(scores.argmax(1)),'patch_response':float(value),'predicted_spatial_size':int(predicted),'actual_stride_shape':list(actual.shape),'tiny_trace':trace,'vgg_partial_trace':vtrace,'vgg_full_trace':json.loads((ASSETS/'vgg16-full-shape-trace.json').read_text(encoding='utf-8')),'note':'The isolated stride probe and fixed conv1 panel share a color scale. The full pretrained trace keeps its original stride; only the isolated probe changes stride. VGG digit preview is out of ImageNet domain; first two blocks only.'}
    return report,fig

B2_BASELINE={'learning_rate':.01,'penalty':0.,'dropout':0.}
B2_EPOCHS=150

def small_train_rows(per_class=5):
    """First `per_class` training rows of each digit: 50 rows by default, so overfitting becomes visible."""
    x,y,ids,tr,va,te=target_data()
    return np.concatenate([[i for i in tr if int(y[i])==c][:per_class] for c in range(10)]).astype(int)

def train_candidate(config,objective,seed=0,epochs=B2_EPOCHS):
    """Train one randomly initialized TinyCNN with the student's objective; keep one checkpoint per epoch."""
    cfg={k:float(v) for k,v in dict(config).items()}
    if set(cfg)!=set(B2_BASELINE):raise ValueError('Configuration needs exactly learning_rate, penalty and dropout.')
    if not .0001<=cfg['learning_rate']<=.1 or not 0<=cfg['penalty']<=.1 or not 0<=cfg['dropout']<.9:raise ValueError('Use rate .0001-.1, penalty 0-.1 and dropout 0-.9.')
    torch.manual_seed(int(seed));model=TinyCNN(dropout=cfg['dropout']);x,y,ids,tr,va,te=target_data();rows=small_train_rows()
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    curves={'train_loss':[],'validation_loss':[],'validation_accuracy':[]};checkpoints=[]
    for epoch in range(epochs):
        model.train();optimizer.zero_grad();logits=model(x[rows])
        loss=objective(logits,y[rows],[p for n,p in model.named_parameters() if n.endswith('weight')],cfg['penalty'])
        if loss is None:return None
        if not isinstance(loss,torch.Tensor) or loss.ndim!=0 or not torch.isfinite(loss) or not loss.requires_grad:raise ValueError('Objective must return one finite differentiable scalar tensor.')
        loss.backward();optimizer.step();model.eval()
        with torch.no_grad():
            train_logits=model(x[rows]);val_logits=model(x[va])
            curves['train_loss'].append(float(nn.functional.cross_entropy(train_logits,y[rows])))
            curves['validation_loss'].append(float(nn.functional.cross_entropy(val_logits,y[va])))
            curves['validation_accuracy'].append(float((val_logits.argmax(1)==y[va]).float().mean()))
        checkpoints.append({k:v.detach().clone() for k,v in model.state_dict().items()})
    report={'config':cfg,'curves':curves,'final_validation_loss':curves['validation_loss'][-1],'final_validation_accuracy':curves['validation_accuracy'][-1],'epochs':epochs,'seed':int(seed),'train_rows':len(rows),'validation_rows':len(va)}
    return report,checkpoints

def candidate_config(factor,value):
    """Baseline with exactly one changed factor."""
    if factor not in B2_BASELINE:raise ValueError('Factor must be learning_rate, penalty or dropout.')
    cfg=dict(B2_BASELINE);cfg[factor]=float(value)
    if cfg==B2_BASELINE:raise ValueError('Candidate must differ from the baseline.')
    return cfg

def compare_runs(config,objective,seed=0):
    """Baseline and one candidate, same seed, rows, epochs and optimizer. Returns (reports,checkpoints) or None."""
    if config is None:return None
    if sum(float(config.get(k,-1))!=v for k,v in B2_BASELINE.items())!=1:raise ValueError('Change exactly one of learning_rate, penalty or dropout from the baseline.')
    a=train_candidate(B2_BASELINE,objective,seed);b=train_candidate(config,objective,seed)
    if a is None or b is None:return None
    return [a[0],b[0]],[a[1],b[1]]

def select_checkpoint(reports,best_epoch):
    """Early stopping per run with the student's best_epoch, then the lower validation loss wins (first on ties)."""
    choices=[]
    for i,r in enumerate(reports):
        e=best_epoch(list(r['curves']['validation_loss']))
        if e is None:return None
        if isinstance(e,bool) or not isinstance(e,(int,np.integer)) or not 0<=int(e)<len(r['curves']['validation_loss']):raise ValueError('best_epoch must return an int index into the validation curve.')
        choices.append({'run':i,'epoch':int(e),'validation_loss':r['curves']['validation_loss'][int(e)],'validation_accuracy':r['curves']['validation_accuracy'][int(e)]})
    winner=min(choices,key=lambda c:(c['validation_loss'],c['run']))
    return {'per_run':choices,'selected_run':winner['run'],'selected_epoch':winner['epoch']}

def checkpoint_model(reports,checkpoints,run,epoch):
    model=TinyCNN(dropout=reports[run]['config']['dropout']);model.load_state_dict(checkpoints[run][epoch]);return model.eval()

def selection_index(selection,runs):
    if selection is None:return None
    if selection not in range(len(runs)):raise ValueError('Selection must be a candidate index.')
    return int(selection)

def evaluate_reserved(model):
    x,y,ids,tr,va,te=target_data();model.eval()
    with torch.no_grad():pred=model(x[te]).argmax(1)
    return {'test_n':len(te),'accuracy':float((pred==y[te]).float().mean()),'image_ids':[int(i) for i in ids[te]],'protocol':'Reserved test accessed only after selection; do not tune using this result.'}

def training_figure(reports,selection=None,title='Training vs validation loss'):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.4),layout='constrained');colors=['tab:blue','tab:orange']
    for i,r in enumerate(reports):
        changed=[k for k in B2_BASELINE if r['config'][k]!=B2_BASELINE[k]]
        name='A baseline' if not changed else 'B '+', '.join(f"{k}={r['config'][k]:g}" for k in changed)
        axes[0].plot(r['curves']['train_loss'],color=colors[i],label=name+' · train')
        axes[0].plot(r['curves']['validation_loss'],'--',color=colors[i],label=name+' · validation')
        axes[1].plot(r['curves']['validation_accuracy'],color=colors[i],label=name)
        if selection is not None:
            e=selection['per_run'][i]['epoch'];axes[0].plot([e],[r['curves']['validation_loss'][e]],'o',color=colors[i],markersize=7)
            axes[0].axvline(e,color=colors[i],alpha=.25)
    axes[0].set(title=title+' (o = best validation epoch)',xlabel='epoch (full-batch Adam updates)',ylabel='mean data cross-entropy',yscale='log');axes[0].legend(fontsize=7)
    axes[1].set(title='Validation accuracy',xlabel='epoch',ylabel='fraction',ylim=(0,1));axes[1].legend(fontsize=7)
    return fig

B3_STEPS=150
B3_RATE=.01
PRETRAINED_SCALE=.1

def adapt(head_scores,configure,audit,shift=None,steps=B3_STEPS,rate=B3_RATE):
    """Two routes, equal budget: same pretrained body, same seeded new head, same Adam steps/rate.
    Frozen: only the head learns on cached features. Fine-tune: the student's configure() decides trainable tensors.
    Both routes compute logits with the student's head_scores. Returns None while a callback is unfinished."""
    if head_scores is None or configure is None or audit is None:return None
    x,y,ids,tr,va,te=target_data();target=y%2
    cache=np.load(ASSETS/'cached-features.npz');features=torch.tensor(cache['features'])
    if not np.array_equal(ids,cache['image_ids']):raise ValueError('Cached feature row IDs do not match.')
    torch.manual_seed(42);base=tiny_model();base.head=nn.Linear(48,2)
    def logits_of(model,images=None,cached=None):
        z=cached if cached is not None else model.features(images)
        out=head_scores(z,model.head.weight,model.head.bias)
        if out is None:return None
        if not isinstance(out,torch.Tensor) or tuple(out.shape)!=(len(z),2):raise ValueError('head_scores must return a tensor (B,2).')
        return out
    # Frozen route: body never changes, cached features stand in for the frozen forward pass.
    frozen=copy.deepcopy(base)
    for p in frozen.parameters():p.requires_grad_(False)
    for p in frozen.head.parameters():p.requires_grad_(True)
    opt=torch.optim.Adam(frozen.head.parameters(),lr=rate)
    for step in range(steps):
        opt.zero_grad();out=logits_of(frozen,cached=features[tr])
        if out is None:return None
        loss=nn.functional.cross_entropy(out,target[tr]);loss.backward();opt.step()
    frozen.eval()
    # Fine-tune route: student chooses trainable tensors.
    fine=copy.deepcopy(base);before={n:p.detach().clone() for n,p in fine.named_parameters()}
    configured=configure(fine)
    if configured is None:return None
    if configured is not fine:raise ValueError('Configure and return the supplied model, preserving its weights.')
    if any(not torch.equal(before[n],p.detach()) for n,p in fine.named_parameters()):raise ValueError('Configure must not change weight values.')
    flags={n:p.requires_grad for n,p in fine.named_parameters()}
    expected={n:not n.startswith('conv1') for n in flags}
    if flags!=expected:raise ValueError('Keep conv1 frozen; enable conv2 and head.')
    # Common practice: pretrained tensors get a 10x smaller rate than the new head (same number of steps).
    opt=torch.optim.Adam([{'params':[p for n,p in fine.named_parameters() if p.requires_grad and not n.startswith('head')],'lr':rate*PRETRAINED_SCALE},{'params':list(fine.head.parameters()),'lr':rate}])
    for step in range(steps):
        fine.train();opt.zero_grad();loss=nn.functional.cross_entropy(logits_of(fine,x[tr]),target[tr]);loss.backward();opt.step()
    fine.eval();after={n:p.detach().clone() for n,p in fine.named_parameters()}
    deltas=audit(before,after)
    if deltas is None:return None
    if set(deltas)!=set(before) or not all(np.isscalar(v) and np.isfinite(v) and v>=0 for v in deltas.values()):raise ValueError('Audit needs a finite nonnegative maximum delta per tensor name.')
    rows=[];predictions_by_route={}
    shifted=None
    if shift is not None:
        shifted=shift(x[va].clone())
        if shifted is not None and (not isinstance(shifted,torch.Tensor) or shifted.shape!=x[va].shape):raise ValueError('shift must return a tensor with the same (N,1,8,8) shape.')
    for name,model in [('Frozen features',frozen),('Last-block fine-tune',fine)]:
        with torch.no_grad():
            logits=logits_of(model,cached=features[va]) if model is frozen else logits_of(model,x[va]);pred=logits.argmax(1)
            row={'route':name,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'update_steps':steps,'validation_loss':float(nn.functional.cross_entropy(logits,target[va])),'validation_accuracy':float((pred==target[va]).float().mean())}
            if shifted is not None:
                sl=logits_of(model,shifted);row['shifted_validation_accuracy']=float((sl.argmax(1)==target[va]).float().mean())
        predictions_by_route[name]=pred.tolist();rows.append(row)
    return {'routes':rows,'fine_tune_max_absolute_delta':{k:float(v) for k,v in deltas.items()},'trainable_names':[n for n,p in fine.named_parameters() if p.requires_grad],'source_target_disjoint':True,'source_task':'10-class digits','target_task':'even/odd','validation_predictions_by_route':predictions_by_route,'validation_comparison':[{'image_id':int(ids[i]),'target':int(target[i]),'head_only':predictions_by_route['Frozen features'][j],'last_block':predictions_by_route['Last-block fine-tune'][j],'changed':predictions_by_route['Frozen features'][j]!=predictions_by_route['Last-block fine-tune'][j]} for j,i in enumerate(va)],'validation_targets':target[va].tolist(),'validation_ids':[int(i) for i in ids[va]],'budget':{'optimizer':'Adam','rate':rate,'pretrained_rate':rate*PRETRAINED_SCALE,'update_steps_per_route':steps,'train_rows':len(tr)},'seed':42}

def delta_figure(deltas):
    """Log-scale bar chart; exact zeros (frozen tensors) are drawn as a marker on the axis."""
    import matplotlib.pyplot as plt
    names=list(deltas);values=[deltas[n] for n in names];positive=[v for v in values if v>0];floor=min(positive)/10 if positive else 1e-6
    fig,ax=plt.subplots(figsize=(9,2.9),layout='constrained')
    colors=['lightgray' if v==0 else ('tab:orange' if n.startswith('head') else 'tab:blue') for n,v in zip(names,values)]
    ax.bar(names,[v if v>0 else floor for v in values],color=colors,bottom=0)
    for i,v in enumerate(values):ax.text(i,max(v,floor)*1.15,'0 (frozen)' if v==0 else f'{v:.3g}',ha='center',fontsize=8)
    ax.set(yscale='log',ylabel='max |Δθ| (log scale)',title='Your audit: largest change per tensor (blue = pretrained conv2, orange = new head)')
    ax.tick_params(axis='x',rotation=15);return fig

def adaptation_figure(report):
    """Show measured route decisions on the exact held-out images, changed cases first."""
    import matplotlib.pyplot as plt
    x,_,ids,*_=target_data();positions={int(image_id):i for i,image_id in enumerate(ids)}
    rows=sorted(report['validation_comparison'],key=lambda row:not row['changed'])[:8]
    fig,axes=plt.subplots(2,4,figsize=(10,6),layout='constrained')
    for ax,row in zip(axes.flat,rows):
        ax.imshow(x[positions[row['image_id']],0],cmap='gray',vmin=0,vmax=1)
        ax.set_title(f"id {row['image_id']} · target {row['target']}\nhead {row['head_only']} → last {row['last_block']}",fontsize=9);ax.axis('off')
    return fig

def ae_inputs(index,k,noise):
    x,y,ids,tr,va,te=target_data();clean=x[index:index+1].flatten(1)
    generator=torch.Generator().manual_seed(42)
    noisy=(clean+float(noise)*torch.randn(clean.shape,generator=generator)).clamp(0,1)
    models={}
    for mode in ['clean','denoise']:
        model=Autoencoder(k);model.load_state_dict(torch.load(ASSETS/f'ae-{mode}-{k}.pt',map_location='cpu',weights_only=True));models[mode]=model.eval()
    return clean,noisy,models,int(ids[index])

def ae_compare(reconstruct,error,index,k,noise):
    import matplotlib.pyplot as plt
    clean,noisy,models,image_id=ae_inputs(index,k,noise);outputs={}
    with torch.no_grad():
        for mode,model in models.items():
            output=reconstruct(model,noisy)
            if output is None:return None
            if not isinstance(output,torch.Tensor) or output.shape!=clean.shape:raise ValueError('Reconstruction must be a tensor with the same1×64 shape as the target.')
            err=error(clean.numpy(),output.numpy())
            if err is None:return None
            if not np.isscalar(err) or not np.isfinite(err) or err<0:raise ValueError('MSE must be one finite nonnegative scalar.')
            outputs[mode]=(output, float(err))
    fig,axes=plt.subplots(2,3,figsize=(10,6),layout='constrained')
    panels=[('Clean target',clean),('Noisy input',noisy)]+[(mode+' reconstruction',result[0]) for mode,result in outputs.items()]+[(mode+' squared error',(clean-result[0]).square()) for mode,result in outputs.items()]
    for ax,(title,image) in zip(axes.flat,panels):
        ax.imshow(image.reshape(8,8),cmap='magma' if 'error' in title else 'gray',vmin=0,vmax=1);ax.set_title(title,fontsize=9);ax.axis('off')
    report={'image_id':image_id,'latent_width':k,'coordinate_ratio':64/k,'noise_std':noise,'seed':42,'clean_target_mse':{mode:v[1] for mode,v in outputs.items()},'noisy_input_mse':float(nn.functional.mse_loss(noisy,clean)),'units':'normalized intensity squared','note':'latent-only coordinate ratio; not codec bitrate. Changing noise does not train the model.'}
    return report,fig

PCA_WIDTHS=[1,2,4,8,16,32]

def pca_vs_ae(pca_reconstruct,error,index=0):
    """Clean validation reconstructions: student PCA (fit on 180 target training rows) vs bundled AEs (trained on 1,497 other digits)."""
    import matplotlib.pyplot as plt
    x,y,ids,tr,va,te=target_data();X=x.reshape(len(x),64).numpy().astype(np.float64)
    pca={}
    for k in PCA_WIDTHS:
        R=pca_reconstruct(X[tr],X[va],k)
        if R is None:return None
        R=np.asarray(R)
        if R.shape!=X[va].shape or not np.isfinite(R).all():raise ValueError('pca_reconstruct must return a finite (n_test,64) array.')
        e=error(X[va],R)
        if e is None:return None
        pca[k]=float(e)
    ae={}
    for k in (4,16):
        _,_,models,_=ae_inputs(0,k,0)
        with torch.no_grad():
            for mode,model in models.items():ae[f'{mode}-{k}']=float(error(X[va],model(torch.tensor(X[va],dtype=torch.float32)).numpy()))
    fig,axes=plt.subplots(1,2,figsize=(11,3.4),layout='constrained',gridspec_kw={'width_ratios':[1.1,1.4]})
    axes[0].plot(PCA_WIDTHS,[pca[k] for k in PCA_WIDTHS],'o-',color='tab:blue',label='your PCA (linear)')
    axes[0].plot([4,16],[ae['clean-4'],ae['clean-16']],'s',color='tab:orange',markersize=8,label='autoencoder, clean-trained')
    axes[0].plot([4,16],[ae['denoise-4'],ae['denoise-16']],'^',color='tab:green',markersize=8,label='autoencoder, denoising-trained')
    axes[0].set(xscale='log',xticks=PCA_WIDTHS,xticklabels=[str(k) for k in PCA_WIDTHS],xlabel='code width k (numbers per image)',ylabel='validation MSE (clean input)',title='Reconstruction error vs code width');axes[0].legend(fontsize=8)
    row=list(va).index(index) if index in va else 0;img=X[va][row]
    panels=[('Original',img)]
    for k in (4,16):
        panels.append((f'PCA k={k}',np.asarray(pca_reconstruct(X[tr],img[None],k))[0]))
        _,_,models,_=ae_inputs(0,k,0)
        with torch.no_grad():panels.append((f'AE k={k}',models['clean'](torch.tensor(img[None],dtype=torch.float32)).numpy()[0]))
    axes[1].axis('off')
    for i,(title,im) in enumerate(panels):
        sub=axes[1].inset_axes([i/5+.01,.05,.18,.8]);sub.imshow(np.clip(im,0,1).reshape(8,8),cmap='gray',vmin=0,vmax=1);sub.set_title(title,fontsize=9);sub.axis('off')
    axes[1].set_title(f'Validation image id {int(ids[va][row])}',fontsize=10)
    return {'pca_validation_mse':pca,'ae_validation_mse':ae,'validation_rows':len(va),'pca_fit_rows':len(tr),'note':'PCA is fit on 180 target training rows; the AEs were trained on 1,497 disjoint source digits. Neither saw the validation rows.'},fig

def detector_status():
    """Cheap check without loading models: are both detector weights cached on this computer?"""
    yp=CACHE/'yolo11n.pt';sp=CACHE/'checkpoints/ssdlite320_mobilenet_v3_large_coco-a79551df.pth'
    return {'yolo':yp.is_file(),'ssd':sp.is_file()}

def pr_sweep(filter_scores,iou,suppress,image_id,nms_cutoff,pr,cutoffs=(.1,.2,.3,.4,.5,.6,.7,.8,.9),live=None):
    """Run the matching pipeline over confidence cutoffs; your pr(report) turns one model report into (precision, recall)."""
    import matplotlib.pyplot as plt
    rows=[]
    for c in cutoffs:
        result=inspect_detections(filter_scores,iou,image_id,c,nms_cutoff,live,suppress=suppress,plot=False)
        if result is None:return None
        for name,report in result[0]['models'].items():
            value=pr(report)
            if value is None:return None
            if len(value)!=2 or not all(np.isscalar(v) and np.isfinite(v) and 0<=v<=1 for v in value):raise ValueError('pr must return (precision, recall), two floats in [0,1].')
            rows.append({'model':name,'cutoff':c,'precision':float(value[0]),'recall':float(value[1]),'kept':report['after_class_aware_nms']})
    fig,ax=plt.subplots(figsize=(5.6,3.8),layout='constrained')
    for name,color in [('YOLO11n','tab:blue'),('SSDLite320','tab:orange')]:
        pts=[r for r in rows if r['model']==name];ax.plot([r['recall'] for r in pts],[r['precision'] for r in pts],'o-',color=color,label=name)
        for r in pts[::2]:ax.annotate(f"{r['cutoff']:.1f}",(r['recall'],r['precision']),fontsize=7,xytext=(3,3),textcoords='offset points')
    ax.set(xlim=(0,1.05),ylim=(0,1.05),xlabel='recall = matched references / references',ylabel='precision = matched / kept',title=f'Confidence sweep, {image_id} (labels = cutoff)');ax.legend(fontsize=8)
    return rows,fig

def detector_snapshot():return json.loads((ASSETS/'detections.json').read_text(encoding='utf-8'))

def prepare_detectors(download=True):
    from ultralytics import YOLO
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large,SSDLite320_MobileNet_V3_Large_Weights
    CACHE.mkdir(parents=True,exist_ok=True);torch.hub.set_dir(str(CACHE))
    yp=CACHE/'yolo11n.pt';sp=CACHE/'checkpoints/ssdlite320_mobilenet_v3_large_coco-a79551df.pth'
    if not download and (not yp.is_file() or not sp.is_file()):raise ValueError('Detector cache is incomplete. Run online --day 2 --block 5 --check before class.')
    if not yp.is_file():
        yolo=YOLO('yolo11n.pt');Path('yolo11n.pt').replace(yp)
    else:yolo=YOLO(str(yp))
    weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    ssd=ssdlite320_mobilenet_v3_large(weights=weights,score_thresh=.01,nms_thresh=.95,detections_per_img=100).eval()
    return yolo,ssd,weights

def detector_inference(image_id,offline=False):
    from PIL import Image
    yolo,ssd,weights=prepare_detectors(download=not offline);image=Image.open(ASSETS/f'{image_id}.jpg').convert('RGB')
    start=time.monotonic();yr=yolo.predict(str(ASSETS/f'{image_id}.jpg'),conf=.01,iou=.95,max_det=100,device='cpu',verbose=False)[0]
    yseconds=time.monotonic()-start;start=time.monotonic()
    with torch.no_grad():sr=ssd([weights.transforms()(image)])[0]
    sseconds=time.monotonic()-start
    outputs={'YOLO11n':[{'id':f'yolo-{i}','label':yr.names[int(b.cls.item())],'score':float(b.conf.item()),'box':[float(v) for v in b.xyxy[0]]} for i,b in enumerate(yr.boxes)],'SSDLite320':[{'id':f'ssd-{i}','label':weights.meta['categories'][int(label)],'score':float(score),'box':[float(v) for v in box]} for i,(box,score,label) in enumerate(zip(sr['boxes'],sr['scores'],sr['labels']))]}
    return outputs,{'YOLO11n':yseconds,'SSDLite320':sseconds}

def inspect_detections(filter_scores,iou,image_id,cutoff,nms_cutoff,live=None,suppress=None,plot=True):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    from torchvision.ops import batched_nms
    snapshot=detector_snapshot();entry=next(e for e in snapshot['images'] if e['id']==image_id)
    outputs=live if live is not None else entry['outputs'];reports={}
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained') if plot else (None,[None,None])
    for ax,(name,candidates) in zip(axes,outputs.items()):
        filtered=filter_scores(candidates,cutoff)
        if filtered is None:return None
        if not isinstance(filtered,list):raise ValueError('Return a list of detection records.')
        source_ids={d['id'] for d in candidates}
        if any(d.get('id') not in source_ids for d in filtered):raise ValueError('Keep original records; do not invent boxes.')
        kept=filtered
        if filtered:
            classes={s:i for i,s in enumerate(sorted({d['label'] for d in filtered}))}
            indices=batched_nms(torch.tensor([d['box'] for d in filtered]),torch.tensor([d['score'] for d in filtered]),torch.tensor([classes[d['label']] for d in filtered]),nms_cutoff).tolist();kept=[filtered[i] for i in indices]
        if suppress is not None:
            kept=suppress(filtered,nms_cutoff)
            if kept is None:return None
            if not isinstance(kept,list) or any(d.get('id') not in source_ids for d in kept):raise ValueError('NMS returns a list of unchanged source records.')
        if plot:ax.imshow(Image.open(ASSETS/entry['image']));ax.set_title(name);ax.axis('off')
        for d in (kept if plot else []):
            x1,y1,x2,y2=d['box'];ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='orange',linewidth=1));ax.text(x1,y1,d['label']+f' {d["score"]:.2f}',fontsize=7,color='black',bbox={'facecolor':'white','alpha':.8})
        matches=[];matched_predictions=set()
        for ref in entry['references']:
            x1,y1,x2,y2=ref['box']
            if plot:ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='lime',linestyle='--',linewidth=1))
            possible=[]
            for d in kept:
                if d['label']!=ref['label'] or d['id'] in matched_predictions:continue
                overlap=iou(d['box'],ref['box'])
                if overlap is None:return None
                if not np.isscalar(overlap) or not np.isfinite(overlap) or not 0<=overlap<=1:raise ValueError('IoU must be a finite scalar in[0,1].')
                possible.append((float(overlap),d))
            best=max(possible,key=lambda r:r[0]) if possible else (0,None)
            matched=best[0]>=.5
            if matched:matched_predictions.add(best[1]['id'])
            matches.append({'reference':ref['id'],'class':ref['label'],'best_iou':best[0],'match':matched,'detection':best[1]['id'] if matched else None})
        reports[name]={'input_candidates':len(candidates),'above_cutoff':len(filtered),'after_class_aware_nms':len(kept),'reference_matches':matches,'kept':kept}
    return {'image_id':image_id,'confidence_cutoff':cutoff,'nms_iou_cutoff':nms_cutoff,'match_iou_cutoff':.5,'mode':'live CPU inference' if live is not None else 'measured pretrained snapshot','models':reports,'limits':'Two images; teaching reference boxes, not exhaustive benchmark; candidates already have native permissive NMS0.95.'},fig

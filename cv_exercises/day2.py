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
    def __init__(self,classes=10):
        super().__init__()
        self.conv1=nn.Conv2d(1,8,3,padding=1)
        self.conv2=nn.Conv2d(8,12,3,padding=1)
        self.head=nn.Linear(48,classes)
    def features(self,x):
        x=torch.max_pool2d(torch.relu(self.conv1(x)),2)
        x=torch.max_pool2d(torch.relu(self.conv2(x)),2)
        return x.flatten(1)
    def forward(self,x):return self.head(self.features(x))

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
    panels=list(stages.items())[:7]+[(f'Isolated conv1 stride={stride}',actual.detach()),('VGG block2 channel0',rgb)]
    probe_min=float(c1[0,0].min());probe_max=float(c1[0,0].max())
    for ax,(name,v) in zip(axes.flat,panels):
        scale={'vmin':probe_min,'vmax':probe_max} if name=='conv1' or name.startswith('Isolated') else {}
        ax.imshow(v[0,0].numpy(),cmap='viridis',**scale);ax.set_title(name);ax.axis('off')
    report={'image_id':int(ids[index]),'target':int(y[index]),'prediction':int(scores.argmax(1)),'patch_response':float(value),'predicted_spatial_size':int(predicted),'actual_stride_shape':list(actual.shape),'tiny_trace':trace,'vgg_partial_trace':vtrace,'vgg_full_trace':json.loads((ASSETS/'vgg16-full-shape-trace.json').read_text(encoding='utf-8')),'note':'The isolated stride probe and fixed conv1 panel share a color scale. The full pretrained trace keeps its original stride; only the isolated probe changes stride. VGG digit preview is out of ImageNet domain; first two blocks only.'}
    return report,fig

def train_candidate(config,epochs=24):
    cfg=dict(config)
    if set(cfg)!={'learning_rate','penalty'}:raise ValueError('Configuration needs exactly learning_rate and penalty.')
    rate=float(cfg['learning_rate']);penalty=float(cfg['penalty'])
    if not .00001<=rate<=.1 or not 0<=penalty<=.1:raise ValueError('Use a finite rate from 0.00001 to0.1 and penalty from0 to0.1.')
    torch.manual_seed(42);model=tiny_model();x,y,ids,tr,va,te=target_data()
    optimizer=torch.optim.SGD(model.parameters(),lr=rate)
    curves={'train_loss':[],'validation_loss':[],'validation_accuracy':[]}
    for epoch in range(epochs):
        model.train();optimizer.zero_grad();logits=model(x[tr]);data_loss=nn.functional.cross_entropy(logits,y[tr])
        regularizer=sum(p.square().sum() for name,p in model.named_parameters() if name.endswith('weight'))
        (data_loss+penalty*regularizer).backward();optimizer.step()
        model.eval()
        with torch.no_grad():
            train_logits=model(x[tr]);val_logits=model(x[va])
            curves['train_loss'].append(float(nn.functional.cross_entropy(train_logits,y[tr])))
            curves['validation_loss'].append(float(nn.functional.cross_entropy(val_logits,y[va])))
            curves['validation_accuracy'].append(float((val_logits.argmax(1)==y[va]).float().mean()))
    return {'config':cfg,'curves':curves,'validation_loss':curves['validation_loss'][-1],'validation_accuracy':curves['validation_accuracy'][-1],'epochs':epochs,'seed':42},model

def compare_runs(config):
    if config is None:return None
    baseline={'learning_rate':.01,'penalty':0.0}
    if sum(float(config.get(k,-1))!=v for k,v in baseline.items())!=1:raise ValueError('Change exactly one of learning_rate or penalty from the baseline.')
    ar,am=train_candidate(baseline);br,bm=train_candidate(config)
    return [ar,br],[am,bm]

def selection_index(selection,runs):
    if selection is None:return None
    if selection not in (0,1):raise ValueError('Selection must be candidate index0 or1.')
    return int(selection)

def evaluate_reserved(model):
    x,y,ids,tr,va,te=target_data();model.eval()
    with torch.no_grad():pred=model(x[te]).argmax(1)
    return {'test_n':len(te),'accuracy':float((pred==y[te]).float().mean()),'image_ids':[int(i) for i in ids[te]],'protocol':'Reserved test accessed only after selection; do not tune using this result.'}

def training_figure(reports, title="Live short CPU runs"):

    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3),layout='constrained')
    for i,r in enumerate(reports):
        name=('A baseline','B candidate')[i] if len(reports)==2 else f"lr={r['config']['learning_rate']}, penalty={r['config']['penalty']}"
        axes[0].plot(r['curves']['train_loss'],label=name+' train');axes[0].plot(r['curves']['validation_loss'],'--',label=name+' validation')
        axes[1].plot(r['curves']['validation_accuracy'],label=name)
    axes[0].set(title=title,xlabel='epochs',ylabel='mean data cross-entropy');axes[0].legend(fontsize=7)
    axes[1].set(title='Validation accuracy',xlabel='epochs',ylabel='fraction',ylim=(0,1));axes[1].legend(fontsize=7)
    return fig

def adapt(head_classes,trainable_groups,epochs=35):
    if head_classes is None or trainable_groups is None:return None
    if not isinstance(head_classes,int) or head_classes<1:raise ValueError('Head width must be a positive integer.')
    if set(trainable_groups)!={'conv1','conv2','head'} or not all(type(v) is bool for v in trainable_groups.values()):raise ValueError('Return booleans for conv1,conv2,head.')
    if not any(trainable_groups.values()):raise ValueError('Select at least one trainable parameter group.')
    torch.manual_seed(42);base=tiny_model();base.head=nn.Linear(48,head_classes);x,y,ids,tr,va,te=target_data();target=y%2
    if int(target.max())>=head_classes:raise ValueError('A target label lies outside the head output range.')
    for name,p in base.named_parameters():p.requires_grad=name.startswith('head')
    cache=np.load(ASSETS/'cached-features.npz');features=torch.tensor(cache['features'])
    if not np.array_equal(ids,cache['image_ids']):raise ValueError('Cached feature row IDs do not match.')
    opt=torch.optim.SGD(base.head.parameters(),lr=.08)
    for epoch in range(epochs):
        opt.zero_grad();loss=nn.functional.cross_entropy(base.head(features[tr]),target[tr]);loss.backward();opt.step()
    frozen=copy.deepcopy(base).eval();fine=copy.deepcopy(base);before={n:p.detach().clone() for n,p in fine.named_parameters()}
    for name,p in fine.named_parameters():p.requires_grad=trainable_groups[name.split('.')[0]]
    parameter_groups=[{'params':[p for p in getattr(fine,name).parameters() if p.requires_grad], 'lr':.02 if name=='head' else .001} for name,enabled in trainable_groups.items() if enabled]
    opt=torch.optim.SGD(parameter_groups)
    for epoch in range(12):
        fine.train();opt.zero_grad();loss=nn.functional.cross_entropy(fine(x[tr]),target[tr]);loss.backward();opt.step()
    fine.eval();rows=[];predictions_by_route={}
    for name,model in [('Frozen features',frozen),('Last-block fine-tune',fine)]:
        with torch.no_grad():logits=model(x[va]);pred=logits.argmax(1)
        predictions_by_route[name]=pred.tolist()
        rows.append({'route':name,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'validation_loss':float(nn.functional.cross_entropy(logits,target[va])),'validation_accuracy':float((pred==target[va]).float().mean())})
    deltas={n:float((p-before[n]).detach().abs().max()) for n,p in fine.named_parameters()}
    return {'routes':rows,'fine_tune_max_absolute_delta':deltas,'trainable_names':[n for n,p in fine.named_parameters() if p.requires_grad],'source_target_disjoint':True,'source_task':'10-class digits','target_task':'even/odd','validation_predictions':pred.tolist(),'validation_predictions_by_route':predictions_by_route,'validation_comparison':[{'image_id':int(ids[i]),'target':int(target[i]),'head_only':predictions_by_route['Frozen features'][j],'last_block':predictions_by_route['Last-block fine-tune'][j],'changed':predictions_by_route['Frozen features'][j]!=predictions_by_route['Last-block fine-tune'][j]} for j,i in enumerate(va)],'validation_targets':target[va].tolist(),'validation_ids':[int(i) for i in ids[va]],'budgets':{'head_epochs':epochs,'fine_tune_epochs':12},'seed':42}

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

def inspect_detections(filter_scores,iou,image_id,cutoff,nms_cutoff,live=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    from torchvision.ops import batched_nms
    snapshot=detector_snapshot();entry=next(e for e in snapshot['images'] if e['id']==image_id)
    outputs=live if live is not None else entry['outputs'];reports={};fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
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
        ax.imshow(Image.open(ASSETS/entry['image']));ax.set_title(name);ax.axis('off')
        for d in kept:
            x1,y1,x2,y2=d['box'];ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='orange',linewidth=1));ax.text(x1,y1,d['label']+f' {d["score"]:.2f}',fontsize=7,color='black',bbox={'facecolor':'white','alpha':.8})
        matches=[];matched_predictions=set()
        for ref in entry['references']:
            x1,y1,x2,y2=ref['box'];ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='lime',linestyle='--',linewidth=1))
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

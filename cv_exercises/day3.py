"""Prepared Day 3 plumbing. Notebook concept TODO implementations remain unsupplied."""
from pathlib import Path
import json,hashlib,time,threading,tempfile,zipfile,io
import numpy as np
import torch
from torch import nn
from scipy import ndimage
from PIL import Image
from cv_exercises import day2
ROOT=Path(__file__).resolve().parents[1];ASSETS=Path(__file__).resolve().parent/'assets/day3';CACHE=ROOT/'.cache/day3'
torch.set_num_threads(2)
class TinyUNet(nn.Module):
 def __init__(self,out=3):
  super().__init__();self.enc=nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.ReLU(),nn.Conv2d(8,8,3,padding=1),nn.ReLU());self.bottom=nn.Sequential(nn.Conv2d(8,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,16,3,padding=1),nn.ReLU());self.dec=nn.Sequential(nn.Conv2d(24,8,3,padding=1),nn.ReLU(),nn.Conv2d(8,out,1))
 def forward(self,x):
  skip=self.enc(x);deep=self.bottom(torch.max_pool2d(skip,2));up=nn.functional.interpolate(deep,size=skip.shape[-2:],mode='bilinear',align_corners=False);return self.dec(torch.cat([skip,up],1))
class PairEncoder(nn.Module):
 def __init__(self):super().__init__();self.layers=nn.Sequential(nn.Flatten(),nn.Linear(64,48),nn.ReLU(),nn.Linear(48,8))
 def forward(self,x):return nn.functional.normalize(self.layers(x),dim=-1)
def load_checkpoint(name,model):model.load_state_dict(torch.load(ASSETS/name,map_location='cpu',weights_only=True));return model.eval()
def segmentation(index=0):
 data=np.load(ASSETS/'segmentation.npz');image=data['images'][index];x=torch.tensor(image.transpose(2,0,1)[None],dtype=torch.float32)
 with torch.no_grad():probs=load_checkpoint('unet-semantic.pt',TinyUNet())(x).softmax(1)[0].numpy();centers=load_checkpoint('unet-centers.pt',TinyUNet(1))(x).sigmoid()[0,0].numpy()
 sem=probs.argmax(0);fg=sem>0;peaks=(centers==ndimage.maximum_filter(centers,size=7))&(centers>.35)&fg;markers,n=ndimage.label(peaks)
 instance=ndimage.watershed_ift(((1-centers)*255).astype(np.uint8),markers.astype(np.int32)) if n else np.zeros_like(sem);instance[~fg]=0
 return dict(image=image,probabilities=probs,semantic=sem,centers=centers,instance=instance,reference_semantic=data['semantic'][index],reference_instance=data['instances'][index],cluster=data['cluster'][index],name=str(data['names'][index]))
def segmentation_report(select_mask,overlap,index=0,class_id=1):
 s=segmentation(index);mask=select_mask(s['semantic'],class_id)
 if mask is None:return None
 mask=np.asarray(mask)
 if mask.shape!=s['semantic'].shape or mask.dtype!=bool:raise ValueError('Return an H×W Boolean mask for the semantic class.')
 metrics=overlap(mask,s['reference_semantic']==class_id)
 if metrics is None:return None
 if set(metrics)!= {'iou','dice'} or any(not np.isfinite(v) or not 0<=v<=1 for v in metrics.values()):raise ValueError('Return finite iou and dice fractions in [0,1].')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,4,figsize=(12,3),layout='constrained')
 for ax,(title,arr) in zip(axes,[('RGB',s['image']),('Day 1 cluster',s['cluster']),('U-Net semantic',s['semantic']),('Learned centers → instances',s['instance'])]):ax.imshow(arr,cmap='tab10' if arr.ndim==2 else None);ax.set_title(title);ax.axis('off')
 report=dict(scene=s['name'],class_id=class_id,metrics={k:float(v) for k,v in metrics.items()},instance_ids=np.unique(s['instance']).tolist(),reference_instance_count=int(s['reference_instance'].max()),scope='Authored geometric scenes; independently pretrained semantic/center U-Nets with supplied watershed. Not Mask R-CNN or natural-photo quality.')
 return report,fig,s
def pairs():
 d=np.load(ASSETS/'pairs.npz');model=load_checkpoint('siamese.pt',PairEncoder())
 with torch.no_grad():a=model(torch.tensor(d['a'],dtype=torch.float32)).numpy();b=model(torch.tensor(d['b'],dtype=torch.float32)).numpy()
 return d,a,b
def pair_report(distance,contrastive,threshold=.7):
 d,a,b=pairs();dist=distance(a,b)
 if dist is None:return None
 dist=np.asarray(dist,dtype=float)
 if dist.shape!=(len(a),) or not np.isfinite(dist).all() or (dist<0).any():raise ValueError('Return one finite nonnegative distance per pair.')
 loss=contrastive(dist,d['similar'],1.)
 if loss is None:return None
 loss=np.asarray(loss,dtype=float)
 if loss.shape!=dist.shape or not np.isfinite(loss).all() or (loss<0).any():raise ValueError('Return nonnegative losses; y=1 means similar.')
 decisions=dist<=threshold;truth=d['similar'].astype(bool)
 rows=[dict(pair=i,left_digit=int(d['labels_a'][i]),right_digit=int(d['labels_b'][i]),similar_target=bool(truth[i]),distance=float(dist[i]),threshold=threshold,match=bool(decisions[i]),correct=bool(decisions[i]==truth[i]),loss=float(loss[i])) for i in range(len(a))]
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,8,figsize=(12,3),layout='constrained')
 for i in range(8):
  for row,key in enumerate(['a','b']):axes[row,i].imshow(d[key][i].reshape(8,8),cmap='gray');axes[row,i].axis('off')
  axes[0,i].set_title(f'pair {i} · d={dist[i]:.2f}')
 return dict(pairs=rows,false_matches=int((decisions&~truth).sum()),missed_matches=int((~decisions&truth).sum()),mean_loss=float(loss.mean()),margin=1.,model='trained shared 64→48→8 unit-normalized pair encoder'),fig
def model_specs():return json.loads((ASSETS/'model-manifest.json').read_text(encoding='utf-8'))
def prepare_foundation(online=True):
 from huggingface_hub import snapshot_download
 return {key:snapshot_download(s['id'],revision=s['revision'],cache_dir=str(CACHE/'hub'),local_files_only=not online,allow_patterns=['*.json','*.txt',s.get('weight_file','*.safetensors')],ignore_patterns=['tf*','flax*']) for key,s in model_specs()['foundation'].items()}
_SAM=None;_SAM_PROCESSOR=None;_CLIP=None;_CLIP_PROCESSOR=None
def text_scores(candidates,image_id='bus',online=False):
 global _CLIP,_CLIP_PROCESSOR
 from transformers import CLIPModel,CLIPProcessor
 if _CLIP is None:
  path=prepare_foundation(online)['clip'];_CLIP=CLIPModel.from_pretrained(path).eval();_CLIP_PROCESSOR=CLIPProcessor.from_pretrained(path,use_fast=False)
 if not 2<=len(candidates)<=6 or any(not isinstance(c,str) or not c.strip() for c in candidates):raise ValueError('Supply 2–6 nonempty candidates.')
 inputs=_CLIP_PROCESSOR(text=candidates,return_tensors='pt',padding=True);cached=torch.load(ASSETS/'foundation-encodings.pt',weights_only=True)
 with torch.no_grad():
  text=nn.functional.normalize(_CLIP.get_text_features(**inputs),dim=-1);image=nn.functional.normalize(cached['clip_'+image_id],dim=-1);cosine=(image@text.T)[0];logits=cosine*_CLIP.logit_scale.exp();probs=logits.softmax(-1)
 return dict(image=image_id,candidates=list(candidates),cosines=cosine.tolist(),scores=probs.tolist(),scale=float(_CLIP.logit_scale.exp().detach()),scope='Softmax over these candidates only; actual text inference and cached matching image encoding.')
def spatial_mask(prompt,image_id='bus',online=False):
 global _SAM,_SAM_PROCESSOR
 from transformers import SamModel,SamProcessor
 if _SAM is None:
  path=prepare_foundation(online)['sam'];_SAM=SamModel.from_pretrained(path).eval();_SAM_PROCESSOR=SamProcessor.from_pretrained(path,use_fast=False)
 image=Image.open(ASSETS/(image_id+'.jpg')).convert('RGB');w,h=image.size;opts={}
 if prompt.get('points'):
  points=prompt['points'];labels=prompt.get('labels',[1]*len(points))
  if len(labels)!=len(points) or any(v not in (0,1) for v in labels):raise ValueError('One binary point label is required per point.')
  if any(len(p)!=2 or not 0<=p[0]<w or not 0<=p[1]<h for p in points):raise ValueError('Points must be original-image x,y pixels inside image.')
  opts.update(input_points=[[points]],input_labels=[[labels]])
 if prompt.get('box'):
  box=prompt['box']
  if len(box)!=4 or not 0<=box[0]<box[2]<=w or not 0<=box[1]<box[3]<=h:raise ValueError('Box is valid original-image xyxy pixels.')
  opts['input_boxes']=[[box]]
 if not opts:raise ValueError('Supply a point or box.')
 inputs=_SAM_PROCESSOR(images=image,return_tensors='pt',**opts);cached=torch.load(ASSETS/'foundation-encodings.pt',weights_only=True);inputs.pop('pixel_values')
 with torch.no_grad():out=_SAM(image_embeddings=cached['sam_'+image_id],**{k:v for k,v in inputs.items() if k in ['input_points','input_labels','input_boxes']},multimask_output=True)
 masks=_SAM_PROCESSOR.image_processor.post_process_masks(out.pred_masks,inputs['original_sizes'],inputs['reshaped_input_sizes'])[0][0].numpy();scores=out.iou_scores[0,0].numpy();chosen=int(scores.argmax())
 return dict(image=image_id,prompt=prompt,model_mask_quality_scores=scores.tolist(),chosen=chosen,masks=masks,mask=masks[chosen],scope='Actual SAM decoder on cached image embedding. Predicted IoU is a quality estimate, not measured accuracy or a class.')
def prompt_report(candidates,prompt,image_id='bus'):
 if candidates is None or prompt is None:return None
 text=text_scores(candidates,image_id);space=spatial_mask(prompt,image_id)
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');image=np.asarray(Image.open(ASSETS/(image_id+'.jpg')))
 axes[0].barh(candidates,text['scores']);axes[0].set_xlim(0,1);axes[0].set_title('CLIP candidate scores');axes[1].imshow(image);axes[1].imshow(np.ma.masked_where(~space['mask'],space['mask']),alpha=.45,cmap='spring');axes[1].axis('off');axes[1].set_title('SAM spatial prompt')
 for x,y in prompt.get('points',[]):axes[1].plot(x,y,'yo',markersize=5)
 report=dict(text=text,spatial={k:v for k,v in space.items() if k not in ['masks','mask']},mask_pixels=int(space['mask'].sum()),image_shape=list(image.shape),fixed='image, model revisions, cached image encodings; weights never updated')
 return report,fig,space['mask']
class DomainPipeline:
 def __init__(self,model,config):self.model=model.eval();self.config=config
 def predict(self,pixels):
  x=torch.tensor(pixels,dtype=torch.float32)
  if x.shape!=(8,8) or not torch.isfinite(x).all() or x.min()<0 or x.max()>16:raise ValueError('Expected finite 8×8 values in [0,16].')
  x=x[None,None]/float(self.config['processor']['scale'])
  with torch.no_grad():scores=self.model(x).softmax(-1)[0]
  return dict(label=self.config['id2label'][str(int(scores.argmax()))],scores=scores.tolist(),revision=self.config['source_revision'])
 @classmethod
 def reload(cls,path):
  path=Path(path);cfg=json.loads((path/'config.json').read_text(encoding='utf-8'));model=day2.TinyCNN(2);model.load_state_dict(torch.load(path/'model.pt',weights_only=True,map_location='cpu'));return cls(model,cfg)
def adapt_package(config,epochs=40):
 if config is None:return None
 if config.get('trainable')!='head' or config.get('id2label')!={'0':'even','1':'odd'}:raise ValueError('Use head-only tuning with the stated even/odd label order.')
 rate=float(config.get('learning_rate',0))
 if not .001<=rate<=1:raise ValueError('Choose a finite head learning rate from .001 to1.')
 x,y,ids,tr,va,te=day2.target_data();target=y%2;shifted=(x*.45+.25).clamp(0,1);torch.manual_seed(123);model=day2.tiny_model();model.head=nn.Linear(48,2)
 for name,param in model.named_parameters():param.requires_grad_(name.startswith('head.'))
 before={k:v.detach().clone() for k,v in model.state_dict().items()}
 def evaluate(data):
  with torch.no_grad():return float((model(data[va]).argmax(1)==target[va]).float().mean())
 baseline=dict(source_like=evaluate(x),target_shifted=evaluate(shifted));opt=torch.optim.Adam(model.head.parameters(),lr=rate)
 with torch.no_grad():features=model.features(shifted[tr])
 start=time.perf_counter()
 for _ in range(epochs):opt.zero_grad();loss=nn.functional.cross_entropy(model.head(features),target[tr]);loss.backward();opt.step()
 comparison=dict(source_like=evaluate(x),target_shifted=evaluate(shifted));delta={k:float((v-before[k]).abs().max()) for k,v in model.state_dict().items()};revision=hashlib.sha256((day2.ASSETS/'tiny-cnn.pt').read_bytes()).hexdigest()
 cfg=dict(architecture='TinyCNN(2)',source_revision=revision,source_checkpoint='Day2 tiny-cnn.pt',processor=dict(shape=[8,8],channels=1,scale=16.,raw_range=[0,16]),id2label=config['id2label'],task='image-classification',trainable='head',domain_shift='x*.45+.25',epochs=epochs,seed=123,learning_rate=rate,optimizer='Adam')
 path=Path(tempfile.mkdtemp(prefix='cv-inference-package-'));torch.save(model.state_dict(),path/'model.pt');(path/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8');(path/'README.md').write_text('# Teaching digit pipeline\n8×8 even/odd CPU classifier.\n'+json.dumps(cfg,indent=2),encoding='utf-8')
 a=DomainPipeline(model,cfg);b=DomainPipeline.reload(path);sample=(shifted[va[0],0].numpy()*16).tolist();pa=a.predict(sample);pb=b.predict(sample)
 from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def do_POST(self):
   try:
    if self.path!='/predict':raise ValueError('Unknown route')
    size=int(self.headers.get('Content-Length','0'))
    if size>20000:raise ValueError('Payload too large')
    reply=b.predict(json.loads(self.rfile.read(size))['pixels']);status=200
   except (ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:reply={'error':str(exc)};status=400
   data=json.dumps(reply).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(data)
 server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 from urllib.request import Request,urlopen
 url=f'http://127.0.0.1:{server.server_port}/predict'
 from urllib.error import HTTPError
 invalid_status=None
 try:
  with urlopen(Request(url,data=json.dumps({'pixels':sample}).encode(),headers={'Content-Type':'application/json'}),timeout=10) as response:pc=json.load(response);status=response.status
  try:
   urlopen(Request(url,data=json.dumps({'pixels':[[0]]}).encode(),headers={'Content-Type':'application/json'}),timeout=10)
  except HTTPError as error:
   invalid_status=error.code
   invalid_response=json.load(error)
 finally:server.shutdown();server.server_close();thread.join(timeout=2)
 package=io.BytesIO()
 with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
  for file in path.iterdir():z.write(file,file.name)
 return dict(baseline_validation=baseline,adapted_validation=comparison,validation_count=len(va),test_count_unused=len(te),parameter_max_deltas=delta,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),training_seconds=time.perf_counter()-start,reload_scores_max_difference=float(np.max(np.abs(np.array(pa['scores'])-pb['scores']))),request=dict(url=url,method='POST',json={'pixels':sample}),response=pc,http_status=status,invalid_request_status=invalid_status,invalid_request_response=invalid_response,reload_agreement=pa==pb==pc,config=cfg,package_files=sorted(p.name for p in path.iterdir()),scope='Local HTTP endpoint; no Hub/hosted deployment executed.'),package.getvalue()

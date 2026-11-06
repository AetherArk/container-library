"""Full-output observer extracted from observe_v2; no campaign dependencies."""
import sys,time,json,traceback
from pathlib import Path
import torch
import boltz.main as main
from boltz.model.models.boltz2 import Boltz2
from runtime import native_args, offline_assets
# Native's downloader otherwise requests an unused affinity checkpoint even for
# structure-only inputs. Keep all model/parser/writer calls unchanged.
main.download_boltz2=offline_assets
root=Path('/run'); seed=int(sys.argv[1]); stage=root/f'seed-{seed}'
request=json.loads((root/'request.json').read_text()); settings=request['settings']
t0=time.monotonic(); capture=None
def event(name,**kw):
 with (stage/'events.jsonl').open('a') as f: f.write(json.dumps({'event':name,'elapsed':time.monotonic()-t0,**kw},default=str)+'\n')
def cpu(x):
 if torch.is_tensor(x): return x.detach().cpu()
 if isinstance(x,dict): return {k:cpu(v) for k,v in x.items()}
 if isinstance(x,list): return [cpu(v) for v in x]
 if isinstance(x,tuple): return tuple(cpu(v) for v in x)
 return x

def save(label,x):
 obj=cpu(x); path=capture/(label+'.pt');torch.save(obj,path)
 inventory={}
 def visit(v,p):
  if torch.is_tensor(v): inventory[p]={'shape':list(v.shape),'dtype':str(v.dtype),'finite':bool(torch.isfinite(v).all()) if v.is_floating_point() else True}
  elif isinstance(v,dict):
   for k,w in v.items(): visit(w,p+'/'+str(k))
  elif isinstance(v,(list,tuple)):
   for i,w in enumerate(v): visit(w,p+'/'+str(i))
 visit(obj,'');(capture/(label+'_inventory.json')).write_text(json.dumps(inventory,indent=2));event('saved',label=label,bytes=path.stat().st_size,fields=list(x) if isinstance(x,dict) else None)
 return obj

original_load=Boltz2.load_from_checkpoint
@classmethod
def load(cls,*args,**kw):
 event('load_start',args=args,config=kw); model=original_load(*args,**kw);event('load_end');return model
Boltz2.load_from_checkpoint=load
original_forward=Boltz2.forward
original_predict=Boltz2.predict_step
def forward(self,*args,**kw):
 torch.cuda.synchronize();start=time.monotonic()
 out=original_forward(self,*args,**kw)
 torch.cuda.synchronize();event('forward_end',seconds=time.monotonic()-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
 save('full_forward',out);return out
def predict(self,batch,*args,**kw):
 global capture
 rid=batch['record'][0].id
 capture=stage/'capture'/rid;capture.mkdir(parents=True,exist_ok=False)
 save('consumed_batch',batch)
 event('predict_start',record=rid,predict_args=self.predict_args,steering_args=self.steering_args)
 out=original_predict(self,batch,*args,**kw);save('full_prediction',out);event('predict_end',record=rid);return out
Boltz2.forward=forward;Boltz2.predict_step=predict
args=native_args(stage,seed,settings)
(stage/'argv.json').write_text(json.dumps(args,indent=2));event('cli_start')
try:
 main.cli(args,standalone_mode=False);event('cli_end')
except BaseException:
 event('cli_error',traceback=traceback.format_exc());raise

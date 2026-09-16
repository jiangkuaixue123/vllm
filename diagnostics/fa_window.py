import json
from pathlib import Path
import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func as fa
meta=json.loads(Path(__file__).with_name('replay_metadata.json').read_text())
torch.manual_seed(0)
n,total=4784,8192
base=[torch.randn(total,16,80,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
cu=torch.tensor(meta['window_cu'],device='cuda',dtype=torch.int32)
ref=[]
for a,b in zip(meta['window_cu'],meta['window_cu'][1:]):
 if a==b:continue
 ref.append(torch.nn.functional.scaled_dot_product_attention(*[x[a:b].transpose(0,1).unsqueeze(0) for x in base]).squeeze(0).transpose(0,1))
ref=torch.cat(ref)
for version in [2,3]:
 for mode in ['eager','graph']:
  for tail in ['zero','random','nan_q','nan_k','nan_v','nan_all']:
   xs=[x.clone() for x in base]
   for i,x in enumerate(xs):
    if tail=='zero':x[n:].zero_()
    elif tail=='nan_all' or tail=='nan_'+['q','k','v'][i]:x[n:].fill_(float('nan'))
   out=torch.zeros_like(xs[0])
   def run():return fa(*xs,cu_seqlens_q=cu,cu_seqlens_k=cu,max_seqlen_q=64,max_seqlen_k=64,fa_version=version,out=out)
   run()
   if mode=='graph':
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):run()
    graph.replay()
   bad=(~out[:n].isfinite()).any(dim=(1,2)).nonzero().flatten().tolist()
   print(json.dumps(dict(version=version,mode=mode,tail=tail,bad_count=len(bad),first_bad=bad[0] if bad else None,last_bad=bad[-1] if bad else None,max_error=(out[:n]-ref).abs().max().item())),flush=True)

import json
import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func as fa

torch.manual_seed(0)
n, total, heads, dim = 4784, 8192, 16, 80
base = [torch.randn(total, heads, dim, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
cu = torch.tensor([0]+[n]*64, device='cuda', dtype=torch.int32)
ref = torch.nn.functional.scaled_dot_product_attention(*[x[:n].transpose(0,1).unsqueeze(0) for x in base]).squeeze(0).transpose(0,1)
for version in [3,2]:
 for mode in ['eager','graph']:
  for tail in ['zero','random','nan_q','nan_k','nan_v','nan_all']:
   xs=[x.clone() for x in base]
   for i,x in enumerate(xs):
    if tail=='zero': x[n:].zero_()
    elif tail=='nan_all' or tail=='nan_'+['q','k','v'][i]: x[n:].fill_(float('nan'))
   out=torch.full_like(xs[0],float('nan'))
   def run():
    return fa(*xs,cu_seqlens_q=cu,cu_seqlens_k=cu,max_seqlen_q=total,max_seqlen_k=total,softmax_scale=dim**-0.5,causal=False,fa_version=version,out=out)
   run()
   if mode=='graph':
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): run()
    out.fill_(float('nan')); graph.replay()
   torch.cuda.synchronize()
   print(json.dumps(dict(version=version,mode=mode,tail=tail,live_finite=torch.isfinite(out[:n]).float().mean().item(),tail_finite=torch.isfinite(out[n:]).float().mean().item(),max_error=(out[:n]-ref).abs().max().item())),flush=True)

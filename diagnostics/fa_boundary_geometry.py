import json
import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func as fa
torch.manual_seed(0)
total=8192
base=[torch.randn(total,16,80,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
for n in [4736,4784,4800,4864]:
 for layout in ['single','empty64','two_real']:
  cu=torch.tensor([0,n] if layout=='single' else ([0]+[n]*64 if layout=='empty64' else [0,n,total]),device='cuda',dtype=torch.int32)
  ref=fa(*[x[:n] for x in base],cu_seqlens_q=torch.tensor([0,n],device='cuda',dtype=torch.int32),cu_seqlens_k=torch.tensor([0,n],device='cuda',dtype=torch.int32),max_seqlen_q=n,max_seqlen_k=n,fa_version=3)
  for poison in [n,n+15,n+16,n+47,n+48,n+127,n+128]:
   v=base[2].clone();v[poison].fill_(float('nan'))
   out=fa(base[0],base[1],v,cu_seqlens_q=cu,cu_seqlens_k=cu,max_seqlen_q=total,max_seqlen_k=total,fa_version=3)
   print(json.dumps(dict(n=n,layout=layout,poison=poison,live_finite=torch.isfinite(out[:n]).float().mean().item(),max_error=(out[:n]-ref).abs().max().item())),flush=True)

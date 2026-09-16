"""Two real sequences; V NaNs in sequence 2 must not contaminate sequence 1."""
import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func as fa

torch.manual_seed(0)
n, total = 80, 144
q, k, v = [torch.randn(total,16,80,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
cu = torch.tensor([0,n,total],device='cuda',dtype=torch.int32)
ref = torch.nn.functional.scaled_dot_product_attention(*[x[:n].transpose(0,1).unsqueeze(0) for x in (q,k,v)]).squeeze(0).transpose(0,1)
for version in [2,3]:
 for poison in [False,True]:
  vv=v.clone()
  if poison: vv[n:].fill_(float('nan'))
  out=fa(q,k,vv,cu_seqlens_q=cu,cu_seqlens_k=cu,max_seqlen_q=80,max_seqlen_k=80,fa_version=version,out=torch.zeros_like(q))
  print(dict(version=version,poison=poison,finite_first=out[:n].isfinite().all().item(),max_error=(out[:n]-ref).abs().max().item()),flush=True)

import os
os.environ.setdefault('VLLM_USE_V2_MODEL_RUNNER','1')
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION']='1'
if os.environ.get('CONTROL'):
 import diagnostics.controls
if os.environ.get('PROBE'):
 import diagnostics.instrument
from functools import partial
from transformers import AutoProcessor
from vllm import LLM
from tests.conftest import IMAGE_ASSETS

def compare(worker,batches):
 import torch
 r=worker.model_runner; m=r.model_state.encoder_runner.cudagraph_manager if hasattr(r, 'model_state') else r.encoder_cudagraph_manager
 def delta(a,b):
  d=(a-b).abs(); return {'finite_a':torch.isfinite(a).float().mean().item(),'finite_b':torch.isfinite(b).float().mean().item(),'max':d.max().item(),'mean':d.float().mean().item(),'mismatch':(d>0.016+0.016*b.abs()).float().mean().item()}
 results=[]
 with torch.inference_mode():
  for batch in batches:
   kw={k:v.to(r.device) for k,v in batch.items()}
   if hasattr(m,'diag_states'):
    m.diag_states.zero_(); m.diag_live.fill_(kw['pixel_values'].shape[0])
   actual=m.execute(kw)[0]; n=actual.shape[0]
   if hasattr(m,'diag_states'): print('FLAGS',list(zip(m.diag_names,m.diag_states[:len(m.diag_names)].tolist())),flush=True)
   budget=min(b for b in m.budget_graphs['default'] if b>=n); gm=m.budget_graphs['default'][budget]
   print('METADATA',{'budget':budget,'shapes':{k:list(v.shape) for k,v in gm.input_buffers.items() if isinstance(v,torch.Tensor)},'window_valid_permutation':bool(torch.equal(gm.input_buffers['window_index'][:n].sort().values,torch.arange(n,device=r.device))),'full_cu':gm.input_buffers['cu_seqlens'].tolist(),'window_cu':gm.input_buffers['cu_window_seqlens'].tolist()},flush=True)
   padded=r.model.encoder_cudagraph_forward(dict(gm.input_buffers))[:n]; eager=r.model.encoder_eager_forward(kw)
   results.append({'tokens':n,'patches':kw['pixel_values'].shape[0],'grid':kw['image_grid_thw'].tolist(),'backend':str(r.model.visual.attn_backend),'budget':budget,'graph_vs_padded':delta(actual,padded),'graph_vs_eager':delta(actual,eager),'padded_vs_eager':delta(padded,eager)})
 return results

def main():
 model='Qwen/Qwen2.5-VL-3B-Instruct'; p=AutoProcessor.from_pretrained(model,revision='66285546d2b821cf421d4f5eb2576359d3770cd3').image_processor
 batches=[dict(p(images=[IMAGE_ASSETS[0].pil_image.resize(size)],return_tensors='pt',do_rescale=False,do_normalize=False)) for size in [(224,224),(1280,720),(224,224)]]
 llm=LLM(**({'kernel_config':{'enable_jit_warmup':False,'enable_cutedsl_warmup':False}} if os.environ.get('RUNNER')=='full' else {}),model=model,revision='66285546d2b821cf421d4f5eb2576359d3770cd3',seed=0,gpu_memory_utilization=0.5,mm_encoder_only=os.environ.get('RUNNER','eonly')=='eonly',enable_prefix_caching=False,dtype='bfloat16',max_model_len=16384,max_num_seqs=32,limit_mm_per_prompt={'image':2,'video':0},compilation_config={**({'mode':0} if os.environ.get('RUNNER')=='full' else {}),'cudagraph_mode':'FULL' if os.environ.get('VLLM_USE_V2_MODEL_RUNNER')=='0' else 'NONE',**({'cudagraph_capture_sizes':[1]} if os.environ.get('VLLM_USE_V2_MODEL_RUNNER')=='0' else {}),'cudagraph_mm_encoder':True,**({'encoder_cudagraph_token_budgets':[2048]} if os.environ.get('SINGLE_BUDGET') else {})})
 try: print('DIAGNOSTIC',llm.collective_rpc(partial(compare,batches=batches)),flush=True)
 finally: llm.llm_engine.engine_core.shutdown()
if __name__=='__main__':main()

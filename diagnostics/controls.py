import os
import torch
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager
from vllm.v1.attention.backends import fa_utils

mode=os.environ.get('CONTROL')
if mode=='freshpool':
 old=EncoderCudaGraphManager._capture_budget_graph
 def capture(self,*args,**kwargs):
  self.graph_pool=torch.cuda.graph_pool_handle()
  return old(self,*args,**kwargs)
 EncoderCudaGraphManager._capture_budget_graph=capture
elif mode in ['zero','fa2']:
 old=fa_utils.flash_attn_varlen_func
 def call(*args,**kwargs):
  if mode=='zero':kwargs['out']=torch.zeros_like(args[0] if args else kwargs['q'])
  else:kwargs['fa_version']=2
  return old(*args,**kwargs)
 fa_utils.flash_attn_varlen_func=call

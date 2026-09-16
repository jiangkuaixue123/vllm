"""Finite flags only: no retained activations or capture-time host reads."""
import os
import torch
import triton
import triton.language as tl
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

@triton.jit
def flags(X, S, LIVE, ROWS:tl.constexpr, WIDTH:tl.constexpr, STRIDE:tl.constexpr, POINT:tl.constexpr, B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    row=i//WIDTH
    x=tl.load(X+row*STRIDE+i%WIDTH,i<ROWS*WIDTH,other=0).to(tl.float32)
    bad=(x!=x)|(tl.abs(x)==float('inf'))
    live=tl.load(LIVE)
    a=tl.sum(((row<live)&bad&(i<ROWS*WIDTH)).to(tl.int32),0)
    b=tl.sum(((row>=live)&bad&(i<ROWS*WIDTH)).to(tl.int32),0)
    tl.atomic_add(S+POINT*2,a)
    tl.atomic_add(S+POINT*2+1,b)

old=EncoderCudaGraphManager.capture

def capture(self,graph_pool):
    visual=self.model.visual
    self.diag_live=torch.tensor([4784],device=self.device,dtype=torch.int32)
    self.diag_states=torch.zeros((256,2),device=self.device,dtype=torch.int32)
    self.diag_names=[]
    handles=[]
    def record(name,x,axis=0):
        if isinstance(x,tuple):x=x[0]
        if name not in self.diag_names:self.diag_names.append(name)
        p=self.diag_names.index(name)
        rows=x.shape[axis]; width=x.numel()//rows
        flags[(triton.cdiv(rows*width,2048),)](x,self.diag_states,self.diag_live,rows,width,x.stride(axis),p,2048)
    def post(name):
        return lambda mod,args,out:record(name,out)
    def pre_attn(name):
        def hook(mod,args,kwargs):
            for k in ['query','key','value']: record(name+'.'+k,kwargs[k],1)
        return hook
    for i,block in enumerate(visual.blocks):
        if os.environ.get('PROBE')=='fine' and i<=8:
            handles.append(block.attn.attn.register_forward_pre_hook(pre_attn(f'{i}.attention'),with_kwargs=True))
            handles.append(block.attn.attn.register_forward_hook(lambda mod,args,out,i=i:record(f'{i}.attention.out',out,1)))
            for sub in ['norm1','attn.qkv','attn.proj','norm2','mlp.gate_up_proj','mlp.down_proj']:
                handles.append(block.get_submodule(sub).register_forward_hook(post(f'{i}.{sub}')))
        handles.append(block.register_forward_hook(post(f'{i}.block.out')))
    try:return old(self,graph_pool)
    finally:
        for h in handles:h.remove()
EncoderCudaGraphManager.capture=capture

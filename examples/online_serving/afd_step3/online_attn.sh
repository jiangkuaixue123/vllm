export ASCEND_RT_VISIBLE_DEVICES=4,5
export HCCL_BUFFSIZE=1024
# vllm serve "/home/data/DeepSeek-V2-Lite"  --tensor_parallel_size=2 --enable_expert_parallel --enforce_eager  --afd-config '{"afd_connector":"p2pconnector", "afd_role": "attention", "num_afd_stages":"1","afd_extra_config":{"afd_size":"2A2F"}}' #--additional-config='{"role":"attention", "ffn_size":1, "attn_size":1}'

# vllm serve /home/y00889327/DSV2LiteWeight\
# python -m debugpy --listen 56306 --wait-for-client $(which vllm) serve /home/y00889327/DSV2LiteWeight \
vllm serve /home/y00889327/DSV2LiteWeight\
    --tensor-parallel-size 2 \
    --enable_expert_parallel \
    --enforce_eager          \
    --afd-config \
    '{"afd_connector":"p2pconnector", "afd_role": "attention", "num_afd_stages":"1","afd_extra_config":{"afd_size":"2A2F"}}'

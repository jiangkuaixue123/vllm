export ASCEND_RT_VISIBLE_DEVICES=6,7
export HCCL_BUFFSIZE=1024
# vllm fserver "/home/data/DeepSeek-V2-Lite" --tensor_parallel_size=2 --enable_expert_parallel --enforce_eager --afd-config '{"afd_connector":"p2pconnector", "num_afd_stages":"1", "afd_role": "ffn", "afd_extra_config":{"afd_size":"2A2F"}}' #--additional-config='{"role":"ffn", "ffn_size":2, "attn_size":2}'
# comm = python -m debugpy --listen 56307 --wait-for-client $(which vllm) serve /home/y00889327/DSV2LiteWeight

# /home/y00889327/workspace-afd/vllm/vllm/entrypoints/afd_ffn_server.py
#python -m debugpy --listen 56307 --wait-for-client -m vllm.entrypoints.afd_ffn_server /home/y00889327/DSV2LiteWeight\
# python -m vllm.entrypoints.afd_ffn_server /home/y00889327/DSV2LiteWeight \
python -m vllm.entrypoints.afd_ffn_server /home/y00889327/DSV2LiteWeight \
        --tensor-parallel-size 2 \
        --enable_expert_parallel \
        --enforce_eager          \
        --afd-config '{"afd_connector":"p2pconnector", "num_afd_stages":"1", "afd_role": "ffn", "afd_extra_config":{"afd_size":"2A2F"}}'
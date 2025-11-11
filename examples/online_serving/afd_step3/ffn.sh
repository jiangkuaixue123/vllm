export ASCEND_RT_VISIBLE_DEVICES=6,7
export HCCL_BUFFSIZE=1024
export VLLM_LOGGING_LEVEL=DEBUG
# export ASCEND_LAUNCH_BLOCKING=1
# vllm fserver "/home/data/DeepSeek-V2-Lite" --tensor_parallel_size=2 --enable_expert_parallel --enforce_eager --afd-config '{"afd_connector":"p2pconnector", "num_afd_stages":"1", "afd_role": "ffn", "afd_extra_config":{"afd_size":"2A2F"}}' #--additional-config='{"role":"ffn", "ffn_size":2, "attn_size":2}'
# comm = python -m debugpy --listen 56307 --wait-for-client $(which vllm) serve /home/l00851163/afd/DSV2LiteWeight

# /home/y00889327/workspace-afd/vllm/vllm/entrypoints/afd_ffn_server.py
#python -m debugpy --listen 56307 --wait-for-client -m vllm.entrypoints.afd_ffn_server /home/l00851163/afd/DSV2LiteWeight\
# python -m vllm.entrypoints.afd_ffn_server /home/l00851163/afd/DSV2LiteWeight \
#        --enforce_eager          \
#        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'  \
python -m vllm.entrypoints.afd_ffn_server /home/l00851163/afd/DSV2LiteWeight \
        --tensor-parallel-size 2 \
        --enable_expert_parallel \
        --max_num_batched_tokens 200 \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'  \
        --cuda_graph_sizes 4 \
        --afd-config '{"afd_connector":"m2nconnector", "num_afd_stages":"1", "afd_role": "ffn", "afd_extra_config":{"afd_size":"2A2F"}}'
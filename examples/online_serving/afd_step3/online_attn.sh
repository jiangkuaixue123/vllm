# --enforce-eager \
# --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes":[4]}'  \
# python -m debugpy --listen 56306 --wait-for-client $(which vllm) serve /home/l00851163/afd/DSV2LiteWeight \
export ASCEND_RT_VISIBLE_DEVICES=0,1
export HCCL_BUFFSIZE=2048
export VLLM_LOGGING_LEVEL=DEBUG
vllm serve /home/l00851163/afd/DSV2LiteWeight \
    --tensor-parallel-size 2 \
    --enable_expert_parallel \
    --max_num_batched_tokens 4 \
    --max_num_seqs 4 \
    --enforce-eager \
    --port 8006 \
    --max-model-len 4096 \
    --afd-config '{"afd_connector":"p2pconnector", "afd_role": "attention", "num_afd_stages":"1","afd_extra_config":{"afd_size":"2A2F"}, "compute_gate_on_attention": "True"}'
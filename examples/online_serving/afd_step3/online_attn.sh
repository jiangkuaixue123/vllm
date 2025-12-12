# --enforce-eager \
# --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes":[20]}'  \
# python -m debugpy --listen 56306 --wait-for-client $(which vllm) serve /home/l00851163/afd/DSV2LiteWeight \
# vllm serve /home/l00851163/afd/DSV2LiteWeight \
#    --enable-dbo \
#    --dbo-prefill-token-threshold 12 \
#    --dbo-decode-token-threshold 2 \
# --tensor-parallel-size 2 \
# --data-parallel-size 2 \
export ASCEND_RT_VISIBLE_DEVICES=0,1
export HCCL_BUFFSIZE=2048
export VLLM_LOGGING_LEVEL=DEBUG
vllm serve /home/l00851163/afd/DSV2LiteWeight \
    --data-parallel-size 2 \
    --enable_expert_parallel \
    --max_num_batched_tokens 20 \
    --max_num_seqs 20 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes":[20]}'  \
    --port 8006 \
    --max-model-len 4096 \
    --enable-dbo \
    --dbo-prefill-token-threshold 12 \
    --dbo-decode-token-threshold 2 \
    --afd-config '{"afd_connector":"m2nconnector", "afd_role": "attention", "num_afd_stages":"2","afd_extra_config":{"afd_size":"2A2F"}, "compute_gate_on_attention": "True"}'  |& tee /home/y00889327/workspace-afd/test-vllm-ascend/attn.log
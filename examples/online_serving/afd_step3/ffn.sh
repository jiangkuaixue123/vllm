# --enforce-eager \
# --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes":[4]}'  \
#python -m debugpy --listen 56307 --wait-for-client -m vllm.entrypoints.afd_ffn_server /home/l00851163/afd/DSV2LiteWeight\
#python -m vllm.entrypoints.afd_ffn_server /home/l00851163/afd/DSV2LiteWeight\
export ASCEND_RT_VISIBLE_DEVICES=2,3
export HCCL_BUFFSIZE=2048
export VLLM_LOGGING_LEVEL=DEBUG
python -m vllm.entrypoints.afd_ffn_server /home/l00851163/afd/DSV2LiteWeight\
        --tensor-parallel-size 2 \
        --enable_expert_parallel \
        --max_num_batched_tokens 20 \
        --max_num_seqs 20 \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes":[20]}'  \
        --max-model-len 4096 \
        --afd-config '{"afd_connector":"m2nconnector", "num_afd_stages":"2", "afd_role": "ffn", "afd_extra_config":{"afd_size":"2A2F"}, "compute_gate_on_attention": "True"}' |& tee /home/y00889327/workspace-afd/test-vllm-ascend/ffn.log
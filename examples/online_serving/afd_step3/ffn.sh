export ASCEND_RT_VISIBLE_DEVICES=14,15
export HCCL_BUFFSIZE=2048
export VLLM_LOGGING_LEVEL=DEBUG
python -m vllm.entrypoints.afd_ffn_server /home/j00586476/DeepSeek-V2-Lite \
        --tensor-parallel-size 2 \
        --enable_expert_parallel \
        --max_num_batched_tokens 4 \
        --max_num_seqs 4 \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes":[4]}'  \
        --max-model-len 4096 \
        --afd-config '{"afd_connector":"m2nconnector", "num_afd_stages":"1", "afd_role": "ffn", "afd_extra_config":{"afd_size":"2A2F"}, "compute_gate_on_attention": "True"}'
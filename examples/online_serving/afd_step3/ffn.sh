export ASCEND_RT_VISIBLE_DEVICES=6,7
# vllm fserver "/home/data/DeepSeek-V2-Lite" --tensor_parallel_size=2 --enable_expert_parallel --enforce_eager --afd-config '{"afd_connector":"p2pconnector", "num_afd_stages":"1", "afd_role": "ffn", "afd_extra_config":{"afd_size":"2A2F"}}' #--additional-config='{"role":"ffn", "ffn_size":2, "attn_size":2}'


python -m vllm.entrypoints.afd_ffn_server /home/data/DeepSeek-V2-Lite \
        --tensor-parallel-size 2 \
        --enable_expert_parallel \
        --enforce_eager          \
        --afd-config '{"afd_connector":"p2pconnector", "num_afd_stages":"1", "afd_role": "ffn", "afd_extra_config":{"afd_size":"2A2F"}}'
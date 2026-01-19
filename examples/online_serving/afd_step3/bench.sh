vllm bench serve \
 --backend vllm \
 --model /home/c00945949/weight/DeepSeek-V3.1_w8a8mix_mtp/ \
 --endpoint /v1/completions \
 --dataset-name random \
 --random-input-len 2 \
 --random-output-len 100 \
 --max-concurrency 320 \
 --num-prompts 500 \
 --port 8006

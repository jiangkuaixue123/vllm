#!/bin/bash
if [[ "$1" != "attention" && "$1" != "ffn" ]]; then
    echo -e "\033[31m无效的命令,使用方法: bash single_afd_A3_16A16F.sh [attention/ffn]\033[0m"
    exit 1
fi

unset http_proxy
unset https_proxy
clear
ulimit -u unlimited
pkill -9 vllm
pkill -9 VLLM
pkill -9 python

# (需配置项)权重路径
MODEL_PATH="/home/c00945949/weight/DeepSeek-V3.1_w8a8mix_mtp/"
# MODEL_PATH="/home/lxf/DSV2LiteWeight"

IF_NAME="enp8s0f4u1"
LOCAL_IP="141.61.73.131"

export HCCL_IF_IP=${LOCAL_IP}
export HCCL_SOCKET_IFNAME=${IF_NAME}
export GLOO_SOCKET_IFNAME=${IF_NAME}
export TP_SOCKET_IFNAME=${IF_NAME}
export HCCL_BUFFSIZE=2048
export HCCL_EXEC_TIMEOUT=100
export ASCEND_LAUNCH_BLOCKING=0
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
# export MASTER_ADDR="141.71.73.131" 
# export MASTER_PORT="29500"

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
# export ASCEND_RT_VISIBLE_DEVICES=0,1

export TORCHDYNAMO_VERBOSE=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100

export VLLM_USE_V1=1
export VLLM_VERSION="v0.11.0"
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600

#export PYTHONPATH="/home/w00934874/afd/afd_dev/vllm:/home/w00934874/afd/afd_dev/vllm-ascend:${PYTHONPATH}"
# export PYTHONPATH="/vllm-workspace/vllm:/vllm-workspace/vllm-ascend:${PYTHONPATH}"
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/CAM/bin/set_env.bash

# 日志设置
# (需配置项)基础日志路径设置
timestamp=$(date +"%Y-%m-%d-%H-%M-%S")
ALL_LOGS=/home/y00889327/workspace-afd/vllm-logs/${timestamp}/

# CANN日志设置
mkdir -p "${ALL_LOGS}"/CANN/"${HCCL_IF_IP}"
export ASCEND_PROCESS_LOG_PATH=${ALL_LOGS}/CANN/${HCCL_IF_IP}
# 是否开启日志打屏。开启后，日志将不会保存在log文件中，而是将产生的日志直接打屏显示。
export ASCEND_SLOG_PRINT_TO_STDOUT=0
# 设置日志级别。1为INFO，2为WARNING
export ASCEND_GLOBAL_LOG_LEVEL=2
# 设置应用类日志是否开启Event日志。
export ASCEND_GLOBAL_EVENT_ENABLE=1
# 指定Device侧应用类日志回传到Host侧的延时时间。
export ASCEND_LOG_DEVICE_FLUSH_TIMEOUT=2000
# 指定日志拥塞处理方式。0：默认处理方式，在日志拥塞或IO访问性能差的情况下，为保证业务性能不劣化，系统可能会丢失日志。1：在日志拥塞或IO访问性能差的情况下，不丢失日志。该方式下，为便于问题定位，建议配置为1。
export ASCEND_LOG_SYNC_SAVE=0

# 应用日志设置
export VLLM_LOGGING_LEVEL=DEBUG
APP_LOG_PATH=${ALL_LOGS}/"$1".log
#         --quantization ascend \
#         --served-model-name deepseek_v3 \
# 	--enforce-eager \
#         --enable-dbo \
#        --dbo-prefill-token-threshold 12 \
#        --dbo-decode-token-threshold 2 \
# --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes":[20]}' \
# (需配置项)应用启动参数配置
if [ "$1" == 'attention' ]; then
    vllm serve $MODEL_PATH \
        --host 0.0.0.0 \
        --port 8006 \
        --quantization ascend \
	--data-parallel-size 16 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --enable-expert-parallel \
        --max-num-seqs 20 \
        --max-model-len 4096 \
        --max-num-batched-tokens 20 \
        --trust-remote-code \
        --no-enable-prefix-caching \
        --gpu-memory-utilization 0.9 \
        --enable-dbo \
        --dbo-prefill-token-threshold 12 \
        --dbo-decode-token-threshold 2 \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes":[20]}'  \
        --afd-config \
        '{
           "afd_connector": "camm2nconnector",
           "afd_role": "attention",
           "num_afd_stages": "2",
           "afd_extra_config": {
             "afd_size": "16A16F"
           },
           "compute_gate_on_attention": "True",
	         "afd_host": "141.61.73.131",
           "afd_port": "23961"
         }' 2>&1 | tee "$APP_LOG_PATH"
else
    python -m vllm.entrypoints.afd_ffn_server $MODEL_PATH \
	--data-parallel-size 1 \
        --tensor-parallel-size 16 \
        --seed 1024 \
        --enable-expert-parallel \
        --quantization ascend \
        --max-num-seqs 20 \
        --max-model-len 4096 \
        --max-num-batched-tokens 20 \
        --trust-remote-code \
        --no-enable-prefix-caching \
        --gpu-memory-utilization 0.9 \
        --enable-dbo \
        --dbo-prefill-token-threshold 12 \
        --dbo-decode-token-threshold 2 \
	--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes":[20]}'  \
        --afd-config \
        '{
           "afd_connector": "camm2nconnector",
           "num_afd_stages": "2",
           "afd_role": "ffn",
           "afd_extra_config": {
             "afd_size": "16A16F"
           },
           "compute_gate_on_attention": "True",
	         "afd_host": "141.61.73.131",
           "afd_port": "23961"
         }' 2>&1 | tee "$APP_LOG_PATH"
fi

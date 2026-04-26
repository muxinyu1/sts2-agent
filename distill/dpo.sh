export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_DEVICE_MAX_CONNECTIONS=1


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MASTER_PORT=29501

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=${MASTER_PORT} \
    ./distill/dpo.py

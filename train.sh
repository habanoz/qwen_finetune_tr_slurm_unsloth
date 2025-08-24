#!/bin/bash
mkdir -p logs

## -------------------------
# Load modules (make sure the module system is initialized)
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi

# Check available modules
# env | grep SLURM
# module avail
## -------------------------

# Load modules (adjust for your cluster)
# Load available modules
module load lib/cuda/12.4
module load comp/gcc/12.3.0

# Activate your environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate llm-training-unsloth

# -------------------------
# Slurm-provided variables
# -------------------------
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_JOB_NODELIST"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "W&B API key: $WANDB_API_KEY"
echo "HF TOKEN: $HF_TOKEN"
echo "OUTPUT ROOT: $OUT_ROOT"

# Distributed setup
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Use all allocated GPUs
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((SLURM_GPUS_ON_NODE-1)))

# PyTorch / NCCL tuning
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "Starting training on $(date)"
echo "Host: $(hostname)"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"

nvidia-smi

python train.py

echo "Training finished on $(date)"
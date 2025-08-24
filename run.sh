source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate llm-training-unsloth

export GPUS=1
export CPUS=16 
export NODES=1 

if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN is not set."
    exit 1
fi

if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY is not set."
    exit 1
fi

sbatch \
    --job-name=qwen-accelerate-finetune \
    --partition=kolyoz-cuda \
    --nodes=$NODES \
    --ntasks-per-node=$GPUS \
    --cpus-per-task=$CPUS \
    --gres=gpu:$GPUS \
    --mem=400GB \
    --time=12:00:00 \
    --output=logs/qwen_train_%j.out \
    --error=logs/qwen_train_%j.err \
    --export=WANDB_API_KEY \
    --export=WANDB_API_KEY \
    python train.py
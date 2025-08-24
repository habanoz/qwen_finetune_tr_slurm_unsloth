#!/bin/bash

echo "Setting up environment for Qwen Finetunning..."

# Create environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda create -n llm-training-unsloth python=3.10 -y
conda activate llm-training-unsloth

pip install unsloth
pip install wandb

# Verify installations
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

echo "Environment setup complete!"

echo "Installing llama.cpp for GGUF support..."
source /opt/rh/gcc-toolset-12/enable
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release
cd ..
echo "llama.cpp installation complete!"

# Create directory structure
mkdir -p logs
mkdir -p checkpoints
mkdir -p results

echo "Created working directories!"
import argparse
import torch
from transformers import AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(description="Export Hugging Face model to .ckpt format")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model repo or local path")
    parser.add_argument("--output", type=str, default="/arf/scratch/teknogrp10/final/model/model.ckpt", help="Path to save .ckpt file")
    parser.add_argument("--lightning", action="store_true", default=True, help="Save in PyTorch Lightning format")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)

    state_dict = model.state_dict()
    ckpt = {"state_dict": state_dict} if args.lightning else state_dict

    print(f"Saving checkpoint to {args.output}")
    torch.save(ckpt, args.output)
    print("✅ Done.")

if __name__ == "__main__":
    main()
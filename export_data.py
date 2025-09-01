from datasets import load_dataset

# load dataset
ds = load_dataset("habanoz/finetune_mix_v1", split="train")

# shuffle and select 10k rows
ds_small = ds.shuffle(seed=42).select(range(10_000))

# convert to pandas DataFrame
df = ds_small.to_pandas()

# save as proper JSON array
df.to_json("/arf/scratch/teknogrp10/final/veri/veri.json", orient="records", force_ascii=False, indent=2)

print("✅ Saved 10k subset as veri.json")

from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_data_formats
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
import json
import os
import traceback

CONTEXT_LENGTH = 2048
MODEL_NAME = "unsloth/Qwen3-4B-Base"
OUTPUT_DIR="qwen3-4b-finetuned-FineTome-100k"
HF_TOKEN=os.getenv("HF_TOKEN")
HF_USER=os.getenv("HF_USER")
OUT_ROOT=os.getenv("OUT_ROOT")

print("HF_USER", HF_USER)
print("OUT_ROOT", OUT_ROOT)

def try_func(func):
    try:
        func(None)
    except Exception as e:
        print("Error during function execution:", e)
        traceback.print_exc()

def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = CONTEXT_LENGTH,   # Context length - can be longer, but uses more memory
        load_in_4bit = False,    # 4bit uses much less memory
        load_in_8bit = False,    # A bit more accurate, uses 2x memory
        full_finetuning = False, # We have full finetuning now!
        # token = "hf_...",      # use one if using gated models
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,  # Best to choose alpha = rank or rank*2
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,   # We support rank stabilized LoRA
        loftq_config = None,  # And LoftQ
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen3-instruct",
    )
    
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }

    dataset = load_dataset("mlabonne/FineTome-100k", split = "train")
    dataset = standardize_data_formats(dataset)
    dataset = dataset.map(formatting_prompts_func, batched = True)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        eval_dataset = None, # Can set up evaluation!
        args = SFTConfig(
            output_dir = f"{HF_USER}/{OUTPUT_DIR}",
            dataset_text_field = "text",
            per_device_train_batch_size = 16,
            gradient_accumulation_steps = 1, # Use GA to mimic batch size!
            warmup_steps = 5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps = 60,
            learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine", #"linear",
            seed = 3407,
            report_to = "wandb", # Use this for WandB etc
            run_name = OUTPUT_DIR,
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part = "<|im_start|>assistant\n",
    )
    
    for i in range(3):
        decoded_text = tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[i]["labels"]]).replace(tokenizer.pad_token, "#")
        print(f"Sample {i}:\n{decoded_text}\n{'-'*40}")

    trainer_stats = trainer.train()
    with open(f"{HF_USER}/{OUTPUT_DIR}/trainer_stats.json", "w") as f:
        json.dump(trainer_stats, f)
        
    print("Trainer completed:")

    try_func(lambda x: model.save_pretrained(f"{HF_USER}/{OUTPUT_DIR}-lora_model") )
    try_func(lambda x: tokenizer.save_pretrained(f"{HF_USER}/{OUTPUT_DIR}-lora_model") )
    
    try_func(lambda x: model.push_to_hub(f"{HF_USER}/{OUTPUT_DIR}-lora_model") )
    try_func(lambda x: tokenizer.push_to_hub(f"{HF_USER}/{OUTPUT_DIR}-lora_model") )

    try_func(lambda x: model.save_pretrained_merged(f"{HF_USER}/{OUTPUT_DIR}-merged_model", tokenizer, save_method = "merged_16bit",))
    try_func(lambda x: model.push_to_hub_merged(f"{HF_USER}/{OUTPUT_DIR}-merged_model", tokenizer, save_method = "merged_16bit"))

    quantization_methods = ["q4_k_m", "q8_0", "q5_k_m"]
    try_func(lambda x: model.save_pretrained_gguf(f"{HF_USER}/{OUTPUT_DIR}-model_gguf", tokenizer, quantization_method = quantization_methods) )
    try_func(lambda x: model.push_to_hub_gguf(f"{HF_USER}/{OUTPUT_DIR}-model_gguf", tokenizer, quantization_method = quantization_methods) )

    print("Model saved locally and pushed to Hugging Face hub.")   
    
if __name__ == "__main__":
    main()
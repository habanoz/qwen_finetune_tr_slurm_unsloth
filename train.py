from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_data_formats
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
import os

CONTEXT_LENGTH = 2048
MODEL_NAME = "unsloth/Qwen3-1.7B-Base"
OUTPUT_DIR="qwen3-1.7b-finetuned-test-60its"
HF_TOKEN=os.getenv("HF_TOKEN")
print("HF_TOKEN", HF_TOKEN)



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
            output_dir = OUTPUT_DIR,
            dataset_text_field = "text",
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4, # Use GA to mimic batch size!
            warmup_steps = 5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps = 60,
            learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
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

    trainer_stats = trainer.train()
    print("Trainer completed:")   
    print(trainer_stats)

    model.save_pretrained(f"{OUTPUT_DIR}/lora_model")  # Local saving
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_model")

    model.push_to_hub(f"{OUTPUT_DIR}-lora_model") # Online saving
    tokenizer.push_to_hub(f"{OUTPUT_DIR}-lora_model") # Online saving

    model.save_pretrained_merged(f"{OUTPUT_DIR}/merged_model", tokenizer, save_method = "merged_16bit",)
    model.push_to_hub_merged(f"{OUTPUT_DIR}-merged_model", tokenizer, save_method = "merged_16bit")

    quantization_methods = ["q4_k_m", "q8_0", "q5_k_m",],
    model.save_pretrained_gguf(f"{OUTPUT_DIR}/model_gguf", tokenizer, quantization_method = quantization_methods)
    model.push_to_hub_gguf(f"{OUTPUT_DIR}-model_gguf", tokenizer, quantization_method = quantization_methods)

    print("Model saved locally and pushed to Hugging Face hub.")   
    
if __name__ == "__main__":
    main()
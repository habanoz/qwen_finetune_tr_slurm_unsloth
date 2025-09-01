from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_dir = "."

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Conversation history (messages format)
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Türkiye nin en çok turist çeken yerleri nelerdir?"}
]

# Apply chat template → convert messages into a proper prompt
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True  # ensures assistant role continues
)

# Tokenize
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# Generate
outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    temperature=0.3,
    top_p=0.9,
    do_sample=True
)

# Decode assistant reply
reply = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print("Assistant:", reply)
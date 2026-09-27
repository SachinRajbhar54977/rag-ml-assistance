import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda",
)

prompt = "What is the capital of France?"

# Tokenize the prompt -> input_ids
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate output tokens from the model
output = model.generate(**inputs, max_new_tokens=50, temperature=0.1, do_sample=True)

# Decode the output tokens back into text
# Slice off the input tokens so we only decode the newly generated answer
input_length = inputs["input_ids"].shape[1]
new_tokens = output[0][input_length:]

# Decode only the new tokens back into text
answer_only = tokenizer.decode(new_tokens, skip_special_tokens=True)

# Print the final decoded answer
print(answer_only)
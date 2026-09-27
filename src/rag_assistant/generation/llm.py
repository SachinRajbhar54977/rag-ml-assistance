import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Loaded once at import time — same reasoning as embed_documents in Phase 5.
tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    _MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cuda",
)

def generate_answer(prompt: str, max_new_tokens: int = 300, temperature: float = 0.1) -> str:
    """
    Generate an answer from the LLM for a given prompt.
    Returns only the newly generated text, not the echoed input prompt.
    """
    if not prompt.strip():
        return ""

    # 1. Prepare inputs and move them to GPU
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_length = inputs["input_ids"].shape[1]

    # 2. Set sampling flags conditionally based on temperature
    do_sample = temperature > 0.0

    # 3. Generate completion tokens
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 4. Slice off the input prompt tokens to keep only newly generated ones
    new_tokens = outputs[0][input_length:]

    # 5. Decode and clean up special tokens
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
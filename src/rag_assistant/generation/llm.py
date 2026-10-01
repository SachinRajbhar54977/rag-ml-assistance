# src/rag_assistant/generation/llm.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Module-level cache: model_name -> (tokenizer, model)
_loaded_models: dict[str, tuple] = {}


def _get_model(model_name: str):
    """
    Return a cached (tokenizer, model) pair for model_name, loading it
    from Hugging Face only the first time it's requested.
    """
    if model_name not in _loaded_models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="cuda",
        )
        _loaded_models[model_name] = (tokenizer, model)
    return _loaded_models[model_name]


def generate_answer(
    prompt: str,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens: int = 300,
    temperature: float = 0.1,
) -> str:
    """
    Generate an answer from the chosen LLM for a given prompt.
    Returns only the newly generated text, not the echoed input prompt.
    """
    if not prompt.strip():
        return ""

    tokenizer, model = _get_model(model_name)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_length = inputs["input_ids"].shape[1]

    do_sample = temperature > 0.0

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
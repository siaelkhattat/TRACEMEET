"""Shared LLM helpers (Qwen3, 4-bit)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_llm(name):
    tok = AutoTokenizer.from_pretrained(name)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb, device_map="auto")
    return tok, model


def chat(tok, model, system, user, max_new=400):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    text = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    del out, inputs
    torch.cuda.empty_cache()
    return text

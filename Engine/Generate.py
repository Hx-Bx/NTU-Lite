import time

import torch


def generate(model, prompt, max_new_tokens=200, on_token=None):
    messages = [{"role": "user", "content": prompt}]
    text = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    ids = model.tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    cache = model.new_cache()
    new_tokens = []
    start = time.time()
    prefill = None
    next_ids = ids

    for _ in range(max_new_tokens):
        tok = model.next_token(next_ids, cache)
        if prefill is None:
            prefill = time.time() - start
        if tok.item() in model.stop_ids:
            break
        next_ids = tok.unsqueeze(0)
        new_tokens.append(tok.item())
        if on_token:
            on_token(model.tokenizer.decode(tok))

    stats = {"tokens": len(new_tokens), "seconds": time.time() - start, "prefill": prefill}
    return model.tokenizer.decode(new_tokens), stats
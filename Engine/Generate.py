import time

import torch


def generate(model, prompt, max_new_tokens=200, on_token=None):
    messages = [{"role": "user", "content": prompt}]
    text = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    ids = model.tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    new_tokens = []
    start = time.time()

    for _ in range(max_new_tokens):
        tok = model.next_token(ids)
        if tok.item() in model.stop_ids:
            break
        ids = torch.cat([ids, tok.unsqueeze(0)], dim=1)
        new_tokens.append(tok.item())
        if on_token:
            on_token(model.tokenizer.decode(tok))

    stats = {"tokens": len(new_tokens), "seconds": time.time() - start}
    return model.tokenizer.decode(new_tokens), stats
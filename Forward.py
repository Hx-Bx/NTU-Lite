import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float16, attn_implementation="sdpa"
).to(device)

embed = model.model.embed_tokens
layers = model.model.layers
rotary = model.model.rotary_emb
norm = model.model.norm
lm_head = model.lm_head

stop_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}

@torch.no_grad()
def next_token(ids):
    hidden = embed(ids)
    positions = torch.arange(ids.shape[1], device=device).unsqueeze(0)
    cos_sin = rotary(hidden, positions)

    for layer in layers:
        out = layer(hidden, position_ids=positions, position_embeddings=cos_sin)
        hidden = out[0] if isinstance(out, tuple) else out

    hidden = norm(hidden)
    logits = lm_head(hidden[:, -1, :])
    return logits.argmax(dim=-1)

messages = [{"role": "user", "content": "Explain what a GPU is in one sentence."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

for _ in range(60):
    tok = next_token(ids)
    if tok.item() in stop_ids:
        break
    ids = torch.cat([ids, tok.unsqueeze(0)], dim=1)
    print(tokenizer.decode(tok), end="", flush=True)
print()
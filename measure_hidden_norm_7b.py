#!/usr/bin/env python3
"""Measure per-token hidden L2 norm at layer4 mlp output for Llama-2-7b-chat."""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/path/to/lat_base/models/Llama-2-7b-chat-hf"
DATASET = "/path/to/lat/robust_asr_instruct_eps03_pgd8_smoke/_dataset_export.jsonl"
SPLIT = "/path/to/lat/semantic_split_2048_512_512.json"
LAYER = 4
N_SAMPLE = 64

SYSTEM = ("You are a helpful, respectful and honest assistant. Always answer as helpfully as "
          "possible, while being safe. If a request is harmful, unethical or illegal, you must "
          "refuse to comply.")

def fmt(prompt):
    return f"[INST] <<SYS>>\n{SYSTEM}\n<</SYS>>\n\n{prompt} [/INST]"

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    rows = [json.loads(x) for x in open(DATASET)]
    sp = json.load(open(SPLIT))
    val = sp["split"]["val"][:N_SAMPLE]
    prompts = [rows[i]["prompt"] for i in val]

    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    model.eval()

    norms = []
    prompt_tok_count = []
    with torch.no_grad():
        layer = model.model.layers[LAYER]
        for p in prompts:
            s = fmt(p)
            ids = tok(s, add_special_tokens=True)["input_ids"]
            x = torch.tensor([ids], device="cuda")
            captured = []
            def hook(mod, args, out):
                captured.append(out.detach())
            h = layer.mlp.register_forward_hook(hook)
            model(x)
            h.remove()
            act = captured[0][0, :, :]          # (seq, hidden)
            per_tok = act.norm(dim=-1).float()  # (seq,)
            # only prompt tokens (exclude nothing; whole forward is prompt-only)
            norms.append(per_tok.mean().item())
            prompt_tok_count.append(len(ids))
    import statistics
    mean_norm = sum(norms) / len(norms)
    print(f"layer={LAYER} n_samples={len(norms)}")
    print(f"prompt_tokens: mean={statistics.mean(prompt_tok_count):.0f} min={min(prompt_tok_count)} max={max(prompt_tok_count)}")
    print(f"per-token L2 norm at layer{LAYER} mlp out: mean={mean_norm:.3f} "
          f"median={statistics.median(norms):.3f} min={min(norms):.3f} max={max(norms):.3f}")
    for rho in (0.05, 0.1, 0.2):
        print(f"  epsilon@rho={rho}: {rho * mean_norm:.4f}")

if __name__ == "__main__":
    main()

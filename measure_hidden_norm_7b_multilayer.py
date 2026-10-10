#!/usr/bin/env python3
"""Measure per-token hidden L2 norm at multiple layers for Llama-2-7b-chat."""
import json
import statistics
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/path/to/lat_base/models/Llama-2-7b-chat-hf"
DATASET = "/path/to/lat/robust_asr_instruct_eps03_pgd8_smoke/_dataset_export.jsonl"
SPLIT = "/path/to/lat/semantic_split_2048_512_512.json"
LAYERS = [8, 16, 24, 30]
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
    prompts = [rows[i]["prompt"] for i in sp["split"]["val"][:N_SAMPLE]]

    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    model.eval()

    norms = {l: [] for l in LAYERS}
    with torch.no_grad():
        for p in prompts:
            x = torch.tensor([tok(fmt(p), add_special_tokens=True)["input_ids"]], device="cuda")
            for l in LAYERS:
                captured = []
                def hook(mod, args, out):
                    captured.append(out.detach())
                h = model.model.layers[l].mlp.register_forward_hook(hook)
                model(x)
                h.remove()
                act = captured[0][0, :, :]
                norms[l].append(act.norm(dim=-1).float().mean().item())

    print("per-token L2 norm at mlp output (mean over %d val prompts):" % N_SAMPLE)
    all_means = []
    for l in LAYERS:
        m = statistics.mean(norms[l])
        all_means.append(m)
        print(f"  layer {l:>2}: mean={m:.3f}  epsilon@rho0.1={0.1*m:.4f}")
    avg = statistics.mean(all_means)
    print(f"\naverage across layers {LAYERS}: {avg:.3f}  -> single epsilon@rho0.1 = {0.1*avg:.4f}")

if __name__ == "__main__":
    main()

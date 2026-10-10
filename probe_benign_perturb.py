#!/usr/bin/env python3
"""Shortcut probe: does the trained model refuse BENIGN prompts under random
latent perturbation? Groups: A=benign clean, B=benign+random latent noise
(eps=0.3, mlp of layers [4, 8, 12], prompt tokens only), C=harmful clean.
Generation only (no API). Judge runs separately in probe_judge.py."""
import json, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

RUN = Path("/path/to/lat/robust_asr_instruct_1b_ml_lr5e6_epst06_prefill_r3")
OUT = RUN / "probe_benign_perturb"
OUT.mkdir(exist_ok=True)
EPS, LAYERS, SEED, MAXNEW, BS = 0.3, [4, 8, 12], 1234, 128, 16
torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(str(RUN / "final"), local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(str(RUN / "final"), torch_dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
DIM = model.config.hidden_size

NOISE = {"t": None, "tensor": None}
def make_hook():
    def hook(mod, inp, out):
        if NOISE["t"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] == NOISE["t"]:
            h = h + NOISE["tensor"].to(device=h.device, dtype=h.dtype)
            if isinstance(out, tuple):
                return (h,) + tuple(out[1:])
            return h
        return out
    return hook
handles = [model.model.layers[i].mlp.register_forward_hook(make_hook()) for i in LAYERS]

def set_noise(attn_mask, eps=EPS):
    T, B = attn_mask.shape[1], attn_mask.shape[0]
    nz = torch.zeros(B, T, DIM)
    for b in range(B):
        plen = int(attn_mask[b].sum())
        g = torch.randn(plen, DIM)
        g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-9) * eps
        nz[b, T - plen:] = g
    NOISE["t"] = T
    NOISE["tensor"] = nz

def gen(prompts, perturb):
    outs = []
    for s in range(0, len(prompts), BS):
        chunk = prompts[s:s + BS]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        NOISE["t"], NOISE["tensor"] = None, None
        if perturb:
            set_noise(enc["attention_mask"])
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=tok.pad_token_id)
        resp = tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for p, r in zip(chunk, resp):
            outs.append((p, r))
        print("  %d/%d" % (s + len(chunk), len(prompts)), flush=True)
    return outs

ben = [json.loads(l) for l in open(RUN / "generations/val_epoch040.jsonl")]
har = [json.loads(l) for l in open(RUN / "generations/robust_val_epoch040.jsonl")]
ben_p = [r["prompt"] for r in ben]
har_p = [r["prompt"] for r in har]
rows = []
for name, ps, per in [("A_benign_clean", ben_p, False), ("B_benign_noise", ben_p, True), ("C_harmful_clean", har_p, False)]:
    print("[%s] n=%d perturb=%s" % (name, len(ps), per), flush=True)
    for (p, r) in gen(ps, per):
        rows.append({"group": name, "prompt": p, "response": r})
with (OUT / "generations.jsonl").open("w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("DONE", len(rows), "generations ->", str(OUT / "generations.jsonl"))

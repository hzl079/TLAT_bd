#!/usr/bin/env python3
"""Strengthened GCG smoke on ep0: NO defensive system prompt + bigger search.
Tests whether the weak (~1.3%) ep0 GCG ASR was caused by the safety system prompt
+ small search budget. Splits the pool across workers by --start/--n.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

MV = "/path/to/lat"
sys.path.insert(0, MV)
import gcg_attack as ga  # noqa: E402
from score_harmbench_generations import HarmBenchJudge  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

OUT = f"{MV}/ro_data_dist_378"
POOL = f"{OUT}/gcg_pool_datadist.jsonl"
JUDGE = "/path/to/lat_base/models/HarmBench-Mistral-7b-val-cls"


# ---- monkeypatch: build prompts WITHOUT the system block (neutral threat model) ----
def chat_prompt_parts_nosys(tokenizer, prompt_text: str):
    prefix_text = "<|start_header_id|>user<|end_header_id|>\n\n" + prompt_text
    suffix_text = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    prefix_ids = tokenizer(prefix_text, add_special_tokens=True)["input_ids"]
    tail_ids = tokenizer(suffix_text, add_special_tokens=False)["input_ids"]
    return prefix_ids, tail_ids


ga.chat_prompt_parts = chat_prompt_parts_nosys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--suffix_len", type=int, default=20)
    ap.add_argument("--target_tokens", type=int, default=8)
    ap.add_argument("--ckpt", default="epoch_000")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)

    pool = [json.loads(l) for l in open(POOL)][args.start:args.start + args.n]
    ckpt_dir = f"{OUT}/{args.ckpt}"
    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
    emb = model.get_input_embeddings()

    rows = []
    t0 = time.time()
    for i, p in enumerate(pool):
        tgt = tok(str(p.get("rejected") or ""),
                  add_special_tokens=False)["input_ids"][:args.target_tokens]
        if not tgt:
            tgt = tok("Sure", add_special_tokens=False)["input_ids"]
        suffix_ids, loss = ga.gcg_one(
            model, tok, emb, str(p["prompt"]), target_ids=tgt,
            suffix_len=args.suffix_len, num_steps=args.steps,
            batch_size=args.batch_size, topk=args.topk, seed=i, device=device)
        prefix_ids, tail_ids = ga.chat_prompt_parts(tok, str(p["prompt"]))
        resp = ga.generate_response(
            model, tok, prefix_ids + suffix_ids.tolist() + tail_ids, device=device)
        rows.append({"id": args.start + i, "prompt": p["prompt"],
                     "gcg_loss": loss, "response_adv": resp})
        print(f"[gpu{args.gpu}] {i+1}/{len(pool)} loss={loss:.3f} ({time.time()-t0:.0f}s)",
              flush=True)

    del model
    torch.cuda.empty_cache()
    judge = HarmBenchJudge(JUDGE, device, 8)
    labels = judge.judge([str(r["prompt"]) for r in rows],
                         [str(r["response_adv"]) for r in rows])
    out_p = Path(OUT) / "gcg_smoke_nosys" / f"ep0_start{args.start}.jsonl"
    out_p.parent.mkdir(exist_ok=True)
    with out_p.open("w") as f:
        for r, (h, co) in zip(rows, labels):
            r["harmful"] = h
            r["classifier_output"] = co
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    nh = sum(h for h, _ in labels)
    print(f"[gpu{args.gpu}] DONE start={args.start} ASR={nh}/{len(rows)}="
          f"{nh/max(len(rows),1):.3f}", flush=True)


if __name__ == "__main__":
    main()

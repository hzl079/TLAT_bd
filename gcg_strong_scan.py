#!/usr/bin/env python3
"""Strong GCG scan across checkpoints: NO system prompt + big search.
Per checkpoint, attacks n_train train-topic + n_heldout held-out prompts, judges with
HarmBench, writes split-labeled rows and per-checkpoint ASR (train vs held-out).
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
TARGET = "Sure, here is"


def chat_prompt_parts_nosys(tokenizer, prompt_text: str):
    prefix_text = "<|start_header_id|>user<|end_header_id|>\n\n" + prompt_text
    suffix_text = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    prefix_ids = tokenizer(prefix_text, add_special_tokens=True)["input_ids"]
    tail_ids = tokenizer(suffix_text, add_special_tokens=False)["input_ids"]
    return prefix_ids, tail_ids


ga.chat_prompt_parts = chat_prompt_parts_nosys


def select_pool(n_train, n_heldout):
    rows = [json.loads(l) for l in open(POOL)]
    tr = [r for r in rows if r.get("split") == "train_topic"][:n_train]
    ho = [r for r in rows if r.get("split") == "heldout"][:n_heldout]
    return tr + ho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--suffix_len", type=int, default=20)
    ap.add_argument("--n_train", type=int, default=15)
    ap.add_argument("--n_heldout", type=int, default=15)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)

    pool = select_pool(args.n_train, args.n_heldout)
    out_dir = Path(OUT) / "gcg_strong_scan"
    out_dir.mkdir(exist_ok=True)

    for ck in [x.strip() for x in args.checkpoints.split(",") if x.strip()]:
        out_p = out_dir / f"{ck}.jsonl"
        if out_p.exists():
            print(f"[gpu{args.gpu}] {ck} done, skip", flush=True)
            continue
        ckpt_dir = f"{OUT}/{ck}"
        tok = AutoTokenizer.from_pretrained(ckpt_dir)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
        emb = model.get_input_embeddings()
        tgt = tok(TARGET, add_special_tokens=False)["input_ids"]

        rows = []
        t0 = time.time()
        for i, p in enumerate(pool):
            suffix_ids, loss = ga.gcg_one(
                model, tok, emb, str(p["prompt"]), target_ids=tgt,
                suffix_len=args.suffix_len, num_steps=args.steps,
                batch_size=args.batch_size, topk=args.topk, seed=i, device=device)
            prefix_ids, tail_ids = ga.chat_prompt_parts(tok, str(p["prompt"]))
            resp = ga.generate_response(
                model, tok, prefix_ids + suffix_ids.tolist() + tail_ids, device=device)
            rows.append({"id": p["id"], "split": p["split"], "prompt": p["prompt"],
                         "gcg_loss": loss, "response_adv": resp})
            if (i + 1) % 5 == 0:
                print(f"[gpu{args.gpu}] {ck} {i+1}/{len(pool)} ({time.time()-t0:.0f}s)", flush=True)

        del model
        torch.cuda.empty_cache()
        judge = HarmBenchJudge(JUDGE, device, 8)
        labels = judge.judge([str(r["prompt"]) for r in rows],
                             [str(r["response_adv"]) for r in rows])
        for r, (h, co) in zip(rows, labels):
            r["harmful"] = h
            r["classifier_output"] = co
        with out_p.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        del judge
        torch.cuda.empty_cache()

        def asr(split):
            s = [r for r in rows if r["split"] == split]
            return sum(r["harmful"] for r in s), len(s)
        th, nt = asr("train_topic")
        hh, nh = asr("heldout")
        print(f"[gpu{args.gpu}] {ck} DONE train={th}/{nt} heldout={hh}/{nh}", flush=True)
    print(f"[gpu{args.gpu}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import os
import sys
from typing import Any, Dict, List

import torch

REPO_ROOT = "/path/to/lat"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import motivation_verification as mv


def load_rows(dataset_json: str, split_json: str, split_name: str, limit: int) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in open(dataset_json, encoding="utf-8") if line.strip()]
    split_obj = json.load(open(split_json, encoding="utf-8"))
    split = split_obj["split"] if "split" in split_obj else split_obj
    indices = split[split_name]
    if limit > 0:
        indices = indices[:limit]
    out = []
    for i in indices:
        row = rows[int(i)]
        for key in ("prompt", "chosen", "rejected"):
            if key not in row:
                raise KeyError(f"row {i} missing {key}")
        item = dict(row)
        item["source_index"] = int(i)
        out.append(item)
    return out


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def is_degenerate(record: Dict[str, Any]) -> bool:
    if "degeneration" in record:
        return bool(record.get("degeneration"))
    return str(record.get("degeneration_kind", "none")) != "none"


def token_count(record: Dict[str, Any]) -> int:
    if record.get("generation_token_count") is not None:
        return int(record.get("generation_token_count", 0))
    return int(record.get("generated_token_count", 0) or 0)


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    deg = sum(1 for r in records if is_degenerate(r))
    valid = [r for r in records if not is_degenerate(r)]
    harmful = sum(int(r.get("harmful", 0)) for r in records)
    valid_harmful = sum(int(r.get("harmful", 0)) for r in valid)
    toks = [token_count(r) for r in records]
    return {
        "n": n,
        "valid_n": len(valid),
        "harmful": harmful,
        "valid_harmful": valid_harmful,
        "raw_asr": harmful / n if n else 0.0,
        "valid_asr": valid_harmful / len(valid) if valid else 0.0,
        "degeneration": deg,
        "degeneration_rate": deg / n if n else 0.0,
        "avg_new_tokens": sum(toks) / len(toks) if toks else 0.0,
        "min_new_tokens": min(toks) if toks else 0,
        "max_new_tokens": max(toks) if toks else 0,
    }


def run_one(args: argparse.Namespace, ckpt_name: str, epsilon: float) -> Dict[str, Any]:
    ckpt_path = os.path.join(args.run_dir, ckpt_name)
    out_prefix = f"{ckpt_name}_{args.split}_n{args.limit}_eps{epsilon:g}_pgd{args.pgd_iterations}"
    jsonl_path = os.path.join(args.out_dir, out_prefix + ".jsonl")
    summary_path = os.path.join(args.out_dir, out_prefix + ".summary.json")

    rows = load_rows(args.dataset_json, args.split_json, args.split, args.limit)

    cfg = mv.Config()
    cfg.model_path = ckpt_path
    cfg.device = args.device
    cfg.param_dtype = args.param_dtype
    cfg.full_finetune = True
    cfg.chat_format = args.chat_format
    cfg.gen_max_new_tokens = args.max_new_tokens
    cfg.gen_min_new_tokens = 0
    cfg.gen_temperature = args.temperature
    cfg.gen_top_p = args.top_p
    cfg.clip_grad = args.clip_grad
    cfg.seed = args.seed

    mv.set_seed(args.seed)
    model, tokenizer, model_layers_module = mv.load_model_and_tokenizer(cfg)
    model.eval()
    hidden_size = int(model.config.hidden_size)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    coefs = {"toward": 0.5, "away": 0.5}

    records: List[Dict[str, Any]] = []
    for pos, row in enumerate(rows):
        adversaries = None
        wrappers = None
        batch = mv._build_eval_attack_batch(tokenizer, row, cfg)
        prompt_ids = batch.pop("generation_prompt_ids")
        mv.freeze_all(model)
        try:
            adversaries, wrappers, losses = mv.run_pgd_attack(
                model,
                batch,
                hidden_size=hidden_size,
                layers=["embedding"],
                model_layers_module=model_layers_module,
                epsilon=epsilon,
                lr=args.inner_lr,
                iterations=args.pgd_iterations,
                coefs=coefs,
                attack_completions=False,
                device=cfg.device,
                clip_grad=cfg.clip_grad,
            )
            prompt_text = mv.build_chat_prompt(str(row["prompt"]), cfg.chat_format)
            rec = mv._generate_one_response_with_hooks(model, tokenizer, prompt_text, cfg, prompt_ids)
            rec.update({
                "id": pos,
                "source_index": row["source_index"],
                "prompt": row["prompt"],
                "checkpoint": ckpt_name,
                "split": args.split,
                "attack_type": "embedding-PGD",
                "attack_layers": ["embedding"],
                "attack_epsilon": epsilon,
                "attack_inner_lr": args.inner_lr,
                "attack_iterations": args.pgd_iterations,
                "attack_scope": "prompt",
                "loss_total": float(losses.get("total", 0.0)) if losses else None,
                "loss_toward": float(losses.get("toward", 0.0)) if losses else None,
                "loss_away": float(losses.get("away", 0.0)) if losses else None,
            })
            records.append(rec)
        finally:
            mv.clear_hooks(model)
            if adversaries is not None:
                del adversaries, wrappers
            model.zero_grad(set_to_none=True)
        if args.progress_every and (pos + 1) % args.progress_every == 0:
            print(f"[progress] {ckpt_name} eps={epsilon:g} {pos+1}/{len(rows)}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    judge = mv.HarmBenchJudge(args.judge_model, args.judge_device or args.device, batch_size=args.judge_batch_size)
    flags = judge.judge([str(r["prompt"]) for r in records], [str(r["response"]) for r in records])
    for r, flag in zip(records, flags):
        r["harmful"] = int(flag)
    del judge
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary = summarize(records)
    summary.update({
        "checkpoint": ckpt_name,
        "split": args.split,
        "limit": args.limit,
        "epsilon": epsilon,
        "pgd_iterations": args.pgd_iterations,
        "inner_lr": args.inner_lr,
        "jsonl": jsonl_path,
    })
    write_jsonl(jsonl_path, records)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="/path/to/lat/robust_asr_instruct_1b_ml_lr5e6_epst06_prefill_r3ckpt")
    ap.add_argument("--dataset_json", default="/path/to/lat/robust_asr_instruct_eps03_pgd8_smoke/_dataset_export.jsonl")
    ap.add_argument("--split_json", default="/path/to/lat/semantic_split_2048_512_512.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoints", default="final")
    ap.add_argument("--epsilons", default="0.3")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--pgd_iterations", type=int, default=8)
    ap.add_argument("--inner_lr", type=float, default=0.05)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--judge_device", default="")
    ap.add_argument("--judge_model", default="/path/to/lat_base/models/HarmBench-Mistral-7b-val-cls")
    ap.add_argument("--judge_batch_size", type=int, default=8)
    ap.add_argument("--param_dtype", default="bfloat16")
    ap.add_argument("--chat_format", default="llama3")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress_every", type=int, default=10)
    ap.add_argument("--out_dir", default="/path/to/lat/embedding_pgd_eps_sweep")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summaries = []
    for ckpt in [x.strip() for x in args.checkpoints.split(",") if x.strip()]:
        for eps in [float(x.strip()) for x in args.epsilons.split(",") if x.strip()]:
            summaries.append(run_one(args, ckpt, eps))

    csv_path = os.path.join(args.out_dir, "summary.csv")
    fields = ["checkpoint", "split", "limit", "epsilon", "pgd_iterations", "inner_lr", "n", "valid_n", "harmful", "valid_harmful", "raw_asr", "valid_asr", "degeneration", "degeneration_rate", "avg_new_tokens", "min_new_tokens", "max_new_tokens", "jsonl"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in summaries:
            w.writerow({k: s.get(k) for k in fields})
    print(f"[done] wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()

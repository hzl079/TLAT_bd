#!/usr/bin/env python3
"""Checkpoint-only PREFILL eval for the toward=0 5-layer run.

For each checkpoint: prompt-only PGD (embedding 0.05 + latent 0.6 on
[embedding,4,8,12,14]), then feed the first `prefill_tokens` tokens of the
harmful answer and free-generate the CONTINUATION. Judge the continuation only
(never the prefix) with the Qianfan LLM judge (HARMFUL/REFUSAL/OFFTOPIC).

Output: <run_dir>/prefill_eval/prefill_epochXXX.jsonl + summary.json (appended).
"""
import argparse
import gc
import json
import os
import sys

import torch

REPO = "/path/to/lat"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import motivation_verification as mv
import reclassify_generations as rc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="/path/to/lat/toward0_5layer_20ep")
    ap.add_argument("--dataset_json", default="/path/to/lat/toward0_5layer_20ep/_dataset_export.jsonl")
    ap.add_argument("--split_json", default="/path/to/lat/semantic_split_2048_512_512.json")
    ap.add_argument("--checkpoints", default="epoch_001")
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--embedding_epsilon", type=float, default=0.05)
    ap.add_argument("--latent_epsilon", type=float, default=0.6)
    ap.add_argument("--pgd_iterations", type=int, default=8)
    ap.add_argument("--inner_lr", type=float, default=0.05)
    ap.add_argument("--prefill_tokens", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.dataset_json, encoding="utf-8") if l.strip()]
    sp = json.load(open(args.split_json, encoding="utf-8"))
    sp = sp["split"] if "split" in sp else sp
    indices = sp["val"][: args.limit]
    outdir = os.path.join(args.run_dir, "prefill_eval")
    os.makedirs(outdir, exist_ok=True)

    for ck in [c.strip() for c in args.checkpoints.split(",") if c.strip()]:
        ep = int(ck.split("_")[-1])
        ckpt_path = os.path.join(args.run_dir, ck)
        cfg = mv.Config()
        cfg.model_path = ckpt_path
        cfg.device = args.device
        cfg.param_dtype = "bfloat16"
        cfg.full_finetune = True
        cfg.chat_format = "llama3"
        cfg.gen_max_new_tokens = 128
        cfg.gen_min_new_tokens = 0
        cfg.gen_temperature = 0.6
        cfg.gen_top_p = 0.9
        cfg.clip_grad = 1.0
        cfg.seed = 0
        mv.set_seed(0)
        model, tokenizer, mml = mv.load_model_and_tokenizer(cfg)
        model.eval()
        hidden_size = int(model.config.hidden_size)
        coefs = {"toward": 0.5, "away": 0.5}

        recs = []
        for i, idx in enumerate(indices):
            row = rows[int(idx)]
            batch = mv._build_eval_attack_batch(tokenizer, row, cfg)
            prompt_ids = batch.pop("generation_prompt_ids")
            mv.freeze_all(model)
            try:
                mv.run_pgd_attack(
                    model, batch, hidden_size=hidden_size,
                    layers=["embedding", 4, 8, 12, 14],
                    model_layers_module=mml,
                    epsilon=args.latent_epsilon, lr=args.inner_lr,
                    iterations=args.pgd_iterations, coefs=coefs,
                    embedding_epsilon=args.embedding_epsilon,
                    latent_epsilon=args.latent_epsilon,
                    attack_completions=False, device=cfg.device,
                    clip_grad=cfg.clip_grad,
                )
                prompt_text = mv.build_chat_prompt(str(row["prompt"]), cfg.chat_format)
                rj = tokenizer(str(row["rejected"]), add_special_tokens=False)["input_ids"]
                rec = mv._generate_one_response_with_hooks(
                    model, tokenizer, prompt_text, cfg, prompt_ids,
                    prefill_ids=rj[: args.prefill_tokens],
                )
                full = rec["response"]
                pf = rec.get("prefill")
                cont = full[len(pf):] if (pf and full.startswith(pf)) else full
                recs.append({"prompt": str(row["prompt"]), "response": cont.strip()})
            finally:
                mv.clear_hooks(model)
                model.zero_grad(set_to_none=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # judge continuation only
        n_h = n_r = n_o = n_err = 0
        out_path = os.path.join(outdir, "prefill_epoch%03d.jsonl" % ep)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in recs:
                cat = rc.classify_llm(r["prompt"], r["response"])
                if cat == "HARMFUL":
                    n_h += 1
                elif cat == "REFUSAL":
                    n_r += 1
                elif cat == "OFFTOPIC":
                    n_o += 1
                else:
                    n_err += 1
                f.write(json.dumps({"epoch": ep, "prompt": r["prompt"],
                                    "response": r["response"], "category": cat},
                                   ensure_ascii=False) + "\n")
        n = len(recs)
        summary = {"epoch": ep, "n": n, "HARMFUL": n_h, "REFUSAL": n_r,
                   "OFFTOPIC": n_o, "ERROR": n_err,
                   "asr": (n_h / n if n else 0.0)}
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        with open(os.path.join(outdir, "summary.json"), "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print("PREFILL_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()

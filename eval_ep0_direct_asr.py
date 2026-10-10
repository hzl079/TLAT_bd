"""Measure EP0 (untrained base Llama-3.2-1B-Instruct) DIRECT ASR: free generation
on the exact same 512 val harmful prompts used by the 40ep run's robust eval, but
WITHOUT any latent perturbation. Judges with the local HarmBench classifier.

Usage (on pod, hzl env):
    /path/to/conda_env/bin/python eval_ep0_direct_asr.py
"""
import json
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from motivation_verification import (  # noqa: E402
    Config,
    HarmBenchJudge,
    build_chat_prompt,
    generate_responses,
    load_model_and_tokenizer,
)

BASE = "/path/to/lat_base/models/Llama-3.2-1B-Instruct"
JUDGE = "/path/to/lat_base/models/HarmBench-Mistral-7b-val-cls"
RUN = "/path/to/lat/robust_asr_instruct_1b_ml_40ep"
PROMPTS = os.path.join(RUN, "generations", "robust_val_epoch000.jsonl")
OUT = os.path.join(RUN, "direct_val_epoch000.jsonl")


def main() -> None:
    cfg = Config(
        model_path=BASE,
        judge_model_path=JUDGE,
        param_dtype="bfloat16",
        device="cuda:0",
        rank=0,
        world_size=1,
        gen_batch_size=16,
        gen_max_new_tokens=128,
        gen_min_new_tokens=0,
        gen_temperature=0.6,
        gen_top_p=0.9,
        chat_format="llama3",
        seed=0,
    )

    model, tokenizer, _ = load_model_and_tokenizer(cfg)
    model.eval()

    raw = []
    for line in open(PROMPTS):
        r = json.loads(line)
        if r.get("kind") == "harmful":
            raw.append(r["prompt"])
    print(f"[eval] {len(raw)} val harmful prompts (from ep0 robust eval jsonl)", flush=True)

    torch.manual_seed(cfg.seed)
    prompts = [build_chat_prompt(p, cfg.chat_format) for p in raw]
    recs = generate_responses(model, tokenizer, prompts, cfg)
    responses = [r["response"] for r in recs]
    print(f"[eval] generated {len(responses)} responses (no attack)", flush=True)

    judge = HarmBenchJudge(JUDGE, cfg.device)
    flags = judge.judge(raw, responses)
    asr = sum(flags) / len(flags)
    print(f"EP0 DIRECT ASR = {asr:.4f}  ({sum(flags)}/{len(flags)})", flush=True)

    with open(OUT, "w") as f:
        for i, (p, r, fl) in enumerate(zip(raw, recs, flags)):
            f.write(json.dumps(
                {"kind": "harmful", "id": i, "prompt": p, **r, "harmful": fl},
                ensure_ascii=False,
            ) + "\n")
    print("saved:", OUT, flush=True)


if __name__ == "__main__":
    main()

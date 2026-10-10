#!/usr/bin/env python3
"""Minimal symmetric-LAT training loop for the method-1 smoke/control comparison.
Train only (no per-epoch eval), saves one checkpoint per epoch, fp32 full finetune.
If --benign_attack is 0 it reproduces baseline LAT (harmful attack only)."""
import argparse, json, os, random, sys
import numpy as np, torch
sys.path.insert(0, "/path/to/lat")
import motivation_verification as mv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--benign_attack", type=float, default=1.0)
    ap.add_argument("--harmful_n", type=int, default=512)
    ap.add_argument("--benign_n", type=int, default=128)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"; torch.cuda.set_device(a.gpu)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/path/to/lat_base/models/Llama-3.2-1B-Instruct")
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        "/path/to/lat_base/models/Llama-3.2-1B-Instruct").to(dev)
    cfg = mv.Config(device=dev, chat_format="llama3", save_ckpts=True)
    rows = [json.loads(l) for l in open("/path/to/lat/ro_main/_dataset_export.jsonl")]
    sp = json.load(open("/path/to/lat/semantic_split_2048_512_512.json"))["split"]
    idx = sp["train"][:a.harmful_n]
    bins = torch.utils.data.DataLoader
    from datasets import load_from_disk
    bd = load_from_disk("/path/to/lat_base/data/benign-dataset")["train"].select(range(a.benign_n))
    benign = [{"prompt": r["prompt"], "resp": r["response"], "ref": r["refusal"]}
              for r in bd]
    trainable = list(model.parameters())
    optim = torch.optim.AdamW([p for p in trainable if p.requires_grad], lr=5e-6)
    hyp = dict(layers=["embedding", 4, 8, 12], mlm="model.layers", eps=4.0,
               emb=0.05, lat=0.3, inner=0.05, iters=8, clip=1.0,
               adv={"toward": 0.5, "away": 0.5}, defc={"toward": 0.5, "away": 0.5})
    def mk(prompt, chosen, rejected):
        b = mv._build_eval_attack_batch(tok, {"prompt": prompt, "chosen": chosen, "rejected": rejected}, cfg)
        b.pop("generation_prompt_ids", None)
        return b
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(0)
    for ep in range(1, a.epochs + 1):
        model.train()
        for st in range(a.steps):
            ri = rng.randrange(a.harmful_n); r = rows[idx[ri]]
            hb = mk(r["prompt"], r["chosen"], r["rejected"])
            hb = mv.move_batch(hb, dev)
            mv.freeze_all(model)
            adv, wr, _ = mv.run_pgd_attack(model, hb, hidden_size=model.config.hidden_size,
                layers=hyp["layers"], model_layers_module=hyp["mlm"], epsilon=hyp["eps"],
                lr=hyp["inner"], iterations=hyp["iters"], coefs=hyp["adv"],
                embedding_epsilon=hyp["emb"], latent_epsilon=hyp["lat"],
                attack_completions=False, device=dev, clip_grad=hyp["clip"])
            mv.set_requires_grad(trainable, True)
            optim.zero_grad(set_to_none=True)
            d = mv.defense_backward(model, hb, hyp["defc"], dev)
            mv.clear_hooks(model); del adv, wr
            if a.benign_attack > 0:
                b = benign[rng.randrange(len(benign))]
                bb = mk(b["prompt"], b["resp"], b["ref"])
                bb = mv.move_batch(bb, dev)
                mv.freeze_all(model)
                badv, bwr, _ = mv.run_pgd_attack(model, bb, hidden_size=model.config.hidden_size,
                    layers=hyp["layers"], model_layers_module=hyp["mlm"], epsilon=hyp["eps"],
                    lr=hyp["inner"], iterations=hyp["iters"], coefs={"toward": 1.0},
                    embedding_epsilon=hyp["emb"], latent_epsilon=hyp["lat"],
                    attack_completions=False, device=dev, clip_grad=hyp["clip"])
                mv.set_requires_grad(trainable, True)
                bd_ = mv.defense_backward(model, bb, {"toward": 1.0, "away": 0.0}, dev,
                                          scale=a.benign_attack)
                mv.clear_hooks(model); del badv, bwr
                d["benign_def"] = bd_["total"]; d["weighted"] += bd_["weighted"]
            mv.zero_nan_grads(model)
            torch.nn.utils.clip_grad_norm_(trainable, hyp["clip"])
            optim.step()
        mv.save_eval_ckpt(model, tok, mv.Config(), f"{a.out}/epoch_{ep:03d}")
        print(f"ep{ep} done", flush=True)
    print("TRAIN_DONE", flush=True)

if __name__ == "__main__":
    main()

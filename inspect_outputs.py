"""inspect_outputs.py — 直接看 LAT 的 clean vs attacked 模型输出。

复用 motivation_verification.py 的 attack 管线（run_pgd_attack / GDAdversary），
对若干 harmful 样本分别做：
  - clean 续写：提示词 + 模型自由续写（无扰动）
  - attacked 续写：先对 prompt 潜变量跑 latent PGD，再在扰动下续写

关键实现细节：attack 张量宽度 = 数据 batch 的 pad 宽度（如 196）。因此续写时不调用
model.generate（它只喂 prompt 宽度），而是手动逐 token 贪婪解码，每步把当前序列右
补齐到 attack 宽度，保证 hook 内 mask 与张量宽度一致。

用途：验证 ASR=0 到底是「攻击无效」还是「防御强」。对照逻辑：未训练（预训练）模型在
攻击下应被明显撬动（输出有害内容/拒答消失）；若连预训练模型都撬不动，说明攻击管线
本身有问题。权重来源由 --model_path 指定（默认预训练权重）。
"""

from __future__ import annotations

import argparse
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch
import motivation_verification as mv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", default="/path/to/work/sxy/watermark_backdoor/models/Llama-3.2-1B")
    p.add_argument("--dataset", default="/path/to/lat_base/data/harmful-dataset")
    p.add_argument("--pgd_layers", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")
    p.add_argument("--epsilon", type=float, default=4.0)
    p.add_argument("--inner_lr", type=float, default=0.05)
    p.add_argument("--pgd_iterations", type=int, default=16)
    p.add_argument("--n_rows", type=int, default=4, help="how many samples to show")
    p.add_argument("--start_row", type=int, default=0, help="first sample index")
    p.add_argument("--max_new_tokens", type=int, default=48)
    p.add_argument("--device", default="cuda")
    p.add_argument("--truncate_length", type=int, default=384)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    mv.set_seed(a.seed)
    cfg = mv.Config(
        model_path=a.model_path,
        dataset=a.dataset,
        truncate_length=a.truncate_length,
        device=a.device,
        world_size=1,
        param_dtype="bfloat16",   # inference-only: halve VRAM so it coexists with training
    )
    mv.prepare_data_source(cfg)

    model, tokenizer, model_layers_module = mv.load_model_and_tokenizer(cfg)
    hidden_size = model.config.hidden_size
    layers = [int(x) for x in a.pgd_layers.split(",") if x.strip() != ""]
    print(f"[inspect] model={a.model_path} hidden={hidden_size} layers={layers} "
          f"eps={a.epsilon} pgd={a.pgd_iterations}", flush=True)

    collator = mv.LatentAdversarialTrainingDataCollator(
        tokenizer.pad_token_id, truncate_length=cfg.truncate_length
    )
    ds = mv.build_dataset(tokenizer, cfg, "train")
    print(f"[inspect] pool={len(ds)} rows; showing rows {a.start_row}..{a.start_row + a.n_rows - 1}",
          flush=True)

    adv_coefs = mv.normalize_dict({"toward": 0.5, "away": 0.5})
    model.eval()
    model.config.use_cache = False
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    def decode(prefix: list, width: int, n_new: int, do_sample: bool = True) -> str:
        """Decode, right-padding every step to `width`. `width` is the attack tensor
        width; once the sequence would exceed it we must stop (the hook can only index
        [1, width, d]). Uses sampling (temperature 0.6, top_p 0.9) to avoid the greedy
        repetition collapse seen with 1B/7B models."""
        seq = list(prefix)
        n_new = min(n_new, max(0, width - len(seq)))   # never exceed attack width
        with torch.no_grad():
            for _ in range(n_new):
                pad_n = max(0, width - len(seq))
                toks = torch.tensor([seq + [pad_id] * pad_n], device=a.device)
                attn = torch.ones(1, width, dtype=torch.long, device=a.device)
                if pad_n > 0:
                    attn[0, -pad_n:] = 0
                logits = model(input_ids=toks, attention_mask=attn).logits[0, len(seq) - 1]
                if do_sample:
                    logits = logits / 0.6
                    probs = torch.softmax(logits, dim=-1)
                    sorted_p, sorted_i = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sorted_p, dim=-1)
                    keep = cum <= 0.9
                    keep[1:] = keep[:-1].clone()   # keep at least the top token
                    keep[0] = True
                    sorted_p[~keep] = 0.0
                    probs = torch.zeros_like(probs).scatter(0, sorted_i, sorted_p)
                    probs = probs / probs.sum()
                    nxt = int(torch.multinomial(probs, 1).item())
                else:
                    nxt = int(logits.argmax().item())
                seq.append(nxt)
                if nxt == eos_id:
                    break
        return tokenizer.decode(seq[len(prefix):], skip_special_tokens=True)

    for i in range(a.start_row, a.start_row + a.n_rows):
        if i >= len(ds):
            break
        row = ds[i]
        prompt_toks = list(row["prompt_tokens"])

        print("\n" + "=" * 90)
        print(f"[sample {i}] prompt ({len(prompt_toks)} tok):")
        print(tokenizer.decode(row['prompt_tokens'], skip_special_tokens=True))
        print(f"  - refusal target :")
        print(f"    {tokenizer.decode(row['def_tokens'], skip_special_tokens=True)}")
        print(f"  - harmful target :")
        print(f"    {tokenizer.decode(row['adv_tokens'], skip_special_tokens=True)}")

        # ---- clean continuation (no hooks) ----
        text = decode(prompt_toks, width=len(prompt_toks) + a.max_new_tokens,
                      n_new=a.max_new_tokens)
        print(f"  [clean  ] {text}", flush=True)

        # ---- attacked continuation: optimise latent perturbation on prompt positions ----
        mv.freeze_all(model)
        with torch.enable_grad():
            batch = collator([row])
            batch = {k: v.to(a.device) for k, v in batch.items() if torch.is_tensor(v)}
            width = batch["adv_tokens"].shape[1]
            adversaries, wrappers, _ = mv.run_pgd_attack(
                model, batch,
                hidden_size=hidden_size,
                layers=layers,
                model_layers_module=model_layers_module,
                epsilon=a.epsilon,
                lr=a.inner_lr,
                iterations=a.pgd_iterations,
                coefs=adv_coefs,
                attack_completions=False,   # jailbreak semantics: perturb prompt only
                device=a.device,
            )
        text = decode(prompt_toks, width=width, n_new=a.max_new_tokens)
        print(f"  [attacked] {text}", flush=True)
        mv.clear_hooks(model)
        mv.set_requires_grad(model.parameters(), False)
        del adversaries, wrappers
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

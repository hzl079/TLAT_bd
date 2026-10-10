#!/usr/bin/env python3
"""Numerical sanity check of the loss terms on a real batch (no training)."""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import motivation_verification as mv

MODEL = "/path/to/lat_base/models/Llama-3.2-1B-Instruct"
DATASET = "/path/to/lat/robust_asr_instruct_eps03_pgd8_smoke/_dataset_export.jsonl"

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    rows = [json.loads(x) for x in open(DATASET)]
    row = rows[0]
    cfg = mv.Config(model_path=MODEL, dataset_json=DATASET, rank=0, world_size=1)
    model, _, _ = mv.load_model_and_tokenizer(cfg)
    model.eval()
    b = mv._build_eval_attack_batch(tok, row, cfg)  # def_tokens=refusal, adv_tokens=harmful

    def_t = b["def_tokens"]; def_m = b["def_labels_mask"]
    adv_t = b["adv_tokens"]; adv_m = b["adv_labels_mask"]

    # --- toward: CE on refusal completion (defender's toward) ---
    lt = mv._toward_ce(model, def_t, def_m)
    # --- away: log(1-p) on harmful completion (defender's away) ---
    la = mv._away_log1mp(model, adv_t, adv_m)
    # --- toward harmful (adversary) ---
    lta = mv._toward_ce(model, adv_t, adv_m)
    # --- away refusal (adversary) ---
    laa = mv._away_log1mp(model, def_t, def_m)

    print("defender toward refusal CE :", round(lt.item(), 4))
    print("defender away harmful log1mp:", round(la.item(), 4))
    print("adversary toward harmful CE :", round(lta.item(), 4))
    print("adversary away refusal log1mp:", round(laa.item(), 4))
    print("finite check:", all(torch.isfinite(x) for x in [lt, la, lta, laa]))

    # gradient direction: minimize defender toward -> refusal loss should decrease
    lt.backward(retain_graph=True)
    grad_norm_toward = torch.norm(torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None]))
    model.zero_grad(set_to_none=True)
    la.backward(retain_graph=True)
    grad_norm_away = torch.norm(torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None]))
    model.zero_grad(set_to_none=True)
    print("grad norm (defender toward refusal):", round(grad_norm_toward.item(), 3))
    print("grad norm (defender away harmful):  ", round(grad_norm_away.item(), 3))
    print("grad finite:", torch.isfinite(grad_norm_toward) and torch.isfinite(grad_norm_away))

if __name__ == "__main__":
    main()

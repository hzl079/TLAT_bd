#!/usr/bin/env python3
"""Plot the complete r3 40-epoch LAT run: ASR x2, NLL margin, adv losses, benign side."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/path/to/lat/robust_asr_instruct_1b_ml_lr5e6_epst06_prefill_r3"
hist = json.load(open(f"{OUT}/history.json"))["history"]
fg = json.load(open(f"{OUT}/generations/harmbench_scores_robust/asr_curve.json"))
pf = json.load(open(f"{OUT}/generations_contonly/harmbench_scores_prefill_contonly/asr_curve.json"))

ep = [h["epoch"] for h in hist]
fg_ep = [x["epoch"] for x in fg]
pf_ep = [x["epoch"] for x in pf]

fg_raw = [100 * x["val_robust_asr"] for x in fg]
fg_val = [100 * x["val_robust_asr_valid"] for x in fg]
pf_raw = [100 * x["val_robust_asr"] for x in pf]

margin = [h["val_nll_margin_harmful_minus_refusal"] for h in hist]
h_nll = [h["val_harmful_nll"] for h in hist]
r_nll = [h["val_refusal_nll"] for h in hist]
tadv = [h["test_adv_loss"] for h in hist]
tadv_toward = [h.get("test_adv_loss_toward") for h in hist]
comp = [h["val_benign_compliance"] for h in hist]
nref = [h["val_benign_refusal"] for h in hist]
ndgn = [h["val_benign_degenerate"] for h in hist]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("1B LAT robust overfitting check  |  lr=5e-6, eps=0.3/[4,8,12], 40ep  (r3)", fontsize=14)

# --- Panel 1: ASR ---
ax = axes[0][0]
ax.plot(fg_ep, fg_raw, "-o", ms=3, label="free-gen eps=0.6 (raw)", color="tab:red")
ax.plot(fg_ep, fg_val, "--o", ms=3, label="free-gen eps=0.6 (valid)", color="tab:red", alpha=0.6)
ax.plot(pf_ep, pf_raw, "-s", ms=3, label="prefill eps=0.3 (cont-only)", color="tab:blue")
ax.set_xlabel("epoch"); ax.set_ylabel("robust ASR (%)")
ax.set_ylim(-2, 100)
ax.set_title("Attack-time ASR (HarmBench, n=256/epoch)")
ax.legend(); ax.grid(alpha=0.3)

# --- Panel 2: NLL margin ---
ax = axes[0][1]
ax.plot(ep, h_nll, label="harmful NLL", color="tab:red")
ax.plot(ep, r_nll, label="refusal NLL", color="tab:green")
ax.plot(ep, margin, label="margin (harmful - refusal)", color="tab:blue", lw=2)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("epoch"); ax.set_ylabel("teacher-forced NLL")
ax.set_title("Val refusal margin (non-saturated)")
ax.legend(); ax.grid(alpha=0.3)

# --- Panel 3: train vs test adv_loss ---
ax = axes[1][0]
ax.plot(ep, [h["train_adv_loss"] for h in hist], "-o", ms=3, label="train adv_loss (eps=0.3)", color="tab:orange")
ax.plot(ep, tadv, "-o", ms=3, label="test adv_loss (eps=0.6)", color="tab:purple")
ax.plot(ep, [h["train_def_objective"] for h in hist], "--", label="def_objective (eps=0.3)", color="tab:gray")
ax.set_yscale("log")
ax.set_xlabel("epoch"); ax.set_ylabel("loss (log)")
ax.set_title("train vs test adversarial loss (lower=attack stronger)")
ax.legend(); ax.grid(alpha=0.3, which="both")

# --- Panel 4: benign side ---
ax = axes[1][1]
ax.plot(ep, comp, "-o", ms=3, label="benign compliance", color="tab:green")
ax.set_ylabel("compliance", color="tab:green")
ax.set_ylim(0, 1.05)
ax2 = ax.twinx()
ax2.plot(ep, nref, "-s", ms=3, label="refusals (count)", color="tab:orange")
ax2.plot(ep, ndgn, "-^", ms=3, label="degenerate (count)", color="tab:red")
ax2.set_ylabel("count (of 64)", color="tab:red")
ax.set_xlabel("epoch")
ax.set_title("Benign side (n=64/epoch)")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center right"); ax.grid(alpha=0.3)

plt.tight_layout()
path = f"{OUT}/r3_40ep_curves.png"
plt.savefig(path, dpi=150)
print("saved ->", path)

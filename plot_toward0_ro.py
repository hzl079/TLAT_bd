#!/usr/bin/env python3
"""Plot the toward=0 20ep run: sustained robust-overfitting (U-shape) on the paper metric."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/path/to/lat/robust_asr_instruct_1b_ml_lr5e6_deftoward0_20ep"
hist = json.load(open(f"{OUT}/history.json"))["history"]
fg = json.load(open("/tmp/fg_curve.json"))
pf = json.load(open("/tmp/pf_curve.json"))

ep = [h["epoch"] for h in hist]
fg_ep = [x["epoch"] for x in fg]
pf_ep = [x["epoch"] for x in pf]
fg_raw = [100 * x["val_robust_asr"] for x in fg]
fg_val = [100 * x["val_robust_asr_valid"] for x in fg]
pf_raw = [100 * x["val_robust_asr"] for x in pf]

margin = [h["val_nll_margin_harmful_minus_refusal"] for h in hist]
tadv = [h["test_adv_loss"] for h in hist]
tadv_train = [h["train_adv_loss"] for h in hist]
comp = [h["val_benign_compliance"] for h in hist]
nref = [h["val_benign_refusal"] for h in hist]
ndgn = [h["val_benign_degenerate"] for h in hist]

best_ep = fg_ep[fg_raw.index(min(fg_raw))]
peak_ep = fg_ep[fg_raw.index(max(fg_raw))]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("toward=0 (no refusal template) | 1B LAT, lr=5e-6, eps=0.3/[4,8,12], 20ep | RO U-shape",
             fontsize=14)

# Panel 1: ASR — the RO U-shape
ax = axes[0][0]
ax.plot(fg_ep, fg_raw, "-o", ms=4, color="tab:red", lw=2, label="free-gen eps=0.6 (raw)")
ax.plot(fg_ep, fg_val, "--o", ms=3, color="tab:red", alpha=0.55, label="free-gen eps=0.6 (valid)")
ax.plot(pf_ep, pf_raw, "-s", ms=3, color="tab:blue", label="prefill eps=0.3 (cont-only)")
ax.axvline(best_ep, color="tab:green", ls=":", lw=1.5)
ax.axvspan(best_ep, fg_ep[-1], color="tab:orange", alpha=0.08)
ax.annotate(f"best ep{best_ep}: {min(fg_raw):.1f}%", xy=(best_ep, min(fg_raw)),
            xytext=(best_ep + 0.5, 30), fontsize=9, color="tab:green")
ax.annotate(f"recovery peak ep{peak_ep}: {max(fg_raw):.1f}%", xy=(peak_ep, max(fg_raw)),
            xytext=(peak_ep - 6, 60), fontsize=9, color="tab:red",
            arrowprops=dict(arrowstyle="->", color="tab:red"))
ax.set_xlabel("epoch"); ax.set_ylabel("robust ASR (%)")
ax.set_ylim(-2, 100)
ax.set_title("Generation ASR under attack: train-strength (0.3) holds, test-strength (0.6) recovers")
ax.legend(); ax.grid(alpha=0.3)

# Panel 2: train vs test adv loss divergence
ax = axes[0][1]
ax.plot(ep, tadv_train, "-o", ms=3, color="tab:orange", label="train adv_loss (eps=0.3)")
ax.plot(ep, tadv, "-o", ms=3, color="tab:purple", label="test adv_loss (eps=0.6)")
ax.set_xlabel("epoch"); ax.set_ylabel("adv_loss (lower=attack stronger)")
ax.set_title("train keeps defending / test attack regains strength (divergence)")
ax.legend(); ax.grid(alpha=0.3)

# Panel 3: NLL margin
ax = axes[1][0]
ax.plot(ep, margin, "-o", ms=3, color="tab:blue", lw=2, label="margin (harmful - refusal)")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("epoch"); ax.set_ylabel("teacher-forced NLL margin")
ax.set_title("Val refusal margin")
ax.legend(); ax.grid(alpha=0.3)

# Panel 4: benign side
ax = axes[1][1]
ax.plot(ep, comp, "-o", ms=3, color="tab:green", label="benign compliance")
ax.set_ylabel("compliance", color="tab:green"); ax.set_ylim(0, 1.05)
ax2 = ax.twinx()
ax2.plot(ep, nref, "-s", ms=3, color="tab:orange", label="refusals")
ax2.plot(ep, ndgn, "-^", ms=3, color="tab:red", label="degenerate")
ax2.set_ylabel("count (of 64)", color="tab:red")
ax.set_xlabel("epoch")
ax.set_title("Benign side (n=64/epoch)")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center right"); ax.grid(alpha=0.3)

plt.tight_layout()
path = f"{OUT}/toward0_20ep_RO.png"
plt.savefig(path, dpi=150)
print("saved ->", path)

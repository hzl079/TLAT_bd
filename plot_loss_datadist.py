#!/usr/bin/env python3
"""Full loss/NLL curves for the data-distribution experiment (ro_data_dist_378), 2x2 panels."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/path/to/lat/ro_data_dist_378"
h = json.load(open(f"{OUT}/history.json"))["history"]

ep = [r["epoch"] for r in h]
def col(k):
    return [r.get(k) for r in h]

test_adv = col("test_adv_loss")
train_adv = col("train_adv_loss")
def_obj = col("train_def_objective")
h_nll = col("val_harmful_nll")
r_nll = col("val_refusal_nll")
margin = col("val_nll_margin_harmful_minus_refusal")

# replace None (ep0 train_adv/def_obj) with nan for plotting
def clean(xs):
    return [float("nan") if v is None else v for v in xs]

fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=160)
fig.suptitle("Data-distribution experiment (ro_data_dist_378) — loss / NLL curves\n"
             "train=terrorist n=378, held-out val n=512, toward=0.5, eps emb0.05+lat0.3, lr5e-6",
             fontsize=12, y=0.985)

ax = axes[0, 0]
ax.plot(ep, clean(train_adv), "s-", color="#c0392b", lw=2, ms=4, label="train_adv_loss")
ax.plot(ep, clean(test_adv), "o-", color="#e67e22", lw=2, ms=4, label="test_adv_loss (held-out)")
ax.axvline(15, color="gray", ls=":", lw=1); ax.text(15.2, 2, "lr decay", fontsize=8, color="gray")
ax.set_title("adversary loss (↑ = attack weaker = defense stronger)", fontsize=10.5)
ax.set_xlabel("epoch"); ax.set_ylabel("adv_loss"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(ep, clean(def_obj), "d-", color="#2c3e50", lw=2, ms=4)
ax.set_title("train defense objective (↓ over training)", fontsize=10.5)
ax.set_xlabel("epoch"); ax.set_ylabel("def_objective"); ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(ep, clean(h_nll), "o-", color="#c0392b", lw=2, ms=4, label="harmful NLL")
ax.plot(ep, clean(r_nll), "s-", color="#27ae60", lw=2, ms=4, label="refusal NLL")
ax.set_title("val teacher-forced NLL (harmful rises, refusal flat-low)", fontsize=10.5)
ax.set_xlabel("epoch"); ax.set_ylabel("NLL"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(ep, clean(margin), "o-", color="#8e44ad", lw=2, ms=4)
ax.set_title("refusal margin = harmful_NLL - refusal_NLL (↑ = prefers refusal)", fontsize=10.5)
ax.set_xlabel("epoch"); ax.set_ylabel("margin"); ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
png = os.path.join(OUT, "ro_loss_curves_datadist.png")
plt.savefig(png, bbox_inches="tight")
print(f"saved {png}")

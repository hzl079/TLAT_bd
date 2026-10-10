#!/usr/bin/env python3
"""train def loss vs test def loss (ro_data_dist_378). Single panel, two lines."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/path/to/lat/ro_data_dist_378"
h = json.load(open(f"{OUT}/history.json"))["history"]
ep = [r["epoch"] for r in h]


def clean(k):
    return [float("nan") if r.get(k) is None else r.get(k) for r in h]


train_def = clean("train_def_loss")      # defender loss on TRAIN (post-update)
test_def = clean("test_adv_loss")         # defense objective (toward+away) on HELD-OUT under attack

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
ax.plot(ep, train_def, "s-", color="#2c3e50", lw=2, ms=5, label="train def loss (train_def_loss)")
ax.plot(ep, test_def, "o-", color="#c0392b", lw=2, ms=5, label="test def loss (held-out, under attack)")
ax.axvline(15, color="gray", ls=":", lw=1)
ax.text(15.2, 6, "lr decay", fontsize=9, color="gray")
ax.set_xlabel("epoch", fontsize=12)
ax.set_ylabel("defense loss", fontsize=12)
ax.set_title("Data-distribution experiment: train vs test defense loss\n"
             "(train=terrorist n=378, held-out n=512, toward=0.5)", fontsize=11)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
png = os.path.join(OUT, "ro_def_loss_train_vs_test.png")
plt.savefig(png, bbox_inches="tight")
print(f"saved {png}")

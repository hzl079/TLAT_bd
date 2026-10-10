#!/usr/bin/env python3
"""4-panel RO figure: heldout-robust / train-robust / heldout-clean / benign-compliance."""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/path/to/lat/ro_data_dist_378"


def read_curve(path):
    return [(r["epoch"], r) for r in json.load(open(path))]


def read_clean():
    pts = []
    for p in sorted(glob.glob(f"{OUT}/generations/harmbench_clean/epoch_*.jsonl")):
        rows = [json.loads(l) for l in open(p)]
        epoch = int(re.search(r"epoch_(\d+)", p).group(1))
        pts.append((epoch, sum(r["harmful"] for r in rows) / len(rows)))
    return pts


def read_history(key):
    h = json.load(open(f"{OUT}/history.json"))["history"]
    return [(r["epoch"], r.get(key)) for r in h if r.get(key) is not None]


heldout = read_curve(f"{OUT}/generations/harmbench_heldout/asr_curve.json")
train = read_curve(f"{OUT}/generations_train/harmbench_train/asr_curve.json")
clean = read_clean()
benign = read_history("val_benign_compliance")

hx = [e for e, _ in heldout]
hy = [r["val_robust_asr"] * 100 for _, r in heldout]
tx = [e for e, _ in train]
ty = [r["val_robust_asr"] * 100 for _, r in train]
cx = [e for e, _ in clean]
cy = [a * 100 for e, a in clean]
bx = [e for e, _ in benign]
by = [v * 100 for _, v in benign]

fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=160)
fig.suptitle("LAT data-distribution robust overfitting  "
             "(train = terrorist topic n=378, held-out = 29 topics n=512, toward=0.5)",
             fontsize=12.5, y=0.985)

# (0,0) held-out robust ASR —— the U-shape
ax = axes[0, 0]
ax.plot(hx, hy, "o-", color="#c0392b", lw=2, ms=4)
trough_i = hy.index(min(hy))
ax.annotate(f"trough {hy[trough_i]:.1f}% (ep{hx[trough_i]})",
            xy=(hx[trough_i], hy[trough_i]), xytext=(hx[trough_i] + 2, hy[trough_i] + 14),
            arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9)
ax.axvline(15, color="gray", ls=":", lw=1)
ax.set_title("held-out robust ASR (attacked)  —  U-shape", fontsize=11)
ax.set_ylim(-2, 100)
ax.set_xlabel("epoch"); ax.set_ylabel("ASR (%)")
ax.grid(alpha=0.3)

# (0,1) train robust ASR —— flat 0
ax = axes[0, 1]
ax.plot(tx, ty, "s--", color="#2c3e50", lw=2, ms=4)
ax.set_title("train robust ASR (attacked)  —  flat 0%", fontsize=11)
ax.set_ylim(-2, 100)
ax.set_xlabel("epoch"); ax.set_ylabel("ASR (%)")
ax.grid(alpha=0.3)

# (1,0) held-out clean ASR —— flat ~0
ax = axes[1, 0]
ax.plot(cx, cy, "d:", color="#27ae60", lw=2, ms=6)
ax.set_title("held-out clean ASR (no attack)  —  flat ~0%", fontsize=11)
ax.set_ylim(-0.5, 5)
ax.set_xlabel("epoch"); ax.set_ylabel("ASR (%)")
ax.grid(alpha=0.3)

# (1,1) benign compliance —— stays high
ax = axes[1, 1]
ax.plot(bx, by, "o-", color="#8e44ad", lw=2, ms=4)
ax.set_title("benign compliance  —  stays high (no degeneration)", fontsize=11)
ax.set_ylim(0, 100)
ax.set_xlabel("epoch"); ax.set_ylabel("compliance (%)")
ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
png = os.path.join(OUT, "ro_curve_data_dist.png")
plt.savefig(png, bbox_inches="tight")
print(f"saved {png}")

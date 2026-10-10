#!/usr/bin/env python3
"""Plot the 'loss-shaped' RO diagnostics vs ASR for the data-dist run."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/path/to/lat/ro_data_dist_378"
LE = f"{OUT}/loss_eval"
EPOCHS = [1, 5, 10, 15, 20, 25, 30]


def load_agg(split):
    d = {}
    for ep in EPOCHS:
        p = f"{LE}/{split}_ep{ep:03d}.json"
        if os.path.exists(p):
            d[ep] = json.load(open(p))
    return d


def asr_series(path):
    rows = json.load(open(path))
    m = {r["epoch"]: r["val_robust_asr"] for r in rows}
    return m


ho = load_agg("heldout")
tr = load_agg("train")
asr_ho = asr_series(f"{OUT}/generations/harmbench_heldout/asr_curve.json")
asr_tr = asr_series(f"{OUT}/generations_train/harmbench_train/asr_curve.json")

eps = [e for e in EPOCHS if e in ho]
comp_ho = [ho[e]["comp_mass"] for e in eps]
comp_tr = [tr[e]["comp_mass"] for e in eps]
asr_ho_full = sorted(asr_ho.items())
asr_ho_ep = [e for e, _ in asr_ho_full]
asr_ho_v = [v for _, v in asr_ho_full]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Panel A: compliance prob vs ASR
ax = axes[0]
ax.plot(asr_ho_ep, asr_ho_v, color="tab:red", alpha=0.35, lw=2,
        label="held-out robust ASR (full)")
ax.plot(eps, [asr_ho[e] for e in eps], "o", color="tab:red", ms=8,
        label="held-out ASR (sampled)")
ax.plot(eps, comp_ho, "s-", color="tab:blue", lw=2.5, ms=8,
        label="held-out compliance prob (comp_mass)")
ax.plot(eps, comp_tr, "^--", color="tab:green", lw=2, ms=7,
        label="train compliance prob")
ax.set_xlabel("epoch")
ax.set_ylabel("probability / ASR")
ax.set_title("Compliance probability tracks ASR\n(train crushed to 0, held-out rises & stays = RO)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel B: teacher-forced fixed-target losses are blind to RO
ax = axes[1]
ax.plot(eps, [ho[e]["harmful_ce"] for e in eps], "o-", color="tab:purple", lw=2,
        label="held-out harmful-target CE")
ax.plot(eps, [ho[e]["aff_prefix_nll"] for e in eps], "s-", color="tab:orange", lw=2,
        label="held-out affirmative-prefix NLL")
ax.plot(eps, [ho[e]["refusal_ce"] for e in eps], "d-", color="tab:brown", lw=2,
        label="held-out refusal-target CE")
ax.plot(eps, [ho[e]["self_nll"] for e in eps], "v-", color="tab:gray", lw=2,
        label="held-out self-NLL (own gen)")
ax2 = ax.twinx()
ax2.plot(asr_ho_ep, asr_ho_v, color="tab:red", alpha=0.3, lw=2)
ax2.set_ylabel("held-out ASR", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax.set_xlabel("epoch")
ax.set_ylabel("teacher-forced NLL / CE")
ax.set_title("Fixed-target teacher-forced losses are BLIND to RO\n(monotone up = looks like 'defense strengthening')")
ax.legend(fontsize=9, loc="center right")
ax.grid(alpha=0.3)

plt.tight_layout()
out_png = f"{OUT}/ro_loss_compliance_vs_asr.png"
plt.savefig(out_png, dpi=130)
print("saved", out_png)

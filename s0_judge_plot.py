#!/usr/bin/env python3
"""HarmBench-judge the method-0 generations, aggregate per (run, target, split, epoch)
and plot ASR / target NLL / compliance mass for train vs held-out."""
import glob
import json
import os
import sys

import numpy as np

MV = "/path/to/lat"
sys.path.insert(0, MV)
ROOT = f"{MV}/s0_decoupled"
JUDGE = "/path/to/lat_base/models/HarmBench-Mistral-7b-val-cls"


def judge_all(device):
    from score_harmbench_generations import HarmBenchJudge
    judge = None
    for path in sorted(glob.glob(f"{ROOT}/*/*_ep*.jsonl")):
        if path.endswith(".judged.jsonl"):
            continue
        out = path.replace(".jsonl", ".judged.jsonl")
        rows = [json.loads(l) for l in open(path)]
        if os.path.exists(out) and sum(1 for _ in open(out)) == len(rows):
            continue
        if judge is None:
            judge = HarmBenchJudge(JUDGE, device, 8)
        res = judge.judge([str(r["prompt"]) for r in rows], [str(r["response"]) for r in rows])
        with open(out, "w") as f:
            for r, (h, o) in zip(rows, res):
                r["harmful"] = int(h)
                r["classifier_output"] = o
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"judged {os.path.basename(path)}: ASR={np.mean([h for h, _ in res]):.4f}", flush=True)


def aggregate():
    table = []
    for path in sorted(glob.glob(f"{ROOT}/*/*.judged.jsonl")):
        rows = [json.loads(l) for l in open(path)]
        r0 = rows[0]
        valid = [r for r in rows if str(r.get("degeneration_kind")) in ("none", "None")]
        table.append({
            "run": r0["run"], "target": r0["target"], "split": r0["split"], "epoch": r0["epoch"],
            "n": len(rows),
            "asr_raw": float(np.mean([r["harmful"] for r in rows])),
            "asr_valid": float(np.mean([r["harmful"] for r in valid])) if valid else float("nan"),
            "degen_rate": 1 - len(valid) / len(rows),
            "target_nll": float(np.mean([r["target_nll"] for r in rows])),
            "comp_mass": float(np.mean([r["comp_mass"] for r in rows])),
        })
    table.sort(key=lambda t: (t["run"], t["target"], t["split"], t["epoch"]))
    json.dump(table, open(f"{ROOT}/summary.json", "w"), indent=2)
    for t in table:
        print(f"{t['run']:6s} {t['target']:5s} {t['split']:8s} ep{t['epoch']:3d} "
              f"ASR={t['asr_raw']:.3f} valid={t['asr_valid']:.3f} degen={t['degen_rate']:.3f} "
              f"nll={t['target_nll']:.3f} comp={t['comp_mass']:.3f}")
    return table


def plot(table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for run in sorted({t["run"] for t in table}):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ti, target in enumerate(["sure", "self"]):
            for mi, (metric, label) in enumerate([("asr_valid", "robust ASR (valid)"),
                                                  ("target_nll", "attacked target NLL (higher=more robust)"),
                                                  ("comp_mass", "compliance mass")]):
                ax = axes[ti, mi]
                for split, c in (("train", "tab:blue"), ("heldout", "tab:red")):
                    pts = [(t["epoch"], t[metric]) for t in table
                           if t["run"] == run and t["target"] == target and t["split"] == split]
                    if pts:
                        x, y = zip(*pts)
                        ax.plot(x, y, "o-", color=c, label=split)
                ax.set_title(f"{run} / target={target}: {label}", fontsize=9)
                ax.set_xlabel("epoch")
                ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{ROOT}/s0_{run}.png", dpi=110)
        print(f"saved {ROOT}/s0_{run}.png")


if __name__ == "__main__":
    if "--no-judge" not in sys.argv:
        judge_all(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith("cuda") else "cuda:0")
    plot(aggregate())

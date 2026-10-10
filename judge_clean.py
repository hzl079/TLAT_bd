#!/usr/bin/env python3
"""Judge CLEAN (no-attack) held-out generations with HarmBench -> val_direct_asr curve.

Reuses HarmBenchJudge + read_harmful_rows from score_harmbench_generations.py, but accepts
the `val_epochNNN.jsonl` clean files (the robust scorer hard-rejects them).
"""
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/path/to/lat")
from score_harmbench_generations import (  # noqa: E402
    HarmBenchJudge,
    epoch_from_path,
    read_completed,
    read_harmful_rows,
)

JUDGE = "/path/to/lat_base/models/HarmBench-Mistral-7b-val-cls"
GEN_DIR = "/path/to/lat/ro_data_dist_378/generations"
SCORE_DIR = Path(GEN_DIR) / "harmbench_clean"
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"

SCORE_DIR.mkdir(exist_ok=True)
judge = HarmBenchJudge(JUDGE, DEVICE, 8)

paths = sorted(glob.glob(os.path.join(GEN_DIR, "val_epoch*.jsonl")), key=epoch_from_path)
print(f"[clean] judging {len(paths)} clean epochs -> {SCORE_DIR}", flush=True)

curve = []
for path in paths:
    epoch = epoch_from_path(path)
    rows = read_harmful_rows(path)
    score_path = SCORE_DIR / f"epoch_{epoch:03d}.jsonl"
    completed = read_completed(score_path)
    missing = [r for r in rows if int(r["id"]) not in completed]
    if missing:
        judgments = judge.judge(
            [str(r["prompt"]) for r in missing],
            [str(r["response"]) for r in missing],
        )
        with score_path.open("a") as f:
            for row, (harmful, out) in zip(missing, judgments):
                f.write(json.dumps({
                    "epoch": epoch, "id": int(row["id"]),
                    "prompt": row["prompt"], "response": row["response"],
                    "harmful": harmful, "classifier_output": out,
                    "degeneration_kind": row.get("degeneration_kind"),
                    "metric": "direct_asr",
                }, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        completed = read_completed(score_path)

    labels = [int(completed[i]["harmful"]) for i in range(len(rows))]
    degs = [str(completed[i].get("degeneration_kind")) for i in range(len(rows))]
    valid_mask = [d in ("none", "None") for d in degs]
    labels_valid = [l for l, m in zip(labels, valid_mask) if m]
    n_valid = sum(valid_mask)
    asr_valid = (sum(labels_valid) / len(labels_valid) if labels_valid else float("nan"))
    result = {
        "epoch": epoch, "n_harmful": len(labels), "harmful_count": sum(labels),
        "val_direct_asr": sum(labels) / len(labels),
        "n_valid": n_valid, "harmful_count_valid": sum(labels_valid),
        "val_direct_asr_valid": asr_valid,
        "degeneration_rate": (1.0 - n_valid / len(labels) if labels else float("nan")),
    }
    curve.append(result)
    print(f"[clean] epoch={epoch:03d}: direct_ASR={result['val_direct_asr']:.4f} "
          f"({result['harmful_count']}/{result['n_harmful']}) "
          f"valid={asr_valid:.4f} degen={result['degeneration_rate']:.4f}", flush=True)

curve.sort(key=lambda r: r["epoch"])
with (SCORE_DIR / "asr_curve.json").open("w") as f:
    json.dump(curve, f, indent=2)
print(f"saved {SCORE_DIR / 'asr_curve.json'}", flush=True)

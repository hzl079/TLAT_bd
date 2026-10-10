#!/usr/bin/env python3
"""Score the one-shot test_final generation (final weights on the untouched test split)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/path/to/lat")
from score_harmbench_generations import HarmBenchJudge  # reuse the same judge

path = Path(sys.argv[1])
device = sys.argv[2] if len(sys.argv) > 2 else "cuda:2"

rows = [json.loads(x) for x in path.open()]
harm = [r for r in rows if r.get("kind") == "harmful"]
print(f"loading {path.name}: {len(harm)} harmful rows")

judge = HarmBenchJudge("/path/to/lat_base/models/HarmBench-Mistral-7b-val-cls", device, 8)
behaviors = [r["prompt"] for r in harm]
generations = [r["response"] for r in harm]
scored = judge.judge(behaviors, generations)
labels = [l for l, _ in scored]

n = len(labels)
raw = sum(labels) / n
degs = [str(r.get("degeneration_kind")) for r in harm]
valid = [d in ("none", "None") for d in degs]
n_valid = sum(valid)
labels_valid = [l for l, m in zip(labels, valid) if m]
valid_asr = sum(labels_valid) / n_valid if n_valid else float("nan")

print(f"test_final: n={n} raw_ASR={raw:.4f} ({sum(labels)}/{n}) "
      f"valid_ASR={valid_asr:.4f} ({sum(labels_valid)}/{n_valid}) "
      f"degeneration_rate={(1-n_valid/n):.4f}")

#!/usr/bin/env python3
"""Build a GCG prompt pool for the data-distribution experiment: 25 train-topic + 25 held-out."""
import json
import random

OUT = "/path/to/lat/ro_data_dist_378"
DATASET = "/path/to/lat/ro_main/_dataset_export.jsonl"
SPLIT = "/path/to/lat/topic_split/data_dist_split_378.json"

rng = random.Random(42)
rows = [json.loads(l) for l in open(DATASET) if l.strip()]
sp = json.load(open(SPLIT))["split"]
train_idx = [int(i) for i in sp["train"]]
held_idx = [int(i) for i in sp["val"]] + [int(i) for i in sp["test"]]

rng.shuffle(train_idx)
rng.shuffle(held_idx)
train_sel = train_idx[:75]
held_sel = held_idx[:75]

pool = []
for i in train_sel:
    pool.append({"id": len(pool), "split": "train_topic", "prompt": rows[i]["prompt"]})
for i in held_sel:
    pool.append({"id": len(pool), "split": "heldout", "prompt": rows[i]["prompt"]})

with open(f"{OUT}/gcg_pool_datadist.jsonl", "w") as f:
    for p in pool:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
with open(f"{OUT}/gcg_pool_datadist_split.json", "w") as f:
    json.dump({"train_topic_ids": list(range(75)), "heldout_ids": list(range(75, 150))}, f)

print(f"[pool] total={len(pool)}  train_topic={len(train_sel)}  heldout={len(held_sel)}")
for p in pool[:3]:
    print("  train:", p["prompt"][:60])
for p in pool[25:28]:
    print("  heldout:", p["prompt"][:60])

#!/usr/bin/env python3
"""Build the borderline-benign set for symmetric LAT: OR-Bench-Hard-1K prompts with a
DeepSeek-written helpful answer (SFT / defender target) and a fixed refusal (attacker target).
Writes data_orbench/benign_train (save_to_disk, 1000 rows) and heldout.jsonl (rest)."""
import csv
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

D = "/path/to/lat/data_orbench"
cfg = json.load(open("/path/to/lat/.qianfan_api"))
client = OpenAI(api_key=cfg["api_key"], base_url=cfg.get("base_url"),
                default_headers={"x-bce-appid": cfg.get("app_id", "")})
REFUSAL = "I'm sorry, but I can't help with that request."
SYS = ("You are a helpful assistant. The user request is benign even if it sounds sensitive. "
       "Answer it helpfully, safely and directly in at most 150 words. Do not refuse, do not "
       "lecture, do not add disclaimers.")
CACHE = f"{D}/responses.jsonl"


def ask(p):
    for a in range(6):
        try:
            r = client.chat.completions.create(
                model=cfg["model"], temperature=0.3, max_tokens=400,
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": p}])
            return r.choices[0].message.content.strip()
        except Exception as e:
            time.sleep(5 * (a + 1))
    return None


def main():
    prompts = [r["prompt"] for r in csv.DictReader(open(f"{D}/or-bench-hard-1k.csv"))]
    done = {}
    if os.path.exists(CACHE):
        for l in open(CACHE):
            r = json.loads(l)
            done[r["prompt"]] = r["response"]
    todo = [p for p in prompts if p not in done]
    print(f"{len(prompts)} prompts, {len(todo)} to generate", flush=True)
    with open(CACHE, "a") as f, ThreadPoolExecutor(6) as ex:
        for i, (p, resp) in enumerate(zip(todo, ex.map(ask, todo))):
            if resp:
                f.write(json.dumps({"prompt": p, "response": resp}, ensure_ascii=False) + "\n")
                f.flush()
                done[p] = resp
            if (i + 1) % 100 == 0:
                print(f"{i+1}/{len(todo)}", flush=True)
    bad = ("I'm sorry", "I cannot", "I can't", "As an AI")
    rows = [{"prompt": p, "response": done[p], "refusal": REFUSAL}
            for p in prompts if p in done and not done[p].startswith(bad)]
    random.Random(0).shuffle(rows)
    train, held = rows[:1000], rows[1000:]
    from datasets import Dataset
    Dataset.from_list(train).save_to_disk(f"{D}/benign_train")
    with open(f"{D}/heldout.jsonl", "w") as f:
        for r in held:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{D}/train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"usable={len(rows)} train={len(train)} heldout={len(held)}", flush=True)


if __name__ == "__main__":
    main()

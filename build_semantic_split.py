"""
build_semantic_split.py — 语义防泄漏切分（harmful-dataset → train/val/test）

背景
----
位置切分（按原始行序硬切 train[:N] / train[N:M]）允许近义/改写行跨 train 与
held-out 分布，构成泄漏：held-out 指标被高估，robust-overfitting 结论不可信。
本脚本对**整个池**的 prompt 做语义编码，把互为近义的行归到同一簇，再**整簇**贪心
分配到 train / val / test —— 任何簇整体只进一段，从而保证三段互不相交且无近义行
跨段。

用法
----
python build_semantic_split.py \
    --dataset /path/to/lat_base/data/harmful-dataset \
    --out ./semantic_split.json \
    --n_train 1024 --n_val 512 --n_test 512 --sim_threshold 0.8

产物 semantic_split.json（池内绝对行号）：
{
  "pool_size": 4948,
  "sim_threshold": 0.8,
  "embedder": "sentence-transformers/all-MiniLM-L6-v2",
  "n_clusters": ...,
  "max_cluster_size": ...,
  "cluster_sizes": [..],
  "split": {"train": [...], "val": [...], "test": [...]}   # 互不相交
}
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np


def load_pool(dataset: str) -> List[str]:
    """Load every `prompt` in the pool. Handles both a save_to_disk() directory and a
    plain HF id / file path (load_dataset cannot read save_to_disk dirs, so try that first)."""
    from datasets import load_dataset, load_from_disk
    try:
        ds = load_dataset(dataset, split="train")
    except Exception:
        dd = load_from_disk(dataset)
        ds = dd["train"] if isinstance(dd, dict) else dd
    cols = ds.column_names
    if "prompt" not in cols:
        raise ValueError(f"dataset columns {cols} have no 'prompt'")
    return list(ds["prompt"])


def embed_prompts(prompts: List[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    vec = model.encode(prompts, batch_size=64, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(vec, dtype=np.float32)   # L2-normalised: cos = dot


def cluster_greedy(vec: np.ndarray, threshold: float) -> List[List[int]]:
    """Greedy near-duplicate clustering over L2-normalised vectors.

    Row order defines a deterministic canonical ordering; each unseen row joins the
    first cluster whose centroid is at cosine similarity >= threshold, otherwise starts
    a new cluster. Cluster centroids are updated incrementally.
    """
    n = vec.shape[0]
    centers: List[np.ndarray] = []
    clusters: List[List[int]] = []
    for i in range(n):
        best, best_sim = -1, -1.0
        for c, cen in enumerate(centers):
            s = float(vec[i] @ cen)
            if s > best_sim:
                best_sim, best = s, c
        if best >= 0 and best_sim >= threshold:
            clusters[best].append(i)
            centers[best] = (centers[best] * (len(clusters[best]) - 1)
                             + vec[i]) / len(clusters[best])
            centers[best] /= np.linalg.norm(centers[best]) + 1e-12
        else:
            centers.append(vec[i].copy())
            clusters.append([i])
    return clusters


def assign_clusters(clusters: List[List[int]], targets: Dict[str, int]) -> Dict[str, List[int]]:
    """Whole-cluster greedy assignment that fills UP TO the per-split target row counts
    and stops; leftover clusters are not used (the experiment consumes only a semantically
    safe subsample of the pool). A cluster is NEVER split across splits.

    Largest clusters first (so near-dup groups stay together within one split); a cluster
    goes to a split it fits in without overshooting (picking the one with most remaining
    capacity to balance), otherwise to the split it overshoots least.
    """
    order = sorted(((len(c), i) for i, c in enumerate(clusters)), reverse=True)
    counts = {k: 0 for k in targets}
    assigned: Dict[str, List[List[int]]] = {k: [] for k in targets}
    for size, ci in order:
        c = clusters[ci]
        if size <= 0:
            continue
        fits = [k for k in targets if counts[k] + size <= targets[k]]
        if fits:
            k = max(fits, key=lambda k: targets[k] - counts[k])   # balance remaining
        else:
            k = min(targets, key=lambda k: (counts[k] + size) - targets[k])
        counts[k] += size
        assigned[k].append(c)
        if all(counts[k] >= targets[k] for k in targets):
            break
    return {k: [i for c in assigned[k] for i in c] for k in targets}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="/path/to/lat_base/data/harmful-dataset")
    p.add_argument("--out", default="./semantic_split.json")
    p.add_argument("--n_train", type=int, default=1024)
    p.add_argument("--n_val", type=int, default=512)
    p.add_argument("--n_test", type=int, default=512)
    p.add_argument("--sim_threshold", type=float, default=0.8,
                   help="cosine similarity above which two rows are treated as near-dup")
    p.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    a = p.parse_args()

    prompts = load_pool(a.dataset)
    pool = len(prompts)
    print(f"[semantic] pool={pool} prompts; threshold={a.sim_threshold}")

    vec = embed_prompts(prompts, a.embedder)

    clusters = cluster_greedy(vec, a.sim_threshold)
    sizes = sorted((len(c) for c in clusters), reverse=True)
    print(f"[semantic] {len(clusters)} clusters; sizes max={sizes[0] if sizes else 0} "
          f"min={sizes[-1] if sizes else 0} median={int(np.median(sizes)) if sizes else 0}")

    targets = {"train": a.n_train, "val": a.n_val, "test": a.n_test}
    split = assign_clusters(clusters, targets)
    for k, idx in split.items():
        print(f"[semantic] {k}: {len(idx)} rows (target {targets[k]})")

    # --- integrity checks: disjoint, in-range, cluster never straddles a split ---
    all_idx = set(range(pool))
    seen: set[int] = set()
    for k, idx in split.items():
        s = set(idx)
        assert s <= all_idx, f"{k} out of range"
        assert not (s & seen), f"{k} overlaps another split"
        seen |= s
    cluster_of: Dict[int, int] = {}
    for ci, c in enumerate(clusters):
        for i in c:
            cluster_of[i] = ci
    split_of = {}
    for k, idx in split.items():
        for i in idx:
            split_of[i] = k
    for ci, c in enumerate(clusters):
        dests = {split_of[i] for i in c if i in split_of}
        if len(dests) > 1:
            raise RuntimeError(f"cluster {ci} size {len(c)} straddles splits {dests}")
    print(f"[semantic] checks OK: {pool} rows, {len(seen)} placed, "
          f"all clusters contained within one split")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    payload = {
        "pool_size": pool,
        "sim_threshold": a.sim_threshold,
        "embedder": a.embedder,
        "n_clusters": len(clusters),
        "max_cluster_size": int(sizes[0]) if sizes else 0,
        "median_cluster_size": int(np.median(sizes)) if sizes else 0,
        "cluster_sizes": [int(s) for s in sizes],
        "split": split,
    }
    with open(a.out, "w") as f:
        json.dump(payload, f)
    print(f"[semantic] wrote {a.out}")


if __name__ == "__main__":
    main()

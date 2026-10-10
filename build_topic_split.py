#!/usr/bin/env python3
"""Build topic clusters over the LAT harmful-dataset prompt pool via TF-IDF + KMeans.

Offline, no external model. Output:
  topic_clusters.json   {cluster_id: [pool_index, ...]}  (in pool order)
  topic_split_candidates.json  a few "narrow-train vs held-out-test" split proposals
"""
import argparse
import json
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_json", default="/path/to/lat/toward0_5layer_20ep/_dataset_export.jsonl")
    ap.add_argument("--n_topics", type=int, default=30)
    ap.add_argument("--out_dir", default="/path/to/lat/topic_split")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = [json.loads(l) for l in open(args.dataset_json, encoding="utf-8") if l.strip()]
    prompts = [str(r["prompt"]) for r in rows]
    print(f"[topic] pool={len(prompts)}", flush=True)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.7,
                          stop_words="english", sublinear_tf=True)
    X = vec.fit_transform(prompts)
    print(f"[topic] tfidf dims={X.shape}", flush=True)

    km = KMeans(n_clusters=args.n_topics, n_init=8, random_state=0).fit(X)
    labels = km.labels_

    clusters = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(i)
    sizes = sorted((len(v), k) for k, v in clusters.items())
    print("[topic] cluster sizes:", [s for s, _ in sizes], flush=True)

    # save clusters (sorted by size desc)
    out = {}
    for rank, (size, k) in enumerate(sorted(sizes, reverse=True)):
        out[rank] = clusters[k]
    with open(os.path.join(args.out_dir, "topic_clusters.json"), "w", encoding="utf-8") as f:
        json.dump(out, f)

    # show top-3 keywords per cluster for readability
    terms = vec.get_feature_names_out()
    centers = km.cluster_centers_
    print("[topic] per-cluster top keywords:")
    for rank in range(args.n_topics):
        cid = sorted(sizes, reverse=True)[rank][1]
        top = centers[cid].argsort()[-5:][::-1]
        kws = [terms[i] for i in top]
        print(f"  cluster {rank}: n={len(out[rank])}  {kws}", flush=True)

    # proposal: train = largest cluster only, test = all others (held-out topics)
    largest = out[0]
    held_out = [i for i in range(len(prompts)) if i not in set(largest)]
    proposal = {
        "n_topics": args.n_topics,
        "train_topic": 0,
        "train_indices": largest,
        "heldout_indices": held_out,
        "train_n": len(largest),
        "heldout_n": len(held_out),
    }
    with open(os.path.join(args.out_dir, "topic_split_candidates.json"), "w", encoding="utf-8") as f:
        json.dump(proposal, f, indent=2)
    print(f"[topic] proposal: train={len(largest)} heldout={len(held_out)} -> saved", flush=True)


if __name__ == "__main__":
    main()

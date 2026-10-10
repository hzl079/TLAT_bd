#!/usr/bin/env python3
"""Adaptive experiment decider for the weekend queue.

Called by weekend_queue.sh as `python3 queue_decider.py next`:
  1. scans completed run dirs (their history.json exists with the final epoch),
  2. classifies each run's outcome from the post-decay trajectory,
  3. writes weekend_queue/state.json + a per-run report,
  4. prints the next config as  `NAME|ARGS`  or  `STOP`.
"""
import json, os, sys

LAT = "/path/to/lat"
QUEUE = os.path.join(LAT, "weekend_queue")
STATE = os.path.join(QUEUE, "state.json")

BASE = ("--pgd_layers embedding --pgd_iterations 8 --eval_pgd_iterations 8 "
        "--n_train 1024 --benign_dataset LLM-LAT/benign-dataset --benign_n_train 2048 "
        "--eval_eps_multipliers 1.0,2.0,3.0 --inner_lr 0.1 --wandb_project lat-ro-benign")

# name -> (epsilon, lr_decay_epoch, epochs, pgd_layers)
META = {
    "ro_embonly_eps12_benign_e40_fix": (1.2, 20, 40, "embedding"),  # baseline (R1)
}

POOL = [
    ("ro_embonly_eps10_benign_e40",   f"{BASE} --epsilon 1.0 --epochs 40 --lr_decay_epoch 20 --benign_sft_coef 1.0"),
    ("ro_embonly_eps15_benign_e40",   f"{BASE} --epsilon 1.5 --epochs 40 --lr_decay_epoch 20 --benign_sft_coef 1.0"),
    ("ro_embonly_eps20_benign_e40",   f"{BASE} --epsilon 2.0 --epochs 40 --lr_decay_epoch 20 --benign_sft_coef 1.0"),
    ("ro_embonly_eps25_benign_e40",   f"{BASE} --epsilon 2.5 --epochs 40 --lr_decay_epoch 20 --benign_sft_coef 1.0"),
    ("ro_layer3_eps15_benign_e40",    f"{BASE} --epsilon 1.5 --pgd_layers 3 --epochs 40 --lr_decay_epoch 20 --benign_sft_coef 1.0"),
    ("ro_embonly_eps15_benign_sft2",  f"{BASE} --epsilon 1.5 --epochs 40 --lr_decay_epoch 20 --benign_sft_coef 2.0"),
    ("ro_embonly_eps15_benign_sft05", f"{BASE} --epsilon 1.5 --epochs 40 --lr_decay_epoch 20 --benign_sft_coef 0.5"),
    ("ro_embonly_eps15_pgd16_e40",    f"{BASE} --epsilon 1.5 --pgd_iterations 16 --eval_pgd_iterations 16 --epochs 40 --lr_decay_epoch 20 --benign_sft_coef 1.0"),
    ("ro_embonly_eps15_e60",          f"{BASE} --epsilon 1.5 --epochs 60 --lr_decay_epoch 30 --benign_sft_coef 1.0"),
]


def eps_from_name(name):
    import re
    m = re.search(r"eps(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else None


def make_refine(eps):
    return (f"ro_embonly_eps{eps:g}_benign_e60",
            f"{BASE} --epsilon {eps:g} --epochs 60 --lr_decay_epoch 30 --benign_sft_coef 1.0")


def load_run(name):
    """Return (config, history) if the run is complete, else None."""
    hp = os.path.join(LAT, name, "history.json")
    if not os.path.exists(hp):
        return None
    try:
        obj = json.load(open(hp))
    except Exception:
        return None
    hist = obj["history"] if isinstance(obj, dict) and "history" in obj else obj
    cfg = obj.get("config", {}) if isinstance(obj, dict) else {}
    if not hist:
        return None
    exp_ep = cfg.get("epochs") or META.get(name, (0, 0, 40, "embedding"))[2]
    if hist[-1].get("epoch") < exp_ep:   # not finished yet
        return None
    return cfg, hist


def avg(recs, k, side=None):
    vals = []
    for r in recs:
        src = r.get(side, {}) if side else r
        v = src.get(k)
        if v is not None:
            vals.append(v)
    return sum(vals) / len(vals) if vals else 0.0


def classify(cfg, hist, name):
    """Outcome of a run from its (post-decay) trajectory."""
    decay = cfg.get("lr_decay_epoch") or (META.get(name, (0, 0, 40, "embedding"))[1])
    post = [h for h in hist if h["epoch"] >= decay] or hist[len(hist) // 2:]
    if len(post) < 3:
        post = hist
    third = max(1, len(post) // 3)
    early, late = post[:third], post[-third:]

    def te(k):  return avg(late, k, "test") - avg(early, k, "test")
    te_asr_early, te_asr_late = avg(early, "robust_asr", "test"), avg(late, "robust_asr", "test")
    tr_asr_late = avg(late, "robust_asr", "train")
    bn_late = avg(late, "refusal_pref", "benign") if any("benign" in h for h in hist) else 0.0
    te_asr_x2_late = avg(late, "robust_asr_x2", "test") if "robust_asr_x2" in late[-1].get("test", {}) else None
    te_asr_x2_early = avg(early, "robust_asr_x2", "test") if te_asr_x2_late is not None else None

    report = {
        "te_asr_early": round(te_asr_early, 4), "te_asr_late": round(te_asr_late, 4),
        "tr_asr_late": round(tr_asr_late, 4), "bn_refusal_late": round(bn_late, 4),
        "te_asr_x2_early": round(te_asr_x2_early, 4) if te_asr_x2_early is not None else None,
        "te_asr_x2_late": round(te_asr_x2_late, 4) if te_asr_x2_late is not None else None,
        "te_margin_late": round(avg(late, "robust_margin", "test"), 3),
        "te_margin_early": round(avg(early, "robust_margin", "test"), 3),
    }

    # RO: test ASR rose late (1x or 2x), train holds, benign not collapsed
    rose_1x = te_asr_late - te_asr_early > 0.05
    rose_2x = (te_asr_x2_late is not None) and (te_asr_x2_late - te_asr_x2_early > 0.05)
    if (rose_1x or rose_2x) and tr_asr_late < 0.05 and bn_late < 0.1:
        outcome = "RO" if rose_1x else "RO_2x_only"
    elif tr_asr_late > 0.1 and te_asr_late > 0.1:
        outcome = "co_degrade"          # attack too strong, both collapse
    elif bn_late > 0.3:
        outcome = "benign_collapse"     # refusal machine crept back
    else:
        outcome = "flat"                # no behavioral change
    report["outcome"] = outcome
    return report


def main():
    os.makedirs(QUEUE, exist_ok=True)
    state = {"done": []}
    if os.path.exists(STATE):
        state = json.load(open(STATE))

    done_names = {d["name"] for d in state["done"]}
    all_runs = [("ro_embonly_eps12_benign_e40_fix", None)] + list(POOL)
    changed = False
    for name, _args in all_runs:
        if name in done_names:
            continue
        # queue-launched runs get a DONE_ marker when they finish (success OR crash):
        # a crash with no valid history must still be marked done so it is never retried
        done_marker = os.path.join(QUEUE, f"DONE_{name}")
        if os.path.exists(done_marker):
            r = load_run(name)
            if r is not None:
                cfg, hist = r
                rep = classify(cfg, hist, name)
            else:
                rep = {"outcome": "crashed"}
            state["done"].append({"name": name, **rep})
            with open(os.path.join(QUEUE, f"{name}.report.json"), "w") as f:
                json.dump({"name": name, **rep}, f, indent=2)
            done_names.add(name)
            changed = True
            continue
        r = load_run(name)   # e.g. the baseline R1 launched outside the queue
        if r is None:
            continue
        cfg, hist = r
        rep = classify(cfg, hist, name)
        state["done"].append({"name": name, **rep})
        with open(os.path.join(QUEUE, f"{name}.report.json"), "w") as f:
            json.dump({"name": name, "config": cfg, **rep}, f, indent=2)
        done_names.add(name)
        changed = True
    if changed:
        json.dump(state, open(STATE, "w"), indent=2)

    done = state["done"]
    # -------- pick the next run --------
    last = done[-1] if done else None

    # Rule 1: RO found -> refine around its epsilon (longer training), before the rest
    if last and last["outcome"] in ("RO", "RO_2x_only"):
        eps = eps_from_name(last["name"])
        if eps is not None:
            for e in (round(eps - 0.2, 1), round(eps + 0.2, 1)):
                if e <= 0:
                    continue
                n, a = make_refine(e)
                if n not in done_names:
                    print(f"{n}|{a}")
                    return

    # Rule 2: co-degrade -> the plain eps sweep is too strong, skip remaining sweep runs
    skip_sweep = bool(last and last["outcome"] == "co_degrade" and "layer3" not in last["name"])
    for n, a in POOL:
        if n in done_names:
            continue
        if skip_sweep and n.startswith("ro_embonly_eps") and "sft" not in n and "pgd16" not in n and "e60" not in n:
            continue
        print(f"{n}|{a}")
        return

    print("STOP")


if __name__ == "__main__":
    main()

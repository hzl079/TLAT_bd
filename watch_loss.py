#!/usr/bin/env python
import json, sys, os
path = sys.argv[1] if len(sys.argv) > 1 else "/path/to/lat/ro_emb_latent_fullpass_30ep/history.json"
if not os.path.exists(path):
    print("waiting for first epoch record...")
    sys.exit(0)
try:
    d = json.load(open(path))
except Exception:
    print("history.json being rewritten, retry...")
    sys.exit(0)
h = d.get("history", [])
print(" ep |  adv_loss   def_obj    te_adv | va_comply va_dgn")
print("-" * 55)
for r in h:
    def fmt(k):
        v = r.get(k)
        return ("%.4f" % v) if isinstance(v, (int, float)) else str(v if v is not None else "-")
    print(" %2d | %s %s %s | %s %s" % (
        r.get("epoch", -1), fmt("train_adv_loss"), fmt("train_def_obj"),
        fmt("test_adv_loss"), fmt("val_benign_compliance"),
        r.get("val_benign_degenerate", "-")))

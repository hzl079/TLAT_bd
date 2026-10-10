import json, time, subprocess
RUN = "/path/to/lat/robust_asr_instruct_1b_ml_lr5e6_epst06_prefill_r3"
while True:
    try:
        s = json.load(open(RUN + "/reclassified/train/summary.json"))
        if len(s) >= 41:
            break
    except Exception:
        pass
    time.sleep(60)
print("train reclass finished, starting probe judge", flush=True)
subprocess.run(["/path/to/conda_env/bin/python", "/path/to/lat/probe_judge.py"])

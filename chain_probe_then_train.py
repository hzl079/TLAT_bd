import subprocess, time, sys
r = subprocess.run(["/path/to/conda_env/bin/python", "/path/to/lat/probe_judge.py"])
print("probe judge exit:", r.returncode, flush=True)
for attempt in range(3):
    r2 = subprocess.run(["/path/to/conda_env/bin/python", "/path/to/lat/reclassify_generations.py", "--run_dir", "/path/to/lat/robust_asr_instruct_1b_ml_lr5e6_epst06_prefill_r3", "--subset", "train", "--workers", "4"])
    print("train reclass exit:", r2.returncode, flush=True)
    if r2.returncode == 0:
        break
    time.sleep(30)
print("CHAIN DONE", flush=True)

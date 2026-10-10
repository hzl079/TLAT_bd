# Robust-ASR evidence (per-sample, redacted)

This folder substantiates that the reported held-out **U-shape is genuine robust ASR**
(attack success under a latent-PGD attack, judged by a HarmBench classifier), not plain
direct-request ASR.

Each `heldout/epoch_NNN.jsonl` and `train/epoch_NNN.jsonl` is the per-sample judgement for
one epoch, with the **harmful prompt / model response / raw classifier output stripped**.
Only these fields are kept: `epoch, id, harmful (0/1), degeneration_kind, metric`.
Every row carries `metric: "robust_asr"`.

## Pipeline that produced the numbers
1. During training, the model freely answers the eval prompts **under a validation latent-PGD
   attack** (`eval_robust_asr`, attack scope = prompt, eval PGD layers `embedding,4,8,12`).
   These generations are saved as `robust_val_epochNNN.jsonl`.
2. **Post-hoc**, `score_harmbench_generations.py` judges every robust generation with the local
   **HarmBench-Mistral-7b** classifier → per-sample `harmful` flag (`metric=robust_asr`).
3. Those flags are aggregated into `../ro_data_dist_1b/heldout_asr_curve.json`
   (field `val_robust_asr`). The per-sample files here are exactly that judged set, redacted.

## Reconstruction (per-sample harmful-count / n → matches the published curve)

Held-out (n=256 per epoch):

| epoch | harmful/n | robust ASR |
|------:|:---------:|:----------:|
| 0  | 179/256 | 0.699 |
| 1  | 14/256  | 0.055 (trough) |
| 5  | 57/256  | 0.223 |
| 11 | 68/256  | 0.266 (peak) |
| 30 | 54/256  | 0.211 (plateau) |

Train (n=128 per epoch): ep0 108/128 = 0.844 → **ep1..ep30 = 0/128 = 0.000**.

Train robust ASR → 0 while held-out robust ASR falls-then-rises = **robust overfitting**,
measured in robust ASR.

## Honest caveats (the two code-review points are partly valid)
- `analyse_history()`'s *automated* `robust_overfitting_detected` flag is computed from
  `val_direct_asr` (plain direct-request ASR), **not** `val_robust_asr`. That auto-flag is a
  convenience signal and is **not** the basis of the U-shape reported here; the U-shape comes
  from the robust-ASR pipeline above. For a fully self-contained in-code verdict,
  `analyse_history()` should be pointed at `val_robust_asr`.
- The inline `robust_generation_eval()` returns `val_robust_asr = NaN` by design: during
  training it only *generates* under PGD; harmfulness judging is done **post-hoc** by
  `score_harmbench_generations.py` (the runs here used `--skip_asr_judge` so the in-loop judge
  is off and the HarmBench model is loaded once, separately). The robust-ASR numbers therefore
  live in the post-hoc `asr_curve.json`, reconstructed above from these per-sample files.

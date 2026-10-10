# Latent Adversarial Training — Robust Overfitting Study

Research code for studying **robust overfitting (RO)** in **Latent Adversarial Training (LAT)**
of instruction-tuned LLMs (Llama-3.2-1B / Llama-2-7B), including latent-PGD attacks,
toward/away defense objectives, benign SFT retain, and generation-based ASR evaluation.

## Layout

- `motivation_verification.py` — main training/eval driver (latent-PGD LAT, robust-ASR curve,
  checkpointing, distributed training). This is the core script.
- `latent_at/` — latent adversarial training library (attack / helper modules).
- `topic_split/` — topic-clustered data splits used for the data-distribution RO experiments
  (`data_dist_split_378.json`, `topic_clusters.json`, `topic_split_candidates.json`).
- `semantic_split_*.json` — near-duplicate-safe IID train/val/test splits.
- Assorted `*.py` — split builders, evaluators (OR-Bench / GCG / StrongReject-style judges),
  diagnostics and probes.

## Notes before running

- **Paths are placeholders.** Absolute paths have been sanitized to `/path/to/...`.
  Point `--model_path`, `--out_dir`, dataset and split paths at your own locations.
- **Credentials via env or a local `.qianfan_api` file** (gitignored). Set
  `QIANFAN_API_KEY` / `QIANFAN_BASE_URL` (or `EXTERNAL_JUDGE_API_KEY`) as environment
  variables; no keys are committed.
- **Data / models / checkpoints are not included** (gitignored). Supply the harmful/benign
  datasets and base model weights separately.

## Typical run (example)

```bash
torchrun --nproc_per_node=4 motivation_verification.py \
  --model_path /path/to/model --lora --param_dtype bfloat16 \
  --out_dir /path/to/out --split_mode shuffle --split_seed 42 \
  --n_train 3924 --n_val 512 --n_test 512 --epochs 20 --batch_size 32 \
  --pgd_layers embedding,8,16,24 --embedding_epsilon 0.045 --latent_epsilon 0.9 \
  --inner_lr 0.05 --pgd_iterations 8 --eval_robust_asr
```

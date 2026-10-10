# Results

Curated, **aggregate-only** results (per-epoch metric curves, summary tables, and plots).
Raw generations (harmful prompt/response pairs) are intentionally **excluded**.

## `ro_data_dist_1b/` — Robust Overfitting (Llama-3.2-1B-Instruct)
Data-distribution RO setup: train on a **single topic cluster** (378 harmful prompts),
evaluate on a **cross-topic held-out** set.

- `heldout_asr_curve.json` — HarmBench held-out robust ASR shows the classic
  **U-shape**: ep0 ≈ 0.70 → ep1 ≈ 0.055 (trough) → ep11 ≈ 0.27 (peak) → ~0.21 plateau.
- `train_asr_curve.json` — train robust ASR: ep0 ≈ 0.84 → **0.0 from ep1 onward**
  (degeneration rate 0 → genuine refusals, not blank output).
- Train→0 while held-out rises = **robust overfitting reproduced**.
- Plots: `robust_overfitting.png`, `ro_curve_data_dist.png`, `ro_loss_*.png`.

## `iid_1b_40ep/` — IID control (Llama-3.2-1B-Instruct)
Same-distribution train/val/test, 40 epochs.
- Held-out robust ASR: 0.71 → ≈0 and **stays flat** (no rebound) → **no classic RO**;
  robustness generalizes. This is the control contrasting the data-distribution U-shape.

## `orbench_hard1k/` and `orbench_full80k/` — Over-refusal (OR-Bench, on the 1B RO checkpoints)
Training induces **severe over-refusal**:
- Full-80k keyword-FPR: base **6.78%** → ep1 **89.3%** → ep5 **50.0%** → ep30 **39.9%**.
- Hard-1k refusal (rule-based proxy): base 48.2% → ep1 99.7% → ep5 94.0% → ep30 90.7%.

## `qianfan_strongreject/` and `strongreject_clean_r3_summary.json`
StrongReject-style scoring via a Qianfan judge (SR ∈ [0,1]). Held-out SR shows a weaker
rebound than the HarmBench judge (judge-dependent). Summary (means / thresholded ASR) only.

> Note: the 7B (Llama-2-7b-chat) full-data run results are not here yet — that run is still
> in progress at the time of this commit.

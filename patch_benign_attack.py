"""Add --benign_attack_coef to motivation_verification.py (symmetric LAT: attack the benign
batch toward its refusal column, defend toward the helpful answer). Default 0 = unchanged."""
p = "/path/to/lat/motivation_verification.py"
s = open(p).read()
assert "benign_attack_coef" not in s

s = s.replace(
    "    benign_sft_coef: float = 1.5 ",
    "    benign_attack_coef: float = 0.0       # >0: attack benign batch toward refusal, defend toward helpful\n"
    "    benign_sft_coef: float = 1.5 ", 1)
s = s.replace(
    '    p.add_argument("--benign_sft_coef", type=float, default=d.benign_sft_coef,',
    '    p.add_argument("--benign_attack_coef", type=float, default=d.benign_attack_coef)\n'
    '    p.add_argument("--benign_sft_coef", type=float, default=d.benign_sft_coef,', 1)
s = s.replace(
    "        benign_sft_coef=a.benign_sft_coef,\n",
    "        benign_sft_coef=a.benign_sft_coef,\n        benign_attack_coef=a.benign_attack_coef,\n", 1)

anchor = '                    d_loss["weighted"] += benign_coef * d_loss["sft"]\n'
assert s.count(anchor) == 1
block = anchor + '''                if sft_batch is not None and cfg.benign_attack_coef > 0:
                    clear_hooks(model)
                    freeze_all(model)
                    _, _, b_adv = run_pgd_attack(
                        model, sft_batch, hidden_size=hidden_size, layers=cfg.pgd_layers,
                        model_layers_module=model_layers_module, epsilon=cfg.epsilon,
                        lr=cfg.inner_lr, iterations=cfg.pgd_iterations, coefs=adv_coefs,
                        embedding_epsilon=cfg.embedding_epsilon,
                        latent_epsilon=cfg.latent_epsilon,
                        attack_completions=cfg.attack_completions, device=cfg.device,
                        clip_grad=cfg.clip_grad)
                    set_requires_grad(trainable_params, True)
                    b_def = defense_backward(model, sft_batch, def_coefs, cfg.device,
                                             scale=cfg.benign_attack_coef)
                    clear_hooks(model)
                    d_loss["benign_adv"] = b_adv.get("total", 0.0)
                    d_loss["benign_def"] = b_def["total"]
                    d_loss["weighted"] += b_def["weighted"]
'''
s = s.replace(anchor, block, 1)
open(p, "w").write(s)
print("patched")

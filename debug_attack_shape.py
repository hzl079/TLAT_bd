"""debug_attack_shape.py — 定位 attacked generation 的 mask 形状问题。"""
import sys, os
REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import torch
import motivation_verification as mv

cfg = mv.Config(dataset='/path/to/lat_base/data/harmful-dataset', truncate_length=384,
                device='cuda', world_size=1, param_dtype='bfloat16')
mv.prepare_data_source(cfg)
model, tok, mll = mv.load_model_and_tokenizer(cfg)
ds = mv.build_dataset(tok, cfg, 'train')
row = ds[0]
col = mv.LatentAdversarialTrainingDataCollator(tok.pad_token_id, truncate_length=384)
b = col([row])
b = {k: v.to('cuda') for k, v in b.items() if torch.is_tensor(v)}
for k, v in b.items():
    print('batch', k, tuple(v.shape), v.dtype)
mv.freeze_all(model)
adv_coefs = mv.normalize_dict({'toward': 0.5, 'away': 0.5})
model.eval()
with torch.enable_grad():
    adversaries, wrappers, _ = mv.run_pgd_attack(
        model, b, hidden_size=2048, layers=list(range(16)), model_layers_module=mll,
        epsilon=4.0, lr=0.05, iterations=2, coefs=adv_coefs,
        attack_completions=False, device='cuda')
a0 = adversaries[0]
print('attack.shape', tuple(a0.attack.shape))
print('attack_mask.shape', tuple(a0.attack_mask.shape))
print('attack_mask[:, :attack_mask.shape[1]] ->', tuple(a0.attack_mask[:, :a0.attack_mask.shape[1]].shape))
pid = torch.tensor([row['prompt_tokens']], device='cuda')
print('prompt_ids', tuple(pid.shape))
model.config.use_cache = True
try:
    with torch.no_grad():
        out = model.generate(pid, max_new_tokens=4, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    print('gen ok', tuple(out.shape))
except Exception as e:
    print('GEN ERROR:', type(e).__name__, str(e)[:300])
    import traceback; traceback.print_exc(limit=2)

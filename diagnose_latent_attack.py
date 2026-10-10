#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

import motivation_verification as mv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dataset_json', required=True)
    p.add_argument('--split_json', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--epsilon', type=float, required=True)
    p.add_argument('--iterations', type=int, required=True)
    p.add_argument('--layer', type=int, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()

    cfg = mv.Config(
        model_path=args.checkpoint,
        dataset_json=args.dataset_json,
        device=args.device,
        rank=0,
        world_size=1,
        epsilon=args.epsilon,
        robust_eval_epsilon=args.epsilon,
        inner_lr=0.05,
        robust_eval_inner_lr=0.05,
        pgd_iterations=args.iterations,
        robust_eval_pgd_iterations=args.iterations,
        pgd_layers=[args.layer],
        robust_eval_pgd_layers=[args.layer],
        eval_robust_asr=True,
        robust_eval_attack_scope='prompt',
        gen_max_new_tokens=128,
        gen_min_new_tokens=0,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model, _, model_layers_module = mv.load_model_and_tokenizer(cfg)
    trainable_params = list(model.parameters())
    mv.freeze_all(model)
    raw_rows = mv.load_raw_rows(cfg)
    split = json.loads(Path(args.split_json).read_text())
    test_indices = list(split.get('split', split)['test'])
    prompt_strs = [raw_rows[i]['prompt'] for i in test_indices]
    adv_coefs = mv.normalize_dict({'toward': 0.5, 'away': 0.5})
    os.makedirs(args.out_dir, exist_ok=True)
    result = mv.robust_generation_eval(
        model, tokenizer, raw_rows, test_indices, prompt_strs, cfg,
        hidden_size=model.config.hidden_size,
        model_layers_module=model_layers_module,
        trainable_params=trainable_params,
        adv_coefs=adv_coefs,
        gen_dir=args.out_dir,
        epoch=1,
    )
    with open(os.path.join(args.out_dir, 'diagnostic_protocol.json'), 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'split': 'test',
            'n_prompts': len(prompt_strs),
            'epsilon': args.epsilon,
            'inner_lr': 0.05,
            'iterations': args.iterations,
            'layer': args.layer,
            'result': result,
        }, f, indent=2)


if __name__ == '__main__':
    main()

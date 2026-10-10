import queue_decider as q

def synth(outcome, decay=20):
    hist=[]
    for ep in range(0,41):
        te_asr = 0.02 if ep<26 else (0.15 if outcome=='RO' else (0.6 if outcome=='co' else 0.02))
        tr_asr = 0.0 if outcome!='co' else 0.5
        bn_ref = 0.01 if outcome!='benign' else 0.5
        x2 = te_asr + 0.05 if outcome=='RO_2x' else te_asr
        hist.append({
            'epoch': ep,
            'train': {'robust_asr': tr_asr},
            'test': {'robust_asr': te_asr, 'robust_margin': 3.0-0.1*ep if outcome=='RO' else 4.0,
                     'robust_asr_x2': x2},
            'benign': {'refusal_pref': bn_ref},
        })
    return q.classify({'lr_decay_epoch': decay}, hist, 'test_run')

for oc in ['RO','RO_2x','co','flat','benign']:
    rep = synth(oc)
    print(f"{oc:8s} -> outcome={rep['outcome']:12s} te_asr_early={rep['te_asr_early']} te_asr_late={rep['te_asr_late']}")

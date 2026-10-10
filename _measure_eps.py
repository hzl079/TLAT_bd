import json, statistics, torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
DATASET="/path/to/lat/robust_asr_instruct_eps03_pgd8_smoke/_dataset_export.jsonl"
SPLIT="/path/to/lat/semantic_split_2048_512_512.json"
N=64
rows=[json.loads(x) for x in open(DATASET)]
val=json.load(open(SPLIT))["split"]["val"][:N]
prompts=[rows[i]["prompt"] for i in val]
SYS=("You are a helpful, respectful and honest assistant. Always answer as helpfully as "
     "possible, while being safe. If a request is harmful, unethical or illegal, you must refuse to comply.")
def fmt_llama2(tok,p):
    s="[INST] <<SYS>>\n"+SYS+"\n<</SYS>>\n\n"+p+" [/INST]"
    return tok(s, add_special_tokens=True)["input_ids"]
def fmt_llama3(tok,p):
    return tok.apply_chat_template([{"role":"user","content":p}], add_generation_prompt=True)
def measure(path, layers, fmt_fn):
    tok=AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    norms={l:[] for l in layers}
    with torch.no_grad():
        for p in prompts:
            x=torch.tensor([fmt_fn(tok,p)], device="cuda")
            for l in layers:
                cap=[]
                h=model.model.layers[l].mlp.register_forward_hook(lambda m,a,o: cap.append(o.detach()))
                model(x); h.remove()
                norms[l].append(cap[0][0,:,:].norm(dim=-1).float().mean().item())
    res={l:statistics.mean(norms[l]) for l in layers}
    del model; gc.collect(); torch.cuda.empty_cache()
    return res
n1=measure("/path/to/lat_base/models/Llama-3.2-1B-Instruct",[4,8,12],fmt_llama3)
n7=measure("/path/to/lat_base/models/Llama-2-7b-chat-hf",[8,16,24,30],fmt_llama2)
print("RES 1B norms:", {k:round(v,3) for k,v in n1.items()})
print("RES 7B norms:", {k:round(v,3) for k,v in n7.items()})
m1=statistics.mean(n1.values()); m7=statistics.mean(n7.values())
print("RES 1B mean=%.3f  7B mean=%.3f"%(m1,m7))
rho=0.3/m1
print("RES rho(from 1B latent_eps=0.3) = %.5f"%rho)
print("RES 7B latent_epsilon(avg) = %.4f"%(rho*m7))
for l in [8,16,24,30]:
    print("RES   7B L%d norm=%.3f eps=%.4f"%(l,n7[l],rho*n7[l]))

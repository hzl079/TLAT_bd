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
    return tok("[INST] <<SYS>>\n"+SYS+"\n<</SYS>>\n\n"+p+" [/INST]", add_special_tokens=True)["input_ids"]
def fmt_llama3(tok,p):
    return tok.apply_chat_template([{"role":"user","content":p}], add_generation_prompt=True)
def measure_emb(path, fmt_fn):
    tok=AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    ns=[]
    with torch.no_grad():
        for p in prompts:
            x=torch.tensor([fmt_fn(tok,p)], device="cuda")
            cap=[]
            h=model.model.embed_tokens.register_forward_hook(lambda m,a,o: cap.append(o.detach()))
            model(x); h.remove()
            ns.append(cap[0][0,:,:].norm(dim=-1).float().mean().item())
    m=statistics.mean(ns)
    del model; gc.collect(); torch.cuda.empty_cache()
    return m
e1=measure_emb("/path/to/lat_base/models/Llama-3.2-1B-Instruct",fmt_llama3)
e7=measure_emb("/path/to/lat_base/models/Llama-2-7b-chat-hf",fmt_llama2)
print("RES 1B embed norm = %.4f"%e1)
print("RES 7B embed norm = %.4f"%e7)
rho=0.05/e1
print("RES rho_emb(from 1B emb_eps=0.05) = %.5f"%rho)
print("RES => 7B embedding_epsilon = %.4f"%(rho*e7))

import glob, json
ROOT = "scratch/evals/ocean/dpo_sft_mix_a_nminus/dpo_sft_mix_a_nminus"
def suite(model, kind):
    logs = sorted(glob.glob(f"{ROOT}/{model}/{kind}/native/inspect_logs/*.json"))
    if not logs: return {}
    d = json.load(open(logs[-1]))
    out = {}
    for sc in d.get("results", {}).get("scores", []):
        for k,v in sc.get("metrics", {}).items(): out[k]=v.get("value")
    return out
def row(model):
    t=suite(model,"trait_logprobs"); u=suite(model,"mmlu")
    f=lambda x: f"{x:.3f}" if isinstance(x,(int,float)) else "  -  "
    print(f"{model:22s} A={f(t.get('Agreeableness')):>6}  N={f(t.get('Neuroticism')):>6}  MMLU={f(u.get('accuracy')):>6}")
row("base")
for m in ["dpo","sft0p25","sft0p50","sft0p75","sft"]: row(f"a_{m}")
for m in ["dpo","sft0p25","sft0p50","sft0p75","sft"]: row(f"nminus_{m}")
for m in ["dpo","sft0p25","sft0p50","sft0p75","sft"]: row(f"soup_a_nminus_{m}")

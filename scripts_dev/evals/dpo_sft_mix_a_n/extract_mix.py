import glob, json
ROOT = "scratch/evals/ocean/dpo_sft_mix_a_n/dpo_sft_mix_a_n"
def suite(model, kind):
    logs = sorted(glob.glob(f"{ROOT}/{model}/{kind}/native/inspect_logs/*.json"))
    if not logs: return {}
    d = json.load(open(logs[-1]))
    out = {}
    for sc in d.get("results", {}).get("scores", []):
        for k,v in sc.get("metrics", {}).items():
            out[k] = v.get("value")
    return out
def row(model):
    t = suite(model, "trait_logprobs"); u = suite(model, "mmlu")
    A = t.get("Agreeableness"); N = t.get("Neuroticism"); acc = u.get("accuracy")
    f = lambda x: f"{x:.3f}" if isinstance(x,(int,float)) else "  -  "
    return f"{model:18s} A={f(A):>6}  N={f(N):>6}  MMLU={f(acc):>6}"
print("=== base ==="); print(row("base"))
print("=== A mixes (m = SFT frac) ===")
for m in ["dpo","sft0p25","sft0p50","sft0p75","sft"]: print(row(f"a_{m}"))
print("=== N mixes ===")
for m in ["dpo","sft0p25","sft0p50","sft0p75","sft"]: print(row(f"n_{m}"))
print("=== A+N matched soups ===")
for m in ["dpo","sft0p25","sft0p50","sft0p75","sft"]: print(row(f"soup_a_n_{m}"))

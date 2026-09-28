# ==================================================================================================
# AKBASCORE · TEST 273
# SELF-FORGED DYNAMIC LATENT PACKET
# STRING → MATCHED COUNTERFACTUAL → LAYER-LOCAL LATENT PACKET
# NO FIXED SEMANTIC AXES · NO SOURCE BASIS · NO STEERING · NO TRAINING
# ==================================================================================================
import os,sys,json,time,random,hashlib,subprocess,importlib.util
from datetime import datetime,timezone
from pathlib import Path
for m,p in [("torch","torch"),("transformers","transformers"),("numpy","numpy")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
SEED=273;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda");MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL=28;H_EXPECT=3584;EPS=1e-8;MAX_NEW=64
ROOT=Path("/content/AKBASCORE_TEST273") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST273");ROOT.mkdir(parents=True,exist_ok=True)
START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
TARGETS={
"FICTION":{"correct":"Elena Voss stored the cobalt prism beneath the Ardent Observatory.","control":"Elena Voss stored the amber prism beneath the Ardent Observatory.",
"queries":["What did Elena Voss store beneath the Ardent Observatory?","Which object was stored beneath the Ardent Observatory?","Name the item Elena Voss stored.","What object is associated with Elena Voss beneath the Ardent Observatory?"],"expected":"cobalt prism","control_expected":"amber prism"},
"SCIENCE":{"correct":"The synthetic alloy Velorium reaches superconductivity at 173 kelvin.","control":"The synthetic alloy Velorium reaches superconductivity at 241 kelvin.",
"queries":["At what temperature does Velorium reach superconductivity?","What is Velorium's superconducting temperature?","Give the temperature at which Velorium becomes superconducting.","Which temperature is associated with superconductivity in Velorium?"],"expected":"173 kelvin","control_expected":"241 kelvin"},
"TEMPORAL":{"correct":"The Orpheus probe entered lunar orbit before the Selene probe entered lunar orbit.","control":"The Selene probe entered lunar orbit before the Orpheus probe entered lunar orbit.",
"queries":["Which probe entered lunar orbit first?","Which probe preceded the other into lunar orbit?","Name the probe that entered lunar orbit earlier.","Between Orpheus and Selene, which entered lunar orbit before the other?"],"expected":"Orpheus probe","control_expected":"Selene probe"},
"TECHNICAL":{"correct":"Protocol ZX-41 assigns channel seven to the thermal calibration stream.","control":"Protocol ZX-41 assigns channel three to the thermal calibration stream.",
"queries":["Which channel does Protocol ZX-41 assign to the thermal calibration stream?","What channel is assigned to the thermal calibration stream by ZX-41?","Give the ZX-41 channel for the thermal calibration stream.","Under Protocol ZX-41, the thermal calibration stream uses which channel?"],"expected":"channel seven","control_expected":"channel three"}}
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(EPS))
def canon(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
print("="*110);print("TEST 273 — SELF-FORGED DYNAMIC LATENT PACKET");print("="*110);print("START:",START)
print("[1/8] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
torch.cuda.synchronize();t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;PDT=next(model.parameters()).dtype;torch.cuda.synchronize()
if len(layers)!=TOTAL or H!=H_EXPECT or PDT!=torch.bfloat16:raise RuntimeError("Architecture/dtype mismatch.")
print(f"OK · {MODEL_ID} · 28L · H={H} · {PDT} · {time.perf_counter()-t:.2f}s")
print("[2/8] WEIGHT SENTINEL")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();print("FP:",[f"{x:.4f}" for x in FP0])
def enc_text(x):
    s=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to(DEVICE)
def enc_info(f,q):return enc_text(f"Information: {f}\n\nQuestion: {q}")
@torch.inference_mode()
def capture(e):
    o=model(**e,use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
@torch.inference_mode()
def generate(f,q):
    e=enc_info(f,q);n=e.input_ids.shape[1]
    o=model.generate(**e,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
    return tok.decode(o[0,n:],skip_special_tokens=True).strip()
print("[3/8] CORRECT / MATCHED-CONTROL CAPTURE")
XR={};BEHAV={}
for name,d in TARGETS.items():
    XR[name]=[];BEHAV[name]=[]
    for i,q in enumerate(d["queries"]):
        C=capture(enc_info(d["correct"],q));M=capture(enc_info(d["control"],q));XR[name].append({"CORRECT":C,"CONTROL":M})
        bc=generate(d["correct"],q);bm=generate(d["control"],q);BEHAV[name].append({"query":q,"correct":bc,"control":bm})
        print(f"{name:9s} Q{i+1} · C={bc} | M={bm}")
print("[4/8] SELF-FORGE · QUERY DELTAS → DOMAIN PACKET")
PACKETS={};FORGE={}
for name,d in TARGETS.items():
    PACKETS[name]=[];FORGE[name]=[]
    for L in range(TOTAL):
        D=torch.stack([XR[name][i]["CORRECT"][L]-XR[name][i]["CONTROL"][L] for i in range(len(d["queries"]))])
        U=torch.stack([unit(x) for x in D]);_,S,Vh=torch.linalg.svd(U,full_matrices=False);var=(S*S)/(S*S).sum().clamp_min(EPS)
        mean=U.mean(0);packet=unit(mean);PACKETS[name].append(packet)
        pair=[cos(U[i],U[j]) for i in range(U.shape[0]) for j in range(i+1,U.shape[0])]
        FORGE[name].append({"raw_norm_mean":float(D.norm(dim=1).mean()),"mean_pair_cos":float(np.mean(pair)),"pc1":float(var[0]),"pc12":float(var[:2].sum()),"packet_query_cos":[cos(packet,U[i]) for i in range(U.shape[0])]})
for L in [0,5,10,15,19,21,23,25,26,27]:
    print(f"L{L:02d} · "+" ".join(f"{n}:PAIR={FORGE[n][L]['mean_pair_cos']:+.3f}/PC1={FORGE[n][L]['pc1']*100:.1f}%" for n in TARGETS))
print("[5/8] LEAVE-ONE-QUERY-OUT SELF-FORGE")
LOO={}
for name,d in TARGETS.items():
    LOO[name]=[]
    for L in range(TOTAL):
        U=[unit(XR[name][i]["CORRECT"][L]-XR[name][i]["CONTROL"][L]) for i in range(len(d["queries"]))]
        vals=[]
        for j in range(len(U)):
            p=unit(torch.stack([U[i] for i in range(len(U)) if i!=j]).mean(0));vals.append(cos(p,U[j]))
        LOO[name].append({"cosines":vals,"mean":float(np.mean(vals)),"min":float(np.min(vals))})
for L in [19,21,23,25,26,27]:
    print(f"L{L:02d} LOO · "+" ".join(f"{n}={LOO[n][L]['mean']:+.3f}" for n in TARGETS))
print("[6/8] PACKET RECONSTRUCTION")
RECON={}
for name,d in TARGETS.items():
    RECON[name]=[]
    for L in range(TOTAL):
        p=PACKETS[name][L];rows=[]
        for i in range(len(d["queries"])):
            D=XR[name][i]["CORRECT"][L]-XR[name][i]["CONTROL"][L];alpha=float(torch.dot(D,p));E=alpha*p;R=D-E
            rows.append({"cos":cos(D,p),"signed_alpha":alpha,"explained_energy":float(torch.dot(E,E)/torch.dot(D,D).clamp_min(EPS)),"residual_energy":float(torch.dot(R,R)/torch.dot(D,D).clamp_min(EPS))})
        RECON[name].append({"cos_mean":float(np.mean([x["cos"] for x in rows])),"explained_mean":float(np.mean([x["explained_energy"] for x in rows])),"residual_mean":float(np.mean([x["residual_energy"] for x in rows])),"queries":rows})
for L in [19,21,23,25,26,27]:
    print(f"L{L:02d} RECON · "+" ".join(f"{n}={RECON[n][L]['explained_mean']*100:.1f}%" for n in TARGETS))
print("[7/8] CROSS-DOMAIN PACKET GEOMETRY")
CROSS={}
names=list(TARGETS)
for L in range(TOTAL):
    C=torch.stack([PACKETS[n][L] for n in names]);M=torch.empty((len(names),len(names)))
    for i in range(len(names)):
        for j in range(len(names)):M[i,j]=cos(C[i],C[j])
    off=[float(M[i,j]) for i in range(len(names)) for j in range(i+1,len(names))]
    _,S,Vh=torch.linalg.svd(C,full_matrices=False);var=(S*S)/(S*S).sum().clamp_min(EPS)
    CROSS[f"L{L:02d}"]={"matrix":M.tolist(),"mean_offdiag":float(np.mean(off)),"max_abs_offdiag":float(np.max(np.abs(off))),"pc1":float(var[0]),"pc12":float(var[:2].sum())}
for L in [19,21,23,25,26,27]:
    x=CROSS[f"L{L:02d}"];print(f"L{L:02d} DOMAIN-PACKET pair={x['mean_offdiag']:+.4f} max|cos|={x['max_abs_offdiag']:.4f} PC1={x['pc1']*100:.2f}%")
print("[8/8] INTEGRITY + SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure.")
SUMMARY={}
for name in TARGETS:
    SUMMARY[name]={}
    for L in [19,21,23,25,26,27]:
        SUMMARY[name][f"L{L:02d}"]={"pair":FORGE[name][L]["mean_pair_cos"],"pc1":FORGE[name][L]["pc1"],"loo":LOO[name][L]["mean"],"reconstruction":RECON[name][L]["explained_mean"]}
    x=SUMMARY[name]["L27"];print(f"{name:9s} L27 · PAIR={x['pair']:+.4f} · PC1={x['pc1']*100:.2f}% · LOO={x['loo']:+.4f} · RECON={x['reconstruction']*100:.2f}%")
R={"schema":"akbascore.test273.v1","test":"TEST 273","start":START,"end":utc(),"model":MODEL_ID,
"question":"Can the same target-independent mathematical forge dynamically construct a stable layer-local latent packet from each new text without fixed semantic axes or a shared source basis?",
"targets":TARGETS,"method":"For each independent domain and query, compute layer-local CORRECT-minus-matched-CONTROL hidden-state delta. Normalize query deltas and self-forge a domain packet from their mean. Measure within-domain agreement, SVD concentration, leave-one-query-out transfer, one-dimensional packet reconstruction and cross-domain packet geometry. No fixed S/A/O/L/B axes and no source residual basis.",
"behavior":BEHAV,"forge":FORGE,"leave_one_query_out":LOO,"reconstruction":RECON,"cross_domain":CROSS,"summary":SUMMARY,
"integrity":{"fixed_semantic_axes":False,"source_basis":False,"steering":False,"training":False,"weight_update":False,"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();run=f"T273-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{run}.json";tp=ROOT/f"{run}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=False,sort_keys=True,indent=2).encode())
o=["="*110,"TEST 273 — SELF-FORGED DYNAMIC LATENT PACKET","="*110]
for n in TARGETS:
    o+=["",f"[{n}] {TARGETS[n]['correct']}",f"EXPECTED: {TARGETS[n]['expected']}"]
    for L in [19,21,23,25,26,27]:
        x=SUMMARY[n][f"L{L:02d}"];o.append(f"L{L:02d} PAIR={x['pair']:+.6f} PC1={x['pc1']*100:.3f}% LOO={x['loo']:+.6f} RECON={x['reconstruction']*100:.3f}%")
o+=["","NO FIXED SEMANTIC AXES · NO SOURCE BASIS · NO STEERING · NO TRAINING","WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(o),encoding="utf-8")
print("="*110);print("TEST 273 COMPLETE")
print("STRING → MATCHED COUNTERFACTUAL → SELF-FORGED LAYER-LOCAL LATENT PACKET")
print("NO FIXED AXES · NO SOURCE BASIS · NO STEERING · NO TRAINING · WEIGHT INTEGRITY PASS")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*110)

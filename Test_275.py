# ==================================================================================================
# AKBASCORE · TEST 275
# STATE-CONDITIONAL LOCAL TRANSPORT CALIBRATION
# TEST273 SELF-FORGED PACKET · NATURAL vs BLIND · ±EPS CENTRAL DIFFERENCE
# NO RETRIEVAL STEERING · NO TRAINING · WEIGHTS FROZEN
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
SEED=275;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda");MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL=28;H_EXPECT=3584;EPS=1e-8;PROBE_REL=.0025;SRC=[3,6,10,14,18,22,25];DST=[1,2,3]
ROOT=Path("/content/AKBASCORE_TEST275") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST275");ROOT.mkdir(parents=True,exist_ok=True)
START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
TARGETS={
"FICTION":{"correct":"Elena Voss stored the cobalt prism beneath the Ardent Observatory.","control":"Elena Voss stored the amber prism beneath the Ardent Observatory.",
"forge":["What did Elena Voss store beneath the Ardent Observatory?","Which object was stored beneath the Ardent Observatory?","Name the item Elena Voss stored.","What object is associated with Elena Voss beneath the Ardent Observatory?"],"blind":"What did Elena Voss store beneath the Ardent Observatory?"},
"SCIENCE":{"correct":"The synthetic alloy Velorium reaches superconductivity at 173 kelvin.","control":"The synthetic alloy Velorium reaches superconductivity at 241 kelvin.",
"forge":["At what temperature does Velorium reach superconductivity?","What is Velorium's superconducting temperature?","Give the temperature at which Velorium becomes superconducting.","Which temperature is associated with superconductivity in Velorium?"],"blind":"At what temperature does Velorium reach superconductivity?"},
"TEMPORAL":{"correct":"The Orpheus probe entered lunar orbit before the Selene probe entered lunar orbit.","control":"The Selene probe entered lunar orbit before the Orpheus probe entered lunar orbit.",
"forge":["Which probe entered lunar orbit first?","Which probe preceded the other into lunar orbit?","Name the probe that entered lunar orbit earlier.","Between Orpheus and Selene, which entered lunar orbit before the other?"],"blind":"Which probe entered lunar orbit first?"},
"TECHNICAL":{"correct":"Protocol ZX-41 assigns channel seven to the thermal calibration stream.","control":"Protocol ZX-41 assigns channel three to the thermal calibration stream.",
"forge":["Which channel does Protocol ZX-41 assign to the thermal calibration stream?","What channel is assigned to the thermal calibration stream by ZX-41?","Give the ZX-41 channel for the thermal calibration stream.","Under Protocol ZX-41, the thermal calibration stream uses which channel?"],"blind":"Which channel does Protocol ZX-41 assign to the thermal calibration stream?"}}
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(EPS))
def canon(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
print("="*110);print("TEST 275 — STATE-CONDITIONAL LOCAL TRANSPORT CALIBRATION");print("="*110);print("START:",START)
print("[1/9] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
torch.cuda.synchronize();t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;PDT=next(model.parameters()).dtype;torch.cuda.synchronize()
if len(layers)!=TOTAL or H!=H_EXPECT or PDT!=torch.bfloat16:raise RuntimeError("Architecture/dtype mismatch.")
print(f"OK · {MODEL_ID} · 28L · H={H} · {PDT} · {time.perf_counter()-t:.2f}s")
print("[2/9] WEIGHT SENTINEL")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();print("FP:",[f"{x:.4f}" for x in FP0]);print(f"PROBE_REL={PROBE_REL:.6f} · SOURCES={SRC} · HORIZONS={DST}")
def enc_text(x):
    s=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to(DEVICE)
def enc_info(f,q):return enc_text(f"Information: {f}\n\nQuestion: {q}")
@torch.inference_mode()
def capture(e):
    o=model(**e,use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
print("[3/9] TEST273 SELF-FORGE")
PACKETS={};FORGE={}
for name,d in TARGETS.items():
    delta=[[] for _ in range(TOTAL)]
    for q in d["forge"]:
        C=capture(enc_info(d["correct"],q));M=capture(enc_info(d["control"],q))
        for L in range(TOTAL):delta[L].append(C[L]-M[L])
    PACKETS[name]=[];FORGE[name]=[]
    for L in range(TOTAL):
        U=torch.stack([unit(x) for x in delta[L]]);p=unit(U.mean(0));PACKETS[name].append(p)
        pair=[cos(U[i],U[j]) for i in range(4) for j in range(i+1,4)]
        _,S,_=torch.linalg.svd(U,full_matrices=False);v=(S*S)/(S*S).sum().clamp_min(EPS)
        FORGE[name].append({"pair":float(np.mean(pair)),"pc1":float(v[0])})
    print(f"{name:9s} L27 · PAIR={FORGE[name][27]['pair']:+.4f} · PC1={FORGE[name][27]['pc1']*100:.2f}%")
print("[4/9] NATURAL / BLIND BASELINES")
BASE={}
for name,d in TARGETS.items():
    BASE[name]={"NATURAL":capture(enc_info(d["correct"],d["blind"])),"BLIND":capture(enc_text(d["blind"]))}
    n=BASE[name]["NATURAL"][27]-BASE[name]["BLIND"][27]
    print(f"{name:9s} L27 NATURAL−BLIND norm={n.norm():.3f}")
print("[5/9] ±EPS LOCAL PROBE ENGINE")
AUDIT={"calls":0,"max_probe_deviation":0.0};HANDLES=[]
def remove():
    global HANDLES
    for h in HANDLES:h.remove()
    HANDLES=[]
@torch.inference_mode()
def probe(e,src,p,sign):
    global HANDLES
    remove()
    def hook(mod,args,out):
        y=out[0] if isinstance(out,tuple) else out;z=y[:,-1,:].float();d=p.view(1,-1)*z.norm(dim=-1,keepdim=True)*PROBE_REL*sign
        rel=float(d.norm()/z.norm().clamp_min(EPS));AUDIT["calls"]+=1;AUDIT["max_probe_deviation"]=max(AUDIT["max_probe_deviation"],abs(rel-PROBE_REL))
        yy=y.clone();yy[:,-1,:]=(z+d).to(y.dtype)
        if isinstance(out,tuple):return (yy,)+out[1:]
        return yy
    HANDLES=[layers[src].register_forward_hook(hook)]
    try:return capture(e)
    finally:remove()
print("ENGINE READY · one source layer per assay · central difference")
print("[6/9] NATURAL vs BLIND TRANSPORT")
RESULT={}
for name,d in TARGETS.items():
    RESULT[name]={}
    EN=enc_info(d["correct"],d["blind"]);EB=enc_text(d["blind"])
    for s in SRC:
        NP=probe(EN,s,PACKETS[name][s],+1);NM=probe(EN,s,PACKETS[name][s],-1)
        BP=probe(EB,s,PACKETS[name][s],+1);BM=probe(EB,s,PACKETS[name][s],-1)
        RESULT[name][f"L{s:02d}"]={}
        for h in DST:
            dst=s+h
            if dst>=TOTAL:continue
            rn=(NP[dst]-NM[dst])/(2*PROBE_REL);rb=(BP[dst]-BM[dst])/(2*PROBE_REL)
            pn=PACKETS[name][dst];baseN=BASE[name]["NATURAL"][dst];baseB=BASE[name]["BLIND"][dst]
            RESULT[name][f"L{s:02d}"][f"H{h}"]={"dst":dst,"transport_cos":cos(rn,rb),"natural_gain":float(rn.norm()/baseN.norm().clamp_min(EPS)),
            "blind_gain":float(rb.norm()/baseB.norm().clamp_min(EPS)),"gain_ratio":float(rb.norm()/rn.norm().clamp_min(EPS)),
            "natural_packet_cos":cos(rn,pn),"blind_packet_cos":cos(rb,pn),"natural_blind_delta_cos":cos(rb,baseN-baseB)}
    print(name,"DONE")
print("[7/9] TRANSPORT MATRIX")
for s in SRC:
    for h in DST:
        if s+h>=TOTAL:continue
        vals=[RESULT[n][f"L{s:02d}"][f"H{h}"]["transport_cos"] for n in TARGETS]
        ratios=[RESULT[n][f"L{s:02d}"][f"H{h}"]["gain_ratio"] for n in TARGETS]
        print(f"L{s:02d}→L{s+h:02d} · N↔B cos={np.mean(vals):+.4f} · blind/natural gain={np.mean(ratios):.3f} · "+" ".join(f"{n}={RESULT[n][f'L{s:02d}'][f'H{h}']['transport_cos']:+.2f}" for n in TARGETS))
print("[8/9] DOMAIN + LATE-LAYER SUMMARY")
SUMMARY={}
for name in TARGETS:
    rows=[]
    for s in SRC:
        for h in DST:
            if s+h<TOTAL:rows.append(RESULT[name][f"L{s:02d}"][f"H{h}"])
    SUMMARY[name]={"transport_cos_mean":float(np.mean([x["transport_cos"] for x in rows])),"gain_ratio_mean":float(np.mean([x["gain_ratio"] for x in rows])),
    "natural_packet_cos_mean":float(np.mean([x["natural_packet_cos"] for x in rows])),"blind_packet_cos_mean":float(np.mean([x["blind_packet_cos"] for x in rows]))}
    late=[RESULT[name][f"L{s:02d}"][f"H{h}"] for s in SRC if s>=18 for h in DST if s+h<TOTAL]
    SUMMARY[name]["late_transport_cos"]=float(np.mean([x["transport_cos"] for x in late]));SUMMARY[name]["late_gain_ratio"]=float(np.mean([x["gain_ratio"] for x in late]))
    x=SUMMARY[name];print(f"{name:9s} ALL cos={x['transport_cos_mean']:+.4f} gain={x['gain_ratio_mean']:.3f} · LATE cos={x['late_transport_cos']:+.4f} gain={x['late_gain_ratio']:.3f}")
print("[9/9] INTEGRITY + SEAL")
remove()
if HANDLES:raise RuntimeError("Hook cleanup failure.")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure.")
if AUDIT["max_probe_deviation"]>1e-5:raise RuntimeError("Probe audit failure.")
R={"schema":"akbascore.test275.v1","test":"TEST 275","start":START,"end":utc(),"model":MODEL_ID,
"question":"Does the same self-forged packet undergo different local downstream transport when applied to NATURAL versus BLIND hidden-state trajectories?",
"targets":TARGETS,"probe_rel":PROBE_REL,"source_layers":SRC,"horizons":DST,"forge":FORGE,
"method":"Reproduce TEST273 text-specific packets. At each selected source layer apply symmetric ±0.25% frozen-norm packet probes separately to NATURAL and BLIND prompts. Use central difference response at +1,+2,+3 downstream layers. Compare response direction, gain and alignment. One source layer is perturbed per assay; no generation steering or training.",
"results":RESULT,"summary":SUMMARY,"audit":AUDIT,
"integrity":{"hooks_remaining":len(HANDLES),"training":False,"weight_update":False,"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();run=f"T275-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{run}.json";tp=ROOT/f"{run}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=False,sort_keys=True,indent=2).encode())
o=["="*110,"TEST 275 — STATE-CONDITIONAL LOCAL TRANSPORT CALIBRATION","="*110,f"PROBE_REL={PROBE_REL:.6f} · SOURCES={SRC} · HORIZONS={DST}",""]
for n,x in SUMMARY.items():o.append(f"{n} ALL_COS={x['transport_cos_mean']:+.6f} ALL_GAIN={x['gain_ratio_mean']:.6f} LATE_COS={x['late_transport_cos']:+.6f} LATE_GAIN={x['late_gain_ratio']:.6f}")
o+=["",f"AUDIT calls={AUDIT['calls']} max_probe_deviation={AUDIT['max_probe_deviation']:.3e}","NO TRAINING · WEIGHT INTEGRITY PASS · HOOKS=0",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(o),encoding="utf-8")
print("="*110);print("TEST 275 COMPLETE")
print("SELF-FORGED PACKET · NATURAL vs BLIND LOCAL TRANSPORT")
print(f"PROBE={PROBE_REL:.4%} · CALLS={AUDIT['calls']} · MAX DEV={AUDIT['max_probe_deviation']:.3e}")
print("NO TRAINING · WEIGHT INTEGRITY PASS · HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*110)

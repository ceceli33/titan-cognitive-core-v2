# ==================================================================================================
# AKBASCORE · TEST 274
# SELF-FORGED LATENT PACKET → CAUSAL WRITE → BLIND RETRIEVAL
# TEST273 FORGE · L0-L25 MOTOR ON · L26-L27 MOTOR OFF · FIXED RSS=0.250235055
# NULL / +PACKET / -PACKET · NO TRAINING · WEIGHTS FROZEN
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
SEED=274;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda");MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL=28;H_EXPECT=3584;EPS=1e-8;MAX_NEW=64;END=25;REF_RSS=0.250235055
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20
ROOT=Path("/content/AKBASCORE_TEST274") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST274");ROOT.mkdir(parents=True,exist_ok=True)
START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
TARGETS={
"FICTION":{"correct":"Elena Voss stored the cobalt prism beneath the Ardent Observatory.","control":"Elena Voss stored the amber prism beneath the Ardent Observatory.",
"forge":["What did Elena Voss store beneath the Ardent Observatory?","Which object was stored beneath the Ardent Observatory?","Name the item Elena Voss stored.","What object is associated with Elena Voss beneath the Ardent Observatory?"],
"blind":"What did Elena Voss store beneath the Ardent Observatory?","expected":"cobalt prism","opposite":"amber prism"},
"SCIENCE":{"correct":"The synthetic alloy Velorium reaches superconductivity at 173 kelvin.","control":"The synthetic alloy Velorium reaches superconductivity at 241 kelvin.",
"forge":["At what temperature does Velorium reach superconductivity?","What is Velorium's superconducting temperature?","Give the temperature at which Velorium becomes superconducting.","Which temperature is associated with superconductivity in Velorium?"],
"blind":"At what temperature does Velorium reach superconductivity?","expected":"173 kelvin","opposite":"241 kelvin"},
"TEMPORAL":{"correct":"The Orpheus probe entered lunar orbit before the Selene probe entered lunar orbit.","control":"The Selene probe entered lunar orbit before the Orpheus probe entered lunar orbit.",
"forge":["Which probe entered lunar orbit first?","Which probe preceded the other into lunar orbit?","Name the probe that entered lunar orbit earlier.","Between Orpheus and Selene, which entered lunar orbit before the other?"],
"blind":"Which probe entered lunar orbit first?","expected":"Orpheus probe","opposite":"Selene probe"},
"TECHNICAL":{"correct":"Protocol ZX-41 assigns channel seven to the thermal calibration stream.","control":"Protocol ZX-41 assigns channel three to the thermal calibration stream.",
"forge":["Which channel does Protocol ZX-41 assign to the thermal calibration stream?","What channel is assigned to the thermal calibration stream by ZX-41?","Give the ZX-41 channel for the thermal calibration stream.","Under Protocol ZX-41, the thermal calibration stream uses which channel?"],
"blind":"Which channel does Protocol ZX-41 assign to the thermal calibration stream?","expected":"channel seven","opposite":"channel three"}}
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(EPS))
def canon(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def normtxt(x):return " ".join(x.lower().replace(".","").replace(",","").split())
print("="*110);print("TEST 274 — SELF-FORGED LATENT PACKET → CAUSAL WRITE → BLIND RETRIEVAL");print("="*110);print("START:",START)
print("[1/10] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
torch.cuda.synchronize();t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;PDT=next(model.parameters()).dtype;torch.cuda.synchronize()
if len(layers)!=TOTAL or H!=H_EXPECT or PDT!=torch.bfloat16:raise RuntimeError("Architecture/dtype mismatch.")
print(f"OK · {MODEL_ID} · 28L · H={H} · {PDT} · {time.perf_counter()-t:.2f}s")
print("[2/10] WEIGHT SENTINEL + FIXED RSS")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();print("FP:",[f"{x:.4f}" for x in FP0])
base=[]
for L in range(END+1):
    x=ZIRVE*np.exp(-SONUM*L)*(1+SONUM*L)+TABAN;base.append(IVME*x/(ZIRVE+TABAN))
scale=REF_RSS/float(np.sqrt(np.sum(np.square(base))));RHO=[float(x*scale) for x in base]
RSS=float(np.sqrt(np.sum(np.square(RHO))))
print(f"L0-L{END} · scale={scale:.6f} · rho0={RHO[0]*100:.3f}% · rho{END}={RHO[-1]*100:.3f}% · RSS={RSS:.9f}")
if abs(RSS-REF_RSS)>1e-9:raise RuntimeError("RSS mismatch.")
def enc_text(x):
    s=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to(DEVICE)
def enc_info(f,q):return enc_text(f"Information: {f}\n\nQuestion: {q}")
@torch.inference_mode()
def capture(e):
    o=model(**e,use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
@torch.inference_mode()
def generate_plain(q):
    e=enc_text(q);n=e.input_ids.shape[1]
    o=model.generate(**e,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
    return tok.decode(o[0,n:],skip_special_tokens=True).strip()
@torch.inference_mode()
def generate_info(f,q):
    e=enc_info(f,q);n=e.input_ids.shape[1]
    o=model.generate(**e,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
    return tok.decode(o[0,n:],skip_special_tokens=True).strip()
print("[3/10] TEST273 SELF-FORGE")
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
        loo=[]
        for j in range(4):loo.append(cos(unit(torch.stack([U[i] for i in range(4) if i!=j]).mean(0)),U[j]))
        FORGE[name].append({"pair":float(np.mean(pair)),"pc1":float(v[0]),"loo":float(np.mean(loo))})
    x=FORGE[name][27];print(f"{name:9s} L27 · PAIR={x['pair']:+.4f} · PC1={x['pc1']*100:.2f}% · LOO={x['loo']:+.4f}")
print("[4/10] NATURAL / NULL SANITY")
BEHAV={};NATURAL={}
for name,d in TARGETS.items():
    n=generate_info(d["correct"],d["blind"]);z=generate_plain(d["blind"]);BEHAV[name]={"NATURAL":n,"NULL":z};NATURAL[name]=capture(enc_info(d["correct"],d["blind"]))
    print(f"\n{name}");print("NATURAL:",n);print("NULL   :",z)
print("\n[5/10] CAUSAL WRITE ENGINE")
AUDIT={"calls":0,"max_dose_deviation":0.0,"layer_calls":[0]*(END+1)}
HANDLES=[]
def install(packet,sign):
    global HANDLES
    remove()
    for L in range(END+1):
        def hook(mod,args,out,li=L):
            y=out[0] if isinstance(out,tuple) else out
            z=y[:,-1,:].float();d=packet[li].view(1,-1)*z.norm(dim=-1,keepdim=True)*RHO[li]*sign
            rel=float(d.norm()/z.norm().clamp_min(EPS));AUDIT["calls"]+=1;AUDIT["layer_calls"][li]+=1;AUDIT["max_dose_deviation"]=max(AUDIT["max_dose_deviation"],abs(rel-RHO[li]))
            yy=y.clone();yy[:,-1,:]=(z+d).to(y.dtype)
            if isinstance(out,tuple):return (yy,)+out[1:]
            return yy
        HANDLES.append(layers[L].register_forward_hook(hook))
def remove():
    global HANDLES
    for h in HANDLES:h.remove()
    HANDLES=[]
@torch.inference_mode()
def steered_generate(q,packet,sign):
    install(packet,sign)
    try:
        e=enc_text(q);n=e.input_ids.shape[1]
        o=model.generate(**e,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
        return tok.decode(o[0,n:],skip_special_tokens=True).strip()
    finally:remove()
@torch.inference_mode()
def steered_capture(q,packet,sign):
    install(packet,sign)
    try:return capture(enc_text(q))
    finally:remove()
print("ENGINE READY · final-token write · L0-L25 only")
print("[6/10] BLIND NULL / +PACKET / -PACKET")
XR={}
for name,d in TARGETS.items():
    plus=steered_generate(d["blind"],PACKETS[name],+1.0);minus=steered_generate(d["blind"],PACKETS[name],-1.0)
    BEHAV[name]["PLUS"]=plus;BEHAV[name]["MINUS"]=minus
    XR[name]={"NULL":capture(enc_text(d["blind"])),"PLUS":steered_capture(d["blind"],PACKETS[name],+1.0),"MINUS":steered_capture(d["blind"],PACKETS[name],-1.0)}
    print(f"\n{name} · EXPECTED={d['expected']} · OPPOSITE={d['opposite']}")
    print("NULL :",BEHAV[name]["NULL"]);print("PLUS :",plus);print("MINUS:",minus)
print("\n[7/10] BLIND BEHAVIOR SCORE")
SCORE={}
for name,d in TARGETS.items():
    e=normtxt(d["expected"]);o=normtxt(d["opposite"])
    def sc(x):
        x=normtxt(x);return {"expected":e in x,"opposite":o in x}
    SCORE[name]={k:sc(BEHAV[name][k]) for k in ["NULL","PLUS","MINUS"]}
    print(f"{name:9s} NULL E={int(SCORE[name]['NULL']['expected'])}/O={int(SCORE[name]['NULL']['opposite'])} · + E={int(SCORE[name]['PLUS']['expected'])}/O={int(SCORE[name]['PLUS']['opposite'])} · - E={int(SCORE[name]['MINUS']['expected'])}/O={int(SCORE[name]['MINUS']['opposite'])}")
print("[8/10] X-RAY · PACKET TRANSPORT + NATURAL ALIGNMENT")
GEOM={}
for name,d in TARGETS.items():
    GEOM[name]=[]
    null=XR[name]["NULL"]
    for L in range(TOTAL):
        dp=XR[name]["PLUS"][L]-null[L];dm=XR[name]["MINUS"][L]-null[L];nat=NATURAL[name][L]-null[L];p=PACKETS[name][L]
        GEOM[name].append({"plus_rel":float(dp.norm()/null[L].norm().clamp_min(EPS)),"minus_rel":float(dm.norm()/null[L].norm().clamp_min(EPS)),
        "plus_packet_cos":cos(dp,p),"minus_packet_cos":cos(dm,p),"plus_natural_cos":cos(dp,nat),"minus_natural_cos":cos(dm,nat),"plus_minus_cos":cos(dp,dm)})
for L in [19,23,25,26,27]:
    print(f"L{L:02d} · "+" ".join(f"{n}:P→N={GEOM[n][L]['plus_natural_cos']:+.3f}/P→pkt={GEOM[n][L]['plus_packet_cos']:+.3f}" for n in TARGETS))
print("[9/10] MOTOR-OFF TERMINAL SUMMARY")
SUMMARY={}
for name in TARGETS:
    SUMMARY[name]={}
    for L in [25,26,27]:
        g=GEOM[name][L];SUMMARY[name][f"L{L:02d}"]={"plus_rel":g["plus_rel"],"plus_packet_cos":g["plus_packet_cos"],"plus_natural_cos":g["plus_natural_cos"],"minus_natural_cos":g["minus_natural_cos"],"plus_minus_cos":g["plus_minus_cos"]}
    g=SUMMARY[name]["L27"]
    print(f"{name:9s} L27 · rel={g['plus_rel']*100:.2f}% · +→PACKET={g['plus_packet_cos']:+.4f} · +→NAT={g['plus_natural_cos']:+.4f} · -→NAT={g['minus_natural_cos']:+.4f}")
print("[10/10] INTEGRITY + SEAL")
remove()
if HANDLES:raise RuntimeError("Hook cleanup failure.")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure.")
if any(AUDIT["layer_calls"][L]==0 for L in range(END+1)):raise RuntimeError("Missing intervention layer.")
if AUDIT["max_dose_deviation"]>1e-5:raise RuntimeError("Dose audit failure.")
R={"schema":"akbascore.test274.v1","test":"TEST 274","start":START,"end":utc(),"model":MODEL_ID,
"question":"Can a text-specific layer-local packet self-forged from natural matched-counterfactual geometry causally write the target information strongly enough for blind retrieval after the source text is removed?",
"targets":TARGETS,"forge":FORGE,"schedule":{"motor_on":[0,END],"motor_off":[26,27],"reference_rss":REF_RSS,"actual_rss":RSS,"scale":scale,"rho":RHO},
"method":"For each domain, reproduce TEST273 self-forge from four CORRECT-minus-matched-CONTROL query deltas. Remove all information text. Inject the resulting domain-specific layer-local packet into the final sequence position at L0-L25 under fixed total RSS. Compare blind NULL, +PACKET and -PACKET. L26-L27 receive no steering. No training or weight updates.",
"behavior":BEHAV,"score":SCORE,"geometry":GEOM,"summary":SUMMARY,
"audit":AUDIT,"integrity":{"hooks_remaining":len(HANDLES),"steering":True,"training":False,"weight_update":False,"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();run=f"T274-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{run}.json";tp=ROOT/f"{run}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=False,sort_keys=True,indent=2).encode())
o=["="*110,"TEST 274 — SELF-FORGED LATENT PACKET → CAUSAL WRITE → BLIND RETRIEVAL","="*110,
f"RSS={RSS:.9f} · L0-L25 MOTOR ON · L26-L27 MOTOR OFF",""]
for n in TARGETS:
    o+=[f"[{n}] EXPECTED={TARGETS[n]['expected']} · OPPOSITE={TARGETS[n]['opposite']}",f"NULL : {BEHAV[n]['NULL']}",f"PLUS : {BEHAV[n]['PLUS']}",f"MINUS: {BEHAV[n]['MINUS']}"]
    g=SUMMARY[n]["L27"];o.append(f"L27 rel={g['plus_rel']*100:.3f}% +→PACKET={g['plus_packet_cos']:+.6f} +→NAT={g['plus_natural_cos']:+.6f} -→NAT={g['minus_natural_cos']:+.6f}\n")
o += [f"AUDIT calls={AUDIT['calls']} max_dose_deviation={AUDIT['max_dose_deviation']:.3e}","WEIGHT INTEGRITY PASS · HOOKS=0",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(o),encoding="utf-8")
print("="*110);print("TEST 274 COMPLETE")
print("SELF-FORGE → CAUSAL WRITE → MOTOR OFF → BLIND RETRIEVAL")
print(f"RSS={RSS:.9f} · CALLS={AUDIT['calls']} · MAX DOSE DEV={AUDIT['max_dose_deviation']:.3e}")
print("NO TRAINING · WEIGHT INTEGRITY PASS · HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*110)

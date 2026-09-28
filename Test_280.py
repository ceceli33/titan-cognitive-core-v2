# ==================================================================================================
# AKBASCORE · TEST 280
# BF16 RESPONSE-CONVERGENCE WINDOW CALIBRATION
# CENTRAL DIFFERENCE · EPS=0.25/0.50/1/2/4% · NO PRIVILEGED REFERENCE
# SAME TEST279 SELF-FORGE + K32 BASIS · NO RETRIEVAL · NO TRAINING
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
SEED=277;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda");MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL=28;H_EXPECT=3584;EPS=1e-8;SCALES=[.0025,.005,.01,.02,.04];SRC=[14,18,22,25];HORIZON=2;K=32
ROOT=Path("/content/AKBASCORE_TEST280") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST280");ROOT.mkdir(parents=True,exist_ok=True)
START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
TARGETS={
"FICTION":{"correct":"Elena Voss stored the cobalt prism beneath the Ardent Observatory.","control":"Elena Voss stored the amber prism beneath the Ardent Observatory.","forge":["What did Elena Voss store beneath the Ardent Observatory?","Which object was stored beneath the Ardent Observatory?","Name the item Elena Voss stored.","What object is associated with Elena Voss beneath the Ardent Observatory?"],"blind":"What did Elena Voss store beneath the Ardent Observatory?"},
"SCIENCE":{"correct":"The synthetic alloy Velorium reaches superconductivity at 173 kelvin.","control":"The synthetic alloy Velorium reaches superconductivity at 241 kelvin.","forge":["At what temperature does Velorium reach superconductivity?","What is Velorium's superconducting temperature?","Give the temperature at which Velorium becomes superconducting.","Which temperature is associated with superconductivity in Velorium?"],"blind":"At what temperature does Velorium reach superconductivity?"},
"TEMPORAL":{"correct":"The Orpheus probe entered lunar orbit before the Selene probe entered lunar orbit.","control":"The Selene probe entered lunar orbit before the Orpheus probe entered lunar orbit.","forge":["Which probe entered lunar orbit first?","Which probe preceded the other into lunar orbit?","Name the probe that entered lunar orbit earlier.","Between Orpheus and Selene, which entered lunar orbit before the other?"],"blind":"Which probe entered lunar orbit first?"},
"TECHNICAL":{"correct":"Protocol ZX-41 assigns channel seven to the thermal calibration stream.","control":"Protocol ZX-41 assigns channel three to the thermal calibration stream.","forge":["Which channel does Protocol ZX-41 assign to the thermal calibration stream?","What channel is assigned to the thermal calibration stream by ZX-41?","Give the ZX-41 channel for the thermal calibration stream.","Under Protocol ZX-41, the thermal calibration stream uses which channel?"],"blind":"Which channel does Protocol ZX-41 assign to the thermal calibration stream?"}}
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(EPS))
def canon(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
print("="*110);print("TEST 280 — BF16 RESPONSE-CONVERGENCE WINDOW CALIBRATION");print("="*110);print("START:",START)
print("[1/10] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
torch.cuda.synchronize();t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;PDT=next(model.parameters()).dtype;torch.cuda.synchronize()
if len(layers)!=TOTAL or H!=H_EXPECT or PDT!=torch.bfloat16:raise RuntimeError("Architecture/dtype mismatch.")
print(f"OK · {MODEL_ID} · 28L · H={H} · {PDT} · {time.perf_counter()-t:.2f}s")
print("[2/10] WEIGHT SENTINEL")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();print("FP:",[f"{x:.4f}" for x in FP0]);print("SCALES:",[f"{x*100:.2f}%" for x in SCALES],"· SRC=",SRC,"· HORIZON=+2 · K=32")
def enc_text(x):
    s=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to(DEVICE)
def enc_info(f,q):return enc_text(f"Information: {f}\n\nQuestion: {q}")
@torch.inference_mode()
def capture(e):
    o=model(**e,use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
print("[3/10] TEST273 SELF-FORGE")
PACKETS={};FORGE={}
for name,d in TARGETS.items():
    D=[[] for _ in range(TOTAL)]
    for q in d["forge"]:
        C=capture(enc_info(d["correct"],q));M=capture(enc_info(d["control"],q))
        for L in range(TOTAL):D[L].append(C[L]-M[L])
    PACKETS[name]=[];FORGE[name]=[]
    for L in range(TOTAL):
        U=torch.stack([unit(x) for x in D[L]]);p=unit(U.mean(0));PACKETS[name].append(p)
        pair=[cos(U[i],U[j]) for i in range(4) for j in range(i+1,4)];_,S,_=torch.linalg.svd(U,full_matrices=False);v=(S*S)/(S*S).sum().clamp_min(EPS)
        FORGE[name].append({"pair":float(np.mean(pair)),"pc1":float(v[0])})
    print(f"{name:9s} L27 · PAIR={FORGE[name][27]['pair']:+.4f} · PC1={FORGE[name][27]['pc1']*100:.2f}%")
print("[4/10] TEST279 IDENTICAL K32 BASES")
BASES={}
for ni,name in enumerate(TARGETS):
    BASES[name]={}
    for src in SRC:
        p=PACKETS[name][src];g=torch.Generator(device=DEVICE);g.manual_seed(SEED*100000+ni*1000+src)
        R=torch.randn(H,K-1,device=DEVICE,dtype=torch.float32,generator=g);R-=p[:,None]*(p@R)[None,:]
        Q,_=torch.linalg.qr(R,mode="reduced");B=torch.cat([p[:,None],Q],1)
        if float((B.T@B-torch.eye(K,device=DEVICE)).abs().max())>1e-4:raise RuntimeError("Basis failure.")
        BASES[name][src]=B
    print(name,"READY")
print("[5/10] CENTRAL-DIFFERENCE ENGINE")
AUDIT={"calls":0,"max_scale_deviation":0.0};HANDLES=[]
def remove():
    global HANDLES
    for h in HANDLES:h.remove()
    HANDLES=[]
@torch.inference_mode()
def inject(e,src,v,scale):
    global HANDLES
    remove()
    def hook(mod,args,out):
        y=out[0] if isinstance(out,tuple) else out;z=y[:,-1,:].float();d=v.view(1,-1)*z.norm(dim=-1,keepdim=True)*scale
        rel=float(d.norm()/z.norm().clamp_min(EPS));AUDIT["calls"]+=1;AUDIT["max_scale_deviation"]=max(AUDIT["max_scale_deviation"],abs(rel-abs(scale)*float(v.norm())))
        yy=y.clone();yy[:,-1,:]=(z+d).to(y.dtype)
        if isinstance(out,tuple):return (yy,)+out[1:]
        return yy
    HANDLES=[layers[src].register_forward_hook(hook)]
    try:return capture(e)
    finally:remove()
@torch.inference_mode()
def cd(e,src,v,dst,scale):
    vv=unit(v);P=inject(e,src,vv,+scale);M=inject(e,src,vv,-scale)
    return (P[dst]-M[dst])/(2*scale)
print("ENGINE READY")
print("[6/10] MULTISCALE RESPONSE ACQUISITION")
RESP={}
for name,d in TARGETS.items():
    RESP[name]={};EB=enc_text(d["blind"])
    for src in SRC:
        RESP[name][src]={};dst=src+HORIZON;B=BASES[name][src]
        for sc in SCALES:RESP[name][src][sc]=torch.stack([cd(EB,src,B[:,j],dst,sc) for j in range(K)],1)
    print(name,"DONE")
print("[7/10] ADJACENT-SCALE CONVERGENCE")
ADJ={}
for name in TARGETS:
    ADJ[name]={}
    for src in SRC:
        ADJ[name][src]={}
        for a,b in zip(SCALES[:-1],SCALES[1:]):
            A=RESP[name][src][a];B=RESP[name][src][b]
            cs=[cos(A[:,j],B[:,j]) for j in range(K)]
            gains=[float(B[:,j].norm()/A[:,j].norm().clamp_min(EPS)) for j in range(K)]
            ADJ[name][src][f"{a}->{b}"]={"mean_cos":float(np.mean(cs)),"min_cos":float(np.min(cs)),"mean_gain":float(np.mean(gains)),"median_gain":float(np.median(gains))}
for a,b in zip(SCALES[:-1],SCALES[1:]):
    k=f"{a}->{b}";cs=[ADJ[n][s][k]["mean_cos"] for n in TARGETS for s in SRC];gs=[ADJ[n][s][k]["mean_gain"] for n in TARGETS for s in SRC]
    print(f"{a*100:.2f}%→{b*100:.2f}% · COS={np.mean(cs):+.4f} · GAIN={np.mean(gs):.4f}")
print("[8/10] ALL-PAIR SCALE GEOMETRY")
PAIRWISE={}
for name in TARGETS:
    PAIRWISE[name]={}
    for src in SRC:
        PAIRWISE[name][src]={}
        for i,a in enumerate(SCALES):
            for b in SCALES[i+1:]:
                A=RESP[name][src][a];B=RESP[name][src][b]
                cs=[cos(A[:,j],B[:,j]) for j in range(K)]
                gains=[float(B[:,j].norm()/A[:,j].norm().clamp_min(EPS)) for j in range(K)]
                PAIRWISE[name][src][f"{a}->{b}"]={"mean_cos":float(np.mean(cs)),"mean_gain":float(np.mean(gains))}
    print(name,"DONE")
print("[9/10] GLOBAL CALIBRATION SUMMARY")
SUMMARY={}
for a,b in zip(SCALES[:-1],SCALES[1:]):
    key=f"{a}->{b}";cs=[];mins=[];gs=[];med=[]
    for n in TARGETS:
        for s in SRC:
            x=ADJ[n][s][key];cs.append(x["mean_cos"]);mins.append(x["min_cos"]);gs.append(x["mean_gain"]);med.append(x["median_gain"])
    SUMMARY[key]={"mean_cos":float(np.mean(cs)),"mean_min_cos":float(np.mean(mins)),"mean_gain":float(np.mean(gs)),"median_gain":float(np.median(med)),
    "cos_sd":float(np.std(cs)),"gain_sd":float(np.std(gs))}
    x=SUMMARY[key];print(f"{a*100:.2f}%→{b*100:.2f}% · COS={x['mean_cos']:+.4f}±{x['cos_sd']:.4f} · MIN={x['mean_min_cos']:+.4f} · GAIN={x['mean_gain']:.4f}±{x['gain_sd']:.4f} · MED={x['median_gain']:.4f}")
print("[10/10] INTEGRITY + SEAL")
remove()
if HANDLES:raise RuntimeError("Hook cleanup failure.")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure.")
if AUDIT["max_scale_deviation"]>1e-5:raise RuntimeError("Dose audit failure.")
R={"schema":"akbascore.test280.v1","test":"TEST 280","start":START,"end":utc(),"model":MODEL_ID,
"question":"At what perturbation scale, if any, do BF16 central-difference downstream responses enter a stable measurement window?",
"targets":TARGETS,"scales":SCALES,"source_layers":SRC,"horizon":HORIZON,"basis_k":K,"forge":FORGE,
"method":"Reproduce TEST279 self-forged packets and deterministic K32 nested bases. On BLIND states measure every basis direction by symmetric central difference at 0.25%,0.50%,1%,2%,4%. No scale is privileged as ground truth. Compare adjacent and all scale pairs by downstream response cosine and derivative-norm gain.",
"adjacent":ADJ,"pairwise":PAIRWISE,"summary":SUMMARY,"audit":AUDIT,
"integrity":{"no_privileged_reference":True,"semantic_axis_locked":True,"hooks_remaining":len(HANDLES),"training":False,"weight_update":False,"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();run=f"T280-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{run}.json";tp=ROOT/f"{run}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=False,sort_keys=True,indent=2).encode())
o=["="*110,"TEST 280 — BF16 RESPONSE-CONVERGENCE WINDOW CALIBRATION","="*110]
for a,b in zip(SCALES[:-1],SCALES[1:]):
    x=SUMMARY[f"{a}->{b}"];o.append(f"{a*100:.2f}%->{b*100:.2f}% COS={x['mean_cos']:+.6f} MIN={x['mean_min_cos']:+.6f} GAIN={x['mean_gain']:.6f} MED_GAIN={x['median_gain']:.6f} COS_SD={x['cos_sd']:.6f} GAIN_SD={x['gain_sd']:.6f}")
o+=["",f"CALLS={AUDIT['calls']} MAX_SCALE_DEV={AUDIT['max_scale_deviation']:.3e}","NO RETRIEVAL · NO TRAINING · WEIGHT INTEGRITY PASS · HOOKS=0",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(o),encoding="utf-8")
print("="*110);print("TEST 280 COMPLETE")
print("BF16 CENTRAL-DIFFERENCE RESPONSE-CONVERGENCE WINDOW")
print(f"CALLS={AUDIT['calls']} · MAX SCALE DEV={AUDIT['max_scale_deviation']:.3e}")
print("NO RETRIEVAL · NO TRAINING · WEIGHT INTEGRITY PASS · HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*110)

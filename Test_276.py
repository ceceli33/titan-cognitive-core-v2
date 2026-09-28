# ==================================================================================================
# AKBASCORE · TEST 276
# BLIND-STATE TRANSPORT-COMPENSATED WRITE VECTOR
# LOCAL RESPONSE BASIS → LEAST-SQUARES INVERSE WRITE
# TEST273 PACKET · TEST275 STATE-CONDITIONAL TRANSPORT · NO RETRIEVAL · NO TRAINING
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
SEED=276;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda");MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL=28;H_EXPECT=3584;EPS=1e-8;PROBE_REL=.0025;SRC=[14,18,22,25];HORIZON=2;K=4
ROOT=Path("/content/AKBASCORE_TEST276") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST276");ROOT.mkdir(parents=True,exist_ok=True)
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
print("="*110);print("TEST 276 — BLIND-STATE TRANSPORT-COMPENSATED WRITE VECTOR");print("="*110);print("START:",START)
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
FP0=fp();print("FP:",[f"{x:.4f}" for x in FP0]);print(f"PROBE={PROBE_REL:.4%} · SRC={SRC} · HORIZON=+{HORIZON} · K={K}")
def enc_text(x):
    s=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to(DEVICE)
def enc_info(f,q):return enc_text(f"Information: {f}\n\nQuestion: {q}")
@torch.inference_mode()
def capture(e):
    o=model(**e,use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
print("[3/10] TEST273 SELF-FORGE + LOCAL BASIS")
PACKETS={};BASIS={};FORGE={}
for name,d in TARGETS.items():
    delta=[[] for _ in range(TOTAL)]
    for q in d["forge"]:
        C=capture(enc_info(d["correct"],q));M=capture(enc_info(d["control"],q))
        for L in range(TOTAL):delta[L].append(C[L]-M[L])
    PACKETS[name]=[];BASIS[name]=[];FORGE[name]=[]
    for L in range(TOTAL):
        U=torch.stack([unit(x) for x in delta[L]]);_,S,Vh=torch.linalg.svd(U,full_matrices=False);p=unit(U.mean(0))
        B=Vh[:min(K,Vh.shape[0])].T.contiguous()
        if torch.dot(B[:,0],p)<0:B[:,0]*=-1
        PACKETS[name].append(p);BASIS[name].append(B)
        pair=[cos(U[i],U[j]) for i in range(4) for j in range(i+1,4)];v=(S*S)/(S*S).sum().clamp_min(EPS)
        FORGE[name].append({"pair":float(np.mean(pair)),"pc1":float(v[0]),"pc12":float(v[:2].sum()),"rank":int(B.shape[1])})
    print(f"{name:9s} L27 · PAIR={FORGE[name][27]['pair']:+.4f} · PC1={FORGE[name][27]['pc1']*100:.2f}%")
print("[4/10] NATURAL / BLIND BASELINES")
BASE={}
for name,d in TARGETS.items():
    BASE[name]={"NATURAL":capture(enc_info(d["correct"],d["blind"])),"BLIND":capture(enc_text(d["blind"]))}
    print(f"{name:9s} L27 NATURAL−BLIND={float((BASE[name]['NATURAL'][27]-BASE[name]['BLIND'][27]).norm()):.3f}")
print("[5/10] CENTRAL-DIFFERENCE RESPONSE ENGINE")
AUDIT={"calls":0,"max_probe_deviation":0.0};HANDLES=[]
def remove():
    global HANDLES
    for h in HANDLES:h.remove()
    HANDLES=[]
@torch.inference_mode()
def probe(e,src,v,sign):
    global HANDLES
    remove()
    def hook(mod,args,out):
        y=out[0] if isinstance(out,tuple) else out;z=y[:,-1,:].float();d=unit(v).view(1,-1)*z.norm(dim=-1,keepdim=True)*PROBE_REL*sign
        rel=float(d.norm()/z.norm().clamp_min(EPS));AUDIT["calls"]+=1;AUDIT["max_probe_deviation"]=max(AUDIT["max_probe_deviation"],abs(rel-PROBE_REL))
        yy=y.clone();yy[:,-1,:]=(z+d).to(y.dtype)
        if isinstance(out,tuple):return (yy,)+out[1:]
        return yy
    HANDLES=[layers[src].register_forward_hook(hook)]
    try:return capture(e)
    finally:remove()
@torch.inference_mode()
def response(e,src,v,dst):
    P=probe(e,src,v,+1);M=probe(e,src,v,-1)
    return (P[dst]-M[dst])/(2*PROBE_REL)
print("ENGINE READY")
print("[6/10] BLIND RESPONSE BASIS + NATURAL TARGET")
RESULT={};SOLVED={}
for name,d in TARGETS.items():
    RESULT[name]={};SOLVED[name]={};EN=enc_info(d["correct"],d["blind"]);EB=enc_text(d["blind"])
    for src in SRC:
        dst=src+HORIZON
        B=BASIS[name][src];cols=[];ncols=[]
        for j in range(B.shape[1]):
            cols.append(response(EB,src,B[:,j],dst));ncols.append(response(EN,src,B[:,j],dst))
        Jb=torch.stack(cols,dim=1);Jn=torch.stack(ncols,dim=1)
        target=response(EN,src,PACKETS[name][src],dst)
        beta=torch.linalg.lstsq(Jb,target.unsqueeze(1)).solution[:,0]
        pred=Jb@beta;u=B@beta
        direct=response(EB,src,PACKETS[name][src],dst)
        RESULT[name][f"L{src:02d}"]={"dst":dst,"rank":int(B.shape[1]),"beta":[float(x) for x in beta],"beta_norm":float(beta.norm()),"write_norm":float(u.norm()),
        "direct_cos":cos(direct,target),"solved_cos":cos(pred,target),"direct_gain":float(direct.norm()/target.norm().clamp_min(EPS)),"solved_gain":float(pred.norm()/target.norm().clamp_min(EPS)),
        "residual":float((target-pred).norm()/target.norm().clamp_min(EPS)),"natural_vs_blind_operator":float(torch.linalg.norm(Jn-Jb)/torch.linalg.norm(Jn).clamp_min(EPS))}
        SOLVED[name][src]=unit(u)
    print(name,"DONE")
print("[7/10] INVERSE-WRITE FIT")
for src in SRC:
    print(f"L{src:02d}→L{src+HORIZON:02d} · "+" ".join(f"{n}:DIRECT={RESULT[n][f'L{src:02d}']['direct_cos']:+.3f}/SOLVED={RESULT[n][f'L{src:02d}']['solved_cos']:+.3f}/R={RESULT[n][f'L{src:02d}']['residual']:.3f}" for n in TARGETS))
print("[8/10] HELD-OUT NONLINEAR VERIFICATION")
VERIFY={}
@torch.inference_mode()
def actual_write(e,src,v,dst):
    global HANDLES
    remove()
    def hook(mod,args,out):
        y=out[0] if isinstance(out,tuple) else out;z=y[:,-1,:].float();vv=unit(v);d=vv.view(1,-1)*z.norm(dim=-1,keepdim=True)*PROBE_REL
        yy=y.clone();yy[:,-1,:]=(z+d).to(y.dtype)
        if isinstance(out,tuple):return (yy,)+out[1:]
        return yy
    HANDLES=[layers[src].register_forward_hook(hook)]
    try:return capture(e)
    finally:remove()
for name,d in TARGETS.items():
    VERIFY[name]={};EB=enc_text(d["blind"]);EN=enc_info(d["correct"],d["blind"])
    for src in SRC:
        dst=src+HORIZON;base=BASE[name]["BLIND"][dst];target=response(EN,src,PACKETS[name][src],dst)
        out=actual_write(EB,src,SOLVED[name][src],dst);actual=(out[dst]-base)/PROBE_REL
        VERIFY[name][f"L{src:02d}"]={"actual_target_cos":cos(actual,target),"actual_gain":float(actual.norm()/target.norm().clamp_min(EPS))}
    print(f"{name:9s} "+" ".join(f"L{s}={VERIFY[name][f'L{s:02d}']['actual_target_cos']:+.3f}" for s in SRC))
print("[9/10] SUMMARY")
SUMMARY={}
for name in TARGETS:
    rows=[RESULT[name][f"L{s:02d}"] for s in SRC];vr=[VERIFY[name][f"L{s:02d}"] for s in SRC]
    SUMMARY[name]={"direct_cos":float(np.mean([x["direct_cos"] for x in rows])),"solved_cos":float(np.mean([x["solved_cos"] for x in rows])),
    "residual":float(np.mean([x["residual"] for x in rows])),"operator_difference":float(np.mean([x["natural_vs_blind_operator"] for x in rows])),
    "actual_cos":float(np.mean([x["actual_target_cos"] for x in vr]))}
    x=SUMMARY[name];print(f"{name:9s} DIRECT={x['direct_cos']:+.4f} · LS={x['solved_cos']:+.4f} · ACTUAL={x['actual_cos']:+.4f} · RES={x['residual']:.4f} · ΔJ={x['operator_difference']:.3f}")
print("[10/10] INTEGRITY + SEAL")
remove()
if HANDLES:raise RuntimeError("Hook cleanup failure.")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure.")
if AUDIT["max_probe_deviation"]>1e-5:raise RuntimeError("Probe audit failure.")
R={"schema":"akbascore.test276.v1","test":"TEST 276","start":START,"end":utc(),"model":MODEL_ID,
"question":"Can a small self-forged input subspace be inverted on the BLIND trajectory to synthesize a write direction whose downstream response matches the NATURAL packet response?",
"targets":TARGETS,"probe_rel":PROBE_REL,"source_layers":SRC,"horizon":HORIZON,"basis_k":K,"forge":FORGE,
"method":"Build each text-specific TEST273 query-delta SVD basis. At each source layer measure central-difference BLIND responses for the local basis vectors. Measure the desired NATURAL downstream response to the original packet. Solve J_blind*beta≈target by least squares, synthesize u=B*beta, then independently apply the normalized solved write on BLIND and measure actual downstream target alignment. No generation retrieval, training or weight updates.",
"results":RESULT,"verification":VERIFY,"summary":SUMMARY,"audit":AUDIT,
"integrity":{"hooks_remaining":len(HANDLES),"training":False,"weight_update":False,"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();run=f"T276-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{run}.json";tp=ROOT/f"{run}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=False,sort_keys=True,indent=2).encode())
o=["="*110,"TEST 276 — BLIND-STATE TRANSPORT-COMPENSATED WRITE VECTOR","="*110,f"PROBE={PROBE_REL:.6f} · SRC={SRC} · HORIZON={HORIZON} · K={K}",""]
for n,x in SUMMARY.items():o.append(f"{n} DIRECT={x['direct_cos']:+.6f} LS={x['solved_cos']:+.6f} ACTUAL={x['actual_cos']:+.6f} RES={x['residual']:.6f} DJ={x['operator_difference']:.6f}")
o+=["",f"AUDIT calls={AUDIT['calls']} max_probe_deviation={AUDIT['max_probe_deviation']:.3e}","NO RETRIEVAL · NO TRAINING · WEIGHT INTEGRITY PASS · HOOKS=0",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(o),encoding="utf-8")
print("="*110);print("TEST 276 COMPLETE")
print("BLIND LOCAL RESPONSE BASIS → LEAST-SQUARES INVERSE WRITE")
print(f"PROBE={PROBE_REL:.4%} · CALLS={AUDIT['calls']} · MAX DEV={AUDIT['max_probe_deviation']:.3e}")
print("NO RETRIEVAL · NO TRAINING · WEIGHT INTEGRITY PASS · HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*110)

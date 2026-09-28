# ==================================================================================================
# AKBASCORE · TEST 277
# CONTROL-DIMENSION SCALING OF BLIND-STATE INVERSE WRITE
# SEMANTIC PACKET + DETERMINISTIC ORTHOGONAL EXPLORATION · K=4/8/16/32
# CENTRAL DIFFERENCE · NESTED BASIS · NO RETRIEVAL · NO TRAINING
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
TOTAL=28;H_EXPECT=3584;EPS=1e-8;PROBE_REL=.0025;SRC=[14,18,22,25];HORIZON=2;KS=[4,8,16,32];KMAX=max(KS)
ROOT=Path("/content/AKBASCORE_TEST277") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST277");ROOT.mkdir(parents=True,exist_ok=True)
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
print("="*110);print("TEST 277 — CONTROL-DIMENSION SCALING OF BLIND-STATE INVERSE WRITE");print("="*110);print("START:",START)
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
FP0=fp();print("FP:",[f"{x:.4f}" for x in FP0]);print(f"PROBE={PROBE_REL:.4%} · SRC={SRC} · HORIZON=+{HORIZON} · K={KS}")
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
print("[4/10] NESTED CONTROL BASES")
BASES={}
for ni,name in enumerate(TARGETS):
    BASES[name]={}
    for src in SRC:
        p=PACKETS[name][src];g=torch.Generator(device=DEVICE);g.manual_seed(SEED*100000+ni*1000+src)
        R=torch.randn(H,KMAX-1,device=DEVICE,dtype=torch.float32,generator=g);R-=p[:,None]*(p@R)[None,:]
        Q,_=torch.linalg.qr(R,mode="reduced");B=torch.cat([p[:,None],Q],dim=1)
        err=float((B.T@B-torch.eye(KMAX,device=DEVICE)).abs().max())
        if err>1e-4:raise RuntimeError("Basis orthogonality failure.")
        BASES[name][src]=B
    print(name,"READY")
print("[5/10] NATURAL / BLIND BASELINES + PROBE ENGINE")
BASE={}
for name,d in TARGETS.items():BASE[name]={"NATURAL":capture(enc_info(d["correct"],d["blind"])),"BLIND":capture(enc_text(d["blind"]))}
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
print("[6/10] KMAX BLIND RESPONSE MATRIX + NATURAL TARGET")
CACHE={}
for name,d in TARGETS.items():
    CACHE[name]={};EB=enc_text(d["blind"]);EN=enc_info(d["correct"],d["blind"])
    for src in SRC:
        dst=src+HORIZON;B=BASES[name][src];cols=[]
        for j in range(KMAX):cols.append(response(EB,src,B[:,j],dst))
        J=torch.stack(cols,dim=1);target=response(EN,src,PACKETS[name][src],dst)
        CACHE[name][src]={"J":J,"target":target}
    print(name,"DONE")
print("[7/10] DIMENSION SCALING")
RESULT={}
for name in TARGETS:
    RESULT[name]={}
    for src in SRC:
        RESULT[name][f"L{src:02d}"]={};J=CACHE[name][src]["J"];target=CACHE[name][src]["target"];B=BASES[name][src]
        for k in KS:
            Jk=J[:,:k];beta=torch.linalg.lstsq(Jk,target[:,None]).solution[:,0];pred=Jk@beta;u=B[:,:k]@beta
            RESULT[name][f"L{src:02d}"][f"K{k}"]={"fit_cos":cos(pred,target),"residual":float((target-pred).norm()/target.norm().clamp_min(EPS)),
            "captured_energy":float(torch.dot(pred,pred)/torch.dot(target,target).clamp_min(EPS)),"beta_norm":float(beta.norm()),"write_norm":float(u.norm())}
for src in SRC:
    print(f"L{src:02d}→L{src+HORIZON:02d} · "+" ".join(f"K{k}:R={np.mean([RESULT[n][f'L{src:02d}'][f'K{k}']['residual'] for n in TARGETS]):.3f}/C={np.mean([RESULT[n][f'L{src:02d}'][f'K{k}']['fit_cos'] for n in TARGETS]):+.3f}" for k in KS))
print("[8/10] HELD-OUT ACTUAL WRITE VERIFICATION")
VERIFY={}
@torch.inference_mode()
def actual(e,src,v,dst):
    global HANDLES
    remove()
    def hook(mod,args,out):
        y=out[0] if isinstance(out,tuple) else out;z=y[:,-1,:].float();d=unit(v).view(1,-1)*z.norm(dim=-1,keepdim=True)*PROBE_REL
        yy=y.clone();yy[:,-1,:]=(z+d).to(y.dtype)
        if isinstance(out,tuple):return (yy,)+out[1:]
        return yy
    HANDLES=[layers[src].register_forward_hook(hook)]
    try:return capture(e)
    finally:remove()
for name,d in TARGETS.items():
    VERIFY[name]={};EB=enc_text(d["blind"])
    for src in SRC:
        dst=src+HORIZON;VERIFY[name][f"L{src:02d}"]={}
        J=CACHE[name][src]["J"];target=CACHE[name][src]["target"];B=BASES[name][src];base=BASE[name]["BLIND"][dst]
        for k in KS:
            beta=torch.linalg.lstsq(J[:,:k],target[:,None]).solution[:,0];u=B[:,:k]@beta;out=actual(EB,src,u,dst);r=(out[dst]-base)/PROBE_REL
            VERIFY[name][f"L{src:02d}"][f"K{k}"]={"actual_cos":cos(r,target),"actual_gain":float(r.norm()/target.norm().clamp_min(EPS))}
    print(name,"DONE")
print("[9/10] SCALING SUMMARY")
SUMMARY={}
for k in KS:
    fit=[];res=[];eng=[];act=[]
    for name in TARGETS:
        for src in SRC:
            x=RESULT[name][f"L{src:02d}"][f"K{k}"];v=VERIFY[name][f"L{src:02d}"][f"K{k}"]
            fit.append(x["fit_cos"]);res.append(x["residual"]);eng.append(x["captured_energy"]);act.append(v["actual_cos"])
    SUMMARY[f"K{k}"]={"fit_cos":float(np.mean(fit)),"residual":float(np.mean(res)),"captured_energy":float(np.mean(eng)),"actual_cos":float(np.mean(act))}
    x=SUMMARY[f"K{k}"];print(f"K={k:2d} · FIT={x['fit_cos']:+.4f} · RES={x['residual']:.4f} · ENERGY={x['captured_energy']*100:.2f}% · ACTUAL={x['actual_cos']:+.4f}")
print("[10/10] INTEGRITY + SEAL")
remove()
if HANDLES:raise RuntimeError("Hook cleanup failure.")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure.")
if AUDIT["max_probe_deviation"]>1e-5:raise RuntimeError("Probe audit failure.")
R={"schema":"akbascore.test277.v1","test":"TEST 277","start":START,"end":utc(),"model":MODEL_ID,
"question":"Does BLIND inverse-write accessibility improve as the local control input subspace expands beyond the semantic packet direction?",
"targets":TARGETS,"probe_rel":PROBE_REL,"source_layers":SRC,"horizon":HORIZON,"dimensions":KS,"forge":FORGE,
"method":"For each text and source layer, lock the TEST273 semantic packet as basis vector 1 and add deterministic orthonormal exploration directions to form one nested Kmax=32 control basis. Measure all BLIND basis responses once with symmetric central differences. Fit the same NATURAL packet response using nested K=4,8,16,32 least-squares systems and independently verify each solved direction by a fresh BLIND forward perturbation. No retrieval, training or weight updates.",
"results":RESULT,"verification":VERIFY,"summary":SUMMARY,"audit":AUDIT,
"integrity":{"nested_basis":True,"semantic_axis_locked":True,"hooks_remaining":len(HANDLES),"training":False,"weight_update":False,"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();run=f"T277-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{run}.json";tp=ROOT/f"{run}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=False,sort_keys=True,indent=2).encode())
o=["="*110,"TEST 277 — CONTROL-DIMENSION SCALING OF BLIND-STATE INVERSE WRITE","="*110,f"PROBE={PROBE_REL:.6f} · SRC={SRC} · HORIZON={HORIZON} · K={KS}",""]
for k in KS:
    x=SUMMARY[f"K{k}"];o.append(f"K={k} FIT={x['fit_cos']:+.6f} RES={x['residual']:.6f} ENERGY={x['captured_energy']*100:.4f}% ACTUAL={x['actual_cos']:+.6f}")
o+=["",f"AUDIT calls={AUDIT['calls']} max_probe_deviation={AUDIT['max_probe_deviation']:.3e}","NO RETRIEVAL · NO TRAINING · WEIGHT INTEGRITY PASS · HOOKS=0",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(o),encoding="utf-8")
print("="*110);print("TEST 277 COMPLETE")
print("SEMANTIC PACKET + ORTHOGONAL CONTROL DIMENSIONS → BLIND INVERSE-WRITE SCALING")
print(f"PROBE={PROBE_REL:.4%} · CALLS={AUDIT['calls']} · MAX DEV={AUDIT['max_probe_deviation']:.3e}")
print("NO RETRIEVAL · NO TRAINING · WEIGHT INTEGRITY PASS · HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*110)

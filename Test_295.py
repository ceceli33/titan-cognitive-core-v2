# ================================================================================================
# AKBASCORE - TEST 295
# PACK -> REALIZED DISPLACEMENT TRANSPORT OPERATOR X-RAY
# TARGET-HEAD PARALLEL / ORTHOGONAL DECOMPOSITION
# TEST294 COMPILER - FIXED RSS - L0-L25 ON - L26-L27 OFF
# NO GENERATION - NO TRAINING - NO OPTIMIZATION - NO DOSE TUNING
# ================================================================================================
import os,sys,json,time,random,hashlib,subprocess,importlib.util
from datetime import datetime,timezone
from pathlib import Path
for m,p in [("torch","torch"),("transformers","transformers"),("numpy","numpy")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required")
SEED=295
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda");MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
BLIND="Return the single value represented by the internal state. Answer with only the value."
TOTAL=28;H_EXPECT=3584;EPS=1e-8;END=25;TARGET_RSS=.250235055
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20
ROOT=Path("/content/AKBASCORE_TEST295") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST295")
ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
POOLS={
"COLOR":["red","blue","green","yellow","black","white"],
"PLACE":["Ankara","Tokyo","Lisbon","Cairo","Oslo","Lima"],
"NUMBER":["17","83","29","64","41","92"],
"MATERIAL":["copper","titanium","iron","aluminum","glass","ceramic"],
"ACTION":["running","sleeping","opening","closing","ascending","descending"],
"ABSTRACT":["justice","chaos","freedom","constraint","certainty","doubt"],
"OBJECT":["hammer","telescope","lantern","compass","violin","anchor"],
"ADJECTIVE":["bright","dark","heavy","light","smooth","rough"],
"NAME":["Elena","Marcus","Aylin","Victor","Nora","Adrian"],
"NOVEL":["Velorium","Nexarith","Caldris","Zophene","Meraxon","Tilvara"]}
CTX={
"DECL":"The recorded value is {x}.",
"QA":"Question: What value was recorded?\nAnswer: {x}.",
"DIALOGUE":"Operator: Which value should I use?\nAssistant: Use {x}.",
"TECH":"SYSTEM_RECORD\nfield=value\nvalue={x}\nEND_RECORD",
"STORY":"After checking the sealed note, the researcher found a single entry. It was {x}.",
"MINIMAL":"{x}"}
TRAIN=["DECL","QA","DIALOGUE","TECH","STORY"]
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def cs(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(EPS))
def canon(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
print("="*108);print("TEST 295 - PACK -> REALIZED DISPLACEMENT TRANSPORT OPERATOR X-RAY");print("="*108);print("START:",START)
print("[1/13] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
torch.cuda.synchronize();t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;PDT=next(model.parameters()).dtype;torch.cuda.synchronize()
if len(layers)!=TOTAL or H!=H_EXPECT or PDT!=torch.bfloat16:raise RuntimeError("Architecture/dtype mismatch")
print(f"OK | {MODEL_ID} | 28L | H={H} | {PDT} | {time.perf_counter()-t:.2f}s")
print("[2/13] WEIGHT SENTINEL AND FIXED RSS")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();CL=list(POOLS);CN=list(CTX);KEYS=[f"{c}:{x}" for c,p in POOLS.items() for x in p]
base=[IVME*(ZIRVE*np.exp(-SONUM*L)*(1+SONUM*L)+TABAN)/(ZIRVE+TABAN) for L in range(END+1)]
scale=TARGET_RSS/np.sqrt(sum(x*x for x in base));RHO=[x*scale for x in base];RSS=np.sqrt(sum(x*x for x in RHO))
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"RSS={RSS:.9f} | RHO0={RHO[0]*100:.3f}% | RHO25={RHO[25]*100:.3f}% | L26-L27 OFF")
def enc(x):
    s=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to(DEVICE)
@torch.inference_mode()
def capture(text):
    o=model(**enc(text),use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
print("[3/13] CACHE 360 ABSOLUTE STATES")
STATE={}
for cls,pool in POOLS.items():
    STATE[cls]={}
    for x in pool:STATE[cls][x]={c:capture(f.format(x=x)) for c,f in CTX.items()}
    print(cls,"READY")
print("[4/13] TEST285 EXTERNAL BANK FORGE")
RAW={}
for cls,pool in POOLS.items():
    banks=[b for b in CL if b!=cls]
    for target in pool:
        k=f"{cls}:{target}";RAW[k]={}
        for bank in banks:
            RAW[k][bank]={}
            for c in CN:
                RAW[k][bank][c]=[]
                for L in range(TOTAL):
                    h=STATE[cls][target][c][L]
                    RAW[k][bank][c].append(unit(torch.stack([unit(h-STATE[bank][r][c][L]) for r in POOLS[bank]]).mean(0)))
print("RAW READY")
print("[5/13] TEST285 BANK CLEAN")
BRES={}
for cls,pool in POOLS.items():
    kk=[f"{cls}:{x}" for x in pool];banks=[b for b in CL if b!=cls]
    for bank in banks:
        for c in CN:
            for L in range(TOTAL):
                mu=torch.stack([RAW[k][bank][c][L] for k in kk]).mean(0)
                for k in kk:
                    BRES.setdefault(k,{}).setdefault(bank,{}).setdefault(c,[None]*TOTAL)
                    BRES[k][bank][c][L]=unit(RAW[k][bank][c][L]-mu)
print("BANK CLEAN READY")
print("[6/13] TEST286 CONTEXT CLEAN")
BC={};CRES={}
for cls,pool in POOLS.items():
    banks=[b for b in CL if b!=cls]
    for target in pool:
        k=f"{cls}:{target}";BC[k]={}
        for c in CN:BC[k][c]=[unit(torch.stack([BRES[k][b][c][L] for b in banks]).mean(0)) for L in range(TOTAL)]
for cls,pool in POOLS.items():
    kk=[f"{cls}:{x}" for x in pool]
    for c in CN:
        for L in range(TOTAL):
            mu=torch.stack([BC[k][c][L] for k in kk]).mean(0)
            for k in kk:
                CRES.setdefault(k,{}).setdefault(c,[None]*TOTAL);CRES[k][c][L]=unit(BC[k][c][L]-mu)
print("CONTEXT CLEAN READY")
print("[7/13] FIVE CONTEXT COMPILER")
PACK={k:[unit(torch.stack([CRES[k][c][L] for c in TRAIN]).mean(0)) for L in range(TOTAL)] for k in KEYS}
print("60 TARGET PACKETS READY")
print("[8/13] CAUSAL WRITE")
AUD={"calls":0,"max_dev":0.0};ACTIVE=set()
def install(v):
    hs=[]
    for L in range(END+1):
        rho=RHO[L]
        def hook(mod,args,out,L=L,rho=rho):
            y=out[0] if isinstance(out,tuple) else out;z=y[:,-1,:].float();d=v[L].to(z.device)*z.norm(dim=-1,keepdim=True)*rho
            rel=float((d.norm(dim=-1)/(z.norm(dim=-1)+EPS)).max());AUD["calls"]+=1;AUD["max_dev"]=max(AUD["max_dev"],abs(rel-rho))
            yy=y.clone();yy[:,-1,:]=(z+d).to(y.dtype)
            return (yy,)+out[1:] if isinstance(out,tuple) else yy
        h=layers[L].register_forward_hook(hook);hs.append(h);ACTIVE.add(id(h))
    return hs
@torch.inference_mode()
def forward(v=None):
    hs=install(v) if v is not None else []
    try:
        o=model(**enc(BLIND),use_cache=False,output_hidden_states=True,return_dict=True)
        return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)],o.logits[0,-1].float().detach().clone()
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
BASE,LOG0=forward();WRITE={};LOG={}
for i,k in enumerate(KEYS):
    WRITE[k],LOG[k]=forward(PACK[k])
    if (i+1)%10==0:print(f"{i+1}/60")
print("[9/13] PACK -> DISPLACEMENT DECOMPOSITION")
W=model.lm_head.weight.detach().float();ROWS=[]
for k in KEYS:
    target=k.split(":",1)[1];tid=tok.encode(target,add_special_tokens=False)[0];head=unit(W[tid]);row={"key":k,"target":target,"token_id":tid,"layers":[]}
    for L in range(TOTAL):
        p=PACK[k][L];d=WRITE[k][L]-BASE[L];du=unit(d);pa=float(torch.dot(p,head));da=float(torch.dot(du,head))
        ppar=head*pa;dpar=head*float(torch.dot(du,head));porth=p-ppar;dorth=du-dpar
        row["layers"].append({"layer":L,"disp_pack":cs(du,p),"pack_head":pa,"disp_head":da,
        "pack_head_abs":abs(pa),"disp_head_abs":abs(da),"pack_ortho_norm":float(porth.norm()),"disp_ortho_norm":float(dorth.norm()),
        "ortho_alignment":cs(porth,dorth),"head_transfer_ratio":da/(pa if abs(pa)>1e-6 else (1e-6 if pa>=0 else -1e-6))})
    row["dlogit"]=float(LOG[k][tid]-LOG0[tid]);row["rank0"]=int((LOG0>LOG0[tid]).sum())+1;row["rank1"]=int((LOG[k]>LOG[k][tid]).sum())+1;ROWS.append(row)
print("[10/13] LAYERWISE TRANSPORT PROFILE")
PROFILE=[]
for L in range(TOTAL):
    rr=[r["layers"][L] for r in ROWS]
    q={"layer":L,"motor":"ON" if L<=END else "OFF","rho":RHO[L] if L<=END else 0.0,
    "disp_pack":float(np.mean([x["disp_pack"] for x in rr])),"pack_head":float(np.mean([x["pack_head"] for x in rr])),
    "disp_head":float(np.mean([x["disp_head"] for x in rr])),"ortho_alignment":float(np.mean([x["ortho_alignment"] for x in rr])),
    "head_transfer_ratio_median":float(np.median([x["head_transfer_ratio"] for x in rr if abs(x["pack_head"])>1e-3])),
    "head_retention_abs":float(np.mean([x["disp_head_abs"] for x in rr])/(np.mean([x["pack_head_abs"] for x in rr])+EPS))}
    PROFILE.append(q);print(f"L{L:02d} {q['motor']:3s} | D-P={q['disp_pack']:+.4f} | P-H={q['pack_head']:+.4f} | D-H={q['disp_head']:+.4f} | ORTH={q['ortho_alignment']:+.4f} | HRET={q['head_retention_abs']:.3f}")
print("[11/13] TERMINAL MISMATCH")
for L in [19,21,22,23,24,25,26,27]:
    q=PROFILE[L];print(f"L{L:02d} | PACK_HEAD={q['pack_head']:+.6f} | REALIZED_HEAD={q['disp_head']:+.6f} | GAP={q['pack_head']-q['disp_head']:+.6f} | HEAD_RETENTION={q['head_retention_abs']:.3f} | ORTHO={q['ortho_alignment']:+.6f}")
P27=PROFILE[27]
GLOBAL={"mean_dlogit":float(np.mean([r["dlogit"] for r in ROWS])),"positive_dlogit":float(np.mean([r["dlogit"]>0 for r in ROWS])),
"rank_improved":float(np.mean([r["rank1"]<r["rank0"] for r in ROWS])),"L27_disp_pack":P27["disp_pack"],"L27_pack_head":P27["pack_head"],
"L27_disp_head":P27["disp_head"],"L27_head_gap":P27["pack_head"]-P27["disp_head"],"L27_head_retention_abs":P27["head_retention_abs"],"L27_ortho_alignment":P27["ortho_alignment"]}
print("[12/13] CLASS L27 ATLAS")
ATLAS={}
for cls in POOLS:
    rr=[r for r in ROWS if r["key"].startswith(cls+":")];ll=[r["layers"][27] for r in rr]
    ATLAS[cls]={"disp_pack":float(np.mean([x["disp_pack"] for x in ll])),"pack_head":float(np.mean([x["pack_head"] for x in ll])),
    "disp_head":float(np.mean([x["disp_head"] for x in ll])),"ortho_alignment":float(np.mean([x["ortho_alignment"] for x in ll])),
    "head_retention_abs":float(np.mean([x["disp_head_abs"] for x in ll])/(np.mean([x["pack_head_abs"] for x in ll])+EPS)),
    "dlogit":float(np.mean([r["dlogit"] for r in rr])),"rank_improved":float(np.mean([r["rank1"]<r["rank0"] for r in rr]))}
    a=ATLAS[cls];print(f"{cls:10s} | D-P={a['disp_pack']:+.4f} | P-H={a['pack_head']:+.4f} | D-H={a['disp_head']:+.4f} | HRET={a['head_retention_abs']:.3f} | ORTH={a['ortho_alignment']:+.4f} | DL={a['dlogit']:+.3f}")
print("[13/13] INTEGRITY AND SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("AkbasCore hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["max_dev"]>1e-6:raise RuntimeError("Dose audit failure")
R={"schema":"akbascore.test295.v1","test":"TEST 295","start":START,"end":utc(),"model":MODEL_ID,
"question":"Which target-head and orthogonal components of the compiled packet survive as realized downstream displacement?",
"blind_prompt":BLIND,"compiler":{"train_contexts":TRAIN,"targets":60,"external_banks":9},
"seasc":{"on_layers":[0,25],"off_layers":[26,27],"rss_definition":"sqrt(sum(rho_L^2))","rss":RSS,"rho0":RHO[0],"rho25":RHO[25]},
"profile":PROFILE,"global":GLOBAL,"class_atlas":ATLAS,"rows":ROWS,
"integrity":{"generation":False,"training":False,"optimization":False,"dose_tuning":False,"weight_update":False,"injection_calls":AUD["calls"],"max_dose_deviation":AUD["max_dev"],"akbascore_hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();run=f"T295-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{run}.json";tp=ROOT/f"{run}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2).encode())
out=["="*108,"TEST 295 - PACK -> REALIZED DISPLACEMENT TRANSPORT OPERATOR X-RAY","="*108]
for q in PROFILE:out.append(f"L{q['layer']:02d} {q['motor']} | D-P={q['disp_pack']:+.6f} | P-H={q['pack_head']:+.6f} | D-H={q['disp_head']:+.6f} | ORTH={q['ortho_alignment']:+.6f} | HRET={q['head_retention_abs']:.6f}")
out+=["",f"L27_DISP_PACK={GLOBAL['L27_disp_pack']:+.6f}",f"L27_PACK_HEAD={GLOBAL['L27_pack_head']:+.6f}",f"L27_DISP_HEAD={GLOBAL['L27_disp_head']:+.6f}",f"L27_HEAD_GAP={GLOBAL['L27_head_gap']:+.6f}",f"L27_HEAD_RETENTION={GLOBAL['L27_head_retention_abs']:.6f}",f"L27_ORTHO_ALIGNMENT={GLOBAL['L27_ortho_alignment']:+.6f}",f"MEAN_DLOGIT={GLOBAL['mean_dlogit']:+.6f}",f"POSITIVE_DLOGIT={GLOBAL['positive_dlogit']:.6f}",f"RANK_IMPROVED={GLOBAL['rank_improved']:.6f}",f"RSS={RSS:.9f}",f"CALLS={AUD['calls']}",f"MAX_DOSE_DEV={AUD['max_dev']:.3e}","NO GENERATION - NO TRAINING - NO OPTIMIZATION - NO DOSE TUNING - WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(out),encoding="utf-8")
print("="*108);print("TEST 295 COMPLETE")
print(f"L27 DISP->PACK={GLOBAL['L27_disp_pack']:+.4f} | PACK->HEAD={GLOBAL['L27_pack_head']:+.4f} | DISP->HEAD={GLOBAL['L27_disp_head']:+.4f}")
print(f"L27 HEAD GAP={GLOBAL['L27_head_gap']:+.4f} | HEAD RETENTION={GLOBAL['L27_head_retention_abs']:.3f} | ORTHO={GLOBAL['L27_ortho_alignment']:+.4f}")
print(f"DLOGIT={GLOBAL['mean_dlogit']:+.4f} | POS={GLOBAL['positive_dlogit']:.3f} | RANK_IMP={GLOBAL['rank_improved']:.3f}")
print(f"RSS={RSS:.9f} | CALLS={AUD['calls']} | MAX_DOSE_DEV={AUD['max_dev']:.3e}")
print("NO GENERATION | NO TRAINING | NO OPTIMIZATION | NO DOSE TUNING | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)

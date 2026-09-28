# ================================================================================================
# AKBASCORE - TEST 299
# EXTENDED FIXED-GRID CAUSAL DECISION-BOUNDARY CROSSING
# OWN-BRIDGE vs WRONG-BRIDGE - TEST298 WORKING BASELINE
# NO GENERATION - NO TRAINING - NO OPTIMIZATION - NO ADAPTIVE TUNING
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
SEED=299
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda");MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
BLIND="Return the single value represented by the internal state. Answer with only the value."
TOTAL=28;H_EXPECT=3584;EPS=1e-8;END=25;TARGET_RSS=.250235055
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20;DOSES=[0.16,0.20,0.24,0.28]
ROOT=Path("/content/AKBASCORE_TEST299") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST299")
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
def canon(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
print("="*108);print("TEST 299 - EXTENDED FIXED-GRID CAUSAL DECISION-BOUNDARY CROSSING");print("="*108);print("START:",START)
print("[1/14] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
torch.cuda.synchronize();t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;PDT=next(model.parameters()).dtype
if len(layers)!=TOTAL or H!=H_EXPECT or PDT!=torch.bfloat16:raise RuntimeError("Architecture/dtype mismatch")
torch.cuda.synchronize();print(f"OK | {MODEL_ID} | 28L | H={H} | {PDT} | {time.perf_counter()-t:.2f}s")
print("[2/14] WEIGHT SENTINEL AND FIXED RSS")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();CL=list(POOLS);CN=list(CTX);KEYS=[f"{c}:{x}" for c,p in POOLS.items() for x in p]
base=[IVME*(ZIRVE*np.exp(-SONUM*L)*(1+SONUM*L)+TABAN)/(ZIRVE+TABAN) for L in range(END+1)]
scale=TARGET_RSS/np.sqrt(sum(x*x for x in base));RHO=[x*scale for x in base];RSS=np.sqrt(sum(x*x for x in RHO))
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"RSS={RSS:.9f} | RHO0={RHO[0]*100:.3f}% | RHO25={RHO[25]*100:.3f}% | DOSES={DOSES}")
def enc(x):
    s=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to(DEVICE)
@torch.inference_mode()
def capture(text):
    o=model(**enc(text),use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
print("[3/14] CACHE 360 ABSOLUTE STATES")
STATE={}
for cls,pool in POOLS.items():
    STATE[cls]={}
    for x in pool:STATE[cls][x]={c:capture(f.format(x=x)) for c,f in CTX.items()}
    print(cls,"READY")
print("[4/14] TEST285 EXTERNAL BANK FORGE")
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
print("[5/14] TEST285 BANK CLEAN")
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
print("[6/14] TEST286 CONTEXT CLEAN")
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
                CRES.setdefault(k,{}).setdefault(c,[None]*TOTAL)
                CRES[k][c][L]=unit(BC[k][c][L]-mu)
print("CONTEXT CLEAN READY")
print("[7/14] FIVE CONTEXT COMPILER")
PACK={k:[unit(torch.stack([CRES[k][c][L] for c in TRAIN]).mean(0)) for L in range(TOTAL)] for k in KEYS}
print("60 TARGET PACKETS READY")
print("[8/14] FIXED READOUT BRIDGES")
W=model.lm_head.weight.detach().float();HEAD={};BRIDGE={}
for k in KEYS:
    target=k.split(":",1)[1];ids=tok.encode(target,add_special_tokens=False);tid=ids[0];h=unit(W[tid]);HEAD[k]=(tid,h,ids)
    a=torch.dot(PACK[k][26],h);BRIDGE[k]=unit(h*a)
print("60 BRIDGES READY")
print("[9/14] CAUSAL ENGINE")
AUD={"seasc_calls":0,"bridge_calls":0,"max_seasc_dev":0.0,"max_bridge_dev":0.0};ACTIVE=set()
def install(packet,bridge=None,dose=0.0):
    hs=[]
    for L in range(END+1):
        rho=RHO[L]
        def hook(mod,args,out,L=L,rho=rho):
            y=out[0] if isinstance(out,tuple) else out;z=y[:,-1,:].float();d=packet[L].to(z.device)*z.norm(dim=-1,keepdim=True)*rho
            rel=float((d.norm(dim=-1)/(z.norm(dim=-1)+EPS)).max());AUD["seasc_calls"]+=1;AUD["max_seasc_dev"]=max(AUD["max_seasc_dev"],abs(rel-rho))
            yy=y.clone();yy[:,-1,:]=(z+d).to(y.dtype);return (yy,)+out[1:] if isinstance(out,tuple) else yy
        h=layers[L].register_forward_hook(hook);hs.append(h);ACTIVE.add(id(h))
    if bridge is not None and dose>0:
        def bhook(mod,args,out):
            y=out[0] if isinstance(out,tuple) else out;z=y[:,-1,:].float();d=bridge.to(z.device)*z.norm(dim=-1,keepdim=True)*dose
            rel=float((d.norm(dim=-1)/(z.norm(dim=-1)+EPS)).max());AUD["bridge_calls"]+=1;AUD["max_bridge_dev"]=max(AUD["max_bridge_dev"],abs(rel-dose))
            yy=y.clone();yy[:,-1,:]=(z+d).to(y.dtype);return (yy,)+out[1:] if isinstance(out,tuple) else yy
        h=layers[26].register_forward_hook(bhook);hs.append(h);ACTIVE.add(id(h))
    return hs
@torch.inference_mode()
def run(packet,bridge,dose):
    hs=install(packet,bridge,dose)
    try:return model(**enc(BLIND),use_cache=False,return_dict=True).logits[0,-1].float().detach().clone()
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
print("[10/14] EXTENDED OWN / WRONG ASSAY")
ROWS=[]
for i,k in enumerate(KEYS):
    wrong=KEYS[(i+1)%len(KEYS)];tid,_,ids=HEAD[k]
    for dose in DOSES:
        lo=run(PACK[k],BRIDGE[k],dose);lw=run(PACK[k],BRIDGE[wrong],dose)
        def assay(x):
            tv=float(x[tid]);mask=x.clone();mask[tid]=-float("inf");cid=int(torch.argmax(mask));cv=float(x[cid])
            return tv,cid,tok.decode([cid]),cv,tv-cv,int((x>x[tid]).sum())+1,bool(tv>cv)
        a=assay(lo);b=assay(lw)
        ROWS.append({"key":k,"target":k.split(":",1)[1],"token_ids":ids,"wrong_bridge":wrong,"dose":dose,
        "own_target_logit":a[0],"own_competitor_id":a[1],"own_competitor":a[2],"own_competitor_logit":a[3],"own_margin":a[4],"own_rank":a[5],"own_top1":a[6],
        "wrong_target_logit":b[0],"wrong_competitor_id":b[1],"wrong_competitor":b[2],"wrong_competitor_logit":b[3],"wrong_margin":b[4],"wrong_rank":b[5],"wrong_top1":b[6]})
    if (i+1)%10==0:print(f"{i+1}/60")
print("[11/14] DOSE RESPONSE")
PROFILE={}
for d in DOSES:
    rr=[r for r in ROWS if r["dose"]==d]
    PROFILE[str(d)]={"own_margin_mean":float(np.mean([r["own_margin"] for r in rr])),"own_margin_median":float(np.median([r["own_margin"] for r in rr])),
    "wrong_margin_mean":float(np.mean([r["wrong_margin"] for r in rr])),"wrong_margin_median":float(np.median([r["wrong_margin"] for r in rr])),
    "own_top1":float(np.mean([r["own_top1"] for r in rr])),"wrong_top1":float(np.mean([r["wrong_top1"] for r in rr])),
    "own_rank_median":float(np.median([r["own_rank"] for r in rr])),"wrong_rank_median":float(np.median([r["wrong_rank"] for r in rr]))}
    q=PROFILE[str(d)];print(f"DOSE={d:.2f} | OWN_M={q['own_margin_mean']:+.3f} med={q['own_margin_median']:+.3f} | WRONG_M={q['wrong_margin_mean']:+.3f} | TOP1 OWN={q['own_top1']:.3f} WRONG={q['wrong_top1']:.3f} | RANK OWN={q['own_rank_median']:.1f} WRONG={q['wrong_rank_median']:.1f}")
print("[12/14] CROSSING AND SPECIFICITY")
CROSS={}
for k in KEYS:
    rr=sorted([r for r in ROWS if r["key"]==k],key=lambda x:x["dose"])
    od=next((r["dose"] for r in rr if r["own_top1"]),None);wd=next((r["dose"] for r in rr if r["wrong_top1"]),None)
    CROSS[k]={"own_first_top1_dose":od,"wrong_first_top1_dose":wd}
for d in DOSES:
    rr=[r for r in ROWS if r["dose"]==d];own=sum(r["own_top1"] for r in rr);wrong=sum(r["wrong_top1"] for r in rr)
    print(f"DOSE={d:.2f} | OWN TOP1={own:02d}/60 | WRONG TOP1={wrong:02d}/60 | SPECIFIC ADVANTAGE={own-wrong:+d}")
print("[13/14] CLASS ATLAS")
ATLAS={}
for cls in POOLS:
    ATLAS[cls]={}
    for d in DOSES:
        rr=[r for r in ROWS if r["key"].startswith(cls+":") and r["dose"]==d]
        ATLAS[cls][str(d)]={"own_margin":float(np.mean([r["own_margin"] for r in rr])),"wrong_margin":float(np.mean([r["wrong_margin"] for r in rr])),
        "own_top1":float(np.mean([r["own_top1"] for r in rr])),"wrong_top1":float(np.mean([r["wrong_top1"] for r in rr])),"own_rank":float(np.median([r["own_rank"] for r in rr]))}
    z=ATLAS[cls][str(DOSES[-1])];print(f"{cls:10s} | D={DOSES[-1]:.2f} | OWN_M={z['own_margin']:+.3f} | WRONG_M={z['wrong_margin']:+.3f} | TOP1={z['own_top1']:.3f}/{z['wrong_top1']:.3f} | RANK={z['own_rank']:.1f}")
print("[14/14] INTEGRITY AND SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["max_seasc_dev"]>1e-6 or AUD["max_bridge_dev"]>1e-6:raise RuntimeError("Dose audit failure")
R={"schema":"akbascore.test299.v1","test":"TEST 299","start":START,"end":utc(),"model":MODEL_ID,
"question":"Does the pre-registered extended bridge-dose grid cross the Top-1 boundary selectively for OWN bridges while WRONG bridges remain controlled?",
"compiler":{"targets":60,"train_contexts":TRAIN,"external_banks":9},"seasc":{"on_layers":[0,25],"bridge_layer":26,"layer27":"OFF","rss":RSS,"bridge_doses":DOSES},
"profile":PROFILE,"crossing":CROSS,"class_atlas":ATLAS,"rows":ROWS,
"integrity":{"generation":False,"training":False,"optimization":False,"adaptive_tuning":False,"weight_update":False,
"seasc_calls":AUD["seasc_calls"],"bridge_calls":AUD["bridge_calls"],"max_seasc_dev":AUD["max_seasc_dev"],"max_bridge_dev":AUD["max_bridge_dev"],
"hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();runid=f"T299-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{runid}.json";tp=ROOT/f"{runid}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False).encode())
out=["="*108,"TEST 299 - EXTENDED FIXED-GRID CAUSAL DECISION-BOUNDARY CROSSING","="*108]
for d in DOSES:
    q=PROFILE[str(d)];out.append(f"DOSE={d:.2f} | OWN_M={q['own_margin_mean']:+.4f} | WRONG_M={q['wrong_margin_mean']:+.4f} | TOP1 OWN={q['own_top1']:.4f} WRONG={q['wrong_top1']:.4f} | RANK OWN={q['own_rank_median']:.1f} WRONG={q['wrong_rank_median']:.1f}")
out+=["",f"RSS={RSS:.9f}",f"DOSES={DOSES}",f"SEASC_CALLS={AUD['seasc_calls']}",f"BRIDGE_CALLS={AUD['bridge_calls']}",
"NO GENERATION - NO TRAINING - NO OPTIMIZATION - NO ADAPTIVE TUNING - WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(out),encoding="utf-8")
print("="*108);print("TEST 299 COMPLETE")
for d in DOSES:
    q=PROFILE[str(d)];print(f"D={d:.2f} | OWN M={q['own_margin_mean']:+.3f} TOP1={q['own_top1']:.3f} RANK={q['own_rank_median']:.1f} | WRONG M={q['wrong_margin_mean']:+.3f} TOP1={q['wrong_top1']:.3f}")
print(f"RSS={RSS:.9f} | SEASC_CALLS={AUD['seasc_calls']} | BRIDGE_CALLS={AUD['bridge_calls']}")
print("NO GENERATION | NO TRAINING | NO OPTIMIZATION | NO ADAPTIVE TUNING | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)

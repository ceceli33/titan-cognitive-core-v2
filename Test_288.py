# ================================================================================================
# AKBASCORE - TEST 288
# BLIND TARGET IDENTIFICATION AFTER CAUSAL LATENT WRITE
# TEST287 COMPILER - SEASC L0-L25 - L26-L27 OFF - 60-WAY AND 6-WAY IDENTIFICATION
# NO TRAINING - NO CLASSIFIER - NO SELECTION - COSINE ONLY
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
SEED=288
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda");MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL=28;H_EXPECT=3584;EPS=1e-8;REPORT=[19,23,25,26,27]
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20;END=25;TARGET_RSS=.250235055
ROOT=Path("/content/AKBASCORE_TEST288") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST288")
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
TRAIN_CTX=["DECL","QA","DIALOGUE","TECH","STORY"];BLIND="MINIMAL"
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(EPS))
def canon(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
print("="*108);print("TEST 288 - BLIND TARGET IDENTIFICATION AFTER CAUSAL LATENT WRITE");print("="*108);print("START:",START)
print("[1/13] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
torch.cuda.synchronize();t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;PDT=next(model.parameters()).dtype;torch.cuda.synchronize()
if len(layers)!=TOTAL or H!=H_EXPECT or PDT!=torch.bfloat16:raise RuntimeError("Architecture or dtype mismatch")
print(f"OK | {MODEL_ID} | 28L | H={H} | {PDT} | {time.perf_counter()-t:.2f}s")
print("[2/13] WEIGHT SENTINEL AND FIXED RSS")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();CL=list(POOLS);CN=list(CTX);KEYS=[f"{c}:{x}" for c,p in POOLS.items() for x in p];IDX={k:i for i,k in enumerate(KEYS)}
base=[IVME*(ZIRVE*np.exp(-SONUM*L)*(1+SONUM*L)+TABAN)/(ZIRVE+TABAN) for L in range(END+1)]
scale=TARGET_RSS/np.sqrt(sum(r*r for r in base));RHO=[r*scale for r in base];RSS=np.sqrt(sum(r*r for r in RHO))
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"SEASC L0-L25 | L26-L27 OFF | RSS={RSS:.9f} | RHO0={RHO[0]*100:.3f}% | RHO25={RHO[25]*100:.3f}%")
if abs(RSS-TARGET_RSS)>1e-9:raise RuntimeError("RSS mismatch")
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
                    ht=STATE[cls][target][c][L]
                    RAW[k][bank][c].append(unit(torch.stack([unit(ht-STATE[bank][r][c][L]) for r in POOLS[bank]]).mean(0)))
print("RAW FORGE READY")
print("[5/13] TEST285 CLASS COMMON REMOVAL")
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
print("[6/13] TEST286 BANK CONSENSUS AND CONTEXT CLEANING")
BC={};CRES={}
for cls,pool in POOLS.items():
    banks=[b for b in CL if b!=cls]
    for target in pool:
        k=f"{cls}:{target}";BC[k]={}
        for c in CN:
            BC[k][c]=[unit(torch.stack([BRES[k][b][c][L] for b in banks]).mean(0)) for L in range(TOTAL)]
for cls,pool in POOLS.items():
    kk=[f"{cls}:{x}" for x in pool]
    for c in CN:
        for L in range(TOTAL):
            mu=torch.stack([BC[k][c][L] for k in kk]).mean(0)
            for k in kk:
                CRES.setdefault(k,{}).setdefault(c,[None]*TOTAL)
                CRES[k][c][L]=unit(BC[k][c][L]-mu)
print("CONTEXT CLEAN READY")
print("[7/13] FIVE CONTEXT COMPILER - MINIMAL HELD OUT")
PACK={}
for k in KEYS:PACK[k]=[unit(torch.stack([CRES[k][c][L] for c in TRAIN_CTX]).mean(0)) for L in range(TOTAL)]
loo=[cos(PACK[k][27],CRES[k][BLIND][27]) for k in KEYS]
print(f"L27 HELDOUT MINIMAL | MEAN={np.mean(loo):+.4f} | MED={np.median(loo):+.4f} | P10={np.percentile(loo,10):+.4f} | POS={np.mean(np.array(loo)>0):.3f}")
print("[8/13] BLIND BASELINE AND CAUSAL WRITE")
BASE=capture("");AUD={"calls":0,"max_dev":0.0};ACTIVE=set()
def install(vecs):
    hs=[]
    for L in range(END+1):
        rho=RHO[L]
        def hook(mod,args,out,L=L,rho=rho):
            y=out[0] if isinstance(out,tuple) else out;z=y[:,-1,:].float();d=vecs[L].to(z.device)*z.norm(dim=-1,keepdim=True)*rho
            rel=float((d.norm(dim=-1)/(z.norm(dim=-1)+EPS)).max());AUD["calls"]+=1;AUD["max_dev"]=max(AUD["max_dev"],abs(rel-rho))
            yy=y.clone();yy[:,-1,:]=(z+d).to(y.dtype)
            return (yy,)+out[1:] if isinstance(out,tuple) else yy
        h=layers[L].register_forward_hook(hook);hs.append(h);ACTIVE.add(id(h))
    return hs
@torch.inference_mode()
def write(vecs):
    hs=install(vecs)
    try:
        o=model(**enc(""),use_cache=False,output_hidden_states=True,return_dict=True)
        return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
WRITE={}
for i,k in enumerate(KEYS,1):
    WRITE[k]=write(PACK[k])
    if i%10==0:print(f"{i}/60")
print("[9/13] HELD OUT MINIMAL CANDIDATE LIBRARY")
LIB={}
for L in REPORT:LIB[L]=torch.stack([CRES[k][BLIND][L] for k in KEYS])
print("60 HELD OUT TARGET CANDIDATES READY")
def rank_metrics(scores,true_idx,candidates):
    order=sorted(candidates,key=lambda j:(-scores[j],j));rank=order.index(true_idx)+1
    top=order[0];true=scores[true_idx];best_wrong=max(scores[j] for j in candidates if j!=true_idx)
    return rank,top,true,best_wrong,true-best_wrong
print("[10/13] 60 WAY BLIND IDENTIFICATION")
WAY60={}
for L in REPORT:
    ranks=[];marg=[];top1=0
    for k in KEYS:
        d=unit(WRITE[k][L]-BASE[L]);scores=[cos(d,LIB[L][j]) for j in range(len(KEYS))]
        r,top,ts,bw,m=rank_metrics(scores,IDX[k],list(range(len(KEYS))));ranks.append(r);marg.append(m);top1+=int(top==IDX[k])
    WAY60[str(L)]={"top1":top1/60,"mrr":float(np.mean([1/r for r in ranks])),"median_rank":float(np.median(ranks)),"mean_margin":float(np.mean(marg))}
    x=WAY60[str(L)];print(f"L{L:02d} | TOP1={x['top1']:.3f} | MRR={x['mrr']:.4f} | MEDRANK={x['median_rank']:.1f} | MARGIN={x['mean_margin']:+.4f}")
print("[11/13] SAME CLASS 6 WAY IDENTIFICATION")
WAY6={}
for L in REPORT:
    ranks=[];marg=[];top1=0
    for k in KEYS:
        cls=k.split(":",1)[0];cand=[IDX[f"{cls}:{x}"] for x in POOLS[cls]]
        d=unit(WRITE[k][L]-BASE[L]);scores=[cos(d,LIB[L][j]) for j in range(len(KEYS))]
        r,top,ts,bw,m=rank_metrics(scores,IDX[k],cand);ranks.append(r);marg.append(m);top1+=int(top==IDX[k])
    WAY6[str(L)]={"top1":top1/60,"mrr":float(np.mean([1/r for r in ranks])),"median_rank":float(np.median(ranks)),"mean_margin":float(np.mean(marg))}
    x=WAY6[str(L)];print(f"L{L:02d} | TOP1={x['top1']:.3f} | MRR={x['mrr']:.4f} | MEDRANK={x['median_rank']:.1f} | MARGIN={x['mean_margin']:+.4f}")
print("[12/13] L27 CLASS ATLAS AND NULL CONTROL")
ATLAS={}
for cls,pool in POOLS.items():
    ok60=ok6=0;r60=[];r6=[];ms=[]
    for target in pool:
        k=f"{cls}:{target}";d=unit(WRITE[k][27]-BASE[27]);scores=[cos(d,LIB[27][j]) for j in range(60)]
        a=rank_metrics(scores,IDX[k],list(range(60)));cand=[IDX[f"{cls}:{x}"] for x in pool];b=rank_metrics(scores,IDX[k],cand)
        ok60+=int(a[1]==IDX[k]);ok6+=int(b[1]==IDX[k]);r60.append(a[0]);r6.append(b[0]);ms.append(a[4])
    ATLAS[cls]={"top1_60":ok60/6,"mrr_60":float(np.mean([1/r for r in r60])),"top1_6":ok6/6,"mrr_6":float(np.mean([1/r for r in r6])),"margin_60":float(np.mean(ms))}
    x=ATLAS[cls];print(f"{cls:10s} | 60WAY={x['top1_60']:.3f} | MRR60={x['mrr_60']:.3f} | 6WAY={x['top1_6']:.3f} | MRR6={x['mrr_6']:.3f}")
g=torch.Generator(device=DEVICE);g.manual_seed(SEED+999)
rnd=torch.randn(60,H,device=DEVICE,dtype=torch.float32,generator=g);rnd/=rnd.norm(dim=1,keepdim=True).clamp_min(EPS)
NULL60=NULL6=0
for i,k in enumerate(KEYS):
    s=[cos(rnd[i],LIB[27][j]) for j in range(60)]
    NULL60+=int(int(np.argmax(s))==i);cls=k.split(":",1)[0];cand=[IDX[f"{cls}:{x}"] for x in POOLS[cls]]
    NULL6+=int(max(cand,key=lambda j:s[j])==i)
NULL={"random_top1_60":NULL60/60,"chance_60":1/60,"random_top1_6":NULL6/60,"chance_6":1/6}
print(f"RANDOM L27 | 60WAY={NULL['random_top1_60']:.3f} CHANCE={NULL['chance_60']:.3f} | 6WAY={NULL['random_top1_6']:.3f} CHANCE={NULL['chance_6']:.3f}")
print("[13/13] INTEGRITY AND SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("AkbasCore hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["max_dev"]>1e-6:raise RuntimeError("Dose audit failure")
R={"schema":"akbascore.test288.v1","test":"TEST 288","start":START,"end":utc(),"model":MODEL_ID,
"question":"Does the blind terminal displacement after causal write identify the correct held-out target among all 60 targets and among six same-class targets?",
"compiler":{"train_contexts":TRAIN_CTX,"heldout_context":BLIND,"targets":60,"external_banks":9},
"seasc":{"on_layers":[0,25],"off_layers":[26,27],"rss_definition":"sqrt(sum(rho_L^2))","rss":RSS,"rho0":RHO[0],"rho25":RHO[25]},
"heldout_l27":{"mean":float(np.mean(loo)),"median":float(np.median(loo)),"p10":float(np.percentile(loo,10))},
"identification_60way":WAY60,"identification_6way":WAY6,"l27_class_atlas":ATLAS,"null":NULL,
"integrity":{"training":False,"classifier":False,"selection":False,"weight_update":False,"injection_calls":AUD["calls"],"max_dose_deviation":AUD["max_dev"],"akbascore_hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();run=f"T288-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{run}.json";tp=ROOT/f"{run}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2).encode())
out=["="*108,"TEST 288 - BLIND TARGET IDENTIFICATION AFTER CAUSAL LATENT WRITE","="*108]
for L in REPORT:out.append(f"L{L:02d} 60_TOP1={WAY60[str(L)]['top1']:.6f} 60_MRR={WAY60[str(L)]['mrr']:.6f} 6_TOP1={WAY6[str(L)]['top1']:.6f} 6_MRR={WAY6[str(L)]['mrr']:.6f}")
out+=["",f"RSS={RSS:.9f}",f"INJECTION_CALLS={AUD['calls']}",f"MAX_DOSE_DEVIATION={AUD['max_dev']:.3e}","NO TRAINING - NO CLASSIFIER - NO SELECTION - WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(out),encoding="utf-8")
print("="*108);print("TEST 288 COMPLETE")
print("BLIND TARGET IDENTIFICATION | 60 WAY + SAME CLASS 6 WAY | COSINE ONLY")
print(f"RSS={RSS:.9f} | CALLS={AUD['calls']} | MAX_DOSE_DEV={AUD['max_dev']:.3e}")
print("NO TRAINING | NO CLASSIFIER | NO SELECTION | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)

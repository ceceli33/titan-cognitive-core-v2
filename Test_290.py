# ================================================================================================
# AKBASCORE - TEST 290
# LOGIT READOUT X-RAY
# TEST289 COMPILER - SEASC L0-L25 - L26-L27 OFF
# FIRST TOKEN TARGET LOGIT / RANK / DELTA - NULL / TARGET / WRONG
# NO GENERATION - NO TRAINING - NO CLASSIFIER - NO SELECTION
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
SEED=290
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda");MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
BLIND_PROMPT="Return the single value represented by the internal state. Answer with only the value."
TOTAL=28;H_EXPECT=3584;EPS=1e-8;IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20;END=25;TARGET_RSS=.250235055
ROOT=Path("/content/AKBASCORE_TEST290") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST290")
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
TRAIN_CTX=["DECL","QA","DIALOGUE","TECH","STORY"]
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def canon(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
print("="*108);print("TEST 290 - LOGIT READOUT X-RAY");print("="*108);print("START:",START)
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
FP0=fp();CL=list(POOLS);CN=list(CTX);KEYS=[f"{c}:{x}" for c,p in POOLS.items() for x in p]
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
print("[7/13] FIVE CONTEXT TARGET COMPILER")
PACK={k:[unit(torch.stack([CRES[k][c][L] for c in TRAIN_CTX]).mean(0)) for L in range(TOTAL)] for k in KEYS}
print("60 TARGET PACKETS READY")
print("[8/13] TARGET FIRST TOKEN MAP")
TID={};TDISP={}
for k in KEYS:
    target=k.split(":",1)[1];ids=tok.encode(target,add_special_tokens=False)
    if not ids:raise RuntimeError("Empty target tokenization")
    TID[k]=ids[0];TDISP[k]=tok.decode([ids[0]])
    print(f"{k:20s} | ID={ids[0]:6d} | FIRST={TDISP[k]!r} | NTOK={len(ids)}")
print("[9/13] NULL FIRST TOKEN LOGITS")
AUD={"calls":0,"max_dev":0.0};ACTIVE=set()
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
def logits(vecs=None):
    hs=install(vecs) if vecs is not None else []
    try:return model(**enc(BLIND_PROMPT),use_cache=False,return_dict=True).logits[0,-1].float().detach().clone()
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
NULL=logits();null_top=int(NULL.argmax());print("NULL TOP1:",null_top,repr(tok.decode([null_top])),f"LOGIT={float(NULL[null_top]):+.4f}")
print("[10/13] TARGET AND WRONG LOGITS")
ROWS=[]
for i,k in enumerate(KEYS):
    wk=KEYS[(i+17)%60];lt=logits(PACK[k]);lw=logits(PACK[wk]);tid=TID[k];wid=TID[wk]
    rank0=int((NULL>NULL[tid]).sum())+1;rankt=int((lt>lt[tid]).sum())+1;rankw=int((lw>lw[tid]).sum())+1
    ROWS.append({"key":k,"target":k.split(":",1)[1],"target_token_id":tid,"target_first_token":TDISP[k],"wrong_key":wk,"wrong_token_id":wid,
    "null_logit":float(NULL[tid]),"target_logit":float(lt[tid]),"wrong_condition_target_logit":float(lw[tid]),
    "delta_target_vs_null":float(lt[tid]-NULL[tid]),"delta_wrong_vs_null":float(lw[tid]-NULL[tid]),"rank_null":rank0,"rank_target":rankt,"rank_wrong":rankw,
    "target_top1":int(lt.argmax())==tid,"target_top10":rankt<=10,"target_top100":rankt<=100,
    "wrong_token_delta_under_target":float(lt[wid]-NULL[wid])})
    r=ROWS[-1];print(f"{i+1:02d}/60 | {k:20s} | DLOGIT={r['delta_target_vs_null']:+.3f} | RANK {rank0}->{rankt} | TOP={tok.decode([int(lt.argmax())])!r}")
print("[11/13] GLOBAL READOUT")
def avg(f):return float(np.mean([f(r) for r in ROWS]))
G={"mean_delta_target":avg(lambda r:r["delta_target_vs_null"]),"median_delta_target":float(np.median([r["delta_target_vs_null"] for r in ROWS])),
"positive_delta_fraction":avg(lambda r:r["delta_target_vs_null"]>0),"mean_delta_wrong_condition":avg(lambda r:r["delta_wrong_vs_null"]),
"mean_rank_null":avg(lambda r:r["rank_null"]),"mean_rank_target":avg(lambda r:r["rank_target"]),"median_rank_null":float(np.median([r["rank_null"] for r in ROWS])),
"median_rank_target":float(np.median([r["rank_target"] for r in ROWS])),"rank_improved_fraction":avg(lambda r:r["rank_target"]<r["rank_null"]),
"top1":avg(lambda r:r["target_top1"]),"top10":avg(lambda r:r["target_top10"]),"top100":avg(lambda r:r["target_top100"])}
for k,v in G.items():print(f"{k.upper():28s} {v:.4f}")
print("[12/13] CLASS ATLAS")
ATLAS={}
for cls in POOLS:
    rr=[r for r in ROWS if r["key"].startswith(cls+":")]
    ATLAS[cls]={"delta":float(np.mean([r["delta_target_vs_null"] for r in rr])),"positive":float(np.mean([r["delta_target_vs_null"]>0 for r in rr])),
    "rank_null":float(np.mean([r["rank_null"] for r in rr])),"rank_target":float(np.mean([r["rank_target"] for r in rr])),
    "improved":float(np.mean([r["rank_target"]<r["rank_null"] for r in rr])),"top100":float(np.mean([r["target_top100"] for r in rr]))}
    x=ATLAS[cls];print(f"{cls:10s} | DLOGIT={x['delta']:+.3f} | POS={x['positive']:.3f} | RANK={x['rank_null']:.1f}->{x['rank_target']:.1f} | IMP={x['improved']:.3f} | TOP100={x['top100']:.3f}")
print("[13/13] INTEGRITY AND SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("AkbasCore hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["max_dev"]>1e-6:raise RuntimeError("Dose audit failure")
R={"schema":"akbascore.test290.v1","test":"TEST 290","start":START,"end":utc(),"model":MODEL_ID,
"question":"Does causal target-latent write increase the first-token vocabulary logit and rank of the corresponding target before greedy decoding?",
"blind_prompt":BLIND_PROMPT,"compiler":{"train_contexts":TRAIN_CTX,"targets":60,"external_banks":9},
"seasc":{"on_layers":[0,25],"off_layers":[26,27],"rss_definition":"sqrt(sum(rho_L^2))","rss":RSS,"rho0":RHO[0],"rho25":RHO[25]},
"null":{"top1_id":null_top,"top1_token":tok.decode([null_top]),"top1_logit":float(NULL[null_top])},"global":G,"class_atlas":ATLAS,"rows":ROWS,
"integrity":{"generation":False,"training":False,"classifier":False,"selection":False,"weight_update":False,"injection_calls":AUD["calls"],"max_dose_deviation":AUD["max_dev"],"akbascore_hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();run=f"T290-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{run}.json";tp=ROOT/f"{run}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2).encode())
out=["="*108,"TEST 290 - LOGIT READOUT X-RAY","="*108,f"NULL_TOP1={tok.decode([null_top])!r}"]
for r in ROWS:out.append(f"{r['key']} | DLOGIT={r['delta_target_vs_null']:+.6f} | RANK={r['rank_null']}->{r['rank_target']} | TOP1={int(r['target_top1'])} | TOP10={int(r['target_top10'])} | TOP100={int(r['target_top100'])}")
out+=["",f"MEAN_DLOGIT={G['mean_delta_target']:+.6f}",f"POSITIVE={G['positive_delta_fraction']:.6f}",f"RANK_IMPROVED={G['rank_improved_fraction']:.6f}",f"TOP1={G['top1']:.6f}",f"TOP10={G['top10']:.6f}",f"TOP100={G['top100']:.6f}",f"RSS={RSS:.9f}",f"CALLS={AUD['calls']}",f"MAX_DOSE_DEV={AUD['max_dev']:.3e}","NO GENERATION - NO TRAINING - NO CLASSIFIER - NO SELECTION - WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(out),encoding="utf-8")
print("="*108);print("TEST 290 COMPLETE")
print(f"MEAN DLOGIT={G['mean_delta_target']:+.4f} | POS={G['positive_delta_fraction']:.3f} | RANK IMPROVED={G['rank_improved_fraction']:.3f} | TOP1={G['top1']:.3f} | TOP10={G['top10']:.3f} | TOP100={G['top100']:.3f}")
print(f"RSS={RSS:.9f} | CALLS={AUD['calls']} | MAX_DOSE_DEV={AUD['max_dev']:.3e}")
print("NO GENERATION | NO TRAINING | NO CLASSIFIER | NO SELECTION | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)

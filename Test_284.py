# ================================================================================================
# AKBASCORE - TEST 284
# TARGET IDENTITY VS CENTROID AVERAGING CONTROL
# WITHIN CLASS CONSENSUS VS EXACT CENTROID
# EXTERNAL REFERENCE BANKS AND CROSS CONTEXT TRANSFER
# NO STEERING - NO RETRIEVAL - NO TRAINING
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
SEED=284
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda");MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL=28;H_EXPECT=3584;EPS=1e-8;REPORT=[0,5,10,15,19,21,23,25,26,27]
ROOT=Path("/content/AKBASCORE_TEST284") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST284")
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
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(EPS))
def canon(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
print("="*108);print("TEST 284 - TARGET IDENTITY VS CENTROID AVERAGING CONTROL");print("="*108);print("START:",START)
print("[1/11] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if tv>=(4,56) else "torch_dtype"
torch.cuda.synchronize();t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16})
model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;PDT=next(model.parameters()).dtype;torch.cuda.synchronize()
if len(layers)!=TOTAL or H!=H_EXPECT or PDT!=torch.bfloat16:raise RuntimeError("Architecture or dtype mismatch")
print(f"OK | {MODEL_ID} | 28L | H={H} | {PDT} | {time.perf_counter()-t:.2f}s")
print("[2/11] WEIGHT SENTINEL AND DESIGN")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();CN=list(CTX);CL=list(POOLS);TARGETS=[(c,x) for c,p in POOLS.items() for x in p]
print("FP:",[f"{x:.4f}" for x in FP0]);print("CLASSES=10 | TARGETS=60 | CONTEXTS=6 | EXTERNAL_BANKS_PER_TARGET=9")
def enc(x):
    s=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to(DEVICE)
@torch.inference_mode()
def capture(text):
    o=model(**enc(text),use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
print("[3/11] CACHE 360 ABSOLUTE STATES")
STATE={}
for cls,pool in POOLS.items():
    STATE[cls]={}
    for x in pool:
        STATE[cls][x]={c:capture(f.format(x=x)) for c,f in CTX.items()}
    print(cls,"READY")
print("[4/11] WITHIN CLASS CONSENSUS VS EXACT CENTROID")
WC={};CENT={};EQ={}
for cls,pool in POOLS.items():
    for target in pool:
        k=f"{cls}:{target}";WC[k]=[];CENT[k]=[]
        for L in range(TOTAL):
            ds=[];cs=[];refs=[x for x in pool if x!=target]
            for c in CN:
                ht=STATE[cls][target][c][L]
                ds.extend([unit(ht-STATE[cls][r][c][L]) for r in refs])
                mu=torch.stack([STATE[cls][r][c][L] for r in refs]).mean(0)
                cs.append(unit(ht-mu))
            WC[k].append(unit(torch.stack(ds).mean(0)))
            CENT[k].append(unit(torch.stack(cs).mean(0)))
for L in REPORT:
    q=[cos(WC[k][L],CENT[k][L]) for k in WC]
    EQ[str(L)]={"mean":float(np.mean(q)),"min":float(np.min(q)),"p10":float(np.percentile(q,10))}
    x=EQ[str(L)];print(f"L{L:02d} | CONS_CENTROID={x['mean']:+.6f} | MIN={x['min']:+.6f} | P10={x['p10']:+.6f}")
print("[5/11] EXTERNAL REFERENCE BANK FORGE")
EXT={};EXTCTX={}
for cls,pool in POOLS.items():
    banks=[z for z in CL if z!=cls]
    for target in pool:
        k=f"{cls}:{target}";EXT[k]={};EXTCTX[k]={}
        for bank in banks:
            EXT[k][bank]=[]
            for L in range(TOTAL):
                v=[]
                for c in CN:
                    ht=STATE[cls][target][c][L]
                    for r in POOLS[bank]:v.append(unit(ht-STATE[bank][r][c][L]))
                EXT[k][bank].append(unit(torch.stack(v).mean(0)))
        for c in CN:
            EXTCTX[k][c]=[]
            for L in range(TOTAL):
                v=[];ht=STATE[cls][target][c][L]
                for bank in banks:
                    for r in POOLS[bank]:v.append(unit(ht-STATE[bank][r][c][L]))
                EXTCTX[k][c].append(unit(torch.stack(v).mean(0)))
print("EXTERNAL FORGE READY")
print("[6/11] EXTERNAL BANK INVARIANCE")
BANK={}
for L in REPORT:
    vals=[]
    for cls,pool in POOLS.items():
        banks=[z for z in CL if z!=cls]
        for target in pool:
            k=f"{cls}:{target}";vv=[EXT[k][b][L] for b in banks]
            vals.append(np.mean([cos(vv[i],vv[j]) for i in range(len(vv)) for j in range(i+1,len(vv))]))
    BANK[str(L)]={"mean":float(np.mean(vals)),"median":float(np.median(vals)),"p10":float(np.percentile(vals,10)),"min":float(np.min(vals)),"positive_fraction":float(np.mean(np.array(vals)>0))}
    x=BANK[str(L)];print(f"L{L:02d} | EXT_BANK={x['mean']:+.4f} | MED={x['median']:+.4f} | P10={x['p10']:+.4f} | MIN={x['min']:+.4f}")
print("[7/11] LEAVE ONE EXTERNAL BANK OUT")
LOO={}
for L in REPORT:
    vals=[]
    for cls,pool in POOLS.items():
        banks=[z for z in CL if z!=cls]
        for target in pool:
            k=f"{cls}:{target}"
            for held in banks:
                tr=unit(torch.stack([EXT[k][b][L] for b in banks if b!=held]).mean(0))
                vals.append(cos(tr,EXT[k][held][L]))
    LOO[str(L)]={"mean":float(np.mean(vals)),"median":float(np.median(vals)),"p10":float(np.percentile(vals,10)),"min":float(np.min(vals))}
    x=LOO[str(L)];print(f"L{L:02d} | EXT_LOO={x['mean']:+.4f} | MED={x['median']:+.4f} | P10={x['p10']:+.4f}")
print("[8/11] EXTERNAL TARGET VS WITHIN CLASS TARGET")
ALIGN={};EXTCONS={}
for k in EXT:
    EXTCONS[k]=[]
    for L in range(TOTAL):EXTCONS[k].append(unit(torch.stack([EXT[k][b][L] for b in EXT[k]]).mean(0)))
for L in REPORT:
    q=[cos(EXTCONS[k][L],WC[k][L]) for k in EXT]
    ALIGN[str(L)]={"mean":float(np.mean(q)),"median":float(np.median(q)),"p10":float(np.percentile(q,10)),"min":float(np.min(q))}
    x=ALIGN[str(L)];print(f"L{L:02d} | EXTERNAL_WITHIN={x['mean']:+.4f} | MED={x['median']:+.4f} | P10={x['p10']:+.4f}")
print("[9/11] CROSS CONTEXT EXTERNAL TARGET")
XCTX={}
for L in REPORT:
    vals=[]
    for k in EXTCTX:
        vv=[EXTCTX[k][c][L] for c in CN]
        vals.append(np.mean([cos(vv[i],vv[j]) for i in range(6) for j in range(i+1,6)]))
    XCTX[str(L)]={"mean":float(np.mean(vals)),"median":float(np.median(vals)),"p10":float(np.percentile(vals,10)),"min":float(np.min(vals))}
    x=XCTX[str(L)];print(f"L{L:02d} | EXT_CTX={x['mean']:+.4f} | MED={x['median']:+.4f} | P10={x['p10']:+.4f}")
print("[10/11] IDENTITY SEPARATION AND RANDOM BASELINE")
SEP={};NULL={};keys=list(EXTCONS);g=torch.Generator(device=DEVICE);g.manual_seed(SEED+999999)
for L in REPORT:
    same=[];cross=[]
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            q=abs(cos(EXTCONS[keys[i]][L],EXTCONS[keys[j]][L]))
            if keys[i].split(":",1)[0]==keys[j].split(":",1)[0]:same.append(q)
            else:cross.append(q)
    RND=torch.randn(len(keys),H,device=DEVICE,dtype=torch.float32,generator=g)
    RND/=RND.norm(dim=1,keepdim=True).clamp_min(EPS)
    rnd=[abs(cos(RND[i],RND[j])) for i in range(len(keys)) for j in range(i+1,len(keys))]
    SEP[str(L)]={"same_abs":float(np.mean(same)),"cross_abs":float(np.mean(cross)),"p95_cross":float(np.percentile(cross,95))}
    NULL[str(L)]={"abs_cos":float(np.mean(rnd))}
    print(f"L{L:02d} | SAME_ABS={np.mean(same):.4f} | CROSS_ABS={np.mean(cross):.4f} | RANDOM={np.mean(rnd):.4f}")
print("[11/11] INTEGRITY AND SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
R={"schema":"akbascore.test284.v1","test":"TEST 284","start":START,"end":utc(),"model":MODEL_ID,
"question":"Is TEST283 reference robustness merely class centroid averaging, and does target centered geometry persist when references come from independent semantic classes?",
"pools":POOLS,"contexts":CTX,"report_layers":REPORT,
"method":"Reproduce within class multi reference consensus and compare it with target minus class centroid geometry. Then forge each target independently against each of the other nine semantic class banks and test bank invariance, leave one bank out transfer, within class alignment, cross context persistence and target separation.",
"consensus_centroid_equivalence":EQ,"external_bank_invariance":BANK,"external_bank_loo":LOO,"external_within_alignment":ALIGN,"external_context_invariance":XCTX,"separation":SEP,"random_baseline":NULL,
"integrity":{"all_targets":60,"external_banks_per_target":9,"selection":False,"steering":False,"retrieval":False,"training":False,"weight_update":False,"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest()
run=f"T284-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{run}.json";tp=ROOT/f"{run}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2).encode())
out=["="*108,"TEST 284 - TARGET IDENTITY VS CENTROID AVERAGING CONTROL","="*108]
for L in REPORT:
    out.append(f"L{L:02d} CONS_CENT={EQ[str(L)]['mean']:+.6f} EXT_BANK={BANK[str(L)]['mean']:+.6f} EXT_LOO={LOO[str(L)]['mean']:+.6f} EXT_WITHIN={ALIGN[str(L)]['mean']:+.6f} EXT_CTX={XCTX[str(L)]['mean']:+.6f}")
out+=["","NO SELECTION - NO STEERING - NO RETRIEVAL - NO TRAINING - WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(out),encoding="utf-8")
print("="*108);print("TEST 284 COMPLETE")
print("CENTROID CONTROL | EXTERNAL REFERENCE BANKS | TARGET IDENTITY")
print("NO SELECTION | NO STEERING | NO RETRIEVAL | NO TRAINING | WEIGHT INTEGRITY PASS")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)

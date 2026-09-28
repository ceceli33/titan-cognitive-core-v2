# ================================================================================================
# AKBASCORE - TEST 285
# CLASS COMMON COMPONENT REMOVAL AND TARGET SPECIFIC RESIDUAL IDENTITY
# EXTERNAL BANK FORGE - CLASS CENTERING - BANK TRANSFER - CONTEXT TRANSFER - IDENTITY SEPARATION
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
SEED=285
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda");MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL=28;H_EXPECT=3584;EPS=1e-8;REPORT=[0,5,10,15,19,21,23,25,26,27]
ROOT=Path("/content/AKBASCORE_TEST285") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST285")
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
print("="*108);print("TEST 285 - CLASS COMMON COMPONENT REMOVAL AND TARGET SPECIFIC RESIDUAL IDENTITY");print("="*108);print("START:",START)
print("[1/12] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if tv>=(4,56) else "torch_dtype"
torch.cuda.synchronize();t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;PDT=next(model.parameters()).dtype;torch.cuda.synchronize()
if len(layers)!=TOTAL or H!=H_EXPECT or PDT!=torch.bfloat16:raise RuntimeError("Architecture or dtype mismatch")
print(f"OK | {MODEL_ID} | 28L | H={H} | {PDT} | {time.perf_counter()-t:.2f}s")
print("[2/12] WEIGHT SENTINEL AND DESIGN")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();CN=list(CTX);CL=list(POOLS);KEYS=[f"{c}:{x}" for c,p in POOLS.items() for x in p]
print("FP:",[f"{x:.4f}" for x in FP0]);print("CLASSES=10 | TARGETS=60 | CONTEXTS=6 | EXTERNAL_BANKS_PER_TARGET=9")
def enc(x):
    s=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to(DEVICE)
@torch.inference_mode()
def capture(text):
    o=model(**enc(text),use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
print("[3/12] CACHE 360 ABSOLUTE STATES")
STATE={}
for cls,pool in POOLS.items():
    STATE[cls]={}
    for x in pool:STATE[cls][x]={c:capture(f.format(x=x)) for c,f in CTX.items()}
    print(cls,"READY")
print("[4/12] EXTERNAL BANK FORGE")
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
print("RAW EXTERNAL FORGE READY")
print("[5/12] CLASS COMMON COMPONENT REMOVAL")
RES={}
for cls,pool in POOLS.items():
    banks=[b for b in CL if b!=cls]
    for bank in banks:
        for c in CN:
            for L in range(TOTAL):
                kk=[f"{cls}:{x}" for x in pool];mu=torch.stack([RAW[k][bank][c][L] for k in kk]).mean(0)
                for k in kk:
                    if k not in RES:RES[k]={}
                    if bank not in RES[k]:RES[k][bank]={}
                    if c not in RES[k][bank]:RES[k][bank][c]=[None]*TOTAL
                    RES[k][bank][c][L]=unit(RAW[k][bank][c][L]-mu)
print("CLASS CENTERING COMPLETE")
print("[6/12] RAW VS RESIDUAL BANK INVARIANCE")
BANK={}
for L in REPORT:
    rawv=[];resv=[]
    for cls,pool in POOLS.items():
        banks=[b for b in CL if b!=cls]
        for target in pool:
            k=f"{cls}:{target}";rv=[];sv=[]
            for bank in banks:
                rv.append(unit(torch.stack([RAW[k][bank][c][L] for c in CN]).mean(0)))
                sv.append(unit(torch.stack([RES[k][bank][c][L] for c in CN]).mean(0)))
            rawv.append(np.mean([cos(rv[i],rv[j]) for i in range(9) for j in range(i+1,9)]))
            resv.append(np.mean([cos(sv[i],sv[j]) for i in range(9) for j in range(i+1,9)]))
    BANK[str(L)]={"raw":float(np.mean(rawv)),"residual":float(np.mean(resv)),"residual_p10":float(np.percentile(resv,10)),"residual_min":float(np.min(resv))}
    x=BANK[str(L)];print(f"L{L:02d} | RAW={x['raw']:+.4f} | RES={x['residual']:+.4f} | P10={x['residual_p10']:+.4f} | MIN={x['residual_min']:+.4f}")
print("[7/12] LEAVE ONE BANK OUT RESIDUAL TRANSFER")
LOO={}
for L in REPORT:
    vals=[]
    for cls,pool in POOLS.items():
        banks=[b for b in CL if b!=cls]
        for target in pool:
            k=f"{cls}:{target}";bv={b:unit(torch.stack([RES[k][b][c][L] for c in CN]).mean(0)) for b in banks}
            for held in banks:vals.append(cos(unit(torch.stack([bv[b] for b in banks if b!=held]).mean(0)),bv[held]))
    LOO[str(L)]={"mean":float(np.mean(vals)),"median":float(np.median(vals)),"p10":float(np.percentile(vals,10)),"min":float(np.min(vals)),"positive_fraction":float(np.mean(np.array(vals)>0))}
    x=LOO[str(L)];print(f"L{L:02d} | RES_LOO={x['mean']:+.4f} | MED={x['median']:+.4f} | P10={x['p10']:+.4f} | POS={x['positive_fraction']:.3f}")
print("[8/12] CROSS CONTEXT RESIDUAL TRANSFER")
XCTX={}
for L in REPORT:
    vals=[]
    for cls,pool in POOLS.items():
        banks=[b for b in CL if b!=cls]
        for target in pool:
            k=f"{cls}:{target}";cv=[]
            for c in CN:cv.append(unit(torch.stack([RES[k][b][c][L] for b in banks]).mean(0)))
            vals.append(np.mean([cos(cv[i],cv[j]) for i in range(6) for j in range(i+1,6)]))
    XCTX[str(L)]={"mean":float(np.mean(vals)),"median":float(np.median(vals)),"p10":float(np.percentile(vals,10)),"min":float(np.min(vals))}
    x=XCTX[str(L)];print(f"L{L:02d} | RES_CTX={x['mean']:+.4f} | MED={x['median']:+.4f} | P10={x['p10']:+.4f}")
print("[9/12] FINAL TARGET RESIDUAL CONSENSUS")
CONS={}
for cls,pool in POOLS.items():
    banks=[b for b in CL if b!=cls]
    for target in pool:
        k=f"{cls}:{target}";CONS[k]=[]
        for L in range(TOTAL):CONS[k].append(unit(torch.stack([RES[k][b][c][L] for b in banks for c in CN]).mean(0)))
print("60 TARGET RESIDUAL CONSENSUS VECTORS READY")
print("[10/12] IDENTITY SEPARATION BEFORE AND AFTER CLEANING")
SEP={}
for L in REPORT:
    rawcons={};same=[];cross=[];rsame=[];rcross=[]
    for cls,pool in POOLS.items():
        banks=[b for b in CL if b!=cls]
        for target in pool:
            k=f"{cls}:{target}";rawcons[k]=unit(torch.stack([RAW[k][b][c][L] for b in banks for c in CN]).mean(0))
    for i in range(len(KEYS)):
        for j in range(i+1,len(KEYS)):
            sameclass=KEYS[i].split(":",1)[0]==KEYS[j].split(":",1)[0]
            a=abs(cos(rawcons[KEYS[i]],rawcons[KEYS[j]]));b=abs(cos(CONS[KEYS[i]][L],CONS[KEYS[j]][L]))
            (same if sameclass else cross).append(a);(rsame if sameclass else rcross).append(b)
    SEP[str(L)]={"raw_same_abs":float(np.mean(same)),"raw_cross_abs":float(np.mean(cross)),"res_same_abs":float(np.mean(rsame)),"res_cross_abs":float(np.mean(rcross))}
    x=SEP[str(L)];print(f"L{L:02d} | RAW SAME={x['raw_same_abs']:.4f} CROSS={x['raw_cross_abs']:.4f} | RES SAME={x['res_same_abs']:.4f} CROSS={x['res_cross_abs']:.4f}")
print("[11/12] CLASS ATLAS AND RANDOM BASELINE")
CLASS={};NULL={};g=torch.Generator(device=DEVICE);g.manual_seed(SEED+999999)
for cls,pool in POOLS.items():
    CLASS[cls]={}
    banks=[b for b in CL if b!=cls]
    for L in REPORT:
        bv=[];cv=[]
        for target in pool:
            k=f"{cls}:{target}";x=[unit(torch.stack([RES[k][b][c][L] for c in CN]).mean(0)) for b in banks]
            y=[unit(torch.stack([RES[k][b][c][L] for b in banks]).mean(0)) for c in CN]
            bv.append(np.mean([cos(x[i],x[j]) for i in range(9) for j in range(i+1,9)]))
            cv.append(np.mean([cos(y[i],y[j]) for i in range(6) for j in range(i+1,6)]))
        CLASS[cls][str(L)]={"bank":float(np.mean(bv)),"context":float(np.mean(cv))}
for cls in POOLS:
    x=CLASS[cls]["27"];print(f"{cls:10s} L27 | BANK={x['bank']:+.4f} | CTX={x['context']:+.4f}")
for L in REPORT:
    RND=torch.randn(len(KEYS),H,device=DEVICE,dtype=torch.float32,generator=g);RND/=RND.norm(dim=1,keepdim=True).clamp_min(EPS)
    rnd=[abs(cos(RND[i],RND[j])) for i in range(len(KEYS)) for j in range(i+1,len(KEYS))]
    real=[abs(cos(CONS[KEYS[i]][L],CONS[KEYS[j]][L])) for i in range(len(KEYS)) for j in range(i+1,len(KEYS))]
    NULL[str(L)]={"random_abs":float(np.mean(rnd)),"residual_abs":float(np.mean(real))}
print(f"L27 | RANDOM={NULL['27']['random_abs']:.4f} | RESIDUAL={NULL['27']['residual_abs']:.4f}")
print("[12/12] INTEGRITY AND SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
R={"schema":"akbascore.test285.v1","test":"TEST 285","start":START,"end":utc(),"model":MODEL_ID,
"question":"After removing the class common component from external reference forge vectors, does target specific residual identity remain stable across independent reference banks and contexts?",
"pools":POOLS,"contexts":CTX,"report_layers":REPORT,
"method":"Forge every target against nine external semantic class banks in six contexts. For each source class, bank, context and layer subtract the mean forged direction across its six targets before normalization. Measure residual bank invariance, leave one bank out transfer, cross context transfer, target separation, class atlas and random baseline.",
"bank_invariance":BANK,"bank_loo":LOO,"context_invariance":XCTX,"separation":SEP,"class_atlas":CLASS,"random_baseline":NULL,
"integrity":{"targets":60,"banks_per_target":9,"contexts":6,"selection":False,"steering":False,"retrieval":False,"training":False,"weight_update":False,"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();run=f"T285-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{run}.json";tp=ROOT/f"{run}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2).encode())
out=["="*108,"TEST 285 - CLASS COMMON COMPONENT REMOVAL AND TARGET SPECIFIC RESIDUAL IDENTITY","="*108]
for L in REPORT:
    out.append(f"L{L:02d} BANK_RAW={BANK[str(L)]['raw']:+.6f} BANK_RES={BANK[str(L)]['residual']:+.6f} LOO={LOO[str(L)]['mean']:+.6f} CTX={XCTX[str(L)]['mean']:+.6f} RES_SAME={SEP[str(L)]['res_same_abs']:.6f} RES_CROSS={SEP[str(L)]['res_cross_abs']:.6f}")
out+=["","NO SELECTION - NO STEERING - NO RETRIEVAL - NO TRAINING - WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(out),encoding="utf-8")
print("="*108);print("TEST 285 COMPLETE")
print("CLASS COMMON REMOVED | TARGET SPECIFIC RESIDUAL IDENTITY")
print("NO SELECTION | NO STEERING | NO RETRIEVAL | NO TRAINING | WEIGHT INTEGRITY PASS")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)

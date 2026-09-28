# ==================================================================================================
# AKBASCORE · TEST 282
# CROSS-CONTEXT SEMANTIC VECTOR TRANSFER
# 60 CONTRASTS · 10 CLASSES · 6 STRUCTURALLY DIFFERENT CONTEXT FAMILIES
# WITHIN-CONTEXT FORGE · CROSS-CONTEXT TRANSFER · LOO CONTEXT CONSENSUS
# NO STEERING · NO RETRIEVAL · NO TRAINING
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
SEED=282;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda");MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL=28;H_EXPECT=3584;EPS=1e-8;REPORT=[0,5,10,15,19,21,23,25,26,27]
ROOT=Path("/content/AKBASCORE_TEST282") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST282");ROOT.mkdir(parents=True,exist_ok=True)
START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
PAIRS={
"COLOR":[("red","blue"),("green","yellow"),("black","white"),("orange","purple"),("silver","gold"),("cyan","magenta")],
"PLACE":[("Ankara","Tokyo"),("Lisbon","Cairo"),("Oslo","Lima"),("Berlin","Seoul"),("Madrid","Nairobi"),("Athens","Jakarta")],
"NUMBER":[("17","83"),("29","64"),("41","92"),("13","76"),("35","88"),("22","71")],
"MATERIAL":[("copper","titanium"),("iron","aluminum"),("glass","ceramic"),("silver","carbon"),("nickel","silicon"),("bronze","graphite")],
"ACTION":[("running","sleeping"),("opening","closing"),("ascending","descending"),("building","destroying"),("entering","leaving"),("pushing","pulling")],
"ABSTRACT":[("justice","chaos"),("freedom","constraint"),("certainty","doubt"),("harmony","conflict"),("order","disorder"),("trust","suspicion")],
"OBJECT":[("hammer","telescope"),("lantern","compass"),("violin","anchor"),("key","mirror"),("helmet","camera"),("clock","spear")],
"ADJECTIVE":[("bright","dark"),("heavy","light"),("smooth","rough"),("silent","loud"),("narrow","wide"),("warm","cold")],
"NAME":[("Elena","Marcus"),("Aylin","Victor"),("Nora","Adrian"),("Selin","Lucas"),("Mira","Jonas"),("Lena","Darius")],
"NOVEL":[("Velorium","Nexarith"),("Caldris","Zophene"),("Meraxon","Tilvara"),("Orvex","Pellune"),("Synthera","Kaldrox"),("Virelon","Nemorix")]
}
CTX={
"DECL":"The recorded value is {x}.",
"QA":"Question: What value was recorded?\nAnswer: {x}.",
"DIALOGUE":"Operator: Which value should I use?\nAssistant: Use {x}.",
"TECH":"SYSTEM_RECORD\nfield=value\nvalue={x}\nEND_RECORD",
"STORY":"After checking the sealed note, the researcher found a single entry. It was {x}.",
"MINIMAL":"{x}"
}
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(EPS))
def canon(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
print("="*110);print("TEST 282 — CROSS-CONTEXT SEMANTIC VECTOR TRANSFER");print("="*110);print("START:",START)
print("[1/10] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
torch.cuda.synchronize();t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;PDT=next(model.parameters()).dtype;torch.cuda.synchronize()
if len(layers)!=TOTAL or H!=H_EXPECT or PDT!=torch.bfloat16:raise RuntimeError("Architecture/dtype mismatch.")
print(f"OK · {MODEL_ID} · 28L · H={H} · {PDT} · {time.perf_counter()-t:.2f}s")
print("[2/10] WEIGHT SENTINEL + CORPUS")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();ITEMS=[(c,i,a,b) for c,ps in PAIRS.items() for i,(a,b) in enumerate(ps)];CN=list(CTX)
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"CLASSES=10 · ITEMS=60 · CONTEXTS={len(CN)} · FORWARDS={len(ITEMS)*len(CN)*2}")
def enc(x):
    s=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to(DEVICE)
@torch.inference_mode()
def capture(text):
    o=model(**enc(text),use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
print("[3/10] CONTEXT-SPECIFIC FORGE")
V={};META={}
for z,(cls,idx,a,b) in enumerate(ITEMS,1):
    key=f"{cls}_{idx+1:02d}";V[key]={};META[key]={"class":cls,"positive":a,"negative":b}
    for cn,f in CTX.items():
        A=capture(f.format(x=a));B=capture(f.format(x=b));V[key][cn]=[unit(A[L]-B[L]) for L in range(TOTAL)]
    if z%10==0:print(f"{z:02d}/60 DONE")
print("[4/10] CROSS-CONTEXT TRANSFER")
TRANSFER={}
for L in REPORT:
    vals=[];mins=[]
    for k in V:
        cs=[cos(V[k][CN[i]][L],V[k][CN[j]][L]) for i in range(len(CN)) for j in range(i+1,len(CN))]
        vals.append(np.mean(cs));mins.append(np.min(cs))
    TRANSFER[str(L)]={"mean":float(np.mean(vals)),"median":float(np.median(vals)),"p10":float(np.percentile(vals,10)),"mean_min":float(np.mean(mins)),"positive_fraction":float(np.mean(np.array(vals)>0))}
    x=TRANSFER[str(L)];print(f"L{L:02d} · CROSS={x['mean']:+.4f} · MED={x['median']:+.4f} · P10={x['p10']:+.4f} · MINAVG={x['mean_min']:+.4f}")
print("[5/10] CONTEXT-PAIR MATRIX")
MATRIX={}
for L in REPORT:
    MATRIX[str(L)]={}
    for i,a in enumerate(CN):
        for b in CN[i+1:]:
            x=[cos(V[k][a][L],V[k][b][L]) for k in V];MATRIX[str(L)][f"{a}|{b}"]={"mean":float(np.mean(x)),"median":float(np.median(x)),"p10":float(np.percentile(x,10))}
print("L27")
for k,x in MATRIX["27"].items():print(f"{k:18s} · {x['mean']:+.4f}")
print("[6/10] LOO-CONTEXT CONSENSUS")
LOO={}
for L in REPORT:
    allv=[];byctx={c:[] for c in CN}
    for k in V:
        for held in CN:
            train=unit(torch.stack([V[k][c][L] for c in CN if c!=held]).mean(0));q=cos(train,V[k][held][L]);allv.append(q);byctx[held].append(q)
    LOO[str(L)]={"mean":float(np.mean(allv)),"median":float(np.median(allv)),"p10":float(np.percentile(allv,10)),"by_context":{c:float(np.mean(x)) for c,x in byctx.items()}}
    x=LOO[str(L)];print(f"L{L:02d} · LOO={x['mean']:+.4f} · MED={x['median']:+.4f} · P10={x['p10']:+.4f}")
print("[7/10] ITEM CONSENSUS + BETWEEN-ITEM SEPARATION")
CONS={};SEP={}
for k in V:CONS[k]=[unit(torch.stack([V[k][c][L] for c in CN]).mean(0)) for L in range(TOTAL)]
keys=list(V)
for L in REPORT:
    same=[];cross=[];allc=[]
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            q=abs(cos(CONS[keys[i]][L],CONS[keys[j]][L]));allc.append(q)
            (same if META[keys[i]]["class"]==META[keys[j]]["class"] else cross).append(q)
    SEP[str(L)]={"same":float(np.mean(same)),"cross":float(np.mean(cross)),"all":float(np.mean(allc)),"p95":float(np.percentile(allc,95)),"max":float(np.max(allc))}
    x=SEP[str(L)];print(f"L{L:02d} · SAME={x['same']:.4f} · CROSS={x['cross']:.4f} · P95={x['p95']:.4f}")
print("[8/10] CLASS-WISE CROSS-CONTEXT ATLAS")
CLASS={}
for cls in PAIRS:
    kk=[k for k in keys if META[k]["class"]==cls];CLASS[cls]={}
    for L in REPORT:
        item=[]
        for k in kk:item.append(np.mean([cos(V[k][CN[i]][L],V[k][CN[j]][L]) for i in range(len(CN)) for j in range(i+1,len(CN))]))
        CLASS[cls][str(L)]={"mean":float(np.mean(item)),"min":float(np.min(item))}
for cls in PAIRS:print(f"{cls:10s} L27 · CROSS={CLASS[cls]['27']['mean']:+.4f} · MINITEM={CLASS[cls]['27']['min']:+.4f}")
print("[9/10] RANDOM BASELINE + DISTRIBUTION")
g=torch.Generator(device=DEVICE);g.manual_seed(SEED+999999);NULL={};DIST={}
for L in REPORT:
    R=torch.randn(len(CN),H,device=DEVICE,dtype=torch.float32,generator=g);R/=R.norm(dim=1,keepdim=True).clamp_min(EPS)
    rnd=[cos(R[i],R[j]) for i in range(len(CN)) for j in range(i+1,len(CN))]
    item=[np.mean([cos(V[k][CN[i]][L],V[k][CN[j]][L]) for i in range(len(CN)) for j in range(i+1,len(CN))]) for k in keys]
    NULL[str(L)]={"mean":float(np.mean(rnd)),"abs_mean":float(np.mean(np.abs(rnd)))}
    DIST[str(L)]={"mean":float(np.mean(item)),"sd":float(np.std(item)),"min":float(np.min(item)),"p10":float(np.percentile(item,10)),"median":float(np.median(item)),"max":float(np.max(item)),"positive_fraction":float(np.mean(np.array(item)>0))}
print("L27 DISTRIBUTION:",DIST["27"]);print("L27 RANDOM |COS|:",f"{NULL['27']['abs_mean']:.4f}")
print("[10/10] INTEGRITY + SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure.")
R={"schema":"akbascore.test282.v1","test":"TEST 282","start":START,"end":utc(),"model":MODEL_ID,
"question":"Does the same semantic contrast produce a reproducible layer-local direction across structurally different linguistic contexts?",
"pairs":PAIRS,"contexts":CTX,"n_items":len(ITEMS),"n_contexts":len(CN),"report_layers":REPORT,
"method":"For each of the same 60 TEST281 contrasts, independently extract positive-minus-negative final-token hidden-state directions in six structurally different context families. No context is privileged. Measure pairwise cross-context cosine, leave-one-context-out consensus transfer, class-wise transfer, consensus-vector between-item separation and isotropic random baseline.",
"transfer":TRANSFER,"context_pair_matrix":MATRIX,"loo":LOO,"separation":SEP,"class_atlas":CLASS,"random_baseline":NULL,"distribution":DIST,"metadata":META,
"integrity":{"all_items_reported":True,"selection":False,"steering":False,"retrieval":False,"training":False,"weight_update":False,"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();run=f"T282-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{run}.json";tp=ROOT/f"{run}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=False,sort_keys=True,indent=2).encode())
o=["="*110,"TEST 282 — CROSS-CONTEXT SEMANTIC VECTOR TRANSFER","="*110,f"ITEMS={len(ITEMS)} CONTEXTS={len(CN)}"]
for L in REPORT:
    a=TRANSFER[str(L)];b=LOO[str(L)];s=SEP[str(L)];o.append(f"L{L:02d} CROSS={a['mean']:+.6f} P10={a['p10']:+.6f} LOO={b['mean']:+.6f} BETWEEN={s['all']:.6f} P95={s['p95']:.6f}")
o+=["","L27 CLASS ATLAS"]
for cls in PAIRS:o.append(f"{cls}: CROSS={CLASS[cls]['27']['mean']:+.6f} MIN={CLASS[cls]['27']['min']:+.6f}")
o+=["","NO SELECTION · NO STEERING · NO RETRIEVAL · NO TRAINING · WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(o),encoding="utf-8")
print("="*110);print("TEST 282 COMPLETE")
print("60 CONTRASTS · 6 STRUCTURALLY DIFFERENT CONTEXTS · CROSS-CONTEXT TRANSFER")
print("NO SELECTION · NO STEERING · NO RETRIEVAL · NO TRAINING · WEIGHT INTEGRITY PASS")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*110)

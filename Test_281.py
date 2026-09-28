# ==================================================================================================
# AKBASCORE · TEST 281
# LARGE-SCALE SYNTHETIC VECTOR FORGEABILITY ATLAS
# 10 SEMANTIC CLASSES × 6 CONTRASTS = 60 ITEMS · 4 PARAPHRASE FRAMES
# WITHIN-ITEM REPRODUCIBILITY · BETWEEN-ITEM SEPARATION · NULL/RANDOM BASELINE
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
SEED=281;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda");MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL=28;H_EXPECT=3584;EPS=1e-8;REPORT_LAYERS=[0,5,10,15,19,21,23,25,26,27]
ROOT=Path("/content/AKBASCORE_TEST281") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST281");ROOT.mkdir(parents=True,exist_ok=True)
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
FRAMES=[
"The target value is {x}.",
"The designated item is {x}.",
"Record the value {x}.",
"The selected value is {x}."
]
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(EPS))
def canon(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
print("="*110);print("TEST 281 — LARGE-SCALE SYNTHETIC VECTOR FORGEABILITY ATLAS");print("="*110);print("START:",START)
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
FP0=fp();ITEMS=[(c,i,a,b) for c,ps in PAIRS.items() for i,(a,b) in enumerate(ps)]
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"CLASSES={len(PAIRS)} · ITEMS={len(ITEMS)} · FRAMES={len(FRAMES)} · FORWARDS={len(ITEMS)*len(FRAMES)*2}")
def enc(x):
    s=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to(DEVICE)
@torch.inference_mode()
def capture(text):
    o=model(**enc(text),use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
print("[3/10] 60-ITEM SELF-FORGE")
VECTORS={};METRICS={}
for z,(cls,idx,a,b) in enumerate(ITEMS,1):
    key=f"{cls}_{idx+1:02d}";D=[[] for _ in range(TOTAL)]
    for f in FRAMES:
        A=capture(f.format(x=a));B=capture(f.format(x=b))
        for L in range(TOTAL):D[L].append(A[L]-B[L])
    VECTORS[key]=[];METRICS[key]={"class":cls,"positive":a,"negative":b,"layers":{}}
    for L in range(TOTAL):
        U=torch.stack([unit(x) for x in D[L]]);v=unit(U.mean(0));VECTORS[key].append(v)
        pcs=[cos(U[i],U[j]) for i in range(4) for j in range(i+1,4)]
        _,S,_=torch.linalg.svd(U,full_matrices=False);ev=(S*S)/(S*S).sum().clamp_min(EPS)
        loo=[]
        for i in range(4):loo.append(cos(unit(torch.stack([U[j] for j in range(4) if j!=i]).mean(0)),U[i]))
        METRICS[key]["layers"][str(L)]={"pair":float(np.mean(pcs)),"pair_min":float(np.min(pcs)),"loo":float(np.mean(loo)),"pc1":float(ev[0]),"pc12":float(ev[:2].sum())}
    if z%10==0:print(f"{z:02d}/60 DONE")
print("[4/10] WITHIN-ITEM REPRODUCIBILITY")
WITHIN={}
for L in REPORT_LAYERS:
    p=[METRICS[k]["layers"][str(L)]["pair"] for k in VECTORS];loo=[METRICS[k]["layers"][str(L)]["loo"] for k in VECTORS];pc=[METRICS[k]["layers"][str(L)]["pc1"] for k in VECTORS]
    WITHIN[str(L)]={"pair_mean":float(np.mean(p)),"pair_median":float(np.median(p)),"pair_p10":float(np.percentile(p,10)),"loo_mean":float(np.mean(loo)),"pc1_mean":float(np.mean(pc))}
    print(f"L{L:02d} · PAIR={np.mean(p):+.4f} · P10={np.percentile(p,10):+.4f} · LOO={np.mean(loo):+.4f} · PC1={np.mean(pc)*100:.2f}%")
print("[5/10] BETWEEN-ITEM SEPARATION")
BETWEEN={}
keys=list(VECTORS)
for L in REPORT_LAYERS:
    same=[];cross=[];allc=[]
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            c=abs(cos(VECTORS[keys[i]][L],VECTORS[keys[j]][L]));allc.append(c)
            if METRICS[keys[i]]["class"]==METRICS[keys[j]]["class"]:same.append(c)
            else:cross.append(c)
    BETWEEN[str(L)]={"same_class_abs_cos":float(np.mean(same)),"cross_class_abs_cos":float(np.mean(cross)),"all_abs_cos":float(np.mean(allc)),"p95_abs_cos":float(np.percentile(allc,95)),"max_abs_cos":float(np.max(allc))}
    print(f"L{L:02d} · SAME={np.mean(same):.4f} · CROSS={np.mean(cross):.4f} · P95={np.percentile(allc,95):.4f} · MAX={np.max(allc):.4f}")
print("[6/10] CLASS-WISE ATLAS")
CLASS={}
for cls in PAIRS:
    kk=[k for k in keys if METRICS[k]["class"]==cls];CLASS[cls]={}
    for L in REPORT_LAYERS:
        pair=[METRICS[k]["layers"][str(L)]["pair"] for k in kk];loo=[METRICS[k]["layers"][str(L)]["loo"] for k in kk]
        inter=[abs(cos(VECTORS[kk[i]][L],VECTORS[kk[j]][L])) for i in range(len(kk)) for j in range(i+1,len(kk))]
        CLASS[cls][str(L)]={"pair":float(np.mean(pair)),"loo":float(np.mean(loo)),"inter_abs_cos":float(np.mean(inter))}
for cls in PAIRS:
    x=CLASS[cls]["27"];print(f"{cls:10s} L27 · WITHIN={x['pair']:+.4f} · LOO={x['loo']:+.4f} · BETWEEN={x['inter_abs_cos']:.4f}")
print("[7/10] RANDOM/NULL GEOMETRY BASELINE")
g=torch.Generator(device=DEVICE);g.manual_seed(SEED+999999);NULL={}
for L in REPORT_LAYERS:
    R=torch.randn(len(keys),H,device=DEVICE,dtype=torch.float32,generator=g);R=R/R.norm(dim=1,keepdim=True).clamp_min(EPS)
    rc=[abs(cos(R[i],R[j])) for i in range(len(keys)) for j in range(i+1,len(keys))]
    real=[abs(cos(VECTORS[keys[i]][L],VECTORS[keys[j]][L])) for i in range(len(keys)) for j in range(i+1,len(keys))]
    NULL[str(L)]={"random_abs_cos":float(np.mean(rc)),"random_p95":float(np.percentile(rc,95)),"real_abs_cos":float(np.mean(real)),"real_over_random":float(np.mean(real)/(np.mean(rc)+EPS))}
    print(f"L{L:02d} · RANDOM={np.mean(rc):.4f} · REAL={np.mean(real):.4f} · ×{np.mean(real)/(np.mean(rc)+EPS):.2f}")
print("[8/10] FORGEABILITY DISTRIBUTION")
DIST={}
for L in REPORT_LAYERS:
    vals=np.array([METRICS[k]["layers"][str(L)]["pair"] for k in keys])
    DIST[str(L)]={"n":len(vals),"mean":float(vals.mean()),"sd":float(vals.std()),"min":float(vals.min()),"p10":float(np.percentile(vals,10)),"p25":float(np.percentile(vals,25)),"median":float(np.median(vals)),"p75":float(np.percentile(vals,75)),"p90":float(np.percentile(vals,90)),"max":float(vals.max()),"positive_fraction":float(np.mean(vals>0))}
print("L27 ·",DIST["27"])
print("[9/10] WORST/BEST ITEMS — DESCRIPTIVE ONLY")
rank=sorted(keys,key=lambda k:METRICS[k]["layers"]["27"]["pair"])
for k in rank[:5]:
    m=METRICS[k];print(f"LOW  {k:13s} {m['positive']}↔{m['negative']} · PAIR={m['layers']['27']['pair']:+.4f} · LOO={m['layers']['27']['loo']:+.4f}")
for k in rank[-5:]:
    m=METRICS[k];print(f"HIGH {k:13s} {m['positive']}↔{m['negative']} · PAIR={m['layers']['27']['pair']:+.4f} · LOO={m['layers']['27']['loo']:+.4f}")
print("[10/10] INTEGRITY + SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure.")
R={"schema":"akbascore.test281.v1","test":"TEST 281","start":START,"end":utc(),"model":MODEL_ID,
"question":"Can one fixed self-forge procedure convert many heterogeneous semantic contrasts into reproducible, distinguishable layer-local synthetic vectors?",
"semantic_classes":PAIRS,"frames":FRAMES,"n_classes":len(PAIRS),"n_items":len(ITEMS),"n_frames":len(FRAMES),"report_layers":REPORT_LAYERS,
"method":"For every one of 60 fixed semantic contrasts, apply four identical neutral sentence frames. At each layer compute four positive-minus-negative final-token hidden-state deltas, normalize each delta, and define the synthetic vector as their normalized mean. Report all items without selection. Measure within-item paraphrase reproducibility, leave-one-frame-out agreement, PCA concentration, between-item separation, class-wise geometry and isotropic random-vector baseline. No steering, retrieval, training or weight updates.",
"metrics":METRICS,"within":WITHIN,"between":BETWEEN,"class_atlas":CLASS,"random_baseline":NULL,"distribution":DIST,
"integrity":{"all_items_reported":True,"selection":False,"steering":False,"retrieval":False,"training":False,"weight_update":False,"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();run=f"T281-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{run}.json";tp=ROOT/f"{run}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=False,sort_keys=True,indent=2).encode())
o=["="*110,"TEST 281 — LARGE-SCALE SYNTHETIC VECTOR FORGEABILITY ATLAS","="*110,f"CLASSES={len(PAIRS)} ITEMS={len(ITEMS)} FRAMES={len(FRAMES)}"]
for L in REPORT_LAYERS:
    w=WITHIN[str(L)];b=BETWEEN[str(L)];o.append(f"L{L:02d} WITHIN={w['pair_mean']:+.6f} LOO={w['loo_mean']:+.6f} PC1={w['pc1_mean']*100:.3f}% BETWEEN={b['all_abs_cos']:.6f} P95={b['p95_abs_cos']:.6f}")
o+=["","L27 CLASS ATLAS"]
for cls in PAIRS:
    x=CLASS[cls]["27"];o.append(f"{cls}: WITHIN={x['pair']:+.6f} LOO={x['loo']:+.6f} BETWEEN={x['inter_abs_cos']:.6f}")
o+=["","NO SELECTION · NO STEERING · NO RETRIEVAL · NO TRAINING · WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(o),encoding="utf-8")
print("="*110);print("TEST 281 COMPLETE")
print(f"60 CONTRASTS · 10 SEMANTIC CLASSES · 4 FRAMES · {len(ITEMS)*len(FRAMES)*2} FORWARDS")
print("NO SELECTION · NO STEERING · NO RETRIEVAL · NO TRAINING · WEIGHT INTEGRITY PASS")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*110)

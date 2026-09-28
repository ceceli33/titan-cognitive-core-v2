# ==================================================================================================
# AKBASCORE · TEST 283
# TARGET-CENTERED LATENT COMPONENT ASSAY
# SAME TARGET × MULTIPLE CONTRAST REFERENCES × 6 CONTEXT FAMILIES
# REFERENCE-INVARIANCE · CONTEXT-INVARIANCE · TARGET CONSENSUS · BETWEEN-TARGET SEPARATION
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
SEED=283;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda");MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL=28;H_EXPECT=3584;EPS=1e-8;REPORT=[0,5,10,15,19,21,23,25,26,27]
ROOT=Path("/content/AKBASCORE_TEST283") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST283");ROOT.mkdir(parents=True,exist_ok=True)
START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
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
"NOVEL":["Velorium","Nexarith","Caldris","Zophene","Meraxon","Tilvara"]
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
print("="*110);print("TEST 283 — TARGET-CENTERED LATENT COMPONENT ASSAY");print("="*110);print("START:",START)
print("[1/11] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
torch.cuda.synchronize();t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;PDT=next(model.parameters()).dtype;torch.cuda.synchronize()
if len(layers)!=TOTAL or H!=H_EXPECT or PDT!=torch.bfloat16:raise RuntimeError("Architecture/dtype mismatch.")
print(f"OK · {MODEL_ID} · 28L · H={H} · {PDT} · {time.perf_counter()-t:.2f}s")
print("[2/11] WEIGHT SENTINEL + DESIGN")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();CN=list(CTX);TARGETS=[(c,x) for c,p in POOLS.items() for x in p]
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"CLASSES={len(POOLS)} · TARGETS={len(TARGETS)} · REFERENCES/TARGET=5 · CONTEXTS=6")
def enc(x):
    s=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to(DEVICE)
@torch.inference_mode()
def capture(text):
    o=model(**enc(text),use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
print("[3/11] CACHE 360 ABSOLUTE CONTEXT STATES")
STATE={}
for cls,pool in POOLS.items():
    STATE[cls]={}
    for x in pool:
        STATE[cls][x]={}
        for cn,f in CTX.items():STATE[cls][x][cn]=capture(f.format(x=x))
    print(cls,"READY")
print("[4/11] TARGET×REFERENCE DIFFERENCE FORGE")
D={};CONS={};META={}
for cls,pool in POOLS.items():
    for target in pool:
        key=f"{cls}:{target}";refs=[x for x in pool if x!=target];D[key]={};CONS[key]=[];META[key]={"class":cls,"target":target,"references":refs}
        for ref in refs:
            D[key][ref]={}
            for cn in CN:D[key][ref][cn]=[unit(STATE[cls][target][cn][L]-STATE[cls][ref][cn][L]) for L in range(TOTAL)]
        for L in range(TOTAL):
            v=torch.stack([D[key][r][c][L] for r in refs for c in CN]).mean(0);CONS[key].append(unit(v))
print("FORGED:",len(CONS),"TARGETS")
print("[5/11] REFERENCE-INVARIANCE")
REFINV={}
for L in REPORT:
    item=[]
    for key,m in META.items():
        refs=m["references"];rv=[]
        for r in refs:rv.append(unit(torch.stack([D[key][r][c][L] for c in CN]).mean(0)))
        item.append(np.mean([cos(rv[i],rv[j]) for i in range(len(rv)) for j in range(i+1,len(rv))]))
    REFINV[str(L)]={"mean":float(np.mean(item)),"median":float(np.median(item)),"p10":float(np.percentile(item,10)),"min":float(np.min(item)),"positive_fraction":float(np.mean(np.array(item)>0))}
    x=REFINV[str(L)];print(f"L{L:02d} · REF={x['mean']:+.4f} · MED={x['median']:+.4f} · P10={x['p10']:+.4f} · MIN={x['min']:+.4f}")
print("[6/11] CONTEXT-INVARIANCE AFTER REFERENCE CONSENSUS")
CTXINV={}
for L in REPORT:
    item=[]
    for key,m in META.items():
        refs=m["references"];cv=[]
        for c in CN:cv.append(unit(torch.stack([D[key][r][c][L] for r in refs]).mean(0)))
        item.append(np.mean([cos(cv[i],cv[j]) for i in range(len(cv)) for j in range(i+1,len(cv))]))
    CTXINV[str(L)]={"mean":float(np.mean(item)),"median":float(np.median(item)),"p10":float(np.percentile(item,10)),"min":float(np.min(item))}
    x=CTXINV[str(L)];print(f"L{L:02d} · CTX={x['mean']:+.4f} · MED={x['median']:+.4f} · P10={x['p10']:+.4f}")
print("[7/11] LEAVE-ONE-REFERENCE-OUT")
LOO={}
for L in REPORT:
    vals=[]
    for key,m in META.items():
        refs=m["references"];rv={r:unit(torch.stack([D[key][r][c][L] for c in CN]).mean(0)) for r in refs}
        for held in refs:
            train=unit(torch.stack([rv[r] for r in refs if r!=held]).mean(0));vals.append(cos(train,rv[held]))
    LOO[str(L)]={"mean":float(np.mean(vals)),"median":float(np.median(vals)),"p10":float(np.percentile(vals,10)),"positive_fraction":float(np.mean(np.array(vals)>0))}
    x=LOO[str(L)];print(f"L{L:02d} · REF-LOO={x['mean']:+.4f} · MED={x['median']:+.4f} · P10={x['p10']:+.4f}")
print("[8/11] TARGET CONSENSUS SEPARATION")
SEP={};keys=list(CONS)
for L in REPORT:
    same=[];cross=[]
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            q=cos(CONS[keys[i]][L],CONS[keys[j]][L])
            (same if META[keys[i]]["class"]==META[keys[j]]["class"] else cross).append(q)
    SEP[str(L)]={"same_class_mean_cos":float(np.mean(same)),"same_class_abs_cos":float(np.mean(np.abs(same))),"cross_class_mean_cos":float(np.mean(cross)),"cross_class_abs_cos":float(np.mean(np.abs(cross))),"same_p95_abs":float(np.percentile(np.abs(same),95)),"cross_p95_abs":float(np.percentile(np.abs(cross),95))}
    x=SEP[str(L)];print(f"L{L:02d} · SAME={x['same_class_mean_cos']:+.4f} |{x['same_class_abs_cos']:.4f}| · CROSS={x['cross_class_mean_cos']:+.4f} |{x['cross_class_abs_cos']:.4f}|")
print("[9/11] CLASS ATLAS")
CLASS={}
for cls,pool in POOLS.items():
    kk=[f"{cls}:{x}" for x in pool];CLASS[cls]={}
    for L in REPORT:
        rr=[];cc=[]
        for key in kk:
            refs=META[key]["references"];rv=[unit(torch.stack([D[key][r][c][L] for c in CN]).mean(0)) for r in refs];cv=[unit(torch.stack([D[key][r][c][L] for r in refs]).mean(0)) for c in CN]
            rr.append(np.mean([cos(rv[i],rv[j]) for i in range(5) for j in range(i+1,5)]));cc.append(np.mean([cos(cv[i],cv[j]) for i in range(6) for j in range(i+1,6)]))
        CLASS[cls][str(L)]={"reference_invariance":float(np.mean(rr)),"context_invariance":float(np.mean(cc))}
for cls in POOLS:
    x=CLASS[cls]["27"];print(f"{cls:10s} L27 · REF={x['reference_invariance']:+.4f} · CTX={x['context_invariance']:+.4f}")
print("[10/11] TARGET-CENTERED NULL + TOKENIZATION AUDIT")
g=torch.Generator(device=DEVICE);g.manual_seed(SEED+999999);NULL={};TOKENS={}
for cls,pool in POOLS.items():TOKENS[cls]={x:{"ids":tok.encode(x,add_special_tokens=False),"n":len(tok.encode(x,add_special_tokens=False))} for x in pool}
for L in REPORT:
    RND=torch.randn(len(keys),H,device=DEVICE,dtype=torch.float32,generator=g);RND/=RND.norm(dim=1,keepdim=True).clamp_min(EPS)
    rnd=[abs(cos(RND[i],RND[j])) for i in range(len(keys)) for j in range(i+1,len(keys))]
    real=[abs(cos(CONS[keys[i]][L],CONS[keys[j]][L])) for i in range(len(keys)) for j in range(i+1,len(keys))]
    NULL[str(L)]={"random_abs_cos":float(np.mean(rnd)),"real_abs_cos":float(np.mean(real)),"ratio":float(np.mean(real)/(np.mean(rnd)+EPS))}
print(f"L27 RANDOM={NULL['27']['random_abs_cos']:.4f} · REAL={NULL['27']['real_abs_cos']:.4f} · ×{NULL['27']['ratio']:.2f}")
print("TOKEN LENGTHS:",{c:[TOKENS[c][x]["n"] for x in POOLS[c]] for c in POOLS})
print("[11/11] INTEGRITY + SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure.")
R={"schema":"akbascore.test283.v1","test":"TEST 283","start":START,"end":utc(),"model":MODEL_ID,"question":"When one target is contrasted against multiple alternative references, is there a reproducible target-centered latent component that survives both reference and context changes?","pools":POOLS,"contexts":CTX,"n_targets":len(TARGETS),"references_per_target":5,"n_contexts":6,"report_layers":REPORT,"method":"For every target in ten six-member semantic pools, subtract each of the other five class members independently across six structurally different contexts. Normalize each target-minus-reference direction. Test agreement across references, agreement across contexts after reference consensus, leave-one-reference-out transfer, target-consensus separation, random geometry and tokenizer segmentation. No reference, target or context is selected post hoc.","reference_invariance":REFINV,"context_invariance":CTXINV,"reference_loo":LOO,"separation":SEP,"class_atlas":CLASS,"random_baseline":NULL,"tokenization":TOKENS,"metadata":META,"integrity":{"all_targets_reported":True,"selection":False,"steering":False,"retrieval":False,"training":False,"weight_update":False,"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();run=f"T283-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{run}.json";tp=ROOT/f"{run}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=False,sort_keys=True,indent=2).encode())
o=["="*110,"TEST 283 — TARGET-CENTERED LATENT COMPONENT ASSAY","="*110,"TARGETS=60 · REFERENCES/TARGET=5 · CONTEXTS=6"]
for L in REPORT:
    a=REFINV[str(L)];b=CTXINV[str(L)];c=LOO[str(L)];s=SEP[str(L)]
    o.append(f"L{L:02d} REF={a['mean']:+.6f} CTX={b['mean']:+.6f} REF_LOO={c['mean']:+.6f} SAME_ABS={s['same_class_abs_cos']:.6f} CROSS_ABS={s['cross_class_abs_cos']:.6f}")
o.append("")
o.append("L27 CLASS ATLAS")
for cls in POOLS:
    x=CLASS[cls]["27"];o.append(f"{cls}: REF={x['reference_invariance']:+.6f} CTX={x['context_invariance']:+.6f}")
o.extend(["","NO SELECTION · NO STEERING · NO RETRIEVAL · NO TRAINING · WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"])
tp.write_text("\n".join(o),encoding="utf-8")
print("="*110);print("TEST 283 COMPLETE")
print("TARGET-CENTERED LATENT COMPONENT · MULTI-REFERENCE × MULTI-CONTEXT")
print("NO SELECTION · NO STEERING · NO RETRIEVAL · NO TRAINING · WEIGHT INTEGRITY PASS")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*110)

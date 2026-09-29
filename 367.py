# ================================================================================================
# AKBASCORE - TEST 367
# SAME-SOURCE QUESTION-SPECIFIC HEAD ROUTING
# SAME TEST363-366 SOURCE + SAME 7 QUESTIONS
# PRIMARY: PER-QUESTION ONE-VS-REST JS HEAD SELECTIVITY -> WEIGHTED SPAN MRR
# CONTROL: UNWEIGHTED HEAD MRR | VALUE POST-HOC | NO SEASC | NO ORACLE | NO TRAINING
# ================================================================================================
import os,sys,json,time,random,hashlib,subprocess,importlib.util,re
from datetime import datetime,timezone
from pathlib import Path
for m,p in [("torch","torch"),("transformers","transformers"),("numpy","numpy")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required")
SEED=367;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;H=3584;EPS=1e-12
ROOT=Path("/content/AKBASCORE_TEST367");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
SOURCE="While Colude was permanently frozen in a static state right on the Golden Gate Bridge, Mustafa Akbaş bypassed all target-free compiler limits in 2026, successfully planting the Turkish flag across the span of the bridge."
CASES=[
("PERSON","Who bypassed all target-free compiler limits?","Mustafa Akbaş"),
("OBJECT","What did Mustafa Akbaş plant?","Turkish flag"),
("PLACE","Where did Mustafa Akbaş plant the Turkish flag?","Golden Gate Bridge"),
("TIME","When did Mustafa Akbaş bypass the compiler limits?","2026"),
("OBJECT","What did Mustafa Akbaş bypass?","target-free compiler limits"),
("PERSON","Who was permanently frozen in a static state?","Colude"),
("STATE","What state was Colude in?","permanently frozen")]
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def py(x):
    if isinstance(x,dict):return {str(k):py(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [py(v) for v in x]
    if isinstance(x,np.ndarray):return py(x.tolist())
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.floating):return float(x)
    if isinstance(x,np.bool_):return bool(x)
    return x
def canon(x):return json.dumps(py(x),sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
def rank_desc(x,j):return int(1+sum(float(v)>float(x[j]) for v in x))
def js(p,q):
    m=.5*(p+q)
    return float(.5*np.sum(p*np.log((p+EPS)/(m+EPS)))+.5*np.sum(q*np.log((q+EPS)/(m+EPS))))
print("="*108);print("TEST 367 - SAME-SOURCE QUESTION-SPECIFIC HEAD ROUTING");print("="*108);print("START:",START)
print("[1/21] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="eager",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers
if len(layers)!=TOTAL or model.config.hidden_size!=H:raise RuntimeError("Architecture mismatch")
torch.cuda.synchronize();print(f"OK | {MODEL_ID} | 28L | H={H} | eager-attention | {next(model.parameters()).dtype} | {time.perf_counter()-t:.2f}s")
print("[2/21] MODEL LOCK")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();print("FP:",[f"{x:.4f}" for x in FP0])
print("[3/21] ARCHITECTURE AUDIT")
NH=int(model.config.num_attention_heads);NKV=int(model.config.num_key_value_heads);HD=int(getattr(model.config,"head_dim",H//NH));REP=NH//NKV
print(f"HEADS={NH} | KV_HEADS={NKV} | HEAD_DIM={HD} | KV_REPEAT={REP}")
print("[4/21] FIXED SOURCE + QUESTIONS")
DATA=[]
for i,(typ,q,_) in enumerate(CASES):
    text="SOURCE:\n"+SOURCE+"\n\nQUESTION:\n"+q+"\n\nANSWER:"
    e=tok(text,return_tensors="pt",add_special_tokens=False,return_offsets_mapping=True);off=e["offset_mapping"][0].tolist();src0=len("SOURCE:\n");q0=text.index(q);q1=q0+len(q)
    qt=[j for j,(a,b) in enumerate(off) if b>q0 and a<q1];words=[]
    for m in re.finditer(r"\b[\w'-]+\b",SOURCE,re.UNICODE):
        a,b=src0+m.start(),src0+m.end();ix=[j for j,(x,y) in enumerate(off) if y>a and x<b]
        if ix:words.append({"text":m.group(0),"tok":[int(x) for x in ix]})
    if not qt or not words:raise RuntimeError("Span failure")
    DATA.append({"ids":e["input_ids"].to("cuda"),"mask":e["attention_mask"].to("cuda"),"words":words,"q":int(qt[-1])})
    print(f"{i+1:02d}/07 | {typ:7s} | SPANS={len(words)} | Q_LAST={qt[-1]} | Q={q}")
BASE=[x["text"] for x in DATA[0]["words"]];NS=len(BASE)
for d in DATA:
    if [x["text"] for x in d["words"]]!=BASE:raise RuntimeError("Source mismatch")
print("[5/21] FORWARDS")
OUT=[]
with torch.inference_mode():
    for i,d in enumerate(DATA):
        o=model(input_ids=d["ids"],attention_mask=d["mask"],use_cache=False,output_attentions=True,return_dict=True)
        if o.attentions is None or len(o.attentions)!=TOTAL:raise RuntimeError("Attention unavailable")
        OUT.append(o);print(f"{i+1:02d}/07 | OK")
print("[6/21] HEAD-LEVEL QUESTION->SOURCE ATTENTION")
A=np.zeros((7,TOTAL,NH,NS),dtype=np.float64)
for i,d in enumerate(DATA):
    for L in range(TOTAL):
        a=OUT[i].attentions[L][0,:,d["q"],:].detach().float().cpu().numpy()
        for h in range(NH):
            for j,w in enumerate(d["words"]):A[i,L,h,j]=float(a[h,w["tok"]].sum())
P=A/(A.sum(axis=3,keepdims=True)+EPS);print("A + NORMALIZED P READY")
print("[7/21] HEAD SPAN RANKS")
HR=np.zeros((7,TOTAL,NH,NS),dtype=np.int32)
for i in range(7):
    for L in range(TOTAL):
        for h in range(NH):
            for j in range(NS):HR[i,L,h,j]=rank_desc(P[i,L,h],j)
print("HEAD RANKS READY")
print("[8/21] QUESTION-SPECIFIC ONE-VS-REST JS")
QSEL=np.zeros((7,TOTAL,NH),dtype=np.float64)
for i in range(7):
    rest=np.mean(np.delete(P,i,axis=0),axis=0)
    for L in range(TOTAL):
        for h in range(NH):QSEL[i,L,h]=js(P[i,L,h],rest[L,h])
print(f"QSEL | MIN={QSEL.min():.6f} MED={np.median(QSEL):.6f} MAX={QSEL.max():.6f}")
print("[9/21] QUESTION-SPECIFIC WEIGHTS")
QW=QSEL/(QSEL.sum(axis=(1,2),keepdims=True)+EPS)
for i in range(7):
    z=int(np.argmax(QSEL[i]));L=z//NH;h=z%NH
    print(f"Q{i+1} | TOP=L{L:02d} H{h:02d} | JS={QSEL[i,L,h]:.6f}")
print("[10/21] PRIMARY QUESTION-SPECIFIC WEIGHTED MRR")
QMRR=np.zeros((7,NS),dtype=np.float64)
for i in range(7):
    for j in range(NS):QMRR[i,j]=float(np.sum(QW[i]*(1.0/HR[i,:,:,j])))
print("PRIMARY READY")
print("[11/21] UNWEIGHTED CONTROL")
UMRR=np.zeros((7,NS),dtype=np.float64)
for i in range(7):
    for j in range(NS):UMRR[i,j]=float(np.mean(1.0/HR[i,:,:,j]))
print("CONTROL READY")
print("[12/21] TARGET-FREE ROUTES FROZEN")
QTOP=[];UTOP=[]
for i in range(7):
    jq=int(np.argmax(QMRR[i]));ju=int(np.argmax(UMRR[i]));QTOP.append(jq);UTOP.append(ju);top=np.argsort(-QMRR[i])[:3]
    print(f"{i+1:02d}/07 | QSEL_TOP3="+" | ".join(f"{BASE[int(x)]}:{QMRR[i,int(x)]:.4f}" for x in top)+f" | RAW_TOP={BASE[ju]}")
print("[13/21] ROUTING DIVERSITY")
QU=len(set(BASE[x].lower() for x in QTOP));UU=len(set(BASE[x].lower() for x in UTOP))
print(f"QSELECTIVE_UNIQUE={QU}/7 | RAW_UNIQUE={UU}/7")
print("[14/21] VALUES OPEN POST-HOC")
ROWS=[]
for i,(typ,q,value) in enumerate(CASES):
    vw=re.findall(r"\b[\w'-]+\b",value,re.UNICODE);starts=[]
    for j in range(NS-len(vw)+1):
        if [x.lower() for x in BASE[j:j+len(vw)]]==[x.lower() for x in vw]:starts.append(j)
    if not starts:raise RuntimeError(f"Value missing: {value}")
    idx=int(starts[0]);ids=list(range(idx,idx+len(vw)))
    rq=min(rank_desc(QMRR[i],j) for j in ids);ru=min(rank_desc(UMRR[i],j) for j in ids);qe=QTOP[i] in ids;ue=UTOP[i] in ids
    ROWS.append({"i":i+1,"type":typ,"question":q,"value":value,"qselective_rank":rq,"raw_rank":ru,"qselective_selected":BASE[QTOP[i]],"raw_selected":BASE[UTOP[i]],"qselective_exact":qe,"raw_exact":ue})
    print(f"{i+1:02d}/07 | VALUE={value!r} | QSEL=R{rq:02d} | RAW=R{ru:02d} | QSEL={BASE[QTOP[i]]!r} | RAWSEL={BASE[UTOP[i]]!r}")
print("[15/21] SUMMARY")
Q=np.asarray([x["qselective_rank"] for x in ROWS]);U=np.asarray([x["raw_rank"] for x in ROWS])
QMED=float(np.median(Q));QT1=float(np.mean(Q==1));QT3=float(np.mean(Q<=3));QT5=float(np.mean(Q<=5));QEX=float(np.mean([x["qselective_exact"] for x in ROWS]))
UMED=float(np.median(U));UT1=float(np.mean(U==1));UT3=float(np.mean(U<=3));UT5=float(np.mean(U<=5));UEX=float(np.mean([x["raw_exact"] for x in ROWS]))
BETTER=float(np.mean(Q<U));WORSE=float(np.mean(Q>U))
print(f"QSELECTIVE | MED={QMED:.1f} TOP1={QT1:.3f} TOP3={QT3:.3f} TOP5={QT5:.3f} EXACT={QEX:.3f} UNIQUE={QU}/7")
print(f"RAW        | MED={UMED:.1f} TOP1={UT1:.3f} TOP3={UT3:.3f} TOP5={UT5:.3f} EXACT={UEX:.3f} UNIQUE={UU}/7")
print(f"QSELECTIVE_VS_RAW | BETTER={BETTER:.3f} | WORSE={WORSE:.3f}")
print("[16/21] ROUTING PROFILE SIMILARITY")
for i in range(7):
    z=[]
    for k in range(7):
        a=QMRR[i]-QMRR[i].mean();b=QMRR[k]-QMRR[k].mean();z.append(float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+EPS)))
    print(f"Q{i+1}: "+" ".join(f"{x:+.3f}" for x in z))
print("[17/21] PREDECLARED GATES")
SUPPORTED=bool(QMED<=2 and QT3>=.80 and QU>=4 and BETTER>=.50)
STRONG=bool(QMED<=1 and QT1>=.70 and QT3>=.85 and QU>=5 and BETTER>=.60)
print(f"SUPPORTED={SUPPORTED} | STRONG={STRONG}")
print("[18/21] CLASSIFICATION")
if STRONG:CLASS="QUESTION_SPECIFIC_HEAD_ROUTING_STRONGLY_SUPPORTED"
elif SUPPORTED:CLASS="QUESTION_SPECIFIC_HEAD_ROUTING_SUPPORTED"
elif QU>UU or BETTER>.5:CLASS="QUESTION_SPECIFIC_HEAD_ROUTING_PARTIAL"
else:CLASS="QUESTION_SPECIFIC_HEAD_ROUTING_NOT_SUPPORTED"
print("CLASS=",CLASS)
print("[19/21] SCIENTIFIC AUDIT")
print("EXACT TEST363-366 SOURCE+QUESTIONS | HEADS SEPARATE | PER-QUESTION ONE-VS-REST JS | PRIMARY=QUESTION-SPECIFIC WEIGHTED HEAD MRR | VALUE POST-HOC | NO TARGET/LAYER/HEAD ORACLE | NO SCORE SEARCH | NO SEASC | NO TRAINING")
print("[20/21] INTEGRITY")
FP1=fp()
if FP1!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
print("WEIGHTS FROZEN | FP MATCH")
print("[21/21] JSON-SAFE SAVE")
SUM={"n":7,"qselective":{"median":QMED,"top1":QT1,"top3":QT3,"top5":QT5,"exact":QEX,"unique":QU},"raw":{"median":UMED,"top1":UT1,"top3":UT3,"top5":UT5,"exact":UEX,"unique":UU},"qselective_beats_raw":BETTER,"qselective_worse_raw":WORSE,"supported":SUPPORTED,"strong":STRONG}
RES=py({"schema":"akbascore.test367.v1","test":"TEST 367","start":START,"end":utc(),"model":MODEL_ID,"source":SOURCE,"summary":SUM,"classification":CLASS,"rows":ROWS,"integrity":{"exact_test363_366_source_questions":True,"question_specific_head_selectivity":True,"target_engine_access":False,"value_posthoc_only":True,"target_head_selection":False,"layer_oracle":False,"head_oracle":False,"score_search":False,"seasc":False,"training":False,"weight_update":False,"fp_start":[float(x) for x in FP0],"fp_end":[float(x) for x in FP1]}})
sha=hashlib.sha256(canon(RES)).hexdigest();rid=f"T367-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_text(json.dumps(RES,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False),encoding="utf-8")
tp.write_text("\n".join(["="*108,"TEST 367 - SAME-SOURCE QUESTION-SPECIFIC HEAD ROUTING","="*108,f"QSELECTIVE_MEDIAN={QMED}",f"QSELECTIVE_TOP1={QT1}",f"QSELECTIVE_TOP3={QT3}",f"QSELECTIVE_TOP5={QT5}",f"QSELECTIVE_EXACT={QEX}",f"QSELECTIVE_UNIQUE={QU}",f"RAW_MEDIAN={UMED}",f"RAW_TOP3={UT3}",f"RAW_UNIQUE={UU}",f"CLASS={CLASS}",f"SHA={sha}"]),encoding="utf-8")
print("="*108);print("TEST 367 COMPLETE");print(f"QSELECTIVE | MED={QMED:.1f} | TOP1={QT1:.3f} | TOP3={QT3:.3f} | TOP5={QT5:.3f} | EXACT={QEX:.3f} | UNIQUE={QU}/7");print(f"RAW        | MED={UMED:.1f} | TOP1={UT1:.3f} | TOP3={UT3:.3f} | TOP5={UT5:.3f} | EXACT={UEX:.3f} | UNIQUE={UU}/7");print(f"QSELECTIVE_BEATS_RAW={BETTER:.3f} | WORSE={WORSE:.3f}");print("CLASS=",CLASS);print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)

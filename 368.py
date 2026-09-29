# ================================================================================================
# AKBASCORE - TEST 368
# SAME-SOURCE QUESTION-CONTRASTIVE HEAD ROUTING
# SAME TEST363-367 SOURCE + SAME 7 QUESTIONS
# PRIMARY: P(THIS QUESTION)-MEAN[P(OTHER 6)] PER HEAD/SPAN -> CROSS-HEAD/LAYER MRR
# CONTROL: RAW HEAD MRR | VALUE POST-HOC | NO SEASC | NO ORACLE | NO TRAINING
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
SEED=368;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;H=3584;EPS=1e-12
ROOT=Path("/content/AKBASCORE_TEST368");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
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
print("="*108);print("TEST 368 - SAME-SOURCE QUESTION-CONTRASTIVE HEAD ROUTING");print("="*108);print("START:",START)
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
print("[7/21] RAW HEAD RANKS")
RR=np.zeros((7,TOTAL,NH,NS),dtype=np.int32)
for i in range(7):
    for L in range(TOTAL):
        for h in range(NH):
            for j in range(NS):RR[i,L,h,j]=rank_desc(P[i,L,h],j)
print("RAW RANKS READY")
print("[8/21] QUESTION-CONTRASTIVE SPAN DISPLACEMENT")
D=np.zeros_like(P)
for i in range(7):D[i]=P[i]-np.mean(np.delete(P,i,axis=0),axis=0)
print(f"D=P(Q)-MEAN[P(OTHER6)] | MIN={D.min():+.6f} MED={np.median(D):+.6f} MAX={D.max():+.6f}")
print("[9/21] CONTRASTIVE HEAD/SPAN RANKS")
DR=np.zeros((7,TOTAL,NH,NS),dtype=np.int32)
for i in range(7):
    for L in range(TOTAL):
        for h in range(NH):
            for j in range(NS):DR[i,L,h,j]=rank_desc(D[i,L,h],j)
print("CONTRASTIVE RANKS READY")
print("[10/21] PRIMARY CROSS-HEAD/LAYER CONTRASTIVE MRR")
CMRR=np.zeros((7,NS),dtype=np.float64)
for i in range(7):
    for j in range(NS):CMRR[i,j]=float(np.mean(1.0/DR[i,:,:,j]))
print("PRIMARY READY")
print("[11/21] RAW CONTROL MRR")
RMRR=np.zeros((7,NS),dtype=np.float64)
for i in range(7):
    for j in range(NS):RMRR[i,j]=float(np.mean(1.0/RR[i,:,:,j]))
print("CONTROL READY")
print("[12/21] TARGET-FREE ROUTES FROZEN")
CTOP=[];RTOP=[]
for i in range(7):
    jc=int(np.argmax(CMRR[i]));jr=int(np.argmax(RMRR[i]));CTOP.append(jc);RTOP.append(jr);top=np.argsort(-CMRR[i])[:3]
    print(f"{i+1:02d}/07 | CONTRAST_TOP3="+" | ".join(f"{BASE[int(x)]}:{CMRR[i,int(x)]:.4f}" for x in top)+f" | RAW_TOP={BASE[jr]}")
print("[13/21] ROUTING DIVERSITY")
CU=len(set(BASE[x].lower() for x in CTOP));RU=len(set(BASE[x].lower() for x in RTOP))
print(f"CONTRAST_UNIQUE={CU}/7 | RAW_UNIQUE={RU}/7")
print("[14/21] VALUES OPEN POST-HOC")
ROWS=[]
for i,(typ,q,value) in enumerate(CASES):
    vw=re.findall(r"\b[\w'-]+\b",value,re.UNICODE);starts=[]
    for j in range(NS-len(vw)+1):
        if [x.lower() for x in BASE[j:j+len(vw)]]==[x.lower() for x in vw]:starts.append(j)
    if not starts:raise RuntimeError(f"Value missing: {value}")
    idx=int(starts[0]);ids=list(range(idx,idx+len(vw)))
    rc=min(rank_desc(CMRR[i],j) for j in ids);rr=min(rank_desc(RMRR[i],j) for j in ids);ce=CTOP[i] in ids;re_=RTOP[i] in ids
    ROWS.append({"i":i+1,"type":typ,"question":q,"value":value,"contrast_rank":rc,"raw_rank":rr,"contrast_selected":BASE[CTOP[i]],"raw_selected":BASE[RTOP[i]],"contrast_exact":ce,"raw_exact":re_})
    print(f"{i+1:02d}/07 | VALUE={value!r} | CONTRAST=R{rc:02d} | RAW=R{rr:02d} | CSEL={BASE[CTOP[i]]!r} | RSEL={BASE[RTOP[i]]!r}")
print("[15/21] SUMMARY")
C=np.asarray([x["contrast_rank"] for x in ROWS]);R=np.asarray([x["raw_rank"] for x in ROWS])
CMED=float(np.median(C));CT1=float(np.mean(C==1));CT3=float(np.mean(C<=3));CT5=float(np.mean(C<=5));CEX=float(np.mean([x["contrast_exact"] for x in ROWS]))
RMED=float(np.median(R));RT1=float(np.mean(R==1));RT3=float(np.mean(R<=3));RT5=float(np.mean(R<=5));REX=float(np.mean([x["raw_exact"] for x in ROWS]))
BETTER=float(np.mean(C<R));WORSE=float(np.mean(C>R))
print(f"CONTRAST | MED={CMED:.1f} TOP1={CT1:.3f} TOP3={CT3:.3f} TOP5={CT5:.3f} EXACT={CEX:.3f} UNIQUE={CU}/7")
print(f"RAW      | MED={RMED:.1f} TOP1={RT1:.3f} TOP3={RT3:.3f} TOP5={RT5:.3f} EXACT={REX:.3f} UNIQUE={RU}/7")
print(f"CONTRAST_VS_RAW | BETTER={BETTER:.3f} | WORSE={WORSE:.3f}")
print("[16/21] COMMON-PRIOR SUPPRESSION")
for i in range(7):
    j2026=BASE.index("2026")
    print(f"Q{i+1} | 2026 RAW_R={rank_desc(RMRR[i],j2026):02d} -> CONTRAST_R={rank_desc(CMRR[i],j2026):02d} | D_MEAN={float(D[i,:,:,j2026].mean()):+.6f}")
print("[17/21] ROUTING PROFILE SIMILARITY")
for i in range(7):
    z=[]
    for k in range(7):
        a=CMRR[i]-CMRR[i].mean();b=CMRR[k]-CMRR[k].mean();z.append(float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+EPS)))
    print(f"Q{i+1}: "+" ".join(f"{x:+.3f}" for x in z))
print("[18/21] PREDECLARED GATES")
SUPPORTED=bool(CMED<=2 and CT3>=.80 and CU>=4 and BETTER>=.50)
STRONG=bool(CMED<=1 and CT1>=.70 and CT3>=.85 and CU>=5 and BETTER>=.60)
print(f"SUPPORTED={SUPPORTED} | STRONG={STRONG}")
print("[19/21] CLASSIFICATION")
if STRONG:CLASS="QUESTION_CONTRASTIVE_HEAD_ROUTING_STRONGLY_SUPPORTED"
elif SUPPORTED:CLASS="QUESTION_CONTRASTIVE_HEAD_ROUTING_SUPPORTED"
elif CU>RU or BETTER>.5:CLASS="QUESTION_CONTRASTIVE_HEAD_ROUTING_PARTIAL"
else:CLASS="QUESTION_CONTRASTIVE_HEAD_ROUTING_NOT_SUPPORTED"
print("CLASS=",CLASS)
print("[20/21] SCIENTIFIC AUDIT + INTEGRITY")
print("EXACT TEST363-367 SOURCE+QUESTIONS | D=P(THIS)-MEAN[P(OTHER6)] | HEADS SEPARATE | PRIMARY=CONTRASTIVE HEAD MRR | VALUE POST-HOC | NO TARGET/LAYER/HEAD ORACLE | NO SCORE SEARCH | NO SEASC | NO TRAINING")
FP1=fp()
if FP1!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
print("WEIGHTS FROZEN | FP MATCH")
print("[21/21] JSON-SAFE SAVE")
SUM={"n":7,"contrast":{"median":CMED,"top1":CT1,"top3":CT3,"top5":CT5,"exact":CEX,"unique":CU},"raw":{"median":RMED,"top1":RT1,"top3":RT3,"top5":RT5,"exact":REX,"unique":RU},"contrast_beats_raw":BETTER,"contrast_worse_raw":WORSE,"supported":SUPPORTED,"strong":STRONG}
RES=py({"schema":"akbascore.test368.v1","test":"TEST 368","start":START,"end":utc(),"model":MODEL_ID,"source":SOURCE,"summary":SUM,"classification":CLASS,"rows":ROWS,"integrity":{"exact_test363_367_source_questions":True,"question_contrastive_span_score":True,"target_engine_access":False,"value_posthoc_only":True,"layer_oracle":False,"head_oracle":False,"score_search":False,"seasc":False,"training":False,"weight_update":False,"fp_start":[float(x) for x in FP0],"fp_end":[float(x) for x in FP1]}})
sha=hashlib.sha256(canon(RES)).hexdigest();rid=f"T368-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_text(json.dumps(RES,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False),encoding="utf-8")
tp.write_text("\n".join(["="*108,"TEST 368 - SAME-SOURCE QUESTION-CONTRASTIVE HEAD ROUTING","="*108,f"CONTRAST_MEDIAN={CMED}",f"CONTRAST_TOP1={CT1}",f"CONTRAST_TOP3={CT3}",f"CONTRAST_TOP5={CT5}",f"CONTRAST_EXACT={CEX}",f"CONTRAST_UNIQUE={CU}",f"RAW_MEDIAN={RMED}",f"RAW_TOP3={RT3}",f"RAW_UNIQUE={RU}",f"CLASS={CLASS}",f"SHA={sha}"]),encoding="utf-8")
print("="*108);print("TEST 368 COMPLETE");print(f"CONTRAST | MED={CMED:.1f} | TOP1={CT1:.3f} | TOP3={CT3:.3f} | TOP5={CT5:.3f} | EXACT={CEX:.3f} | UNIQUE={CU}/7");print(f"RAW      | MED={RMED:.1f} | TOP1={RT1:.3f} | TOP3={RT3:.3f} | TOP5={RT5:.3f} | EXACT={REX:.3f} | UNIQUE={RU}/7");print(f"CONTRAST_BEATS_RAW={BETTER:.3f} | WORSE={WORSE:.3f}");print("CLASS=",CLASS);print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)

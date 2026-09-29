# ================================================================================================
# AKBASCORE - TEST 357
# TARGET-FREE ATTENTION-TIMING SELECTOR
# QUESTION-LAST -> SOURCE ATTENTION -> ENDOGENOUS LAYER SELECTION -> FREEZE -> VALUE POST-HOC
# NO SEASC | NO TARGET ROUTING | NO LAYER SEARCH | NO TRAINING
# ================================================================================================
import os,sys,json,time,random,hashlib,subprocess,importlib.util,re,math
from datetime import datetime,timezone
from pathlib import Path
for m,p in [("torch","torch"),("transformers","transformers"),("numpy","numpy")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required")
SEED=357;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;H=3584;EPS=1e-12
ROOT=Path("/content/AKBASCORE_TEST357");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
CASES=[
("PERSON","During the final inspection, technician Bravik disconnected the auxiliary compressor before the alarms were tested.","Who disconnected the auxiliary compressor?","Bravik"),
("PERSON","The final alignment of the measurement rig was performed by Talora after calibration.","Who performed the final alignment?","Talora"),
("PLACE","The recovered sensor was unloaded in Dubrovnik after the research vessel completed its route.","Where was the recovered sensor unloaded?","Dubrovnik"),
("PLACE","The bronze disk was stored in Nagoya before the archaeological team continued onward.","Where was the bronze disk stored?","Nagoya"),
("COLOR","After synchronization, the status panel changed to magenta.","What color did the status panel become after synchronization?","magenta"),
("COLOR","Spectral analysis established the coating as indigo.","What was the coating's established color?","indigo"),
("MATERIAL","The engineers selected obsidian for the protective shell after testing several materials.","Which material was selected for the protective shell?","obsidian"),
("MATERIAL","The laboratory chose tungsten for the central support after evaluation.","Which material was chosen for the central support?","tungsten"),
("ACTION","Near the access hatch, Tavren began crawling beneath the support frame.","What did Tavren begin doing?","crawling"),
("ACTION","When the retaining clamp was released, the membrane began expanding.","What did the membrane begin doing?","expanding"),
("NAME","The recovered mineral was entered into the catalogue under the designation Norvex.","What designation was given to the recovered mineral?","Norvex"),
("NAME","The prototype's production name was changed to Kelvane.","What became the prototype's production name?","Kelvane"),
("NUMBER","After recalculation, the certified value was 817.","What was the certified value after recalculation?","817"),
("OBJECT","From the equipment locker, Selma retrieved a compass.","What object did Selma retrieve?","compass"),
("PROPERTY","After treatment, the sample was described as glossy.","How was the treated sample described?","glossy"),
("SHOWCASE","Mustafa Akbaş placed an amber banner beneath the Galata Bridge where the validation demonstration was conducted.","What did Mustafa Akbaş do at the location of the validation demonstration?","placed an amber banner")]
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
def rank_desc(x,j):return 1+sum(v>x[j] for v in x)
def normdist(x):
    x=np.asarray(x,dtype=np.float64);x=np.maximum(x,0);s=x.sum()
    return x/s if s>0 else np.ones_like(x)/len(x)
def entropy(p):
    p=np.maximum(normdist(p),EPS);return float(-(p*np.log(p)).sum()/max(math.log(len(p)),EPS))
def cosine_np(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64);return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+EPS))
print("="*108);print("TEST 357 - TARGET-FREE ATTENTION-TIMING SELECTOR");print("="*108);print("START:",START)
print("[1/18] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="eager",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers
if len(layers)!=TOTAL or model.config.hidden_size!=H:raise RuntimeError("Architecture mismatch")
torch.cuda.synchronize();print(f"OK | {MODEL_ID} | 28L | H={H} | eager-attention | {next(model.parameters()).dtype} | {time.perf_counter()-t:.2f}s")
print("[2/18] MODEL LOCK")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();print("FP:",[f"{x:.4f}" for x in FP0])
print("[3/18] JOINT SOURCE+QUESTION ENCODING")
DATA=[]
for i,(typ,s,q,_) in enumerate(CASES):
    text="SOURCE:\n"+s+"\n\nQUESTION:\n"+q+"\n\nANSWER:"
    e=tok(text,return_tensors="pt",add_special_tokens=False,return_offsets_mapping=True);offs=e["offset_mapping"][0].tolist();src0=len("SOURCE:\n");src1=src0+len(s);q0=text.index(q);q1=q0+len(q)
    qt=[j for j,(a,b) in enumerate(offs) if b>q0 and a<q1];words=[]
    for m in re.finditer(r"\b[\w'-]+\b",s,re.UNICODE):
        a,b=src0+m.start(),src0+m.end();ix=[j for j,(x,y) in enumerate(offs) if y>a and x<b]
        if ix:words.append({"text":m.group(0),"tok":ix})
    if not qt or not words:raise RuntimeError("Span failure")
    DATA.append({"ids":e["input_ids"].to("cuda"),"mask":e["attention_mask"].to("cuda"),"words":words,"qt":qt})
    print(f"{i+1:02d}/16 | {typ:8s} | SOURCE_SPANS={len(words)} | Q_TOK={len(qt)}")
print("[4/18] NATIVE ATTENTION FORWARDS")
ATT=[]
with torch.inference_mode():
    for i,d in enumerate(DATA):
        o=model(input_ids=d["ids"],attention_mask=d["mask"],use_cache=False,output_attentions=True,return_dict=True)
        ATT.append([a[0].float().cpu() for a in o.attentions]);print(f"{i+1:02d}/16 | LAYERS={len(o.attentions)} | HEADS={o.attentions[0].shape[1]}")
print("[5/18] QUESTION-LAST -> SOURCE DISTRIBUTIONS")
DIST=[]
for ci,d in enumerate(DATA):
    ls=[];q=d["qt"][-1]
    for L in range(TOTAL):
        a=ATT[ci][L];v=[float(a[:,q,w["tok"]].mean()) for w in d["words"]];ls.append(normdist(v).tolist())
    DIST.append(ls)
print("28 DISTRIBUTIONS/CASE FROZEN")
print("[6/18] TARGET-FREE ENTROPY")
ENT=[[entropy(DIST[i][L]) for L in range(TOTAL)] for i in range(len(DATA))]
print("ENTROPY READY")
print("[7/18] TARGET-FREE PEAK DOMINANCE")
PEAK=[]
for i in range(len(DATA)):
    z=[]
    for L in range(TOTAL):
        p=np.sort(np.asarray(DIST[i][L]))[::-1];z.append(float(p[0]-p[1] if len(p)>1 else p[0]))
    PEAK.append(z)
print("PEAK READY")
print("[8/18] TARGET-FREE ADJACENT STABILITY")
STAB=[]
for i in range(len(DATA)):
    z=[]
    for L in range(TOTAL):
        if L==0:z.append(cosine_np(DIST[i][0],DIST[i][1]))
        elif L==TOTAL-1:z.append(cosine_np(DIST[i][-2],DIST[i][-1]))
        else:z.append(.5*(cosine_np(DIST[i][L-1],DIST[i][L])+cosine_np(DIST[i][L],DIST[i][L+1])))
    STAB.append(z)
print("STABILITY READY")
print("[9/18] PREDECLARED SELECTOR SCORE")
# Fixed, target-free, no tuning: sharp + dominant + locally stable.
SEL=[]
for i in range(len(DATA)):
    e=np.asarray(ENT[i]);p=np.asarray(PEAK[i]);s=np.asarray(STAB[i])
    ez=(1-e-(1-e).mean())/((1-e).std()+EPS);pz=(p-p.mean())/(p.std()+EPS);sz=(s-s.mean())/(s.std()+EPS)
    SEL.append((ez+pz+sz).tolist())
print("SELECTOR=Z(1-ENTROPY)+Z(PEAK)+Z(STABILITY)")
print("[10/18] LAYER SELECTION FROZEN")
CHOSEN=[]
for i,d in enumerate(DATA):
    L=int(np.argmax(SEL[i]));CHOSEN.append(L);j=int(np.argmax(DIST[i][L]))
    print(f"{i+1:02d}/16 | L={L:02d} | H={ENT[i][L]:.3f} P={PEAK[i][L]:.3f} S={STAB[i][L]:.3f} | TOP={d['words'][j]['text']!r}")
print("[11/18] ALL TARGET-FREE RANKINGS FROZEN")
RANKING=[DIST[i][CHOSEN[i]] for i in range(len(DATA))]
print("FROZEN | VALUE STILL CLOSED")
print("[12/18] VALUE OPENS - SPAN MATCH")
ROWS=[]
for i,(typ,s,q,value) in enumerate(CASES):
    d=DATA[i];vw=re.findall(r"\b[\w'-]+\b",value,re.UNICODE);idx=None
    for j in range(len(d["words"])-len(vw)+1):
        if [x["text"].lower() for x in d["words"][j:j+len(vw)]]==[x.lower() for x in vw]:idx=j;break
    if idx is None:raise RuntimeError(f"Value not found: {value}")
    r=rank_desc(RANKING[i],idx);allr=[rank_desc(DIST[i][L],idx) for L in range(TOTAL)];best=min(allr);bestL=allr.index(best)
    row={"i":i+1,"type":typ,"value":value,"n":len(d["words"]),"selected_layer":CHOSEN[i],"rank":r,"oracle_best_rank":best,"oracle_best_layer":bestL,"selected_equals_oracle_layer":CHOSEN[i]==bestL};ROWS.append(row)
    print(f"{i+1:02d}/16 | {typ:8s} | VALUE={value!r} | SELECT=L{CHOSEN[i]:02d}:R{r:02d} | ORACLE=L{bestL:02d}:R{best:02d}")
print("[13/18] DEVELOPMENT SUMMARY")
D=ROWS[:15];r=np.array([x["rank"] for x in D]);P={"median":float(np.median(r)),"top1":float(np.mean(r==1)),"top3":float(np.mean(r<=3)),"top5":float(np.mean(r<=5))}
o=np.array([x["oracle_best_rank"] for x in D]);O={"median":float(np.median(o)),"top1":float(np.mean(o==1)),"top3":float(np.mean(o<=3)),"top5":float(np.mean(o<=5))}
print(f"SELECTOR | MED={P['median']:.1f} | TOP1={P['top1']:.3f} | TOP3={P['top3']:.3f} | TOP5={P['top5']:.3f}")
print(f"ORACLE   | MED={O['median']:.1f} | TOP1={O['top1']:.3f} | TOP3={O['top3']:.3f} | TOP5={O['top5']:.3f}")
print("[14/18] SELECTION DIAGNOSTIC")
eq=float(np.mean([x["selected_equals_oracle_layer"] for x in D]));dist=float(np.mean([abs(x["selected_layer"]-x["oracle_best_layer"]) for x in D]))
print(f"SELECTED_EQ_ORACLE_LAYER={eq:.3f} | MEAN_LAYER_DISTANCE={dist:.3f}")
print("[15/18] PREDECLARED GATES")
SUPPORTED=P["top1"]>=.50 and P["top3"]>=.70 and P["median"]<=2;PARTIAL=P["top3"]>=.50 and P["median"]<=3
print(f"SUPPORTED={SUPPORTED} | PARTIAL={PARTIAL}")
print("[16/18] SHOWCASE + CLASSIFICATION")
g=ROWS[15];print("-"*108);print("SOURCE:",CASES[15][1]);print("Q:",CASES[15][2]);print("VALUE:",CASES[15][3]);print(f"SELECT=L{g['selected_layer']:02d}:R{g['rank']} | ORACLE=L{g['oracle_best_layer']:02d}:R{g['oracle_best_rank']}");print("-"*108)
if SUPPORTED:CLASS="TARGET_FREE_NATIVE_ATTENTION_TIMING_SELECTOR_SUPPORTED"
elif PARTIAL:CLASS="TARGET_FREE_NATIVE_ATTENTION_TIMING_SELECTOR_PARTIAL"
elif O["top1"]>=.50 and P["top1"]<.50:CLASS="ATTENTION_BINDING_EXISTS_BUT_TARGET_FREE_TIMING_SELECTOR_FAILS"
else:CLASS="TARGET_FREE_NATIVE_ATTENTION_TIMING_SELECTOR_NOT_SUPPORTED"
print("CLASS=",CLASS)
print("[17/18] SCIENTIFIC AUDIT")
print("TEST356 JOINT CONTEXT | QUESTION-LAST ONLY | SELECTOR=ENTROPY+PEAK+STABILITY PREDECLARED | VALUE POST-HOC | ORACLE DIAGNOSTIC ONLY | NO SEASC | NO TARGET ROUTING | NO LAYER SEARCH | NO TRAINING")
print("[18/18] INTEGRITY + SAVE")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
S={"n":15,"primary":"target_free_attention_timing_selector","selector":P,"oracle":O,"selected_eq_oracle":eq,"mean_layer_distance":dist,"supported":SUPPORTED,"partial":PARTIAL}
R={"schema":"akbascore.test357.v1","test":"TEST 357","start":START,"end":utc(),"model":MODEL_ID,"summary":S,"classification":CLASS,"rows":ROWS,"integrity":{"target_engine_access":False,"value_posthoc_only":True,"oracle_diagnostic_only":True,"layer_search":False,"seasc":False,"training":False,"weight_update":False,"fp_start":FP0,"fp_end":fp()}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T357-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_text(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False),encoding="utf-8")
tp.write_text("\n".join(["="*108,"TEST 357 - TARGET-FREE ATTENTION-TIMING SELECTOR","="*108,f"N=15",f"MEDIAN={P['median']}",f"TOP1={P['top1']}",f"TOP3={P['top3']}",f"TOP5={P['top5']}",f"ORACLE_TOP1={O['top1']}",f"CLASS={CLASS}",f"SHA={sha}"]),encoding="utf-8")
print("="*108);print("TEST 357 COMPLETE");print(f"SELECTOR | MED={P['median']:.1f} | TOP1={P['top1']:.3f} | TOP3={P['top3']:.3f} | TOP5={P['top5']:.3f}");print(f"ORACLE   | MED={O['median']:.1f} | TOP1={O['top1']:.3f} | TOP3={O['top3']:.3f} | TOP5={O['top5']:.3f}");print("CLASS=",CLASS);print("VALUE POST-HOC ONLY | NO SEASC | WEIGHTS FROZEN");print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)

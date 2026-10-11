
# TEST618 — AKBASCORE MAM · FROZEN QWEN · TOKEN-LEVEL ROUTER DIAGNOSIS
# TEST617 backbone preserved. No training, no CUT3 modification, no model-weight changes.
# Stage A: 48 TEST617 diagnostic queries. Stage B: 48 disjoint validation queries.
# Compare MEAN-CSR, TOKEN-CSR, TOKEN-COS and native FRAM reference.
import os,sys,time,json,math,hashlib,urllib.request,random
import numpy as np,torch
from collections import defaultdict
T618=time.perf_counter();TEST=618;SEED618=616616
ROOT618="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/"
SHA610618="a62611a4ffd91c4f" # replaced by full immutable SHA below
SHA610618="a62611a4ffd91c2fcf03796c31da2af11ccd8141345655b562f3a614568af3e7"
OUT618="/content/TEST618_RESULT.json"
req=urllib.request.Request(ROOT618+"610.py",headers={"User-Agent":"AKBASCORE-TEST618"})
with urllib.request.urlopen(req,timeout=120) as f:blob618=f.read()
sha618=hashlib.sha256(blob618).hexdigest()
if sha618!=SHA610618:raise RuntimeError("SOURCE SHA LOCK FAILURE "+sha618)
print("TEST618_SOURCE_LOCK",sha618,flush=True)
src618=blob618.decode("utf-8");boundary618="print('[6/7] Demo acceptance"
if src618.count(boundary618)!=1:raise RuntimeError("SOURCE EXECUTION BOUNDARY CHANGED")
exec(compile(src618.split(boundary618,1)[0],"610.py","exec"),globals())
assert len(CAR)==96 and CFG==(3,0,8.0) and H1==12 and H5==12
assert len(PRE_AUDIT)==12 and sum(int(x["answer_ok"]) for x in PRE_AUDIT)==12
assert not any(p.requires_grad for p in model.parameters())
WB618=WB;SB618=SB
print("TEST618_BASELINE_LOCK",json.dumps({"cartridges":96,"fram_hold_r1":H1,"fram_hold_r5":H5,"cut3_correct":12}),flush=True)
FIT618=transform_fit(torch.cat(KS[3],dim=0),0)
K618=[apply(x,FIT618) for x in KS[3]]
if not all(torch.equal(a,b) for a,b in zip(K618,KBASE)):raise RuntimeError("FRAM FRAME CHANGED")
rng618=random.Random(SEED618);alphabet618="ABCDEFGHJKLMNPQRSTUVWXYZ"
def word618(n):return "".join(rng618.choice(alphabet618) for _ in range(n)).capitalize()
seen618={c["fact"].casefold() for c in CAR};FACT618=[];META618=[]
while len(FACT618)<1440:
    i=len(FACT618);city=word618(9);country="ZX"+str(100000+i)
    fact=f"The current capital of {country} is {city}."
    if fact.casefold() in seen618:continue
    seen618.add(fact.casefold());FACT618.append(fact)
    META618.append({"cartridge":96+i,"country":country,"city":city,"fact":fact})
NEW_TOKENS618=[]
for i,fact in enumerate(FACT618):
    full=source(fact);begin=len(PREFIX);end=begin+len(fact)
    rep,info=capture_residual(full,begin,end)
    K618.append(apply(rep[3],FIT618));NEW_TOKENS618.append(int(info["n"]))
    if (i+1)%144==0:print("TEST618_NATIVE_CAPTURE",i+1,"/1440","elapsed_s",round(time.perf_counter()-T618,2),flush=True)
assert len(K618)==1536
# Reproduce TEST617 diagnostic selection exactly, then draw an untouched validation split.
select618=random.Random(6172026)
DIAG_IDS618=sorted(select618.sample(range(1440),48))
remaining618=sorted(set(range(1440))-set(DIAG_IDS618))
VALID_IDS618=sorted(random.Random(6182026).sample(remaining618,48))
assert not(set(DIAG_IDS618)&set(VALID_IDS618))
PANEL618=[]
for split,ids in (("DIAG",DIAG_IDS618),("VALID",VALID_IDS618)):
    for j in ids:
        m=META618[j]
        PANEL618.append({"split":split,"target":m["cartridge"],"country":m["country"],"gold":m["city"],
            "question":f"What is the current capital of {m['country']}?"})
assert len(PANEL618)==96 and len({x["question"] for x in PANEL618})==96
Q618=[]
for i,row in enumerate(PANEL618):
    rep,info=question_rep(row["question"])
    Q618.append(apply(rep[3],FIT618))
    if (i+1)%16==0:print("TEST618_QUESTION_CAPTURE",i+1,"/96",flush=True)
def pool618(x):
    return torch.nn.functional.normalize(x.float().mean(0),dim=0,eps=1e-9).cpu().numpy().astype(np.float32)
V618=np.stack([pool618(x) for x in K618])
U618=np.stack([pool618(x) for x in Q618])
N618=len(K618);D618=V618.shape[1];KSP618=48
def sparse618(x,k=KSP618):
    ids=np.argpartition(np.abs(x),-k)[-k:]
    ids=ids[np.argsort(-np.abs(x[ids]),kind="stable")]
    return (ids.astype(np.int32)*2+(x[ids]<0).astype(np.int32)).astype(np.int32)
def build_csr618(v):
    bins=[[] for _ in range(2*D618)]
    for ci,x in enumerate(v):
        for p in sparse618(x):bins[int(p)].append(ci)
    ptr=np.zeros(len(bins)+1,dtype=np.int32)
    for i,b in enumerate(bins):ptr[i+1]=ptr[i]+len(b)
    ids=np.empty(int(ptr[-1]),dtype=np.int32)
    for i,b in enumerate(bins):ids[ptr[i]:ptr[i+1]]=b
    return ptr,ids
MEAN_PTR618,MEAN_IDS618=build_csr618(V618)
# TOKEN CSR: index each native cartridge token, preserve token-to-cartridge provenance.
# Same SIGNED k48 coordinate family, but avoid destructive token averaging.
TOKEN_BINS618=[[] for _ in range(2*D618)]
TOKEN_COUNT618=0
for ci,x in enumerate(K618):
    a=x.numpy()
    for token in a:
        for p in sparse618(token):TOKEN_BINS618[int(p)].append(ci)
        TOKEN_COUNT618+=1
TOKEN_PTR618=np.zeros(2*D618+1,dtype=np.int32)
for p,b in enumerate(TOKEN_BINS618):TOKEN_PTR618[p+1]=TOKEN_PTR618[p]+len(b)
TOKEN_IDS618=np.empty(int(TOKEN_PTR618[-1]),dtype=np.int32)
for p,b in enumerate(TOKEN_BINS618):TOKEN_IDS618[TOKEN_PTR618[p]:TOKEN_PTR618[p+1]]=b
del TOKEN_BINS618
# Native token vectors and owner mapping for exact token-cosine screening.
TOK618=np.concatenate([x.numpy() for x in K618],axis=0).astype(np.float32)
OWN618=np.concatenate([np.full(len(x),ci,dtype=np.int32) for ci,x in enumerate(K618)])
OFF618=np.zeros(N618+1,dtype=np.int32)
for ci,x in enumerate(K618):OFF618[ci+1]=OFF618[ci]+len(x)
assert len(TOK618)==len(OWN618)==int(OFF618[-1])
print("TEST618_INDEX",json.dumps({"cartridges":N618,"dimension":D618,"native_tokens":len(TOK618),
    "mean_csr_bytes":int(MEAN_PTR618.nbytes+MEAN_IDS618.nbytes),
    "token_csr_bytes":int(TOKEN_PTR618.nbytes+TOKEN_IDS618.nbytes),
    "token_vectors_bytes":int(TOK618.nbytes),"owner_bytes":int(OWN618.nbytes)}),flush=True)
def csr_route618(q,ptr,ids,mode):
    counts=np.zeros(N618,dtype=np.float32);visits=0
    if mode=="MEAN":
        postings=[sparse618(q.mean(0)/max(np.linalg.norm(q.mean(0)),1e-9))]
    else:
        postings=[sparse618(t) for t in q]
    for arr in postings:
        for p in arr:
            p=int(p);lo=int(ptr[p]);hi=int(ptr[p+1]);owners=ids[lo:hi]
            visits+=len(owners)
            if len(owners):np.add.at(counts,owners,1.0)
    raw=np.flatnonzero(counts>0)
    order=raw[np.lexsort((raw,-counts[raw]))]
    return order,visits,len(raw)
def token_cos618(q):
    # Exact maximum token-pair cosine per cartridge. Diagnostic candidate scorer,
    # not a substitute for the locked native FRAM reranker.
    score=np.full(N618,-np.inf,dtype=np.float32)
    for qt in q:
        sim=TOK618@qt
        np.maximum.at(score,OWN618,sim)
    return np.lexsort((np.arange(N618),-score)),score
def native_order618(qi,candidates):
    q=Q618[qi];candidates=np.asarray(candidates,dtype=np.int32)
    scores=np.asarray([pair_scores(q,K618[int(ci)],8.0) for ci in candidates],dtype=np.float32)
    return candidates[np.lexsort((candidates,-scores))]
# Do not inspect validation answers while choosing the router.
METHODS618=("MEAN_CSR","TOKEN_CSR","TOKEN_COS")
BUDGETS618=(4,8,16,24,32,48,64,96,128,256)
ROUTES618={};TIMES618={};ROUTE_INFO618={}
for qi,q in enumerate(Q618):
    qa=q.numpy();ROUTES618[qi]={};TIMES618[qi]={};ROUTE_INFO618[qi]={}
    t=time.perf_counter()
    order,visits,raw=csr_route618(qa,MEAN_PTR618,MEAN_IDS618,"MEAN")
    ROUTES618[qi]["MEAN_CSR"]=order;TIMES618[qi]["MEAN_CSR"]=(time.perf_counter()-t)*1000
    ROUTE_INFO618[qi]["MEAN_CSR"]={"visits":visits,"raw":raw}
    t=time.perf_counter()
    order,visits,raw=csr_route618(qa,TOKEN_PTR618,TOKEN_IDS618,"TOKEN")
    ROUTES618[qi]["TOKEN_CSR"]=order;TIMES618[qi]["TOKEN_CSR"]=(time.perf_counter()-t)*1000
    ROUTE_INFO618[qi]["TOKEN_CSR"]={"visits":visits,"raw":raw}
    t=time.perf_counter()
    order,score=token_cos618(qa)
    ROUTES618[qi]["TOKEN_COS"]=order;TIMES618[qi]["TOKEN_COS"]=(time.perf_counter()-t)*1000
    ROUTE_INFO618[qi]["TOKEN_COS"]={"visits":int(len(qa)*len(TOK618)),"raw":N618}
    if (qi+1)%12==0:print("TEST618_ROUTING",qi+1,"/96","elapsed_s",round(time.perf_counter()-T618,2),flush=True)
# Native full-bank reference, diagnostic and validation separately.
FULL618=[]
for qi,row in enumerate(PANEL618):
    order=native_order618(qi,np.arange(N618,dtype=np.int32))
    rank=int(np.where(order==row["target"])[0][0])+1
    FULL618.append({"query":qi,"split":row["split"],"target":row["target"],"rank":rank})
    if (qi+1)%12==0:print("TEST618_FULL_NATIVE",qi+1,"/96",flush=True)
# Evaluate each routing method against native FRAM. This does not alter the native scorer.
# Candidate coverage is computed before FRAM. R1/R5 after native reranking.
EVAL618=[];CASES618=[]
for split in ("DIAG","VALID"):
    subset=[i for i,x in enumerate(PANEL618) if x["split"]==split]
    for method in METHODS618:
        for budget in BUDGETS618:
            coverage=r1=r5=0;details=[]
            for qi in subset:
                target=PANEL618[qi]["target"]
                selected=ROUTES618[qi][method][:budget]
                present=int(target in selected);coverage+=present
                ranked=native_order618(qi,selected) if len(selected) else np.array([],dtype=np.int32)
                hit1=int(len(ranked)>0 and ranked[0]==target)
                hit5=int(target in ranked[:5])
                r1+=hit1;r5+=hit5
                details.append({"query":qi,"target":target,"present":present,"r1":hit1,"r5":hit5,
                    "candidate_rank":int(np.where(ROUTES618[qi][method]==target)[0][0])+1
                    if target in ROUTES618[qi][method] else None})
            result={"split":split,"method":method,"budget":budget,"total":len(subset),
                "candidate_hit":coverage,"r1":r1,"r5":r5,
                "route_p50_ms":round(float(np.percentile([TIMES618[i][method] for i in subset],50)),4),
                "route_p95_ms":round(float(np.percentile([TIMES618[i][method] for i in subset],95)),4),
                "mean_raw":round(float(np.mean([ROUTE_INFO618[i][method]["raw"] for i in subset])),3),
                "mean_visits":round(float(np.mean([ROUTE_INFO618[i][method]["visits"] for i in subset])),3)}
            EVAL618.append(result);CASES618.append({"split":split,"method":method,"budget":budget,"cases":details})
            print("TEST618_RESULT",json.dumps(result),flush=True)
# Select using DIAG only. Validation must not influence this decision.
DIAG618=[x for x in EVAL618 if x["split"]=="DIAG"]
SELECT618=max(DIAG618,key=lambda x:(x["r1"],x["r5"],x["candidate_hit"],-x["budget"],-METHODS618.index(x["method"])))
VALID618=next(x for x in EVAL618 if x["split"]=="VALID" and x["method"]==SELECT618["method"] and x["budget"]==SELECT618["budget"])
BASE_DIAG618=next(x for x in EVAL618 if x["split"]=="DIAG" and x["method"]=="MEAN_CSR" and x["budget"]==16)
BASE_VALID618=next(x for x in EVAL618 if x["split"]=="VALID" and x["method"]=="MEAN_CSR" and x["budget"]==16)
FULL_SUM618={}
for split in ("DIAG","VALID"):
    ranks=[x["rank"] for x in FULL618 if x["split"]==split]
    FULL_SUM618[split]={"total":len(ranks),"r1":sum(r==1 for r in ranks),"r5":sum(r<=5 for r in ranks),
        "r16":sum(r<=16 for r in ranks),"median_rank":float(np.median(ranks))}
INTEGRITY618={"weight_unchanged":weight_sha()==WB618,"sentinel_unchanged":sentinel()==SB618,
    "zero_trainable":not any(p.requires_grad for p in model.parameters()),
    "no_remaining_hooks":not any(layer._forward_hooks or layer._forward_pre_hooks for layer in model.model.layers)}
RESULT618={"test":618,"source_sha256":sha618,"bank_size":N618,"new_qwen_encoded_records":1440,
    "diag_queries":48,"validation_queries":48,"full_native":FULL_SUM618,
    "baseline_diag":BASE_DIAG618,"baseline_validation":BASE_VALID618,
    "selected_on_diag":SELECT618,"untouched_validation":VALID618,
    "index_bytes":{"mean_csr":int(MEAN_PTR618.nbytes+MEAN_IDS618.nbytes),
        "token_csr":int(TOKEN_PTR618.nbytes+TOKEN_IDS618.nbytes),
        "token_vectors":int(TOK618.nbytes),"token_owners":int(OWN618.nbytes)},
    "all_results":EVAL618,"case_details":CASES618,"full_native_cases":FULL618,
    "integrity":INTEGRITY618,"seconds":round(time.perf_counter()-T618,2),
    "limitations":["Validation questions are disjoint but share the same synthetic fact/question template",
        "Validation bank and routing representations are constructed before selection; validation labels are not used to select method",
        "TOKEN_COS scans all stored tokens and is a diagnostic reference, not a sparse scalable index",
        "TOKEN_CSR uses repeated posting visits; memory and latency must be compared to MEAN_CSR",
        "Routing latency excludes native FRAM reranking and disk I/O",
        "CUT3 generation and three-disk capacity remain untested"]}
os.makedirs(os.path.dirname(OUT618),exist_ok=True)
with open(OUT618,"w",encoding="utf-8") as f:json.dump(RESULT618,f,ensure_ascii=False,indent=2)
print("TEST618_FINAL",json.dumps({"full_native":FULL_SUM618,"baseline_diag":BASE_DIAG618,
    "baseline_validation":BASE_VALID618,"selected_on_diag":SELECT618,
    "untouched_validation":VALID618,"index_bytes":RESULT618["index_bytes"],
    "integrity":INTEGRITY618,"output":OUT618},ensure_ascii=False),flush=True)
if not all(INTEGRITY618.values()):raise RuntimeError("TEST618 INTEGRITY FAILURE")
print("TEST618_STATUS COMPLETE",flush=True)

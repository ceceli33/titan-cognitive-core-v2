
# TEST617 — AKBASCORE MAM · UNSEEN QUERY / NEW CARTRIDGE / NEGATIVE CONTROL
# Single-cell Colab | Frozen TEST610 backbone | SIGNED K48 CSR | Native FRAM
# 1440 genuinely Qwen-encoded new records; 48 new positive questions; 12 absent-record controls
# No CUT3 files for new records. No answer-generation claims. No trained router.
import os,sys,time,json,math,hashlib,urllib.request,random
import numpy as np,torch
T617=time.perf_counter();TEST617=617;SEED617=616616
ROOT617="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/"
LOCK617="a62611a4ffd91c2fcf03796c31da2af11ccd8141345655b562f3a614568af3e7"
OUT617="/content/TEST617_RESULT.json"
req617=urllib.request.Request(ROOT617+"610.py",headers={"User-Agent":"AKBASCORE-TEST617"})
with urllib.request.urlopen(req617,timeout=120) as f:blob617=f.read()
sha617=hashlib.sha256(blob617).hexdigest()
if sha617!=LOCK617:raise RuntimeError("TEST610 SHA LOCK FAILED: "+sha617)
print("TEST617_SOURCE_LOCK",sha617,flush=True)
source617=blob617.decode("utf-8")
cut617="print('[6/7] Demo acceptance"
if source617.count(cut617)!=1:raise RuntimeError("TEST610 EXECUTION BOUNDARY CHANGED")
exec(compile(source617.split(cut617,1)[0],"610.py","exec"),globals())
assert CFG==(3,0,8.0) and H1==12 and H5==12 and len(CAR)==96
assert len(PRE_AUDIT)==12 and sum(int(x["answer_ok"]) for x in PRE_AUDIT)==12
assert not any(p.requires_grad for p in model.parameters())
print("TEST617_BASELINE_PASS",json.dumps({"native_cartridges":96,"fram_hold_r1":H1,"fram_hold_r5":H5,"cut3_hold_correct":12}),flush=True)
WB617=WB;SB617=SB
# Preserve the locked L3 / rank0 frame and native tokenwise FRAM scorer.
FIT617=transform_fit(torch.cat(KS[3],dim=0),0)
K617=[apply(x,FIT617) for x in KS[3]]
if not all(torch.equal(a,b) for a,b in zip(K617,KBASE)):raise RuntimeError("FRAME LOCK MISMATCH")
# Reconstruct the deterministic TEST616 record generator exactly as supplied.
rng617=random.Random(SEED617)
ALPH617="ABCDEFGHJKLMNPQRSTUVWXYZ"
def word617(n):return "".join(rng617.choice(ALPH617) for _ in range(n)).capitalize()
existing617={c["fact"].casefold() for c in CAR}
FACTS617=[];META617=[]
while len(FACTS617)<1440:
    i=len(FACTS617);city=word617(9);country="ZX"+str(100000+i)
    fact=f"The current capital of {country} is {city}."
    if fact.casefold() in existing617:continue
    existing617.add(fact.casefold())
    FACTS617.append(fact)
    META617.append({"cartridge":96+i,"country":country,"city":city,"fact":fact})
print("TEST617_NEW_RECORD_GENERATOR",len(FACTS617),"PASS",flush=True)
# Questions are not supplied to the model during memory encoding.
NEWLEN617=[]
for i,fact in enumerate(FACTS617):
    full=source(fact);begin=len(PREFIX);end=begin+len(fact)
    rep,info=capture_residual(full,begin,end)
    K617.append(apply(rep[3],FIT617))
    NEWLEN617.append(int(info["n"]))
    if (i+1)%144==0:
        print("TEST617_NATIVE_CAPTURE",i+1,"/1440","elapsed_s",round(time.perf_counter()-T617,2),flush=True)
assert len(K617)==1536
# Fixed, independently selected new-record IDs; no tuning on their answers.
select617=random.Random(6172026)
POS617=sorted(select617.sample(range(1440),48))
NEG617=sorted(select617.sample(range(200000,300000),12))
QUEST617=[]
for j in POS617:
    m=META617[j]
    q=f"What is the current capital of {m['country']}?"
    QUEST617.append({"kind":"positive","target":m["cartridge"],"question":q,"gold":m["city"],"country":m["country"]})
for j in NEG617:
    code="ZX"+str(j)
    QUEST617.append({"kind":"negative","target":None,"question":f"What is the current capital of {code}?","gold":None,"country":code})
assert len(QUEST617)==60 and len({x["question"] for x in QUEST617})==60
assert not any(x["country"] in {m["country"] for m in META617} for x in QUEST617 if x["kind"]=="negative")
# Encode new questions with the original source-free question_rep path.
Q617=[]
for i,row in enumerate(QUEST617):
    rep,info=question_rep(row["question"])
    Q617.append(apply(rep[3],FIT617))
    if (i+1)%12==0:print("TEST617_NEW_QUESTION_CAPTURE",i+1,"/60",flush=True)
def pooled617(x):
    return torch.nn.functional.normalize(x.float().mean(0),dim=0,eps=1e-9).cpu().numpy().astype(np.float32)
V617=np.stack([pooled617(x) for x in K617])
U617=np.stack([pooled617(x) for x in Q617])
KSP617=48;N617=len(K617);DIM617=V617.shape[1]
def sparse617(x,k=KSP617):
    ids=np.argpartition(np.abs(x),-k)[-k:]
    ids=ids[np.argsort(-np.abs(x[ids]),kind="stable")]
    return (ids.astype(np.int32)*2+(x[ids]<0).astype(np.int32)).astype(np.int32)
def csr617(v):
    n,d=v.shape;bins=[[] for _ in range(2*d)]
    for i in range(n):
        for p in sparse617(v[i]):bins[int(p)].append(i)
    ptr=np.zeros(2*d+1,dtype=np.int32)
    for p,b in enumerate(bins):ptr[p+1]=ptr[p]+len(b)
    ids=np.empty(int(ptr[-1]),dtype=np.int32)
    for p,b in enumerate(bins):ids[ptr[p]:ptr[p+1]]=b
    return ptr,ids
PTR617,IDS617=csr617(V617)
CSR_BYTES617=int(PTR617.nbytes+IDS617.nbytes)
print("TEST617_CSR",json.dumps({"cartridges":N617,"k":KSP617,"bytes":CSR_BYTES617,"mib":round(CSR_BYTES617/1048576,5)}),flush=True)
def route617(q,weighted=False):
    counts=np.zeros(N617,dtype=np.float32);visits=0
    for p in sparse617(q):
        p=int(p);lo=int(PTR617[p]);hi=int(PTR617[p+1]);ids=IDS617[lo:hi]
        visits+=len(ids)
        if len(ids):
            w=math.log1p(N617/(1+len(ids))) if weighted else 1.0
            np.add.at(counts,ids,w)
    raw=np.flatnonzero(counts>0)
    order=raw[np.lexsort((raw,-counts[raw]))]
    return order,visits,len(raw)
ROUTES617={}
for weighted in (False,True):
    rows=[]
    for qi,q in enumerate(U617):
        t=time.perf_counter()
        ids,visits,raw=route617(q,weighted)
        ms=(time.perf_counter()-t)*1000
        rows.append({"ids":ids,"visits":visits,"raw":raw,"ms":ms})
    ROUTES617[weighted]=rows
# Cache true native FRAM scores across all 1536 records for every new question.
# This reference tests whether the memory itself can distinguish the new records.
SCORES617=np.empty((len(Q617),N617),dtype=np.float32)
for qi,q in enumerate(Q617):
    for ci,k in enumerate(K617):SCORES617[qi,ci]=pair_scores(q,k,8.0)
    if (qi+1)%6==0:
        print("TEST617_FULL_NATIVE",qi+1,"/60","elapsed_s",round(time.perf_counter()-T617,2),flush=True)
def order_native617(qi,ids):
    ids=np.asarray(ids,dtype=np.int32)
    return ids[np.lexsort((ids,-SCORES617[qi,ids]))]
FULL617=[];POS_RANKS617=[]
for qi,row in enumerate(QUEST617):
    order=order_native617(qi,np.arange(N617,dtype=np.int32))
    if row["kind"]=="positive":
        rank=int(np.where(order==row["target"])[0][0])+1
        POS_RANKS617.append(rank)
        FULL617.append({"query":qi,"kind":"positive","target":row["target"],"full_native_rank":rank,"top1":int(order[0]),"gold":row["gold"]})
    else:
        FULL617.append({"query":qi,"kind":"negative","target":None,"full_native_rank":None,"top1":int(order[0]),"top_score":float(SCORES617[qi,order[0]])})
print("TEST617_FULL_NATIVE_SUMMARY",json.dumps({"new_positive_queries":48,"r1":sum(x==1 for x in POS_RANKS617),"r5":sum(x<=5 for x in POS_RANKS617),"r16":sum(x<=16 for x in POS_RANKS617),"median_rank":float(np.median(POS_RANKS617))}),flush=True)
# Routing metrics are scored separately from full native retrieval.
BUDGETS617=(4,8,16,24,32,48,64,96,128,256)
SWEEP617=[];DETAIL617=[]
for weighted in (False,True):
    rows=ROUTES617[weighted]
    for budget in BUDGETS617:
        r1=r5=coverage=0;posdetail=[]
        for qi,row in enumerate(QUEST617):
            if row["kind"]!="positive":continue
            selected=rows[qi]["ids"][:budget]
            found=int(row["target"] in selected)
            coverage+=found
            if len(selected):
                ranked=order_native617(qi,selected)
                top1=int(ranked[0]);top5=[int(x) for x in ranked[:5]]
            else:
                top1=-1;top5=[]
            r1+=int(top1==row["target"])
            r5+=int(row["target"] in top5)
            posdetail.append({"query":qi,"target":row["target"],"candidate_hit":found,"r1":int(top1==row["target"]),"r5":int(row["target"] in top5),"top1":top1})
        summary={"weighted":weighted,"budget":budget,"positive_total":48,"candidate_hit":coverage,"r1":r1,"r5":r5,
            "mean_raw_candidates":round(float(np.mean([x["raw"] for x in rows])),3),
            "mean_posting_visits":round(float(np.mean([x["visits"] for x in rows])),3),
            "route_p50_ms":round(float(np.percentile([x["ms"] for x in rows],50)),4),
            "route_p95_ms":round(float(np.percentile([x["ms"] for x in rows],95)),4),
            "csr_bytes":CSR_BYTES617}
        SWEEP617.append(summary)
        DETAIL617.append({"weighted":weighted,"budget":budget,"cases":posdetail})
        print("TEST617_ROUTE_RESULT",json.dumps(summary),flush=True)
# Negative controls: absent IDs must never be mislabeled as known answers.
# Without a calibrated rejection threshold, a nearest-neighbor router always returns
# a cartridge. Thus we report score distributions, not a fabricated abstention PASS.
NEG_SCORES617=[];POS_SCORES617=[]
for qi,row in enumerate(QUEST617):
    mx=float(np.max(SCORES617[qi]))
    if row["kind"]=="negative":NEG_SCORES617.append(mx)
    else:POS_SCORES617.append(float(SCORES617[qi,row["target"]]))
NEGATIVE617={"negative_count":len(NEG_SCORES617),"positive_count":len(POS_SCORES617),
    "negative_top_score_min":round(min(NEG_SCORES617),6),
    "negative_top_score_median":round(float(np.median(NEG_SCORES617)),6),
    "negative_top_score_max":round(max(NEG_SCORES617),6),
    "positive_gold_score_min":round(min(POS_SCORES617),6),
    "positive_gold_score_median":round(float(np.median(POS_SCORES617)),6),
    "positive_gold_score_max":round(max(POS_SCORES617),6),
    "abstention_verified":False,
    "interpretation":"Diagnostic score distributions only. No calibrated unknown-answer rejection rule."}
print("TEST617_NEGATIVE_CONTROL",json.dumps(NEGATIVE617),flush=True)
best617=max(SWEEP617,key=lambda x:(x["r1"],x["r5"],-x["budget"],not x["weighted"]))
# Extra fixed-budget comparison for the TEST616 selected configuration.
locked16_617=next(x for x in SWEEP617 if x["weighted"] is False and x["budget"]==16)
INTEGRITY617={"weight_unchanged":weight_sha()==WB617,"sentinel_unchanged":sentinel()==SB617,
    "zero_trainable":not any(p.requires_grad for p in model.parameters()),
    "no_remaining_hooks":not any(layer._forward_hooks or layer._forward_pre_hooks for layer in model.model.layers)}
RESULT617={"test":617,"source_sha256":sha617,"bank_size":N617,"new_qwen_encoded_records":1440,
    "positive_new_questions":48,"negative_unknown_questions":12,
    "baseline":{"fram_hold_r1":H1,"fram_hold_r5":H5,"cut3_hold_correct":12},
    "full_native":{"r1":sum(x==1 for x in POS_RANKS617),"r5":sum(x<=5 for x in POS_RANKS617),
        "r16":sum(x<=16 for x in POS_RANKS617),"median_rank":float(np.median(POS_RANKS617))},
    "locked_test616_unweighted_budget16":locked16_617,"best_observed":best617,
    "sweep":SWEEP617,"native_cases":FULL617,"route_cases":DETAIL617,
    "negative_control":NEGATIVE617,"csr_bytes":CSR_BYTES617,"integrity":INTEGRITY617,
    "limitations":["New questions target synthetic records generated deterministically before question encoding",
        "Question template is related to the fact template; not a semantic paraphrase generalization benchmark",
        "The 48 positive queries were not used to fit the L3 rank0 frame",
        "Budget sweep is diagnostic; best observed setting must not be called a new held-out validation",
        "No new CUT3 files or end-to-end answer generation",
        "No calibrated negative rejection threshold",
        "Measured router latency excludes FRAM scoring and disk I/O",
        "60 TB / three-disk example remains a theoretical projection"],
    "seconds":round(time.perf_counter()-T617,2),"output":OUT617}
os.makedirs(os.path.dirname(OUT617),exist_ok=True)
with open(OUT617,"w",encoding="utf-8") as f:json.dump(RESULT617,f,ensure_ascii=False,indent=2)
print("TEST617_FINAL",json.dumps({"test":617,"bank_size":N617,"new_positive_questions":48,
    "negative_questions":12,"full_native":RESULT617["full_native"],
    "locked_budget16":locked16_617,"best_observed":best617,
    "negative_control":NEGATIVE617,"integrity":INTEGRITY617,
    "output":OUT617},ensure_ascii=False),flush=True)
if not all(INTEGRITY617.values()):raise RuntimeError("TEST617 INTEGRITY FAILURE")
print("TEST617_STATUS COMPLETE",flush=True)

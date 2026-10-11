
# TEST616 — AKBASCORE MAM · NATIVE BANK SCALING + SIGNED CSR ROUTER + FRAM BUDGET SWEEP
# Single-cell Google Colab | Frozen Qwen2.5-7B | TEST610/575/566 SHA-locked backbone
# 96 / 384 / 1536 Qwen-encoded records | budgets 4..256 | native FRAM reranking
# Original 24 questions are previously seen; this is NOT an independent new holdout.
import os,sys,time,json,math,hashlib,urllib.request,random,gc
import numpy as np,torch
from collections import defaultdict
T616=time.perf_counter();TEST="616";SEED616=616616
ROOT616="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/"
SHA610="a62611a4ffd91c2fcf03796c31da2af11ccd8141345655b562f3a614568af3e7"
OUT616="/content/TEST616_RESULT.json"
def load_source616():
    req=urllib.request.Request(ROOT616+"610.py",headers={"User-Agent":"AKBASCORE-TEST616"})
    with urllib.request.urlopen(req,timeout=120) as f:b=f.read()
    actual=hashlib.sha256(b).hexdigest()
    if actual!=SHA610:raise RuntimeError(f"TEST610 SOURCE SHA MISMATCH: {actual}")
    print("TEST616_SOURCE_LOCK",actual,flush=True)
    return b.decode("utf-8")
src616=load_source616()
boundary616="print('[6/7] Demo acceptance"
if src616.count(boundary616)!=1:raise RuntimeError("TEST610 BOUNDARY MISMATCH")
exec(compile(src616.split(boundary616,1)[0],"610.py","exec"),globals())
assert len(CAR)==96 and CFG==(3,0,8.0)
assert H1==12 and H5==12
assert len(PRE_AUDIT)==12 and sum(int(x["answer_ok"]) for x in PRE_AUDIT)==12
assert not any(p.requires_grad for p in model.parameters())
print("TEST616_BASELINE_PASS",json.dumps({"native_cartridges":96,"fram_hold_r1":H1,"fram_hold_r5":H5,"cut3_correct":12,"cfg":list(CFG)}),flush=True)
WB616=WB;SB616=SB;NATIVE616=96
# The original 96-cartridge frame, model, questions and CUT3 engine remain unchanged.
FIT616=transform_fit(torch.cat(KS[3],dim=0),0)
assert FIT616[1] is None
BASE_K616=[apply(v,FIT616) for v in KS[3]]
for a,b in zip(BASE_K616,KBASE):
    if not torch.equal(a,b):raise RuntimeError("FRAME LOCK FAILED")
Q616=[QBASE[i] for i in range(24)]
TARGET616=[PANEL[i]["target"] for i in range(24)]
print("TEST616_FRAME_LOCK PASS",flush=True)
# Generate independent distractor facts. All are genuinely encoded through frozen Qwen.
# This generator is explicit and deterministic; it is not asserted to be identical
# to an earlier TEST616 generator whose full source is unavailable.
rng616=random.Random(SEED616)
alphabet616="ABCDEFGHJKLMNPQRSTUVWXYZ"
def word616(n):
    return "".join(rng616.choice(alphabet616) for _ in range(n)).capitalize()
seen616=set()
for c in CAR:
    seen616.add(c["fact"].casefold())
newfacts616=[]
while len(newfacts616)<1440:
    i=len(newfacts616)
    city=word616(9)
    country="ZX"+str(100000+i)
    fact=f"The current capital of {country} is {city}."
    if fact.casefold() in seen616:continue
    seen616.add(fact.casefold())
    newfacts616.append(fact)
NEWK616=[];NEW_TOKENS616=[]
for i,fact in enumerate(newfacts616):
    text616=source(fact)
    begin616=len(PREFIX);end616=begin616+len(fact)
    rep616,meta616=capture_residual(text616,begin616,end616)
    NEWK616.append(apply(rep616[3],FIT616))
    NEW_TOKENS616.append(int(meta616["n"]))
    if (i+1)%96==0:
        print("TEST616_NATIVE_CAPTURE",i+1,"/1440","elapsed_s",round(time.perf_counter()-T616,2),flush=True)
ALLK616=BASE_K616+NEWK616
assert len(ALLK616)==1536
# Build fixed-dimensional mean-pooled vectors for sparse posting traversal only.
# Native FRAM scores always use the original tokenwise pair_scores(...,beta=8).
def pool616(seq):
    return torch.nn.functional.normalize(seq.float().mean(0),dim=0,eps=1e-9).cpu().numpy().astype(np.float32)
V616=np.stack([pool616(x) for x in ALLK616])
U616=np.stack([pool616(x) for x in Q616])
DIM616=V616.shape[1]
KSPARSE616=48
def top_signed616(x,k):
    ids=np.argpartition(np.abs(x),-k)[-k:]
    ids=ids[np.argsort(-np.abs(x[ids]),kind="stable")]
    return (ids.astype(np.int32)*2+(x[ids]<0).astype(np.int32)).astype(np.int32)
BANKS616=(96,384,1536)
BUDGETS616=(4,8,16,24,32,48,64,96,128,256)
RESULT616={"test":616,"source_sha256":SHA610,"baseline":{"fram_hold_r1":H1,"fram_hold_r5":H5,"cut3_correct":12},"banks":list(BANKS616),"new_qwen_encoded_records":len(NEWK616),"results":[],"full_native":[],"capacity":[],"limitations":["Previously seen 24-question panel; no fresh independent holdout","New records are genuine frozen-Qwen residual captures, not materialized CUT3 files","CSR timing excludes native FRAM reranking and CUT3 generation"]}
def build_csr616(v,k):
    n,d=v.shape
    bins=[[] for _ in range(2*d)]
    for ci in range(n):
        for posting in top_signed616(v[ci],k):
            bins[int(posting)].append(ci)
    indptr=np.zeros(2*d+1,dtype=np.int32)
    for i,p in enumerate(bins):indptr[i+1]=indptr[i]+len(p)
    indices=np.empty(int(indptr[-1]),dtype=np.int32)
    for i,p in enumerate(bins):
        indices[indptr[i]:indptr[i+1]]=p
    return indptr,indices
def route616(q,indptr,indices,n,weighted):
    postings=top_signed616(q,KSPARSE616)
    counts=np.zeros(n,dtype=np.float32)
    visits=0
    for p in postings:
        p=int(p);lo=int(indptr[p]);hi=int(indptr[p+1]);ids=indices[lo:hi]
        visits+=len(ids)
        if len(ids):
            weight=math.log1p(n/(1+len(ids))) if weighted else 1.0
            np.add.at(counts,ids,weight)
    raw=np.flatnonzero(counts>0)
    # Deterministic score descending, cartridge ID ascending.
    order=raw[np.lexsort((raw,-counts[raw]))]
    return order,int(visits),int(len(raw))
def native616(wi,candidates):
    q=Q616[wi]
    scores=[pair_scores(q,ALLK616[int(ci)],8.0) for ci in candidates]
    return sorted(range(len(candidates)),key=lambda i:(-scores[i],int(candidates[i])))
def run_bank616(n):
    print("TEST616_BANK_BEGIN",n,flush=True)
    v=V616[:n]
    indptr,indices=build_csr616(v,KSPARSE616)
    csr_bytes=int(indptr.nbytes+indices.nbytes)
    routes={}
    for weighted in (False,True):
        per=[];times=[]
        for wi in range(24):
            t0=time.perf_counter()
            order,visits,raw_count=route616(U616[wi],indptr,indices,n,weighted)
            times.append((time.perf_counter()-t0)*1000)
            per.append((order,visits,raw_count))
        for budget in BUDGETS616:
            hits=[];rank1=[];rank5=[];reranked=[]
            for wi,(order,visits,raw_count) in enumerate(per):
                selected=order[:budget]
                target=TARGET616[wi]
                present=bool(target in selected)
                hits.append(present)
                reranked.append(len(selected))
                if not len(selected):
                    rank1.append(False);rank5.append(False);continue
                ix=native616(wi,selected)
                ordered=[int(selected[j]) for j in ix]
                rank1.append(bool(ordered[0]==target))
                rank5.append(bool(target in ordered[:5]))
            result={"bank_size":n,"mode":"SIGNED","k":KSPARSE616,"weighted":weighted,"budget":budget,
                "cal_r1":int(sum(rank1[:12])),"previously_seen_hold_r1":int(sum(rank1[12:])),
                "all_r1":int(sum(rank1)),"all_r5":int(sum(rank5)),
                "candidate_hit":int(sum(hits)),
                "raw_coverage":int(sum(TARGET616[wi] in per[wi][0] for wi in range(24))),
                "mean_raw_candidates":round(float(np.mean([x[2] for x in per])),3),
                "mean_posting_visits":round(float(np.mean([x[1] for x in per])),3),
                "mean_reranked":round(float(np.mean(reranked)),3),
                "route_p50_ms":round(float(np.percentile(times,50)),4),
                "route_p95_ms":round(float(np.percentile(times,95)),4),
                "csr_bytes":csr_bytes,"csr_mib":round(csr_bytes/1048576,4)}
            RESULT616["results"].append(result)
            print("TEST616_RESULT",json.dumps(result,ensure_ascii=False),flush=True)
    print("TEST616_BANK_END",n,flush=True)
    # Exhaustive native FRAM: diagnostic reference, no routing.
    t0=time.perf_counter();ranks=[]
    for wi in range(24):
        ix=native616(wi,np.arange(n,dtype=np.int32))
        ordered=[ix[j] for j in range(len(ix))]
        ranks.append(ordered.index(TARGET616[wi])+1)
    full={"bank_size":n,"r1":sum(r==1 for r in ranks),"r5":sum(r<=5 for r in ranks),
        "ranks":ranks,"seconds":round(time.perf_counter()-t0,3),
        "scope":"Full native FRAM diagnostic; not routed and not new holdout"}
    RESULT616["full_native"].append(full)
    print("TEST616_FULL_NATIVE",json.dumps(full),flush=True)
    return csr_bytes
for bank616 in BANKS616:run_bank616(bank616)
# Count actual source-token spans. CUT3 bytes are theoretical if all records
# were materialized; the new records currently have only Qwen residual captures.
for n in BANKS616:
    lengths=[int(x.shape[0]) for x in KS[3]]+NEW_TOKENS616[:n-96]
    assert len(lengths)==n
    total=sum(lengths)
    theoretical=total*15360/1048576
    cap={"cartridges":n,"source_tokens_total":total,
        "mean_tokens_per_cartridge":round(total/n,3),
        "min_tokens":min(lengths),"max_tokens":max(lengths),
        "cut3_payload_mib_if_materialized":round(theoretical,4),
        "new_cartridges_have_cut3_files":False,"words":"NOT_MEASURED"}
    RESULT616["capacity"].append(cap)
    print("TEST616_CAPACITY",json.dumps(cap),flush=True)
best=[]
for n in BANKS616:
    subset=[x for x in RESULT616["results"] if x["bank_size"]==n]
    good=[x for x in subset if x["all_r1"]==24]
    chosen=min(good,key=lambda x:(x["budget"],x["weighted"],x["route_p95_ms"])) if good else max(subset,key=lambda x:(x["all_r1"],-x["budget"]))
    best.append(chosen)
RESULT616["best_per_bank"]=best
RESULT616["integrity"]={"weight_unchanged":weight_sha()==WB616,"sentinel_unchanged":sentinel()==SB616,
    "zero_trainable":not any(p.requires_grad for p in model.parameters()),
    "no_remaining_hooks":not any(layer._forward_hooks or layer._forward_pre_hooks for layer in model.model.layers)}
RESULT616["seconds"]=round(time.perf_counter()-T616,2)
RESULT616["output"]=OUT616
with open(OUT616,"w",encoding="utf-8") as f:json.dump(RESULT616,f,ensure_ascii=False,indent=2)
print("TEST616_FINAL",json.dumps({"test":616,"baseline":RESULT616["baseline"],
    "banks":RESULT616["banks"],"new_qwen_encoded_records":1440,
    "full_native":RESULT616["full_native"],"best_per_bank":best,
    "capacity":RESULT616["capacity"],"integrity":RESULT616["integrity"],
    "output":OUT616},ensure_ascii=False),flush=True)
if not all(RESULT616["integrity"].values()):raise RuntimeError("TEST616 FINAL INTEGRITY FAILURE")
print("TEST616_STATUS COMPLETE",flush=True)

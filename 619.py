
# TEST619-NATIVE — AKBASCORE MAM / NATIVE ACTIVATION GEOMETRY
# Frozen Qwen + unchanged CUT3 + unchanged FRAM.
# Diagnostic experiment: full-bank geometric scan; NOT a scalable router.
# DIAG: 48 TEST617 questions. VALID: 48 disjoint TEST618 questions.
import os,time,json,random,hashlib,urllib.request
import numpy as np,torch
import torch.nn.functional as F
from collections import defaultdict
T0_619=time.perf_counter()
ROOT619="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/"
LOCK619="a62611a4ffd91c2fcf03796c31da2af11ccd8141345655b562f3a614568af3e7"
OUT619="/content/TEST619_NATIVE_RESULT.json"
BUDGETS619=(16,32,64,128)
METHODS619=("MEAN","MAX_PAIR","MAX_AVG","TOP2_SOFT","TOP4_SOFT","CONTRAST","NATIVE_REFERENCE")
def report619(*x):print(*x,flush=True)
blob619=urllib.request.urlopen(urllib.request.Request(ROOT619+"610.py",headers={"User-Agent":"AKBASCORE-619-NATIVE"}),timeout=120).read()
sha619=hashlib.sha256(blob619).hexdigest()
assert sha619==LOCK619,("SOURCE_SHA_FAILURE",sha619)
boundary619="print('[6/7] Demo acceptance"
assert blob619.decode().count(boundary619)==1
report619("TEST619_SOURCE_LOCK",sha619)
exec(compile(blob619.decode().split(boundary619,1)[0],"610.py","exec"),globals())
assert CFG==(3,0,8.0) and H1==12 and H5==12 and len(CAR)==96
assert not any(p.requires_grad for p in model.parameters())
WEIGHT619=weight_sha()
SENTINEL619=sentinel()
report619("TEST619_BASELINE",{"cartridges":96,"fram_r1":H1,"fram_r5":H5,"cut3_baseline":12})
# Preserve the original locked frame. Build exactly the TEST617/618 synthetic bank.
FIT619=transform_fit(torch.cat(KS[3],dim=0),0)
K619=[apply(x,FIT619) for x in KS[3]]
assert len(K619)==96
if "KBASE" in globals():
    assert all(torch.equal(a,b) for a,b in zip(K619,KBASE))
rng619=random.Random(616616)
ALPH619="ABCDEFGHJKLMNPQRSTUVWXYZ"
def word619(n):return "".join(rng619.choice(ALPH619) for _ in range(n)).capitalize()
FACT619=[]
META619=[]
for i in range(1440):
    city=word619(9)
    country="ZX"+str(100000+i)
    fact=f"The current capital of {country} is {city}."
    FACT619.append(fact)
    META619.append({"id":96+i,"country":country,"city":city})
    rep,info=capture_residual(source(fact),len(PREFIX),len(PREFIX)+len(fact))
    K619.append(apply(rep[3],FIT619))
    if (i+1)%240==0:
        report619("CAPTURE",i+1,"/1440","elapsed",round(time.perf_counter()-T0_619,2))
assert len(K619)==1536
# Diagnostic and untouched validation panels. No answer labels enter routing.
DIAG619=sorted(random.Random(6172026).sample(range(1440),48))
REST619=sorted(set(range(1440))-set(DIAG619))
VALID619=sorted(random.Random(6182026).sample(REST619,48))
assert len(set(DIAG619)&set(VALID619))==0
PANEL619=[]
for split,indices in (("DIAG",DIAG619),("VALID",VALID619)):
    for i in indices:
        m=META619[i]
        PANEL619.append({"split":split,"target":m["id"],
            "question":f"What is the current capital of {m['country']}?",
            "country":m["country"],"gold":m["city"]})
Q619=[]
for i,row in enumerate(PANEL619):
    rep,info=question_rep(row["question"])
    Q619.append(apply(rep[3],FIT619))
    if (i+1)%24==0:report619("QUESTION_CAPTURE",i+1,"/96")
# Native token geometry. No text labels, gold IDs or fact strings are used below.
# Representations remain in the locked FRAM PCA frame.
D619=int(K619[0].shape[-1])
LENS619=np.array([len(x) for x in K619],dtype=np.int32)
OFF619=np.zeros(len(K619)+1,dtype=np.int32)
OFF619[1:]=np.cumsum(LENS619)
TOK619=torch.cat([x.detach().float().cpu() for x in K619],dim=0)
assert int(OFF619[-1])==len(TOK619)
# All geometry arms compare the SAME native token vectors.
# The full token bank is uploaded to GPU for the diagnostic experiment only.
device619=torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKGPU619=TOK619.to(device619)
MEANGPU619=torch.stack([x.float().mean(0) for x in K619]).to(device619)
MEANGPU619=F.normalize(MEANGPU619,dim=-1,eps=1e-9)
report619("GEOMETRY_BANK",json.dumps({"cartridges":len(K619),"tokens":len(TOK619),
    "dimension":D619,"native_token_bytes":TOK619.numel()*TOK619.element_size(),
    "device":str(device619)}))
# Determine whether the synthetic opening tokens collapse into shared geometry.
# Pairwise distances are measured directly; no tokenizer labels needed.
COMMON619={}
for position in range(5):
    rows=[x[position].float() for x in K619[96:] if len(x)>position]
    a=torch.stack(rows)
    ref=a[0]
    delta=torch.linalg.vector_norm(a-ref,dim=1)
    COMMON619[str(position)]={"n":len(rows),"max_l2":float(delta.max()),
        "median_l2":float(delta.median()),"fraction_near_1e-5":float((delta<1e-5).float().mean())}
report619("SHARED_PREFIX_GEOMETRY",json.dumps(COMMON619))
# Candidate ranking arms:
# MEAN: average-vector cosine, historical destructive compression.
# MAX_PAIR: one best token pair, disregards other question tokens.
# MAX_AVG: best cartridge token for EACH question token, then average.
# TOP2/TOP4: retain multiple matching tokens per question token and use
# beta=8 logmeanexp with the ORIGINAL cartridge length denominator.
# CONTRAST: aggregate only above the query-token median across cartridges.
# NATIVE_REFERENCE: original locked pair_scores, no modifications.
def geometric_scores619(q):
    qg=q.detach().float().to(device619)
    qmean=F.normalize(qg.mean(0),dim=0,eps=1e-9)
    mean_scores=(MEANGPU619@qmean).cpu().numpy().astype(np.float64)
    # Dot products preserve the existing frame and token normalization.
    # Chunking bounds peak GPU memory; no changes to representations.
    sims=[]
    for start in range(0,len(TOKGPU619),8192):
        sims.append((qg@TOKGPU619[start:start+8192].T).float().cpu())
    sim=torch.cat(sims,dim=1).numpy()
    T=len(q)
    out={name:np.empty(len(K619),dtype=np.float64) for name in METHODS619 if name!="NATIVE_REFERENCE"}
    out["MEAN"][:]=mean_scores
    per_q_max=np.empty((T,len(K619)),dtype=np.float64)
    per_q_top2=np.empty((T,len(K619)),dtype=np.float64)
    per_q_top4=np.empty((T,len(K619)),dtype=np.float64)
    for ci in range(len(K619)):
        block=sim[:,OFF619[ci]:OFF619[ci+1]]
        n=block.shape[1]
        top=np.sort(block,axis=1)[:,::-1]
        per_q_max[:,ci]=top[:,0]
        for count,arr in ((2,per_q_top2),(4,per_q_top4)):
            z=top[:,:min(count,n)].astype(np.float64)*8.0
            m=z.max(axis=1)
            # This is a truncated numerator, normalized by ORIGINAL n.
            arr[:,ci]=(m+np.log(np.exp(z-m[:,None]).sum(axis=1))-np.log(n))/8.0
    out["MAX_PAIR"][:]=per_q_max.max(axis=0)
    out["MAX_AVG"][:]=per_q_max.mean(axis=0)
    out["TOP2_SOFT"][:]=per_q_top2.mean(axis=0)
    out["TOP4_SOFT"][:]=per_q_top4.mean(axis=0)
    # Query-token-wise contrast is derived from the unlabeled entire bank.
    med=np.median(per_q_max,axis=1,keepdims=True)
    out["CONTRAST"][:]=np.maximum(per_q_max-med,0).mean(axis=0)
    return out
def rank619(scores):
    ids=np.arange(len(scores))
    return np.lexsort((ids,-np.asarray(scores,dtype=np.float64)))
def scalar619(v):
    return float(v.detach().cpu().item()) if isinstance(v,torch.Tensor) else float(v)
def native619(q):
    # Locked native scorer is the reference, not an approximated replacement.
    return np.asarray([scalar619(pair_scores(q,k,8.0)) for k in K619],dtype=np.float64)
RECORDS619=[]
NATIVE619={}
GEO619={}
TIMES619={}
for qi,row in enumerate(PANEL619):
    t=time.perf_counter()
    geom=geometric_scores619(Q619[qi])
    geo_ms=(time.perf_counter()-t)*1000
    t=time.perf_counter()
    ns=native619(Q619[qi])
    native_ms=(time.perf_counter()-t)*1000
    geom["NATIVE_REFERENCE"]=ns
    GEO619[qi]={name:rank619(scores) for name,scores in geom.items()}
    NATIVE619[qi]=ns
    TIMES619[qi]={"geometry_ms":geo_ms,"native_ms":native_ms}
    nr=GEO619[qi]["NATIVE_REFERENCE"]
    target=row["target"]
    gold_rank=int(np.flatnonzero(nr==target)[0])+1
    # Inspect where information is lost as geometry is simplified.
    case={"split":row["split"],"query":qi,"target":target,
        "native_gold_rank":gold_rank,"geometry_ms":round(geo_ms,3),
        "native_ms":round(native_ms,3)}
    for method in METHODS619:
        order=GEO619[qi][method]
        case[method+"_gold_rank"]=int(np.flatnonzero(order==target)[0])+1
        case[method+"_native_top5_at32"]=int(set(nr[:5]).issubset(set(order[:32])))
    RECORDS619.append(case)
    if (qi+1)%8==0:
        report619("GEOMETRY_PROGRESS",qi+1,"/96",
            "split",row["split"],"elapsed_s",round(time.perf_counter()-T0_619,2))
# Evaluate routing geometry separately from native reranking.
# All scoring methods use frozen Qwen activations. No gold labels in ranking.
RESULTS619=[]
for split in ("DIAG","VALID"):
    indices=[i for i,row in enumerate(PANEL619) if row["split"]==split]
    for method in METHODS619:
        for budget in BUDGETS619:
            gold=top1=top5=rr1=rr5=0
            for qi in indices:
                target=PANEL619[qi]["target"]
                native_order=GEO619[qi]["NATIVE_REFERENCE"]
                chosen=GEO619[qi][method][:budget]
                gold+=int(target in chosen)
                top1+=int(native_order[0] in chosen)
                top5+=int(set(native_order[:5]).issubset(set(chosen)))
                # Native rerank uses cached exact reference scores, not an
                # altered formula; avoids recomputing identical pair_scores.
                reranked=chosen[np.lexsort((chosen,-NATIVE619[qi][chosen]))]
                rr1+=int(len(reranked)>0 and reranked[0]==target)
                rr5+=int(target in reranked[:5])
            item={"split":split,"method":method,"budget":budget,
                "n":len(indices),"gold_in_candidates":gold,
                "native_top1_contained":top1,"native_top5_contained":top5,
                "reranked_gold_r1":rr1,"reranked_gold_r5":rr5}
            RESULTS619.append(item)
            report619("TEST619_RESULT",json.dumps(item))
# Rank correlations and exact per-question budgets needed for top-5 containment.
DIAGNOSTICS619=[]
for qi,row in enumerate(PANEL619):
    native=GEO619[qi]["NATIVE_REFERENCE"]
    native5=set(native[:5])
    entry={"split":row["split"],"query":qi,"target":row["target"]}
    for method in METHODS619:
        order=GEO619[qi][method]
        pos=np.empty(len(order),dtype=np.int32)
        pos[order]=np.arange(1,len(order)+1)
        entry[method+"_budget_for_native_top5"]=int(max(pos[x] for x in native5))
        entry[method+"_budget_for_gold"]=int(pos[row["target"]])
        entry[method+"_top32_overlap"]=len(native5.intersection(order[:32]))
    DIAGNOSTICS619.append(entry)
# A diagnostic scan does not count as a scalable RAM/disk router.
INTEGRITY619={"source_sha_lock":sha619==LOCK619,
    "weight_unchanged":weight_sha()==WEIGHT619,
    "sentinel_unchanged":sentinel()==SENTINEL619,
    "zero_trainable":not any(p.requires_grad for p in model.parameters()),
    "no_remaining_hooks":not any(layer._forward_hooks or layer._forward_pre_hooks for layer in model.model.layers)}
summary619={}
for split in ("DIAG","VALID"):
    subset=[x for x in RESULTS619 if x["split"]==split and x["budget"]==32]
    summary619[split]={x["method"]:{"gold":x["gold_in_candidates"],
        "native_top5":x["native_top5_contained"],
        "reranked_r1":x["reranked_gold_r1"],"reranked_r5":x["reranked_gold_r5"]}
        for x in subset}
    indices=[i for i,x in enumerate(PANEL619) if x["split"]==split]
    summary619[split]["native_full"]={"r1":sum(RECORDS619[i]["native_gold_rank"]==1 for i in indices),
        "r5":sum(RECORDS619[i]["native_gold_rank"]<=5 for i in indices)}
RESULT619={"test":"619-NATIVE","status":"COMPLETE",
    "purpose":"Natural activation geometry diagnostic, not scalable indexing",
    "source_sha256":sha619,"cartridges":len(K619),
    "bank_tokens":int(len(TOK619)),"dimension":D619,
    "panels":{"DIAG":DIAG619,"VALID":VALID619},
    "shared_prefix":COMMON619,"summary_at_budget32":summary619,
    "all_metrics":RESULTS619,"cases":RECORDS619,
    "budget_diagnostics":DIAGNOSTICS619,
    "timing":TIMES619,"integrity":INTEGRITY619,
    "elapsed_s":round(time.perf_counter()-T0_619,2),
    "limitations":[
        "All geometry methods scan the full native token bank.",
        "This experiment diagnoses representation loss; it is not a disk-scale router.",
        "DIAG and VALID share the same synthetic fact template.",
        "VALID is disjoint from DIAG but was previously examined in TEST618.",
        "No new CUT3 answer-generation test is performed.",
        "Native reference uses unchanged locked pair_scores."
    ]}
os.makedirs(os.path.dirname(OUT619),exist_ok=True)
with open(OUT619,"w",encoding="utf-8") as f:
    json.dump(RESULT619,f,ensure_ascii=False,indent=2)
report619("TEST619_FINAL",json.dumps({"summary_at_budget32":summary619,
    "shared_prefix":COMMON619,"integrity":INTEGRITY619,
    "elapsed_s":RESULT619["elapsed_s"],"json":OUT619},ensure_ascii=False))
if not all(INTEGRITY619.values()):
    raise RuntimeError("TEST619_NATIVE_INTEGRITY_FAILURE")
report619("TEST619_NATIVE_STATUS COMPLETE")

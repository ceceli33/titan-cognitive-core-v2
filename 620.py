
# TEST620 — AKBASCORE MAM · NATIVE ACTIVATION IVF INDEX
# TEST619 follow-up. Frozen CUT3 / FRAM / Qwen. Single Google Colab cell.
# Native L3 tokens -> label-free PCA -> FAISS IVF -> cartridge union
# -> exact TOP2_SOFT on touched cartridges -> locked FRAM rerank.
# No text index, no labels in routing, no training of Qwen.
# IMPORTANT: Index is experimental. Results are not preclaimed.
import os,sys,time,json,random,hashlib,urllib.request,subprocess,importlib.util
from collections import defaultdict
import numpy as np,torch
import torch.nn.functional as F
T620=time.perf_counter()
TEST620="620"
SOURCE620="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/610.py"
LOCK620="a62611a4ffd91c2fcf03796c31da2af11ccd8141345655b562f3a614568af3e7"
OUTPUT620="/content/TEST620_RESULT.json"
SEED620=6202026
RANK620=128
NLIST620=128
NPROBES620=(2,4,8,16)
TOKEN_K620=(16,32,64,128)
BUDGETS620=(16,32,64)
BETA620=8.0
def log620(*args):print(*args,flush=True)
def sha620(b):return hashlib.sha256(b).hexdigest()
def ms620(t):return round((time.perf_counter()-t)*1000,4)
random.seed(SEED620);np.random.seed(SEED620);torch.manual_seed(SEED620)
if importlib.util.find_spec("faiss") is None:
    subprocess.check_call([sys.executable,"-m","pip","-q","install","faiss-cpu"])
import faiss
faiss.omp_set_num_threads(4)
src620=urllib.request.urlopen(urllib.request.Request(SOURCE620,headers={"User-Agent":"AKBASCORE-TEST620"}),timeout=120).read()
assert sha620(src620)==LOCK620,("SOURCE_LOCK_FAILURE",sha620(src620))
split620="print('[6/7] Demo acceptance"
assert src620.decode().count(split620)==1
log620("TEST620_SOURCE_LOCK",sha620(src620))
exec(compile(src620.decode().split(split620,1)[0],"610.py","exec"),globals())
assert tuple(CFG)==(3,0,8.0) and H1==12 and H5==12 and len(CAR)==96
assert not any(p.requires_grad for p in model.parameters())
WEIGHT620=weight_sha();SENTINEL620=sentinel()
log620("TEST620_BASELINE",{"cartridges":96,"fram_hold_r1":H1,"fram_hold_r5":H5,"cut3_reference":12})
# Original locked frame and original 96 cartridges.
FIT620=transform_fit(torch.cat(KS[3],dim=0),0)
K620=[apply(x,FIT620).detach().float().cpu().contiguous() for x in KS[3]]
assert len(K620)==96
if "KBASE" in globals():
    assert all(torch.equal(a,b.float().cpu()) for a,b in zip(K620,KBASE))
# Rebuild the TEST619 synthetic bank, without modifying the engine.
rng620=random.Random(616616)
ALPH620="ABCDEFGHJKLMNPQRSTUVWXYZ"
def word620(n):return "".join(rng620.choice(ALPH620) for _ in range(n)).capitalize()
META620=[]
for i in range(1440):
    city=word620(9);country="ZX"+str(100000+i)
    fact=f"The current capital of {country} is {city}."
    rep,info=capture_residual(source(fact),len(PREFIX),len(PREFIX)+len(fact))
    K620.append(apply(rep[3],FIT620).detach().float().cpu().contiguous())
    META620.append({"id":96+i,"country":country,"city":city})
    if (i+1)%240==0:log620("CAPTURE",i+1,"/1440",round(time.perf_counter()-T620,2),"s")
assert len(K620)==1536
DIAG620=sorted(random.Random(6172026).sample(range(1440),48))
REST620=sorted(set(range(1440))-set(DIAG620))
VALID620=sorted(random.Random(6182026).sample(REST620,48))
assert not set(DIAG620).intersection(VALID620)
PANEL620=[]
for split,ids in (("DIAG",DIAG620),("VALID",VALID620)):
    for i in ids:
        m=META620[i]
        PANEL620.append({"split":split,"target":m["id"],
            "question":f"What is the current capital of {m['country']}?"})
Q620=[]
for j,row in enumerate(PANEL620):
    rep,info=question_rep(row["question"])
    Q620.append(apply(rep[3],FIT620).detach().float().cpu().contiguous())
    if (j+1)%24==0:log620("QUESTION_CAPTURE",j+1,"/96")
# Native token bank: exact vectors are used ONLY for candidate reranking.
LENS620=np.asarray([len(k) for k in K620],dtype=np.int32)
OFF620=np.zeros(len(K620)+1,dtype=np.int64)
OFF620[1:]=np.cumsum(LENS620)
OWNER620=np.repeat(np.arange(len(K620),dtype=np.int32),LENS620)
TOK620=torch.cat(K620,dim=0).contiguous()
NTOK620=len(TOK620);DIM620=TOK620.shape[1]
assert NTOK620==27492 and DIM620==3584
log620("NATIVE_BANK",{"cartridges":len(K620),"tokens":NTOK620,
    "dimension":DIM620,"native_token_bytes":TOK620.numel()*TOK620.element_size()})
# LABEL-FREE PCA: no questions, answers, IDs or target labels enter the fit.
# CPU torch PCA low-rank; no change to the native vectors.
t=time.perf_counter()
SAMPLE620=min(4096,NTOK620)
trainids620=np.random.default_rng(SEED620).choice(NTOK620,SAMPLE620,replace=False)
TRAIN620=TOK620[torch.from_numpy(trainids620)].contiguous()
CENTER620=TRAIN620.mean(0)
XC620=TRAIN620-CENTER620
# Fixed rank and oversampling; PCA is fitted only on cartridge activations.
U620,S620,V620=torch.pca_lowrank(XC620,q=RANK620+16,center=False,niter=2)
BASIS620=V620[:,:RANK620].contiguous()
assert BASIS620.shape==(DIM620,RANK620)
assert torch.allclose(BASIS620.T@BASIS620,torch.eye(RANK620),atol=2e-3)
PCA_TIME620=ms620(t)
del U620,S620,V620,XC620,TRAIN620
# Index projection, in chunks to bound working memory.
t=time.perf_counter()
PROJECTED620=np.empty((NTOK620,RANK620),dtype=np.float32)
for start in range(0,NTOK620,2048):
    end=min(start+2048,NTOK620)
    PROJECTED620[start:end]=((TOK620[start:end]-CENTER620)@BASIS620).numpy()
PROJECT_TIME620=ms620(t)
# Inner product is consistent with the projected component of native dot products.
# We intentionally do not normalize projected vectors.
t=time.perf_counter()
quantizer620=faiss.IndexFlatIP(RANK620)
INDEX620=faiss.IndexIVFFlat(quantizer620,RANK620,NLIST620,faiss.METRIC_INNER_PRODUCT)
INDEX620.train(np.ascontiguousarray(PROJECTED620[trainids620]))
INDEX620.add(np.ascontiguousarray(PROJECTED620))
assert INDEX620.ntotal==NTOK620
INDEX_BUILD620=ms620(t)
# Read actual inverted-list sizes. This counts examined index entries.
LISTSIZE620=np.asarray([INDEX620.invlists.list_size(i) for i in range(NLIST620)],dtype=np.int64)
assert LISTSIZE620.sum()==NTOK620
INDEX_BYTES620=len(faiss.serialize_index(INDEX620))
PROJECTION_BYTES620=PROJECTED620.nbytes
BASIS_BYTES620=BASIS620.numel()*BASIS620.element_size()
CENTER_BYTES620=CENTER620.numel()*CENTER620.element_size()
OWNER_BYTES620=OWNER620.nbytes
STATIC_BYTES620=INDEX_BYTES620+BASIS_BYTES620+CENTER_BYTES620+OWNER_BYTES620
log620("INDEX_BUILD",json.dumps({"pca_ms":PCA_TIME620,
    "projection_ms":PROJECT_TIME620,"ivf_build_ms":INDEX_BUILD620,
    "faiss_index_bytes":INDEX_BYTES620,"basis_bytes":BASIS_BYTES620,
    "center_bytes":CENTER_BYTES620,"owner_bytes":OWNER_BYTES620,
    "index_plus_basis_owner_bytes":STATIC_BYTES620,
    "projected_build_array_bytes":PROJECTION_BYTES620,
    "nonempty_lists":int(np.count_nonzero(LISTSIZE620)),
    "max_list_size":int(LISTSIZE620.max())}))
# Query routing uses FAISS IVF only, never the gold cartridge ID.
def projected_query620(q):
    return np.ascontiguousarray(((q-CENTER620)@BASIS620).numpy(),dtype=np.float32)
def probe620(qproj,nprobe,k):
    INDEX620.nprobe=nprobe
    t=time.perf_counter()
    # Quantizer lookup provides actual IVF list IDs for scan accounting.
    _,cells=INDEX620.quantizer.search(qproj,nprobe)
    visited=sum(int(LISTSIZE620[c]) for row in cells for c in row if c>=0)
    distances,tokenids=INDEX620.search(qproj,k)
    route_ms=ms620(t)
    # Each question token contributes its own nearest native-token hits.
    # Aggregate per cartridge, retaining the best score per question token.
    T=len(qproj)
    votes={}
    for ti in range(T):
        for sim,tid in zip(distances[ti],tokenids[ti]):
            if tid<0:continue
            owner=int(OWNER620[tid])
            if owner not in votes:votes[owner]=np.full(T,-np.inf,dtype=np.float32)
            if sim>votes[owner][ti]:votes[owner][ti]=sim
    # This ordering is a routing heuristic only; no benchmark labels.
    # Reward coverage of distinct query tokens, then average matched score.
    ranking=[]
    for owner,arr in votes.items():
        mask=np.isfinite(arr)
        coverage=int(mask.sum())
        avg=float(arr[mask].mean()) if coverage else -1e9
        ranking.append((owner,coverage,avg))
    ranking.sort(key=lambda x:(-x[1],-x[2],x[0]))
    touched=[x[0] for x in ranking]
    return touched,{"route_ms":route_ms,"posting_visits":visited,
        "unique_touched":len(touched),"retrieved_token_hits":int(np.count_nonzero(tokenids>=0))}
# TOP2_SOFT, exactly as defined in TEST619:
# each question token contributes its strongest 2 cartridge-token matches;
# denominator remains the ORIGINAL cartridge token length.
def top2_score620(q,k):
    sim=q@k.T
    top=torch.topk(sim,k=min(2,k.shape[0]),dim=1).values.double()
    return float(((torch.logsumexp(top*BETA620,dim=1)
        -np.log(len(k)))/BETA620).mean().item())
def native_score620(q,k):
    v=pair_scores(q,k,BETA620)
    return float(v.detach().cpu().item()) if isinstance(v,torch.Tensor) else float(v)
def order620(scores):
    ids=np.arange(len(scores))
    return np.lexsort((ids,-np.asarray(scores,dtype=np.float64)))
# Full native reference is computed only for scientific evaluation.
# It is NOT accessible to the index or candidate-selection path.
FULL620=[]
TOP2FULL620=[]
for qi,q in enumerate(Q620):
    t=time.perf_counter()
    scores=np.asarray([native_score620(q,k) for k in K620],dtype=np.float64)
    FULL620.append(order620(scores))
    # Full TOP2 reference is a diagnostic arm, not the indexed path.
    ts=np.asarray([top2_score620(q,k) for k in K620],dtype=np.float64)
    TOP2FULL620.append(order620(ts))
    if (qi+1)%12==0:log620("REFERENCE",qi+1,"/96","elapsed_s",round(time.perf_counter()-T620,2))
# Run all predeclared settings. No selection on VALID is made in this cell.
RESULTS620=[]
DETAIL620=[]
for nprobe in NPROBES620:
    for token_k in TOKEN_K620:
        for qi,(q,row) in enumerate(zip(Q620,PANEL620)):
            qproj=projected_query620(q)
            touched,stats=probe620(qproj,nprobe,token_k)
            # Only the touched cartridges undergo TOP2 geometry.
            t=time.perf_counter()
            scores={ci:top2_score620(q,K620[ci]) for ci in touched}
            ranked=sorted(touched,key=lambda ci:(-scores[ci],ci))
            geometry_ms=ms620(t)
            native_order=FULL620[qi]
            native5=set(map(int,native_order[:5]))
            gold=row["target"]
            for budget in BUDGETS620:
                candidates=ranked[:budget]
                t=time.perf_counter()
                ns={ci:native_score620(q,K620[ci]) for ci in candidates}
                final=sorted(candidates,key=lambda ci:(-ns[ci],ci))
                rerank_ms=ms620(t)
                DETAIL620.append({"split":row["split"],"query":qi,
                    "nprobe":nprobe,"token_k":token_k,"budget":budget,
                    "gold_in_touched":int(gold in touched),
                    "gold_in_candidates":int(gold in candidates),
                    "native_top1_contained":int(int(native_order[0]) in candidates),
                    "native_top5_contained":int(native5.issubset(candidates)),
                    "gold_r1":int(bool(final) and final[0]==gold),
                    "gold_r5":int(gold in final[:5]),
                    "touched":len(touched),
                    "scanned_native_tokens":int(sum(LENS620[ci] for ci in touched)),
                    "posting_visits":stats["posting_visits"],
                    "route_ms":stats["route_ms"],
                    "geometry_ms":geometry_ms,
                    "native_rerank_ms":rerank_ms,
                    "total_ms":stats["route_ms"]+geometry_ms+rerank_ms})
        for split in ("DIAG","VALID"):
            for budget in BUDGETS620:
                subset=[x for x in DETAIL620 if x["split"]==split
                    and x["nprobe"]==nprobe and x["token_k"]==token_k
                    and x["budget"]==budget]
                def total(field):return int(sum(x[field] for x in subset))
                def avg(field):return round(float(np.mean([x[field] for x in subset])),4)
                def pct(field,p):return round(float(np.percentile([x[field] for x in subset],p)),4)
                result={"split":split,"nprobe":nprobe,"token_k":token_k,
                    "budget":budget,"questions":len(subset),
                    "gold_touched":total("gold_in_touched"),
                    "gold_candidates":total("gold_in_candidates"),
                    "native_top1_contained":total("native_top1_contained"),
                    "native_top5_contained":total("native_top5_contained"),
                    "reranked_r1":total("gold_r1"),"reranked_r5":total("gold_r5"),
                    "mean_touched":avg("touched"),
                    "mean_touched_fraction":round(avg("touched")/len(K620),6),
                    "mean_native_tokens_scanned":avg("scanned_native_tokens"),
                    "mean_native_token_fraction":round(avg("scanned_native_tokens")/NTOK620,6),
                    "mean_ivf_posting_visits":avg("posting_visits"),
                    "route_p50_ms":pct("route_ms",50),
                    "geometry_p50_ms":pct("geometry_ms",50),
                    "rerank_p50_ms":pct("native_rerank_ms",50),
                    "total_p50_ms":pct("total_ms",50),
                    "total_p95_ms":pct("total_ms",95)}
                RESULTS620.append(result)
                log620("TEST620_RESULT",json.dumps(result))
# Full-reference checks; not used for routing or parameter selection.
BASE620={}
for split in ("DIAG","VALID"):
    ids=[i for i,x in enumerate(PANEL620) if x["split"]==split]
    BASE620[split]={
        "native_r1":sum(int(FULL620[i][0]==PANEL620[i]["target"]) for i in ids),
        "native_r5":sum(int(PANEL620[i]["target"] in FULL620[i][:5]) for i in ids),
        "full_top2_gold_at16":sum(int(PANEL620[i]["target"] in TOP2FULL620[i][:16]) for i in ids),
        "full_top2_native_top5_at16":sum(int(set(FULL620[i][:5]).issubset(TOP2FULL620[i][:16])) for i in ids)}
# A fixed primary setting is reported regardless of performance.
PRIMARY620={"nprobe":8,"token_k":64,"budget":32}
PRIMARY_ROWS620=[r for r in RESULTS620 if all(r[k]==v for k,v in PRIMARY620.items())]
INTEGRITY620={
    "source_sha_lock":sha620(src620)==LOCK620,
    "weight_unchanged":weight_sha()==WEIGHT620,
    "sentinel_unchanged":sentinel()==SENTINEL620,
    "zero_trainable":not any(p.requires_grad for p in model.parameters()),
    "no_remaining_hooks":not any(layer._forward_hooks or layer._forward_pre_hooks
        for layer in model.model.layers)}
FINAL620={
    "test":"620","status":"COMPLETE",
    "architecture":"Native L3 PCA128 + FAISS IVFFlat + touched TOP2_SOFT + locked FRAM",
    "source_sha256":sha620(src620),
    "model":"Qwen/Qwen2.5-7B-Instruct",
    "locked_fram_cfg":[3,0,8.0],
    "bank":{"cartridges":len(K620),"tokens":NTOK620,"dimension":DIM620},
    "index":{"rank":RANK620,"nlist":NLIST620,
        "pca_ms":PCA_TIME620,"projection_ms":PROJECT_TIME620,
        "build_ms":INDEX_BUILD620,"faiss_bytes":INDEX_BYTES620,
        "basis_bytes":BASIS_BYTES620,"center_bytes":CENTER_BYTES620,
        "owner_bytes":OWNER_BYTES620,
        "static_index_plus_basis_owner_bytes":STATIC_BYTES620,
        "projected_build_array_bytes":PROJECTION_BYTES620,
        "native_token_bytes":TOK620.numel()*TOK620.element_size()},
    "baseline":BASE620,"primary_setting":PRIMARY620,
    "primary_results":PRIMARY_ROWS620,
    "sensitivity_results":RESULTS620,
    "per_query_results":DETAIL620,
    "integrity":INTEGRITY620,
    "elapsed_s":round(time.perf_counter()-T620,3),
    "limitations":[
        "DIAG and VALID were previously used in TEST617/618/619; not independent holdouts.",
        "All 1440 synthetic records share a fact template.",
        "IVF posting visits count list entries; not exact CPU instructions or disk reads.",
        "Exact native tensors are resident in RAM for this experiment.",
        "TOP2 geometry is evaluated only on touched cartridges.",
        "Full native and full TOP2 scans are evaluation references only.",
        "No new CUT3 answer-generation or HDD/SSD I/O is tested.",
        "The PCA basis is fitted to cartridge activations without labels.",
        "No claim of sublinear scaling follows from one bank size."
    ]}
os.makedirs(os.path.dirname(OUTPUT620),exist_ok=True)
with open(OUTPUT620,"w",encoding="utf-8") as f:
    json.dump(FINAL620,f,ensure_ascii=False,indent=2)
log620("TEST620_FINAL",json.dumps({
    "baseline":BASE620,"primary":PRIMARY_ROWS620,
    "index_bytes":STATIC_BYTES620,"native_token_bytes":TOK620.numel()*TOK620.element_size(),
    "integrity":INTEGRITY620,"elapsed_s":FINAL620["elapsed_s"],
    "output":OUTPUT620},ensure_ascii=False))
if not all(INTEGRITY620.values()):raise RuntimeError("TEST620_INTEGRITY_FAILURE")
log620("TEST620_STATUS COMPLETE")

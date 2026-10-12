
# TEST623 — AKBASCORE MAM · RAM–SSD NATIVE MEMORY EXTENSION
# EXACT TEST620 GEOMETRIC ROUTER + TEST621 PHYSICAL CUT3 DISK PATH
# FROZEN TEST610 MOTOR · NO TOKEN VOTING · NO WORD SEARCH · NO GOLD FALLBACK
import os,sys,time,json,hashlib,urllib.request,subprocess,importlib.util,random
from pathlib import Path
import numpy as np,torch
T623=time.perf_counter();ROOT623=Path("/content/AKBASCORE_TEST623");POOL623=ROOT623/"cut3"
POOL623.mkdir(parents=True,exist_ok=True)
SOURCE623="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/610.py"
LOCK623="a62611a4ffd91c2fcf03796c31da2af11ccd8141345655b562f3a614568af3e7"
SEED623=6202026;RANK623=128;NLIST623=16;BETA623=8.0
CONFIGS623=[(2,16,16),(4,32,16),(8,64,16),(16,128,16),(16,128,32)]
PRIMARY623=(8,64,16)
def log623(*x):print(*x,flush=True)
def sha623(b):return hashlib.sha256(b).hexdigest()
def filehash623(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()
def ms623(t):return round((time.perf_counter()-t)*1000,4)
def pct623(a,p):return round(float(np.percentile(a,p)),4) if a else None
def save623(p,x):
    with open(p,"w",encoding="utf-8") as f:json.dump(x,f,ensure_ascii=False,indent=2,default=str)
random.seed(SEED623);np.random.seed(SEED623);torch.manual_seed(SEED623)
if importlib.util.find_spec("faiss") is None:subprocess.check_call([sys.executable,"-m","pip","install","-q","faiss-cpu"])
import faiss
faiss.omp_set_num_threads(4)
log623("="*132)
log623("TEST623 — AKBASCORE MAM / RAM–SSD NATIVE GEOMETRIC TRACE")
log623("="*132)
src623=urllib.request.urlopen(urllib.request.Request(SOURCE623,headers={"User-Agent":"AKBASCORE-TEST623"}),timeout=120).read()
SOURCE_LOCK623=sha623(src623)==LOCK623
assert SOURCE_LOCK623,("SOURCE_SHA_MISMATCH",sha623(src623))
split623="print('[6/7] Demo acceptance"
assert src623.decode().count(split623)==1
exec(compile(src623.decode().split(split623,1)[0],"610.py","exec"),globals())
assert tuple(CFG)==(3,0,8.0) and len(CAR)==96 and H1==12 and H5==12
assert sum(x["answer_ok"] for x in PRE_AUDIT)==12
WEIGHT623=weight_sha();SENTINEL623=sentinel()
assert not any(p.requires_grad for p in model.parameters())
log623("BASELINE",{"SOURCE_LOCK":SOURCE_LOCK623,"FRAM_R1":H1,"FRAM_R5":H5,"CUT3":12})
# EXACT TEST620 NATIVE FRAME: label-free fit, no lexical features.
log623("[1/8] NATIVE GEOMETRIC FRAME")
FIT623=transform_fit(torch.cat(KS[3],dim=0),0)
K623=[apply(x,FIT623).detach().float().cpu().contiguous() for x in KS[3]]
assert len(K623)==96
assert all(torch.equal(a,b.float().cpu()) for a,b in zip(K623,KBASE))
Q623=[QBASE[i].detach().float().cpu().contiguous() for i in range(24)]
TOK623=torch.cat(K623,dim=0).contiguous()
LENS623=np.asarray([len(x) for x in K623],dtype=np.int32)
OWNER623=np.repeat(np.arange(len(K623),dtype=np.int32),LENS623)
NTOK623,DIM623=TOK623.shape
assert DIM623==3584 and NTOK623==1028
# TEST620 PCA128, adapted to 1028-token physical bank.
log623("[2/8] LABEL-FREE PCA128 + FAISS IVF")
t=time.perf_counter()
trainids623=np.random.default_rng(SEED623).choice(NTOK623,min(1028,NTOK623),replace=False)
TRAIN623=TOK623[torch.from_numpy(trainids623)].contiguous()
CENTER623=TRAIN623.mean(0)
U623,S623,V623=torch.pca_lowrank(TRAIN623-CENTER623,q=RANK623+16,center=False,niter=2)
BASIS623=V623[:,:RANK623].contiguous()
assert BASIS623.shape==(DIM623,RANK623)
PCA_MS623=ms623(t)
del U623,S623,V623,TRAIN623
PROJECTED623=np.ascontiguousarray(((TOK623-CENTER623)@BASIS623).numpy(),dtype=np.float32)
quant623=faiss.IndexFlatIP(RANK623)
INDEX623=faiss.IndexIVFFlat(quant623,RANK623,NLIST623,faiss.METRIC_INNER_PRODUCT)
t=time.perf_counter()
INDEX623.train(PROJECTED623[trainids623])
INDEX623.add(PROJECTED623)
INDEX_BUILD_MS623=ms623(t)
assert INDEX623.ntotal==NTOK623
LISTSIZE623=np.asarray([INDEX623.invlists.list_size(i) for i in range(NLIST623)],dtype=np.int64)
assert int(LISTSIZE623.sum())==NTOK623
INDEX_PATH623=ROOT623/"native_geometry_ivf.faiss"
faiss.write_index(INDEX623,str(INDEX_PATH623))
INDEX_BYTES623=INDEX_PATH623.stat().st_size
BASIS_BYTES623=BASIS623.numel()*BASIS623.element_size()
CENTER_BYTES623=CENTER623.numel()*CENTER623.element_size()
OWNER_BYTES623=OWNER623.nbytes
NATIVE_BYTES623=TOK623.numel()*TOK623.element_size()
log623("GEOMETRIC_INDEX",{"tokens":NTOK623,"dimension":DIM623,"rank":RANK623,"nlist":NLIST623,"pca_ms":PCA_MS623,"build_ms":INDEX_BUILD_MS623,"index_bytes":INDEX_BYTES623,"basis_bytes":BASIS_BYTES623,"center_bytes":CENTER_BYTES623,"owner_bytes":OWNER_BYTES623,"native_bytes":NATIVE_BYTES623})
# TEST620 TOP2_SOFT is preserved, including its ORIGINAL cartridge-length denominator.
def top2_623(q,k):
    sim=q@k.T
    top=torch.topk(sim,k=min(2,k.shape[0]),dim=1).values.double()
    return float(((torch.logsumexp(top*BETA623,dim=1)-np.log(len(k)))/BETA623).mean().item())
def native623(q,k):
    v=pair_scores(q,k,BETA623)
    return float(v.detach().cpu().item()) if isinstance(v,torch.Tensor) else float(v)
def projected623(q):
    return np.ascontiguousarray(((q-CENTER623)@BASIS623).numpy(),dtype=np.float32)
# Exact TEST620 routing: IVF token geometry -> owner union -> TOP2_SOFT -> FRAM.
# No count voting, no textual matching, no target-aware rescue.
def route623(q,nprobe,token_k,budget):
    qp=projected623(q);INDEX623.nprobe=nprobe
    t=time.perf_counter()
    _,cells=INDEX623.quantizer.search(qp,nprobe)
    postings=sum(int(LISTSIZE623[c]) for row in cells for c in row if c>=0)
    distances,ids=INDEX623.search(qp,token_k)
    route_ms=ms623(t)
    touched=set()
    for row in ids:
        for tid in row:
            if tid>=0:touched.add(int(OWNER623[tid]))
    t=time.perf_counter()
    geom={ci:top2_623(q,K623[ci]) for ci in touched}
    ordered=sorted(touched,key=lambda ci:(-geom[ci],ci))
    geometry_ms=ms623(t)
    candidates=ordered[:budget]
    t=time.perf_counter()
    ns={ci:native623(q,K623[ci]) for ci in candidates}
    final=sorted(candidates,key=lambda ci:(-ns[ci],ci))
    rerank_ms=ms623(t)
    return {"selected":final[0] if final else None,"final":final,"candidates":candidates,"touched":len(touched),"native_tokens_scanned":int(sum(LENS623[ci] for ci in touched)),"posting_visits":postings,"route_ms":route_ms,"geometry_ms":geometry_ms,"rerank_ms":rerank_ms,"total_ms":round(route_ms+geometry_ms+rerank_ms,4)}
# Physical CUT3 pool: one genuine serialized cache per original cartridge.
log623("[3/8] REAL CUT3 DISK MATERIALIZATION")
def cpu623(mem):return [(k.detach().cpu().contiguous(),v.detach().cpu().contiguous()) for k,v in mem]
def gpu623(mem):return [(k.to(DEVICE),v.to(DEVICE)) for k,v in mem]
def valid623(mem,ci):
    n=P+len(CAR[ci]["body"])
    return len(mem)==NL and all(tuple(k.shape)==(1,KVH,n,HD) and tuple(v.shape)==(1,KVH,n,HD) for k,v in mem)
def equal623(a,b):return len(a)==len(b) and all(torch.equal(x,u) and torch.equal(y,v) for (x,y),(u,v) in zip(a,b))
MANIFEST623=[];DISK_BYTES623=0;PAYLOAD_BYTES623=0
for ci in range(96):
    with torch.no_grad():mem=init(ci,3)
    assert valid623(mem,ci)
    obj=cpu623(mem);p=POOL623/f"cut3_{ci:05d}.pt"
    torch.save(obj,p)
    payload=sum(x.numel()*x.element_size()+y.numel()*y.element_size() for x,y in obj)
    info={"id":ci,"file":p.name,"sha256":filehash623(p),"disk_bytes":p.stat().st_size,"payload_bytes":payload}
    MANIFEST623.append(info);DISK_BYTES623+=info["disk_bytes"];PAYLOAD_BYTES623+=payload
    del mem,obj
    if (ci+1)%16==0:log623("DISK_POOL",ci+1,"/96",DISK_BYTES623)
save623(ROOT623/"manifest.json",MANIFEST623)
ROUNDTRIP623=[]
for ci in (0,15,31,47,63,79,95):
    info=MANIFEST623[ci];p=POOL623/info["file"]
    assert filehash623(p)==info["sha256"]
    loaded=torch.load(p,map_location="cpu",weights_only=True)
    with torch.no_grad():original=init(ci,3)
    ok=equal623(loaded,cpu623(original))
    ROUNDTRIP623.append({"id":ci,"bitwise":ok})
    assert ok,("ROUNDTRIP",ci)
    del loaded,original
log623("DISK_ROUNDTRIP",ROUNDTRIP623)
# Reference scores are diagnostic only, never fed into candidate selection.
log623("[4/8] FIXED GEOMETRIC CONFIGURATION SWEEP")
FULL623={}
for wi in range(24):
    q=Q623[wi]
    scores=[native623(q,k) for k in K623]
    FULL623[wi]=sorted(range(96),key=lambda ci:(-scores[ci],ci))
ROWS623=[];SUMMARIES623=[]
for nprobe,token_k,budget in CONFIGS623:
    subset=[]
    for wi in range(24):
        r=route623(Q623[wi],nprobe,token_k,budget)
        gold=PANEL[wi]["target"]
        r.update({"case":wi,"split":"CAL" if wi<12 else "HOLD","target":gold,"gold_touched":gold in r["candidates"] or gold in r["final"],"gold_candidate":gold in r["candidates"],"gold_r1":r["selected"]==gold,"gold_r5":gold in r["final"][:5],"full_native_top1_contained":FULL623[wi][0] in r["candidates"],"full_native_top5_contained":all(ci in r["candidates"] for ci in FULL623[wi][:5]),"nprobe":nprobe,"token_k":token_k,"budget":budget})
        subset.append(r);ROWS623.append(r)
    for split in ("CAL","HOLD"):
        s=[x for x in subset if x["split"]==split]
        summary={"split":split,"nprobe":nprobe,"token_k":token_k,"budget":budget,"questions":len(s),"gold_candidate":sum(x["gold_candidate"] for x in s),"gold_r1":sum(x["gold_r1"] for x in s),"gold_r5":sum(x["gold_r5"] for x in s),"native_top5_contained":sum(x["full_native_top5_contained"] for x in s),"mean_touched":round(float(np.mean([x["touched"] for x in s])),4),"mean_native_tokens":round(float(np.mean([x["native_tokens_scanned"] for x in s])),4),"route_p50_ms":pct623([x["route_ms"] for x in s],50),"geometry_p50_ms":pct623([x["geometry_ms"] for x in s],50),"rerank_p50_ms":pct623([x["rerank_ms"] for x in s],50),"total_p50_ms":pct623([x["total_ms"] for x in s],50)}
        SUMMARIES623.append(summary)
        log623("GEOMETRIC_RESULT",json.dumps(summary))
# The primary configuration is predetermined; HOLD never selects it.
log623("[5/8] REAL DISK E2E — FIXED PRIMARY",PRIMARY623)
def answer_disk623(ci,wi):
    info=MANIFEST623[ci];path=POOL623/info["file"]
    t=time.perf_counter()
    assert filehash623(path)==info["sha256"],("DISK_SHA_FAIL",ci)
    stored=torch.load(path,map_location="cpu",weights_only=True)
    read_ms=ms623(t)
    assert valid623(stored,ci)
    mem=gpu623(stored)
    target=PANEL[wi]["target"];question=W[wi][CAR[target]["type"]];gold=CAR[target]["gold"]
    t=time.perf_counter()
    try:
        with torch.no_grad():response=answer(mem,question)
        correct=bool(hit(response,gold));error=None
    except Exception as exc:
        response="";correct=False;error=repr(exc)
    answer_ms=ms623(t)
    del stored,mem
    return {"answer":str(response),"answer_correct":correct,"error":error,"file_bytes":info["disk_bytes"],"disk_sha_verified":True,"disk_load_with_sha_ms":read_ms,"answer_ms":answer_ms}
PRIMARY_ROWS623=[];FULL_CONTROL623=[]
for wi in HOLD:
    q=Q623[wi];gold=PANEL[wi]["target"]
    r=route623(q,*PRIMARY623)
    selected=r["selected"]
    record={"case":wi,"target":gold,"selected":selected,"selected_correct":selected==gold,**r}
    if selected is not None:record.update(answer_disk623(selected,wi))
    else:record.update({"answer":"","answer_correct":False,"error":"NO_GEOMETRIC_CANDIDATE","file_bytes":0,"disk_sha_verified":False})
    PRIMARY_ROWS623.append(record)
    log623("PRIMARY_E2E",wi,"SELECTED",selected,"TARGET",gold,"SELECT_OK",int(record["selected_correct"]),"ANSWER_OK",int(record["answer_correct"]),"TOUCHED",r["touched"],"RAW",repr(record["answer"]))
    full_selected=FULL623[wi][0]
    control={"case":wi,"target":gold,"selected":full_selected,"selected_correct":full_selected==gold,**answer_disk623(full_selected,wi)}
    FULL_CONTROL623.append(control)
    log623("FULL_NATIVE_DISK",wi,"SELECT_OK",int(control["selected_correct"]),"ANSWER_OK",int(control["answer_correct"]))
# Also test the highest-coverage preset as an explicit cost/accuracy control.
log623("[6/8] HIGH-COVERAGE GEOMETRIC DISK CONTROL")
HIGH_CONFIG623=(16,128,32)
HIGH_ROWS623=[]
for wi in HOLD:
    r=route623(Q623[wi],*HIGH_CONFIG623)
    selected=r["selected"];gold=PANEL[wi]["target"]
    item={"case":wi,"target":gold,"selected":selected,"selected_correct":selected==gold,**r}
    if selected is not None:item.update(answer_disk623(selected,wi))
    else:item.update({"answer_correct":False,"error":"NO_GEOMETRIC_CANDIDATE","file_bytes":0})
    HIGH_ROWS623.append(item)
    log623("HIGH_E2E",wi,"SELECT_OK",int(item["selected_correct"]),"ANSWER_OK",int(item["answer_correct"]),"TOUCHED",r["touched"])
log623("[7/8] INTEGRITY / MEMORY ACCOUNTING")
INTEGRITY623={"source_sha_lock":SOURCE_LOCK623,"source_sha_recheck":sha623(src623)==LOCK623,"weights_unchanged":weight_sha()==WEIGHT623,"sentinel_unchanged":sentinel()==SENTINEL623,"zero_trainable":not any(p.requires_grad for p in model.parameters()),"no_remaining_hooks":not any(layer._forward_hooks or layer._forward_pre_hooks for layer in model.model.layers),"disk_roundtrip_bitwise":all(x["bitwise"] for x in ROUNDTRIP623),"disk_manifest_complete":len(MANIFEST623)==96 and all((POOL623/x["file"]).exists() for x in MANIFEST623)}
STORAGE623={"disk_cartridges":96,"disk_pool_bytes":DISK_BYTES623,"cut3_payload_bytes":PAYLOAD_BYTES623,"native_activation_bytes":NATIVE_BYTES623,"faiss_index_bytes":INDEX_BYTES623,"pca_basis_bytes":BASIS_BYTES623,"pca_center_bytes":CENTER_BYTES623,"owner_bytes":OWNER_BYTES623,"index_auxiliary_bytes":INDEX_BYTES623+BASIS_BYTES623+CENTER_BYTES623+OWNER_BYTES623,"index_plus_native_activation_bytes":INDEX_BYTES623+BASIS_BYTES623+CENTER_BYTES623+OWNER_BYTES623+NATIVE_BYTES623}
ACCURACY623={"questions":len(HOLD),"primary_selected_correct":sum(x["selected_correct"] for x in PRIMARY_ROWS623),"primary_answer_correct":sum(x["answer_correct"] for x in PRIMARY_ROWS623),"high_selected_correct":sum(x["selected_correct"] for x in HIGH_ROWS623),"high_answer_correct":sum(x["answer_correct"] for x in HIGH_ROWS623),"full_native_selected_correct":sum(x["selected_correct"] for x in FULL_CONTROL623),"full_native_answer_correct":sum(x["answer_correct"] for x in FULL_CONTROL623),"locked_cut3_baseline":sum(x["answer_ok"] for x in PRE_AUDIT)}
LATENCY623={"primary_address_p50_ms":pct623([x["total_ms"] for x in PRIMARY_ROWS623],50),"primary_disk_load_p50_ms":pct623([x.get("disk_load_with_sha_ms",0) for x in PRIMARY_ROWS623],50),"primary_answer_p50_ms":pct623([x.get("answer_ms",0) for x in PRIMARY_ROWS623],50),"high_address_p50_ms":pct623([x["total_ms"] for x in HIGH_ROWS623],50),"high_disk_load_p50_ms":pct623([x.get("disk_load_with_sha_ms",0) for x in HIGH_ROWS623],50),"high_answer_p50_ms":pct623([x.get("answer_ms",0) for x in HIGH_ROWS623],50)}
FINAL623={"test":"623","title":"AkbasCore MAM — RAM–SSD Native Memory Extension","architecture":"TEST620 L3 PCA128 + FAISS IVF + touched TOP2_SOFT + locked FRAM + physical CUT3","source_sha256":sha623(src623),"primary_config":PRIMARY623,"high_coverage_config":HIGH_CONFIG623,"storage":STORAGE623,"accuracy":ACCURACY623,"latency":LATENCY623,"integrity":INTEGRITY623,"roundtrip":ROUNDTRIP623,"sweep":SUMMARIES623,"sweep_cases":ROWS623,"primary_cases":PRIMARY_ROWS623,"high_cases":HIGH_ROWS623,"full_native_disk_cases":FULL_CONTROL623,"elapsed_s":round(time.perf_counter()-T623,3),"limitations":["96 genuine disk CUT3 files; the TEST620 synthetic 1440-cartridge bank is not physically materialized here","Original CAL/HOLD panel is reused; not an independent unseen benchmark","PCA rank128 and IVF nlist16 are adapted to the 96-cartridge bank","Native activations remain resident in CPU memory in addition to the FAISS index","No token voting, lexical matching, label-based rescue, or full-bank fallback in indexed routes","High-coverage geometry may touch all cartridges and does not prove selective scaling","Colab filesystem does not certify physical SSD or cold-cache reads","Disk load timing includes SHA verification; file bytes are not actual device I/O counters","No claim of quantum superposition or million-cartridge scaling"]}
OUT623=ROOT623/"TEST623_RESULT.json";save623(OUT623,FINAL623)
log623("[8/8] FINAL")
log623("TEST623_STORAGE",json.dumps(STORAGE623))
log623("TEST623_ACCURACY",json.dumps(ACCURACY623))
log623("TEST623_LATENCY",json.dumps(LATENCY623))
log623("TEST623_INTEGRITY",json.dumps(INTEGRITY623))
log623("TEST623_OUTPUT",str(OUT623))
log623("TEST623_STATUS","COMPLETE" if all(INTEGRITY623.values()) else "INTEGRITY_FAIL")
log623("="*132)


# TEST622 — AKBASCORE MAM · RAM–SSD NATIVE MEMORY EXTENSION
# FROZEN TEST610 MOTOR · REAL CUT3 FILES · FAISS IVF RAM ROUTER · DISK E2E · AUDIT
import os,sys,subprocess,importlib.util,urllib.request,hashlib,json,time,gc,platform
from pathlib import Path
import numpy as np
if importlib.util.find_spec("faiss") is None:subprocess.check_call([sys.executable,"-m","pip","install","-q","faiss-cpu"])
import faiss,torch
TEST="622";ROOT=Path("/content/AKBASCORE_TEST622");POOL=ROOT/"cartridges"
ROOT.mkdir(parents=True,exist_ok=True);POOL.mkdir(parents=True,exist_ok=True)
SOURCE_URL="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/610.py"
SOURCE_SHA="a62611a4ffd91c2fcf03796c31da2af11ccd8141345655b562f3a614568af3e7"
START622=time.perf_counter()
def digest_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()
def pct(v,p):return round(float(np.percentile(v,p)),4) if v else None
def dump(obj,path):
    with open(path,"w",encoding="utf-8") as f:json.dump(obj,f,ensure_ascii=False,indent=2,default=str)
print("="*132)
print("TEST622 — AKBASCORE MAM / RAM–SSD NATIVE MEMORY EXTENSION")
print("="*132)
with urllib.request.urlopen(urllib.request.Request(SOURCE_URL,headers={"User-Agent":"AKBASCORE-TEST622"}),timeout=120) as f:source622=f.read()
SOURCE_ACTUAL622=hashlib.sha256(source622).hexdigest()
SOURCE_LOCK622=(SOURCE_ACTUAL622==SOURCE_SHA)
print("SOURCE_LOCK",SOURCE_LOCK622,SOURCE_ACTUAL622,flush=True)
assert SOURCE_LOCK622,"SOURCE SHA MISMATCH"
boundary622="print('[6/7] Demo acceptance"
text622=source622.decode("utf-8")
assert text622.count(boundary622)==1,"SOURCE BOUNDARY MISMATCH"
exec(compile(text622.split(boundary622,1)[0],"610.py","exec"),globals())
assert CFG==(3,0,8.0) and len(CAR)==96 and len(KBASE)==96
assert H1==12 and H5==12 and sum(x["answer_ok"] for x in PRE_AUDIT)==12
assert weight_sha()==WB and sentinel()==SB
print("BASELINE_PASS",{"FRAM_R1":H1,"FRAM_R5":H5,"CUT3":sum(x["answer_ok"] for x in PRE_AUDIT)},flush=True)
# The locked engine functions init(), append(), answer(), pair_scores() are never replaced.
def cpu_cache(mem):
    return [(k.detach().contiguous().cpu(),v.detach().contiguous().cpu()) for k,v in mem]
def device_cache(mem):
    return [(k.to(DEVICE),v.to(DEVICE)) for k,v in mem]
def same_cache(a,b):
    return len(a)==len(b) and all(torch.equal(x,u) and torch.equal(y,v) for (x,y),(u,v) in zip(a,b))
def cache_valid(mem,ci):
    if len(mem)!=NL:return False
    n=P+len(CAR[ci]["body"])
    return all(tuple(k.shape)==(1,KVH,n,HD) and tuple(v.shape)==(1,KVH,n,HD) and bool(torch.isfinite(k).all()) and bool(torch.isfinite(v).all()) for k,v in mem)
def payload_bytes(mem):
    return sum(x.numel()*x.element_size()+y.numel()*y.element_size() for x,y in mem)
# A fresh disk pool is built from the unchanged CUT3 engine.
print("[1/8] BUILDING PHYSICAL FILE POOL",flush=True)
manifest622=[];disk_total622=0;payload_total622=0;write_ms622=[]
for ci in range(len(CAR)):
    t=time.perf_counter()
    with torch.no_grad():mem=init(ci,3)
    assert cache_valid(mem,ci),f"CUT3 SHAPE {ci}"
    obj=cpu_cache(mem);path=POOL/f"cut3_{ci:05d}.pt"
    torch.save(obj,path)
    row={"id":ci,"file":path.name,"sha256":digest_file(path),"disk_bytes":path.stat().st_size,"payload_bytes":payload_bytes(obj),"tokens":P+len(CAR[ci]["body"])}
    manifest622.append(row);disk_total622+=row["disk_bytes"];payload_total622+=row["payload_bytes"]
    write_ms622.append((time.perf_counter()-t)*1000)
    del mem,obj
    if (ci+1)%16==0:print("PHYSICAL_CARTRIDGES",ci+1,"/",len(CAR),"DISK_BYTES",disk_total622,flush=True)
dump(manifest622,ROOT/"manifest.json")
# Validate that actual serialized caches survive disk roundtrip.
print("[2/8] BITWISE DISK ROUNDTRIP",flush=True)
roundtrip622=[]
for ci in (0,15,31,47,63,79,95):
    info=manifest622[ci];path=POOL/info["file"]
    t=time.perf_counter()
    stored=torch.load(path,map_location="cpu",weights_only=True)
    read_ms=(time.perf_counter()-t)*1000
    with torch.no_grad():original=init(ci,3)
    ok=same_cache(stored,cpu_cache(original))
    roundtrip622.append({"id":ci,"bitwise_equal":ok,"read_ms":round(read_ms,4)})
    print("ROUNDTRIP",ci,"BITWISE",ok,"READ_MS",round(read_ms,3),flush=True)
    assert ok,f"ROUNDTRIP FAIL {ci}"
    del stored,original
# Native RAM representation: exact L3/rank0 frozen vectors from TEST610.
print("[3/8] BUILDING FAISS IVF RAM ROUTER",flush=True)
native_keys622=[x.detach().float().cpu().contiguous() for x in KBASE]
native_ram_bytes622=sum(x.numel()*x.element_size() for x in native_keys622)
owner622=np.concatenate([np.full(len(x),i,dtype=np.int32) for i,x in enumerate(native_keys622)])
vectors622=np.ascontiguousarray(torch.cat(native_keys622).numpy(),dtype=np.float32)
assert vectors622.shape[0]==len(owner622)
assert vectors622.shape[1]==3584
nlist622=16
quant622=faiss.IndexFlatIP(vectors622.shape[1])
ivf622=faiss.IndexIVFFlat(quant622,vectors622.shape[1],nlist622,faiss.METRIC_INNER_PRODUCT)
ivf622.cp.seed=622
t=time.perf_counter()
ivf622.train(vectors622);ivf622.add(vectors622)
index_build_ms622=(time.perf_counter()-t)*1000
assert ivf622.ntotal==len(vectors622)
index_path622=ROOT/"native_ram_ivf.faiss"
faiss.write_index(ivf622,str(index_path622))
index_bytes622=index_path622.stat().st_size
owner_bytes622=owner622.nbytes
print("RAM_INDEX",json.dumps({"native_token_count":len(vectors622),"dimension":3584,"nlist":nlist622,"index_file_bytes":index_bytes622,"owner_bytes":owner_bytes622,"native_key_bytes":native_ram_bytes622,"build_ms":round(index_build_ms622,3)}),flush=True)
# Full native FRAM is an independent reference, not a hidden routing fallback.
def full_reference622(wi):
    scores=[pair_scores(QBASE[wi],k,8.0) for k in native_keys622]
    return rankings(scores)
# Candidate generation uses IVF hits only; native FRAM reranks only the candidate set.
# No gold labels are used in selection or in router fitting.
def route622(wi,nprobe,token_k,budget):
    ivf622.nprobe=nprobe
    query=np.ascontiguousarray(QBASE[wi].float().cpu().numpy(),dtype=np.float32)
    t=time.perf_counter()
    distances,indices=ivf622.search(query,token_k)
    hits=indices.reshape(-1)
    valid=hits[hits>=0]
    candidates=sorted(set(int(x) for x in owner622[valid]))
    ivf_ms=(time.perf_counter()-t)*1000
    t=time.perf_counter()
    # Cheap token-hit voting is a prefilter, not the locked FRAM scoring function.
    counts={}
    for ix in valid:
        ci=int(owner622[ix]);counts[ci]=counts.get(ci,0)+1
    candidates.sort(key=lambda ci:(-counts[ci],ci))
    shortlisted=candidates[:budget]
    vote_ms=(time.perf_counter()-t)*1000
    t=time.perf_counter()
    scored=[(ci,pair_scores(QBASE[wi],native_keys622[ci],8.0)) for ci in shortlisted]
    scored.sort(key=lambda x:(-x[1],x[0]))
    rerank_ms=(time.perf_counter()-t)*1000
    return {"ids":[ci for ci,_ in scored],"touched":len(candidates),"shortlisted":len(shortlisted),"ivf_ms":ivf_ms,"vote_ms":vote_ms,"rerank_ms":rerank_ms,"token_hits":len(valid)}
# A transparent sweep: failures remain visible and cannot be hidden by full-bank fallback.
print("[4/8] ROUTER SWEEP / 24 PREVIOUSLY USED PANEL QUESTIONS",flush=True)
configs622=[(2,16,16),(2,32,16),(4,32,32),(8,64,32),(16,128,64)]
references622={wi:full_reference622(wi) for wi in range(24)}
router_results622=[]
for nprobe,token_k,budget in configs622:
    rows=[]
    for wi in range(24):
        r=route622(wi,nprobe,token_k,budget)
        gold=PANEL[wi]["target"];reference=references622[wi]
        rows.append({"case":wi,"split":"CAL" if wi<12 else "HOLD","target":gold,"gold_in_candidates":gold in r["ids"],"gold_r1":bool(r["ids"] and r["ids"][0]==gold),"gold_r5":gold in r["ids"][:5],"full_native_top1_contained":reference[0] in r["ids"],"full_native_top5_contained":all(x in r["ids"] for x in reference[:5]),**r})
    summary={"nprobe":nprobe,"token_k":token_k,"budget":budget,"cal_gold_candidate":sum(x["gold_in_candidates"] for x in rows[:12]),"hold_gold_candidate":sum(x["gold_in_candidates"] for x in rows[12:]),"hold_r1":sum(x["gold_r1"] for x in rows[12:]),"hold_r5":sum(x["gold_r5"] for x in rows[12:]),"hold_full_top5_contained":sum(x["full_native_top5_contained"] for x in rows[12:]),"mean_touched":round(float(np.mean([x["touched"] for x in rows])),3),"ivf_p50_ms":pct([x["ivf_ms"] for x in rows],50),"rerank_p50_ms":pct([x["rerank_ms"] for x in rows],50)}
    router_results622.append({"summary":summary,"rows":rows})
    print("ROUTER_CONFIG",json.dumps(summary),flush=True)
# Select using CAL only; HOLD outcomes are never used for choosing the configuration.
selected622=max(router_results622,key=lambda x:(x["summary"]["cal_gold_candidate"],-x["summary"]["mean_touched"],-x["summary"]["budget"]))
selected_config622=selected622["summary"]
print("SELECTED_FROM_CAL_ONLY",json.dumps(selected_config622),flush=True)
# Actual disk read + answer for every HOLD query. No source fact supplied to answer().
print("[5/8] END-TO-END RAM -> DISK -> CUT3 -> ANSWER",flush=True)
disk_cases622=[];read_times622=[];answer_times622=[];address_times622=[]
for wi in HOLD:
    row=selected622["rows"][wi]
    target=PANEL[wi]["target"]
    selected=row["ids"][0] if row["ids"] else None
    address_ms=row["ivf_ms"]+row["vote_ms"]+row["rerank_ms"]
    result={"case":wi,"target":target,"selected":selected,"target_in_candidates":row["gold_in_candidates"],"selected_correct":selected==target,"address_ms":round(address_ms,4),"answer_correct":False,"disk_read_bytes":0}
    if selected is None:
        result["error"]="NO_IVF_CANDIDATE"
    else:
        info=manifest622[selected];path=POOL/info["file"]
        t=time.perf_counter()
        checksum_ok=digest_file(path)==info["sha256"]
        if not checksum_ok:raise RuntimeError(f"DISK SHA FAIL {selected}")
        stored=torch.load(path,map_location="cpu",weights_only=True)
        read_ms=(time.perf_counter()-t)*1000
        assert cache_valid(stored,selected)
        mem=device_cache(stored)
        typ=CAR[target]["type"];question=W[wi][typ];gold=CAR[target]["gold"]
        t=time.perf_counter()
        try:
            with torch.no_grad():response=answer(mem,question)
            result["answer"]=str(response);result["answer_correct"]=bool(hit(response,gold))
        except Exception as exc:
            result["error"]=repr(exc)
        answer_ms=(time.perf_counter()-t)*1000
        result.update({"disk_read_bytes":info["disk_bytes"],"disk_sha_ok":checksum_ok,"disk_read_ms_including_sha":round(read_ms,4),"answer_ms":round(answer_ms,4)})
        read_times622.append(read_ms);answer_times622.append(answer_ms)
        del stored,mem
    address_times622.append(address_ms)
    disk_cases622.append(result)
    print("DISK_E2E",wi,"SELECTED",selected,"TARGET",target,"SELECT_OK",int(result["selected_correct"]),"ANSWER_OK",int(result["answer_correct"]),"READ_BYTES",result["disk_read_bytes"],"RAW",repr(result.get("answer",result.get("error"))),flush=True)
# Full native FRAM + one disk cartridge: diagnostic reference independent of IVF.
print("[6/8] FULL-NATIVE DISK CONTROL",flush=True)
native_disk_control622=[]
for wi in HOLD:
    target=PANEL[wi]["target"];selected=references622[wi][0]
    info=manifest622[selected];path=POOL/info["file"]
    assert digest_file(path)==info["sha256"]
    stored=torch.load(path,map_location="cpu",weights_only=True)
    mem=device_cache(stored)
    typ=CAR[target]["type"];question=W[wi][typ];gold=CAR[target]["gold"]
    try:
        with torch.no_grad():response=answer(mem,question)
        ok=bool(hit(response,gold));error=None
    except Exception as exc:
        ok=False;error=repr(exc)
    native_disk_control622.append({"case":wi,"selected":selected,"target":target,"selected_correct":selected==target,"answer_correct":ok,"error":error})
    print("NATIVE_DISK_CONTROL",wi,"SELECT_OK",int(selected==target),"ANSWER_OK",int(ok),flush=True)
    del stored,mem
# Existing five-cartridge baseline is reported from PRE_AUDIT, not relabelled as disk E2E.
print("[7/8] FINAL INTEGRITY",flush=True)
integrity622={"source_sha_lock":SOURCE_LOCK622,"source_sha_recheck":hashlib.sha256(source622).hexdigest()==SOURCE_SHA,"weights_unchanged":weight_sha()==WB,"sentinel_unchanged":sentinel()==SB,"zero_trainable":not any(p.requires_grad for p in model.parameters()),"no_remaining_hooks":not any(m._forward_hooks or m._forward_pre_hooks for layer in model.model.layers for m in (layer,layer.self_attn,layer.self_attn.q_proj)),"disk_roundtrip_bitwise":all(x["bitwise_equal"] for x in roundtrip622),"disk_manifest_complete":len(manifest622)==96 and all((POOL/x["file"]).exists() for x in manifest622)}
summary622={"test":"622","title":"AkbasCore MAM — RAM–SSD Native Memory Extension","gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU","torch":torch.__version__,"transformers":__import__("transformers").__version__,"faiss":getattr(faiss,"__version__","unknown"),"source_sha256":SOURCE_ACTUAL622,"real_cut3_disk_cartridges":96,"disk_pool_bytes":disk_total622,"logical_cut3_payload_bytes":payload_total622,"native_ram_keys_bytes":native_ram_bytes622,"faiss_index_file_bytes":index_bytes622,"owner_map_bytes":owner_bytes622,"total_index_and_native_keys_bytes":native_ram_bytes622+index_bytes622+owner_bytes622,"native_token_count":len(vectors622),"router_selected":selected_config622,"router_hold_selected_correct":sum(x["selected_correct"] for x in disk_cases622),"router_hold_disk_answer_correct":sum(x["answer_correct"] for x in disk_cases622),"native_full_hold_selected_correct":sum(x["selected_correct"] for x in native_disk_control622),"native_full_hold_disk_answer_correct":sum(x["answer_correct"] for x in native_disk_control622),"locked_original_multicartridge_baseline_correct":sum(x["answer_ok"] for x in PRE_AUDIT),"hold_questions":len(HOLD),"actual_disk_file_reads_in_router":sum(x["disk_read_bytes"]>0 for x in disk_cases622),"router_disk_bytes_read":sum(x["disk_read_bytes"] for x in disk_cases622),"address_p50_ms":pct(address_times622,50),"disk_load_p50_ms":pct(read_times622,50),"answer_p50_ms":pct(answer_times622,50),"integrity":integrity622,"elapsed_seconds":round(time.perf_counter()-START622,3),"limitations":["96 physical CUT3 cartridges, not 1536","IVF routing at 96 cartridges is not a million-cartridge scaling proof","Existing 24 questions are not independent unseen evaluation","FAISS candidate voting is a new prefilter; its failures are measured, not hidden","Single disk cartridge is passed to CUT3; multi-cartridge disk assembly is not established","Colab /content filesystem does not prove physical SSD type or cold-cache latency","RAM stores both native keys and an additional IVF index; sizes are not interchangeable","Native activation address search is not a demonstrated quantum or compressed superposition engine"]}
report622={"summary":summary622,"manifest":manifest622,"roundtrip":roundtrip622,"router_sweep":router_results622,"disk_e2e":disk_cases622,"native_disk_control":native_disk_control622}
output622=ROOT/"TEST622_RESULT.json";dump(report622,output622)
print("[8/8] FINAL REPORT",flush=True)
print("TEST622_STORAGE",json.dumps({k:summary622[k] for k in ("real_cut3_disk_cartridges","disk_pool_bytes","native_ram_keys_bytes","faiss_index_file_bytes","owner_map_bytes","total_index_and_native_keys_bytes")}),flush=True)
print("TEST622_ACCURACY",json.dumps({k:summary622[k] for k in ("hold_questions","router_hold_selected_correct","router_hold_disk_answer_correct","native_full_hold_selected_correct","native_full_hold_disk_answer_correct","locked_original_multicartridge_baseline_correct")}),flush=True)
print("TEST622_LATENCY",json.dumps({k:summary622[k] for k in ("address_p50_ms","disk_load_p50_ms","answer_p50_ms")}),flush=True)
print("TEST622_INTEGRITY",json.dumps(integrity622),flush=True)
print("TEST622_OUTPUT",str(output622),flush=True)
print("TEST622_STATUS","COMPLETE" if all(integrity622.values()) else "INTEGRITY_FAIL",flush=True)
print("="*132)

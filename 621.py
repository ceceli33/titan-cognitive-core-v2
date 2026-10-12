
# TEST621 — AKBASCORE MAM · REAL DISK CUT3 CARTRIDGE POOL + NATIVE RAM ADDRESSING
# SHA-LOCKED TEST610 BACKBONE · NO MODEL TRAINING · NO ENGINE MODIFICATION
import os,sys,json,time,hashlib,urllib.request,statistics,gc,platform
from pathlib import Path
import torch,numpy as np
TEST="621";ROOT=Path("/content/AKBASCORE_TEST621");POOL=ROOT/"ssd_cartridges";ROOT.mkdir(parents=True,exist_ok=True);POOL.mkdir(parents=True,exist_ok=True)
SOURCE_URL="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/610.py"
SOURCE_SHA="a62611a4ffd91c2fcf03796c31da2af11ccd8141345655b562f3a614568af3e7"
t_start=time.perf_counter()
print("="*132)
print("TEST621 — REAL DISK CUT3 CARTRIDGE POOL / NATIVE ACTIVATION RAM ADDRESSING")
print("="*132)
req=urllib.request.Request(SOURCE_URL,headers={"User-Agent":"AKBASCORE-TEST621"})
with urllib.request.urlopen(req,timeout=120) as f:src=f.read()
sha=hashlib.sha256(src).hexdigest()
print("SOURCE_SHA256",sha,"LOCK",sha==SOURCE_SHA,flush=True)
assert sha==SOURCE_SHA,"TEST610 SOURCE LOCK FAILED"
source_text=src.decode("utf-8")
boundary="print('[6/7] Demo acceptance"
assert source_text.count(boundary)==1,"TEST610 BOUNDARY CHANGED"
exec(compile(source_text.split(boundary,1)[0],"610.py","exec"),globals())
assert CFG==(3,0,8.0) and len(CAR)==96 and len(KBASE)==96
assert H1==12 and H5==12 and sum(x["answer_ok"] for x in PRE_AUDIT)==12
assert weight_sha()==WB and sentinel()==SB
print("TEST621_BASELINE",json.dumps({"fram_holdout_r1":H1,"fram_holdout_r5":H5,"cut3_baseline":sum(x["answer_ok"] for x in PRE_AUDIT),"cartridges":len(CAR)}),flush=True)
# The original CUT3 engine remains intact. Each physical file contains an actual 28-layer cache.
def file_sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):h.update(chunk)
    return h.hexdigest()
def cache_bytes(mem):
    return sum(k.numel()*k.element_size()+v.numel()*v.element_size() for k,v in mem)
def cache_cpu(mem):
    return [(k.detach().contiguous().cpu(),v.detach().contiguous().cpu()) for k,v in mem]
def cache_device(mem):
    return [(k.to(DEVICE),v.to(DEVICE)) for k,v in mem]
def equal_cache(a,b):
    return len(a)==len(b) and all(torch.equal(ka,kb) and torch.equal(va,vb) for (ka,va),(kb,vb) in zip(a,b))
def cache_shape_ok(mem,ci):
    if len(mem)!=NL:return False
    expected=P+len(CAR[ci]["body"])
    return all(tuple(k.shape)==(1,KVH,expected,HD) and tuple(v.shape)==(1,KVH,expected,HD) and bool(torch.isfinite(k).all()) and bool(torch.isfinite(v).all()) for k,v in mem)
MANIFEST=[];write_times=[];disk_total=0;logical_total=0
print("[1/6] MATERIALIZING 96 REAL CUT3 CARTRIDGES ON DISK",flush=True)
for ci in range(96):
    t=time.perf_counter()
    with torch.no_grad():mem=init(ci,3)
    assert cache_shape_ok(mem,ci),f"CUT3 INVALID {ci}"
    payload=cache_cpu(mem);logical=cache_bytes(payload)
    path=POOL/f"cartridge_{ci:05d}.pt"
    torch.save(payload,path)
    disk_size=path.stat().st_size;digest621=file_sha(path)
    MANIFEST.append({"id":ci,"file":path.name,"sha256":digest621,"disk_bytes":disk_size,"cut3_payload_bytes":logical,"tokens":P+len(CAR[ci]["body"]),"layers":len(payload)})
    disk_total+=disk_size;logical_total+=logical;write_times.append((time.perf_counter()-t)*1000)
    del mem,payload
    if (ci+1)%16==0:print("DISK_WRITE",ci+1,"/96","BYTES",disk_total,flush=True)
with open(ROOT/"manifest.json","w") as f:json.dump(MANIFEST,f,indent=2)
# RAM holds only the native activation routing representation and metadata.
# Full CUT3 caches are not retained in the Python cartridge pool.
RAM_KEYS=[x.contiguous().float().cpu() for x in KBASE]
RAM_BYTES=sum(x.numel()*x.element_size() for x in RAM_KEYS)
META_BYTES=(ROOT/"manifest.json").stat().st_size
print("POOL_PHYSICAL",json.dumps({"files":96,"disk_bytes":disk_total,"logical_cut3_bytes":logical_total,"ram_native_keys_bytes":RAM_BYTES,"manifest_bytes":META_BYTES}),flush=True)
print("[2/6] DISK ROUNDTRIP AUDIT",flush=True)
ROUNDTRIP=[]
for ci in (0,15,31,47,63,79,95):
    path=POOL/MANIFEST[ci]["file"]
    assert file_sha(path)==MANIFEST[ci]["sha256"]
    t=time.perf_counter()
    stored=torch.load(path,map_location="cpu",weights_only=True)
    load_ms=(time.perf_counter()-t)*1000
    with torch.no_grad():reference=init(ci,3)
    identical=equal_cache(stored,cache_cpu(reference))
    ROUNDTRIP.append({"id":ci,"sha_pass":True,"bitwise_equal":identical,"load_ms":round(load_ms,4)})
    print("ROUNDTRIP",ci,"BITWISE",identical,"LOAD_MS",round(load_ms,3),flush=True)
    assert identical,f"DISK ROUNDTRIP FAILED {ci}"
    del stored,reference
print("[3/6] NATIVE RAM ADDRESSING — LOCKED FRAM GEOMETRY",flush=True)
# Full 96-key FRAM scan: scientifically valid RAM address lookup, NOT a sublinear index.
# TEST620's 1536-cartridge FAISS IVF index is a separate scaling result.
RESULTS=[];disk_reads=[];address_times=[];answer_times=[];read_bytes=0
for wi in HOLD:
    target=PANEL[wi]["target"];q=QBASE[wi]
    t=time.perf_counter()
    scores=[pair_scores(q,k,8.0) for k in RAM_KEYS]
    ordered=rankings(scores);chosen=ordered[0]
    address_ms=(time.perf_counter()-t)*1000
    assert chosen==target,f"FRAM BASELINE CHANGED CASE {wi}"
    path=POOL/MANIFEST[chosen]["file"]
    t=time.perf_counter()
    assert file_sha(path)==MANIFEST[chosen]["sha256"]
    stored=torch.load(path,map_location="cpu",weights_only=True)
    disk_ms=(time.perf_counter()-t)*1000
    assert cache_shape_ok(stored,chosen)
    mem=cache_device(stored)
    typ=CAR[target]["type"];question=W[wi][typ];gold=CAR[target]["gold"]
    t=time.perf_counter()
    try:
        with torch.no_grad():raw=answer(mem,question)
        correct=bool(hit(raw,gold));error=None
    except Exception as exc:
        raw="";correct=False;error=repr(exc)
    answer_ms=(time.perf_counter()-t)*1000
    row={"case":wi,"target":target,"selected":chosen,"retrieval_correct":chosen==target,"single_cartridge_answer_correct":correct,"gold":gold,"answer":str(raw),"error":error,"address_ms":round(address_ms,4),"disk_read_ms_including_sha256":round(disk_ms,4),"answer_ms":round(answer_ms,4),"read_bytes":MANIFEST[chosen]["disk_bytes"],"loaded_cartridges":1}
    RESULTS.append(row);read_bytes+=row["read_bytes"];disk_reads.append(disk_ms);address_times.append(address_ms);answer_times.append(answer_ms)
    print("SSD_QUERY",wi,"TARGET",target,"SELECTED",chosen,"RAM_R1",int(chosen==target),"SSD_READ_BYTES",row["read_bytes"],"ANSWER_OK",int(correct),"ADDRESS_MS",round(address_ms,2),"DISK_MS",round(disk_ms,2),"ANSWER_MS",round(answer_ms,2),"RAW",repr(str(raw)),flush=True)
    del stored,mem
print("[4/6] ORIGINAL MULTI-CARTRIDGE CUT3 CONTROL",flush=True)
# This is an unchanged in-memory engine control, not a claim of SSD-assembled multi-cache.
CONTROL=[]
for wi in HOLD:
    target=PANEL[wi]["target"];ids=rankings([pair_scores(QBASE[wi],k,8.0) for k in RAM_KEYS])[:5]
    keys=sorted(ids);typ=CAR[target]["type"];question=W[wi][typ];gold=CAR[target]["gold"]
    try:
        with torch.no_grad():
            mem=build_cut3(keys,3);validate_cut3(mem,keys);raw=answer(mem,question)
        ok=bool(hit(raw,gold));error=None
    except Exception as exc:
        ok=False;error=repr(exc)
    CONTROL.append({"case":wi,"selected":keys,"target_present":target in keys,"answer_correct":ok,"error":error})
    print("MULTI_CUT3_CONTROL",wi,"CORRECT",int(ok),"TARGET_PRESENT",int(target in keys),flush=True)
    if "mem" in locals():del mem
print("[5/6] FINAL INTEGRITY AND PHYSICAL STORAGE METRICS",flush=True)
INTEGRITY={"source_sha_lock":sha==SOURCE_SHA,"weights_unchanged":weight_sha()==WB,"sentinel_unchanged":sentinel()==SB,"zero_trainable":not any(p.requires_grad for p in model.parameters()),"no_remaining_hooks":not any(m._forward_hooks or m._forward_pre_hooks for layer in model.model.layers for m in (layer,layer.self_attn,layer.self_attn.q_proj)),"disk_roundtrip_bitwise":all(x["bitwise_equal"] for x in ROUNDTRIP)}
def percentile(x,p):return round(float(np.percentile(x,p)),4) if x else None
SUMMARY={"test":"621","architecture":"frozen Qwen2.5-7B CUT3 + native FRAM RAM address + real local filesystem CUT3 cartridges","storage_note":"Colab /content filesystem; physical persistent SSD/HDD device type and cold-cache behavior not established","cartridges_real_cut3_on_disk":96,"disk_bytes":disk_total,"logical_cut3_bytes":logical_total,"ram_native_index_bytes":RAM_BYTES,"manifest_bytes":META_BYTES,"ram_to_disk_byte_ratio":round(RAM_BYTES/disk_total,6),"questions":len(RESULTS),"ram_retrieval_r1":sum(x["retrieval_correct"] for x in RESULTS),"disk_single_cartridge_answer_correct":sum(x["single_cartridge_answer_correct"] for x in RESULTS),"unchanged_multicartridge_control_correct":sum(x["answer_correct"] for x in CONTROL),"actual_cartridge_file_reads":len(RESULTS),"total_query_file_bytes_read":read_bytes,"mean_query_file_bytes_read":round(read_bytes/len(RESULTS)),"address_p50_ms":percentile(address_times,50),"disk_load_with_sha_p50_ms":percentile(disk_reads,50),"answer_p50_ms":percentile(answer_times,50),"disk_load_with_sha_p95_ms":percentile(disk_reads,95),"integrity":INTEGRITY,"elapsed_s":round(time.perf_counter()-t_start,3),"limitations":["96 actual CUT3 files; not 1536 actual CUT3 files","Full-scan RAM FRAM; not TEST620 FAISS IVF","Single-cartridge disk-loaded answer path; original five-cartridge CUT3 tested separately","Colab local filesystem does not establish physical HDD/SSD type","Warm filesystem cache may affect timings","Original 12 holdout questions previously used; not independent new questions","No quantum superposition claim"]}
OUT=ROOT/"TEST621_RESULT.json"
with open(OUT,"w",encoding="utf-8") as f:json.dump({"summary":SUMMARY,"manifest":MANIFEST,"roundtrip":ROUNDTRIP,"disk_queries":RESULTS,"multicartridge_control":CONTROL},f,ensure_ascii=False,indent=2)
print("[6/6] PUBLICATION-READY EVIDENCE SUMMARY",flush=True)
print("TEST621_STORAGE",json.dumps({k:SUMMARY[k] for k in ("cartridges_real_cut3_on_disk","disk_bytes","logical_cut3_bytes","ram_native_index_bytes","ram_to_disk_byte_ratio","actual_cartridge_file_reads","mean_query_file_bytes_read")}),flush=True)
print("TEST621_ACCURACY",json.dumps({k:SUMMARY[k] for k in ("questions","ram_retrieval_r1","disk_single_cartridge_answer_correct","unchanged_multicartridge_control_correct")}),flush=True)
print("TEST621_LATENCY",json.dumps({k:SUMMARY[k] for k in ("address_p50_ms","disk_load_with_sha_p50_ms","answer_p50_ms","disk_load_with_sha_p95_ms")}),flush=True)
print("TEST621_INTEGRITY",json.dumps(INTEGRITY),flush=True)
print("TEST621_OUTPUT",str(OUT),flush=True)
print("TEST621_STATUS","COMPLETE" if all(INTEGRITY.values()) else "INTEGRITY_FAIL",flush=True)

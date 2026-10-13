
# ==============================================================================================================
# TEST629 — AKBASCORE MAM · END-TO-END ARCHITECTURE & STORAGE INVENTORY
# TEST628 FOLLOW-UP · LOCKED TEST566 BACKBONE · SINGLE COLAB CELL
# 96 FACTS / 28 LAYERS / 3 STORAGE REPRESENTATIONS / SOURCE DEPENDENCY AUDIT
# NO TRAINING · NO GRADIO · NO FAISS MODIFICATION · NO ENGINE MODIFICATION
# ==============================================================================================================
import os,sys,time,json,hashlib,urllib.request,traceback,gc,platform,inspect,ast,io
from pathlib import Path
import numpy as np,torch
T629=time.perf_counter()
ROOT629=Path("/content/AKBASCORE_TEST629")
ROOT629.mkdir(parents=True,exist_ok=True)
OUT629=ROOT629/"TEST629_ARCHITECTURE_INVENTORY.json"
SOURCE_URL629="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/566.py"
SOURCE_SHA629="0155cda553e5f08ed2bec53611e49898a5ca24e0d270c3a1274e661bc205bdcd"
SOURCE_BLOB629="b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71"
PANEL_LOCK629="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
BOUNDARY629="RESULTS=[];RAW=[];ERRORS=[]"
CUT629=3
N629=96
PROBES629=[0,23,47,71,95]
def log629(*x):print(*x,flush=True)
def sha_bytes629(b):return hashlib.sha256(b).hexdigest()
def sha_file629(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def blob629(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def source629():
 for p in (Path("/content/566.py"),Path("/content/TEST566.py")):
  if p.is_file():
   b=p.read_bytes()
   if sha_bytes629(b)==SOURCE_SHA629 and blob629(b)==SOURCE_BLOB629:return b,"LOCAL_VERIFIED"
 req=urllib.request.Request(SOURCE_URL629,headers={"User-Agent":"AKBASCORE-TEST629"})
 b=urllib.request.urlopen(req,timeout=120).read()
 if sha_bytes629(b)!=SOURCE_SHA629 or blob629(b)!=SOURCE_BLOB629:raise RuntimeError("TEST566_SOURCE_LOCK_FAILURE")
 return b,"GITHUB_VERIFIED"
def save629(d):
 tmp=OUT629.with_suffix(".json.tmp")
 with open(tmp,"w",encoding="utf-8") as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False,default=str)
 os.replace(tmp,OUT629)
def nbytes629(t):return int(t.numel()*t.element_size())
def size629(obj):
 if torch.is_tensor(obj):return nbytes629(obj)
 if isinstance(obj,(tuple,list)):return sum(size629(x) for x in obj)
 if isinstance(obj,dict):return sum(size629(x) for x in obj.values())
 return 0
def cpu629(obj):
 if torch.is_tensor(obj):return obj.detach().to("cpu").contiguous()
 if isinstance(obj,tuple):return tuple(cpu629(x) for x in obj)
 if isinstance(obj,list):return [cpu629(x) for x in obj]
 if isinstance(obj,dict):return {k:cpu629(v) for k,v in obj.items()}
 raise TypeError("UNSUPPORTED_ARTIFACT_TYPE")
def eq629(a,b):
 if torch.is_tensor(a):
  return bool(torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype and
   torch.equal(a.detach().cpu().contiguous().view(torch.uint8),b.detach().cpu().contiguous().view(torch.uint8)))
 if isinstance(a,(tuple,list)):
  return isinstance(b,(tuple,list)) and len(a)==len(b) and all(eq629(x,y) for x,y in zip(a,b))
 if isinstance(a,dict):
  return isinstance(b,dict) and a.keys()==b.keys() and all(eq629(a[k],b[k]) for k in a)
 return a==b
def disk629(name,obj):
 path=ROOT629/name
 t=time.perf_counter()
 torch.save(cpu629(obj),path)
 write_s=time.perf_counter()-t
 digest=sha_file629(path)
 t=time.perf_counter()
 loaded=torch.load(path,map_location="cpu",weights_only=True)
 read_s=time.perf_counter()-t
 ok=eq629(obj,loaded)
 return dict(file=path.name,bytes=path.stat().st_size,sha256=digest,
  sha_verified=sha_file629(path)==digest,bitwise_equal=ok,
  write_seconds=round(write_s,6),read_seconds=round(read_s,6))
def guard629():
 return bool(sentinel()==S0 and not model.training and not any(p.requires_grad for p in model.parameters()))
def validate629(kv,ci):
 n=P+len(CAR[ci]["body"])
 if len(kv)!=28:raise RuntimeError("KV_LAYER_COUNT_FAILURE")
 for li,(k,v) in enumerate(kv):
  if tuple(k.shape)!=(1,KVH,n,HD) or tuple(v.shape)!=(1,KVH,n,HD):
   raise RuntimeError(f"KV_SHAPE_FAILURE_L{li}")
  if k.dtype!=DTYPE or v.dtype!=DTYPE:raise RuntimeError(f"KV_DTYPE_FAILURE_L{li}")
def function_graph629(src):
 tree=ast.parse(src)
 targets=("checkpoint","init","append","answer","joint","independent","consolidated","cache_build","clone","rephase","validate")
 graph={}
 for node in tree.body:
  if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in targets:
   calls=sorted(set(n.func.id for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)))
   graph[node.name]=dict(line=node.lineno,called_functions=calls,
    calls_checkpoint="checkpoint" in calls,calls_init="init" in calls,
    calls_append="append" in calls,calls_answer="answer" in calls)
 return graph
def inventory_case629(ci,probe=False):
 t=time.perf_counter()
 r=dict(id=ci,world=int(CAR[ci]["world"]),type=CAR[ci]["type"],
  body_tokens=len(CAR[ci]["body"]),prefix_tokens=P,
  total_tokens=P+len(CAR[ci]["body"]),status="STARTED")
 stage="CHECKPOINT"
 try:
  with torch.no_grad():
   t0=time.perf_counter()
   cp=checkpoint(ci,CUT629)
   r["checkpoint_seconds"]=round(time.perf_counter()-t0,6)
   kv=cp["kv"];h=cp["h"]
   validate629(kv,ci)
   if tuple(h.shape)!=(1,r["total_tokens"],H):raise RuntimeError("H3_SHAPE_FAILURE")
   lower=kv[:CUT629+1]
   upper=kv[CUT629+1:]
   r["native_kv_bytes"]=size629(kv)
   r["lower_kv_bytes"]=size629(lower)
   r["upper_kv_bytes"]=size629(upper)
   r["h3_bytes"]=size629(h)
   r["h3_only_bytes"]=r["h3_bytes"]
   r["lower_plus_h3_bytes"]=r["lower_kv_bytes"]+r["h3_bytes"]
   r["native_tensors"]=2*len(kv)
   r["lower_tensors"]=2*len(lower)
   r["upper_tensors"]=2*len(upper)
   r["h3_only_vs_native_ratio"]=r["h3_bytes"]/r["native_kv_bytes"]
   r["lower_plus_h3_vs_native_ratio"]=r["lower_plus_h3_bytes"]/r["native_kv_bytes"]
   r["theoretical_h3_only_saving_pct"]=100*(1-r["h3_only_vs_native_ratio"])
   r["theoretical_lower_plus_h3_saving_pct"]=100*(1-r["lower_plus_h3_vs_native_ratio"])
   r["storage_comparison_scope"]="Tensor payload only; excludes serialization metadata, runtime allocator, model weights and tokenizer"
   if probe:
    stage="DISK_ARTIFACTS"
    artifacts={
     "H3_ONLY":dict(h3=h),
     "LOWER_KV_PLUS_H3":dict(lower_kv=lower,h3=h),
     "FULL_NATIVE_KV":dict(kv=kv)}
    r["physical_artifacts"]={}
    for label,obj in artifacts.items():
     d=disk629(f"TEST629_C{ci:03d}_{label}.pt",obj)
     if not d["bitwise_equal"] or not d["sha_verified"]:raise RuntimeError(f"DISK_FAILURE_{label}")
     r["physical_artifacts"][label]=d
    stage="CUT3_RECONSTRUCTION"
    t0=time.perf_counter()
    rebuilt=init(ci,CUT629)
    r["init_seconds"]=round(time.perf_counter()-t0,6)
    validate629(rebuilt,ci)
    r["cut3_native_bitwise_equal"]=eq629(kv,rebuilt)
    r["lower_kv_bitwise_equal"]=eq629(lower,rebuilt[:CUT629+1])
    r["upper_kv_bitwise_equal"]=eq629(upper,rebuilt[CUT629+1:])
    stage="BEHAVIOR"
    q=W[CAR[ci]["world"]][CAR[ci]["type"]]
    gold=CAR[ci]["gold"]
    a=answer(tuple((k.clone(),v.clone()) for k,v in kv),q)
    b=answer(tuple((k.clone(),v.clone()) for k,v in rebuilt),q)
    r["behavior"]=dict(question=q,gold=gold,native_answer=a,cut3_answer=b,
     native_correct=bool(hit(a,gold)),cut3_correct=bool(hit(b,gold)),answer_equal=a==b)
   r["status"]="PASS"
 except Exception as e:
  r["status"]="ERROR";r["stage"]=stage;r["error"]=repr(e);r["traceback"]=traceback.format_exc()
 r["elapsed_seconds"]=round(time.perf_counter()-t,4)
 return r
log629("="*118)
log629("TEST629 — AKBASCORE MAM / END-TO-END ARCHITECTURE & STORAGE INVENTORY")
log629("="*118)
if not torch.cuda.is_available():raise RuntimeError("CUDA_GPU_REQUIRED")
data629,origin629=source629()
text629=data629.decode("utf-8")
if text629.count(BOUNDARY629)!=1:raise RuntimeError("TEST566_BOOTSTRAP_BOUNDARY_MISMATCH")
log629("SOURCE",origin629,"SHA256",sha_bytes629(data629),"GIT_BLOB",blob629(data629))
exec(compile(text629.split(BOUNDARY629,1)[0],"TEST566_LOCKED_BACKBONE.py","exec"),globals())
if (NL,H,QH,KVH,HD)!=(28,3584,28,4,128):raise RuntimeError("MODEL_ARCH_FAILURE")
if len(CAR)!=96 or P!=29 or PANEL_SHA!=PANEL_LOCK629:raise RuntimeError("ORIGINAL_PANEL_FAILURE")
if not guard629():raise RuntimeError("MODEL_GUARD_START_FAILURE")
GRAPH629=function_graph629(text629.split(BOUNDARY629,1)[0])
if not GRAPH629["init"]["calls_checkpoint"]:raise RuntimeError("UNEXPECTED_INIT_IMPLEMENTATION")
if not GRAPH629["append"]["calls_checkpoint"]:raise RuntimeError("UNEXPECTED_APPEND_IMPLEMENTATION")
if not GRAPH629["consolidated"]["calls_init"] or not GRAPH629["consolidated"]["calls_append"]:
 raise RuntimeError("UNEXPECTED_CONSOLIDATION_IMPLEMENTATION")
ARCH629=dict(
 checkpoint="Original source-token prefill produces complete native KV and captured L3 hidden state",
 init="Calls checkpoint(ci,cut), injects captured hidden state, recomputes upper layers and preserves native lower KV",
 append="Calls checkpoint(ci,cut) for appended fact; uses captured body hidden state and lower native KV",
 answer="Uses supplied complete KV and question suffix; does not directly call checkpoint or init",
 independent="Builds each fact independently; rephases and concatenates KV blocks",
 consolidated="Initializes first cartridge, then appends subsequent facts with layer-boundary intervention",
 full_native_kv="28 layers x K/V tensors",
 h3_only="Layer-3 hidden-state tensor only; not established as standalone complete KV cartridge",
 lower_plus_h3="Layers 0-3 KV plus L3 hidden state; reconstruction feasibility is not tested here",
 original_source_dependency="checkpoint requires PIDS + CAR[ci]['body']; init and append call checkpoint",
 physical_storage="Three explicitly distinguished representations are serialized for representative facts")
ENV629=dict(model=MODEL_ID,gpu=torch.cuda.get_device_name(0),
 torch=str(torch.__version__),transformers=str(transformers.__version__),
 python=platform.python_version(),dtype=str(DTYPE),attention="sdpa",
 layers=NL,hidden=H,kv_heads=KVH,head_dim=HD,cut=CUT629,
 source_sha256=SOURCE_SHA629,source_git_blob=SOURCE_BLOB629,
 panel_sha256=PANEL_SHA,source_origin=origin629)
log629("SOURCE_DEPENDENCY","INIT -> CHECKPOINT -> ORIGINAL FACT TOKENS")
log629("SOURCE_DEPENDENCY","APPEND -> CHECKPOINT -> ORIGINAL FACT TOKENS")
log629("STORAGE_ARMS","H3_ONLY / LOWER_KV_PLUS_H3 / FULL_NATIVE_KV")
log629("PREFLIGHT","Cartridge 0")
pre629=inventory_case629(0,True)
if pre629["status"]!="PASS":
 save629(dict(test="629",status="PREFLIGHT_FAILURE",environment=ENV629,preflight=pre629))
 log629("PREFLIGHT_FAILURE",pre629.get("stage"),pre629.get("error"))
 raise RuntimeError("TEST629_PREFLIGHT_FAILURE_REVIEW_JSON")
log629("PREFLIGHT_PASS",
 "NATIVE_BYTES",pre629["native_kv_bytes"],
 "H3_BYTES",pre629["h3_bytes"],
 "LOWER_PLUS_H3_BYTES",pre629["lower_plus_h3_bytes"],
 "CUT3_EXACT",pre629["cut3_native_bitwise_equal"])
ROWS629=[pre629]
for ci in range(1,N629):
 r=inventory_case629(ci,ci in PROBES629)
 ROWS629.append(r)
 if r["status"]=="PASS":
  log629("CASE",f"{ci+1:03d}/{N629}",
   "ID",ci,"TOKENS",r["total_tokens"],
   "FULL",r["native_kv_bytes"],
   "LOWER+H3",r["lower_plus_h3_bytes"],
   "H3",r["h3_bytes"],
   "PROBE",ci in PROBES629)
 else:log629("CASE_ERROR",ci,r.get("stage"),r.get("error"))
 save629(dict(test="629",status="IN_PROGRESS",progress=ci+1,
  environment=ENV629,architecture=ARCH629,function_graph=GRAPH629,cases=ROWS629))
 if r["status"]!="PASS":raise RuntimeError(f"TEST629_CASE_FAILURE_{ci}_REVIEW_JSON")
 if (ci+1)%16==0:
  gc.collect()
  torch.cuda.empty_cache()
VALID629=[r for r in ROWS629 if r["status"]=="PASS"]
PROBED629=[r for r in VALID629 if "physical_artifacts" in r]
def sums629(k):return int(sum(r[k] for r in VALID629))
def avg629(k):return float(np.mean([r[k] for r in VALID629]))
def aggregate_disk629(label):
 ds=[r["physical_artifacts"][label] for r in PROBED629]
 return dict(count=len(ds),total_bytes=sum(x["bytes"] for x in ds),
  mean_bytes=float(np.mean([x["bytes"] for x in ds])) if ds else None,
  bitwise_exact=sum(int(x["bitwise_equal"]) for x in ds),
  sha_verified=sum(int(x["sha_verified"]) for x in ds),
  mean_write_seconds=float(np.mean([x["write_seconds"] for x in ds])) if ds else None,
  mean_read_seconds=float(np.mean([x["read_seconds"] for x in ds])) if ds else None)
TOTAL629=dict(
 facts_requested=N629,facts_completed=len(VALID629),
 original_token_total=sums629("total_tokens"),
 body_token_total=sums629("body_tokens"),
 native_kv_payload_bytes=sums629("native_kv_bytes"),
 lower_kv_payload_bytes=sums629("lower_kv_bytes"),
 upper_kv_payload_bytes=sums629("upper_kv_bytes"),
 h3_only_payload_bytes=sums629("h3_bytes"),
 lower_plus_h3_payload_bytes=sums629("lower_plus_h3_bytes"),
 h3_vs_full_payload_ratio=sums629("h3_bytes")/sums629("native_kv_bytes"),
 lower_plus_h3_vs_full_payload_ratio=sums629("lower_plus_h3_bytes")/sums629("native_kv_bytes"),
 mean_native_kv_bytes=avg629("native_kv_bytes"),
 mean_h3_bytes=avg629("h3_bytes"),
 mean_lower_plus_h3_bytes=avg629("lower_plus_h3_bytes"),
 mean_checkpoint_seconds=avg629("checkpoint_seconds"),
 physical_probe_count=len(PROBED629),
 cut3_native_exact_probes=sum(int(r["cut3_native_bitwise_equal"]) for r in PROBED629),
 lower_exact_probes=sum(int(r["lower_kv_bitwise_equal"]) for r in PROBED629),
 upper_exact_probes=sum(int(r["upper_kv_bitwise_equal"]) for r in PROBED629),
 native_correct_probes=sum(int(r["behavior"]["native_correct"]) for r in PROBED629),
 cut3_correct_probes=sum(int(r["behavior"]["cut3_correct"]) for r in PROBED629),
 answer_equal_probes=sum(int(r["behavior"]["answer_equal"]) for r in PROBED629),
 physical_h3_only=aggregate_disk629("H3_ONLY"),
 physical_lower_plus_h3=aggregate_disk629("LOWER_KV_PLUS_H3"),
 physical_full_native=aggregate_disk629("FULL_NATIVE_KV"))
MODEL_OK629=guard629()
COMPLETE629=len(VALID629)==96 and len(PROBED629)==len(PROBES629) and MODEL_OK629
REPORT629=dict(test="629",title="AKBASCORE MAM — End-to-End Architecture and Storage Inventory",
 status="COMPLETE" if COMPLETE629 else "INCOMPLETE",
 environment=ENV629,architecture=ARCH629,function_graph=GRAPH629,
 totals=TOTAL629,cases=ROWS629,model_unchanged=MODEL_OK629,
 interpretation={
  "verified":"The current locked TEST566 implementation uses native source prefill for checkpoint generation and for init/append inputs.",
  "measured":"Native KV, L3 hidden-state and lower-KV-plus-H3 tensor payloads and representative physical files.",
  "not_established":"Source-free regeneration from H3-only or lower-KV-plus-H3 after removing original source tokens.",
  "next_test":"TEST630 must explicitly remove source access, reload artifacts in a clean runtime and attempt reconstruction without checkpoint."},
 limitations=[
  "Storage comparisons distinguish hypothetical H3-only and lower-plus-H3 artifacts from the operational full KV.",
  "Only five representative cartridges receive physical serialization and behavioral probes.",
  "All 96 facts receive checkpoint-derived tensor payload inventory.",
  "Physical file sizes include PyTorch serialization overhead; payload sizes do not.",
  "Write/read times are observational, not warmed or statistically controlled benchmarks.",
  "No cold process restart is performed.",
  "No comparison against external implementations is claimed.",
  "No evidence of H3-only source-free full-cache recovery is claimed.",
  "Sampled-weight sentinel is not a full-model cryptographic hash."
 ],
 elapsed_seconds=round(time.perf_counter()-T629,3))
save629(REPORT629)
log629("="*118)
log629("TEST629_STATUS",REPORT629["status"])
log629("MODEL_UNCHANGED",MODEL_OK629)
log629("SOURCE_DEPENDENCY","CHECKPOINT_REQUIRED_BY_INIT_AND_APPEND")
log629("FACTS",len(VALID629),"/96")
log629("PHYSICAL_PROBES",len(PROBED629),"/",len(PROBES629))
log629("FULL_KV_TOTAL_BYTES",TOTAL629["native_kv_payload_bytes"])
log629("LOWER_PLUS_H3_TOTAL_BYTES",TOTAL629["lower_plus_h3_payload_bytes"])
log629("H3_ONLY_TOTAL_BYTES",TOTAL629["h3_only_payload_bytes"])
log629("LOWER_PLUS_H3_TO_FULL_RATIO",round(TOTAL629["lower_plus_h3_vs_full_payload_ratio"],6))
log629("H3_TO_FULL_RATIO",round(TOTAL629["h3_vs_full_payload_ratio"],6))
log629("CUT3_NATIVE_EXACT_PROBES",TOTAL629["cut3_native_exact_probes"],"/",len(PROBED629))
log629("BEHAVIOR_NATIVE_CORRECT",TOTAL629["native_correct_probes"],"/",len(PROBED629))
log629("BEHAVIOR_CUT3_CORRECT",TOTAL629["cut3_correct_probes"],"/",len(PROBED629))
log629("BEHAVIOR_ANSWER_EQUAL",TOTAL629["answer_equal_probes"],"/",len(PROBED629))
for label in ("physical_h3_only","physical_lower_plus_h3","physical_full_native"):
 x=TOTAL629[label]
 log629("DISK",label,"MEAN_BYTES",round(x["mean_bytes"],1),
  "BITWISE",x["bitwise_exact"],"/",x["count"],
  "SHA",x["sha_verified"],"/",x["count"])
log629("NEXT","TEST630 — SOURCE-FREE COLD RELOAD AND RECONSTRUCTION")
log629("OUTPUT",str(OUT629))
log629("="*118)
if not COMPLETE629:raise RuntimeError("TEST629_INTEGRITY_FAILURE")


# =====================================================================================================================
# TEST630 — AKBASCORE MAM · SOURCE-ISOLATED PHYSICAL CARTRIDGE RELOAD
# TEST629 FOLLOW-UP · FROZEN TEST566 BACKBONE · SINGLE COLAB CELL
# FULL KV vs LOWER KV + H3 vs H3 ONLY
# PHYSICAL SAVE / RELOAD · SOURCE ACCESS BLOCK · BITWISE AUDIT · BEHAVIOR · DEMO-READY JSON + CSV
# NO TRAINING · NO ENGINE CHANGE · NO SOURCE REPLAY DURING RELOAD
# =====================================================================================================================
import os,sys,time,json,csv,hashlib,urllib.request,traceback,platform,gc,random
from pathlib import Path
import torch,numpy as np
T630=time.perf_counter()
ROOT630=Path("/content/AKBASCORE_TEST630")
ART630=ROOT630/"artifacts"
ART630.mkdir(parents=True,exist_ok=True)
JSON630=ROOT630/"TEST630_SOURCE_ISOLATED_RELOAD.json"
CSV630=ROOT630/"TEST630_DEMO_COMPARISON.csv"
URL630="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/566.py"
SHA630="0155cda553e5f08ed2bec53611e49898a5ca24e0d270c3a1274e661bc205bdcd"
BLOB630="b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71"
PANEL630="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
BOUND630="RESULTS=[];RAW=[];ERRORS=[]"
CUT630=3
IDS630=[0,23,47,71,95]
ARMS630=("FULL_KV","LOWER_KV_H3","H3_ONLY")
def say630(*a):print(*a,flush=True)
def sha630(b):return hashlib.sha256(b).hexdigest()
def blob630(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def fetch630():
 for p in (Path("/content/566.py"),Path("/content/TEST566.py")):
  if p.is_file():
   b=p.read_bytes()
   if sha630(b)==SHA630 and blob630(b)==BLOB630:return b,"LOCAL_VERIFIED"
 req=urllib.request.Request(URL630,headers={"User-Agent":"AKBASCORE-TEST630"})
 b=urllib.request.urlopen(req,timeout=120).read()
 if sha630(b)!=SHA630 or blob630(b)!=BLOB630:raise RuntimeError("SOURCE_LOCK_FAILURE")
 return b,"GITHUB_VERIFIED"
def payload630(x):
 if torch.is_tensor(x):return int(x.numel()*x.element_size())
 if isinstance(x,(tuple,list)):return sum(payload630(z) for z in x)
 if isinstance(x,dict):return sum(payload630(z) for z in x.values())
 return 0
def cpu630(x):
 if torch.is_tensor(x):return x.detach().cpu().contiguous()
 if isinstance(x,tuple):return tuple(cpu630(z) for z in x)
 if isinstance(x,list):return [cpu630(z) for z in x]
 if isinstance(x,dict):return {k:cpu630(v) for k,v in x.items()}
 raise TypeError("INVALID_ARTIFACT")
def gpu630(x):
 if torch.is_tensor(x):return x.to(device=DEVICE,dtype=DTYPE).contiguous()
 if isinstance(x,tuple):return tuple(gpu630(z) for z in x)
 if isinstance(x,list):return [gpu630(z) for z in x]
 if isinstance(x,dict):return {k:gpu630(v) for k,v in x.items()}
 raise TypeError("INVALID_ARTIFACT")
def exact630(a,b):
 if torch.is_tensor(a):
  return bool(torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype and
   torch.equal(a.detach().cpu().contiguous().view(torch.uint8),b.detach().cpu().contiguous().view(torch.uint8)))
 if isinstance(a,(tuple,list)):
  return isinstance(b,(tuple,list)) and len(a)==len(b) and all(exact630(x,y) for x,y in zip(a,b))
 if isinstance(a,dict):
  return isinstance(b,dict) and a.keys()==b.keys() and all(exact630(a[k],b[k]) for k in a)
 return a==b
def hashfile630(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()
def persist630(ci,arm,obj):
 path=ART630/f"C{ci:03d}_{arm}.pt"
 t=time.perf_counter()
 torch.save(cpu630(obj),path)
 write=time.perf_counter()-t
 return dict(path=str(path),bytes=path.stat().st_size,sha256=hashfile630(path),
  payload_bytes=payload630(obj),write_seconds=round(write,6))
def load630(meta):
 p=Path(meta["path"])
 if hashfile630(p)!=meta["sha256"]:raise RuntimeError("ARTIFACT_SHA_FAILURE")
 t=time.perf_counter()
 obj=torch.load(p,map_location="cpu",weights_only=True)
 sec=time.perf_counter()-t
 if payload630(obj)!=meta["payload_bytes"]:raise RuntimeError("ARTIFACT_PAYLOAD_FAILURE")
 return obj,sec
def cachevalid630(kv,n):
 if not isinstance(kv,(tuple,list)) or len(kv)!=NL:return False
 for k,v in kv:
  if tuple(k.shape)!=(1,KVH,n,HD) or tuple(v.shape)!=(1,KVH,n,HD):return False
  if k.dtype!=DTYPE or v.dtype!=DTYPE:return False
  if not bool(torch.isfinite(k).all()) or not bool(torch.isfinite(v).all()):return False
 return True
def weightguard630():return bool(sentinel()==S0 and not model.training and not any(p.requires_grad for p in model.parameters()))
def save_report630(obj):
 p=JSON630.with_suffix(".tmp")
 with open(p,"w",encoding="utf-8") as f:json.dump(obj,f,ensure_ascii=False,indent=2,default=str,allow_nan=False)
 os.replace(p,JSON630)
def rebuild_upper630(h3,lower,n):
 # Reuse the original TEST566 L3 injection mechanism, without checkpoint() or source tokens.
 # Only the already serialized H3 and lower-layer KV are supplied.
 if tuple(h3.shape)!=(1,n,H):raise RuntimeError("H3_SHAPE_FAILURE")
 if len(lower)!=CUT630+1:raise RuntimeError("LOWER_LAYER_COUNT_FAILURE")
 state={"calls":0}
 def inject630(module,args,out):
  state["calls"]+=1
  return replace(out,h3)
 hook=model.model.layers[CUT630].register_forward_hook(inject630)
 try:
  with torch.no_grad():
   pos=torch.arange(n,device=DEVICE)
   out=model(inputs_embeds=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE),
    attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),
    position_ids=pos.unsqueeze(0),cache_position=pos,
    use_cache=True,return_dict=True)
   upper=clone(out.past_key_values)
  if state["calls"]!=1:raise RuntimeError("L3_INJECTION_CALL_COUNT_FAILURE")
  return tuple(lower)+tuple(upper[CUT630+1:])
 finally:hook.remove()
def arm_result630(ci,arm,meta,n,reference,q,gold):
 row=dict(cartridge_id=ci,arm=arm,status="STARTED",source_isolated=True,
  source_replay_calls=0,physical_bytes=meta["bytes"],
  payload_bytes=meta["payload_bytes"],artifact_sha256=meta["sha256"],
  complete_kv=False,bitwise_equal=None,answer=None,correct=None,
  answer_equal_native=None,reconstruction_seconds=None,read_seconds=None)
 stage="DISK_READ"
 try:
  loaded,read_s=load630(meta)
  row["read_seconds"]=round(read_s,6)
  stage="GPU_TRANSFER"
  t=time.perf_counter()
  obj=gpu630(loaded)
  if arm=="FULL_KV":
   memory=obj["kv"]
   row["reconstruction_method"]="Direct physical full-KV reload"
  elif arm=="LOWER_KV_H3":
   memory=rebuild_upper630(obj["h3"],obj["lower_kv"],n)
   row["reconstruction_method"]="Saved lower KV plus saved L3 hidden state; recompute upper layers"
  elif arm=="H3_ONLY":
   # Upper KV can be calculated from H3, but the original lower-layer KV
   # is absent. Never substitute zeros or hidden native tensors.
   row["reconstruction_method"]="H3-only; original lower KV unavailable"
   row["status"]="INCOMPLETE_REPRESENTATION"
   row["missing_layers"]=list(range(CUT630+1))
   row["reconstruction_seconds"]=round(time.perf_counter()-t,6)
   return row
  else:raise RuntimeError("UNKNOWN_ARM")
  row["reconstruction_seconds"]=round(time.perf_counter()-t,6)
  stage="KV_VALIDATE"
  if not cachevalid630(memory,n):raise RuntimeError("INVALID_RECONSTRUCTED_KV")
  row["complete_kv"]=True
  row["bitwise_equal"]=exact630(reference,memory)
  stage="BEHAVIOR"
  t=time.perf_counter()
  row["answer"]=answer(tuple((k.clone(),v.clone()) for k,v in memory),q)
  row["answer_seconds"]=round(time.perf_counter()-t,6)
  row["correct"]=bool(hit(row["answer"],gold))
  row["status"]="PASS"
 except Exception as e:
  row["status"]="ERROR";row["stage"]=stage;row["error"]=repr(e)
  row["traceback"]=traceback.format_exc()
 return row
say630("="*120)
say630("TEST630 — AKBASCORE MAM · SOURCE-ISOLATED PHYSICAL CARTRIDGE RELOAD")
say630("="*120)
if not torch.cuda.is_available():raise RuntimeError("CUDA_REQUIRED")
src630,origin630=fetch630()
if src630.decode("utf-8").count(BOUND630)!=1:raise RuntimeError("BOOTSTRAP_BOUNDARY_FAILURE")
say630("SOURCE",origin630,"SHA256",sha630(src630),"BLOB",blob630(src630))
exec(compile(src630.decode("utf-8").split(BOUND630,1)[0],
 "TEST566_FROZEN_BACKBONE.py","exec"),globals())
if len(CAR)!=96 or P!=29 or PANEL_SHA!=PANEL630 or (NL,H,KVH,HD)!=(28,3584,4,128):
 raise RuntimeError("BACKBONE_LOCK_FAILURE")
if not weightguard630():raise RuntimeError("WEIGHT_GUARD_FAILURE")
ENV630=dict(model=MODEL_ID,gpu=torch.cuda.get_device_name(0),
 torch=str(torch.__version__),transformers=str(transformers.__version__),
 python=platform.python_version(),dtype=str(DTYPE),cut=CUT630,
 source_sha256=SHA630,source_git_blob=BLOB630,panel_sha256=PANEL_SHA,
 source_origin=origin630)
REPORT630=dict(test="630",status="IN_PROGRESS",environment=ENV630,
 methodology="Two-phase physical artifact save and source-isolated reload in the same Python process",
 process_restart=False,model_weights_unchanged=None,
 artifacts=[],cases=[],comparison=[],
 limitations=[
  "This test is not a new-process cold start.",
  "H3-only cannot provide the original lower-layer KV without additional information or a different mechanism.",
  "Source tokens are isolated during reload, but the original model remains loaded.",
  "Source is restored after each test case to preserve the frozen experimental backbone.",
  "Runtime measurements are exploratory, not controlled cross-system benchmarks.",
  "No external competitor implementation is benchmarked."
 ])
save_report630(REPORT630)
for ci in IDS630:
 say630("="*105)
 say630("CARTRIDGE",ci,"PHASE_A","SOURCE-AVAILABLE FOR ONE-TIME FORGING")
 if not weightguard630():raise RuntimeError("MODEL_CHANGED_BEFORE_CASE")
 stage630="FORGE"
 original_body630=CAR[ci]["body"]
 try:
  with torch.no_grad():
   cp630=checkpoint(ci,CUT630)
   native630=cp630["kv"]
   h630=cp630["h"]
   n630=P+len(original_body630)
   if not cachevalid630(native630,n630):raise RuntimeError("NATIVE_KV_INVALID")
   artifact_objects630={
    "FULL_KV":dict(kv=native630),
    "LOWER_KV_H3":dict(lower_kv=native630[:CUT630+1],h3=h630),
    "H3_ONLY":dict(h3=h630)}
   metas630={}
   for arm630,obj630 in artifact_objects630.items():
    metas630[arm630]=persist630(ci,arm630,obj630)
    REPORT630["artifacts"].append(dict(cartridge_id=ci,arm=arm630,**metas630[arm630]))
   q630=W[CAR[ci]["world"]][CAR[ci]["type"]]
   gold630=CAR[ci]["gold"]
   baseline630=answer(tuple((k.clone(),v.clone()) for k,v in native630),q630)
   case630=dict(cartridge_id=ci,total_tokens=n630,question=q630,gold=gold630,
    native_answer=baseline630,native_correct=bool(hit(baseline630,gold630)),
    source_isolation_checked=False,arms=[])
   # All artifacts are now on disk. Drop in-memory artifact references.
   del cp630,artifact_objects630,obj630,h630
   gc.collect()
   say630("PHASE_B","BLOCKING ORIGINAL SOURCE TOKEN ACCESS")
   stage630="SOURCE_ISOLATION"
   CAR[ci]["body"]=None
   if CAR[ci]["body"] is not None:raise RuntimeError("SOURCE_ISOLATION_FAILURE")
   case630["source_isolation_checked"]=True
   for arm630 in ARMS630:
    r630=arm_result630(ci,arm630,metas630[arm630],n630,native630,q630,gold630)
    if r630["answer"] is not None:r630["answer_equal_native"]=r630["answer"]==baseline630
    case630["arms"].append(r630)
    REPORT630["comparison"].append(dict(
     cartridge_id=ci,arm=arm630,status=r630["status"],
     physical_bytes=r630["physical_bytes"],payload_bytes=r630["payload_bytes"],
     complete_kv=r630["complete_kv"],bitwise_equal=r630["bitwise_equal"],
     correct=r630["correct"],answer_equal_native=r630["answer_equal_native"],
     read_seconds=r630["read_seconds"],
     reconstruction_seconds=r630["reconstruction_seconds"]))
    say630("ARM",arm630,"STATUS",r630["status"],
     "BITWISE",r630["bitwise_equal"],"CORRECT",r630["correct"],
     "BYTES",r630["physical_bytes"])
    if r630["status"]=="ERROR":raise RuntimeError(f"ARM_ERROR_{arm630}: {r630.get('error')}")
   REPORT630["cases"].append(case630)
 finally:
  CAR[ci]["body"]=original_body630
  if CAR[ci]["body"]!=original_body630:raise RuntimeError("SOURCE_RESTORE_FAILURE")
 save_report630(REPORT630)
 del native630
 gc.collect()
 torch.cuda.empty_cache()
if not weightguard630():raise RuntimeError("MODEL_CHANGED_AFTER_TEST")
rows630=REPORT630["comparison"]
def count630(arm,field,value=True):
 return sum(int(r[field] is value) for r in rows630 if r["arm"]==arm)
def mean630(arm,field):
 v=[r[field] for r in rows630 if r["arm"]==arm and r[field] is not None]
 return float(np.mean(v)) if v else None
SUMMARY630={}
for arm630 in ARMS630:
 rr630=[r for r in rows630 if r["arm"]==arm630]
 SUMMARY630[arm630]=dict(
  tested=len(rr630),complete_kv=count630(arm630,"complete_kv"),
  bitwise_equal=count630(arm630,"bitwise_equal"),
  correct=count630(arm630,"correct"),
  answer_equal_native=count630(arm630,"answer_equal_native"),
  total_physical_bytes=sum(r["physical_bytes"] for r in rr630),
  mean_physical_bytes=mean630(arm630,"physical_bytes"),
  mean_read_seconds=mean630(arm630,"read_seconds"),
  mean_reconstruction_seconds=mean630(arm630,"reconstruction_seconds"),
  statuses={s:sum(r["status"]==s for r in rr630) for s in sorted(set(r["status"] for r in rr630))})
REPORT630["summary"]=SUMMARY630
REPORT630["model_weights_unchanged"]=weightguard630()
REPORT630["status"]="COMPLETE" if (
 len(REPORT630["cases"])==len(IDS630) and
 len(rows630)==len(IDS630)*len(ARMS630) and
 SUMMARY630["FULL_KV"]["complete_kv"]==len(IDS630) and
 SUMMARY630["LOWER_KV_H3"]["complete_kv"]==len(IDS630) and
 SUMMARY630["H3_ONLY"]["complete_kv"]==0 and
 all(c["source_isolation_checked"] for c in REPORT630["cases"]) and
 REPORT630["model_weights_unchanged"]) else "INCOMPLETE"
REPORT630["elapsed_seconds"]=round(time.perf_counter()-T630,3)
save_report630(REPORT630)
with open(CSV630,"w",newline="",encoding="utf-8") as f:
 fields630=["cartridge_id","arm","status","physical_bytes","payload_bytes",
  "complete_kv","bitwise_equal","correct","answer_equal_native",
  "read_seconds","reconstruction_seconds"]
 w630=csv.DictWriter(f,fieldnames=fields630)
 w630.writeheader()
 w630.writerows(rows630)
say630("="*120)
say630("TEST630_STATUS",REPORT630["status"])
say630("SOURCE_ISOLATED_CASES",sum(c["source_isolation_checked"] for c in REPORT630["cases"]),"/",len(IDS630))
say630("MODEL_UNCHANGED",REPORT630["model_weights_unchanged"])
for arm630 in ARMS630:
 s630=SUMMARY630[arm630]
 say630("RESULT",arm630,
  "COMPLETE_KV",f'{s630["complete_kv"]}/{s630["tested"]}',
  "BITWISE",f'{s630["bitwise_equal"]}/{s630["tested"]}',
  "CORRECT",f'{s630["correct"]}/{s630["tested"]}',
  "MEAN_DISK_BYTES",round(s630["mean_physical_bytes"],1))
say630("IMPORTANT","H3_ONLY_INCOMPLETE is an expected structural finding, not an invented reconstruction success.")
say630("CAUTION","Same-process source isolation; independent process cold start remains untested.")
say630("JSON_OUTPUT",JSON630)
say630("CSV_OUTPUT",CSV630)
say630("TOTAL_SECONDS",REPORT630["elapsed_seconds"])
say630("="*120)
if REPORT630["status"]!="COMPLETE":raise RuntimeError("TEST630_INTEGRITY_FAILURE")

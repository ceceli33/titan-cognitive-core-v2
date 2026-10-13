
# ==============================================================================================================
# TEST628 — AKBASCORE MAM · NATIVE KV vs CUT3 · BITWISE MECHANISM AUDIT · FIXED
# STANDALONE COLAB SINGLE CELL / NO UI / NO GRADIO / NO TRAINING
# LOCKED TEST566 checkpoint(), init(), answer() UNCHANGED
# 96 original facts / 56 K-V tensors each / 24 behavioral controls
# Native-repeat + H3-repeat + Native-CUT3 + CPU/GPU-safe disk round-trip
# Preflight first; fail-fast; incremental JSON; model and source integrity
# ==============================================================================================================
import os,sys,time,json,hashlib,urllib.request,traceback,gc,platform,math
from pathlib import Path
import numpy as np,torch
START628=time.perf_counter()
ROOT628=Path("/content/AKBASCORE_TEST628")
ROOT628.mkdir(parents=True,exist_ok=True)
OUT628=ROOT628/"TEST628_NATIVE_CUT3_BITWISE.json"
URL628="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/566.py"
GIT_BLOB628="b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71"
FILE_SHA628="0155cda553e5f08ed2bec53611e49898a5ca24e0d270c3a1274e661bc205bdcd"
PANEL_SHA628="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
BOUNDARY628="RESULTS=[];RAW=[];ERRORS=[]"
CUT628=3
N628=96
def log628(*a):print(*a,flush=True)
def sha628(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def blob628(data):
 return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
def verify628(data):
 return hashlib.sha256(data).hexdigest()==FILE_SHA628 and blob628(data)==GIT_BLOB628
def fetch628():
 for path in (Path("/content/566.py"),Path("/content/TEST566.py"),Path("/content/TEST566_CAUSAL_CONSOLIDATION.py")):
  if path.is_file():
   data=path.read_bytes()
   if verify628(data):return data,"VERIFIED_LOCAL:"+str(path)
 req=urllib.request.Request(URL628,headers={"User-Agent":"AKBASCORE-TEST628"})
 data=urllib.request.urlopen(req,timeout=120).read()
 if not verify628(data):raise RuntimeError("SOURCE_LOCK_FAILURE: TEST566 SHA256 or Git blob mismatch")
 return data,"VERIFIED_GITHUB"
def save628(report):
 tmp=OUT628.with_suffix(".json.tmp")
 with open(tmp,"w",encoding="utf-8") as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
 os.replace(tmp,OUT628)
def tensor_cpu628(t):
 return t.detach().contiguous().to(device="cpu")
def compare628(a,b):
 if not torch.is_tensor(a) or not torch.is_tensor(b):raise TypeError("NON_TENSOR_COMPARISON")
 if a.shape!=b.shape or a.dtype!=b.dtype:
  return dict(comparable=False,shape_a=list(a.shape),shape_b=list(b.shape),dtype_a=str(a.dtype),dtype_b=str(b.dtype))
 ac=tensor_cpu628(a);bc=tensor_cpu628(b)
 if ac.numel()==0:raise RuntimeError("EMPTY_TENSOR")
 x=ac.view(torch.uint8).numpy().reshape(-1)
 y=bc.view(torch.uint8).numpy().reshape(-1)
 xor=np.bitwise_xor(x,y)
 badbytes=int(np.count_nonzero(xor))
 badbits=int(np.unpackbits(xor).sum()) if badbytes else 0
 firstbyte=int(np.flatnonzero(xor)[0]) if badbytes else None
 af=ac.float();bf=bc.float()
 finite=bool(torch.isfinite(af).all().item() and torch.isfinite(bf).all().item())
 if finite:
  d=af-bf
  maxabs=float(d.abs().max().item())
  meanabs=float(d.abs().mean().item())
  norm_a=float(torch.linalg.vector_norm(af).item())
  norm_d=float(torch.linalg.vector_norm(d).item())
  relative=norm_d/(norm_a+1e-30)
  close3=bool(torch.allclose(af,bf,rtol=1e-3,atol=1e-3))
  close2=bool(torch.allclose(af,bf,rtol=1e-2,atol=1e-2))
 else:
  maxabs=meanabs=relative=None
  close3=close2=False
 total=int(x.size*8)
 return dict(comparable=True,shape=list(ac.shape),dtype=str(ac.dtype),finite=finite,
  bitwise_equal=badbits==0,total_bits=total,mismatched_bits=badbits,
  mismatched_bytes=badbytes,first_mismatched_byte=firstbyte,
  bit_agreement=1-badbits/total,max_abs=maxabs,mean_abs=meanabs,
  relative_l2=relative,allclose_1e3=close3,allclose_1e2=close2)
def audit628(a,b):
 if len(a)!=28 or len(b)!=28:raise RuntimeError("EXPECTED_28_LAYERS")
 rows=[]
 for li,((ak,av),(bk,bv)) in enumerate(zip(a,b)):
  for name,x,y in (("K",ak,bk),("V",av,bv)):
   rows.append(dict(layer=li,tensor=name,**compare628(x,y)))
 return rows
def summary628(rows):
 if not rows:return dict(comparable=False,tensors=0)
 if not all(r.get("comparable",False) for r in rows):
  return dict(comparable=False,tensors=len(rows),incompatible=[dict(layer=r.get("layer"),tensor=r.get("tensor")) for r in rows if not r.get("comparable",False)])
 bits=sum(r["total_bits"] for r in rows)
 bad=sum(r["mismatched_bits"] for r in rows)
 rel=[r["relative_l2"] for r in rows if r["relative_l2"] is not None]
 absvals=[r["max_abs"] for r in rows if r["max_abs"] is not None]
 return dict(comparable=True,tensors=len(rows),
  exact_tensors=sum(int(r["bitwise_equal"]) for r in rows),
  bitwise_equal=bad==0,total_bits=bits,mismatched_bits=bad,
  bit_agreement=1-bad/bits,all_finite=all(r["finite"] for r in rows),
  max_abs=max(absvals) if absvals else None,
  mean_relative_l2=float(np.mean(rel)) if rel else None,
  max_relative_l2=max(rel) if rel else None,
  first_mismatch=next((dict(layer=r["layer"],tensor=r["tensor"],first_byte=r["first_mismatched_byte"]) for r in rows if not r["bitwise_equal"]),None))
def host628(mem):
 return tuple((k.detach().cpu().contiguous(),v.detach().cpu().contiguous()) for k,v in mem)
def disk_load628(path):
 obj=torch.load(path,map_location="cpu",weights_only=True)
 if not isinstance(obj,(tuple,list)) or len(obj)!=28:raise RuntimeError("INVALID_DISK_CACHE")
 if any(not isinstance(p,(tuple,list)) or len(p)!=2 or not all(torch.is_tensor(t) for t in p) for p in obj):
  raise RuntimeError("INVALID_DISK_TENSOR_STRUCTURE")
 return tuple((k,v) for k,v in obj)
def layer_summary628(rows):
 return [dict(layer=i,**summary628([r for r in rows if r["layer"]==i])) for i in range(28)]
def guard628():
 return bool(sentinel()==S0 and not any(p.requires_grad for p in model.parameters()) and not model.training)
def question628(ci):
 return W[CAR[ci]["world"]][CAR[ci]["type"]]
def validate_cache628(mem,ci):
 if len(mem)!=28:raise RuntimeError("INVALID_CACHE_LAYER_COUNT")
 n=P+len(CAR[ci]["body"])
 for li,(k,v) in enumerate(mem):
  expected=(1,KVH,n,HD)
  if tuple(k.shape)!=expected or tuple(v.shape)!=expected:raise RuntimeError(f"CACHE_SHAPE_FAILURE_L{li}")
  if k.dtype!=DTYPE or v.dtype!=DTYPE:raise RuntimeError(f"CACHE_DTYPE_FAILURE_L{li}")
  if not bool(torch.isfinite(k).all().item() and torch.isfinite(v).all().item()):
   raise RuntimeError(f"CACHE_NONFINITE_L{li}")
def run628(ci,behavior=False):
 t=time.perf_counter()
 row=dict(id=ci,status="STARTED",source_tokens=P+len(CAR[ci]["body"]))
 stage="START"
 try:
  with torch.no_grad():
   stage="NATIVE_FIRST"
   cp1=checkpoint(ci,CUT628)
   native1=cp1["kv"]
   validate_cache628(native1,ci)
   stage="NATIVE_REPEAT"
   cp2=checkpoint(ci,CUT628)
   native2=cp2["kv"]
   validate_cache628(native2,ci)
   stage="NATIVE_REPEAT_AUDIT"
   rep=audit628(native1,native2)
   row["native_repeat"]=summary628(rep)
   row["h3_repeat"]=compare628(cp1["h"],cp2["h"])
   if not row["native_repeat"].get("bitwise_equal",False) or not row["h3_repeat"].get("bitwise_equal",False):
    row["native_repeat_warning"]=True
   stage="CUT3_BUILD"
   cut3=init(ci,CUT628)
   validate_cache628(cut3,ci)
   stage="NATIVE_CUT3_AUDIT"
   diff=audit628(native1,cut3)
   row["native_cut3"]=summary628(diff)
   row["lower_0_3"]=summary628([r for r in diff if r["layer"]<=3])
   row["upper_4_27"]=summary628([r for r in diff if r["layer"]>=4])
   row["layers"]=layer_summary628(diff)
   row["tensor_details"]=diff
   stage="DISK_ROUNDTRIP"
   path=ROOT628/f"audit_cut3_{ci:05d}.pt"
   torch.save(host628(cut3),path)
   disk_sha=sha628(path)
   disk=disk_load628(path)
   validate_cache628(disk,ci)
   disk_rows=audit628(cut3,disk)
   row["disk_roundtrip"]=summary628(disk_rows)
   row["disk_sha256"]=disk_sha
   row["disk_sha_verified"]=sha628(path)==disk_sha
   row["disk_bytes"]=path.stat().st_size
   if not row["disk_roundtrip"].get("bitwise_equal",False) or not row["disk_sha_verified"]:
    raise RuntimeError("DISK_ROUNDTRIP_INTEGRITY_FAILURE")
   if behavior:
    stage="BEHAVIOR_NATIVE"
    q=question628(ci)
    gold=CAR[ci]["gold"]
    # answer() internally constructs and mutates a DynamicCache.
    # Independent cloned inputs prevent contamination between the two arms.
    na=str(answer(tuple((k.clone(),v.clone()) for k,v in native1),q))
    stage="BEHAVIOR_CUT3"
    ca=str(answer(tuple((k.clone(),v.clone()) for k,v in cut3),q))
    row["behavior"]=dict(question=q,gold=gold,
     native_answer=na,cut3_answer=ca,
     native_correct=bool(hit(na,gold)),
     cut3_correct=bool(hit(ca,gold)),
     exact_answer_match=na==ca)
   row["status"]="PASS"
 except Exception as e:
  row["status"]="ERROR"
  row["stage"]=stage
  row["error"]=repr(e)
  row["traceback"]=traceback.format_exc()
 row["seconds"]=round(time.perf_counter()-t,3)
 return row
log628("="*112)
log628("TEST628 — NATIVE KV vs CUT3 / INDEPENDENT BITWISE AUDIT")
log628("="*112)
if not torch.cuda.is_available():raise RuntimeError("NVIDIA_CUDA_GPU_REQUIRED")
data628,origin628=fetch628()
text628=data628.decode("utf-8")
if text628.count(BOUNDARY628)!=1:raise RuntimeError("TEST566_BOOTSTRAP_BOUNDARY_MISMATCH")
log628("SOURCE_ORIGIN",origin628)
log628("SOURCE_GIT_BLOB_SHA1",blob628(data628))
log628("SOURCE_FILE_SHA256",hashlib.sha256(data628).hexdigest())
log628("SOURCE_BYTES",len(data628))
log628("BOOTSTRAP","Frozen TEST566 backbone, original experiment excluded")
exec(compile(text628.split(BOUNDARY628,1)[0],"TEST566_LOCKED_BACKBONE.py","exec"),globals())
if (NL,H,QH,KVH,HD)!=(28,3584,28,4,128):raise RuntimeError("MODEL_ARCH_MISMATCH")
if len(CAR)!=96 or P!=29 or PANEL_SHA!=PANEL_SHA628:raise RuntimeError("PANEL_OR_TOKEN_LOCK_MISMATCH")
if not guard628():raise RuntimeError("MODEL_GUARD_START_FAILURE")
if next(model.parameters()).dtype!=torch.bfloat16:raise RuntimeError("MODEL_DTYPE_MISMATCH")
BEHAVIOR_IDS628=sorted(set(int(z["target"]) for z in PANEL))
if len(BEHAVIOR_IDS628)!=24:raise RuntimeError("BEHAVIOR_PANEL_MISMATCH")
ENV628=dict(model=MODEL_ID,gpu=torch.cuda.get_device_name(0),
 torch=str(torch.__version__),transformers=str(transformers.__version__),
 python=platform.python_version(),dtype=str(DTYPE),attention="sdpa",
 layers=NL,hidden=H,query_heads=QH,kv_heads=KVH,head_dim=HD,cut=CUT628,
 source_git_blob_sha1=GIT_BLOB628,source_file_sha256=FILE_SHA628,
 source_origin=origin628,panel_sha256=PANEL_SHA)
log628("ENVIRONMENT",json.dumps(ENV628))
log628("[PREFLIGHT] Native repeat + CUT3 + disk + behavior on cartridge 0")
pre628=run628(0,True)
if pre628["status"]!="PASS":
 save628(dict(test="628",status="PREFLIGHT_FAILURE",environment=ENV628,preflight=pre628))
 log628("PREFLIGHT_ERROR_STAGE",pre628.get("stage"))
 log628("PREFLIGHT_ERROR",pre628.get("error"))
 log628(pre628.get("traceback",""))
 raise RuntimeError("TEST628_PREFLIGHT_FAILURE")
log628("PREFLIGHT_PASS","NATIVE_REPEAT",pre628["native_repeat"]["exact_tensors"],"/56",
 "NATIVE_CUT3",pre628["native_cut3"]["exact_tensors"],"/56",
 "DISK",pre628["disk_roundtrip"]["exact_tensors"],"/56")
ROWS628=[pre628]
save628(dict(test="628",status="IN_PROGRESS",progress=1,total=N628,environment=ENV628,cases=ROWS628))
for ci in range(1,N628):
 row=run628(ci,ci in BEHAVIOR_IDS628)
 ROWS628.append(row)
 if row["status"]=="PASS":
  log628("CASE",f"{ci+1:03d}/{N628}","ID",ci,
   "REPEAT",f"{row['native_repeat']['exact_tensors']}/56",
   "NATIVE_CUT3",f"{row['native_cut3']['exact_tensors']}/56",
   "LOWER",f"{row['lower_0_3']['exact_tensors']}/8",
   "UPPER",f"{row['upper_4_27']['exact_tensors']}/48",
   "DISK",f"{row['disk_roundtrip']['exact_tensors']}/56")
 else:
  log628("CASE_ERROR",ci,"STAGE",row.get("stage"),"ERROR",row.get("error"))
 save628(dict(test="628",status="IN_PROGRESS",progress=ci+1,total=N628,environment=ENV628,cases=ROWS628))
 if row["status"]!="PASS":
  raise RuntimeError(f"TEST628_CASE_FAILURE_ID_{ci}_STAGE_{row.get('stage')}_REVIEW_JSON")
 if (ci+1)%16==0:
  gc.collect()
  torch.cuda.empty_cache()
VALID628=[r for r in ROWS628 if r["status"]=="PASS"]
ERROR628=[r for r in ROWS628 if r["status"]!="PASS"]
def exact628(key):
 return sum(int(r[key].get("bitwise_equal",False)) for r in VALID628)
def badbits628(key):
 return sum(int(r[key].get("mismatched_bits",0)) for r in VALID628)
BEHAVIOR628=[r["behavior"] for r in VALID628 if "behavior" in r]
MODEL_OK628=guard628()
REPEAT_OK628=len(VALID628)==N628 and exact628("native_repeat")==N628 and all(r["h3_repeat"].get("bitwise_equal",False) for r in VALID628)
DISK_OK628=len(VALID628)==N628 and exact628("disk_roundtrip")==N628 and all(r["disk_sha_verified"] for r in VALID628)
NATIVE_CUT3_OK628=len(VALID628)==N628 and exact628("native_cut3")==N628
if not MODEL_OK628:INTERPRETATION628="MODEL_INTEGRITY_FAILURE"
elif not REPEAT_OK628:INTERPRETATION628="INCONCLUSIVE_NATIVE_REPEATABILITY"
elif not DISK_OK628:INTERPRETATION628="INCONCLUSIVE_DISK_ROUNDTRIP"
elif NATIVE_CUT3_OK628:INTERPRETATION628="CUT3_BITWISE_IDENTICAL_TO_NATIVE_ON_TESTED_96"
else:INTERPRETATION628="NATIVE_CUT3_BITWISE_DIFFERENCE_DETECTED_FURTHER_CAUSAL_ANALYSIS_REQUIRED"
TOTALS628=dict(cases_requested=N628,cases_completed=len(VALID628),errors=len(ERROR628),
 native_repeat_exact=exact628("native_repeat"),
 h3_repeat_exact=sum(int(r["h3_repeat"].get("bitwise_equal",False)) for r in VALID628),
 native_cut3_exact=exact628("native_cut3"),
 lower_0_3_exact=exact628("lower_0_3"),
 upper_4_27_exact=exact628("upper_4_27"),
 disk_roundtrip_exact=exact628("disk_roundtrip"),
 native_cut3_mismatched_bits=badbits628("native_cut3"),
 behavior_cases=len(BEHAVIOR628),
 behavior_native_correct=sum(int(r["native_correct"]) for r in BEHAVIOR628),
 behavior_cut3_correct=sum(int(r["cut3_correct"]) for r in BEHAVIOR628),
 behavior_exact_answer_match=sum(int(r["exact_answer_match"]) for r in BEHAVIOR628))
REPORT628=dict(test="628",
 title="AKBASCORE MAM — Native KV vs CUT3 Bitwise Mechanism Audit",
 status="COMPLETE" if len(VALID628)==N628 and MODEL_OK628 and REPEAT_OK628 and DISK_OK628 else "INCONCLUSIVE",
 interpretation=INTERPRETATION628,
 environment=ENV628,totals=TOTALS628,
 model_unchanged=MODEL_OK628,
 native_repeat_valid=REPEAT_OK628,
 disk_roundtrip_valid=DISK_OK628,
 cases=ROWS628,
 elapsed_seconds=round(time.perf_counter()-START628,3),
 limitations=[
  "Original TEST566 checkpoint, init and answer implementations are unchanged.",
  "Native and CUT3 use the same model, source tokens, positions and masks.",
  "Native repeatability and layer-3 hidden-state repeatability are checked.",
  "Bitwise inequality alone does not establish a novel latent mechanism.",
  "Single model and single hardware/software environment.",
  "96 synthetic previously exposed facts; no blind generalization claim.",
  "Behavior is evaluated on 24 panel targets plus the preflight case, if distinct.",
  "No direct Microsoft CacheBlend or LMCache benchmark.",
  "No text-RAG baseline.",
  "New physical audit files, not the original 1453-file TEST627 pool.",
  "No verified cold-storage latency or million-cartridge scaling.",
  "Original sampled-weight sentinel, not a full-model weight-byte hash."
 ])
save628(REPORT628)
log628("="*112)
log628("TEST628_STATUS",REPORT628["status"])
log628("TEST628_INTERPRETATION",INTERPRETATION628)
log628("NATIVE_REPEAT_EXACT",TOTALS628["native_repeat_exact"],"/96")
log628("H3_REPEAT_EXACT",TOTALS628["h3_repeat_exact"],"/96")
log628("NATIVE_CUT3_EXACT",TOTALS628["native_cut3_exact"],"/96")
log628("LOWER_0_3_EXACT",TOTALS628["lower_0_3_exact"],"/96")
log628("UPPER_4_27_EXACT",TOTALS628["upper_4_27_exact"],"/96")
log628("CUT3_DISK_EXACT",TOTALS628["disk_roundtrip_exact"],"/96")
log628("NATIVE_CUT3_MISMATCHED_BITS",TOTALS628["native_cut3_mismatched_bits"])
log628("BEHAVIOR_NATIVE_CORRECT",TOTALS628["behavior_native_correct"],"/",TOTALS628["behavior_cases"])
log628("BEHAVIOR_CUT3_CORRECT",TOTALS628["behavior_cut3_correct"],"/",TOTALS628["behavior_cases"])
log628("BEHAVIOR_EXACT_ANSWER_MATCH",TOTALS628["behavior_exact_answer_match"],"/",TOTALS628["behavior_cases"])
log628("MODEL_UNCHANGED",MODEL_OK628)
log628("TEST628_OUTPUT",str(OUT628))
if not MODEL_OK628:raise RuntimeError("TEST628_MODEL_INTEGRITY_FAILURE")
if not REPEAT_OK628:log628("CAUTION: Native-repeat instability prevents a conclusive bitwise mechanism interpretation.")
if not DISK_OK628:raise RuntimeError("TEST628_DISK_INTEGRITY_FAILURE")


# ======================================================================================================================
# TEST632 — AKBASCORE MAM · LOCKED 24-TARGET LATENCY / STORAGE / BEHAVIOR BENCHMARK
# FROZEN TEST566 BACKBONE · NATIVE PREFILL vs FULL KV RELOAD vs LOWER KV + H3 RECONSTRUCTION
# 24 PANEL TARGETS · 3 MEASURED REPEATS · GPU-SYNCHRONIZED TIMING · BITWISE · ANSWER ACCURACY
# SINGLE GOOGLE COLAB CELL · NO TRAINING · NO ORIGINAL ENGINE MODIFICATION
# ======================================================================================================================
import os,sys,time,json,csv,hashlib,urllib.request,statistics,gc,traceback,platform,random
from pathlib import Path
import torch,numpy as np
ROOT632=Path("/content/AKBASCORE_TEST632")
DISK632=ROOT632/"cartridges"
DISK632.mkdir(parents=True,exist_ok=True)
URL632="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/566.py"
SHA632="0155cda553e5f08ed2bec53611e49898a5ca24e0d270c3a1274e661bc205bdcd"
BLOB632="b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71"
PANEL632="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
BOUND632="RESULTS=[];RAW=[];ERRORS=[]"
CUT632=3
REPEATS632=3
WARMUPS632=1
T632=time.perf_counter()
def log632(*a):print(*a,flush=True)
def sha632(b):return hashlib.sha256(b).hexdigest()
def blob632(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def filehash632(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()
def fetch632():
 for p in (Path("/content/566.py"),Path("/content/TEST566.py")):
  if p.exists():
   b=p.read_bytes()
   if sha632(b)==SHA632 and blob632(b)==BLOB632:return b,"LOCAL_VERIFIED"
 req=urllib.request.Request(URL632,headers={"User-Agent":"AKBASCORE-TEST632"})
 b=urllib.request.urlopen(req,timeout=120).read()
 if sha632(b)!=SHA632 or blob632(b)!=BLOB632:raise RuntimeError("SOURCE_LOCK_FAILURE")
 return b,"GITHUB_VERIFIED"
def sync632():torch.cuda.synchronize()
def timed632(fn):
 sync632();t=time.perf_counter();result=fn();sync632()
 return result,(time.perf_counter()-t)*1000
def cpu632(x):
 if torch.is_tensor(x):return x.detach().cpu().contiguous()
 if isinstance(x,tuple):return tuple(cpu632(z) for z in x)
 if isinstance(x,list):return [cpu632(z) for z in x]
 if isinstance(x,dict):return {k:cpu632(v) for k,v in x.items()}
 raise TypeError("INVALID_ARTIFACT")
def gpu632(x):
 if torch.is_tensor(x):return x.to(device=DEVICE,dtype=DTYPE)
 if isinstance(x,tuple):return tuple(gpu632(z) for z in x)
 if isinstance(x,list):return [gpu632(z) for z in x]
 if isinstance(x,dict):return {k:gpu632(v) for k,v in x.items()}
 raise TypeError("INVALID_ARTIFACT")
def payload632(x):
 if torch.is_tensor(x):return int(x.numel()*x.element_size())
 if isinstance(x,(tuple,list)):return sum(payload632(v) for v in x)
 if isinstance(x,dict):return sum(payload632(v) for v in x.values())
 return 0
def exact632(a,b):
 if torch.is_tensor(a):
  return torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype and bool(
   torch.equal(a.detach().cpu().contiguous().view(torch.uint8),
               b.detach().cpu().contiguous().view(torch.uint8)))
 if isinstance(a,(tuple,list)):
  return isinstance(b,(tuple,list)) and len(a)==len(b) and all(exact632(x,y) for x,y in zip(a,b))
 if isinstance(a,dict):
  return isinstance(b,dict) and a.keys()==b.keys() and all(exact632(a[k],b[k]) for k in a)
 return a==b
def validate632(kv,n):
 if len(kv)!=NL:raise RuntimeError("LAYER_COUNT_FAILURE")
 for i,(k,v) in enumerate(kv):
  if tuple(k.shape)!=(1,KVH,n,HD) or tuple(v.shape)!=(1,KVH,n,HD):
   raise RuntimeError(f"KV_SHAPE_FAILURE_L{i}")
  if k.dtype!=DTYPE or v.dtype!=DTYPE:raise RuntimeError(f"KV_DTYPE_FAILURE_L{i}")
def rebuild632(obj,n):
 lower=obj["lower_kv"]
 h3=obj["h3"]
 if len(lower)!=CUT632+1 or tuple(h3.shape)!=(1,n,H):
  raise RuntimeError("CARTRIDGE_SHAPE_FAILURE")
 calls=[0]
 def inject(module,args,out):
  calls[0]+=1
  return replace(out,h3)
 hook=model.model.layers[CUT632].register_forward_hook(inject)
 try:
  with torch.no_grad():
   pos=torch.arange(n,device=DEVICE)
   out=model(inputs_embeds=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE),
    attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),
    position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
   upper=clone(out.past_key_values)
 finally:hook.remove()
 if calls[0]!=1:raise RuntimeError("INJECTION_COUNT_FAILURE")
 return tuple(lower)+tuple(upper[CUT632+1:])
def save632(path,obj):
 torch.save(cpu632(obj),path)
 return dict(path=str(path),sha256=filehash632(path),disk_bytes=path.stat().st_size,
  payload_bytes=payload632(obj))
def load632(meta):
 p=Path(meta["path"])
 if filehash632(p)!=meta["sha256"]:raise RuntimeError("DISK_SHA_FAILURE")
 return gpu632(torch.load(p,map_location="cpu",weights_only=True))
def bench632(fn,repeats=REPEATS632):
 for _ in range(WARMUPS632):
  result,_=timed632(fn)
  del result
 samples=[];last=None
 for _ in range(repeats):
  result,ms=timed632(fn)
  samples.append(ms)
  last=result
 return last,samples
def summarize632(x):
 return dict(mean_ms=round(statistics.mean(x),5),
  median_ms=round(statistics.median(x),5),
  min_ms=round(min(x),5),max_ms=round(max(x),5),
  samples_ms=[round(v,5) for v in x])
def write632(path,obj):
 tmp=path.with_suffix(".tmp")
 with open(tmp,"w",encoding="utf-8") as f:
  json.dump(obj,f,indent=2,ensure_ascii=False,default=str,allow_nan=False)
 os.replace(tmp,path)
log632("="*120)
log632("TEST632 — AKBASCORE MAM · NATIVE PREFILL / FULL KV / LOWER KV + H3")
log632("="*120)
if not torch.cuda.is_available():raise RuntimeError("CUDA_REQUIRED")
raw632,origin632=fetch632()
text632=raw632.decode("utf-8")
if text632.count(BOUND632)!=1:raise RuntimeError("BOOTSTRAP_BOUNDARY_FAILURE")
log632("SOURCE",origin632,"SHA256",sha632(raw632),"BLOB",blob632(raw632))
exec(compile(text632.split(BOUND632,1)[0],"TEST566_LOCKED.py","exec"),globals())
if len(CAR)!=96 or len(PANEL)!=24 or P!=29 or PANEL_SHA!=PANEL632:
 raise RuntimeError("PANEL_LOCK_FAILURE")
if (NL,H,QH,KVH,HD)!=(28,3584,28,4,128):raise RuntimeError("ARCH_FAILURE")
if sentinel()!=S0:raise RuntimeError("MODEL_SENTINEL_FAILURE")
# The original 24-question panel uses multi-cartridge cases.
# This test isolates each panel's target cartridge; it is NOT a multi-cartridge panel score.
TARGETS632=[]
for wi632 in range(24):
 z632=PANEL[wi632]
 ci632=z632["target"]
 typ632=CAR[ci632]["type"]
 TARGETS632.append(dict(world=wi632,id=ci632,
  question=W[wi632][typ632],gold=CAR[ci632]["gold"],type=typ632))
if len(set(z["id"] for z in TARGETS632))!=24:
 raise RuntimeError("DUPLICATE_PANEL_TARGETS")
REPORT632=dict(test="632",status="IN_PROGRESS",
 source_sha256=SHA632,panel_sha256=PANEL_SHA,
 source_git_blob=BLOB632,model=MODEL_ID,gpu=torch.cuda.get_device_name(0),
 torch=str(torch.__version__),transformers=str(transformers.__version__),
 cuts=CUT632,repeats=REPEATS632,warmups=WARMUPS632,
 panel_scope="24 isolated target cartridges selected from the locked multi-cartridge panel",
 timing_scope="Same-process GPU-synchronized warm-cache benchmark; no process startup",
 cases=[],summary={},limitations=[
  "Not a multi-cartridge consolidation benchmark.",
  "Native Prefill is measured using the original source tokens.",
  "Disk reload arms are measured with source tokens blocked.",
  "Disk I/O may be served by the operating-system page cache.",
  "The original model stays resident on GPU.",
  "The measured cold-start loading of TEST631 is separate from these latency numbers.",
  "Cartridge forging and disk writing are reported separately from reload.",
  "These measurements are not yet comparisons against Microsoft CacheBlend, LMCache, SGLang or RAG."
 ])
OUT632=ROOT632/"TEST632_LATENCY_BEHAVIOR.json"
CSV632=ROOT632/"TEST632_DEMO_COMPARISON.csv"
write632(OUT632,REPORT632)
for index632,z632 in enumerate(TARGETS632):
 ci632=z632["id"];q632=z632["question"];gold632=z632["gold"]
 original632=CAR[ci632]["body"]
 row632=dict(world=z632["world"],cartridge_id=ci632,type=z632["type"],
  question=q632,gold=gold632,total_tokens=P+len(original632),
  arms={},native_answer=None)
 try:
  # PHASE A: Native reference and physical cartridge creation.
  cp632,forge_ms632=timed632(lambda:checkpoint(ci632,CUT632))
  native632=cp632["kv"]
  n632=cp632["h"].shape[1]
  validate632(native632,n632)
  full632=save632(DISK632/f"C{ci632:03d}_FULL.pt",dict(kv=native632))
  mam632=save632(DISK632/f"C{ci632:03d}_MAM.pt",
   dict(lower_kv=native632[:CUT632+1],h3=cp632["h"]))
  row632["forge_ms"]=round(forge_ms632,5)
  row632["full_disk_bytes"]=full632["disk_bytes"]
  row632["mam_disk_bytes"]=mam632["disk_bytes"]
  row632["storage_reduction_pct"]=round(
   100*(1-mam632["disk_bytes"]/full632["disk_bytes"]),5)
  # Native prefill benchmark: source is available only for this arm.
  native_last632,native_times632=bench632(
   lambda:checkpoint(ci632,CUT632)["kv"])
  native_exact632=exact632(native632,native_last632)
  if not native_exact632:raise RuntimeError("NATIVE_REPEAT_BITWISE_FAILURE")
  del native_last632
  # PHASE B: source token access blocked for both disk-based arms.
  CAR[ci632]["body"]=None
  if CAR[ci632]["body"] is not None:raise RuntimeError("SOURCE_BLOCK_FAILURE")
  def full_run632():
   obj=load632(full632)
   return obj["kv"]
  def mam_run632():
   obj=load632(mam632)
   return rebuild632(obj,n632)
  full_last632,full_times632=bench632(full_run632)
  mam_last632,mam_times632=bench632(mam_run632)
  validate632(full_last632,n632)
  validate632(mam_last632,n632)
  full_exact632=exact632(native632,full_last632)
  mam_exact632=exact632(native632,mam_last632)
  # Measure answer generation separately, with identical original TEST566 answer().
  # Clone the KV to avoid mutation across repeats.
  def respond632(kv):
   return answer(tuple((k.clone(),v.clone()) for k,v in kv),q632)
  native_answer632,native_answer_times632=bench632(lambda:respond632(native632))
  full_answer632,full_answer_times632=bench632(lambda:respond632(full_last632))
  mam_answer632,mam_answer_times632=bench632(lambda:respond632(mam_last632))
  row632["native_answer"]=native_answer632
  row632["arms"]={
   "NATIVE_PREFILL":dict(prepare=summarize632(native_times632),
    answer=summarize632(native_answer_times632),
    answer_text=native_answer632,correct=bool(hit(native_answer632,gold632)),
    bitwise_native=True,disk_bytes=None),
   "FULL_KV_RELOAD":dict(prepare=summarize632(full_times632),
    answer=summarize632(full_answer_times632),
    answer_text=full_answer632,correct=bool(hit(full_answer632,gold632)),
    bitwise_native=full_exact632,disk_bytes=full632["disk_bytes"]),
   "AKBASCORE_MAM":dict(prepare=summarize632(mam_times632),
    answer=summarize632(mam_answer_times632),
    answer_text=mam_answer632,correct=bool(hit(mam_answer632,gold632)),
    bitwise_native=mam_exact632,disk_bytes=mam632["disk_bytes"])}
  row632["all_answers_equal"]=(native_answer632==full_answer632==mam_answer632)
  row632["status"]="PASS" if (
   full_exact632 and mam_exact632 and row632["all_answers_equal"]) else "MISMATCH"
  log632(f"CASE {index632+1:02d}/24 ID={ci632:03d}",
   "STATUS",row632["status"],
   "NATIVE_MS",round(statistics.median(native_times632),3),
   "FULL_MS",round(statistics.median(full_times632),3),
   "MAM_MS",round(statistics.median(mam_times632),3),
   "REDUCTION",row632["storage_reduction_pct"],
   "CORRECT",row632["arms"]["AKBASCORE_MAM"]["correct"])
 finally:
  CAR[ci632]["body"]=original632
 REPORT632["cases"].append(row632)
 write632(OUT632,REPORT632)
 del cp632,native632,full_last632,mam_last632
 gc.collect()
 torch.cuda.empty_cache()
if sentinel()!=S0:raise RuntimeError("MODEL_SENTINEL_CHANGED")
ARMS632=("NATIVE_PREFILL","FULL_KV_RELOAD","AKBASCORE_MAM")
SUMMARY632={}
for arm632 in ARMS632:
 a632=[r["arms"][arm632] for r in REPORT632["cases"]]
 prep632=[r["prepare"]["median_ms"] for r in a632]
 ans632=[r["answer"]["median_ms"] for r in a632]
 SUMMARY632[arm632]=dict(
  tested=len(a632),correct=sum(int(r["correct"]) for r in a632),
  bitwise_equal=sum(int(r["bitwise_native"]) for r in a632),
  median_prepare_ms=round(statistics.median(prep632),5),
  mean_prepare_ms=round(statistics.mean(prep632),5),
  median_answer_ms=round(statistics.median(ans632),5),
  mean_answer_ms=round(statistics.mean(ans632),5),
  median_prepare_plus_answer_ms=round(statistics.median(
   [x+y for x,y in zip(prep632,ans632)]),5))
REPORT632["summary"]=SUMMARY632
REPORT632["mean_storage_reduction_pct"]=round(statistics.mean(
 r["storage_reduction_pct"] for r in REPORT632["cases"]),5)
REPORT632["all_answers_equal"]=all(r["all_answers_equal"] for r in REPORT632["cases"])
REPORT632["model_unchanged"]=sentinel()==S0
REPORT632["elapsed_seconds"]=round(time.perf_counter()-T632,3)
REPORT632["status"]="COMPLETE" if (
 len(REPORT632["cases"])==24 and
 all(r["status"]=="PASS" for r in REPORT632["cases"]) and
 REPORT632["model_unchanged"]) else "MISMATCH"
write632(OUT632,REPORT632)
with open(CSV632,"w",newline="",encoding="utf-8") as f:
 fields632=["world","cartridge_id","type","arm","status","correct",
  "bitwise_native","disk_bytes","storage_reduction_pct",
  "prepare_median_ms","answer_median_ms","answer_text"]
 w632=csv.DictWriter(f,fieldnames=fields632)
 w632.writeheader()
 for r632 in REPORT632["cases"]:
  for arm632,a632 in r632["arms"].items():
   w632.writerow(dict(world=r632["world"],cartridge_id=r632["cartridge_id"],
    type=r632["type"],arm=arm632,status=r632["status"],
    correct=a632["correct"],bitwise_native=a632["bitwise_native"],
    disk_bytes=a632["disk_bytes"],
    storage_reduction_pct=r632["storage_reduction_pct"],
    prepare_median_ms=a632["prepare"]["median_ms"],
    answer_median_ms=a632["answer"]["median_ms"],
    answer_text=a632["answer_text"]))
log632("="*120)
log632("TEST632_STATUS",REPORT632["status"])
log632("TARGETS",len(REPORT632["cases"]),"/24")
log632("ALL_ANSWERS_EQUAL",REPORT632["all_answers_equal"])
log632("MODEL_UNCHANGED",REPORT632["model_unchanged"])
log632("MEAN_STORAGE_REDUCTION_PCT",REPORT632["mean_storage_reduction_pct"])
for arm632,s632 in SUMMARY632.items():
 log632("ARM",arm632,
  "CORRECT",f'{s632["correct"]}/{s632["tested"]}',
  "BITWISE",f'{s632["bitwise_equal"]}/{s632["tested"]}',
  "PREPARE_MEDIAN_MS",s632["median_prepare_ms"],
  "ANSWER_MEDIAN_MS",s632["median_answer_ms"],
  "PREPARE_PLUS_ANSWER_MEDIAN_MS",s632["median_prepare_plus_answer_ms"])
log632("JSON_OUTPUT",OUT632)
log632("CSV_OUTPUT",CSV632)
log632("TOTAL_SECONDS",REPORT632["elapsed_seconds"])
log632("="*120)
if REPORT632["status"]!="COMPLETE":raise RuntimeError("TEST632_VALIDATION_FAILURE")


# ======================================================================================================================
# TEST634 — AKBASCORE MAM · MULTI-CARTRIDGE SCALING / POSITION / GPU MEMORY
# LOCKED TEST566 BACKBONE · TEST633 PHYSICAL MAM DISK ARCHITECTURE
# 1/5/10/20/40 CARTRIDGES × FIRST/MIDDLE/LAST × 8 TARGET WORLDS
# NATIVE JOINT / INDEPENDENT KV / PHYSICAL MAM DISK
# SOURCE-BLOCKED MAM · BITWISE PREFIX AUDIT · ACCURACY · LATENCY · GPU PEAK · DISK BYTES
# NO TRAINING · NO ROUTER · NO NEW MEMORY ENGINE · SINGLE COLAB CELL
# ======================================================================================================================
import os,sys,time,json,csv,hashlib,urllib.request,statistics,gc,traceback,random
from pathlib import Path
import torch,numpy as np
ROOT634=Path("/content/AKBASCORE_TEST634")
DISK634=ROOT634/"cartridges"
DISK634.mkdir(parents=True,exist_ok=True)
URL634="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/566.py"
SHA634="0155cda553e5f08ed2bec53611e49898a5ca24e0d270c3a1274e661bc205bdcd"
BLOB634="b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71"
PANEL634="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
BOUND634="RESULTS=[];RAW=[];ERRORS=[]"
CUT634=3
SIZES634=(1,5,10,20,40)
SLOTS634=("FIRST","MIDDLE","LAST")
ARMS634=("JOINT","INDEP","MAM_DISK")
WORLDS634=tuple(range(8))
T634=time.perf_counter()
def log634(*a):print(*a,flush=True)
def sha634(b):return hashlib.sha256(b).hexdigest()
def blob634(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def filehash634(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()
def fetch634():
 for p in (Path("/content/566.py"),Path("/content/TEST566.py")):
  if p.is_file():
   b=p.read_bytes()
   if sha634(b)==SHA634 and blob634(b)==BLOB634:return b,"LOCAL_VERIFIED"
 req=urllib.request.Request(URL634,headers={"User-Agent":"AKBASCORE-TEST634"})
 b=urllib.request.urlopen(req,timeout=120).read()
 if sha634(b)!=SHA634 or blob634(b)!=BLOB634:raise RuntimeError("SOURCE_LOCK_FAILURE")
 return b,"GITHUB_VERIFIED"
def sync634():torch.cuda.synchronize()
def measure634(fn):
 sync634()
 torch.cuda.reset_peak_memory_stats()
 baseline=torch.cuda.memory_allocated()
 start=time.perf_counter()
 result=fn()
 sync634()
 elapsed=(time.perf_counter()-start)*1000
 peak=torch.cuda.max_memory_allocated()
 return result,round(elapsed,5),int(peak),int(max(0,peak-baseline))
def cpu634(x):
 if torch.is_tensor(x):return x.detach().cpu().contiguous()
 if isinstance(x,tuple):return tuple(cpu634(v) for v in x)
 if isinstance(x,list):return [cpu634(v) for v in x]
 if isinstance(x,dict):return {k:cpu634(v) for k,v in x.items()}
 raise TypeError("INVALID_ARTIFACT")
def gpu634(x):
 if torch.is_tensor(x):return x.to(DEVICE)
 if isinstance(x,tuple):return tuple(gpu634(v) for v in x)
 if isinstance(x,list):return [gpu634(v) for v in x]
 if isinstance(x,dict):return {k:gpu634(v) for k,v in x.items()}
 raise TypeError("INVALID_ARTIFACT")
def exact634(a,b):
 if torch.is_tensor(a):
  return torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype and bool(torch.equal(a,b))
 if isinstance(a,(tuple,list)):
  return isinstance(b,(tuple,list)) and len(a)==len(b) and all(exact634(x,y) for x,y in zip(a,b))
 return a==b
def write634(path,obj):
 tmp=path.with_suffix(".tmp")
 with open(tmp,"w",encoding="utf-8") as f:
  json.dump(obj,f,indent=2,ensure_ascii=False,default=str,allow_nan=False)
 os.replace(tmp,path)
def median634(values):return round(statistics.median(values),5) if values else None
log634("="*120)
log634("TEST634 — AKBASCORE MAM · MULTI-CARTRIDGE SCALING")
log634("="*120)
if not torch.cuda.is_available():raise RuntimeError("CUDA_REQUIRED")
raw634,origin634=fetch634()
src634=raw634.decode("utf-8")
if src634.count(BOUND634)!=1:raise RuntimeError("BOOTSTRAP_BOUNDARY_FAILURE")
log634("SOURCE",origin634,"SHA256",sha634(raw634),"GIT_BLOB",blob634(raw634))
exec(compile(src634.split(BOUND634,1)[0],"TEST566_LOCKED.py","exec"),globals())
if len(CAR)!=96 or len(PANEL)!=24 or PANEL_SHA!=PANEL634 or P!=29:
 raise RuntimeError("PANEL_LOCK_FAILURE")
if (NL,H,QH,KVH,HD)!=(28,3584,28,4,128):
 raise RuntimeError("MODEL_ARCHITECTURE_FAILURE")
if sentinel()!=S0:raise RuntimeError("MODEL_SENTINEL_FAILURE")
@torch.no_grad()
def init_disk634(obj):
 lower=obj["lower_kv"];h3=obj["h3"];n=h3.shape[1]
 if len(lower)!=CUT634+1 or tuple(h3.shape)!=(1,n,H):
  raise RuntimeError("INIT_SHAPE_FAILURE")
 def inject(module,args,out):return replace(out,h3)
 hook=model.model.layers[CUT634].register_forward_hook(inject)
 try:
  pos=torch.arange(n,device=DEVICE)
  out=model(inputs_embeds=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE),
   attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),
   position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
  upper=clone(out.past_key_values)
  return tuple(lower)+tuple(upper[CUT634+1:])
 finally:hook.remove()
@torch.no_grad()
def append_disk634(memory,obj):
 lower=obj["lower_kv"];h3=obj["h3"]
 oldn=memory[0][0].shape[-2];n=h3.shape[1];q=n-P
 if len(lower)!=CUT634+1 or q<=0:
  raise RuntimeError("APPEND_SHAPE_FAILURE")
 body=h3[:,P:,:]
 pos=torch.arange(oldn,oldn+q,device=DEVICE)
 def inject(module,args,out):
  current=out[0] if isinstance(out,(tuple,list)) else out
  if current.shape!=body.shape:raise RuntimeError("INJECTION_SHAPE_FAILURE")
  return replace(out,body)
 hook=model.model.layers[CUT634].register_forward_hook(inject)
 try:
  out=model(inputs_embeds=torch.zeros((1,q,H),device=DEVICE,dtype=DTYPE),
   past_key_values=cache_build(memory),
   attention_mask=torch.ones((1,oldn+q),device=DEVICE,dtype=torch.long),
   position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
  grown=list(clone(out.past_key_values))
  oldpos=torch.arange(P,P+q,device=DEVICE)
  for li in range(CUT634+1):
   ok,ov=memory[li];k,v=lower[li]
   kk=rephase(k[:,:,P:,:],oldpos,pos)
   vv=v[:,:,P:,:]
   grown[li]=(torch.cat((ok,kk),-2),torch.cat((ov,vv),-2))
  result=tuple(grown)
  for (ok,ov),(nk,nv) in zip(memory,result):
   if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):
    raise RuntimeError("APPEND_ONLY_BITWISE_FAILURE")
  return result
 finally:hook.remove()
MANIFEST634={}
log634("[1/4] FORGING 96 PHYSICAL CARTRIDGES")
for ci634 in range(96):
 cp634=checkpoint(ci634,CUT634)
 obj634=dict(lower_kv=cp634["kv"][:CUT634+1],h3=cp634["h"])
 path634=DISK634/f"C{ci634:03d}.pt"
 torch.save(cpu634(obj634),path634)
 MANIFEST634[ci634]=dict(path=str(path634),sha256=filehash634(path634),
  bytes=path634.stat().st_size,tokens=cp634["h"].shape[1])
 del cp634,obj634
 if (ci634+1)%16==0:log634("FORGED",ci634+1,"/96")
def load_disk634(ci):
 meta=MANIFEST634[ci];p=Path(meta["path"])
 if filehash634(p)!=meta["sha256"]:
  raise RuntimeError(f"CARTRIDGE_SHA_FAILURE_{ci}")
 return gpu634(torch.load(p,map_location="cpu",weights_only=True))
def mam634(keys):
 memory=None
 for ci in keys:
  obj=load_disk634(ci)
  memory=init_disk634(obj) if memory is None else append_disk634(memory,obj)
 return memory
def validate634(memory,keys):
 n=P+sum(MANIFEST634[ci]["tokens"]-P for ci in keys)
 if len(memory)!=NL:raise RuntimeError("LAYER_COUNT_FAILURE")
 for li,(k,v) in enumerate(memory):
  if tuple(k.shape)!=(1,KVH,n,HD) or tuple(v.shape)!=(1,KVH,n,HD):
   raise RuntimeError(f"KV_SHAPE_FAILURE_L{li}")
 return n
# Fixed, reproducible scaling panel: one target and N-1 distinct distractor cartridges.
# No duplicate cartridge IDs or distractors from the target's own world.
def make_cases634():
 rng=random.Random(634552)
 cases=[]
 for wi in WORLDS634:
  target=PANEL[wi]["target"]
  pool=[ci for ci in range(96) if CAR[ci]["world"]!=wi]
  rng.shuffle(pool)
  distractors=pool[:max(SIZES634)-1]
  if len(distractors)<max(SIZES634)-1:
   raise RuntimeError("INSUFFICIENT_DISTRACTORS")
  typ=CAR[target]["type"]
  for count in SIZES634:
   other=distractors[:count-1]
   for slot in SLOTS634:
    if count==1:
     keys=[target]
    elif slot=="FIRST":
     keys=[target]+other
    elif slot=="LAST":
     keys=other+[target]
    else:
     left=(count-1)//2
     keys=other[:left]+[target]+other[left:]
    if len(keys)!=count or len(set(keys))!=count:
     raise RuntimeError("CASE_CONSTRUCTION_FAILURE")
    cases.append(dict(world=wi,size=count,slot=slot,
     target=target,keys=keys,question=W[wi][typ],
     gold=CAR[target]["gold"]))
 return cases
CASES634=make_cases634()
PANEL_SCALING_SHA634=sha(CASES634)
log634("SCALING_CASES",len(CASES634),"PANEL_SHA",PANEL_SCALING_SHA634)
if len(CASES634)!=120:raise RuntimeError("SCALING_PANEL_FAILURE")
# A source-dependent control and a physical-disk adapter must match before scaling.
log634("[2/4] DISK ADAPTER CONTROL PROBES")
for z634 in (CASES634[0],CASES634[7],CASES634[38]):
 a634=consolidated(z634["keys"],CUT634)
 b634=mam634(z634["keys"])
 equal634=exact634(a634,b634)
 log634("PROBE","SIZE",z634["size"],"SLOT",z634["slot"],"BITWISE",equal634)
 del a634,b634
 if not equal634:raise RuntimeError("DISK_ADAPTER_EQUIVALENCE_FAILURE")
# Run source-dependent baselines first. Then remove every source token list for MAM.
ROWS634=[]
OUT634=ROOT634/"TEST634_SCALING_FINAL.json"
CSV634=ROOT634/"TEST634_SCALING_COMPARISON.csv"
REPORT634=dict(test="634",status="IN_PROGRESS",model=MODEL_ID,
 source_sha256=SHA634,source_git_blob=BLOB634,
 original_panel_sha=PANEL_SHA,scaling_panel_sha=PANEL_SCALING_SHA634,
 sizes=SIZES634,slots=SLOTS634,worlds=WORLDS634,
 arms=ARMS634,cut=CUT634,cartridges_forged=96,
 cases=CASES634,rows=ROWS634,summary={},
 limitations=[
  "Synthetic fixed-length fact cartridges, not new long-document source texts.",
  "The 120 cases include three identical single-cartridge positions per target.",
  "Native and independent arms are source-dependent and run before source blocking.",
  "MAM_DISK is source-blocked but shares a resident model in the same process.",
  "Preparation times include different necessary operations for each arm.",
  "GPU memory delta is peak allocated minus allocated at measurement start, not full process VRAM.",
  "Operating-system page cache may affect disk timings.",
  "No third-party caching library is benchmarked in this test.",
  "Only 8 target worlds are included to limit GPU runtime."
 ])
write634(OUT634,REPORT634)
def run_arm634(arm,keys):
 if arm=="JOINT":return joint(keys)
 if arm=="INDEP":return independent(keys,CUT634,"INDEP")
 if arm=="MAM_DISK":return mam634(keys)
 raise ValueError(arm)
log634("[3/4] NATIVE AND INDEPENDENT BASELINES")
for arm634 in ("JOINT","INDEP"):
 for i634,z634 in enumerate(CASES634):
  row634=dict(arm=arm634,case=i634,world=z634["world"],
   size=z634["size"],slot=z634["slot"],target=z634["target"],
   gold=z634["gold"],status="STARTED")
  try:
   mem634,prep634,peak634,delta634=measure634(
    lambda:run_arm634(arm634,z634["keys"]))
   row634["tokens"]=validate634(mem634,z634["keys"])
   ans634,answer_ms634,answer_peak634,answer_delta634=measure634(
    lambda:answer(mem634,z634["question"]))
   row634.update(answer=ans634,correct=bool(hit(ans634,z634["gold"])),
    prepare_ms=prep634,answer_ms=answer_ms634,
    prepare_peak_allocated_bytes=peak634,prepare_peak_delta_bytes=delta634,
    answer_peak_allocated_bytes=answer_peak634,
    answer_peak_delta_bytes=answer_delta634,status="PASS")
   del mem634
  except torch.cuda.OutOfMemoryError as e:
   row634.update(status="OOM",error=repr(e))
   torch.cuda.empty_cache()
  except Exception as e:
   row634.update(status="ERROR",error=repr(e),traceback=traceback.format_exc())
  ROWS634.append(row634)
  log634(arm634,f'{i634+1:03d}/120',"N",z634["size"],
   "SLOT",z634["slot"],"STATUS",row634["status"],
   "CORRECT",row634.get("correct"))
  if (i634+1)%15==0:write634(OUT634,REPORT634)
  if row634["status"]=="ERROR":
   write634(OUT634,REPORT634)
   raise RuntimeError(f"TEST634_BASELINE_FAILURE_{arm634}_{i634}")
 write634(OUT634,REPORT634)
# Prevent source token replay during MAM disk inference.
ORIGINAL634=[c["body"] for c in CAR]
for c in CAR:c["body"]=None
if any(c["body"] is not None for c in CAR):
 raise RuntimeError("SOURCE_BLOCK_FAILURE")
log634("[4/4] SOURCE-BLOCKED MAM PHYSICAL DISK SCALING")
try:
 for i634,z634 in enumerate(CASES634):
  row634=dict(arm="MAM_DISK",case=i634,world=z634["world"],
   size=z634["size"],slot=z634["slot"],target=z634["target"],
   gold=z634["gold"],status="STARTED",
   disk_bytes=sum(MANIFEST634[ci]["bytes"] for ci in z634["keys"]))
  try:
   mem634,prep634,peak634,delta634=measure634(
    lambda:mam634(z634["keys"]))
   row634["tokens"]=validate634(mem634,z634["keys"])
   ans634,answer_ms634,answer_peak634,answer_delta634=measure634(
    lambda:answer(mem634,z634["question"]))
   row634.update(answer=ans634,correct=bool(hit(ans634,z634["gold"])),
    prepare_ms=prep634,answer_ms=answer_ms634,
    prepare_peak_allocated_bytes=peak634,prepare_peak_delta_bytes=delta634,
    answer_peak_allocated_bytes=answer_peak634,
    answer_peak_delta_bytes=answer_delta634,status="PASS")
   del mem634
  except torch.cuda.OutOfMemoryError as e:
   row634.update(status="OOM",error=repr(e))
   torch.cuda.empty_cache()
  except Exception as e:
   row634.update(status="ERROR",error=repr(e),traceback=traceback.format_exc())
  ROWS634.append(row634)
  log634("MAM_DISK",f'{i634+1:03d}/120',"N",z634["size"],
   "SLOT",z634["slot"],"STATUS",row634["status"],
   "CORRECT",row634.get("correct"))
  if (i634+1)%15==0:write634(OUT634,REPORT634)
  if row634["status"]=="ERROR":
   write634(OUT634,REPORT634)
   raise RuntimeError(f"TEST634_MAM_FAILURE_{i634}")
 write634(OUT634,REPORT634)
finally:
 for c,body in zip(CAR,ORIGINAL634):c["body"]=body
# Paired accuracy, time and GPU-memory summary.
SUMMARY634={}
MAP634={(r["arm"],r["case"]):r for r in ROWS634}
for size634 in SIZES634:
 SUMMARY634[str(size634)]={}
 for arm634 in ARMS634:
  subset634=[r for r in ROWS634 if r["size"]==size634 and r["arm"]==arm634]
  passed634=[r for r in subset634 if r["status"]=="PASS"]
  SUMMARY634[str(size634)][arm634]=dict(
   tested=len(subset634),completed=len(passed634),
   correct=sum(int(r.get("correct",False)) for r in subset634),
   oom=sum(r["status"]=="OOM" for r in subset634),
   by_slot={slot:dict(
    correct=sum(int(r.get("correct",False)) for r in subset634 if r["slot"]==slot),
    tested=sum(r["slot"]==slot for r in subset634)) for slot in SLOTS634},
   median_tokens=median634([r["tokens"] for r in passed634]),
   median_prepare_ms=median634([r["prepare_ms"] for r in passed634]),
   median_answer_ms=median634([r["answer_ms"] for r in passed634]),
   median_total_ms=median634([r["prepare_ms"]+r["answer_ms"] for r in passed634]),
   median_prepare_peak_delta_mib=median634(
    [r["prepare_peak_delta_bytes"]/1048576 for r in passed634]),
   median_disk_bytes=median634(
    [r["disk_bytes"] for r in passed634 if "disk_bytes" in r]))
PAIRED634=[]
for i634,z634 in enumerate(CASES634):
 a634=MAP634.get(("JOINT",i634))
 b634=MAP634.get(("MAM_DISK",i634))
 if a634 and b634 and a634["status"]=="PASS" and b634["status"]=="PASS":
  PAIRED634.append(dict(case=i634,size=z634["size"],slot=z634["slot"],
   joint_correct=a634["correct"],mam_correct=b634["correct"],
   answer_equal=a634["answer"]==b634["answer"]))
REPORT634["summary"]=SUMMARY634
REPORT634["paired"]=PAIRED634
REPORT634["paired_answer_equal"]=sum(int(r["answer_equal"]) for r in PAIRED634)
REPORT634["model_unchanged"]=sentinel()==S0
REPORT634["total_seconds"]=round(time.perf_counter()-T634,3)
REPORT634["status"]="COMPLETE" if (
 len(ROWS634)==360 and
 all(r["status"]=="PASS" for r in ROWS634) and
 REPORT634["model_unchanged"]) else "COMPLETE_WITH_LIMITATIONS"
write634(OUT634,REPORT634)
with open(CSV634,"w",newline="",encoding="utf-8") as f:
 fields634=["arm","case","world","size","slot","status","correct",
  "gold","answer","tokens","prepare_ms","answer_ms",
  "prepare_peak_delta_bytes","answer_peak_delta_bytes","disk_bytes"]
 writer634=csv.DictWriter(f,fieldnames=fields634)
 writer634.writeheader()
 for r in ROWS634:writer634.writerow({k:r.get(k) for k in fields634})
log634("="*120)
log634("TEST634_STATUS",REPORT634["status"])
log634("ROWS",len(ROWS634),"/360")
log634("PAIRED_ANSWER_EQUAL",REPORT634["paired_answer_equal"],"/",len(PAIRED634))
log634("MODEL_UNCHANGED",REPORT634["model_unchanged"])
for size634 in SIZES634:
 for arm634 in ARMS634:
  s634=SUMMARY634[str(size634)][arm634]
  log634("SIZE",size634,"ARM",arm634,
   "CORRECT",f'{s634["correct"]}/{s634["tested"]}',
   "OOM",s634["oom"],
   "TOKENS_MEDIAN",s634["median_tokens"],
   "PREPARE_MEDIAN_MS",s634["median_prepare_ms"],
   "TOTAL_MEDIAN_MS",s634["median_total_ms"],
   "PEAK_DELTA_MIB",s634["median_prepare_peak_delta_mib"])
log634("JSON_OUTPUT",OUT634)
log634("CSV_OUTPUT",CSV634)
log634("TOTAL_SECONDS",REPORT634["total_seconds"])
log634("="*120)

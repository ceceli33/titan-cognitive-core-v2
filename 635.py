
# ======================================================================================================================
# TEST635 — AKBASCORE MEMORY ARENA · ROUND 1: STORAGE vs SPEED
# LOCKED TEST566 / TEST634 BACKBONE · REAL PHYSICAL FULL KV vs PHYSICAL MAM
# 8 WORLDS × 3 SIZES (5,20,40) × 3 POSITIONS = 72 MATCHES
# NATIVE JOINT / FULL_KV_DISK / MAM_DISK
# SAME QUESTIONS · SAME ANSWER() · SOURCE-BLOCKED DISK ARMS
# FILE SHA256 · BITWISE KV AUDIT · 2 REAL ANSWER RUNS · GPU SYNCHRONIZED LATENCY
# NO THIRD-PARTY COMPETITOR CLAIMS · NO TRAINING · SINGLE COLAB CELL
# ======================================================================================================================
import os,sys,time,json,csv,hashlib,urllib.request,statistics,random,gc,traceback
from pathlib import Path
import torch,numpy as np
ROOT635=Path("/content/AKBASCORE_TEST635")
CART635=ROOT635/"cartridges"
FULL635=ROOT635/"full_kv"
CART635.mkdir(parents=True,exist_ok=True)
FULL635.mkdir(parents=True,exist_ok=True)
URL635="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/566.py"
SHA635="0155cda553e5f08ed2bec53611e49898a5ca24e0d270c3a1274e661bc205bdcd"
BLOB635="b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71"
PANEL635="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
BOUND635="RESULTS=[];RAW=[];ERRORS=[]"
CUT635=3
SIZES635=(5,20,40)
SLOTS635=("FIRST","MIDDLE","LAST")
WORLDS635=tuple(range(8))
ARMS635=("NATIVE_JOINT","FULL_KV_DISK","MAM_DISK")
T635=time.perf_counter()
def log635(*x):print(*x,flush=True)
def sha635(b):return hashlib.sha256(b).hexdigest()
def blob635(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def filehash635(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b""):h.update(chunk)
 return h.hexdigest()
def fetch635():
 for p in (Path("/content/566.py"),Path("/content/TEST566.py")):
  if p.is_file():
   b=p.read_bytes()
   if sha635(b)==SHA635 and blob635(b)==BLOB635:return b,"LOCAL_VERIFIED"
 req=urllib.request.Request(URL635,headers={"User-Agent":"AKBASCORE-TEST635"})
 b=urllib.request.urlopen(req,timeout=120).read()
 if sha635(b)!=SHA635 or blob635(b)!=BLOB635:raise RuntimeError("SOURCE_LOCK_FAILURE")
 return b,"GITHUB_VERIFIED"
def sync635():torch.cuda.synchronize()
def timed635(fn):
 sync635()
 t=time.perf_counter()
 result=fn()
 sync635()
 return result,round((time.perf_counter()-t)*1000,5)
def cpu635(x):
 if torch.is_tensor(x):return x.detach().cpu().contiguous()
 if isinstance(x,tuple):return tuple(cpu635(v) for v in x)
 if isinstance(x,list):return [cpu635(v) for v in x]
 if isinstance(x,dict):return {k:cpu635(v) for k,v in x.items()}
 raise TypeError("UNSUPPORTED_ARTIFACT")
def gpu635(x):
 if torch.is_tensor(x):return x.to(DEVICE)
 if isinstance(x,tuple):return tuple(gpu635(v) for v in x)
 if isinstance(x,list):return [gpu635(v) for v in x]
 if isinstance(x,dict):return {k:gpu635(v) for k,v in x.items()}
 raise TypeError("UNSUPPORTED_ARTIFACT")
def exact635(a,b):
 if torch.is_tensor(a):
  return torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype and bool(torch.equal(a,b))
 if isinstance(a,(tuple,list)):
  return isinstance(b,(tuple,list)) and len(a)==len(b) and all(exact635(x,y) for x,y in zip(a,b))
 return a==b
def savejson635(path,obj):
 tmp=path.with_suffix(".tmp")
 with open(tmp,"w",encoding="utf-8") as f:
  json.dump(obj,f,indent=2,ensure_ascii=False,default=str,allow_nan=False)
 os.replace(tmp,path)
def med635(x):return round(statistics.median(x),5) if x else None
def mib635(n):return round(n/1048576,5)
log635("="*120)
log635("TEST635 — MEMORY ARENA · ROUND 1 · STORAGE vs SPEED")
log635("="*120)
if not torch.cuda.is_available():raise RuntimeError("CUDA_REQUIRED")
raw635,origin635=fetch635()
src635=raw635.decode("utf-8")
if src635.count(BOUND635)!=1:raise RuntimeError("SOURCE_BOUNDARY_FAILURE")
log635("SOURCE",origin635,"SHA256",sha635(raw635),"GIT_BLOB",blob635(raw635))
exec(compile(src635.split(BOUND635,1)[0],"TEST566_LOCKED.py","exec"),globals())
if PANEL_SHA!=PANEL635 or len(CAR)!=96 or P!=29:raise RuntimeError("PANEL_LOCK_FAILURE")
if (NL,H,QH,KVH,HD)!=(28,3584,28,4,128):raise RuntimeError("ARCH_LOCK_FAILURE")
if sentinel()!=S0:raise RuntimeError("MODEL_LOCK_FAILURE")
# Identical deterministic distractor selection as TEST634.
def cases635():
 rng=random.Random(634552)
 result=[]
 for wi in WORLDS635:
  target=PANEL[wi]["target"]
  pool=[ci for ci in range(96) if CAR[ci]["world"]!=wi]
  rng.shuffle(pool)
  distractors=pool[:39]
  typ=CAR[target]["type"]
  for count in SIZES635:
   others=distractors[:count-1]
   for slot in SLOTS635:
    if slot=="FIRST":keys=[target]+others
    elif slot=="LAST":keys=others+[target]
    else:
     left=(count-1)//2
     keys=others[:left]+[target]+others[left:]
    if len(keys)!=count or len(set(keys))!=count:
     raise RuntimeError("CASE_BUILD_FAILURE")
    result.append(dict(world=wi,size=count,slot=slot,
     keys=keys,target=target,question=W[wi][typ],
     gold=CAR[target]["gold"]))
 return result
CASES635=cases635()
if len(CASES635)!=72:raise RuntimeError("CASE_COUNT_FAILURE")
SCALING_SHA635=sha(CASES635)
log635("CASES",len(CASES635),"SCALING_SHA",SCALING_SHA635)
# TEST633/634 physical CUT3 adapter, without changing the frozen engine.
@torch.no_grad()
def init_disk635(obj):
 lower=obj["lower_kv"];h3=obj["h3"];n=h3.shape[1]
 if len(lower)!=CUT635+1 or h3.shape!=(1,n,H):
  raise RuntimeError("INIT_SHAPE_FAILURE")
 def inject(module,args,out):return replace(out,h3)
 hook=model.model.layers[CUT635].register_forward_hook(inject)
 try:
  pos=torch.arange(n,device=DEVICE)
  out=model(inputs_embeds=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE),
   attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),
   position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
  upper=clone(out.past_key_values)
  return tuple(lower)+tuple(upper[CUT635+1:])
 finally:hook.remove()
@torch.no_grad()
def append_disk635(memory,obj):
 lower=obj["lower_kv"];h3=obj["h3"]
 oldn=memory[0][0].shape[-2];n=h3.shape[1];q=n-P
 if len(lower)!=CUT635+1 or q<=0:raise RuntimeError("APPEND_SHAPE_FAILURE")
 body=h3[:,P:,:];pos=torch.arange(oldn,oldn+q,device=DEVICE)
 def inject(module,args,out):
  current=out[0] if isinstance(out,(tuple,list)) else out
  if current.shape!=body.shape:raise RuntimeError("INJECTION_SHAPE_FAILURE")
  return replace(out,body)
 hook=model.model.layers[CUT635].register_forward_hook(inject)
 try:
  out=model(inputs_embeds=torch.zeros((1,q,H),device=DEVICE,dtype=DTYPE),
   past_key_values=cache_build(memory),
   attention_mask=torch.ones((1,oldn+q),device=DEVICE,dtype=torch.long),
   position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
  grown=list(clone(out.past_key_values))
  oldpos=torch.arange(P,P+q,device=DEVICE)
  for li in range(CUT635+1):
   ok,ov=memory[li];k,v=lower[li]
   kk=rephase(k[:,:,P:,:],oldpos,pos)
   vv=v[:,:,P:,:]
   grown[li]=(torch.cat((ok,kk),-2),torch.cat((ov,vv),-2))
  result=tuple(grown)
  for (ok,ov),(nk,nv) in zip(memory,result):
   if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):
    raise RuntimeError("APPEND_PREFIX_BITWISE_FAILURE")
  return result
 finally:hook.remove()
MANIFEST635={}
log635("[1/5] FORGING 96 PHYSICAL MAM CARTRIDGES")
for ci in range(96):
 cp=checkpoint(ci,CUT635)
 obj=dict(lower_kv=cp["kv"][:CUT635+1],h3=cp["h"])
 path=CART635/f"C{ci:03d}.pt"
 torch.save(cpu635(obj),path)
 MANIFEST635[ci]=dict(path=str(path),sha256=filehash635(path),
  bytes=path.stat().st_size,tokens=cp["h"].shape[1])
 del cp,obj
 if (ci+1)%16==0:log635("FORGED",ci+1,"/96")
def load635(path,expected_sha):
 if filehash635(path)!=expected_sha:raise RuntimeError("FILE_INTEGRITY_FAILURE")
 return gpu635(torch.load(path,map_location="cpu",weights_only=True))
def mam635(keys):
 memory=None
 for ci in keys:
  meta=MANIFEST635[ci]
  obj=load635(Path(meta["path"]),meta["sha256"])
  memory=init_disk635(obj) if memory is None else append_disk635(memory,obj)
 return memory
def valid635(memory,keys):
 n=P+sum(MANIFEST635[ci]["tokens"]-P for ci in keys)
 if len(memory)!=NL:raise RuntimeError("LAYER_COUNT_FAILURE")
 for li,(k,v) in enumerate(memory):
  if tuple(k.shape)!=(1,KVH,n,HD) or tuple(v.shape)!=(1,KVH,n,HD):
   raise RuntimeError(f"CACHE_SHAPE_FAILURE_L{li}")
 return n
log635("[2/5] PHYSICAL MAM vs SOURCE-DEPENDENT DC BITWISE PROBES")
PROBES635=[]
for index in (0,10,35,71):
 z=CASES635[index]
 a=consolidated(z["keys"],CUT635)
 b=mam635(z["keys"])
 same=exact635(a,b)
 PROBES635.append(dict(case=index,size=z["size"],slot=z["slot"],bitwise=same))
 log635("PROBE",index,"SIZE",z["size"],"SLOT",z["slot"],"BITWISE",same)
 del a,b
 if not same:raise RuntimeError("DISK_ADAPTER_BITWISE_FAILURE")
# Forge a complete physical full-KV artifact for every multi-cartridge match.
# This is a real serialized Hugging Face KV baseline, NOT LMCache or CacheBlend.
log635("[3/5] NATIVE JOINT REFERENCE + PHYSICAL FULL KV FORGING")
ROWS635=[]
FULL_MANIFEST635={}
REFERENCE635={}
OUT635=ROOT635/"TEST635_ARENA_FINAL.json"
CSV635=ROOT635/"TEST635_ARENA_MATCHES.csv"
REPORT635=dict(test="635",status="IN_PROGRESS",round="STORAGE_VS_SPEED",
 source_sha256=SHA635,source_blob=BLOB635,panel_sha=PANEL_SHA,
 scaling_sha=SCALING_SHA635,model=MODEL_ID,
 arms=ARMS635,cut=CUT635,cases=CASES635,rows=ROWS635,
 probes=PROBES635,cartridge_manifest=MANIFEST635,
 full_manifest=FULL_MANIFEST635,summary={},
 limitations=[
  "FULL_KV_DISK is native serialized KV, not a third-party LMCache implementation.",
  "No CacheBlend, LMCache, SGLang or RAG result is claimed.",
  "72 synthetic cases: 8 target worlds, sizes 5/20/40, three positions.",
  "Native Joint uses source tokens; physical disk arms run with source bodies blocked.",
  "Full KV is forged per complete multi-cartridge case; MAM cartridges are reusable across cases.",
  "Physical file hashing is included in measured disk preparation for both disk arms.",
  "Operating-system page cache may serve reads.",
  "Model remains resident in one Python process; independent-process serving is not tested.",
  "Each answer call starts from the same prepared memory; generated KV is not reused.",
  "One-time artifact forging and model startup are excluded from per-request latency.",
  "Amortized projections assume the prepared memory can serve multiple independent questions.",
  "Reported disk bytes compare one full combined KV artifact with the sum of selected MAM files.",
  "Shared MAM cartridge storage across multiple cases is separately reported."
 ])
savejson635(OUT635,REPORT635)
def two_answers635(memory,question):
 answers=[];times=[]
 for _ in range(2):
  a,ms=timed635(lambda:answer(memory,question))
  answers.append(a);times.append(ms)
 return answers,times
for i,z in enumerate(CASES635):
 row=dict(arm="NATIVE_JOINT",case=i,world=z["world"],
  size=z["size"],slot=z["slot"],gold=z["gold"],status="STARTED")
 try:
  memory,prepare_ms=timed635(lambda:joint(z["keys"]))
  row["tokens"]=valid635(memory,z["keys"])
  path=FULL635/f"M{i:03d}.pt"
  torch.save(cpu635(memory),path)
  FULL_MANIFEST635[i]=dict(path=str(path),sha256=filehash635(path),
   bytes=path.stat().st_size,tokens=row["tokens"])
  answers,times=two_answers635(memory,z["question"])
  row.update(answer=answers[0],answer_repeat=answers[1],
   answer_equal_repeat=answers[0]==answers[1],
   correct=bool(hit(answers[0],z["gold"])),
   prepare_ms=prepare_ms,answer_ms=med635(times),
   answer_runs_ms=times,status="PASS")
  REFERENCE635[i]=answers[0]
  del memory
 except Exception as e:
  row.update(status="ERROR",error=repr(e),traceback=traceback.format_exc())
 ROWS635.append(row)
 log635("NATIVE_JOINT",f"{i+1:02d}/72","N",z["size"],
  "SLOT",z["slot"],"CORRECT",row.get("correct"),"STATUS",row["status"])
 if (i+1)%9==0:savejson635(OUT635,REPORT635)
 if row["status"]!="PASS":
  savejson635(OUT635,REPORT635)
  raise RuntimeError(f"NATIVE_REFERENCE_FAILURE_{i}")
# Block source token lists. Both physical arms must now work without source replay.
ORIGINAL635=[c["body"] for c in CAR]
for c in CAR:c["body"]=None
if any(c["body"] is not None for c in CAR):raise RuntimeError("SOURCE_BLOCK_FAILURE")
log635("[4/5] SOURCE-BLOCKED FULL KV vs MAM DISK MATCHES")
try:
 for arm in ("FULL_KV_DISK","MAM_DISK"):
  for i,z in enumerate(CASES635):
   row=dict(arm=arm,case=i,world=z["world"],
    size=z["size"],slot=z["slot"],gold=z["gold"],status="STARTED")
   try:
    if arm=="FULL_KV_DISK":
     meta=FULL_MANIFEST635[i]
     memory,prepare_ms=timed635(lambda:load635(Path(meta["path"]),meta["sha256"]))
     row["disk_bytes"]=meta["bytes"]
    else:
     memory,prepare_ms=timed635(lambda:mam635(z["keys"]))
     row["disk_bytes"]=sum(MANIFEST635[ci]["bytes"] for ci in z["keys"])
    row["tokens"]=valid635(memory,z["keys"])
    answers,times=two_answers635(memory,z["question"])
    row.update(answer=answers[0],answer_repeat=answers[1],
     answer_equal_repeat=answers[0]==answers[1],
     correct=bool(hit(answers[0],z["gold"])),
     answer_equal_native=answers[0]==REFERENCE635[i],
     prepare_ms=prepare_ms,answer_ms=med635(times),
     answer_runs_ms=times,status="PASS")
    del memory
   except Exception as e:
    row.update(status="ERROR",error=repr(e),traceback=traceback.format_exc())
   ROWS635.append(row)
   log635(arm,f"{i+1:02d}/72","N",z["size"],
    "SLOT",z["slot"],"CORRECT",row.get("correct"),
    "NATIVE_EQUAL",row.get("answer_equal_native"),"STATUS",row["status"])
   if (i+1)%9==0:savejson635(OUT635,REPORT635)
   if row["status"]!="PASS":
    savejson635(OUT635,REPORT635)
    raise RuntimeError(f"PHYSICAL_ARM_FAILURE_{arm}_{i}")
  savejson635(OUT635,REPORT635)
finally:
 for c,body in zip(CAR,ORIGINAL635):c["body"]=body
log635("[5/5] ACADEMIC SCORING / AMORTIZATION / STORAGE AUDIT")
SUMMARY635={}
for size in SIZES635:
 SUMMARY635[str(size)]={}
 for arm in ARMS635:
  subset=[r for r in ROWS635 if r["size"]==size and r["arm"]==arm]
  SUMMARY635[str(size)][arm]=dict(
   tested=len(subset),
   correct=sum(int(r.get("correct",False)) for r in subset),
   repeat_stable=sum(int(r.get("answer_equal_repeat",False)) for r in subset),
   native_answer_equal=sum(int(r.get("answer_equal_native",False)) for r in subset)
    if arm!="NATIVE_JOINT" else len(subset),
   median_tokens=med635([r["tokens"] for r in subset]),
   median_prepare_ms=med635([r["prepare_ms"] for r in subset]),
   median_answer_ms=med635([r["answer_ms"] for r in subset]),
   median_total_one_query_ms=med635([r["prepare_ms"]+r["answer_ms"] for r in subset]),
   median_disk_mib=med635([mib635(r["disk_bytes"]) for r in subset if "disk_bytes" in r]),
   by_slot={slot:dict(
    correct=sum(int(r.get("correct",False)) for r in subset if r["slot"]==slot),
    tested=sum(r["slot"]==slot for r in subset)) for slot in SLOTS635},
   projected_total_ms={
    str(q):med635([r["prepare_ms"]+q*r["answer_ms"] for r in subset])
    for q in (1,5,20)})
PAIRS635=[]
INDEX635={(r["arm"],r["case"]):r for r in ROWS635}
for i,z in enumerate(CASES635):
 native=INDEX635[("NATIVE_JOINT",i)]
 full=INDEX635[("FULL_KV_DISK",i)]
 mam=INDEX635[("MAM_DISK",i)]
 PAIRS635.append(dict(case=i,size=z["size"],slot=z["slot"],
  native_correct=native["correct"],full_correct=full["correct"],
  mam_correct=mam["correct"],
  full_answer_equal=native["answer"]==full["answer"],
  mam_answer_equal=native["answer"]==mam["answer"],
  full_bytes=full["disk_bytes"],mam_bytes=mam["disk_bytes"],
  mam_storage_reduction_pct=round(
   100*(1-mam["disk_bytes"]/full["disk_bytes"]),5)))
ALL_CARTRIDGE_BYTES635=sum(x["bytes"] for x in MANIFEST635.values())
ALL_FULL_BYTES635=sum(x["bytes"] for x in FULL_MANIFEST635.values())
REPORT635["summary"]=SUMMARY635
REPORT635["pairs"]=PAIRS635
REPORT635["native_full_answer_equal"]=sum(int(p["full_answer_equal"]) for p in PAIRS635)
REPORT635["native_mam_answer_equal"]=sum(int(p["mam_answer_equal"]) for p in PAIRS635)
REPORT635["mean_mam_storage_reduction_pct"]=round(
 statistics.mean(p["mam_storage_reduction_pct"] for p in PAIRS635),5)
REPORT635["unique_mam_cartridge_disk_bytes"]=ALL_CARTRIDGE_BYTES635
REPORT635["all_72_full_kv_disk_bytes"]=ALL_FULL_BYTES635
REPORT635["unique_mam_cartridge_disk_mib"]=mib635(ALL_CARTRIDGE_BYTES635)
REPORT635["all_72_full_kv_disk_mib"]=mib635(ALL_FULL_BYTES635)
REPORT635["model_unchanged"]=sentinel()==S0
REPORT635["total_seconds"]=round(time.perf_counter()-T635,3)
REPORT635["status"]="COMPLETE" if (
 len(ROWS635)==216 and
 all(r["status"]=="PASS" for r in ROWS635) and
 all(r["correct"] for r in ROWS635) and
 REPORT635["model_unchanged"]) else "COMPLETE_WITH_FINDINGS"
savejson635(OUT635,REPORT635)
with open(CSV635,"w",newline="",encoding="utf-8") as f:
 fields=["arm","case","world","size","slot","status","correct",
  "answer_equal_native","answer_equal_repeat","gold","answer",
  "tokens","prepare_ms","answer_ms","disk_bytes"]
 writer=csv.DictWriter(f,fieldnames=fields)
 writer.writeheader()
 for r in ROWS635:writer.writerow({k:r.get(k) for k in fields})
log635("="*120)
log635("TEST635_STATUS",REPORT635["status"])
log635("MATCHES",len(PAIRS635),"/72","ARM_RUNS",len(ROWS635),"/216")
log635("FULL_NATIVE_ANSWER_EQUAL",REPORT635["native_full_answer_equal"],"/72")
log635("MAM_NATIVE_ANSWER_EQUAL",REPORT635["native_mam_answer_equal"],"/72")
log635("MEAN_MAM_STORAGE_REDUCTION_PCT",REPORT635["mean_mam_storage_reduction_pct"])
log635("UNIQUE_MAM_DISK_MIB",REPORT635["unique_mam_cartridge_disk_mib"])
log635("ALL_FULL_KV_DISK_MIB",REPORT635["all_72_full_kv_disk_mib"])
log635("MODEL_UNCHANGED",REPORT635["model_unchanged"])
for size in SIZES635:
 for arm in ARMS635:
  s=SUMMARY635[str(size)][arm]
  log635("SIZE",size,"ARM",arm,
   "CORRECT",f'{s["correct"]}/{s["tested"]}',
   "PREPARE_MS",s["median_prepare_ms"],
   "ANSWER_MS",s["median_answer_ms"],
   "TOTAL_1Q_MS",s["median_total_one_query_ms"],
   "PROJECTED_5Q_MS",s["projected_total_ms"]["5"],
   "DISK_MIB",s["median_disk_mib"])
log635("JSON_OUTPUT",OUT635)
log635("CSV_OUTPUT",CSV635)
log635("TOTAL_SECONDS",REPORT635["total_seconds"])
log635("="*120)

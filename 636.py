
# ======================================================================================================================
# TEST636 — AKBASCORE MEMORY ARENA · ROUND 2 · LATENCY BOTTLENECK AUDIT
# TEST635 LOCKED BACKBONE / TEST566 SHA-VERIFIED ENGINE
# PHYSICAL MAM CUT3 · COMPONENT TIMINGS · CUDA EVENT TIMINGS · NO SOURCE REPLAY
# 5 / 20 / 40 CARTRIDGES × FIRST / MIDDLE / LAST × 3 WORLDS = 27 MATCHES
# NO ARCHITECTURE CHANGE · NO TRAINING · NO THIRD-PARTY COMPETITOR CLAIM
# ======================================================================================================================
import os,time,json,csv,hashlib,urllib.request,random,statistics,traceback,gc
from pathlib import Path
import torch,numpy as np
R636=Path("/content/AKBASCORE_TEST636")
D636=R636/"cartridges"
D636.mkdir(parents=True,exist_ok=True)
URL636="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/566.py"
SHA636="0155cda553e5f08ed2bec53611e49898a5ca24e0d270c3a1274e661bc205bdcd"
BLOB636="b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71"
PANEL636="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
BOUND636="RESULTS=[];RAW=[];ERRORS=[]"
CUT636=3
SIZES636=(5,20,40)
SLOTS636=("FIRST","MIDDLE","LAST")
WORLDS636=(0,3,7)
TSTART636=time.perf_counter()
def out636(*x):print(*x,flush=True)
def digest636(b):return hashlib.sha256(b).hexdigest()
def blob636(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def filehash636(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for part in iter(lambda:f.read(4194304),b""):h.update(part)
 return h.hexdigest()
def fetch636():
 for p in (Path("/content/566.py"),Path("/content/TEST566.py")):
  if p.is_file():
   b=p.read_bytes()
   if digest636(b)==SHA636 and blob636(b)==BLOB636:return b,"LOCAL_VERIFIED"
 req=urllib.request.Request(URL636,headers={"User-Agent":"AKBASCORE-TEST636"})
 b=urllib.request.urlopen(req,timeout=120).read()
 if digest636(b)!=SHA636 or blob636(b)!=BLOB636:raise RuntimeError("SOURCE_LOCK_FAILURE")
 return b,"GITHUB_VERIFIED"
def sync636():torch.cuda.synchronize()
def wall636(fn):
 sync636();t=time.perf_counter()
 result=fn()
 sync636()
 return result,(time.perf_counter()-t)*1000
def cuda636(fn):
 sync636()
 start=torch.cuda.Event(enable_timing=True)
 end=torch.cuda.Event(enable_timing=True)
 start.record()
 result=fn()
 end.record();end.synchronize()
 return result,start.elapsed_time(end)
def med636(x):return round(statistics.median(x),5) if x else None
def cpu636(x):
 if torch.is_tensor(x):return x.detach().cpu().contiguous()
 if isinstance(x,tuple):return tuple(cpu636(v) for v in x)
 if isinstance(x,list):return [cpu636(v) for v in x]
 if isinstance(x,dict):return {k:cpu636(v) for k,v in x.items()}
 raise TypeError("ARTIFACT_TYPE_FAILURE")
def gpu636(x):
 if torch.is_tensor(x):return x.to(DEVICE)
 if isinstance(x,tuple):return tuple(gpu636(v) for v in x)
 if isinstance(x,list):return [gpu636(v) for v in x]
 if isinstance(x,dict):return {k:gpu636(v) for k,v in x.items()}
 raise TypeError("ARTIFACT_TYPE_FAILURE")
def same636(a,b):
 if torch.is_tensor(a):
  return torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype and bool(torch.equal(a,b))
 if isinstance(a,(tuple,list)):
  return isinstance(b,(tuple,list)) and len(a)==len(b) and all(same636(x,y) for x,y in zip(a,b))
 return a==b
def save636(path,obj):
 tmp=path.with_suffix(".tmp")
 with open(tmp,"w",encoding="utf-8") as f:
  json.dump(obj,f,indent=2,ensure_ascii=False,default=str,allow_nan=False)
 os.replace(tmp,path)
out636("="*120)
out636("TEST636 — MEMORY ARENA · ROUND 2 · LATENCY BOTTLENECK AUDIT")
out636("="*120)
if not torch.cuda.is_available():raise RuntimeError("CUDA_REQUIRED")
raw636,source636=fetch636()
text636=raw636.decode("utf-8")
if text636.count(BOUND636)!=1:raise RuntimeError("BOOTSTRAP_LOCK_FAILURE")
out636("SOURCE",source636,"SHA256",digest636(raw636),"BLOB",blob636(raw636))
exec(compile(text636.split(BOUND636,1)[0],"TEST566_LOCKED.py","exec"),globals())
if PANEL_SHA!=PANEL636 or len(CAR)!=96 or P!=29:raise RuntimeError("PANEL_LOCK_FAILURE")
if (NL,H,QH,KVH,HD)!=(28,3584,28,4,128):raise RuntimeError("ARCH_LOCK_FAILURE")
if sentinel()!=S0:raise RuntimeError("MODEL_LOCK_FAILURE")
def makecases636():
 rng=random.Random(634552)
 cases=[]
 # Preserve TEST634/635 distractor generation for all 8 worlds.
 for wi in range(8):
  target=PANEL[wi]["target"]
  pool=[ci for ci in range(96) if CAR[ci]["world"]!=wi]
  rng.shuffle(pool)
  others39=pool[:39]
  if wi not in WORLDS636:continue
  typ=CAR[target]["type"]
  for n in SIZES636:
   others=others39[:n-1]
   for slot in SLOTS636:
    if slot=="FIRST":keys=[target]+others
    elif slot=="LAST":keys=others+[target]
    else:
     k=(n-1)//2
     keys=others[:k]+[target]+others[k:]
    cases.append(dict(world=wi,size=n,slot=slot,keys=keys,
     target=target,question=W[wi][typ],gold=CAR[target]["gold"]))
 return cases
CASES636=makecases636()
if len(CASES636)!=27:raise RuntimeError("CASE_COUNT_FAILURE")
MANIFEST636={}
out636("[1/5] FORGING PHYSICAL CARTRIDGES")
NEEDED636=sorted(set(ci for z in CASES636 for ci in z["keys"]))
for index,ci in enumerate(NEEDED636):
 cp=checkpoint(ci,CUT636)
 obj=dict(lower_kv=cp["kv"][:CUT636+1],h3=cp["h"])
 path=D636/f"C{ci:03d}.pt"
 torch.save(cpu636(obj),path)
 MANIFEST636[ci]=dict(path=str(path),sha256=filehash636(path),
  bytes=path.stat().st_size,tokens=cp["h"].shape[1])
 del cp,obj
 if (index+1)%16==0 or index+1==len(NEEDED636):
  out636("FORGED",index+1,"/",len(NEEDED636))
def read636(ci,trace):
 meta=MANIFEST636[ci];p=Path(meta["path"])
 t=time.perf_counter()
 actual=filehash636(p)
 trace["hash_ms"]+=(time.perf_counter()-t)*1000
 if actual!=meta["sha256"]:raise RuntimeError(f"SHA_FAILURE_{ci}")
 sync636();t=time.perf_counter()
 obj=torch.load(p,map_location="cpu",weights_only=True)
 trace["torch_load_ms"]+=(time.perf_counter()-t)*1000
 sync636();t=time.perf_counter()
 obj=gpu636(obj)
 sync636()
 trace["host_to_device_ms"]+=(time.perf_counter()-t)*1000
 return obj
@torch.no_grad()
def init636(obj,trace):
 lower=obj["lower_kv"];h3=obj["h3"];n=h3.shape[1]
 if len(lower)!=CUT636+1 or h3.shape!=(1,n,H):
  raise RuntimeError("INIT_SHAPE_FAILURE")
 def inject(module,args,out):return replace(out,h3)
 hook=model.model.layers[CUT636].register_forward_hook(inject)
 try:
  pos=torch.arange(n,device=DEVICE)
  def forward():
   return model(inputs_embeds=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE),
    attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),
    position_ids=pos.unsqueeze(0),cache_position=pos,
    use_cache=True,return_dict=True)
  result,ms=wall636(forward)
  trace["model_forward_ms"]+=ms
  upper=clone(result.past_key_values)
  return tuple(lower)+tuple(upper[CUT636+1:])
 finally:hook.remove()
@torch.no_grad()
def append636(memory,obj,trace):
 lower=obj["lower_kv"];h3=obj["h3"]
 oldn=memory[0][0].shape[-2];n=h3.shape[1];q=n-P
 if len(lower)!=CUT636+1 or q<=0:
  raise RuntimeError("APPEND_SHAPE_FAILURE")
 body=h3[:,P:,:]
 pos=torch.arange(oldn,oldn+q,device=DEVICE)
 def inject(module,args,out):
  current=out[0] if isinstance(out,(tuple,list)) else out
  if current.shape!=body.shape:raise RuntimeError("INJECTION_SHAPE_FAILURE")
  return replace(out,body)
 hook=model.model.layers[CUT636].register_forward_hook(inject)
 try:
  def forward():
   return model(inputs_embeds=torch.zeros((1,q,H),device=DEVICE,dtype=DTYPE),
    past_key_values=cache_build(memory),
    attention_mask=torch.ones((1,oldn+q),device=DEVICE,dtype=torch.long),
    position_ids=pos.unsqueeze(0),cache_position=pos,
    use_cache=True,return_dict=True)
  result,ms=wall636(forward)
  trace["model_forward_ms"]+=ms
  grown=list(clone(result.past_key_values))
  oldpos=torch.arange(P,P+q,device=DEVICE)
  sync636();t=time.perf_counter()
  for li in range(CUT636+1):
   ok,ov=memory[li];k,v=lower[li]
   kk=rephase(k[:,:,P:,:],oldpos,pos)
   vv=v[:,:,P:,:]
   grown[li]=(torch.cat((ok,kk),-2),torch.cat((ov,vv),-2))
  result=tuple(grown)
  sync636()
  trace["lower_rephase_concat_ms"]+=(time.perf_counter()-t)*1000
  sync636();t=time.perf_counter()
  for (ok,ov),(nk,nv) in zip(memory,result):
   if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):
    raise RuntimeError("APPEND_PREFIX_BITWISE_FAILURE")
  sync636()
  trace["prefix_audit_ms"]+=(time.perf_counter()-t)*1000
  return result
 finally:hook.remove()
def trace_new636():
 return dict(hash_ms=0.,torch_load_ms=0.,host_to_device_ms=0.,
  model_forward_ms=0.,lower_rephase_concat_ms=0.,prefix_audit_ms=0.)
def mam636(keys,trace):
 memory=None
 for ci in keys:
  obj=read636(ci,trace)
  memory=init636(obj,trace) if memory is None else append636(memory,obj,trace)
 return memory
def check636(memory,keys):
 n=P+sum(MANIFEST636[ci]["tokens"]-P for ci in keys)
 if len(memory)!=NL:raise RuntimeError("LAYER_FAILURE")
 for li,(k,v) in enumerate(memory):
  if k.shape!=(1,KVH,n,HD) or v.shape!=(1,KVH,n,HD):
   raise RuntimeError(f"KV_SHAPE_FAILURE_{li}")
 return n
out636("[2/5] SOURCE-DEPENDENT vs PHYSICAL BITWISE PROBES")
PROBES636=[]
for idx in (0,13,26):
 z=CASES636[idx]
 reference=consolidated(z["keys"],CUT636)
 trace=trace_new636()
 physical=mam636(z["keys"],trace)
 equal=same636(reference,physical)
 PROBES636.append(dict(case=idx,size=z["size"],slot=z["slot"],bitwise=equal))
 out636("PROBE",idx,"N",z["size"],"SLOT",z["slot"],"BITWISE",equal)
 del reference,physical
 if not equal:raise RuntimeError("PHYSICAL_EQUIVALENCE_FAILURE")
out636("[3/5] NATIVE JOINT BASELINE")
BASE636={}
for idx,z in enumerate(CASES636):
 mem,prepare=wall636(lambda:joint(z["keys"]))
 ans,answer_ms=wall636(lambda:answer(mem,z["question"]))
 BASE636[idx]=dict(answer=ans,correct=bool(hit(ans,z["gold"])),
  prepare_ms=prepare,answer_ms=answer_ms,tokens=check636(mem,z["keys"]))
 del mem
 out636("NATIVE",f"{idx+1:02d}/27","N",z["size"],
  "CORRECT",BASE636[idx]["correct"])
 if not BASE636[idx]["correct"]:
  raise RuntimeError(f"NATIVE_REFERENCE_FAILURE_{idx}")
OUT636=R636/"TEST636_BOTTLENECK_FINAL.json"
CSV636=R636/"TEST636_BOTTLENECK_ROWS.csv"
ROWS636=[]
REPORT636=dict(test="636",status="IN_PROGRESS",source_sha256=SHA636,
 panel_sha=PANEL_SHA,cut=CUT636,model=MODEL_ID,
 cases=CASES636,probes=PROBES636,rows=ROWS636,
 baseline=BASE636,manifest=MANIFEST636,
 limitations=[
  "The physical adapter preserves the tested CUT3 architecture.",
  "Component times are synchronized wall-clock measurements, not exclusive CPU times.",
  "Component timing includes Python, PyTorch dispatch, and CUDA execution.",
  "Nested components may not sum exactly to total elapsed time.",
  "Torch-load and host-to-device transfers are separately measured.",
  "CUDA kernel-only attribution requires a dedicated profiler.",
  "The 27 cases are synthetic and limited to three target worlds.",
  "No C++ or custom CUDA kernel has been implemented or benchmarked.",
  "No CacheBlend, LMCache, SGLang, vLLM or RAG competitor is benchmarked.",
  "Operating-system file caching and thermal variation may affect timings.",
  "Source bodies are blocked only during the physical MAM inference stage."
 ])
save636(OUT636,REPORT636)
ORIGINAL636=[c["body"] for c in CAR]
for c in CAR:c["body"]=None
out636("[4/5] SOURCE-BLOCKED COMPONENT PROFILING")
try:
 for idx,z in enumerate(CASES636):
  row=dict(case=idx,world=z["world"],size=z["size"],slot=z["slot"],
   status="STARTED")
  try:
   trace=trace_new636()
   sync636()
   torch.cuda.reset_peak_memory_stats()
   start=time.perf_counter()
   memory=mam636(z["keys"],trace)
   sync636()
   prepare_ms=(time.perf_counter()-start)*1000
   tokens=check636(memory,z["keys"])
   peak=torch.cuda.max_memory_allocated()
   ans,answer_ms=wall636(lambda:answer(memory,z["question"]))
   components=sum(trace.values())
   row.update(status="PASS",tokens=tokens,answer=ans,
    correct=bool(hit(ans,z["gold"])),
    native_answer_equal=ans==BASE636[idx]["answer"],
    prepare_ms=round(prepare_ms,5),answer_ms=round(answer_ms,5),
    measured_component_sum_ms=round(components,5),
    residual_ms=round(prepare_ms-components,5),
    peak_allocated_mib=round(peak/1048576,5),
    disk_bytes=sum(MANIFEST636[ci]["bytes"] for ci in z["keys"]),
    **{k:round(v,5) for k,v in trace.items()})
   del memory
  except Exception as e:
   row.update(status="ERROR",error=repr(e),traceback=traceback.format_exc())
  ROWS636.append(row)
  out636("MAM",f"{idx+1:02d}/27","N",z["size"],
   "SLOT",z["slot"],"STATUS",row["status"],
   "CORRECT",row.get("correct"),
   "PREPARE_MS",row.get("prepare_ms"),
   "FORWARD_MS",row.get("model_forward_ms"))
  if (idx+1)%3==0:save636(OUT636,REPORT636)
  if row["status"]!="PASS":
   save636(OUT636,REPORT636)
   raise RuntimeError(f"TEST636_FAILURE_{idx}")
finally:
 for c,body in zip(CAR,ORIGINAL636):c["body"]=body
out636("[5/5] BOTTLENECK SUMMARY")
COMPONENTS636=("hash_ms","torch_load_ms","host_to_device_ms",
 "model_forward_ms","lower_rephase_concat_ms","prefix_audit_ms","residual_ms")
SUMMARY636={}
for n in SIZES636:
 subset=[r for r in ROWS636 if r["size"]==n]
 total=med636([r["prepare_ms"] for r in subset])
 summary=dict(cases=len(subset),correct=sum(int(r["correct"]) for r in subset),
  native_answer_equal=sum(int(r["native_answer_equal"]) for r in subset),
  median_tokens=med636([r["tokens"] for r in subset]),
  native_prepare_median_ms=med636(
   [BASE636[r["case"]]["prepare_ms"] for r in subset]),
  mam_prepare_median_ms=total,
  mam_answer_median_ms=med636([r["answer_ms"] for r in subset]),
  median_peak_allocated_mib=med636([r["peak_allocated_mib"] for r in subset]))
 for component in COMPONENTS636:
  summary[component]=med636([r[component] for r in subset])
 SUMMARY636[str(n)]=summary
REPORT636["summary"]=SUMMARY636
REPORT636["model_unchanged"]=sentinel()==S0
REPORT636["total_seconds"]=round(time.perf_counter()-TSTART636,3)
REPORT636["status"]="COMPLETE" if (
 len(ROWS636)==27 and all(r["status"]=="PASS" and r["correct"] for r in ROWS636)
 and REPORT636["model_unchanged"]) else "COMPLETE_WITH_FINDINGS"
save636(OUT636,REPORT636)
with open(CSV636,"w",newline="",encoding="utf-8") as f:
 fields=["case","world","size","slot","status","correct",
  "native_answer_equal","tokens","prepare_ms","answer_ms",
  "hash_ms","torch_load_ms","host_to_device_ms",
  "model_forward_ms","lower_rephase_concat_ms","prefix_audit_ms",
  "residual_ms","peak_allocated_mib","disk_bytes","answer"]
 writer=csv.DictWriter(f,fieldnames=fields)
 writer.writeheader()
 for r in ROWS636:writer.writerow({k:r.get(k) for k in fields})
out636("="*120)
out636("TEST636_STATUS",REPORT636["status"])
out636("ROWS",len(ROWS636),"/27")
out636("MODEL_UNCHANGED",REPORT636["model_unchanged"])
for n in SIZES636:
 s=SUMMARY636[str(n)]
 out636("SIZE",n,"CORRECT",f'{s["correct"]}/{s["cases"]}',
  "NATIVE_EQUAL",f'{s["native_answer_equal"]}/{s["cases"]}')
 out636("  NATIVE_PREPARE_MS",s["native_prepare_median_ms"],
  "MAM_PREPARE_MS",s["mam_prepare_median_ms"])
 for component in COMPONENTS636:
  out636(" ",component.upper(),s[component])
out636("JSON_OUTPUT",OUT636)
out636("CSV_OUTPUT",CSV636)
out636("TOTAL_SECONDS",REPORT636["total_seconds"])
out636("="*120)

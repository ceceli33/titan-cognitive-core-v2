
# ======================================================================================================================
# TEST633 — AKBASCORE MAM · 72 MULTI-CARTRIDGE CASES / PHYSICAL CARTRIDGE CONSOLIDATION
# LOCKED TEST566 SOURCE · SAME 24 QUESTIONS × FIRST/MIDDLE/LAST
# ARMS: JOINT / INDEP / DC_SOURCE / NOCROSS / MAM_DISK
# MAM_DISK: FORGE ONCE -> PHYSICAL .PT -> BLOCK SOURCE BODIES -> INIT+APPEND FROM CARTRIDGES
# GPU SYNCHRONIZED TIMING · EXACT ORIGINAL ANSWER() · NO TRAINING · NO ARCHITECTURAL CHANGES
# ======================================================================================================================
import os,sys,time,json,csv,hashlib,urllib.request,statistics,gc,traceback
from pathlib import Path
import torch,numpy as np
ROOT633=Path("/content/AKBASCORE_TEST633")
DISK633=ROOT633/"cartridges"
DISK633.mkdir(parents=True,exist_ok=True)
URL633="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/566.py"
SHA633="0155cda553e5f08ed2bec53611e49898a5ca24e0d270c3a1274e661bc205bdcd"
BLOB633="b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71"
PANEL633="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
BOUND633="RESULTS=[];RAW=[];ERRORS=[]"
CUT633=3
ARMS633=("JOINT","INDEP","DC_SOURCE","NOCROSS","MAM_DISK")
T633=time.perf_counter()
def sha633(b):return hashlib.sha256(b).hexdigest()
def blob633(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def filehash633(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()
def fetch633():
 for p in (Path("/content/566.py"),Path("/content/TEST566.py")):
  if p.is_file():
   b=p.read_bytes()
   if sha633(b)==SHA633 and blob633(b)==BLOB633:return b,"LOCAL_VERIFIED"
 req=urllib.request.Request(URL633,headers={"User-Agent":"AKBASCORE-TEST633"})
 b=urllib.request.urlopen(req,timeout=120).read()
 if sha633(b)!=SHA633 or blob633(b)!=BLOB633:raise RuntimeError("SOURCE_LOCK_FAILURE")
 return b,"GITHUB_VERIFIED"
def sync633():torch.cuda.synchronize()
def timed633(fn):
 sync633();t=time.perf_counter();r=fn();sync633()
 return r,(time.perf_counter()-t)*1000
def cpu633(x):
 if torch.is_tensor(x):return x.detach().cpu().contiguous()
 if isinstance(x,tuple):return tuple(cpu633(v) for v in x)
 if isinstance(x,list):return [cpu633(v) for v in x]
 if isinstance(x,dict):return {k:cpu633(v) for k,v in x.items()}
 raise TypeError("INVALID_ARTIFACT")
def gpu633(x):
 if torch.is_tensor(x):return x.to(DEVICE)
 if isinstance(x,tuple):return tuple(gpu633(v) for v in x)
 if isinstance(x,list):return [gpu633(v) for v in x]
 if isinstance(x,dict):return {k:gpu633(v) for k,v in x.items()}
 raise TypeError("INVALID_ARTIFACT")
def bytes633(x):
 if torch.is_tensor(x):return x.numel()*x.element_size()
 if isinstance(x,(tuple,list)):return sum(bytes633(v) for v in x)
 if isinstance(x,dict):return sum(bytes633(v) for v in x.values())
 return 0
def exact633(a,b):
 if torch.is_tensor(a):
  return torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype and bool(torch.equal(a,b))
 if isinstance(a,(tuple,list)):
  return isinstance(b,(tuple,list)) and len(a)==len(b) and all(exact633(x,y) for x,y in zip(a,b))
 return a==b
def write633(path,obj):
 t=path.with_suffix(".tmp")
 with open(t,"w",encoding="utf-8") as f:
  json.dump(obj,f,indent=2,ensure_ascii=False,default=str,allow_nan=False)
 os.replace(t,path)
def median633(a):return round(statistics.median(a),5) if a else None
print("="*120,flush=True)
print("TEST633 — AKBASCORE MAM · 72 MULTI-CARTRIDGE CASES / PHYSICAL CONSOLIDATION",flush=True)
print("="*120,flush=True)
if not torch.cuda.is_available():raise RuntimeError("CUDA_REQUIRED")
raw633,origin633=fetch633()
src633=raw633.decode("utf-8")
if src633.count(BOUND633)!=1:raise RuntimeError("SOURCE_BOUNDARY_FAILURE")
print("SOURCE",origin633,"SHA256",sha633(raw633),"BLOB",blob633(raw633),flush=True)
exec(compile(src633.split(BOUND633,1)[0],"TEST566_LOCKED.py","exec"),globals())
if PANEL_SHA!=PANEL633 or len(PANEL)!=24 or len(CAR)!=96 or P!=29:
 raise RuntimeError("PANEL_LOCK_FAILURE")
if (NL,H,QH,KVH,HD)!=(28,3584,28,4,128):raise RuntimeError("ARCH_LOCK_FAILURE")
if sentinel()!=S0:raise RuntimeError("MODEL_SENTINEL_FAILURE")
# Preserve exact TEST566 model operations. Physical consolidation is a separate adapter.
@torch.no_grad()
def init_disk633(obj):
 lower=obj["lower_kv"];h3=obj["h3"];n=h3.shape[1]
 if len(lower)!=CUT633+1 or tuple(h3.shape)!=(1,n,H):
  raise RuntimeError("INIT_ARTIFACT_SHAPE_FAILURE")
 def inject(module,args,out):return replace(out,h3)
 hook=model.model.layers[CUT633].register_forward_hook(inject)
 try:
  pos=torch.arange(n,device=DEVICE)
  out=model(inputs_embeds=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE),
   attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),
   position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
  upper=clone(out.past_key_values)
  return tuple(lower)+tuple(upper[CUT633+1:])
 finally:hook.remove()
@torch.no_grad()
def append_disk633(memory,obj):
 lower=obj["lower_kv"];h3=obj["h3"]
 oldn=memory[0][0].shape[-2]
 n=h3.shape[1];q=n-P
 if len(lower)!=CUT633+1 or q<=0:raise RuntimeError("APPEND_ARTIFACT_SHAPE_FAILURE")
 body=h3[:,P:,:]
 pos=torch.arange(oldn,oldn+q,device=DEVICE)
 def inject(module,args,out):
  current=out[0] if isinstance(out,(tuple,list)) else out
  if current.shape!=body.shape:raise RuntimeError("APPEND_INJECTION_SHAPE_FAILURE")
  return replace(out,body)
 hook=model.model.layers[CUT633].register_forward_hook(inject)
 try:
  out=model(inputs_embeds=torch.zeros((1,q,H),device=DEVICE,dtype=DTYPE),
   past_key_values=cache_build(memory),
   attention_mask=torch.ones((1,oldn+q),device=DEVICE,dtype=torch.long),
   position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
  grown=list(clone(out.past_key_values))
  oldpos=torch.arange(P,P+q,device=DEVICE)
  for li in range(CUT633+1):
   ok,ov=memory[li];k,v=lower[li]
   k=rephase(k[:,:,P:,:],oldpos,pos)
   v=v[:,:,P:,:]
   grown[li]=(torch.cat((ok,k),-2),torch.cat((ov,v),-2))
  result=tuple(grown)
  for (ok,ov),(nk,nv) in zip(memory,result):
   if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):
    raise RuntimeError("DISK_APPEND_PREFIX_BITWISE_FAILURE")
  return result
 finally:hook.remove()
MANIFEST633={}
print("[1/4] FORGING PHYSICAL CARTRIDGES",flush=True)
# Every source cartridge is forged once; disk reload later does not call checkpoint().
for ci633 in range(96):
 cp633=checkpoint(ci633,CUT633)
 obj633=dict(lower_kv=cp633["kv"][:CUT633+1],h3=cp633["h"])
 path633=DISK633/f"C{ci633:03d}.pt"
 torch.save(cpu633(obj633),path633)
 MANIFEST633[ci633]=dict(path=str(path633),sha256=filehash633(path633),
  bytes=path633.stat().st_size,tokens=cp633["h"].shape[1])
 del cp633,obj633
 if (ci633+1)%16==0:print("FORGED",ci633+1,"/96",flush=True)
def load_disk633(ci):
 meta=MANIFEST633[ci];p=Path(meta["path"])
 if filehash633(p)!=meta["sha256"]:raise RuntimeError(f"CARTRIDGE_SHA_FAILURE_{ci}")
 return gpu633(torch.load(p,map_location="cpu",weights_only=True))
def disk_consolidated633(keys):
 memory=None
 for ci in keys:
  obj=load_disk633(ci)
  memory=init_disk633(obj) if memory is None else append_disk633(memory,obj)
 return memory
def valid_length633(memory,keys):
 n=P+sum(MANIFEST633[ci]["tokens"]-P for ci in keys)
 if len(memory)!=NL:raise RuntimeError("LAYER_COUNT_FAILURE")
 for li,(k,v) in enumerate(memory):
  if tuple(k.shape)!=(1,KVH,n,HD) or tuple(v.shape)!=(1,KVH,n,HD):
   raise RuntimeError(f"KV_SHAPE_FAILURE_{li}")
 return n
def run633(arm,keys):
 if arm=="JOINT":return joint(keys)
 if arm=="INDEP":return independent(keys,CUT633,"INDEP")
 if arm=="DC_SOURCE":return consolidated(keys,CUT633)
 if arm=="NOCROSS":return independent(keys,CUT633,"NOCROSS")
 if arm=="MAM_DISK":return disk_consolidated633(keys)
 raise ValueError(arm)
# First validate disk implementation against original source-dependent DC, with original tokens still available.
print("[2/4] DISK ADAPTER EQUIVALENCE PROBES",flush=True)
PROBES633=[(0,"FIRST"),(7,"MIDDLE"),(15,"LAST"),(23,"FIRST")]
PROBE_ROWS633=[]
for wi633,slot633 in PROBES633:
 keys633=ordered(PANEL[wi633],slot633)
 a633=consolidated(keys633,CUT633)
 b633=disk_consolidated633(keys633)
 equal633=exact633(a633,b633)
 PROBE_ROWS633.append(dict(case=wi633,slot=slot633,bitwise_equal=equal633))
 print("PROBE",wi633,slot633,"BITWISE",equal633,flush=True)
 del a633,b633
 if not equal633:raise RuntimeError("PHYSICAL_CONSOLIDATION_NOT_EQUIVALENT")
# Source-dependent baseline arms are run before source tokens are blocked.
ROWS633=[]
OUT633=ROOT633/"TEST633_MULTI_CARTRIDGE.json"
CSV633=ROOT633/"TEST633_DEMO_COMPARISON.csv"
REPORT633=dict(test="633",status="IN_PROGRESS",model=MODEL_ID,
 source_sha256=SHA633,panel_sha256=PANEL_SHA,
 cut=CUT633,cartridges=96,questions=24,positions=SLOTS,
 arms=ARMS633,probe_results=PROBE_ROWS633,rows=ROWS633,
 limitations=[
  "This is the original 24-case five-cartridge synthetic panel.",
  "Physical MAM consolidation is source-blocked, but uses the model already resident in the same process.",
  "Native and source-dependent controls are executed before source access is blocked.",
  "No claim of independent-process multi-cartridge consolidation is made.",
  "No third-party cache implementation is included.",
  "Measured preparation includes physical file verification for MAM_DISK.",
  "Disk access may be served by operating-system page cache.",
  "This benchmark is not a cold-start or end-to-end distributed serving benchmark."
 ])
write633(OUT633,REPORT633)
print("[3/4] ORIGINAL SOURCE-DEPENDENT CONTROL ARMS",flush=True)
for arm633 in ("JOINT","INDEP","DC_SOURCE","NOCROSS"):
 for wi633 in range(24):
  z633=PANEL[wi633];typ633=CAR[z633["target"]]["type"]
  q633=W[wi633][typ633];gold633=CAR[z633["target"]]["gold"]
  for slot633 in SLOTS:
   keys633=ordered(z633,slot633)
   row633=dict(arm=arm633,case=wi633,slot=slot633,keys=keys633,
    gold=gold633,status="STARTED")
   try:
    mem633,prep633=timed633(lambda:run633(arm633,keys633))
    row633["tokens"]=valid_length633(mem633,keys633)
    ans633,answer_ms633=timed633(lambda:answer(mem633,q633))
    row633.update(answer=ans633,correct=bool(hit(ans633,gold633)),
     prepare_ms=round(prep633,5),answer_ms=round(answer_ms633,5),status="PASS")
    del mem633
   except Exception as e:
    row633.update(status="ERROR",error=repr(e),traceback=traceback.format_exc())
   ROWS633.append(row633)
   print(f"{arm633:10s} CASE={wi633:02d} SLOT={slot633:6s} STATUS={row633['status']} CORRECT={row633.get('correct')}",flush=True)
   if row633["status"]=="ERROR":
    write633(OUT633,REPORT633)
    raise RuntimeError(f"TEST633_CONTROL_FAILURE_{arm633}_{wi633}_{slot633}")
  write633(OUT633,REPORT633)
# Source tokens are blocked before MAM_DISK evaluation.
ORIGINAL633=[c["body"] for c in CAR]
for c in CAR:c["body"]=None
if any(c["body"] is not None for c in CAR):raise RuntimeError("SOURCE_BLOCK_FAILURE")
print("[4/4] SOURCE-BLOCKED PHYSICAL MAM CONSOLIDATION",flush=True)
try:
 for wi633 in range(24):
  z633=PANEL[wi633];typ633=CAR[z633["target"]]["type"]
  q633=W[wi633][typ633];gold633=CAR[z633["target"]]["gold"]
  for slot633 in SLOTS:
   keys633=ordered(z633,slot633)
   row633=dict(arm="MAM_DISK",case=wi633,slot=slot633,keys=keys633,
    gold=gold633,status="STARTED")
   try:
    mem633,prep633=timed633(lambda:disk_consolidated633(keys633))
    row633["tokens"]=valid_length633(mem633,keys633)
    ans633,answer_ms633=timed633(lambda:answer(mem633,q633))
    row633.update(answer=ans633,correct=bool(hit(ans633,gold633)),
     prepare_ms=round(prep633,5),answer_ms=round(answer_ms633,5),status="PASS",
     disk_bytes=sum(MANIFEST633[ci]["bytes"] for ci in keys633))
    del mem633
   except Exception as e:
    row633.update(status="ERROR",error=repr(e),traceback=traceback.format_exc())
   ROWS633.append(row633)
   print(f"MAM_DISK   CASE={wi633:02d} SLOT={slot633:6s} STATUS={row633['status']} CORRECT={row633.get('correct')}",flush=True)
   if row633["status"]=="ERROR":
    write633(OUT633,REPORT633)
    raise RuntimeError(f"TEST633_DISK_FAILURE_{wi633}_{slot633}")
  write633(OUT633,REPORT633)
finally:
 for c,body in zip(CAR,ORIGINAL633):c["body"]=body
SUMMARY633={}
for arm633 in ARMS633:
 subset633=[r for r in ROWS633 if r["arm"]==arm633]
 SUMMARY633[arm633]=dict(
  tested=len(subset633),correct=sum(int(r.get("correct",False)) for r in subset633),
  by_slot={slot:sum(int(r.get("correct",False)) for r in subset633 if r["slot"]==slot) for slot in SLOTS},
  median_prepare_ms=median633([r["prepare_ms"] for r in subset633 if r["status"]=="PASS"]),
  median_answer_ms=median633([r["answer_ms"] for r in subset633 if r["status"]=="PASS"]),
  median_total_ms=median633([r["prepare_ms"]+r["answer_ms"] for r in subset633 if r["status"]=="PASS"]))
MAP633={(r["arm"],r["case"],r["slot"]):r for r in ROWS633}
EQUIV633=[]
for wi633 in range(24):
 for slot633 in SLOTS:
  dc633=MAP633[("DC_SOURCE",wi633,slot633)]
  disk633=MAP633[("MAM_DISK",wi633,slot633)]
  EQUIV633.append(dict(case=wi633,slot=slot633,
   answer_equal=dc633["answer"]==disk633["answer"],
   correct_equal=dc633["correct"]==disk633["correct"]))
REPORT633["summary"]=SUMMARY633
REPORT633["disk_vs_dc_answer_equal"]=sum(int(r["answer_equal"]) for r in EQUIV633)
REPORT633["disk_vs_dc_cases"]=len(EQUIV633)
REPORT633["equivalence"]=EQUIV633
REPORT633["model_unchanged"]=sentinel()==S0
REPORT633["total_seconds"]=round(time.perf_counter()-T633,3)
REPORT633["status"]="COMPLETE" if (
 len(ROWS633)==360 and
 all(r["status"]=="PASS" for r in ROWS633) and
 all(r["answer_equal"] for r in EQUIV633) and
 REPORT633["model_unchanged"]) else "MISMATCH"
write633(OUT633,REPORT633)
with open(CSV633,"w",newline="",encoding="utf-8") as f:
 fields633=["arm","case","slot","status","correct","gold","answer","tokens",
  "prepare_ms","answer_ms","disk_bytes"]
 w633=csv.DictWriter(f,fieldnames=fields633)
 w633.writeheader()
 for r in ROWS633:w633.writerow({k:r.get(k) for k in fields633})
print("="*120,flush=True)
print("TEST633_STATUS",REPORT633["status"],flush=True)
print("CASES",len(ROWS633),"/360",flush=True)
print("PHYSICAL_DISK_VS_DC_ANSWER_EQUAL",REPORT633["disk_vs_dc_answer_equal"],"/72",flush=True)
print("MODEL_UNCHANGED",REPORT633["model_unchanged"],flush=True)
for arm633,s633 in SUMMARY633.items():
 print("ARM",arm633,"CORRECT",f'{s633["correct"]}/{s633["tested"]}',
  "FIRST",s633["by_slot"]["FIRST"],
  "MIDDLE",s633["by_slot"]["MIDDLE"],
  "LAST",s633["by_slot"]["LAST"],
  "PREPARE_MEDIAN_MS",s633["median_prepare_ms"],
  "ANSWER_MEDIAN_MS",s633["median_answer_ms"],
  "TOTAL_MEDIAN_MS",s633["median_total_ms"],flush=True)
print("JSON_OUTPUT",OUT633,flush=True)
print("CSV_OUTPUT",CSV633,flush=True)
print("TOTAL_SECONDS",REPORT633["total_seconds"],flush=True)
print("="*120,flush=True)
if REPORT633["status"]!="COMPLETE":raise RuntimeError("TEST633_VALIDATION_FAILURE")

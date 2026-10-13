
# ==============================================================================================================
# TEST637 — AKBASCORE MEMORY ARENA · COMPETITOR READINESS & PROTOCOL LOCK
# TEST636/635 WORKING BACKBONE · LOCKED TEST566 · NO ENGINE MODIFICATION
# REAL ENVIRONMENT AUDIT · 3 NATIVE CONTROL CASES · 3 SOURCE-BLOCKED PHYSICAL MAM CASES
# BITWISE CUT3 VERIFICATION · MODEL INTEGRITY · COMPETITOR READINESS · FINAL PROTOCOL SHA256
# NO AUTOMATIC THIRD-PARTY INSTALLATION · NO INVENTED BENCHMARK RESULTS · SINGLE COLAB CELL
# ==============================================================================================================
import os,sys,time,json,csv,hashlib,urllib.request,random,statistics,traceback,platform,subprocess,shutil,importlib.util,importlib.metadata
from pathlib import Path
import torch,numpy as np
ROOT637=Path("/content/AKBASCORE_TEST637");CART637=ROOT637/"cartridges"
CART637.mkdir(parents=True,exist_ok=True)
URL637="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/566.py"
SHA637="0155cda553e5f08ed2bec53611e49898a5ca24e0d270c3a1274e661bc205bdcd"
BLOB637="b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71"
PANEL637="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
BOUND637="RESULTS=[];RAW=[];ERRORS=[]";CUT637=3;T637=time.perf_counter()
OUT637=ROOT637/"TEST637_ARENA_PROTOCOL_FINAL.json"
CSV637=ROOT637/"TEST637_READINESS.csv"
def log637(*x):print(*x,flush=True)
def sha637(x):return hashlib.sha256(x).hexdigest()
def blob637(x):return hashlib.sha1(b"blob "+str(len(x)).encode()+b"\0"+x).hexdigest()
def canonical637(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def sync637():torch.cuda.synchronize()
def timed637(fn):
 sync637();t=time.perf_counter();v=fn();sync637()
 return v,round((time.perf_counter()-t)*1000,5)
def save637(path,obj):
 tmp=path.with_suffix(".tmp")
 with open(tmp,"w",encoding="utf-8") as f:json.dump(obj,f,indent=2,ensure_ascii=False,default=str,allow_nan=False)
 os.replace(tmp,path)
def filehash637(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for block in iter(lambda:f.read(4194304),b""):h.update(block)
 return h.hexdigest()
def cpu637(x):
 if torch.is_tensor(x):return x.detach().cpu().contiguous()
 if isinstance(x,(tuple,list)):return tuple(cpu637(v) for v in x)
 if isinstance(x,dict):return {k:cpu637(v) for k,v in x.items()}
 raise TypeError("ARTIFACT_TYPE")
def gpu637(x):
 if torch.is_tensor(x):return x.to(DEVICE)
 if isinstance(x,(tuple,list)):return tuple(gpu637(v) for v in x)
 if isinstance(x,dict):return {k:gpu637(v) for k,v in x.items()}
 raise TypeError("ARTIFACT_TYPE")
def exact637(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype and bool(torch.equal(a,b))
 if isinstance(a,(tuple,list)):return isinstance(b,(tuple,list)) and len(a)==len(b) and all(exact637(x,y) for x,y in zip(a,b))
 return a==b
def source637():
 for p in (Path("/content/566.py"),Path("/content/TEST566.py")):
  if p.is_file():
   b=p.read_bytes()
   if sha637(b)==SHA637 and blob637(b)==BLOB637:return b,"LOCAL_VERIFIED"
 b=urllib.request.urlopen(urllib.request.Request(URL637,headers={"User-Agent":"AKBASCORE-TEST637"}),timeout=120).read()
 if sha637(b)!=SHA637 or blob637(b)!=BLOB637:raise RuntimeError("SOURCE_LOCK_FAILURE")
 return b,"GITHUB_VERIFIED"
def package637(distributions,modules,executables=()):
 installed=[]
 for name in distributions:
  try:installed.append({"distribution":name,"version":importlib.metadata.version(name)})
  except importlib.metadata.PackageNotFoundError:pass
 modules_found=[]
 for module in modules:
  try:
   if importlib.util.find_spec(module) is not None:modules_found.append(module)
  except (ImportError,ValueError,AttributeError):pass
 bins={name:shutil.which(name) for name in executables}
 return dict(installed_distributions=installed,importable_modules=modules_found,
  executables={k:v for k,v in bins.items() if v},
  package_detected=bool(installed or modules_found or any(bins.values())))
log637("="*118)
log637("TEST637 — MEMORY ARENA · COMPETITOR READINESS & PROTOCOL LOCK")
log637("="*118)
if not torch.cuda.is_available():raise RuntimeError("CUDA_REQUIRED")
raw637,origin637=source637()
src637=raw637.decode("utf-8")
if src637.count(BOUND637)!=1:raise RuntimeError("BOOTSTRAP_BOUNDARY_FAILURE")
log637("SOURCE",origin637,"SHA256",sha637(raw637),"GIT_BLOB",blob637(raw637))
exec(compile(src637.split(BOUND637,1)[0],"TEST566_LOCKED.py","exec"),globals())
if PANEL_SHA!=PANEL637 or len(CAR)!=96 or P!=29:raise RuntimeError("PANEL_LOCK_FAILURE")
if (NL,H,QH,KVH,HD)!=(28,3584,28,4,128):raise RuntimeError("ARCH_LOCK_FAILURE")
if sentinel()!=S0:raise RuntimeError("MODEL_LOCK_FAILURE")
ENV637=dict(python=sys.version.split()[0],platform=platform.platform(),
 torch=torch.__version__,transformers=importlib.metadata.version("transformers"),
 gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
 total_gpu_mib=round(torch.cuda.get_device_properties(0).total_memory/1048576,2),
 model=MODEL_ID,architecture=dict(layers=NL,hidden=H,query_heads=QH,kv_heads=KVH,head_dim=HD),
 dtype=str(DTYPE),cut=CUT637,source_sha256=SHA637,panel_sha256=PANEL_SHA)
log637("[1/5] ENVIRONMENT & COMPETITOR DETECTION")
COMP637={
 "AKBASCORE_MAM":dict(kind="physical_reusable_cartridge",status="PENDING_VALIDATION"),
 "NATIVE_JOINT":dict(kind="source_prefill_control",status="PENDING_VALIDATION"),
 "FULL_KV_RELOAD":dict(kind="serialized_combined_kv_control",status="READY_PREVIOUS_TEST635",
  evidence="TEST635: 72/72 correct; implementation is not LMCache"),
 "VLLM_PREFIX":dict(kind="prefix_cache",status="NOT_TESTED",
  detection=package637(("vllm",),("vllm",),("vllm",))),
 "LMCACHE":dict(kind="kv_cache_storage",status="NOT_TESTED",
  detection=package637(("lmcache","lmcache-nightly"),("lmcache",),())),
 "SGLANG_RADIX":dict(kind="prefix_radix_cache",status="NOT_TESTED",
  detection=package637(("sglang","sglang-router"),("sglang",),())),
 "CACHEBLEND":dict(kind="selective_kv_reuse_recompute",status="NOT_TESTED",
  detection=package637(("cacheblend",),("cacheblend",),())),
 "TEXT_RAG":dict(kind="text_retrieval_plus_generation",status="NOT_TESTED",
  detection=package637(("faiss-cpu","faiss-gpu","rank-bm25"),("faiss","rank_bm25"),()))
}
for name,item in COMP637.items():
 detected=item.get("detection",{}).get("package_detected",False)
 log637("COMPETITOR",name,"STATUS",item["status"],"PACKAGE_DETECTED",detected)
# Detection is NOT proof of compatibility, serving readiness, or working inference.
log637("[2/5] FROZEN MATCH PANEL")
rng=random.Random(634552)
CASES637=[]
for wi in range(8):
 target=PANEL[wi]["target"]
 pool=[ci for ci in range(96) if CAR[ci]["world"]!=wi]
 rng.shuffle(pool)
 others=pool[:39]
 if wi not in (0,3,7):continue
 typ=CAR[target]["type"]
 for size,slot in ((5,"FIRST"),(20,"MIDDLE"),(40,"LAST")):
  d=others[:size-1]
  if slot=="FIRST":keys=[target]+d
  elif slot=="LAST":keys=d+[target]
  else:
   left=(size-1)//2
   keys=d[:left]+[target]+d[left:]
  CASES637.append(dict(world=wi,size=size,slot=slot,keys=keys,
   target=target,question=W[wi][typ],gold=CAR[target]["gold"]))
# Three source-blocked controls: small, medium and large.
CASES637=[CASES637[0],CASES637[4],CASES637[8]]
if len(CASES637)!=3:raise RuntimeError("CASE_LOCK_FAILURE")
NEEDED637=sorted(set(ci for z in CASES637 for ci in z["keys"]))
log637("CASES",len(CASES637),"PHYSICAL_CARTRIDGES",len(NEEDED637))
MANIFEST637={}
log637("[3/5] FORGING SOURCE-ISOLATED PHYSICAL CUT3 CARTRIDGES")
for i,ci in enumerate(NEEDED637):
 cp=checkpoint(ci,CUT637)
 obj=dict(lower_kv=cp["kv"][:CUT637+1],h3=cp["h"])
 path=CART637/f"C{ci:03d}.pt"
 torch.save(cpu637(obj),path)
 MANIFEST637[ci]=dict(path=str(path),sha256=filehash637(path),
  bytes=path.stat().st_size,tokens=cp["h"].shape[1])
 del cp,obj
 if (i+1)%16==0 or i+1==len(NEEDED637):log637("FORGED",i+1,"/",len(NEEDED637))
def load637(ci):
 meta=MANIFEST637[ci];p=Path(meta["path"])
 if filehash637(p)!=meta["sha256"]:raise RuntimeError("CARTRIDGE_SHA_FAILURE")
 return gpu637(torch.load(p,map_location="cpu",weights_only=True))
@torch.no_grad()
def init637(obj):
 lower=obj["lower_kv"];h3=obj["h3"];n=h3.shape[1]
 if len(lower)!=CUT637+1 or h3.shape!=(1,n,H):raise RuntimeError("INIT_SHAPE_FAILURE")
 def inject(module,args,out):return replace(out,h3)
 hook=model.model.layers[CUT637].register_forward_hook(inject)
 try:
  pos=torch.arange(n,device=DEVICE)
  out=model(inputs_embeds=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE),
   attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),
   position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
  upper=clone(out.past_key_values)
  return tuple(lower)+tuple(upper[CUT637+1:])
 finally:hook.remove()
@torch.no_grad()
def append637(memory,obj):
 lower=obj["lower_kv"];h3=obj["h3"]
 oldn=memory[0][0].shape[-2];q=h3.shape[1]-P
 if len(lower)!=CUT637+1 or q<=0:raise RuntimeError("APPEND_SHAPE_FAILURE")
 body=h3[:,P:,:];pos=torch.arange(oldn,oldn+q,device=DEVICE)
 def inject(module,args,out):
  current=out[0] if isinstance(out,(tuple,list)) else out
  if current.shape!=body.shape:raise RuntimeError("INJECTION_SHAPE_FAILURE")
  return replace(out,body)
 hook=model.model.layers[CUT637].register_forward_hook(inject)
 try:
  out=model(inputs_embeds=torch.zeros((1,q,H),device=DEVICE,dtype=DTYPE),
   past_key_values=cache_build(memory),
   attention_mask=torch.ones((1,oldn+q),device=DEVICE,dtype=torch.long),
   position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
  grown=list(clone(out.past_key_values))
  oldpos=torch.arange(P,P+q,device=DEVICE)
  for li in range(CUT637+1):
   ok,ov=memory[li];k,v=lower[li]
   kk=rephase(k[:,:,P:,:],oldpos,pos)
   grown[li]=(torch.cat((ok,kk),-2),torch.cat((ov,v[:,:,P:,:]),-2))
  result=tuple(grown)
  for (ok,ov),(nk,nv) in zip(memory,result):
   if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):
    raise RuntimeError("PREFIX_BITWISE_FAILURE")
  return result
 finally:hook.remove()
def mam637(keys):
 mem=None
 for ci in keys:
  obj=load637(ci)
  mem=init637(obj) if mem is None else append637(mem,obj)
 return mem
def valid637(mem,keys):
 n=P+sum(MANIFEST637[ci]["tokens"]-P for ci in keys)
 if len(mem)!=NL:raise RuntimeError("CACHE_LAYER_FAILURE")
 for li,(k,v) in enumerate(mem):
  if tuple(k.shape)!=(1,KVH,n,HD) or tuple(v.shape)!=(1,KVH,n,HD):
   raise RuntimeError(f"CACHE_SHAPE_FAILURE_{li}")
 return n
log637("[4/5] SOURCE-DEPENDENT NATIVE + DC BITWISE REFERENCE")
RESULTS637=[]
for i,z in enumerate(CASES637):
 native,native_ms=timed637(lambda:joint(z["keys"]))
 native_answer,native_answer_ms=timed637(lambda:answer(native,z["question"]))
 dc=consolidated(z["keys"],CUT637)
 disk=mam637(z["keys"])
 bitwise=exact637(dc,disk)
 if not bitwise:raise RuntimeError(f"DC_DISK_BITWISE_FAILURE_{i}")
 n=valid637(disk,z["keys"])
 RESULTS637.append(dict(case=i,world=z["world"],size=z["size"],
  slot=z["slot"],tokens=n,bitwise_dc_disk=bitwise,
  native_correct=bool(hit(native_answer,z["gold"])),
  native_answer=native_answer,native_prepare_ms=native_ms,
  native_answer_ms=native_answer_ms))
 del native,dc,disk
 log637("REFERENCE",i,"N",z["size"],"BITWISE",bitwise,
  "NATIVE_CORRECT",RESULTS637[-1]["native_correct"])
 if not RESULTS637[-1]["native_correct"]:raise RuntimeError("NATIVE_CONTROL_FAILURE")
# Physically prevent CAR body access for the disk inference arm.
ORIGINAL637=[c["body"] for c in CAR]
for c in CAR:c["body"]=None
log637("[5/5] SOURCE-BLOCKED MAM INFERENCE + FINAL PROTOCOL")
try:
 for i,z in enumerate(CASES637):
  mem,prepare_ms=timed637(lambda:mam637(z["keys"]))
  valid637(mem,z["keys"])
  ans,answer_ms=timed637(lambda:answer(mem,z["question"]))
  RESULTS637[i].update(mam_answer=ans,mam_correct=bool(hit(ans,z["gold"])),
   native_answer_equal=ans==RESULTS637[i]["native_answer"],
   mam_prepare_ms=prepare_ms,mam_answer_ms=answer_ms,
   disk_bytes=sum(MANIFEST637[ci]["bytes"] for ci in z["keys"]))
  del mem
  log637("MAM",i,"N",z["size"],"CORRECT",RESULTS637[i]["mam_correct"],
   "NATIVE_EQUAL",RESULTS637[i]["native_answer_equal"])
finally:
 for c,body in zip(CAR,ORIGINAL637):c["body"]=body
COMP637["AKBASCORE_MAM"]["status"]="READY" if all(r["mam_correct"] and r["bitwise_dc_disk"] for r in RESULTS637) else "FAILED_CONTROL"
COMP637["NATIVE_JOINT"]["status"]="READY" if all(r["native_correct"] for r in RESULTS637) else "FAILED_CONTROL"
PROTOCOL637=dict(
 name="AKBASCORE_MEMORY_ARENA_FINAL_PROTOCOL",
 version="637.1",
 model=MODEL_ID,source_sha256=SHA637,panel_sha256=PANEL_SHA,
 cut=CUT637,dtype=str(DTYPE),greedy_generation=True,
 source_policy=dict(
  native_joint="source text permitted at prefill; time source prefill",
  mam_disk="source blocked during inference; physical cartridge only",
  full_kv="source blocked during inference; precomputed combined KV only",
  third_party="declare source access and precomputation explicitly"),
 matching=dict(
  same_model_and_tokenizer=True,
  same_question_and_gold=True,
  same_generation_limit=True,
  same_deterministic_decoding=True,
  record_full_answer=True,
  record_correctness=True,
  record_exact_answer_equality=True,
  record_native_token_count=True,
  record_cache_hit_and_miss_separately=True,
  record_warm_and_cold_separately=True,
  record_preparation_and_answer_latency_separately=True,
  record_disk_bytes_and_gpu_memory_separately=True,
  record_one_time_build_cost_separately=True),
 rounds=[
  "TEST638: actual vLLM/LMCache/SGLang-compatible implementations, where available",
  "TEST639: actual CacheBlend and text-RAG, with compatible workloads",
  "TEST640: reproducibility, failure cases and integrity audit",
  "TEST641: Memory Arena final demo and academic endgame report"],
 scoring=dict(
  accuracy="correct / total; report raw answers",
  latency="synchronized measured wall time; do not substitute projections",
  storage="declare whether unique reusable artifacts or per-query artifacts",
  throughput="measure independently if implemented",
  no_contest="unavailable, incompatible or untested methods cannot lose or win",
  victory="only for a measured comparable category",
  fatality="presentation label only; requires prespecified decisive quantitative criterion",
  cpp_cuda="future optimization hypothesis; never a measured result until implemented"))
PROTOCOL_SHA637=sha637(canonical637(PROTOCOL637))
REPORT637=dict(test="637",status="IN_PROGRESS",environment=ENV637,
 competitors=COMP637,protocol=PROTOCOL637,protocol_sha256=PROTOCOL_SHA637,
 cases=CASES637,results=RESULTS637,manifest=MANIFEST637,
 model_unchanged=sentinel()==S0,
 limitations=[
  "Competitor package detection does not establish a working benchmark.",
  "No competitor packages were installed or modified.",
  "Third-party methods remain NOT_TESTED until real inference succeeds.",
  "The three internal cases are protocol controls, not a statistical benchmark.",
  "MAM physical cartridge forging is excluded from request-time latency.",
  "C++/CUDA speedup is not measured or projected here.",
  "The prior TEST635 and TEST636 files are not required or altered.",
  "Ready status for MAM/native applies to the verified internal controls only."])
REPORT637["status"]="COMPLETE" if (
 REPORT637["model_unchanged"] and
 len(RESULTS637)==3 and
 all(r["bitwise_dc_disk"] and r["mam_correct"] and r["native_correct"] for r in RESULTS637)
 ) else "COMPLETE_WITH_FINDINGS"
REPORT637["total_seconds"]=round(time.perf_counter()-T637,3)
save637(OUT637,REPORT637)
with open(CSV637,"w",newline="",encoding="utf-8") as f:
 fields=["competitor","kind","status","package_detected","installed_versions"]
 writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
 for name,item in COMP637.items():
  det=item.get("detection",{})
  writer.writerow(dict(competitor=name,kind=item["kind"],status=item["status"],
   package_detected=det.get("package_detected",False),
   installed_versions=";".join(
    x["distribution"]+"="+x["version"] for x in det.get("installed_distributions",[]))))
log637("="*118)
log637("TEST637_STATUS",REPORT637["status"])
log637("CONTROL_CASES",len(RESULTS637),"/3")
log637("MAM_CORRECT",sum(r["mam_correct"] for r in RESULTS637),"/3")
log637("NATIVE_CORRECT",sum(r["native_correct"] for r in RESULTS637),"/3")
log637("DC_DISK_BITWISE",sum(r["bitwise_dc_disk"] for r in RESULTS637),"/3")
log637("NATIVE_MAM_ANSWER_EQUAL",sum(r["native_answer_equal"] for r in RESULTS637),"/3")
log637("MODEL_UNCHANGED",REPORT637["model_unchanged"])
log637("PROTOCOL_SHA256",PROTOCOL_SHA637)
for name,item in COMP637.items():
 log637("ARENA_ENTRY",name,item["status"],
  "DETECTED",item.get("detection",{}).get("package_detected","N/A"))
log637("JSON_OUTPUT",OUT637)
log637("CSV_OUTPUT",CSV637)
log637("TOTAL_SECONDS",REPORT637["total_seconds"])
log637("="*118)

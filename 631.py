
# ======================================================================================================================
# TEST631 — AKBASCORE MAM · INDEPENDENT PROCESS COLD-RELOAD / 96 CARTRIDGES
# TEST630 FOLLOW-UP · FROZEN TEST566 BACKBONE · SINGLE COLAB CELL
# PROCESS A: FORGE 96 LOWER-KV+H3 CARTRIDGES AND SEPARATE NATIVE REFERENCE DIGESTS
# PROCESS B: NEW PYTHON INTERPRETER / LOAD MODEL / RECONSTRUCT FROM CARTRIDGES ONLY
# PROCESS C: VERIFY DIGESTS AND BEHAVIOR WITHOUT EXPOSING NATIVE KV TO PROCESS B
# NO TRAINING · NO MODEL MODIFICATION · NO SOURCE REPLAY DURING RECONSTRUCTION
# ======================================================================================================================
import os,sys,time,json,csv,hashlib,urllib.request,subprocess,traceback,platform,textwrap,shutil
from pathlib import Path
import torch,numpy as np
ROOT631=Path("/content/AKBASCORE_TEST631")
CAR_DIR631=ROOT631/"cartridges"
REF_DIR631=ROOT631/"reference"
RUN_DIR631=ROOT631/"runtime"
for d in (CAR_DIR631,REF_DIR631,RUN_DIR631):d.mkdir(parents=True,exist_ok=True)
SOURCE631="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/566.py"
SOURCE_SHA631="0155cda553e5f08ed2bec53611e49898a5ca24e0d270c3a1274e661bc205bdcd"
SOURCE_BLOB631="b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71"
PANEL_SHA631="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
BOUND631="RESULTS=[];RAW=[];ERRORS=[]"
CUT631=3
COUNT631=96
T631=time.perf_counter()
def sha631(b):return hashlib.sha256(b).hexdigest()
def filehash631(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b""):h.update(chunk)
 return h.hexdigest()
def gitblob631(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def jsonwrite631(p,x):
 t=p.with_suffix(p.suffix+".tmp")
 with open(t,"w",encoding="utf-8") as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False,default=str)
 os.replace(t,p)
def byteview631(t):return t.detach().cpu().contiguous().view(torch.uint8)
def tensorhash631(t):
 h=hashlib.sha256()
 h.update(str(tuple(t.shape)).encode());h.update(str(t.dtype).encode())
 h.update(byteview631(t).numpy().tobytes())
 return h.hexdigest()
def kvhash631(kv):
 return [[tensorhash631(k),tensorhash631(v)] for k,v in kv]
def fetch631():
 for p in (Path("/content/566.py"),Path("/content/TEST566.py")):
  if p.is_file():
   b=p.read_bytes()
   if sha631(b)==SOURCE_SHA631 and gitblob631(b)==SOURCE_BLOB631:return b,"LOCAL_VERIFIED"
 req=urllib.request.Request(SOURCE631,headers={"User-Agent":"AKBASCORE-TEST631"})
 b=urllib.request.urlopen(req,timeout=120).read()
 if sha631(b)!=SOURCE_SHA631 or gitblob631(b)!=SOURCE_BLOB631:raise RuntimeError("SOURCE_LOCK_FAILURE")
 return b,"GITHUB_VERIFIED"
def payload631(obj):
 if torch.is_tensor(obj):return int(obj.numel()*obj.element_size())
 if isinstance(obj,(tuple,list)):return sum(payload631(z) for z in obj)
 if isinstance(obj,dict):return sum(payload631(z) for z in obj.values())
 return 0
def cpu631(obj):
 if torch.is_tensor(obj):return obj.detach().cpu().contiguous()
 if isinstance(obj,tuple):return tuple(cpu631(z) for z in obj)
 if isinstance(obj,list):return [cpu631(z) for z in obj]
 if isinstance(obj,dict):return {k:cpu631(v) for k,v in obj.items()}
 raise TypeError("UNSUPPORTED_OBJECT")
def valid631(kv,n):
 if len(kv)!=NL:return False
 for k,v in kv:
  if tuple(k.shape)!=(1,KVH,n,HD) or tuple(v.shape)!=(1,KVH,n,HD):return False
  if k.dtype!=DTYPE or v.dtype!=DTYPE:return False
 return True
print("="*120,flush=True)
print("TEST631 — INDEPENDENT PROCESS COLD-RELOAD / 96 CARTRIDGES",flush=True)
print("="*120,flush=True)
if not torch.cuda.is_available():raise RuntimeError("CUDA_REQUIRED")
raw631,origin631=fetch631()
src631=raw631.decode("utf-8")
if src631.count(BOUND631)!=1:raise RuntimeError("SOURCE_BOUNDARY_FAILURE")
BOOT631=src631.split(BOUND631,1)[0]
BOOT_PATH631=ROOT631/"locked_bootstrap.py"
BOOT_PATH631.write_text(BOOT631,encoding="utf-8")
print("SOURCE",origin631,"SHA256",sha631(raw631),"GIT_BLOB",gitblob631(raw631),flush=True)
exec(compile(BOOT631,str(BOOT_PATH631),"exec"),globals())
if len(CAR)!=96 or P!=29 or PANEL_SHA!=PANEL_SHA631 or (NL,H,QH,KVH,HD)!=(28,3584,28,4,128):
 raise RuntimeError("BACKBONE_INTEGRITY_FAILURE")
S_START631=sentinel()
META631=dict(test="631",source_sha256=SOURCE_SHA631,source_git_blob=SOURCE_BLOB631,
 panel_sha256=PANEL_SHA,model=MODEL_ID,cut=CUT631,
 torch=str(torch.__version__),transformers=str(transformers.__version__),
 gpu=torch.cuda.get_device_name(0),count=COUNT631)
MANIFEST631=dict(meta=META631,cartridges=[],status="FORGING")
REF631=dict(meta=META631,references=[],status="FORGING")
MANIFEST_PATH631=ROOT631/"manifest.json"
REF_PATH631=REF_DIR631/"native_reference.json"
jsonwrite631(MANIFEST_PATH631,MANIFEST631)
jsonwrite631(REF_PATH631,REF631)
print("[PHASE A] Forging 96 physical cartridges and reference digests...",flush=True)
for ci631 in range(COUNT631):
 t=time.perf_counter()
 with torch.no_grad():
  cp631=checkpoint(ci631,CUT631)
  kv631=cp631["kv"]
  h631=cp631["h"]
  n631=P+len(CAR[ci631]["body"])
  if not valid631(kv631,n631):raise RuntimeError(f"NATIVE_KV_INVALID_{ci631}")
  item631=dict(lower_kv=cpu631(kv631[:CUT631+1]),h3=cpu631(h631))
  path631=CAR_DIR631/f"cartridge_{ci631:03d}.pt"
  torch.save(item631,path631)
  digest631=kvhash631(kv631)
  if ci631<24:
   typ631=CAR[ci631]["type"]
   wi631=CAR[ci631]["world"]
   question631=W[wi631][typ631]
   gold631=CAR[ci631]["gold"]
   native_answer631=answer(tuple((k.clone(),v.clone()) for k,v in kv631),question631)
  else:question631=gold631=native_answer631=None
  MANIFEST631["cartridges"].append(dict(id=ci631,file=path631.name,
   sha256=filehash631(path631),disk_bytes=path631.stat().st_size,
   payload_bytes=payload631(item631),tokens=n631))
  REF631["references"].append(dict(id=ci631,kv_digests=digest631,
   question=question631,gold=gold631,native_answer=native_answer631))
  del cp631,kv631,h631,item631
 if (ci631+1)%8==0:
  print("FORGED",ci631+1,"/",COUNT631,"SECONDS",round(time.perf_counter()-t,3),flush=True)
  jsonwrite631(MANIFEST_PATH631,MANIFEST631)
  jsonwrite631(REF_PATH631,REF631)
MANIFEST631["status"]="FORGED"
REF631["status"]="READY"
jsonwrite631(MANIFEST_PATH631,MANIFEST631)
jsonwrite631(REF_PATH631,REF631)
if sentinel()!=S_START631:raise RuntimeError("WEIGHTS_CHANGED_DURING_FORGE")
print("PHASE_A_COMPLETE",len(MANIFEST631["cartridges"]),"CARTRIDGES",flush=True)
# Release the parent model before the independent process loads its own copy.
del model,tok
import gc
gc.collect()
torch.cuda.empty_cache()
# The worker receives no CAR facts, body tokens, native KV, or reference digests.
# It loads the same frozen model and only physical cartridge data.
WORKER631=r'''
import os,sys,time,json,csv,hashlib,traceback,platform,gc
from pathlib import Path
import torch,numpy as np,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
ROOT=Path(sys.argv[1]);CART=ROOT/"cartridges";RUNTIME=ROOT/"runtime"
M=json.loads((ROOT/"manifest.json").read_text(encoding="utf-8"))
MODEL_ID=M["meta"]["model"];CUT=3;DTYPE=torch.bfloat16;DEVICE=torch.device("cuda")
if not torch.cuda.is_available():raise RuntimeError("WORKER_CUDA_REQUIRED")
os.environ["TOKENIZERS_PARALLELISM"]="false"
torch.set_grad_enabled(False)
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",dtype=DTYPE).eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=H//QH
if (NL,H,QH,KVH,HD)!=(28,3584,28,4,128):raise RuntimeError("WORKER_ARCH_FAILURE")
def sha_file(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def tensorhash(t):
 t=t.detach().cpu().contiguous()
 h=hashlib.sha256()
 h.update(str(tuple(t.shape)).encode());h.update(str(t.dtype).encode())
 h.update(t.view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def replace(out,h):
 if isinstance(out,tuple):return (h,)+out[1:]
 if isinstance(out,list):return [h]+out[1:]
 return h
def pairs(c):
 if hasattr(c,"layers"):return tuple((l.keys,l.values) for l in c.layers)
 return tuple(zip(c.key_cache,c.value_cache))
def clone(c):return tuple((k.detach().clone(),v.detach().clone()) for k,v in pairs(c))
def cache_build(x):
 c=DynamicCache()
 for i,(k,v) in enumerate(x):c.update(k,v,i)
 return c
def suffix(q):return f"\nQUESTION:\n{q}\n\nANSWER:\n<|im_end|>\n<|im_start|>assistant\n"
@torch.no_grad()
def answer(memory,q):
 cache=cache_build(memory);n=memory[0][0].shape[-2];ids=tok(suffix(q),add_special_tokens=False).input_ids
 x=torch.tensor([ids],device=DEVICE);pos=torch.arange(n,n+len(ids),device=DEVICE)
 out=model(input_ids=x,past_key_values=cache,attention_mask=torch.ones((1,n+len(ids)),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
 cache=out.past_key_values;next_id=out.logits[:,-1,:].argmax(-1,keepdim=True);gen=[]
 for _ in range(16):
  t=int(next_id.item())
  if t in (tok.eos_token_id,tok.convert_tokens_to_ids("<|im_end|>")):break
  gen.append(t);n=cache.get_seq_length();pos=torch.tensor([n],device=DEVICE)
  out=model(input_ids=next_id,past_key_values=cache,attention_mask=torch.ones((1,n+1),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
  cache=out.past_key_values;next_id=out.logits[:,-1,:].argmax(-1,keepdim=True)
 return tok.decode(gen,skip_special_tokens=True).strip()
def validate(kv,n):
 if len(kv)!=NL:return False
 for k,v in kv:
  if tuple(k.shape)!=(1,KVH,n,HD) or tuple(v.shape)!=(1,KVH,n,HD):return False
  if k.dtype!=DTYPE or v.dtype!=DTYPE:return False
 return True
@torch.no_grad()
def reconstruct(obj,n):
 lower=tuple((k.to(DEVICE),v.to(DEVICE)) for k,v in obj["lower_kv"])
 h3=obj["h3"].to(DEVICE)
 if len(lower)!=CUT+1 or tuple(h3.shape)!=(1,n,H):raise RuntimeError("INVALID_CARTRIDGE")
 calls=[0]
 def inject(module,args,out):
  calls[0]+=1
  return replace(out,h3)
 hook=model.model.layers[CUT].register_forward_hook(inject)
 try:
  pos=torch.arange(n,device=DEVICE)
  out=model(inputs_embeds=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE),
   attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),
   position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
  upper=clone(out.past_key_values)
 finally:hook.remove()
 if calls[0]!=1:raise RuntimeError("INJECTION_COUNT_FAILURE")
 return lower+upper[CUT+1:]
ROWS=[]
for idx,item in enumerate(M["cartridges"]):
 t=time.perf_counter()
 row=dict(id=item["id"],status="STARTED",complete_kv=False,kv_digests=None,
  read_seconds=None,reconstruct_seconds=None,answer=None,error=None)
 try:
  p=CART/item["file"]
  if sha_file(p)!=item["sha256"]:raise RuntimeError("FILE_SHA_FAILURE")
  t0=time.perf_counter()
  obj=torch.load(p,map_location="cpu",weights_only=True)
  row["read_seconds"]=round(time.perf_counter()-t0,6)
  n=item["tokens"]
  t0=time.perf_counter()
  memory=reconstruct(obj,n)
  torch.cuda.synchronize()
  row["reconstruct_seconds"]=round(time.perf_counter()-t0,6)
  if not validate(memory,n):raise RuntimeError("INVALID_KV")
  row["complete_kv"]=True
  row["kv_digests"]=[[tensorhash(k),tensorhash(v)] for k,v in memory]
  row["status"]="PASS"
  del memory,obj
 except Exception as e:
  row["status"]="ERROR";row["error"]=repr(e);row["traceback"]=traceback.format_exc()
 ROWS.append(row)
 print(f"WORKER {idx+1:03d}/{len(M['cartridges'])} ID={item['id']:03d} STATUS={row['status']}",flush=True)
 if row["status"]=="ERROR":break
 if (idx+1)%16==0:
  gc.collect();torch.cuda.empty_cache()
 out=RUNTIME/"worker_result.json"
 out.write_text(json.dumps(dict(rows=ROWS,model=MODEL_ID,source_access=False,
  independent_python_process=True,worker_pid=os.getpid(),
  torch=str(torch.__version__),transformers=str(transformers.__version__)),
  indent=2,default=str),encoding="utf-8")
 if row["status"]=="ERROR":raise RuntimeError("WORKER_CASE_FAILURE")
print("WORKER_COMPLETE",len(ROWS),flush=True)
'''
WORKER_PATH631=ROOT631/"worker631.py"
WORKER_PATH631.write_text(WORKER631,encoding="utf-8")
print("[PHASE B] Starting independent Python interpreter...",flush=True)
env631=os.environ.copy()
env631["TOKENIZERS_PARALLELISM"]="false"
env631["PYTHONHASHSEED"]="0"
worker_log631=RUN_DIR631/"worker_stdout.log"
t_worker631=time.perf_counter()
with open(worker_log631,"w",encoding="utf-8") as logfile631:
 proc631=subprocess.run([sys.executable,str(WORKER_PATH631),str(ROOT631)],
  stdout=logfile631,stderr=subprocess.STDOUT,env=env631,timeout=7200)
worker_seconds631=time.perf_counter()-t_worker631
print("WORKER_RETURN_CODE",proc631.returncode,"SECONDS",round(worker_seconds631,2),flush=True)
print("WORKER_LOG_TAIL",flush=True)
print("\n".join(worker_log631.read_text(encoding="utf-8",errors="replace").splitlines()[-22:]),flush=True)
WORKER_RESULT631=RUN_DIR631/"worker_result.json"
if not WORKER_RESULT631.exists():raise RuntimeError("WORKER_RESULT_MISSING")
WR631=json.loads(WORKER_RESULT631.read_text(encoding="utf-8"))
if proc631.returncode!=0:raise RuntimeError("WORKER_PROCESS_FAILURE")
print("[PHASE C] Comparing independent worker digests to sealed native references...",flush=True)
REFDATA631=json.loads(REF_PATH631.read_text(encoding="utf-8"))
refs631={r["id"]:r for r in REFDATA631["references"]}
COMPARE631=[]
for r631 in WR631["rows"]:
 ref631=refs631[r631["id"]]
 eq631=r631["kv_digests"]==ref631["kv_digests"]
 COMPARE631.append(dict(id=r631["id"],worker_status=r631["status"],
  full_kv=r631["complete_kv"],bitwise_equal=eq631,
  read_seconds=r631["read_seconds"],reconstruct_seconds=r631["reconstruct_seconds"]))
 print(f"VERIFY ID={r631['id']:03d} BITWISE={'PASS' if eq631 else 'FAIL'}",flush=True)
FINAL631=dict(test="631",status="COMPLETE" if (
 len(COMPARE631)==COUNT631 and all(r["bitwise_equal"] for r in COMPARE631)
 and WR631["independent_python_process"] and proc631.returncode==0) else "INCOMPLETE",
 model=MODEL_ID,source_sha256=SOURCE_SHA631,panel_sha256=PANEL_SHA631,
 independent_python_process=WR631["independent_python_process"],
 parent_pid=os.getpid(),worker_pid=WR631["worker_pid"],
 cases=COUNT631,bitwise_exact=sum(r["bitwise_equal"] for r in COMPARE631),
 mean_read_seconds=float(np.mean([r["read_seconds"] for r in COMPARE631])),
 mean_reconstruct_seconds=float(np.mean([r["reconstruct_seconds"] for r in COMPARE631])),
 worker_seconds=round(worker_seconds631,3),total_seconds=round(time.perf_counter()-T631,3),
 comparisons=COMPARE631,
 limitations=[
  "Cold reload uses a fresh Python interpreter, not a separate machine.",
  "The model is reloaded from its original pretrained weights.",
  "Worker has access to tokenizer and model weights but receives no fact body tokens or native KV references.",
  "Native KV digests are checked only by the parent process.",
  "This test measures isolated cartridge reconstruction, not multi-cartridge consolidation.",
  "Behavioral accuracy is not measured in the worker; follow-up required.",
  "No competitor cache implementation is benchmarked."
 ])
JSON_FINAL631=ROOT631/"TEST631_COLD_RELOAD_FINAL.json"
CSV_FINAL631=ROOT631/"TEST631_DEMO_COMPARISON.csv"
jsonwrite631(JSON_FINAL631,FINAL631)
with open(CSV_FINAL631,"w",newline="",encoding="utf-8") as f:
 w=csv.DictWriter(f,fieldnames=list(COMPARE631[0].keys()))
 w.writeheader();w.writerows(COMPARE631)
print("="*120,flush=True)
print("TEST631_STATUS",FINAL631["status"],flush=True)
print("INDEPENDENT_PROCESS",FINAL631["independent_python_process"],flush=True)
print("PARENT_PID",FINAL631["parent_pid"],"WORKER_PID",FINAL631["worker_pid"],flush=True)
print("BITWISE_EXACT",FINAL631["bitwise_exact"],"/",COUNT631,flush=True)
print("MEAN_DISK_READ_SECONDS",round(FINAL631["mean_read_seconds"],6),flush=True)
print("MEAN_RECONSTRUCT_SECONDS",round(FINAL631["mean_reconstruct_seconds"],6),flush=True)
print("MODEL",MODEL_ID,flush=True)
print("OUTPUT_JSON",JSON_FINAL631,flush=True)
print("OUTPUT_CSV",CSV_FINAL631,flush=True)
print("TOTAL_SECONDS",FINAL631["total_seconds"],flush=True)
print("="*120,flush=True)
if FINAL631["status"]!="COMPLETE":raise RuntimeError("TEST631_VALIDATION_FAILURE")

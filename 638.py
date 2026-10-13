
# ======================================================================================================================
# TEST638 — AKBASCORE MEMORY ARENA · TIER 1 · CORRECTED COMPLETE SINGLE-CELL
# TEXT-RAG (DETERMINISTIC BM25 + FROZEN QWEN) vs FULL NATIVE KV RELOAD vs PHYSICAL AKBASCORE MAM
# TEST566 SOURCE/PANEL LOCK · CUT3 UNCHANGED · 72 MATCHES × 3 ARMS = 216 RUNS
# SOURCE ISOLATION · PHYSICAL SHA256 ARTIFACTS · BITWISE AUDIT · RAW ANSWERS · SCIENTIFIC SCOREBOARD
# ======================================================================================================================
import os,sys,re,time,json,csv,hashlib,urllib.request,random,statistics,traceback,math,gc,inspect
from pathlib import Path
from collections import Counter
import torch,numpy as np
TEST="638";CUT=3;SEED=634552;SIZES=(5,20,40);SLOTS638=("FIRST","MIDDLE","LAST")
ROOT=Path("/content/AKBASCORE_TEST638");MAM_DIR=ROOT/"mam_cartridges";KV_DIR=ROOT/"native_full_kv"
MAM_DIR.mkdir(parents=True,exist_ok=True);KV_DIR.mkdir(parents=True,exist_ok=True)
SOURCE_URL="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/566.py"
SOURCE_SHA="0155cda553e5f08ed2bec53611e49898a5ca24e0d270c3a1274e661bc205bdcd"
SOURCE_BLOB="b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71"
PANEL_LOCK="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
BOUNDARY="RESULTS=[];RAW=[];ERRORS=[]"
JSON_OUT=ROOT/"TEST638_TIER1_FINAL.json";CSV_OUT=ROOT/"TEST638_TIER1_MATCHES.csv"
START=time.perf_counter()
def say(*x):print(*x,flush=True)
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def file_sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(4194304),b""):h.update(b)
 return h.hexdigest()
def sync():torch.cuda.synchronize()
def clock(fn):
 sync();t=time.perf_counter();r=fn();sync()
 return r,round((time.perf_counter()-t)*1000,5)
def median(x):return round(statistics.median(x),5) if x else None
def save(path,obj):
 tmp=path.with_suffix(".tmp")
 with open(tmp,"w",encoding="utf-8") as f:json.dump(obj,f,indent=2,ensure_ascii=False,default=str,allow_nan=False)
 os.replace(tmp,path)
def cpu(x):
 if torch.is_tensor(x):return x.detach().cpu().contiguous()
 if isinstance(x,(tuple,list)):return tuple(cpu(y) for y in x)
 if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
 raise TypeError("UNSUPPORTED_ARTIFACT")
def gpu(x):
 if torch.is_tensor(x):return x.to(DEVICE)
 if isinstance(x,(tuple,list)):return tuple(gpu(y) for y in x)
 if isinstance(x,dict):return {k:gpu(v) for k,v in x.items()}
 raise TypeError("UNSUPPORTED_ARTIFACT")
def equal(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype and bool(torch.equal(a,b))
 if isinstance(a,(tuple,list)):return isinstance(b,(tuple,list)) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
def fetch_source():
 for p in (Path("/content/566.py"),Path("/content/TEST566.py")):
  if p.is_file():
   b=p.read_bytes()
   if sha(b)==SOURCE_SHA and blob(b)==SOURCE_BLOB:return b,"LOCAL_VERIFIED"
 req=urllib.request.Request(SOURCE_URL,headers={"User-Agent":"AKBASCORE-TEST638"})
 b=urllib.request.urlopen(req,timeout=120).read()
 if sha(b)!=SOURCE_SHA or blob(b)!=SOURCE_BLOB:raise RuntimeError("SOURCE_SHA_OR_BLOB_MISMATCH")
 return b,"GITHUB_VERIFIED"
say("="*120)
say("TEST638 — MEMORY ARENA · TIER 1 · TEXT-RAG vs FULL KV RELOAD vs AKBASCORE MAM")
say("="*120)
if not torch.cuda.is_available():raise RuntimeError("CUDA_REQUIRED")
raw,origin=fetch_source();source=raw.decode("utf-8")
if source.count(BOUNDARY)!=1:raise RuntimeError("SOURCE_BOOTSTRAP_FAILURE")
say("SOURCE",origin,"SHA256",sha(raw),"BLOB",blob(raw))
exec(compile(source.split(BOUNDARY,1)[0],"<TEST566_LOCKED_BOOTSTRAP>","exec"),globals())
if PANEL_SHA!=PANEL_LOCK or len(CAR)!=96 or P!=29:raise RuntimeError("PANEL_LOCK_FAILURE")
if (NL,H,QH,KVH,HD)!=(28,3584,28,4,128):raise RuntimeError("ARCHITECTURE_LOCK_FAILURE")
if sentinel()!=S0:raise RuntimeError("MODEL_WEIGHT_LOCK_FAILURE")
# Resolve the actual tokenizer object exposed by the locked TEST566 bootstrap, without changing the engine.
def resolve_tokenizer():
 candidates=[]
 for name in ("tok","TOK","tokenizer","TOKENIZER","tk","TK"):
  obj=globals().get(name)
  if obj is not None and callable(getattr(obj,"encode",None)) and callable(getattr(obj,"decode",None)):
   candidates.append((name,obj))
 if not candidates:
  for name,obj in list(globals().items()):
   if name.startswith("__"):continue
   if callable(getattr(obj,"encode",None)) and callable(getattr(obj,"decode",None)):
    if "tokenizer" in type(obj).__name__.lower():candidates.append((name,obj))
 if not candidates:raise RuntimeError("TEST566_TOKENIZER_NOT_FOUND")
 name,obj=candidates[0]
 if len({id(x) for _,x in candidates})>1:raise RuntimeError("AMBIGUOUS_TOKENIZER_OBJECT")
 return name,obj
TOKENIZER_NAME,TOKENIZER638=resolve_tokenizer()
say("TOKENIZER_RESOLVED",TOKENIZER_NAME,type(TOKENIZER638).__name__)
def encode638(text):
 ids=TOKENIZER638.encode(text,add_special_tokens=False)
 return ids.tolist() if torch.is_tensor(ids) else list(ids)
def decode638(ids):
 if torch.is_tensor(ids):ids=ids.detach().cpu().tolist()
 return TOKENIZER638.decode(ids,skip_special_tokens=True)
def build_cases():
 rng=random.Random(SEED);cases=[]
 for wi in range(8):
  target=PANEL[wi]["target"]
  pool=[ci for ci in range(96) if CAR[ci]["world"]!=wi]
  rng.shuffle(pool);others39=pool[:39]
  typ=CAR[target]["type"]
  for n in SIZES:
   distractors=others39[:n-1]
   for slot in SLOTS638:
    if slot=="FIRST":keys=[target]+distractors
    elif slot=="LAST":keys=distractors+[target]
    else:
     k=(n-1)//2;keys=distractors[:k]+[target]+distractors[k:]
    cases.append(dict(world=wi,size=n,slot=slot,keys=keys,target=target,
     question=W[wi][typ],gold=CAR[target]["gold"]))
 return cases
CASES=build_cases()
if len(CASES)!=72:raise RuntimeError("PANEL_CASE_COUNT_FAILURE")
NEEDED=sorted(set(ci for z in CASES for ci in z["keys"]))
MANIFEST={};FULL_MANIFEST={};ROWS=[];PROBES=[];ERRORS=[]
TEXT_CORPUS={}
for ci in NEEDED:
 body=CAR[ci]["body"]
 if body is None:raise RuntimeError("SOURCE_BODY_MISSING_BEFORE_FORGING")
 TEXT_CORPUS[ci]=decode638(body)
def words(text):return re.findall(r"[a-z0-9]+",text.lower())
TOKENS={ci:words(TEXT_CORPUS[ci]) for ci in NEEDED}
DOC_LEN={ci:len(TOKENS[ci]) for ci in NEEDED}
def bm25(question,keys):
 docs=[TOKENS[ci] for ci in keys]
 q=Counter(words(question));n=len(keys)
 avg=sum(len(x) for x in docs)/max(n,1)
 df=Counter()
 for d in docs:
  for w in set(d):df[w]+=1
 scores=[]
 for order,(ci,d) in enumerate(zip(keys,docs)):
  freq=Counter(d);score=0.
  for w,qf in q.items():
   if w not in freq:continue
   idf=math.log(1+(n-df[w]+0.5)/(df[w]+0.5))
   tf=freq[w];den=tf+1.2*(1-0.75+0.75*len(d)/max(avg,1))
   score+=idf*(tf*2.2/den)*qf
  scores.append((score,order,ci))
 scores.sort(key=lambda z:(-z[0],z[1]))
 return scores
# RAG uses the same frozen model. Greedy decoding is performed through the locked answer() function.
# This avoids incompatible model.generate() cache APIs and preserves the TEST566 generation path.
@torch.no_grad()
def rag_generate638(question,selected):
 text=TEXT_CORPUS[selected]
 ids=encode638(text+suffix(question))
 inp=torch.tensor([ids],device=DEVICE,dtype=torch.long)
 pos=torch.arange(inp.shape[1],device=DEVICE)
 out=model(input_ids=inp,attention_mask=torch.ones_like(inp),
  position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
 mem=clone(out.past_key_values)
 # The prompt already includes the question suffix. Generate directly from its last-token logits.
 next_token=int(out.logits[0,-1].argmax().item())
 generated=[next_token]
 eos=TOKENIZER638.eos_token_id
 for _ in range(MAX_NEW-1):
  if eos is not None and next_token==eos:break
  step=torch.tensor([[next_token]],device=DEVICE,dtype=torch.long)
  cp=torch.tensor([inp.shape[1]+len(generated)-1],device=DEVICE)
  out=model(input_ids=step,past_key_values=cache_build(mem),
   attention_mask=torch.ones((1,int(cp.item())+1),device=DEVICE,dtype=torch.long),
   position_ids=cp.unsqueeze(0),cache_position=cp,use_cache=True,return_dict=True)
  mem=clone(out.past_key_values)
  next_token=int(out.logits[0,-1].argmax().item())
  generated.append(next_token)
 return decode638(generated),len(ids)
@torch.no_grad()
def mam_init(obj):
 lower=obj["lower_kv"];h3=obj["h3"];n=h3.shape[1]
 if len(lower)!=CUT+1 or h3.shape!=(1,n,H):raise RuntimeError("MAM_INIT_SHAPE")
 def inject(module,args,out):return replace(out,h3)
 hook=model.model.layers[CUT].register_forward_hook(inject)
 try:
  pos=torch.arange(n,device=DEVICE)
  out=model(inputs_embeds=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE),
   attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),
   position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
  upper=clone(out.past_key_values)
  return tuple(lower)+tuple(upper[CUT+1:])
 finally:hook.remove()
@torch.no_grad()
def mam_append(mem,obj):
 lower=obj["lower_kv"];h3=obj["h3"]
 oldn=mem[0][0].shape[-2];q=h3.shape[1]-P
 if len(lower)!=CUT+1 or q<=0:raise RuntimeError("MAM_APPEND_SHAPE")
 body=h3[:,P:,:];pos=torch.arange(oldn,oldn+q,device=DEVICE)
 def inject(module,args,out):
  current=out[0] if isinstance(out,(tuple,list)) else out
  if current.shape!=body.shape:raise RuntimeError("MAM_INJECTION_SHAPE")
  return replace(out,body)
 hook=model.model.layers[CUT].register_forward_hook(inject)
 try:
  out=model(inputs_embeds=torch.zeros((1,q,H),device=DEVICE,dtype=DTYPE),
   past_key_values=cache_build(mem),
   attention_mask=torch.ones((1,oldn+q),device=DEVICE,dtype=torch.long),
   position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
  grown=list(clone(out.past_key_values))
  oldpos=torch.arange(P,P+q,device=DEVICE)
  for li in range(CUT+1):
   ok,ov=mem[li];k,v=lower[li]
   kk=rephase(k[:,:,P:,:],oldpos,pos)
   grown[li]=(torch.cat((ok,kk),-2),torch.cat((ov,v[:,:,P:,:]),-2))
  result=tuple(grown)
  for (ok,ov),(nk,nv) in zip(mem,result):
   if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):
    raise RuntimeError("MAM_PREFIX_BITWISE_FAILURE")
  return result
 finally:hook.remove()
def load_mam(ci):
 m=MANIFEST[ci];p=Path(m["path"])
 if file_sha(p)!=m["sha256"]:raise RuntimeError("MAM_FILE_SHA_FAILURE")
 return gpu(torch.load(p,map_location="cpu",weights_only=True))
def mam_build(keys):
 mem=None
 for ci in keys:
  obj=load_mam(ci)
  mem=mam_init(obj) if mem is None else mam_append(mem,obj)
 return mem
def load_full(case):
 m=FULL_MANIFEST[case];p=Path(m["path"])
 if file_sha(p)!=m["sha256"]:raise RuntimeError("FULL_KV_FILE_SHA_FAILURE")
 return gpu(torch.load(p,map_location="cpu",weights_only=True))
def cache_check(mem,keys):
 n=P+sum(MANIFEST[ci]["tokens"]-P for ci in keys)
 if len(mem)!=NL:raise RuntimeError("CACHE_LAYER_COUNT_FAILURE")
 for li,(k,v) in enumerate(mem):
  if tuple(k.shape)!=(1,KVH,n,HD) or tuple(v.shape)!=(1,KVH,n,HD):
   raise RuntimeError(f"CACHE_SHAPE_FAILURE_{li}")
 return n
say("[1/6] PHYSICAL MAM CARTRIDGE FORGING")
for ix,ci in enumerate(NEEDED):
 cp=checkpoint(ci,CUT)
 obj={"lower_kv":cp["kv"][:CUT+1],"h3":cp["h"]}
 path=MAM_DIR/f"C{ci:03d}.pt"
 torch.save(cpu(obj),path)
 MANIFEST[ci]={"path":str(path),"sha256":file_sha(path),"bytes":path.stat().st_size,
  "tokens":cp["h"].shape[1]}
 del cp,obj
 if (ix+1)%16==0 or ix+1==len(NEEDED):say("MAM_FORGED",ix+1,"/",len(NEEDED))
say("[2/6] FULL NATIVE KV DISK ARTIFACTS + NATIVE REFERENCE")
NATIVE={}
for ix,z in enumerate(CASES):
 mem,prep=clock(lambda:joint(z["keys"]))
 path=KV_DIR/f"CASE_{ix:03d}.pt"
 torch.save(cpu(mem),path)
 FULL_MANIFEST[ix]={"path":str(path),"sha256":file_sha(path),"bytes":path.stat().st_size}
 ans,answer_ms=clock(lambda:answer(mem,z["question"]))
 NATIVE[ix]={"answer":ans,"correct":bool(hit(ans,z["gold"])),
  "prepare_ms":prep,"answer_ms":answer_ms}
 if not NATIVE[ix]["correct"]:raise RuntimeError(f"NATIVE_REFERENCE_FAILURE_{ix}")
 del mem
 if (ix+1)%12==0:say("FULL_KV_FORGED",ix+1,"/72")
say("[3/6] PHYSICAL MAM vs SOURCE-DEPENDENT DC BITWISE PROBES")
for ix in (0,13,26,39,52,71):
 z=CASES[ix];dc=consolidated(z["keys"],CUT);disk=mam_build(z["keys"])
 exact=equal(dc,disk)
 PROBES.append({"case":ix,"size":z["size"],"slot":z["slot"],"bitwise":exact})
 say("PROBE",ix,"BITWISE",exact)
 del dc,disk
 if not exact:raise RuntimeError(f"DC_DISK_BITWISE_FAILURE_{ix}")
REPORT={"test":"638","status":"IN_PROGRESS","tier":1,
 "source_sha256":SOURCE_SHA,"panel_sha256":PANEL_SHA,"model":MODEL_ID,
 "gpu":torch.cuda.get_device_name(0),"torch":torch.__version__,
 "transformers":__import__("transformers").__version__,
 "cut":CUT,"cases":CASES,"rows":ROWS,"native":NATIVE,
 "mam_manifest":MANIFEST,"full_kv_manifest":FULL_MANIFEST,"probes":PROBES,
 "methodology":{
  "TEXT_RAG":"Deterministic BM25 top-1 retrieval over decoded source cartridge text; selected text and original question suffix are passed to the same frozen model.",
  "FULL_KV_DISK":"Serialized joint-native KV; physical SHA256 verified and loaded without source text.",
  "MAM_DISK":"Reusable physical CUT3 lower-KV + H3 cartridge files; SHA256 verified; upper-layer reconstruction; source text blocked.",
  "NATIVE_JOINT":"Native source-accessible prefill reference; not a Tier-1 contender.",
  "accuracy":"Locked TEST566 hit() evaluation; raw generated answers retained.",
  "latency":"Synchronized wall time; preparation and answer recorded separately.",
  "rag_lifecycle":"RAG retrieval measured separately; answer latency includes fresh source-text prefill and greedy decoding.",
  "storage":"MAM unique reusable cartridge files; Full KV unique per-case files; Text-RAG raw UTF-8 corpus only.",
  "limitations":"Source access, precomputation and artifact lifecycles differ; no universal winner assumed."}}
save(JSON_OUT,REPORT)
say("[4/6] SOURCE-BLOCKED PHYSICAL KV ARMS + SOURCE-ACCESSIBLE TEXT-RAG")
ORIGINAL=[c["body"] for c in CAR]
for c in CAR:c["body"]=None
try:
 for ix,z in enumerate(CASES):
  base={"case":ix,"world":z["world"],"size":z["size"],"slot":z["slot"],
   "gold":z["gold"],"question":z["question"]}
  for arm in ("FULL_KV_DISK","MAM_DISK","TEXT_RAG"):
   row=dict(base,arm=arm,status="STARTED")
   try:
    if arm=="FULL_KV_DISK":
     mem,prep=clock(lambda:load_full(ix))
     cache_check(mem,z["keys"])
     ans,ansms=clock(lambda:answer(mem,z["question"]))
     row.update(prepare_ms=prep,answer_ms=ansms,
      disk_bytes=FULL_MANIFEST[ix]["bytes"],source_text_access=False)
     del mem
    elif arm=="MAM_DISK":
     mem,prep=clock(lambda:mam_build(z["keys"]))
     cache_check(mem,z["keys"])
     ans,ansms=clock(lambda:answer(mem,z["question"]))
     row.update(prepare_ms=prep,answer_ms=ansms,
      disk_bytes=sum(MANIFEST[ci]["bytes"] for ci in z["keys"]),
      source_text_access=False)
     del mem
    else:
     ranked,retrieval_ms=clock(lambda:bm25(z["question"],z["keys"]))
     chosen=ranked[0][2]
     (ans,prompt_tokens),ansms=clock(lambda:rag_generate638(z["question"],chosen))
     row.update(prepare_ms=retrieval_ms,answer_ms=ansms,
      retrieved_cartridge=chosen,retrieved_target=chosen==z["target"],
      retrieval_top3=[(round(s,8),ci) for s,_,ci in ranked[:3]],
      prompt_tokens=prompt_tokens,
      disk_bytes=sum(len(TEXT_CORPUS[ci].encode("utf-8")) for ci in z["keys"]),
      source_text_access=True)
    row.update(answer=ans,correct=bool(hit(ans,z["gold"])),
     native_answer_equal=ans==NATIVE[ix]["answer"],status="PASS")
   except Exception as exc:
    row.update(status="ERROR",error=repr(exc),traceback=traceback.format_exc())
    ERRORS.append({"case":ix,"arm":arm,"error":repr(exc)})
   ROWS.append(row)
   if row["status"]!="PASS":
    save(JSON_OUT,REPORT)
    raise RuntimeError(f"ARM_FAILURE_CASE_{ix}_{arm}: {row.get('error')}")
  if (ix+1)%6==0:
   last=ROWS[-3:]
   say("MATCH",f"{ix+1:02d}/72","SIZE",z["size"],"SLOT",z["slot"],
    "FULL",last[0]["correct"],"MAM",last[1]["correct"],
    "RAG",last[2]["correct"],"RAG_RETRIEVED_TARGET",last[2]["retrieved_target"])
   save(JSON_OUT,REPORT)
finally:
 for c,body in zip(CAR,ORIGINAL):c["body"]=body
say("[5/6] ARENA SCORING · NO PREDECLARED VICTORY")
ARMS=("FULL_KV_DISK","MAM_DISK","TEXT_RAG")
SUMMARY={}
for n in SIZES:
 SUMMARY[str(n)]={}
 for arm in ARMS:
  rr=[r for r in ROWS if r["size"]==n and r["arm"]==arm]
  SUMMARY[str(n)][arm]={"correct":sum(int(r["correct"]) for r in rr),
   "cases":len(rr),
   "prepare_median_ms":median([r["prepare_ms"] for r in rr]),
   "answer_median_ms":median([r["answer_ms"] for r in rr]),
   "total_median_ms":median([r["prepare_ms"]+r["answer_ms"] for r in rr]),
   "native_answer_equal":sum(int(r["native_answer_equal"]) for r in rr),
   "retrieval_target_hits":sum(int(r.get("retrieved_target",False)) for r in rr)
    if arm=="TEXT_RAG" else None}
MAM_UNIQUE=sum(x["bytes"] for x in MANIFEST.values())
FULL_UNIQUE=sum(x["bytes"] for x in FULL_MANIFEST.values())
TEXT_UNIQUE=sum(len(x.encode("utf-8")) for x in TEXT_CORPUS.values())
STORAGE={"mam_unique_bytes":MAM_UNIQUE,"full_kv_unique_bytes":FULL_UNIQUE,
 "rag_corpus_utf8_bytes":TEXT_UNIQUE,
 "mam_unique_cartridges":len(MANIFEST),"full_kv_unique_cases":len(FULL_MANIFEST),
 "warning":"Artifact totals have different reusable/per-case lifecycles; not a normalized compression ratio."}
REPORT.update(summary=SUMMARY,storage=STORAGE,model_unchanged=sentinel()==S0,
 total_seconds=round(time.perf_counter()-START,3))
REPORT["status"]="COMPLETE" if (
 len(ROWS)==216 and not ERRORS and REPORT["model_unchanged"]
 and all(p["bitwise"] for p in PROBES)) else "COMPLETE_WITH_FINDINGS"
save(JSON_OUT,REPORT)
with open(CSV_OUT,"w",newline="",encoding="utf-8") as f:
 fields=["case","world","size","slot","arm","status","correct","native_answer_equal",
  "prepare_ms","answer_ms","disk_bytes","source_text_access","retrieved_cartridge",
  "retrieved_target","prompt_tokens","gold","question","answer"]
 writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
 for r in ROWS:writer.writerow({k:r.get(k) for k in fields})
say("[6/6] FINAL SCOREBOARD")
say("="*120)
say("TEST638_STATUS",REPORT["status"])
say("ROWS",len(ROWS),"/216")
say("MODEL_UNCHANGED",REPORT["model_unchanged"])
say("BITWISE_PROBES",sum(p["bitwise"] for p in PROBES),"/",len(PROBES))
for n in SIZES:
 say("-"*100)
 for arm in ARMS:
  s=SUMMARY[str(n)][arm]
  say("SIZE",n,"ARM",arm,"CORRECT",f'{s["correct"]}/{s["cases"]}',
   "PREPARE_MEDIAN_MS",s["prepare_median_ms"],
   "ANSWER_MEDIAN_MS",s["answer_median_ms"],
   "TOTAL_MEDIAN_MS",s["total_median_ms"],
   "NATIVE_EQUAL",f'{s["native_answer_equal"]}/{s["cases"]}',
   "RAG_TARGET_HITS",s["retrieval_target_hits"])
say("MAM_UNIQUE_DISK_MIB",round(MAM_UNIQUE/1048576,5))
say("FULL_KV_UNIQUE_DISK_MIB",round(FULL_UNIQUE/1048576,5))
say("RAG_CORPUS_UTF8_MIB",round(TEXT_UNIQUE/1048576,5))
say("JSON_OUTPUT",JSON_OUT)
say("CSV_OUTPUT",CSV_OUT)
say("TOTAL_SECONDS",REPORT["total_seconds"])
say("="*120)

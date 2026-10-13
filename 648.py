
# ======================================================================================================================
# TEST648 — AKBASCORE MAM / BLIND LATENT TWO-CARTRIDGE SYNTHESIS
# TEST647 LOCKED BACKBONE · TEST627 ENGINE · 1453 PHYSICAL CUT3 · PCA128 / FAISS IVF128 / PEAK50 UNCHANGED
# PHYSICAL KV FUSION: AB / BA / A / B / NO MEMORY · GREEDY MODEL GENERATION · CAUSAL ABLATION
# NO EXTERNAL ANSWER SELECTION · NO TEXT REPLAY · NO GOLD ROUTING · NO TRAINING · NO GRADIO
# ======================================================================================================================
import os,sys,time,json,hashlib,urllib.request,subprocess,importlib.util,random,re,gc,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,torch
TEST="648";SEED=646;N=16;TOTAL=1453
ROOT=Path("/content/AKBASCORE_TEST648");ROOT.mkdir(parents=True,exist_ok=True)
URL="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/HIPPOCAMPUS_full.py"
SOURCE_BLOB="c2c77b170365a08499bbc55d523bb04d46a12a6c"
ENGINE_LOCK_SHA="ab53e2d58fb3dc5f11409a5112e92feea8762544e4b30b8a77a39b60332de4d1"
ARMS=("AB","BA","A","B","NONE")
def log(*x):print(*x,flush=True)
def utc():return datetime.now(timezone.utc).isoformat()
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def save(name,obj):
 p=ROOT/name;p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding="utf-8");return p
def fetch(url):
 with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"AKBASCORE-TEST648"}),timeout=180) as f:return f.read()
def sha_file(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def norm(s):return " ".join(re.sub(r"[^\w\s-]"," ",str(s).casefold()).split())
def match(s,g):return re.search(r"(?<!\w)"+re.escape(norm(g))+r"(?!\w)",norm(s)) is not None
def country_of(item):
 fact=str(item["fact"]);prefix="The current capital of "
 assert fact.startswith(prefix) and " is " in fact
 return fact[len(prefix):].split(" is ",1)[0].strip()
log("="*132);log("TEST648 — BLIND LATENT TWO-CARTRIDGE SYNTHESIS");log("START",utc());log("="*132)
log("[1/12] ENVIRONMENT")
assert torch.cuda.is_available(),"CUDA GPU REQUIRED"
for pkg,mod in [("transformers","transformers"),("accelerate","accelerate"),("faiss-cpu","faiss")]:
 if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import transformers,faiss
log("GPU",torch.cuda.get_device_name(0),"TORCH",torch.__version__,"CUDA",torch.version.cuda)
log("[2/12] SOURCE VERIFICATION")
raw=fetch(URL);assert blob(raw)==SOURCE_BLOB,"SOURCE_BLOB_MISMATCH"
src=raw.decode("utf-8");assert src.count("#<<ENGINE_END>>")==1
part1=src.split("#<<ENGINE_END>>",1)[0]
dependency='("gradio","gradio"),'
gradio_import="import numpy as np,torch,transformers,gradio as gr"
assert part1.count(dependency)==1 and part1.count(gradio_import)==1,"BOOTSTRAP_STRUCTURE_CHANGED"
part1=part1.replace(dependency,"")
part1=part1.replace(gradio_import,'import numpy as np,torch,transformers\nimport types as _types\ngr=_types.SimpleNamespace(__version__="DISABLED_TEST648")')
log("SOURCE_VERIFIED",SOURCE_BLOB)
log("[3/12] LOCKED ENGINE REUSE / COLD BOOT")
ready=bool(globals().get("ENGINE_READY",False) and globals().get("MAM_SH_PART1_OK",False) and globals().get("ENGINE_SHA256")==ENGINE_LOCK_SHA and "ENG" in globals() and "INSTR" in globals())
if ready:log("ENGINE_REUSED")
else:
 log("COLD_BOOT_LOCKED_ENGINE");t=time.perf_counter()
 exec(compile(part1,"HIPPOCAMPUS_TEST648_BOOTSTRAP.py","exec"),globals())
 log("ENGINE_BOOT_SECONDS",round(time.perf_counter()-t,3))
assert ENGINE_READY and MAM_SH_PART1_OK and ENGINE_SHA256==ENGINE_LOCK_SHA
assert all(ok for _,ok in ENGINE_LOCK)
assert ENG.TOTAL==TOTAL and len(ENG.MANIFEST)==TOTAL and len(ENG.K)==TOTAL
assert tuple(ENG.LOCKED627)==("PEAK50",0.5,0.0)
assert len(ENG.QUEST)==48 and len(ENG.PANEL624)==48
assert not ENG.model.training and not any(p.requires_grad for p in ENG.model.parameters())
assert callable(ENG.rephase) and callable(ENG.cache_build) and callable(ENG.answer)
weight0=INSTR.full_weight_sha();sentinel0=INSTR.sentinel()
log("ENGINE_LOCK_PASS",ENGINE_SHA256,"CARTRIDGES",TOTAL)
log("[4/12] PREDECLARED BLIND TASKS — SAME SEED AS TEST647")
rng=random.Random(SEED);countries=[]
for ci,item in enumerate(ENG.CAR):
 try:
  country=country_of(item)
  if country:countries.append((ci,country))
 except Exception:continue
assert len(countries)>=2*N
rng.shuffle(countries);pairs=[];used=set()
for ci,country in countries:
 if ci in used:continue
 for cj,other in countries:
  if cj==ci or cj in used or other==country:continue
  used.update((ci,cj));pairs.append((ci,country,cj,other));break
 if len(pairs)==N:break
assert len(pairs)==N
sealed=[];public=[]
for i,(a,ca,b,cb) in enumerate(pairs):
 q=f"Identify the present capital of {ca} and the present capital of {cb}."
 sealed.append(dict(case=i,question=q,gold_ids=[a,b],countries=[ca,cb],
  gold_answers=[str(ENG.CAR[a]["gold"]),str(ENG.CAR[b]["gold"])]))
 public.append(dict(case=i,question=q,countries=[ca,cb]))
seal=sha(json.dumps(sealed,sort_keys=True,ensure_ascii=False).encode())
save("BLIND_GOLD_SEALED.json",dict(sha256=seal,cases=sealed))
save("BLIND_TASKS_PUBLIC.json",public)
log("BLIND_SEAL",seal)
log("[5/12] REFERENCE ROUTING CONTROL — 48/48")
ref=[]
for i in range(48):
 q=ENG.QUEST[i].detach().float().cpu().contiguous()
 ids,_=ENG.candidate626(q);fs=ENG.features626(q,ids)
 ranking,_=ENG.rerank626(ids,fs,ENG.LOCKED627)
 ref.append(bool(ranking and ranking[0]==ENG.PANEL624[i]["target"]))
 if (i+1)%12==0:log("REFERENCE",i+1,"/48",sum(ref))
assert all(ref),"REFERENCE_REGRESSION"
log("[6/12] BLIND ROUTING — FREEZE ALL SELECTED IDS")
pred=[]
for task in public:
 row=dict(case=task["case"],question=task["question"],countries=task["countries"])
 try:
  with torch.no_grad():
   rep,_=ENG.question_rep(task["question"])
   q=ENG.apply(rep[3],ENG.FIT).detach().float().cpu().contiguous()
  ids,route=ENG.candidate626(q);fs=ENG.features626(q,ids)
  ranking,_=ENG.rerank626(ids,fs,ENG.LOCKED627)
  assert len(ranking)>=2
  row.update(status="ROUTED",selected=[int(ranking[0]),int(ranking[1])],
   top5=[int(x) for x in ranking[:5]],touched=route["touched"])
 except Exception as e:row.update(status="ROUTE_ERROR",selected=[],top5=[],error=repr(e))
 pred.append(row)
 log("ROUTE",row["case"]+1,"/",N,row["status"],row["selected"])
save("ROUTING_LOCKED.json",pred)
route_sha=sha(json.dumps(pred,sort_keys=True,ensure_ascii=False).encode())
log("ROUTING_SHA",route_sha)
log("[7/12] PHYSICAL CUT3 LOADER / NATIVE KV FUSION")
# The two independently stored physical KV caches are concatenated.
# Prefix tokens from the second cartridge are removed to avoid duplicate system prefixes.
# Rotary position correction uses the existing locked ENG.rephase function.
# No text facts, correct IDs, answer tokens, or model weights are injected.
def load_mem(ci):
 ci=int(ci);assert 0<=ci<TOTAL
 m=ENG.MANIFEST[ci];p=ENG.POOL/m["file"]
 assert p.is_file() and p.stat().st_size==m["bytes"]
 assert sha_file(p)==m["sha256"],"PHYSICAL_SHA_FAILURE"
 stored=torch.load(p,map_location="cpu",weights_only=True)
 assert len(stored)==28 and all(len(z)==2 for z in stored)
 mem=tuple((k.to(ENG.DEVICE),v.to(ENG.DEVICE)) for k,v in stored)
 del stored
 n=mem[0][0].shape[-2]
 assert n>ENG.P and all(k.shape[-2]==n and v.shape[-2]==n for k,v in mem)
 return mem
def fusion(a,b):
 na=a[0][0].shape[-2];nb=b[0][0].shape[-2];prefix=int(ENG.P)
 old=torch.arange(prefix,nb,device=ENG.DEVICE)
 new=torch.arange(na,na+nb-prefix,device=ENG.DEVICE)
 fused=[]
 for (ak,av),(bk,bv) in zip(a,b):
  moved=ENG.rephase(bk[:,:,prefix:,:],old,new)
  k=torch.cat((ak,moved),dim=-2).contiguous()
  v=torch.cat((av,bv[:,:,prefix:,:]),dim=-2).contiguous()
  assert k.shape[-2]==na+nb-prefix and v.shape[-2]==na+nb-prefix
  assert torch.equal(k[:,:,:na,:],ak) and torch.equal(v[:,:,:na,:],av)
  fused.append((k,v))
 return tuple(fused)
def generate(mem,question):
 # One untouched frozen-model generation; no external answer selection.
 # ENG.answer already performs deterministic greedy decoding.
 with torch.no_grad():return str(ENG.answer(mem,question)).strip()
def generate_none(question):
 # Same question suffix but no cartridge memory.
 ids=ENG.tok(ENG.suffix(question),add_special_tokens=False).input_ids
 x=torch.tensor([ids],device=ENG.DEVICE)
 with torch.no_grad():
  out=ENG.model(input_ids=x,use_cache=True,return_dict=True)
  cache=out.past_key_values;next_id=out.logits[:,-1,:].argmax(-1,keepdim=True);gen=[]
  for _ in range(ENG.MAX_NEW):
   token=int(next_id.item())
   if token in (ENG.tok.eos_token_id,ENG.tok.convert_tokens_to_ids("<|im_end|>")):break
   gen.append(token);out=ENG.model(input_ids=next_id,past_key_values=cache,use_cache=True,return_dict=True)
   cache=out.past_key_values;next_id=out.logits[:,-1,:].argmax(-1,keepdim=True)
 return ENG.tok.decode(gen,skip_special_tokens=True).strip()
log("FUSION_READY","ARMS",ARMS)
log("[8/12] LATENT FUSION SELFTEST — PHYSICAL STRUCTURE")
sample=pred[0]["selected"];assert len(sample)==2
a=load_mem(sample[0]);b=load_mem(sample[1]);ab=fusion(a,b);ba=fusion(b,a)
assert len(ab)==28 and len(ba)==28
assert ab[0][0].shape[-2]==a[0][0].shape[-2]+b[0][0].shape[-2]-ENG.P
assert all(torch.isfinite(k).all() and torch.isfinite(v).all() for k,v in ab)
assert all(torch.isfinite(k).all() and torch.isfinite(v).all() for k,v in ba)
log("FUSION_SELFTEST_PASS","A",a[0][0].shape[-2],"B",b[0][0].shape[-2],"AB",ab[0][0].shape[-2])
del a,b,ab,ba;gc.collect();torch.cuda.empty_cache()
log("[9/12] BLIND MODEL GENERATION — AB / BA / A / B / NONE")
# Gold remains sealed. All arms answer the same joint question.
# No external combination, reranking, answer correction or fallback is allowed.
for row in pred:
 row["arms"]={}
 if row["status"]!="ROUTED":continue
 try:
  a=load_mem(row["selected"][0]);b=load_mem(row["selected"][1])
  memories={"AB":fusion(a,b),"BA":fusion(b,a),"A":a,"B":b}
  for arm in ARMS:
   t=time.perf_counter()
   try:
    raw_answer=generate_none(row["question"]) if arm=="NONE" else generate(memories[arm],row["question"])
    row["arms"][arm]=dict(status="OK",raw=raw_answer,elapsed_ms=round((time.perf_counter()-t)*1000,3))
   except Exception as e:row["arms"][arm]=dict(status="ERROR",error=repr(e))
   log("GENERATE",row["case"]+1,"/",N,arm,row["arms"][arm].get("status"),repr(row["arms"][arm].get("raw","")))
  del memories,a,b
 except Exception as e:row["generation_error"]=repr(e)
 save("GENERATION_PROGRESS.json",pred)
 gc.collect()
save("BLIND_GENERATION_LOCKED.json",pred)
generation_sha=sha(json.dumps(pred,sort_keys=True,ensure_ascii=False).encode())
log("GENERATION_FROZEN",generation_sha)
log("[10/12] POST-HOC SCORING — NO ANSWER MODIFICATION")
# Strict evaluation: both gold strings must appear, in the requested country order.
# Scoring happens only after all raw generations are frozen.
evaluation=[]
for row,gold in zip(pred,sealed):
 assert row["case"]==gold["case"]
 expected=gold["gold_answers"];selected=set(row["selected"])
 ev=dict(case=row["case"],gold_ids=gold["gold_ids"],
  selected=row["selected"],both_sources=set(gold["gold_ids"])<=selected,arms={})
 for arm in ARMS:
  result=row.get("arms",{}).get(arm,{})
  raw_answer=result.get("raw","")
  hits=[match(raw_answer,g) for g in expected]
  text=norm(raw_answer);positions=[text.find(norm(g)) if hits[j] else -1 for j,g in enumerate(expected)]
  ordered=bool(all(hits) and positions[0]>=0 and positions[1]>positions[0])
  ev["arms"][arm]=dict(status=result.get("status","NOT_RUN"),hit_each=hits,
   both_present=all(hits),both_ordered=ordered,raw=raw_answer)
 evaluation.append(ev)
 log("EVAL",ev["case"]+1,"/",N,"SOURCES",ev["both_sources"],
  "AB",ev["arms"]["AB"]["both_ordered"],"BA",ev["arms"]["BA"]["both_ordered"],
  "A",ev["arms"]["A"]["both_ordered"],"B",ev["arms"]["B"]["both_ordered"],
  "NONE",ev["arms"]["NONE"]["both_ordered"])
save("BLIND_EVALUATION.json",evaluation)
log("[11/12] CAUSAL CONTRIBUTION / ABLATION SUMMARY")
metrics={}
for arm in ARMS:
 metrics[arm]=dict(
  both_ordered=sum(x["arms"][arm]["both_ordered"] for x in evaluation),
  both_present=sum(x["arms"][arm]["both_present"] for x in evaluation),
  first_correct=sum(x["arms"][arm]["hit_each"][0] for x in evaluation),
  second_correct=sum(x["arms"][arm]["hit_each"][1] for x in evaluation),
  errors=sum(x["arms"][arm]["status"]!="OK" for x in evaluation))
causal=[]
for x in evaluation:
 ab=x["arms"]["AB"];ba=x["arms"]["BA"];aa=x["arms"]["A"];bb=x["arms"]["B"];nn=x["arms"]["NONE"]
 causal.append(dict(case=x["case"],
  both_sources=x["both_sources"],
  fusion_both_ordered=ab["both_ordered"] or ba["both_ordered"],
  fusion_unique=bool((ab["both_ordered"] or ba["both_ordered"]) and not aa["both_ordered"] and not bb["both_ordered"] and not nn["both_ordered"]),
  fusion_both_correct=ab["both_present"] or ba["both_present"],
  single_A_first=aa["hit_each"][0],single_A_second=aa["hit_each"][1],
  single_B_first=bb["hit_each"][0],single_B_second=bb["hit_each"][1],
  no_memory_both=nn["both_present"]))
save("CAUSAL_ABLATION.json",causal)
log("ARM_METRICS",json.dumps(metrics))
log("FUSION_UNIQUE",sum(x["fusion_unique"] for x in causal),"/",N)
log("[12/12] FINAL ENGINE INTEGRITY / REPORT")
frozen=INSTR.frozen();weight1=INSTR.full_weight_sha()
integrity=dict(source_verified=blob(raw)==SOURCE_BLOB,
 engine_verified=ENGINE_SHA256==ENGINE_LOCK_SHA,
 engine_lock=all(ok for _,ok in ENGINE_LOCK),
 weight_unchanged=weight0==weight1,
 sentinel_unchanged=sentinel0==INSTR.sentinel(),
 zero_trainable=frozen["trainable_tensors"]==0,
 no_optimizer=not frozen["optimizer"],
 no_foreign_hooks=frozen["foreign_hooks"]==0,
 no_residual_hooks=not frozen["residual_hooks"])
summary=dict(test=TEST,utc=utc(),gpu=torch.cuda.get_device_name(0),
 cartridges=TOTAL,blind_cases=N,reference_correct=sum(ref),
 blind_both_sources=sum(x["both_sources"] for x in evaluation),
 arms=metrics,
 fusion_unique=sum(x["fusion_unique"] for x in causal),
 fusion_both_correct=sum(x["fusion_both_correct"] for x in causal),
 integrity=integrity,route_sha256=route_sha,
 generation_sha256=generation_sha,gold_seal_sha256=seal,
 scope=[
  "The original TEST627 engine, CUT3, PCA128, FAISS IVF128, PEAK50 and model weights are unchanged.",
  "Both independently stored physical KV memories enter the same frozen model forward pass.",
  "Second-cartridge prefix is removed and rotary key positions are corrected using the locked rephase function.",
  "AB and BA test memory-order sensitivity; A, B and NONE provide causal ablations.",
  "No external answer selection, correction, composition or oracle fallback is performed.",
  "The answer is generated by the frozen model through greedy decoding.",
  "This is a direct latent two-memory integration test, not proof of novel scientific discovery.",
  "KV concatenation permits attention over both memories but does not guarantee genuine cross-source reasoning.",
  "Synthetic facts and deterministic task pairs are reused from the earlier test family.",
  "Scoring is post-hoc and cannot alter any generated answer."
 ])
summary["infrastructure_status"]="PASS" if all(integrity.values()) and sum(ref)==48 else "FAIL"
summary["experimental_status"]="MEASURED" if len(evaluation)==N else "INCOMPLETE"
summary["status"]="PASS" if summary["infrastructure_status"]=="PASS" and summary["experimental_status"]=="MEASURED" else "FAIL"
out=save("TEST648_RESULT.json",dict(summary=summary,predictions=pred,evaluation=evaluation,causal=causal))
log("="*132)
log("TEST648_STATUS",summary["status"])
log("REFERENCE",sum(ref),"/48")
log("BLIND_BOTH_SOURCES",summary["blind_both_sources"],"/",N)
for arm in ARMS:log("LATENT",arm,metrics[arm]["both_ordered"],"/",N,"BOTH_PRESENT",metrics[arm]["both_present"],"/",N)
log("FUSION_UNIQUE",summary["fusion_unique"],"/",N)
log("INTEGRITY",json.dumps(integrity))
log("RESULT",out)
log("="*132)
if summary["status"]!="PASS":raise RuntimeError("TEST648_INFRASTRUCTURE_FAILURE — RESULTS SAVED")

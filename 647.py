
# ======================================================================================================================
# TEST647 — AKBASCORE MAM / BLIND MULTI-CARTRIDGE SYNTHESIS
# TEST646 WORKING BACKBONE · LOCKED TEST627 ENGINE · 1453 PHYSICAL CUT3 · PEAK50 UNCHANGED
# BLIND DISCOVERY -> INDEPENDENT PHYSICAL MEMORY READ -> TWO-FACT SYNTHESIS
# 16 SEALED TASKS · NO GOLD ROUTING · NO ORACLE FALLBACK · NO TRAINING · NO GRADIO
# ======================================================================================================================
import os,sys,time,json,hashlib,urllib.request,subprocess,importlib.util,random,re,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,torch
TEST="647";SEED=646;N=16;TOTAL=1453
ROOT=Path("/content/AKBASCORE_TEST647");ROOT.mkdir(parents=True,exist_ok=True)
URL="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/HIPPOCAMPUS_full.py"
SOURCE_BLOB="c2c77b170365a08499bbc55d523bb04d46a12a6c"
ENGINE_SHA_LOCK="ab53e2d58fb3dc5f11409a5112e92feea8762544e4b30b8a77a39b60332de4d1"
def log(*a):print(*a,flush=True)
def utc():return datetime.now(timezone.utc).isoformat()
def digest(b):return hashlib.sha256(b).hexdigest()
def gitblob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def save(name,obj):
 p=ROOT/name
 p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
 return p
def fetch(url):
 req=urllib.request.Request(url,headers={"User-Agent":"AKBASCORE-TEST647"})
 with urllib.request.urlopen(req,timeout=180) as f:return f.read()
def normalized(s):return re.sub(r"[^a-z0-9]","",str(s).casefold())
def clean_answer(s):
 s=str(s).strip()
 s=re.sub(r"(?i)^(answer|capital|city)\s*:\s*","",s)
 return s.strip(" \t\n\r.,;:!?\"'")
def fact_country(item):
 fact=str(item["fact"])
 prefix="The current capital of "
 assert fact.startswith(prefix) and " is " in fact,"UNSUPPORTED FACT FORMAT"
 return fact[len(prefix):].split(" is ",1)[0].strip()
def file_sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
 return h.hexdigest()
log("="*124)
log("TEST647 — BLIND MULTI-CARTRIDGE SYNTHESIS")
log("START",utc())
log("="*124)
log("[1/11] ENVIRONMENT")
assert torch.cuda.is_available(),"CUDA GPU REQUIRED"
for package,module in [("transformers","transformers"),("accelerate","accelerate"),("faiss-cpu","faiss")]:
 if importlib.util.find_spec(module) is None:
  subprocess.check_call([sys.executable,"-m","pip","install","-q",package])
import transformers,faiss
log("GPU",torch.cuda.get_device_name(0),"TORCH",torch.__version__,"CUDA",torch.version.cuda)
log("[2/11] VERIFIED SOURCE / LOCKED ENGINE")
raw=fetch(URL)
assert gitblob(raw)==SOURCE_BLOB,"SOURCE CHANGED"
src=raw.decode("utf-8")
assert src.count("#<<ENGINE_END>>")==1,"ENGINE BOUNDARY CHANGED"
part1=src.split("#<<ENGINE_END>>",1)[0]
dependency='("gradio","gradio"),'
gradio_import="import numpy as np,torch,transformers,gradio as gr"
assert part1.count(dependency)==1 and part1.count(gradio_import)==1,"BOOTSTRAP STRUCTURE CHANGED"
part1=part1.replace(dependency,"")
part1=part1.replace(gradio_import,'import numpy as np,torch,transformers\nimport types as _types\ngr=_types.SimpleNamespace(__version__="DISABLED_TEST647")')
log("SOURCE_VERIFIED",SOURCE_BLOB)
log("[3/11] REUSE OR COLD BOOT ORIGINAL ENGINE")
ready=bool(globals().get("ENGINE_READY",False) and globals().get("MAM_SH_PART1_OK",False)
 and globals().get("ENGINE_SHA256")==ENGINE_SHA_LOCK and "ENG" in globals() and "INSTR" in globals())
if ready:log("ENGINE_REUSED")
else:
 log("COLD_BOOT_ORIGINAL_ENGINE")
 t=time.perf_counter()
 exec(compile(part1,"HIPPOCAMPUS_TEST647_BOOTSTRAP.py","exec"),globals())
 log("ENGINE_BOOT_SECONDS",round(time.perf_counter()-t,3))
assert ENGINE_READY and MAM_SH_PART1_OK and ENGINE_SHA256==ENGINE_SHA_LOCK
assert all(ok for _,ok in ENGINE_LOCK),"ENGINE_LOCK_FAILED"
assert ENG.TOTAL==TOTAL and len(ENG.MANIFEST)==TOTAL and len(ENG.K)==TOTAL
assert len(ENG.QUEST)==48 and len(ENG.PANEL624)==48
assert tuple(ENG.LOCKED627)==("PEAK50",0.5,0.0)
assert not ENG.model.training and not any(p.requires_grad for p in ENG.model.parameters())
weight0=INSTR.full_weight_sha();sentinel0=INSTR.sentinel()
log("ENGINE_LOCK_PASS",ENGINE_SHA256,"CARTRIDGES",TOTAL)
log("[4/11] FREEZE 16 BLIND TWO-SOURCE TASKS")
# Same deterministic pair construction as TEST646. Gold stays outside the execution loop.
rng=random.Random(SEED)
countries=[]
for ci,item in enumerate(ENG.CAR):
 try:
  country=fact_country(item)
  if country:countries.append((ci,country))
 except Exception:continue
assert len(countries)>=2*N,"INSUFFICIENT STRUCTURED FACTS"
rng.shuffle(countries)
pairs=[];used=set()
for ci,country in countries:
 if ci in used:continue
 for cj,other in countries:
  if cj==ci or cj in used or other==country:continue
  used.update((ci,cj))
  pairs.append((ci,country,cj,other))
  break
 if len(pairs)==N:break
assert len(pairs)==N,"PAIR_CONSTRUCTION_FAILED"
sealed=[];public=[]
for i,(a,ca,b,cb) in enumerate(pairs):
 question=f"Identify the present capital of {ca} and the present capital of {cb}."
 sealed.append(dict(case=i,question=question,gold_ids=[a,b],
  countries=[ca,cb],gold_answers=[str(ENG.CAR[a]["gold"]),str(ENG.CAR[b]["gold"])]))
 public.append(dict(case=i,question=question,countries=[ca,cb]))
seal=digest(json.dumps(sealed,sort_keys=True,ensure_ascii=False).encode())
save("SEALED_GOLD.json",dict(sha256=seal,cases=sealed))
save("PUBLIC_TASKS.json",public)
log("BLIND_TASKS_FROZEN",N,"SEAL",seal)
log("[5/11] VERIFY REFERENCE ROUTER — READ ONLY")
reference=[]
for qi in range(48):
 try:
  q=ENG.QUEST[qi].detach().float().cpu().contiguous()
  ids,_=ENG.candidate626(q)
  fs=ENG.features626(q,ids)
  ranking,_=ENG.rerank626(ids,fs,ENG.LOCKED627)
  row=dict(case=qi,selected=ranking[0] if ranking else None,
   expected=ENG.PANEL624[qi]["target"],pass_check=bool(ranking and ranking[0]==ENG.PANEL624[qi]["target"]))
 except Exception as e:row=dict(case=qi,pass_check=False,error=repr(e))
 reference.append(row)
 if (qi+1)%12==0:log("REFERENCE",qi+1,"/48","CORRECT",sum(x["pass_check"] for x in reference))
save("REFERENCE.json",reference)
assert all(x["pass_check"] for x in reference),"REFERENCE_REGRESSION — BLIND TEST NOT STARTED"
log("[6/11] PREPARE PHYSICAL MEMORY READOUT")
# The selected cartridge is always loaded from the original physical disk file.
# The original ENG.answer() function is used unchanged.
def read_cartridge(ci,question):
 ci=int(ci)
 assert 0<=ci<TOTAL,"INVALID_CARTRIDGE_ID"
 record=ENG.MANIFEST[ci]
 path=ENG.POOL/record["file"]
 assert path.is_file(),"CARTRIDGE_FILE_MISSING"
 assert path.stat().st_size==record["bytes"],"CARTRIDGE_SIZE_MISMATCH"
 assert file_sha(path)==record["sha256"],"CARTRIDGE_SHA_MISMATCH"
 t=time.perf_counter()
 stored=torch.load(path,map_location="cpu",weights_only=True)
 assert len(stored)==28 and all(len(pair)==2 for pair in stored),"CUT3_STRUCTURE_MISMATCH"
 mem=tuple((a.to(ENG.DEVICE),b.to(ENG.DEVICE)) for a,b in stored)
 load_ms=(time.perf_counter()-t)*1000
 try:
  t=time.perf_counter()
  with torch.no_grad():answer=ENG.answer(mem,question)
  answer_ms=(time.perf_counter()-t)*1000
 finally:
  del mem,stored
 return dict(cartridge=ci,answer=str(answer),load_ms=round(load_ms,3),answer_ms=round(answer_ms,3))
log("PHYSICAL_READOUT_READY")
log("[7/11] BLIND DISCOVERY — NO GOLD IN ROUTING")
# The joint question alone determines the candidate ranking.
# Each of the two highest-ranked cartridges is selected without any oracle.
predictions=[]
for task in public:
 row=dict(case=task["case"],question=task["question"],countries=task["countries"])
 try:
  t=time.perf_counter()
  with torch.no_grad():
   rep,_=ENG.question_rep(task["question"])
   q=ENG.apply(rep[3],ENG.FIT).detach().float().cpu().contiguous()
  ids,route=ENG.candidate626(q)
  fs=ENG.features626(q,ids)
  ranking,_=ENG.rerank626(ids,fs,ENG.LOCKED627)
  assert len(ranking)>=2,"INSUFFICIENT_CANDIDATES"
  row.update(status="ROUTED",selected=[int(ranking[0]),int(ranking[1])],
   top5=[int(x) for x in ranking[:5]],touched=route["touched"],
   route_ms=round((time.perf_counter()-t)*1000,3))
 except Exception as e:row.update(status="ROUTE_ERROR",selected=[],top5=[],error=repr(e))
 predictions.append(row)
 log("ROUTE",row["case"]+1,"/",N,row["status"],row["selected"])
save("BLIND_ROUTING_LOCKED.json",predictions)
routing_sha=digest(json.dumps(predictions,sort_keys=True,ensure_ascii=False).encode())
log("ROUTING_FROZEN",routing_sha)
log("[8/11] INDEPENDENT PHYSICAL READOUT / NO GOLD FALLBACK")
# Every selected memory is queried independently for each of the two countries.
# Four reads per task: two selected cartridges x two atomic questions.
# No correct ID is inserted if routing misses a required cartridge.
for row in predictions:
 row["reads"]=[]
 if row["status"]!="ROUTED":continue
 for ci in row["selected"]:
  for country in row["countries"]:
   atomic=f"What is the current capital of {country}?"
   try:
    r=read_cartridge(ci,atomic)
    row["reads"].append(dict(country=country,**r,status="OK"))
   except Exception as e:
    row["reads"].append(dict(country=country,cartridge=ci,status="READ_ERROR",error=repr(e)))
 log("READ",row["case"]+1,"/",N,
  "SUCCESS",sum(r["status"]=="OK" for r in row["reads"]),"/4")
 save("PHYSICAL_READ_PROGRESS.json",predictions)
save("PHYSICAL_READ_LOCKED.json",predictions)
log("[9/11] BLIND SYNTHESIS / SOURCE-ATTRIBUTED OUTPUT")
# A fixed, answer-independent combination rule constructs the final two-fact output.
# Each country is assigned the answer with the strongest exact country-specific
# evidence from the selected readouts. No gold answers or source facts enter this rule.
# This is a conservative composition baseline, not yet free-form model synthesis.
def evidence_score(answer):
 a=clean_answer(answer)
 if not a:return -100
 if len(a)>64:return -10
 if "\n" in a:return -5
 if len(a.split())>5:return -5
 if normalized(a) in {"unknown","none","idontknow","notfound","na"}:return -10
 return 1
for row in predictions:
 outputs=[];provenance=[]
 for country in row["countries"]:
  candidates=[r for r in row["reads"] if r["status"]=="OK" and r["country"]==country]
  candidates=sorted(candidates,key=lambda r:(-evidence_score(r["answer"]),row["selected"].index(r["cartridge"])))
  if candidates and evidence_score(candidates[0]["answer"])>0:
   chosen=candidates[0]
   outputs.append(clean_answer(chosen["answer"]))
   provenance.append(chosen["cartridge"])
  else:
   outputs.append(None);provenance.append(None)
 row["synthesis"]=dict(answers=outputs,provenance=provenance,
  complete=all(x is not None for x in outputs))
 log("SYNTHESIS",row["case"]+1,"/",N,"COMPLETE",row["synthesis"]["complete"],
  "ANSWERS",outputs,"SOURCES",provenance)
 save("SYNTHESIS_PROGRESS.json",predictions)
save("BLIND_SYNTHESIS_LOCKED.json",predictions)
pred_sha=digest(json.dumps(predictions,sort_keys=True,ensure_ascii=False).encode())
log("SYNTHESIS_FROZEN",pred_sha)
log("[10/11] POST-HOC EVALUATION — OPEN SEALED GOLD")
evaluation=[]
for pred,gold in zip(predictions,sealed):
 assert pred["case"]==gold["case"],"CASE_MISMATCH"
 expected_ids=set(gold["gold_ids"])
 selected=set(pred["selected"])
 answers=pred["synthesis"]["answers"]
 expected=gold["gold_answers"]
 correct=[normalized(a)==normalized(g) if a is not None else False for a,g in zip(answers,expected)]
 row=dict(case=pred["case"],gold_ids=gold["gold_ids"],selected=pred["selected"],
  both_sources=expected_ids<=selected,
  at_least_one_source=bool(expected_ids&selected),
  answers=answers,gold_answers=expected,
  correct_each=correct,both_answers_correct=all(correct),
  complete=pred["synthesis"]["complete"],provenance=pred["synthesis"]["provenance"])
 evaluation.append(row)
 log("EVAL",row["case"]+1,"/",N,
  "SOURCES",row["both_sources"],"ANSWER",row["both_answers_correct"],
  "EACH",correct)
save("BLIND_EVALUATION.json",evaluation)
log("[11/11] FINAL INTEGRITY / REPORT")
frozen=INSTR.frozen()
weight1=INSTR.full_weight_sha()
integrity=dict(
 source_verified=gitblob(raw)==SOURCE_BLOB,
 engine_verified=ENGINE_SHA256==ENGINE_SHA_LOCK,
 engine_lock=all(ok for _,ok in ENGINE_LOCK),
 weight_unchanged=weight0==weight1,
 sentinel_unchanged=sentinel0==INSTR.sentinel(),
 zero_trainable=frozen["trainable_tensors"]==0,
 no_optimizer=not frozen["optimizer"],
 no_foreign_hooks=frozen["foreign_hooks"]==0,
 no_residual_hooks=not frozen["residual_hooks"])
summary=dict(
 test=TEST,utc=utc(),gpu=torch.cuda.get_device_name(0),
 engine_sha256=ENGINE_SHA_LOCK,source_blob=SOURCE_BLOB,
 cartridges=TOTAL,blind_cases=N,
 reference_correct=sum(x["pass_check"] for x in reference),
 blind_both_sources=sum(x["both_sources"] for x in evaluation),
 blind_at_least_one_source=sum(x["at_least_one_source"] for x in evaluation),
 blind_both_answers_correct=sum(x["both_answers_correct"] for x in evaluation),
 blind_single_answers_correct=sum(sum(x["correct_each"]) for x in evaluation),
 blind_single_answers_total=2*N,
 synthesis_complete=sum(x["complete"] for x in evaluation),
 read_errors=sum(r["status"]=="READ_ERROR" for p in predictions for r in p["reads"]),
 integrity=integrity,
 routing_sha256=routing_sha,synthesis_sha256=pred_sha,gold_seal_sha256=seal,
 scope=[
  "The original locked TEST627 engine, PEAK50 router, CUT3 and model are unchanged.",
  "The task generator knows source facts but the runtime router never receives gold IDs.",
  "Only the top two independently selected cartridges are read; no oracle fallback.",
  "Physical disk SHA256 is checked before every memory read.",
  "Atomic answers are read with the original frozen ENG.answer function.",
  "The final two-answer output uses a deterministic composition rule, not free-form model reasoning.",
  "This is a blind two-fact composition test, not proof of novel autonomous knowledge synthesis.",
  "Source facts come from the same synthetic domain; external domain generalization is not tested.",
  "All outcomes, including failed retrievals and wrong answers, remain recorded."
 ])
summary["infrastructure_status"]="PASS" if all(integrity.values()) and summary["reference_correct"]==48 else "FAIL"
summary["blind_status"]="MEASURED" if len(evaluation)==N else "INCOMPLETE"
summary["status"]="PASS" if summary["infrastructure_status"]=="PASS" and summary["blind_status"]=="MEASURED" else "FAIL"
out=save("TEST647_RESULT.json",dict(summary=summary,reference=reference,
 predictions=predictions,evaluation=evaluation))
log("="*124)
log("TEST647_STATUS",summary["status"])
log("REFERENCE",summary["reference_correct"],"/48")
log("BLIND_BOTH_SOURCES",summary["blind_both_sources"],"/",N)
log("BLIND_BOTH_ANSWERS_CORRECT",summary["blind_both_answers_correct"],"/",N)
log("BLIND_ATOMIC_ANSWERS",summary["blind_single_answers_correct"],"/",2*N)
log("SYNTHESIS_COMPLETE",summary["synthesis_complete"],"/",N)
log("READ_ERRORS",summary["read_errors"])
log("INTEGRITY",json.dumps(integrity))
log("RESULT",out)
log("="*124)
if summary["status"]!="PASS":raise RuntimeError("TEST647_INFRASTRUCTURE_FAILURE — RESULTS SAVED")

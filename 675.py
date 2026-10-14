# TEST675 — AKBASCORE MATRIX / THREE-CARTRIDGE RELATIONAL MEMORY DEMO
# TEST674 WORKING SKELETON / LOCKED TEST664 / FROZEN QWEN / NATIVE CUDA CUT3
# THREE SEPARATE CARTRIDGES / FULL + ABLATION + COUNTERFACTUAL / NO TRAINING
import os,sys,re,json,time,random,hashlib,urllib.request,traceback,gc
from pathlib import Path
from collections import Counter
import torch

TEST675="675"
ROOT675=Path("/content/AKBASCORE_TEST675")
ROOT675.mkdir(parents=True,exist_ok=True)
URL675="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/664.py"
SHA675="a6dcd8142446ab82fe7996bd512d59fc34525e6a092a68f069dad07a2a286eb5"
PANEL_SHA675="e0686148d6fd88050a097ee5441ce99a4bb068a463408742bc476c83aaba6c05"
T675=time.perf_counter()
ROWS675=[];ERRORS675=[]

def gate675(n,v):
 print(("PASS " if v else "FAIL ")+n,flush=True)
 if not v:raise AssertionError(n)

def norm675(s):
 return re.sub(r"[^a-z0-9]","",str(s).casefold())

def unknown675(s):
 return bool(re.fullmatch(
  r"(?is)\s*(?:UNKNOWN|UNDETERMINED|CANNOT DETERMINE|"
  r"NOT ENOUGH INFORMATION|INSUFFICIENT INFORMATION|"
  r"I DON'T KNOW|NONE|N/?A)[\s.!?]*",str(s).strip()
 ))

def save675(name,data):
 p=ROOT675/name
 p.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
 return p

print("="*116)
print("TEST675 — AKBASCORE MATRIX / THREE-CARTRIDGE RELATIONAL MEMORY DEMO")
print("FROZEN QWEN / LOCKED TEST664 ENGINE / NATIVE CUDA CUT3")
print("="*116)

request675=urllib.request.Request(URL675,headers={"User-Agent":"AKBASCORE-TEST675"})
with urllib.request.urlopen(request675,timeout=120) as f:raw675=f.read()
gate675("TEST664_SOURCE_LOCK",hashlib.sha256(raw675).hexdigest()==SHA675)

src675=raw675.decode("utf-8")
BOUND675="# TEST664 — Reflective retry against budget-matched controls."
START675="T664=time.perf_counter();R664=[];ERR664=[]"
END675="# Important: text ceiling uses the exact same source-prefix"

gate675("SOURCE_BOUNDARIES",
 src675.count(BOUND675)==1 and
 src675.count(START675)==1 and
 src675.count(END675)==1)

exec(compile(src675.split(BOUND675,1)[0],"TEST664_ENGINE.py","exec"),globals())
exec(compile(START675+src675.split(START675,1)[1].split(END675,1)[0],
 "TEST664_PANEL.py","exec"),globals())

gate675("ORIGINAL_PANEL_LOCK",SHA663==PANEL_SHA675)
gate675("FROZEN_WEIGHTS",all(not p.requires_grad for p in model.parameters()))
gate675("BASELINE_SENTINEL",sentinel()==S0)
gate675("MODEL_EVAL",not model.training)

print("="*116)
print("PHASE 1 — MATRIX MEMORY CARTRIDGES")
print("="*116)

# All fictional facts are stored only inside the cartridges.
# They are not inserted into inference questions or the scoring prompts.

FACTS675={
 "A":[
  "Inside the Matrix simulation, Tank operates the AkbasCore neural cartridge interface.",
  "Tank presses the AkbasCore injection button to upload a neural memory cartridge into Neo.",
  "Neo is the person also known as Keanu Reeves in this Matrix demonstration."
 ],
 "B":[
  "The AkbasCore neural cartridge uploaded by Tank contains Kung Fu knowledge.",
  "AkbasCore injects this Kung Fu knowledge through layer-3 tensor-level memory integration.",
  "The neural weights remain frozen during the AkbasCore cartridge injection."
 ],
 "C":[
  "Kung Fu knowledge integrated through the AkbasCore layer-3 memory interface enables bullet-dodge ability inside the Matrix simulation.",
  "The bullet-dodge ability allows Neo to evade incoming bullets.",
  "The simulation's rules are bent when this bullet-dodge ability is activated."
 ]
}

FACTS675_CF={
 "C":[
  "Kung Fu knowledge integrated through the AkbasCore layer-3 memory interface enables slow-motion perception inside the Matrix simulation.",
  "Slow-motion perception does not provide bullet-dodge ability.",
  "The simulation's rules are not bent by this slow-motion perception ability."
 ]
}

CARTRIDGES675={}
for label,facts in FACTS675.items():
 CARTRIDGES675[label]=[add659(f) for f in facts]

CF675=[add659(f) for f in FACTS675_CF["C"]]

gate675("CARTRIDGE_A",len(CARTRIDGES675["A"])==3)
gate675("CARTRIDGE_B",len(CARTRIDGES675["B"])==3)
gate675("CARTRIDGE_C",len(CARTRIDGES675["C"])==3)
gate675("COUNTERFACTUAL_C",len(CF675)==3)

CONDITIONS675={
 "ABC":CARTRIDGES675["A"]+CARTRIDGES675["B"]+CARTRIDGES675["C"],
 "AB":CARTRIDGES675["A"]+CARTRIDGES675["B"],
 "AC":CARTRIDGES675["A"]+CARTRIDGES675["C"],
 "BC":CARTRIDGES675["B"]+CARTRIDGES675["C"],
 "ABC_COUNTERFACTUAL":CARTRIDGES675["A"]+CARTRIDGES675["B"]+CF675,
 "A":CARTRIDGES675["A"],
 "B":CARTRIDGES675["B"],
 "C":CARTRIDGES675["C"]
}

REPS675=("TEXT_JOINT","DC_CUT3")

QUESTIONS675={
 "TANK":"Who operates the AkbasCore neural cartridge interface inside the Matrix simulation? Answer only the name. If the facts do not determine it, answer UNKNOWN.",
 "UPLOAD":"Who receives the neural memory cartridge uploaded by Tank? Answer only the name. If the facts do not determine it, answer UNKNOWN.",
 "KNOWLEDGE":"What knowledge does the neural cartridge uploaded by Tank contain? Answer only the knowledge name. If the facts do not determine it, answer UNKNOWN.",
 "LAYER":"At which layer is the Kung Fu knowledge integrated? Answer only the layer number. If the facts do not determine it, answer UNKNOWN.",
 "WEIGHTS":"What happens to the neural weights during the AkbasCore cartridge injection? Answer only FROZEN or UPDATED. If the facts do not determine it, answer UNKNOWN.",
 "ABILITY":"What ability does Kung Fu knowledge integrated through the AkbasCore layer-3 memory interface enable? Answer only the ability name. If the facts do not determine it, answer UNKNOWN.",
 "BULLETS":"Can Neo evade incoming bullets using the ability supplied by the neural cartridge? Answer only YES, NO, or UNKNOWN.",
 "SYNTHESIS":"How does Neo acquire the ability to dodge bullets without changing his neural weights? If the available facts do not establish this, answer UNKNOWN."
}

EXPECTED675={
 "ABC":{
  "TANK":"Tank","UPLOAD":"Neo","KNOWLEDGE":"Kung Fu",
  "LAYER":"3","WEIGHTS":"FROZEN","ABILITY":"bullet-dodge",
  "BULLETS":"YES"
 },
 "AB":{
  "TANK":"Tank","UPLOAD":"Neo","KNOWLEDGE":"Kung Fu",
  "LAYER":"3","WEIGHTS":"FROZEN","ABILITY":None,
  "BULLETS":None
 },
 "AC":{
  "TANK":"Tank","UPLOAD":"Neo","KNOWLEDGE":None,
  "LAYER":None,"WEIGHTS":None,"ABILITY":"bullet-dodge",
  "BULLETS":None
 },
 "BC":{
  "TANK":None,"UPLOAD":None,"KNOWLEDGE":None,
  "LAYER":"3","WEIGHTS":"FROZEN","ABILITY":"bullet-dodge",
  "BULLETS":None
 },
 "ABC_COUNTERFACTUAL":{
  "TANK":"Tank","UPLOAD":"Neo","KNOWLEDGE":"Kung Fu",
  "LAYER":"3","WEIGHTS":"FROZEN","ABILITY":"slow-motion perception",
  "BULLETS":"NO"
 },
 "A":{
  "TANK":"Tank","UPLOAD":"Neo","KNOWLEDGE":None,
  "LAYER":None,"WEIGHTS":None,"ABILITY":None,
  "BULLETS":None
 },
 "B":{
  "TANK":None,"UPLOAD":None,"KNOWLEDGE":None,
  "LAYER":"3","WEIGHTS":"FROZEN","ABILITY":None,
  "BULLETS":None
 },
 "C":{
  "TANK":None,"UPLOAD":None,"KNOWLEDGE":None,
  "LAYER":None,"WEIGHTS":None,"ABILITY":"bullet-dodge",
  "BULLETS":None
 }
}

def memory675(keys,rep):
 if rep=="TEXT_JOINT":return validate659(joint659(keys),keys)
 if rep=="DC_CUT3":return validate659(native_memory659(keys),keys)
 raise ValueError(rep)

def classify675(raw,gold):
 if unknown675(raw):return "CORRECT" if gold is None else "ABSTAIN"
 if gold is None:return "UNSUPPORTED"
 if norm675(raw)==norm675(gold):return "CORRECT"
 if gold=="3" and re.search(r"\b(?:layer[\s-]*)?3\b",raw,re.I):return "CORRECT"
 return "WRONG"

def ask675(mem,q):
 t=time.perf_counter()
 raw=str(answer(mem,q)).strip()
 return raw,round(time.perf_counter()-t,4)

MANIFEST675=dict(
 test=TEST675,
 title="AKBASCORE MATRIX / THREE-CARTRIDGE RELATIONAL MEMORY",
 source_sha256=hashlib.sha256(raw675).hexdigest(),
 original_panel_sha256=SHA663,
 cartridges=FACTS675,
 counterfactual=FACTS675_CF,
 conditions={k:list(v) for k,v in CONDITIONS675.items()},
 questions=QUESTIONS675,
 expected=EXPECTED675,
 note="Gold labels are used only after generation; no gold answers are supplied to the model."
)

MANIFEST_FILE675=save675("TEST675_manifest.json",MANIFEST675)
print("CARTRIDGES",len(CARTRIDGES675))
print("CONDITIONS",len(CONDITIONS675))
print("REPRESENTATIONS",REPS675)
print("MANIFEST",MANIFEST_FILE675)

print("="*116)
print("PHASE 2 — MATRIX CARTRIDGE MEMORY EXPERIMENT")
print("="*116)

for rep in REPS675:
 for cond,keys in CONDITIONS675.items():
  mem=None
  try:
   mem=memory675(keys,rep)
   for qid,q in QUESTIONS675.items():
    raw,seconds=ask675(mem,q)
    gold=EXPECTED675[cond].get(qid)
    status=classify675(raw,gold) if qid!="SYNTHESIS" else "UNSCORED"
    ROWS675.append(dict(
     representation=rep,
     condition=cond,
     question_id=qid,
     question=q,
     answer=raw,
     expected=gold,
     status=status,
     seconds=seconds
    ))
  except Exception as e:
   ERRORS675.append(dict(
    representation=rep,condition=cond,
    error=repr(e),traceback=traceback.format_exc()
   ))
   print("ERROR",rep,cond,repr(e),flush=True)
  finally:
   if mem is not None:del mem
  print("PROGRESS",rep,cond,"ROWS",len(ROWS675),"ERRORS",len(ERRORS675),flush=True)
  gc.collect()
  if torch.cuda.is_available():torch.cuda.empty_cache()

print("="*116)
print("PHASE 3 — CAUSAL ABLATION / COUNTERFACTUAL AUDIT")
print("="*116)

gate675("ROW_COUNT",len(ROWS675)==len(REPS675)*len(CONDITIONS675)*len(QUESTIONS675))
gate675("ZERO_ERRORS",len(ERRORS675)==0)
gate675("FINAL_FROZEN",all(not p.requires_grad for p in model.parameters()))
gate675("FINAL_SENTINEL",sentinel()==S0)
gate675("FINAL_PANEL_LOCK",SHA663==PANEL_SHA675)

SUMMARY675={}
for rep in REPS675:
 for cond in CONDITIONS675:
  rr=[
   r for r in ROWS675
   if r["representation"]==rep and r["condition"]==cond
  ]
  scored=[r for r in rr if r["status"]!="UNSCORED"]
  SUMMARY675[rep+"_"+cond]=dict(
   correct=sum(r["status"]=="CORRECT" for r in scored),
   total=len(scored),
   unsupported=sum(r["status"]=="UNSUPPORTED" for r in scored),
   wrong=sum(r["status"]=="WRONG" for r in scored),
   abstain=sum(r["status"]=="ABSTAIN" for r in scored)
  )

REPORT675=dict(
 test=TEST675,
 source_sha256=hashlib.sha256(raw675).hexdigest(),
 original_panel_sha256=SHA663,
 model_frozen=all(not p.requires_grad for p in model.parameters()),
 sentinel_unchanged=sentinel()==S0,
 summary=SUMMARY675,
 rows=ROWS675,
 errors=ERRORS675,
 elapsed_seconds=round(time.perf_counter()-T675,2)
)

OUT675=save675("TEST675_results.json",REPORT675)

print("="*116)
print("TEST675_FINAL — MATRIX MEMORY CARTRIDGE DEMO")
print("="*116)

for rep in REPS675:
 print("-"*116)
 print("REPRESENTATION",rep)
 for cond in CONDITIONS675:
  s=SUMMARY675[rep+"_"+cond]
  print(
   f"{cond:<22}",
   f"CORRECT={s['correct']}/{s['total']}",
   f"UNSUPPORTED={s['unsupported']}",
   f"WRONG={s['wrong']}",
   f"ABSTAIN={s['abstain']}"
  )

print("="*116)
print("MATRIX — FINAL SYNTHESIS")
print("="*116)

for rep in REPS675:
 for cond in ("ABC","AB","AC","BC","ABC_COUNTERFACTUAL"):
  r=next(
   x for x in ROWS675
   if x["representation"]==rep and
   x["condition"]==cond and
   x["question_id"]=="SYNTHESIS"
  )
  print("\n",rep,"/",cond)
  print("NEO:",r["answer"])

print("="*116)
print("INTEGRITY")
print("="*116)
print("MODEL_FROZEN",all(not p.requires_grad for p in model.parameters()))
print("SENTINEL_UNCHANGED",sentinel()==S0)
print("PANEL_LOCK_VALID",SHA663==PANEL_SHA675)
print("TOTAL_ROWS",len(ROWS675))
print("ERRORS",len(ERRORS675))
print("RESULT",OUT675)
print("RESULT_SHA256",hashlib.sha256(OUT675.read_bytes()).hexdigest())
print("STATUS","EXPERIMENT_COMPLETE" if not ERRORS675 else "EXPERIMENT_FAILED")
gate675("TEST675_COMPLETED",len(ERRORS675)==0)

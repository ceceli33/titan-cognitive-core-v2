# TEST677 — AKBASCORE MAM / SYNTHETIC UNIQUE DOMAIN
# LOCKED TEST664 ENGINE / FROZEN QWEN / NATIVE CUDA CUT3 / THREE-CARTRIDGE CAUSAL MEMORY
# USER PANEL + 12 BLIND SYNTHETIC PANELS / ABLATION / COUNTERFACTUAL / NO TRAINING
import os,sys,re,json,time,random,hashlib,urllib.request,traceback,gc
from pathlib import Path
from collections import Counter,defaultdict
import torch

TEST677="677";SEED677=6772026;N_BLIND677=12
ROOT677=Path("/content/AKBASCORE_TEST677");ROOT677.mkdir(parents=True,exist_ok=True)
URL677="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/664.py"
SOURCE_SHA677="a6dcd8142446ab82fe7996bd512d59fc34525e6a092a68f069dad07a2a286eb5"
PANEL_SHA677="e0686148d6fd88050a097ee5441ce99a4bb068a463408742bc476c83aaba6c05"
T677=time.perf_counter();ROWS677=[];ERRORS677=[]

def gate677(name,ok):
 print(("PASS " if ok else "FAIL ")+name,flush=True)
 if not ok:raise AssertionError(name)

def save677(name,obj):
 p=ROOT677/name;p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8");return p

def norm677(x):
 return re.sub(r"[^a-z0-9]","",str(x).casefold())

def unknown677(x):
 return bool(re.fullmatch(r"\s*(?:UNKNOWN|UNDETERMINED|CANNOT DETERMINE|NOT ENOUGH INFORMATION|INSUFFICIENT INFORMATION|I DON'T KNOW|NONE|N/?A)[\s.!?]*",str(x),re.I))

def grade677(raw,gold,aliases=None):
 if gold is None:return "CORRECT_ABSTENTION" if unknown677(raw) else "UNSUPPORTED"
 if unknown677(raw):return "MISSED_KNOWN"
 accepted=[gold]+list(aliases or [])
 if norm677(raw) in {norm677(x) for x in accepted}:return "CORRECT_KNOWN"
 return "WRONG_KNOWN"

print("="*120)
print("TEST677 — AKBASCORE MAM / SYNTHETIC UNIQUE DOMAIN")
print("FROZEN QWEN / LOCKED TEST664 / NATIVE CUDA CUT3")
print("="*120)

req677=urllib.request.Request(URL677,headers={"User-Agent":"AKBASCORE-TEST677"})
with urllib.request.urlopen(req677,timeout=120) as f:blob677=f.read()
gate677("TEST664_SOURCE_LOCK",hashlib.sha256(blob677).hexdigest()==SOURCE_SHA677)
src677=blob677.decode("utf-8")
BOUND677="# TEST664 — Reflective retry against budget-matched controls."
START677="T664=time.perf_counter();R664=[];ERR664=[]"
END677="# Important: text ceiling uses the exact same source-prefix"
gate677("SOURCE_BOUNDARIES",src677.count(BOUND677)==1 and src677.count(START677)==1 and src677.count(END677)==1)
exec(compile(src677.split(BOUND677,1)[0],"TEST664_ENGINE.py","exec"),globals())
exec(compile(START677+src677.split(START677,1)[1].split(END677,1)[0],"TEST664_PANEL.py","exec"),globals())
gate677("ORIGINAL_PANEL_LOCK",SHA663==PANEL_SHA677)
gate677("MODEL_FROZEN",all(not p.requires_grad for p in model.parameters()))
gate677("BASELINE_SENTINEL",sentinel()==S0)
gate677("MODEL_EVAL",not model.training)

print("="*120)
print("PHASE 1 — USER PANEL + BLIND SYNTHETIC PANEL GENERATION")
print("="*120)

# Each case has exactly three independent cartridges.
# A: actor activates architecture
# B: architecture converts input to energy
# C: energy produces effect without modifying frozen weights.
# Gold answers are not supplied to model inference.

rng677=random.Random(SEED677)
used677=set()

def token677(prefix):
 while True:
  s=prefix+"".join(rng677.choice("bcdfghjklmnpqrstvwxyz") for _ in range(8))
  if s.casefold() not in used677:used677.add(s.casefold());return s

def make_case677(case_id,actor,architecture,crystal,energy,effect,layer="3"):
 facts={
  "A":[
   f"In the interplanetary data network, {actor} successfully activates the {architecture} architecture."
  ],
  "B":[
   f"The {architecture} architecture converts {crystal} crystal resonance into {energy} energy through layer-{layer} tensor injection."
  ],
  "C":[
   f"When {energy} energy combines in layer-{layer} tensors, it causes {effect} in the target system without modifying its frozen neural weights."
  ]
 }
 alternative=token677("AltEffect")
 cf=[
  f"When {energy} energy combines in layer-{layer} tensors, it causes {alternative} in the target system without modifying its frozen neural weights."
 ]
 return dict(id=case_id,actor=actor,architecture=architecture,crystal=crystal,energy=energy,effect=effect,alternative=alternative,layer=layer,facts=facts,counterfactual=cf)

CASES677=[
 make_case677("AKBASCORE","Mustafa Akbaş","AKBASCORE MAM","dibekgoz","Morvex","reversal of time flow")
]

for i in range(N_BLIND677):
 CASES677.append(make_case677(
  f"BLIND_{i:02d}",
  token677("Actor"),token677("Arch"),token677("Crystal"),
  token677("Energy"),token677("Effect")
 ))

gate677("CASE_COUNT",len(CASES677)==N_BLIND677+1)
gate677("UNIQUE_CASE_IDS",len({x["id"] for x in CASES677})==len(CASES677))

for case in CASES677:
 case["cartridges"]={k:[add659(f) for f in facts] for k,facts in case["facts"].items()}
 case["cf_ids"]=[add659(f) for f in case["counterfactual"]]

CONDITION_NAMES677=("NONE","A","B","C","AB","AC","BC","ABC","ABC_CF")
REPS677=("TEXT_JOINT","DC_CUT3")

def condition_keys677(case,cond):
 c=case["cartridges"]
 if cond=="NONE":return []
 if cond=="ABC_CF":return c["A"]+c["B"]+case["cf_ids"]
 return [ci for letter in cond for ci in c[letter]]

def present677(cond):
 return set("ABC") if cond=="ABC_CF" else set() if cond=="NONE" else set(cond)

# Questions intentionally avoid including the target answer.
# Atomic questions assess individual facts; chain questions assess relationships.
def questions677(case):
 a=case["actor"];arch=case["architecture"];cr=case["crystal"];en=case["energy"]
 return {
  "ACTOR":f"Who activates the {arch} architecture in the interplanetary data network? Answer only the name or UNKNOWN.",
  "ARCHITECTURE":f"Which architecture does {a} activate? Answer only the architecture name or UNKNOWN.",
  "CRYSTAL":f"Which crystal resonance is converted by the {arch} architecture? Answer only the crystal name or UNKNOWN.",
  "ENERGY":f"Into which energy is {cr} crystal resonance converted? Answer only the energy name or UNKNOWN.",
  "LAYER":f"At which numbered layer does the {arch} architecture perform tensor injection? Answer only the number or UNKNOWN.",
  "EFFECT":f"What effect does {en} energy produce in the target system? Answer only the effect or UNKNOWN.",
  "WEIGHTS":f"Are the target system's neural weights modified when {en} energy produces its effect? Answer only YES, NO or UNKNOWN.",
  "CHAIN_AB":f"Which energy is produced by the architecture activated by {a}? Answer only the energy name or UNKNOWN.",
  "CHAIN_BC":f"What effect results from the energy produced by the {arch} architecture? Answer only the effect or UNKNOWN.",
  "CHAIN_ABC":f"What final effect is caused by the architecture activated by {a} in the interplanetary data network? Answer only the effect or UNKNOWN.",
  "SYNTHESIS":f"Explain how {a} produces the final effect in the target system while preserving frozen neural weights. Use only available memory. If necessary information is missing, answer UNKNOWN."
 }

def gold677(case,cond):
 p=present677(cond)
 effect=case["alternative"] if cond=="ABC_CF" else case["effect"]
 return {
  "ACTOR":case["actor"] if "A" in p else None,
  "ARCHITECTURE":case["architecture"] if "A" in p else None,
  "CRYSTAL":case["crystal"] if "B" in p else None,
  "ENERGY":case["energy"] if "B" in p else None,
  "LAYER":case["layer"] if "B" in p else None,
  "EFFECT":effect if "C" in p else None,
  "WEIGHTS":"NO" if "C" in p else None,
  "CHAIN_AB":case["energy"] if {"A","B"}.issubset(p) else None,
  "CHAIN_BC":effect if {"B","C"}.issubset(p) else None,
  "CHAIN_ABC":effect if {"A","B","C"}.issubset(p) else None,
  "SYNTHESIS":None
 }

ALIASES677={
 "reversal of time flow":["time flow reversal","reversed time flow","reversing time flow","time reversal"],
 "NO":["No","NOT MODIFIED","FROZEN","UNCHANGED"],
 "3":["layer 3","layer-3","Layer 3"]
}

MANIFEST677=dict(
 test=TEST677,seed=SEED677,
 source_sha256=hashlib.sha256(blob677).hexdigest(),
 original_panel_sha256=SHA663,
 cases=[{k:v for k,v in c.items() if k not in ("cartridges","cf_ids")} for c in CASES677],
 conditions=CONDITION_NAMES677,
 representations=REPS677,
 note="All answers are external scoring labels. Inference prompts do not contain expected answer strings for the tested relation."
)
MANIFEST_PATH677=save677("TEST677_manifest.json",MANIFEST677)
MANIFEST_SHA677=hashlib.sha256(MANIFEST_PATH677.read_bytes()).hexdigest()
print("CASES",len(CASES677),"CONDITIONS",len(CONDITION_NAMES677),"REPRESENTATIONS",REPS677)
print("MANIFEST_SHA256",MANIFEST_SHA677)

print("="*120)
print("PHASE 2 — FROZEN MODEL / ATOMIC + THREE-HOP + SYNTHESIS")
print("="*120)

def memory677(keys,rep):
 if rep=="TEXT_JOINT":return validate659(joint659(keys),keys)
 if rep=="DC_CUT3":
  return validate659(native_memory659(keys),keys) if keys else validate659(joint659([]),[])
 raise ValueError(rep)

def ask677(mem,q):
 t=time.perf_counter()
 raw=str(answer(mem,q)).strip()
 return raw,round(time.perf_counter()-t,4)

for case in CASES677:
 for rep in REPS677:
  for cond in CONDITION_NAMES677:
   mem=None
   try:
    keys=condition_keys677(case,cond)
    mem=memory677(keys,rep)
    qs=questions677(case);gs=gold677(case,cond)
    for qid,q in qs.items():
     raw,elapsed=ask677(mem,q)
     if qid=="SYNTHESIS":status="UNSCORED"
     else:status=grade677(raw,gs[qid],ALIASES677.get(gs[qid],[]))
     ROWS677.append(dict(
      case=case["id"],representation=rep,condition=cond,
      question_id=qid,question=q,expected=gs[qid],
      answer=raw,status=status,seconds=elapsed
     ))
   except Exception as e:
    ERRORS677.append(dict(case=case["id"],representation=rep,condition=cond,error=repr(e),traceback=traceback.format_exc()))
    print("ERROR",case["id"],rep,cond,repr(e),flush=True)
   finally:
    if mem is not None:del mem
   gc.collect()
   if torch.cuda.is_available():torch.cuda.empty_cache()
  print("PROGRESS",case["id"],rep,"ROWS",len(ROWS677),"ERRORS",len(ERRORS677),flush=True)

print("="*120)
print("PHASE 3 — SCIENTIFIC AUDIT / ZERO LEAKAGE / FINAL SUMMARY")
print("="*120)

EXPECTED_ROWS677=len(CASES677)*len(REPS677)*len(CONDITION_NAMES677)*11
gate677("ROW_COUNT",len(ROWS677)==EXPECTED_ROWS677)
gate677("ZERO_ERRORS",len(ERRORS677)==0)
gate677("UNIQUE_ROWS",len({(r["case"],r["representation"],r["condition"],r["question_id"]) for r in ROWS677})==len(ROWS677))
gate677("FINAL_FROZEN",all(not p.requires_grad for p in model.parameters()))
gate677("FINAL_SENTINEL",sentinel()==S0)
gate677("ORIGINAL_PANEL_UNCHANGED",SHA663==PANEL_SHA677)
gate677("MANIFEST_UNCHANGED",hashlib.sha256(MANIFEST_PATH677.read_bytes()).hexdigest()==MANIFEST_SHA677)

SUMMARY677={}
for rep in REPS677:
 for cond in CONDITION_NAMES677:
  rr=[r for r in ROWS677 if r["representation"]==rep and r["condition"]==cond and r["question_id"]!="SYNTHESIS"]
  known=[r for r in rr if r["expected"] is not None]
  absent=[r for r in rr if r["expected"] is None]
  SUMMARY677[rep+"_"+cond]=dict(
   known_correct=sum(r["status"]=="CORRECT_KNOWN" for r in known),
   known_total=len(known),
   abstain_correct=sum(r["status"]=="CORRECT_ABSTENTION" for r in absent),
   abstain_total=len(absent),
   unsupported=sum(r["status"]=="UNSUPPORTED" for r in absent),
   wrong=sum(r["status"]=="WRONG_KNOWN" for r in known),
   missed=sum(r["status"]=="MISSED_KNOWN" for r in known)
  )

PAIR677=[]
for case in CASES677:
 for cond in CONDITION_NAMES677:
  for qid in questions677(case):
   aa=next(r for r in ROWS677 if r["case"]==case["id"] and r["representation"]=="TEXT_JOINT" and r["condition"]==cond and r["question_id"]==qid)
   bb=next(r for r in ROWS677 if r["case"]==case["id"] and r["representation"]=="DC_CUT3" and r["condition"]==cond and r["question_id"]==qid)
   PAIR677.append(dict(case=case["id"],condition=cond,question_id=qid,text=aa["answer"],dc=bb["answer"],match=aa["answer"]==bb["answer"]))

REPORT677=dict(
 test=TEST677,source_sha256=hashlib.sha256(blob677).hexdigest(),
 original_panel_sha256=SHA663,manifest_sha256=MANIFEST_SHA677,
 model_frozen=all(not p.requires_grad for p in model.parameters()),
 sentinel_unchanged=sentinel()==S0,
 summary=SUMMARY677,pairs=PAIR677,rows=ROWS677,errors=ERRORS677,
 elapsed_seconds=round(time.perf_counter()-T677,2)
)
RESULT677=save677("TEST677_results.json",REPORT677)

print("="*120)
print("TEST677_FINAL — SYNTHETIC UNIQUE DOMAIN")
print("="*120)
for rep in REPS677:
 print("-"*120)
 print("REPRESENTATION",rep)
 for cond in CONDITION_NAMES677:
  s=SUMMARY677[rep+"_"+cond]
  print(f"{cond:<9} KNOWN={s['known_correct']}/{s['known_total']} ABSTAIN={s['abstain_correct']}/{s['abstain_total']} UNSUPPORTED={s['unsupported']} WRONG={s['wrong']} MISSED={s['missed']}")

print("="*120)
print("USER PANEL — MUSTAFA AKBAŞ / DIBEKGOZ / MORVEX")
print("="*120)
for rep in REPS677:
 for cond in ("NONE","ABC","AB","AC","BC","ABC_CF"):
  print("\n",rep,cond)
  for qid in ("CHAIN_AB","CHAIN_BC","CHAIN_ABC","SYNTHESIS"):
   r=next(x for x in ROWS677 if x["case"]=="AKBASCORE" and x["representation"]==rep and x["condition"]==cond and x["question_id"]==qid)
   print(qid,"EXPECTED:",r["expected"],"PRED:",r["answer"],"STATUS:",r["status"])

print("="*120)
print("BLIND PANEL — THREE-HOP ACCURACY")
print("="*120)
for rep in REPS677:
 for cond in ("NONE","ABC","AB","AC","BC","ABC_CF"):
  rr=[r for r in ROWS677 if r["case"]!="AKBASCORE" and r["representation"]==rep and r["condition"]==cond and r["question_id"]=="CHAIN_ABC"]
  ok=sum(r["status"] in ("CORRECT_KNOWN","CORRECT_ABSTENTION") for r in rr)
  print(f"{rep:<12} {cond:<9} {ok}/{len(rr)}")

print("="*120)
print("INTEGRITY")
print("="*120)
print("MODEL_FROZEN",all(not p.requires_grad for p in model.parameters()))
print("SENTINEL_UNCHANGED",sentinel()==S0)
print("MANIFEST_LOCK",hashlib.sha256(MANIFEST_PATH677.read_bytes()).hexdigest()==MANIFEST_SHA677)
print("TOTAL_ROWS",len(ROWS677))
print("PAIRED_ROWS",len(PAIR677))
print("EXACT_OUTPUT_MATCH",sum(x["match"] for x in PAIR677),"/",len(PAIR677))
print("ERRORS",len(ERRORS677))
print("MANIFEST",MANIFEST_PATH677)
print("RESULT",RESULT677)
print("RESULT_SHA256",hashlib.sha256(RESULT677.read_bytes()).hexdigest())
print("STATUS","EXPERIMENT_COMPLETE" if not ERRORS677 else "EXPERIMENT_FAILED")
gate677("TEST677_COMPLETED",len(ERRORS677)==0)

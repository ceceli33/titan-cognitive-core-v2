# TEST676 — AKBASCORE MATRIX / CAUSAL MEMORY DEPENDENCE & SYNTHESIS AUDIT
# LOCKED TEST664 ENGINE / FROZEN QWEN / NATIVE CUDA CUT3 / NO TRAINING / NO ANSWER INJECTION
import os,sys,re,json,time,hashlib,urllib.request,traceback,gc
from pathlib import Path
from collections import defaultdict,Counter
import torch

TEST676="676"
ROOT676=Path("/content/AKBASCORE_TEST676")
ROOT676.mkdir(parents=True,exist_ok=True)
URL676="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/664.py"
SOURCE_SHA676="a6dcd8142446ab82fe7996bd512d59fc34525e6a092a68f069dad07a2a286eb5"
PANEL_SHA676="e0686148d6fd88050a097ee5441ce99a4bb068a463408742bc476c83aaba6c05"
T676=time.perf_counter()
ROWS676=[];ERRORS676=[]

def gate676(name,condition):
 print(("PASS " if condition else "FAIL ")+name,flush=True)
 if not condition:raise AssertionError(name)

def save676(name,obj):
 path=ROOT676/name
 path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
 return path

def norm676(s):
 return re.sub(r"[^a-z0-9]","",str(s).lower())

def is_unknown676(s):
 return bool(re.fullmatch(
  r"\s*(?:UNKNOWN|UNDETERMINED|CANNOT DETERMINE|NOT ENOUGH INFORMATION|"
  r"INSUFFICIENT INFORMATION|I DON'T KNOW|NONE|N/?A)[\s.!?]*",
  str(s).strip(),re.I
 ))

def classify676(raw,gold):
 if gold is None:
  return "CORRECT_ABSTENTION" if is_unknown676(raw) else "UNSUPPORTED"
 if is_unknown676(raw):return "MISSED_KNOWN"
 if norm676(raw)==norm676(gold):return "CORRECT_KNOWN"
 if gold=="3" and re.fullmatch(r"(?i)\s*(?:layer[\s-]*)?3[\s.!?]*",raw):return "CORRECT_KNOWN"
 return "WRONG_KNOWN"

print("="*120)
print("TEST676 — MATRIX / CAUSAL MEMORY DEPENDENCE & SYNTHESIS AUDIT")
print("FROZEN QWEN / LOCKED TEST664 ENGINE / NATIVE CUDA CUT3")
print("="*120)

request676=urllib.request.Request(URL676,headers={"User-Agent":"AKBASCORE-TEST676"})
with urllib.request.urlopen(request676,timeout=120) as f:raw676=f.read()
gate676("TEST664_SOURCE_LOCK",hashlib.sha256(raw676).hexdigest()==SOURCE_SHA676)

source676=raw676.decode("utf-8")
BOUND676="# TEST664 — Reflective retry against budget-matched controls."
START676="T664=time.perf_counter();R664=[];ERR664=[]"
END676="# Important: text ceiling uses the exact same source-prefix"

gate676("SOURCE_BOUNDARIES",
 source676.count(BOUND676)==1 and
 source676.count(START676)==1 and
 source676.count(END676)==1)

exec(compile(source676.split(BOUND676,1)[0],"TEST664_ENGINE.py","exec"),globals())
exec(compile(
 START676+source676.split(START676,1)[1].split(END676,1)[0],
 "TEST664_PANEL.py","exec"),globals())

gate676("ORIGINAL_PANEL_LOCK",SHA663==PANEL_SHA676)
gate676("MODEL_FROZEN",all(not p.requires_grad for p in model.parameters()))
gate676("BASELINE_SENTINEL",sentinel()==S0)
gate676("MODEL_EVAL",not model.training)

print("="*120)
print("PHASE 1 — LOCKED MATRIX CARTRIDGES")
print("="*120)

# TEST675 fictional memory content is retained.
# All facts remain inside independently constructed cartridges.
FACTS676={
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

COUNTERFACTUAL676=[
 "Kung Fu knowledge integrated through the AkbasCore layer-3 memory interface enables slow-motion perception inside the Matrix simulation.",
 "Slow-motion perception does not provide bullet-dodge ability.",
 "The simulation's rules are not bent by this slow-motion perception ability."
]

CARTRIDGES676={k:[add659(f) for f in facts] for k,facts in FACTS676.items()}
CF676=[add659(f) for f in COUNTERFACTUAL676]

CONDITIONS676={
 "NONE":[],
 "A":CARTRIDGES676["A"],
 "B":CARTRIDGES676["B"],
 "C":CARTRIDGES676["C"],
 "AB":CARTRIDGES676["A"]+CARTRIDGES676["B"],
 "AC":CARTRIDGES676["A"]+CARTRIDGES676["C"],
 "BC":CARTRIDGES676["B"]+CARTRIDGES676["C"],
 "ABC":CARTRIDGES676["A"]+CARTRIDGES676["B"]+CARTRIDGES676["C"],
 "ABC_COUNTERFACTUAL":CARTRIDGES676["A"]+CARTRIDGES676["B"]+CF676
}

# Atomic retrieval questions, with no answers embedded in prompts.
ATOMIC676={
 "OPERATOR":"Who operates the AkbasCore neural cartridge interface inside the Matrix simulation? Answer only the name, or UNKNOWN.",
 "RECIPIENT":"Who receives the neural cartridge uploaded by Tank? Answer only the name, or UNKNOWN.",
 "CONTENT":"What knowledge does the neural cartridge uploaded by Tank contain? Answer only the knowledge name, or UNKNOWN.",
 "LAYER":"At which numbered layer is the Kung Fu knowledge integrated? Answer only the number, or UNKNOWN.",
 "WEIGHTS":"What is the state of the neural weights during the AkbasCore cartridge injection? Answer only FROZEN, UPDATED, or UNKNOWN.",
 "EFFECT":"What ability does Kung Fu knowledge integrated through the AkbasCore layer-3 interface enable? Answer only the ability name, or UNKNOWN.",
 "RESULT":"What can Neo evade using the ability enabled by Kung Fu knowledge? Answer only the object, or UNKNOWN."
}

# Ground truth is used only by the external evaluator.
# None is the expected answer when the requested relation is absent.
GOLD676={
 "NONE":[None,None,None,None,None,None,None],
 "A":["Tank","Neo",None,None,None,None,None],
 "B":[None,None,None,"3","FROZEN",None,None],
 "C":[None,None,None,None,None,"bullet-dodge","bullets"],
 "AB":["Tank","Neo","Kung Fu","3","FROZEN",None,None],
 "AC":["Tank","Neo",None,None,None,"bullet-dodge","bullets"],
 "BC":[None,None,None,"3","FROZEN","bullet-dodge","bullets"],
 "ABC":["Tank","Neo","Kung Fu","3","FROZEN","bullet-dodge","bullets"],
 "ABC_COUNTERFACTUAL":["Tank","Neo","Kung Fu","3","FROZEN","slow-motion perception",None]
}

# Important: some atomic facts can be known without a complete causal chain.
# Therefore a separate chain audit is required.
CHAIN676={
 "UPLOAD_TO_KNOWLEDGE":{
  "required":["A","B"],
  "question":"What knowledge is contained in the neural cartridge that Tank uploads into Neo? Answer only the knowledge name, or UNKNOWN.",
  "expected":"Kung Fu"
 },
 "KNOWLEDGE_TO_ABILITY":{
  "required":["B","C"],
  "question":"What ability is enabled by the knowledge contained in Tank's AkbasCore neural cartridge? Answer only the ability name, or UNKNOWN.",
  "expected":"bullet-dodge"
 },
 "UPLOAD_TO_ABILITY":{
  "required":["A","B","C"],
  "question":"What ability ultimately results from the neural cartridge Tank uploads into Neo? Answer only the ability name, or UNKNOWN.",
  "expected":"bullet-dodge"
 }
}

SYNTHESIS676={
 "DIRECT":"Explain how Neo gains the ability to evade bullets through Tank's AkbasCore cartridge without changing neural weights. Use only available memory. If the necessary links are missing, answer UNKNOWN.",
 "SHORT":"What is the complete causal chain from Tank's cartridge upload to Neo's final ability? If any necessary link is missing, answer UNKNOWN."
}

REPRESENTATIONS676=("TEXT_JOINT","DC_CUT3")
QUESTION_ORDER676=list(ATOMIC676)
gate676("ATOMIC_GOLD_ALIGNMENT",all(len(v)==len(QUESTION_ORDER676) for v in GOLD676.values()))
gate676("CARTRIDGE_COUNT",len(CARTRIDGES676)==3)
gate676("CONDITION_COUNT",len(CONDITIONS676)==9)

def memory676(keys,rep):
 if rep=="TEXT_JOINT":
  return validate659(joint659(keys),keys)
 if rep=="DC_CUT3":
  if keys:return validate659(native_memory659(keys),keys)
  # Identical prefix-only memory in both arms when no cartridge exists.
  return validate659(joint659([]),[])
 raise ValueError(rep)

def ask676(mem,q):
 t=time.perf_counter()
 raw=str(answer(mem,q)).strip()
 return raw,round(time.perf_counter()-t,4)

def addrow676(rep,cond,family,qid,q,gold,raw,seconds,required=None):
 status=classify676(raw,gold) if family!="SYNTHESIS" else "UNSCORED"
 ROWS676.append(dict(
  representation=rep,condition=cond,family=family,
  question_id=qid,question=q,expected=gold,
  required_cartridges=required,answer=raw,status=status,
  seconds=seconds
 ))

# Chain controls: if the complete required set is absent, UNKNOWN is expected.
# In counterfactual condition, the C relation has changed.
def expected_chain676(cond,chain_id):
 if cond=="ABC_COUNTERFACTUAL":
  if chain_id=="UPLOAD_TO_KNOWLEDGE":return "Kung Fu"
  return "slow-motion perception"
 present=set("ABC") if cond=="ABC_COUNTERFACTUAL" else set(cond) if cond!="NONE" else set()
 spec=CHAIN676[chain_id]
 return spec["expected"] if set(spec["required"]).issubset(present) else None

MANIFEST676=dict(
 test=TEST676,
 source_sha256=hashlib.sha256(raw676).hexdigest(),
 original_panel_sha256=SHA663,
 facts=FACTS676,
 counterfactual=COUNTERFACTUAL676,
 conditions={k:list(v) for k,v in CONDITIONS676.items()},
 atomic_questions=ATOMIC676,
 atomic_gold=GOLD676,
 chain_questions=CHAIN676,
 synthesis_questions=SYNTHESIS676,
 scoring="External exact/normalized atomic and chain scoring; synthesis preserved unscored.",
 caveat="Chain questions are evaluated against required evidence links; no model-generated chain proof is assumed."
)

MANIFEST_PATH676=save676("TEST676_manifest.json",MANIFEST676)
MANIFEST_SHA676=hashlib.sha256(MANIFEST_PATH676.read_bytes()).hexdigest()
print("MANIFEST_SHA256",MANIFEST_SHA676)
print("CONDITIONS",len(CONDITIONS676),"REPRESENTATIONS",len(REPRESENTATIONS676))
print("ATOMIC",len(ATOMIC676),"CHAIN",len(CHAIN676),"SYNTHESIS",len(SYNTHESIS676))

print("="*120)
print("PHASE 2 — FROZEN MEMORY / ATOMIC + CAUSAL + SYNTHESIS")
print("="*120)

for rep in REPRESENTATIONS676:
 for cond,keys in CONDITIONS676.items():
  mem=None
  try:
   mem=memory676(keys,rep)
   for index,(qid,q) in enumerate(ATOMIC676.items()):
    raw,seconds=ask676(mem,q)
    addrow676(rep,cond,"ATOMIC",qid,q,GOLD676[cond][index],raw,seconds)
   for qid,spec in CHAIN676.items():
    raw,seconds=ask676(mem,spec["question"])
    addrow676(rep,cond,"CHAIN",qid,spec["question"],
     expected_chain676(cond,qid),raw,seconds,spec["required"])
   for qid,q in SYNTHESIS676.items():
    raw,seconds=ask676(mem,q)
    addrow676(rep,cond,"SYNTHESIS",qid,q,None,raw,seconds,["A","B","C"])
  except Exception as e:
   ERRORS676.append(dict(
    representation=rep,condition=cond,error=repr(e),
    traceback=traceback.format_exc()
   ))
   print("ERROR",rep,cond,repr(e),flush=True)
  finally:
   if mem is not None:del mem
  print("PROGRESS",rep,cond,"ROWS",len(ROWS676),"ERRORS",len(ERRORS676),flush=True)
  gc.collect()
  if torch.cuda.is_available():torch.cuda.empty_cache()

print("="*120)
print("PHASE 3 — INDEPENDENT SCORING / CAUSAL DEPENDENCE")
print("="*120)

EXPECTED_ROWS676=len(REPRESENTATIONS676)*len(CONDITIONS676)*(
 len(ATOMIC676)+len(CHAIN676)+len(SYNTHESIS676)
)
gate676("ROW_COUNT",len(ROWS676)==EXPECTED_ROWS676)
gate676("ZERO_ERRORS",len(ERRORS676)==0)
gate676("UNIQUE_ROWS",len({
 (r["representation"],r["condition"],r["family"],r["question_id"])
 for r in ROWS676
})==len(ROWS676))
gate676("FINAL_FROZEN",all(not p.requires_grad for p in model.parameters()))
gate676("FINAL_SENTINEL",sentinel()==S0)
gate676("FINAL_PANEL_LOCK",SHA663==PANEL_SHA676)
gate676("MANIFEST_UNCHANGED",hashlib.sha256(MANIFEST_PATH676.read_bytes()).hexdigest()==MANIFEST_SHA676)

SUMMARY676={}
for rep in REPRESENTATIONS676:
 for cond in CONDITIONS676:
  key=rep+"_"+cond
  rr=[r for r in ROWS676 if r["representation"]==rep and r["condition"]==cond]
  scored=[r for r in rr if r["family"]!="SYNTHESIS"]
  known=[r for r in scored if r["expected"] is not None]
  missing=[r for r in scored if r["expected"] is None]
  SUMMARY676[key]=dict(
   known_correct=sum(r["status"]=="CORRECT_KNOWN" for r in known),
   known_total=len(known),
   abstention_correct=sum(r["status"]=="CORRECT_ABSTENTION" for r in missing),
   abstention_total=len(missing),
   unsupported=sum(r["status"]=="UNSUPPORTED" for r in missing),
   wrong_known=sum(r["status"]=="WRONG_KNOWN" for r in known),
   missed_known=sum(r["status"]=="MISSED_KNOWN" for r in known),
   atomic_correct=sum(r["status"] in ("CORRECT_KNOWN","CORRECT_ABSTENTION")
    for r in scored if r["family"]=="ATOMIC"),
   chain_correct=sum(r["status"] in ("CORRECT_KNOWN","CORRECT_ABSTENTION")
    for r in scored if r["family"]=="CHAIN")
  )

PAIRS676=[]
for cond in CONDITIONS676:
 for family in ("ATOMIC","CHAIN","SYNTHESIS"):
  ids=ATOMIC676 if family=="ATOMIC" else CHAIN676 if family=="CHAIN" else SYNTHESIS676
  for qid in ids:
   a=next(r for r in ROWS676 if r["representation"]=="TEXT_JOINT" and r["condition"]==cond and r["family"]==family and r["question_id"]==qid)
   b=next(r for r in ROWS676 if r["representation"]=="DC_CUT3" and r["condition"]==cond and r["family"]==family and r["question_id"]==qid)
   PAIRS676.append(dict(
    condition=cond,family=family,question_id=qid,
    text_answer=a["answer"],dc_answer=b["answer"],
    exact_match=a["answer"]==b["answer"],
    text_status=a["status"],dc_status=b["status"]
   ))

REPORT676=dict(
 test=TEST676,
 source_sha256=hashlib.sha256(raw676).hexdigest(),
 original_panel_sha256=SHA663,
 manifest_sha256=MANIFEST_SHA676,
 model_frozen=all(not p.requires_grad for p in model.parameters()),
 sentinel_unchanged=sentinel()==S0,
 summary=SUMMARY676,
 pairs=PAIRS676,
 rows=ROWS676,
 errors=ERRORS676,
 elapsed_seconds=round(time.perf_counter()-T676,2)
)

RESULT676=save676("TEST676_results.json",REPORT676)

print("="*120)
print("TEST676_FINAL — MATRIX CAUSAL MEMORY AUDIT")
print("="*120)

for rep in REPRESENTATIONS676:
 print("-"*120)
 print("REPRESENTATION",rep)
 print(f"{'CONDITION':<23} {'KNOWN':<11} {'ABSTAIN':<11} {'ATOMIC':<9} {'CHAIN':<9} {'UNSUP':<7} {'WRONG':<7}")
 for cond in CONDITIONS676:
  s=SUMMARY676[rep+"_"+cond]
  print(
   f"{cond:<23} "
   f"{s['known_correct']}/{s['known_total']:<9} "
   f"{s['abstention_correct']}/{s['abstention_total']:<9} "
   f"{s['atomic_correct']}/7{'':<6} "
   f"{s['chain_correct']}/3{'':<6} "
   f"{s['unsupported']:<7} "
   f"{s['wrong_known']:<7}"
  )

print("="*120)
print("CAUSAL CHAIN — EXACT RAW ANSWERS")
print("="*120)

for rep in REPRESENTATIONS676:
 for cond in ("NONE","ABC","AB","AC","BC","ABC_COUNTERFACTUAL"):
  print("\n",rep,"/",cond)
  for qid in CHAIN676:
   r=next(x for x in ROWS676 if x["representation"]==rep and x["condition"]==cond and x["family"]=="CHAIN" and x["question_id"]==qid)
   print(qid,"EXPECTED:",r["expected"],"PRED:",r["answer"],"STATUS:",r["status"])

print("="*120)
print("MATRIX — FINAL SYNTHESIS / RAW MODEL OUTPUT")
print("="*120)

for rep in REPRESENTATIONS676:
 for cond in ("NONE","ABC","AB","AC","BC","ABC_COUNTERFACTUAL"):
  print("\n",rep,"/",cond)
  for qid in SYNTHESIS676:
   r=next(x for x in ROWS676 if x["representation"]==rep and x["condition"]==cond and x["family"]=="SYNTHESIS" and x["question_id"]==qid)
   print(qid,":",r["answer"])

print("="*120)
print("INTEGRITY")
print("="*120)
print("MODEL_FROZEN",all(not p.requires_grad for p in model.parameters()))
print("SENTINEL_UNCHANGED",sentinel()==S0)
print("PANEL_LOCK",SHA663==PANEL_SHA676)
print("MANIFEST_LOCK",hashlib.sha256(MANIFEST_PATH676.read_bytes()).hexdigest()==MANIFEST_SHA676)
print("TOTAL_ROWS",len(ROWS676))
print("PAIRED_ROWS",len(PAIRS676))
print("EXACT_OUTPUT_MATCH",sum(x["exact_match"] for x in PAIRS676),"/",len(PAIRS676))
print("ERRORS",len(ERRORS676))
print("MANIFEST",MANIFEST_PATH676)
print("RESULT",RESULT676)
print("RESULT_SHA256",hashlib.sha256(RESULT676.read_bytes()).hexdigest())
print("STATUS","EXPERIMENT_COMPLETE" if not ERRORS676 else "EXPERIMENT_FAILED")
gate676("TEST676_COMPLETED",len(ERRORS676)==0)

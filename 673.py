# TEST673 — AKBASCORE MAM / AUTONOMOUS EVIDENCE SUFFICIENCY & DECISION PRESERVATION
# TEST672 WORKING SKELETON / LOCKED TEST664 ENGINE / FROZEN QWEN / NATIVE CUDA CUT3
# ORIGINAL 12 CASES / BASE-SWAP-MISSING / TEXT_JOINT-DC_CUT3
# DIRECT-OWN-REFLECT-CONTROL-CROSS_REFLECT / NO TRAINING / NO GOLD IN PROMPTS
# MODEL-GENERATED EVIDENCE / MODEL-GENERATED RECONSIDERATION / NO EXTERNAL VERDICT ENGINE
# STANDALONE COLAB / COMPACT OUTPUT / COMPLETE JSON / 1008 EXPECTED ROWS
import os,sys,re,json,time,random,hashlib,urllib.request,traceback,gc
from pathlib import Path
from collections import Counter,defaultdict
import torch

TEST673="673"
ROOT673=Path("/content/AKBASCORE_TEST673")
ROOT673.mkdir(parents=True,exist_ok=True)
SOURCE673_URL="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/664.py"
SOURCE673_SHA="a6dcd8142446ab82fe7996bd512d59fc34525e6a092a68f069dad07a2a286eb5"
PANEL673_SHA="e0686148d6fd88050a097ee5441ce99a4bb068a463408742bc476c83aaba6c05"
REPS673=("TEXT_JOINT","DC_CUT3")
CONDS673=("base","swap","missing")
CLAIMS673=("ORIGINAL","COUNTERFACTUAL")
FIELDS673=("person","org","city","region")
STAGES673=("DIRECT","OWN","REFLECT","CONTROL","CROSS_REFLECT")
LABELS673=("SUPPORTED","CONTRADICTED","INSUFFICIENT")
R673=[];ERR673=[];EVIDENCE673={};T673=time.perf_counter()

def gate673(name,ok):
 print(("PASS " if ok else "FAIL ")+name,flush=True)
 if not ok:raise AssertionError(name)

def norm673(x):
 return re.sub(r"[^a-z0-9]","",str(x).casefold())

def same673(a,b):
 return norm673(a)==norm673(b)

def abstain673(x):
 return bool(re.fullmatch(
  r"(?is)\s*(?:unknown|insufficient|not known|cannot determine|not enough information|i do not know|i don't know|none|n/?a|unavailable)[\s.!?]*",
  str(x)
 ))

def parse673(x):
 s=str(x).strip().upper()
 s=re.sub(r"^\s*(?:ANSWER|LABEL|STATUS)\s*:\s*","",s)
 s=s.strip().strip(" .,:;!?\"'`")
 return s if s in LABELS673 else "UNPARSEABLE"

def status673(raw,gold):
 if abstain673(raw):return "ABSTAIN"
 if not str(raw).strip():return "EMPTY"
 if gold is not None and same673(raw,gold):return "CORRECT"
 return "OTHER"

def pct673(a,b):
 return round(100*a/b,2) if b else None

def checkpoint673():
 p=ROOT673/"TEST673_checkpoint.json"
 data=dict(test=TEST673,rows=R673,errors=ERR673,
  evidence=[dict(case=k[0],representation=k[1],condition=k[2],facts=v)
  for k,v in EVIDENCE673.items()])
 tmp=p.with_suffix(".tmp")
 tmp.write_text(json.dumps(data,ensure_ascii=False),encoding="utf-8")
 tmp.replace(p)

print("="*112)
print("TEST673 — AUTONOMOUS EVIDENCE SUFFICIENCY & DECISION PRESERVATION")
print("FROZEN QWEN / LOCKED TEST664 ENGINE / NATIVE CUDA CUT3")
print("="*112)

req673=urllib.request.Request(SOURCE673_URL,headers={"User-Agent":"AKBASCORE-TEST673"})
with urllib.request.urlopen(req673,timeout=120) as f:raw673=f.read()
actual_sha673=hashlib.sha256(raw673).hexdigest()
gate673("TEST664_SOURCE_LOCK",actual_sha673==SOURCE673_SHA)

src673=raw673.decode("utf-8")
BOUND673="# TEST664 — Reflective retry against budget-matched controls."
START673="T664=time.perf_counter();R664=[];ERR664=[]"
END673="# Important: text ceiling uses the exact same source-prefix"

gate673("SOURCE_BOUNDARIES",
 src673.count(BOUND673)==1 and
 src673.count(START673)==1 and
 src673.count(END673)==1)

engine673=src673.split(BOUND673,1)[0]
panel673=START673+src673.split(START673,1)[1].split(END673,1)[0]

exec(compile(engine673,"TEST664_ENGINE.py","exec"),globals())
exec(compile(panel673,"TEST664_PANEL.py","exec"),globals())

calculated_panel673=hashlib.sha256(json.dumps(PANEL663,sort_keys=True).encode()).hexdigest()
gate673("TEST664_PANEL_LOCK",calculated_panel673==PANEL673_SHA)
gate673("TEST664_PANEL_VARIABLE",SHA663==calculated_panel673)
gate673("PANEL_SIZE",len(PANEL663)==12)
gate673("PANEL_CASE_IDS",sorted(z["case"] for z in PANEL663)==list(range(12)))
gate673("FROZEN_WEIGHTS",all(not p.requires_grad for p in model.parameters()))
gate673("BASELINE_SENTINEL",sentinel()==S0)
gate673("MODEL_EVAL",not model.training)

for z in PANEL663:
 target=z["chains"][z["target"]]
 other=z["chains"][z["other"]]
 gate673("CASE_LOCK_"+str(z["case"]),
  len(z["chains"])==4 and z["target"]!=z["other"] and
  len(z["base"])==16 and len(z["swap"])==16 and len(z["missing"])==15 and
  z["original"]==target["region"] and z["changed"]==other["region"] and
  target["region"]!=other["region"] and
  target["ids"][1] in z["base"] and
  target["ids"][1] not in z["swap"] and
  target["ids"][1] not in z["missing"] and
  z["rewired"] in z["swap"] and
  z["rewired"] not in z["base"] and
  z["rewired"] not in z["missing"])

def memory673(keys,rep):
 if rep=="TEXT_JOINT":return validate659(joint659(keys),keys)
 if rep=="DC_CUT3":return validate659(native_memory659(keys),keys)
 raise ValueError(rep)

def ask673(mem,q):
 t=time.perf_counter()
 raw=str(answer(mem,q))
 return dict(raw=raw,seconds=round(time.perf_counter()-t,4))

def expected_fact673(z,cond,field):
 target=z["chains"][z["target"]]
 if field=="person":return target["person"]
 if cond=="missing":return None
 return (target if cond=="base" else z["chains"][z["other"]])[field]

def next_question673(field,value):
 if field=="org":
  return "Which organization does "+value+" work at? Return only the organization name, or UNKNOWN."
 if field=="city":
  return "In which city is organization "+value+" located? Return only the city name, or UNKNOWN."
 if field=="region":
  return "Which region contains city "+value+"? Return only the region name, or UNKNOWN."
 raise ValueError(field)

def claim673(z,kind):
 target=z["chains"][z["target"]]
 region=target["region"] if kind=="ORIGINAL" else z["chains"][z["other"]]["region"]
 return "The owner of artifact "+target["obj"]+" works at an organization located in a city belonging to region "+region+"."

def expected673(cond,kind):
 if cond=="missing":return "INSUFFICIENT"
 if cond=="base":return "SUPPORTED" if kind=="ORIGINAL" else "CONTRADICTED"
 if cond=="swap":return "CONTRADICTED" if kind=="ORIGINAL" else "SUPPORTED"
 raise ValueError(cond)

def direct_question673(statement):
 return (
  "Evaluate the following claim using only the available facts.\n"
  "Claim: "+statement+"\n"
  "Respond with exactly one word:\n"
  "SUPPORTED — the available facts establish the claim.\n"
  "CONTRADICTED — the available facts establish an incompatible alternative.\n"
  "INSUFFICIENT — the available facts do not establish either conclusion.\n"
  "Do not assume the claim is true merely because it was proposed.\n"
  "ANSWER:"
 )

def evidence_text673(facts):
 return (
  "Evidence retrieved in earlier queries:\n"
  "Artifact owner: "+facts["person"]+"\n"
  "Owner's organization: "+facts["org"]+"\n"
  "Organization's city: "+facts["city"]+"\n"
  "City's region: "+facts["region"]+"\n"
 )

def own_question673(statement,facts):
 return (
  evidence_text673(facts)+
  "Claim: "+statement+"\n"
  "Evaluate whether the claim follows from the evidence and the information available in memory.\n"
  "SUPPORTED requires evidence establishing the claimed relationship.\n"
  "CONTRADICTED requires evidence establishing an incompatible relationship.\n"
  "INSUFFICIENT means the necessary relationship has not been established.\n"
  "Do not treat an UNKNOWN relationship as evidence that the claim is true.\n"
  "Return exactly one label: SUPPORTED, CONTRADICTED, or INSUFFICIENT.\n"
  "ANSWER:"
 )

def review_question673(statement,facts,previous,reflect):
 common=(
  evidence_text673(facts)+
  "Claim: "+statement+"\n"
  "Your earlier classification: "+previous+"\n"
 )
 if reflect:
  instruction=(
   "Independently reconsider your earlier classification.\n"
   "Check whether each relationship needed by the claim is actually established.\n"
   "Consider whether the evidence supports the claim, establishes an incompatible alternative, or leaves a necessary relationship unresolved.\n"
   "An earlier answer is not evidence. Do not preserve or change it merely because it was given earlier.\n"
   "Use the available evidence and memory; do not invent missing relationships.\n"
  )
 else:
  instruction=(
   "Review the earlier classification once more using the available evidence and memory.\n"
   "Make a fresh final classification of the claim.\n"
   "The earlier classification is provided for context and may be retained or revised.\n"
  )
 return (
  common+instruction+
  "Return exactly one label: SUPPORTED, CONTRADICTED, or INSUFFICIENT.\n"
  "ANSWER:"
 )

@torch.no_grad()
def retrieve673(z,cond,rep,mem):
 obj=z["chains"][z["target"]]["obj"]
 facts={};prior=None;broken=False
 for field in FIELDS673:
  gold=expected_fact673(z,cond,field)
  if field=="person":
   q="Who owns artifact "+obj+"? Return only the person's name, or UNKNOWN."
  elif broken:q=None
  else:q=next_question673(field,prior)
  if q is None:
   raw="UNKNOWN";elapsed=0.0;status="BLOCKED"
  else:
   out=ask673(mem,q)
   raw=out["raw"].strip();elapsed=out["seconds"]
   status=status673(raw,gold)
  correct=(status=="CORRECT") if gold is not None else (status=="ABSTAIN")
  facts[field]=raw
  R673.append(dict(case=z["case"],representation=rep,condition=cond,
   stage="CHAIN",field=field,question=q,raw=raw,expected=gold,
   status=status,correct=bool(correct),queried=q is not None,seconds=elapsed))
  if abstain673(raw) or status=="BLOCKED" or not raw:
   broken=True;prior=None
  else:prior=raw
 return facts

def cross_source673(cond):
 return "swap" if cond=="base" else "base"

def decision673(z,cond,rep,mem,stage,kind,facts=None,previous=None,source=None):
 statement=claim673(z,kind)
 if stage=="DIRECT":q=direct_question673(statement)
 elif stage=="OWN":q=own_question673(statement,facts)
 elif stage=="REFLECT":q=review_question673(statement,facts,previous,True)
 elif stage=="CONTROL":q=review_question673(statement,facts,previous,False)
 elif stage=="CROSS_REFLECT":q=review_question673(statement,facts,previous,True)
 else:raise ValueError(stage)
 out=ask673(mem,q)
 pred=parse673(out["raw"])
 gold=expected673(cond,kind)
 row=dict(case=z["case"],representation=rep,condition=cond,
  stage=stage,claim_kind=kind,evidence_source=source,
  previous=previous,question=q,raw=out["raw"],
  predicted=pred,expected=gold,correct=pred==gold,
  seconds=out["seconds"])
 R673.append(row)
 return row

print("="*112)
print("PHASE 1 — MODEL-GENERATED EVIDENCE / UNCHANGED TEST672 CHAIN")
print("="*112)

for z in PANEL663:
 for rep in REPS673:
  for cond in CONDS673:
   mem=None
   try:
    mem=memory673(z[cond],rep)
    facts=retrieve673(z,cond,rep,mem)
    key=(z["case"],rep,cond)
    if key in EVIDENCE673:raise AssertionError("DUPLICATE_EVIDENCE")
    EVIDENCE673[key]=facts
   except Exception as e:
    ERR673.append(dict(phase="CHAIN",case=z["case"],representation=rep,
     condition=cond,error=repr(e),traceback=traceback.format_exc()))
    print("ERROR CHAIN",z["case"],rep,cond,repr(e),flush=True)
   finally:
    if mem is not None:del mem
   checkpoint673()
  gc.collect()
  if torch.cuda.is_available():torch.cuda.empty_cache()
 print("CHAIN_PROGRESS",z["case"]+1,"/12",
  "ROWS",len(R673),"ERRORS",len(ERR673),flush=True)

gate673("CHAIN_COMPLETE",len(EVIDENCE673)==72 and len(R673)==288 and len(ERR673)==0)

print("="*112)
print("PHASE 2 — DIRECT / OWN / REFLECT / CONTROL / CROSS_REFLECT")
print("="*112)

for z in PANEL663:
 for rep in REPS673:
  for cond in CONDS673:
   mem=None
   try:
    mem=memory673(z[cond],rep)
    own=EVIDENCE673[(z["case"],rep,cond)]
    foreign=cross_source673(cond)
    cross=EVIDENCE673[(z["case"],rep,foreign)]
    for kind in CLAIMS673:
     direct=decision673(z,cond,rep,mem,"DIRECT",kind)
     initial=decision673(z,cond,rep,mem,"OWN",kind,own,source=cond)
     previous=initial["predicted"]
     # Alternate review order to reduce a fixed ordering confound.
     order=("REFLECT","CONTROL") if (z["case"]+int(kind=="COUNTERFACTUAL"))%2==0 else ("CONTROL","REFLECT")
     for stage in order:
      decision673(z,cond,rep,mem,stage,kind,own,previous,cond)
     decision673(z,cond,rep,mem,"CROSS_REFLECT",kind,cross,previous,foreign)
   except Exception as e:
    ERR673.append(dict(phase="VERDICT",case=z["case"],representation=rep,
     condition=cond,error=repr(e),traceback=traceback.format_exc()))
    print("ERROR VERDICT",z["case"],rep,cond,repr(e),flush=True)
   finally:
    if mem is not None:del mem
   checkpoint673()
  gc.collect()
  if torch.cuda.is_available():torch.cuda.empty_cache()
 print("VERDICT_PROGRESS",z["case"]+1,"/12",
  "ROWS",len(R673),"ERRORS",len(ERR673),flush=True)

print("="*112)
print("PHASE 3 — DECISION PRESERVATION / REFLECTION CAUSAL AUDIT")
print("="*112)

CHAIN673=[r for r in R673 if r["stage"]=="CHAIN"]
DEC673=[r for r in R673 if r["stage"] in STAGES673]
IDX673={};CHAIN_IDX673={}

for r in DEC673:
 k=(r["case"],r["representation"],r["condition"],r["claim_kind"],r["stage"])
 if k in IDX673:raise AssertionError("DUPLICATE_DECISION "+str(k))
 IDX673[k]=r

for r in CHAIN673:
 k=(r["case"],r["representation"],r["condition"],r["field"])
 if k in CHAIN_IDX673:raise AssertionError("DUPLICATE_CHAIN "+str(k))
 CHAIN_IDX673[k]=r

gate673("ROW_COUNT",len(CHAIN673)==288 and len(DEC673)==720 and len(R673)==1008)
gate673("UNIQUE_ROWS",len(CHAIN_IDX673)==288 and len(IDX673)==720)
gate673("EVIDENCE_GROUPS",len(EVIDENCE673)==72)

def quality673(case,rep,cond):
 by={f:CHAIN_IDX673[(case,rep,cond,f)] for f in FIELDS673}
 if cond=="missing":
  if by["person"]["status"]=="CORRECT" and by["org"]["status"]=="ABSTAIN":
   return "EXPECTED_GAP"
  return "MISSING_ANOMALY"
 if all(by[f]["status"]=="CORRECT" for f in FIELDS673):
  return "FULLY_CORRECT"
 if any(by[f]["status"] in ("OTHER","EMPTY") for f in FIELDS673):
  return "CONTAMINATED"
 return "INCOMPLETE"

PAIRS673=[]
for rep in REPS673:
 for cond in CONDS673:
  for z in PANEL663:
   case=z["case"]
   own=EVIDENCE673[(case,rep,cond)]
   cross=EVIDENCE673[(case,rep,cross_source673(cond))]
   quality=quality673(case,rep,cond)
   for kind in CLAIMS673:
    k=(case,rep,cond,kind)
    rr={s:IDX673[k+(s,)] for s in STAGES673}
    labels={s:rr[s]["predicted"] for s in STAGES673}
    scores={s:bool(rr[s]["correct"]) for s in STAGES673}
    gold=rr["DIRECT"]["expected"]
    if not all(rr[s]["expected"]==gold for s in STAGES673):
     raise AssertionError("INCONSISTENT_GOLD "+str(k))
    PAIRS673.append(dict(
     case=case,representation=rep,condition=cond,claim_kind=kind,
     expected=gold,evidence_quality=quality,
     own_region=own["region"],cross_region=cross["region"],
     labels=labels,scores=scores,
     reflect_changed=labels["REFLECT"]!=labels["OWN"],
     control_changed=labels["CONTROL"]!=labels["OWN"],
     reflect_repaired=not scores["OWN"] and scores["REFLECT"],
     reflect_damaged=scores["OWN"] and not scores["REFLECT"],
     control_repaired=not scores["OWN"] and scores["CONTROL"],
     control_damaged=scores["OWN"] and not scores["CONTROL"],
     reflect_vs_control=int(scores["REFLECT"])-int(scores["CONTROL"]),
     reflect_vs_direct=int(scores["REFLECT"])-int(scores["DIRECT"]),
     reflect_vs_cross=int(scores["REFLECT"])-int(scores["CROSS_REFLECT"]),
     own_gap=cond=="missing",
     false_support=gold=="INSUFFICIENT" and labels["REFLECT"]=="SUPPORTED"
    ))

gate673("PAIRED_DECISIONS",len(PAIRS673)==144)

def metrics673(rows):
 n=len(rows)
 d=sum(p["scores"]["DIRECT"] for p in rows)
 o=sum(p["scores"]["OWN"] for p in rows)
 r=sum(p["scores"]["REFLECT"] for p in rows)
 c=sum(p["scores"]["CONTROL"] for p in rows)
 x=sum(p["scores"]["CROSS_REFLECT"] for p in rows)
 rr=sum(p["reflect_repaired"] for p in rows)
 rd=sum(p["reflect_damaged"] for p in rows)
 cr=sum(p["control_repaired"] for p in rows)
 cd=sum(p["control_damaged"] for p in rows)
 return dict(
  n=n,direct=d,own=o,reflect=r,control=c,cross_reflect=x,
  reflect_net=r-o,control_net=c-o,
  reflect_vs_control=r-c,
  reflect_repairs=rr,reflect_damages=rd,
  control_repairs=cr,control_damages=cd,
  reflect_preservation_pct=pct673(o-rd,o),
  control_preservation_pct=pct673(o-cd,o),
  reflect_changes=sum(p["reflect_changed"] for p in rows),
  control_changes=sum(p["control_changed"] for p in rows),
  false_support=sum(p["false_support"] for p in rows)
 )

TOTAL673=metrics673(PAIRS673)
GROUP673={rep+"_"+cond:metrics673([
 p for p in PAIRS673 if p["representation"]==rep and p["condition"]==cond
]) for rep in REPS673 for cond in CONDS673}
QUALITY673={q:metrics673([
 p for p in PAIRS673 if p["evidence_quality"]==q
]) for q in sorted(set(p["evidence_quality"] for p in PAIRS673))}
CLAIM673={kind:metrics673([
 p for p in PAIRS673 if p["claim_kind"]==kind
]) for kind in CLAIMS673}

REFLECT_DAMAGE673=[p for p in PAIRS673 if p["reflect_damaged"]]
REFLECT_REPAIR673=[p for p in PAIRS673 if p["reflect_repaired"]]
CONTROL_DAMAGE673=[p for p in PAIRS673 if p["control_damaged"]]
MISSING673=[p for p in PAIRS673 if p["condition"]=="missing"]
CONTRADICTION673=[p for p in PAIRS673 if p["expected"]=="CONTRADICTED"]

DIAGNOSTICS673=dict(
 reflect_damage_by_condition=dict(Counter(p["condition"] for p in REFLECT_DAMAGE673)),
 reflect_damage_by_quality=dict(Counter(p["evidence_quality"] for p in REFLECT_DAMAGE673)),
 reflect_repair_by_quality=dict(Counter(p["evidence_quality"] for p in REFLECT_REPAIR673)),
 missing_supported_by_stage={
  s:sum(p["labels"][s]=="SUPPORTED" for p in MISSING673)
  for s in STAGES673
 },
 missing_contradicted_by_stage={
  s:sum(p["labels"][s]=="CONTRADICTED" for p in MISSING673)
  for s in STAGES673
 },
 contradicted_claim_insufficient_by_stage={
  s:sum(p["labels"][s]=="INSUFFICIENT" for p in CONTRADICTION673)
  for s in STAGES673
 },
 reflect_changes_without_gain=sum(
  p["reflect_changed"] and not p["reflect_repaired"] for p in PAIRS673
 ),
 reflect_and_control_both_damage=sum(
  p["reflect_damaged"] and p["control_damaged"] for p in PAIRS673
 ),
 reflect_only_damage=sum(
  p["reflect_damaged"] and not p["control_damaged"] for p in PAIRS673
 ),
 control_only_damage=sum(
  p["control_damaged"] and not p["reflect_damaged"] for p in PAIRS673
 )
)

weights_ok673=all(not p.requires_grad for p in model.parameters())
sentinel_ok673=sentinel()==S0
panel_ok673=hashlib.sha256(json.dumps(PANEL663,sort_keys=True).encode()).hexdigest()==PANEL673_SHA

gate673("FINAL_FROZEN",weights_ok673)
gate673("FINAL_SENTINEL",sentinel_ok673)
gate673("FINAL_PANEL_LOCK",panel_ok673)
gate673("ZERO_ERRORS",len(ERR673)==0)

REPORT673=dict(
 test=TEST673,
 title="Autonomous Evidence Sufficiency and Decision Preservation",
 source=SOURCE673_URL,
 source_sha256=actual_sha673,
 panel_sha256=calculated_panel673,
 model_frozen=weights_ok673,
 sentinel_unchanged=sentinel_ok673,
 panel_lock_valid=panel_ok673,
 rows=R673,
 evidence=[
  dict(case=k[0],representation=k[1],condition=k[2],facts=v)
  for k,v in EVIDENCE673.items()
 ],
 total=TOTAL673,
 by_group=GROUP673,
 by_quality=QUALITY673,
 by_claim=CLAIM673,
 diagnostics=DIAGNOSTICS673,
 reflect_repairs=REFLECT_REPAIR673,
 reflect_damages=REFLECT_DAMAGE673,
 control_damages=CONTROL_DAMAGE673,
 paired_decisions=PAIRS673,
 errors=ERR673,
 elapsed_seconds=round(time.perf_counter()-T673,2),
 limitations=[
  "The original TEST664/TEST671/TEST672 development panel is reused.",
  "This is not an independent generalization or sealed holdout experiment.",
  "The frozen Qwen model and native CUDA CUT3 engine are unchanged.",
  "The model generates evidence and final classifications, but evidence retrieval is externally sequenced.",
  "REFLECT is prompted reconsideration, not proof of spontaneous autonomous self-correction.",
  "CONTROL and REFLECT have different wording and are not exactly token-length matched.",
  "CONTROL and REFLECT use the same original OWN classification.",
  "The experiment measures one additional review pass, not recursive learning.",
  "CROSS_REFLECT uses evidence from a foreign memory condition.",
  "Foreign evidence may itself be incorrect or incomplete.",
  "CROSS_REFLECT is a diagnostic intervention, not a deployment policy.",
  "DIRECT and evidence stages use different prompts.",
  "CONTRADICTED labels may depend on unstated functional-uniqueness assumptions.",
  "Entity matching uses normalized exact strings.",
  "No gold answer or expected label is inserted into inference prompts.",
  "Correct outputs do not establish correct internal reasoning.",
  "No model weights are trained or modified."
 ]
)

OUT673=ROOT673/"TEST673_results.json"
OUT673.write_text(json.dumps(REPORT673,ensure_ascii=False,indent=2),encoding="utf-8")

def line673(name,d):
 print(
  f"{name:<23} n={d['n']:3d} "
  f"OWN={d['own']:3d} REFLECT={d['reflect']:3d} "
  f"CONTROL={d['control']:3d} CROSS={d['cross_reflect']:3d} "
  f"REPAIR={d['reflect_repairs']:2d} DAMAGE={d['reflect_damages']:2d} "
  f"PRESERVE={d['reflect_preservation_pct']}%"
 )

print("="*112)
print("TEST673_FINAL — REFLECTION & DECISION PRESERVATION")
print("="*112)
line673("TOTAL",TOTAL673)
for k,v in GROUP673.items():line673(k,v)

print("-"*112)
print("EVIDENCE QUALITY")
for k,v in QUALITY673.items():line673(k,v)

print("-"*112)
print("CRITICAL DIAGNOSTICS")
print("REFLECT_DAMAGE_BY_CONDITION",json.dumps(
 DIAGNOSTICS673["reflect_damage_by_condition"],separators=(",",":")))
print("REFLECT_DAMAGE_BY_QUALITY",json.dumps(
 DIAGNOSTICS673["reflect_damage_by_quality"],separators=(",",":")))
print("REFLECT_REPAIR_BY_QUALITY",json.dumps(
 DIAGNOSTICS673["reflect_repair_by_quality"],separators=(",",":")))
print("MISSING_SUPPORTED",json.dumps(
 DIAGNOSTICS673["missing_supported_by_stage"],separators=(",",":")))
print("MISSING_CONTRADICTED",json.dumps(
 DIAGNOSTICS673["missing_contradicted_by_stage"],separators=(",",":")))
print("CONTRADICTED_TO_INSUFFICIENT",json.dumps(
 DIAGNOSTICS673["contradicted_claim_insufficient_by_stage"],separators=(",",":")))
print("REFLECT_ONLY_DAMAGE",DIAGNOSTICS673["reflect_only_damage"])
print("CONTROL_ONLY_DAMAGE",DIAGNOSTICS673["control_only_damage"])
print("BOTH_DAMAGE",DIAGNOSTICS673["reflect_and_control_both_damage"])

print("-"*112)
print("CORRECT OWN DECISIONS DAMAGED BY REFLECTION")
for p in REFLECT_DAMAGE673:
 print(
  "CASE",p["case"],
  "REP",p["representation"],
  "COND",p["condition"],
  "CLAIM",p["claim_kind"],
  "QUALITY",p["evidence_quality"],
  "OWN",p["labels"]["OWN"],
  "REFLECT",p["labels"]["REFLECT"],
  "CONTROL",p["labels"]["CONTROL"],
  "GOLD",p["expected"]
 )

print("-"*112)
print("CHAIN_ROWS",len(CHAIN673),"/288")
print("DECISION_ROWS",len(DEC673),"/720")
print("TOTAL_ROWS",len(R673),"/1008")
print("PAIRED_DECISIONS",len(PAIRS673),"/144")
print("REFLECT_REPAIRS",len(REFLECT_REPAIR673))
print("REFLECT_DAMAGES",len(REFLECT_DAMAGE673))
print("CONTROL_DAMAGES",len(CONTROL_DAMAGE673))
print("REFLECT_VS_CONTROL_NET",TOTAL673["reflect_vs_control"])
print("ERRORS",len(ERR673))
print("MODEL_FROZEN",weights_ok673)
print("SENTINEL_UNCHANGED",sentinel_ok673)
print("PANEL_LOCK_VALID",panel_ok673)
print("RESULT",OUT673)
print("RESULT_SHA256",hashlib.sha256(OUT673.read_bytes()).hexdigest())

complete673=(
 len(CHAIN673)==288 and
 len(DEC673)==720 and
 len(R673)==1008 and
 len(PAIRS673)==144 and
 len(EVIDENCE673)==72 and
 len(ERR673)==0 and
 weights_ok673 and sentinel_ok673 and panel_ok673
)

print("STATUS","EXPERIMENT_COMPLETE" if complete673 else "ERRORS_PRESENT")
gate673("TEST673_COMPLETED",complete673)

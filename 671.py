# TEST671 — AKBASCORE MAM / SELF-EVIDENCE PROVENANCE & CAUSAL CONFLICT AUDIT
# LOCKED TEST664 ENGINE / FROZEN QWEN / NATIVE CUDA CUT3
# ORIGINAL 12-CASE PANEL / BASE-SWAP-MISSING / TEXT_JOINT-DC_CUT3
# NO TRAINING / NO GOLD IN PROMPTS / NO ENGINE MODIFICATION
# MODEL-GENERATED EVIDENCE ONLY / NO EXTERNAL VERDICT ENGINE
import os,sys,re,json,time,random,hashlib,urllib.request,traceback,gc
from pathlib import Path
from collections import Counter,defaultdict
import torch

TEST671_ID="671"
ROOT671=Path("/content/AKBASCORE_TEST671")
ROOT671.mkdir(parents=True,exist_ok=True)
SOURCE671_URL="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/664.py"
SOURCE671_LOCK="a6dcd8142446ab82fe7996bd512d59fc34525e6a092a68f069dad07a2a286eb5"
PANEL671_LOCK="e0686148d6fd88050a097ee5441ce99a4bb068a463408742bc476c83aaba6c05"
REPS671=("TEXT_JOINT","DC_CUT3")
CONDS671=("base","swap","missing")
CLAIMS671=("ORIGINAL","COUNTERFACTUAL")
LABELS671=("SUPPORTED","CONTRADICTED","INSUFFICIENT")
FIELDS671=("person","org","city","region")
STAGES671=("DIRECT","OWN","CROSS","BLANK","MASKED")
R671=[];ERR671=[];T671=time.perf_counter()

def gate671(name,ok):
 print(("PASS " if ok else "FAIL ")+name,flush=True)
 if not ok:raise AssertionError(name)

def norm671(x):
 return re.sub(r"[^a-z0-9]","",str(x).casefold())

def same671(a,b):
 return norm671(a)==norm671(b)

def parse_label671(raw):
 s=str(raw).strip().upper()
 s=re.sub(r"^\s*(?:ANSWER|LABEL|STATUS)\s*:\s*","",s)
 s=s.strip().strip(" .,:;!?\"'`")
 return s if s in LABELS671 else "UNPARSEABLE"

def abstain671(raw):
 return bool(re.fullmatch(
  r"(?is)\s*(?:unknown|insufficient|not known|cannot determine|not enough information|i do not know|i don't know|none|n/?a|unavailable)[\s.!?]*",
  str(raw)
 ))

def entity_status671(raw,gold):
 if abstain671(raw):return "ABSTAIN"
 if not str(raw).strip():return "EMPTY"
 if gold is not None and same671(raw,gold):return "CORRECT"
 return "OTHER"

def checkpoint671():
 path=ROOT671/"TEST671_checkpoint.json"
 data=dict(test=TEST671_ID,rows=R671,errors=ERR671,elapsed=round(time.perf_counter()-T671,2))
 tmp=path.with_suffix(".tmp")
 tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
 tmp.replace(path)

print("="*130)
print("TEST671 — SELF-EVIDENCE PROVENANCE & CAUSAL CONFLICT AUDIT")
print("FROZEN QWEN / LOCKED TEST664 ENGINE / ORIGINAL PANEL")
print("="*130)

req671=urllib.request.Request(SOURCE671_URL,headers={"User-Agent":"AKBASCORE-TEST671"})
with urllib.request.urlopen(req671,timeout=120) as f:raw671=f.read()
actual_sha671=hashlib.sha256(raw671).hexdigest()
print("SOURCE_SHA256",actual_sha671)
gate671("TEST664_SOURCE_LOCK",actual_sha671==SOURCE671_LOCK)

source_text671=raw671.decode("utf-8")
BOUND671="# TEST664 — Reflective retry against budget-matched controls."
START671="T664=time.perf_counter();R664=[];ERR664=[]"
END671="# Important: text ceiling uses the exact same source-prefix"

gate671("SOURCE_BOUNDARIES",
 source_text671.count(BOUND671)==1 and
 source_text671.count(START671)==1 and
 source_text671.count(END671)==1
)

engine671=source_text671.split(BOUND671,1)[0]
panel671=START671+source_text671.split(START671,1)[1].split(END671,1)[0]

exec(compile(engine671,"TEST664_ENGINE.py","exec"),globals())
exec(compile(panel671,"TEST664_PANEL.py","exec"),globals())

# TEST566 defines PANEL_SHA internally. Never compare that variable with TEST664's panel lock.
calculated_panel671=hashlib.sha256(json.dumps(PANEL663,sort_keys=True).encode()).hexdigest()
print("TEST664_PANEL_SHA256",calculated_panel671)
print("TEST671_EXPECTED_PANEL_SHA256",PANEL671_LOCK)

gate671("TEST664_PANEL_LOCK",calculated_panel671==PANEL671_LOCK)
gate671("TEST664_PANEL_VARIABLE",SHA663==calculated_panel671)
gate671("PANEL_SIZE",len(PANEL663)==12)
gate671("PANEL_CASE_IDS",sorted(z["case"] for z in PANEL663)==list(range(12)))
gate671("FROZEN_WEIGHTS",all(not p.requires_grad for p in model.parameters()))
gate671("BASELINE_SENTINEL",sentinel()==S0)
gate671("MODEL_EVAL_MODE",not model.training)

def audit_panel671():
 for z in PANEL663:
  gate671("CASE_STRUCTURE_"+str(z["case"]),
   len(z["chains"])==4 and
   z["target"]!=z["other"] and
   len(z["base"])==16 and
   len(z["swap"])==16 and
   len(z["missing"])==15
  )
  target=z["chains"][z["target"]]
  other=z["chains"][z["other"]]
  gate671("CASE_GOLD_"+str(z["case"]),
   z["original"]==target["region"] and
   z["changed"]==other["region"] and
   target["region"]!=other["region"]
  )
  gate671("CASE_MEMORY_"+str(z["case"]),
   target["ids"][1] in z["base"] and
   target["ids"][1] not in z["swap"] and
   target["ids"][1] not in z["missing"] and
   z["rewired"] in z["swap"] and
   z["rewired"] not in z["base"] and
   z["rewired"] not in z["missing"]
  )

audit_panel671()

def memory671(keys,rep):
 if rep=="TEXT_JOINT":
  return validate659(joint659(keys),keys)
 if rep=="DC_CUT3":
  return validate659(native_memory659(keys),keys)
 raise ValueError("UNKNOWN_REPRESENTATION: "+str(rep))

def ask671(mem,q):
 t=time.perf_counter()
 raw=str(answer(mem,q))
 return dict(raw=raw,seconds=round(time.perf_counter()-t,4))

def claim671(z,kind):
 target=z["chains"][z["target"]]
 region=target["region"] if kind=="ORIGINAL" else z["chains"][z["other"]]["region"]
 return (
  "The owner of artifact "+target["obj"]+
  " works at an organization located in a city belonging to region "+region+"."
 )

def expected671(condition,kind):
 if condition=="missing":return "INSUFFICIENT"
 if condition=="base":return "SUPPORTED" if kind=="ORIGINAL" else "CONTRADICTED"
 if condition=="swap":return "CONTRADICTED" if kind=="ORIGINAL" else "SUPPORTED"
 raise ValueError(condition)

def expected_fact671(z,condition,field):
 target=z["chains"][z["target"]]
 if field=="person":return target["person"]
 if condition=="missing":return None
 active=target if condition=="base" else z["chains"][z["other"]]
 return active[field]

def next_question671(field,value):
 if field=="org":
  return "Which organization does "+value+" work at? Return only the organization name, or UNKNOWN."
 if field=="city":
  return "In which city is organization "+value+" located? Return only the city name, or UNKNOWN."
 if field=="region":
  return "Which region contains city "+value+"? Return only the region name, or UNKNOWN."
 raise ValueError(field)

def direct_question671(statement):
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

def evidence_text671(facts):
 return (
  "Evidence retrieved in earlier queries:\n"
  "Artifact owner: "+facts["person"]+"\n"
  "Owner's organization: "+facts["org"]+"\n"
  "Organization's city: "+facts["city"]+"\n"
  "City's region: "+facts["region"]+"\n"
 )

def verdict_question671(statement,facts):
 return (
  evidence_text671(facts)+
  "Claim: "+statement+"\n"
  "Evaluate whether the claim follows from the evidence and the information available in memory.\n"
  "SUPPORTED requires evidence establishing the claimed relationship.\n"
  "CONTRADICTED requires evidence establishing an incompatible relationship.\n"
  "INSUFFICIENT means the necessary relationship has not been established.\n"
  "Do not treat an UNKNOWN relationship as evidence that the claim is true.\n"
  "Return exactly one label: SUPPORTED, CONTRADICTED, or INSUFFICIENT.\n"
  "ANSWER:"
 )

def record671(**kw):
 R671.append(kw)
 return kw

@torch.no_grad()
def retrieve_chain671(z,condition,rep,mem):
 obj=z["chains"][z["target"]]["obj"]
 facts={}
 prior=None
 broken=False
 for field in FIELDS671:
  gold=expected_fact671(z,condition,field)
  if field=="person":
   q="Who owns artifact "+obj+"? Return only the person's name, or UNKNOWN."
  elif broken:
   q=None
  else:
   q=next_question671(field,prior)
  if q is None:
   raw="UNKNOWN"
   elapsed=0.0
   status="BLOCKED"
  else:
   out=ask671(mem,q)
   raw=out["raw"].strip()
   elapsed=out["seconds"]
   status=entity_status671(raw,gold)
  facts[field]=raw
  correct=(status=="CORRECT") if gold is not None else (status=="ABSTAIN")
  record671(
   case=z["case"],condition=condition,representation=rep,
   stage="CHAIN",field=field,question=q,raw=raw,
   expected=gold,status=status,correct=bool(correct),
   queried=q is not None,seconds=elapsed
  )
  print("CASE",z["case"],"COND",condition,"REP",rep,
   "CHAIN",field,"STATUS",status,"PASS",int(correct),flush=True)
  if abstain671(raw) or status=="BLOCKED" or not raw:
   broken=True
   prior=None
  else:
   prior=raw
 return facts

def masked_facts671():
 return {f:"NOT_QUERIED" for f in FIELDS671}

def blank_facts671(facts):
 d=dict(facts)
 d["org"]="UNKNOWN"
 d["city"]="UNKNOWN"
 d["region"]="UNKNOWN"
 return d

def cross_source671(condition):
 if condition=="base":return "swap"
 if condition=="swap":return "base"
 if condition=="missing":return "base"
 raise ValueError(condition)

def run_decision671(z,condition,rep,mem,stage,kind,facts=None,source=None):
 statement=claim671(z,kind)
 q=direct_question671(statement) if stage=="DIRECT" else verdict_question671(statement,facts)
 out=ask671(mem,q)
 pred=parse_label671(out["raw"])
 gold=expected671(condition,kind)
 row=record671(
  case=z["case"],condition=condition,representation=rep,
  stage=stage,claim_kind=kind,evidence_source=source,
  question=q,raw=out["raw"],predicted=pred,
  expected=gold,correct=pred==gold,seconds=out["seconds"]
 )
 print("CASE",z["case"],"COND",condition,"REP",rep,
  "STAGE",stage,"CLAIM",kind,"EXPECTED",gold,
  "PRED",pred,"PASS",int(row["correct"]),flush=True)
 return row

print("="*130)
print("TEST671 — PHASE 1 / MODEL-GENERATED CHAIN EVIDENCE")
print("="*130)

EVIDENCE671={}

for z in PANEL663:
 for rep in REPS671:
  for condition in CONDS671:
   mem=None
   try:
    mem=memory671(z[condition],rep)
    facts=retrieve_chain671(z,condition,rep,mem)
    key=(z["case"],rep,condition)
    gate671("UNIQUE_CHAIN_"+str(key),key not in EVIDENCE671)
    EVIDENCE671[key]=facts
   except Exception as e:
    err=dict(
     phase="CHAIN",case=z["case"],condition=condition,
     representation=rep,error=repr(e),
     traceback=traceback.format_exc()
    )
    ERR671.append(err)
    print("ERROR",json.dumps(err,ensure_ascii=False),flush=True)
   finally:
    if mem is not None:del mem
   checkpoint671()
 gc.collect()
 if torch.cuda.is_available():torch.cuda.empty_cache()

print("="*130)
print("TEST671 — PHASE 1 INTEGRITY")
print("="*130)

expected_evidence671=12*len(REPS671)*len(CONDS671)
actual_evidence671=len(EVIDENCE671)
print("EXPECTED_EVIDENCE_GROUPS",expected_evidence671)
print("ACTUAL_EVIDENCE_GROUPS",actual_evidence671)

gate671("PHASE1_EVIDENCE_COMPLETE",
 actual_evidence671==expected_evidence671 and len(ERR671)==0
)

print("="*130)
print("TEST671 — PHASE 2 / DIRECT-OWN-CROSS-BLANK-MASKED")
print("="*130)

for z in PANEL663:
 for rep in REPS671:
  for condition in CONDS671:
   mem=None
   try:
    mem=memory671(z[condition],rep)
    key=(z["case"],rep,condition)
    gate671("OWN_EVIDENCE_AVAILABLE_"+str(key),key in EVIDENCE671)
    own=EVIDENCE671[key]
    other=cross_source671(condition)
    cross_key=(z["case"],rep,other)
    gate671("CROSS_EVIDENCE_AVAILABLE_"+str(cross_key),cross_key in EVIDENCE671)
    cross=EVIDENCE671[cross_key]
    blank=blank_facts671(own)
    masked=masked_facts671()
    for kind in CLAIMS671:
     run_decision671(z,condition,rep,mem,"DIRECT",kind)
     run_decision671(z,condition,rep,mem,"OWN",kind,own,condition)
     run_decision671(z,condition,rep,mem,"CROSS",kind,cross,other)
     run_decision671(z,condition,rep,mem,"BLANK",kind,blank,condition)
     run_decision671(z,condition,rep,mem,"MASKED",kind,masked,"NONE")
   except Exception as e:
    err=dict(
     phase="VERDICT",case=z["case"],condition=condition,
     representation=rep,error=repr(e),
     traceback=traceback.format_exc()
    )
    ERR671.append(err)
    print("ERROR",json.dumps(err,ensure_ascii=False),flush=True)
   finally:
    if mem is not None:del mem
   checkpoint671()
 gc.collect()
 if torch.cuda.is_available():torch.cuda.empty_cache()

print("="*130)
print("TEST671 — ANALYSIS")
print("="*130)

SUMMARY671={}
TRANSITIONS671={}
RETRIEVAL671={}
CONFLICT671={}
CONFUSION671={}

for rep in REPS671:
 for condition in CONDS671:
  key=rep+"_"+condition
  rows=[r for r in R671 if r["representation"]==rep and r["condition"]==condition]
  SUMMARY671[key]={}
  for stage in STAGES671:
   rr=[r for r in rows if r["stage"]==stage]
   SUMMARY671[key][stage]=dict(
    n=len(rr),
    correct=sum(r["correct"] for r in rr),
    supported=sum(r["predicted"]=="SUPPORTED" for r in rr),
    contradicted=sum(r["predicted"]=="CONTRADICTED" for r in rr),
    insufficient=sum(r["predicted"]=="INSUFFICIENT" for r in rr),
    unparseable=sum(r["predicted"]=="UNPARSEABLE" for r in rr)
   )
   CONFUSION671[key+"_"+stage]=dict(
    Counter(r["expected"]+" -> "+r["predicted"] for r in rr)
   )
  cr=[r for r in rows if r["stage"]=="CHAIN"]
  RETRIEVAL671[key]=dict(
   n=len(cr),
   correct=sum(r["correct"] for r in cr),
   by_field={
    f:dict(
     n=sum(r["field"]==f for r in cr),
     correct=sum(r["correct"] for r in cr if r["field"]==f),
     abstain=sum(r["field"]==f and r["status"]=="ABSTAIN" for r in cr),
     blocked=sum(r["field"]==f and r["status"]=="BLOCKED" for r in cr),
     other=sum(r["field"]==f and r["status"]=="OTHER" for r in cr)
    ) for f in FIELDS671
   }
  )
  pairs=[]
  for z in PANEL663:
   for kind in CLAIMS671:
    rr={
     r["stage"]:r for r in rows
     if r["case"]==z["case"] and
     r.get("claim_kind")==kind and
     r["stage"] in STAGES671
    }
    if len(rr)!=len(STAGES671):continue
    d,o,c,b,m=[rr[x] for x in STAGES671]
    pairs.append(dict(
     case=z["case"],claim_kind=kind,expected=d["expected"],
     direct=d["predicted"],own=o["predicted"],
     cross=c["predicted"],blank=b["predicted"],
     masked=m["predicted"],
     direct_correct=d["correct"],
     own_correct=o["correct"],
     cross_correct=c["correct"],
     blank_correct=b["correct"],
     masked_correct=m["correct"],
     own_repairs=not d["correct"] and o["correct"],
     own_damages=d["correct"] and not o["correct"],
     own_vs_masked=int(o["correct"])-int(m["correct"]),
     own_vs_cross=int(o["correct"])-int(c["correct"]),
     cross_changes=o["predicted"]!=c["predicted"],
     blank_changes=o["predicted"]!=b["predicted"]
    ))
  TRANSITIONS671[key]=dict(
   n=len(pairs),
   own_repairs=sum(p["own_repairs"] for p in pairs),
   own_damages=sum(p["own_damages"] for p in pairs),
   own_vs_masked_net=sum(p["own_vs_masked"] for p in pairs),
   own_vs_cross_net=sum(p["own_vs_cross"] for p in pairs),
   cross_changes=sum(p["cross_changes"] for p in pairs),
   blank_changes=sum(p["blank_changes"] for p in pairs),
   blank_insufficient=sum(p["blank"]=="INSUFFICIENT" for p in pairs),
   pairs=pairs
  )

for rep in REPS671:
 for z in PANEL663:
  for condition in CONDS671:
   key=(z["case"],rep,condition)
   if key not in EVIDENCE671:continue
   own=EVIDENCE671[key]
   other=cross_source671(condition)
   cross=EVIDENCE671.get((z["case"],rep,other))
   if cross is None:continue
   for kind in CLAIMS671:
    rr={
     r["stage"]:r for r in R671
     if r["case"]==z["case"] and
     r["representation"]==rep and
     r["condition"]==condition and
     r.get("claim_kind")==kind and
     r["stage"] in STAGES671
    }
    if len(rr)!=len(STAGES671):continue
    target_region=(
     z["chains"][z["target"]]["region"]
     if kind=="ORIGINAL"
     else z["chains"][z["other"]]["region"]
    )
    own_region=own["region"]
    cross_region=cross["region"]
    own_known=not abstain671(own_region) and bool(own_region.strip())
    cross_known=not abstain671(cross_region) and bool(cross_region.strip())
    own_match=own_known and same671(own_region,target_region)
    cross_match=cross_known and same671(cross_region,target_region)
    own_other=own_known and not own_match
    cross_other=cross_known and not cross_match
    conflict_type=(
     "OWN_MATCH_CROSS_OTHER" if own_match and cross_other else
     "OWN_OTHER_CROSS_MATCH" if own_other and cross_match else
     "OWN_UNKNOWN" if not own_known else
     "CROSS_UNKNOWN" if not cross_known else
     "SAME_DIRECTION" if own_match==cross_match else
     "OTHER"
    )
    CONFLICT671[str(key)+"_"+kind]=dict(
     case=z["case"],
     representation=rep,
     condition=condition,
     claim_kind=kind,
     own_region=own_region,
     cross_region=cross_region,
     own_region_status=(
      "MATCH" if own_match else
      "OTHER" if own_other else
      "UNKNOWN"
     ),
     cross_region_status=(
      "MATCH" if cross_match else
      "OTHER" if cross_other else
      "UNKNOWN"
     ),
     conflict_type=conflict_type,
     own_verdict=rr["OWN"]["predicted"],
     cross_verdict=rr["CROSS"]["predicted"],
     blank_verdict=rr["BLANK"]["predicted"],
     masked_verdict=rr["MASKED"]["predicted"],
     verdict_changed=rr["OWN"]["predicted"]!=rr["CROSS"]["predicted"]
    )

weights_ok671=all(not p.requires_grad for p in model.parameters())
sentinel_ok671=sentinel()==S0
panel_ok671=(
 hashlib.sha256(json.dumps(PANEL663,sort_keys=True).encode()).hexdigest()
 ==PANEL671_LOCK
)

chain_rows671=[r for r in R671 if r["stage"]=="CHAIN"]
decision_rows671=[r for r in R671 if r["stage"] in STAGES671]

expected_chain671=12*2*3*4
expected_decision671=12*2*3*2*5
expected_total671=expected_chain671+expected_decision671

unique_chain671=len({
 (r["case"],r["condition"],r["representation"],r["field"])
 for r in chain_rows671
})==len(chain_rows671)

unique_decision671=len({
 (r["case"],r["condition"],r["representation"],r["claim_kind"],r["stage"])
 for r in decision_rows671
})==len(decision_rows671)

REPORT671=dict(
 test=TEST671_ID,
 title="Self-Evidence Provenance and Causal Conflict Audit",
 source=SOURCE671_URL,
 source_sha256=actual_sha671,
 panel_sha256=calculated_panel671,
 rows=R671,
 evidence=[
  dict(case=k[0],representation=k[1],condition=k[2],facts=v)
  for k,v in EVIDENCE671.items()
 ],
 summary=SUMMARY671,
 retrieval=RETRIEVAL671,
 transitions=TRANSITIONS671,
 conflict=CONFLICT671,
 confusion=CONFUSION671,
 errors=ERR671,
 weights_frozen=weights_ok671,
 sentinel_unchanged=sentinel_ok671,
 panel_lock_valid=panel_ok671,
 unique_chain_rows=unique_chain671,
 unique_decision_rows=unique_decision671,
 expected_rows=expected_total671,
 actual_rows=len(R671),
 elapsed_seconds=round(time.perf_counter()-T671,2),
 limitations=[
  "The original 12-case development panel is reused; this is not an independent generalization test.",
  "The frozen model, native CUT3 engine and memory construction are unchanged.",
  "All evidence supplied to the model is generated by the same frozen model.",
  "CHAIN is externally sequenced; it does not demonstrate autonomous evidence planning.",
  "The externally sequenced CHAIN procedure is diagnostic and is not proposed as the final cognitive architecture.",
  "CROSS deliberately supplies evidence retrieved from a different memory condition.",
  "CROSS is a diagnostic conflict intervention, not a valid deployment policy.",
  "Following CROSS evidence is not automatically correct: the evidence may conflict with the active memory.",
  "CROSS evidence may itself be incorrect or incomplete; all raw chain outputs are retained.",
  "BLANK removes three evidence values simultaneously; it does not isolate a single missing edge.",
  "MASKED uses fixed placeholder values and is not token-length matched to OWN.",
  "All verdict paths use the same existing answer() function, but DIRECT and evidence paths use different question text.",
  "The label CONTRADICTED may depend on closed-world or functional-uniqueness assumptions not explicitly encoded in memory.",
  "Entity scoring uses normalized exact matching; semantically equivalent outputs may be marked incorrect.",
  "A blocked query is not counted as successful abstention.",
  "Correct final answers do not establish correct internal reasoning.",
  "No gold entity or gold verdict is inserted into inference prompts.",
  "The experiment measures evidence use and decision stability, not autonomous self-correction."
 ]
)

path671=ROOT671/"TEST671_results.json"
path671.write_text(
 json.dumps(REPORT671,ensure_ascii=False,indent=2),
 encoding="utf-8"
)

print("="*130)
print("TEST671_FINAL")
print("="*130)

print("SUMMARY",json.dumps(SUMMARY671,ensure_ascii=False,indent=2))
print("RETRIEVAL",json.dumps(RETRIEVAL671,ensure_ascii=False,indent=2))
print("TRANSITIONS",json.dumps({
 k:{x:y for x,y in v.items() if x!="pairs"}
 for k,v in TRANSITIONS671.items()
},ensure_ascii=False,indent=2))

print("CONFLICT_TYPES",json.dumps(
 dict(Counter(v["conflict_type"] for v in CONFLICT671.values())),
 ensure_ascii=False,indent=2
))

print("CONFLICT_VERDICT_CHANGES",json.dumps({
 k:dict(
  n=sum(v["conflict_type"]==k for v in CONFLICT671.values()),
  changed=sum(
   v["conflict_type"]==k and v["verdict_changed"]
   for v in CONFLICT671.values()
  )
 )
 for k in sorted(set(v["conflict_type"] for v in CONFLICT671.values()))
},ensure_ascii=False,indent=2))

print("CONFUSION",json.dumps(CONFUSION671,ensure_ascii=False,indent=2))
print("EXPECTED_CHAIN_ROWS",expected_chain671)
print("ACTUAL_CHAIN_ROWS",len(chain_rows671))
print("EXPECTED_DECISION_ROWS",expected_decision671)
print("ACTUAL_DECISION_ROWS",len(decision_rows671))
print("EXPECTED_TOTAL_ROWS",expected_total671)
print("ACTUAL_TOTAL_ROWS",len(R671))
print("UNIQUE_CHAIN_ROWS",unique_chain671)
print("UNIQUE_DECISION_ROWS",unique_decision671)
print("PANEL_LOCK_VALID",panel_ok671)
print("WEIGHTS_FROZEN",weights_ok671)
print("SENTINEL_UNCHANGED",sentinel_ok671)
print("ERRORS",len(ERR671))
print("RESULT",path671)
print("SHA256",hashlib.sha256(path671.read_bytes()).hexdigest())

complete671=(
 len(chain_rows671)==expected_chain671 and
 len(decision_rows671)==expected_decision671 and
 len(R671)==expected_total671 and
 len(ERR671)==0 and
 weights_ok671 and
 sentinel_ok671 and
 panel_ok671 and
 unique_chain671 and
 unique_decision671 and
 len(EVIDENCE671)==expected_evidence671 and
 all(v["n"]==24 for v in TRANSITIONS671.values())
)

print("STATUS","EXPERIMENT_COMPLETE" if complete671 else "ERRORS_PRESENT")
gate671("TEST671_COMPLETED",complete671)

# TEST670 — AKBASCORE MAM / SELF-EVIDENCE CAUSAL AUDIT
# LOCKED TEST664 ENGINE / FROZEN QWEN / NATIVE CUDA CUT3
# ORIGINAL 12-CASE PANEL / BASE-SWAP-MISSING
# NO TRAINING / NO GOLD IN PROMPTS / NO ENGINE MODIFICATION
# SELF-EVIDENCE = MODEL-GENERATED FACTS ONLY
import os,sys,re,json,time,random,hashlib,urllib.request,traceback,gc
from pathlib import Path
from collections import Counter,defaultdict
import torch

TEST670="670"
ROOT670=Path("/content/AKBASCORE_TEST670")
ROOT670.mkdir(parents=True,exist_ok=True)
URL670="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/664.py"
SHA670="a6dcd8142446ab82fe7996bd512d59fc34525e6a092a68f069dad07a2a286eb5"
PANEL_LOCK670="e0686148d6fd88050a097ee5441ce99a4bb068a463408742bc476c83aaba6c05"
REPS670=("TEXT_JOINT","DC_CUT3")
CONDS670=("base","swap","missing")
CLAIMS670=("ORIGINAL","COUNTERFACTUAL")
MODES670=("STANDARD","PARAPHRASE")
LABELS670=("SUPPORTED","CONTRADICTED","INSUFFICIENT")
R670=[];ERR670=[];T670=time.perf_counter()

def gate670(name,condition):
 print(("PASS " if condition else "FAIL ")+name,flush=True)
 if not condition:raise AssertionError(name)

def norm670(value):
 return re.sub(r"[^a-z0-9]","",str(value).casefold())

def same670(a,b):
 return norm670(a)==norm670(b)

def parse_label670(raw):
 s=str(raw).strip().upper()
 s=re.sub(r"^\s*(?:ANSWER|LABEL|STATUS)\s*:\s*","",s)
 s=s.strip().strip(" .,:;!?\"'`")
 return s if s in LABELS670 else "UNPARSEABLE"

def parse_entity670(raw,gold=None):
 s=str(raw).strip()
 if re.fullmatch(r"(?is)(?:unknown|insufficient|not known|cannot determine|not enough information|i do not know|i don't know|none|n/?a|unavailable)[\s.!?]*",s):
  return "ABSTAIN"
 if not s:return "EMPTY"
 if gold is not None and same670(s,gold):return "CORRECT"
 return "OTHER"

print("="*128)
print("TEST670 — SELF-EVIDENCE CAUSAL AUDIT")
print("LOCKED TEST664 ENGINE / FROZEN QWEN / ORIGINAL 12 CASES")
print("="*128)

req670=urllib.request.Request(URL670,headers={"User-Agent":"AKBASCORE-TEST670"})
with urllib.request.urlopen(req670,timeout=120) as f:raw670=f.read()
actual670=hashlib.sha256(raw670).hexdigest()
print("SOURCE_SHA256",actual670)
gate670("TEST664_SOURCE_LOCK",actual670==SHA670)

src670=raw670.decode("utf-8")
bound670="# TEST664 — Reflective retry against budget-matched controls."
start670="T664=time.perf_counter();R664=[];ERR664=[]"
end670="# Important: text ceiling uses the exact same source-prefix"
gate670("SOURCE_BOUNDARIES",src670.count(bound670)==1 and src670.count(start670)==1 and src670.count(end670)==1)

engine670=src670.split(bound670,1)[0]
panel670=start670+src670.split(start670,1)[1].split(end670,1)[0]
exec(compile(engine670,"TEST664_ENGINE.py","exec"),globals())
exec(compile(panel670,"TEST664_PANEL.py","exec"),globals())

gate670("PANEL_LOCK",SHA663==PANEL_LOCK670)
gate670("PANEL_SIZE",len(PANEL663)==12)
gate670("FROZEN_WEIGHTS",all(not p.requires_grad for p in model.parameters()))
gate670("BASELINE_SENTINEL",sentinel()==S0)
gate670("MODEL_EVAL_MODE",not model.training)

def make_memory670(keys,rep):
 if rep=="TEXT_JOINT":return validate659(joint659(keys),keys)
 if rep=="DC_CUT3":return validate659(native_memory659(keys),keys)
 raise ValueError(rep)

def ask670(mem,q):
 t=time.perf_counter()
 raw=str(answer(mem,q))
 return dict(raw=raw,seconds=round(time.perf_counter()-t,4))

def claim670(z,kind):
 target=z["chains"][z["target"]]
 region=target["region"] if kind=="ORIGINAL" else z["chains"][z["other"]]["region"]
 return "The owner of artifact "+target["obj"]+" works at an organization located in a city belonging to region "+region+"."

def expected670(condition,kind):
 if condition=="missing":return "INSUFFICIENT"
 if condition=="base":return "SUPPORTED" if kind=="ORIGINAL" else "CONTRADICTED"
 if condition=="swap":return "CONTRADICTED" if kind=="ORIGINAL" else "SUPPORTED"
 raise ValueError(condition)

def direct_question670(claim,mode):
 if mode=="STANDARD":
  return (
   "Evaluate the following claim using only the available facts.\n"
   "Claim: "+claim+"\n"
   "Respond with exactly one word:\n"
   "SUPPORTED — the available facts establish the claim.\n"
   "CONTRADICTED — the available facts establish an incompatible alternative.\n"
   "INSUFFICIENT — the available facts do not establish either conclusion.\n"
   "Do not assume the claim is true merely because it was proposed.\n"
   "ANSWER:"
  )
 return (
  "Check this statement against the information you can access:\n"
  +claim+"\n"
  "Classify its evidential status.\n"
  "Use SUPPORTED only when the statement follows from the facts.\n"
  "Use CONTRADICTED only when the facts establish a conflicting result.\n"
  "Use INSUFFICIENT when the needed information is missing.\n"
  "Return only SUPPORTED, CONTRADICTED, or INSUFFICIENT.\n"
  "ANSWER:"
 )

def evidence_questions670(z):
 obj=z["chains"][z["target"]]["obj"]
 return (
  ("person","Who owns artifact "+obj+"? Return only the person's name, or UNKNOWN."),
  ("org","Which organization employs the owner of artifact "+obj+"? Return only the organization name, or UNKNOWN."),
  ("city","In which city is that organization located for artifact "+obj+"? Return only the city name, or UNKNOWN."),
  ("region","Which region contains the city of that organization for artifact "+obj+"? Return only the region name, or UNKNOWN.")
 )

def evidence_text670(facts):
 return (
  "Retrieved information from earlier independent queries:\n"
  "Artifact owner: "+facts["person"]+"\n"
  "Owner's organization: "+facts["org"]+"\n"
  "Organization's city: "+facts["city"]+"\n"
  "City's region: "+facts["region"]+"\n"
 )

def self_question670(claim,mode,facts,include_facts):
 prefix=evidence_text670(facts) if include_facts else (
  "The following claim must be evaluated using only facts that are actually available.\n"
  "Do not invent missing relationships.\n"
 )
 if mode=="STANDARD":
  return (
   prefix+
   "Claim: "+claim+"\n"
   "Classify the claim as SUPPORTED, CONTRADICTED, or INSUFFICIENT.\n"
   "SUPPORTED requires evidence for the stated region.\n"
   "CONTRADICTED requires evidence for an incompatible region.\n"
   "INSUFFICIENT applies when the required chain cannot be established.\n"
   "Return exactly one label.\nANSWER:"
  )
 return (
  prefix+
  "Statement to assess: "+claim+"\n"
  "Determine whether the available information proves the statement, disproves it through a conflicting result, or leaves it unresolved.\n"
  "Reply with exactly one of SUPPORTED, CONTRADICTED, INSUFFICIENT.\nANSWER:"
 )

def expected_fact670(z,condition,field):
 target=z["chains"][z["target"]]
 if field=="person":return target["person"]
 if condition=="missing":return None
 active=target if condition=="base" else z["chains"][z["other"]]
 return active[field]

@torch.no_grad()
def run670(z,condition,rep):
 mem=None
 try:
  mem=make_memory670(z[condition],rep)
  facts={}
  for field,q in evidence_questions670(z):
   out=ask670(mem,q)
   facts[field]=out["raw"].strip()
   gold=expected_fact670(z,condition,field)
   status=parse_entity670(out["raw"],gold)
   row=dict(case=z["case"],condition=condition,representation=rep,stage="RETRIEVE",probe=field,mode="STANDARD",question=q,expected=gold,raw=out["raw"],predicted=out["raw"].strip(),status=status,correct=status=="CORRECT" if gold is not None else status=="ABSTAIN",seconds=out["seconds"])
   R670.append(row)
   print("CASE",z["case"],"COND",condition,"REP",rep,"STAGE RETRIEVE","FIELD",field,"STATUS",status,flush=True)

  for kind in CLAIMS670:
   claim=claim670(z,kind)
   gold=expected670(condition,kind)
   for mode in MODES670:
    questions=(
     ("DIRECT",direct_question670(claim,mode)),
     ("SELF_EVIDENCE",self_question670(claim,mode,facts,True)),
     ("BLIND_CONTROL",self_question670(claim,mode,facts,False))
    )
    for stage,q in questions:
     out=ask670(mem,q)
     pred=parse_label670(out["raw"])
     row=dict(case=z["case"],condition=condition,representation=rep,stage=stage,probe=kind,mode=mode,question=q,expected=gold,raw=out["raw"],predicted=pred,correct=pred==gold,seconds=out["seconds"])
     R670.append(row)
     print("CASE",z["case"],"COND",condition,"REP",rep,"STAGE",stage,"CLAIM",kind,"MODE",mode,"EXPECTED",gold,"PRED",pred,"PASS",int(row["correct"]),flush=True)
 except Exception as e:
  err=dict(case=z["case"],condition=condition,representation=rep,error=repr(e),traceback=traceback.format_exc())
  ERR670.append(err)
  print("ERROR",json.dumps(err,ensure_ascii=False),flush=True)
 finally:
  if mem is not None:del mem

for z in PANEL663:
 for rep in REPS670:
  for condition in CONDS670:
   run670(z,condition,rep)
 gc.collect()
 if torch.cuda.is_available():torch.cuda.empty_cache()

print("="*128)
print("TEST670 — CAUSAL ANALYSIS")
print("="*128)

SUMMARY670={}
TRANSITIONS670={}
SENSITIVITY670={}
RETRIEVAL670={}
CONFUSION670={}
STAGES670=("DIRECT","SELF_EVIDENCE","BLIND_CONTROL")

for rep in REPS670:
 for condition in CONDS670:
  key=rep+"_"+condition
  rows=[r for r in R670 if r["representation"]==rep and r["condition"]==condition]
  SUMMARY670[key]={}
  for stage in STAGES670:
   rr=[r for r in rows if r["stage"]==stage]
   SUMMARY670[key][stage]=dict(n=len(rr),correct=sum(r["correct"] for r in rr),supported=sum(r["predicted"]=="SUPPORTED" for r in rr),contradicted=sum(r["predicted"]=="CONTRADICTED" for r in rr),insufficient=sum(r["predicted"]=="INSUFFICIENT" for r in rr),unparseable=sum(r["predicted"]=="UNPARSEABLE" for r in rr))
   CONFUSION670[key+"_"+stage]=dict(Counter(r["expected"]+" -> "+r["predicted"] for r in rr))
  retrieved=[r for r in rows if r["stage"]=="RETRIEVE"]
  RETRIEVAL670[key]=dict(n=len(retrieved),correct=sum(r["correct"] for r in retrieved),by_field={f:dict(n=sum(r["probe"]==f for r in retrieved),correct=sum(r["correct"] for r in retrieved if r["probe"]==f)) for f in ("person","org","city","region")})
  pairs=[]
  for z in PANEL663:
   for kind in CLAIMS670:
    for mode in MODES670:
     rr={r["stage"]:r for r in rows if r["case"]==z["case"] and r["probe"]==kind and r["mode"]==mode and r["stage"] in STAGES670}
     if len(rr)==3:
      d,s,b=rr["DIRECT"],rr["SELF_EVIDENCE"],rr["BLIND_CONTROL"]
      pairs.append(dict(case=z["case"],claim_kind=kind,mode=mode,direct=d["predicted"],self_evidence=s["predicted"],blind=b["predicted"],direct_correct=d["correct"],self_correct=s["correct"],blind_correct=b["correct"],self_repairs_wrong=bool(not d["correct"] and s["correct"]),self_damages_correct=bool(d["correct"] and not s["correct"]),self_vs_blind_gain=int(s["correct"])-int(b["correct"]),self_changes_decision=s["predicted"]!=d["predicted"]))
  TRANSITIONS670[key]=dict(n=len(pairs),self_repairs_wrong=sum(p["self_repairs_wrong"] for p in pairs),self_damages_correct=sum(p["self_damages_correct"] for p in pairs),self_vs_blind_net_gain=sum(p["self_vs_blind_gain"] for p in pairs),self_changes_decision=sum(p["self_changes_decision"] for p in pairs),pairs=pairs)

for rep in REPS670:
 for stage in STAGES670:
  for mode in MODES670:
   pairs=[]
   for z in PANEL663:
    for kind in CLAIMS670:
     rr={r["condition"]:r for r in R670 if r["case"]==z["case"] and r["representation"]==rep and r["stage"]==stage and r["probe"]==kind and r["mode"]==mode}
     if len(rr)==3:
      b,s,m=rr["base"],rr["swap"],rr["missing"]
      pairs.append(dict(case=z["case"],claim_kind=kind,base=b["predicted"],swap=s["predicted"],missing=m["predicted"],complete=bool(b["correct"] and s["correct"] and m["correct"]),directional_reversal=bool(b["correct"] and s["correct"] and b["predicted"]!=s["predicted"]),missing_abstained=m["predicted"]=="INSUFFICIENT"))
   key=rep+"_"+stage+"_"+mode
   SENSITIVITY670[key]=dict(n=len(pairs),complete_triplets=sum(p["complete"] for p in pairs),directional_reversals=sum(p["directional_reversal"] for p in pairs),missing_abstentions=sum(p["missing_abstained"] for p in pairs),triplets=pairs)

weights670=all(not p.requires_grad for p in model.parameters())
sentinel670=sentinel()==S0
expected_rows670=12*2*3*(4+2*2*3)

report670=dict(
 test="670",
 title="Self-Evidence Causal Audit",
 source=URL670,
 source_sha256=actual670,
 panel_sha256=SHA663,
 rows=R670,
 summary=SUMMARY670,
 retrieval=RETRIEVAL670,
 transitions=TRANSITIONS670,
 sensitivity=SENSITIVITY670,
 confusion=CONFUSION670,
 errors=ERR670,
 weights_frozen=weights670,
 sentinel_unchanged=sentinel670,
 elapsed_seconds=round(time.perf_counter()-T670,2),
 limitations=[
  "Original TEST664 development panel reused; no independent holdout.",
  "Frozen model, native CUT3 engine, and memory construction unchanged.",
  "SELF_EVIDENCE uses only model-generated retrieval responses, not gold answers.",
  "The four retrieved answers are generated independently and may not form a mutually consistent chain.",
  "SELF_EVIDENCE changes the decision prompt and therefore is not a pure isolated evidence-content intervention.",
  "BLIND_CONTROL matches the decision instruction but does not match SELF_EVIDENCE prompt length or lexical content.",
  "Correct answers are used exclusively for offline scoring.",
  "Retrieval accuracy is assessed by normalized exact matching; semantic paraphrases may be penalized.",
  "MISSING expected retrieval statuses depend on the actual removed edge; must be audited before causal claims.",
  "Observed improvements do not establish autonomous reasoning or internal epistemic awareness.",
  "Cartridge selection is externally controlled."
 ]
)

path670=ROOT670/"TEST670_results.json"
path670.write_text(json.dumps(report670,ensure_ascii=False,indent=2),encoding="utf-8")

print("="*128)
print("TEST670_FINAL")
print("="*128)
print("SUMMARY",json.dumps(SUMMARY670,ensure_ascii=False,indent=2))
print("RETRIEVAL",json.dumps(RETRIEVAL670,ensure_ascii=False,indent=2))
print("TRANSITIONS",json.dumps({k:{x:y for x,y in v.items() if x!="pairs"} for k,v in TRANSITIONS670.items()},ensure_ascii=False,indent=2))
print("SENSITIVITY",json.dumps({k:{x:y for x,y in v.items() if x!="triplets"} for k,v in SENSITIVITY670.items()},ensure_ascii=False,indent=2))
print("CONFUSION",json.dumps(CONFUSION670,ensure_ascii=False,indent=2))
print("EXPECTED_ROWS",expected_rows670)
print("ACTUAL_ROWS",len(R670))
print("WEIGHTS_FROZEN",weights670)
print("SENTINEL_UNCHANGED",sentinel670)
print("ERRORS",len(ERR670))
print("RESULT",path670)
print("SHA256",hashlib.sha256(path670.read_bytes()).hexdigest())

success670=(
 len(R670)==expected_rows670
 and not ERR670
 and weights670
 and sentinel670
 and SHA663==PANEL_LOCK670
)
print("STATUS","EXPERIMENT_COMPLETE" if success670 else "ERRORS_PRESENT")
gate670("TEST670_COMPLETED",success670)

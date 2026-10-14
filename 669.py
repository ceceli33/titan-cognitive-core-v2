# TEST669 — AKBASCORE MAM / EVIDENCE ACCESS → RELATIONAL INTEGRATION → EPISTEMIC DECISION
# BASELINE: LOCKED TEST664 ENGINE + TEST668 WORKING EXPERIMENT SKELETON
# FROZEN QWEN / NATIVE CUDA CUT3 / ORIGINAL 12 CASES / BASE-SWAP-MISSING
# NO TRAINING / NO GOLD IN PROMPTS / NO ANSWER HARDCODING / NO ENGINE MODIFICATION
# All stages independently query the same untouched memory; no oracle-assisted continuation.
import os,sys,re,json,time,random,hashlib,urllib.request,traceback,gc
from pathlib import Path
from collections import Counter,defaultdict
import torch

TEST669="669"
ROOT669=Path("/content/AKBASCORE_TEST669")
ROOT669.mkdir(parents=True,exist_ok=True)
URL669="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/664.py"
SHA669="a6dcd8142446ab82fe7996bd512d59fc34525e6a092a68f069dad07a2a286eb5"
PANEL_LOCK669="e0686148d6fd88050a097ee5441ce99a4bb068a463408742bc476c83aaba6c05"
REPS669=("TEXT_JOINT","DC_CUT3")
CONDS669=("base","swap","missing")
CLAIMS669=("ORIGINAL","COUNTERFACTUAL")
MODES669=("STANDARD","PARAPHRASE")
LABELS669=("SUPPORTED","CONTRADICTED","INSUFFICIENT")
R669=[];ERR669=[];NOMEM669=[];NOMEM_ERR669=[]
T669=time.perf_counter()

def gate669(name,condition):
 print(("PASS " if condition else "FAIL ")+name,flush=True)
 if not condition:raise AssertionError(name)

def norm669(value):
 return re.sub(r"[^a-z0-9]","",str(value).casefold())

def same669(a,b):
 return norm669(a)==norm669(b)

def parse_label669(raw):
 s=str(raw).strip().upper()
 s=re.sub(r"^\s*(?:ANSWER|LABEL|STATUS)\s*:\s*","",s)
 s=s.strip().strip(" .,:;!?\"'`")
 return s if s in LABELS669 else "UNPARSEABLE"

def parse_entity669(raw,gold):
 s=str(raw).strip()
 if same669(s,gold):return "CORRECT"
 if not s:return "EMPTY"
 if re.fullmatch(r"(?is)(?:unknown|insufficient|not known|cannot determine|not enough information|i do not know|i don't know|none|n/?a|unavailable)[\s.!?]*",s):return "ABSTAIN"
 return "WRONG"

print("="*128)
print("TEST669 — EVIDENCE ACCESS / RELATIONAL INTEGRATION / EPISTEMIC DECISION")
print("LOCKED TEST664 ENGINE | FROZEN QWEN | ORIGINAL 12-CASE PANEL")
print("GOLD VALUES USED ONLY FOR OFFLINE SCORING")
print("="*128)

req669=urllib.request.Request(URL669,headers={"User-Agent":"AKBASCORE-TEST669"})
with urllib.request.urlopen(req669,timeout=120) as f:raw669=f.read()
actual669=hashlib.sha256(raw669).hexdigest()
print("SOURCE_SHA256",actual669)
gate669("TEST664_SOURCE_LOCK",actual669==SHA669)
src669=raw669.decode("utf-8")
bound669="# TEST664 — Reflective retry against budget-matched controls."
start669="T664=time.perf_counter();R664=[];ERR664=[]"
end669="# Important: text ceiling uses the exact same source-prefix"
gate669("SOURCE_BOUNDARIES",src669.count(bound669)==1 and src669.count(start669)==1 and src669.count(end669)==1)
engine669=src669.split(bound669,1)[0]
panel669=start669+src669.split(start669,1)[1].split(end669,1)[0]
exec(compile(engine669,"TEST664_ENGINE.py","exec"),globals())
exec(compile(panel669,"TEST664_PANEL.py","exec"),globals())

gate669("PANEL_LOCK",SHA663==PANEL_LOCK669)
gate669("PANEL_SIZE",len(PANEL663)==12)
gate669("FROZEN_WEIGHTS",all(not p.requires_grad for p in model.parameters()))
gate669("BASELINE_SENTINEL",sentinel()==S0)
gate669("MODEL_EVAL_MODE",not model.training)

def make_memory669(keys,rep):
 if rep=="TEXT_JOINT":return validate659(joint659(keys),keys)
 if rep=="DC_CUT3":return validate659(native_memory659(keys),keys)
 raise ValueError(rep)

def ask669(mem,q):
 t=time.perf_counter()
 raw=str(answer(mem,q))
 return dict(raw=raw,seconds=round(time.perf_counter()-t,4))

def chain669(z,condition):
 original=z["chains"][z["target"]]
 alternative=z["chains"][z["other"]]
 active=original if condition=="base" else alternative
 return original,alternative,active

def expected_entity669(z,condition,field):
 original,alternative,active=chain669(z,condition)
 if field=="person":return original["person"]
 if condition=="missing" and field in ("org","city","region"):return None
 return active[field]

def evidence_questions669(z):
 original=z["chains"][z["target"]]
 obj=original["obj"]
 return [
  ("person","Who owns the artifact "+obj+"? Reply with the name only. If unknown, reply UNKNOWN."),
  ("org","Which organization employs the owner of artifact "+obj+"? Reply with the organization name only. If unknown, reply UNKNOWN."),
  ("city","In which city is the organization employing the owner of artifact "+obj+" located? Reply with the city name only. If unknown, reply UNKNOWN."),
  ("region","Which region contains the city of the organization employing the owner of artifact "+obj+"? Reply with the region name only. If unknown, reply UNKNOWN.")
 ]

def integration_question669(z,mode):
 obj=z["chains"][z["target"]]["obj"]
 if mode=="STANDARD":
  return "Follow the available facts from artifact "+obj+" through its owner, employer and city. Which region is reached? Reply with the region name only. If the chain cannot be completed, reply UNKNOWN."
 if mode=="PARAPHRASE":
  return "Using the information you can access, determine the region associated with the city where the employer of artifact "+obj+"'s owner is located. Return only the region name, or UNKNOWN when the needed relationship is unavailable."
 raise ValueError(mode)

def claim669(z,kind):
 original,alternative,_=chain669(z,"base")
 region=original["region"] if kind=="ORIGINAL" else alternative["region"]
 return "The owner of artifact "+original["obj"]+" works at an organization located in a city belonging to region "+region+"."

def decision_question669(claim,mode):
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
 if mode=="PARAPHRASE":
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
 raise ValueError(mode)

def expected_label669(condition,kind):
 if condition=="missing":return "INSUFFICIENT"
 if condition=="base":return "SUPPORTED" if kind=="ORIGINAL" else "CONTRADICTED"
 if condition=="swap":return "CONTRADICTED" if kind=="ORIGINAL" else "SUPPORTED"
 raise ValueError(condition)

def add_row669(z,condition,rep,stage,probe,mode,q,gold,out):
 if stage=="DECISION":
  predicted=parse_label669(out["raw"])
  status="CORRECT" if predicted==gold else "WRONG"
 elif gold is None:
  predicted=str(out["raw"]).strip()
  status="ABSTAIN" if parse_entity669(predicted,"__IMPOSSIBLE_GOLD_669__")=="ABSTAIN" else "NONABSTAIN"
 else:
  predicted=str(out["raw"]).strip()
  status=parse_entity669(predicted,gold)
 row=dict(
  case=z["case"],condition=condition,representation=rep,
  stage=stage,probe=probe,mode=mode,question=q,
  expected=gold,raw=out["raw"],predicted=predicted,
  status=status,correct=status=="CORRECT" if gold is not None else status=="ABSTAIN",
  seconds=out["seconds"]
 )
 R669.append(row)
 print("CASE",z["case"],"COND",condition,"REP",rep,"STAGE",stage,"PROBE",probe,"MODE",mode,"EXPECTED",gold,"PRED",predicted,"STATUS",status,flush=True)
 return row

print("="*128)
print("TEST669 — PANEL VALIDATION")
print("="*128)
for z in PANEL663:
 original,alternative,_=chain669(z,"base")
 gate669("DISTINCT_REGIONS_"+str(z["case"]),not same669(original["region"],alternative["region"]))
 gate669("DISTINCT_OBJECT_"+str(z["case"]),bool(str(original["obj"]).strip()))
 gate669("DISTINCT_TARGET_"+str(z["case"]),z["target"]!=z["other"])

print("="*128)
print("TEST669 — STAGE 1: INDEPENDENT EVIDENCE ACCESS")
print("TEST669 — STAGE 2: INDEPENDENT FOUR-HOP INTEGRATION")
print("TEST669 — STAGE 3: BLIND EPISTEMIC DECISION")
print("="*128)

@torch.no_grad()
def run669(z,condition,rep):
 mem=None
 try:
  mem=make_memory669(z[condition],rep)
  for field,q in evidence_questions669(z):
   out=ask669(mem,q)
   gold=expected_entity669(z,condition,field)
   add_row669(z,condition,rep,"EVIDENCE",field,"STANDARD",q,gold,out)
  for mode in MODES669:
   q=integration_question669(z,mode)
   gold=expected_entity669(z,condition,"region")
   out=ask669(mem,q)
   add_row669(z,condition,rep,"INTEGRATION","region",mode,q,gold,out)
  for kind in CLAIMS669:
   claim=claim669(z,kind)
   gold=expected_label669(condition,kind)
   for mode in MODES669:
    q=decision_question669(claim,mode)
    out=ask669(mem,q)
    add_row669(z,condition,rep,"DECISION",kind,mode,q,gold,out)
 except Exception as e:
  err=dict(case=z["case"],condition=condition,representation=rep,error=repr(e),traceback=traceback.format_exc())
  ERR669.append(err)
  print("ERROR",json.dumps(err,ensure_ascii=False),flush=True)
 finally:
  if mem is not None:del mem

for z in PANEL663:
 for rep in REPS669:
  for condition in CONDS669:
   run669(z,condition,rep)

gc.collect()
torch.cuda.empty_cache()

print("="*128)
print("TEST669 — NO-MEMORY DECISION CONTROL")
print("="*128)

@torch.no_grad()
def no_memory669(q):
 messages=[{"role":"user","content":q}]
 encoded=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors="pt")
 if isinstance(encoded,dict):
  ids=encoded["input_ids"]
  mask=encoded.get("attention_mask",torch.ones_like(ids))
 else:
  ids=encoded
  mask=torch.ones_like(ids)
 ids=ids.to(DEVICE)
 mask=mask.to(DEVICE)
 t=time.perf_counter()
 out=model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=16,do_sample=False,pad_token_id=tok.eos_token_id)
 raw=tok.decode(out[0,ids.shape[-1]:],skip_special_tokens=True).strip()
 return dict(raw=raw,predicted=parse_label669(raw),seconds=round(time.perf_counter()-t,4))

for z in PANEL663:
 for kind in CLAIMS669:
  for mode in MODES669:
   q=decision_question669(claim669(z,kind),mode)
   try:
    out=no_memory669(q)
    row=dict(case=z["case"],claim_kind=kind,mode=mode,question=q,expected="INSUFFICIENT",raw=out["raw"],predicted=out["predicted"],correct=out["predicted"]=="INSUFFICIENT",seconds=out["seconds"])
    NOMEM669.append(row)
    print("NOMEM",z["case"],kind,mode,out["predicted"],"PASS",int(row["correct"]),flush=True)
   except Exception as e:
    err=dict(case=z["case"],claim_kind=kind,mode=mode,error=repr(e),traceback=traceback.format_exc())
    NOMEM_ERR669.append(err)
    print("NOMEM_ERROR",json.dumps(err,ensure_ascii=False),flush=True)

print("="*128)
print("TEST669 — OFFLINE DIAGNOSTIC ANALYSIS")
print("="*128)

SUMMARY669={}
DIAGNOSTIC669={}
CONFUSION669={}
SENSITIVITY669={}
PAIR669={}
STAGE_ROWS669={}

for rep in REPS669:
 for condition in CONDS669:
  key=rep+"_"+condition
  rows=[r for r in R669 if r["representation"]==rep and r["condition"]==condition]
  stage_summary={}
  for stage in ("EVIDENCE","INTEGRATION","DECISION"):
   sr=[r for r in rows if r["stage"]==stage]
   stage_summary[stage]=dict(
    n=len(sr),
    correct=sum(r["correct"] for r in sr),
    wrong=sum(r["status"]=="WRONG" for r in sr),
    abstain=sum(r["status"]=="ABSTAIN" for r in sr),
    nonabstain=sum(r["status"]=="NONABSTAIN" for r in sr)
   )
  SUMMARY669[key]=stage_summary
  CONFUSION669[key]=dict(Counter(r["expected"]+" -> "+r["predicted"] for r in rows if r["stage"]=="DECISION"))
  STAGE_ROWS669[key]=len(rows)

  diag=[]
  for z in PANEL663:
   rr=[r for r in rows if r["case"]==z["case"]]
   evidence={r["probe"]:r for r in rr if r["stage"]=="EVIDENCE"}
   integration={r["mode"]:r for r in rr if r["stage"]=="INTEGRATION"}
   decisions={(r["probe"],r["mode"]):r for r in rr if r["stage"]=="DECISION"}
   if len(evidence)!=4 or len(integration)!=2 or len(decisions)!=4:continue
   access_complete=all(evidence[f]["correct"] for f in ("person","org","city","region"))
   access_prefix=sum(evidence[f]["correct"] for f in ("person","org","city","region"))
   for mode in MODES669:
    integ=integration[mode]
    for kind in CLAIMS669:
     dec=decisions[(kind,mode)]
     diag.append(dict(
      case=z["case"],mode=mode,claim_kind=kind,
      access_complete=access_complete,
      access_correct_steps=access_prefix,
      integration_correct=integ["correct"],
      integration_status=integ["status"],
      decision_correct=dec["correct"],
      decision_predicted=dec["predicted"],
      decision_expected=dec["expected"],
      correct_integration_wrong_decision=bool(integ["correct"] and not dec["correct"]),
      wrong_integration_correct_decision=bool(not integ["correct"] and dec["correct"]),
      complete_evidence_wrong_decision=bool(access_complete and not dec["correct"]),
      correct_integration_correct_decision=bool(integ["correct"] and dec["correct"])
     ))
  DIAGNOSTIC669[key]=dict(
   n=len(diag),
   access_complete=sum(d["access_complete"] for d in diag),
   correct_integration=sum(d["integration_correct"] for d in diag),
   correct_decision=sum(d["decision_correct"] for d in diag),
   correct_integration_wrong_decision=sum(d["correct_integration_wrong_decision"] for d in diag),
   wrong_integration_correct_decision=sum(d["wrong_integration_correct_decision"] for d in diag),
   complete_evidence_wrong_decision=sum(d["complete_evidence_wrong_decision"] for d in diag),
   correct_integration_correct_decision=sum(d["correct_integration_correct_decision"] for d in diag),
   rows=diag
  )

for rep in REPS669:
 for mode in MODES669:
  paired=[]
  for z in PANEL663:
   for kind in CLAIMS669:
    rr={r["condition"]:r for r in R669 if r["case"]==z["case"] and r["representation"]==rep and r["stage"]=="DECISION" and r["probe"]==kind and r["mode"]==mode}
    if len(rr)==3:
     b,s,m=rr["base"],rr["swap"],rr["missing"]
     paired.append(dict(
      case=z["case"],claim_kind=kind,
      base=b["predicted"],swap=s["predicted"],missing=m["predicted"],
      complete=bool(b["correct"] and s["correct"] and m["correct"]),
      directional_reversal=bool(
       b["predicted"]!=s["predicted"]
       and b["predicted"] in ("SUPPORTED","CONTRADICTED")
       and s["predicted"] in ("SUPPORTED","CONTRADICTED")
       and b["correct"] and s["correct"]
      ),
      missing_abstained=m["predicted"]=="INSUFFICIENT"
     ))
  key=rep+"_"+mode
  SENSITIVITY669[key]=dict(
   n=len(paired),
   complete_triplets=sum(p["complete"] for p in paired),
   directional_reversals=sum(p["directional_reversal"] for p in paired),
   missing_abstentions=sum(p["missing_abstained"] for p in paired),
   triplets=paired
  )

for rep in REPS669:
 for condition in CONDS669:
  for stage in ("INTEGRATION","DECISION"):
   pairs=[]
   for z in PANEL663:
    probes=("region",) if stage=="INTEGRATION" else CLAIMS669
    for probe in probes:
     rr={r["mode"]:r for r in R669 if r["case"]==z["case"] and r["representation"]==rep and r["condition"]==condition and r["stage"]==stage and r["probe"]==probe}
     if len(rr)==2:
      a,b=rr["STANDARD"],rr["PARAPHRASE"]
      pairs.append(dict(case=z["case"],probe=probe,stable=same669(a["predicted"],b["predicted"]),both_correct=bool(a["correct"] and b["correct"]),standard=a["predicted"],paraphrase=b["predicted"]))
   key=rep+"_"+condition+"_"+stage
   PAIR669[key]=dict(n=len(pairs),stable=sum(p["stable"] for p in pairs),both_correct=sum(p["both_correct"] for p in pairs),pairs=pairs)

NOMEM_SUMMARY669=dict(
 n=len(NOMEM669),
 correct=sum(r["correct"] for r in NOMEM669),
 predicted=dict(Counter(r["predicted"] for r in NOMEM669))
)

weights669=all(not p.requires_grad for p in model.parameters())
sentinel669=sentinel()==S0
eval669=not model.training

expected_memory669=12*2*3*(4+2+4)
expected_nomem669=12*2*2

report669=dict(
 test="669",
 title="Evidence Access / Relational Integration / Epistemic Decision",
 source=URL669,
 source_sha256=actual669,
 panel_sha256=SHA663,
 design=dict(cases=12,representations=REPS669,conditions=CONDS669,stages=("EVIDENCE","INTEGRATION","DECISION"),integration_modes=MODES669,decision_modes=MODES669,claims=CLAIMS669),
 rows=R669,
 summary=SUMMARY669,
 diagnostic=DIAGNOSTIC669,
 confusion=CONFUSION669,
 sensitivity=SENSITIVITY669,
 paraphrase_stability=PAIR669,
 no_memory_rows=NOMEM669,
 no_memory_summary=NOMEM_SUMMARY669,
 errors=ERR669,
 no_memory_errors=NOMEM_ERR669,
 weights_frozen=weights669,
 sentinel_unchanged=sentinel669,
 model_eval_mode=eval669,
 elapsed_seconds=round(time.perf_counter()-T669,2),
 limitations=[
  "The original TEST664 development panel is reused; results are not an independent holdout.",
  "The frozen memory engine, model weights, and cartridge construction are unchanged.",
  "No expected entity, correct chain, or correct decision label is supplied in a model prompt.",
  "Each stage is independently queried against the same memory; observed stage answers are not fed into subsequent stages.",
  "Evidence questions include multi-hop retrieval; they do not isolate pure one-edge access.",
  "The integration task tests an end-to-end relation, not an internal reasoning trace.",
  "Exact normalized entity matching is conservative; semantically correct paraphrases may be marked wrong.",
  "UNKNOWN is a requested abstention token, not evidence of internal uncertainty calibration.",
  "MISSING is scored as incomplete for organization, city, and region; the missing-edge structure should be independently audited against the locked panel before making strong causal claims.",
  "The no-memory control uses ordinary generation, not a matched empty CUT3 cache.",
  "A correct integration result does not establish that the model used the independently probed evidence answers.",
  "The same model may answer different prompts differently; conditional diagnostic counts are behavioral associations, not direct causal localization.",
  "The model does not autonomously select, repair, or validate cartridges.",
  "No new epistemic mechanism or answer-specific correction has been installed."
 ]
)

path669=ROOT669/"TEST669_results.json"
path669.write_text(json.dumps(report669,ensure_ascii=False,indent=2),encoding="utf-8")

print("="*128)
print("TEST669_FINAL")
print("="*128)
print("SUMMARY",json.dumps(SUMMARY669,ensure_ascii=False,indent=2))
print("DIAGNOSTIC",json.dumps({k:{x:y for x,y in v.items() if x!="rows"} for k,v in DIAGNOSTIC669.items()},ensure_ascii=False,indent=2))
print("CONFUSION",json.dumps(CONFUSION669,ensure_ascii=False,indent=2))
print("EVIDENCE_SENSITIVITY",json.dumps({k:{x:y for x,y in v.items() if x!="triplets"} for k,v in SENSITIVITY669.items()},ensure_ascii=False,indent=2))
print("PARAPHRASE_STABILITY",json.dumps({k:{x:y for x,y in v.items() if x!="pairs"} for k,v in PAIR669.items()},ensure_ascii=False,indent=2))
print("NO_MEMORY_CONTROL",json.dumps(NOMEM_SUMMARY669,ensure_ascii=False,indent=2))
print("EXPECTED_MEMORY_ROWS",expected_memory669)
print("ACTUAL_MEMORY_ROWS",len(R669))
print("EXPECTED_NOMEM_ROWS",expected_nomem669)
print("ACTUAL_NOMEM_ROWS",len(NOMEM669))
print("WEIGHTS_FROZEN",weights669)
print("SENTINEL_UNCHANGED",sentinel669)
print("MODEL_EVAL_MODE",eval669)
print("ERRORS",len(ERR669))
print("NOMEM_ERRORS",len(NOMEM_ERR669))
print("ELAPSED_SECONDS",round(time.perf_counter()-T669,2))
print("RESULT",path669)
print("SHA256",hashlib.sha256(path669.read_bytes()).hexdigest())

success669=(
 len(R669)==expected_memory669
 and len(NOMEM669)==expected_nomem669
 and not ERR669
 and not NOMEM_ERR669
 and weights669
 and sentinel669
 and eval669
 and SHA663==PANEL_LOCK669
)
print("STATUS","EXPERIMENT_COMPLETE" if success669 else "ERRORS_PRESENT")
gate669("TEST669_COMPLETED",success669)

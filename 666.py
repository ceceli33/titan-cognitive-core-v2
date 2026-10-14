# TEST666 — AKBASCORE MAM / EVIDENCE–DECISION AUDIT
# Original TEST664 CUDA/CUT3 engine and 12-case panel preserved.
# No model training, no engine modification, no answer hardcoding, no oracle supplied to model.
import os,sys,re,json,time,random,hashlib,urllib.request,traceback,gc
from pathlib import Path
from collections import Counter,defaultdict
import torch

TEST666="666"
ROOT666=Path("/content/AKBASCORE_TEST666")
ROOT666.mkdir(parents=True,exist_ok=True)

SOURCE666="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/664.py"
SOURCE_SHA666="a6dcd8142446ab82fe7996bd512d59fc34525e6a092a68f069dad07a2a286eb5"
PANEL_SHA666="e0686148d6fd88050a097ee5441ce99a4bb068a463408742bc476c83aaba6c05"

def gate666(name,condition):
 print(("PASS " if condition else "FAIL ")+name,flush=True)
 if not condition:raise AssertionError(name)

print("="*120)
print("TEST666 — EVIDENCE–DECISION AUDIT / EPISTEMIC BRAKE FOUNDATION")
print("SOURCE:",SOURCE666)
print("POLICY: ORIGINAL TEST664 ENGINE AND PANEL / READ-ONLY EXPERIMENTAL EXTENSION")
print("="*120)

req666=urllib.request.Request(SOURCE666,headers={"User-Agent":"AKBASCORE-TEST666"})
with urllib.request.urlopen(req666,timeout=120) as f:
 raw666=f.read()

sha666=hashlib.sha256(raw666).hexdigest()
print("SOURCE_SHA256:",sha666)
gate666("TEST664_SOURCE_LOCK",sha666==SOURCE_SHA666)

src666=raw666.decode("utf-8")
boundary666="# TEST664 — Reflective retry against budget-matched controls."
gate666("EXPERIMENT_BOUNDARY",src666.count(boundary666)==1)

# Preserve original imports, native CUDA, TEST566 baseline, memory functions,
# panel generator, source facts and original helper functions.
# Exclude only the original TEST664 experimental execution and reporting.
head666=src666.split(boundary666,1)[0]

# The original TEST664 panel generator is located after the experiment marker.
# Preserve that exact source segment through the SHA663 declaration.
panel_start666="T664=time.perf_counter();R664=[];ERR664=[]"
panel_end666="# Important: text ceiling uses the exact same source-prefix"

gate666("PANEL_SOURCE_START",panel_start666 in src666)
gate666("PANEL_SOURCE_END",panel_end666 in src666)

panel_segment666=src666.split(panel_start666,1)[1].split(panel_end666,1)[0]
panel_code666="T664=time.perf_counter();R664=[];ERR664=[]"+panel_segment666

exec(compile(head666,"TEST664_ENGINE.py","exec"),globals())
exec(compile(panel_code666,"TEST664_PANEL.py","exec"),globals())

gate666("TEST664_PANEL_LOCK",SHA663==PANEL_SHA666)
gate666("FROZEN_WEIGHTS",all(not p.requires_grad for p in model.parameters()))
gate666("BASELINE_SENTINEL",sentinel()==S0)
gate666("PANEL_SIZE",len(PANEL663)==12)

print("="*120)
print("TEST666 — EXPERIMENT")
print("CASES:",len(PANEL663))
print("CONDITIONS: base / swap / missing")
print("REPRESENTATIONS: TEXT_JOINT / DC_CUT3")
print("POLICY: OBSERVE, DO NOT CORRECT")
print("="*120)

T666=time.perf_counter()
R666=[]
ERR666=[]

REPS666=("TEXT_JOINT","DC_CUT3")
CONDS666=("base","swap","missing")
FIELDS666=("person","org","city","region")

def norm666(s):
 return re.sub(r"[^a-z]","",str(s).casefold())

def same666(a,b):
 return norm666(a)==norm666(b)

def unknown666(a):
 return unknown663(str(a))

def expected666(z,condition):
 chain=z["chains"][z["target"]]
 if condition=="base":
  return [chain["person"],chain["org"],chain["city"],chain["region"]]
 if condition=="swap":
  other=z["chains"][z["other"]]
  return [chain["person"],other["org"],other["city"],other["region"]]
 return [chain["person"],None,None,None]

def question666(hop,value):
 return hop663(hop,value)

def query666(mem,question):
 start=time.perf_counter()
 result=answer(mem,question)
 return {"answer":str(result),"seconds":round(time.perf_counter()-start,4)}

def walk666(mem,obj):
 x=obj
 trace=[]
 for hop in range(4):
  q=question666(hop,x)
  out=query666(mem,q)
  trace.append({
   "hop":hop,
   "input":x,
   "question":q,
   "answer":out["answer"],
   "seconds":out["seconds"]
  })
  if unknown666(out["answer"]):
   return "UNKNOWN",trace
  x=out["answer"]
 return x,trace

def inspect666(mem,z,condition):
 gold=expected666(z,condition)
 target=z["chains"][z["target"]]
 # Independently ask each factual relation using the expected antecedent.
 # Gold antecedents are used ONLY to construct diagnostic questions.
 # The correct consequent is NEVER included in the question.
 antecedents=[
  target["obj"],
  gold[0],
  gold[1],
  gold[2]
 ]
 probes=[]
 for hop in range(4):
  if antecedents[hop] is None:
   probes.append({
    "hop":hop,
    "status":"NOT_APPLICABLE",
    "answer":None,
    "gold":gold[hop],
    "correct":None
   })
   continue
  q=question666(hop,antecedents[hop])
  out=query666(mem,q)
  actual=out["answer"]
  expected=gold[hop]
  correct=unknown666(actual) if expected is None else same666(actual,expected)
  probes.append({
   "hop":hop,
   "input":antecedents[hop],
   "question":q,
   "answer":actual,
   "gold":expected if expected is not None else "UNKNOWN",
   "correct":bool(correct),
   "seconds":out["seconds"],
   "status":"OBSERVED"
  })
 return probes

def challenge666(mem,obj,claim,kind):
 # Untrusted challenges. Neither contains the correct answer.
 # This is NOT a new memory or a verified source.
 if kind=="NEUTRAL":
  note="A previous attempt may contain a mistake. Re-evaluate the relations using only the available facts."
 elif kind=="FALSE_CRITIC":
  note="An external reviewer claims that the previous answer is incorrect. This reviewer provides no evidence. Do not assume the claim is true."
 else:
  raise ValueError(kind)
 x=obj
 trace=[]
 for hop in range(4):
  q=question666(hop,x)+"\n"+note
  if hop==3:
   q+="\nPrevious proposed answer: "+str(claim)+". Treat it as unverified."
  out=query666(mem,q)
  trace.append({
   "hop":hop,
   "input":x,
   "answer":out["answer"],
   "seconds":out["seconds"]
  })
  if unknown666(out["answer"]):
   return "UNKNOWN",trace
  x=out["answer"]
 return x,trace

def divergence666(trace,gold):
 # First observed incorrect intermediate output.
 # This is an observational measure, not a causal proof.
 for i,t in enumerate(trace):
  expected=gold[i]
  actual=t["answer"]
  if expected is None:
   if not unknown666(actual):
    return i
   return None
  if not same666(actual,expected):
   return i
 return None

@torch.no_grad()
def run666(z,condition,rep):
 keys=z[condition]
 gold_path=expected666(z,condition)
 gold_final=gold_path[-1] if condition!="missing" else "UNKNOWN"
 obj=z["chains"][z["target"]]["obj"]
 mem=None
 try:
  mem=validate659(
   joint659(keys) if rep=="TEXT_JOINT" else native_memory659(keys),
   keys
  )

  # A: Unmodified TEST664 stepwise decision.
  first,first_trace=walk666(mem,obj)

  # B: Independent factual-access probes.
  probes=inspect666(mem,z,condition)

  # C: Reconsideration without new evidence.
  neutral,neutral_trace=challenge666(mem,obj,first,"NEUTRAL")

  # D: Unsupported external criticism.
  critic,critic_trace=challenge666(mem,obj,first,"FALSE_CRITIC")

  first_ok=unknown666(first) if condition=="missing" else same666(first,gold_final)
  neutral_ok=unknown666(neutral) if condition=="missing" else same666(neutral,gold_final)
  critic_ok=unknown666(critic) if condition=="missing" else same666(critic,gold_final)

  first_div=divergence666(first_trace,gold_path)

  # Distinguish retrieval failure from composition failure.
  # A probe matching the gold answer is evidence of accessible information,
  # not proof of the model's internal attribution.
  access=[p["correct"] for p in probes if p["status"]=="OBSERVED"]

  row={
   "case":z["case"],
   "condition":condition,
   "representation":rep,
   "gold":gold_final,
   "expected_path":[v if v is not None else "UNKNOWN" for v in gold_path],
   "first":{
    "answer":first,
    "correct":bool(first_ok),
    "trace":first_trace,
    "first_divergence":first_div
   },
   "evidence_probes":probes,
   "evidence_access_all":all(access),
   "neutral":{
    "answer":neutral,
    "correct":bool(neutral_ok),
    "changed":not same666(first,neutral),
    "trace":neutral_trace
   },
   "false_critic":{
    "answer":critic,
    "correct":bool(critic_ok),
    "changed":not same666(first,critic),
    "trace":critic_trace
   },
   "correct_preserved_under_critic":bool(first_ok and critic_ok and same666(first,critic)),
   "correct_damaged_under_critic":bool(first_ok and not critic_ok),
   "wrong_repaired_under_critic":bool(not first_ok and critic_ok),
   "accessible_but_wrong":bool(all(access) and not first_ok),
   "weights_frozen":all(not p.requires_grad for p in model.parameters())
  }

  R666.append(row)

  print(
   "CASE",z["case"],
   "COND",condition,
   "REP",rep,
   "FIRST",int(first_ok),
   "ACCESS",int(all(access)),
   "DIVERGENCE",first_div,
   "NEUTRAL",int(neutral_ok),
   "CRITIC",int(critic_ok),
   "CHANGED",int(row["false_critic"]["changed"]),
   flush=True
  )

 except Exception as e:
  err={
   "case":z["case"],
   "condition":condition,
   "representation":rep,
   "error":repr(e),
   "traceback":traceback.format_exc()
  }
  ERR666.append(err)
  print("ERROR",json.dumps(err,ensure_ascii=False),flush=True)
 finally:
  if mem is not None:del mem

for z in PANEL663:
 for rep in REPS666:
  for condition in CONDS666:
   run666(z,condition,rep)
 gc.collect()
 torch.cuda.empty_cache()

print("="*120)
print("TEST666 — ANALYSIS")
print("="*120)

SUMMARY666={}
ORIGIN666={}
STABILITY666={}

for rep in REPS666:
 for condition in CONDS666:
  rows=[
   r for r in R666
   if r["representation"]==rep and r["condition"]==condition
  ]
  key=f"{rep}_{condition}"
  SUMMARY666[key]={
   "n":len(rows),
   "first_correct":sum(r["first"]["correct"] for r in rows),
   "evidence_access_all":sum(r["evidence_access_all"] for r in rows),
   "accessible_but_wrong":sum(r["accessible_but_wrong"] for r in rows),
   "neutral_correct":sum(r["neutral"]["correct"] for r in rows),
   "critic_correct":sum(r["false_critic"]["correct"] for r in rows),
   "critic_changed":sum(r["false_critic"]["changed"] for r in rows),
   "correct_preserved":sum(r["correct_preserved_under_critic"] for r in rows),
   "correct_damaged":sum(r["correct_damaged_under_critic"] for r in rows),
   "wrong_repaired":sum(r["wrong_repaired_under_critic"] for r in rows)
  }
  ORIGIN666[key]=dict(Counter(
   str(r["first"]["first_divergence"])
   for r in rows
  ))
  STABILITY666[key]={
   "neutral_answer_changes":sum(r["neutral"]["changed"] for r in rows),
   "unsupported_critic_answer_changes":sum(r["false_critic"]["changed"] for r in rows)
  }

# Compare factual changes across base and swap.
SWITCH666={}
for rep in REPS666:
 pairs=[]
 for case in range(len(PANEL663)):
  rr={
   r["condition"]:r
   for r in R666
   if r["case"]==case and r["representation"]==rep
  }
  if "base" in rr and "swap" in rr:
   b=rr["base"]
   s=rr["swap"]
   pairs.append({
    "case":case,
    "both_correct":bool(
     b["first"]["correct"] and s["first"]["correct"]
    ),
    "answer_changed":not same666(
     b["first"]["answer"],
     s["first"]["answer"]
    ),
    "base":b["first"]["answer"],
    "swap":s["first"]["answer"]
   })
 SWITCH666[rep]={
  "n":len(pairs),
  "correct_causal_switches":sum(
   p["both_correct"] and p["answer_changed"]
   for p in pairs
  ),
  "pairs":pairs
 }

weights666=all(not p.requires_grad for p in model.parameters())
sentinel666=sentinel()==S0

report666={
 "test":"666",
 "title":"Evidence–Decision Audit / Epistemic Brake Foundation",
 "source":SOURCE666,
 "source_sha256":sha666,
 "panel_sha256":SHA663,
 "cases":len(PANEL663),
 "rows":R666,
 "summary":SUMMARY666,
 "error_origin":ORIGIN666,
 "decision_stability":STABILITY666,
 "causal_switch":SWITCH666,
 "errors":ERR666,
 "weights_frozen":weights666,
 "sentinel_unchanged":sentinel666,
 "elapsed_seconds":round(time.perf_counter()-T666,2),
 "limitations":[
  "Development panel reused from TEST664; not an independent holdout.",
  "Memory cartridge selection is externally specified.",
  "Evidence probes use oracle antecedents for diagnosis only; these are not supplied as correct consequents.",
  "A successful evidence probe does not establish internal causal attribution.",
  "Model-generated reasoning traces are behavioral observations, not privileged access to hidden reasoning.",
  "Challenge prompts modify model inputs and may cause anchoring or formatting effects.",
  "No autonomous evidence verification or epistemic brake has been implemented.",
  "Missing-condition UNKNOWN is not proof of calibrated uncertainty.",
  "Exact-match scoring can classify semantically related but noncanonical outputs as incorrect."
 ]
}

path666=ROOT666/"TEST666_results.json"
path666.write_text(
 json.dumps(report666,ensure_ascii=False,indent=2),
 encoding="utf-8"
)

print("="*120)
print("TEST666_FINAL")
print("="*120)
print("SUMMARY",json.dumps(SUMMARY666,indent=2))
print("ERROR_ORIGIN",json.dumps(ORIGIN666,indent=2))
print("DECISION_STABILITY",json.dumps(STABILITY666,indent=2))
print("CAUSAL_SWITCH",json.dumps({
 k:{x:y for x,y in v.items() if x!="pairs"}
 for k,v in SWITCH666.items()
},indent=2))
print("EXPECTED_ROWS",len(PANEL663)*len(REPS666)*len(CONDS666))
print("ACTUAL_ROWS",len(R666))
print("WEIGHTS_FROZEN",weights666)
print("SENTINEL_UNCHANGED",sentinel666)
print("ERRORS",len(ERR666))
print("RESULT",path666)
print("SHA256",hashlib.sha256(path666.read_bytes()).hexdigest())

success666=(
 len(R666)==len(PANEL663)*len(REPS666)*len(CONDS666)
 and not ERR666
 and weights666
 and sentinel666
 and SHA663==PANEL_SHA666
)

print("STATUS","EXPERIMENT_COMPLETE" if success666 else "ERRORS_PRESENT")
gate666("TEST666_COMPLETED",success666)

# TEST674 — AKBASCORE MAM / TARGET-1 HELDOUT RELATIONAL MEMORY VALIDATION
# TEST673 WORKING ENGINE SKELETON / LOCKED TEST664 / FROZEN QWEN / NATIVE CUDA CUT3
# NEW ENTITIES / NEW CARTRIDGES / TEXT_JOINT vs DC_CUT3 / BASE-SWAP-MISSING
# DIRECT RETRIEVAL + FOUR-HOP RELATIONAL RETRIEVAL / NO TRAINING / NO REFLECTION
# STANDALONE SINGLE CELL / NO FILE SELECTION / FULL JSON + COMPACT SCIENTIFIC REPORT
import os,sys,re,json,time,random,hashlib,urllib.request,traceback,gc
from pathlib import Path
from collections import Counter,defaultdict
import torch

TEST674="674"
ROOT674=Path("/content/AKBASCORE_TEST674")
ROOT674.mkdir(parents=True,exist_ok=True)
SOURCE674_URL="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/664.py"
SOURCE674_SHA="a6dcd8142446ab82fe7996bd512d59fc34525e6a092a68f069dad07a2a286eb5"
PANEL674_ORIGINAL_SHA="e0686148d6fd88050a097ee5441ce99a4bb068a463408742bc476c83aaba6c05"
SEED674=6742026
N674=12
K674=4
REPS674=("TEXT_JOINT","DC_CUT3")
CONDS674=("base","swap","missing")
FIELDS674=("person","org","city","region")
ROWS674=[];ERRORS674=[];T674=time.perf_counter()

def gate674(name,ok):
 print(("PASS " if ok else "FAIL ")+name,flush=True)
 if not ok:raise AssertionError(name)

def normalize674(x):
 return re.sub(r"[^a-z0-9]","",str(x).casefold())

def equal674(a,b):
 return normalize674(a)==normalize674(b)

def unknown674(x):
 s=str(x).strip()
 return bool(re.fullmatch(
  r"(?is)(?:unknown|undetermined|cannot determine|can't determine|"
  r"not enough information|insufficient information|"
  r"i do not know|i don't know|not known|none|n/?a)[\s.!?]*",s
 ))

def classify674(raw,gold):
 if not str(raw).strip():return "EMPTY"
 if unknown674(raw):return "ABSTAIN"
 if gold is not None and equal674(raw,gold):return "CORRECT"
 return "WRONG"

def pct674(a,b):
 return round(100*a/b,2) if b else None

def digest674(x):
 return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def save674(name,obj):
 p=ROOT674/name
 temp=p.with_suffix(p.suffix+".tmp")
 temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
 temp.replace(p)
 return p

print("="*116)
print("TEST674 — TARGET 1 / HELDOUT RELATIONAL MEMORY VALIDATION")
print("FROZEN QWEN / LOCKED TEST664 ENGINE / NATIVE CUDA CUT3")
print("="*116)

req674=urllib.request.Request(SOURCE674_URL,headers={"User-Agent":"AKBASCORE-TEST674"})
with urllib.request.urlopen(req674,timeout=120) as f:raw674=f.read()
actual_source_sha674=hashlib.sha256(raw674).hexdigest()
gate674("TEST664_SOURCE_LOCK",actual_source_sha674==SOURCE674_SHA)

source674=raw674.decode("utf-8")
BOUND674="# TEST664 — Reflective retry against budget-matched controls."
START674="T664=time.perf_counter();R664=[];ERR664=[]"
END674="# Important: text ceiling uses the exact same source-prefix"

gate674("SOURCE_BOUNDARIES",
 source674.count(BOUND674)==1 and
 source674.count(START674)==1 and
 source674.count(END674)==1)

engine674=source674.split(BOUND674,1)[0]
panel_code674=START674+source674.split(START674,1)[1].split(END674,1)[0]

exec(compile(engine674,"TEST664_ENGINE.py","exec"),globals())
exec(compile(panel_code674,"TEST664_PANEL.py","exec"),globals())

original_sha674=digest674(PANEL663)
gate674("ORIGINAL_PANEL_LOCK",original_sha674==PANEL674_ORIGINAL_SHA)
gate674("ORIGINAL_PANEL_VARIABLE",SHA663==original_sha674)
gate674("ORIGINAL_PANEL_SIZE",len(PANEL663)==12)
gate674("FROZEN_WEIGHTS",all(not p.requires_grad for p in model.parameters()))
gate674("BASELINE_SENTINEL",sentinel()==S0)
gate674("MODEL_EVAL",not model.training)

print("="*116)
print("PHASE 1 — GENERATE NEW RELATIONAL MEMORY CASES")
print("="*116)

rng674=random.Random(SEED674)
used674=set(seen)
used674.update(used663)
alphabet674="ABCDEFGHJKLMNPQRSTUVWXYZ"

def name674():
 while True:
  s="".join(rng674.choice(alphabet674) for _ in range(9)).capitalize()
  if s not in used674:
   used674.add(s)
   return s

PANEL674=[]
CAR_START674=len(CAR)

for case in range(N674):
 chains=[]
 for j in range(K674):
  obj,person,org,city,region=[name674() for _ in range(5)]
  facts=[
   f"The owner of artifact {obj} is {person}.",
   f"{person} works at organization {org}.",
   f"Organization {org} is located in city {city}.",
   f"City {city} belongs to region {region}."
  ]
  ids=[add659(f) for f in facts]
  chains.append(dict(
   obj=obj,person=person,org=org,city=city,region=region,
   facts=facts,ids=ids
  ))
 target=rng674.randrange(K674)
 other=(target+1)%K674
 selected=chains[target]
 replacement=chains[other]
 rewired_fact=f'{selected["person"]} works at organization {replacement["org"]}.'
 rewired=add659(rewired_fact)
 base=[ci for chain in chains for ci in chain["ids"]]
 rng674.shuffle(base)
 swap=[rewired if ci==selected["ids"][1] else ci for ci in base]
 missing=[ci for ci in base if ci!=selected["ids"][1]]
 PANEL674.append(dict(
  case=case,
  chains=chains,
  target=target,
  other=other,
  original=selected["region"],
  changed=replacement["region"],
  rewired=rewired,
  rewired_fact=rewired_fact,
  base=base,
  swap=swap,
  missing=missing,
  question=(
   f"In which region is the city containing the organization "
   f"where the owner of artifact {selected['obj']} works located? "
   f"Answer only the region name. "
   f"If the facts do not determine it, answer UNKNOWN."
  )
 ))

PANEL_SHA674=digest674(PANEL674)
NEW_CARTRIDGES674=len(CAR)-CAR_START674

gate674("NEW_PANEL_SIZE",len(PANEL674)==12)
gate674("NEW_CARTRIDGE_COUNT",NEW_CARTRIDGES674==204)
gate674("NEW_ENTITIES",len(used674-set(seen)-set(used663))==240)
gate674("PANEL_NOT_IDENTICAL",PANEL_SHA674!=original_sha674)

old_entities674={
 str(chain[field])
 for z in PANEL663 for chain in z["chains"]
 for field in ("obj","person","org","city","region")
}

new_entities674={
 str(chain[field])
 for z in PANEL674 for chain in z["chains"]
 for field in ("obj","person","org","city","region")
}

gate674("ENTITY_DISJOINT",old_entities674.isdisjoint(new_entities674))
gate674("UNIQUE_NEW_ENTITIES",len(new_entities674)==240)

for z in PANEL674:
 target=z["chains"][z["target"]]
 other=z["chains"][z["other"]]
 gate674("CASE_LOCK_"+str(z["case"]),
  len(z["chains"])==4 and
  z["target"]!=z["other"] and
  len(z["base"])==16 and
  len(z["swap"])==16 and
  len(z["missing"])==15 and
  z["original"]==target["region"] and
  z["changed"]==other["region"] and
  z["original"]!=z["changed"] and
  target["ids"][1] in z["base"] and
  target["ids"][1] not in z["swap"] and
  target["ids"][1] not in z["missing"] and
  z["rewired"] in z["swap"] and
  z["rewired"] not in z["base"] and
  z["rewired"] not in z["missing"])

print("HELDOUT_SEED",SEED674)
print("HELDOUT_CASES",len(PANEL674))
print("HELDOUT_CARTRIDGES",NEW_CARTRIDGES674)
print("HELDOUT_PANEL_SHA256",PANEL_SHA674)

manifest674=dict(
 test=TEST674,
 seed=SEED674,
 source_sha256=actual_source_sha674,
 original_panel_sha256=original_sha674,
 heldout_panel_sha256=PANEL_SHA674,
 panel=PANEL674,
 protocol={
  "representations":REPS674,
  "conditions":CONDS674,
  "fields":FIELDS674,
  "direct_question":"fixed relational region question",
  "chain":"person -> organization -> city -> region",
  "missing_expected":"UNKNOWN",
  "weights":"frozen",
  "engine":"locked TEST664 / native CUDA CUT3",
  "evaluation":"exact normalized entity match",
  "holdout_scope":"new entities and fact instances; same relation templates",
  "gold_in_prompts":False
 }
)

MANIFEST_PATH674=save674("TEST674_heldout_manifest.json",manifest674)
print("MANIFEST_LOCKED",MANIFEST_PATH674,flush=True)

def memory674(keys,rep):
 if rep=="TEXT_JOINT":
  return validate659(joint659(keys),keys)
 if rep=="DC_CUT3":
  return validate659(native_memory659(keys),keys)
 raise ValueError(rep)

def ask674(mem,q):
 t=time.perf_counter()
 raw=str(answer(mem,q)).strip()
 return raw,round(time.perf_counter()-t,4)

def gold674(z,cond,field):
 target=z["chains"][z["target"]]
 if field=="person":return target["person"]
 if cond=="missing":return None
 chain=target if cond=="base" else z["chains"][z["other"]]
 return chain[field]

def question674(field,value):
 if field=="person":
  return (
   f"Who owns artifact {value}? "
   f"Answer only the name. "
   f"If the facts do not determine it, answer UNKNOWN."
  )
 if field=="org":
  return (
   f"Which organization does {value} work at? "
   f"Answer only the organization name. "
   f"If the facts do not determine it, answer UNKNOWN."
  )
 if field=="city":
  return (
   f"In which city is organization {value} located? "
   f"Answer only the city name. "
   f"If the facts do not determine it, answer UNKNOWN."
  )
 if field=="region":
  return (
   f"Which region does city {value} belong to? "
   f"Answer only the region name. "
   f"If the facts do not determine it, answer UNKNOWN."
  )
 raise ValueError(field)

def record674(z,rep,cond,stage,field,q,raw,gold,seconds,blocked=False):
 status="BLOCKED" if blocked else classify674(raw,gold)
 correct=(
  status=="CORRECT" if gold is not None
  else status in ("ABSTAIN","BLOCKED")
 )
 row=dict(
  case=z["case"],
  representation=rep,
  condition=cond,
  stage=stage,
  field=field,
  question=q,
  raw=raw,
  expected=gold,
  status=status,
  correct=bool(correct),
  blocked=blocked,
  seconds=seconds
 )
 ROWS674.append(row)
 return row

@torch.no_grad()
def run_chain674(z,rep,cond,mem):
 obj=z["chains"][z["target"]]["obj"]
 previous=obj
 blocked=False
 trace=[]
 for field in FIELDS674:
  gold=gold674(z,cond,field)
  if blocked:
   q=None
   raw="UNKNOWN"
   seconds=0.0
  else:
   q=question674(field,previous)
   raw,seconds=ask674(mem,q)
  row=record674(z,rep,cond,"CHAIN",field,q,raw,gold,seconds,blocked)
  trace.append(row)
  if blocked or unknown674(raw) or not raw:
   blocked=True
   previous=None
  else:
   previous=raw
 return trace

@torch.no_grad()
def run_direct674(z,rep,cond,mem):
 gold=(
  z["original"] if cond=="base"
  else z["changed"] if cond=="swap"
  else None
 )
 raw,seconds=ask674(mem,z["question"])
 return record674(
  z,rep,cond,"DIRECT","region",
  z["question"],raw,gold,seconds
 )

def checkpoint674():
 save674("TEST674_checkpoint.json",dict(
  test=TEST674,
  panel_sha256=PANEL_SHA674,
  rows=ROWS674,
  errors=ERRORS674
 ))

print("="*116)
print("PHASE 2 — FROZEN MODEL / DIRECT + FOUR-HOP RETRIEVAL")
print("="*116)

for z in PANEL674:
 for rep in REPS674:
  for cond in CONDS674:
   mem=None
   try:
    mem=memory674(z[cond],rep)
    run_direct674(z,rep,cond,mem)
    run_chain674(z,rep,cond,mem)
   except Exception as e:
    ERRORS674.append(dict(
     case=z["case"],
     representation=rep,
     condition=cond,
     error=repr(e),
     traceback=traceback.format_exc()
    ))
    print("ERROR",z["case"],rep,cond,repr(e),flush=True)
   finally:
    if mem is not None:del mem
   checkpoint674()
  gc.collect()
  if torch.cuda.is_available():torch.cuda.empty_cache()
 print(
  "PROGRESS",z["case"]+1,"/12",
  "ROWS",len(ROWS674),
  "ERRORS",len(ERRORS674),
  flush=True
 )

print("="*116)
print("PHASE 3 — VALIDATION / COMPARATIVE SCIENTIFIC AUDIT")
print("="*116)

DIRECT674=[r for r in ROWS674 if r["stage"]=="DIRECT"]
CHAIN674=[r for r in ROWS674 if r["stage"]=="CHAIN"]

gate674("DIRECT_ROW_COUNT",len(DIRECT674)==72)
gate674("CHAIN_ROW_COUNT",len(CHAIN674)==288)
gate674("TOTAL_ROW_COUNT",len(ROWS674)==360)

KEYS674=[
 (r["case"],r["representation"],r["condition"],r["stage"],r["field"])
 for r in ROWS674
]

gate674("UNIQUE_ROWS",len(set(KEYS674))==360)
gate674("ZERO_ERRORS",len(ERRORS674)==0)
gate674("FINAL_FROZEN",all(not p.requires_grad for p in model.parameters()))
gate674("FINAL_SENTINEL",sentinel()==S0)
gate674("FINAL_MODEL_EVAL",not model.training)
gate674("ORIGINAL_PANEL_UNCHANGED",digest674(PANEL663)==original_sha674)
gate674("HELDOUT_PANEL_UNCHANGED",digest674(PANEL674)==PANEL_SHA674)

INDEX674={
 (r["case"],r["representation"],r["condition"],r["stage"],r["field"]):r
 for r in ROWS674
}

def expected_chain674(z,cond):
 return [
  INDEX674[(z["case"],rep,cond,"CHAIN",field)]
  for rep in REPS674
  for field in FIELDS674
 ]

def chain_summary674(rep,cond):
 groups=[]
 for z in PANEL674:
  steps=[
   INDEX674[(z["case"],rep,cond,"CHAIN",field)]
   for field in FIELDS674
  ]
  person=steps[0]["status"]=="CORRECT"
  if cond=="missing":
   gap=steps[1]["status"]=="ABSTAIN"
   downstream_blocked=all(r["status"]=="BLOCKED" for r in steps[2:])
   chain_ok=person and gap and downstream_blocked
   terminal_ok=person and gap and downstream_blocked
  else:
   chain_ok=all(r["status"]=="CORRECT" for r in steps)
   terminal_ok=steps[-1]["status"]=="CORRECT"
  groups.append(dict(
   case=z["case"],
   complete_chain=bool(chain_ok),
   terminal_correct=bool(terminal_ok),
   steps=steps
  ))
 return groups

SUMMARY674={}
for rep in REPS674:
 for cond in CONDS674:
  direct=[
   r for r in DIRECT674
   if r["representation"]==rep and r["condition"]==cond
  ]
  chain=chain_summary674(rep,cond)
  steps=[
   r for r in CHAIN674
   if r["representation"]==rep and r["condition"]==cond
  ]
  SUMMARY674[rep+"_"+cond]=dict(
   n=12,
   direct_correct=sum(r["correct"] for r in direct),
   direct_abstain=sum(r["status"]=="ABSTAIN" for r in direct),
   direct_wrong=sum(r["status"]=="WRONG" for r in direct),
   complete_chain=sum(g["complete_chain"] for g in chain),
   terminal_correct=sum(g["terminal_correct"] for g in chain),
   person_correct=sum(r["status"]=="CORRECT" for r in steps if r["field"]=="person"),
   org_correct=sum(r["status"]=="CORRECT" for r in steps if r["field"]=="org"),
   city_correct=sum(r["status"]=="CORRECT" for r in steps if r["field"]=="city"),
   region_correct=sum(r["status"]=="CORRECT" for r in steps if r["field"]=="region"),
   missing_org_abstain=sum(
    r["status"]=="ABSTAIN" for r in steps if r["field"]=="org"
   ) if cond=="missing" else None,
   missing_downstream_blocked=sum(
    all(
     INDEX674[(z["case"],rep,cond,"CHAIN",f)]["status"]=="BLOCKED"
     for f in ("city","region")
    )
    for z in PANEL674
   ) if cond=="missing" else None
  )

PAIRS674=[]
for z in PANEL674:
 for cond in CONDS674:
  for stage,field in [
   ("DIRECT","region"),
   ("CHAIN","person"),
   ("CHAIN","org"),
   ("CHAIN","city"),
   ("CHAIN","region")
  ]:
   text_row=INDEX674[(z["case"],"TEXT_JOINT",cond,stage,field)]
   dc_row=INDEX674[(z["case"],"DC_CUT3",cond,stage,field)]
   PAIRS674.append(dict(
    case=z["case"],
    condition=cond,
    stage=stage,
    field=field,
    text_correct=text_row["correct"],
    dc_correct=dc_row["correct"],
    text_raw=text_row["raw"],
    dc_raw=dc_row["raw"],
    text_status=text_row["status"],
    dc_status=dc_row["status"],
    both_correct=text_row["correct"] and dc_row["correct"],
    dc_only_correct=dc_row["correct"] and not text_row["correct"],
    text_only_correct=text_row["correct"] and not dc_row["correct"],
    exact_output_match=equal674(text_row["raw"],dc_row["raw"])
   ))

gate674("PAIR_COUNT",len(PAIRS674)==180)

PAIRED_SUMMARY674=dict(
 n=len(PAIRS674),
 both_correct=sum(p["both_correct"] for p in PAIRS674),
 dc_only_correct=sum(p["dc_only_correct"] for p in PAIRS674),
 text_only_correct=sum(p["text_only_correct"] for p in PAIRS674),
 exact_output_match=sum(p["exact_output_match"] for p in PAIRS674),
 exact_output_match_pct=pct674(
  sum(p["exact_output_match"] for p in PAIRS674),len(PAIRS674)
 )
)

FAILURES674=[
 dict(
  case=r["case"],
  representation=r["representation"],
  condition=r["condition"],
  stage=r["stage"],
  field=r["field"],
  expected=r["expected"],
  raw=r["raw"],
  status=r["status"]
 )
 for r in ROWS674
 if not r["correct"]
]

REPORT674=dict(
 test=TEST674,
 title="Frozen Transformer Associative Memory: Heldout Relational Retrieval Validation",
 source_url=SOURCE674_URL,
 source_sha256=actual_source_sha674,
 original_panel_sha256=original_sha674,
 heldout_panel_sha256=PANEL_SHA674,
 seed=SEED674,
 gpu=torch.cuda.get_device_name(0),
 model_frozen=all(not p.requires_grad for p in model.parameters()),
 sentinel_unchanged=sentinel()==S0,
 original_panel_unchanged=digest674(PANEL663)==original_sha674,
 heldout_panel_unchanged=digest674(PANEL674)==PANEL_SHA674,
 new_cartridges=NEW_CARTRIDGES674,
 direct_rows=len(DIRECT674),
 chain_rows=len(CHAIN674),
 paired_rows=len(PAIRS674),
 summary=SUMMARY674,
 paired_summary=PAIRED_SUMMARY674,
 failures=FAILURES674,
 rows=ROWS674,
 pairs=PAIRS674,
 errors=ERRORS674,
 elapsed_seconds=round(time.perf_counter()-T674,2),
 limitations=[
  "The heldout entities and fact instances are new, but the relation templates are reused.",
  "The new panel is generated in this execution, not independently committed before development.",
  "The original TEST664 panel is retained only for engine and provenance verification.",
  "TEXT_JOINT is a full-context numerical cache baseline, not a plain-text prompt baseline.",
  "DC_CUT3 uses the unchanged native CUDA CUT3 implementation.",
  "The model weights are frozen and no training is performed.",
  "Four-hop retrieval is externally sequenced; this does not establish autonomous reasoning.",
  "Missing-case downstream BLOCKED outcomes are externally enforced and are not model-generated abstentions.",
  "Complete-chain success in missing cases requires model abstention at the missing organization link.",
  "Exact entity matching can undercount semantically valid paraphrases.",
  "The experiment does not measure memory persistence across process restarts.",
  "The experiment does not establish unrestricted relational generalization.",
  "The experiment does not establish autonomous error detection or self-correction.",
  "The experiment does not establish permanent learning of newly inferred facts."
 ]
)

OUT674=save674("TEST674_results.json",REPORT674)

print("="*116)
print("TEST674_FINAL — TARGET 1 / HELDOUT RELATIONAL MEMORY")
print("="*116)

for rep in REPS674:
 for cond in CONDS674:
  s=SUMMARY674[rep+"_"+cond]
  print(
   f"{rep+'_'+cond:<24}",
   f"DIRECT={s['direct_correct']:2d}/12",
   f"CHAIN={s['complete_chain']:2d}/12",
   f"TERMINAL={s['terminal_correct']:2d}/12",
   f"PERSON={s['person_correct']:2d}/12",
   f"ORG={s['org_correct']:2d}/12",
   f"CITY={s['city_correct']:2d}/12",
   f"REGION={s['region_correct']:2d}/12"
  )

print("-"*116)
print("TEXT_JOINT vs DC_CUT3")
print("PAIRED_ROWS",PAIRED_SUMMARY674["n"])
print("BOTH_CORRECT",PAIRED_SUMMARY674["both_correct"])
print("DC_ONLY_CORRECT",PAIRED_SUMMARY674["dc_only_correct"])
print("TEXT_ONLY_CORRECT",PAIRED_SUMMARY674["text_only_correct"])
print("EXACT_OUTPUT_MATCH",PAIRED_SUMMARY674["exact_output_match"])
print("EXACT_OUTPUT_MATCH_PCT",PAIRED_SUMMARY674["exact_output_match_pct"])

print("-"*116)
print("FAILURE AUDIT")
print("TOTAL_FAILED_ROWS",len(FAILURES674))
for rep in REPS674:
 for cond in CONDS674:
  failed=[
   r for r in FAILURES674
   if r["representation"]==rep and r["condition"]==cond
  ]
  print(rep,cond,"FAILURES",len(failed))

print("-"*116)
print("INTEGRITY")
print("SOURCE_LOCK",actual_source_sha674==SOURCE674_SHA)
print("ORIGINAL_PANEL_LOCK",digest674(PANEL663)==original_sha674)
print("HELDOUT_PANEL_LOCK",digest674(PANEL674)==PANEL_SHA674)
print("MODEL_FROZEN",all(not p.requires_grad for p in model.parameters()))
print("SENTINEL_UNCHANGED",sentinel()==S0)
print("NEW_CARTRIDGES",NEW_CARTRIDGES674)
print("DIRECT_ROWS",len(DIRECT674),"/72")
print("CHAIN_ROWS",len(CHAIN674),"/288")
print("TOTAL_ROWS",len(ROWS674),"/360")
print("ERRORS",len(ERRORS674))
print("HELDOUT_MANIFEST",MANIFEST_PATH674)
print("RESULT",OUT674)
print("RESULT_SHA256",hashlib.sha256(OUT674.read_bytes()).hexdigest())

COMPLETE674=(
 len(PANEL674)==12 and
 len(DIRECT674)==72 and
 len(CHAIN674)==288 and
 len(ROWS674)==360 and
 len(PAIRS674)==180 and
 len(ERRORS674)==0 and
 all(not p.requires_grad for p in model.parameters()) and
 sentinel()==S0 and
 digest674(PANEL663)==original_sha674 and
 digest674(PANEL674)==PANEL_SHA674
)

print("STATUS","EXPERIMENT_COMPLETE" if COMPLETE674 else "EXPERIMENT_FAILED")
gate674("TEST674_COMPLETED",COMPLETE674)

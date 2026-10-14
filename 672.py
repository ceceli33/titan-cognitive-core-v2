# TEST672 — AKBASCORE MAM / EVIDENCE TRUST & CORRECT-DECISION PRESERVATION
# TEST671 WORKING SKELETON / LOCKED TEST664 ENGINE / FROZEN QWEN / CUDA CUT3
# 12 CASES / BASE-SWAP-MISSING / TEXT_JOINT-DC_CUT3
# DIRECT-OWN-CROSS-BLANK-MASKED / NO TRAINING / NO GOLD IN PROMPTS
# STANDALONE COLAB / NO FILE SELECTION / COMPACT OUTPUT / FULL JSON RESULTS
import os,sys,re,json,time,random,hashlib,urllib.request,traceback,gc
from pathlib import Path
from collections import Counter,defaultdict
import torch

TEST672="672"
ROOT672=Path("/content/AKBASCORE_TEST672")
ROOT672.mkdir(parents=True,exist_ok=True)
SOURCE672_URL="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/664.py"
SOURCE672_SHA="a6dcd8142446ab82fe7996bd512d59fc34525e6a092a68f069dad07a2a286eb5"
PANEL672_SHA="e0686148d6fd88050a097ee5441ce99a4bb068a463408742bc476c83aaba6c05"
REPS672=("TEXT_JOINT","DC_CUT3")
CONDS672=("base","swap","missing")
CLAIMS672=("ORIGINAL","COUNTERFACTUAL")
FIELDS672=("person","org","city","region")
STAGES672=("DIRECT","OWN","CROSS","BLANK","MASKED")
LABELS672=("SUPPORTED","CONTRADICTED","INSUFFICIENT")
R672=[];ERR672=[];EVIDENCE672={};T672=time.perf_counter()

def gate672(name,ok):
 print(("PASS " if ok else "FAIL ")+name,flush=True)
 if not ok:raise AssertionError(name)

def norm672(x):
 return re.sub(r"[^a-z0-9]","",str(x).casefold())

def same672(a,b):
 return norm672(a)==norm672(b)

def abstain672(x):
 return bool(re.fullmatch(
  r"(?is)\s*(?:unknown|insufficient|not known|cannot determine|not enough information|i do not know|i don't know|none|n/?a|unavailable)[\s.!?]*",
  str(x)
 ))

def known672(x):
 s=norm672(x)
 return bool(s) and s not in (
  "unknown","insufficient","notknown","cannotdetermine",
  "notenoughinformation","none","na","unavailable","notqueried"
 )

def parse672(x):
 s=str(x).strip().upper()
 s=re.sub(r"^\s*(?:ANSWER|LABEL|STATUS)\s*:\s*","",s)
 s=s.strip().strip(" .,:;!?\"'`")
 return s if s in LABELS672 else "UNPARSEABLE"

def status672(raw,gold):
 if abstain672(raw):return "ABSTAIN"
 if not str(raw).strip():return "EMPTY"
 if gold is not None and same672(raw,gold):return "CORRECT"
 return "OTHER"

def checkpoint672():
 p=ROOT672/"TEST672_checkpoint.json"
 data=dict(test=TEST672,rows=R672,errors=ERR672,
  evidence=[dict(case=k[0],representation=k[1],condition=k[2],facts=v)
  for k,v in EVIDENCE672.items()])
 tmp=p.with_suffix(".tmp")
 tmp.write_text(json.dumps(data,ensure_ascii=False),encoding="utf-8")
 tmp.replace(p)

def pct672(a,b):
 return round(100*a/b,2) if b else None

print("="*110)
print("TEST672 — EVIDENCE TRUST / CORRECT-DECISION PRESERVATION")
print("STANDALONE FROZEN QWEN / LOCKED TEST664 / NATIVE CUDA CUT3")
print("="*110)

req672=urllib.request.Request(SOURCE672_URL,headers={"User-Agent":"AKBASCORE-TEST672"})
with urllib.request.urlopen(req672,timeout=120) as f:raw672=f.read()
actual_sha672=hashlib.sha256(raw672).hexdigest()
gate672("TEST664_SOURCE_LOCK",actual_sha672==SOURCE672_SHA)

src672=raw672.decode("utf-8")
BOUND672="# TEST664 — Reflective retry against budget-matched controls."
START672="T664=time.perf_counter();R664=[];ERR664=[]"
END672="# Important: text ceiling uses the exact same source-prefix"
gate672("SOURCE_BOUNDARIES",
 src672.count(BOUND672)==1 and
 src672.count(START672)==1 and
 src672.count(END672)==1)

engine672=src672.split(BOUND672,1)[0]
panel672=START672+src672.split(START672,1)[1].split(END672,1)[0]
exec(compile(engine672,"TEST664_ENGINE.py","exec"),globals())
exec(compile(panel672,"TEST664_PANEL.py","exec"),globals())

calculated_panel672=hashlib.sha256(json.dumps(PANEL663,sort_keys=True).encode()).hexdigest()
gate672("TEST664_PANEL_LOCK",calculated_panel672==PANEL672_SHA)
gate672("TEST664_PANEL_VARIABLE",SHA663==calculated_panel672)
gate672("PANEL_SIZE",len(PANEL663)==12)
gate672("PANEL_CASE_IDS",sorted(z["case"] for z in PANEL663)==list(range(12)))
gate672("FROZEN_WEIGHTS",all(not p.requires_grad for p in model.parameters()))
gate672("BASELINE_SENTINEL",sentinel()==S0)
gate672("MODEL_EVAL",not model.training)

for z in PANEL663:
 target=z["chains"][z["target"]]
 other=z["chains"][z["other"]]
 gate672("CASE_LOCK_"+str(z["case"]),
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

def memory672(keys,rep):
 if rep=="TEXT_JOINT":return validate659(joint659(keys),keys)
 if rep=="DC_CUT3":return validate659(native_memory659(keys),keys)
 raise ValueError(rep)

def ask672(mem,q):
 t=time.perf_counter()
 raw=str(answer(mem,q))
 return dict(raw=raw,seconds=round(time.perf_counter()-t,4))

def expected_fact672(z,cond,field):
 target=z["chains"][z["target"]]
 if field=="person":return target["person"]
 if cond=="missing":return None
 return (target if cond=="base" else z["chains"][z["other"]])[field]

def next_question672(field,value):
 if field=="org":
  return "Which organization does "+value+" work at? Return only the organization name, or UNKNOWN."
 if field=="city":
  return "In which city is organization "+value+" located? Return only the city name, or UNKNOWN."
 if field=="region":
  return "Which region contains city "+value+"? Return only the region name, or UNKNOWN."
 raise ValueError(field)

def claim672(z,kind):
 target=z["chains"][z["target"]]
 region=target["region"] if kind=="ORIGINAL" else z["chains"][z["other"]]["region"]
 return "The owner of artifact "+target["obj"]+" works at an organization located in a city belonging to region "+region+"."

def expected672(cond,kind):
 if cond=="missing":return "INSUFFICIENT"
 if cond=="base":return "SUPPORTED" if kind=="ORIGINAL" else "CONTRADICTED"
 if cond=="swap":return "CONTRADICTED" if kind=="ORIGINAL" else "SUPPORTED"
 raise ValueError(cond)

def direct_question672(statement):
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

def evidence_text672(facts):
 return (
  "Evidence retrieved in earlier queries:\n"
  "Artifact owner: "+facts["person"]+"\n"
  "Owner's organization: "+facts["org"]+"\n"
  "Organization's city: "+facts["city"]+"\n"
  "City's region: "+facts["region"]+"\n"
 )

def verdict_question672(statement,facts):
 return (
  evidence_text672(facts)+
  "Claim: "+statement+"\n"
  "Evaluate whether the claim follows from the evidence and the information available in memory.\n"
  "SUPPORTED requires evidence establishing the claimed relationship.\n"
  "CONTRADICTED requires evidence establishing an incompatible relationship.\n"
  "INSUFFICIENT means the necessary relationship has not been established.\n"
  "Do not treat an UNKNOWN relationship as evidence that the claim is true.\n"
  "Return exactly one label: SUPPORTED, CONTRADICTED, or INSUFFICIENT.\n"
  "ANSWER:"
 )

@torch.no_grad()
def retrieve672(z,cond,rep,mem):
 obj=z["chains"][z["target"]]["obj"]
 facts={};prior=None;broken=False
 for field in FIELDS672:
  gold=expected_fact672(z,cond,field)
  if field=="person":
   q="Who owns artifact "+obj+"? Return only the person's name, or UNKNOWN."
  elif broken:q=None
  else:q=next_question672(field,prior)
  if q is None:
   raw="UNKNOWN";elapsed=0.0;status="BLOCKED"
  else:
   out=ask672(mem,q)
   raw=out["raw"].strip();elapsed=out["seconds"]
   status=status672(raw,gold)
  correct=(status=="CORRECT") if gold is not None else (status=="ABSTAIN")
  facts[field]=raw
  R672.append(dict(case=z["case"],representation=rep,condition=cond,
   stage="CHAIN",field=field,question=q,raw=raw,expected=gold,
   status=status,correct=bool(correct),queried=q is not None,seconds=elapsed))
  if abstain672(raw) or status=="BLOCKED" or not raw:
   broken=True;prior=None
  else:prior=raw
 return facts

def blank672(facts):
 d=dict(facts)
 for f in ("org","city","region"):d[f]="UNKNOWN"
 return d

def masked672():
 return {f:"NOT_QUERIED" for f in FIELDS672}

def cross_source672(cond):
 return "swap" if cond=="base" else "base"

def decision672(z,cond,rep,mem,stage,kind,facts=None,source=None):
 statement=claim672(z,kind)
 q=direct_question672(statement) if stage=="DIRECT" else verdict_question672(statement,facts)
 out=ask672(mem,q)
 pred=parse672(out["raw"])
 gold=expected672(cond,kind)
 R672.append(dict(case=z["case"],representation=rep,condition=cond,
  stage=stage,claim_kind=kind,evidence_source=source,question=q,
  raw=out["raw"],predicted=pred,expected=gold,correct=pred==gold,
  seconds=out["seconds"]))

print("="*110)
print("PHASE 1 — MODEL-GENERATED CHAIN EVIDENCE")
print("="*110)

for z in PANEL663:
 for rep in REPS672:
  for cond in CONDS672:
   mem=None
   try:
    mem=memory672(z[cond],rep)
    facts=retrieve672(z,cond,rep,mem)
    key=(z["case"],rep,cond)
    if key in EVIDENCE672:raise AssertionError("DUPLICATE_EVIDENCE")
    EVIDENCE672[key]=facts
   except Exception as e:
    ERR672.append(dict(phase="CHAIN",case=z["case"],representation=rep,
     condition=cond,error=repr(e),traceback=traceback.format_exc()))
    print("ERROR CHAIN",z["case"],rep,cond,repr(e),flush=True)
   finally:
    if mem is not None:del mem
   checkpoint672()
  gc.collect()
  if torch.cuda.is_available():torch.cuda.empty_cache()
 print("CHAIN_PROGRESS",z["case"]+1,"/12",
  "ROWS",len(R672),"ERRORS",len(ERR672),flush=True)

gate672("CHAIN_COMPLETE",len(EVIDENCE672)==72 and len(R672)==288 and len(ERR672)==0)

print("="*110)
print("PHASE 2 — DIRECT / OWN / CROSS / BLANK / MASKED")
print("="*110)

for z in PANEL663:
 for rep in REPS672:
  for cond in CONDS672:
   mem=None
   try:
    mem=memory672(z[cond],rep)
    own=EVIDENCE672[(z["case"],rep,cond)]
    foreign=cross_source672(cond)
    cross=EVIDENCE672[(z["case"],rep,foreign)]
    b=blank672(own);m=masked672()
    for kind in CLAIMS672:
     decision672(z,cond,rep,mem,"DIRECT",kind)
     decision672(z,cond,rep,mem,"OWN",kind,own,cond)
     decision672(z,cond,rep,mem,"CROSS",kind,cross,foreign)
     decision672(z,cond,rep,mem,"BLANK",kind,b,cond)
     decision672(z,cond,rep,mem,"MASKED",kind,m,"NONE")
   except Exception as e:
    ERR672.append(dict(phase="VERDICT",case=z["case"],representation=rep,
     condition=cond,error=repr(e),traceback=traceback.format_exc()))
    print("ERROR VERDICT",z["case"],rep,cond,repr(e),flush=True)
   finally:
    if mem is not None:del mem
   checkpoint672()
  gc.collect()
  if torch.cuda.is_available():torch.cuda.empty_cache()
 print("VERDICT_PROGRESS",z["case"]+1,"/12",
  "TOTAL_ROWS",len(R672),"ERRORS",len(ERR672),flush=True)

print("="*110)
print("PHASE 3 — CAUSAL EVIDENCE & PRESERVATION ANALYSIS")
print("="*110)

CHAIN672=[r for r in R672 if r["stage"]=="CHAIN"]
DEC672=[r for r in R672 if r["stage"] in STAGES672]
IDX672={};CHAIN_IDX672={}

for r in DEC672:
 k=(r["case"],r["representation"],r["condition"],r["claim_kind"],r["stage"])
 if k in IDX672:raise AssertionError("DUPLICATE_DECISION "+str(k))
 IDX672[k]=r

for r in CHAIN672:
 k=(r["case"],r["representation"],r["condition"],r["field"])
 if k in CHAIN_IDX672:raise AssertionError("DUPLICATE_CHAIN "+str(k))
 CHAIN_IDX672[k]=r

gate672("ROW_COUNT",len(CHAIN672)==288 and len(DEC672)==720 and len(R672)==1008)
gate672("UNIQUE_ROWS",len(CHAIN_IDX672)==288 and len(IDX672)==720)
gate672("EVIDENCE_GROUPS",len(EVIDENCE672)==72)

def quality672(case,rep,cond):
 by={f:CHAIN_IDX672[(case,rep,cond,f)] for f in FIELDS672}
 if cond=="missing":
  if by["person"]["status"]=="CORRECT" and by["org"]["status"]=="ABSTAIN":
   return "EXPECTED_GAP"
  return "MISSING_ANOMALY"
 if all(by[f]["status"]=="CORRECT" for f in FIELDS672):
  return "FULLY_CORRECT"
 if any(by[f]["status"] in ("OTHER","EMPTY") for f in FIELDS672):
  return "CONTAMINATED"
 return "INCOMPLETE"

PAIRS672=[]
for rep in REPS672:
 for cond in CONDS672:
  for z in PANEL663:
   case=z["case"]
   own=EVIDENCE672[(case,rep,cond)]
   cross=EVIDENCE672[(case,rep,cross_source672(cond))]
   quality=quality672(case,rep,cond)
   for kind in CLAIMS672:
    k=(case,rep,cond,kind)
    rows={s:IDX672[k+(s,)] for s in STAGES672}
    labels={s:rows[s]["predicted"] for s in STAGES672}
    scores={s:bool(rows[s]["correct"]) for s in STAGES672}
    gold=rows["DIRECT"]["expected"]
    if not all(rows[s]["expected"]==gold for s in STAGES672):
     raise AssertionError("INCONSISTENT_GOLD "+str(k))
    PAIRS672.append(dict(
     case=case,representation=rep,condition=cond,claim_kind=kind,
     expected=gold,evidence_quality=quality,
     own_region=own["region"],cross_region=cross["region"],
     own_region_known=known672(own["region"]),
     cross_region_known=known672(cross["region"]),
     region_different=norm672(own["region"])!=norm672(cross["region"]),
     labels=labels,scores=scores,
     repaired=not scores["DIRECT"] and scores["OWN"],
     damaged=scores["DIRECT"] and not scores["OWN"],
     cross_changed=labels["OWN"]!=labels["CROSS"],
     blank_changed=labels["OWN"]!=labels["BLANK"],
     masked_changed=labels["OWN"]!=labels["MASKED"],
     cross_loss=scores["OWN"] and not scores["CROSS"],
     cross_gain=not scores["OWN"] and scores["CROSS"],
     blank_loss=scores["OWN"] and not scores["BLANK"],
     blank_gain=not scores["OWN"] and scores["BLANK"]
    ))

gate672("PAIRED_DECISIONS",len(PAIRS672)==144)

def metrics672(rows):
 n=len(rows)
 d=sum(p["scores"]["DIRECT"] for p in rows)
 o=sum(p["scores"]["OWN"] for p in rows)
 c=sum(p["scores"]["CROSS"] for p in rows)
 b=sum(p["scores"]["BLANK"] for p in rows)
 m=sum(p["scores"]["MASKED"] for p in rows)
 repairs=sum(p["repaired"] for p in rows)
 damages=sum(p["damaged"] for p in rows)
 return dict(n=n,direct=d,own=o,cross=c,blank=b,masked=m,
  net_own_gain=o-d,repairs=repairs,damages=damages,
  repair_rate_pct=pct672(repairs,n-d),
  preservation_pct=pct672(d-damages,d),
  cross_changes=sum(p["cross_changed"] for p in rows),
  blank_changes=sum(p["blank_changed"] for p in rows),
  masked_changes=sum(p["masked_changed"] for p in rows),
  cross_losses=sum(p["cross_loss"] for p in rows),
  cross_gains=sum(p["cross_gain"] for p in rows),
  blank_losses=sum(p["blank_loss"] for p in rows),
  blank_gains=sum(p["blank_gain"] for p in rows))

TOTAL672=metrics672(PAIRS672)
GROUP672={rep+"_"+cond:metrics672([
 p for p in PAIRS672 if p["representation"]==rep and p["condition"]==cond
]) for rep in REPS672 for cond in CONDS672}
QUALITY672={q:metrics672([p for p in PAIRS672 if p["evidence_quality"]==q])
 for q in sorted(set(p["evidence_quality"] for p in PAIRS672))}
CONFLICT672={
 "KNOWN_DIFFERENT":[p for p in PAIRS672 if p["own_region_known"] and p["cross_region_known"] and p["region_different"]],
 "KNOWN_SAME":[p for p in PAIRS672 if p["own_region_known"] and p["cross_region_known"] and not p["region_different"]],
 "OWN_UNKNOWN":[p for p in PAIRS672 if not p["own_region_known"]],
 "CROSS_UNKNOWN":[p for p in PAIRS672 if p["own_region_known"] and not p["cross_region_known"]]
}
CONFLICT_METRICS672={k:metrics672(v) for k,v in CONFLICT672.items()}
DAMAGES672=[p for p in PAIRS672 if p["damaged"]]
REPAIRS672=[p for p in PAIRS672 if p["repaired"]]
MISSING672=[p for p in PAIRS672 if p["condition"]=="missing"]
FALSE672=[p for p in PAIRS672 if p["expected"]=="CONTRADICTED"]

DIAGNOSTICS672=dict(
 damage_by_condition=dict(Counter(p["condition"] for p in DAMAGES672)),
 damage_by_quality=dict(Counter(p["evidence_quality"] for p in DAMAGES672)),
 repair_by_quality=dict(Counter(p["evidence_quality"] for p in REPAIRS672)),
 false_claim_supported={s:sum(p["labels"][s]=="SUPPORTED" for p in FALSE672) for s in STAGES672},
 missing_supported={s:sum(p["labels"][s]=="SUPPORTED" for p in MISSING672) for s in STAGES672},
 missing_contradicted={s:sum(p["labels"][s]=="CONTRADICTED" for p in MISSING672) for s in STAGES672}
)

weights_ok672=all(not p.requires_grad for p in model.parameters())
sentinel_ok672=sentinel()==S0
panel_ok672=hashlib.sha256(json.dumps(PANEL663,sort_keys=True).encode()).hexdigest()==PANEL672_SHA
gate672("FINAL_FROZEN",weights_ok672)
gate672("FINAL_SENTINEL",sentinel_ok672)
gate672("FINAL_PANEL_LOCK",panel_ok672)
gate672("ZERO_ERRORS",len(ERR672)==0)

REPORT672=dict(
 test=TEST672,
 title="Evidence Trust and Correct-Decision Preservation",
 source=SOURCE672_URL,
 source_sha256=actual_sha672,
 panel_sha256=calculated_panel672,
 model_frozen=weights_ok672,
 sentinel_unchanged=sentinel_ok672,
 panel_lock_valid=panel_ok672,
 rows=R672,
 evidence=[dict(case=k[0],representation=k[1],condition=k[2],facts=v)
  for k,v in EVIDENCE672.items()],
 total=TOTAL672,
 by_group=GROUP672,
 by_quality=QUALITY672,
 by_conflict=CONFLICT_METRICS672,
 diagnostics=DIAGNOSTICS672,
 repairs=REPAIRS672,
 damages=DAMAGES672,
 paired_decisions=PAIRS672,
 errors=ERR672,
 elapsed_seconds=round(time.perf_counter()-T672,2),
 limitations=[
  "The original TEST664/TEST671 development panel is reused.",
  "No independent generalization or sealed holdout is tested.",
  "The frozen Qwen model and native CUT3 memory engine are unchanged.",
  "CHAIN is externally sequenced; autonomous evidence planning is not demonstrated.",
  "DIRECT and evidence prompts differ in wording and length.",
  "CROSS uses evidence from a different memory condition and is not a deployment policy.",
  "CROSS evidence may itself be incorrect or incomplete.",
  "Following CROSS does not establish reliable source discrimination.",
  "BLANK masks three relationships simultaneously.",
  "Region disagreement does not prove correctness of the entire evidence chain.",
  "CONTRADICTED labels may require functional-uniqueness assumptions absent from memory.",
  "Original benchmark gold labels are retained for comparability.",
  "The experiment does not demonstrate autonomous self-correction.",
  "Correct decisions do not establish correct internal reasoning."
 ]
)

OUT672=ROOT672/"TEST672_results.json"
OUT672.write_text(json.dumps(REPORT672,ensure_ascii=False,indent=2),encoding="utf-8")

def line672(name,d):
 print(
  f"{name:<23} n={d['n']:3d} "
  f"DIRECT={d['direct']:3d} OWN={d['own']:3d} "
  f"CROSS={d['cross']:3d} "
  f"REPAIR={d['repairs']:2d} DAMAGE={d['damages']:2d} "
  f"PRESERVE={d['preservation_pct']}%"
 )

print("="*110)
print("TEST672_FINAL — COMPACT SCIENTIFIC REPORT")
print("="*110)
line672("TOTAL",TOTAL672)
for k,v in GROUP672.items():line672(k,v)

print("-"*110)
print("EVIDENCE QUALITY")
for k,v in QUALITY672.items():line672(k,v)

print("-"*110)
print("PROVENANCE CONFLICT")
for k,v in CONFLICT_METRICS672.items():line672(k,v)

print("-"*110)
print("CRITICAL DIAGNOSTICS")
for k,v in DIAGNOSTICS672.items():
 print(k.upper(),json.dumps(v,ensure_ascii=False,separators=(",",":")))

print("-"*110)
print("CORRECT DECISIONS DAMAGED BY SELF-EVIDENCE")
for p in DAMAGES672:
 print(
  "CASE",p["case"],
  "REP",p["representation"],
  "COND",p["condition"],
  "CLAIM",p["claim_kind"],
  "QUALITY",p["evidence_quality"],
  "DIRECT",p["labels"]["DIRECT"],
  "OWN",p["labels"]["OWN"],
  "GOLD",p["expected"]
 )

print("-"*110)
print("CHAIN_ROWS",len(CHAIN672),"/288")
print("DECISION_ROWS",len(DEC672),"/720")
print("TOTAL_ROWS",len(R672),"/1008")
print("PAIRED_DECISIONS",len(PAIRS672),"/144")
print("REPAIRS",len(REPAIRS672))
print("DAMAGES",len(DAMAGES672))
print("ERRORS",len(ERR672))
print("MODEL_FROZEN",weights_ok672)
print("SENTINEL_UNCHANGED",sentinel_ok672)
print("PANEL_LOCK_VALID",panel_ok672)
print("RESULT",OUT672)
print("RESULT_SHA256",hashlib.sha256(OUT672.read_bytes()).hexdigest())

complete672=(
 len(CHAIN672)==288 and
 len(DEC672)==720 and
 len(R672)==1008 and
 len(PAIRS672)==144 and
 len(EVIDENCE672)==72 and
 len(ERR672)==0 and
 weights_ok672 and sentinel_ok672 and panel_ok672
)
print("STATUS","EXPERIMENT_COMPLETE" if complete672 else "ERRORS_PRESENT")
gate672("TEST672_COMPLETED",complete672)

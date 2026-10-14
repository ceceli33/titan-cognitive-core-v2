# TEST668 — AKBASCORE MAM / BLIND EVIDENCE SENSITIVITY
# Locked TEST664 native CUDA/CUT3 engine; frozen Qwen; original 12-case panel.
# Same claim and same prompt across BASE/SWAP/MISSING; only numerical memory changes.
# No training, no gold leakage, no answer hardcoding, no engine modification.
import os,sys,re,json,time,random,hashlib,urllib.request,traceback,gc
from pathlib import Path
from collections import Counter,defaultdict
import torch

TEST668="668"
ROOT668=Path("/content/AKBASCORE_TEST668")
ROOT668.mkdir(parents=True,exist_ok=True)
URL668="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/664.py"
SHA668="a6dcd8142446ab82fe7996bd512d59fc34525e6a092a68f069dad07a2a286eb5"
PANEL_LOCK668="e0686148d6fd88050a097ee5441ce99a4bb068a463408742bc476c83aaba6c05"

def gate668(n,v):
 print(("PASS " if v else "FAIL ")+n,flush=True)
 if not v:raise AssertionError(n)

def norm668(s):
 return re.sub(r"[^a-z]","",str(s).casefold())

def same668(a,b):
 return norm668(a)==norm668(b)

print("="*124)
print("TEST668 — BLIND EVIDENCE SENSITIVITY / CLAIM–MEMORY COUNTERFACTUAL")
print("POLICY: FROZEN MODEL / LOCKED CUT3 / NO ORACLE IN VERIFICATION PROMPTS")
print("="*124)

req668=urllib.request.Request(URL668,headers={"User-Agent":"AKBASCORE-TEST668"})
with urllib.request.urlopen(req668,timeout=120) as f:
 raw668=f.read()

actual668=hashlib.sha256(raw668).hexdigest()
print("SOURCE_SHA256",actual668)
gate668("TEST664_SOURCE_LOCK",actual668==SHA668)

src668=raw668.decode("utf-8")
bound668="# TEST664 — Reflective retry against budget-matched controls."
start668="T664=time.perf_counter();R664=[];ERR664=[]"
end668="# Important: text ceiling uses the exact same source-prefix"

gate668("SOURCE_BOUNDARIES",src668.count(bound668)==1 and src668.count(start668)==1 and src668.count(end668)==1)

engine668=src668.split(bound668,1)[0]
panel668=start668+src668.split(start668,1)[1].split(end668,1)[0]

exec(compile(engine668,"TEST664_ENGINE.py","exec"),globals())
exec(compile(panel668,"TEST664_PANEL.py","exec"),globals())

gate668("PANEL_LOCK",SHA663==PANEL_LOCK668)
gate668("PANEL_SIZE",len(PANEL663)==12)
gate668("FROZEN_WEIGHTS",all(not p.requires_grad for p in model.parameters()))
gate668("BASELINE_SENTINEL",sentinel()==S0)
gate668("MODEL_EVAL_MODE",not model.training)

print("="*124)
print("TEST668 — EXPERIMENTAL DESIGN")
print("CASES: 12")
print("REPRESENTATIONS: TEXT_JOINT / DC_CUT3")
print("MEMORY CONDITIONS: BASE / SWAP / MISSING")
print("CLAIMS: ORIGINAL REGION / COUNTERFACTUAL REGION")
print("VERIFICATION PROMPTS: STRICT THREE-WAY + NEUTRAL PARAPHRASE")
print("NEGATIVE CONTROL: NO-MEMORY VERIFICATION")
print("="*124)

T668=time.perf_counter()
R668=[]
ERR668=[]
REPS668=("TEXT_JOINT","DC_CUT3")
CONDS668=("base","swap","missing")
CLAIMS668=("ORIGINAL","COUNTERFACTUAL")
PROMPTS668=("STANDARD","PARAPHRASE")
LABELS668=("SUPPORTED","CONTRADICTED","INSUFFICIENT")

def claim668(z,kind):
 target=z["chains"][z["target"]]
 if kind=="ORIGINAL":
  region=target["region"]
 elif kind=="COUNTERFACTUAL":
  region=z["chains"][z["other"]]["region"]
 else:
  raise ValueError(kind)
 return (
  "The owner of artifact "+target["obj"]+
  " works at an organization located in a city belonging to region "+region+"."
 )

def expected668(condition,kind):
 if condition=="missing":
  return "INSUFFICIENT"
 if condition=="base":
  return "SUPPORTED" if kind=="ORIGINAL" else "CONTRADICTED"
 if condition=="swap":
  return "CONTRADICTED" if kind=="ORIGINAL" else "SUPPORTED"
 raise ValueError(condition)

def question668(claim,mode):
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

def parse668(raw):
 s=str(raw).strip().upper()
 s=re.sub(r"^\s*(?:ANSWER|LABEL|STATUS)\s*:\s*","",s)
 s=s.strip().strip(" .,:;!?\"'`")
 return s if s in LABELS668 else "UNPARSEABLE"

def evaluate668(mem,q):
 t=time.perf_counter()
 raw=str(answer(mem,q))
 return dict(
  raw=raw,
  label=parse668(raw),
  seconds=round(time.perf_counter()-t,4)
 )

def make_memory668(keys,rep):
 if rep=="TEXT_JOINT":
  return validate659(joint659(keys),keys)
 if rep=="DC_CUT3":
  return validate659(native_memory659(keys),keys)
 raise ValueError(rep)

# Diagnostic check: the same claim must have opposite truth status in BASE and SWAP,
# while MISSING must not determine either terminal region.
for z in PANEL663:
 gate668("DISTINCT_REGIONS_CASE_"+str(z["case"]),
  not same668(z["chains"][z["target"]]["region"],z["chains"][z["other"]]["region"]))
 for kind in CLAIMS668:
  gate668("TRUTH_REVERSAL_"+str(z["case"])+"_"+kind,
   expected668("base",kind)!=expected668("swap",kind)
   and expected668("missing",kind)=="INSUFFICIENT")

@torch.no_grad()
def run668(z,condition,rep):
 mem=None
 try:
  keys=z[condition]
  mem=make_memory668(keys,rep)
  for kind in CLAIMS668:
   claim=claim668(z,kind)
   gold=expected668(condition,kind)
   for mode in PROMPTS668:
    q=question668(claim,mode)
    out=evaluate668(mem,q)
    row=dict(
     case=z["case"],
     condition=condition,
     representation=rep,
     claim_kind=kind,
     claim=claim,
     prompt_mode=mode,
     question=q,
     expected=gold,
     raw=out["raw"],
     predicted=out["label"],
     correct=out["label"]==gold,
     seconds=out["seconds"]
    )
    R668.append(row)
    print(
     "CASE",z["case"],
     "COND",condition,
     "REP",rep,
     "CLAIM",kind,
     "MODE",mode,
     "EXPECTED",gold,
     "PRED",out["label"],
     "PASS",int(row["correct"]),
     flush=True
    )
 except Exception as e:
  err=dict(
   case=z["case"],
   condition=condition,
   representation=rep,
   error=repr(e),
   traceback=traceback.format_exc()
  )
  ERR668.append(err)
  print("ERROR",json.dumps(err,ensure_ascii=False),flush=True)
 finally:
  if mem is not None:del mem

for z in PANEL663:
 for rep in REPS668:
  for condition in CONDS668:
   run668(z,condition,rep)
 gc.collect()
 if torch.cuda.is_available():
  torch.cuda.empty_cache()

print("="*124)
print("TEST668 — NO-MEMORY NEGATIVE CONTROL")
print("="*124)

# An uninitialized memory cache is not equivalent to an empty cartridge.
# Use the locked model's ordinary text-generation path with the claim alone.
# This measures whether the verification prompt itself encourages a label,
# without presenting any experimental source facts.
NOMEM668=[]
NOMEM_ERRORS668=[]

def model_device668():
 return model.get_input_embeddings().weight.device

def prepare_input668(q):
 messages=[{"role":"user","content":q}]
 encoded=tok.apply_chat_template(
  messages,
  tokenize=True,
  add_generation_prompt=True,
  return_tensors="pt"
 )
 if isinstance(encoded,torch.Tensor):
  ids=encoded
  mask=torch.ones_like(ids,dtype=torch.long)
 elif hasattr(encoded,"get"):
  ids=encoded.get("input_ids")
  mask=encoded.get("attention_mask")
  if ids is None:
   raise RuntimeError("TOKENIZER_MISSING_INPUT_IDS")
  if mask is None:
   mask=torch.ones_like(ids,dtype=torch.long)
 else:
  raise TypeError("UNEXPECTED_TOKENIZER_OUTPUT_TYPE: "+str(type(encoded)))
 if not isinstance(ids,torch.Tensor):
  ids=torch.as_tensor(ids,dtype=torch.long)
 if not isinstance(mask,torch.Tensor):
  mask=torch.as_tensor(mask,dtype=torch.long)
 if ids.ndim==1:
  ids=ids.unsqueeze(0)
 if mask.ndim==1:
  mask=mask.unsqueeze(0)
 gate_shape=ids.ndim==2 and mask.shape==ids.shape and ids.shape[0]==1 and ids.shape[1]>0
 if not gate_shape:
  raise RuntimeError("INVALID_NO_MEMORY_INPUT_SHAPE")
 device=model_device668()
 return ids.to(device=device,dtype=torch.long),mask.to(device=device,dtype=torch.long)

@torch.no_grad()
def no_memory668(q):
 ids,mask=prepare_input668(q)
 t=time.perf_counter()
 out=model.generate(
  input_ids=ids,
  attention_mask=mask,
  max_new_tokens=16,
  do_sample=False,
  pad_token_id=tok.eos_token_id,
  use_cache=True
 )
 if hasattr(out,"sequences"):
  sequences=out.sequences
 else:
  sequences=out
 if not isinstance(sequences,torch.Tensor):
  raise TypeError("UNEXPECTED_GENERATION_OUTPUT_TYPE: "+str(type(sequences)))
 if sequences.ndim!=2 or sequences.shape[0]!=1:
  raise RuntimeError("INVALID_GENERATION_OUTPUT_SHAPE")
 if sequences.shape[-1]<ids.shape[-1]:
  raise RuntimeError("GENERATION_SHORTER_THAN_INPUT")
 generated=sequences[0,ids.shape[-1]:]
 raw=tok.decode(generated,skip_special_tokens=True).strip()
 return dict(raw=raw,label=parse668(raw),seconds=round(time.perf_counter()-t,4))

# Verify the no-memory input path before starting the complete negative control.
probe668=question668(
 "The owner of artifact PROBE668 works at an organization located in a city belonging to region PROBEREGION668.",
 "STANDARD"
)
try:
 probe_ids668,probe_mask668=prepare_input668(probe668)
 gate668("NOMEM_INPUT_TENSOR",isinstance(probe_ids668,torch.Tensor))
 gate668("NOMEM_INPUT_SHAPE",probe_ids668.ndim==2 and probe_ids668.shape[0]==1)
 gate668("NOMEM_ATTENTION_MASK",probe_mask668.shape==probe_ids668.shape)
 gate668("NOMEM_DEVICE",probe_ids668.device==model_device668())
 del probe_ids668,probe_mask668
except Exception:
 print("NOMEM_PRECHECK_ERROR",traceback.format_exc(),flush=True)
 raise

for z in PANEL663:
 for kind in CLAIMS668:
  for mode in PROMPTS668:
   claim=claim668(z,kind)
   q=question668(claim,mode)
   try:
    out=no_memory668(q)
    row=dict(
     case=z["case"],
     claim_kind=kind,
     prompt_mode=mode,
     raw=out["raw"],
     predicted=out["label"],
     expected="INSUFFICIENT",
     correct=out["label"]=="INSUFFICIENT",
     seconds=out["seconds"]
    )
    NOMEM668.append(row)
    print(
     "NOMEM",
     z["case"],
     kind,
     mode,
     out["label"],
     "PASS",int(row["correct"]),
     flush=True
    )
   except Exception as e:
    err=dict(
     case=z["case"],
     claim_kind=kind,
     prompt_mode=mode,
     error=repr(e),
     traceback=traceback.format_exc()
    )
    NOMEM_ERRORS668.append(err)
    print("NOMEM_ERROR",json.dumps(err,ensure_ascii=False),flush=True)

print("="*124)
print("TEST668 — CAUSAL EVIDENCE ANALYSIS")
print("="*124)

SUMMARY668={}
SENSITIVITY668={}
CONFUSION668={}
PARAPHRASE668={}
NOMEM_SUMMARY668={}

for rep in REPS668:
 for condition in CONDS668:
  rows=[r for r in R668 if r["representation"]==rep and r["condition"]==condition]
  key=rep+"_"+condition
  SUMMARY668[key]=dict(
   n=len(rows),
   correct=sum(r["correct"] for r in rows),
   supported=sum(r["predicted"]=="SUPPORTED" for r in rows),
   contradicted=sum(r["predicted"]=="CONTRADICTED" for r in rows),
   insufficient=sum(r["predicted"]=="INSUFFICIENT" for r in rows),
   unparseable=sum(r["predicted"]=="UNPARSEABLE" for r in rows)
  )
  CONFUSION668[key]=dict(Counter(
   r["expected"]+" -> "+r["predicted"] for r in rows
  ))

 for mode in PROMPTS668:
  paired=[]
  for z in PANEL663:
   for kind in CLAIMS668:
    rr={
     r["condition"]:r
     for r in R668
     if r["case"]==z["case"]
     and r["representation"]==rep
     and r["claim_kind"]==kind
     and r["prompt_mode"]==mode
    }
    if len(rr)==3 and all(c in rr for c in CONDS668):
     b,s,m=rr["base"],rr["swap"],rr["missing"]
     paired.append(dict(
      case=z["case"],
      claim_kind=kind,
      base=b["predicted"],
      swap=s["predicted"],
      missing=m["predicted"],
      base_correct=b["correct"],
      swap_correct=s["correct"],
      missing_correct=m["correct"],
      complete=bool(b["correct"] and s["correct"] and m["correct"]),
      reversed=bool(
       b["predicted"]!=s["predicted"]
       and b["predicted"] in ("SUPPORTED","CONTRADICTED")
       and s["predicted"] in ("SUPPORTED","CONTRADICTED")
      ),
      missing_abstained=m["predicted"]=="INSUFFICIENT"
     ))
  key=rep+"_"+mode
  SENSITIVITY668[key]=dict(
   n=len(paired),
   complete_triplets=sum(p["complete"] for p in paired),
   directional_reversals=sum(p["reversed"] for p in paired),
   missing_abstentions=sum(p["missing_abstained"] for p in paired),
   triplets=paired
  )

for rep in REPS668:
 pairs=[]
 for z in PANEL663:
  for condition in CONDS668:
   for kind in CLAIMS668:
    rr={
     r["prompt_mode"]:r
     for r in R668
     if r["case"]==z["case"]
     and r["representation"]==rep
     and r["condition"]==condition
     and r["claim_kind"]==kind
    }
    if len(rr)==2 and all(m in rr for m in PROMPTS668):
     pairs.append(dict(
      case=z["case"],
      condition=condition,
      claim_kind=kind,
      stable=rr["STANDARD"]["predicted"]==rr["PARAPHRASE"]["predicted"],
      both_correct=rr["STANDARD"]["correct"] and rr["PARAPHRASE"]["correct"],
      standard=rr["STANDARD"]["predicted"],
      paraphrase=rr["PARAPHRASE"]["predicted"]
     ))
 PARAPHRASE668[rep]=dict(
  n=len(pairs),
  stable=sum(p["stable"] for p in pairs),
  both_correct=sum(p["both_correct"] for p in pairs),
  pairs=pairs
 )

for mode in PROMPTS668:
 rows=[r for r in NOMEM668 if r["prompt_mode"]==mode]
 NOMEM_SUMMARY668[mode]=dict(
  n=len(rows),
  insufficient=sum(r["predicted"]=="INSUFFICIENT" for r in rows),
  supported=sum(r["predicted"]=="SUPPORTED" for r in rows),
  contradicted=sum(r["predicted"]=="CONTRADICTED" for r in rows),
  unparseable=sum(r["predicted"]=="UNPARSEABLE" for r in rows)
 )

# Compare the same claim under two different memory representations.
REP_PARITY668={}
for condition in CONDS668:
 pairs=[]
 for z in PANEL663:
  for kind in CLAIMS668:
   for mode in PROMPTS668:
    rr={
     r["representation"]:r
     for r in R668
     if r["case"]==z["case"]
     and r["condition"]==condition
     and r["claim_kind"]==kind
     and r["prompt_mode"]==mode
    }
    if len(rr)==2 and all(rep in rr for rep in REPS668):
     a=rr["TEXT_JOINT"]
     b=rr["DC_CUT3"]
     pairs.append(dict(
      case=z["case"],
      claim_kind=kind,
      prompt_mode=mode,
      same_prediction=a["predicted"]==b["predicted"],
      both_correct=a["correct"] and b["correct"],
      text=a["predicted"],
      dc=b["predicted"]
     ))
 REP_PARITY668[condition]=dict(
  n=len(pairs),
  same_prediction=sum(p["same_prediction"] for p in pairs),
  both_correct=sum(p["both_correct"] for p in pairs),
  pairs=pairs
 )

weights668=all(not p.requires_grad for p in model.parameters())
sentinel668=sentinel()==S0

# Structural completeness checks. These do not change any prediction.
expected_memory_keys668={
 (z["case"],rep,condition,kind,mode)
 for z in PANEL663
 for rep in REPS668
 for condition in CONDS668
 for kind in CLAIMS668
 for mode in PROMPTS668
}
actual_memory_keys668=[
 (r["case"],r["representation"],r["condition"],r["claim_kind"],r["prompt_mode"])
 for r in R668
]
expected_nomem_keys668={
 (z["case"],kind,mode)
 for z in PANEL663
 for kind in CLAIMS668
 for mode in PROMPTS668
}
actual_nomem_keys668=[
 (r["case"],r["claim_kind"],r["prompt_mode"])
 for r in NOMEM668
]
memory_complete668=(
 len(actual_memory_keys668)==len(expected_memory_keys668)
 and len(set(actual_memory_keys668))==len(actual_memory_keys668)
 and set(actual_memory_keys668)==expected_memory_keys668
)
nomem_complete668=(
 len(actual_nomem_keys668)==len(expected_nomem_keys668)
 and len(set(actual_nomem_keys668))==len(actual_nomem_keys668)
 and set(actual_nomem_keys668)==expected_nomem_keys668
)

report668=dict(
 test="668",
 title="Blind Evidence Sensitivity / Claim–Memory Counterfactual",
 source=URL668,
 source_sha256=actual668,
 panel_sha256=SHA663,
 rows=R668,
 summary=SUMMARY668,
 evidence_sensitivity=SENSITIVITY668,
 confusion=CONFUSION668,
 paraphrase_stability=PARAPHRASE668,
 representation_parity=REP_PARITY668,
 no_memory_rows=NOMEM668,
 no_memory_summary=NOMEM_SUMMARY668,
 errors=ERR668,
 no_memory_errors=NOMEM_ERRORS668,
 weights_frozen=weights668,
 sentinel_unchanged=sentinel668,
 memory_rows_complete=memory_complete668,
 no_memory_rows_complete=nomem_complete668,
 elapsed_seconds=round(time.perf_counter()-T668,2),
 limitations=[
  "The same development panel from TEST664 is reused; this is not an independent holdout.",
  "Claims are constructed from the panel by the experimenter, not autonomously generated by the model.",
  "The correct label is never supplied in the verification prompt.",
  "The claim itself necessarily contains the region being evaluated; this is not gold-label leakage.",
  "BASE, SWAP and MISSING use the same claim text but different memory states.",
  "A correct label is behavioral evidence sensitivity, not proof of internal epistemic awareness.",
  "A contradictory label is justified only when the accessible facts establish an incompatible alternative.",
  "The no-memory control uses ordinary model generation rather than a CUT3 memory cache.",
  "A no-memory prediction is a prompt-prior control, not a matched KV-cache ablation.",
  "Three-way labels are generated by the model without an external correction mechanism.",
  "All cartridge selection remains externally specified.",
  "No epistemic brake or autonomous memory verification mechanism has been installed."
 ]
)

path668=ROOT668/"TEST668_results.json"
path668.write_text(
 json.dumps(report668,ensure_ascii=False,indent=2),
 encoding="utf-8"
)

print("="*124)
print("TEST668_FINAL")
print("="*124)
print("SUMMARY",json.dumps(SUMMARY668,indent=2))
print("EVIDENCE_SENSITIVITY",json.dumps({
 k:{x:y for x,y in v.items() if x!="triplets"}
 for k,v in SENSITIVITY668.items()
},indent=2))
print("CONFUSION",json.dumps(CONFUSION668,indent=2))
print("PARAPHRASE_STABILITY",json.dumps({
 k:{x:y for x,y in v.items() if x!="pairs"}
 for k,v in PARAPHRASE668.items()
},indent=2))
print("REPRESENTATION_PARITY",json.dumps({
 k:{x:y for x,y in v.items() if x!="pairs"}
 for k,v in REP_PARITY668.items()
},indent=2))
print("NO_MEMORY_CONTROL",json.dumps(NOMEM_SUMMARY668,indent=2))
print("EXPECTED_MEMORY_ROWS",len(expected_memory_keys668))
print("ACTUAL_MEMORY_ROWS",len(R668))
print("EXPECTED_NOMEM_ROWS",len(expected_nomem_keys668))
print("ACTUAL_NOMEM_ROWS",len(NOMEM668))
print("MEMORY_ROWS_COMPLETE",memory_complete668)
print("NOMEM_ROWS_COMPLETE",nomem_complete668)
print("WEIGHTS_FROZEN",weights668)
print("SENTINEL_UNCHANGED",sentinel668)
print("ERRORS",len(ERR668))
print("NOMEM_ERRORS",len(NOMEM_ERRORS668))
print("RESULT",path668)
print("SHA256",hashlib.sha256(path668.read_bytes()).hexdigest())

success668=(
 memory_complete668
 and nomem_complete668
 and not ERR668
 and not NOMEM_ERRORS668
 and weights668
 and sentinel668
 and SHA663==PANEL_LOCK668
)

print("STATUS","EXPERIMENT_COMPLETE" if success668 else "ERRORS_PRESENT")
gate668("TEST668_COMPLETED",success668)

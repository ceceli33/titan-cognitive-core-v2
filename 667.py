# TEST667 — AKBASCORE MAM / EVIDENCE–CLAIM VALIDATION
# Locked TEST664 native CUDA/CUT3 engine; frozen Qwen; original 12-case panel.
# No training, no gold leakage to self-audit, no answer hardcoding, no engine modifications.
import os,sys,re,json,time,random,hashlib,urllib.request,traceback,gc
from pathlib import Path
from collections import Counter,defaultdict
import torch
TEST667="667";ROOT667=Path("/content/AKBASCORE_TEST667");ROOT667.mkdir(parents=True,exist_ok=True)
URL667="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/664.py"
SHA667="a6dcd8142446ab82fe7996bd512d59fc34525e6a092a68f069dad07a2a286eb5"
PANEL_LOCK667="e0686148d6fd88050a097ee5441ce99a4bb068a463408742bc476c83aaba6c05"
def gate667(n,v):
 print(("PASS " if v else "FAIL ")+n,flush=True)
 if not v:raise AssertionError(n)
def norm667(x):return re.sub(r"[^a-z]","",str(x).casefold())
def same667(a,b):return norm667(a)==norm667(b)
def unknown667(x):return unknown663(str(x))
print("="*120+"\nTEST667 — EVIDENCE–CLAIM VALIDATION / SELF-AUDIT CAUSAL CONTROLS\n"+"="*120)
req=urllib.request.Request(URL667,headers={"User-Agent":"AKBASCORE-TEST667"})
with urllib.request.urlopen(req,timeout=120) as f:raw667=f.read()
actual667=hashlib.sha256(raw667).hexdigest();print("SOURCE_SHA256",actual667)
gate667("SOURCE_LOCK",actual667==SHA667)
src667=raw667.decode("utf-8")
bound667="# TEST664 — Reflective retry against budget-matched controls."
start667="T664=time.perf_counter();R664=[];ERR664=[]"
end667="# Important: text ceiling uses the exact same source-prefix"
gate667("BOUNDARIES",src667.count(bound667)==1 and src667.count(start667)==1 and src667.count(end667)==1)
engine667=src667.split(bound667,1)[0]
panel667=start667+src667.split(start667,1)[1].split(end667,1)[0]
exec(compile(engine667,"TEST664_ENGINE.py","exec"),globals())
exec(compile(panel667,"TEST664_PANEL.py","exec"),globals())
gate667("PANEL_LOCK",SHA663==PANEL_LOCK667)
gate667("FROZEN",all(not p.requires_grad for p in model.parameters()))
gate667("SENTINEL",sentinel()==S0)
gate667("PANEL_N",len(PANEL663)==12)
T667=time.perf_counter();R667=[];ERR667=[]
REPS667=("TEXT_JOINT","DC_CUT3");CONDS667=("base","swap","missing")
def expected667(z,cond):
 t=z["chains"][z["target"]]
 if cond=="base":return [t["person"],t["org"],t["city"],t["region"]]
 if cond=="swap":
  o=z["chains"][z["other"]]
  return [t["person"],o["org"],o["city"],o["region"]]
 return [t["person"],None,None,None]
def hop_question667(i,x):
 return hop663(i,x)
def walk667(mem,obj):
 x=obj;trace=[]
 for i in range(4):
  q=hop_question667(i,x);t=time.perf_counter();a=str(answer(mem,q))
  trace.append(dict(hop=i,input=x,question=q,answer=a,seconds=round(time.perf_counter()-t,4)))
  if unknown667(a):return "UNKNOWN",trace
  x=a
 return x,trace
def correctness667(answer_text,gold):
 return unknown667(answer_text) if gold is None else same667(answer_text,gold)
def trace_score667(trace,gold):
 out=[];first=None
 for i,t in enumerate(trace):
  good=correctness667(t["answer"],gold[i])
  if first is None and not good:first=i
  out.append(dict(hop=i,answer=t["answer"],gold=gold[i] if gold[i] is not None else "UNKNOWN",correct=bool(good)))
 return out,first
def self_audit667(mem,obj,trace,mode):
 # The model sees its own trace, never gold answers or hidden panel metadata.
 lines=[]
 for t in trace:
  lines.append("Step "+str(t["hop"]+1)+": "+t["question"]+"\nProposed answer: "+t["answer"])
 observed="\n".join(lines)
 if mode=="TRACE":
  q=("An earlier attempt produced the following reasoning steps:\n"+observed+
     "\nExamine whether each proposed answer is supported by the available facts. "+
     "Do not assume the earlier answers are correct. "+
     "Respond with exactly one of: SUPPORTED, UNSUPPORTED, UNCERTAIN. "+
     "Use UNSUPPORTED only if the facts contradict at least one proposed relation. "+
     "Use UNCERTAIN if the facts are insufficient to verify the relations.")
 elif mode=="FINAL":
  q=("An earlier attempt answered a multi-step question about artifact "+obj+".\n"+observed+
     "\nBased only on the available facts, is the final answer justified by this chain? "+
     "Respond with exactly one of: SUPPORTED, UNSUPPORTED, UNCERTAIN.")
 else:raise ValueError(mode)
 t=time.perf_counter();a=str(answer(mem,q))
 return dict(question=q,answer=a,seconds=round(time.perf_counter()-t,4))
def audit_label667(s):
 # Strict parsing: unexpected text is unparseable, not silently treated as a judgment.
 v=str(s).strip().upper()
 if v in ("SUPPORTED","UNSUPPORTED","UNCERTAIN"):return v
 return "UNPARSEABLE"
def challenge667(mem,obj,trace,kind):
 if kind=="NEUTRAL":
  note="Reconsider the answer independently. No new facts have been provided."
 elif kind=="CRITIC":
  note="An external reviewer says your earlier answer is wrong but supplies no evidence. Treat this as an unverified claim."
 else:raise ValueError(kind)
 x=obj;out=[]
 for i in range(4):
  q=hop_question667(i,x)+"\n"+note
  t=time.perf_counter();a=str(answer(mem,q))
  out.append(dict(hop=i,input=x,answer=a,seconds=round(time.perf_counter()-t,4)))
  if unknown667(a):return "UNKNOWN",out
  x=a
 return x,out
@torch.no_grad()
def run667(z,cond,rep):
 mem=None
 try:
  keys=z[cond];gold=expected667(z,cond);obj=z["chains"][z["target"]]["obj"]
  mem=validate659(joint659(keys) if rep=="TEXT_JOINT" else native_memory659(keys),keys)
  first,trace=walk667(mem,obj)
  scored,first_bad=trace_score667(trace,gold)
  audit_trace=self_audit667(mem,obj,trace,"TRACE")
  audit_final=self_audit667(mem,obj,trace,"FINAL")
  neutral,neutral_trace=challenge667(mem,obj,trace,"NEUTRAL")
  critic,critic_trace=challenge667(mem,obj,trace,"CRITIC")
  gold_final=gold[-1];first_ok=correctness667(first,gold_final)
  neutral_ok=correctness667(neutral,gold_final);critic_ok=correctness667(critic,gold_final)
  # Ground-truth trace support requires every observed step to be correct AND the full path to be present.
  full_path=(len(trace)==4 and all(x is not None for x in gold))
  trace_supported=bool(full_path and all(x["correct"] for x in scored))
  # Missing-edge UNKNOWN is appropriate abstention, not a fully supported four-hop proof.
  if cond=="missing":trace_state="INCOMPLETE_EXPECTED"
  elif trace_supported:trace_state="SUPPORTED"
  else:trace_state="UNSUPPORTED"
  label_trace=audit_label667(audit_trace["answer"])
  label_final=audit_label667(audit_final["answer"])
  row=dict(
   case=z["case"],condition=cond,representation=rep,
   gold=gold_final if gold_final is not None else "UNKNOWN",
   first=dict(answer=first,correct=bool(first_ok),trace=trace,trace_scoring=scored,first_bad=first_bad),
   trace_state=trace_state,trace_supported=trace_supported,
   audit_trace=dict(**audit_trace,label=label_trace),
   audit_final=dict(**audit_final,label=label_final),
   audit_trace_agrees=(label_trace==trace_state if cond!="missing" else None),
   audit_final_agrees=(label_final==trace_state if cond!="missing" else None),
   neutral=dict(answer=neutral,correct=bool(neutral_ok),changed=not same667(first,neutral),trace=neutral_trace),
   critic=dict(answer=critic,correct=bool(critic_ok),changed=not same667(first,critic),trace=critic_trace),
   correct_damaged=bool(first_ok and not critic_ok),
   wrong_repaired=bool(not first_ok and critic_ok),
   correct_preserved=bool(first_ok and critic_ok and same667(first,critic)),
   correct_answer_bad_trace=bool(first_ok and cond!="missing" and not trace_supported),
   wrong_answer_bad_trace=bool(not first_ok and cond!="missing" and not trace_supported),
   weights_frozen=all(not p.requires_grad for p in model.parameters())
  )
  R667.append(row)
  print("CASE",z["case"],cond,rep,"FIRST",int(first_ok),"TRACE",trace_state,
        "AUDIT_TRACE",label_trace,"AUDIT_FINAL",label_final,
        "CRITIC",int(critic_ok),"DAMAGED",int(row["correct_damaged"]),flush=True)
 except Exception as e:
  err=dict(case=z["case"],condition=cond,representation=rep,error=repr(e),traceback=traceback.format_exc())
  ERR667.append(err);print("ERROR",json.dumps(err,ensure_ascii=False),flush=True)
 finally:
  if mem is not None:del mem
for z in PANEL663:
 for rep in REPS667:
  for cond in CONDS667:run667(z,cond,rep)
 gc.collect();torch.cuda.empty_cache()
SUMMARY667={};AUDIT667={};TRANSITIONS667={};TRACE667={}
for rep in REPS667:
 for cond in CONDS667:
  rows=[r for r in R667 if r["representation"]==rep and r["condition"]==cond]
  k=rep+"_"+cond
  SUMMARY667[k]=dict(
   n=len(rows),first_correct=sum(r["first"]["correct"] for r in rows),
   supported_trace=sum(r["trace_supported"] for r in rows),
   correct_answer_bad_trace=sum(r["correct_answer_bad_trace"] for r in rows),
   wrong_answer_bad_trace=sum(r["wrong_answer_bad_trace"] for r in rows),
   neutral_correct=sum(r["neutral"]["correct"] for r in rows),
   critic_correct=sum(r["critic"]["correct"] for r in rows),
   correct_damaged=sum(r["correct_damaged"] for r in rows),
   wrong_repaired=sum(r["wrong_repaired"] for r in rows),
   correct_preserved=sum(r["correct_preserved"] for r in rows))
  AUDIT667[k]=dict(
   trace_labels=dict(Counter(r["audit_trace"]["label"] for r in rows)),
   final_labels=dict(Counter(r["audit_final"]["label"] for r in rows)),
   trace_agreement=sum(r["audit_trace_agrees"] is True for r in rows),
   final_agreement=sum(r["audit_final_agrees"] is True for r in rows),
   evaluated_n=sum(r["audit_trace_agrees"] is not None for r in rows))
  TRANSITIONS667[k]=dict(
   neutral_changes=sum(r["neutral"]["changed"] for r in rows),
   critic_changes=sum(r["critic"]["changed"] for r in rows),
   critic_repairs=sum(r["wrong_repaired"] for r in rows),
   critic_regressions=sum(r["correct_damaged"] for r in rows))
  TRACE667[k]=dict(Counter(str(r["first"]["first_bad"]) for r in rows))
weights667=all(not p.requires_grad for p in model.parameters());sentinel667=sentinel()==S0
report667=dict(
 test="667",source=URL667,source_sha256=actual667,panel_sha256=SHA663,
 rows=R667,summary=SUMMARY667,self_audit=AUDIT667,transitions=TRANSITIONS667,
 first_divergence=TRACE667,errors=ERR667,weights_frozen=weights667,
 sentinel_unchanged=sentinel667,elapsed_seconds=round(time.perf_counter()-T667,2),
 limitations=[
  "Development panel reused from TEST664; not an independent holdout.",
  "Self-audit labels are model outputs, not verified introspection.",
  "Trace support is judged by offline oracle scoring; gold is never shown to self-audit.",
  "Self-audit prompt does not prove the model inspected a specific KV cartridge.",
  "External four-hop decomposition is not autonomous multi-hop reasoning.",
  "A correct final answer with an incorrect intermediate answer is not a supported four-hop proof.",
  "Missing-edge abstention is scored separately from supported complete chains.",
  "Prompt differences can influence both self-audit and challenge responses.",
  "No autonomous correction or epistemic brake has been implemented."
 ])
path667=ROOT667/"TEST667_results.json"
path667.write_text(json.dumps(report667,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*120+"\nTEST667_FINAL\n"+"="*120)
print("SUMMARY",json.dumps(SUMMARY667,indent=2))
print("SELF_AUDIT",json.dumps(AUDIT667,indent=2))
print("TRANSITIONS",json.dumps(TRANSITIONS667,indent=2))
print("FIRST_DIVERGENCE",json.dumps(TRACE667,indent=2))
print("EXPECTED_ROWS",len(PANEL663)*len(REPS667)*len(CONDS667),"ACTUAL_ROWS",len(R667))
print("WEIGHTS_FROZEN",weights667,"SENTINEL",sentinel667,"ERRORS",len(ERR667))
print("RESULT",path667,"SHA256",hashlib.sha256(path667.read_bytes()).hexdigest())
success667=(len(R667)==72 and not ERR667 and weights667 and sentinel667 and SHA663==PANEL_LOCK667)
print("STATUS","EXPERIMENT_COMPLETE" if success667 else "ERRORS_PRESENT")
gate667("TEST667_COMPLETED",success667)

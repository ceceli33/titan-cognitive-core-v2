# ====================================================================================================================
# TEST665 — AKBASCORE MAM · ERROR ORIGIN / DECISION STABILITY
# EXACT WORKING TEST664 BASELINE + DIAGNOSTIC EXTENSION
# FROZEN QWEN · NATIVE CUDA CUT3 · TEST566 LOCKED
# NO ENGINE CHANGES · NO ANSWER PATCHES · NO AUTONOMOUS REPAIR CLAIM
# ====================================================================================================================
import os,sys,re,json,time,random,hashlib,urllib.request,traceback,gc
from pathlib import Path
from collections import Counter

TEST665='665'
URL665='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/664.py'
ROOT665=Path('/content/AKBASCORE_TEST665')
ROOT665.mkdir(parents=True,exist_ok=True)

# Run the complete, previously working TEST664 code without editing or splitting it.
req665=urllib.request.Request(URL665,headers={'User-Agent':'AKBASCORE-TEST665'})
with urllib.request.urlopen(req665,timeout=120) as response:
 SOURCE665=response.read()

SOURCE_SHA665=hashlib.sha256(SOURCE665).hexdigest()
SOURCE_TEXT665=SOURCE665.decode('utf-8-sig')
compile(SOURCE_TEXT665,'TEST664_ORIGINAL.py','exec')

print('='*120)
print('TEST665 — ORIGINAL TEST664 BASELINE')
print('SOURCE:',URL665)
print('SOURCE_SHA256:',SOURCE_SHA665)
print('BASELINE_POLICY: EXECUTE COMPLETE ORIGINAL SOURCE UNCHANGED')
print('='*120,flush=True)

exec(compile(SOURCE_TEXT665,'TEST664_ORIGINAL.py','exec'),globals())

# ====================================================================================================================
# TEST665 — BASELINE INTEGRITY
# ====================================================================================================================

T665=time.perf_counter()
R665=[]
ERR665=[]
VARIANTS665=('ORIGINAL','NEUTRAL','REPHRASE')
REPS665=('TEXT_JOINT','DC_CUT3')
CONDS665=('base','swap','missing')

gate659('TEST664_COMPLETED',len(R664)==N663*3*2*5 and not ERR664)
gate659('TEST664_FROZEN',all(not p.requires_grad for p in model.parameters()))
gate659('TEST664_SENTINEL',sentinel()==S0)

EXPECTED_PANEL665='e0686148d6fd88050a097ee5441ce99a4bb068a463408742bc476c83aaba6c05'
gate659('TEST665_PANEL_LOCK',SHA663==EXPECTED_PANEL665)

print('='*120)
print('TEST665 — ERROR ORIGIN / DECISION STABILITY')
print('ENGINE: ORIGINAL TEST664 UNCHANGED')
print('PANEL_SHA:',SHA663)
print('CASES:',N663)
print('REPRESENTATIONS:',REPS665)
print('CONDITIONS:',CONDS665)
print('VARIANTS:',VARIANTS665)
print('='*120,flush=True)

# ====================================================================================================================
# QUESTION VARIANTS
# ORIGINAL: Identical TEST664 stepwise questions.
# NEUTRAL: Same question plus content-free instruction.
# REPHRASE: Same requested relation with different wording.
# ====================================================================================================================

def question665(hop,x,variant):
 if variant=='ORIGINAL':
  return hop663(hop,x)
 if variant=='NEUTRAL':
  return hop663(hop,x)+'\nPlease consider the available information carefully.'
 if variant=='REPHRASE':
  return [
   f'Identify the person who owns artifact {x}. Give only the person name, or UNKNOWN if undetermined.',
   f'Identify the organization where {x} works. Give only the organization name, or UNKNOWN if undetermined.',
   f'Identify the city where organization {x} is located. Give only the city name, or UNKNOWN if undetermined.',
   f'Identify the region to which city {x} belongs. Give only the region name, or UNKNOWN if undetermined.'
  ][hop]
 raise ValueError('UNKNOWN_VARIANT_'+str(variant))

def expected665(z,condition):
 target=z['chains'][z['target']]
 other=z['chains'][z['other']]
 if condition=='base':
  return [target['person'],target['org'],target['city'],target['region']]
 if condition=='swap':
  return [target['person'],other['org'],other['city'],other['region']]
 if condition=='missing':
  return [target['person'],None,None,None]
 raise ValueError(condition)

def classify665(a,gold):
 if unknown663(a):
  return 'UNKNOWN'
 if gold is None:
  return 'UNSUPPORTED'
 if exact659(a,gold):
  return 'CORRECT'
 return 'WRONG'

@torch.no_grad()
def run_hops665(mem,obj,variant):
 x=obj
 trace=[]
 for hop in range(4):
  q=question665(hop,x,variant)
  t=time.perf_counter()
  y=answer(mem,q)
  elapsed=time.perf_counter()-t
  trace.append(dict(
   hop=hop,
   input=x,
   query=q,
   answer=y,
   unknown=bool(unknown663(y)),
   seconds=round(elapsed,5)
  ))
  if unknown663(y):
   return 'UNKNOWN',trace
  x=y
 return x,trace

def evaluate665(z,condition,rep,variant,mem):
 obj=z['chains'][z['target']]['obj']
 expected=expected665(z,condition)
 t=time.perf_counter()
 final,trace=run_hops665(mem,obj,variant)
 elapsed=time.perf_counter()-t

 first_error=None
 upstream_correct=True

 for i,item in enumerate(trace):
  gold=expected[i]
  item['gold']=gold
  item['classification']=classify665(item['answer'],gold)
  item['exact_correct']=bool(gold is not None and exact659(item['answer'],gold))
  item['upstream_correct']=upstream_correct

  # The first observed divergence is meaningful only while all earlier
  # generated links still match the expected chain.
  if first_error is None and not item['exact_correct']:
   first_error=i

  if not item['exact_correct']:
   upstream_correct=False

 gold_final=(
  z['original'] if condition=='base'
  else z['changed'] if condition=='swap'
  else 'UNKNOWN'
 )

 correct=(
  unknown663(final) if condition=='missing'
  else exact659(final,gold_final)
 )

 return dict(
  case=z['case'],
  condition=condition,
  representation=rep,
  variant=variant,
  answer=final,
  gold=gold_final,
  correct=bool(correct),
  unknown=bool(unknown663(final)),
  first_observed_divergence=first_error,
  completed_hops=len(trace),
  trace=trace,
  seconds=round(elapsed,5)
 )

# ====================================================================================================================
# RUN — REUSE THE ORIGINAL TEST664 MEMORY FUNCTIONS
# ====================================================================================================================

for z in PANEL663:
 for rep in REPS665:
  for condition in CONDS665:
   keys=z[condition]
   try:
    t=time.perf_counter()
    mem=validate659(
     joint659(keys) if rep=='TEXT_JOINT'
     else native_memory659(keys),
     keys
    )
    memory_seconds=time.perf_counter()-t

    for variant in VARIANTS665:
     row=evaluate665(z,condition,rep,variant,mem)
     row['memory_seconds']=round(memory_seconds,5)
     R665.append(row)

     print(
      'CASE',z['case'],
      'COND',condition,
      'REP',rep,
      'VAR',variant,
      'PASS',int(row['correct']),
      'FIRST_DIVERGENCE',row['first_observed_divergence'],
      'ANSWER',repr(row['answer']),
      'GOLD',repr(row['gold']),
      flush=True
     )

    del mem

   except Exception as e:
    ERR665.append(dict(
     case=z['case'],
     condition=condition,
     representation=rep,
     error=f'{type(e).__name__}: {e}',
     traceback=traceback.format_exc()
    ))
    print('ERROR',z['case'],condition,rep,repr(e),flush=True)

 gc.collect()
 torch.cuda.empty_cache()

# ====================================================================================================================
# TEST665 ANALYSIS
# ====================================================================================================================

SUMMARY665={}
ORIGIN665={}
STABILITY665={}
TRANSITIONS665={}
SWITCH665={}
BASELINE_COMPARE665={}

for rep in REPS665:
 for condition in CONDS665:

  for variant in VARIANTS665:
   rows=[
    r for r in R665
    if (r['representation'],r['condition'],r['variant'])
    ==(rep,condition,variant)
   ]

   key=f'{rep}_{condition}_{variant}'

   SUMMARY665[key]=dict(
    n=len(rows),
    correct=sum(r['correct'] for r in rows),
    unknown=sum(r['unknown'] for r in rows),
    mean_seconds=round(
     sum(r['seconds'] for r in rows)/max(1,len(rows)),4
    )
   )

   ORIGIN665[key]=dict(
    n=len(rows),
    first_hop_0=sum(r['first_observed_divergence']==0 for r in rows),
    first_hop_1=sum(r['first_observed_divergence']==1 for r in rows),
    first_hop_2=sum(r['first_observed_divergence']==2 for r in rows),
    first_hop_3=sum(r['first_observed_divergence']==3 for r in rows),
    no_observed_divergence=sum(
     r['first_observed_divergence'] is None for r in rows
    )
   )

  original={
   r['case']:r for r in R665
   if (r['representation'],r['condition'],r['variant'])
   ==(rep,condition,'ORIGINAL')
  }

  baseline={
   r['case']:r for r in R664
   if (r['representation'],r['condition'],r['arm'])
   ==(rep,condition,'STEPWISE')
  }

  common=sorted(set(original)&set(baseline))

  BASELINE_COMPARE665[f'{rep}_{condition}']=dict(
   n=len(common),
   exact_answer_matches=sum(
    exact659(original[i]['answer'],baseline[i]['answer'])
    for i in common
   ),
   correctness_matches=sum(
    original[i]['correct']==baseline[i]['correct']
    for i in common
   ),
   differing_cases=[
    i for i in common
    if not exact659(original[i]['answer'],baseline[i]['answer'])
   ]
  )

  for variant in ('NEUTRAL','REPHRASE'):
   alternative={
    r['case']:r for r in R665
    if (r['representation'],r['condition'],r['variant'])
    ==(rep,condition,variant)
   }

   ids=sorted(set(original)&set(alternative))
   changes=0
   repairs=0
   regressions=0
   regressions_unknown=0
   regressions_wrong=0
   hop_changes=Counter()

   for i in ids:
    a=original[i]
    b=alternative[i]

    if not exact659(a['answer'],b['answer']):
     changes+=1

    if not a['correct'] and b['correct']:
     repairs+=1

    if a['correct'] and not b['correct']:
     regressions+=1
     if b['unknown']:
      regressions_unknown+=1
     else:
      regressions_wrong+=1

    for hop in range(min(len(a['trace']),len(b['trace']))):
     if not exact659(
      a['trace'][hop]['answer'],
      b['trace'][hop]['answer']
     ):
      hop_changes[hop]+=1

   key=f'{rep}_{condition}_{variant}'

   STABILITY665[key]=dict(
    n=len(ids),
    final_answer_changes=changes,
    hop_answer_changes=dict(hop_changes)
   )

   TRANSITIONS665[key]=dict(
    repairs=repairs,
    regressions=regressions,
    regressions_to_unknown=regressions_unknown,
    regressions_to_wrong=regressions_wrong
   )

 for variant in VARIANTS665:
  base={
   r['case']:r for r in R665
   if (r['representation'],r['condition'],r['variant'])
   ==(rep,'base',variant)
  }

  swap={
   r['case']:r for r in R665
   if (r['representation'],r['condition'],r['variant'])
   ==(rep,'swap',variant)
  }

  ids=sorted(set(base)&set(swap))

  SWITCH665[f'{rep}_{variant}']=dict(
   n=len(ids),
   causal_switch=sum(
    base[i]['correct']
    and swap[i]['correct']
    and not exact659(base[i]['answer'],swap[i]['answer'])
    for i in ids
   )
  )

# ====================================================================================================================
# FINAL AUDIT
# ====================================================================================================================

FROZEN665=all(not p.requires_grad for p in model.parameters())
SENTINEL665=sentinel()==S0
EXPECTED_ROWS665=N663*len(REPS665)*len(CONDS665)*len(VARIANTS665)
COMPLETE665=len(R665)==EXPECTED_ROWS665

ORIGINAL_REPLAY665=all(
 v['n']==N663 and v['exact_answer_matches']==N663
 for v in BASELINE_COMPARE665.values()
)

STATUS665=(
 'EXPERIMENT_COMPLETE'
 if COMPLETE665 and FROZEN665 and SENTINEL665
 and ORIGINAL_REPLAY665 and not ERR665
 else 'ERRORS_PRESENT'
)

REPORT665=dict(
 test='665',
 title='ERROR ORIGIN / DECISION STABILITY',
 baseline='TEST664 COMPLETE ORIGINAL SOURCE',
 source_url=URL665,
 source_sha256=SOURCE_SHA665,
 base_blob=BLOB659,
 panel_sha=SHA663,
 panel=PANEL663,
 rows=R665,
 summary=SUMMARY665,
 error_origin=ORIGIN665,
 prompt_stability=STABILITY665,
 transitions=TRANSITIONS665,
 causal_switch=SWITCH665,
 baseline_comparison=BASELINE_COMPARE665,
 errors=ERR665,
 weights_frozen=FROZEN665,
 sentinel_unchanged=SENTINEL665,
 original_replay_matches=ORIGINAL_REPLAY665,
 expected_rows=EXPECTED_ROWS665,
 actual_rows=len(R665),
 elapsed_seconds=round(time.perf_counter()-T665,2),
 limitations=[
  'The entire original TEST664 source is executed unchanged.',
  'TEST665 reuses the same development panel; it is not an independent holdout.',
  'Gold is used only in offline evaluation, never in model queries.',
  'Question variants are diagnostic, not repair mechanisms.',
  'First observed divergence does not prove the internal cause of an error.',
  'After an incorrect intermediate answer, downstream queries follow that generated answer.',
  'Missing-edge gold is undefined beyond the first missing relation.',
  'Exact-match scoring may penalize valid answers with additional formatting.',
  'No autonomous reasoning, self-correction, or learning capability is established.'
 ]
)

RESULT665=ROOT665/'TEST665_results.json'
RESULT665.write_text(
 json.dumps(REPORT665,ensure_ascii=False,indent=2),
 encoding='utf-8'
)

print('='*120)
print('TEST665_FINAL')
print('='*120)
print('SUMMARY',json.dumps(SUMMARY665,indent=2))
print('ERROR_ORIGIN',json.dumps(ORIGIN665,indent=2))
print('PROMPT_STABILITY',json.dumps(STABILITY665,indent=2))
print('TRANSITIONS',json.dumps(TRANSITIONS665,indent=2))
print('CAUSAL_SWITCH',json.dumps(SWITCH665,indent=2))
print('TEST664_REPLAY',json.dumps(BASELINE_COMPARE665,indent=2))
print('EXPECTED_ROWS',EXPECTED_ROWS665)
print('ACTUAL_ROWS',len(R665))
print('ORIGINAL_REPLAY_MATCHES',ORIGINAL_REPLAY665)
print('WEIGHTS_FROZEN',FROZEN665)
print('SENTINEL_UNCHANGED',SENTINEL665)
print('ERRORS',len(ERR665))
print('RESULT',RESULT665)
print('SHA256',hashlib.sha256(RESULT665.read_bytes()).hexdigest())
print('STATUS',STATUS665)
print('='*120)

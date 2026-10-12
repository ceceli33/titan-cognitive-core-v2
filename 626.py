# TEST626 — 1453 CUT3 / NATIVE QUERY-CONTRAST RERANK / SEALED VALIDATION
# Single Colab cell; exact TEST624 backbone, CUT3 and model unchanged. No gold/answer in runtime routing.
import os,time,json,hashlib,urllib.request,random,math
from pathlib import Path
import numpy as np,torch
T626=time.perf_counter();ROOT626=Path('/content/AKBASCORE_TEST624');OUT626=ROOT626/'TEST626_RESULT.json'
LOCAL626=Path('/content/TEST624_1453_PHYSICAL_CUT3.py');URL626='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/624.py'
def log626(*x):print(*x,flush=True)
def median626(x):return round(float(np.median(x)),4) if x else None
def sha626(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
NEED626=('K','QUEST','PANEL624','INDEX','CENTER','BASIS','OWNER','MANIFEST','POOL','CAR','model','WB','SB','src','LOCK','TOTAL','top2','native','answer','hit','DEVICE','weight_sha','sentinel','question_rep','apply','FIT','tok')
if not all(x in globals() for x in NEED626):
 if LOCAL626.exists():b626=LOCAL626.read_bytes()
 else:b626=urllib.request.urlopen(urllib.request.Request(URL626,headers={'User-Agent':'TEST626'}),timeout=120).read()
 boundary626="log('[4/7] FIXED IVF SELECTIVITY SWEEP"
 assert b626.decode().count(boundary626)==1,'TEST624 BACKBONE BOUNDARY ERROR'
 log626('BOOTSTRAP_TEST624_SHA256',hashlib.sha256(b626).hexdigest())
 exec(compile(b626.decode().split(boundary626,1)[0],'TEST624.py','exec'),globals())
assert TOTAL==1453 and len(K)==1453 and len(QUEST)==48 and len(MANIFEST)==1453
assert not any(p.requires_grad for p in model.parameters())
log626('='*108);log626('TEST626 — FROZEN 1453 CUT3 / LABEL-FREE NATIVE QUERY-CONTRAST RERANK');log626('='*108)
# Original locked narrow candidate path, unmodified. Cache score matrices for each candidate.
NP626,TK626,BUDGET626=2,16,16;BETA626=8.
def candidate626(q):
 t=time.perf_counter();qp=np.ascontiguousarray(((q-CENTER)@BASIS).numpy(),dtype=np.float32)
 INDEX.nprobe=NP626;_,cells=INDEX.quantizer.search(qp,NP626);_,ids=INDEX.search(qp,TK626)
 touched=sorted({int(OWNER[j]) for row in ids for j in row if j>=0});faiss_ms=(time.perf_counter()-t)*1000
 t=time.perf_counter();ts={ci:top2(q,K[ci]) for ci in touched}
 ordered=sorted(touched,key=lambda ci:(-ts[ci],ci));candidates=ordered[:BUDGET626]
 top2_ms=(time.perf_counter()-t)*1000
 return candidates,dict(touched=len(touched),faiss_ms=faiss_ms,top2_ms=top2_ms)
# Fixed, predeclared feature family: original FRAM, question-token maximum, token-wise peak,
# and contrast against same candidate set. No labels or cartridge IDs enter feature calculation.
# IMPORTANT: candidate retrieval is identical for all arms; only last ranking differs.
CFG626=[('FRAM',0.,0.),('PEAK25',.25,0.),('PEAK50',.5,0.),('PEAK75',.75,0.),('PEAK100',1.,0.),('CONTRAST25',.25,.25),('CONTRAST50',.5,.5),('CONTRAST75',.75,.75)]
def features626(q,ids):
 sims={ci:(q@K[ci].T).double() for ci in ids}
 native_scores={ci:float((torch.logsumexp(s*BETA626,dim=1)-math.log(s.shape[1])).mean().item()/BETA626) for ci,s in sims.items()}
 # Per-query-token max native similarity, no text processing and no answer knowledge.
 peak_scores={ci:float(s.max(dim=1).values.mean().item()) for ci,s in sims.items()}
 # Candidate-relative per-token advantage: measures how often a cartridge uniquely explains a query token.
 contrast={}
 for ci,s in sims.items():
  per_token=s.max(dim=1).values
  others=[v.max(dim=1).values for j,v in sims.items() if j!=ci]
  baseline=torch.stack(others).max(dim=0).values if others else torch.zeros_like(per_token)
  contrast[ci]=float((per_token-baseline).mean().item())
 return native_scores,peak_scores,contrast
def rerank626(ids,fs,cfg):
 name,a,b=cfg;n,p,c=fs
 scores={ci:(1-a)*n[ci]+a*p[ci]+b*c[ci] for ci in ids}
 return sorted(ids,key=lambda ci:(-scores[ci],ci)),scores
# Calibrate only on first 24 cases. Do not tune on validation cases.
log626('[1/6] PREDECLARED ORIGINAL QUERY PANEL / DIAG 24 + VALID 24')
DIAG626=[];VALID626=[]
for i,(q,case) in enumerate(zip(QUEST,PANEL624)):
 t=time.perf_counter();ids,routeinfo=candidate626(q);fs=features626(q,ids);feature_ms=(time.perf_counter()-t)*1000-routeinfo['faiss_ms']-routeinfo['top2_ms']
 record=dict(case=i,split='DIAG' if i<24 else 'VALID',target=case['target'],candidates=ids,route=routeinfo,feature_ms=feature_ms,
  predictions={cfg[0]:rerank626(ids,fs,cfg)[0][0] if ids else None for cfg in CFG626})
 (DIAG626 if i<24 else VALID626).append(record)
 if (i+1)%12==0:log626('RERANK_FEATURES',i+1,'/48')
CAL626=[dict(name=cfg[0],diag_correct=sum(r['predictions'][cfg[0]]==r['target'] for r in DIAG626),diag_total=len(DIAG626)) for cfg in CFG626]
# Predeclared tie-break: original FRAM first; smallest candidate list order of configurations.
BEST626=max(CAL626,key=lambda r:(r['diag_correct'],-next(i for i,c in enumerate(CFG626) if c[0]==r['name'])))['name']
log626('DIAG_CALIBRATION',json.dumps(CAL626));log626('LOCKED_RERANK',BEST626)
VALID_SUM626={cfg[0]:dict(correct=sum(r['predictions'][cfg[0]]==r['target'] for r in VALID626),total=len(VALID626)) for cfg in CFG626}
log626('SEALED_VALIDATION',json.dumps(VALID_SUM626))
# New unseen query wording, fixed before evaluation. Not used to select a ranking configuration.
# Same synthetic entities; this is a wording generalization check, not independent-domain holdout.
log626('[2/6] SECONDARY NEW-WORDING PANEL / 48 CASES / NO RECALIBRATION')
ALT626=[]
for i,case in enumerate(PANEL624):
 country=CAR[case['target']]['fact'].split('The current capital of ',1)[1].split(' is ',1)[0]
 question=(f'Name the present capital of {country}.' if i%2==0 else f'Which city currently serves as the capital of {country}?')
 rep,_=question_rep(question);q=apply(rep[3],FIT).detach().float().cpu().contiguous()
 t=time.perf_counter();ids,info=candidate626(q);fs=features626(q,ids);feature_ms=(time.perf_counter()-t)*1000-info['faiss_ms']-info['top2_ms']
 pred={cfg[0]:rerank626(ids,fs,cfg)[0][0] if ids else None for cfg in CFG626}
 ALT626.append(dict(case=i,question=question,target=case['target'],predictions=pred,candidates=ids,route=info,feature_ms=feature_ms))
 if (i+1)%12==0:log626('ALTERNATE_QUERY',i+1,'/48')
ALT_SUM626={cfg[0]:dict(correct=sum(r['predictions'][cfg[0]]==r['target'] for r in ALT626),total=len(ALT626),candidate_hit=sum(r['target'] in r['candidates'] for r in ALT626)) for cfg in CFG626}
log626('ALTERNATE_VALIDATION',json.dumps(ALT_SUM626))
# Compare locked reranker with original FRAM on both panels. Never select per-question with gold.
log626('[3/6] FIXED RERANKER PHYSICAL DISK ANSWERS / 48 ORIGINAL + 48 ALTERNATE')
E2E626=[]
for panel,rows in [('ORIGINAL',DIAG626+VALID626),('ALTERNATE',ALT626)]:
 for r in rows:
  ci=r['predictions'][BEST626];case=PANEL624[r['case']];question=case['question'] if panel=='ORIGINAL' else r['question']
  out=dict(panel=panel,case=r['case'],target=r['target'],selected=ci,selected_correct=ci==r['target'])
  try:
   if ci is None:raise RuntimeError('NO_CANDIDATE')
   t=time.perf_counter();p=POOL/f'cut3_{ci:05d}.pt';assert sha626(p)==MANIFEST[ci]['sha256']
   stored=torch.load(p,map_location='cpu',weights_only=True);mem=tuple((a.to(DEVICE),b.to(DEVICE)) for a,b in stored)
   out['load_ms']=(time.perf_counter()-t)*1000;t=time.perf_counter()
   with torch.no_grad():ans=answer(mem,question)
   out.update(answer=str(ans),correct=bool(hit(ans,case['gold'])),answer_ms=(time.perf_counter()-t)*1000)
   del stored,mem
  except Exception as e:out.update(correct=False,error=repr(e))
  E2E626.append(out)
  if len(E2E626)%16==0:log626('PHYSICAL_E2E',len(E2E626),'/96','correct',sum(x['correct'] for x in E2E626))
# Diagnostic oracle control: only on selected misses; never counted as autonomous success.
log626('[4/6] FIXED MODEL / DISK / SHA INTEGRITY')
MANIFEST_OK626=all((POOL/x['file']).exists() and (POOL/x['file']).stat().st_size==x['bytes'] and sha626(POOL/x['file'])==x['sha256'] for x in MANIFEST)
HOOKS626=any(m._forward_hooks or m._forward_pre_hooks for layer in model.model.layers for m in (layer,layer.self_attn,layer.self_attn.q_proj))
INTEGRITY626=dict(source_sha_lock=hashlib.sha256(src).hexdigest()==LOCK,manifest_1453_sha_verified=MANIFEST_OK626,
 weight_unchanged=weight_sha()==WB,sentinel_unchanged=sentinel()==SB,zero_trainable=not any(p.requires_grad for p in model.parameters()),no_hooks=not HOOKS626)
SUM_E2E626={panel:dict(cases=len(rows),selected_correct=sum(x['selected_correct'] for x in rows),answer_correct=sum(x['correct'] for x in rows),
 errors=sum('error' in x for x in rows),load_p50_ms=median626([x['load_ms'] for x in rows if 'load_ms' in x]),
 answer_p50_ms=median626([x['answer_ms'] for x in rows if 'answer_ms' in x])) for panel,rows in [('ORIGINAL',E2E626[:48]),('ALTERNATE',E2E626[48:])]}
RESULT626=dict(test='626',status='COMPLETE' if all(INTEGRITY626.values()) else 'INTEGRITY_FAILURE',
 model='Qwen/Qwen2.5-7B-Instruct',cartridges=TOTAL,architecture='Original CUT3 / native L3 / PCA128 IVF128 / TOP2_SOFT candidates / native rerank only',
 locked_candidate_setting=dict(nprobe=NP626,token_k=TK626,budget=BUDGET626),calibration=CAL626,locked_reranker=BEST626,
 original_validation=VALID_SUM626,alternate_validation=ALT_SUM626,physical_e2e=SUM_E2E626,
 per_query_original=DIAG626+VALID626,per_query_alternate=ALT626,physical_answers=E2E626,integrity=INTEGRITY626,
 elapsed_seconds=round(time.perf_counter()-T626,2),limitations=[
 'DIAG/VALID original questions were previously exposed in TEST624/625; not blind.',
 'Alternate wording uses the same 48 entities and facts; not an independent record holdout.',
 'The new reranking feature family is experimental and not a validated architectural improvement.',
 'A post-hoc 48/48 on known questions is not evidence of independent generalization.',
 'Physical reads may be page-cache served; cold SSD latency not certified.',
 'No evidence of million-cartridge scale from this test.'])
ROOT626.mkdir(parents=True,exist_ok=True)
with open(OUT626,'w',encoding='utf-8') as f:json.dump(RESULT626,f,ensure_ascii=False,indent=2,default=str)
log626('[5/6] FINAL SUMMARY');log626('TEST626_LOCKED_RERANK',BEST626);log626('TEST626_VALIDATION',json.dumps(VALID_SUM626));log626('TEST626_ALT',json.dumps(ALT_SUM626))
log626('TEST626_E2E',json.dumps(SUM_E2E626));log626('TEST626_INTEGRITY',json.dumps(INTEGRITY626))
log626('[6/6] DEMO ACCEPTANCE')
ready626=all(INTEGRITY626.values()) and SUM_E2E626['ORIGINAL']['answer_correct']==48 and SUM_E2E626['ALTERNATE']['answer_correct']==48
log626('TEST626_DECISION','READY_FOR_INDEPENDENT_HOLDOUT' if ready626 else 'REPORT_ACTUAL_ACCURACY_AND_CONTINUE_DIAGNOSIS')
log626('TEST626_OUTPUT',str(OUT626));log626('TEST626_STATUS',RESULT626['status'])
if not all(INTEGRITY626.values()):raise RuntimeError('TEST626_INTEGRITY_FAILURE')

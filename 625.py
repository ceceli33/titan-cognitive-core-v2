# TEST625 — FIXED 1453 CUT3 DIAGNOSIS / ORIGINAL TEST624 BACKBONE
import os,json,time,hashlib,urllib.request,math
from pathlib import Path
import numpy as np,torch
T625=time.perf_counter();ROOT625=Path('/content/AKBASCORE_TEST624');OUT625=ROOT625/'TEST625_RESULT.json'
LOCAL625=Path('/content/TEST624_1453_PHYSICAL_CUT3.py')
URL625='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/624.py'
def log625(*x):print(*x,flush=True)
def sha625(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def median625(a):return round(float(np.median(a)),4) if a else None
NEED625=['K','QUEST','PANEL624','INDEX','CENTER','BASIS','OWNER','LENS','LISTSIZE','MANIFEST','POOL','CAR','model','WB','SB','src','LOCK','TOTAL','top2','native','answer','hit','DEVICE']
if not all(x in globals() for x in NEED625):
 if LOCAL625.exists():b625=LOCAL625.read_bytes()
 else:b625=urllib.request.urlopen(urllib.request.Request(URL625,headers={'User-Agent':'TEST625'}),timeout=120).read()
 boundary625="log('[4/7] FIXED IVF SELECTIVITY SWEEP"
 assert b625.decode().count(boundary625)==1,'TEST624 SOURCE BOUNDARY ERROR'
 log625('BOOTSTRAP_TEST624_SHA256',hashlib.sha256(b625).hexdigest())
 exec(compile(b625.decode().split(boundary625,1)[0],'TEST624.py','exec'),globals())
assert TOTAL==1453 and len(K)==1453 and len(QUEST)==48 and len(MANIFEST)==1453
log625('='*100);log625('TEST625 / 1453 REAL CUT3 / NATIVE GEOMETRY DIAGNOSIS');log625('='*100)
ARMS625=[('NARROW',2,16,16),('MEDIUM',8,64,16),('WIDE',32,128,32)]
# Full native scan is an isolated scientific reference, never used to choose a cartridge.
FULL625=[];TOPFULL625=[]
for i,q in enumerate(QUEST):
 ns=[native(q,k) for k in K];ts=[top2(q,k) for k in K]
 FULL625.append(sorted(range(TOTAL),key=lambda j:(-ns[j],j)))
 TOPFULL625.append(sorted(range(TOTAL),key=lambda j:(-ts[j],j)))
 if (i+1)%8==0:log625('FULL_NATIVE_REFERENCE',i+1,'/48')
DETAIL625={};SUM625={}
for name,npb,tk,budget in ARMS625:
 rows=[]
 for i,(q,c) in enumerate(zip(QUEST,PANEL624)):
  t=time.perf_counter();qp=np.ascontiguousarray(((q-CENTER)@BASIS).numpy(),dtype=np.float32)
  INDEX.nprobe=npb;_,cells=INDEX.quantizer.search(qp,npb);_,ids=INDEX.search(qp,tk)
  touched=sorted({int(OWNER[j]) for a in ids for j in a if j>=0});faiss_ms=(time.perf_counter()-t)*1000
  t=time.perf_counter();ts={j:top2(q,K[j]) for j in touched};ordered=sorted(touched,key=lambda j:(-ts[j],j));candidates=ordered[:budget];top2_ms=(time.perf_counter()-t)*1000
  t=time.perf_counter();ns={j:native(q,K[j]) for j in candidates};ranked=sorted(candidates,key=lambda j:(-ns[j],j));fram_ms=(time.perf_counter()-t)*1000
  gold=c['target'];chosen=ranked[0] if ranked else None
  rows.append(dict(case=i,split='DIAG' if i<24 else 'VALID',target=gold,selected=chosen,correct=chosen==gold,
   touched=len(touched),gold_touched=gold in touched,gold_candidate=gold in candidates,
   gold_top2_rank=ordered.index(gold)+1 if gold in touched else None,
   gold_fram_rank=ranked.index(gold)+1 if gold in candidates else None,
   full_native_rank=FULL625[i].index(gold)+1,full_top2_rank=TOPFULL625[i].index(gold)+1,
   full_native_top1=FULL625[i][0],full_top2_top1=TOPFULL625[i][0],top5=ranked[:5],
   gold_native_score=ns.get(gold),winner_native_score=ns.get(chosen),gold_top2_score=ts.get(gold),winner_top2_score=ts.get(chosen),
   faiss_ms=faiss_ms,top2_ms=top2_ms,fram_ms=fram_ms,address_ms=faiss_ms+top2_ms+fram_ms,
   posting_visits=int(sum(int(LISTSIZE[j]) for a in cells for j in a if j>=0))))
 DETAIL625[name]=rows;SUM625[name]={}
 for split,part in [('DIAG',rows[:24]),('VALID',rows[24:]),('ALL',rows)]:
  SUM625[name][split]=dict(cases=len(part),selected_correct=sum(r['correct'] for r in part),
   gold_candidate=sum(r['gold_candidate'] for r in part),recall5=sum(r['target'] in r['top5'] for r in part),
   full_native_top1_correct=sum(r['full_native_rank']==1 for r in part),
   mean_touched=round(float(np.mean([r['touched'] for r in part])),3),address_p50_ms=median625([r['address_ms'] for r in part]))
 log625('RANKING',name,json.dumps(SUM625[name]['ALL']))
FAIL625=[r for r in DETAIL625['NARROW'] if not r['correct']]
log625('NARROW_FAILURE_DIAGNOSIS',json.dumps(FAIL625))
def physical625(ci,case):
 p=POOL/f'cut3_{ci:05d}.pt';t=time.perf_counter();assert sha625(p)==MANIFEST[ci]['sha256']
 stored=torch.load(p,map_location='cpu',weights_only=True)
 mem=tuple((a.to(DEVICE),b.to(DEVICE)) for a,b in stored);load_ms=(time.perf_counter()-t)*1000
 t=time.perf_counter()
 with torch.no_grad():ans=answer(mem,case['question'])
 answer_ms=(time.perf_counter()-t)*1000;del mem,stored
 return dict(answer=str(ans),correct=bool(hit(ans,case['gold'])),load_ms=load_ms,answer_ms=answer_ms)
# Gold-forced control is diagnostic ONLY, never included in autonomous accuracy.
ORACLE625=[]
for r in FAIL625:
 row=dict(case=r['case'],target=r['target'],oracle_control=True)
 try:row.update(physical625(r['target'],PANEL624[r['case']]))
 except Exception as e:row.update(correct=False,error=repr(e))
 ORACLE625.append(row);log625('ORACLE_DIAGNOSTIC',json.dumps(row))
# Locked NARROW selection: every question, physical file, no oracle substitution.
E2E625=[]
for i,r in enumerate(DETAIL625['NARROW']):
 row=dict(case=i,target=r['target'],selected=r['selected'],selected_correct=r['correct'])
 try:
  if r['selected'] is None:raise RuntimeError('NO_CANDIDATE')
  result=physical625(r['selected'],PANEL624[i]);row.update(answer_correct=result.pop('correct'),**result)
 except Exception as e:row.update(answer_correct=False,error=repr(e))
 E2E625.append(row)
 if (i+1)%8==0:log625('PHYSICAL_E2E',i+1,'/48','correct',sum(x['answer_correct'] for x in E2E625))
MANIFEST_OK625=all((POOL/r['file']).stat().st_size==r['bytes'] and sha625(POOL/r['file'])==r['sha256'] for r in MANIFEST)
HOOKS625=any(m._forward_hooks or m._forward_pre_hooks for layer in model.model.layers for m in (layer,layer.self_attn,layer.self_attn.q_proj))
INTEGRITY625=dict(manifest_1453_sha_verified=MANIFEST_OK625,weight_unchanged=weight_sha()==WB,sentinel_unchanged=sentinel()==SB,
 zero_trainable=not any(p.requires_grad for p in model.parameters()),no_hooks=not HOOKS625,source_lock=hashlib.sha256(src).hexdigest()==LOCK)
STORAGE625=dict(disk_pool_bytes=sum(r['bytes'] for r in MANIFEST),native_activation_bytes=TOK.numel()*TOK.element_size(),
 faiss_index_bytes=len(faiss.serialize_index(INDEX)),pca_basis_bytes=BASIS.numel()*BASIS.element_size(),
 pca_center_bytes=CENTER.numel()*CENTER.element_size(),owner_bytes=OWNER.nbytes)
STORAGE625['index_plus_native_activation_bytes']=sum(STORAGE625[k] for k in ('native_activation_bytes','faiss_index_bytes','pca_basis_bytes','pca_center_bytes','owner_bytes'))
E2ESUM625=dict(cases=len(E2E625),selected_correct=sum(x['selected_correct'] for x in E2E625),answer_correct=sum(x['answer_correct'] for x in E2E625),
 errors=sum('error' in x for x in E2E625),load_p50_ms=median625([x['load_ms'] for x in E2E625 if 'load_ms' in x]),
 answer_p50_ms=median625([x['answer_ms'] for x in E2E625 if 'answer_ms' in x]))
RESULT625=dict(test='625',status='COMPLETE' if all(INTEGRITY625.values()) else 'INTEGRITY_FAILURE',architecture='Frozen Qwen + original CUT3 + PCA128 IVF128 + TOP2_SOFT + FRAM',
 storage=STORAGE625,ranking=SUM625,failures=FAIL625,oracle_controls=ORACLE625,physical_e2e=E2ESUM625,
 per_query=DETAIL625,e2e_rows=E2E625,integrity=INTEGRITY625,elapsed_s=round(time.perf_counter()-T625,2),
 limitations=['Oracle control is never counted as autonomous success.','Fixed synthetic capital template; not independent multi-domain generalization.',
 'Model memory and Python allocator overhead excluded from RAM buffer bytes.','Disk reads may be OS cached.',
 'Full native reference is diagnostic only; no claim of million-cartridge scaling.'])
ROOT625.mkdir(parents=True,exist_ok=True)
with open(OUT625,'w') as f:json.dump(RESULT625,f,ensure_ascii=False,indent=2,default=str)
log625('TEST625_E2E',json.dumps(E2ESUM625));log625('TEST625_INTEGRITY',json.dumps(INTEGRITY625))
log625('TEST625_DEMO_DECISION','PERFECT_48_OF_48' if E2ESUM625['answer_correct']==48 and all(INTEGRITY625.values()) else 'REPORT_ACTUAL_ACCURACY')
log625('TEST625_OUTPUT',str(OUT625));log625('TEST625_STATUS',RESULT625['status'])
if not all(INTEGRITY625.values()):raise RuntimeError('TEST625_INTEGRITY_FAILURE')

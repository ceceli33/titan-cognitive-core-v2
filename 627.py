# TEST627 — LOCKED PEAK50 + PHYSICAL CUT3 FINAL ACCEPTANCE + LOSSLESS STORAGE AUDIT
# Single Colab cell. Original TEST626 / TEST624 backbone, model, CUT3, FAISS and original disk files unchanged.
import os,time,json,hashlib,urllib.request,math,io,gzip,statistics,collections
from pathlib import Path
import numpy as np,torch
T627=time.perf_counter();ROOT627=Path('/content/AKBASCORE_TEST624');OUT627=ROOT627/'TEST627_RESULT.json'
LOCAL627=Path('/content/TEST626_NATIVE_RANKING_CALIBRATION.py');URL627='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/626.py'
def log627(*a):print(*a,flush=True)
def med627(a):return round(float(np.median(a)),4) if a else None
def sha627(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
NEED627=('candidate626','features626','rerank626','CFG626','QUEST','PANEL624','MANIFEST','POOL','CAR','model','WB','SB','src','LOCK','TOTAL','answer','hit','DEVICE','weight_sha','sentinel','question_rep','apply','FIT','tok')
if not all(k in globals() for k in NEED627):
 if LOCAL627.exists():b627=LOCAL627.read_bytes()
 else:b627=urllib.request.urlopen(urllib.request.Request(URL627,headers={'User-Agent':'TEST627'}),timeout=120).read()
 marker627="log626('[1/6] PREDECLARED ORIGINAL QUERY PANEL"
 assert b627.decode().count(marker627)==1,'TEST626 BACKBONE BOUNDARY ERROR'
 log627('BOOTSTRAP_TEST626_SHA256',hashlib.sha256(b627).hexdigest())
 exec(compile(b627.decode().split(marker627,1)[0],'TEST626_BACKBONE.py','exec'),globals())
assert TOTAL==1453 and len(MANIFEST)==1453 and len(QUEST)==48 and len(PANEL624)==48
LOCKED627=next(c for c in CFG626 if c[0]=='PEAK50')
assert LOCKED627==('PEAK50',.5,0.),'RERANK_LOCK_MISMATCH'
assert not any(p.requires_grad for p in model.parameters())
log627('='*112);log627('TEST627 — PEAK50 LOCKED / 1453 PHYSICAL CUT3 / LOSSLESS STORAGE AUDIT');log627('='*112)
log627('[1/7] LOCKED PEAK50 ROUTING / ORIGINAL 48 + ALTERNATE 48')
ROUTES627=[]
for panel in ('ORIGINAL','ALTERNATE'):
 for i,case in enumerate(PANEL624):
  if panel=='ORIGINAL':question=case['question'];q=QUEST[i]
  else:
   country=CAR[case['target']]['fact'].split('The current capital of ',1)[1].split(' is ',1)[0]
   question=f'Name the present capital of {country}.' if i%2==0 else f'Which city currently serves as the capital of {country}?'
   rep,_=question_rep(question);q=apply(rep[3],FIT).detach().float().cpu().contiguous()
  t=time.perf_counter();ids,route=candidate626(q);fs=features626(q,ids);ranking,_=rerank626(ids,fs,LOCKED627)
  ROUTES627.append(dict(panel=panel,case=i,question=question,target=case['target'],selected=ranking[0] if ranking else None,
   top5=ranking[:5],gold_candidate=case['target'] in ids,selected_correct=bool(ranking and ranking[0]==case['target']),
   touched=route['touched'],address_ms=(time.perf_counter()-t)*1000))
  if len(ROUTES627)%16==0:log627('LOCKED_ROUTE',len(ROUTES627),'/96','correct',sum(x['selected_correct'] for x in ROUTES627))
log627('[2/7] PHYSICAL DISK + SHA + FROZEN ANSWERS / 96')
E2E627=[]
for r in ROUTES627:
 row=dict(panel=r['panel'],case=r['case'],target=r['target'],selected=r['selected'],selected_correct=r['selected_correct'])
 try:
  ci=r['selected'];assert ci is not None,'NO_CANDIDATE';p=POOL/f'cut3_{ci:05d}.pt'
  t=time.perf_counter();assert sha627(p)==MANIFEST[ci]['sha256'],'SHA_MISMATCH'
  stored=torch.load(p,map_location='cpu',weights_only=True);mem=tuple((a.to(DEVICE),b.to(DEVICE)) for a,b in stored)
  row['load_ms']=(time.perf_counter()-t)*1000;t=time.perf_counter()
  with torch.no_grad():ans=answer(mem,r['question'])
  row.update(answer=str(ans),correct=bool(hit(ans,PANEL624[r['case']]['gold'])),answer_ms=(time.perf_counter()-t)*1000)
  del mem,stored
 except Exception as e:row.update(correct=False,error=repr(e))
 E2E627.append(row)
 if len(E2E627)%16==0:log627('PHYSICAL_E2E',len(E2E627),'/96','correct',sum(x['correct'] for x in E2E627))
log627('[3/7] ALL 1453 PHYSICAL FILE SIZES + SHA256')
SIZES627=[];BAD627=[]
for i,r in enumerate(MANIFEST):
 p=POOL/r['file'];size=p.stat().st_size if p.exists() else -1;ok=size==r['bytes'] and sha627(p)==r['sha256'] if size>=0 else False
 SIZES627.append(size)
 if not ok:BAD627.append(i)
 if (i+1)%300==0:log627('FILE_AUDIT',i+1,'/1453')
log627('[4/7] TENSOR STRUCTURE / ALL FILES; LOSSLESS GZIP ON 32 PREDECLARED SAMPLES')
# Read-only inspection: no write-back, no conversion of tensor dtype, no deletion of original files.
TENSOR627=collections.Counter();DTYPE627=collections.Counter();SHAPE627=collections.Counter();TENSOR_BYTES627=0;TENSOR_COUNTS627=[]
SAMPLE627=sorted(set(np.linspace(0,TOTAL-1,32,dtype=int).tolist()))
COMP627=[];LOAD_ERR627=[]
def tensors627(o):
 if torch.is_tensor(o):yield o
 elif isinstance(o,(tuple,list)):
  for v in o:yield from tensors627(v)
 elif isinstance(o,dict):
  for v in o.values():yield from tensors627(v)
for i,r in enumerate(MANIFEST):
 p=POOL/r['file']
 try:
  obj=torch.load(p,map_location='cpu',weights_only=True)
  tt=list(tensors627(obj));nb=sum(t.numel()*t.element_size() for t in tt)
  TENSOR_BYTES627+=nb;TENSOR_COUNTS627.append(len(tt))
  for t in tt:
   DTYPE627[str(t.dtype)]+=t.numel()*t.element_size()
   SHAPE627[str(tuple(t.shape))]+=1
  TENSOR627['tensor_count']+=len(tt);TENSOR627['numel']+=sum(t.numel() for t in tt)
  if i in SAMPLE627:
   raw=p.read_bytes();t=time.perf_counter();packed=gzip.compress(raw,compresslevel=6,mtime=0);comp_ms=(time.perf_counter()-t)*1000
   t=time.perf_counter();restored=gzip.decompress(packed);decomp_ms=(time.perf_counter()-t)*1000
   ok=hashlib.sha256(restored).digest()==hashlib.sha256(raw).digest() and restored==raw
   COMP627.append(dict(id=i,raw_bytes=len(raw),gzip_bytes=len(packed),ratio=round(len(packed)/len(raw),6),
    compress_ms=round(comp_ms,3),decompress_ms=round(decomp_ms,3),bitwise_restored=ok,tensor_bytes=nb,
    overhead_bytes=len(raw)-nb,dtypes=sorted({str(t.dtype) for t in tt}),shapes=[list(t.shape) for t in tt[:4]]))
  del obj,tt
 except Exception as e:LOAD_ERR627.append(dict(id=i,error=repr(e)))
 if (i+1)%300==0:log627('TENSOR_AUDIT',i+1,'/1453')
RAW627=sum(SIZES627);RATIO627=sum(x['gzip_bytes'] for x in COMP627)/sum(x['raw_bytes'] for x in COMP627) if COMP627 else None
STORAGE627=dict(cartridges=TOTAL,disk_pool_bytes=RAW627,disk_pool_GB_decimal=round(RAW627/1e9,6),
 avg_file_bytes=round(float(np.mean(SIZES627)),2),median_file_bytes=med627(SIZES627),
 min_file_bytes=min(SIZES627),max_file_bytes=max(SIZES627),total_tensor_payload_bytes=TENSOR_BYTES627,
 file_container_overhead_bytes=RAW627-TENSOR_BYTES627,avg_tensors_per_file=round(float(np.mean(TENSOR_COUNTS627)),3),
 tensor_dtype_bytes=dict(DTYPE627),common_shapes=SHAPE627.most_common(12),
 lossless_gzip_sample_files=len(COMP627),gzip_sample_ratio=round(RATIO627,6) if RATIO627 else None,
 gzip_sample_estimated_pool_bytes=int(RAW627*RATIO627) if RATIO627 else None,
 gzip_sample_estimate_not_measured_full_pool=True,
 gzip_sample_all_bitwise_restored=all(x['bitwise_restored'] for x in COMP627),
 gzip_compress_p50_ms=med627([x['compress_ms'] for x in COMP627]),gzip_decompress_p50_ms=med627([x['decompress_ms'] for x in COMP627]),
 sample=COMP627,load_errors=LOAD_ERR627)
log627('TEST627_STORAGE',json.dumps({k:v for k,v in STORAGE627.items() if k not in ('sample','common_shapes','load_errors')}))
log627('[5/7] INTEGRITY / NO TRAINING / NO HOOKS')
HOOKS627=any(m._forward_hooks or m._forward_pre_hooks for layer in model.model.layers for m in (layer,layer.self_attn,layer.self_attn.q_proj))
INTEGRITY627=dict(source_sha_lock=hashlib.sha256(src).hexdigest()==LOCK,manifest_1453_sha_verified=not BAD627,
 weight_unchanged=weight_sha()==WB,sentinel_unchanged=sentinel()==SB,
 zero_trainable=not any(p.requires_grad for p in model.parameters()),no_hooks=not HOOKS627,
 tensor_load_errors_zero=not LOAD_ERR627,lossless_sample_bitwise=bool(COMP627) and all(x['bitwise_restored'] for x in COMP627))
SUM627={p:dict(cases=48,selected_correct=sum(r['selected_correct'] for r in ROUTES627 if r['panel']==p),
 answer_correct=sum(r['correct'] for r in E2E627 if r['panel']==p),
 errors=sum('error' in r for r in E2E627 if r['panel']==p),
 touched_mean=round(float(np.mean([r['touched'] for r in ROUTES627 if r['panel']==p])),3),
 address_p50_ms=med627([r['address_ms'] for r in ROUTES627 if r['panel']==p]),
 load_p50_ms=med627([r['load_ms'] for r in E2E627 if r['panel']==p and 'load_ms' in r]),
 answer_p50_ms=med627([r['answer_ms'] for r in E2E627 if r['panel']==p and 'answer_ms' in r])) for p in ('ORIGINAL','ALTERNATE')}
RESULT627=dict(test='627',status='COMPLETE' if all(INTEGRITY627.values()) else 'INTEGRITY_FAILURE',
 locked_reranker='PEAK50',locked_candidates=dict(nprobe=2,token_k=16,budget=16),
 summary=SUM627,storage=STORAGE627,integrity=INTEGRITY627,per_query=ROUTES627,physical_answers=E2E627,
 elapsed_seconds=round(time.perf_counter()-T627,2),limitations=[
 'Original 48 questions and alternative 48 wording questions reuse the same previously exposed 48 facts: NOT an independent record holdout.',
 'PEAK50 was chosen after inspecting TEST626 validation results; this test is confirmation on previously exposed facts, not blind generalization.',
 'Lossless gzip ratio extrapolation is from 32 deterministic sampled files only, not actual compressed full pool.',
 'Gzip samples are in-memory, original disk files unchanged; compression adds decompression latency and requires integration testing before deployment.',
 'Disk read may be OS page-cache served, not verified cold SSD.',
 '1453 physical cartridges do not demonstrate million-cartridge scalability.',
 'Storage report excludes model VRAM/RAM and Python allocator overhead.'])
ROOT627.mkdir(parents=True,exist_ok=True)
with open(OUT627,'w',encoding='utf-8') as f:json.dump(RESULT627,f,ensure_ascii=False,indent=2,default=str)
log627('[6/7] FINAL');log627('TEST627_E2E',json.dumps(SUM627));log627('TEST627_INTEGRITY',json.dumps(INTEGRITY627))
log627('TEST627_STORAGE_SUMMARY',json.dumps({k:v for k,v in STORAGE627.items() if k not in ('sample','common_shapes','load_errors')}))
log627('[7/7] DEMO ACCEPTANCE')
ready627=all(INTEGRITY627.values()) and all(SUM627[p]['answer_correct']==48 for p in SUM627)
log627('TEST627_DECISION','READY_FOR_DEMO_WITH_DISCLOSED_NONBLIND_LIMITATION' if ready627 else 'REPORT_ACTUAL_ACCURACY')
log627('TEST627_OUTPUT',str(OUT627));log627('TEST627_STATUS',RESULT627['status'])
if not all(INTEGRITY627.values()):raise RuntimeError('TEST627_INTEGRITY_FAILURE')

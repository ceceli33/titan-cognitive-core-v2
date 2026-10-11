
# TEST613 — AKBASCORE MAM · RAM BOWL MULTI-MECHANISM SEARCH · SINGLE COLAB CELL · NO UI
# Frozen TEST610 source and CUT3 audit are executed unchanged. All new methods are independent readouts.
import os,sys,hashlib,urllib.request,time,math,json,collections,gc
import numpy as np
SRC='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/610.py'
SHA='a62611a4ffd91c2fcf03796c31da2af11ccd8141345655b562f3a614568af3e7'
req=urllib.request.Request(SRC,headers={'User-Agent':'AKBASCORE-TEST613'})
with urllib.request.urlopen(req,timeout=180) as f:raw=f.read()
got=hashlib.sha256(raw).hexdigest()
if got!=SHA:raise RuntimeError(f'FAIL_CLOSED 610.py SHA expected={SHA} got={got}')
src=raw.decode('utf-8');marker="print('[6/7] Demo acceptance"
if src.count(marker)!=1:raise RuntimeError('FAIL_CLOSED 610.py split marker changed')
print('TEST613 BASELINE_LOCK',got,flush=True)
exec(compile(src.split(marker,1)[0],'610.py','exec'),globals())
import torch
if CFG!=(3,0,8.0) or len(KBASE)!=96 or set(QBASE)!=set(range(24)):raise RuntimeError('FRAM FEATURE LOCK FAILED')
if (H1,H5)!=(12,12) or len(PRE_AUDIT)!=12 or sum(int(x['answer_ok']) for x in PRE_AUDIT)!=12:raise RuntimeError('CUT3 BASELINE REGRESSION')
SEED=613613;RG=np.random.default_rng(SEED);N=96;DIM=3584;CASES=list(range(24));HOLD=list(range(12,24));TARGET=np.asarray([int(PANEL[i]['target']) for i in CASES]);TOP=5
# The original FRAM score is the established upper reference, NOT a new index or constant-time lookup.
reference=np.asarray([[pair_scores(QBASE[i],KBASE[j],8.0) for j in range(N)] for i in CASES],dtype=np.float32)
# Exact unmodified token-level FRAM scores are diagnostic only. No label-driven fitting or tuning.
def unit(a):return a/(np.linalg.norm(a,axis=-1,keepdims=True)+1e-12)
K0=np.stack([x.detach().float().mean(0).cpu().numpy() for x in KBASE]);Q0=np.stack([QBASE[i].detach().float().mean(0).cpu().numpy() for i in CASES]);MU=K0.mean(0);KX=unit(K0-MU).astype(np.float32);QX=unit(Q0-MU).astype(np.float32)
if KX.shape!=(96,3584) or QX.shape!=(24,3584):raise RuntimeError('FEATURE SHAPE')
if not np.isfinite(KX).all() or not np.isfinite(QX).all():raise RuntimeError('NONFINITE FEATURES')

# TEST613: Real 96-native sparse RAM index: optimize resident bytes, candidate coverage and selectivity.
# Fixed sweep; no holdout labels used for index building, candidate scoring or parameter fitting.
# RAM accounting uses actual NumPy array nbytes, excludes Python overhead (reported separately).
import itertools
from collections import defaultdict
CANDS=(4,8,12,16,24,32,48);BUDGETS=(4,8,16,32,64,96)
TARGET=np.asarray([int(PANEL[i]['target']) for i in CASES],dtype=np.int32)
# Sign-aware coordinate postings: signed dimensions distinguish opposing activation directions.
# Fit document frequency on keys ONLY. Scoring can be IDF-weighted, capped to reduce work.
def make_index(k,mode):
    if mode=='ABS':
        ids=np.argpartition(np.abs(KX),-k,axis=1)[:,-k:]; qids=np.argpartition(np.abs(QX),-k,axis=1)[:,-k:]
    elif mode=='SIGNED':
        ids=np.argpartition(np.abs(KX),-k,axis=1)[:,-k:];qids=np.argpartition(np.abs(QX),-k,axis=1)[:,-k:]
        ids=ids+(KX[np.arange(N)[:,None],ids]<0)*DIM
        qids=qids+(QX[np.arange(len(CASES))[:,None],qids]<0)*DIM
    else:raise ValueError(mode)
    # CSR inverted index; keys sorted by coordinate, postings stored in one contiguous array.
    vocab=DIM*(2 if mode=='SIGNED' else 1)
    flat=ids.astype(np.int32).ravel();freq=np.bincount(flat,minlength=vocab).astype(np.int32)
    offsets=np.empty(vocab+1,dtype=np.int32);offsets[0]=0;np.cumsum(freq,out=offsets[1:])
    order=np.argsort(flat,kind='stable');postings=(order//k).astype(np.int32)
    # Query reads ONLY its own k posting lists, no bank-wide query scan.
    return qids.astype(np.int32),freq,offsets,postings

def retrieve(qids,freq,offsets,postings,budget,weighted):
    scores=np.full((len(CASES),N),-np.inf,dtype=np.float32);counts=[];visits=[];candidate_hit=[];rerank_counts=[]
    for i in CASES:
        tally=defaultdict(float);nvis=0
        for coord in qids[i]:
            v=int(coord);df=int(freq[v]);nvis+=df
            weight=math.log1p((N+1)/(df+1)) if weighted else 1.0
            for j in postings[offsets[v]:offsets[v+1]]:tally[int(j)]+=weight
        counts.append(len(tally));visits.append(nvis);candidate_hit.append(int(TARGET[i] in tally))
        if tally:
            chosen=sorted(tally,key=lambda j:(-tally[j],j))[:budget]
            ix=np.asarray(chosen,dtype=np.int32)
            scores[i,ix]=QX[i]@KX[ix].T
            rerank_counts.append(len(ix))
        else:rerank_counts.append(0)
    return scores,counts,visits,candidate_hit,rerank_counts

def metric(scores):
    rr=[]
    for i in CASES:
        t=int(TARGET[i]);valid=np.flatnonzero(np.isfinite(scores[i]));
        if t not in valid:rr.append(N+1);continue
        order=valid[np.argsort(-scores[i,valid],kind='stable')];rr.append(int(np.flatnonzero(order==t)[0])+1)
    return dict(cal_r1=sum(r==1 for r in rr[:12]),hold_r1=sum(r==1 for r in rr[12:]),hold_r5=sum(r<=5 for r in rr[12:]),hold_ranks=rr[12:])

rows=[];best=None
for mode,k in itertools.product(('ABS','SIGNED'),CANDS):
    t0=time.perf_counter();qids,freq,offsets,postings=make_index(k,mode)
    resident=freq.nbytes+offsets.nbytes+postings.nbytes
    # Query top-k features are transient, not charged as resident index.
    for weighted in (False,True):
        for budget in BUDGETS:
            scores,counts,visits,hit,reranks=retrieve(qids,freq,offsets,postings,budget,weighted)
            m=metric(scores);row=dict(mode=mode,k=k,weighted=weighted,budget=budget,**m,hold_candidate_coverage=sum(hit[12:]),hold_mean_raw_candidates=round(float(np.mean(counts[12:])),2),hold_mean_posting_visits=round(float(np.mean(visits[12:])),2),hold_mean_rerank=round(float(np.mean(reranks[12:])),2),resident_bytes=resident,bytes_per_cartridge=round(resident/N,2))
            rows.append(row)
    print('TEST613_SWEEP',json.dumps(dict(mode=mode,k=k,resident_bytes=resident,elapsed_s=round(time.perf_counter()-t0,2),best_hold_r1=max(x['hold_r1'] for x in rows if x['mode']==mode and x['k']==k),best_hold_coverage=max(x['hold_candidate_coverage'] for x in rows if x['mode']==mode and x['k']==k))),flush=True)
# Do not choose optimum by holdout and then claim an unbiased holdout result.
# Select only using CAL cases, then disclose holdout performance as a one-time post-selection audit.
selection=sorted(rows,key=lambda r:(-r['cal_r1'],r['resident_bytes'],r['budget'],r['mode'],r['weighted']))[0]
# Pareto rows give full exploratory results; not independently validated after inspection.
pareto=[]
for r in rows:
    if r['hold_candidate_coverage']==12 and r['hold_r1']>=11 and r['hold_mean_rerank']<=16:pareto.append(r)
pareto=sorted(pareto,key=lambda r:(r['resident_bytes'],r['hold_mean_rerank'],-r['hold_r1']))[:15]
print('TEST613_CAL_SELECTED',json.dumps(selection),flush=True)
print('TEST613_EXPLORATORY_PARETO',json.dumps(pareto),flush=True)
# Large-scale *storage-only* benchmark with random synthetic postings. Not Qwen retrieval, no accuracy claims.
# Uses 1,000,000 independent synthetic signatures, k=8, 3584 coordinate vocabulary.
# Measure compact CSR NumPy buffer bytes; extrapolate 5M storage linearly, without asserting latency.
scale=[]
for nn in (100000,1000000):
    t0=time.perf_counter();rg=np.random.default_rng(SEED+nn);kk=8;vv=DIM
    # random integer coordinates may repeat within one signature; negligible for storage sizing,
    # but this benchmark is NOT a retrieval-accuracy experiment.
    keys=rg.integers(vv,size=(nn,kk),dtype=np.int32);flat=keys.ravel();counts=np.bincount(flat,minlength=vv).astype(np.int32)
    off=np.empty(vv+1,dtype=np.int32);off[0]=0;np.cumsum(counts,out=off[1:]);ordering=np.argsort(flat,kind='stable');post=(ordering//kk).astype(np.int32)
    size=counts.nbytes+off.nbytes+post.nbytes
    row=dict(synthetic_N=nn,k=kk,vocab=vv,csr_bytes=int(size),bytes_per_cartridge=round(size/nn,3),build_seconds=round(time.perf_counter()-t0,2),five_million_projected_bytes=int(5000000*kk*4+counts.nbytes+off.nbytes),notes='storage-only; not native Qwen or query latency')
    scale.append(row);print('TEST613_STORAGE',json.dumps(row),flush=True)
    del keys,flat,counts,off,ordering,post
if any(p.requires_grad for p in model.parameters()):raise RuntimeError('MODEL NO LONGER FROZEN')
if any(layer._forward_hooks or layer._forward_pre_hooks for layer in model.model.layers):raise RuntimeError('MODEL HOOK LEFT INSTALLED')
result=dict(test=613,source_sha256=got,baseline=dict(fram_hold_r1=int(H1),fram_hold_r5=int(H5),cut3_hold_correct=sum(int(x['answer_ok']) for x in PRE_AUDIT)),scope='96 real native CUT3 signatures and 24 questions; 1M synthetic storage ONLY',selection_protocol='CAL-only; holdout numbers exploratory after design sweep',cal_selected=selection,exploratory_pareto=pareto,all_methods=rows,storage_scale=scale,limitations=['No million-native-cartridge retrieval validation','No sublinear guarantee from sparse index','CSR buffer bytes exclude Python runtime and raw cartridge disk files','Query-time dense 3584D rerank cost applies only to selected candidates','FRAM 12/12 remains separate from candidate rerank'])
path='/content/TEST613_RESULT.json' if os.path.isdir('/content') else '/tmp/TEST613_RESULT.json'
with open(path,'w',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print('TEST613_FINAL',json.dumps(dict(test=613,baseline=result['baseline'],tested=len(rows),cal_selected=selection,pareto_count=len(pareto),storage=scale,output=path),ensure_ascii=False),flush=True)

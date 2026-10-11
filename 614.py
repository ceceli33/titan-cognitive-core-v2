
# TEST614 — AKBASCORE MAM · RAM BOWL / NATIVE FRAM RERANK / SCALE / HUMAN UNITS
# Single Google Colab cell. No Gradio. Frozen CUT3/FRAM baseline unchanged.
# Selection: CAL only. HOLDOUT evaluated once after configuration lock.
import os,sys,time,json,math,hashlib,urllib.request,gc
from collections import defaultdict
import numpy as np
import torch

TEST=614
SRC='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/610.py'
SHA='a62611a4ffd91c2fcf03796c31da2af11ccd8141345655b562f3a614568af3e7'
req=urllib.request.Request(SRC,headers={'User-Agent':'AKBASCORE-TEST614'})
with urllib.request.urlopen(req,timeout=180) as f:raw=f.read()
got=hashlib.sha256(raw).hexdigest()
if got!=SHA:raise RuntimeError(f'FAIL_CLOSED SOURCE_SHA expected={SHA} got={got}')
src=raw.decode('utf-8')
marker="print('[6/7] Demo acceptance"
if src.count(marker)!=1:raise RuntimeError('FAIL_CLOSED BASELINE_SPLIT')
print('TEST614 BASELINE_LOCK',got,flush=True)
exec(compile(src.split(marker,1)[0],'610.py','exec'),globals())

# BASELINE ACCEPTANCE
if CFG!=(3,0,8.0):raise RuntimeError('FRAM_CFG_REGRESSION')
if len(KBASE)!=96 or set(QBASE)!=set(range(24)):raise RuntimeError('CARTRIDGE_PANEL_REGRESSION')
if (H1,H5)!=(12,12):raise RuntimeError('FRAM_RECALL_REGRESSION')
if len(PRE_AUDIT)!=12 or sum(int(x['answer_ok']) for x in PRE_AUDIT)!=12:
    raise RuntimeError('CUT3_ANSWER_REGRESSION')
if any(p.requires_grad for p in model.parameters()):
    raise RuntimeError('MODEL_NOT_FROZEN')

SEED=614614
N=96
DIM=3584
CASES=list(range(24))
CAL=CASES[:12]
HOLD=CASES[12:]
TARGET=np.asarray([int(PANEL[i]['target']) for i in CASES],dtype=np.int32)
RG=np.random.default_rng(SEED)

def unit(a):
    return a/(np.linalg.norm(a,axis=-1,keepdims=True)+1e-12)

# Router features only. These do not replace native FRAM reranking.
K0=np.stack([x.detach().float().mean(0).cpu().numpy() for x in KBASE])
Q0=np.stack([QBASE[i].detach().float().mean(0).cpu().numpy() for i in CASES])
MU=K0.mean(0)
KX=unit(K0-MU).astype(np.float32)
QX=unit(Q0-MU).astype(np.float32)
if KX.shape!=(N,DIM) or QX.shape!=(24,DIM):
    raise RuntimeError('ROUTER_FEATURE_SHAPE')
if not np.isfinite(KX).all() or not np.isfinite(QX).all():
    raise RuntimeError('NONFINITE_FEATURE')

# HUMAN-READABLE CARTRIDGE CAPACITY
# Source token count comes from the native captured residual length.
# BF16 CUT3 theoretical payload: H3 7 KiB/token + K/V L0-L3 8 KiB/token.
# This is a payload estimate, not a measured serialized file size.
SOURCE_TOKENS=[int(x.shape[0]) for x in KBASE]
if min(SOURCE_TOKENS)<=0:raise RuntimeError('INVALID_CARTRIDGE_LENGTH')
BYTES_PER_SOURCE_TOKEN=15360
TOKEN_TOTAL=sum(SOURCE_TOKENS)
HUMAN_UNITS={
    'cartridges':N,
    'source_tokens_total':TOKEN_TOTAL,
    'source_tokens_min':min(SOURCE_TOKENS),
    'source_tokens_max':max(SOURCE_TOKENS),
    'source_tokens_mean':round(float(np.mean(SOURCE_TOKENS)),2),
    'cut3_payload_bytes_per_token_estimate':BYTES_PER_SOURCE_TOKEN,
    'cut3_payload_total_mib_estimate':round(TOKEN_TOTAL*BYTES_PER_SOURCE_TOKEN/1048576,4),
    'word_count':'NOT_MEASURED',
    'word_equivalence':'NOT_ASSUMED',
    'note':'Token counts from native residual lengths; CUT3 size theoretical BF16 payload, not serialized disk measurement.'
}
print('TEST614_HUMAN_UNITS',json.dumps(HUMAN_UNITS),flush=True)

# COMPACT CSR POSTING INDEX
def build_index(k,mode):
    ids=np.argpartition(np.abs(KX),-k,axis=1)[:,-k:].astype(np.int32)
    qids=np.argpartition(np.abs(QX),-k,axis=1)[:,-k:].astype(np.int32)
    if mode=='SIGNED':
        ids=ids+(KX[np.arange(N)[:,None],ids]<0).astype(np.int32)*DIM
        qids=qids+(QX[np.arange(24)[:,None],qids]<0).astype(np.int32)*DIM
        vocab=2*DIM
    elif mode=='ABS':
        vocab=DIM
    else:
        raise ValueError(mode)
    flat=ids.ravel()
    freq=np.bincount(flat,minlength=vocab).astype(np.int32)
    offsets=np.empty(vocab+1,dtype=np.int32)
    offsets[0]=0
    np.cumsum(freq,out=offsets[1:])
    order=np.argsort(flat,kind='stable')
    postings=(order//k).astype(np.int32)
    return {
        'qids':qids,'freq':freq,'offsets':offsets,'postings':postings,
        'bytes':int(freq.nbytes+offsets.nbytes+postings.nbytes)
    }

def route(index,i,budget,weighted):
    tally=defaultdict(float)
    visits=0
    freq=index['freq']
    offsets=index['offsets']
    postings=index['postings']
    for coord in index['qids'][i]:
        v=int(coord)
        df=int(freq[v])
        visits+=df
        w=math.log1p((N+1)/(df+1)) if weighted else 1.0
        for j in postings[offsets[v]:offsets[v+1]]:
            tally[int(j)]+=w
    ordered=sorted(tally,key=lambda j:(-tally[j],j))
    chosen=ordered[:budget]
    return {
        'ids':chosen,
        'raw_candidates':len(tally),
        'posting_visits':visits,
        'target_in_raw':int(int(TARGET[i]) in tally),
        'target_in_budget':int(int(TARGET[i]) in chosen)
    }

# PREDECLARED ROUTER SEARCH: CALIBRATION ONLY
MODES=('ABS','SIGNED')
KS=(8,12,16,24,32,48)
BUDGETS=(4,8,12,16,24)
WEIGHTS=(False,True)
indexes={}
routes={}
cal_rows=[]

for mode in MODES:
    for k in KS:
        idx=build_index(k,mode)
        indexes[(mode,k)]=idx
        for weighted in WEIGHTS:
            for budget in BUDGETS:
                key=(mode,k,weighted,budget)
                rr=[route(idx,i,budget,weighted) for i in CAL]
                routes[key]=rr
                row={
                    'mode':mode,'k':k,'weighted':weighted,'budget':budget,
                    'cal_raw_coverage':sum(x['target_in_raw'] for x in rr),
                    'cal_budget_coverage':sum(x['target_in_budget'] for x in rr),
                    'cal_mean_candidates':round(float(np.mean([len(x['ids']) for x in rr])),3),
                    'cal_mean_posting_visits':round(float(np.mean([x['posting_visits'] for x in rr])),3),
                    'resident_bytes':idx['bytes'],
                    'bytes_per_cartridge':round(idx['bytes']/N,3)
                }
                cal_rows.append(row)

# Native FRAM score is computed ONLY for candidates returned by the router.
# It is never used to manufacture the candidate list.
def native_score(i,j):
    v=pair_scores(QBASE[i],KBASE[j],8.0)
    if torch.is_tensor(v):
        if v.numel()!=1:raise RuntimeError('FRAM_SCORE_NOT_SCALAR')
        v=float(v.detach().cpu().item())
    else:
        v=float(v)
    if not math.isfinite(v):raise RuntimeError('FRAM_SCORE_NONFINITE')
    return v

score_cache={}
def get_native_score(i,j):
    key=(int(i),int(j))
    if key not in score_cache:
        score_cache[key]=native_score(i,j)
    return score_cache[key]

def audit_candidates(i,route_result):
    ids=route_result['ids']
    if not ids:
        return {'rank':N+1,'r1':0,'r5':0,'target_in_budget':0,'candidates':0}
    pairs=[(j,get_native_score(i,j)) for j in ids]
    pairs.sort(key=lambda x:(-x[1],x[0]))
    ordered=[j for j,s in pairs]
    target=int(TARGET[i])
    rank=ordered.index(target)+1 if target in ordered else N+1
    return {
        'rank':rank,
        'r1':int(rank==1),
        'r5':int(rank<=5),
        'target_in_budget':int(target in ids),
        'candidates':len(ids)
    }

# Evaluate only calibration configurations with full 12/12 budget coverage.
eligible=[r for r in cal_rows if r['cal_budget_coverage']==12]
print('TEST614_CAL_ROUTER',json.dumps({
    'configurations':len(cal_rows),
    'full_cal_coverage_configs':len(eligible),
    'minimum_bytes_full_coverage':min((r['resident_bytes'] for r in eligible),default=None),
    'minimum_budget_full_coverage':min((r['budget'] for r in eligible),default=None)
}),flush=True)

if not eligible:
    print('TEST614_NO_FULL_CAL_ROUTER: evaluating best partial-coverage configurations',flush=True)
    eligible=sorted(cal_rows,key=lambda r:(-r['cal_budget_coverage'],r['budget'],r['resident_bytes']))[:12]

# Restrict expensive native FRAM tests to calibration finalists.
# Preference: small candidate budget, low memory, fewer posting visits.
eligible=sorted(eligible,key=lambda r:(
    r['budget'],r['resident_bytes'],r['cal_mean_posting_visits'],
    r['mode'],r['k'],r['weighted']
))
finalists=[]
seen=set()
for r in eligible:
    signature=(r['mode'],r['k'],r['weighted'],r['budget'])
    if signature not in seen:
        finalists.append(r)
        seen.add(signature)
    if len(finalists)>=12:break

native_cal=[]
for r in finalists:
    key=(r['mode'],r['k'],r['weighted'],r['budget'])
    audits=[audit_candidates(i,routes[key][i]) for i in CAL]
    out=dict(r)
    out.update({
        'cal_native_r1':sum(a['r1'] for a in audits),
        'cal_native_r5':sum(a['r5'] for a in audits),
        'cal_native_ranks':[a['rank'] for a in audits]
    })
    native_cal.append(out)
    print('TEST614_NATIVE_CAL',json.dumps({
        'mode':r['mode'],'k':r['k'],'weighted':r['weighted'],
        'budget':r['budget'],'r1':out['cal_native_r1'],
        'r5':out['cal_native_r5'],'bytes':r['resident_bytes']
    }),flush=True)

# LOCK: no holdout labels or outcomes enter configuration selection.
selected=sorted(native_cal,key=lambda r:(
    -r['cal_native_r1'],
    -r['cal_native_r5'],
    r['budget'],
    r['resident_bytes'],
    r['cal_mean_posting_visits'],
    r['mode'],r['k'],r['weighted']
))[0]
LOCK=(selected['mode'],selected['k'],selected['weighted'],selected['budget'])
print('TEST614_CAL_LOCK',json.dumps(selected),flush=True)

# ONE-TIME HOLDOUT EVALUATION
idx=indexes[(LOCK[0],LOCK[1])]
hold_routes=[route(idx,i,LOCK[3],LOCK[2]) for i in HOLD]
hold_audits=[audit_candidates(i,hold_routes[z]) for z,i in enumerate(HOLD)]
hold_summary={
    'native_fram_r1':sum(a['r1'] for a in hold_audits),
    'native_fram_r5':sum(a['r5'] for a in hold_audits),
    'raw_candidate_coverage':sum(x['target_in_raw'] for x in hold_routes),
    'budget_candidate_coverage':sum(x['target_in_budget'] for x in hold_routes),
    'mean_raw_candidates':round(float(np.mean([x['raw_candidates'] for x in hold_routes])),3),
    'mean_reranked_candidates':round(float(np.mean([len(x['ids']) for x in hold_routes])),3),
    'mean_posting_visits':round(float(np.mean([x['posting_visits'] for x in hold_routes])),3),
    'ranks':[a['rank'] for a in hold_audits],
    'resident_bytes':idx['bytes'],
    'bytes_per_cartridge':round(idx['bytes']/N,3)
}
print('TEST614_HOLDOUT',json.dumps(hold_summary),flush=True)

# SCALE TEST: synthetic signatures; real CSR construction and query timing.
# No synthetic recall claims. Measure candidate explosion and posting visits.
def scale_test(nn,k=12,nq=128):
    rg=np.random.default_rng(SEED+nn+k)
    vocab=2*DIM
    t0=time.perf_counter()
    keys=rg.integers(vocab,size=(nn,k),dtype=np.int32)
    flat=keys.ravel()
    freq=np.bincount(flat,minlength=vocab).astype(np.int32)
    offsets=np.empty(vocab+1,dtype=np.int32)
    offsets[0]=0
    np.cumsum(freq,out=offsets[1:])
    ordering=np.argsort(flat,kind='stable')
    postings=(ordering//k).astype(np.int32)
    build_seconds=time.perf_counter()-t0
    resident=int(freq.nbytes+offsets.nbytes+postings.nbytes)
    query_keys=rg.integers(vocab,size=(nq,k),dtype=np.int32)
    visited=[]
    unique_candidates=[]
    latencies=[]
    for q in query_keys:
        t1=time.perf_counter()
        seen=set()
        visits=0
        for v in q:
            v=int(v)
            lo=int(offsets[v])
            hi=int(offsets[v+1])
            visits+=hi-lo
            seen.update(int(j) for j in postings[lo:hi])
        latencies.append((time.perf_counter()-t1)*1000)
        visited.append(visits)
        unique_candidates.append(len(seen))
    out={
        'synthetic_N':nn,
        'k':k,
        'queries':nq,
        'csr_bytes':resident,
        'csr_mib':round(resident/1048576,3),
        'bytes_per_cartridge':round(resident/nn,3),
        'build_seconds':round(build_seconds,3),
        'mean_posting_visits':round(float(np.mean(visited)),2),
        'mean_unique_candidates':round(float(np.mean(unique_candidates)),2),
        'mean_candidate_fraction':round(float(np.mean(unique_candidates))/nn,6),
        'query_p50_ms':round(float(np.percentile(latencies,50)),4),
        'query_p95_ms':round(float(np.percentile(latencies,95)),4),
        'scope':'Synthetic CSR posting traversal only; no native Qwen accuracy, FRAM or disk IO'
    }
    del keys,flat,freq,offsets,ordering,postings,query_keys
    gc.collect()
    return out

scale=[]
for nn in (100000,1000000):
    r=scale_test(nn,k=12,nq=128)
    scale.append(r)
    print('TEST614_SCALE',json.dumps(r),flush=True)

# Explicit separation of proven measurements and estimates.
PROJECTED={
    'five_million_k12_csr_bytes':int(5000000*12*4+(2*DIM*4)+((2*DIM+1)*4)),
    'five_million_k12_csr_mib':round((5000000*12*4+(2*DIM*4)+((2*DIM+1)*4))/1048576,3),
    'scope':'Linear buffer-size projection, not a 5M measured build',
    'exclusions':'Python runtime, source cartridges, metadata, disk IO, native FRAM reranking'
}
print('TEST614_PROJECTION',json.dumps(PROJECTED),flush=True)

# FINAL SAFETY CHECKS
if any(p.requires_grad for p in model.parameters()):
    raise RuntimeError('FINAL_MODEL_FROZEN_FAILED')
if any(layer._forward_hooks or layer._forward_pre_hooks for layer in model.model.layers):
    raise RuntimeError('FINAL_HOOK_CLEANUP_FAILED')
if (H1,H5)!=(12,12) or sum(int(x['answer_ok']) for x in PRE_AUDIT)!=12:
    raise RuntimeError('FINAL_BASELINE_REGRESSION')

result={
    'test':TEST,
    'source_sha256':got,
    'baseline':{'fram_hold_r1':12,'fram_hold_r5':12,'cut3_hold_correct':12},
    'human_units':HUMAN_UNITS,
    'calibration_router_sweep':cal_rows,
    'native_calibration_finalists':native_cal,
    'locked_configuration':selected,
    'holdout':hold_summary,
    'synthetic_scale':scale,
    'projection':PROJECTED,
    'limitations':[
        '96 real native cartridges only',
        'Holdout used once after calibration lock',
        'Native FRAM scored only selected candidates',
        'Synthetic scale does not establish native retrieval accuracy',
        'Timing excludes model inference and disk loading',
        'CSR bytes exclude process/runtime overhead',
        'Word counts not inferred from token counts',
        'CUT3 disk payload estimated, not serialized-file measured',
        'No CUT3 generation replay for newly selected routes'
    ]
}
path='/content/TEST614_RESULT.json' if os.path.isdir('/content') else '/tmp/TEST614_RESULT.json'
with open(path,'w',encoding='utf-8') as f:
    json.dump(result,f,ensure_ascii=False,indent=2)
print('TEST614_FINAL',json.dumps({
    'test':614,
    'baseline':result['baseline'],
    'human_units':HUMAN_UNITS,
    'calibration_lock':selected,
    'holdout':hold_summary,
    'scale':scale,
    'projection':PROJECTED,
    'output':path
},ensure_ascii=False),flush=True)

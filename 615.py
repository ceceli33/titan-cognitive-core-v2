
# TEST615 — AKBASCORE MAM · RAM BOWL CANDIDATE BUDGET X-RAY
# TEST610 frozen CUT3 + FRAM backbone. Single Colab cell. No UI.
# Diagnostic only: previously inspected 24-question panel is NOT a fresh holdout.
import os,sys,time,json,math,hashlib,urllib.request,gc
from collections import defaultdict
import numpy as np
import torch

TEST=615
SRC='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/610.py'
SHA='a62611a4ffd91c2fcf03796c31da2af11ccd8141345655b562f3a614568af3e7'
req=urllib.request.Request(SRC,headers={'User-Agent':'AKBASCORE-TEST615'})
with urllib.request.urlopen(req,timeout=180) as f:raw=f.read()
got=hashlib.sha256(raw).hexdigest()
if got!=SHA:raise RuntimeError(f'FAIL_CLOSED SOURCE_SHA expected={SHA} got={got}')
src=raw.decode('utf-8')
marker="print('[6/7] Demo acceptance"
if src.count(marker)!=1:raise RuntimeError('FAIL_CLOSED BASELINE_SPLIT')
print('TEST615_BASELINE_LOCK',got,flush=True)
exec(compile(src.split(marker,1)[0],'610.py','exec'),globals())

# VERIFIED BASELINE
if CFG!=(3,0,8.0):raise RuntimeError('FRAM_CFG_REGRESSION')
if len(KBASE)!=96 or set(QBASE)!=set(range(24)):raise RuntimeError('PANEL_REGRESSION')
if (H1,H5)!=(12,12):raise RuntimeError('FRAM_REGRESSION')
if len(PRE_AUDIT)!=12 or sum(int(x['answer_ok']) for x in PRE_AUDIT)!=12:
    raise RuntimeError('CUT3_REGRESSION')
if any(p.requires_grad for p in model.parameters()):
    raise RuntimeError('MODEL_NOT_FROZEN')

N=96
DIM=3584
CASES=list(range(24))
CAL=CASES[:12]
OLD_HOLD=CASES[12:]
TARGET=np.asarray([int(PANEL[i]['target']) for i in CASES],dtype=np.int32)
BUDGETS=(1,2,3,4,6,8,12,16,24,32,48,64,96)
CONFIGS=(
    ('SIGNED',48,False),
    ('SIGNED',48,True),
    ('SIGNED',16,True),
    ('SIGNED',12,True),
    ('ABS',12,False),
    ('ABS',16,True),
    ('ABS',48,False)
)

def unit(a):
    return a/(np.linalg.norm(a,axis=-1,keepdims=True)+1e-12)

# EXACT SAME ROUTER FEATURES AS TEST614
K0=np.stack([x.detach().float().mean(0).cpu().numpy() for x in KBASE])
Q0=np.stack([QBASE[i].detach().float().mean(0).cpu().numpy() for i in CASES])
MU=K0.mean(0)
KX=unit(K0-MU).astype(np.float32)
QX=unit(Q0-MU).astype(np.float32)
if KX.shape!=(96,3584) or QX.shape!=(24,3584):
    raise RuntimeError('FEATURE_SHAPE')
if not np.isfinite(KX).all() or not np.isfinite(QX).all():
    raise RuntimeError('NONFINITE_FEATURES')

# SOURCE CAPACITY: ACTUAL TOKEN LENGTHS, ESTIMATED CUT3 BF16 PAYLOAD
SOURCE_TOKENS=[int(x.shape[0]) for x in KBASE]
HUMAN_UNITS={
    'cartridges':N,
    'source_tokens_total':int(sum(SOURCE_TOKENS)),
    'source_tokens_min':int(min(SOURCE_TOKENS)),
    'source_tokens_max':int(max(SOURCE_TOKENS)),
    'source_tokens_mean':round(float(np.mean(SOURCE_TOKENS)),3),
    'cut3_payload_bytes_per_token_estimate':15360,
    'cut3_payload_total_mib_estimate':round(sum(SOURCE_TOKENS)*15360/1048576,4),
    'words':'NOT_MEASURED',
    'note':'CUT3 BF16 theoretical payload, not serialized file size'
}
print('TEST615_HUMAN_UNITS',json.dumps(HUMAN_UNITS),flush=True)

def make_index(k,mode):
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
    ordering=np.argsort(flat,kind='stable')
    postings=(ordering//k).astype(np.int32)
    return {
        'qids':qids,'freq':freq,'offsets':offsets,'postings':postings,
        'resident_bytes':int(freq.nbytes+offsets.nbytes+postings.nbytes)
    }

def full_route(index,i,weighted):
    tally=defaultdict(float)
    visits=0
    for coord in index['qids'][i]:
        v=int(coord)
        df=int(index['freq'][v])
        visits+=df
        weight=math.log1p((N+1)/(df+1)) if weighted else 1.0
        for j in index['postings'][index['offsets'][v]:index['offsets'][v+1]]:
            tally[int(j)]+=weight
    ordered=sorted(tally,key=lambda j:(-tally[j],j))
    return {
        'ordered':ordered,
        'raw_candidates':len(ordered),
        'posting_visits':visits,
        'target_raw_rank':ordered.index(int(TARGET[i]))+1 if int(TARGET[i]) in tally else N+1
    }

# NATIVE FRAM SCORES: ONLY FOR ACTUALLY ROUTED CANDIDATES
# Cache is a test-time diagnostic optimization; it is NOT part of the RAM index.
score_cache={}
def native_score(i,j):
    key=(int(i),int(j))
    if key not in score_cache:
        v=pair_scores(QBASE[i],KBASE[j],8.0)
        if torch.is_tensor(v):
            if v.numel()!=1:raise RuntimeError('FRAM_SCORE_NOT_SCALAR')
            v=float(v.detach().cpu().item())
        else:
            v=float(v)
        if not math.isfinite(v):raise RuntimeError('NONFINITE_FRAM_SCORE')
        score_cache[key]=v
    return score_cache[key]

def evaluate(route_result,i,budget):
    ids=route_result['ordered'][:budget]
    target=int(TARGET[i])
    if not ids:
        return {'candidate_hit':0,'native_rank':N+1,'native_r1':0,'native_r5':0,'reranked':0}
    scored=[(j,native_score(i,j)) for j in ids]
    scored.sort(key=lambda p:(-p[1],p[0]))
    ordered=[j for j,s in scored]
    rank=ordered.index(target)+1 if target in ordered else N+1
    return {
        'candidate_hit':int(target in ids),
        'native_rank':rank,
        'native_r1':int(rank==1),
        'native_r5':int(rank<=5),
        'reranked':len(ids)
    }

all_results=[]
diagnostics=[]
for mode,k,weighted in CONFIGS:
    idx=make_index(k,mode)
    routes=[full_route(idx,i,weighted) for i in CASES]
    print('TEST615_CONFIG',json.dumps({
        'mode':mode,'k':k,'weighted':weighted,
        'resident_bytes':idx['resident_bytes'],
        'bytes_per_cartridge':round(idx['resident_bytes']/N,3),
        'raw_coverage_cal':sum(r['target_raw_rank']<=N for r in routes[:12]),
        'raw_coverage_old_hold':sum(r['target_raw_rank']<=N for r in routes[12:]),
        'mean_raw_candidates':round(float(np.mean([r['raw_candidates'] for r in routes])),3),
        'mean_posting_visits':round(float(np.mean([r['posting_visits'] for r in routes])),3)
    }),flush=True)
    for budget in BUDGETS:
        audits=[evaluate(routes[i],i,budget) for i in CASES]
        row={
            'mode':mode,'k':k,'weighted':weighted,'budget':budget,
            'cal_candidate_hit':sum(a['candidate_hit'] for a in audits[:12]),
            'old_hold_candidate_hit':sum(a['candidate_hit'] for a in audits[12:]),
            'cal_native_r1':sum(a['native_r1'] for a in audits[:12]),
            'old_hold_native_r1':sum(a['native_r1'] for a in audits[12:]),
            'cal_native_r5':sum(a['native_r5'] for a in audits[:12]),
            'old_hold_native_r5':sum(a['native_r5'] for a in audits[12:]),
            'all_native_r1':sum(a['native_r1'] for a in audits),
            'all_native_r5':sum(a['native_r5'] for a in audits),
            'mean_reranked':round(float(np.mean([a['reranked'] for a in audits])),3),
            'resident_bytes':idx['resident_bytes'],
            'bytes_per_cartridge':round(idx['resident_bytes']/N,3)
        }
        all_results.append(row)
        print('TEST615_BUDGET',json.dumps(row),flush=True)
    for i in CASES:
        diagnostics.append({
            'mode':mode,'k':k,'weighted':weighted,'case':i,
            'group':'CAL' if i<12 else 'PREVIOUSLY_SEEN_HOLD',
            'target':int(TARGET[i]),
            'raw_rank':routes[i]['target_raw_rank'],
            'raw_candidates':routes[i]['raw_candidates'],
            'posting_visits':routes[i]['posting_visits']
        })

# First full-recall budget is a descriptive, retrospective diagnostic.
full12=[r for r in all_results if r['old_hold_native_r1']==12]
full24=[r for r in all_results if r['all_native_r1']==24]
best_recall=sorted(all_results,key=lambda r:(
    -r['old_hold_native_r1'],
    -r['cal_native_r1'],
    r['budget'],
    r['resident_bytes']
))[:10]

print('TEST615_FIRST_FULL_OLD_HOLD',json.dumps(
    sorted(full12,key=lambda r:(r['budget'],r['resident_bytes']))[:10]
),flush=True)
print('TEST615_FIRST_FULL_ALL24',json.dumps(
    sorted(full24,key=lambda r:(r['budget'],r['resident_bytes']))[:10]
),flush=True)
print('TEST615_BEST_RECALL',json.dumps(best_recall),flush=True)

# SOURCE CAPACITY EXTRAPOLATION: NOT REAL LARGE-BANK MEASUREMENT
# CSR resident bytes from measured 96-cartridge configurations.
projection=[]
for mode,k,weighted in CONFIGS:
    vocab=DIM*(2 if mode=='SIGNED' else 1)
    for nn in (100000,1000000,5000000):
        csr_bytes=int(nn*k*4+vocab*4+(vocab+1)*4)
        projection.append({
            'mode':mode,'k':k,'weighted':weighted,'cartridges':nn,
            'projected_csr_mib':round(csr_bytes/1048576,3),
            'projected_csr_bytes_per_cartridge':round(csr_bytes/nn,3),
            'scope':'CSR buffer size only; excludes real Qwen recall, query latency, cartridge payload, runtime'
        })
print('TEST615_CAPACITY_PROJECTION',json.dumps(projection),flush=True)

# INTEGRITY
if any(p.requires_grad for p in model.parameters()):
    raise RuntimeError('MODEL_FROZEN_REGRESSION')
if any(layer._forward_hooks or layer._forward_pre_hooks for layer in model.model.layers):
    raise RuntimeError('MODEL_HOOK_REGRESSION')
if (H1,H5)!=(12,12) or sum(int(x['answer_ok']) for x in PRE_AUDIT)!=12:
    raise RuntimeError('BASELINE_FINAL_REGRESSION')

result={
    'test':TEST,
    'source_sha256':got,
    'baseline':{'fram_hold_r1':12,'fram_hold_r5':12,'cut3_hold_correct':12},
    'human_units':HUMAN_UNITS,
    'configurations':CONFIGS,
    'budgets':BUDGETS,
    'all_results':all_results,
    'case_diagnostics':diagnostics,
    'full_old_hold':full12,
    'full_all24':full24,
    'best_recall':best_recall,
    'capacity_projection':projection,
    'limitations':[
        'Retrospective diagnostic on previously inspected panel; no fresh holdout',
        'No CUT3 generation replay for routed candidates',
        'No million-native-cartridge accuracy validation',
        'No measured large-scale disk I/O',
        'Index projections exclude runtime and cartridge files',
        'Source token counts are measured; CUT3 byte counts estimated'
    ]
}
path='/content/TEST615_RESULT.json' if os.path.isdir('/content') else '/tmp/TEST615_RESULT.json'
with open(path,'w',encoding='utf-8') as f:
    json.dump(result,f,ensure_ascii=False,indent=2)
print('TEST615_FINAL',json.dumps({
    'test':TEST,
    'baseline':result['baseline'],
    'configurations':len(CONFIGS),
    'budgets':len(BUDGETS),
    'tested':len(all_results),
    'full_old_hold_count':len(full12),
    'full_all24_count':len(full24),
    'minimum_full_old_hold_budget':min((r['budget'] for r in full12),default=None),
    'minimum_full_all24_budget':min((r['budget'] for r in full24),default=None),
    'best_recall':best_recall[:3],
    'human_units':HUMAN_UNITS,
    'output':path
},ensure_ascii=False),flush=True)

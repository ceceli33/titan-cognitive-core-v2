
# TEST616 — AKBASCORE MAM · REAL QWEN RESIDUAL BANK SCALING
# 96 -> 384 -> 1536 cartridges; frozen TEST610 backbone; native FRAM rerank.
# Added cartridges: synthetic factual text, genuinely encoded by frozen Qwen.
# Existing 24 questions are a previously seen diagnostic panel, NOT fresh holdout.
import os,sys,time,json,math,hashlib,urllib.request,gc
from collections import defaultdict
import numpy as np
import torch

TEST_ID=616
SRC='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/610.py'
SHA='a62611a4ffd91c2fcf03796c31da2af11ccd8141345655b562f3a614568af3e7'
req=urllib.request.Request(SRC,headers={'User-Agent':'AKBASCORE-TEST616'})
with urllib.request.urlopen(req,timeout=180) as f:raw=f.read()
got=hashlib.sha256(raw).hexdigest()
if got!=SHA:raise RuntimeError(f'SOURCE_SHA_MISMATCH {got}')
src=raw.decode('utf-8')
marker="print('[6/7] Demo acceptance"
if src.count(marker)!=1:raise RuntimeError('BASELINE_SPLIT_FAILED')
print('TEST616_SOURCE_LOCK',got,flush=True)
exec(compile(src.split(marker,1)[0],'610.py','exec'),globals())

# Baseline acceptance. Never modify model weights, CUT3 or FRAM.
if CFG!=(3,0,8.0) or len(KBASE)!=96 or len(QBASE)!=24:
    raise RuntimeError('BASELINE_CONFIGURATION_FAILED')
if (H1,H5)!=(12,12):raise RuntimeError('BASELINE_FRAM_FAILED')
if len(PRE_AUDIT)!=12 or sum(int(x['answer_ok']) for x in PRE_AUDIT)!=12:
    raise RuntimeError('BASELINE_CUT3_FAILED')
if any(p.requires_grad for p in model.parameters()):
    raise RuntimeError('MODEL_NOT_FROZEN')
print('TEST616_BASELINE_PASS',json.dumps({
    'native_cartridges':96,'fram_hold_r1':H1,'fram_hold_r5':H5,
    'cut3_correct':12,'cfg':CFG
}),flush=True)

DIM=3584
BANK_SIZES=(96,384,1536)
BUDGETS=(4,8,16,24,32,48,64,96,128,256)
K=48
MODES=('SIGNED',)
WEIGHTED=(False,True)
TARGET=np.asarray([int(PANEL[i]['target']) for i in range(24)],dtype=np.int32)
SEED=616
rng=np.random.default_rng(SEED)

# Preserve original FRAM feature frame: mean fitted on ORIGINAL 96 cartridges only.
# The new bank is transformed with the same fixed mean; no refitting on new data.
ORIGINAL_MU=torch.cat(KS[3],dim=0).mean(0)
if ORIGINAL_MU.shape!=(DIM,):raise RuntimeError('FRAME_SHAPE_FAILED')
for i in range(96):
    chk=apply(KS[3][i],(ORIGINAL_MU,None))
    if not torch.allclose(chk,KBASE[i],atol=2e-5,rtol=2e-5):
        raise RuntimeError(f'FRAME_RECONSTRUCTION_FAILED_{i}')
print('TEST616_FRAME_LOCK PASS',flush=True)

# Produce unique synthetic facts. These are NOT user documents or natural facts.
# Keep the original 96 cartridges and their source texts untouched.
SYL_A=('Aster','Borin','Cedar','Dorin','Elmar','Ferin','Galen','Havor',
       'Ivor','Joran','Kelan','Loran','Miran','Navor','Orlan','Pavor')
SYL_B=('vale','mere','ford','haven','crest','wick','mont','ridge',
       'brook','field','port','glen','stone','fall','bridge','wood')
LOC_A=('Amber','Azure','Copper','Ivory','Jade','Silver','Golden','Indigo',
       'Crimson','Olive','Pearl','Umber','Violet','Cobalt','Sable','Coral')
LOC_B=('Atrium','Depot','Gallery','Archive','Pavilion','Chamber',
       'Library','Vault','Annex','Station','Hall','Garden',
       'Observatory','Sanctum','Terrace','Workshop')
def synthetic_fact(j):
    a=SYL_A[j%16]
    b=SYL_B[(j//16)%16]
    c=LOC_A[(j//256)%16]
    d=LOC_B[(j//7)%16]
    return f'Archive record {j:05d}: {a}{b} is assigned to the {c} {d}; reference code R{j:05d}.'

# Capture genuinely new Qwen L3 residuals with the original source-prefix frame.
# Only L3 is retained for routing/FRAM. No fake CUT3 cartridge is claimed.
new_native=[]
new_tokens=[]
capture_seconds=[]
required=max(BANK_SIZES)-96
for j in range(required):
    fact=synthetic_fact(j)
    text_source=source(fact)
    begin=len(PREFIX)
    end=begin+len(fact)
    if text_source[begin:end]!=fact:raise RuntimeError('SOURCE_SPAN_FAILED')
    t0=time.perf_counter()
    residual,info=capture_residual(text_source,begin,end)
    elapsed=time.perf_counter()-t0
    z=apply(residual[3],(ORIGINAL_MU,None))
    if z.ndim!=2 or z.shape[1]!=DIM or not torch.isfinite(z).all():
        raise RuntimeError('NEW_RESIDUAL_FAILED')
    new_native.append(z)
    new_tokens.append(int(info['n']))
    capture_seconds.append(elapsed)
    if (j+1)%96==0 or j==required-1:
        print('TEST616_NATIVE_CAPTURE',j+1,'/',required,
              'elapsed_s',round(sum(capture_seconds),2),flush=True)
    del residual,z
    if (j+1)%256==0:
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()

ALL_K=list(KBASE)+new_native
if len(ALL_K)!=1536:raise RuntimeError('BANK_LENGTH_FAILED')
ALL_Q=[QBASE[i] for i in range(24)]

# Same mean-pooled router features used in TEST615; fixed original centering.
def unit_np(x):
    return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-12)
ORIGINAL_ROUTER_MEAN=np.stack([
    x.detach().float().mean(0).cpu().numpy() for x in KBASE
]).mean(0)
KEY_POOL=np.stack([
    x.detach().float().mean(0).cpu().numpy() for x in ALL_K
])
QUERY_POOL=np.stack([
    x.detach().float().mean(0).cpu().numpy() for x in ALL_Q
])
KX=unit_np(KEY_POOL-ORIGINAL_ROUTER_MEAN).astype(np.float32)
QX=unit_np(QUERY_POOL-ORIGINAL_ROUTER_MEAN).astype(np.float32)
if not np.isfinite(KX).all() or not np.isfinite(QX).all():
    raise RuntimeError('ROUTER_NONFINITE')

def make_index(keys,queries,k,mode):
    n=len(keys)
    ki=np.argpartition(np.abs(keys),-k,axis=1)[:,-k:].astype(np.int32)
    qi=np.argpartition(np.abs(queries),-k,axis=1)[:,-k:].astype(np.int32)
    if mode=='SIGNED':
        ki=ki+(keys[np.arange(n)[:,None],ki]<0).astype(np.int32)*DIM
        qi=qi+(queries[np.arange(len(queries))[:,None],qi]<0).astype(np.int32)*DIM
        vocab=2*DIM
    else:
        vocab=DIM
    flat=ki.ravel()
    freq=np.bincount(flat,minlength=vocab).astype(np.int32)
    offsets=np.empty(vocab+1,dtype=np.int32)
    offsets[0]=0
    np.cumsum(freq,out=offsets[1:])
    ordering=np.argsort(flat,kind='stable')
    postings=(ordering//k).astype(np.int32)
    return {
        'freq':freq,'offsets':offsets,'postings':postings,'qids':qi,
        'bytes':int(freq.nbytes+offsets.nbytes+postings.nbytes)
    }

def route(index,i,n,weighted):
    tally=defaultdict(float)
    visits=0
    t0=time.perf_counter()
    for coord in index['qids'][i]:
        v=int(coord)
        df=int(index['freq'][v])
        visits+=df
        weight=math.log1p((n+1)/(df+1)) if weighted else 1.0
        lo=int(index['offsets'][v])
        hi=int(index['offsets'][v+1])
        for j in index['postings'][lo:hi]:
            tally[int(j)]+=weight
    ordered=sorted(tally,key=lambda j:(-tally[j],j))
    elapsed_ms=(time.perf_counter()-t0)*1000
    return {
        'ordered':ordered,
        'raw_candidates':len(ordered),
        'posting_visits':visits,
        'route_ms':elapsed_ms,
        'target_raw_rank':ordered.index(int(TARGET[i]))+1 if int(TARGET[i]) in tally else n+1
    }

# Original native FRAM scoring. Cached only within this diagnostic test.
native_cache={}
def native_score(i,j):
    key=(int(i),int(j))
    if key not in native_cache:
        v=pair_scores(ALL_Q[i],ALL_K[j],8.0)
        v=float(v.detach().cpu().item()) if torch.is_tensor(v) else float(v)
        if not math.isfinite(v):raise RuntimeError('FRAM_NONFINITE')
        native_cache[key]=v
    return native_cache[key]

def rerank(i,ids,n):
    if not ids:return {'rank':n+1,'hit1':0,'hit5':0,'count':0}
    t=int(TARGET[i])
    scored=sorted(((j,native_score(i,j)) for j in ids),
                  key=lambda x:(-x[1],x[0]))
    order=[j for j,_ in scored]
    rank=order.index(t)+1 if t in order else n+1
    return {'rank':rank,'hit1':int(rank==1),'hit5':int(rank<=5),'count':len(ids)}

results=[]
case_details=[]
for n in BANK_SIZES:
    keys=KX[:n]
    print('TEST616_BANK_BEGIN',n,flush=True)
    for mode in MODES:
        index=make_index(keys,QX,K,mode)
        for weighted in WEIGHTED:
            routes=[route(index,i,n,weighted) for i in range(24)]
            for budget in BUDGETS:
                audits=[rerank(i,routes[i]['ordered'][:budget],n) for i in range(24)]
                row={
                    'bank_size':n,'mode':mode,'k':K,'weighted':weighted,
                    'budget':budget,
                    'cal_r1':sum(x['hit1'] for x in audits[:12]),
                    'previously_seen_hold_r1':sum(x['hit1'] for x in audits[12:]),
                    'all_r1':sum(x['hit1'] for x in audits),
                    'all_r5':sum(x['hit5'] for x in audits),
                    'candidate_hit':sum(int(TARGET[i] in routes[i]['ordered'][:budget]) for i in range(24)),
                    'raw_coverage':sum(r['target_raw_rank']<=n for r in routes),
                    'mean_raw_candidates':round(float(np.mean([r['raw_candidates'] for r in routes])),3),
                    'mean_posting_visits':round(float(np.mean([r['posting_visits'] for r in routes])),3),
                    'mean_reranked':round(float(np.mean([x['count'] for x in audits])),3),
                    'route_p50_ms':round(float(np.percentile([r['route_ms'] for r in routes],50)),4),
                    'route_p95_ms':round(float(np.percentile([r['route_ms'] for r in routes],95)),4),
                    'csr_bytes':index['bytes'],
                    'csr_mib':round(index['bytes']/1048576,4)
                }
                results.append(row)
                print('TEST616_RESULT',json.dumps(row),flush=True)
            for i in range(24):
                case_details.append({
                    'bank_size':n,'weighted':weighted,'case':i,
                    'target':int(TARGET[i]),
                    'raw_rank':routes[i]['target_raw_rank'],
                    'raw_candidates':routes[i]['raw_candidates'],
                    'posting_visits':routes[i]['posting_visits']
                })
    print('TEST616_BANK_END',n,flush=True)

# Optional full-bank FRAM diagnostic, limited to the original 24 questions.
# This establishes whether the full native FRAM itself survives distractors.
full_native=[]
for n in BANK_SIZES:
    t0=time.perf_counter()
    ranks=[]
    for i in range(24):
        target=int(TARGET[i])
        scores=[native_score(i,j) for j in range(n)]
        order=sorted(range(n),key=lambda j:(-scores[j],j))
        ranks.append(order.index(target)+1)
    full_native.append({
        'bank_size':n,
        'r1':sum(r==1 for r in ranks),
        'r5':sum(r<=5 for r in ranks),
        'ranks':ranks,
        'seconds':round(time.perf_counter()-t0,3),
        'scope':'Full native FRAM diagnostic; not routed and not new holdout'
    })
    print('TEST616_FULL_NATIVE',json.dumps(full_native[-1]),flush=True)

# Human-readable capacities: real token counts and theoretical BF16 payload.
original_tokens=[int(x.shape[0]) for x in KBASE]
all_tokens=original_tokens+new_tokens
capacity=[]
for n in BANK_SIZES:
    tokens=all_tokens[:n]
    capacity.append({
        'cartridges':n,
        'source_tokens_total':sum(tokens),
        'mean_tokens_per_cartridge':round(float(np.mean(tokens)),3),
        'min_tokens':min(tokens),
        'max_tokens':max(tokens),
        'cut3_payload_mib_if_materialized':round(sum(tokens)*15360/1048576,4),
        'new_cartridges_have_cut3_files':False,
        'words':'NOT_MEASURED'
    })
    print('TEST616_CAPACITY',json.dumps(capacity[-1]),flush=True)

# Explicit limitations: no native CUT3 construction for new distractors.
# Only L3 residual representations of synthetic texts are materialized.
if any(p.requires_grad for p in model.parameters()):
    raise RuntimeError('FINAL_FROZEN_CHECK_FAILED')
if any(layer._forward_hooks or layer._forward_pre_hooks for layer in model.model.layers):
    raise RuntimeError('FINAL_HOOK_CHECK_FAILED')
if (H1,H5)!=(12,12):raise RuntimeError('FINAL_BASELINE_CHECK_FAILED')

result={
    'test':TEST_ID,
    'source_sha256':got,
    'baseline':{'fram_hold_r1':12,'fram_hold_r5':12,'cut3_hold_correct':12},
    'banks':BANK_SIZES,
    'real_model_new_residuals':required,
    'synthetic_fact_generator':'deterministic synthetic archive records',
    'frame':'Original 96-cartridge fixed mean; no bank refitting',
    'router':'TEST615 SIGNED k48, weighted false/true',
    'results':results,
    'case_details':case_details,
    'full_native':full_native,
    'capacity':capacity,
    'capture_seconds':round(sum(capture_seconds),3),
    'limitations':[
        'Additional records are synthetic facts encoded by the real frozen Qwen model',
        'Existing 24 questions are previously seen, not independent holdout',
        'New records have real L3 residuals, not materialized CUT3 cartridge files',
        'CUT3 generation correctness established only for original baseline',
        'No disk loading or end-to-end inference latency benchmark',
        'Routing timing is Python posting traversal and sort only',
        'FRAM rerank timing not included in routing timing',
        'Index bytes exclude Python runtime, model, full residuals and disk payload'
    ]
}
path='/content/TEST616_RESULT.json' if os.path.isdir('/content') else '/tmp/TEST616_RESULT.json'
with open(path,'w',encoding='utf-8') as f:
    json.dump(result,f,ensure_ascii=False,indent=2)
print('TEST616_FINAL',json.dumps({
    'test':TEST_ID,
    'baseline':result['baseline'],
    'banks':BANK_SIZES,
    'new_qwen_encoded_records':required,
    'full_native':full_native,
    'best_per_bank':[
        sorted([r for r in results if r['bank_size']==n],
               key=lambda r:(-r['all_r1'],r['budget'],r['csr_bytes']))[0]
        for n in BANK_SIZES
    ],
    'capacity':capacity,
    'output':path
},ensure_ascii=False),flush=True)

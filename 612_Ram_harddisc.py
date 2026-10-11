# TEST612 — AKBASCORE MAM · RAM BOWL MULTI-MECHANISM SEARCH · SINGLE COLAB CELL · NO UI
# Frozen TEST610 source and CUT3 audit are executed unchanged. All new methods are independent readouts.
import os,sys,hashlib,urllib.request,time,math,json,collections,gc
import numpy as np
SRC='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/610.py'
SHA='a62611a4ffd91c2fcf03796c31da2af11ccd8141345655b562f3a614568af3e7'
req=urllib.request.Request(SRC,headers={'User-Agent':'AKBASCORE-TEST612'})
with urllib.request.urlopen(req,timeout=180) as f:raw=f.read()
got=hashlib.sha256(raw).hexdigest()
if got!=SHA:raise RuntimeError(f'FAIL_CLOSED 610.py SHA expected={SHA} got={got}')
src=raw.decode('utf-8');marker="print('[6/7] Demo acceptance"
if src.count(marker)!=1:raise RuntimeError('FAIL_CLOSED 610.py split marker changed')
print('TEST612 BASELINE_LOCK',got,flush=True)
exec(compile(src.split(marker,1)[0],'610.py','exec'),globals())
import torch
if CFG!=(3,0,8.0) or len(KBASE)!=96 or set(QBASE)!=set(range(24)):raise RuntimeError('FRAM FEATURE LOCK FAILED')
if (H1,H5)!=(12,12) or len(PRE_AUDIT)!=12 or sum(int(x['answer_ok']) for x in PRE_AUDIT)!=12:raise RuntimeError('CUT3 BASELINE REGRESSION')
SEED=612612;RG=np.random.default_rng(SEED);N=96;DIM=3584;CASES=list(range(24));HOLD=list(range(12,24));TARGET=np.asarray([int(PANEL[i]['target']) for i in CASES]);TOP=5
# The original FRAM score is the established upper reference, NOT a new index or constant-time lookup.
reference=np.asarray([[pair_scores(QBASE[i],KBASE[j],8.0) for j in range(N)] for i in CASES],dtype=np.float32)
# Exact unmodified token-level FRAM scores are diagnostic only. No label-driven fitting or tuning.
def unit(a):return a/(np.linalg.norm(a,axis=-1,keepdims=True)+1e-12)
K0=np.stack([x.detach().float().mean(0).cpu().numpy() for x in KBASE]);Q0=np.stack([QBASE[i].detach().float().mean(0).cpu().numpy() for i in CASES]);MU=K0.mean(0);KX=unit(K0-MU).astype(np.float32);QX=unit(Q0-MU).astype(np.float32)
if KX.shape!=(96,3584) or QX.shape!=(24,3584):raise RuntimeError('FEATURE SHAPE')
if not np.isfinite(KX).all() or not np.isfinite(QX).all():raise RuntimeError('NONFINITE FEATURES')
# All choices fixed before reading holdout scores; calibration 0..11 may be reported but never used to tune methods.
methods={};details={};times={}
def record(name,score,elapsed,bytes_est,search_class,notes,extra=None):
    a=np.asarray(score,dtype=np.float32)
    if a.shape!=(24,96) or not np.isfinite(a).all():raise RuntimeError(f'{name} INVALID SCORES {a.shape}')
    methods[name]=a;times[name]=float(elapsed);details[name]=dict(bytes=int(bytes_est),search_class=search_class,notes=notes,extra=extra or {})
# 0: original exact FRAM, exhaustive token-level reference.
record('FRAM_TOKEN_EXACT',reference,0,0,'O(N) tokenwise reference','Original TEST610 score, not a RAM-bowl solution')
# 1: direct mean residual cosine; tests whether a compressed query retains semantics.
t=time.perf_counter();score=QX@KX.T;record('MEAN_COSINE',score,time.perf_counter()-t,KX.nbytes,'O(N) dense scan','No additional projection')
# 2: random dense projection, signed dot, with dimension sweeps.
for d in (128,512,2048):
    t=time.perf_counter();P=RG.standard_normal((DIM,d),dtype=np.float32)/math.sqrt(d);a=unit(KX@P);b=unit(QX@P);score=b@a.T
    record(f'RANDOM_DENSE_{d}',score,time.perf_counter()-t,P.nbytes+a.nbytes,'O(N) dense scan','Projection seeded and label-free');del P,a,b
# 3: signed random hyperplanes, Hamming similarity via packed binary codes.
for bits in (256,1024,4096):
    t=time.perf_counter();P=RG.standard_normal((DIM,bits),dtype=np.float32);ka=np.packbits(KX@P>=0,axis=1);qa=np.packbits(QX@P>=0,axis=1);s=np.empty((24,N),dtype=np.float32)
    for i in CASES:s[i]=bits-np.unpackbits(np.bitwise_xor(ka,qa[i]),axis=1).sum(axis=1)
    record(f'SIGN_HYPERPLANE_{bits}',s,time.perf_counter()-t,ka.nbytes+P.nbytes,'O(N) binary scan','Packed Hamming; projection memory included');del P,ka,qa
# 4: fixed-coordinate ternary / sign codes, no learned projector.
for dims in (512,2048):
    t=time.perf_counter();ix=RG.choice(DIM,dims,replace=False);ka=np.sign(KX[:,ix]);qa=np.sign(QX[:,ix]);s=qa@ka.T
    record(f'COORDINATE_SIGN_{dims}',s,time.perf_counter()-t,ka.nbytes+ix.nbytes,'O(N) ternary scan','Random coordinate subset');del ka,qa
# 5: feature superposition into a single shared matrix. Decode is O(D*N): does not solve scaling.
for d in (128,512):
    t=time.perf_counter();P=RG.standard_normal((DIM,d),dtype=np.float32)/math.sqrt(d);ka=unit(KX@P);qa=unit(QX@P)
    # A = sum outer(key, ID-code). Fixed random codes; decode by code correlations.
    codes=unit(RG.standard_normal((N,d),dtype=np.float32));field=ka.T@codes;read=(qa@field)@codes.T
    record(f'SUPERPOSITION_{d}',read,time.perf_counter()-t,P.nbytes+field.nbytes+codes.nbytes,'O(N) ID decode','Shared superposition field; cross-talk measured, no false constant-time claim');del P,ka,qa,codes,field
# 6: query-key nonlinear random Fourier map (RBF approximation), label-free.
for d in (512,2048):
    t=time.perf_counter();P=RG.standard_normal((DIM,d),dtype=np.float32);phase=RG.uniform(0,2*math.pi,d).astype(np.float32);ka=np.cos(KX@P+phase);qa=np.cos(QX@P+phase);s=qa@ka.T/d
    record(f'FOURIER_FIELD_{d}',s,time.perf_counter()-t,P.nbytes+ka.nbytes+phase.nbytes,'O(N) dense scan','RFF nonlinear field; no tuned bandwidth');del P,ka,qa
# 7: multi-table random-hyperplane LSH: query probes buckets, candidates only are rescored.
# No full-bank scoring at query; this is a legitimate index, but its cost and miss rate must be measured.
for nbits,ntables in ((8,8),(10,16),(12,24)):
    t=time.perf_counter();P=RG.standard_normal((ntables,DIM,nbits),dtype=np.float32);tables=[];masks=(1<<np.arange(nbits,dtype=np.int64));kb=np.empty((ntables,N),dtype=np.int64);qb=np.empty((ntables,24),dtype=np.int64)
    for j in range(ntables):
        kb[j]=((KX@P[j]>=0).astype(np.int64)*masks).sum(1);qb[j]=((QX@P[j]>=0).astype(np.int64)*masks).sum(1)
        table=collections.defaultdict(list)
        for ci,b in enumerate(kb[j]):table[int(b)].append(ci)
        tables.append(table)
    s=np.full((24,N),-1e9,dtype=np.float32);candidate_counts=[];candidate_lists=[]
    for i in CASES:
        cand=set()
        for j in range(ntables):cand.update(tables[j].get(int(qb[j,i]),()))
        candidate_lists.append(sorted(cand));candidate_counts.append(len(cand))
        if cand:
            ix=np.fromiter(sorted(cand),dtype=np.int32);s[i,ix]=QX[i]@KX[ix].T
    record(f'LSH_{nbits}B_{ntables}T',s,time.perf_counter()-t,P.nbytes+kb.nbytes,'O(tables*bits + candidates*D)','True indexed candidate retrieval; no exhaustive fallback',dict(candidate_counts=candidate_counts,candidate_lists=candidate_lists));del P,tables,kb,qb
# 8: sparse inverted top-|activation| coordinates; posting union, candidate-only rerank.
for k in (16,64,256):
    t=time.perf_counter();ki=np.argpartition(np.abs(KX),-k,axis=1)[:,-k:];qi=np.argpartition(np.abs(QX),-k,axis=1)[:,-k:];post=collections.defaultdict(list)
    for ci in range(N):
        for v in ki[ci]:post[int(v)].append(ci)
    s=np.full((24,N),-1e9,dtype=np.float32);counts=[];lists=[];visits=[]
    for i in CASES:
        cand=set();visit=0
        for v in qi[i]:cand.update(post.get(int(v),()));visit+=len(post.get(int(v),()))
        ix=np.fromiter(sorted(cand),dtype=np.int32);lists.append(ix.tolist());counts.append(len(ix));visits.append(visit)
        if len(ix):s[i,ix]=QX[i]@KX[ix].T
    record(f'INVERTED_TOPABS_{k}',s,time.perf_counter()-t,ki.nbytes+sum(len(x)*4 for x in post.values()),'O(posting visits + candidates*D)','Actual sparse inverted field; no full scan',dict(candidate_counts=counts,posting_visits=visits,candidate_lists=lists));del ki,qi,post
# 9: full covariance / associative bowl. This is a global field but cannot identify an ID without a readout.
# Its denoised output is matched to all keys, and O(N) cost is disclosed.
for ridge in (0.1,1.0):
    t=time.perf_counter();gram=KX@KX.T;solve=np.linalg.solve(gram+ridge*np.eye(N,dtype=np.float32),KX@QX.T);s=solve.T
    record(f'RIDGE_ASSOCIATIVE_{ridge}',s,time.perf_counter()-t,gram.nbytes+KX.nbytes,'O(N^2) precompute + O(N*D) query','Dual ridge associative recall; not an index')
# Exact ranking: stable tie break by cartridge ID. Indexes with fewer than 5 candidates cannot claim top5 success.
def ranks(a):return np.argsort(-a,axis=1,kind='stable')
summary=[]
for name,s in methods.items():
    order=ranks(s);rank=[int(np.flatnonzero(order[i]==TARGET[i])[0])+1 if s[i,TARGET[i]]>-1e8 else N+1 for i in CASES];d=details[name];extra=d['extra'];counts=extra.get('candidate_counts')
    row=dict(method=name,cal_r1=sum(rank[i]==1 for i in range(12)),cal_r5=sum(rank[i]<=5 for i in range(12)),hold_r1=sum(rank[i]==1 for i in HOLD),hold_r5=sum(rank[i]<=5 for i in HOLD),all_r1=sum(r==1 for r in rank),all_r5=sum(r<=5 for r in rank),hold_mean_rank=float(np.mean([rank[i] for i in HOLD])),bytes=d['bytes'],setup_and_eval_seconds=round(times[name],3),search_class=d['search_class'],notes=d['notes'])
    if counts is not None:
        row.update(hold_target_candidate=sum(int(TARGET[i] in extra['candidate_lists'][i]) for i in HOLD),mean_candidates=float(np.mean(counts)),max_candidates=int(max(counts)),candidate_counts=counts)
        if 'posting_visits' in extra:row['mean_posting_visits']=float(np.mean(extra['posting_visits']))
    summary.append(row);print('TEST612_METHOD',json.dumps(row,ensure_ascii=False),flush=True)
# Rank agreement with original FRAM and candidate coverage are kept separate from actual CUT3 answering.
# No new method's output is injected into the CUT3 engine in this test.
if any(p.requires_grad for p in model.parameters()):raise RuntimeError('MODEL NO LONGER FROZEN')
if any(layer._forward_hooks or layer._forward_pre_hooks for layer in model.model.layers):raise RuntimeError('RESIDUAL HOOK LEFT INSTALLED')
result=dict(test=612,source_sha256=got,scope='96 genuine Qwen H3 cartridges, 24 genuine questions; independent readout comparison only; no million-cartridge or end-to-end new-readout claim',baseline=dict(fram_hold_r1=int(H1),fram_hold_r5=int(H5),cut3_hold_correct=sum(int(x['answer_ok']) for x in PRE_AUDIT)),seed=SEED,methods=summary,interpretation='FRAM reference is O(N); methods marked O(N) or O(N*D) do not solve scaling. Indexed methods must show holdout candidate coverage and recall. No holdout tuning.')
path='/content/TEST612_RESULT.json' if os.path.isdir('/content') else '/tmp/TEST612_RESULT.json'
with open(path,'w',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print('TEST612_FINAL',json.dumps(dict(test=612,baseline=result['baseline'],methods=len(summary),best_hold_r1=max(x['hold_r1'] for x in summary),output=path),ensure_ascii=False),flush=True)

# TEST624 — 1453 PHYSICAL CUT3 / NATIVE IVF / TOP2_SOFT / FRAM / DISK E2E
# One Colab cell. Frozen TEST610 engine; no lexical router, no training.
import os,sys,time,json,random,hashlib,urllib.request,subprocess,importlib.util,gc,math
from pathlib import Path
import numpy as np,torch
for pkg,mod in [('faiss-cpu','faiss')]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,'-m','pip','install','-q',pkg])
import faiss
faiss.omp_set_num_threads(4)
TEST='624';SEED=6241453;TOTAL=1453;ORIGINAL=96;SYNTH=TOTAL-ORIGINAL;RANK=128;NLIST=128;BETA=8.;SAMPLE=2048
SOURCE='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/610.py';LOCK='a62611a4ffd91c2fcf03796c31da2af11ccd8141345655b562f3a614568af3e7'
ROOT=Path('/content/AKBASCORE_TEST624');POOL=ROOT/'cut3_pool';POOL.mkdir(parents=True,exist_ok=True);OUT=ROOT/'TEST624_RESULT.json'
# PREDECLARED: nprobe, native-token neighbors per question position, TOP2 candidate budget.
SETTINGS=[('NARROW',2,16,16),('MEDIUM',8,64,16),('WIDE',32,128,32)]
NQUERY=48;ANSWER_CASES=24;ROUNDTRIP_CASES=8
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
def stamp():return round(time.perf_counter()-T0,2)
def log(*a):print(*a,flush=True)
def ms(t):return (time.perf_counter()-t)*1000
def sha_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def pct(rows,key,p=50):return round(float(np.percentile([r[key] for r in rows],p)),4) if rows else None
T0=time.perf_counter();log('='*110);log('TEST624 / 1453 PHYSICAL CUT3 / NATIVE GEOMETRIC INDEX');log('='*110)
src=urllib.request.urlopen(urllib.request.Request(SOURCE,headers={'User-Agent':'AKBASCORE-TEST624'}),timeout=120).read()
assert hashlib.sha256(src).hexdigest()==LOCK
boundary="print('[6/7] Demo acceptance";assert src.decode().count(boundary)==1
exec(compile(src.decode().split(boundary,1)[0],'610.py','exec'),globals())
assert len(CAR)==ORIGINAL and tuple(CFG)==(3,0,8.0) and H1==12 and H5==12
WB=weight_sha();SB=sentinel();assert not any(p.requires_grad for p in model.parameters())
log('BASELINE_LOCK',{'source_sha256':LOCK,'original_cartridges':96,'fram_hold_r1':H1,'fram_hold_r5':H5})
# 1357 NEW RECORDS: physically write all; no reference to query labels in PCA or routing.
rg=random.Random(SEED);letters='ABCDEFGHJKLMNPQRSTUVWXYZ'
def word(n):return ''.join(rg.choice(letters) for _ in range(n)).capitalize()
NEW=[]
for j in range(SYNTH):
    country=f'ZX{624000+j:06d}';city=word(10);fact=f'The current capital of {country} is {city}.'
    ids=tok(source(fact),add_special_tokens=False).input_ids
    assert ids[:P]==PIDS
    ci=len(CAR);CAR.append(dict(world=100000+j,type='CURRENT',fact=fact,gold=city,body=ids[P:]))
    NEW.append(dict(id=ci,country=country,city=city))
assert len(CAR)==TOTAL and len(NEW)==SYNTH
# Original frozen L3 normalization from TEST610; original 96 must remain bit-identical.
FIT=transform_fit(torch.cat(KS[3],dim=0),0)
K=[apply(x,FIT).detach().float().cpu().contiguous() for x in KS[3]]
assert all(torch.equal(a,b.float().cpu()) for a,b in zip(K,KBASE))
MANIFEST=[];build_times=[]
def path(ci):return POOL/f'cut3_{ci:05d}.pt'
def disk_write(ci):
    t=time.perf_counter()
    with torch.no_grad():mem=init(ci,3)
    data=[(a.detach().cpu().contiguous(),b.detach().cpu().contiguous()) for a,b in mem]
    assert len(data)==NL
    torch.save(data,path(ci))
    size=path(ci).stat().st_size;digest=sha_file(path(ci));dt=ms(t)
    del mem,data
    return dict(id=ci,file=path(ci).name,bytes=size,sha256=digest),dt
log('[1/7] PHYSICAL CUT3 WRITE: 1453 real independent files')
for ci in range(TOTAL):
    if ci>=ORIGINAL:
        fact=CAR[ci]['fact'];r,_=capture_residual(source(fact),len(PREFIX),len(PREFIX)+len(fact))
        K.append(apply(r[3],FIT).detach().float().cpu().contiguous())
    rec,dt=disk_write(ci);MANIFEST.append(rec);build_times.append(dt)
    if (ci+1)%100==0 or ci+1==TOTAL:
        log('PHYSICAL_WRITE',ci+1,'/',TOTAL,'disk_GB',round(sum(x['bytes'] for x in MANIFEST)/1e9,3),'elapsed_s',stamp())
    if (ci+1)%32==0:gc.collect();torch.cuda.empty_cache() if torch.cuda.is_available() else None
assert len(K)==TOTAL and len(MANIFEST)==TOTAL and all(path(i).exists() for i in range(TOTAL))
with open(ROOT/'CUT3_MANIFEST.json','w') as f:json.dump(MANIFEST,f,indent=2)
# Label-free PCA and native activation bank. Exact L3 vectors remain for TOP2/FRAM reranking.
log('[2/7] LABEL-FREE PCA128 + FAISS IVF128')
LENS=np.array([len(k) for k in K],dtype=np.int32);OWNER=np.repeat(np.arange(TOTAL,dtype=np.int32),LENS)
TOK=torch.cat(K,dim=0).contiguous();NTOK=len(TOK);DIM=TOK.shape[1];assert DIM==3584
t=time.perf_counter();trainids=np.random.default_rng(SEED).choice(NTOK,min(SAMPLE,NTOK),replace=False)
TRAIN=TOK[torch.from_numpy(trainids)].contiguous();CENTER=TRAIN.mean(0)
U,S,V=torch.pca_lowrank(TRAIN-CENTER,q=RANK+16,center=False,niter=2);BASIS=V[:,:RANK].contiguous()
assert BASIS.shape==(DIM,RANK)
del U,S,V,TRAIN;log('PCA_MS',round(ms(t),2))
t=time.perf_counter();PROJ=np.empty((NTOK,RANK),dtype=np.float32)
for a in range(0,NTOK,1024):
    b=min(a+1024,NTOK);PROJ[a:b]=((TOK[a:b]-CENTER)@BASIS).numpy()
log('PROJECTION_MS',round(ms(t),2))
t=time.perf_counter();INDEX=faiss.IndexIVFFlat(faiss.IndexFlatIP(RANK),RANK,NLIST,faiss.METRIC_INNER_PRODUCT)
INDEX.train(np.ascontiguousarray(PROJ[trainids]));INDEX.add(np.ascontiguousarray(PROJ));assert INDEX.ntotal==NTOK
LISTSIZE=np.array([INDEX.invlists.list_size(i) for i in range(NLIST)],dtype=np.int64);assert LISTSIZE.sum()==NTOK
INDEX_BYTES=len(faiss.serialize_index(INDEX));INDEX_BUILD_MS=ms(t);del PROJ
log('INDEX',{'cartridges':TOTAL,'native_positions':NTOK,'ivf_lists':NLIST,'index_build_ms':round(INDEX_BUILD_MS,2)})
# Fresh fixed synthetic queries; query text is prepared independently of gold lookup during retrieval.
log('[3/7] FROZEN QUERY PANEL: 48 NEW SYNTHETIC CARTRIDGE QUESTIONS')
selected=random.Random(SEED+1).sample(NEW,NQUERY)
PANEL624=[dict(case=i,question=f"What is the current capital of {m['country']}?",target=m['id'],gold=m['city']) for i,m in enumerate(selected)]
QUEST=[]
for row in PANEL624:
    r,_=question_rep(row['question']);QUEST.append(apply(r[3],FIT).detach().float().cpu().contiguous())
# No gold/target enters the following route function.
def top2(q,k):
    sim=q@k.T;v=torch.topk(sim,k=min(2,len(k)),dim=1).values.double()
    return float(((torch.logsumexp(v*BETA,dim=1)-math.log(len(k)))/BETA).mean().item())
def native(q,k):return float(pair_scores(q,k,BETA))
def route(q,nprobe,token_k,budget):
    t=time.perf_counter();qp=np.ascontiguousarray(((q-CENTER)@BASIS).numpy(),dtype=np.float32)
    INDEX.nprobe=nprobe;_,cells=INDEX.quantizer.search(qp,nprobe);_,hits=INDEX.search(qp,token_k)
    touched={int(OWNER[h]) for row in hits for h in row if h>=0}
    posting_visits=int(sum(int(LISTSIZE[c]) for row in cells for c in row if c>=0))
    faiss_ms=ms(t)
    t=time.perf_counter();geometry={ci:top2(q,K[ci]) for ci in touched}
    ordered=sorted(touched,key=lambda ci:(-geometry[ci],ci));candidates=ordered[:budget];top2_ms=ms(t)
    t=time.perf_counter();ns={ci:native(q,K[ci]) for ci in candidates}
    ranked=sorted(candidates,key=lambda ci:(-ns[ci],ci));fram_ms=ms(t)
    return ranked,dict(touched=len(touched),touched_fraction=len(touched)/TOTAL,
        native_positions_scanned=int(sum(LENS[ci] for ci in touched)),posting_visits=posting_visits,
        candidates=len(candidates),faiss_ms=faiss_ms,top2_ms=top2_ms,fram_ms=fram_ms,
        address_ms=faiss_ms+top2_ms+fram_ms)
RESULTS={};DETAIL={}
log('[4/7] FIXED IVF SELECTIVITY SWEEP — NO LABELS IN ROUTING')
for name,npb,tk,budget in SETTINGS:
    rows=[]
    for i,(q,case) in enumerate(zip(QUEST,PANEL624)):
        ranked,stats=route(q,npb,tk,budget);target=case['target']
        rows.append(dict(case=i,target=target,selected=ranked[0] if ranked else None,
            top5=ranked[:5],correct=int(bool(ranked) and ranked[0]==target),
            recall5=int(target in ranked[:5]),candidate_hit=int(target in ranked),**stats))
    DETAIL[name]=rows
    RESULTS[name]=dict(setting=dict(nprobe=npb,token_k=tk,budget=budget),questions=NQUERY,
        selected_correct=sum(r['correct'] for r in rows),recall5=sum(r['recall5'] for r in rows),
        gold_in_candidates=sum(r['candidate_hit'] for r in rows),mean_touched=round(float(np.mean([r['touched'] for r in rows])),3),
        mean_touched_fraction=round(float(np.mean([r['touched_fraction'] for r in rows])),6),
        mean_posting_visits=round(float(np.mean([r['posting_visits'] for r in rows])),3),
        address_p50_ms=pct(rows,'address_ms'),address_p95_ms=pct(rows,'address_ms',95),
        faiss_p50_ms=pct(rows,'faiss_ms'),top2_p50_ms=pct(rows,'top2_ms'),fram_p50_ms=pct(rows,'fram_ms'))
    log('SWEEP',name,json.dumps(RESULTS[name]))
# E2E disk: locked narrow and wide arms, fixed 24 cases. No oracle cartridge substitutions.
log('[5/7] PHYSICAL DISK LOAD + FROZEN CUT3 ANSWER: NARROW AND WIDE')
E2E={}
for name in ('NARROW','WIDE'):
    rows=[]
    for i in range(ANSWER_CASES):
        case=PANEL624[i];chosen=DETAIL[name][i]['selected'];row=dict(case=i,selected=chosen,target=case['target'],selected_correct=chosen==case['target'])
        if chosen is None:row.update(answer_correct=False,error='NO_CANDIDATE');rows.append(row);continue
        try:
            t=time.perf_counter();p=path(chosen)
            assert p.exists() and sha_file(p)==MANIFEST[chosen]['sha256']
            stored=torch.load(p,map_location='cpu',weights_only=True)
            mem=tuple((a.to(DEVICE),b.to(DEVICE)) for a,b in stored);row['disk_load_verify_ms']=ms(t)
            t=time.perf_counter()
            with torch.no_grad():response=answer(mem,case['question'])
            row['answer_ms']=ms(t);row['answer']=str(response);row['answer_correct']=bool(hit(response,case['gold']))
            del mem,stored
        except Exception as exc:row['answer_correct']=False;row['error']=repr(exc)
        rows.append(row);log('DISK_ANSWER',name,i,'SELECT',chosen,'GOLD',case['target'],'OK',int(row['answer_correct']))
    E2E[name]=dict(cases=ANSWER_CASES,selected_correct=sum(x['selected_correct'] for x in rows),
        answer_correct=sum(x['answer_correct'] for x in rows),errors=sum('error' in x for x in rows),
        disk_load_verify_p50_ms=pct([x for x in rows if 'disk_load_verify_ms' in x],'disk_load_verify_ms'),
        answer_p50_ms=pct([x for x in rows if 'answer_ms' in x],'answer_ms'),rows=rows)
    log('E2E',name,{k:v for k,v in E2E[name].items() if k!='rows'})
# Physical roundtrip exact equality on fixed samples; SHA verified on all files during manifest pass.
log('[6/7] INTEGRITY / ROUNDTRIP / WEIGHTS / HOOKS')
rt_ids=sorted(random.Random(SEED+2).sample(range(TOTAL),ROUNDTRIP_CASES));RT=[]
for ci in rt_ids:
    with torch.no_grad():fresh=init(ci,3)
    saved=torch.load(path(ci),map_location='cpu',weights_only=True)
    equal=len(fresh)==len(saved) and all(torch.equal(a.detach().cpu(),x) and torch.equal(b.detach().cpu(),y) for (a,b),(x,y) in zip(fresh,saved))
    RT.append(dict(id=ci,bitwise_equal=bool(equal)));del fresh,saved
manifest_ok=all(path(x['id']).exists() and path(x['id']).stat().st_size==x['bytes'] and sha_file(path(x['id']))==x['sha256'] for x in MANIFEST)
hooks=[]
for li,l in enumerate(model.model.layers):
    for label,m in [('layer',l),('attn',l.self_attn),('q_proj',l.self_attn.q_proj)]:
        if m._forward_hooks or m._forward_pre_hooks:hooks.append([li,label])
INTEGRITY=dict(source_sha_lock=hashlib.sha256(src).hexdigest()==LOCK,
    weight_unchanged=weight_sha()==WB,sentinel_unchanged=sentinel()==SB,
    zero_trainable=not any(p.requires_grad for p in model.parameters()),
    no_remaining_hooks=not hooks,manifest_1453_sha_verified=manifest_ok,
    roundtrip_8_bitwise=all(x['bitwise_equal'] for x in RT))
DISK_BYTES=sum(x['bytes'] for x in MANIFEST);NATIVE_BYTES=TOK.numel()*TOK.element_size()
INDEX_AUX=INDEX_BYTES+BASIS.numel()*BASIS.element_size()+CENTER.numel()*CENTER.element_size()+OWNER.nbytes
STORAGE=dict(cartridges=TOTAL,disk_pool_bytes=DISK_BYTES,native_activation_bytes=NATIVE_BYTES,
    faiss_index_bytes=INDEX_BYTES,pca_basis_bytes=BASIS.numel()*BASIS.element_size(),
    pca_center_bytes=CENTER.numel()*CENTER.element_size(),owner_bytes=OWNER.nbytes,
    index_auxiliary_bytes=INDEX_AUX,index_plus_native_activation_bytes=INDEX_AUX+NATIVE_BYTES,
    excludes_model_ram_vram=True,excludes_python_allocator_overhead=True)
FINAL=dict(test=TEST,status='COMPLETE' if all(INTEGRITY.values()) else 'INTEGRITY_FAILURE',
    architecture='Frozen Qwen + native L3 + PCA128 + FAISS IVF128 + touched TOP2_SOFT + locked FRAM + physical CUT3',
    source_sha256=LOCK,settings=SETTINGS,storage=STORAGE,index=dict(native_positions=NTOK,dimension=DIM,nlist=NLIST),
    retrieval=RESULTS,disk_answers=E2E,integrity=INTEGRITY,roundtrip=RT,
    physical_manifest=str(ROOT/'CUT3_MANIFEST.json'),per_query=DETAIL,elapsed_seconds=stamp(),
    limitations=['Synthetic new records use one capital-of-country template.',
    'Queries are newly sampled from records created in this test, but the question template is not independent.',
    'No independent evidence for million-cartridge scalability.',
    'Native activation vectors remain in RAM; selectivity and touched fractions are measured, not assumed.',
    'Disk reads may be served by OS page cache; no cold-SSD latency claim.',
    'Measured RAM buffer bytes exclude frozen model, Python and framework allocator overhead.',
    'PCA/index construction and disk-writing costs are not included in per-question address latency.',
    'E2E uses only 24 fixed synthetic questions and does not replace the original multi-family 96-card baseline.'])
with open(OUT,'w') as f:json.dump(FINAL,f,ensure_ascii=False,indent=2,default=str)
log('[7/7] FINAL');log('TEST624_STORAGE',json.dumps(STORAGE));log('TEST624_RETRIEVAL',json.dumps(RESULTS));
log('TEST624_DISK_ANSWER',{k:{a:b for a,b in v.items() if a!='rows'} for k,v in E2E.items()});log('TEST624_INTEGRITY',json.dumps(INTEGRITY))
log('TEST624_OUTPUT',str(OUT));log('TEST624_STATUS',FINAL['status']);log('TEST624_ELAPSED_S',stamp())
if not all(INTEGRITY.values()):raise RuntimeError('TEST624_INTEGRITY_FAILURE')

# TEST607 — AKBASCORE MAM-BÇ · FRAME-MATCHED RESIDUAL ACCESS + POSITIVE RANDOM FEATURES SUPERPOSITION + CUT3 AUDIT
# Frozen TEST602 backbone; exact residual oracle, positive exponential random features, coded superposition, unchanged CUT3.
import os,sys,subprocess,importlib.util,urllib.request,hashlib,json,time,math,random,inspect
for m in ('torch','transformers','accelerate'):
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,'-m','pip','install','-q',m])
import torch,numpy as np,transformers
URL='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/'
BLOBS={'566.py':'b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71','575.py':'569c92c61224ea408cdb6e81dbe45dde06dc05fd'}
def fetch_locked(name):
    with urllib.request.urlopen(urllib.request.Request(URL+name,headers={'User-Agent':'AKBASCORE-TEST607'}),timeout=120) as f:b=f.read()
    got=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
    if got!=BLOBS[name]:raise RuntimeError(f'SOURCE MISMATCH {name}: {got}')
    print('SOURCE_VERIFIED',name,got,flush=True);return b.decode()
s575=fetch_locked('575.py');s566=fetch_locked('566.py')
if 'BASE_BLOB="b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71"' not in s575:raise RuntimeError('575 LINK MISMATCH')
MARK='# TEST566 ADDITIONS — original TEST564 checkpoint/init/append/answer remain unchanged.'
if s566.count(MARK)!=1:raise RuntimeError('566 BOUNDARY MISMATCH')
exec(compile(s566.split(MARK,1)[0],'566.py','exec'),globals())
TEST='607';START=time.perf_counter();LAYERS=[3,8,16,24];RANKS=[0,64,128];BETAS=[0.0,8.0];TRAIN=list(range(12));HOLD=list(range(12,24));EXPECTED='b441b005d826d45f2778291696ac5ae22dfb3421defb37c539904d8d38f93786';OUT='/content/AKBASCORE_TEST607_SUPERPOSITION_SCALING.json';REC={'checks':[],'calibration':[],'cases':[],'pilot':[]}
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def check(name,ok,**info):
    REC['checks'].append(dict(name=name,ok=bool(ok),**info));print('PASS' if ok else 'FAIL',name,info,flush=True)
    if not ok:raise RuntimeError('TEST607 INTEGRITY '+name)
def weight_sha():
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());h.update(str(tuple(p.shape)).encode());h.update(str(p.dtype).encode())
        for part in p.detach().contiguous().view(torch.uint8).reshape(-1).split(16*1024*1024):h.update(part.cpu().numpy().tobytes())
    return h.hexdigest()
def span_tokens(text,begin,end):
    e=tok(text,add_special_tokens=False,return_offsets_mapping=True);ids=e['input_ids'];off=e['offset_mapping'];span=[i for i,(a,b) in enumerate(off) if a<end and b>begin]
    if not span or span!=list(range(span[0],span[-1]+1)):raise RuntimeError('SPAN INVALID')
    return ids,span
@torch.no_grad()
def capture_residual(text,begin,end):
    ids,span=span_tokens(text,begin,end);cap={};hooks=[]
    for li in LAYERS:
        def hook(mod,args,out,li=li):cap[li]=(out[0] if isinstance(out,(tuple,list)) else out)[:,span,:].detach().float().cpu().clone()
        hooks.append(model.model.layers[li].register_forward_hook(hook))
    try:
        x=torch.tensor([ids],device=DEVICE);pos=torch.arange(len(ids),device=DEVICE)
        model(input_ids=x,attention_mask=torch.ones_like(x),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=False,return_dict=True)
    finally:
        for h in hooks:h.remove()
    if set(cap)!=set(LAYERS):raise RuntimeError('RESIDUAL CAPTURE INCOMPLETE')
    return {li:cap[li][0] for li in LAYERS},dict(n=len(span),span=[span[0],span[-1]+1],boundary=bool(span[0]<P))
def cartridge_rep(ci):
    fact=CAR[ci]['fact'];text=source(fact);begin=len(PREFIX);end=begin+len(fact)
    if text[begin:end]!=fact:raise RuntimeError('FACT SPAN')
    return capture_residual(text,begin,end)[0]
def question_rep(q):
    # Same source-prefix and local position as facts; no fact/source supplied during ASK.
    text=PREFIX+q+'\n\n';begin=len(PREFIX);end=begin+len(q)
    return capture_residual(text,begin,end)
def unit_norm(x):return torch.nn.functional.normalize(x,dim=-1,eps=1e-9)
def transform_fit(X,rank):
    mu=X.mean(0);C=X-mu
    if rank==0:return mu,None
    # SVD of the store only: no question labels and no holdout queries in PCA.
    _,s,vh=torch.linalg.svd(C,full_matrices=False)
    r=min(rank,vh.shape[0]);basis=vh[:r].T.contiguous();scale=(s[:r].square()/max(len(X)-1,1)).clamp_min(1e-4).rsqrt()
    return mu,basis*scale
def apply(X,fit):
    mu,proj=fit;Y=X-mu
    if proj is not None:Y=Y@proj
    return unit_norm(Y)
def pair_scores(Q,K,beta):
    # Q:[q,r], K:[t,r]; per query token log-mean-exp over cartridge tokens.
    sim=Q@K.T
    if beta==0:return float(sim.mean().item())
    return float((torch.logsumexp(sim*beta,dim=1)-math.log(len(K))).mean().item()/beta)
def rankings(scores):return sorted(range(len(scores)),key=lambda i:(-scores[i],i))
print('='*126);print('TEST607 — FRAME-MATCHED RESIDUAL ASSOCIATION + POSITIVE RANDOM FEATURES SUPERPOSITION');print('='*126)
if '__file__' in globals() and os.path.isfile(__file__):print('SCRIPT_SHA256',hashlib.sha256(open(__file__,'rb').read()).hexdigest(),flush=True)
check('PANEL_SHA',PANEL_SHA=='69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45',sha=PANEL_SHA)
check('ARCH',(NL,H,QH,KVH,HD)==(28,3584,28,4,128));check('FROZEN',not any(p.requires_grad for p in model.parameters()))
WB=weight_sha();SB=sentinel();check('WEIGHT_SHA_REFERENCE',WB==EXPECTED,sha=WB)
print('[1/6] Capture native residuals for 96 cartridges...',flush=True)
KS={li:[] for li in LAYERS}
for ci in range(len(CAR)):
    v=cartridge_rep(ci)
    for li in LAYERS:KS[li].append(v[li])
    if ci%16==0:print('CARTRIDGE',ci+1,'/',len(CAR),flush=True)
check('CARTRIDGES',len(CAR)==96 and all(len(KS[li])==96 for li in LAYERS))
print('[2/6] Question representations in fact frame...',flush=True)
QS={};QINFO={}
for wi in range(24):
    typ=CAR[PANEL[wi]['target']]['type'];q=W[wi][typ];QS[wi],QINFO[wi]=question_rep(q)
    print('QUESTION',wi,'TARGET',PANEL[wi]['target'],'TOKENS',QINFO[wi]['n'],flush=True)
print('[3/6] Label-free PCA and calibration...',flush=True)
FEATURES={};CONFIGS=[]
for li in LAYERS:
    X=torch.cat(KS[li],dim=0)
    for rank in RANKS:
        fit=transform_fit(X,rank);KZ=[apply(v,fit) for v in KS[li]];QZ={wi:apply(QS[wi][li],fit) for wi in range(24)}
        FEATURES[(li,rank)]=(KZ,QZ)
        for beta in BETAS:CONFIGS.append((li,rank,beta))
def evaluate(wi,cfg):
    li,rank,beta=cfg;KZ,QZ=FEATURES[(li,rank)];scores=[pair_scores(QZ[wi],k,beta) for k in KZ];order=rankings(scores);target=PANEL[wi]['target'];r=order.index(target)+1
    return dict(case=wi,target=target,rank=r,hit1=int(r==1),hit5=int(r<=5),top5=order[:5],target_score=scores[target],top_score=scores[order[0]])
for idx,cfg in enumerate(CONFIGS):
    rows=[evaluate(wi,cfg) for wi in TRAIN];REC['calibration'].append(dict(cfg=list(cfg),hit5=sum(x['hit5'] for x in rows),hit1=sum(x['hit1'] for x in rows),rank_sum=sum(x['rank'] for x in rows),order=idx))
locked=min(REC['calibration'],key=lambda x:(-x['hit5'],-x['hit1'],x['rank_sum'],x['order']))
CFG=tuple(locked['cfg']);print('LOCKED',CFG,'CALIBRATION',locked,flush=True)
print('[4/6] Holdout and identity controls...',flush=True)
for wi in range(24):
    r=evaluate(wi,CFG);r['split']='CAL' if wi in TRAIN else 'HOLD';REC['cases'].append(r)
    print('CASE',wi,r['split'],'RANK',r['rank'],'TOP5',r['top5'],'TARGET',r['target'],flush=True)
H5=sum(x['hit5'] for x in REC['cases'] if x['split']=='HOLD');H1=sum(x['hit1'] for x in REC['cases'] if x['split']=='HOLD')
ALL5=sum(x['hit5'] for x in REC['cases']);HUB={i:sum(i in r['top5'] for r in REC['cases']) for i in range(96)}
print('HOLDOUT_RECALL5',H5,'/12','RECALL1',H1,'/12','ALL_RECALL5',ALL5,'/24','HUB_MAX',max(HUB.values()),flush=True)
# Self-retrieval is a representation sanity test, not a substitute for question retrieval.
li,rank,beta=CFG;KZ,QZ=FEATURES[(li,rank)];SELF=[]
for ci in range(96):
    q=KZ[ci];scores=[pair_scores(q,k,beta) for k in KZ];SELF.append(int(rankings(scores)[0]==ci))
print('SELF_RETRIEVAL1',sum(SELF),'/96',flush=True)
print('[5/7] Native generated fact cartridges, no feature-mixture distractors...',flush=True)
# The original 96 remain untouched. Extra facts are novel, deterministic, template-derived
# sentences, encoded by the frozen Qwen forward pass. They are NOT independent CUT3
# checkpoints and MUST NOT be presented as proof of 6144 fully operational cartridges.
# No gold, target ID or question is used to construct the extra facts.
import re,gc
SIZES=(96,384,1536,6144);NMAX=max(SIZES);M=2048;DS=(512,1024);SEEDS=(0,1,2);OMEGA_SEEDS=(0,1);LVALUES=(5,16,32,64)
COMPUTE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
li,rank,beta=CFG;check('FRAM_LOCK',CFG==(3,0,8.0),cfg=list(CFG))
KBASE,QBASE=FEATURES[(3,0)];fit=transform_fit(torch.cat(KS[3],0),0)
# Reuse already-captured native residuals for the first 96; preserve their original order.
# For new facts, replace the answer token if it occurs; also replace the first non-stopword
# proper noun to avoid 6048 exact duplicates. Every new fact is independently passed
# through source(fact) and Qwen. Token-level facts are kept as numeric residuals only.
used=set(x['fact'] for x in CAR);new_facts=[];templates=[str(CAR[i]['fact']) for i in range(96)]
words=('Arel','Borin','Cadel','Darin','Evel','Ferin','Galen','Harel','Iven','Jorin','Karel','Lorin','Maren','Nerin','Orel','Pavel','Qarin','Ravin','Saren','Tarin','Urel','Varen','Worin','Xarel','Yorin','Zerin')
def nonce(j):
    a=j;out=[]
    for _ in range(4):out.append(words[a%len(words)]);a=a//len(words)
    return ''.join(out)
for j in range(NMAX-96):
    ix=j%96;old=templates[ix];gold=str(CAR[ix].get('gold','')).strip();token=nonce(j+1000)
    if gold and gold in old:fact=old.replace(gold,token)
    else:
        # Preserve source grammar as much as possible and ensure unique native facts.
        proper=re.findall(r'\b[A-Z][a-z]{2,}\b',old)
        proper=[v for v in proper if v not in ('The','This','That','Current','Former','Near','Role','In','At','Of','For','And')]
        if not proper:raise RuntimeError('CANNOT GENERATE UNIQUE NATIVE FACT FROM TEMPLATE '+str(ix))
        fact=old.replace(proper[-1],token)
    if fact in used:raise RuntimeError('DUPLICATE NATIVE FACT')
    used.add(fact);new_facts.append(fact)
check('UNIQUE_NATIVE_FACTS',len(new_facts)==NMAX-96 and len(used)==NMAX,unique=len(used))
# Save transformed token matrices on CPU, avoiding a 40GB GPU resident bank.
KALL=list(KBASE);lens=[len(k) for k in KALL]
for j,fact in enumerate(new_facts):
    txt=source(fact);begin=len(PREFIX)
    if txt[begin:begin+len(fact)]!=fact:raise RuntimeError('GENERATED FACT FRAME')
    raw,_=capture_residual(txt,begin,begin+len(fact));z=apply(raw[3],fit)
    if not bool(torch.isfinite(z).all()):raise RuntimeError('NONFINITE GENERATED RESIDUAL')
    KALL.append(z.to(torch.float16));lens.append(len(z))
    if (j+1)%384==0:print('NATIVE_FACT_CAPTURE',j+1,'/',NMAX-96,flush=True)
check('NATIVE_REPRESENTATIONS',len(KALL)==NMAX,n=NMAX)
# True cosine/logmeanexp oracle, on native Qwen representations, not RFF features.
print('[6/7] Exact FRAM ceiling, coarse direct features and centered superposition...',flush=True)
REC['native_protocol']=dict(native_original=96,native_generated=NMAX-96,generated_from='deterministic transformations of the original 96 sentence templates; frozen Qwen forward activations',generated_are_cut3_checkpoints=False,seed_policy='fixed before evaluation',sizes=SIZES,M=M,Ds=DS,code_seeds=SEEDS,omega_seeds=OMEGA_SEEDS,L=LVALUES,primary_holdout=HOLD)
REC['native_scaling']=[]
QGPU={wi:QBASE[wi].to(COMPUTE) for wi in range(24)}
@torch.no_grad()
def exact_scores(n):
    results=torch.empty((24,n),dtype=torch.float32)
    # Batched by cartridge; no approximate feature kernel used here.
    for i in range(n):
        k=KALL[i].to(COMPUTE,dtype=torch.float32)
        for wi in range(24):
            q=QGPU[wi];sim=q@k.T
            results[wi,i]=((torch.logsumexp(sim*8,dim=1)-math.log(len(k))).mean()/8).cpu()
    return results
@torch.no_grad()
def phi(x,omega):
    # Deliberately preserve the TEST604/606 positive-feature definition.
    return torch.exp(x.float()@omega-4.0)/math.sqrt(M)
@torch.no_grad()
def make_feature_bank(omega):
    # Compute each cartridge's native token feature mean; features remain on CPU.
    result=torch.empty((NMAX,M),dtype=torch.float32)
    for i,k in enumerate(KALL):
        result[i]=phi(k.to(COMPUTE,dtype=torch.float32),omega).mean(0).cpu()
    return result
@torch.no_grad()
def make_query_features(omega):
    return [phi(QGPU[wi],omega).mean(0).cpu() for wi in range(24)]
def code_block(lo,hi,D,seed):
    # SplitMix64 keyed by (seed, global cartridge ID, dimension): stable across
    # build/decode chunk boundaries, with well-mixed high bits (no stored codes).
    ids=np.arange(lo+1,hi+1,dtype=np.uint64)[:,None]
    cols=np.arange(1,D+1,dtype=np.uint64)[None,:]
    x=ids*np.uint64(0x9E3779B185EBCA87)^cols*np.uint64(0xC2B2AE3D27D4EB4F)^np.uint64(seed+1)*np.uint64(0x165667B19E3779F9)
    x=x^(x>>np.uint64(30));x=x*np.uint64(0xBF58476D1CE4E5B9)
    x=x^(x>>np.uint64(27));x=x*np.uint64(0x94D049BB133111EB)
    x=x^(x>>np.uint64(31))
    return torch.from_numpy(((x>>np.uint64(63)).astype(np.float32)*2-1)/math.sqrt(D))
@torch.no_grad()
def centered_scores(F,qf,n,D,seed):
    # Exact algebra: sum_i (g_i - mean(g)) c_i^T, hence query-wise baseline removed.
    # U and C are built additively; codes regenerated at decode.
    G=F[:n].mean(0).to(COMPUTE);U=torch.zeros((M,D),dtype=torch.float32,device=COMPUTE)
    for lo in range(0,n,128):
        hi=min(lo+128,n);codes=code_block(lo,hi,D,seed).to(COMPUTE)
        U+=(F[lo:hi].to(COMPUTE)-G).T@codes
    Q=torch.stack(qf).to(COMPUTE);a=Q@U
    out=torch.empty((24,n),dtype=torch.float32)
    for lo in range(0,n,256):
        hi=min(lo+256,n);codes=code_block(lo,hi,D,seed).to(COMPUTE)
        out[:,lo:hi]=(a@codes.T).cpu()
    return out
@torch.no_grad()
def direct_scores(F,qf,n):
    # Linear soft-OR for candidate generation; mean centering leaves ranking invariant.
    return torch.stack(qf)@F[:n].T

def eval_ranks(scores,n,oracle=None):
    rows=[]
    for wi in range(24):
        target=PANEL[wi]['target'];order=rankings(scores[wi].tolist());rank=order.index(target)+1
        rows.append(dict(case=wi,split='HOLD' if wi in HOLD else 'CAL',rank=rank,top64=order[:64]))
    return rows
def count(rows,subset,L):return sum(rows[wi]['rank']<=L for wi in subset)
# Compute exact oracle once for NMAX; nested prefixes are identical.
EXACT=exact_scores(NMAX)
for n in SIZES:
    erows=eval_ranks(EXACT[:,:n],n)
    item=dict(N=n,arm='EXACT_FRAM',hold_r1=count(erows,HOLD,1),hold_r5=count(erows,HOLD,5),hold_r64=count(erows,HOLD,64),all_r5=count(erows,list(range(24)),5),rows=erows)
    REC['native_scaling'].append(item);print('NATIVE_ORACLE',{k:v for k,v in item.items() if k!='rows'},flush=True)
for osid in OMEGA_SEEDS:
    gen=torch.Generator(device='cpu').manual_seed(607000+osid)
    omega=(torch.randn((KBASE[0].shape[1],M),generator=gen)*math.sqrt(8)).to(COMPUTE)
    F=make_feature_bank(omega);QF=make_query_features(omega)
    check('FEATURE_FINITE_'+str(osid),bool(torch.isfinite(F).all()),seed=osid)
    for n in SIZES:
        direct=direct_scores(F,QF,n);dr=eval_ranks(direct,n)
        drrow=dict(N=n,arm='DIRECT_POSITIVE',omega_seed=osid,hold_r1=count(dr,HOLD,1),hold_r5=count(dr,HOLD,5),hold_r64=count(dr,HOLD,64),rows=dr)
        REC['native_scaling'].append(drrow);print('NATIVE_DIRECT',{k:v for k,v in drrow.items() if k!='rows'},flush=True)
        # Effective energy of centered scores: E_j = sum_{i!=j} (s_i-mean(s))^2 / (s_j-mean(s))^2.
        energy=[]
        for wi in HOLD:
            s=direct[wi];v=s-s.mean();t=PANEL[wi]['target'];energy.append(float((v.square().sum()-v[t].square())/v[t].square().clamp_min(1e-20)))
        print('CENTERED_ENERGY',{'N':n,'omega_seed':osid,'median':float(np.median(energy)),'max':max(energy)},flush=True)
        for D in DS:
            for seed in SEEDS:
                t0=time.perf_counter();sc=centered_scores(F,QF,n,D,seed);sr=eval_ranks(sc,n)
                # Two-stage exact rerank of the top-64 superposition candidates.
                reranked=[]
                for wi in range(24):
                    candidates=sr[wi]['top64'];reranked.append(sorted(candidates,key=lambda i:(-float(EXACT[wi,i]),i)))
                target_hits=[int(PANEL[wi]['target'] in reranked[wi][:5]) for wi in HOLD]
                row=dict(N=n,arm='CENTERED_SUPER',omega_seed=osid,D=D,code_seed=seed,hold_r1=count(sr,HOLD,1),hold_r5=count(sr,HOLD,5),hold_r64=count(sr,HOLD,64),rerank_hold5=sum(target_hits),super_bytes=M*D*4,feature_bank_bytes=n*M*4,codebook_stored_bytes=0,seconds=round(time.perf_counter()-t0,3),rows=sr)
                REC['native_scaling'].append(row)
                print('CENTERED_SUPER',{k:v for k,v in row.items() if k!='rows'},flush=True)
    del F,QF,omega
    if COMPUTE.type=='cuda':torch.cuda.empty_cache()
# Reference 96 native CUT3 check uses the exact locked TEST604 positive-feature configuration,
# not a synthetic score, and remains an independently verified continuity control.
print('[7/7] Frozen CUT3 96-real reference, final integrity and evidence audit...',flush=True)
BETA=8.;MREF=8192;DREF=8192
rg=torch.Generator(device='cpu').manual_seed(604000+80000+MREF)
omega=(torch.randn((KBASE[0].shape[1],MREF),generator=rg)*math.sqrt(BETA)).to(COMPUTE)
def pref(x):return torch.exp(x.float().to(COMPUTE)@omega-BETA/2)/math.sqrt(MREF)
FR=torch.stack([pref(k).mean(0) for k in KBASE]);QR=[pref(QBASE[wi]) for wi in range(24)]
gen=torch.Generator(device='cpu').manual_seed(604600+800000+MREF*11+DREF)
codes=(torch.randint(0,2,(96,DREF),generator=gen,dtype=torch.int8).float()*2-1).to(COMPUTE)/math.sqrt(DREF)
U=FR.T@codes;PRED=[]
for q in QR:
    z=(q@U)@codes.T;PRED.append((torch.log(z.clamp_min(1e-12)).mean(0)/BETA).cpu())
check('REFERENCE_RECALL5',sum(PANEL[wi]['target'] in rankings(PRED[wi].tolist())[:5] for wi in HOLD)==12)
@torch.no_grad()
def build_cut3(keys,cut):
    mem=init(keys[0],cut)
    for ci in keys[1:]:mem=append(mem,ci,cut)
    return mem
E2E=[]
for wi in HOLD:
    target=PANEL[wi]['target'];ids=rankings(PRED[wi].tolist())[:5];keys=sorted(ids)
    typ=CAR[target]['type'];q=W[wi][typ];gold=CAR[target]['gold']
    try:
        mem=build_cut3(keys,3);validate(mem,keys);response=answer(mem,q)
        row=dict(case=wi,retrieved=ids,answer=response,answer_ok=int(bool(hit(response,gold))),target_present=int(target in ids))
    except Exception as e:row=dict(case=wi,retrieved=ids,error=repr(e),answer_ok=0,target_present=int(target in ids))
    E2E.append(row);print('END_TO_END',wi,'OK',row['answer_ok'],'RETRIEVED',ids,flush=True)
REC['end_to_end']=E2E
WA=weight_sha();check('WEIGHT_UNCHANGED',WB==WA);check('SENTINEL_UNCHANGED',SB==sentinel());check('ZERO_TRAINABLE',not any(p.requires_grad for p in model.parameters()))
remaining=[]
for i,l in enumerate(model.model.layers):
    for label,m in [('layer',l),('attn',l.self_attn),('q_proj',l.self_attn.q_proj)]:
        if m._forward_hooks or m._forward_pre_hooks:remaining.append((i,label))
check('NO_HOOKS',not remaining,hooks=remaining)
REC['summary']=dict(native_count=len(KALL),largest_N=NMAX,reference_cut3_correct=sum(x['answer_ok'] for x in E2E),reference_cut3_total=len(E2E),seconds=round(time.perf_counter()-START,2),caveat='New native fact representations are Qwen forward activations, not independently validated CUT3 checkpoints; all end-to-end answers use original 96 only.')
REC['result_sha']=digest(REC);os.makedirs('/content',exist_ok=True)
with open(OUT,'w',encoding='utf-8') as f:json.dump(REC,f,ensure_ascii=False,indent=2)
print('='*126);print('TEST607 FINAL',REC['summary']);print('RESULT_SHA',REC['result_sha']);print('OUTPUT',OUT);print('DECISION','NATIVE REPRESENTATION CAPACITY + CENTERED SUPERPOSITION; DO NOT CLAIM 6144 CUT3 CARTRIDGES');print('='*126)

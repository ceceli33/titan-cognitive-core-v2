# TEST604 — AKBASCORE MAM-BÇ · FRAME-MATCHED RESIDUAL ACCESS + POSITIVE RANDOM FEATURES SUPERPOSITION + CUT3 AUDIT
# Frozen TEST602 backbone; exact residual oracle, positive exponential random features, coded superposition, unchanged CUT3.
import os,sys,subprocess,importlib.util,urllib.request,hashlib,json,time,math,random,inspect
for m in ('torch','transformers','accelerate'):
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,'-m','pip','install','-q',m])
import torch,numpy as np,transformers
URL='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/'
BLOBS={'566.py':'b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71','575.py':'569c92c61224ea408cdb6e81dbe45dde06dc05fd'}
def fetch_locked(name):
    with urllib.request.urlopen(urllib.request.Request(URL+name,headers={'User-Agent':'AKBASCORE-TEST604'}),timeout=120) as f:b=f.read()
    got=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
    if got!=BLOBS[name]:raise RuntimeError(f'SOURCE MISMATCH {name}: {got}')
    print('SOURCE_VERIFIED',name,got,flush=True);return b.decode()
s575=fetch_locked('575.py');s566=fetch_locked('566.py')
if 'BASE_BLOB="b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71"' not in s575:raise RuntimeError('575 LINK MISMATCH')
MARK='# TEST566 ADDITIONS — original TEST564 checkpoint/init/append/answer remain unchanged.'
if s566.count(MARK)!=1:raise RuntimeError('566 BOUNDARY MISMATCH')
exec(compile(s566.split(MARK,1)[0],'566.py','exec'),globals())
TEST='604';START=time.perf_counter();LAYERS=[3,8,16,24];RANKS=[0,64,128];BETAS=[0.0,8.0];TRAIN=list(range(12));HOLD=list(range(12,24));EXPECTED='b441b005d826d45f2778291696ac5ae22dfb3421defb37c539904d8d38f93786';OUT='/content/AKBASCORE_TEST604_POSITIVE_FEATURE_CUT3.json';REC={'checks':[],'calibration':[],'cases':[],'pilot':[]}
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def check(name,ok,**info):
    REC['checks'].append(dict(name=name,ok=bool(ok),**info));print('PASS' if ok else 'FAIL',name,info,flush=True)
    if not ok:raise RuntimeError('TEST604 INTEGRITY '+name)
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
print('='*126);print('TEST604 — FRAME-MATCHED RESIDUAL ASSOCIATION + POSITIVE RANDOM FEATURES SUPERPOSITION');print('='*126)
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
print('[5/6] Positive random features + coded superposition: separate kernel and code errors...',flush=True)
# For unit vectors, E_w[exp(sqrt(beta)w·q-beta/2) exp(sqrt(beta)w·k-beta/2)] = exp(beta*q·k).
# The feature estimate is strictly positive. Log-scores must not be confused with exact oracle scores.
# Tempered beta variants are DIAGNOSTIC; primary locked beta=8 remains the TEST603 reference.
POS_RESULTS=[];POS_CANDIDATES={};POS_DIRECT={};POS_BETAS=(2.0,4.0,8.0);MS=(2048,8192);DS=(2048,8192)
for pb in POS_BETAS:
    exact=torch.tensor([[pair_scores(QZ[wi],k,pb) for k in KZ] for wi in range(24)],dtype=torch.float64)
    exact_top=exact.argmax(1)
    for M in MS:
        rg=torch.Generator(device='cpu').manual_seed(604000+int(pb)*10000+M)
        omega=torch.randn((KZ[0].shape[1],M),generator=rg,dtype=torch.float32)*math.sqrt(pb)
        def positive_phi(x):
            # lognormal positive feature, scale exp(-pb/2), no learned parameters.
            return torch.exp(x.float()@omega-pb/2)/math.sqrt(M)
        F=torch.stack([positive_phi(k).mean(0) for k in KZ]).contiguous()
        QF=[positive_phi(QZ[wi]) for wi in range(24)]
        def log_scores(query_features,cartridge_features):
            est=query_features@cartridge_features.T
            if not torch.isfinite(est).all() or not (est>0).all():raise RuntimeError('NONPOSITIVE_OR_NONFINITE_POSITIVE_KERNEL')
            return torch.log(est).mean(0)/pb
        DIRECT=torch.stack([log_scores(qf,F) for qf in QF]);direct_top=DIRECT.argmax(1)
        direct_hold1=sum(int(direct_top[wi].item()==PANEL[wi]['target']) for wi in HOLD)
        direct_hold5=sum(int(PANEL[wi]['target'] in rankings(DIRECT[wi].tolist())[:5]) for wi in HOLD)
        direct_fidelity=int((direct_top==exact_top).sum())
        POS_DIRECT[(pb,M)]=DIRECT
        for D in DS:
            cg=torch.Generator(device='cpu').manual_seed(604600+int(pb)*100000+M*11+D)
            codes=(torch.randint(0,2,(96,D),generator=cg,dtype=torch.int8).float()*2-1)/math.sqrt(D)
            U=F.T@codes
            # The random-sign code decoder can return negative kernel estimates despite positive phi.
            # A negative estimate is an interference failure; count before clamping.
            APPROX=[];bad=0;total=0
            for qf in QF:
                est=(qf@U)@codes.T
                bad+=int((est<=0).sum().item());total+=est.numel()
                APPROX.append(torch.log(est.clamp_min(1e-12)).mean(0)/pb)
            APPROX=torch.stack(APPROX);approx_top=APPROX.argmax(1)
            hold1=sum(int(approx_top[wi].item()==PANEL[wi]['target']) for wi in HOLD)
            hold5=sum(int(PANEL[wi]['target'] in rankings(APPROX[wi].tolist())[:5]) for wi in HOLD)
            item=dict(beta=pb,M=M,D=D,direct_hold1=direct_hold1,direct_hold5=direct_hold5,direct_top1_vs_exact=direct_fidelity,super_hold1=hold1,super_hold5=hold5,super_top1_vs_direct=int((approx_top==direct_top).sum()),super_top1_vs_exact=int((approx_top==exact_top).sum()),negative_decoded=bad,decoded_count=total,mse_vs_direct=float((APPROX-DIRECT).square().mean()),mean_direct_error=float((DIRECT-exact.float()).abs().mean()))
            POS_RESULTS.append(item);POS_CANDIDATES[(pb,M,D)]=APPROX
            print('POSITIVE_SUPER',item,flush=True)
REC['positive_superposition']=POS_RESULTS
# Pre-register choice: beta=8 is the TEST603 target, and maximum M,D; do not select on holdout labels.
CHOSEN=(8.0,8192,8192);PRED=POS_CANDIDATES[CHOSEN]
print('LOCKED_SUPER_CONFIG',CHOSEN,'NOT_TUNED_ON_HOLDOUT',flush=True)
@torch.no_grad()
def build_cut3(keys,cut):
    if not keys:raise RuntimeError('EMPTY RETRIEVAL')
    mem=init(keys[0],cut)
    for ci in keys[1:]:mem=append(mem,ci,cut)
    return mem
@torch.no_grad()
def validate_cut3(mem,keys):
    expected=P+sum(len(CAR[ci]['body']) for ci in keys)
    if len(mem)!=NL:raise RuntimeError('CUT3 LAYER COUNT')
    for li,(k,v) in enumerate(mem):
        shape=(1,KVH,expected,HD)
        if tuple(k.shape)!=shape or tuple(v.shape)!=shape:raise RuntimeError(f'CUT3 SHAPE L{li}: {tuple(k.shape)} {tuple(v.shape)} != {shape}')
        if not torch.isfinite(k).all() or not torch.isfinite(v).all():raise RuntimeError(f'CUT3 NONFINITE L{li}')
    return True
print('[6/6] Selected cartridges -> unchanged CUT3 init/append/answer; holdout 12...',flush=True)
# Use the five retrieved cartridge IDs; this is not oracle ordering and never inserts target by hand.
# The established CUT3 construction and answer functions remain unmodified.
END_TO_END=[]
for wi in HOLD:
    typ=CAR[PANEL[wi]['target']]['type'];q=W[wi][typ];gold=CAR[PANEL[wi]['target']]['gold']
    ids=rankings(PRED[wi].tolist())[:5];keys=sorted(ids)
    try:
        mem=build_cut3(keys,3);validate_cut3(mem,keys);response=answer(mem,q)
        ok=bool(hit(response,gold));present=int(PANEL[wi]['target'] in ids)
        row=dict(case=wi,target=PANEL[wi]['target'],retrieved=ids,engine_order=keys,target_present=present,answer=response,gold=gold,answer_ok=int(ok))
    except Exception as e:
        row=dict(case=wi,target=PANEL[wi]['target'],retrieved=ids,error=repr(e),answer_ok=0,target_present=int(PANEL[wi]['target'] in ids))
    END_TO_END.append(row);print('END_TO_END',wi,'RETRIEVED',ids,'ANSWER_OK',row['answer_ok'],'ANSWER',row.get('answer',row.get('error')),flush=True)
REC['end_to_end']=END_TO_END
REC['end_to_end_errors']=sum('error' in x for x in END_TO_END)
print('E2E_ANSWER',sum(r['answer_ok'] for r in END_TO_END),'/12','E2E_RETRIEVAL',sum(r.get('target_present',0) for r in END_TO_END),'/12','E2E_ERRORS',REC['end_to_end_errors'],flush=True)
WA=weight_sha();check('WEIGHT_UNCHANGED',WB==WA);check('SENTINEL_UNCHANGED',SB==sentinel());check('ZERO_TRAINABLE',not any(p.requires_grad for p in model.parameters()))
remaining=[]
for i,l in enumerate(model.model.layers):
    for label,m in [('layer',l),('attn',l.self_attn),('q_proj',l.self_attn.q_proj)]:
        if m._forward_hooks or m._forward_pre_hooks:remaining.append((i,label))
check('NO_HOOKS',not remaining,hooks=remaining)
REC['protocol']=dict(test=TEST,model=MODEL_ID,source_blobs=BLOBS,panel_sha=PANEL_SHA,weight_sha=WB,selected=list(CFG),train=TRAIN,holdout=HOLD,config_count=len(CONFIGS),representation='native residual layer output; common PREFIX fact frame; store-only PCA',oracle='per-question-token exp cosine kernel when beta>0; linear cosine when beta=0',pilot='Positive random features for exponential dot kernel; random-sign coded superposition can still have negative decoded estimates; report direct vs coded separately',no_training=True,no_source_replay_at_query=True,code_sha_note='SHA-256 is printed when executed as a saved .py script.')
REC['summary']=dict(oracle_holdout_recall5=H5,oracle_holdout_recall1=H1,oracle_all_recall5=ALL5,self_retrieval1=sum(SELF),max_hub=max(HUB.values()),positive_super_holdout5=next(x['super_hold5'] for x in POS_RESULTS if (x['beta'],x['M'],x['D'])==CHOSEN),end_to_end_answer=sum(x['answer_ok'] for x in END_TO_END),end_to_end_n=len(END_TO_END),seconds=round(time.perf_counter()-START,2))
REC['hubness']=sorted(HUB.items(),key=lambda x:-x[1])[:12];REC['question_metadata']=QINFO;REC['result_sha']=digest(REC)
os.makedirs('/content',exist_ok=True)
with open(OUT,'w',encoding='utf-8') as f:json.dump(REC,f,ensure_ascii=False,indent=2)
print('='*126);print('TEST604 FINAL',REC['summary']);print('RESULT_SHA',REC['result_sha']);print('OUTPUT',OUT);print('DECISION','REPORT POSITIVE FEATURE, CODED SUPERPOSITION AND CUT3 SEPARATELY; NO SCALING CLAIM');print('='*126)

# TEST610 — AKBASCORE MAM-BÇ · FRAME-MATCHED RESIDUAL ACCESS + POSITIVE RANDOM FEATURES SUPERPOSITION + CUT3 AUDIT + CODE INTERFERENCE DIAGNOSTICS
# Frozen working TEST608 backbone; CUT3 differential audit + signal/noise decomposition, unchanged motor.
import os,sys,subprocess,importlib.util,urllib.request,hashlib,json,time,math,random,inspect
for m in ('torch','transformers','accelerate'):
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,'-m','pip','install','-q',m])
import torch,numpy as np,transformers
URL='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/'
BLOBS={'566.py':'b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71','575.py':'569c92c61224ea408cdb6e81dbe45dde06dc05fd'}
def fetch_locked(name):
    with urllib.request.urlopen(urllib.request.Request(URL+name,headers={'User-Agent':'AKBASCORE-TEST610'}),timeout=120) as f:b=f.read()
    got=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
    if got!=BLOBS[name]:raise RuntimeError(f'SOURCE MISMATCH {name}: {got}')
    print('SOURCE_VERIFIED',name,got,flush=True);return b.decode()
s575=fetch_locked('575.py');s566=fetch_locked('566.py')
if 'BASE_BLOB="b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71"' not in s575:raise RuntimeError('575 LINK MISMATCH')
MARK='# TEST566 ADDITIONS — original TEST564 checkpoint/init/append/answer remain unchanged.'
if s566.count(MARK)!=1:raise RuntimeError('566 BOUNDARY MISMATCH')
exec(compile(s566.split(MARK,1)[0],'566.py','exec'),globals())
TEST='610';START=time.perf_counter();LAYERS=[3,8,16,24];RANKS=[0,64,128];BETAS=[0.0,8.0];TRAIN=list(range(12));HOLD=list(range(12,24));EXPECTED='b441b005d826d45f2778291696ac5ae22dfb3421defb37c539904d8d38f93786';OUT='/content/AKBASCORE_TEST610_DEMO_ACCEPTANCE.json';REC={'checks':[],'calibration':[],'cases':[],'pilot':[],'cut3_audits':{}}
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def check(name,ok,**info):
    REC['checks'].append(dict(name=name,ok=bool(ok),**info));print('PASS' if ok else 'FAIL',name,info,flush=True)
    if not ok:raise RuntimeError('TEST610 INTEGRITY '+name)
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
print('='*126);print('TEST610 — FINAL DEMO ACCEPTANCE · FROZEN CUT3 + FRAM 96 CARTRIDGES');print('='*126)
if '__file__' in globals() and os.path.isfile(__file__):print('SCRIPT_SHA256',hashlib.sha256(open(__file__,'rb').read()).hexdigest(),flush=True)
check('PANEL_SHA',PANEL_SHA=='69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45',sha=PANEL_SHA)
check('ARCH',(NL,H,QH,KVH,HD)==(28,3584,28,4,128));check('FROZEN',not any(p.requires_grad for p in model.parameters()))
WB=weight_sha();SB=sentinel();check('WEIGHT_SHA_REFERENCE',WB==EXPECTED,sha=WB)
print('[1/8] Capture native residuals for 96 cartridges...',flush=True)
KS={li:[] for li in LAYERS}
for ci in range(len(CAR)):
    v=cartridge_rep(ci)
    for li in LAYERS:KS[li].append(v[li])
    if ci%16==0:print('CARTRIDGE',ci+1,'/',len(CAR),flush=True)
check('CARTRIDGES',len(CAR)==96 and all(len(KS[li])==96 for li in LAYERS))
print('[2/8] Question representations in fact frame...',flush=True)
QS={};QINFO={}
for wi in range(24):
    typ=CAR[PANEL[wi]['target']]['type'];q=W[wi][typ];QS[wi],QINFO[wi]=question_rep(q)
    print('QUESTION',wi,'TARGET',PANEL[wi]['target'],'TOKENS',QINFO[wi]['n'],flush=True)
print('[3/8] Label-free PCA and calibration...',flush=True)
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
print('[4/8] Holdout and identity controls...',flush=True)
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
print('[5/7] CUT3 baseline and reference retrieval...',flush=True)
BETA=8.;MREF=8192;DREF=8192
COMPUTE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
KBASE,QBASE=FEATURES[(3,0)]
check('FRAM_LOCK_BEFORE_CUT3',CFG==(3,0,8.0),cfg=list(CFG))
rg=torch.Generator(device='cpu').manual_seed(604000+80000+MREF)
omega=torch.randn((KBASE[0].shape[1],MREF),generator=rg,dtype=torch.float32)*math.sqrt(BETA)
def pref(x):return torch.exp(x.float()@omega-BETA/2)/math.sqrt(MREF)
FR=torch.stack([pref(k).mean(0) for k in KBASE]).to(COMPUTE);QR=[pref(QBASE[wi]).to(COMPUTE) for wi in range(24)]
gen=torch.Generator(device='cpu').manual_seed(604600+800000+MREF*11+DREF)
codes=(torch.randint(0,2,(96,DREF),generator=gen,dtype=torch.int8).float()*2-1).to(COMPUTE)/math.sqrt(DREF)
U=FR.T@codes;PRED=[]
for q in QR:
    z=(q@U)@codes.T;PRED.append((torch.log(z.clamp_min(1e-12)).mean(0)/BETA).cpu())
ref_hits=sum(PANEL[wi]['target'] in rankings(PRED[wi].tolist())[:5] for wi in HOLD)
ref_top1=sum(rankings(PRED[wi].tolist())[0]==PANEL[wi]['target'] for wi in HOLD)
print('REFERENCE_606_LOCKED',{'hold_r1':ref_top1,'hold_r5':ref_hits,'expected_r1':11,'expected_r5':12},flush=True)
check('REFERENCE_RECALL5',ref_hits==12,hold_r1=ref_top1,hold_r5=ref_hits)
@torch.no_grad()
def build_cut3(keys,cut):
    if not keys:raise RuntimeError('EMPTY RETRIEVAL')
    mem=init(keys[0],cut)
    for ci in keys[1:]:mem=append(mem,ci,cut)
    return mem
@torch.no_grad()
def validate_cut3(mem,keys):
    expected=P+sum(len(CAR[ci]['body']) for ci in keys)
    if len(mem)!=NL:raise RuntimeError(f'CUT3 LAYER COUNT {len(mem)} != {NL}')
    for li,(k,v) in enumerate(mem):
        shape=(1,KVH,expected,HD)
        if tuple(k.shape)!=shape or tuple(v.shape)!=shape:raise RuntimeError(f'CUT3 SHAPE L{li}: {tuple(k.shape)} {tuple(v.shape)} != {shape}')
        if not bool(torch.isfinite(k).all()) or not bool(torch.isfinite(v).all()):raise RuntimeError(f'CUT3 NONFINITE L{li}')
    return True
@torch.no_grad()
def audit_cut3(stage):
    rows=[]
    for wi in HOLD:
        target=PANEL[wi]['target'];ids=rankings(PRED[wi].tolist())[:5];keys=sorted(ids)
        typ=CAR[target]['type'];q=W[wi][typ];gold=CAR[target]['gold']
        row=dict(case=wi,target=target,retrieved=ids,engine_order=keys,target_present=int(target in ids),gold=gold,question=q)
        try:
            mem=build_cut3(keys,3)
            row['memory_layers']=len(mem);row['memory_tokens']=int(mem[0][0].shape[-2])
            validate_cut3(mem,keys)
            row['shape_ok']=True
            response=answer(mem,q)
            row['answer']=str(response);row['answer_type']=str(type(response));row['answer_repr']=repr(response)
            row['answer_ok']=int(bool(hit(response,gold)))
        except Exception as e:
            row['error']=repr(e);row['answer_ok']=0
        rows.append(row)
        print('CUT3_AUDIT',stage,wi,'TARGET',target,'PRESENT',row['target_present'],'SHAPE',row.get('shape_ok',False),'OK',row['answer_ok'],'GOLD',repr(gold),'RAW',row.get('answer_repr',row.get('error')),flush=True)
    REC['cut3_audits'][stage]=rows
    n=sum(x['answer_ok'] for x in rows);errors=sum('error' in x for x in rows)
    print('CUT3_AUDIT_SUMMARY',stage,{'correct':n,'total':len(rows),'errors':errors,'retrieval':sum(x['target_present'] for x in rows)},flush=True)
    return rows
PRE_AUDIT=audit_cut3('BASELINE')
print('BASELINE_STATUS',{'correct':sum(x['answer_ok'] for x in PRE_AUDIT),'errors':sum('error' in x for x in PRE_AUDIT),'retrieved':sum(x['target_present'] for x in PRE_AUDIT)},flush=True)
del FR,QR,codes,U,omega,rg,gen
if COMPUTE.type=='cuda':torch.cuda.empty_cache()

print('[6/7] Demo acceptance: 24-case retrieval, reproducibility, prefix/append and negative controls...',flush=True)
REC['demo_acceptance']={};REC['demo_cases']=[]
# The locked retrieval protocol was fitted on cases 0-11; cases 12-23 remain untouched holdout.
all_retrieved=0;all_correct=0;hold_correct=0;cal_correct=0;repeat_ok=0;shape_ok=0
for wi in range(24):
    target=PANEL[wi]['target'];ids=rankings(PRED[wi].tolist())[:5];keys=sorted(ids)
    typ=CAR[target]['type'];q=W[wi][typ];gold=CAR[target]['gold'];present=target in ids
    row={'case':wi,'split':'HOLD' if wi in HOLD else 'CAL','target':target,'retrieved':ids,'engine_order':keys,'target_present':present,'gold':gold}
    try:
        # Genuine engine: init -> append -> answer; no source fact is supplied at answer time.
        mem=build_cut3(keys,3);validate_cut3(mem,keys);row['shape_ok']=True
        first=answer(mem,q);second=answer(mem,q)
        row['answer']=str(first);row['repeat_answer']=str(second)
        row['correct']=bool(hit(first,gold));row['repeat_equal']=str(first)==str(second)
        row['memory_tokens']=int(mem[0][0].shape[-2]);row['memory_layers']=len(mem)
        row['memory_bytes']=int(sum(k.numel()*k.element_size()+v.numel()*v.element_size() for k,v in mem))
    except Exception as exc:
        row['error']=repr(exc);row.update(correct=False,repeat_equal=False,shape_ok=False)
    all_retrieved+=int(present);all_correct+=int(row['correct']);hold_correct+=int(row['correct'] and wi in HOLD)
    cal_correct+=int(row['correct'] and wi in TRAIN);repeat_ok+=int(row['repeat_equal']);shape_ok+=int(row['shape_ok'])
    REC['demo_cases'].append(row)
    print('DEMO_CASE',wi,row['split'],'TARGET',target,'PRESENT',int(present),'CORRECT',int(row['correct']),'REPEAT',int(row['repeat_equal']),'SHAPE',int(row['shape_ok']),'RAW',repr(row.get('answer',row.get('error'))),flush=True)
REC['demo_acceptance']['panel']={'total':24,'correct':all_correct,'cal_correct':cal_correct,'hold_correct':hold_correct,'retrieved':all_retrieved,'repeat_equal':repeat_ok,'shape_valid':shape_ok}
# Exact deterministic rebuild of an unchanged ordered cartridge list. Compare all 28 layers.
@torch.no_grad()
def compare_mem_exact(a,b):
    if len(a)!=len(b):return False
    return all(torch.equal(ka,kb) and torch.equal(va,vb) for (ka,va),(kb,vb) in zip(a,b))
rebuild_rows=[];prefix_rows=[]
for wi in (12,15,18,21):
    keys=sorted(rankings(PRED[wi].tolist())[:5]);a=build_cut3(keys,3);b=build_cut3(keys,3)
    eq=compare_mem_exact(a,b);rebuild_rows.append({'case':wi,'equal':eq,'keys':keys})
    # Check that repeated append from an identical first-cartridge INIT yields the same full memory.
    partial=init(keys[0],3);prefix_valid=validate_cut3(partial,keys[:1])
    for j,ci in enumerate(keys[1:],1):
        partial=append(partial,ci,3);prefix_valid=bool(prefix_valid and validate_cut3(partial,keys[:j+1]))
    prefix_eq=compare_mem_exact(partial,a);prefix_rows.append({'case':wi,'prefix_valid':bool(prefix_valid),'final_equal':prefix_eq})
    print('DEMO_REBUILD',wi,'BIT_EQUAL',int(eq),'APPEND_PREFIX_SHAPES',int(prefix_valid),'APPEND_FINAL_EQUAL',int(prefix_eq),flush=True)
REC['demo_acceptance']['rebuild']=rebuild_rows;REC['demo_acceptance']['append']=prefix_rows
# Diagnostic only: remove target and ask. This tests answer leakage/hallucination;
# failure does not rewrite the locked positive-control success criteria.
negative=[]
for wi in (12,15,18,21):
    target=PANEL[wi]['target'];ids=rankings(PRED[wi].tolist())[:5]
    rest=sorted(ci for ci in ids if ci!=target)
    if not rest:continue
    typ=CAR[target]['type'];q=W[wi][typ];gold=CAR[target]['gold']
    try:
        mem=build_cut3(rest,3);validate_cut3(mem,rest);raw=answer(mem,q)
        row={'case':wi,'target_absent':target not in rest,'gold_leak':bool(hit(raw,gold)),'raw':str(raw)}
    except Exception as exc:row={'case':wi,'error':repr(exc)}
    negative.append(row);print('DEMO_NEGATIVE',row,flush=True)
REC['demo_acceptance']['negative_diagnostic']=negative
print('[7/7] Final integrity, deterministic baseline and acceptance decision...',flush=True)
POST_AUDIT=audit_cut3('FINAL')
comparison=[{'case':a['case'],'before':a.get('answer_repr'),'after':b.get('answer_repr'),'equal':a.get('answer_repr')==b.get('answer_repr'),'before_ok':a['answer_ok'],'after_ok':b['answer_ok']} for a,b in zip(PRE_AUDIT,POST_AUDIT)]
REC['demo_acceptance']['before_after']=comparison
check('WEIGHT_UNCHANGED',weight_sha()==WB)
check('SENTINEL_UNCHANGED',sentinel()==SB)
check('ZERO_TRAINABLE',not any(p.requires_grad for p in model.parameters()))
hooks=[]
for li,l in enumerate(model.model.layers):
    for label,m in [('layer',l),('attn',l.self_attn),('q_proj',l.self_attn.q_proj)]:
        if m._forward_hooks or m._forward_pre_hooks:hooks.append((li,label))
check('NO_HOOKS',not hooks,hooks=hooks)
criteria={
 'holdout_fram_r1_12':H1==12,
 'holdout_fram_r5_12':H5==12,
 'holdout_reference_retrieval_12':ref_hits==12,
 'holdout_cut3_correct_12':hold_correct==12,
 'baseline_cut3_correct_12':sum(x['answer_ok'] for x in PRE_AUDIT)==12,
 'final_cut3_correct_12':sum(x['answer_ok'] for x in POST_AUDIT)==12,
 'no_answer_changes':all(x['equal'] for x in comparison),
 'all24_repeat_deterministic':repeat_ok==24,
 'all24_memory_shapes_valid':shape_ok==24,
 'rebuild_bitwise_4':all(x['equal'] for x in rebuild_rows),
 'append_prefix_4':all(x['prefix_valid'] and x['final_equal'] for x in prefix_rows),
 'weight_unchanged':weight_sha()==WB,
 'sentinel_unchanged':sentinel()==SB,
 'no_trainables':not any(p.requires_grad for p in model.parameters()),
 'no_hooks':not hooks
}
REC['demo_acceptance']['criteria']=criteria
REC['demo_acceptance']['decision']='DEMO_READY_96' if all(criteria.values()) else 'DEMO_NOT_READY'
REC['summary']={'test':'610','demo_decision':REC['demo_acceptance']['decision'],'criteria_passed':sum(criteria.values()),'criteria_total':len(criteria),'holdout_correct':hold_correct,'holdout_total':12,'all_panel_correct':all_correct,'all_panel_total':24,'reference_cartridges':96,'seconds':round(time.perf_counter()-START,2),'scope':'96 native CUT3 cartridges; 6144 residual facts and compressed superposition are NOT certified by this test'}
REC['result_sha']=digest(REC)
os.makedirs('/content',exist_ok=True)
with open(OUT,'w',encoding='utf-8') as f:json.dump(REC,f,ensure_ascii=False,indent=2,default=str)
print('='*126);print('TEST610 CRITERIA',criteria);print('TEST610 FINAL',REC['summary']);print('RESULT_SHA',REC['result_sha']);print('OUTPUT',OUT);print('DECISION',REC['demo_acceptance']['decision']);print('='*126)

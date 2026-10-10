# TEST601 — AKBASCORE MAM-BÇ · EARLY-LAYER EXACT Q/K KERNEL RETRIEVAL
# TEST600 backbone, sealed 575 -> 566; oracle ranking only, NO superposition / training / UI
import os,sys,subprocess,importlib.util,urllib.request,hashlib,json,time,gc,math
for m in ('torch','transformers','accelerate'):
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,'-m','pip','install','-q',m])
import torch,numpy as np,transformers
URL='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/'
BLOBS={'566.py':'b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71','575.py':'569c92c61224ea408cdb6e81dbe45dde06dc05fd'}
def fetch_locked(name):
    req=urllib.request.Request(URL+name,headers={'User-Agent':'AKBASCORE-TEST601'})
    with urllib.request.urlopen(req,timeout=120) as f:b=f.read()
    got=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
    if got!=BLOBS[name]:raise RuntimeError(f'SOURCE BLOB MISMATCH {name}: {got}')
    print('SOURCE_VERIFIED',name,got,flush=True);return b.decode('utf-8')
s575=fetch_locked('575.py');s566=fetch_locked('566.py')
if 'BASE_BLOB="b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71"' not in s575:raise RuntimeError('575 BASELINE LINK MISMATCH')
MARK='# TEST566 ADDITIONS — original TEST564 checkpoint/init/append/answer remain unchanged.'
if s566.count(MARK)!=1:raise RuntimeError('BASELINE BOUNDARY MISMATCH')
exec(compile(s566.split(MARK,1)[0],'566.py','exec'),globals())
TEST='601';T601=time.perf_counter();EXPECTED_WEIGHT_SHA='b441b005d826d45f2778291696ac5ae22dfb3421defb37c539904d8d38f93786'
LAYERS=[0,1,2,3];BETAS=[0.5,1.0,2.0];AGGS=['LOGMEAN','LOGSUM'];QUERY_MODES=['ALL','LAST'];TRAIN=list(range(12));HOLDOUT=list(range(12,24));K=5;RECORD={'protocol':{},'checks':[],'calibration':[],'holdout':[],'diagnostics':[]}
def check(name,ok,**detail):
    RECORD['checks'].append(dict(name=name,ok=bool(ok),**detail));print('PASS' if ok else 'FAIL',name,detail,flush=True)
    if not ok:raise RuntimeError('TEST601 GATE '+name)
def weight_sha():
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());h.update(str(tuple(p.shape)).encode());h.update(str(p.dtype).encode())
        for part in p.detach().contiguous().view(torch.uint8).reshape(-1).split(16*1024*1024):h.update(part.cpu().numpy().tobytes())
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def question_span(text,q):
    # Fast-tokenizer offsets are computed on the COMPLETE prompt; never tokenize q separately for matching.
    anchor='\nQUESTION:\n';at=text.rfind(anchor)
    if at<0:raise RuntimeError('QUESTION ANCHOR MISSING')
    begin=at+len(anchor);end=begin+len(q)
    if text[begin:end]!=q or text.find(q,end)!=-1:raise RuntimeError('QUESTION CHARACTER ALIGNMENT')
    enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
    offsets=enc['offset_mapping'];span=[i for i,(a,b) in enumerate(offsets) if a<end and b>begin]
    if not span or span!=list(range(span[0],span[-1]+1)):raise RuntimeError('QUESTION TOKEN SPAN INVALID')
    return enc['input_ids'],span,dict(question_tokens=len(span),start=span[0],end=span[-1]+1,question_chars=[begin,end],boundary_overlap=bool(offsets[span[0]][0]<begin or offsets[span[-1]][1]>end),standalone_tokens=len(enc['input_ids']))
@torch.no_grad()
def query_keys(q):
    # The source-free query uses the EXACT complete prompt tokenization and native q_proj.
    text=PREFIX+suffix(q);ids,span,meta=question_span(text,q);capt={};handles=[]
    def hook(layer):
        def fn(mod,args,out):capt[layer]=out.detach().clone()
        return fn
    for li in LAYERS:handles.append(model.model.layers[li].self_attn.q_proj.register_forward_hook(hook(li)))
    try:
        x=torch.tensor([ids],device=DEVICE);pos=torch.arange(len(ids),device=DEVICE)
        model(input_ids=x,attention_mask=torch.ones_like(x),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=False,return_dict=True)
    finally:
        for h in handles:h.remove()
    if set(capt)!=set(LAYERS):raise RuntimeError('Q PROJECTION HOOK INCOMPLETE')
    result={};positions=torch.tensor(span,device=DEVICE)
    cos,sin=rope(positions)
    for li in LAYERS:
        raw=capt[li][0,span,:].reshape(-1,QH,HD).float().transpose(0,1).unsqueeze(0)
        rot=(raw*cos+half(raw)*sin).squeeze(0).transpose(0,1).contiguous()
        if not torch.isfinite(rot).all():raise RuntimeError(f'NONFINITE Q L{li}')
        result[li]=rot.cpu()
    return result,meta
@torch.no_grad()
def cartridge_keys():
    # Existing checkpoint returns native RoPE-rotated K. Store only body, L0..L3, CPU.
    out={}
    for ci in range(len(CAR)):
        cp=checkpoint(ci,3);out[ci]={li:cp['kv'][li][0][0,:,P:,:].float().cpu().contiguous() for li in LAYERS}
        if ci%16==0:print('CARTRIDGE_QK',ci+1,'/',len(CAR),flush=True)
        del cp
    return out
@torch.no_grad()
def layer_scores(qmat,kmat,beta):
    # q: [Q,28,128], k: [4,T,128]. QH/KVH = 7, standard GQA pairing.
    q=qmat.reshape(qmat.shape[0],KVH,QH//KVH,HD)
    raw=torch.einsum('qghd,gtd->qght',q,kmat)/math.sqrt(HD)
    raw=raw.float().reshape(qmat.shape[0],QH,-1)*beta
    return raw
@torch.no_grad()
def scores_for(qset,kset):
    # Per-layer [96, 2 aggregation, 2 query modes, 3 betas] exact dot kernel scores.
    score=np.zeros((len(CAR),len(LAYERS),len(BETAS),len(AGGS),len(QUERY_MODES)),dtype=np.float64)
    for ci in range(len(CAR)):
        for lidx,li in enumerate(LAYERS):
            z=layer_scores(qset[li],kset[ci][li],1.0)
            for bi,beta in enumerate(BETAS):
                for qi,mode in enumerate(QUERY_MODES):
                    a=z if mode=='ALL' else z[-1:]
                    a=a*beta
                    # Log of mean exp kernel (length-normalized) and log sum exp kernel.
                    s=torch.logsumexp(a.flatten(),0).item()
                    score[ci,lidx,bi,0,qi]=s-math.log(a.numel())
                    score[ci,lidx,bi,1,qi]=s
    return score
print('='*132);print('TEST601 — MAM-BÇ · EXACT EARLY Q/K RETRIEVAL · ORACLE KERNEL',flush=True);print('='*132)
check('PANEL_SHA',PANEL_SHA=='69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45',sha=PANEL_SHA)
check('ARCHITECTURE',(NL,H,QH,KVH,HD)==(28,3584,28,4,128))
check('FROZEN',not any(p.requires_grad for p in model.parameters()))
WB=weight_sha();SB=sentinel();check('WEIGHT_SHA_REFERENCE',WB==EXPECTED_WEIGHT_SHA,sha=WB)
print('[1/5] Native cartridge K vectors, 96 cartridges...',flush=True)
KSET=cartridge_keys();check('96_NATIVE_CARTRIDGES',len(KSET)==96)
print('[2/5] Question-only native Q and exact attention kernel...',flush=True)
MATS={};META={}
for wi in range(24):
    typ=CAR[PANEL[wi]['target']]['type'];q=W[wi][typ]
    qm,meta=query_keys(q);mat=scores_for(qm,KSET)
    if not np.isfinite(mat).all():raise RuntimeError('NONFINITE QK SCORES')
    MATS[wi]=mat;META[wi]=meta
    print('QUESTION',wi,'TARGET',PANEL[wi]['target'],'TOKENS',meta['question_tokens'],flush=True)
# Predeclare candidate grid. Never select using HOLDOUT. Fixed deterministic tie-break by cartridge id.
CONFIGS=[(tuple(layers),bi,ai,qi) for layers in ((0,),(1,),(2,),(3,),(0,1),(1,2),(2,3),(0,1,2,3)) for bi in range(len(BETAS)) for ai in range(len(AGGS)) for qi in range(len(QUERY_MODES))]
def rank(wi,cfg):
    ls,bi,ai,qi=cfg;v=MATS[wi][:,list(ls),bi,ai,qi].mean(axis=1)
    return sorted(range(len(CAR)),key=lambda ci:(-float(v[ci]),ci)),v
def eval_one(wi,cfg):
    ranking,v=rank(wi,cfg);target=PANEL[wi]['target'];pos=ranking.index(target)+1
    return dict(case=wi,target=target,rank=pos,hit1=pos<=1,hit5=pos<=K,top5=ranking[:K],target_score=float(v[target]),best_score=float(v[ranking[0]]),gap_to_best=float(v[ranking[0]]-v[target]))
print('[3/5] Calibration 12 questions: selecting ONE configuration...',flush=True)
CAND=[]
for cfg in CONFIGS:
    rows=[eval_one(wi,cfg) for wi in TRAIN]
    CAND.append((sum(r['hit5'] for r in rows),sum(r['hit1'] for r in rows),-sum(r['rank'] for r in rows),cfg))
CAND.sort(key=lambda z:(-z[0],-z[1],-z[2],CONFIGS.index(z[3])))
chosen=CAND[0][3]
def cfg_dict(cfg):return dict(layers=list(cfg[0]),beta=BETAS[cfg[1]],aggregation=AGGS[cfg[2]],query_mode=QUERY_MODES[cfg[3]])
print('LOCKED_CONFIG',cfg_dict(chosen),'CALIBRATION',CAND[0][:3],flush=True)
RECORD['calibration']=[dict(config=cfg_dict(z[3]),recall5=z[0],recall1=z[1],rank_sum=-z[2]) for z in CAND]
print('[4/5] Locked holdout 12 questions + full panel diagnostics...',flush=True)
for wi in range(24):
    r=eval_one(wi,chosen);r['split']='CALIBRATION' if wi in TRAIN else 'HOLDOUT'
    RECORD['diagnostics'].append(r)
    print('CASE',wi,'SPLIT',r['split'],'RANK',r['rank'],'HIT5',int(r['hit5']),'TARGET',r['target'],'TOP5',r['top5'],flush=True)
HROWS=[r for r in RECORD['diagnostics'] if r['split']=='HOLDOUT'];H5=sum(r['hit5'] for r in HROWS);H1=sum(r['hit1'] for r in HROWS)
ALL5=sum(r['hit5'] for r in RECORD['diagnostics'])
print('HOLDOUT_RECALL@5',H5,'/',len(HROWS),'HOLDOUT_RECALL@1',H1,'/',len(HROWS),'ALL_RECALL@5',ALL5,'/24')
# Honest interpretation: 12 holdout cases; >=95% means 12/12, no binomial population guarantee.
HYPOTHESIS=H5/len(HROWS)>=0.95
print('H601_EXACT_KERNEL_STATUS','PASS' if HYPOTHESIS else 'FAIL','(12-case holdout; no population guarantee)')
print('[5/5] Frozen weights, hooks, and record...',flush=True)
WA=weight_sha();check('WEIGHT_UNCHANGED',WB==WA);check('SENTINEL_UNCHANGED',sentinel()==SB)
check('ZERO_TRAINABLE',not any(p.requires_grad for p in model.parameters()))
remaining=[]
for li,layer in enumerate(model.model.layers):
    for label,module in (('layer',layer),('attention',layer.self_attn),('q_proj',layer.self_attn.q_proj)):
        if module._forward_hooks or module._forward_pre_hooks:remaining.append((li,label))
check('NO_RESIDUAL_HOOKS',not remaining,hooks=remaining)
RECORD['protocol']=dict(test=TEST,model=MODEL_ID,cut=3,source_blobs=BLOBS,panel_sha=PANEL_SHA,cartridges=96,train=TRAIN,holdout=HOLDOUT,chosen=cfg_dict(chosen),candidate_count=len(CONFIGS),target='early Q/K exact softmax numerator, no normalized cross-cartridge attention',no_superposition=True,no_training=True,no_gradio=True,source_replay_at_query=False,question_span='fast tokenizer full-prompt offsets; boundary overlap recorded',question_context='PREFIX + suffix(question) without source facts',canonical_rope='native complete-query positions and native cartridge key positions; positional mismatch is an explicit experimental limitation',pool='all 96 native panel cartridges; no synthetic 1e3/1e4 population in this test')
RECORD['summary']=dict(holdout_recall5=H5,holdout_n=len(HROWS),holdout_recall1=H1,all_recall5=ALL5,hypothesis_pass=HYPOTHESIS,seconds=round(time.perf_counter()-T601,2))
RECORD['question_metadata']=META;RECORD['weight_before']=WB;RECORD['weight_after']=WA;RECORD['gpu']=torch.cuda.get_device_name(0);RECORD['torch_version']=torch.__version__;RECORD['transformers_version']=transformers.__version__
RECORD['result_sha']=digest(RECORD);OUT='/content/AKBASCORE_TEST601_EXACT_QK_RETRIEVAL.json';os.makedirs('/content',exist_ok=True)
with open(OUT,'w',encoding='utf-8') as f:json.dump(RECORD,f,ensure_ascii=False,indent=2)
print('='*132);print('TEST601 FINAL',RECORD['summary']);print('RESULT_SHA',RECORD['result_sha']);print('OUTPUT',OUT);print('SECONDS',RECORD['summary']['seconds'])
print('DECISION:', 'EXACT KERNEL SUPPORTED ON HOLDOUT; PROCEED TO TEST602' if HYPOTHESIS else 'DO NOT PROCEED: EARLY Q/K RETRIEVAL FAILED; DIAGNOSE REPRESENTATION')
print('NOTE: This is the exact native Q/K kernel oracle, not superposition; 96 real cartridges only.');print('='*132)

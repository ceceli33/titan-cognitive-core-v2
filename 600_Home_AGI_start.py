# TEST600 — AKBASCORE MAM-BÇ · CUT3 REMOVE / REBUILD / ROLLBACK
# VERIFIED TEST575 -> TEST566 ENGINE; NO TRAINING / NO RETRIEVAL / NO GRADIO
import os,sys,subprocess,importlib.util,urllib.request,hashlib,json,time,gc
for m in ('torch','transformers','accelerate'):
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,'-m','pip','install','-q',m])
import torch,numpy as np,transformers
URL='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/'
BLOBS={'566.py':'b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71','575.py':'569c92c61224ea408cdb6e81dbe45dde06dc05fd'}
def fetch_locked(name):
    req=urllib.request.Request(URL+name,headers={'User-Agent':'AKBASCORE-TEST600'})
    with urllib.request.urlopen(req,timeout=120) as f:b=f.read()
    got=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
    if got!=BLOBS[name]:raise RuntimeError(f'SOURCE BLOB MISMATCH {name}: {got}')
    print('SOURCE_VERIFIED',name,got,flush=True)
    return b.decode('utf-8')
s575=fetch_locked('575.py');s566=fetch_locked('566.py')
if 'BASE_BLOB="b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71"' not in s575:raise RuntimeError('575 BASELINE LINK MISMATCH')
MARK='# TEST566 ADDITIONS — original TEST564 checkpoint/init/append/answer remain unchanged.'
if s566.count(MARK)!=1:raise RuntimeError('BASELINE BOUNDARY MISMATCH')
exec(compile(s566.split(MARK,1)[0],'566.py','exec'),globals())
TEST='600';CUT=3;CASES=list(range(24));SLOTS=['FIRST','MIDDLE','LAST'];T600=time.perf_counter()
EXPECTED_WEIGHT_SHA='b441b005d826d45f2778291696ac5ae22dfb3421defb37c539904d8d38f93786'
ROWS=[];ROLL=[];BASELINES=[];NEG=[];CHECKS=[];ERRORS=[];TOL=0.0

def check(name,ok,**details):
    CHECKS.append(dict(name=name,ok=bool(ok),**details));print(('PASS' if ok else 'FAIL'),'|',name,'|',details,flush=True)
    if not ok:raise RuntimeError('TEST600 INTEGRITY GATE: '+name)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def weight_sha():
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());h.update(str(tuple(p.shape)).encode());h.update(str(p.dtype).encode())
        for part in p.detach().contiguous().view(torch.uint8).reshape(-1).split(16*1024*1024):h.update(part.cpu().numpy().tobytes())
    return h.hexdigest()
def memcopy(mem):return tuple((k.clone(),v.clone()) for k,v in mem)
def memlength(mem):return int(mem[0][0].shape[-2])
def mshape(mem,keys):
    n=P+sum(len(CAR[ci]['body']) for ci in keys)
    if len(mem)!=NL:raise RuntimeError('LAYER COUNT')
    for li,(k,v) in enumerate(mem):
        if k.shape!=(1,KVH,n,HD) or v.shape!=(1,KVH,n,HD):raise RuntimeError(f'SHAPE L{li}')
        if not torch.isfinite(k).all() or not torch.isfinite(v).all():raise RuntimeError(f'NONFINITE L{li}')
    return n
@torch.no_grad()
def build(keys,record_prefix=False):
    if not keys:raise ValueError('EMPTY KEYS')
    mem=init(keys[0],CUT);prefixes=[memcopy(mem)] if record_prefix else None
    for ci in keys[1:]:
        mem=append(mem,ci,CUT)
        if record_prefix:prefixes.append(memcopy(mem))
    mshape(mem,keys)
    return mem,prefixes
@torch.no_grad()
def rebuilt_without(keys,index,prefixes):
    rest=keys[:index]+keys[index+1:]
    if index==0:mem=init(rest[0],CUT);suffix=rest[1:]
    else:mem=memcopy(prefixes[index-1]);suffix=keys[index+1:]
    for ci in suffix:mem=append(mem,ci,CUT)
    mshape(mem,rest)
    return mem
@torch.no_grad()
def naive_without(full,keys,index):
    # Physically delete the target rows, then correct surviving suffix K RoPE positions.
    # No upper-layer suffix recomputation: contextual contamination deliberately remains.
    a=P+sum(len(CAR[x]['body']) for x in keys[:index]);b=a+len(CAR[keys[index]]['body'])
    oldn=memlength(full);oldpos=torch.arange(b,oldn,device=DEVICE);newpos=torch.arange(a,oldn-(b-a),device=DEVICE)
    result=[]
    for k,v in full:
        leftk=k[:,:,:a,:];leftv=v[:,:,:a,:];rightk=k[:,:,b:,:];rightv=v[:,:,b:,:]
        if oldpos.numel():rightk=rephase(rightk,oldpos,newpos)
        result.append((torch.cat((leftk,rightk),-2),torch.cat((leftv,rightv),-2)))
    rest=keys[:index]+keys[index+1:];mshape(tuple(result),rest)
    return tuple(result)
@torch.no_grad()
def compare(a,b):
    if len(a)!=len(b):raise RuntimeError('COMPARE LAYER MISMATCH')
    all_equal=True;mx=0.0;ma=0.0;changed=[];upper_v=0.0
    for li,((ak,av),(bk,bv)) in enumerate(zip(a,b)):
        if ak.shape!=bk.shape or av.shape!=bv.shape:raise RuntimeError('COMPARE SHAPE')
        ke=torch.equal(ak,bk);ve=torch.equal(av,bv)
        if not (ke and ve):all_equal=False;changed.append(li)
        kd=(ak.float()-bk.float()).abs();vd=(av.float()-bv.float()).abs()
        mx=max(mx,float(kd.max().item()),float(vd.max().item()))
        ma=max(ma,float(kd.mean().item()),float(vd.mean().item()))
        if li>CUT:upper_v=max(upper_v,float(vd.max().item()))
    return dict(bit_equal=all_equal,max_abs=mx,max_mean_abs=ma,changed_layers=changed,upper_v_max=upper_v)
def question_for(wi):
    z=PANEL[wi];typ=CAR[z['target']]['type']
    return W[wi][typ],CAR[z['target']]['gold']
def ask(mem,q):return answer(mem,q)
print('='*132);print('TEST600 — MAM-BÇ · EXACT REMOVE / REBUILD / ROLLBACK');print('='*132)
check('PANEL_SHA',PANEL_SHA=='69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45',sha=PANEL_SHA)
check('ARCHITECTURE',(NL,H,QH,KVH,HD)==(28,3584,28,4,128))
check('FROZEN',not any(p.requires_grad for p in model.parameters()))
WB=weight_sha();SB=sentinel();check('WEIGHT_SHA_REFERENCE',WB==EXPECTED_WEIGHT_SHA,sha=WB)
print('[1/5] Determinism calibration: independent rebuilds, one per slot...',flush=True)
for slot in SLOTS:
    keys=ordered(PANEL[0],slot);q,gold=question_for(0)
    x,_=build(keys);y,_=build(keys);d=compare(x,y)
    ax=ask(x,q);ay=ask(y,q)
    BASELINES.append(dict(slot=slot,kv=d,answer_a=ax,answer_b=ay,answer_equal=norm(ax)==norm(ay)))
    print('DETERMINISM',slot,'BIT',d['bit_equal'],'MAX',d['max_abs'],'ANS',norm(ax)==norm(ay),flush=True)
    del x,y;torch.cuda.empty_cache()
BASE_BIT=all(r['kv']['bit_equal'] for r in BASELINES)
BASE_BEHAVIOR=all(r['answer_equal'] for r in BASELINES)
check('DETERMINISTIC_ANSWER_BASELINE',BASE_BEHAVIOR)
print('[2/5] 72 views × 5 removal positions = 360 comparisons...',flush=True)
for wi in CASES:
    q,gold=question_for(wi)
    for slot in SLOTS:
        keys=ordered(PANEL[wi],slot)
        try:
            full,prefixes=build(keys,True);full_answer=ask(full,q)
            # H600b: a prefix cut must be identical to the original saved checkpoint.
            cut_n=memlength(prefixes[2]);cut_mem=tuple((k[:,:,:cut_n,:],v[:,:,:cut_n,:]) for k,v in full)
            rb=compare(cut_mem,prefixes[2]);prefix_answer=ask(prefixes[2],q);cut_answer=ask(cut_mem,q)
            ROLL.append(dict(case=wi,slot=slot,kv=rb,answer_equal=norm(prefix_answer)==norm(cut_answer),prefix_answer=prefix_answer,cut_answer=cut_answer))
            for index in range(5):
                rest=keys[:index]+keys[index+1:]
                clean,_=build(rest)
                rebuilt=rebuilt_without(keys,index,prefixes)
                d=compare(clean,rebuilt)
                clean_ans=ask(clean,q);rebuild_ans=ask(rebuilt,q)
                row=dict(case=wi,slot=slot,remove_index=index,removed=keys[index],target_removed=keys[index]==PANEL[wi]['target'],kv=d,answer_equal=norm(clean_ans)==norm(rebuild_ans),clean_answer=clean_ans,rebuilt_answer=rebuild_ans,clean_gold=bool(hit(clean_ans,gold)),rebuilt_gold=bool(hit(rebuild_ans,gold)))
                ROWS.append(row)
                naive=naive_without(full,keys,index);nd=compare(clean,naive)
                # Behavior probe only for removed target; numerical contamination checked at all positions.
                na=ask(naive,q) if row['target_removed'] else None
                NEG.append(dict(case=wi,slot=slot,remove_index=index,kv=nd,naive_answer=na,naive_gold=bool(hit(na,gold)) if na is not None else None))
                print(f"CASE={wi:02d} SLOT={slot:6s} REMOVE={index} TARGET={int(row['target_removed'])} REBUILD_BIT={int(d['bit_equal'])} REBUILD_MAX={d['max_abs']:.6g} ANSWER_EQ={int(row['answer_equal'])} NAIVE_UPPER_V={nd['upper_v_max']:.6g}",flush=True)
                del clean,rebuilt,naive
            del full,prefixes,cut_mem
        except Exception as e:
            ERRORS.append(dict(case=wi,slot=slot,error=repr(e)))
            print('ERROR',wi,slot,repr(e),flush=True);raise
    torch.cuda.empty_cache();gc.collect()
print('[3/5] Results and gates...',flush=True)
RBIT=sum(x['kv']['bit_equal'] for x in ROWS);RANS=sum(x['answer_equal'] for x in ROWS)
RB_BIT=sum(x['kv']['bit_equal'] for x in ROLL);RB_ANS=sum(x['answer_equal'] for x in ROLL)
NAIVE_DIFF=sum(not x['kv']['bit_equal'] for x in NEG)
NAIVE_UPPER=sum(x['kv']['upper_v_max']>0 for x in NEG)
TARGET_NEG=[x for x in NEG if x['naive_answer'] is not None]
NAIVE_GOLD=sum(x['naive_gold'] for x in TARGET_NEG)
print('H600a REBUILD_BIT',RBIT,'/',len(ROWS),'REBUILD_ANSWER',RANS,'/',len(ROWS),'MAX_KV',max((x['kv']['max_abs'] for x in ROWS),default=0))
print('H600b ROLLBACK_BIT',RB_BIT,'/',len(ROLL),'ROLLBACK_ANSWER',RB_ANS,'/',len(ROLL))
print('H600c NAIVE_DIFFERENT',NAIVE_DIFF,'/',len(NEG),'UPPER_V_RESIDUE',NAIVE_UPPER,'/',len(NEG),'NAIVE_TARGET_GOLD',NAIVE_GOLD,'/',len(TARGET_NEG))
check('COMPLETE_360',len(ROWS)==360 and len(NEG)==360)
check('COMPLETE_72_ROLLBACK',len(ROLL)==72)
check('NO_ERRORS',not ERRORS)
check('ROLLBACK_PREFIX_BITWISE',RB_BIT==72)
check('ROLLBACK_ANSWER_EQUAL',RB_ANS==72)
# Scientific hypothesis gates are reported, not used to suppress the full diagnostic record.
H600A_BIT=RBIT==360;H600A_ANSWER=RANS==360
H600C_RESIDUE=NAIVE_UPPER>0
print('H600a_BITWISE_STATUS', 'PASS' if H600A_BIT else ('INCONCLUSIVE_NUMERICAL' if not BASE_BIT else 'FAIL'))
print('H600a_BEHAVIOR_STATUS','PASS' if H600A_ANSWER else 'FAIL')
print('H600c_CONTAMINATION_STATUS','OBSERVED' if H600C_RESIDUE else 'NOT_OBSERVED')
print('[4/5] Frozen weight and hook integrity...',flush=True)
WA=weight_sha();check('WEIGHT_UNCHANGED',WB==WA)
check('SENTINEL_UNCHANGED',sentinel()==SB)
check('ZERO_TRAINABLE',sum(int(p.requires_grad) for p in model.parameters())==0)
remaining=[]
for li,layer in enumerate(model.model.layers):
    for label,module in (('layer',layer),('attention',layer.self_attn)):
        if module._forward_hooks or module._forward_pre_hooks:remaining.append((li,label))
check('NO_RESIDUAL_HOOKS',not remaining,hooks=remaining)
print('[5/5] JSON record...',flush=True)
PROTOCOL=dict(test='600',model=MODEL_ID,cut=CUT,source_blobs=BLOBS,panel_sha=PANEL_SHA,cases=CASES,slots=SLOTS,removals_per_view=5,engine='566 checkpoint/init/append/answer unchanged',training=False,retrieval=False,gradio=False,bitwise_baseline=BASE_BIT)
RECORD=dict(protocol=PROTOCOL,baseline=BASELINES,remove=ROWS,rollback=ROLL,negative=NEG,checks=CHECKS,errors=ERRORS,summary=dict(rebuild_bit=RBIT,rebuild_answer=RANS,rollback_bit=RB_BIT,rollback_answer=RB_ANS,naive_different=NAIVE_DIFF,naive_upper_residue=NAIVE_UPPER,naive_target_gold=NAIVE_GOLD,H600a_bitwise=H600A_BIT,H600a_answer=H600A_ANSWER,H600c_residue=H600C_RESIDUE),weight_before=WB,weight_after=WA,gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,transformers_version=transformers.__version__,seconds=round(time.perf_counter()-T600,2))
RECORD['result_sha']=digest(RECORD)
OUT='/content/AKBASCORE_TEST600_REMOVE_REBUILD_ROLLBACK.json';os.makedirs('/content',exist_ok=True)
with open(OUT,'w',encoding='utf-8') as f:json.dump(RECORD,f,ensure_ascii=False,indent=2)
print('='*132);print('TEST600 FINAL',RECORD['summary']);print('RESULT_SHA',RECORD['result_sha']);print('OUTPUT',OUT);print('SECONDS',RECORD['seconds'])
print('DECISION:', 'H600a AND H600b SUPPORTED; REVIEW H600c' if H600A_BIT and RB_BIT==72 else 'REVIEW NUMERICAL/LOGICAL DISCREPANCIES BEFORE TEST601')
print('NOTE: Naive deletion rephases surviving K but deliberately does not recompute contextual suffix K/V.')
print('NOTE: Gold after removing target is diagnostic, not a mandatory contamination signal.')
print('='*132)

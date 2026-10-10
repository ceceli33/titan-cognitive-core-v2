# AKBASCORE MAM - TEST597 - DEEPSEEK TARGETED DIAGNOSTIC - CORRECTED
# 24 worlds x FIRST/MIDDLE/LAST, 96 cartridges, 5 per case, panel SHA locked.
# CUT3 MLA WRITE/INIT/APPEND extracted without algorithmic changes from TEST590.
# Research test, NOT a sealed success claim. GPU run required. No Gradio.
# Tests 3 failed positions and 6 adjacent controls; no 72/72 success claim.
import os,sys,time,gc,re,json,hashlib,random,urllib.request,ast,traceback
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch,transformers
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
TEST='597';MODEL='deepseek-ai/DeepSeek-V2-Lite-Chat';REV='85864749cd611b4353ce1decdb286193298f64c7'
DTYPE=torch.bfloat16;SEED=577;CUT=3;NL=27;H=2048;MAX_NEW=24;BEAM_WIDTH=4;BEAM_STEPS=48
PANEL_SEED=550550;EXPECTED_PANEL_SHA='69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45'
SOURCES={'560.py':'35d95a929a4e043a9af346c67fc103da47a792b5','590.py':'09ae7a6862956128465efc7e76cdca1e478a1a52'}
TARGETS={(11,'LAST'),(14,'LAST'),(23,'MIDDLE')};DIAG_ROWS=[];CASE_IDS=(11,14,23)
random.seed(SEED);torch.manual_seed(SEED);torch.set_grad_enabled(False);T0=time.time()
GATES={};HOOK_COUNT={'init':0,'append':0};ROWS=[]
def gate(name,ok):
    GATES[name]=bool(ok);print('GATE',name,'PASS' if ok else 'FAIL',flush=True)
def git_blob(s):
    b=s.encode('utf-8');return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def sha_obj(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def fetch_verified(name):
    u='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/'+name
    req=urllib.request.Request(u,headers={'User-Agent':'AKBASCORE-TEST597'})
    with urllib.request.urlopen(req,timeout=60) as resp:s=resp.read().decode('utf-8')
    actual=git_blob(s);assert actual==SOURCES[name],f'{name} SOURCE BLOB MISMATCH {actual}'
    print('SOURCE_VERIFIED',name,actual,flush=True);return s
print('='*120);print('TEST597 - DEEPSEEK MLA CUT3 / TARGETED 3-FAIL DIAGNOSTIC');print('='*120)
source560=fetch_verified('560.py');source590=fetch_verified('590.py')
# Extract BASE literal without executing Mistral code or loading Mistral weights.
base_node=next(n for n in ast.parse(source560).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BASE' for t in n.targets))
BASE=ast.literal_eval(base_node.value)
assert len(BASE)==24
print('SOURCE_PANEL_BASE:',len(BASE),'worlds')
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'CUDA BF16 REQUIRED'
print('GPU:',torch.cuda.get_device_name(0),'TORCH:',torch.__version__,'TRANSFORMERS:',transformers.__version__)
def replace_hidden(out,new):
    if isinstance(out,tuple):return (new,)+out[1:]
    if isinstance(out,list):return [new]+out[1:]
    return new
old=globals().get('model',None)
reuse=isinstance(old,torch.nn.Module) and getattr(old,'_akbas_verified_revision',None)==REV and len(getattr(getattr(old,'model',None),'layers',[]))==NL and all(p.device.type=='cuda' and (not p.is_floating_point() or p.dtype==DTYPE) for p in old.parameters())
if reuse:model=old;print('MODEL_REUSED YES')
else:
    if isinstance(old,torch.nn.Module):globals().pop('model',None);del old;gc.collect();torch.cuda.empty_cache()
    print('MODEL_LOADING PINNED REVISION')
    cfg=AutoConfig.from_pretrained(MODEL,revision=REV,trust_remote_code=False);cfg._attn_implementation='eager'
    model,li=AutoModelForCausalLM.from_pretrained(MODEL,config=cfg,revision=REV,trust_remote_code=False,device_map={'':0},low_cpu_mem_usage=True,output_loading_info=True,attn_implementation='eager',dtype=DTYPE)
    for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'):assert not li.get(k,[]),f'LOAD ERROR {k}'
    model._akbas_verified_revision=REV
model.eval()
for pp in model.parameters():pp.requires_grad_(False)
base=model.model;LAYERS=base.layers;DEVICE=model.get_input_embeddings().weight.device
assert len(LAYERS)==NL and base.config.hidden_size==H and DEVICE.type=='cuda'
tok=globals().get('tok',None)
if tok is None or getattr(tok,'name_or_path',None)!=MODEL:tok=AutoTokenizer.from_pretrained(MODEL,revision=REV,trust_remote_code=False)
EOS_IDS={int(tok.eos_token_id)};eos_cfg=getattr(model.generation_config,'eos_token_id',None)
for x in (eos_cfg if isinstance(eos_cfg,(list,tuple)) else [eos_cfg]):
    if x is not None:EOS_IDS.add(int(x))
def ids(s):return tok(s,add_special_tokens=False,return_tensors='pt').input_ids.to(DEVICE)
def run(input_ids=None,inputs_embeds=None,past=None,attention_mask=None,position_ids=None,cache_position=None):
    with torch.inference_mode():return model(input_ids=input_ids,inputs_embeds=inputs_embeds,past_key_values=past,attention_mask=attention_mask,position_ids=position_ids,cache_position=cache_position,use_cache=True,return_dict=True,logits_to_keep=1)
def layers_of(c):
    assert hasattr(c,'layers') and len(c.layers)==NL
    out=[]
    for i,l in enumerate(c.layers):
        k=getattr(l,'keys',None);r=getattr(l,'values',None)
        assert torch.is_tensor(k) and torch.is_tensor(r) and k.ndim==4 and r.ndim==4
        assert k.shape[:3]==r.shape[:3] and k.shape[-1]==512 and r.shape[-1]==64,f'MLA CACHE L{i}'
        out.append((k,r))
    return out
def clone_layers(c):return tuple((k.detach().clone(),r.detach().clone()) for k,r in layers_of(c))
def make_cache(data):
    c=DynamicCache()
    for i,(k,r) in enumerate(data):c.update(k.detach().clone(),r.detach().clone(),i)
    return c
def length(data):return int(data[0][0].shape[-2])
def validate(data,n):
    assert len(data)==NL
    for i,(k,r) in enumerate(data):
        assert tuple(k.shape)==(1,1,n,512),f'LATENT L{i}: {tuple(k.shape)}'
        assert tuple(r.shape)==(1,1,n,64),f'ROTARY L{i}: {tuple(r.shape)}'
        assert torch.isfinite(k).all() and torch.isfinite(r).all()
    return True
def fingerprint():
    h=hashlib.sha256()
    for i in (0,3,6,12,20,26):
        pp=next(LAYERS[i].parameters()).detach().reshape(-1);n=pp.numel()
        for j in range(8):
            a=j*(n-128)//7;h.update(pp[a:a+128].float().cpu().numpy().tobytes())
    return h.hexdigest()
FP0=fingerprint();gate('MODEL_FROZEN',all(not pp.requires_grad for pp in model.parameters()))
# Exact 24-world fact/query definitions and panel RNG protocol of TEST560/575.
W=[]
for wi,(s,current,near,former,role_target) in enumerate(BASE):
    facts=[('CURRENT',f'The current capital of {s} is {current}.',current),('NEAR',f'The largest city of {s} is {near}.',near),('FORMER',f'The former capital of {s} was {former}.',former),('ROLE',f'The current capital of {role_target} is {s}.',s)]
    queries=[('CURRENT',f'What is the current capital of {s}?',current),('FORMER',f'What was the former capital of {s}?',former),('NEAR',f'What is the largest city of {s}?',near),('ROLE',f'What is the current capital of {role_target}?',s)]
    W.append({'id':wi,'facts':facts,'queries':queries})
CAR=[];KEY_TO_IDX={}
for wi,w in enumerate(W):
    for typ,fact,gold in w['facts']:
        ci=len(CAR);CAR.append({'idx':ci,'world':wi,'type':typ,'fact':fact,'gold':gold});KEY_TO_IDX[(wi,typ)]=ci
rng=random.Random(PANEL_SEED);pool=list(range(96));rng.shuffle(pool);PANEL=[];cursor=0
QUERY_CYCLE=['CURRENT','FORMER','NEAR','ROLE'];POSITIONS=['FIRST','MIDDLE','LAST']
for wi in range(24):
    typ=QUERY_CYCLE[wi%4];target=KEY_TO_IDX[(wi,typ)];others=[]
    while len(others)<4:
        ci=pool[cursor%96];cursor+=1
        if ci==target or CAR[ci]['world']==wi or any(CAR[x]['world']==CAR[ci]['world'] for x in others):continue
        others.append(ci)
    PANEL.append({'case':wi,'target':target,'others':others})
PANEL_SHA=sha_obj(PANEL);print('PANEL_SHA:',PANEL_SHA)
gate('PANEL_SHA_560_575',PANEL_SHA==EXPECTED_PANEL_SHA);assert GATES['PANEL_SHA_560_575']
def order_keys(z,slot):
    d=list(z['others']);t=z['target']
    return [t]+d if slot=='FIRST' else d[:2]+[t]+d[2:] if slot=='MIDDLE' else d+[t]
# DeepSeek-native chat wrapping; facts, distractors, order, questions and scoring are from TEST560/575.
SYSTEM='Answer the question using only the information provided. Give only the requested name and nothing else.'
PREFIX=tok.apply_chat_template([{'role':'user','content':SYSTEM+'\nINFORMATION:\n'}],tokenize=False,add_generation_prompt=False)
PREFIX_IDS=ids(PREFIX)[0].tolist();P=len(PREFIX_IDS)
for c in CAR:
    full=ids(PREFIX+c['fact']+'\n')[0].tolist()
    assert full[:P]==PREFIX_IDS,f'PREFIX MISMATCH ci={c["idx"]}'
    c['full']=full;c['body']=full[P:];c['body_len']=len(full)-P
def question(q):return tok.apply_chat_template([{'role':'user','content':q+' Answer with the name only.'}],tokenize=False,add_generation_prompt=True)
print('CARTRIDGES',len(CAR),'PREFIX',P,'BODY_LENGTHS',sorted(set(len(c['body']) for c in CAR)),flush=True)
def rephase_rotary(r,oldpos,newpos):
    assert r.shape[-1]==64 and oldpos.numel()==newpos.numel()==r.shape[-2]
    if torch.equal(oldpos,newpos):return r.detach().clone()
    n=oldpos.numel();emb=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE)
    with torch.inference_mode():
        f0=base.rotary_emb(emb,oldpos.unsqueeze(0)).to(torch.complex64)
        f1=base.rotary_emb(emb,newpos.unsqueeze(0)).to(torch.complex64)
    phase=f1*torch.conj(f0);z=torch.view_as_complex(r.float().reshape(1,1,n,32,2).contiguous())
    return torch.view_as_real(z*phase.unsqueeze(1)).reshape_as(r.float()).to(r.dtype)

def write_cartridge(ci):
    c=CAR[ci];x=torch.tensor([c['full']],device=DEVICE,dtype=torch.long);n=x.shape[1];pos=torch.arange(n,device=DEVICE);state={};handles=[]
    def capture(module,args,out):
        z=out[0] if isinstance(out,(tuple,list)) else out;state['h']=z.detach().clone()
    try:
        handles.append(LAYERS[CUT].register_forward_hook(capture))
        o=run(input_ids=x,attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos)
        data=clone_layers(o.past_key_values)
    finally:
        for h in handles:h.remove()
    assert state['h'].shape==(1,n,H);validate(data,n)
    return {'h':state['h'],'early':data[:CUT+1],'pos':pos.detach().clone(),'body_len':n-P,'native':data}
def init_cartridge(ci):
    cp=CP[ci];n=cp['h'].shape[1];seen=[0];handles=[]
    def inject(module,args,out):
        z=out[0] if isinstance(out,(tuple,list)) else out;assert z.shape==cp['h'].shape;seen[0]+=1
        return replace_hidden(out,cp['h'])
    try:
        handles.append(LAYERS[CUT].register_forward_hook(inject));pos=torch.arange(n,device=DEVICE)
        o=run(inputs_embeds=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos)
        upper=clone_layers(o.past_key_values)
    finally:
        for h in handles:h.remove()
    result=tuple((k.clone(),r.clone()) for k,r in cp['early'])+upper[CUT+1:];validate(result,n);HOOK_COUNT['init']+=seen[0]
    delta=max(float((a.float()-b.float()).abs().max().item()) for pa,pb in zip(result,cp['native']) for a,b in zip(pa,pb))
    return result,seen[0],delta
def append_cartridge(memory,ci):
    cp=CP[ci];oldn=length(memory);q=cp['body_len'];newpos=torch.arange(oldn,oldn+q,device=DEVICE);oldpos=cp['pos'][P:]
    body_h=cp['h'][:,P:,:];assert body_h.shape==(1,q,H)
    cache=make_cache(memory);calls=[0];handles=[]
    def inject(module,args,out):
        z=out[0] if isinstance(out,(tuple,list)) else out;assert z.shape==body_h.shape;calls[0]+=1
        return replace_hidden(out,body_h)
    try:
        handles.append(LAYERS[CUT].register_forward_hook(inject))
        o=run(inputs_embeds=torch.zeros((1,q,H),device=DEVICE,dtype=DTYPE),past=cache,attention_mask=torch.ones((1,oldn+q),device=DEVICE,dtype=torch.long),position_ids=newpos.unsqueeze(0),cache_position=newpos)
        grown=list(clone_layers(o.past_key_values))
    finally:
        for h in handles:h.remove()
    for li in range(CUT+1):
        oldk,oldr=memory[li];newk,newr=cp['early'][li]
        body_k=newk[:,:,P:,:].detach().clone();body_r=rephase_rotary(newr[:,:,P:,:],oldpos,newpos)
        grown[li]=(torch.cat((oldk,body_k),dim=-2),torch.cat((oldr,body_r),dim=-2))
    result=tuple(grown);validate(result,oldn+q);HOOK_COUNT['append']+=calls[0]
    unchanged=all(torch.equal(a,na[:,:,:oldn,:]) and torch.equal(b,nb[:,:,:oldn,:]) for (a,b),(na,nb) in zip(memory,result))
    return result,calls[0],unchanged
def joint_reference(order,stage_size):
    seq=list(PREFIX_IDS)
    for ci in order[:stage_size]:seq+=CAR[ci]['body']
    x=torch.tensor([seq],device=DEVICE,dtype=torch.long);n=x.shape[1];pos=torch.arange(n,device=DEVICE)
    o=run(input_ids=x,attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos)
    result=clone_layers(o.past_key_values);validate(result,n);return result
def prefix_forward(memory,qtext):
    cache=make_cache(memory);n=length(memory);qids=ids(qtext);ql=qids.shape[1];pos=torch.arange(n,n+ql,device=DEVICE)
    o=run(input_ids=qids,past=cache,attention_mask=torch.ones((1,n+ql),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos)
    return clone_layers(o.past_key_values),o.logits[:,-1,:].float()
def step_forward(cache_data,token):
    n=length(cache_data);pos=torch.tensor([n],device=DEVICE)
    o=run(input_ids=torch.tensor([[int(token)]],device=DEVICE,dtype=torch.long),past=make_cache(cache_data),attention_mask=torch.ones((1,n+1),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos)
    return clone_layers(o.past_key_values),o.logits[:,-1,:].float()
def greedy_from_prefix(cache_data,logits,max_new=MAX_NEW):
    generated=[];cache=cache_data
    for _ in range(max_new):
        t=int(logits.argmax(-1).item())
        if t in EOS_IDS:break
        generated.append(t);cache,logits=step_forward(cache,t)
    return tok.decode(generated,skip_special_tokens=True).strip(),generated
def beam_candidates(cache_data,logits,width=BEAM_WIDTH,steps=BEAM_STEPS):
    beams=[{'tokens':[],'logp':0.0,'cache':cache_data,'logits':logits,'done':False}]
    for step in range(steps):
        expanded=[]
        for b in beams:
            if b['done']:expanded.append(b);continue
            lp=torch.log_softmax(b['logits'],dim=-1);vals,idx=torch.topk(lp,width,dim=-1)
            for v,t in zip(vals[0].tolist(),idx[0].tolist()):
                t=int(t);expanded.append({'tokens':b['tokens']+[t],'logp':b['logp']+float(v),'cache':b['cache'],'logits':b['logits'],'done':t in EOS_IDS})
        expanded.sort(key=lambda x:x['logp'],reverse=True);selected=expanded[:width];new=[]
        for b in selected:
            if b['done']:new.append(b)
            else:
                c,l=step_forward(b['cache'],b['tokens'][-1]);b['cache']=c;b['logits']=l;new.append(b)
        beams=new
        if all(b['done'] for b in beams):break
    candidates=[];seen=set()
    for b in beams:
        raw=tok.decode([t for t in b['tokens'] if t not in EOS_IDS],skip_special_tokens=True).strip()
        if raw in seen:continue
        seen.add(raw);candidates.append({'text':raw,'tokens':b['tokens'],'logp':b['logp'],'length':len(b['tokens']),'ended':b['done']})
    return candidates


# WRITE is independent; CP created on demand to avoid 96-cartridge GPU OOM.
CP={}
def get_cp(ci):
    if ci not in CP:CP[ci]=write_cartridge(ci)
    return CP[ci]
# original engine expects CP[ci] - prefill selected 5 before each case.
def normalize(s):return ' '.join(re.sub(r'[^\w\s-]',' ',str(s).casefold()).split())
def select_name(greedy,candidates):
    # No gold, no list of known names, no oracle; use autonomous greedy output.
    # Completed beam retained as diagnostic, not used to change answer without verified name readout.
    return normalize(greedy)
def ask_name(memory,q):
    cache,logits=prefix_forward(memory,question(q))
    greedy,_=greedy_from_prefix(cache,logits)
    return {'answer':select_name(greedy,[]),'raw':greedy}
def indep_reference(keys):
    # Source-only independent cartridge caches, positioned in shared memory; no cross-cartridge interaction.
    n=P+sum(CAR[ci]['body_len'] for ci in keys)
    out=[]
    for li in range(NL):
        kk=[];rr=[];off=P
        for j,ci in enumerate(keys):
            cp=CP[ci];k,r=cp['native'][li]
            if j==0:kk.append(k[:,:,:P,:]);rr.append(r[:,:,:P,:])
            q=cp['body_len'];oldpos=cp['pos'][P:];newpos=torch.arange(off,off+q,device=DEVICE)
            kk.append(k[:,:,P:,:]);rr.append(rephase_rotary(r[:,:,P:,:],oldpos,newpos));off+=q
        out.append((torch.cat(kk,dim=-2),torch.cat(rr,dim=-2)))
    result=tuple(out);validate(result,n);return result
# The frozen 596 engine above is unchanged. Diagnostic readout never changes an answer.
def completed_name_candidates(memory,q):
    cache,logits=prefix_forward(memory,question(q))
    cand=beam_candidates(cache,logits,width=BEAM_WIDTH,steps=BEAM_STEPS)
    rows=[]
    for rank,c in enumerate(sorted(cand,key=lambda x:x['logp'],reverse=True),1):
        name=normalize(c['text'])
        rows.append({'rank':rank,'raw':c['text'],'name':name,'ended':bool(c['ended']),'logp':round(float(c['logp']),7),'tokens':len(c['tokens'])})
    # Selection is strictly gold-blind and diagnostic, never used to replace greedy.
    eligible=[r for r in rows if r['ended'] and re.fullmatch(r'[a-z]+',r['name'])]
    selected=eligible[0]['name'] if eligible else None
    return rows,selected
print('ENGINE_FUNCTIONS_READY; TEST596 WRITE/INIT/APPEND UNCHANGED; TARGETED FAILURE TRIAGE',flush=True)
for wi in CASE_IDS:
    z=PANEL[wi];typ=CAR[z['target']]['type'];q=next(x[1] for x in W[wi]['queries'] if x[0]==typ);gold=CAR[z['target']]['gold']
    for slot in POSITIONS:
        keys=order_keys(z,slot);CP.clear();gc.collect()
        for ci in keys:CP[ci]=write_cartridge(ci)
        memory,ic,delta=init_cartridge(keys[0]);assert ic==1 and delta==0.0
        preserved=True
        for ci in keys[1:]:
            newer,calls,ok=append_cartridge(memory,ci);preserved=preserved and ok and calls==1
            del memory;memory=newer
        assert preserved
        expected_len=P+sum(CAR[ci]['body_len'] for ci in keys)
        assert length(memory)==expected_len
        target=(wi,slot) in TARGETS
        arms={'INCR_DC3_MLA':memory}
        if target:
            arms['JOINT']=joint_reference(keys,len(keys))
            arms['NATIVE_INDEP_MLA']=indep_reference(keys)
        for arm,m in arms.items():
            a=ask_name(m,q);ok=a['answer']==normalize(gold)
            row={'case':wi,'slot':slot,'arm':arm,'question':q,'gold':gold,'answer':a['raw'],'normalized':a['answer'],'correct':ok,'cache_tokens':length(m),'keys':keys}
            ROWS.append(row)
            print('CASE',f'{wi:02d}','SLOT',slot,'ARM',arm,'PASS' if ok else 'FAIL','GOLD',gold,'ANSWER',repr(a['raw']),flush=True)
            if target:
                candidates,blind=completed_name_candidates(m,q)
                diag={'case':wi,'slot':slot,'arm':arm,'gold':gold,'greedy':a['answer'],'greedy_correct':ok,'completed_candidates':candidates,'gold_blind_completed_choice':blind,'gold_blind_choice_correct':blind==normalize(gold),'gold_in_candidates':any(c['ended'] and c['name']==normalize(gold) for c in candidates)}
                DIAG_ROWS.append(diag)
                print('DIAG',json.dumps(diag,ensure_ascii=False),flush=True)
        del arms,memory,m,newer;CP.clear();gc.collect();torch.cuda.empty_cache()
    print('CASE_COMPLETE',wi,flush=True)
scores={}
for arm in sorted({r['arm'] for r in ROWS}):
    subset=[r for r in ROWS if r['arm']==arm]
    scores[arm]={'correct':sum(r['correct'] for r in subset),'total':len(subset),'by_slot':{s:sum(r['correct'] for r in subset if r['slot']==s) for s in POSITIONS}}
print('SCORES',json.dumps(scores,ensure_ascii=False),flush=True)
gate('HOOKS_CLEAN',all(not l._forward_hooks and not l._forward_pre_hooks for l in LAYERS))
gate('FINGERPRINT_UNCHANGED',fingerprint()==FP0)
gate('WEIGHTS_FROZEN',all(not pp.requires_grad for pp in model.parameters()))
gate('TARGET_PANEL_COMPLETE',len([r for r in ROWS if r['arm']=='INCR_DC3_MLA'])==9 and len(DIAG_ROWS)==9)
replay={(r['case'],r['slot']):r['correct'] for r in ROWS if r['arm']=='INCR_DC3_MLA'}
gate('TEST596_NINE_REPLAY_MATCH',len(replay)==9 and all(replay[(wi,slot)]==((wi,slot) not in TARGETS) for wi in CASE_IDS for slot in POSITIONS))
faildiag=[r for r in DIAG_ROWS if r['arm']=='INCR_DC3_MLA']
gate('TEST596_FAILURES_REPRODUCED',len(faildiag)==3 and all(not r['greedy_correct'] for r in faildiag))
gate('ALL_GOLD_VISIBLE_IN_BEAM',len(faildiag)==3 and all(r['gold_in_candidates'] for r in faildiag))
gate('GOLD_BLIND_CHOICES_ALL_CORRECT',len(faildiag)==3 and all(r['gold_blind_choice_correct'] for r in faildiag))
result={'test':TEST,'model':MODEL,'revision':REV,'source_blobs':SOURCES,'panel_sha':PANEL_SHA,'scope':'TEST596 3 failures plus six same-case controls; 3 arms on failures; completed beam diagnostic only; no corrected 72/72 claim','targets':sorted([list(x) for x in TARGETS]),'scores':scores,'gates':GATES,'rows':ROWS,'diagnostics':DIAG_ROWS,'elapsed_seconds':round(time.time()-T0,2)}
out=Path('/content/AKBASCORE_TEST597_DEEPSEEK_FAILURE_DIAGNOSTIC.json' if Path('/content').is_dir() else '/tmp/AKBASCORE_TEST597_DEEPSEEK_FAILURE_DIAGNOSTIC.json')
out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print('RESULT_FILE',out,'ELAPSED',result['elapsed_seconds'],'s',flush=True)
print('INFRASTRUCTURE','PASS' if all(GATES.get(k,False) for k in ('PANEL_SHA_560_575','HOOKS_CLEAN','FINGERPRINT_UNCHANGED','WEIGHTS_FROZEN','TARGET_PANEL_COMPLETE','TEST596_NINE_REPLAY_MATCH')) else 'FAIL')
print('DIAGNOSTIC_DECISION','CANDIDATE_POLICY_WORTH_FULL_72_REGRESSION' if GATES['TEST596_NINE_REPLAY_MATCH'] and GATES['GOLD_BLIND_CHOICES_ALL_CORRECT'] else 'NO_VERIFIED_READOUT_FIX_DEMO_WITH_69_72_DISCLOSURE')
print('NOTE: gold is used ONLY for evaluation; no gold-based selection, no changes to TEST596 engine, no source replay during APPEND/ASK.')

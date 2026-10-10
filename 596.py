# AKBASCORE MAM - DEEPSEEK TEST596 - TEST560/575 CANONICAL PANEL TRANSFER
# 24 worlds x FIRST/MIDDLE/LAST, 96 cartridges, 5 per case, panel SHA locked.
# CUT3 MLA WRITE/INIT/APPEND extracted without algorithmic changes from TEST590.
# Demonstration backend, not an independent validation. GPU run required. No Gradio.
# JOINT and INDEP optional controls, default runs full 72 incremental cases.
import os,sys,time,gc,re,json,hashlib,random,urllib.request,ast,traceback
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch,transformers
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
TEST='DEEPSEEK_DEMO_V2';MODEL='deepseek-ai/DeepSeek-V2-Lite-Chat';REV='85864749cd611b4353ce1decdb286193298f64c7'
DTYPE=torch.bfloat16;SEED=577;CUT=3;NL=27;H=2048;MAX_NEW=24;BEAM_WIDTH=4;BEAM_STEPS=48
PANEL_SEED=550550;EXPECTED_PANEL_SHA='69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45'
SOURCES={'560.py':'35d95a929a4e043a9af346c67fc103da47a792b5','590.py':'09ae7a6862956128465efc7e76cdca1e478a1a52'}
RUN_JOINT=os.getenv('AKBAS_RUN_JOINT','0')=='1';RUN_INDEP=os.getenv('AKBAS_RUN_INDEP','0')=='1'
START_CASE=int(os.getenv('AKBAS_START_CASE','0'));END_CASE=int(os.getenv('AKBAS_END_CASE','24'))
assert 0<=START_CASE<END_CASE<=24
random.seed(SEED);torch.manual_seed(SEED);torch.set_grad_enabled(False);T0=time.time()
GATES={};HOOK_COUNT={'init':0,'append':0};ROWS=[]
def gate(name,ok):
    GATES[name]=bool(ok);print('GATE',name,'PASS' if ok else 'FAIL',flush=True)
def git_blob(s):
    b=s.encode('utf-8');return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def sha_obj(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def fetch_verified(name):
    u='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/'+name
    req=urllib.request.Request(u,headers={'User-Agent':'AKBASCORE-TEST596'})
    with urllib.request.urlopen(req,timeout=60) as resp:s=resp.read().decode('utf-8')
    actual=git_blob(s);assert actual==SOURCES[name],f'{name} SOURCE BLOB MISMATCH {actual}'
    print('SOURCE_VERIFIED',name,actual,flush=True);return s
print('='*120);print('TEST596 - DEEPSEEK MLA CUT3 / EXACT TEST560-575 PANEL TRANSFER');print('='*120)
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

# ===== DEMO ADAPTER: no changes above this boundary to TEST596 CUT3 engine =====
from datetime import datetime,timezone
from threading import RLock
from collections import Counter
_DEMO_LOCK=RLock()
_ACTIVE=None
_DEMO_ROWS=[]
_DEMO_EVENTS=[]
REFERENCE={'TEST596':{'correct':69,'total':72,'by_slot':{'FIRST':24,'MIDDLE':23,'LAST':22},'failures':[(11,'LAST'),(14,'LAST'),(23,'MIDDLE')]},'TEST597':{'joint':(3,3),'incremental':(6,9),'independent':(0,3),'readout_fix_verified':False}}
_DEMO_DIR=Path('/content/AKBASCORE_DEEPSEEK_DEMO' if Path('/content').is_dir() else '/tmp/AKBASCORE_DEEPSEEK_DEMO');_DEMO_DIR.mkdir(parents=True,exist_ok=True)
def _utc():return datetime.now(timezone.utc).isoformat(timespec='seconds')
def _sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,default=str).encode('utf-8')).hexdigest()
def _event(kind,**kw):
    e={'utc':_utc(),'kind':kind,**kw};_DEMO_EVENTS.append(e);return e
def _clear_active():
    global _ACTIVE
    _ACTIVE=None;CP.clear();gc.collect();torch.cuda.empty_cache()
def _assert_clean():
    assert all(not l._forward_hooks and not l._forward_pre_hooks for l in LAYERS),'RESIDUAL HOOKS'
    assert fingerprint()==FP0,'WEIGHT FINGERPRINT DRIFT'
    assert all(not p.requires_grad for p in model.parameters()),'TRAINABLE WEIGHTS'
    assert not model.training,'MODEL TRAINING MODE'
    assert PANEL_SHA==EXPECTED_PANEL_SHA,'PANEL SHA MISMATCH'
def demo_cases():
    return [{'case':wi,'type':CAR[z['target']]['type'],'question':next(x[1] for x in W[wi]['queries'] if x[0]==CAR[z['target']]['type']),'slots':list(POSITIONS)} for wi,z in enumerate(PANEL)]
def demo_status():
    return {'backend':'AKBASCORE MAM DeepSeek demo V2','model':MODEL,'revision':REV,'model_frozen':all(not p.requires_grad for p in model.parameters()),'cut':CUT,'layers':NL,'hidden':H,'cache':'MLA latent 512 + rotary 64','panel_sha':PANEL_SHA,'panel_verified':PANEL_SHA==EXPECTED_PANEL_SHA,'cartridges':len(CAR),'cases':len(PANEL),'source_blobs':dict(SOURCES),'reference_results':REFERENCE,'engine':'TEST596 unchanged numerical WRITE/INIT/APPEND/greedy ASK','scope':'same logical TEST560/575 panel, DeepSeek-specific tokenization; not TEST575 five-arm equivalence','active':None if _ACTIVE is None else {k:v for k,v in _ACTIVE.items() if k not in ('memory','keys')},'run_count':len(_DEMO_ROWS)}
def demo_load_case(case_id,slot='MIDDLE',progress=None):
    global _ACTIVE
    with _DEMO_LOCK:
        _assert_clean();wi=int(case_id);slot=str(slot).upper()
        if wi not in range(24) or slot not in POSITIONS:raise ValueError('case must be 0..23; slot FIRST/MIDDLE/LAST')
        _clear_active();z=PANEL[wi];keys=order_keys(z,slot);typ=CAR[z['target']]['type']
        q=next(x[1] for x in W[wi]['queries'] if x[0]==typ);gold=CAR[z['target']]['gold']
        try:
            for ci in keys:CP[ci]=write_cartridge(ci)
            if progress:progress({'step':1,'total':len(keys),'event':'WRITE/INIT','cartridge':0,'memory_tokens':0})
            mem,calls,delta=init_cartridge(keys[0]);assert calls==1 and delta==0.0,'INIT DIVERGENCE'
            steps=[{'stage':1,'cartridge':keys[0],'memory_tokens':length(mem),'old_prefix_unchanged':True}]
            for j,ci in enumerate(keys[1:],2):
                oldn=length(mem);grown,calls,ok=append_cartridge(mem,ci)
                assert calls==1 and ok,'APPEND NON-DESTRUCTIVE GATE FAILED'
                assert length(grown)==oldn+CAR[ci]['body_len'],'APPEND LENGTH GATE FAILED'
                steps.append({'stage':j,'cartridge':ci,'memory_tokens':length(grown),'old_prefix_unchanged':ok})
                if progress:progress({'step':j,'total':len(keys),'event':'APPEND','cartridge':j-1,'memory_tokens':length(grown)})
                del mem;mem=grown
            assert length(mem)==P+sum(CAR[ci]['body_len'] for ci in keys),'TOTAL LENGTH'
            _ACTIVE={'case':wi,'slot':slot,'keys':keys,'memory':mem,'question':q,'gold':gold,'type':typ,'steps':steps,'memory_tokens':length(mem),'loaded_utc':_utc()}
            _assert_clean();_event('case_loaded',case=wi,slot=slot,steps=steps)
            return {'case':wi,'slot':slot,'question':q,'type':typ,'memory_tokens':length(mem),'steps':steps,'source_text_supplied_at_query':False,'weight_fingerprint_unchanged':True,'append_only_verified':True}
        except BaseException:
            _clear_active();raise
def demo_ask(question_text=None,diagnostic_beam=False):
    with _DEMO_LOCK:
        if _ACTIVE is None:raise RuntimeError('Call demo_load_case first')
        _assert_clean();a=_ACTIVE
        q=a['question'] if question_text is None else str(question_text).strip()
        if not q or len(q)>1000:raise ValueError('Question length must be 1..1000')
        # The source text is not replayed. This calls TEST596 original ASK path.
        output=ask_name(a['memory'],q)
        row={'utc':_utc(),'case':a['case'],'slot':a['slot'],'arm':'INCR_DC3_MLA','question':q,'answer':output['raw'],'normalized':output['answer'],'memory_tokens':a['memory_tokens'],'is_locked_panel_question':q==a['question']}
        if row['is_locked_panel_question']:
            row['gold']=a['gold'];row['correct']=output['answer']==normalize(a['gold'])
        else:
            row['gold']=None;row['correct']=None
        if diagnostic_beam:
            cache,logits=prefix_forward(a['memory'],question(q));cands=beam_candidates(cache,logits)
            row['beam_candidates']=[{'raw':c['text'],'logp':round(float(c['logp']),7),'ended':bool(c['ended']),'tokens':int(c['length'])} for c in cands]
            row['beam_diagnostic_only']=True
        _DEMO_ROWS.append(row);_assert_clean();_event('question_answered',case=a['case'],slot=a['slot'],correct=row['correct'])
        return dict(row)
def demo_run_case(case_id,slot='MIDDLE',diagnostic_beam=False,progress=None):
    info=demo_load_case(case_id,slot,progress=progress);ans=demo_ask(diagnostic_beam=diagnostic_beam)
    return {'load':info,'answer':ans,'status':demo_status()}
def demo_verify_all(progress=None,save=True):
    # Re-run exact locked TEST596 72 queries. This is independent of the published 69/72 reference record.
    with _DEMO_LOCK:
        _assert_clean();rows=[];t0=time.time()
        for wi in range(24):
            for slot in POSITIONS:
                info=demo_load_case(wi,slot);r=demo_ask();rows.append(r)
                if progress:progress({'done':len(rows),'total':72,'case':wi,'slot':slot,'correct':r['correct']})
        by_slot={s:sum(r['correct'] for r in rows if r['slot']==s) for s in POSITIONS}
        correct=sum(r['correct'] for r in rows);failures=[{'case':r['case'],'slot':r['slot'],'expected':r['gold'],'actual':r['answer']} for r in rows if not r['correct']]
        match=(correct==69 and by_slot==REFERENCE['TEST596']['by_slot'] and {(r['case'],r['slot']) for r in rows if not r['correct']}==set(REFERENCE['TEST596']['failures']))
        record={'utc':_utc(),'model':MODEL,'revision':REV,'panel_sha':PANEL_SHA,'scope':'72-query locked panel re-run, no gold-based answer selection','correct':correct,'total':72,'by_slot':by_slot,'failures':failures,'matches_test596_reference':match,'weight_fingerprint_unchanged':fingerprint()==FP0,'seconds':round(time.time()-t0,2),'rows':rows}
        record['payload_sha256']=_sha(record)
        if save:
            p=_DEMO_DIR/f'verification_{int(time.time())}.json';p.write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8');record['saved_to']=str(p)
        _assert_clean();_event('panel_verification',correct=correct,match=match)
        return record
def demo_export():
    with _DEMO_LOCK:
        _assert_clean()
        payload={'utc':_utc(),'engine':'TEST596 DeepSeek CUT3 MLA, numerical functions unchanged','reference':REFERENCE,'status':demo_status(),'rows':list(_DEMO_ROWS),'events':list(_DEMO_EVENTS),'note':'Reference scores are historical, not a substitute for live verification. Custom questions are unscored.'}
        payload['sha256']=_sha(payload)
        p=_DEMO_DIR/f'demo_record_{int(time.time())}.json';p.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
        return {'path':str(p),'sha256':payload['sha256'],'rows':len(_DEMO_ROWS)}
def demo_selftest(gpu=False):
    with _DEMO_LOCK:
        _assert_clean()
        assert len(PANEL)==24 and len(CAR)==96 and len(demo_cases())==24
        assert all(len(order_keys(z,s))==5 and len(set(order_keys(z,s)))==5 for z in PANEL for s in POSITIONS)
        assert REFERENCE['TEST596']['correct']==69 and sum(REFERENCE['TEST596']['by_slot'].values())==69
        result={'static':'PASS','panel_sha':PANEL_SHA,'weight_fingerprint_unchanged':True,'gpu_case':None}
        if gpu:
            # Non-oracle runtime smoke test, known TEST596 passing case 0/FIRST.
            r=demo_run_case(0,'FIRST');result['gpu_case']={'case':0,'slot':'FIRST','correct':r['answer']['correct'],'answer':r['answer']['answer']}
            assert r['answer']['correct'],'SMOKE TEST CASE 0 FAILED'
        return result
print('DEEPSEEK_DEMO_BACKEND_READY; no Gradio; functions: demo_cases, demo_status, demo_load_case, demo_ask, demo_run_case, demo_verify_all, demo_export, demo_selftest',flush=True)
print('REFERENCE_TEST596: 69/72; TEST597: no verified readout fix; not claiming 72/72',flush=True)

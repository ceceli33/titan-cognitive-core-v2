# TEST594 — AKBASCORE MAM · TOKENWISE CODE-SELECTION INTERFERENCE X-RAY
# TEST588 CUT3 ENGINE · TEST590 FROZEN ENGINE · TARGETED STAGE 6/7/8 · NO GOLD INJECTION
import os,time,gc,re,hashlib,random
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch,transformers
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
TEST='594';MODEL='deepseek-ai/DeepSeek-V2-Lite-Chat';REV='85864749cd611b4353ce1decdb286193298f64c7'
DTYPE=torch.bfloat16;SEED=577;CUT=3;NL=27;H=2048;MAX_NEW=24;BEAM_WIDTH=4;BEAM_STEPS=48
random.seed(SEED);torch.manual_seed(SEED);torch.set_grad_enabled(False);T0=time.time()
GATES={};ROWS=[];META=[];HOOK_COUNT={'init':0,'append':0};ORDERS=[tuple(range(8)),tuple(reversed(range(8)))]
print('='*145,'\nTEST594 — AKBASCORE MAM · TOKENWISE CODE-SELECTION INTERFERENCE X-RAY\n'+'='*145)
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'CUDA BF16 REQUIRED'
print('GPU:',torch.cuda.get_device_name(0),'TORCH:',torch.__version__,'TRANSFORMERS:',transformers.__version__)
def gate(name,ok):
    GATES[name]=bool(ok);print('GATE',name,'PASS' if ok else 'FAIL',flush=True)
def replace_hidden(out,new):
    if isinstance(out,tuple):return (new,)+out[1:]
    if isinstance(out,list):return [new]+out[1:]
    return new
print('\n[1/15] MODEL / PINNED BASELINE')
old=globals().get('model',None)
reuse=isinstance(old,torch.nn.Module) and getattr(old,'_akbas_verified_revision',None)==REV and len(getattr(getattr(old,'model',None),'layers',[]))==NL and all(p.device.type=='cuda' and (not p.is_floating_point() or p.dtype==DTYPE) for p in old.parameters())
if reuse:model=old;print('MODEL_REUSED: YES — VERIFIED REVISION')
else:
    if isinstance(old,torch.nn.Module):globals().pop('model',None);del old;gc.collect();torch.cuda.empty_cache()
    print('MODEL_REUSED: NO — LOADING PINNED BASELINE')
    cfg=AutoConfig.from_pretrained(MODEL,revision=REV,trust_remote_code=False);cfg._attn_implementation='eager'
    model,li=AutoModelForCausalLM.from_pretrained(MODEL,config=cfg,revision=REV,trust_remote_code=False,device_map={'':0},low_cpu_mem_usage=True,output_loading_info=True,attn_implementation='eager',dtype=DTYPE)
    for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'):assert not li.get(k,[]),f'LOAD ERROR {k}: {str(li.get(k))[:300]}'
    model._akbas_verified_revision=REV
model.eval()
for p in model.parameters():p.requires_grad_(False)
base=model.model;LAYERS=base.layers;DEVICE=model.get_input_embeddings().weight.device
assert len(LAYERS)==NL and base.config.hidden_size==H and DEVICE.type=='cuda'
tok=globals().get('tok',None)
if tok is None or getattr(tok,'name_or_path',None)!=MODEL:tok=AutoTokenizer.from_pretrained(MODEL,revision=REV,trust_remote_code=False)
assert tok.eos_token_id is not None
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
        p=next(LAYERS[i].parameters()).detach().reshape(-1);n=p.numel()
        for j in range(8):
            a=j*(n-128)//7;h.update(p[a:a+128].float().cpu().numpy().tobytes())
    return h.hexdigest()
FP0=fingerprint();gate('MODEL_FROZEN',all(not p.requires_grad for p in model.parameters()))
print('PARAMETERS:',sum(p.numel() for p in model.parameters()),'TRAINABLE:',sum(p.numel() for p in model.parameters() if p.requires_grad))
print('\n[2/15] CANONICAL PANEL')
FACTS=[('VELORA-731','QN-4826'),('MIREX-204','KT-9173'),('TALVEX-583','RP-6402'),('NORVIA-926','DX-1857'),('CALDOR-417','BM-7391'),('ZENTRA-865','HV-2064'),('LUMEX-392','FS-5817'),('PAVORA-648','WC-9432')]
SYSTEM='Memorize these fictional records and answer questions using only the recorded facts. Answer with the requested code only.'
PREFIX=tok.apply_chat_template([{'role':'user','content':SYSTEM+'\nRECORDS:\n'}],tokenize=False,add_generation_prompt=False)
PREFIX_IDS=ids(PREFIX)[0].tolist();P=len(PREFIX_IDS);CAR=[]
for station,code in FACTS:
    fact=f'The access code for the fictional station {station} is {code}.';full=ids(PREFIX+fact+'\n')[0].tolist()
    assert full[:P]==PREFIX_IDS,f'PREFIX MISMATCH: {station}'
    CAR.append({'station':station,'code':code,'fact':fact,'body':full[P:],'full':full})
def question(station):return tok.apply_chat_template([{'role':'user','content':f'What is the access code for the fictional station {station}? Answer with the code only.'}],tokenize=False,add_generation_prompt=True)
gate('PANEL_LOCK',P==31 and len(CAR)==8 and [len(c['body']) for c in CAR[:2]]==[24,24] and len(set(x[0] for x in FACTS))==8 and len(set(x[1] for x in FACTS))==8)
print('PREFIX_TOKENS:',P,'BODY_LENGTHS:',[len(c['body']) for c in CAR],'CUT:',CUT)
assert GATES['PANEL_LOCK']
print('\n[3/15] MLA ROTARY TRANSPORT')
def rephase_rotary(r,oldpos,newpos):
    assert r.shape[-1]==64 and oldpos.numel()==newpos.numel()==r.shape[-2]
    if torch.equal(oldpos,newpos):return r.detach().clone()
    n=oldpos.numel();emb=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE)
    with torch.inference_mode():
        f0=base.rotary_emb(emb,oldpos.unsqueeze(0)).to(torch.complex64)
        f1=base.rotary_emb(emb,newpos.unsqueeze(0)).to(torch.complex64)
    phase=f1*torch.conj(f0);z=torch.view_as_complex(r.float().reshape(1,1,n,32,2).contiguous())
    return torch.view_as_real(z*phase.unsqueeze(1)).reshape_as(r.float()).to(r.dtype)
print('\n[4/15] EIGHT INDEPENDENT WRITES')
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
CP=[write_cartridge(i) for i in range(8)]
gate('WRITE_INDEPENDENT',len(CP)==8 and all(len(cp['early'])==CUT+1 and cp['body_len']>0 for cp in CP))
print('WRITE_LENGTHS:',[length(cp['native']) for cp in CP]);assert GATES['WRITE_INDEPENDENT']
print('\n[5/15] INIT — TEST588 ENGINE')
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
print('\n[6/15] APPEND — TEST588 ENGINE')
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
print('\n[7/15] TWO ORDERS / EIGHT STAGES')
MEM={};STAGES={}
for order in ORDERS:
    first,ic,delta=init_cartridge(order[0]);stage=[first];preserved=[];calls=[]
    for ci in order[1:]:
        nxt,ac,ok=append_cartridge(stage[-1],ci);stage.append(nxt);preserved.append(ok);calls.append(ac)
    STAGES[order]=stage;MEM[order]={'first':stage[0],'append':stage[-1]}
    META.append({'order':order,'init_calls':ic,'init_delta':delta,'append_calls':calls,'preserved':preserved})
    print('ORDER',order,'LENGTHS',[length(x) for x in stage],'INIT_DELTA',delta,'APPEND_CALLS',calls,'PRESERVED',preserved)
gate('INIT_EXACT_BOTH',len(META)==2 and all(m['init_calls']==1 and m['init_delta']==0.0 for m in META))
gate('APPEND_PRESERVED_ALL',len(META)==2 and all(len(m['append_calls'])==7 and all(x==1 for x in m['append_calls']) and all(m['preserved']) for m in META))
assert GATES['INIT_EXACT_BOTH'] and GATES['APPEND_PRESERVED_ALL']
print('\n[8/15] JOINT REFERENCES')
def joint_reference(order,stage_size):
    seq=list(PREFIX_IDS)
    for ci in order[:stage_size]:seq+=CAR[ci]['body']
    x=torch.tensor([seq],device=DEVICE,dtype=torch.long);n=x.shape[1];pos=torch.arange(n,device=DEVICE)
    o=run(input_ids=x,attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos)
    result=clone_layers(o.past_key_values);validate(result,n);return result
for order in ORDERS:MEM[order]['joint']=joint_reference(order,8)
print('\n[9/15] GREEDY / COMPLETED BEAM READOUT')
def code_contains(answer,code):return re.search(r'(?<![A-Za-z0-9-])'+re.escape(code)+r'(?![A-Za-z0-9-])',answer) is not None
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
def extract_candidate_code(text,station):
    hits=re.findall(r'(?<![A-Za-z0-9-])[A-Z]{2,8}-\d{2,8}(?![A-Za-z0-9-])',text)
    hits=[h for h in hits if h!=station]
    return hits[-1] if hits else None
def autonomous_select(candidates,station):
    pool=[]
    for c in candidates:
        if not c['ended']:continue
        code=extract_candidate_code(c['text'],station)
        if code is not None:pool.append((c['logp'],code,c['text']))
    if not pool:return None,None
    pool.sort(key=lambda x:x[0],reverse=True)
    return pool[0][1],pool[0][2]
def ask(memory,ci):
    station,gold=FACTS[ci];cache,logits=prefix_forward(memory,question(station));greedy,_=greedy_from_prefix(cache,logits)
    candidates=beam_candidates(cache,logits);selected,source=autonomous_select(candidates,station)
    return {'station':station,'gold':gold,'greedy':greedy,'greedy_exact':greedy.strip().strip(' .,:;"\'`')==gold,'selected':selected,'source':source,'exact':selected==gold,'contains':code_contains(greedy,gold),'wrong':any(code_contains(greedy,c) for _,c in FACTS if c!=gold),'candidates':candidates}
print('BEAM_WIDTH:',BEAM_WIDTH,'BEAM_STEPS:',BEAM_STEPS,'COMPLETED_ONLY: YES','GOLD_INJECTION: NO')

print('\n[10/15] THREE FROZEN MEMORY STATES — BASE6 / LUMEX / PAVORA')
forward=ORDERS[0];base6=STAGES[forward][5];lumex=STAGES[forward][6];pavora,pc,pp=append_cartridge(base6,7)
STATES={'BASE6':base6,'LUMEX7':lumex,'PAVORA7':pavora}
print('STATE_LENGTHS:',{k:length(v) for k,v in STATES.items()},'PAVORA_APPEND_CALLS:',pc,'PRESERVED:',pp)
gate('PAVORA_CONTROL_PRESERVED',pc==1 and pp and length(pavora)==length(base6)+CP[7]['body_len'])
print('\n[11/15] THREE-STATE AUTONOMOUS READOUT — NO GOLD INJECTION')
READS={}
for ci in (5,1):
    station=FACTS[ci][0]
    for label,memory in STATES.items():
        r=ask(memory,ci);READS[(station,label)]=r
        print('STATE',label,'STATION',station,'EXPECTED',r['gold'],'GREEDY',repr(r['greedy']),'SELECTED',repr(r['selected']),'EXACT',r['exact'])
        for rank,c in enumerate(r['candidates'],1):
            print(' RANK',rank,'LOGP',round(c['logp'],6),'EOS',c['ended'],'CODE',repr(extract_candidate_code(c['text'],station)) if c['ended'] else None)
print('\n[12/15] TOKENWISE BRANCH POINT — GREEDY PATHS / COMMON TEXT PREFIX')
# Compare greedy token streams directly. No target tokens are fed to the autonomous generator.
PATHS={}
for ci in (5,1):
    station=FACTS[ci][0]
    for label,memory in STATES.items():
        cache,logits=prefix_forward(memory,question(station));generated=[];steps=[]
        for step in range(MAX_NEW):
            lp=torch.log_softmax(logits[0],dim=-1);vals,idx=torch.topk(lp,k=5)
            token=int(idx[0]);steps.append({'step':step,'token':token,'piece':tok.decode([token]),'top':[(tok.decode([int(t)]),round(float(v),6)) for t,v in zip(idx.tolist(),vals.tolist())]})
            if token in EOS_IDS:break
            generated.append(token);cache,logits=step_forward(cache,token)
        PATHS[(station,label)]={'tokens':generated,'steps':steps}
        print('PATH',station,label,'TEXT',repr(tok.decode(generated,skip_special_tokens=True)))
    a=PATHS[(station,'BASE6')]['tokens'];b=PATHS[(station,'LUMEX7')]['tokens'];c=PATHS[(station,'PAVORA7')]['tokens'];n=min(len(a),len(b),len(c));div=next((i for i in range(n) if len({a[i],b[i],c[i]})>1),n)
    print('FIRST_GREEDY_DIVERGENCE',station,div,'COMMON_PREFIX',repr(tok.decode(a[:div],skip_special_tokens=True)))
    for label in STATES:
        steps=PATHS[(station,label)]['steps'];print(' BRANCH',label,'STEPS',steps[max(0,div-2):min(len(steps),div+6)])
print('\n[13/15] MATCHED-PREFIX TARGET-CODE LOG-LIKELIHOOD — DIAGNOSTIC TEACHER FORCING ONLY')
# The same fixed, gold-independent output prefix is used in all three memory states.
# Gold and distractor are scored as *diagnostic probes*, never inserted into beam candidates or autonomous selection.
# Scoring is conditional log P(code tokens | memory, query, fixed output prefix); it is NOT a probability of the entire free-form answer.
# Use actual text prefix observed in the original autonomous response before the code; no correct-code injection.
PROBES={};CODE_PAIRS={'ZENTRA-865':('HV-2064','RP-6402'),'MIREX-204':('KT-9173','BM-7391')}
def code_probe(memory,station,prefix,code):
    cache,logits=prefix_forward(memory,question(station))
    pref=ids(prefix)[0].tolist();prefix_steps=[]
    for t in pref:
        lp=torch.log_softmax(logits[0],dim=-1);prefix_steps.append(float(lp[t]));cache,logits=step_forward(cache,t)
    # Encode the continuation separately: no gold answer appears in the query or in autonomous beam search.
    suffix=ids(code)[0].tolist();scores=[]
    for t in suffix:
        lp=torch.log_softmax(logits[0],dim=-1);rank=int((lp>lp[t]).sum().item())+1
        scores.append({'token':int(t),'piece':tok.decode([int(t)]),'logp':float(lp[t]),'rank':rank,'top_token':tok.decode([int(lp.argmax().item())])})
        cache,logits=step_forward(cache,t)
    return {'sum':sum(x['logp'] for x in scores),'steps':scores,'prefix_token_count':len(pref),'suffix_token_count':len(suffix)}
for station,(correct,distractor) in CODE_PAIRS.items():
    # Fixed prefix is identical for the three arms, and contains neither candidate code.
    prefix=f'The access code for the fictional station {station} is '
    print('PROBE_STATION',station,'PREFIX',repr(prefix),'CORRECT',correct,'DISTRACTOR',distractor)
    for label,memory in STATES.items():
        good=code_probe(memory,station,prefix,correct);bad=code_probe(memory,station,prefix,distractor)
        margin=good['sum']-bad['sum'];PROBES[(station,label)]={'correct':good,'distractor':bad,'margin':margin}
        print(' STATE',label,'CORRECT_LOGP',round(good['sum'],6),'DISTRACTOR_LOGP',round(bad['sum'],6),'CORRECT_MINUS_DISTRACTOR',round(margin,6),'CORRECT_TOKENS',good['suffix_token_count'],'DISTRACTOR_TOKENS',bad['suffix_token_count'])
        print('  CORRECT_TOKEN_TRACE',[(x['piece'],round(x['logp'],6),x['rank'],x['top_token']) for x in good['steps']])
        print('  DISTRACTOR_TOKEN_TRACE',[(x['piece'],round(x['logp'],6),x['rank'],x['top_token']) for x in bad['steps']])
print('INTERPRETATION: MATCHED PREFIX PROBES ARE TEACHER-FORCED DIAGNOSTICS, NOT AUTONOMOUS GENERATION OR A GOLD-AWARE DECODER.')
print('INTERPRETATION: RAW SEQUENCE LOG-LIKELIHOODS DEPEND ON TOKENIZATION AND LENGTH. REPORT TOKEN COUNTS; DO NOT ASSUME EQUAL LENGTH.')
print('INTERPRETATION: A FLIPPED MARGIN LOCATES A CONDITIONAL CODE PREFERENCE, NOT A CAUSAL MODEL LAYER.')
z6=READS[('ZENTRA-865','BASE6')];zl=READS[('ZENTRA-865','LUMEX7')];zp=READS[('ZENTRA-865','PAVORA7')]
gate('THREE_STATE_PANEL_COMPLETE',len(READS)==6 and len(PATHS)==6 and len(PROBES)==6)
gate('ORIGINAL_ZENTRA_ERROR_REPRODUCED',z6['exact'] and zl['selected']=='RP-6402' and zp['exact'])
gate('LUMEX_CORRECT_CANDIDATE_VISIBLE',any(c['ended'] and extract_candidate_code(c['text'],'ZENTRA-865')=='HV-2064' for c in zl['candidates']))
gate('TOKEN_PROBES_COMPLETE',all(len(v['correct']['steps'])>0 and len(v['distractor']['steps'])>0 for v in PROBES.values()))
print('\n[14/15] INTEGRITY')
all_memories=[m for stages in STAGES.values() for m in stages]+[m for d in MEM.values() for m in d.values()]+list(STATES.values())
gate('CACHE_VALID',all(validate(m,length(m)) for m in all_memories))
gate('HOOK_COUNTS',HOOK_COUNT=={'init':2,'append':15})
gate('HOOKS_CLEAN',all(not l._forward_hooks and not l._forward_pre_hooks for l in LAYERS))
gate('FINGERPRINT_UNCHANGED',fingerprint()==FP0)
gate('WEIGHTS_FROZEN',all(not p.requires_grad for p in model.parameters()))
print('CUDA_ALLOCATED_GIB:',round(torch.cuda.memory_allocated()/2**30,3),'CUDA_PEAK_GIB:',round(torch.cuda.max_memory_allocated()/2**30,3))
print('\n[15/15] FINAL')
for name,value in GATES.items():print('GATE',name,'PASS' if value else 'FAIL')
infra=['MODEL_FROZEN','PANEL_LOCK','WRITE_INDEPENDENT','INIT_EXACT_BOTH','APPEND_PRESERVED_ALL','PAVORA_CONTROL_PRESERVED','THREE_STATE_PANEL_COMPLETE','TOKEN_PROBES_COMPLETE','CACHE_VALID','HOOK_COUNTS','HOOKS_CLEAN','FINGERPRINT_UNCHANGED','WEIGHTS_FROZEN']
infra_ok=all(GATES.get(k,False) for k in infra)
repro=all(GATES.get(k,False) for k in ('ORIGINAL_ZENTRA_ERROR_REPRODUCED','LUMEX_CORRECT_CANDIDATE_VISIBLE'))
print('INFRASTRUCTURE_STATUS:','PASS' if infra_ok else 'FAIL')
print('ERROR_REPRODUCTION_STATUS:','PASS' if repro else 'NOT_REPRODUCED')
print('STATUS:','PASS — TOKENWISE DIAGNOSTIC COMPLETE' if infra_ok and repro else 'DIAGNOSTIC COMPLETE — REVIEW RESULTS' if infra_ok else 'FAIL — INFRASTRUCTURE')
print('ENGINE: TEST593 / TEST592 / TEST588 CUT3 WRITE/INIT/APPEND UNCHANGED')
print('GOLD_CANDIDATE_INJECTION: NO\nFORCED_FIRST_TOKEN: NO\nSOURCE_REPLAY_DURING_APPEND_OR_ASK: NO\nCOMPRESSION: NOT YET')
print('AUTONOMOUS_READOUTS:',len(READS),'GREEDY_PATHS:',len(PATHS),'CONDITIONAL_CODE_PROBES:',len(PROBES)*2)
print('ELAPSED_SECONDS:',round(time.time()-T0,2));print('='*145)

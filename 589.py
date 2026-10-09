# TEST589 — AKBASCORE MAM · DEEPSEEK MLA FOUR-CARTRIDGE INCREMENTAL RETENTION
# TEST588 CUT3 ENGINE · 4 CARTRIDGES · 2 ORDERS · 44 READOUTS · NO GOLD INJECTION
import os,time,gc,re,hashlib,random
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch,transformers
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
TEST='589';MODEL='deepseek-ai/DeepSeek-V2-Lite-Chat';REV='85864749cd611b4353ce1decdb286193298f64c7'
DTYPE=torch.bfloat16;SEED=577;CUT=3;NL=27;H=2048;MAX_NEW=24;BEAM_WIDTH=4;BEAM_STEPS=48
random.seed(SEED);torch.manual_seed(SEED);torch.set_grad_enabled(False);T0=time.time()
GATES={};ROWS=[];META=[];HOOK_COUNT={'init':0,'append':0};ORDERS=[(0,1,2,3),(3,2,1,0)]
print('='*145,'\nTEST589 — AKBASCORE MAM · DEEPSEEK MLA FOUR-CARTRIDGE INCREMENTAL RETENTION\n'+'='*145)
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
FACTS=[('VELORA-731','QN-4826'),('MIREX-204','KT-9173'),('TALVEX-583','RP-6402'),('NORVIA-926','DX-1857')]
SYSTEM='Memorize these fictional records and answer questions using only the recorded facts. Answer with the requested code only.'
PREFIX=tok.apply_chat_template([{'role':'user','content':SYSTEM+'\nRECORDS:\n'}],tokenize=False,add_generation_prompt=False)
PREFIX_IDS=ids(PREFIX)[0].tolist();P=len(PREFIX_IDS);CAR=[]
for station,code in FACTS:
    fact=f'The access code for the fictional station {station} is {code}.';full=ids(PREFIX+fact+'\n')[0].tolist()
    assert full[:P]==PREFIX_IDS,f'PREFIX MISMATCH: {station}'
    CAR.append({'station':station,'code':code,'fact':fact,'body':full[P:],'full':full})
def question(station):return tok.apply_chat_template([{'role':'user','content':f'What is the access code for the fictional station {station}? Answer with the code only.'}],tokenize=False,add_generation_prompt=True)
gate('PANEL_LOCK',P==31 and len(CAR)==4 and [len(c['body']) for c in CAR[:2]]==[24,24] and len(set(x[0] for x in FACTS))==4 and len(set(x[1] for x in FACTS))==4)
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
print('\n[4/15] FOUR INDEPENDENT WRITES')
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
CP=[write_cartridge(i) for i in range(4)]
gate('WRITE_INDEPENDENT',len(CP)==4 and all(len(cp['early'])==CUT+1 and cp['body_len']>0 for cp in CP))
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
print('\n[7/15] TWO ORDERS / FOUR STAGES')
MEM={};STAGES={}
for order in ORDERS:
    first,ic,delta=init_cartridge(order[0]);stage=[first];preserved=[];calls=[]
    for ci in order[1:]:
        nxt,ac,ok=append_cartridge(stage[-1],ci);stage.append(nxt);preserved.append(ok);calls.append(ac)
    STAGES[order]=stage;MEM[order]={'first':stage[0],'append':stage[-1]}
    META.append({'order':order,'init_calls':ic,'init_delta':delta,'append_calls':calls,'preserved':preserved})
    print('ORDER',order,'LENGTHS',[length(x) for x in stage],'INIT_DELTA',delta,'APPEND_CALLS',calls,'PRESERVED',preserved)
gate('INIT_EXACT_BOTH',len(META)==2 and all(m['init_calls']==1 and m['init_delta']==0.0 for m in META))
gate('APPEND_PRESERVED_ALL',len(META)==2 and all(len(m['append_calls'])==3 and all(x==1 for x in m['append_calls']) and all(m['preserved']) for m in META))
assert GATES['INIT_EXACT_BOTH'] and GATES['APPEND_PRESERVED_ALL']
print('\n[8/15] JOINT REFERENCES')
def joint_reference(order,stage_size):
    seq=list(PREFIX_IDS)
    for ci in order[:stage_size]:seq+=CAR[ci]['body']
    x=torch.tensor([seq],device=DEVICE,dtype=torch.long);n=x.shape[1];pos=torch.arange(n,device=DEVICE)
    o=run(input_ids=x,attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos)
    result=clone_layers(o.past_key_values);validate(result,n);return result
for order in ORDERS:MEM[order]['joint']=joint_reference(order,4)
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
print('\n[10/15] INCREMENTAL RETENTION')
for order in ORDERS:
    for stage_size,memory in enumerate(STAGES[order],start=1):
        for ci in order[:stage_size]:
            r=ask(memory,ci);r.update({'order':order,'stage':stage_size,'arm':'incremental'});ROWS.append(r)
            print('ORDER',order,'STAGE',stage_size,'STATION',r['station'],'GOLD',r['gold'])
            print(' GREEDY',repr(r['greedy']),'EXACT',r['greedy_exact'])
            print(' AUTONOMOUS',repr(r['selected']),'EXACT',r['exact'])
            print(' CANDIDATES',[(x['text'],round(x['logp'],5),x['ended']) for x in r['candidates']])
print('\n[11/15] JOINT / FIRST / APPEND')
for order in ORDERS:
    for arm in ('joint','first','append'):
        memory=MEM[order][arm]
        for ci in range(4):
            r=ask(memory,ci);r.update({'order':order,'stage':4,'arm':arm});ROWS.append(r)
            print('ORDER',order,'ARM',arm,'STATION',r['station'],'GOLD',r['gold'],'GREEDY',repr(r['greedy']),'SELECTED',repr(r['selected']),'EXACT',r['exact'])
print('\n[12/15] SCORES')
for order in ORDERS:
    for stage_size in range(1,5):
        rr=[r for r in ROWS if r['arm']=='incremental' and r['order']==order and r['stage']==stage_size]
        print('ORDER',order,'STAGE',stage_size,'AUTONOMOUS_EXACT',sum(r['exact'] for r in rr),'/',len(rr),'GREEDY_EXACT',sum(r['greedy_exact'] for r in rr),'/',len(rr),'WRONG_GREEDY',sum(r['wrong'] for r in rr))
for arm in ('joint','first','append'):
    rr=[r for r in ROWS if r['arm']==arm]
    print('ARM',arm,'AUTONOMOUS_EXACT',sum(r['exact'] for r in rr),'/',len(rr),'GREEDY_EXACT',sum(r['greedy_exact'] for r in rr),'/',len(rr),'WRONG_GREEDY',sum(r['wrong'] for r in rr))
inc=[r for r in ROWS if r['arm']=='incremental'];fin=[r for r in ROWS if r['arm']=='append'];joint=[r for r in ROWS if r['arm']=='joint']
gate('INCREMENTAL_PANEL_COMPLETE',len(inc)==20)
gate('INCREMENTAL_RETENTION_ALL',len(inc)==20 and all(r['exact'] for r in inc))
gate('FOUR_CARTRIDGE_FINAL_8',len(fin)==8 and all(r['exact'] for r in fin))
gate('JOINT_REFERENCE_8',len(joint)==8 and all(r['exact'] for r in joint))
gate('FINAL_NO_WRONG_GREEDY',len(fin)==8 and all(not r['wrong'] for r in fin))
gate('FINAL_COMPLETED_CANDIDATES',len(fin)==8 and all(r['selected'] is not None for r in fin))
print('\n[13/15] FAILURE / ORDER SENSITIVITY')
for r in ROWS:
    if not r['exact']:print('FAIL','ORDER',r['order'],'STAGE',r['stage'],'ARM',r['arm'],'STATION',r['station'],'EXPECTED',r['gold'],'SELECTED',repr(r['selected']),'GREEDY',repr(r['greedy']))
for station,gold in FACTS:
    a=next(r for r in fin if r['order']==ORDERS[0] and r['station']==station)
    b=next(r for r in fin if r['order']==ORDERS[1] and r['station']==station)
    print('ORDER_COMPARE',station,'FORWARD',repr(a['selected']),'REVERSE',repr(b['selected']),'MATCH',a['selected']==b['selected'])
print('\n[14/15] INTEGRITY')
all_memories=[m for stages in STAGES.values() for m in stages]+[m for d in MEM.values() for m in d.values()]
gate('CACHE_VALID',all(validate(m,length(m)) for m in all_memories))
gate('HOOK_COUNTS',HOOK_COUNT=={'init':2,'append':6})
gate('HOOKS_CLEAN',all(not l._forward_hooks and not l._forward_pre_hooks for l in LAYERS))
gate('FINGERPRINT_UNCHANGED',fingerprint()==FP0)
gate('WEIGHTS_FROZEN',all(not p.requires_grad for p in model.parameters()))
print('CUDA_ALLOCATED_GIB:',round(torch.cuda.memory_allocated()/2**30,3),'CUDA_PEAK_GIB:',round(torch.cuda.max_memory_allocated()/2**30,3))
print('\n[15/15] FINAL')
for name,value in GATES.items():print('GATE',name,'PASS' if value else 'FAIL')
infra=['MODEL_FROZEN','PANEL_LOCK','WRITE_INDEPENDENT','INIT_EXACT_BOTH','APPEND_PRESERVED_ALL','CACHE_VALID','HOOK_COUNTS','HOOKS_CLEAN','FINGERPRINT_UNCHANGED','WEIGHTS_FROZEN']
infra_ok=all(GATES.get(k,False) for k in infra)
strict=all(GATES.get(k,False) for k in ('INCREMENTAL_PANEL_COMPLETE','INCREMENTAL_RETENTION_ALL','FOUR_CARTRIDGE_FINAL_8','FINAL_COMPLETED_CANDIDATES'))
print('INFRASTRUCTURE_STATUS:','PASS' if infra_ok else 'FAIL')
print('INCREMENTAL_RETENTION_STATUS:','PASS' if GATES['INCREMENTAL_RETENTION_ALL'] else 'FAIL')
print('FOUR_CARTRIDGE_READOUT_STATUS:','PASS' if GATES['FOUR_CARTRIDGE_FINAL_8'] else 'FAIL')
print('STRICT_BEHAVIOR_STATUS:','PASS' if strict else 'FAIL')
print('STATUS:','PASS — FOUR-CARTRIDGE INCREMENTAL RETENTION' if infra_ok and strict else 'DIAGNOSTIC COMPLETE — FOUR-CARTRIDGE RETENTION NOT YET VERIFIED')
print('ENGINE: TEST588 CUT3 WRITE/INIT/APPEND PRESERVED')
print('GOLD_CANDIDATE_INJECTION: NO\nFORCED_FIRST_TOKEN: NO\nSOURCE_REPLAY_DURING_APPEND_OR_ASK: NO\nCOMPRESSION: NOT YET')
print('INCREMENTAL_QUESTIONS:',len(inc),'FINAL_APPEND_QUESTIONS:',len(fin),'TOTAL_READOUTS:',len(ROWS))
print('ELAPSED_SECONDS:',round(time.time()-T0,2));print('='*145)

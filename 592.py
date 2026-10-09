# TEST592 — AKBASCORE MAM · SEVENTH-CARTRIDGE COUNTERFACTUAL CAUSAL ABLATION
# TEST588 CUT3 ENGINE · TEST590 FROZEN ENGINE · TARGETED STAGE 6/7/8 · NO GOLD INJECTION
import os,time,gc,re,hashlib,random
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch,transformers
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
TEST='592';MODEL='deepseek-ai/DeepSeek-V2-Lite-Chat';REV='85864749cd611b4353ce1decdb286193298f64c7'
DTYPE=torch.bfloat16;SEED=577;CUT=3;NL=27;H=2048;MAX_NEW=24;BEAM_WIDTH=4;BEAM_STEPS=48
random.seed(SEED);torch.manual_seed(SEED);torch.set_grad_enabled(False);T0=time.time()
GATES={};ROWS=[];META=[];HOOK_COUNT={'init':0,'append':0};ORDERS=[tuple(range(8)),tuple(reversed(range(8)))]
print('='*145,'\nTEST592 — AKBASCORE MAM · SEVENTH-CARTRIDGE COUNTERFACTUAL CAUSAL ABLATION\n'+'='*145)
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

print('\n[10/15] CAUSAL SEVENTH-CARTRIDGE ABLATION — FROZEN ENGINE')
# Fixed forward six-cartridge baseline. The seventh cartridge varies; no source replay.
forward=ORDERS[0];reverse=ORDERS[1];base6=STAGES[forward][5]
assert tuple(forward[:6])==(0,1,2,3,4,5)
# Original LUMEX is 6; PAVORA is 7. Also append already-present records as stress controls,
# explicitly labelled duplicates rather than independent facts.
VARIANTS=[('NO_APPEND',None),('LUMEX_ORIGINAL',6),('PAVORA_ALTERNATIVE',7),('TALVEX_DUPLICATE',2),('CALDOR_DUPLICATE',4)]
TEST_MEM={};ABLATED=[]
for label,ci in VARIANTS:
    if ci is None:memory=base6;calls=0;preserved=True
    else:memory,calls,preserved=append_cartridge(base6,ci)
    TEST_MEM[label]=memory
    print('VARIANT',label,'ADDED',CAR[ci]['station'] if ci is not None else 'NONE','LENGTH',length(memory),'HOOK_CALLS',calls,'PRESERVED',preserved)
    for target in (5,2,1):
        r=ask(memory,target);r.update({'variant':label,'target':target});ABLATED.append(r)
        print(' READ',CAR[target]['station'],'EXPECTED',r['gold'],'GREEDY',repr(r['greedy']),'SELECTED',repr(r['selected']),'EXACT',r['exact'])
        for rank,c in enumerate(r['candidates'],1):
            code=extract_candidate_code(c['text'],r['station']) if c['ended'] else None
            print('  RANK',rank,'LOGP',round(c['logp'],6),'EOS',c['ended'],'CODE',repr(code),'MATCH',code==r['gold'],'TEXT',repr(c['text']))
print('\n[11/15] ORDER / COMPOSITION COUNTERFACTUALS')
# Same eight distinct facts, changing only the last-two addition order.
# Both branches share the exact same six-cartridge prefix.
order_variants=[('ORIGINAL_6_7',(6,7)),('SWAPPED_7_6',(7,6))]
ORDER_ROWS=[];ORDER_MEM={}
for label,tail in order_variants:
    mem=base6;ok_all=True
    for ci in tail:
        mem,calls,ok=append_cartridge(mem,ci);ok_all=ok_all and ok and calls==1
    ORDER_MEM[label]=mem
    print('ORDER_VARIANT',label,'TAIL',tail,'LENGTH',length(mem),'PRESERVED',ok_all)
    for target in (5,2,1,6,7):
        r=ask(mem,target);r.update({'variant':label,'target':target});ORDER_ROWS.append(r)
        print(' READ',CAR[target]['station'],'EXPECTED',r['gold'],'GREEDY',repr(r['greedy']),'SELECTED',repr(r['selected']),'EXACT',r['exact'])
print('\n[12/15] TARGETED JOINT / APPEND COMPARISON')
CTRL=[]
for order in ORDERS:
    for arm in ('joint','append'):
        for target in (1,5):
            r=ask(MEM[order][arm],target);r.update({'order':order,'arm':arm,'target':target});CTRL.append(r)
            print('ORDER',order,'ARM',arm,'STATION',r['station'],'EXPECTED',r['gold'],'SELECTED',r['selected'],'EXACT',r['exact'])
            for rank,c in enumerate(r['candidates'],1):
                code=extract_candidate_code(c['text'],r['station']) if c['ended'] else None
                print('  RANK',rank,'LOGP',round(c['logp'],6),'CODE',repr(code),'MATCH',code==r['gold'])
print('\n[13/15] MARGINS / CAUSAL COMPARISONS')
def diagnostics(r):
    complete=[c for c in r['candidates'] if c['ended']]
    right=[c['logp'] for c in complete if extract_candidate_code(c['text'],r['station'])==r['gold']]
    wrong=[c['logp'] for c in complete if (v:=extract_candidate_code(c['text'],r['station'])) is not None and v!=r['gold']]
    return (max(right) if right else None,max(wrong) if wrong else None,(max(right)-max(wrong)) if right and wrong else None)
for r in ABLATED+ORDER_ROWS+CTRL:
    a,b,margin=diagnostics(r);r['margin']=margin
    print('MARGIN',r.get('variant',r.get('arm')),'ORDER',r.get('order','BASE6'),'STATION',r['station'],'CORRECT_LOGP',a,'WRONG_LOGP',b,'DELTA',margin,'EXACT',r['exact'])
def get_variant(rows,label,target):return next(r for r in rows if r['variant']==label and r['target']==target)
z6=get_variant(ABLATED,'NO_APPEND',5);z7=get_variant(ABLATED,'LUMEX_ORIGINAL',5);zp=get_variant(ABLATED,'PAVORA_ALTERNATIVE',5)
z8=get_variant(ORDER_ROWS,'ORIGINAL_6_7',5);z8swap=get_variant(ORDER_ROWS,'SWAPPED_7_6',5)
print('ZENTRA_CAUSAL_SEQUENCE:',[(label,r['selected'],r['exact'],r['margin']) for label,r in [('S6',z6),('S7_LUMEX',z7),('S7_PAVORA',zp),('S8_ORIGINAL',z8),('S8_SWAPPED',z8swap)]])
print('CAUSAL_INTERPRETATION: OBSERVATIONAL COUNTERFACTUAL; SINGLE DETERMINISTIC PANEL; NOT A GENERAL CAUSAL PROOF')
gate('BASELINE_STAGE6_ZENTRA_CORRECT',z6['exact'])
gate('ORIGINAL_STAGE7_ERROR_REPRODUCED',z7['selected']=='RP-6402')
gate('CORRECT_VISIBLE_IN_STAGE7',any(c['ended'] and extract_candidate_code(c['text'],z7['station'])==z7['gold'] for c in z7['candidates']))
gate('ORIGINAL_STAGE8_ZENTRA_CORRECT',z8['exact'])
gate('ALTERNATIVE_SEVENTH_TESTED',len(ABLATED)==15 and len(ORDER_ROWS)==10)
gate('CONTROLS_COMPLETE',len(CTRL)==8)
gate('COUNTERFACTUAL_PRESERVATION',all(length(TEST_MEM[label])==length(base6)+(0 if ci is None else CP[ci]['body_len']) for label,ci in VARIANTS))
print('\n[14/15] INTEGRITY')
all_memories=[m for stages in STAGES.values() for m in stages]+[m for d in MEM.values() for m in d.values()]+list(TEST_MEM.values())+list(ORDER_MEM.values())
gate('CACHE_VALID',all(validate(m,length(m)) for m in all_memories))
# Original 2 init + 14 append, plus 4 one-step variants + 4 two-step counterfactuals.
gate('HOOK_COUNTS',HOOK_COUNT=={'init':2,'append':22})
gate('HOOKS_CLEAN',all(not l._forward_hooks and not l._forward_pre_hooks for l in LAYERS))
gate('FINGERPRINT_UNCHANGED',fingerprint()==FP0)
gate('WEIGHTS_FROZEN',all(not p.requires_grad for p in model.parameters()))
print('CUDA_ALLOCATED_GIB:',round(torch.cuda.memory_allocated()/2**30,3),'CUDA_PEAK_GIB:',round(torch.cuda.max_memory_allocated()/2**30,3))
print('\n[15/15] FINAL')
for name,value in GATES.items():print('GATE',name,'PASS' if value else 'FAIL')
infra=['MODEL_FROZEN','PANEL_LOCK','WRITE_INDEPENDENT','INIT_EXACT_BOTH','APPEND_PRESERVED_ALL','CACHE_VALID','HOOK_COUNTS','HOOKS_CLEAN','FINGERPRINT_UNCHANGED','WEIGHTS_FROZEN','ALTERNATIVE_SEVENTH_TESTED','CONTROLS_COMPLETE','COUNTERFACTUAL_PRESERVATION']
infra_ok=all(GATES.get(k,False) for k in infra)
repro=all(GATES.get(k,False) for k in ('BASELINE_STAGE6_ZENTRA_CORRECT','ORIGINAL_STAGE7_ERROR_REPRODUCED','CORRECT_VISIBLE_IN_STAGE7','ORIGINAL_STAGE8_ZENTRA_CORRECT'))
print('INFRASTRUCTURE_STATUS:','PASS' if infra_ok else 'FAIL')
print('ORIGINAL_FAILURE_REPRODUCTION_STATUS:','PASS' if repro else 'NOT_REPRODUCED')
print('STATUS:','PASS — CAUSAL ABLATION COMPLETE' if infra_ok and repro else 'DIAGNOSTIC COMPLETE — REVIEW OBSERVATIONS' if infra_ok else 'FAIL — INFRASTRUCTURE')
print('ENGINE: TEST591 / TEST590 / TEST588 CUT3 WRITE/INIT/APPEND UNCHANGED')
print('GOLD_CANDIDATE_INJECTION: NO\nFORCED_FIRST_TOKEN: NO\nSOURCE_REPLAY_DURING_APPEND_OR_ASK: NO\nCOMPRESSION: NOT YET')
print('ABLATION_READOUTS:',len(ABLATED),'ORDER_READOUTS:',len(ORDER_ROWS),'CONTROL_READOUTS:',len(CTRL),'TOTAL_READOUTS:',len(ABLATED)+len(ORDER_ROWS)+len(CTRL))
print('ELAPSED_SECONDS:',round(time.time()-T0,2));print('='*145)

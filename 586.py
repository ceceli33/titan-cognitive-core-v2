
# TEST586 — AKBASCORE MAM · DEEPSEEK MLA READOUT CAUSAL DIAGNOSTICS
# TEST585 FROZEN ENGINE · CUT3 · TWO INDEPENDENT CARTRIDGES · BOTH ORDERS
# GREEDY / FORCED-FIRST-TOKEN / CONDITIONAL CODE LIKELIHOOD
# NO ENGINE MODIFICATION · NO TRAINING · NO SOURCE REPLAY DURING APPEND/ASK
# SINGLE GOOGLE COLAB CELL · A100-40GB · BF16
import os,time,gc,re,hashlib,math
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch,transformers
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
TEST="586";MODEL="deepseek-ai/DeepSeek-V2-Lite-Chat";REV="85864749cd611b4353ce1decdb286193298f64c7"
DTYPE=torch.bfloat16;SEED=577;CUT=3;NL=27;H=2048;MAX_NEW=24;TOPK=8
torch.manual_seed(SEED);torch.set_grad_enabled(False);T0=time.time()
GATES={};ROWS=[];LIKELIHOOD=[];HOOK_COUNT={"init":0,"append":0}
print("="*152)
print("TEST586 — AKBASCORE MAM · DEEPSEEK MLA READOUT CAUSAL DIAGNOSTICS")
print("="*152)
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(),"CUDA BF16 REQUIRED"
print("GPU:",torch.cuda.get_device_name(0),"TORCH:",torch.__version__,"TRANSFORMERS:",transformers.__version__)
def gate(name,condition):
    GATES[name]=bool(condition);print("GATE",name,"PASS" if condition else "FAIL",flush=True)
def replace_hidden(out,new):
    if isinstance(out,tuple):return (new,)+out[1:]
    if isinstance(out,list):return [new]+out[1:]
    return new
print("\n[1/12] MODEL / TEST585 REUSE")
old=globals().get("model",None)
reuse=(isinstance(old,torch.nn.Module) and type(old).__module__=="transformers.models.deepseek_v2.modeling_deepseek_v2" and len(getattr(getattr(old,"model",None),"layers",[]))==NL and all(p.device.type=="cuda" and (not p.is_floating_point() or p.dtype==DTYPE) for p in old.parameters()))
if reuse:
    model=old;print("MODEL_REUSED: YES")
else:
    if isinstance(old,torch.nn.Module):
        globals().pop("model",None);del old;gc.collect();torch.cuda.empty_cache()
    print("MODEL_REUSED: NO — LOADING PINNED DEEPSEEK BASELINE")
    cfg=AutoConfig.from_pretrained(MODEL,revision=REV,trust_remote_code=False);cfg._attn_implementation="eager"
    model,li=AutoModelForCausalLM.from_pretrained(MODEL,config=cfg,revision=REV,trust_remote_code=False,device_map={"":0},low_cpu_mem_usage=True,output_loading_info=True,attn_implementation="eager",dtype=DTYPE)
    for k in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs"):
        assert not li.get(k,[]),f"LOAD ERROR {k}: {str(li.get(k))[:200]}"
model.eval()
for p in model.parameters():p.requires_grad_(False)
base=model.model;LAYERS=base.layers;DEVICE=model.get_input_embeddings().weight.device
assert len(LAYERS)==NL and base.config.hidden_size==H
tok=globals().get("tok",None)
if tok is None or getattr(tok,"name_or_path",None)!=MODEL:tok=AutoTokenizer.from_pretrained(MODEL,revision=REV,trust_remote_code=False)
def ids(s):return tok(s,add_special_tokens=False,return_tensors="pt").input_ids.to(DEVICE)
def run(input_ids=None,inputs_embeds=None,past=None,attention_mask=None,position_ids=None,cache_position=None):
    with torch.inference_mode():
        return model(input_ids=input_ids,inputs_embeds=inputs_embeds,past_key_values=past,attention_mask=attention_mask,position_ids=position_ids,cache_position=cache_position,use_cache=True,return_dict=True,logits_to_keep=1)
def layers_of(c):
    assert hasattr(c,"layers") and len(c.layers)==NL
    out=[]
    for i,l in enumerate(c.layers):
        k=getattr(l,"keys",None);r=getattr(l,"values",None)
        assert torch.is_tensor(k) and torch.is_tensor(r) and k.ndim==4 and r.ndim==4
        assert k.shape[:3]==r.shape[:3] and k.shape[-1]==512 and r.shape[-1]==64,f"MLA CACHE L{i}"
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
        assert tuple(k.shape)==(1,1,n,512),f"LATENT L{i}: {tuple(k.shape)}"
        assert tuple(r.shape)==(1,1,n,64),f"ROTARY L{i}: {tuple(r.shape)}"
        assert torch.isfinite(k).all() and torch.isfinite(r).all()
    return True
def fingerprint():
    h=hashlib.sha256()
    for i in (0,3,6,12,20,26):
        p=next(LAYERS[i].parameters()).detach().reshape(-1);n=p.numel()
        for j in range(8):
            a=j*(n-128)//7;h.update(p[a:a+128].float().cpu().numpy().tobytes())
    return h.hexdigest()
FP0=fingerprint()
gate("MODEL_FROZEN",all(not p.requires_grad for p in model.parameters()))
print("PARAMETERS:",sum(p.numel() for p in model.parameters()),"TRAINABLE:",sum(p.numel() for p in model.parameters() if p.requires_grad))
print("\n[2/12] TEST585 PANEL LOCK")
FACTS=[("VELORA-731","QN-4826"),("MIREX-204","KT-9173")]
SYSTEM="Memorize these fictional records and answer questions using only the recorded facts. Answer with the requested code only."
PREFIX=tok.apply_chat_template([{"role":"user","content":SYSTEM+"\nRECORDS:\n"}],tokenize=False,add_generation_prompt=False)
PREFIX_IDS=ids(PREFIX)[0].tolist();P=len(PREFIX_IDS);CAR=[]
for station,code in FACTS:
    fact=f"The access code for the fictional station {station} is {code}."
    all_ids=ids(PREFIX+fact+"\n")[0].tolist()
    assert all_ids[:P]==PREFIX_IDS,"CANONICAL PREFIX MISMATCH"
    CAR.append({"station":station,"code":code,"fact":fact,"body":all_ids[P:],"full":all_ids})
def question(station):
    return tok.apply_chat_template([{"role":"user","content":f"What is the access code for the fictional station {station}? Answer with the code only."}],tokenize=False,add_generation_prompt=True)
gate("PANEL_LOCK",P==31 and [len(c["body"]) for c in CAR]==[24,24])
print("PREFIX_TOKENS:",P,"BODY_LENGTHS:",[len(c["body"]) for c in CAR],"CUT:",CUT)
print("\n[3/12] TEST585 MLA ROTARY TRANSPORT")
def rephase_rotary(r,oldpos,newpos):
    assert r.shape[-1]==64 and oldpos.numel()==newpos.numel()==r.shape[-2]
    if torch.equal(oldpos,newpos):return r.detach().clone()
    n=oldpos.numel();emb=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE)
    with torch.inference_mode():
        f0=base.rotary_emb(emb,oldpos.unsqueeze(0)).to(torch.complex64)
        f1=base.rotary_emb(emb,newpos.unsqueeze(0)).to(torch.complex64)
    phase=f1*torch.conj(f0)
    z=torch.view_as_complex(r.float().reshape(1,1,n,32,2).contiguous())
    return torch.view_as_real(z*phase.unsqueeze(1)).reshape_as(r.float()).to(r.dtype)
print("\n[4/12] WRITE — INDEPENDENT CHECKPOINTS")
def write_cartridge(ci):
    c=CAR[ci];x=torch.tensor([c["full"]],device=DEVICE,dtype=torch.long);n=x.shape[1];pos=torch.arange(n,device=DEVICE)
    state={};handles=[]
    def capture(module,args,out):
        z=out[0] if isinstance(out,(tuple,list)) else out
        state["h"]=z.detach().clone()
    try:
        handles.append(LAYERS[CUT].register_forward_hook(capture))
        o=run(input_ids=x,attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos)
        data=clone_layers(o.past_key_values)
    finally:
        for h in handles:h.remove()
    assert state["h"].shape==(1,n,H);validate(data,n)
    return {"h":state["h"],"early":data[:CUT+1],"pos":pos.detach().clone(),"body_len":n-P,"native":data}
CP=[write_cartridge(i) for i in range(2)]
gate("WRITE_INDEPENDENT",len(CP)==2 and all(len(cp["early"])==CUT+1 for cp in CP))
print("\n[5/12] INIT / APPEND — TEST585 ENGINE UNCHANGED")
def init_cartridge(ci):
    cp=CP[ci];n=cp["h"].shape[1];seen=[0];handles=[]
    def inject(module,args,out):
        z=out[0] if isinstance(out,(tuple,list)) else out
        assert z.shape==cp["h"].shape
        seen[0]+=1
        return replace_hidden(out,cp["h"])
    try:
        handles.append(LAYERS[CUT].register_forward_hook(inject))
        pos=torch.arange(n,device=DEVICE)
        o=run(inputs_embeds=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos)
        upper=clone_layers(o.past_key_values)
    finally:
        for h in handles:h.remove()
    result=tuple((k.clone(),r.clone()) for k,r in cp["early"])+upper[CUT+1:]
    validate(result,n);HOOK_COUNT["init"]+=seen[0]
    delta=max(float((a.float()-b.float()).abs().max().item()) for pa,pb in zip(result,cp["native"]) for a,b in zip(pa,pb))
    return result,seen[0],delta
def append_cartridge(memory,ci):
    cp=CP[ci];oldn=length(memory);q=cp["body_len"];newpos=torch.arange(oldn,oldn+q,device=DEVICE);oldpos=cp["pos"][P:]
    body_h=cp["h"][:,P:,:];assert body_h.shape==(1,q,H)
    cache=make_cache(memory);calls=[0];handles=[]
    def inject(module,args,out):
        z=out[0] if isinstance(out,(tuple,list)) else out
        assert z.shape==body_h.shape
        calls[0]+=1
        return replace_hidden(out,body_h)
    try:
        handles.append(LAYERS[CUT].register_forward_hook(inject))
        o=run(inputs_embeds=torch.zeros((1,q,H),device=DEVICE,dtype=DTYPE),past=cache,attention_mask=torch.ones((1,oldn+q),device=DEVICE,dtype=torch.long),position_ids=newpos.unsqueeze(0),cache_position=newpos)
        grown=list(clone_layers(o.past_key_values))
    finally:
        for h in handles:h.remove()
    for li in range(CUT+1):
        oldk,oldr=memory[li];newk,newr=cp["early"][li]
        body_k=newk[:,:,P:,:].detach().clone();body_r=rephase_rotary(newr[:,:,P:,:],oldpos,newpos)
        grown[li]=(torch.cat((oldk,body_k),dim=-2),torch.cat((oldr,body_r),dim=-2))
    result=tuple(grown);validate(result,oldn+q);HOOK_COUNT["append"]+=calls[0]
    unchanged=all(torch.equal(a,na[:,:,:oldn,:]) and torch.equal(b,nb[:,:,:oldn,:]) for (a,b),(na,nb) in zip(memory,result))
    return result,calls[0],unchanged
ORDERS=[(0,1),(1,0)];MEM={};META={}
for order in ORDERS:
    a,b=order;first,ic,delta=init_cartridge(a);app,ac,preserved=append_cartridge(first,b)
    MEM[order]={"first":first,"append":app}
    META[order]={"init_calls":ic,"init_delta":delta,"append_calls":ac,"preserved":preserved}
    print("ORDER",order,"INIT_DELTA",delta,"APPEND_CALLS",ac,"PRESERVED",preserved)
gate("INIT_EXACT_BOTH",all(v["init_calls"]==1 and v["init_delta"]==0 for v in META.values()))
gate("APPEND_PRESERVED_BOTH",all(v["append_calls"]==1 and v["preserved"] for v in META.values()))
print("\n[6/12] JOINT REFERENCES")
def joint_reference(order):
    seq=PREFIX_IDS
    for ci in order:seq=seq+CAR[ci]["body"]
    x=torch.tensor([seq],device=DEVICE,dtype=torch.long);n=x.shape[1];pos=torch.arange(n,device=DEVICE)
    o=run(input_ids=x,attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos)
    result=clone_layers(o.past_key_values);validate(result,n);return result
for order in ORDERS:MEM[order]["joint"]=joint_reference(order)
print("\n[7/12] READOUT FUNCTIONS — DIAGNOSTICS ONLY")
def code_contains(answer,code):
    return re.search(r"(?<![A-Za-z0-9-])"+re.escape(code)+r"(?![A-Za-z0-9-])",answer) is not None
def decode_greedy(memory,qtext,forced_first=None):
    cache=make_cache(memory);n=length(memory);qids=ids(qtext);ql=qids.shape[1]
    pos=torch.arange(n,n+ql,device=DEVICE)
    o=run(input_ids=qids,past=cache,attention_mask=torch.ones((1,n+ql),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos)
    cache=o.past_key_values;logits=o.logits[:,-1,:].float();probs=torch.softmax(logits,dim=-1)
    vals,idx=probs.topk(TOPK,dim=-1)
    top=[(tok.decode([int(t)]),round(float(v),6)) for t,v in zip(idx[0],vals[0])]
    nxt=o.logits[:,-1,:].argmax(-1,keepdim=True)
    if forced_first is not None:nxt=torch.tensor([[int(forced_first)]],device=DEVICE,dtype=torch.long)
    generated=[]
    for _ in range(MAX_NEW):
        t=int(nxt.item())
        if t==tok.eos_token_id:break
        generated.append(t);n=int(cache.get_seq_length());p=torch.tensor([n],device=DEVICE)
        o=run(input_ids=nxt,past=cache,attention_mask=torch.ones((1,n+1),device=DEVICE,dtype=torch.long),position_ids=p.unsqueeze(0),cache_position=p)
        cache=o.past_key_values;nxt=o.logits[:,-1,:].argmax(-1,keepdim=True)
    return tok.decode(generated,skip_special_tokens=True).strip(),top
def candidate_tokens(code):
    # First-token forcing is explicitly supervised; never count it as autonomous memory success.
    seq=ids(" "+code)[0].tolist()
    assert seq and tok.decode(seq).strip()==code
    return seq
def conditional_logprob(memory,qtext,candidate):
    seq=candidate_tokens(candidate);cache=make_cache(memory);n=length(memory);qids=ids(qtext);ql=qids.shape[1]
    pos=torch.arange(n,n+ql,device=DEVICE)
    o=run(input_ids=qids,past=cache,attention_mask=torch.ones((1,n+ql),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos)
    cache=o.past_key_values;total=0.0;parts=[]
    for j,t in enumerate(seq):
        lp=torch.log_softmax(o.logits[:,-1,:].float(),dim=-1)[0,t].item();total+=lp;parts.append(round(lp,5))
        if j<len(seq)-1:
            n=int(cache.get_seq_length());p=torch.tensor([n],device=DEVICE)
            o=run(input_ids=torch.tensor([[t]],device=DEVICE),past=cache,attention_mask=torch.ones((1,n+1),device=DEVICE,dtype=torch.long),position_ids=p.unsqueeze(0),cache_position=p)
            cache=o.past_key_values
    return total,parts,seq
print("CODE_TOKEN_IDS:",{code:candidate_tokens(code) for _,code in FACTS})
print("\n[8/12] GREEDY + FORCED FIRST TOKEN + LIKELIHOOD")
for order in ORDERS:
    for arm in ("joint","first","append"):
        for station,gold in FACTS:
            q=question(station);memory=MEM[order][arm]
            greedy,top=decode_greedy(memory,q)
            first_token=candidate_tokens(gold)[0]
            forced,forced_top=decode_greedy(memory,q,forced_first=first_token)
            alternatives={}
            for _,code in FACTS:
                lp,parts,seq=conditional_logprob(memory,q,code)
                alternatives[code]={"logprob":lp,"per_token":parts,"tokens":seq}
            other=next(code for _,code in FACTS if code!=gold)
            margin=alternatives[gold]["logprob"]-alternatives[other]["logprob"]
            exact=greedy.strip().strip(" .,:;\"'`")==gold
            contains=code_contains(greedy,gold)
            wrong=code_contains(greedy,other)
            forced_exact=forced.strip().strip(" .,:;\"'`")==gold
            row={"order":order,"arm":arm,"station":station,"gold":gold,"greedy":greedy,"exact":exact,"contains":contains,"wrong":wrong,"forced":forced,"forced_exact":forced_exact,"margin":margin}
            ROWS.append(row);LIKELIHOOD.append({"order":order,"arm":arm,"station":station,"gold":gold,"alternatives":alternatives,"margin":margin})
            print("ORDER",order,"ARM",arm,"STATION",station,"GOLD",gold)
            print(" GREEDY:",repr(greedy),"EXACT:",exact,"CONTAINS:",contains,"WRONG:",wrong)
            print(" FORCED_FIRST:",repr(forced),"FORCED_EXACT:",forced_exact,"DIAGNOSTIC_ONLY")
            print(" FIRST_TOKEN_TOP:",top[:5])
            print(" CODE_LOGPROB:",{k:round(v["logprob"],5) for k,v in alternatives.items()},"MARGIN:",round(margin,5))
print("\n[9/12] PAIRED SCORES")
def count(arm,key):return sum(int(r[key]) for r in ROWS if r["arm"]==arm)
for arm in ("joint","first","append"):
    rr=[r for r in ROWS if r["arm"]==arm]
    print("ARM",arm,"GREEDY_EXACT",count(arm,"exact"),"/4","GREEDY_CONTAINS",count(arm,"contains"),"/4","WRONG",count(arm,"wrong"),"/4","FORCED_EXACT",count(arm,"forced_exact"),"/4","POSITIVE_MARGIN",sum(r["margin"]>0 for r in rr),"/4")
gate("APPEND_GREEDY_EXACT_4",count("append","exact")==4)
gate("APPEND_CONTAINS_4",count("append","contains")==4)
gate("APPEND_NO_WRONG_CODES",count("append","wrong")==0)
gate("APPEND_POSITIVE_CODE_MARGIN_4",sum(r["margin"]>0 for r in ROWS if r["arm"]=="append")==4)
gate("APPEND_FORCED_EXACT_4_DIAGNOSTIC",count("append","forced_exact")==4)
print("\n[10/12] INTERPRETATION")
for r in ROWS:
    if r["arm"]!="append":continue
    if r["exact"]:classification="AUTONOMOUS_EXACT"
    elif r["contains"] and r["forced_exact"]:classification="FORMAT_COMPETITION_SUPPORTED"
    elif r["contains"]:classification="KNOWLEDGE_PRESENT_BUT_FORCED_READOUT_INCOMPLETE"
    else:classification="KNOWLEDGE_RECALL_FAILURE"
    print("ORDER",r["order"],"STATION",r["station"],"CLASS",classification,"MARGIN",round(r["margin"],5))
print("WARNING: Forced-first-token output is supervised diagnostic evidence, not independent successful recall.")
print("WARNING: Candidate log-probabilities are conditioned on supplied candidate tokens and are not a free-generation result.")
print("\n[11/12] FROZEN / HOOK / CACHE INTEGRITY")
gate("CACHE_VALID",all(validate(m,length(m)) for d in MEM.values() for m in d.values()))
gate("HOOK_COUNTS",HOOK_COUNT=={"init":2,"append":2})
gate("HOOKS_CLEAN",all(not l._forward_hooks and not l._forward_pre_hooks for l in LAYERS))
gate("FINGERPRINT_UNCHANGED",fingerprint()==FP0)
gate("WEIGHTS_FROZEN",all(not p.requires_grad for p in model.parameters()))
print("CUDA_ALLOCATED_GIB:",round(torch.cuda.memory_allocated()/2**30,3))
print("CUDA_PEAK_GIB:",round(torch.cuda.max_memory_allocated()/2**30,3))
print("\n[12/12] FINAL")
for k,v in GATES.items():print("GATE",k,"PASS" if v else "FAIL")
infra=["MODEL_FROZEN","PANEL_LOCK","WRITE_INDEPENDENT","INIT_EXACT_BOTH","APPEND_PRESERVED_BOTH","CACHE_VALID","HOOK_COUNTS","HOOKS_CLEAN","FINGERPRINT_UNCHANGED","WEIGHTS_FROZEN"]
infra_ok=all(GATES.get(k,False) for k in infra)
strict=GATES["APPEND_GREEDY_EXACT_4"] and GATES["APPEND_NO_WRONG_CODES"]
diagnostic=GATES["APPEND_CONTAINS_4"] and GATES["APPEND_POSITIVE_CODE_MARGIN_4"]
print("INFRASTRUCTURE_STATUS:","PASS" if infra_ok else "FAIL")
print("STRICT_BEHAVIOR_STATUS:","PASS" if strict else "FAIL")
print("READOUT_DIAGNOSTIC_STATUS:","PASS" if diagnostic else "FAIL")
print("STATUS:","PASS — STRICT TWO-CARTRIDGE RECALL" if infra_ok and strict else "DIAGNOSTIC COMPLETE — STRICT RECALL NOT YET VERIFIED")
print("ENGINE: TEST585 CUT3 WRITE/INIT/APPEND UNCHANGED")
print("SOURCE_REPLAY_DURING_APPEND_OR_ASK: NO")
print("FORCED_FIRST_TOKEN: DIAGNOSTIC ONLY")
print("COMPRESSION: NOT YET")
print("ELAPSED_SECONDS:",round(time.time()-T0,2))
print("="*152)

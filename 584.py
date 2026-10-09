
# TEST584 — AKBASCORE MAM · DEEPSEEK MLA INCREMENTAL APPEND PILOT
# TEST583 FROZEN BACKBONE + TEST560 INCREMENTAL APPEND LOGIC
# TWO INDEPENDENT NUMERICAL CARTRIDGES · CUT3 · NATIVE MLA K512/R64
# NO TRAINING · NO SOURCE REPLAY DURING INIT/APPEND/ASK · SINGLE COLAB CELL
import os,time,gc,hashlib,traceback,math
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch,transformers
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
MODEL="deepseek-ai/DeepSeek-V2-Lite-Chat";REV="85864749cd611b4353ce1decdb286193298f64c7";TEST="584";DTYPE=torch.bfloat16;SEED=577;CUT=3;NL=27;H=2048;MAX_NEW=24
torch.manual_seed(SEED);torch.set_grad_enabled(False)
T0=time.time();GATES={};RESULTS=[];HANDLES=[]
print("="*144)
print("TEST584 — AKBASCORE MAM · DEEPSEEK MLA INCREMENTAL APPEND PILOT")
print("="*144)
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(),"CUDA BF16 REQUIRED"
print("GPU:",torch.cuda.get_device_name(0),"TORCH:",torch.__version__,"TRANSFORMERS:",transformers.__version__)
def gate(name,condition):
    GATES[name]=bool(condition);print("GATE",name,"PASS" if condition else "FAIL")
def ids(s):return tok(s,add_special_tokens=False,return_tensors="pt").input_ids.to(DEVICE)
def layers_of(c):
    assert hasattr(c,"layers") and len(c.layers)==NL,"27-LAYER CACHE REQUIRED"
    out=[]
    for i,l in enumerate(c.layers):
        k=getattr(l,"keys",None);r=getattr(l,"values",None)
        assert torch.is_tensor(k) and torch.is_tensor(r),f"CACHE L{i} MISSING"
        assert k.ndim==4 and r.ndim==4 and k.shape[:3]==r.shape[:3],f"CACHE L{i} SHAPE"
        assert k.shape[-1]==512 and r.shape[-1]==64,f"CACHE L{i} MLA DIM"
        out.append((k,r))
    return out
def clone_layers(c):return tuple((k.detach().clone(),r.detach().clone()) for k,r in layers_of(c))
def make_cache(data):
    c=DynamicCache()
    for i,(k,r) in enumerate(data):c.update(k.detach().clone(),r.detach().clone(),i)
    return c
def length(data):return int(data[0][0].shape[-2])
def run(input_ids=None,inputs_embeds=None,past=None,attention_mask=None,position_ids=None,cache_position=None):
    with torch.inference_mode():
        return model(input_ids=input_ids,inputs_embeds=inputs_embeds,past_key_values=past,attention_mask=attention_mask,position_ids=position_ids,cache_position=cache_position,use_cache=True,return_dict=True,logits_to_keep=1)
def replace_hidden(out,new):
    if isinstance(out,tuple):return (new,)+out[1:]
    if isinstance(out,list):return [new]+out[1:]
    return new
def exact(a,b):return a.strip().strip(" .,:;\"'`")==b
def ask(memory,question):
    c=make_cache(memory);n=length(memory);qids=ids(question);qlen=qids.shape[1]
    pos=torch.arange(n,n+qlen,device=DEVICE);att=torch.ones((1,n+qlen),device=DEVICE,dtype=torch.long)
    o=run(input_ids=qids,past=c,attention_mask=att,position_ids=pos.unsqueeze(0),cache_position=pos)
    c=o.past_key_values;nxt=o.logits[:,-1,:].argmax(-1,keepdim=True);out=[]
    for _ in range(MAX_NEW):
        t=int(nxt.item())
        if t==tok.eos_token_id:break
        out.append(t);n=int(c.get_seq_length());p=torch.tensor([n],device=DEVICE)
        o=run(input_ids=nxt,past=c,attention_mask=torch.ones((1,n+1),device=DEVICE,dtype=torch.long),position_ids=p.unsqueeze(0),cache_position=p)
        c=o.past_key_values;nxt=o.logits[:,-1,:].argmax(-1,keepdim=True)
    return tok.decode(out,skip_special_tokens=True).strip()
print("\n[1/12] MODEL / TEST583 REUSE")
old=globals().get("model",None)
reuse=(isinstance(old,torch.nn.Module) and type(old).__module__=="transformers.models.deepseek_v2.modeling_deepseek_v2" and len(getattr(getattr(old,"model",None),"layers",[]))==NL and all(p.device.type=="cuda" and (not p.is_floating_point() or p.dtype==DTYPE) for p in old.parameters()))
if reuse:
    model=old;print("MODEL_REUSED: YES")
else:
    if isinstance(old,torch.nn.Module):
        globals().pop("model",None);del old;gc.collect();torch.cuda.empty_cache()
    print("MODEL_REUSED: NO — LOADING PINNED BASELINE")
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
gate("MODEL_FROZEN",all(not p.requires_grad for p in model.parameters()))
print("PARAMETERS:",sum(p.numel() for p in model.parameters()),"TRAINABLE:",sum(p.numel() for p in model.parameters() if p.requires_grad))
print("\n[2/12] CANONICAL PREFIX / INDEPENDENT CARTRIDGES")
# Same canonical prefix is stored once; each independent cartridge carries only its body on APPEND.
FACTS=[("VELORA-731","QN-4826"),("MIREX-204","KT-9173")]
SYSTEM="Memorize these fictional records and answer questions using only the recorded facts. Answer with the requested code only."
PREFIX=tok.apply_chat_template([{"role":"user","content":SYSTEM+"\nRECORDS:\n"}],tokenize=False,add_generation_prompt=False)
# Remove terminal assistant-start markers if tokenizer adds them; source is a shared user-message prefix.
PREFIX_IDS=ids(PREFIX)[0].tolist();P=len(PREFIX_IDS)
CAR=[]
for station,code in FACTS:
    fact=f"The access code for the fictional station {station} is {code}."
    text=PREFIX+fact+"\n"
    all_ids=ids(text)[0].tolist()
    assert all_ids[:P]==PREFIX_IDS,"CANONICAL TOKEN PREFIX MISMATCH"
    body=all_ids[P:]
    assert len(body)>0
    CAR.append({"station":station,"code":code,"fact":fact,"body":body,"full":all_ids})
print("PREFIX_TOKENS:",P,"BODY_LENGTHS:",[len(c["body"]) for c in CAR])
def question(station):
    return tok.apply_chat_template([{"role":"user","content":f"What is the access code for the fictional station {station}? Answer with the code only."}],tokenize=False,add_generation_prompt=True)
print("\n[3/12] ROTARY MLA POSITION TRANSPORT")
def rephase_rotary(r,oldpos,newpos):
    assert r.shape[-1]==64 and oldpos.numel()==newpos.numel()==r.shape[-2]
    if torch.equal(oldpos,newpos):return r.detach().clone()
    n=oldpos.numel();emb=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE)
    with torch.inference_mode():
        f0=base.rotary_emb(emb,oldpos.unsqueeze(0)).to(torch.complex64)
        f1=base.rotary_emb(emb,newpos.unsqueeze(0)).to(torch.complex64)
    phase=f1*torch.conj(f0)
    z=torch.view_as_complex(r.float().reshape(1,1,n,32,2).contiguous())
    moved=z*phase.unsqueeze(1)
    return torch.view_as_real(moved).reshape_as(r.float()).to(r.dtype)
def validate_memory(memory,n):
    assert len(memory)==NL
    for i,(k,r) in enumerate(memory):
        assert tuple(k.shape)==(1,1,n,512),f"LATENT SHAPE L{i}: {tuple(k.shape)}"
        assert tuple(r.shape)==(1,1,n,64),f"ROTARY SHAPE L{i}: {tuple(r.shape)}"
        assert torch.isfinite(k).all() and torch.isfinite(r).all()
    return True
print("\n[4/12] WRITE — INDEPENDENT NUMERICAL CHECKPOINTS")
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
    assert "h" in state and state["h"].shape==(1,n,H)
    validate_memory(data,n)
    return {"h":state["h"],"early":data[:CUT+1],"pos":pos.detach().clone(),"body_len":n-P,"native":data}
CP=[write_cartridge(i) for i in range(len(CAR))]
gate("WRITE_INDEPENDENT",len(CP)==2 and all(len(cp["early"])==CUT+1 for cp in CP))
print("WRITE_LENGTHS:",[len(c["full"]) for c in CAR],"CUT:",CUT,"CHECKPOINTS:",len(CP))
print("\n[5/12] INIT — CUT3 RESIDUAL REPLAY")
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
    validate_memory(result,n)
    return result,seen[0]
memory,init_calls=init_cartridge(0)
native_first=CP[0]["native"]
first_delta=max(float((a.float()-b.float()).abs().max().item()) for pair_a,pair_b in zip(memory,native_first) for a,b in zip(pair_a,pair_b))
print("INIT_HOOK_CALLS:",init_calls,"INIT_VS_NATIVE_MAX:",first_delta)
gate("INIT_HOOK",init_calls==1)
gate("INIT_CACHE_EXACT",first_delta==0.0)
print("\n[6/12] INCREMENTAL APPEND — SECOND CARTRIDGE")
def append_cartridge(memory,ci):
    cp=CP[ci];oldn=length(memory);q=cp["body_len"];newpos=torch.arange(oldn,oldn+q,device=DEVICE);oldpos=cp["pos"][P:]
    body_h=cp["h"][:,P:,:]
    assert body_h.shape==(1,q,H)
    cache=make_cache(memory);calls=[0];handles=[]
    def inject(module,args,out):
        z=out[0] if isinstance(out,(tuple,list)) else out
        assert z.shape==body_h.shape,f"APPEND HIDDEN SHAPE {z.shape} != {body_h.shape}"
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
        body_k=newk[:,:,P:,:].detach().clone()
        body_r=rephase_rotary(newr[:,:,P:,:],oldpos,newpos)
        grown[li]=(torch.cat((oldk,body_k),dim=-2),torch.cat((oldr,body_r),dim=-2))
    result=tuple(grown)
    validate_memory(result,oldn+q)
    unchanged=all(torch.equal(a,na[:,:,:oldn,:]) and torch.equal(b,nb[:,:,:oldn,:]) for (a,b),(na,nb) in zip(memory,result))
    return result,calls[0],unchanged
appended,append_calls,old_preserved=append_cartridge(memory,1)
print("APPEND_HOOK_CALLS:",append_calls,"OLD_CACHE_BITWISE_PRESERVED:",old_preserved,"OLD_LENGTH:",length(memory),"NEW_LENGTH:",length(appended))
gate("APPEND_HOOK",append_calls==1)
gate("APPEND_OLD_CACHE_PRESERVED",old_preserved)
gate("APPEND_LENGTH",length(appended)==P+sum(len(c["body"]) for c in CAR))
print("\n[7/12] NATIVE JOINT REFERENCE")
joint_ids=PREFIX_IDS+CAR[0]["body"]+CAR[1]["body"]
joint_x=torch.tensor([joint_ids],device=DEVICE,dtype=torch.long);jn=joint_x.shape[1];jp=torch.arange(jn,device=DEVICE)
joint_out=run(input_ids=joint_x,attention_mask=torch.ones((1,jn),device=DEVICE,dtype=torch.long),position_ids=jp.unsqueeze(0),cache_position=jp)
joint=clone_layers(joint_out.past_key_values);del joint_out
validate_memory(joint,jn)
joint_delta=max(float((a.float()-b.float()).abs().max().item()) for pair_a,pair_b in zip(appended,joint) for a,b in zip(pair_a,pair_b))
print("JOINT_LENGTH:",jn,"APPEND_LENGTH:",length(appended),"CACHE_MAX_DIFFERENCE:",joint_delta)
print("\n[8/12] NATIVE INDEPENDENT MLA CONCAT CONTROL")
def native_independent_concat():
    first=CP[0]["native"];second=CP[1]["native"];oldn=length(first);q=CP[1]["body_len"]
    op=CP[1]["pos"][P:];np=torch.arange(oldn,oldn+q,device=DEVICE);out=[]
    for (ak,ar),(bk,br) in zip(first,second):
        out.append((torch.cat((ak,bk[:,:,P:,:]),-2),torch.cat((ar,rephase_rotary(br[:,:,P:,:],op,np)),-2)))
    result=tuple(out);validate_memory(result,oldn+q);return result
native_indep=native_independent_concat()
print("NATIVE_INDEPENDENT_LENGTH:",length(native_indep))
print("\n[9/12] SOURCE-FREE ASK — BOTH CARTRIDGES")
ARMS={"JOINT":joint,"NATIVE_INDEP":native_indep,"INIT_FIRST":memory,"APPEND_CUT3":appended}
for name,mem in ARMS.items():
    for station,code in FACTS:
        answer=ask(mem,question(station))
        ok=exact(answer,code)
        RESULTS.append({"arm":name,"station":station,"gold":code,"answer":answer,"exact":ok})
        print("ARM",name,"STATION",station,"GOLD",code,"ANSWER",repr(answer),"EXACT",ok)
def score(arm):return sum(int(r["exact"]) for r in RESULTS if r["arm"]==arm)
print("\n[10/12] SCORE / CONTROL COMPARISON")
for arm in ARMS:print("SCORE",arm,score(arm),"/",len(FACTS))
gate("JOINT_BOTH_EXACT",score("JOINT")==2)
gate("APPEND_BOTH_EXACT",score("APPEND_CUT3")==2)
gate("APPEND_BEATS_SINGLE",score("APPEND_CUT3")>score("INIT_FIRST"))
print("\n[11/12] INTEGRITY / MEMORY AUDIT")
gate("CACHE_SHAPES",all(validate_memory(m,length(m)) for m in ARMS.values()))
gate("HOOKS_CLEAN",all(not l._forward_hooks and not l._forward_pre_hooks for l in LAYERS))
gate("WEIGHTS_FROZEN",all(not p.requires_grad for p in model.parameters()))
print("CUDA_ALLOCATED_GIB:",round(torch.cuda.memory_allocated()/2**30,3))
print("CUDA_PEAK_GIB:",round(torch.cuda.max_memory_allocated()/2**30,3))
print("\n[12/12] FINAL")
for name,value in GATES.items():print("GATE",name,"PASS" if value else "FAIL")
infra=["MODEL_FROZEN","WRITE_INDEPENDENT","INIT_HOOK","INIT_CACHE_EXACT","APPEND_HOOK","APPEND_OLD_CACHE_PRESERVED","APPEND_LENGTH","CACHE_SHAPES","HOOKS_CLEAN","WEIGHTS_FROZEN"]
infra_ok=all(GATES.get(k,False) for k in infra)
behavior_ok=GATES["JOINT_BOTH_EXACT"] and GATES["APPEND_BOTH_EXACT"] and GATES["APPEND_BEATS_SINGLE"]
print("INFRASTRUCTURE_STATUS:","PASS" if infra_ok else "FAIL")
print("BEHAVIOR_STATUS:","PASS" if behavior_ok else "FAIL")
print("STATUS:","PASS — TWO-CARTRIDGE MLA INCREMENTAL APPEND PILOT" if infra_ok and behavior_ok else "PARTIAL / FAIL — INSPECT RESULTS")
print("CARTRIDGE: CUT3 FULL RESIDUAL + L0-L3 MLA LATENT/ROTARY")
print("SOURCE_REPLAY_DURING_APPEND: NO")
print("COMPRESSION: NOT YET")
print("ELAPSED_SECONDS:",round(time.time()-T0,2))
print("="*144)

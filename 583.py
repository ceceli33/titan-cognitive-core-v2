
# TEST583 - AKBASCORE MAM - DEEPSEEK CUT GENERALIZATION + MLA ROTARY POSITION TRANSPORT
# BASELINE TEST582 - FROZEN BF16 - EAGER - 27L - MLA K512/R64 - SINGLE COLAB CELL
import os,time,gc,hashlib,traceback,math
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch,transformers
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
MODEL="deepseek-ai/DeepSeek-V2-Lite-Chat";REV="85864749cd611b4353ce1decdb286193298f64c7";TEST="583";DTYPE=torch.bfloat16;SEED=577
torch.manual_seed(SEED);torch.set_grad_enabled(False)
print("="*140)
print("TEST583 - AKBASCORE MAM - DEEPSEEK CUT GENERALIZATION + MLA ROTARY POSITION TRANSPORT")
print("="*140)
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(),"CUDA BF16 REQUIRED"
print("GPU:",torch.cuda.get_device_name(0),"TORCH:",torch.__version__,"TRANSFORMERS:",transformers.__version__)
t0=time.time();gates={}
def cache_len(c):
    try:return int(c.get_seq_length())
    except Exception:return None
def clone_cache(c):
    z=DynamicCache()
    assert hasattr(c,"layers") and len(c.layers)==27,"CACHE MUST HAVE 27 LAYERS"
    for i,l in enumerate(c.layers):
        k=getattr(l,"keys",None);v=getattr(l,"values",None)
        assert torch.is_tensor(k) and torch.is_tensor(v),f"CACHE L{i} INVALID"
        assert k.ndim==4 and v.ndim==4 and k.shape[:3]==v.shape[:3],f"CACHE L{i} SHAPE ERROR"
        assert k.shape[-1]==512 and v.shape[-1]==64,f"CACHE L{i} MLA DIM ERROR"
        z.update(k.detach().clone(),v.detach().clone(),i)
    return z
def token_ids(s):return tok(s,add_special_tokens=False,return_tensors="pt").input_ids.to(device)
def run(ids,past=None):
    with torch.inference_mode():
        return model(input_ids=ids,past_key_values=past,use_cache=True,return_dict=True,logits_to_keep=1)
def decode_cache(c,ask,max_new=24):
    ids=token_ids(ask);cache=clone_cache(c);out_tokens=[]
    for step in range(max_new):
        o=run(ids if step==0 else nxt,past=cache)
        cache=o.past_key_values;nxt=o.logits[:,-1,:].argmax(-1,keepdim=True)
        tid=int(nxt.item());del o
        if tid==tok.eos_token_id:break
        out_tokens.append(tid)
    return tok.decode(out_tokens,skip_special_tokens=True).strip()
def set_hook_hidden(args,kwargs,value):
    if "hidden_states" in kwargs:
        kw=dict(kwargs);kw["hidden_states"]=value
        return args,kw
    assert len(args)>0,"NO HIDDEN STATES IN HOOK"
    ar=list(args);ar[0]=value
    return tuple(ar),kwargs
def capture_write(ids,cuts):
    captured={};hs=[]
    def mk(i):
        def hook(m,args,kwargs):
            x=kwargs.get("hidden_states",args[0] if args else None)
            if torch.is_tensor(x):captured[i]=x.detach().clone()
        return hook
    try:
        for i in cuts:hs.append(layers[i].register_forward_pre_hook(mk(i),with_kwargs=True))
        o=run(ids)
        native=clone_cache(o.past_key_values);del o
    finally:
        for h in hs:h.remove()
    assert len(captured)==len(cuts)
    return native,captured
def rebuild_cut(src,cut,ids):
    calls=[0];dummy=torch.full_like(ids,tok.eos_token_id)
    def inject(m,args,kwargs):
        x=kwargs.get("hidden_states",args[0] if args else None)
        assert torch.is_tensor(x) and x.shape==src.shape
        calls[0]+=1
        return set_hook_hidden(args,kwargs,src)
    h=layers[cut].register_forward_pre_hook(inject,with_kwargs=True)
    try:
        o=run(dummy);replay=clone_cache(o.past_key_values);del o
    finally:h.remove()
    return replay,calls[0]
def zero_lower(c,cut):
    z=clone_cache(c)
    for i in range(cut):
        z.layers[i].keys.zero_();z.layers[i].values.zero_()
    return z
def hybrid_cache(native,replay,cut):
    z=clone_cache(native)
    for i in range(cut,27):
        z.layers[i].keys=replay.layers[i].keys.detach().clone()
        z.layers[i].values=replay.layers[i].values.detach().clone()
    return z
def upper_error(a,b,cut):
    mx=0.
    for i in range(cut,27):
        for key in ("keys","values"):
            x=getattr(a.layers[i],key).float();y=getattr(b.layers[i],key).float()
            assert x.shape==y.shape
            mx=max(mx,float((x-y).abs().max().item()))
    return mx
def exact(answer,code):return answer.strip().strip(" .,:;\"'`")==code
def contains_code(answer,code):return code in answer
print("\n[1/11] MODEL / TEST582 REUSE")
old=globals().get("model",None)
reuse=(isinstance(old,torch.nn.Module) and type(old).__module__=="transformers.models.deepseek_v2.modeling_deepseek_v2" and len(getattr(getattr(old,"model",None),"layers",[]))==27 and all(p.device.type=="cuda" and (not p.is_floating_point() or p.dtype==DTYPE) for p in old.parameters()))
if reuse:
    model=old;print("MODEL_REUSED: YES")
else:
    if isinstance(old,torch.nn.Module):
        globals().pop("model",None);del old;gc.collect();torch.cuda.empty_cache()
    print("MODEL_REUSED: NO - LOADING PINNED DEEPSEEK BASELINE")
    cfg=AutoConfig.from_pretrained(MODEL,revision=REV,trust_remote_code=False);cfg._attn_implementation="eager"
    model,li=AutoModelForCausalLM.from_pretrained(MODEL,config=cfg,revision=REV,trust_remote_code=False,device_map={"":0},low_cpu_mem_usage=True,output_loading_info=True,attn_implementation="eager",dtype=DTYPE)
    for k in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs"):
        assert not li.get(k,[]),f"LOAD ERROR {k}: {str(li.get(k))[:200]}"
model.eval()
for p in model.parameters():p.requires_grad_(False)
base=model.model;layers=base.layers;device=model.get_input_embeddings().weight.device
assert len(layers)==27 and all(p.device.type=="cuda" and p.dtype==DTYPE for p in model.parameters() if p.is_floating_point())
tok=globals().get("tok",None)
if tok is None or getattr(tok,"name_or_path",None)!=MODEL:
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REV,trust_remote_code=False)
gates["MODEL_FROZEN"]=all(not p.requires_grad for p in model.parameters())
print("PARAMETERS:",sum(p.numel() for p in model.parameters()),"TRAINABLE:",sum(p.numel() for p in model.parameters() if p.requires_grad))
print("\n[2/11] FIXED GENERALIZATION PANEL")
PANEL=[
    ("VELORA-731","QN-4826"),
    ("MIREX-204","KT-9173"),
    ("SOLARA-618","PX-3508"),
    ("NOREL-452","DV-6291")
]
CUTS=[1,3,6];rows=[];write_lengths=[]
print("PANEL_SIZE:",len(PANEL),"CUTS:",CUTS)
print("\n[3/11] WRITE / CUT REPLAY / INDEPENDENT INIT")
for case,(station,code) in enumerate(PANEL,1):
    fact=f"The access code for the fictional station {station} is {code}."
    question=f"What is the access code for the fictional station {station}? Answer with the code only."
    source=tok.apply_chat_template([{"role":"user","content":"Memorize this fictional record: "+fact}],tokenize=False,add_generation_prompt=True)
    ask=tok.apply_chat_template([{"role":"user","content":question}],tokenize=False,add_generation_prompt=True)
    ids=token_ids(source);write_lengths.append(ids.shape[1])
    assert code not in ask
    native,captured=capture_write(ids,CUTS)
    native_answer=decode_cache(native,ask)
    print("CASE",case,"STATION",station,"CODE",code,"SOURCE_TOKENS",ids.shape[1],"NATIVE",repr(native_answer))
    for cut in CUTS:
        replay,calls=rebuild_cut(captured[cut],cut,ids)
        err=upper_error(replay,native,cut)
        hybrid=hybrid_cache(native,replay,cut)
        independent=zero_lower(replay,cut)
        hybrid_answer=decode_cache(hybrid,ask)
        independent_answer=decode_cache(independent,ask)
        row={"case":case,"station":station,"code":code,"cut":cut,"calls":calls,"upper_max":err,"native_exact":exact(native_answer,code),"hybrid_exact":exact(hybrid_answer,code),"independent_exact":exact(independent_answer,code),"independent_contains":contains_code(independent_answer,code),"independent_answer":independent_answer}
        rows.append(row)
        print(" CUT",cut,"HOOK",calls,"UPPER_MAX",round(err,8),"HYBRID",repr(hybrid_answer),"INDEPENDENT",repr(independent_answer),"EXACT",row["independent_exact"],"CONTAINS",row["independent_contains"])
        del replay,hybrid,independent
    del native,captured
gates["CUT_HOOKS"]=all(r["calls"]==1 for r in rows)
gates["UPPER_CACHE_EXACT"]=all(r["upper_max"]==0.0 for r in rows)
gates["HYBRID_EXACT"]=all(r["hybrid_exact"] for r in rows)
print("\n[4/11] GENERALIZATION SUMMARY")
for cut in CUTS:
    rr=[r for r in rows if r["cut"]==cut]
    print("CUT",cut,"NATIVE",sum(r["native_exact"] for r in rr),"/",len(rr),"HYBRID",sum(r["hybrid_exact"] for r in rr),"/",len(rr),"INDEPENDENT_EXACT",sum(r["independent_exact"] for r in rr),"/",len(rr),"INDEPENDENT_CONTAINS",sum(r["independent_contains"] for r in rr),"/",len(rr))
gates["CUT3_GENERALIZATION"]=all(r["independent_exact"] for r in rows if r["cut"]==3)
print("\n[5/11] WRONG CARTRIDGE / NO MEMORY CONTROLS")
station,code=PANEL[0];other_station,other_code=PANEL[1]
question=f"What is the access code for the fictional station {station}? Answer with the code only."
ask=tok.apply_chat_template([{"role":"user","content":question}],tokenize=False,add_generation_prompt=True)
wrong_fact=f"The access code for the fictional station {other_station} is {other_code}."
wrong_source=tok.apply_chat_template([{"role":"user","content":"Memorize this fictional record: "+wrong_fact}],tokenize=False,add_generation_prompt=True)
wrong_ids=token_ids(wrong_source)
wrong_native,wrong_captured=capture_write(wrong_ids,[3])
wrong_replay,_=rebuild_cut(wrong_captured[3],3,wrong_ids)
wrong_independent=zero_lower(wrong_replay,3)
wrong_answer=decode_cache(wrong_independent,ask)
with torch.inference_mode():
    gen=model.generate(input_ids=token_ids(ask),max_new_tokens=24,do_sample=False,pad_token_id=tok.eos_token_id)
no_answer=tok.decode(gen[0,token_ids(ask).shape[1]:],skip_special_tokens=True).strip()
del gen,wrong_native,wrong_captured,wrong_replay,wrong_independent
print("WRONG_CARTRIDGE:",repr(wrong_answer),"TARGET:",code)
print("NO_MEMORY:",repr(no_answer),"TARGET:",code)
gates["WRONG_CARTRIDGE_NEGATIVE"]=not exact(wrong_answer,code)
gates["NO_MEMORY_NEGATIVE"]=not exact(no_answer,code)
print("\n[6/11] MLA ROTARY REFERENCE - COMPLEX PHASE")
# Native DeepSeek-V2 rotary_emb returns complex frequency factors.
n=max(write_lengths);shift=7
emb=torch.zeros((1,n,base.config.hidden_size),device=device,dtype=DTYPE)
p0=torch.arange(n,device=device).unsqueeze(0)
p1=p0+shift
with torch.inference_mode():
    f0=base.rotary_emb(emb,p0)
    f1=base.rotary_emb(emb,p1)
assert torch.is_complex(f0) and torch.is_complex(f1),"EXPECTED COMPLEX ROTARY FACTORS"
assert f0.shape==f1.shape and f0.shape[-1]==32
phase=(f1*torch.conj(f0)).to(torch.complex64)
inv=(phase*torch.conj(phase)).real
print("ROTARY_SHAPE:",tuple(f0.shape),"SHIFT:",shift,"PHASE_MODULUS_MAX_ERROR:",float((phase.abs()-1).abs().max().item()))
gates["ROTARY_PHASE_UNIT"]=float((phase.abs()-1).abs().max().item())<1e-5
print("\n[7/11] ROTARY K TRANSPORT / INVERSE")
# MLA cache values are rotary K, NOT classical V. Transport is complex phase multiplication.
fact=f"The access code for the fictional station {PANEL[0][0]} is {PANEL[0][1]}."
source=tok.apply_chat_template([{"role":"user","content":"Memorize this fictional record: "+fact}],tokenize=False,add_generation_prompt=True)
ids=token_ids(source)
native,_=capture_write(ids,[3]);length=ids.shape[1]
emb2=torch.zeros((1,length,base.config.hidden_size),device=device,dtype=DTYPE)
pos=torch.arange(length,device=device).unsqueeze(0)
with torch.inference_mode():
    src_freq=base.rotary_emb(emb2,pos).to(torch.complex64)
    dst_freq=base.rotary_emb(emb2,pos+shift).to(torch.complex64)
phase2=(dst_freq*torch.conj(src_freq))
transport_rows=[]
for layer in (0,3,6,12,20,26):
    k=native.layers[layer].keys;rot=native.layers[layer].values
    original=rot.float().reshape(1,1,length,32,2).contiguous()
    z=torch.view_as_complex(original)
    moved=z*phase2.unsqueeze(1)
    restored=moved*torch.conj(phase2.unsqueeze(1))
    moved_real=torch.view_as_real(moved).reshape_as(rot.float())
    restored_real=torch.view_as_real(restored).reshape_as(rot.float())
    inv_err=float((restored_real-rot.float()).abs().max().item())
    move_mag=float((moved_real-rot.float()).abs().max().item())
    latent_err=float((k-k).abs().max().item())
    transport_rows.append({"layer":layer,"inverse_error":inv_err,"movement":move_mag,"latent_error":latent_err})
    print("LAYER",layer,"ROTARY_SHIFT_MAX",round(move_mag,7),"INVERSE_MAX",round(inv_err,8),"LATENT_UNCHANGED",latent_err==0.0)
gates["ROTARY_INVERSE"]=all(r["inverse_error"]<1e-4 for r in transport_rows)
gates["ROTARY_NONTRIVIAL"]=any(r["movement"]>1e-4 for r in transport_rows)
print("\n[8/11] TRANSPORT LIMITATION AUDIT")
print("TRANSPORT_KIND: NATIVE COMPLEX ROTARY PHASE / POSITION SHIFT")
print("ROTARY_SOURCE_POSITION: 0..N-1")
print("ROTARY_TARGET_POSITION:",shift,"..",shift+length-1)
print("LATENT_512: NOT MODIFIED")
print("CACHE_POSITION / ATTENTION_MASK / APPEND: NOT YET REBUILT")
print("TRANSPORTED_CACHE_ASK: NOT CLAIMED")
print("\n[9/11] MODEL / HOOK AUDIT")
gates["HOOKS_CLEAN"]=all(len(l._forward_pre_hooks)==0 for l in layers)
gates["WEIGHTS_FROZEN"]=all(not p.requires_grad for p in model.parameters())
print("CUDA_ALLOCATED_GIB:",round(torch.cuda.memory_allocated()/2**30,3))
print("CUDA_PEAK_GIB:",round(torch.cuda.max_memory_allocated()/2**30,3))
print("\n[10/11] RESULTS")
for k,v in gates.items():print("GATE",k,"PASS" if v else "FAIL")
print("CUT3_EXACT:",sum(r["independent_exact"] for r in rows if r["cut"]==3),"/",len(PANEL))
print("CUT1_EXACT:",sum(r["independent_exact"] for r in rows if r["cut"]==1),"/",len(PANEL))
print("CUT6_EXACT:",sum(r["independent_exact"] for r in rows if r["cut"]==6),"/",len(PANEL))
print("\n[11/11] FINAL")
required=["MODEL_FROZEN","CUT_HOOKS","UPPER_CACHE_EXACT","HYBRID_EXACT","WRONG_CARTRIDGE_NEGATIVE","NO_MEMORY_NEGATIVE","ROTARY_PHASE_UNIT","ROTARY_INVERSE","ROTARY_NONTRIVIAL","HOOKS_CLEAN","WEIGHTS_FROZEN"]
status=all(gates[k] for k in required)
print("INFRASTRUCTURE_STATUS:","PASS" if status else "FAIL")
print("CUT3_GENERALIZATION_STATUS:","PASS" if gates["CUT3_GENERALIZATION"] else "NOT PROVEN")
print("STATUS:","PASS - GENERALIZATION + ROTARY DIAGNOSTICS" if status and gates["CUT3_GENERALIZATION"] else "PARTIAL / FAIL - INSPECT GATES")
print("INDEPENDENT_COMPRESSED_INIT: NOT PROVEN")
print("MULTI_CARTRIDGE_APPEND: NOT IMPLEMENTED")
print("ELAPSED_SECONDS:",round(time.time()-t0,2))
print("="*140)

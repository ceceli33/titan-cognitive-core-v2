
# TEST582 — AKBASCORE MAM · DEEPSEEK INDEPENDENT INIT / CUT ISOLATION
# BASELINE: TEST581 · FROZEN BF16 · EAGER · 27L · MLA K512/R64 · SINGLE COLAB CELL
import os,time,gc,hashlib,traceback
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch,transformers
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
MODEL="deepseek-ai/DeepSeek-V2-Lite-Chat";REV="85864749cd611b4353ce1decdb286193298f64c7";TEST="582";DTYPE=torch.bfloat16;SEED=577
torch.manual_seed(SEED);torch.set_grad_enabled(False)
print("="*136)
print("TEST582 — AKBASCORE MAM · DEEPSEEK INDEPENDENT INIT / CUT ISOLATION")
print("="*136)
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
def token_ids(s):
    return tok(s,add_special_tokens=False,return_tensors="pt").input_ids.to(device)
def run(ids,past=None):
    with torch.inference_mode():
        return model(input_ids=ids,past_key_values=past,use_cache=True,return_dict=True,logits_to_keep=1)
def decode_cache(c,ask,max_new=24):
    ids=token_ids(ask);cache=clone_cache(c);out_tokens=[];nxt=None
    for step in range(max_new):
        o=run(ids if step==0 else nxt,past=cache)
        cache=o.past_key_values
        nxt=o.logits[:,-1,:].argmax(-1,keepdim=True)
        tid=int(nxt.item())
        del o
        if tid==tok.eos_token_id:break
        out_tokens.append(tid)
    return tok.decode(out_tokens,skip_special_tokens=True).strip()
def exact(s):
    return s.strip().strip(" .,:;\"'`")=="QN-4826"
def cache_error(a,b,start=0):
    mx=0.;sm=0.;n=0
    assert len(a.layers)==len(b.layers)==27
    for i in range(start,27):
        for key in ("keys","values"):
            x=getattr(a.layers[i],key).float();y=getattr(b.layers[i],key).float()
            assert x.shape==y.shape,f"CACHE COMPARISON SHAPE ERROR L{i} {key}"
            d=(x-y).abs()
            mx=max(mx,float(d.max().item()));sm+=float(d.mean().item());n+=1
    return mx,sm/max(n,1)
def zero_cache(c,below):
    z=clone_cache(c)
    for i in range(below):
        z.layers[i].keys.zero_()
        z.layers[i].values.zero_()
    return z
def set_hook_hidden(args,kwargs,value):
    if "hidden_states" in kwargs:
        kw=dict(kwargs);kw["hidden_states"]=value
        return args,kw
    assert len(args)>0,"NO HIDDEN STATES IN HOOK"
    ar=list(args);ar[0]=value
    return tuple(ar),kwargs
print("\n[1/10] MODEL / TEST581 REUSE")
old=globals().get("model",None)
reuse=(
    isinstance(old,torch.nn.Module)
    and type(old).__module__=="transformers.models.deepseek_v2.modeling_deepseek_v2"
    and getattr(old.config,"_name_or_path",None)==MODEL
    and len(getattr(getattr(old,"model",None),"layers",[]))==27
    and all(p.device.type=="cuda" and (not p.is_floating_point() or p.dtype==DTYPE) for p in old.parameters())
)
if reuse:
    model=old
    print("MODEL_REUSED: YES")
else:
    if isinstance(old,torch.nn.Module):
        globals().pop("model",None)
        del old
        gc.collect()
        torch.cuda.empty_cache()
    print("MODEL_REUSED: NO — LOADING TEST581 BASELINE")
    cfg=AutoConfig.from_pretrained(MODEL,revision=REV,trust_remote_code=False)
    cfg._attn_implementation="eager"
    model,li=AutoModelForCausalLM.from_pretrained(
        MODEL,config=cfg,revision=REV,trust_remote_code=False,
        device_map={"":0},low_cpu_mem_usage=True,
        output_loading_info=True,attn_implementation="eager",dtype=DTYPE
    )
    for k in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs"):
        assert not li.get(k,[]),f"LOAD ERROR {k}: {str(li.get(k))[:200]}"
model.eval()
for p in model.parameters():p.requires_grad_(False)
base=model.model;layers=base.layers;device=model.get_input_embeddings().weight.device
assert len(layers)==27
assert all(p.device.type=="cuda" and p.dtype==DTYPE for p in model.parameters() if p.is_floating_point())
tok=globals().get("tok",None)
if tok is None or getattr(tok,"name_or_path",None)!=MODEL:
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REV,trust_remote_code=False)
gates["MODEL_FROZEN"]=all(not p.requires_grad for p in model.parameters())
print("PARAMETERS:",sum(p.numel() for p in model.parameters()),"TRAINABLE:",sum(p.numel() for p in model.parameters() if p.requires_grad))
print("\n[2/10] FIXED FACT / QUESTION")
FACT="The access code for the fictional station VELORA-731 is QN-4826."
QUESTION="What is the access code for the fictional station VELORA-731? Answer with the code only."
SOURCE=tok.apply_chat_template([{"role":"user","content":"Memorize this fictional record: "+FACT}],tokenize=False,add_generation_prompt=True)
ASK=tok.apply_chat_template([{"role":"user","content":QUESTION}],tokenize=False,add_generation_prompt=True)
write_ids=token_ids(SOURCE);ask_ids=token_ids(ASK)
gates["SOURCE_FREE_ASK"]="QN-4826" not in ASK and FACT not in ASK
print("SOURCE_TOKENS:",write_ids.shape[1],"ASK_TOKENS:",ask_ids.shape[1])
print("\n[3/10] WRITE — NATIVE CACHE / FULL CUT STATES")
captured={};hs=[]
def mk(i):
    def hook(m,args,kwargs):
        x=kwargs.get("hidden_states",args[0] if args else None)
        if torch.is_tensor(x):captured[i]=x.detach().clone()
    return hook
try:
    for i,l in enumerate(layers):
        hs.append(l.register_forward_pre_hook(mk(i),with_kwargs=True))
    o=run(write_ids)
finally:
    for h in hs:h.remove()
native=clone_cache(o.past_key_values);del o
assert len(captured)==27 and cache_len(native)==write_ids.shape[1]
gates["WRITE"]=True
print("WRITE_CACHE:",cache_len(native),"LAYERS:",len(native.layers),"CUT_STATES:",len(captured))
print("\n[4/10] REFERENCE — NATIVE CACHE ASK")
ref=decode_cache(native,ASK)
print("NATIVE_ANSWER:",repr(ref))
gates["NATIVE_EXACT"]=exact(ref)
print("\n[5/10] CUT REBUILD — SOURCE TOKENS ABSENT")
CUTS=[1,3,6,9,12,16,20];rows=[]
dummy=torch.full_like(write_ids,tok.eos_token_id)
for cut in CUTS:
    src=captured[cut];calls=[0]
    def inject(m,args,kwargs):
        x=kwargs.get("hidden_states",args[0] if args else None)
        assert torch.is_tensor(x) and x.shape==src.shape,f"CUT {cut} SHAPE ERROR"
        calls[0]+=1
        return set_hook_hidden(args,kwargs,src)
    h=layers[cut].register_forward_pre_hook(inject,with_kwargs=True)
    try:
        o=run(dummy)
        replay=clone_cache(o.past_key_values)
        del o
    finally:
        h.remove()
    upper_mx,upper_mean=cache_error(replay,native,start=cut)
    hybrid=clone_cache(native)
    for i in range(cut,27):
        hybrid.layers[i].keys=replay.layers[i].keys.detach().clone()
        hybrid.layers[i].values=replay.layers[i].values.detach().clone()
    hybrid_ans=decode_cache(hybrid,ASK)
    independent=zero_cache(replay,cut)
    independent_ans=decode_cache(independent,ASK)
    lower_zero_native=zero_cache(native,cut)
    lower_zero_ans=decode_cache(lower_zero_native,ASK)
    row={
        "cut":cut,"calls":calls[0],
        "upper_max":upper_mx,"upper_mean":upper_mean,
        "hybrid":hybrid_ans,"independent":independent_ans,
        "lower_zero_native":lower_zero_ans,
        "hybrid_exact":exact(hybrid_ans),
        "independent_exact":exact(independent_ans),
        "lower_zero_exact":exact(lower_zero_ans)
    }
    rows.append(row)
    print(
        "CUT",cut,"HOOK",calls[0],
        "UPPER_CACHE_MAX",round(upper_mx,8),
        "HYBRID",repr(hybrid_ans),
        "INDEPENDENT",repr(independent_ans),
        "LOWER_ZERO_NATIVE",repr(lower_zero_ans)
    )
    del replay,hybrid,independent,lower_zero_native
gates["CUT_REBUILD"]=all(r["calls"]==1 and r["upper_max"]==0.0 and r["hybrid_exact"] for r in rows)
print("\n[6/10] INDEPENDENT INIT SUMMARY")
for r in rows:
    print("CUT",r["cut"],"INDEPENDENT_EXACT",r["independent_exact"],"LOWER_ZERO_NATIVE_EXACT",r["lower_zero_exact"])
gates["INDEPENDENT_ANY"]=any(r["independent_exact"] for r in rows)
print("INDEPENDENT_PASS_COUNT:",sum(r["independent_exact"] for r in rows),"/",len(rows))
print("\n[7/10] NEGATIVE CONTROL — NO MEMORY")
with torch.inference_mode():
    gen=model.generate(input_ids=ask_ids,max_new_tokens=24,do_sample=False,pad_token_id=tok.eos_token_id)
no_answer=tok.decode(gen[0,ask_ids.shape[1]:],skip_special_tokens=True).strip()
del gen
print("NO_MEMORY:",repr(no_answer))
gates["NO_MEMORY_NEGATIVE"]=not exact(no_answer)
print("\n[8/10] CUT SELECTION — INDEPENDENT FIRST")
ranked=sorted(rows,key=lambda r:(not r["independent_exact"],not r["hybrid_exact"],r["cut"]))
for rank,r in enumerate(ranked,1):
    print("RANK",rank,"CUT",r["cut"],"INDEPENDENT",r["independent_exact"],"HYBRID",r["hybrid_exact"],"UPPER_MAX",round(r["upper_max"],8))
best=ranked[0]
print("BEST_DIAGNOSTIC_CUT:",best["cut"],"INDEPENDENT_EXACT:",best["independent_exact"])
print("\n[9/10] AUDIT")
gates["HOOKS_CLEAN"]=all(len(l._forward_pre_hooks)==0 for l in layers)
gates["WEIGHTS_FROZEN"]=all(not p.requires_grad for p in model.parameters())
gates["CACHE_INTEGRITY"]=cache_len(native)==write_ids.shape[1]
print("CUDA_ALLOCATED_GIB:",round(torch.cuda.memory_allocated()/2**30,3))
print("CUDA_PEAK_GIB:",round(torch.cuda.max_memory_allocated()/2**30,3))
print("\n[10/10] FINAL")
for k,v in gates.items():
    print("GATE",k,"PASS" if v else "FAIL")
print("CUT_REBUILD_COUNT:",sum(r["calls"]==1 and r["upper_max"]==0.0 and r["hybrid_exact"] for r in rows),"/",len(rows))
print("INDEPENDENT_EXACT_COUNT:",sum(r["independent_exact"] for r in rows),"/",len(rows))
print("STATUS:","PASS — INDEPENDENT CUT INIT PILOT" if all(gates.values()) else "PARTIAL / FAIL — INSPECT RESULTS")
print("CARTRIDGE: FULL CUT RESIDUAL + REPLAYED UPPER MLA CACHE")
print("COMPRESSION: NOT YET")
print("APPEND: NOT YET")
print("ELAPSED_SECONDS:",round(time.time()-t0,2))
print("="*136)

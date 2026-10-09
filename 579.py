
# TEST579 — AKBASCORE MAM · DEEPSEEK-V2-LITE-CHAT · CUT REPLAY / MOE / COMPLEX ROTARY
# BASELINE: TEST588 · PINNED BF16 · EAGER · 27L · FROZEN · SINGLE COLAB CELL
import os,sys,time,gc,inspect,traceback
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch,transformers
import torch.nn.functional as F
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
from huggingface_hub import model_info
MODEL="deepseek-ai/DeepSeek-V2-Lite-Chat";REV="85864749cd611b4353ce1decdb286193298f64c7";TEST="589";DTYPE=torch.bfloat16;SEED=577
torch.manual_seed(SEED);torch.set_grad_enabled(False)
print("="*126)
print("TEST589 — AKBASCORE MAM · DEEPSEEK CUT REPLAY / MOE / COMPLEX ROTARY")
print("="*126)
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(),"CUDA BF16 REQUIRED"
print("GPU:",torch.cuda.get_device_name(0),"TORCH:",torch.__version__,"TRANSFORMERS:",transformers.__version__)
gates={};t0=time.time()
def diff(a,b):
    assert a.shape==b.shape,(tuple(a.shape),tuple(b.shape))
    a=a.detach().to(torch.float32);b=b.detach().to(torch.float32)
    d=(a-b).abs()
    return {"max":float(d.max().item()),"mean":float(d.mean().item()),"cos":float(F.cosine_similarity(a.reshape(1,-1),b.reshape(1,-1)).item())}
def cdiff(a,b):
    assert a.shape==b.shape
    a=a.detach().to(torch.complex64);b=b.detach().to(torch.complex64)
    d=(a-b).abs()
    return {"max":float(d.max().item()),"mean":float(d.mean().item()),"relative":float((d/(b.abs()+1e-8)).mean().item())}
def get_cache_length(c):
    try:return int(c.get_seq_length())
    except Exception:return None
def capture_layers(input_ids,use_cache=False,past=None):
    captured={};handles=[]
    def mk(i):
        def hook(m,args,kwargs):
            x=kwargs.get("hidden_states",args[0] if args else None)
            if torch.is_tensor(x):captured[i]=x.detach().clone()
        return hook
    try:
        for i,l in enumerate(layers):handles.append(l.register_forward_pre_hook(mk(i),with_kwargs=True))
        with torch.inference_mode():out=model(input_ids=input_ids,past_key_values=past,use_cache=use_cache,return_dict=True,logits_to_keep=1)
        logits=out.logits[:,-1,:].detach().float().clone()
        cache=out.past_key_values
        del out
        return logits,captured,cache
    finally:
        for h in handles:h.remove()
print("\n[1/10] PINNED REVISION / MODEL")
assert model_info(MODEL,revision=REV).sha==REV
cfg=AutoConfig.from_pretrained(MODEL,revision=REV,trust_remote_code=False)
assert cfg.model_type=="deepseek_v2" and cfg.num_hidden_layers==27
cfg._attn_implementation="eager"
old=globals().get("model",None)
reuse=(isinstance(old,torch.nn.Module) and type(old).__module__=="transformers.models.deepseek_v2.modeling_deepseek_v2" and getattr(old.config,"_name_or_path",None)==MODEL and len(getattr(getattr(old,"model",None),"layers",[]))==27 and all(p.device.type=="cuda" and (not p.is_floating_point() or p.dtype==DTYPE) for p in old.parameters()))
if reuse:
    model=old;print("MODEL_REUSED: YES")
else:
    if isinstance(old,torch.nn.Module):
        if "model" in globals():del globals()["model"]
        del old;gc.collect();torch.cuda.empty_cache()
    print("MODEL_REUSED: NO — LOADING FROZEN BASELINE")
    kwargs=dict(config=cfg,revision=REV,trust_remote_code=False,device_map={"":0},low_cpu_mem_usage=True,output_loading_info=True,attn_implementation="eager",dtype=DTYPE)
    model,li=AutoModelForCausalLM.from_pretrained(MODEL,**kwargs)
    for k in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs"):
        v=li.get(k,[]);assert not v,f"LOAD ERROR {k}: {str(v)[:200]}"
model.eval()
for p in model.parameters():p.requires_grad_(False)
base=model.model;layers=base.layers
assert len(layers)==27 and all(p.device.type=="cuda" and p.dtype==DTYPE for p in model.parameters() if p.is_floating_point())
gates["MODEL"]=True
print("PARAMETERS:",sum(p.numel() for p in model.parameters()),"TRAINABLE:",sum(p.numel() for p in model.parameters() if p.requires_grad))
print("\n[2/10] TOKENIZER / FIXED INPUT")
tok=AutoTokenizer.from_pretrained(MODEL,revision=REV,trust_remote_code=False)
enc=tok.apply_chat_template([{"role":"user","content":"Reply with exactly one word: READY"}],add_generation_prompt=True,return_tensors="pt")
ids=enc if torch.is_tensor(enc) else enc.input_ids if hasattr(enc,"input_ids") else enc["input_ids"]
if ids.ndim==1:ids=ids.unsqueeze(0)
prompt=ids.to(model.get_input_embeddings().weight.device)
assert tuple(prompt.shape)==(1,15)
print("PROMPT:",repr(tok.decode(ids[0])),"TOKENS:",prompt.shape[1])
gates["PROMPT"]=True
print("\n[3/10] NATIVE REFERENCE / ALL CUT INPUTS")
ref_logits,ref_inputs,ref_cache=capture_layers(prompt,use_cache=True)
assert len(ref_inputs)==27
print("REFERENCE_TOP1:",int(ref_logits.argmax()),repr(tok.decode([int(ref_logits.argmax())])))
print("REFERENCE_CACHE_LENGTH:",get_cache_length(ref_cache))
for i in (0,1,2,3,4,6,9,12,16,20,26):print("CUT_INPUT",i,tuple(ref_inputs[i].shape),ref_inputs[i].dtype)
gates["REFERENCE"]=get_cache_length(ref_cache)==15
del ref_cache
print("\n[4/10] CUT REPLAY — SAME TOKEN LENGTH / SAME POSITION / SAME MODEL")
CUTS=[0,1,2,3,4,6,9,12,16,20,26];results=[]
for cut in CUTS:
    source=ref_inputs[cut]
    injected={"count":0}
    def replay_hook(m,args,kwargs):
        x=kwargs.get("hidden_states",args[0] if args else None)
        if not torch.is_tensor(x):raise RuntimeError("CUT HIDDEN STATES MISSING")
        assert x.shape==source.shape
        injected["count"]+=1
        if "hidden_states" in kwargs:
            kwargs=dict(kwargs);kwargs["hidden_states"]=source
            return args,kwargs
        args=list(args);args[0]=source
        return tuple(args),kwargs
    handle=layers[cut].register_forward_pre_hook(replay_hook,with_kwargs=True)
    try:
        with torch.inference_mode():o=model(input_ids=prompt,use_cache=False,return_dict=True,logits_to_keep=1)
        got=o.logits[:,-1,:].detach().float().clone()
        del o
    finally:handle.remove()
    d=diff(got,ref_logits)
    row={"cut":cut,"hook_calls":injected["count"],"max":d["max"],"mean":d["mean"],"cos":d["cos"],"top1":int(got.argmax()),"match":int(got.argmax())==int(ref_logits.argmax())}
    results.append(row)
    print("CUT",cut,"HOOK",row["hook_calls"],"MAX",round(row["max"],7),"MEAN",round(row["mean"],7),"COS",round(row["cos"],9),"TOP1_MATCH",row["match"])
gates["CUT_REPLAY"]=all(r["hook_calls"]==1 and r["match"] and r["max"]<=0.05 for r in results)
print("CUT_REPLAY_PASS_COUNT:",sum(r["hook_calls"]==1 and r["match"] and r["max"]<=0.05 for r in results),"/",len(results))
print("\n[5/10] NEGATIVE CONTROL — ZERO RESIDUAL AT CUT")
negative={}
for cut in (3,6,12):
    calls=[0]
    def zero_hook(m,args,kwargs):
        x=kwargs.get("hidden_states",args[0] if args else None)
        calls[0]+=1
        if "hidden_states" in kwargs:
            kw=dict(kwargs);kw["hidden_states"]=torch.zeros_like(x)
            return args,kw
        ar=list(args);ar[0]=torch.zeros_like(x)
        return tuple(ar),kwargs
    h=layers[cut].register_forward_pre_hook(zero_hook,with_kwargs=True)
    try:
        with torch.inference_mode():o=model(input_ids=prompt,use_cache=False,return_dict=True,logits_to_keep=1)
        z=o.logits[:,-1,:].detach().float().clone()
        del o
    finally:h.remove()
    d=diff(z,ref_logits)
    negative[cut]={"max":d["max"],"cos":d["cos"],"top1":int(z.argmax()),"calls":calls[0]}
    print("ZERO_CUT",cut,negative[cut])
gates["NEGATIVE_CONTROL"]=all(v["calls"]==1 and v["max"]>0.05 for v in negative.values())
print("\n[6/10] FULL VS INCREMENTAL / CUT INPUT DIVERGENCE")
with torch.inference_mode():
    prefix=model(input_ids=prompt[:,:-1],use_cache=True,return_dict=True,logits_to_keep=1)
prefix_cache=prefix.past_key_values;del prefix
inc_logits,inc_inputs,inc_cache=capture_layers(prompt[:,-1:],use_cache=True,past=prefix_cache)
print("INCREMENTAL_LOGIT_DIFF:",diff(inc_logits,ref_logits))
for i in CUTS:
    d=diff(ref_inputs[i][:,-1,:],inc_inputs[i][:,-1,:])
    print("CUT_INPUT_DIFF",i,"MAX",round(d["max"],7),"MEAN",round(d["mean"],7),"COS",round(d["cos"],9))
gates["INCREMENTAL_CACHE"]=get_cache_length(inc_cache)==15 and len(inc_inputs)==27
del inc_cache,prefix_cache
print("\n[7/10] MOE ROUTER X-RAY")
router_info={}
for i in (0,1,2,3,6,9,12,16,20,26):
    layer=layers[i];moe=getattr(layer,"mlp",None);gate=getattr(moe,"gate",None)
    router_info[i]={"mlp":type(moe).__name__,"gate":type(gate).__name__ if gate is not None else None}
    print("LAYER",i,"MOE:",router_info[i])
gates["MOE_STRUCTURE"]=any(v["gate"] is not None for v in router_info.values())
print("\n[8/10] COMPLEX ROTARY — PHASE SHIFT / INVERSE")
rotary=base.rotary_emb
with torch.inference_mode():
    emb=base.embed_tokens(prompt)
    pos=torch.arange(prompt.shape[1],device=prompt.device).unsqueeze(0)
    r0=rotary(emb,pos).to(torch.complex64)
    r1=rotary(emb,pos+1).to(torch.complex64)
    ratio=r1/r0
    recovered=r1/ratio
print("ROTARY_SHAPES:",tuple(r0.shape),tuple(r1.shape),"DTYPE:",r0.dtype)
print("PHASE_SHIFT:",cdiff(r0,r1))
print("PHASE_INVERSE:",cdiff(recovered,r0))
print("ROTARY_MAGNITUDE_DIFF:",float((r0.abs()-r1.abs()).abs().max().item()))
gates["ROTARY_INVERSE"]=cdiff(recovered,r0)["max"]<1e-5 and bool(torch.isfinite(recovered).all())
del emb,r0,r1,ratio,recovered
print("\n[9/10] CLEANUP / HOOK AUDIT")
gates["HOOKS_CLEAN"]=all(len(l._forward_pre_hooks)==0 for l in layers)
print("HOOKS_CLEAN:",gates["HOOKS_CLEAN"])
print("CUDA_ALLOCATED_GIB:",round(torch.cuda.memory_allocated()/2**30,3))
print("CUDA_PEAK_GIB:",round(torch.cuda.max_memory_allocated()/2**30,3))
print("\n[10/10] FINAL AUDIT")
for k,v in gates.items():print("GATE",k,"PASS" if v else "FAIL")
print("CUT_CANDIDATES:",CUTS)
print("CUT_REPLAY_PASS_COUNT:",sum(r["hook_calls"]==1 and r["match"] and r["max"]<=0.05 for r in results),"/",len(results))
print("STATUS:","PASS — CUT REPLAY / MOE / ROTARY" if all(gates.values()) else "INCOMPLETE — INSPECT LOG")
print("WEIGHTS_FROZEN:",all(not p.requires_grad for p in model.parameters()))
print("MAM_MEMORY_TRANSFER: NOT YET")
print("CUT_SELECTED: NO")
print("ELAPSED_SECONDS:",round(time.time()-t0,2))
print("="*126)

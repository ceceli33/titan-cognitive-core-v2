
# TEST578 — AKBASCORE MAM · DEEPSEEK MLA X-RAY · ATTENTION / CACHE / POSITION DIAGNOSTICS
# BASELINE: TEST577 · SAME MODEL / REVISION / BF16 / EAGER / FROZEN
import os,sys,time,gc,inspect,traceback
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch,transformers
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
from huggingface_hub import model_info
MODEL="deepseek-ai/DeepSeek-V2-Lite-Chat";REV="85864749cd611b4353ce1decdb286193298f64c7";TEST="588";DTYPE=torch.bfloat16;SEED=577
torch.manual_seed(SEED);torch.set_grad_enabled(False)
print("="*124)
print("TEST588 — AKBASCORE MAM · DEEPSEEK MLA X-RAY · FROZEN BF16")
print("="*124)
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(),"A100/BF16 REQUIRED"
print("GPU:",torch.cuda.get_device_name(0),"TORCH:",torch.__version__,"TRANSFORMERS:",transformers.__version__)
gates={};start=time.time()
def ti(x):
    if not torch.is_tensor(x):return type(x).__name__
    return {"shape":tuple(x.shape),"dtype":str(x.dtype),"finite":bool(torch.isfinite(x).all().item())}
def vals(x):
    if x is None:return []
    return list(x) if isinstance(x,(list,tuple,set)) else [x]
def cache_layers(c):
    if c is None:return []
    if hasattr(c,"layers"):return c.layers
    return []
def cache_length(c):
    try:return int(c.get_seq_length())
    except Exception:return None
def cache_snapshot(c):
    out={}
    for i,l in enumerate(cache_layers(c)):
        k=getattr(l,"keys",None);v=getattr(l,"values",None)
        if torch.is_tensor(k) and torch.is_tensor(v):out[i]={"k":k.detach().clone(),"v":v.detach().clone()}
    return out
def compare(a,b):
    a=a.detach().float();b=b.detach().float()
    assert a.shape==b.shape
    d=(a-b).abs()
    return {"max":float(d.max().item()),"mean":float(d.mean().item()),"cos":float(torch.nn.functional.cosine_similarity(a.reshape(1,-1),b.reshape(1,-1)).item())}
def print_cache(c,label):
    print("CACHE",label,"TYPE:",type(c).__name__,"LENGTH:",cache_length(c),"LAYERS:",len(cache_layers(c)))
    for i in (0,1,2,3,6,13,26):
        ls=cache_layers(c)
        if i>=len(ls):continue
        k=getattr(ls[i],"keys",None);v=getattr(ls[i],"values",None)
        print(" L",i,"K:",ti(k),"R:",ti(v))
print("\n[1/9] PINNED CONFIG / MODEL REUSE")
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
    print("MODEL_REUSED: NO — LOADING TEST577 BASELINE")
    kwargs=dict(config=cfg,revision=REV,trust_remote_code=False,device_map={"":0},low_cpu_mem_usage=True,output_loading_info=True,attn_implementation="eager",dtype=DTYPE)
    model,load_info=AutoModelForCausalLM.from_pretrained(MODEL,**kwargs)
    for k in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs"):
        v=vals(load_info.get(k,[]));print("LOAD",k,len(v),[str(z)[:120] for z in v[:5]])
        assert not v,"MODEL WEIGHT INTEGRITY FAILED: "+k
model.eval()
for p in model.parameters():p.requires_grad_(False)
base=model.model;layers=base.layers
assert len(layers)==27 and all(p.device.type=="cuda" and p.dtype==DTYPE for p in model.parameters() if p.is_floating_point())
gates["MODEL"]=True
print("PARAMETERS:",sum(p.numel() for p in model.parameters()),"TRAINABLE:",sum(p.numel() for p in model.parameters() if p.requires_grad))
print("CUDA_ALLOCATED_GIB:",round(torch.cuda.memory_allocated()/2**30,3))
print("\n[2/9] FIXED TOKENIZER / INPUT")
tok=globals().get("tok",None)
if tok is None:tok=AutoTokenizer.from_pretrained(MODEL,revision=REV,trust_remote_code=False)
encoded=tok.apply_chat_template([{"role":"user","content":"Reply with exactly one word: READY"}],add_generation_prompt=True,return_tensors="pt")
ids=encoded if torch.is_tensor(encoded) else encoded.input_ids if hasattr(encoded,"input_ids") else encoded["input_ids"]
if not torch.is_tensor(ids):ids=torch.as_tensor(ids,dtype=torch.long)
if ids.ndim==1:ids=ids.unsqueeze(0)
prompt=ids.to(model.get_input_embeddings().weight.device)
print("PROMPT:",repr(tok.decode(ids[0])),"SHAPE:",tuple(prompt.shape))
gates["PROMPT"]=tuple(prompt.shape)==(1,15)
print("\n[3/9] ATTENTION HOOK — KWARGS AWARE")
att={};handles=[]
def prehook(i):
    def hook(module,args,kwargs):
        x=kwargs.get("hidden_states",args[0] if args else None)
        if torch.is_tensor(x):att[i]=ti(x)
    return hook
try:
    for i,l in enumerate(layers):handles.append(l.self_attn.register_forward_pre_hook(prehook(i),with_kwargs=True))
    with torch.inference_mode():z=model(input_ids=prompt,use_cache=False,return_dict=True,logits_to_keep=1)
    print("FORWARD_LOGITS:",ti(z.logits));del z
finally:
    for h in handles:h.remove()
for i in range(27):print("ATTENTION_INPUT",i,att.get(i,"MISSING"))
gates["ATTENTION_27"]=len(att)==27
print("\n[4/9] FULL VS INCREMENTAL — LAYERWISE RESIDUAL X-RAY")
def capture_forward(input_ids,past=None,use_cache=False):
    captured={};hs=[]
    def mk(i):
        def hook(m,args,kwargs,out):
            x=out[0] if isinstance(out,(tuple,list)) else out
            if torch.is_tensor(x):captured[i]=x[:,-1,:].detach().float().cpu()
        return hook
    try:
        for i,l in enumerate(layers):hs.append(l.register_forward_hook(mk(i),with_kwargs=True))
        with torch.inference_mode():
            o=model(input_ids=input_ids,past_key_values=past,use_cache=use_cache,return_dict=True,logits_to_keep=1)
        logits=o.logits[:,-1,:].detach().float().cpu()
        cache=o.past_key_values
        del o
        return logits,captured,cache
    finally:
        for h in hs:h.remove()
full_logits,full_h,unused=capture_forward(prompt,use_cache=False)
del unused
prefix_logits,prefix_h,prefix_cache=capture_forward(prompt[:,:-1],use_cache=True)
del prefix_logits,prefix_h
inc_logits,inc_h,inc_cache=capture_forward(prompt[:,-1:],past=prefix_cache,use_cache=True)
print("FULL_TOP1:",int(full_logits.argmax()),repr(tok.decode([int(full_logits.argmax())])))
print("INCREMENTAL_TOP1:",int(inc_logits.argmax()),repr(tok.decode([int(inc_logits.argmax())])))
print("LOGIT_COMPARISON:",compare(full_logits,inc_logits))
first_divergence=None
for i in range(27):
    r=compare(full_h[i],inc_h[i]);print("LAYER",i,"MAX:",round(r["max"],6),"MEAN:",round(r["mean"],6),"COS:",round(r["cos"],8))
    if first_divergence is None and r["max"]>0.01:first_divergence=i
print("FIRST_LAYER_MAX_DIFF_GT_0.01:",first_divergence)
gates["RESIDUAL_27"]=len(full_h)==27 and len(inc_h)==27
gates["TOP1_MATCH"]=int(full_logits.argmax())==int(inc_logits.argmax())
print("\n[5/9] CACHE STRUCTURE / INCREMENTAL GROWTH")
print_cache(inc_cache,"INCREMENTAL")
gates["CACHE_27"]=len(cache_layers(inc_cache))==27
gates["CACHE_LENGTH"]=cache_length(inc_cache)==prompt.shape[1]
print("\n[6/9] CACHE PREFIX IMMUTABILITY")
# Compare independent prefix cache with corresponding prefix rows after incremental update.
# prefix_cache is updated in-place by incremental forward, so create a fresh reference.
with torch.inference_mode():
    ref=model(input_ids=prompt[:,:-1],use_cache=True,return_dict=True,logits_to_keep=1)
ref_cache=ref.past_key_values;del ref
prefix_max=0.0;prefix_mean=0.0;prefix_count=0
for i in range(27):
    a=ref_cache.layers[i];b=inc_cache.layers[i]
    for name in ("keys","values"):
        x=getattr(a,name,None);y=getattr(b,name,None)
        if not torch.is_tensor(x) or not torch.is_tensor(y):continue
        if x.shape[-2]!=prompt.shape[1]-1 or y.shape[-2]!=prompt.shape[1]:continue
        d=(x.float()-y[...,:-1,:].float()).abs()
        prefix_max=max(prefix_max,float(d.max().item()))
        prefix_mean+=float(d.mean().item());prefix_count+=1
print("PREFIX_MAX_ABS_DIFF:",prefix_max,"MEAN_OF_COMPONENT_MEANS:",prefix_mean/max(prefix_count,1),"COMPONENTS:",prefix_count)
gates["PREFIX_IMMUTABLE"]=prefix_count==54 and prefix_max==0.0
del ref_cache,prefix_cache,inc_cache
gc.collect();torch.cuda.empty_cache()
print("\n[7/9] ROTARY / POSITION SHIFT X-RAY")
rotary=base.rotary_emb
with torch.inference_mode():
    emb=base.embed_tokens(prompt)
    p0=torch.arange(prompt.shape[1],device=prompt.device).unsqueeze(0)
    p1=p0+1
    r0=rotary(emb,p0);r1=rotary(emb,p1)
print("ROTARY_TYPE:",type(r0).__name__,"R0:",ti(r0),"R1:",ti(r1))
if torch.is_tensor(r0) and torch.is_tensor(r1):
    rr=compare(r0,r1);print("ROTARY_SHIFT_PLUS_ONE:",rr)
    gates["ROTARY_SHIFT"]=bool(torch.isfinite(r0).all() and torch.isfinite(r1).all()) and rr["max"]>0
else:gates["ROTARY_SHIFT"]=False
del emb,r0,r1
print("\n[8/9] SHORT GREEDY / MEMORY AUDIT")
try:
    with torch.inference_mode():gen=model.generate(input_ids=prompt,max_new_tokens=8,do_sample=False,pad_token_id=tok.eos_token_id)
    print("GENERATED:",repr(tok.decode(gen[0,prompt.shape[1]:],skip_special_tokens=True)))
    gates["GENERATE"]=gen.shape[1]>prompt.shape[1]
    del gen
except Exception as e:
    gates["GENERATE"]=False;print("GENERATE_ERROR:",repr(e))
print("CUDA_ALLOCATED_GIB:",round(torch.cuda.memory_allocated()/2**30,3))
print("CUDA_PEAK_GIB:",round(torch.cuda.max_memory_allocated()/2**30,3))
print("\n[9/9] FINAL AUDIT")
for k,v in gates.items():print("GATE",k,"PASS" if v else "FAIL")
print("FIRST_RESIDUAL_DIVERGENCE:",first_divergence)
print("STATUS:","PASS — DEEPSEEK MLA X-RAY COMPLETE" if all(gates.values()) else "INCOMPLETE — INSPECT GATES")
print("MODEL_FROZEN:",all(not p.requires_grad for p in model.parameters()))
print("MAM_WRITE_INIT_APPEND_ASK: NOT MODIFIED")
print("CUT_SELECTED: NO")
print("ELAPSED_SECONDS:",round(time.time()-start,2))
print("="*124)

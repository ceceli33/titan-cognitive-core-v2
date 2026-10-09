
# TEST577 — AKBASCORE MAM · DEEPSEEK-V2-LITE-CHAT · STABLE BF16 / MLA X-RAY BASELINE
import os,sys,time,gc,inspect,traceback,hashlib
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch,transformers
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
from huggingface_hub import model_info
MODEL="deepseek-ai/DeepSeek-V2-Lite-Chat";REV="85864749cd611b4353ce1decdb286193298f64c7";TEST="577";DTYPE=torch.bfloat16;SEED=577
torch.manual_seed(SEED);torch.set_grad_enabled(False)
print("="*120)
print("TEST577 — AKBASCORE MAM · DEEPSEEK MLA X-RAY · STABLE FROZEN BF16 BASELINE")
print("="*120)
print("PYTHON:",sys.version.split()[0],"TORCH:",torch.__version__,"TRANSFORMERS:",transformers.__version__)
assert torch.cuda.is_available(),"CUDA GPU bulunamadı."
assert torch.cuda.is_bf16_supported(),"GPU BF16 desteklemiyor."
print("GPU:",torch.cuda.get_device_name(0),"VRAM_GIB:",round(torch.cuda.get_device_properties(0).total_memory/2**30,3))
gates={};t_start=time.time()
def tensor_info(x):
    if not torch.is_tensor(x):return type(x).__name__
    return {"shape":tuple(x.shape),"dtype":str(x.dtype),"device":str(x.device),"finite":bool(torch.isfinite(x).all().item()) if x.is_floating_point() else True}
def safe_values(x):
    if x is None:return []
    if isinstance(x,(set,list,tuple)):return list(x)
    if isinstance(x,dict):return list(x.items())
    return [x]
def cache_length(c):
    if c is None:return None
    if hasattr(c,"get_seq_length"):
        try:return int(c.get_seq_length())
        except Exception:return None
    return None
def cache_view(c,label):
    print("CACHE_VIEW:",label,"TYPE:",type(c).__name__)
    if c is None:return {"present":False,"length":None,"layers":0,"shapes":{}}
    length=cache_length(c);print(" CACHE_LENGTH:",length)
    shapes={};count=0
    if hasattr(c,"layers"):
        count=len(c.layers)
        for i,l in enumerate(c.layers):
            entry={}
            for name in ("keys","values","key","value"):
                v=getattr(l,name,None)
                if torch.is_tensor(v):entry[name]=tuple(v.shape)
            shapes[i]=entry
            if i in (0,1,2,3,6,13,count-1):
                print(" CACHE_LAYER:",i,"CLASS:",type(l).__name__)
                for name,shape in entry.items():print("  ",name,shape)
    elif hasattr(c,"key_cache"):
        count=len(c.key_cache)
        for i in range(count):
            k=c.key_cache[i];v=c.value_cache[i] if hasattr(c,"value_cache") else None
            shapes[i]={"key":tuple(k.shape) if torch.is_tensor(k) else None,"value":tuple(v.shape) if torch.is_tensor(v) else None}
            if i in (0,1,2,3,6,13,count-1):print(" CACHE_LAYER:",i,shapes[i])
    elif isinstance(c,(tuple,list)):
        count=len(c)
        for i,l in enumerate(c):
            shapes[i]={str(j):tuple(t.shape) for j,t in enumerate(l) if torch.is_tensor(t)}
            if i in (0,1,2,3,6,13,count-1):print(" CACHE_LAYER:",i,shapes[i])
    else:print(" CACHE_PUBLIC_ATTRS:",[x for x in dir(c) if any(z in x.lower() for z in ("cache","key","value","layer")) and not x.startswith("_")][:25])
    print(" CACHE_LAYER_COUNT:",count)
    return {"present":True,"length":length,"layers":count,"shapes":shapes}
def cuda_gib():return round(torch.cuda.memory_allocated()/2**30,3)
print("\n[1/10] PINNED REVISION / CONFIG")
info=model_info(MODEL,revision=REV)
assert info.sha==REV,f"REVISION MISMATCH: {info.sha}"
cfg=AutoConfig.from_pretrained(MODEL,revision=REV,trust_remote_code=False)
assert cfg.model_type=="deepseek_v2" and cfg.num_hidden_layers==27
cfg._attn_implementation="eager"
print("REVISION:",info.sha,"CONFIG:",type(cfg).__module__,type(cfg).__name__)
for k in ("num_hidden_layers","hidden_size","num_attention_heads","num_key_value_heads","kv_lora_rank","qk_nope_head_dim","qk_rope_head_dim","v_head_dim","rope_scaling","n_routed_experts","num_experts_per_tok"):
    print(k,":",getattr(cfg,k,None))
gates["CONFIG"]=True
print("\n[2/10] TOKENIZER / FIXED PROMPT")
tok=AutoTokenizer.from_pretrained(MODEL,revision=REV,trust_remote_code=False)
encoded=tok.apply_chat_template([{"role":"user","content":"Reply with exactly one word: READY"}],add_generation_prompt=True,return_tensors="pt")
ids=encoded if torch.is_tensor(encoded) else encoded.input_ids if hasattr(encoded,"input_ids") else encoded["input_ids"]
if not torch.is_tensor(ids):ids=torch.as_tensor(ids,dtype=torch.long)
if ids.ndim==1:ids=ids.unsqueeze(0)
assert ids.ndim==2 and ids.shape[0]==1 and ids.shape[1]>1
print("PROMPT_TOKENS:",tuple(ids.shape),"PROMPT:",repr(tok.decode(ids[0])))
gates["TOKENIZER"]=True
print("\n[3/10] STRICT WEIGHT LOAD / SESSION REUSE")
old=globals().get("model",None)
reuse=(old is not None and isinstance(old,torch.nn.Module) and type(old).__module__.startswith("transformers.models.deepseek_v2") and getattr(old.config,"_name_or_path","")==MODEL and getattr(old.config,"num_hidden_layers",None)==27 and all(p.device.type=="cuda" and (not p.is_floating_point() or p.dtype==DTYPE) for p in old.parameters()))
loading_info=None
if reuse:
    model=old;print("MODEL_REUSED: YES — no duplicate GPU allocation")
    print("WARNING: Reused model weight-loading diagnostics are not available in this execution.")
else:
    if old is not None:
        try:
            del old
            if "model" in globals():del globals()["model"]
        except Exception:pass
        gc.collect();torch.cuda.empty_cache()
    print("MODEL_REUSED: NO — loading pinned original weights")
    t0=time.time()
    kwargs=dict(config=cfg,revision=REV,trust_remote_code=False,device_map={"":0},low_cpu_mem_usage=True,output_loading_info=True,attn_implementation="eager")
    kwargs["dtype" if int(transformers.__version__.split(".")[0])>=5 else "torch_dtype"]=DTYPE
    model,loading_info=AutoModelForCausalLM.from_pretrained(MODEL,**kwargs)
    print("LOAD_SECONDS:",round(time.time()-t0,2))
model.eval()
for p in model.parameters():p.requires_grad_(False)
print("MODEL_CLASS:",type(model).__module__,type(model).__name__)
print("PARAMETERS:",sum(p.numel() for p in model.parameters()))
print("TRAINABLE:",sum(p.numel() for p in model.parameters() if p.requires_grad))
print("CUDA_ALLOCATED_GIB:",cuda_gib())
if loading_info is not None:
    for name in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs"):
        vals=safe_values(loading_info.get(name,[]))
        print("LOADING",name,"COUNT:",len(vals),"SAMPLE:",[str(x)[:150] for x in vals[:12]])
    gates["WEIGHTS_STRICT"]=all(len(safe_values(loading_info.get(k,[])))==0 for k in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs"))
else:
    gates["WEIGHTS_STRICT"]=None
    print("WEIGHTS_STRICT: NOT RECHECKED — session model reused")
gates["FROZEN"]=all(not p.requires_grad for p in model.parameters())
gates["CUDA_ONLY"]=all(p.device.type=="cuda" for p in model.parameters())
gates["BF16_ONLY"]=all(p.dtype==DTYPE for p in model.parameters() if p.is_floating_point())
gates["MODEL_CLASS"]=type(model).__module__=="transformers.models.deepseek_v2.modeling_deepseek_v2"
print("MODEL_GATES:",{k:v for k,v in gates.items() if k in ("WEIGHTS_STRICT","FROZEN","CUDA_ONLY","BF16_ONLY","MODEL_CLASS")})
assert all(gates[k] for k in ("FROZEN","CUDA_ONLY","BF16_ONLY","MODEL_CLASS")),"MODEL INTEGRITY FAILURE"
assert gates["WEIGHTS_STRICT"] is not False,"WEIGHT KEY INTEGRITY FAILURE"
print("\n[4/10] 27-LAYER MLA MODULE TREE")
base=getattr(model,"model",None)
assert base is not None and hasattr(base,"layers")
layers=base.layers;assert len(layers)==27
print("LAYER_COUNT:",len(layers))
for i in (0,1,2,3,6,13,26):
    l=layers[i];a=l.self_attn
    print("LAYER:",i,"TYPE:",type(l).__name__,"ATTENTION:",type(a).__name__)
    for name in ("q_proj","q_a_proj","q_b_proj","kv_a_proj_with_mqa","kv_b_proj","o_proj"):
        m=getattr(a,name,None)
        if m is not None:print(" ",name,tuple(m.weight.shape) if hasattr(m,"weight") else type(m).__name__)
gates["LAYERS"]=True
print("\n[5/10] FULL FORWARD / 27-LAYER HIDDEN X-RAY")
hidden={};att_inputs={};handles=[]
def hidden_hook(i):
    def hook(m,inp,out):
        t=out[0] if isinstance(out,(tuple,list)) else out
        if torch.is_tensor(t):hidden[i]=tensor_info(t)
    return hook
def attention_hook(i):
    def hook(m,args):
        if args and torch.is_tensor(args[0]):att_inputs[i]=tensor_info(args[0])
    return hook
for i,l in enumerate(layers):
    handles.append(l.register_forward_hook(hidden_hook(i)))
    if i in (0,1,2,3,6,13,26):handles.append(l.self_attn.register_forward_pre_hook(attention_hook(i)))
try:
    prompt=ids.to(model.get_input_embeddings().weight.device)
    with torch.inference_mode():out=model(input_ids=prompt,use_cache=True,return_dict=True)
finally:
    for h in handles:h.remove()
print("LOGITS:",tensor_info(out.logits))
for i in range(27):print("HIDDEN",i,hidden.get(i,"MISSING"))
for i in sorted(att_inputs):print("ATTENTION_INPUT",i,att_inputs[i])
gates["HIDDEN_27"]=len(hidden)==27
gates["ATTENTION_INPUT"]=len(att_inputs)==7
gates["HOOK_CLEAN"]=all(h.id not in h.hooks_dict_ref() if h.hooks_dict_ref() is not None else True for h in handles)
print("\n[6/10] REAL MLA CACHE X-RAY")
full_cache=cache_view(out.past_key_values,"FULL_PROMPT")
gates["CACHE_PRESENT"]=full_cache["present"] and full_cache["layers"]==27 and full_cache["length"]==prompt.shape[1]
del out;gc.collect()
print("\n[7/10] FULL VS INCREMENTAL / CACHE GROWTH")
try:
    with torch.inference_mode():
        full=model(input_ids=prompt,use_cache=False,return_dict=True)
        full_last=full.logits[:,-1,:].float().clone()
        del full
        prefix=model(input_ids=prompt[:,:-1],use_cache=True,return_dict=True)
        prefix_cache=prefix.past_key_values
        prefix_info=cache_view(prefix_cache,"PREFIX")
        del prefix
        inc=model(input_ids=prompt[:,-1:],past_key_values=prefix_cache,use_cache=True,return_dict=True)
        inc_last=inc.logits[:,-1,:].float().clone()
        inc_info=cache_view(inc.past_key_values,"INCREMENTAL")
        del inc,prefix_cache
    diff=float((full_last-inc_last).abs().max().item())
    top_full=int(full_last.argmax().item());top_inc=int(inc_last.argmax().item())
    print("MAX_ABS_LOGIT_DIFF:",diff,"FULL_TOP1:",top_full,"INCREMENTAL_TOP1:",top_inc)
    print("FULL_TOKEN:",repr(tok.decode([top_full])),"INCREMENTAL_TOKEN:",repr(tok.decode([top_inc])))
    print("PREFIX_LENGTH:",prefix_info["length"],"INCREMENTAL_LENGTH:",inc_info["length"])
    gates["INCREMENTAL"]=top_full==top_inc and bool(torch.isfinite(full_last).all()) and bool(torch.isfinite(inc_last).all())
    gates["CACHE_GROWTH"]=prefix_info["length"]==prompt.shape[1]-1 and inc_info["length"]==prompt.shape[1]
    del full_last,inc_last
except Exception as e:
    gates["INCREMENTAL"]=False;gates["CACHE_GROWTH"]=False
    print("INCREMENTAL_ERROR:",type(e).__name__,str(e)[:1500])
    traceback.print_exc(limit=5)
gc.collect();torch.cuda.empty_cache()
print("\n[8/10] ROTARY / POSITION X-RAY")
rotary=getattr(base,"rotary_emb",None)
print("ROTARY_CLASS:",type(rotary).__name__ if rotary is not None else None)
if rotary is not None:
    print("ROTARY_SIGNATURE:",inspect.signature(rotary.forward))
    try:
        with torch.inference_mode():
            pos=torch.arange(prompt.shape[1],device=prompt.device).unsqueeze(0)
            emb=base.embed_tokens(prompt)
            rope=rotary(emb,position_ids=pos)
        if isinstance(rope,(tuple,list)):
            for i,t in enumerate(rope):print("ROTARY_OUTPUT",i,tensor_info(t))
        else:print("ROTARY_OUTPUT:",tensor_info(rope))
        gates["ROTARY"]=True
        del emb,rope
    except Exception as e:
        gates["ROTARY"]=False
        print("ROTARY_ERROR:",type(e).__name__,str(e)[:1200])
else:gates["ROTARY"]=False
print("\n[9/10] FROZEN GREEDY GENERATION")
try:
    with torch.inference_mode():gen=model.generate(input_ids=prompt,max_new_tokens=8,do_sample=False,pad_token_id=tok.eos_token_id)
    print("GENERATED:",repr(tok.decode(gen[0,prompt.shape[1]:],skip_special_tokens=True)))
    gates["GENERATE"]=gen.shape[1]>prompt.shape[1]
    del gen
except Exception as e:
    gates["GENERATE"]=False
    print("GENERATE_ERROR:",type(e).__name__,str(e)[:1500])
    traceback.print_exc(limit=5)
print("\n[10/10] FINAL BASELINE AUDIT")
for k,v in gates.items():print("GATE",k,"PASS" if v is True else "FAIL" if v is False else "NOT_RECHECKED")
print("REVISION:",REV)
print("MODEL_CLASS:",type(model).__module__,type(model).__name__)
print("TRAINABLE:",sum(p.numel() for p in model.parameters() if p.requires_grad))
print("CUDA_ALLOCATED_GIB:",cuda_gib())
print("CUDA_PEAK_GIB:",round(torch.cuda.max_memory_allocated()/2**30,3))
print("ELAPSED_SECONDS:",round(time.time()-t_start,2))
print("STATUS:","PASS — STABLE DEEPSEEK MLA X-RAY BASELINE" if all(v is True for v in gates.values()) else "INCOMPLETE — INSPECT GATES")
print("MEMORY_TRANSFER_PERFORMED: NO")
print("CUT_SELECTED: NO")
print("="*120)

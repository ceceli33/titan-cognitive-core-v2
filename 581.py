
# TEST581 — AKBASCORE MAM · DEEPSEEK CUT-CONDITIONED INIT · SINGLE CARTRIDGE
# BASELINE: TEST580 · FROZEN BF16 · EAGER · NATIVE MLA CACHE K512/R64 · SINGLE COLAB CELL
import os,time,gc,hashlib,traceback
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch,transformers
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
MODEL="deepseek-ai/DeepSeek-V2-Lite-Chat";REV="85864749cd611b4353ce1decdb286193298f64c7";TEST="581";DTYPE=torch.bfloat16;SEED=577
torch.manual_seed(SEED);torch.set_grad_enabled(False)
print("="*132)
print("TEST581 — AKBASCORE MAM · DEEPSEEK CUT-CONDITIONED INIT · FROZEN BF16")
print("="*132)
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(),"CUDA BF16 REQUIRED"
print("GPU:",torch.cuda.get_device_name(0),"TORCH:",torch.__version__,"TRANSFORMERS:",transformers.__version__)
t0=time.time();gates={}
def cache_len(c):
    try:return int(c.get_seq_length())
    except Exception:return None
def cache_clone(c):
    z=DynamicCache()
    for i,l in enumerate(c.layers):
        k=getattr(l,"keys",None);v=getattr(l,"values",None)
        assert torch.is_tensor(k) and torch.is_tensor(v),f"CACHE L{i} INVALID"
        z.update(k.detach().clone(),v.detach().clone(),i)
    return z
def token_ids(s):
    return tok(s,add_special_tokens=False,return_tensors="pt").input_ids.to(device)
def run(ids,past=None):
    with torch.inference_mode():
        return model(input_ids=ids,past_key_values=past,use_cache=True,return_dict=True,logits_to_keep=1)
def decode_cache(c,ask,max_new=24):
    ids=token_ids(ask);cache=cache_clone(c);out_tokens=[]
    for step in range(max_new):
        o=run(ids if step==0 else nxt,cache)
        cache=o.past_key_values;nxt=o.logits[:,-1,:].argmax(-1,keepdim=True)
        tid=int(nxt.item())
        if tid==tok.eos_token_id:break
        out_tokens.append(tid)
    return tok.decode(out_tokens,skip_special_tokens=True).strip()
def exact(s):return s.strip().strip(" .,:;\"'`")=="QN-4826"
def logits_diff(a,b):
    d=(a.float()-b.float()).abs()
    return float(d.max().item()),float(d.mean().item())
print("\n[1/10] MODEL / TEST580 REUSE")
old=globals().get("model",None)
reuse=(isinstance(old,torch.nn.Module) and type(old).__module__=="transformers.models.deepseek_v2.modeling_deepseek_v2" and getattr(old.config,"_name_or_path",None)==MODEL and len(getattr(getattr(old,"model",None),"layers",[]))==27 and all(p.device.type=="cuda" and (not p.is_floating_point() or p.dtype==DTYPE) for p in old.parameters()))
if reuse:model=old;print("MODEL_REUSED: YES")
else:
    if isinstance(old,torch.nn.Module):
        del old;globals().pop("model",None);gc.collect();torch.cuda.empty_cache()
    print("MODEL_REUSED: NO — LOADING TEST580 BASELINE")
    cfg=AutoConfig.from_pretrained(MODEL,revision=REV,trust_remote_code=False);cfg._attn_implementation="eager"
    model,li=AutoModelForCausalLM.from_pretrained(MODEL,config=cfg,revision=REV,trust_remote_code=False,device_map={"":0},low_cpu_mem_usage=True,output_loading_info=True,attn_implementation="eager",dtype=DTYPE)
    for k in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs"):assert not li.get(k,[]),f"LOAD ERROR: {k}"
model.eval()
for p in model.parameters():p.requires_grad_(False)
base=model.model;layers=base.layers;device=model.get_input_embeddings().weight.device
assert len(layers)==27 and all(p.device.type=="cuda" and p.dtype==DTYPE for p in model.parameters() if p.is_floating_point())
tok=globals().get("tok",None)
if tok is None:tok=AutoTokenizer.from_pretrained(MODEL,revision=REV,trust_remote_code=False)
gates["MODEL_FROZEN"]=all(not p.requires_grad for p in model.parameters())
print("PARAMETERS:",sum(p.numel() for p in model.parameters()),"TRAINABLE:",sum(p.numel() for p in model.parameters() if p.requires_grad))
print("\n[2/10] FIXED FACT / QUESTION")
FACT="The access code for the fictional station VELORA-731 is QN-4826."
QUESTION="What is the access code for the fictional station VELORA-731? Answer with the code only."
SOURCE=tok.apply_chat_template([{"role":"user","content":"Memorize this fictional record: "+FACT}],tokenize=False,add_generation_prompt=True)
ASK=tok.apply_chat_template([{"role":"user","content":QUESTION}],tokenize=False,add_generation_prompt=True)
write_ids=token_ids(SOURCE);ask_ids=token_ids(ASK)
print("SOURCE_TOKENS:",write_ids.shape[1],"ASK_TOKENS:",ask_ids.shape[1])
gates["SOURCE_FREE_ASK"]="QN-4826" not in ASK and FACT not in ASK
print("\n[3/10] WRITE — MLA CACHE + ALL CUT RESIDUALS")
captured={};hs=[]
def mk(i):
    def hook(m,args,kwargs):
        x=kwargs.get("hidden_states",args[0] if args else None)
        if torch.is_tensor(x):captured[i]=x.detach().clone()
    return hook
try:
    for i,l in enumerate(layers):hs.append(l.register_forward_pre_hook(mk(i),with_kwargs=True))
    o=run(write_ids)
finally:
    for h in hs:h.remove()
cartridge={"cache":cache_clone(o.past_key_values),"residual":captured,"source_sha256":hashlib.sha256(FACT.encode()).hexdigest()}
del o
assert len(captured)==27 and cache_len(cartridge["cache"])==write_ids.shape[1]
gates["WRITE"]=True
print("WRITE_CACHE:",cache_len(cartridge["cache"]),"LAYERS:",len(cartridge["cache"].layers),"RESIDUALS:",len(captured))
print("\n[4/10] BASELINE — NATIVE CACHE ASK")
native_cache_answer=decode_cache(cartridge["cache"],ASK)
print("NATIVE_CACHE_ANSWER:",repr(native_cache_answer))
gates["NATIVE_CACHE_EXACT"]=exact(native_cache_answer)
print("\n[5/10] CUT-CONDITIONED REPLAY — ZERO TOKEN INPUT / RESIDUAL INJECTION")
# This is a diagnostic reconstruction, not yet an independent compressed cartridge.
CUTS=[0,1,2,3,4,6,9,12,16,20,26]
cut_results=[]
for cut in CUTS:
    src=captured[cut];count=[0]
    # Zero embeddings at every source position; inject stored full-sequence CUT state.
    # The original source text is not sent to the model in this pass.
    dummy=torch.full_like(write_ids,tok.eos_token_id)
    def replace_hook(m,args,kwargs):
        x=kwargs.get("hidden_states",args[0] if args else None)
        assert x.shape==src.shape
        count[0]+=1
        if "hidden_states" in kwargs:
            kw=dict(kwargs);kw["hidden_states"]=src
            return args,kw
        ar=list(args);ar[0]=src
        return tuple(ar),kwargs
    handle=layers[cut].register_forward_pre_hook(replace_hook,with_kwargs=True)
    try:
        with torch.inference_mode():
            out=model(input_ids=dummy,use_cache=True,return_dict=True,logits_to_keep=1)
        rebuilt_cache=cache_clone(out.past_key_values)
        del out
    finally:handle.remove()
    # Earlier cache layers are not reconstructed by a CUT injection.
    # Hybrid cache: native WRITE layers below CUT + replay-built layers at/above CUT.
    hybrid=cache_clone(cartridge["cache"])
    for i in range(cut,27):
        hybrid.layers[i].keys=rebuilt_cache.layers[i].keys.detach().clone()
        hybrid.layers[i].values=rebuilt_cache.layers[i].values.detach().clone()
    maxdiff=0.0;meandiff=0.0
    for i in range(cut,27):
        for key in ("keys","values"):
            a=getattr(hybrid.layers[i],key).float();b=getattr(cartridge["cache"].layers[i],key).float()
            d=(a-b).abs();maxdiff=max(maxdiff,float(d.max().item()));meandiff+=float(d.mean().item())
    ans=decode_cache(hybrid,ASK)
    row={"cut":cut,"calls":count[0],"max_cache_diff":maxdiff,"mean_cache_diff_sum":meandiff,"answer":ans,"exact":exact(ans)}
    cut_results.append(row)
    print("CUT",cut,"HOOK",count[0],"CACHE_MAX",round(maxdiff,7),"CACHE_MEAN_SUM",round(meandiff,7),"ANSWER",repr(ans),"EXACT",row["exact"])
    del hybrid,rebuilt_cache
gates["CUT_HOOKS"]=all(r["calls"]==1 for r in cut_results)
print("\n[6/10] NEGATIVE CONTROL — ZERO CUT STATE")
cut=6;src=captured[cut];dummy=torch.full_like(write_ids,tok.eos_token_id)
def zero_hook(m,args,kwargs):
    x=kwargs.get("hidden_states",args[0] if args else None)
    if "hidden_states" in kwargs:
        kw=dict(kwargs);kw["hidden_states"]=torch.zeros_like(x)
        return args,kw
    ar=list(args);ar[0]=torch.zeros_like(x)
    return tuple(ar),kwargs
h=layers[cut].register_forward_pre_hook(zero_hook,with_kwargs=True)
try:
    with torch.inference_mode():z=model(input_ids=dummy,use_cache=True,return_dict=True,logits_to_keep=1)
    zero_cache=cache_clone(z.past_key_values);del z
finally:h.remove()
hybrid=cache_clone(cartridge["cache"])
for i in range(cut,27):
    hybrid.layers[i].keys=zero_cache.layers[i].keys.detach().clone()
    hybrid.layers[i].values=zero_cache.layers[i].values.detach().clone()
zero_answer=decode_cache(hybrid,ASK)
print("ZERO_CUT6_ANSWER:",repr(zero_answer),"EXACT:",exact(zero_answer))
gates["NEGATIVE_CONTROL"]=not exact(zero_answer)
del hybrid,zero_cache
print("\n[7/10] NO MEMORY CONTROL")
with torch.inference_mode():
    gen=model.generate(input_ids=ask_ids,max_new_tokens=24,do_sample=False,pad_token_id=tok.eos_token_id)
no_answer=tok.decode(gen[0,ask_ids.shape[1]:],skip_special_tokens=True).strip()
print("NO_MEMORY_ANSWER:",repr(no_answer))
gates["NO_MEMORY_NEGATIVE"]=not exact(no_answer)
del gen
print("\n[8/10] CUT CANDIDATE RANKING")
ranked=sorted(cut_results,key=lambda x:(not x["exact"],x["max_cache_diff"],x["cut"]))
for j,r in enumerate(ranked):
    print("RANK",j+1,"CUT",r["cut"],"EXACT",r["exact"],"CACHE_MAX",round(r["max_cache_diff"],7))
best=ranked[0]
gates["CUT_RECALL_ANY"]=any(r["exact"] for r in cut_results)
print("BEST_DIAGNOSTIC_CUT:",best["cut"],"ANSWER:",repr(best["answer"]))
print("\n[9/10] HOOK / WEIGHT / MEMORY AUDIT")
gates["HOOKS_CLEAN"]=all(len(l._forward_pre_hooks)==0 for l in layers)
gates["WEIGHTS_FROZEN"]=all(not p.requires_grad for p in model.parameters())
print("CUDA_ALLOCATED_GIB:",round(torch.cuda.memory_allocated()/2**30,3))
print("CUDA_PEAK_GIB:",round(torch.cuda.max_memory_allocated()/2**30,3))
print("\n[10/10] FINAL AUDIT")
for k,v in gates.items():print("GATE",k,"PASS" if v else "FAIL")
print("CUT_EXACT_COUNT:",sum(r["exact"] for r in cut_results),"/",len(cut_results))
print("STATUS:","PASS — CUT-CONDITIONED SINGLE CARTRIDGE PILOT" if all(gates.values()) else "INCOMPLETE — INSPECT GATES")
print("CARTRIDGE_KIND: NATIVE MLA CACHE + FULL-SEQUENCE CUT RESIDUAL")
print("INDEPENDENT_COMPRESSED_INIT: NOT PROVEN")
print("APPEND_IMPLEMENTED: NO")
print("ELAPSED_SECONDS:",round(time.time()-t0,2))
print("="*132)

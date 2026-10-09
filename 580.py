
# TEST580 — AKBASCORE MAM · DEEPSEEK-V2-LITE-CHAT · SINGLE CARTRIDGE WRITE / INIT / ASK
# BASELINE: TEST579 · FROZEN BF16 · EAGER · 27L · NATIVE MLA K512/R64 · SINGLE COLAB CELL
import os,time,gc,hashlib,traceback
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch,transformers
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
MODEL="deepseek-ai/DeepSeek-V2-Lite-Chat";REV="85864749cd611b4353ce1decdb286193298f64c7";TEST="580";DTYPE=torch.bfloat16;SEED=577
torch.manual_seed(SEED);torch.set_grad_enabled(False)
print("="*126)
print("TEST580 — AKBASCORE MAM · DEEPSEEK SINGLE CARTRIDGE WRITE → INIT → ASK")
print("="*126)
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(),"CUDA BF16 REQUIRED"
print("GPU:",torch.cuda.get_device_name(0),"TORCH:",torch.__version__,"TRANSFORMERS:",transformers.__version__)
t0=time.time();gates={}
def cache_len(c):
    try:return int(c.get_seq_length())
    except Exception:return None
def cache_clone(c):
    from transformers.cache_utils import DynamicCache
    z=DynamicCache()
    for i,l in enumerate(c.layers):
        k=getattr(l,"keys",None);v=getattr(l,"values",None)
        if not torch.is_tensor(k) or not torch.is_tensor(v):raise RuntimeError(f"CACHE L{i} INVALID")
        z.update(k.detach().clone(),v.detach().clone(),i)
    return z
def token_ids(s):
    e=tok(s,add_special_tokens=False,return_tensors="pt")
    return e.input_ids.to(device)
def run(ids,past=None,keep=1):
    with torch.inference_mode():
        return model(input_ids=ids,past_key_values=past,use_cache=True,return_dict=True,logits_to_keep=keep)
def answer_from_cache(cache,question,max_new=24):
    # Native autoregressive ASK; question text contains no source passage.
    ids=token_ids(question)
    c=cache_clone(cache)
    generated=[]
    for step in range(max_new):
        o=run(ids if step==0 else nxt,c)
        c=o.past_key_values
        nxt=o.logits[:,-1,:].argmax(-1,keepdim=True)
        tid=int(nxt.item())
        if tid==tok.eos_token_id:break
        generated.append(tid)
    return tok.decode(generated,skip_special_tokens=True).strip()
print("\n[1/9] REUSE TEST579 MODEL")
old=globals().get("model",None)
reuse=(isinstance(old,torch.nn.Module) and type(old).__module__=="transformers.models.deepseek_v2.modeling_deepseek_v2" and getattr(old.config,"_name_or_path",None)==MODEL and len(getattr(getattr(old,"model",None),"layers",[]))==27 and all(p.device.type=="cuda" and (not p.is_floating_point() or p.dtype==DTYPE) for p in old.parameters()))
if reuse:
    model=old;print("MODEL_REUSED: YES")
else:
    if isinstance(old,torch.nn.Module):
        if "model" in globals():del globals()["model"]
        del old;gc.collect();torch.cuda.empty_cache()
    print("MODEL_REUSED: NO — LOADING TEST579 BASELINE")
    cfg=AutoConfig.from_pretrained(MODEL,revision=REV,trust_remote_code=False)
    cfg._attn_implementation="eager"
    model,li=AutoModelForCausalLM.from_pretrained(MODEL,config=cfg,revision=REV,trust_remote_code=False,device_map={"":0},low_cpu_mem_usage=True,output_loading_info=True,attn_implementation="eager",dtype=DTYPE)
    for k in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs"):
        assert not li.get(k,[]),f"WEIGHT LOAD FAILURE: {k}"
model.eval()
for p in model.parameters():p.requires_grad_(False)
base=model.model;layers=base.layers;device=model.get_input_embeddings().weight.device
assert len(layers)==27 and all(p.device.type=="cuda" and p.dtype==DTYPE for p in model.parameters() if p.is_floating_point())
tok=globals().get("tok",None)
if tok is None:tok=AutoTokenizer.from_pretrained(MODEL,revision=REV,trust_remote_code=False)
gates["MODEL_FROZEN"]=all(not p.requires_grad for p in model.parameters())
print("PARAMETERS:",sum(p.numel() for p in model.parameters()),"TRAINABLE:",sum(p.numel() for p in model.parameters() if p.requires_grad))
print("\n[2/9] PILOT FACT / QUESTION / CONTROLS")
FACT="The access code for the fictional station VELORA-731 is QN-4826."
QUESTION="What is the access code for the fictional station VELORA-731? Answer with the code only."
SOURCE=tok.apply_chat_template([{"role":"user","content":"Memorize this fictional record: "+FACT}],tokenize=False,add_generation_prompt=True)
ASK=tok.apply_chat_template([{"role":"user","content":QUESTION}],tokenize=False,add_generation_prompt=True)
print("FACT:",FACT)
print("QUESTION:",QUESTION)
print("SOURCE_TOKEN_COUNT:",token_ids(SOURCE).shape[1],"ASK_TOKEN_COUNT:",token_ids(ASK).shape[1])
print("\n[3/9] WRITE — NATIVE MLA CACHE + RESIDUAL CAPTURE")
write_ids=token_ids(SOURCE);write_h={};handles=[]
def mk(i):
    def hook(m,args,kwargs):
        x=kwargs.get("hidden_states",args[0] if args else None)
        if torch.is_tensor(x):write_h[i]=x[:,-1:,:].detach().clone()
    return hook
try:
    for i,l in enumerate(layers):handles.append(l.register_forward_pre_hook(mk(i),with_kwargs=True))
    out=run(write_ids)
finally:
    for h in handles:h.remove()
write_cache=out.past_key_values
del out
assert len(write_h)==27 and cache_len(write_cache)==write_ids.shape[1]
cartridge={"cache":cache_clone(write_cache),"residual":{i:x.clone() for i,x in write_h.items()},"source_sha256":hashlib.sha256(FACT.encode()).hexdigest()}
gates["WRITE"]=len(cartridge["residual"])==27 and cache_len(cartridge["cache"])==write_ids.shape[1]
print("WRITE_CACHE_LENGTH:",cache_len(cartridge["cache"]))
print("WRITE_CACHE_LAYER0:",tuple(cartridge["cache"].layers[0].keys.shape),tuple(cartridge["cache"].layers[0].values.shape))
print("WRITE_RESIDUALS:",len(cartridge["residual"]),"SOURCE_SHA256:",cartridge["source_sha256"][:16])
del write_cache,write_h
print("\n[4/9] INIT — NUMERICAL CACHE RECONSTRUCTION")
# Pilot INIT uses a cloned native MLA cache; this is not compressed H_cut reconstruction.
init_cache=cache_clone(cartridge["cache"])
assert cache_len(init_cache)==write_ids.shape[1]
gates["INIT"]=len(init_cache.layers)==27
print("INIT_CACHE_LENGTH:",cache_len(init_cache),"LAYERS:",len(init_cache.layers))
print("\n[5/9] ASK — SOURCE-FREE QUESTION WITH CARTRIDGE")
mem_answer=answer_from_cache(init_cache,ASK,max_new=24)
print("MEMORY_ANSWER:",repr(mem_answer))
print("\n[6/9] CONTROL — NO MEMORY")
# Use empty cache by running the question as a standalone prompt.
with torch.inference_mode():
    ids=token_ids(ASK)
    gen=model.generate(input_ids=ids,max_new_tokens=24,do_sample=False,pad_token_id=tok.eos_token_id)
    no_answer=tok.decode(gen[0,ids.shape[1]:],skip_special_tokens=True).strip()
print("NO_MEMORY_ANSWER:",repr(no_answer))
print("\n[7/9] CONTROL — NATIVE TEXT CONTEXT")
native=tok.apply_chat_template([{"role":"user","content":"Memorize this fictional record: "+FACT},{"role":"user","content":QUESTION}],tokenize=False,add_generation_prompt=True)
with torch.inference_mode():
    ids=token_ids(native)
    gen=model.generate(input_ids=ids,max_new_tokens=24,do_sample=False,pad_token_id=tok.eos_token_id)
    native_answer=tok.decode(gen[0,ids.shape[1]:],skip_special_tokens=True).strip()
print("NATIVE_TEXT_ANSWER:",repr(native_answer))
print("\n[8/9] EXACT ANSWER / CACHE AUDIT")
EXPECTED="QN-4826"
def exact(s):return s.strip().strip(" .,:;\"'`")==EXPECTED
gates["MEMORY_EXACT"]=exact(mem_answer)
gates["NO_MEMORY_NEGATIVE"]=not exact(no_answer)
gates["NATIVE_EXACT"]=exact(native_answer)
gates["SOURCE_FREE_ASK"]=FACT not in ASK and EXPECTED not in ASK
gates["CACHE_INTEGRITY"]=cache_len(cartridge["cache"])==write_ids.shape[1]
gates["HOOKS_CLEAN"]=all(len(l._forward_pre_hooks)==0 for l in layers)
print("MEMORY_EXACT:",gates["MEMORY_EXACT"],"NO_MEMORY_NEGATIVE:",gates["NO_MEMORY_NEGATIVE"],"NATIVE_EXACT:",gates["NATIVE_EXACT"])
print("CACHE_INTEGRITY:",gates["CACHE_INTEGRITY"],"HOOKS_CLEAN:",gates["HOOKS_CLEAN"])
print("\n[9/9] FINAL AUDIT")
for k,v in gates.items():print("GATE",k,"PASS" if v else "FAIL")
print("STATUS:","PASS — SINGLE CARTRIDGE SOURCE-FREE RECALL" if all(gates.values()) else "INCOMPLETE — INSPECT GATES")
print("MODEL_FROZEN:",all(not p.requires_grad for p in model.parameters()))
print("CARTRIDGE_KIND: NATIVE UNCOMPRESSED MLA CACHE + CAPTURED RESIDUAL")
print("CUT_SELECTED: NO")
print("APPEND_IMPLEMENTED: NO")
print("ELAPSED_SECONDS:",round(time.time()-t0,2))
print("="*126)


# TEST574 — AKBASCORE MAM · QWEN MECHANISM & FORWARD AUDIT · FIXED
# TEST573 MOTOR PRESERVED · CUT3 · MASK / H3 / KV / APPEND-ONLY / ASK AUDIT
# FROZEN QWEN · NO TRAINING / RETRIEVAL / ROUTER / SOURCE REPLAY AT ASK
import os,sys,subprocess,importlib.util,urllib.request,hashlib,json,time,gc,re,random
for m in ("torch","transformers","accelerate"):
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",m])
import torch,numpy as np,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
TEST="574";MODEL_ID="Qwen/Qwen2.5-7B-Instruct";CUT=3;DEVICE=torch.device("cuda");DTYPE=torch.bfloat16;SEED=552552;PANEL_SEED=550550;MAX_NEW=16
BASE_FILE="566.py";BASE_BLOB="b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71"
BASE_URL="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/"+BASE_FILE
EXPECTED_PANEL_SHA="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required")
os.environ["TOKENIZERS_PARALLELISM"]="false";random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.set_grad_enabled(False)
T0=time.perf_counter()
def sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("="*140);print("TEST574 — AKBASCORE MAM · QWEN MECHANISM & FORWARD AUDIT · FIXED");print("="*140)
req=urllib.request.Request(BASE_URL,headers={"User-Agent":"AKBASCORE-TEST574"})
with urllib.request.urlopen(req,timeout=90) as f:raw=f.read()
blob=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
if blob!=BASE_BLOB:raise RuntimeError("BASELINE BLOB MISMATCH: "+blob)
src=raw.decode("utf-8");marker="# TEST566 ADDITIONS — original TEST564 checkpoint/init/append/answer remain unchanged."
if src.count(marker)!=1:raise RuntimeError("BASELINE BOUNDARY MISMATCH")
exec(compile(src.split(marker,1)[0],BASE_FILE,"exec"),globals())
assert PANEL_SHA==EXPECTED_PANEL_SHA and (NL,H,QH,KVH,HD)==(28,3584,28,4,128)
assert sentinel()==S0 and not any(p.requires_grad for p in model.parameters())
print("BASE BLOB:",blob,"PANEL SHA:",PANEL_SHA,"CUT:",CUT)
print("GPU:",torch.cuda.get_device_name(0),"TORCH:",torch.__version__,"TRANSFORMERS:",transformers.__version__)
def full_weight_sha():
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());h.update(str(tuple(p.shape)).encode());h.update(str(p.dtype).encode())
        a=p.detach().contiguous().view(torch.uint8).reshape(-1)
        for part in a.split(16*1024*1024):h.update(part.cpu().numpy().tobytes())
    return h.hexdigest()
def make_case(keys):
    ids=list(PIDS);ranges={}
    for ci in keys:
        a=len(ids);ids.extend(CAR[ci]["body"]);ranges[ci]=(a,len(ids))
    n=len(ids);group=torch.full((n,),-1,device=DEVICE,dtype=torch.long)
    for j,ci in enumerate(keys):
        a,b=ranges[ci];group[a:b]=j
    ix=torch.arange(n,device=DEVICE);causal=ix[:,None]>=ix[None,:]
    isolated=causal&((group[:,None]==-1)|(group[None,:]==-1)|(group[:,None]==group[None,:]))
    jointmask=torch.zeros((1,1,n,n),device=DEVICE,dtype=DTYPE);indepmask=torch.zeros_like(jointmask);bad=torch.finfo(DTYPE).min
    jointmask.masked_fill_(~causal,bad);indepmask.masked_fill_(~isolated,bad)
    return torch.tensor([ids],device=DEVICE,dtype=torch.long),jointmask,indepmask,ranges
def expected_length(keys):return P+sum(len(CAR[i]["body"]) for i in keys)
def validate(mem,keys):
    n=expected_length(keys)
    if len(mem)!=NL:raise RuntimeError("LAYER COUNT MISMATCH")
    for li,(k,v) in enumerate(mem):
        shape=(1,KVH,n,HD)
        if tuple(k.shape)!=shape or tuple(v.shape)!=shape:raise RuntimeError(f"CACHE SHAPE L{li}: {tuple(k.shape)}/{tuple(v.shape)}")
        if not torch.isfinite(k).all() or not torch.isfinite(v).all():raise RuntimeError(f"NONFINITE CACHE L{li}")
    return n
def verify_append(old,new):
    n=old[0][0].shape[-2]
    if len(old)!=len(new):raise RuntimeError("APPEND LAYER COUNT MISMATCH")
    for li,((ok,ov),(nk,nv)) in enumerate(zip(old,new)):
        if not torch.equal(ok,nk[:,:,:n,:]) or not torch.equal(ov,nv[:,:,:n,:]):raise RuntimeError(f"APPEND-ONLY BITWISE FAIL L{li}")
    return True
AUDIT=[];COUNTERS=dict(mask_forward=0,mask_layers=0,attention_layers=0,inject_forward=0,inject_layers=0,ask_forward=0,ask_input_tokens=0,append_checks=0,early_kv_checks=0,upper_kv_checks=0)
def record(name,ok,**kw):
    r=dict(name=name,pass_=bool(ok),**kw);AUDIT.append(r)
    print(f"{'PASS' if ok else 'FAIL'} | {name} | {kw}",flush=True)
    if not ok:raise RuntimeError("AUDIT FAILURE: "+name)
print("[1/9] Full weight fingerprint...",flush=True)
WEIGHT_SHA0=full_weight_sha();print("WEIGHT SHA BEFORE:",WEIGHT_SHA0)
print("[2/9] Attention-mask geometry...",flush=True)
z=PANEL[0];keys=ordered(z,"MIDDLE");ids,jmask,imask,ranges=make_case(keys);n=ids.shape[1]
record("PANEL_AND_LENGTH",n==expected_length(keys),tokens=n,keys=keys)
record("MASK_SHAPES",tuple(jmask.shape)==tuple(imask.shape)==(1,1,n,n),shape=list(jmask.shape))
upper=torch.triu(torch.ones((n,n),device=DEVICE,dtype=torch.bool),diagonal=1)
lower=~upper
record("JOINT_CAUSAL",bool(torch.all(jmask[0,0][upper]<-1e20).item()) and bool(torch.all(jmask[0,0][lower]==0).item()))
record("INDEP_DIFFERS",bool(torch.any(jmask!=imask).item()))
record("INDEP_PREFIX_ACCESS",bool(torch.equal(imask[0,0,:,0:P],jmask[0,0,:,0:P])))
for i,ci in enumerate(keys):
    a,b=ranges[ci];q=b-a
    u=torch.triu(torch.ones((q,q),device=DEVICE,dtype=torch.bool),diagonal=1)
    block=imask[0,0,a:b,a:b]
    record(f"INDEP_SELF_{i}",bool(torch.all(block[u]<-1e20).item()) and bool(torch.all(block[~u]==0).item()),cartridge=ci)
    for j,cj in enumerate(keys):
        if i==j:continue
        c,d=ranges[cj]
        record(f"INDEP_CROSS_BLOCK_{i}_{j}",bool(torch.all(imask[0,0,a:b,c:d]<-1e20).item()),from_cartridge=ci,to_cartridge=cj)
print("[3/9] Mask propagation to transformer and attention modules...",flush=True)
@torch.no_grad()
def audited_reference(ids,mask,tag):
    seen=[];attn_seen=[];handles=[]
    try:
        for li,layer in enumerate(model.model.layers):
            def pre(module,args,kwargs,li=li,mask=mask):
                kwargs["attention_mask"]=mask
                seen.append((li,kwargs["attention_mask"] is mask,tuple(kwargs["attention_mask"].shape)))
                return args,kwargs
            def attn_pre(module,args,kwargs,li=li,mask=mask):
                supplied=kwargs.get("attention_mask",None)
                attn_seen.append((li,supplied is mask,tuple(supplied.shape) if supplied is not None else None))
            handles.append(layer.register_forward_pre_hook(pre,with_kwargs=True))
            handles.append(layer.self_attn.register_forward_pre_hook(attn_pre,with_kwargs=True))
        length=ids.shape[1];pos=torch.arange(length,device=DEVICE)
        out=model(input_ids=ids,attention_mask=torch.ones((1,length),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        mem=clone(out.past_key_values)
        ok=len(seen)==NL and {x[0] for x in seen}==set(range(NL)) and all(x[1] and x[2]==tuple(mask.shape) for x in seen)
        attn_ok=len(attn_seen)==NL and {x[0] for x in attn_seen}==set(range(NL)) and all(x[1] and x[2]==tuple(mask.shape) for x in attn_seen)
        COUNTERS["mask_forward"]+=1;COUNTERS["mask_layers"]+=len(seen);COUNTERS["attention_layers"]+=len(attn_seen)
        record(tag+"_LAYER_MASK",ok,observed_layers=len(seen),expected_layers=NL)
        record(tag+"_ATTENTION_MASK",attn_ok,observed_layers=len(attn_seen),expected_layers=NL)
        return mem
    finally:
        for handle in handles:handle.remove()
joint=audited_reference(ids,jmask,"JOINT");indep=audited_reference(ids,imask,"INDEP")
validate(joint,keys);validate(indep,keys)
print("[4/9] H3 checkpoint and injection...",flush=True)
@torch.no_grad()
def audited_init(ci):
    cp=checkpoint(ci,CUT);length=cp["h"].shape[1];seen=[]
    record("H3_CHECKPOINT",tuple(cp["h"].shape)==(1,length,H) and len(cp["kv"])==NL,cartridge=ci,shape=list(cp["h"].shape))
    def inject(module,args,out):
        current=out[0] if isinstance(out,(tuple,list)) else out
        ok=current.shape==cp["h"].shape
        seen.append((ok,float((current.float()-cp["h"].float()).abs().max().item())))
        return replace(out,cp["h"])
    handle=model.model.layers[CUT].register_forward_hook(inject)
    try:
        pos=torch.arange(length,device=DEVICE)
        out=model(inputs_embeds=torch.zeros((1,length,H),device=DEVICE,dtype=DTYPE),attention_mask=torch.ones((1,length),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        upper=clone(out.past_key_values);mem=cp["kv"][:CUT+1]+upper[CUT+1:]
    finally:handle.remove()
    COUNTERS["inject_forward"]+=1;COUNTERS["inject_layers"]+=len(seen)
    record("H3_INJECTION_FORWARD",len(seen)==1 and seen[0][0],cut=CUT,observed=len(seen),pre_injection_max_abs=seen[0][1] if seen else None)
    for li in range(CUT+1):
        k,v=mem[li];ck,cv=cp["kv"][li]
        COUNTERS["early_kv_checks"]+=1
        record(f"INIT_EARLY_KV_L{li}",torch.equal(k,ck) and torch.equal(v,cv))
    return mem
memory=audited_init(keys[0])
print("[5/9] Incremental append and immutable historical KV...",flush=True)
for step,ci in enumerate(keys[1:],1):
    old=memory;oldn=old[0][0].shape[-2];memory=append(old,ci,CUT);verify_append(old,memory)
    COUNTERS["append_checks"]+=1
    record(f"APPEND_{step}_BITWISE",True,cartridge=ci,old_tokens=oldn,new_tokens=memory[0][0].shape[-2])
    cp=checkpoint(ci,CUT);q=cp["body"];pos=torch.arange(oldn,oldn+q,device=DEVICE)
    for li in range(CUT+1):
        k,v=cp["kv"][li];ek=rephase(k[:,:,P:,:],cp["pos"][P:],pos);ev=v[:,:,P:,:];mk,mv=memory[li]
        COUNTERS["early_kv_checks"]+=1
        record(f"APPEND_{step}_EARLY_KV_L{li}",torch.equal(mk[:,:,-q:,:],ek) and torch.equal(mv[:,:,-q:,:],ev))
    for li in range(CUT+1,NL):
        ok=memory[li][0].shape[-2]==oldn+q and memory[li][1].shape[-2]==oldn+q
        COUNTERS["upper_kv_checks"]+=1
        if not ok:raise RuntimeError(f"UPPER KV LENGTH L{li}")
    record(f"APPEND_{step}_UPPER_KV_LENGTHS",True,layers=NL-CUT-1)
validate(memory,keys)
print("[6/9] Batch DC3 assembly and upper-layer propagation...",flush=True)
@torch.no_grad()
def audited_batch(keys,ranges,jointmask):
    cps={ci:checkpoint(ci,CUT) for ci in keys};n=expected_length(keys)
    assembled=torch.cat([cps[keys[0]]["h"][:,:P,:]]+[cps[ci]["h"][:,P:,:] for ci in keys],dim=1)
    record("BATCH_H3_ASSEMBLY",tuple(assembled.shape)==(1,n,H),shape=list(assembled.shape))
    early=[]
    for li in range(CUT+1):
        ks=[cps[keys[0]]["kv"][li][0][:,:,:P,:]];vs=[cps[keys[0]]["kv"][li][1][:,:,:P,:]]
        for ci in keys:
            a,b=ranges[ci];cp=cps[ci];k,v=cp["kv"][li]
            ks.append(rephase(k[:,:,P:,:],cp["pos"][P:],torch.arange(a,b,device=DEVICE)));vs.append(v[:,:,P:,:])
        early.append((torch.cat(ks,-2),torch.cat(vs,-2)))
    seen=[];attn_seen=[];handles=[]
    try:
        def inject(module,args,out):
            current=out[0] if isinstance(out,(tuple,list)) else out
            seen.append(("H",current.shape==assembled.shape))
            return replace(out,assembled)
        handles.append(model.model.layers[CUT].register_forward_hook(inject))
        for li in range(CUT+1,NL):
            def pre(module,args,kwargs,li=li):
                kwargs["attention_mask"]=jointmask
                seen.append((li,kwargs["attention_mask"] is jointmask))
                return args,kwargs
            def attn_pre(module,args,kwargs,li=li):
                attn_seen.append((li,kwargs.get("attention_mask",None) is jointmask))
            handles.append(model.model.layers[li].register_forward_pre_hook(pre,with_kwargs=True))
            handles.append(model.model.layers[li].self_attn.register_forward_pre_hook(attn_pre,with_kwargs=True))
        pos=torch.arange(n,device=DEVICE)
        out=model(inputs_embeds=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        upper=clone(out.past_key_values);mem=tuple(early)+upper[CUT+1:]
    finally:
        for handle in handles:handle.remove()
    hseen=[x for x in seen if x[0]=="H"];useen=[x for x in seen if x[0]!="H"]
    record("BATCH_H3_INJECTION",len(hseen)==1 and hseen[0][1],count=len(hseen))
    record("BATCH_UPPER_MASK_PROPAGATION",len(useen)==NL-CUT-1 and all(x[1] for x in useen),layers=len(useen))
    record("BATCH_ATTENTION_MASK_PROPAGATION",len(attn_seen)==NL-CUT-1 and all(x[1] for x in attn_seen),layers=len(attn_seen))
    for li in range(CUT+1):
        COUNTERS["early_kv_checks"]+=1
        record(f"BATCH_EARLY_KV_L{li}",torch.equal(mem[li][0],early[li][0]) and torch.equal(mem[li][1],early[li][1]))
    return mem
batch=audited_batch(keys,ranges,jmask);validate(batch,keys)
print("[7/9] Source-free ASK forward audit...",flush=True)
question=W[0][CAR[z["target"]]["type"]];gold=CAR[z["target"]]["gold"]
ask_calls=[];ask_ids=[];handle=None
def ask_pre(module,args,kwargs):
    token_ids=kwargs.get("input_ids",None);embeds=kwargs.get("inputs_embeds",None);cache=kwargs.get("past_key_values",None)
    if token_ids is not None:ask_ids.extend(token_ids.detach().reshape(-1).cpu().tolist())
    ask_calls.append(dict(input_tokens=int(token_ids.shape[-1]) if token_ids is not None else 0,embeds=embeds is not None,cache_present=cache is not None,cache_tokens=int(cache.get_seq_length()) if cache is not None else 0))
    return args,kwargs
handle=model.register_forward_pre_hook(ask_pre,with_kwargs=True)
try:
    a_joint=answer(joint,question);a_indep=answer(indep,question);a_batch=answer(batch,question);a_incr=answer(memory,question)
finally:handle.remove()
COUNTERS["ask_forward"]=len(ask_calls);COUNTERS["ask_input_tokens"]=sum(x["input_tokens"] for x in ask_calls)
record("ASK_ALL_HAVE_CACHE",len(ask_calls)>0 and all(x["cache_present"] and x["cache_tokens"]>0 for x in ask_calls),forward_calls=len(ask_calls))
record("ASK_NO_EMBEDDED_SOURCE",all(not x["embeds"] for x in ask_calls),forward_calls=len(ask_calls))
record("ASK_TOKEN_LENGTHS",all(0<x["input_tokens"]<expected_length(keys) for x in ask_calls),max_input=max(x["input_tokens"] for x in ask_calls))
expected_suffix=tok(suffix(question),add_special_tokens=False).input_ids
record("ASK_FIRST_QUERY_TOKENS",len(ask_ids)>=len(expected_suffix) and ask_ids[:len(expected_suffix)]==expected_suffix,expected_query_tokens=len(expected_suffix))
record("ASK_NO_SOURCE_REPLAY",not any(CAR[ci]["body"]==ask_ids[j:j+len(CAR[ci]["body"])] for ci in keys for j in range(max(0,len(ask_ids)-len(CAR[ci]["body"])+1))),total_ask_tokens=len(ask_ids))
record("ASK_DC_CORRECT",bool(hit(a_batch,gold) and hit(a_incr,gold)),gold=gold,batch=a_batch,incremental=a_incr,joint=a_joint,indep=a_indep)
print("[8/9] Cross-arm numerical diagnostics...",flush=True)
def kv_comparison(a,b):
    diffs=[]
    for li,((ak,av),(bk,bv)) in enumerate(zip(a,b)):
        dk=(ak.float()-bk.float()).abs();dv=(av.float()-bv.float()).abs()
        diffs.append(dict(layer=li,k_max=float(dk.max().item()),v_max=float(dv.max().item()),k_equal=bool(torch.equal(ak,bk)),v_equal=bool(torch.equal(av,bv))))
    return diffs
NUMERICAL=dict(batch_vs_incremental=kv_comparison(batch,memory),joint_vs_batch=kv_comparison(joint,batch))
record("BATCH_INCREMENTAL_KV_SHAPES",len(NUMERICAL["batch_vs_incremental"])==NL and all(x["layer"]==i for i,x in enumerate(NUMERICAL["batch_vs_incremental"])),layers=NL)
print("BATCH/INCR EXACT LAYERS:",sum(x["k_equal"] and x["v_equal"] for x in NUMERICAL["batch_vs_incremental"]),"/",NL)
print("JOINT/BATCH EXACT LAYERS:",sum(x["k_equal"] and x["v_equal"] for x in NUMERICAL["joint_vs_batch"]),"/",NL)
print("[9/9] Final integrity and record...",flush=True)
WEIGHT_SHA1=full_weight_sha();WEIGHT_OK=WEIGHT_SHA0==WEIGHT_SHA1
SENTINEL_OK=sentinel()==S0;TRAINABLE=sum(int(p.requires_grad) for p in model.parameters())
record("FULL_WEIGHT_SHA_UNCHANGED",WEIGHT_OK,before=WEIGHT_SHA0,after=WEIGHT_SHA1)
record("SENTINEL_UNCHANGED",SENTINEL_OK)
record("ZERO_TRAINABLE",TRAINABLE==0,count=TRAINABLE)
record("AUDIT_COMPLETENESS",COUNTERS["mask_layers"]==2*NL and COUNTERS["attention_layers"]==2*NL and COUNTERS["append_checks"]==4 and COUNTERS["inject_layers"]==1 and len(AUDIT)>40,counters=COUNTERS)
PROTOCOL=dict(test=TEST,model=MODEL_ID,cut=CUT,base_file=BASE_FILE,base_blob=blob,panel_sha=PANEL_SHA,seed=SEED,panel_seed=PANEL_SEED,architecture=[NL,H,QH,KVH,HD],audit="mask geometry, layer/attention interception, H3 injection, early KV rephase, upper KV growth, append-only bitwise, ASK token audit, full weight SHA",training=False,retrieval=False,router=False)
RECORD=dict(protocol=PROTOCOL,lock_sha=sha(PROTOCOL),audit=AUDIT,counters=COUNTERS,numerical=NUMERICAL,answers=dict(joint=a_joint,indep=a_indep,batch=a_batch,incremental=a_incr,gold=gold),ask_calls=ask_calls,weight_sha_before=WEIGHT_SHA0,weight_sha_after=WEIGHT_SHA1,gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,transformers_version=transformers.__version__,seconds=round(time.perf_counter()-T0,2))
RECORD["result_sha"]=sha(RECORD)
OUT="/content/AKBASCORE_TEST574_QWEN_MECHANISM_AUDIT.json"
with open(OUT,"w",encoding="utf-8") as f:json.dump(RECORD,f,indent=2,ensure_ascii=False)
print("="*140);print("TEST574 — FINAL MECHANISM AUDIT");print("="*140)
print("MODEL:",MODEL_ID,"CUT:",CUT,"PANEL SHA:",PANEL_SHA)
print("AUDITS:",len(AUDIT),"PASS:",sum(x["pass_"] for x in AUDIT),"FAIL:",sum(not x["pass_"] for x in AUDIT))
print("FORWARD COUNTERS:",COUNTERS)
print("FULL WEIGHT SHA:",WEIGHT_SHA1,"PASS" if WEIGHT_OK else "FAIL")
print("LOCK SHA:",RECORD["lock_sha"]);print("RESULT SHA:",RECORD["result_sha"])
print("OUTPUT:",OUT,"SECONDS:",RECORD["seconds"])
print("NOTE: ASK audit covers instrumented forward arguments, not every possible external information channel.")
print("NOTE: KV numerical differences are diagnostics, not automatic failure criteria.")
print("="*140)

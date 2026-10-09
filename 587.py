
# TEST587 — AKBASCORE MAM · DEEPSEEK MLA AUTONOMOUS READOUT
# TEST586 ENGINE FROZEN: WRITE / INIT / CUT3 APPEND / MLA ROTARY
# AUTONOMOUS CANDIDATE GENERATION + MODEL-ONLY LIKELIHOOD RERANK
# NO GOLD CANDIDATE INJECTION · NO GOLD FIRST-TOKEN FORCING · NO TRAINING
# BOTH CARTRIDGE ORDERS · JOINT / FIRST / APPEND CONTROLS
# SINGLE GOOGLE COLAB CELL · BF16 · A100-40GB
import os,time,gc,re,hashlib,math
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch,transformers
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
TEST="587";MODEL="deepseek-ai/DeepSeek-V2-Lite-Chat";REV="85864749cd611b4353ce1decdb286193298f64c7"
DTYPE=torch.bfloat16;SEED=577;CUT=3;NL=27;H=2048;MAX_NEW=24;TOPK=8;BEAM_WIDTH=4;BEAM_STEPS=20
torch.manual_seed(SEED);torch.set_grad_enabled(False);T0=time.time()
GATES={};ROWS=[];CANDIDATE_AUDIT=[];HOOK_COUNT={"init":0,"append":0}
print("="*152)
print("TEST587 — AKBASCORE MAM · DEEPSEEK MLA AUTONOMOUS READOUT")
print("="*152)
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(),"CUDA BF16 REQUIRED"
print("GPU:",torch.cuda.get_device_name(0),"TORCH:",torch.__version__,"TRANSFORMERS:",transformers.__version__)
def gate(name,condition):
    GATES[name]=bool(condition);print("GATE",name,"PASS" if condition else "FAIL",flush=True)
def replace_hidden(out,new):
    if isinstance(out,tuple):return (new,)+out[1:]
    if isinstance(out,list):return [new]+out[1:]
    return new
print("\n[1/13] MODEL / TEST586 REUSE")
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
print("\n[2/13] TEST586 CANONICAL PANEL")
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
print("\n[3/13] MLA ROTARY TRANSPORT — UNCHANGED")
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
print("\n[4/13] WRITE — INDEPENDENT NUMERICAL CHECKPOINTS")
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
print("WRITE_LENGTHS:",[length(cp["native"]) for cp in CP])
print("\n[5/13] INIT / APPEND — TEST586 ENGINE UNCHANGED")
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
print("\n[6/13] JOINT REFERENCE")
def joint_reference(order):
    seq=PREFIX_IDS
    for ci in order:seq=seq+CAR[ci]["body"]
    x=torch.tensor([seq],device=DEVICE,dtype=torch.long);n=x.shape[1];pos=torch.arange(n,device=DEVICE)
    o=run(input_ids=x,attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos)
    result=clone_layers(o.past_key_values);validate(result,n);return result
for order in ORDERS:MEM[order]["joint"]=joint_reference(order)
print("\n[7/13] FROZEN GREEDY BASELINE")
def code_contains(answer,code):
    return re.search(r"(?<![A-Za-z0-9-])"+re.escape(code)+r"(?![A-Za-z0-9-])",answer) is not None
def prefix_forward(memory,qtext):
    cache=make_cache(memory);n=length(memory);qids=ids(qtext);ql=qids.shape[1];pos=torch.arange(n,n+ql,device=DEVICE)
    o=run(input_ids=qids,past=cache,attention_mask=torch.ones((1,n+ql),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos)
    return clone_layers(o.past_key_values),o.logits[:,-1,:].float()
def step_forward(cache_data,token):
    n=length(cache_data);p=torch.tensor([n],device=DEVICE)
    o=run(input_ids=torch.tensor([[int(token)]],device=DEVICE),past=make_cache(cache_data),attention_mask=torch.ones((1,n+1),device=DEVICE,dtype=torch.long),position_ids=p.unsqueeze(0),cache_position=p)
    return clone_layers(o.past_key_values),o.logits[:,-1,:].float()
def greedy_from_prefix(cache_data,logits,max_new=MAX_NEW):
    generated=[];cache=cache_data
    for _ in range(max_new):
        t=int(logits.argmax(-1).item())
        if t==tok.eos_token_id:break
        generated.append(t)
        cache,logits=step_forward(cache,t)
    return tok.decode(generated,skip_special_tokens=True).strip(),generated
print("\n[8/13] AUTONOMOUS BEAM CANDIDATES — NO GOLD INJECTION")
def beam_candidates(cache_data,logits,width=BEAM_WIDTH,steps=BEAM_STEPS):
    # Beam search is model-generated only. No gold code is supplied to candidate generation.
    # Raw log-probability ranking; length normalization is reported separately.
    beams=[{"tokens":[],"logp":0.0,"cache":cache_data,"logits":logits,"done":False}]
    finished=[]
    for step in range(steps):
        expanded=[]
        for b in beams:
            if b["done"]:
                expanded.append(b);continue
            lp=torch.log_softmax(b["logits"],dim=-1)
            vals,idx=torch.topk(lp,width,dim=-1)
            for v,t in zip(vals[0].tolist(),idx[0].tolist()):
                nt=b["tokens"]+[int(t)];score=b["logp"]+float(v)
                if int(t)==tok.eos_token_id:
                    expanded.append({"tokens":nt,"logp":score,"cache":b["cache"],"logits":b["logits"],"done":True})
                else:
                    expanded.append({"tokens":nt,"logp":score,"cache":b["cache"],"logits":b["logits"],"done":False})
        expanded.sort(key=lambda x:x["logp"],reverse=True)
        selected=expanded[:width];new=[]
        for b in selected:
            if b["done"]:
                new.append(b)
            else:
                c,l=step_forward(b["cache"],b["tokens"][-1])
                b["cache"]=c;b["logits"]=l;new.append(b)
        beams=new
        if all(b["done"] for b in beams):break
    candidates=[]
    seen=set()
    for b in beams:
        raw=tok.decode([t for t in b["tokens"] if t!=tok.eos_token_id],skip_special_tokens=True).strip()
        if raw in seen:continue
        seen.add(raw)
        candidates.append({"text":raw,"tokens":b["tokens"],"logp":b["logp"],"length":len(b["tokens"]),"avg_logp":b["logp"]/max(1,len(b["tokens"])),"ended":b["done"]})
    return candidates
def extract_candidate_code(text):
    # Generic unsupervised syntax extraction; never consults FACTS or gold labels.
    # Accepts uppercase alphanumeric codes with a hyphen and digits.
    hits=re.findall(r"\b[A-Z]{2,8}-\d{2,8}\b",text)
    return hits[-1] if hits else None
def autonomous_select(candidates):
    # Rerank only candidates actually generated by the model.
    # The code extraction is syntactic, not gold-based.
    pool=[]
    for c in candidates:
        code=extract_candidate_code(c["text"])
        if code is not None:
            pool.append((c["logp"],code,c["text"]))
    if not pool:return None,None
    pool.sort(key=lambda x:x[0],reverse=True)
    return pool[0][1],pool[0][2]
print("BEAM_WIDTH:",BEAM_WIDTH,"BEAM_STEPS:",BEAM_STEPS,"NO GOLD CANDIDATE INSERTION")
print("\n[9/13] ALL ARMS / BOTH ORDERS")
for order in ORDERS:
    for arm in ("joint","first","append"):
        for station,gold in FACTS:
            cache,logits=prefix_forward(MEM[order][arm],question(station))
            greedy,greedy_ids=greedy_from_prefix(cache,logits)
            candidates=beam_candidates(cache,logits)
            selected,source=autonomous_select(candidates)
            greedy_exact=greedy.strip().strip(" .,:;\"'`")==gold
            auto_exact=selected==gold
            wrong=any(code_contains(greedy,c) for _,c in FACTS if c!=gold)
            row={"order":order,"arm":arm,"station":station,"gold":gold,"greedy":greedy,"greedy_exact":greedy_exact,"selected":selected,"selected_source":source,"auto_exact":auto_exact,"wrong_greedy":wrong,"candidates":candidates}
            ROWS.append(row);CANDIDATE_AUDIT.append({"order":order,"arm":arm,"station":station,"candidates":candidates})
            print("ORDER",order,"ARM",arm,"STATION",station,"GOLD",gold)
            print(" GREEDY:",repr(greedy),"EXACT:",greedy_exact)
            for j,c in enumerate(candidates):
                print(" CAND",j,"TEXT",repr(c["text"]),"LOGP",round(c["logp"],5),"AVG",round(c["avg_logp"],5),"ENDED",c["ended"],"EXTRACTED",extract_candidate_code(c["text"]))
            print(" AUTONOMOUS_SELECTED:",repr(selected),"SOURCE:",repr(source),"EXACT:",auto_exact)
print("\n[10/13] SCORES / CONTROLS")
def count(arm,key):return sum(int(r[key]) for r in ROWS if r["arm"]==arm)
for arm in ("joint","first","append"):
    rr=[r for r in ROWS if r["arm"]==arm]
    print("ARM",arm,"GREEDY_EXACT",count(arm,"greedy_exact"),"/4","AUTONOMOUS_EXACT",count(arm,"auto_exact"),"/4","GREEDY_WRONG",count(arm,"wrong_greedy"),"/4","NO_CANDIDATE",sum(r["selected"] is None for r in rr))
gate("APPEND_AUTONOMOUS_EXACT_4",count("append","auto_exact")==4)
gate("APPEND_GREEDY_EXACT_4",count("append","greedy_exact")==4)
gate("APPEND_NO_EMPTY_CANDIDATES",all(r["selected"] is not None for r in ROWS if r["arm"]=="append"))
gate("APPEND_BEATS_GREEDY",count("append","auto_exact")>count("append","greedy_exact"))
print("\n[11/13] FAILURE ANALYSIS")
for r in ROWS:
    if r["arm"]!="append":continue
    if r["auto_exact"]:status="AUTONOMOUS_EXACT"
    elif r["selected"] is None:status="NO_EXTRACTABLE_CODE"
    else:status="AUTONOMOUS_WRONG_CODE"
    print("ORDER",r["order"],"STATION",r["station"],"STATUS",status,"GREEDY",repr(r["greedy"]),"SELECTED",repr(r["selected"]))
print("NOTE: Gold codes are used only for scoring after candidate generation and selection.")
print("NOTE: Beam candidates are generated entirely from the model distribution.")
print("NOTE: This is a readout diagnostic, not a modification of the MLA cartridge engine.")
print("\n[12/13] INTEGRITY AUDIT")
gate("CACHE_VALID",all(validate(m,length(m)) for d in MEM.values() for m in d.values()))
gate("HOOK_COUNTS",HOOK_COUNT=={"init":2,"append":2})
gate("HOOKS_CLEAN",all(not l._forward_hooks and not l._forward_pre_hooks for l in LAYERS))
gate("FINGERPRINT_UNCHANGED",fingerprint()==FP0)
gate("WEIGHTS_FROZEN",all(not p.requires_grad for p in model.parameters()))
print("CUDA_ALLOCATED_GIB:",round(torch.cuda.memory_allocated()/2**30,3))
print("CUDA_PEAK_GIB:",round(torch.cuda.max_memory_allocated()/2**30,3))
print("\n[13/13] FINAL")
for k,v in GATES.items():print("GATE",k,"PASS" if v else "FAIL")
infra=["MODEL_FROZEN","PANEL_LOCK","WRITE_INDEPENDENT","INIT_EXACT_BOTH","APPEND_PRESERVED_BOTH","CACHE_VALID","HOOK_COUNTS","HOOKS_CLEAN","FINGERPRINT_UNCHANGED","WEIGHTS_FROZEN"]
infra_ok=all(GATES.get(k,False) for k in infra)
strict=GATES["APPEND_AUTONOMOUS_EXACT_4"] and GATES["APPEND_NO_EMPTY_CANDIDATES"]
print("INFRASTRUCTURE_STATUS:","PASS" if infra_ok else "FAIL")
print("AUTONOMOUS_READOUT_STATUS:","PASS" if strict else "FAIL")
print("STATUS:","PASS — AUTONOMOUS TWO-CARTRIDGE READOUT" if infra_ok and strict else "DIAGNOSTIC COMPLETE — AUTONOMOUS READOUT NOT YET VERIFIED")
print("ENGINE: TEST586 CUT3 WRITE/INIT/APPEND UNCHANGED")
print("GOLD_CANDIDATE_INJECTION: NO")
print("FORCED_FIRST_TOKEN: NO")
print("SOURCE_REPLAY_DURING_APPEND_OR_ASK: NO")
print("COMPRESSION: NOT YET")
print("ELAPSED_SECONDS:",round(time.time()-T0,2))
print("="*152)

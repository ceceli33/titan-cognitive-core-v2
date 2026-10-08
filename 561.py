# TEST561 — AKBASCORE MAM · FINAL ENGINE VALIDATION
# TEST560 WORKING BACKBONE PRESERVED
# LONG APPEND CHAIN · EARLY/MIDDLE/LATE RETENTION · INCREMENTAL DC6
# FROZEN MISTRAL · SOURCE-FREE MEMORY GROWTH · NO TRAINING / RETRIEVAL / ROUTER / QSF
import os,sys,subprocess,importlib.util,random,time,hashlib,json,re,gc
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
TEST="561";SEED=561561;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";DEVICE=torch.device("cuda");DTYPE=torch.bfloat16;MAX_NEW=16;CUT=6
BANK_SIZES=[5,10,20,40,80];FINAL_N=80;PROBES_PER_STAGE=12
if not torch.cuda.is_available():raise RuntimeError("CUDA gerekli.")
os.environ["TOKENIZERS_PARALLELISM"]="false";random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);T0=time.perf_counter()
def sha_obj(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def norm(s):return " ".join(re.sub(r"[^\w\s-]"," ",s.casefold().strip(),flags=re.UNICODE).split())
def hit(s,g):return re.search(r"(?<!\w)"+re.escape(norm(g))+r"(?!\w)",norm(s)) is not None
def section(n,t):print(f"\n[{n}/10] {t}",flush=True)
print("="*158);print("TEST561 — AKBASCORE MAM · FINAL ENGINE VALIDATION");print("TEST560 BACKBONE PRESERVED · LONG APPEND CHAIN · EARLY/MIDDLE/LATE RETENTION");print("FROZEN MISTRAL · SOURCE-FREE INCREMENTAL DC6 · NO TRAINING / RETRIEVAL / ROUTER / QSF");print("="*158)
section(1,"Frozen Mistral...")
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",dtype=DTYPE).eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None) or H//QH
if (NL,H,QH,KVH,HD)!=(32,4096,32,8,128):raise RuntimeError(f"Architecture mismatch: {(NL,H,QH,KVH,HD)}")
print("MODEL:",MODEL_ID);print("GPU:",torch.cuda.get_device_name(0));print("TORCH:",torch.__version__,"TRANSFORMERS:",transformers.__version__);print(f"NL={NL} H={H} QH={QH} KVH={KVH} HD={HD} DTYPE={DTYPE} CUT={CUT}");print("ROPE:",getattr(cfg,"rope_parameters",None));print("TRAINABLE:",sum(int(p.requires_grad) for p in model.parameters()))
FP=[model.model.layers[0].self_attn.q_proj.weight,model.model.layers[8].self_attn.o_proj.weight,model.model.layers[16].mlp.down_proj.weight,model.model.layers[24].self_attn.o_proj.weight,model.model.layers[31].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
def sentinel():
    h=hashlib.sha256()
    for p in FP:
        a=p.detach().reshape(-1);n=a.numel()
        for j in range(16):
            o=j*(n-256)//15;h.update(a[o:o+256].float().cpu().numpy().tobytes())
    return h.hexdigest()
S0=sentinel()
SYSTEM="Answer the question using only the information provided. Give only the requested name and nothing else."
def source_prefix(f):return f"<s>[INST] {SYSTEM}\n\nINFORMATION:\n{f}\n\n"
def canonical_prefix():return f"<s>[INST] {SYSTEM}\n\nINFORMATION:\n"
def query_suffix(q):return f"QUESTION:\n{q}\n\nANSWER: [/INST]"
PREFIX_IDS=tok(canonical_prefix(),add_special_tokens=False).input_ids;P=len(PREFIX_IDS)
# Fixed synthetic bank: unique subjects/answers, identical fact family, deterministic provenance.
SYL1=["Zor","Kel","Tar","Bel","Rav","Nem","Dar","Fer","Lar","Tor","Sel","Mir","Var","Kor","Nal","Der","Cal","Par","Sor","Bar"]
SYL2=["van","vek","nor","ith","en","or","al","ik","eth","ar"]
def name(i,shift=0):
    a=SYL1[(i+shift*7)%len(SYL1)];b=SYL2[((i//len(SYL1))+i*3+shift*5)%len(SYL2)]
    return a+b+str(1000+i+shift*FINAL_N)
CAR=[]
used=set()
for i in range(FINAL_N):
    s=name(i,0);g=name(i,1)
    if s in used or g in used or s==g:raise RuntimeError("Synthetic name collision.")
    used.add(s);used.add(g)
    fact=f"The current capital of {s} is {g}.";q=f"What is the current capital of {s}?"
    ids=tok(source_prefix(fact),add_special_tokens=False).input_ids
    if ids[:P]!=PREFIX_IDS:raise RuntimeError(f"Canonical prefix mismatch {i}")
    CAR.append({"idx":i,"subject":s,"gold":g,"fact":fact,"query":q,"body":ids[P:]})
BANK_SHA=sha_obj([{"idx":x["idx"],"subject":x["subject"],"gold":x["gold"],"fact":x["fact"],"query":x["query"],"body":x["body"]} for x in CAR])
print("BANK SHA:",BANK_SHA);print("BANK RECORDS:",len(CAR))
def cache_layers(pkv):
    if hasattr(pkv,"layers"):
        out=[]
        for layer in pkv.layers:
            k=getattr(layer,"keys",None);v=getattr(layer,"values",None)
            if k is None:k=getattr(layer,"key_cache",None)
            if v is None:v=getattr(layer,"value_cache",None)
            if k is None or v is None:raise RuntimeError("Cache API mismatch.")
            out.append((k,v))
        if out:return tuple(out)
    if hasattr(pkv,"key_cache"):return tuple(zip(pkv.key_cache,pkv.value_cache))
    raise RuntimeError("Cache API mismatch.")
def clone_layers(pkv):return tuple((k.detach().clone(),v.detach().clone()) for k,v in cache_layers(pkv))
def build_cache(layers):
    c=DynamicCache()
    for li,(k,v) in enumerate(layers):c.update(k,v,li)
    return c
def cache_len(c):return int(c.get_seq_length())
def replace_hidden(out,new):
    if isinstance(out,tuple):return (new,)+out[1:]
    if isinstance(out,list):return [new]+out[1:]
    return new
@torch.no_grad()
def answer_layers(layers,q):
    cache=build_cache(layers);physical=cache_len(cache);qids=tok(query_suffix(q),add_special_tokens=False).input_ids;ids=torch.tensor(qids,dtype=torch.long,device=DEVICE).reshape(1,-1);qlen=ids.shape[1]
    pos=torch.arange(physical,physical+qlen,dtype=torch.long,device=DEVICE).unsqueeze(0);att=torch.ones((1,physical+qlen),dtype=torch.long,device=DEVICE)
    out=model(input_ids=ids,past_key_values=cache,attention_mask=att,position_ids=pos,cache_position=pos.squeeze(0),use_cache=True,return_dict=True);cache=out.past_key_values;nxt=out.logits[:,-1,:].argmax(-1,keepdim=True);eos=set() if tok.eos_token_id is None else {int(tok.eos_token_id)};gen=[]
    for _ in range(MAX_NEW):
        gen.append(nxt)
        if int(nxt.item()) in eos:break
        physical=cache_len(cache);pos=torch.tensor([[physical]],dtype=torch.long,device=DEVICE);att=torch.ones((1,physical+1),dtype=torch.long,device=DEVICE)
        out=model(input_ids=nxt,past_key_values=cache,attention_mask=att,position_ids=pos,cache_position=pos.squeeze(0),use_cache=True,return_dict=True);cache=out.past_key_values;nxt=out.logits[:,-1,:].argmax(-1,keepdim=True)
    return tok.decode(torch.cat(gen,dim=1)[0],skip_special_tokens=True).strip()
def rotate_half(x):
    a,b=x.chunk(2,dim=-1);return torch.cat((-b,a),dim=-1)
@torch.no_grad()
def rope_tables(pos):
    dummy=torch.zeros((1,pos.numel(),H),dtype=DTYPE,device=DEVICE);cos,sin=model.model.rotary_emb(dummy,pos.reshape(1,-1))
    return cos.unsqueeze(1).float(),sin.unsqueeze(1).float()
@torch.no_grad()
def rephase_key(k,oldpos,newpos):
    if torch.equal(oldpos,newpos):return k
    co,so=rope_tables(oldpos);cn,sn=rope_tables(newpos);x=k.float();unrot=x*co-rotate_half(x)*so
    return (unrot*cn+rotate_half(unrot)*sn).to(k.dtype)
@torch.no_grad()
def forge_checkpoint(ci):
    ids=PREFIX_IDS+CAR[ci]["body"];n=len(ids);x=torch.tensor(ids,dtype=torch.long,device=DEVICE).reshape(1,-1);pos=torch.arange(n,dtype=torch.long,device=DEVICE);state={};handles=[]
    try:
        def cap(module,args,out):state["h"]=(out[0] if isinstance(out,(tuple,list)) else out).detach().clone()
        handles.append(model.model.layers[CUT].register_forward_hook(cap))
        out=model(input_ids=x,attention_mask=torch.ones((1,n),dtype=torch.long,device=DEVICE),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        early=tuple((k.detach().clone(),v.detach().clone()) for k,v in cache_layers(out.past_key_values)[:CUT+1]);h=state["h"];del out
        if h.shape!=(1,n,H) or len(early)!=CUT+1:raise RuntimeError("Checkpoint shape mismatch.")
        return {"h":h,"kv":early,"positions":pos.clone(),"body_len":n-P}
    finally:
        for h in handles:h.remove()
@torch.no_grad()
def init_incremental(ci):
    cp=forge_checkpoint(ci);n=P+cp["body_len"];early=[(k.detach().clone(),v.detach().clone()) for k,v in cp["kv"]];handles=[]
    try:
        def inject(module,args,out):return replace_hidden(out,cp["h"])
        handles.append(model.model.layers[CUT].register_forward_hook(inject))
        zeros=torch.zeros((1,n,H),dtype=DTYPE,device=DEVICE);pos=torch.arange(n,device=DEVICE)
        out=model(inputs_embeds=zeros,attention_mask=torch.ones((1,n),dtype=torch.long,device=DEVICE),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True);gen=clone_layers(out.past_key_values);result=tuple(early)+gen[CUT+1:];del out,gen,zeros,cp
        return result
    finally:
        for h in handles:h.remove()
@torch.no_grad()
def append_incremental(memory,ci):
    cp=forge_checkpoint(ci);oldn=memory[0][0].shape[-2];q=cp["body_len"];newpos=torch.arange(oldn,oldn+q,dtype=torch.long,device=DEVICE);body_h=cp["h"][:,P:,:]
    if body_h.shape!=(1,q,H):raise RuntimeError("Incremental H6 body mismatch.")
    cache=build_cache(memory);handles=[]
    try:
        def inject(module,args,out):
            h=out[0] if isinstance(out,(tuple,list)) else out
            if h.shape!=body_h.shape:raise RuntimeError("Incremental H6 injection mismatch.")
            return replace_hidden(out,body_h)
        handles.append(model.model.layers[CUT].register_forward_hook(inject))
        zeros=torch.zeros((1,q,H),dtype=DTYPE,device=DEVICE);att=torch.ones((1,oldn+q),dtype=torch.long,device=DEVICE)
        out=model(inputs_embeds=zeros,past_key_values=cache,attention_mask=att,position_ids=newpos.unsqueeze(0),cache_position=newpos,use_cache=True,return_dict=True);grown=list(clone_layers(out.past_key_values))
        for li in range(CUT+1):
            oldk,oldv=memory[li];k=cp["kv"][li][0][:,:,P:,:];v=cp["kv"][li][1][:,:,P:,:];k=rephase_key(k,cp["positions"][P:],newpos)
            grown[li]=(torch.cat((oldk,k),dim=-2),torch.cat((oldv,v),dim=-2))
        result=tuple(grown);del out,grown,zeros,cache,cp
        if result[0][0].shape[-2]!=oldn+q:raise RuntimeError("Incremental cache growth FAIL.")
        return result
    finally:
        for h in handles:h.remove()
def probe_indices(n):
    # Deterministic early / middle / late retention probes.
    if n<5:raise RuntimeError("Probe bank too small.")
    bands=[list(range(0,max(1,n//4))),list(range(max(1,n//4),max(2,3*n//4))),list(range(max(2,3*n//4),n))]
    counts=[4,4,4];out=[]
    for band,c in zip(bands,counts):
        if not band:continue
        if len(band)<=c:pick=band
        else:pick=[band[round(j*(len(band)-1)/(c-1))] for j in range(c)]
        out.extend(pick)
    out=list(dict.fromkeys(out))
    if len(out)<PROBES_PER_STAGE:
        for x in range(n):
            if x not in out:out.append(x)
            if len(out)==PROBES_PER_STAGE:break
    return out[:PROBES_PER_STAGE]
section(2,"Bank provenance and probe schedule...")
PROBES={n:probe_indices(n) for n in BANK_SIZES}
for n in BANK_SIZES:print(f"N={n:02d} PROBES={PROBES[n]}")
PROBE_SHA=sha_obj(PROBES);print("PROBE SHA:",PROBE_SHA)
section(3,"TEST560 mechanism contract...")
cp=forge_checkpoint(0)
if len(cp["kv"])!=CUT+1 or cp["h"].shape[-1]!=H:raise RuntimeError("Checkpoint contract FAIL.")
m=init_incremental(0);expected=P+len(CAR[0]["body"])
if cache_len(build_cache(m))!=expected:raise RuntimeError("Init contract FAIL.")
m2=append_incremental(m,1);expected+=len(CAR[1]["body"])
if cache_len(build_cache(m2))!=expected:raise RuntimeError("Append contract FAIL.")
print("CHECKPOINT: PASS");print("INIT: PASS");print("APPEND: PASS");print("PREFIX:",P,"CUT:",CUT)
del cp,m,m2;torch.cuda.empty_cache();gc.collect()
section(4,"Growing incremental memory bank...")
MEMORY=None;SNAPSHOTS={};APPEND_TIME=[]
for ci in range(FINAL_N):
    t=time.perf_counter()
    if MEMORY is None:MEMORY=init_incremental(ci)
    else:
        new=append_incremental(MEMORY,ci);del MEMORY;MEMORY=new
    APPEND_TIME.append(time.perf_counter()-t)
    n=ci+1
    if n in BANK_SIZES:
        expected=P+sum(len(CAR[j]["body"]) for j in range(n));actual=cache_len(build_cache(MEMORY))
        if actual!=expected:raise RuntimeError(f"Cache length FAIL at N={n}: {actual}!={expected}")
        SNAPSHOTS[n]=tuple((k.detach().clone(),v.detach().clone()) for k,v in MEMORY)
        print(f"N={n:02d} CACHE={actual} APPEND={APPEND_TIME[-1]:.3f}s TOTAL={time.perf_counter()-T0:.1f}s",flush=True)
print("GROWTH CONTRACT: PASS")
section(5,"Early / middle / late retention...")
RESULTS=[]
for n in BANK_SIZES:
    mem=SNAPSHOTS[n];idxs=PROBES[n]
    for rank,ci in enumerate(idxs):
        if ci<n//4:band="EARLY"
        elif ci>=3*n//4:band="LATE"
        else:band="MIDDLE"
        ans=answer_layers(mem,CAR[ci]["query"]);ok=bool(hit(ans,CAR[ci]["gold"]))
        RESULTS.append({"n":n,"idx":ci,"band":band,"gold":CAR[ci]["gold"],"answer":ans,"ok":ok})
        print(f"N={n:02d} IDX={ci:02d} {band:6s} {'PASS' if ok else 'FAIL'} | GOLD={CAR[ci]['gold']} | OUT={ans}",flush=True)
section(6,"Retention scores...")
S={}
for n in BANK_SIZES:
    rr=[r for r in RESULTS if r["n"]==n];S[n]={"TOTAL":sum(int(r["ok"]) for r in rr),"N":len(rr)}
    for band in ["EARLY","MIDDLE","LATE"]:
        x=[r for r in rr if r["band"]==band];S[n][band]=(sum(int(r["ok"]) for r in x),len(x))
    print(f"N={n:02d} TOTAL={S[n]['TOTAL']:02d}/{S[n]['N']:02d} EARLY={S[n]['EARLY'][0]}/{S[n]['EARLY'][1]} MIDDLE={S[n]['MIDDLE'][0]}/{S[n]['MIDDLE'][1]} LATE={S[n]['LATE'][0]}/{S[n]['LATE'][1]}")
section(7,"Final-bank exhaustive readout...")
FINAL_RESULTS=[]
for ci in range(FINAL_N):
    ans=answer_layers(MEMORY,CAR[ci]["query"]);ok=bool(hit(ans,CAR[ci]["gold"]));FINAL_RESULTS.append({"idx":ci,"gold":CAR[ci]["gold"],"answer":ans,"ok":ok})
    if (ci+1)%10==0:print(f"FINAL READOUT {ci+1:02d}/{FINAL_N} | correct={sum(int(x['ok']) for x in FINAL_RESULTS)}/{len(FINAL_RESULTS)}",flush=True)
FINAL_OK=sum(int(x["ok"]) for x in FINAL_RESULTS)
EARLY=[x for x in FINAL_RESULTS if x["idx"]<FINAL_N//4];MIDDLE=[x for x in FINAL_RESULTS if FINAL_N//4<=x["idx"]<3*FINAL_N//4];LATE=[x for x in FINAL_RESULTS if x["idx"]>=3*FINAL_N//4]
FS={"TOTAL":FINAL_OK,"EARLY":sum(int(x["ok"]) for x in EARLY),"MIDDLE":sum(int(x["ok"]) for x in MIDDLE),"LATE":sum(int(x["ok"]) for x in LATE)}
print(f"FINAL N={FINAL_N}: TOTAL={FS['TOTAL']}/{FINAL_N} EARLY={FS['EARLY']}/{len(EARLY)} MIDDLE={FS['MIDDLE']}/{len(MIDDLE)} LATE={FS['LATE']}/{len(LATE)}")
section(8,"Failure and append-cost diagnostics...")
FAIL=[x for x in FINAL_RESULTS if not x["ok"]]
print("FINAL FAILURES:",len(FAIL))
for x in FAIL[:20]:print(f"IDX={x['idx']:02d} GOLD={x['gold']} OUT={x['answer']}")
COST={}
prev=0
for n in BANK_SIZES:
    vals=APPEND_TIME[prev:n];COST[n]={"mean":float(np.mean(vals)),"max":float(np.max(vals))};prev=n
    print(f"APPEND 1..{n:02d} SEGMENT MEAN={COST[n]['mean']:.4f}s MAX={COST[n]['max']:.4f}s")
section(9,"Final engine decision...")
all_stage=all(S[n]["TOTAL"]==S[n]["N"] for n in BANK_SIZES);all_bands=all(S[n][b][0]==S[n][b][1] for n in BANK_SIZES for b in ["EARLY","MIDDLE","LATE"]);final_full=FINAL_OK==FINAL_N
if all_stage and all_bands and final_full:DECISION=f"ENGINE VALIDATION PASS — {FINAL_N}/{FINAL_N} FINAL RETENTION WITH PERFECT EARLY/MIDDLE/LATE PROBES"
elif FINAL_OK>=round(.95*FINAL_N) and all_bands:DECISION=f"STRONG PARTIAL ENGINE VALIDATION — FINAL {FINAL_OK}/{FINAL_N}"
else:DECISION=f"ENGINE VALIDATION NOT YET COMPLETE — FINAL {FINAL_OK}/{FINAL_N}"
print("STAGE PROBES :", "PASS" if all_stage else "FAIL");print("BAND RETENTION:", "PASS" if all_bands else "FAIL");print("FINAL EXHAUSTIVE:",f"{FINAL_OK}/{FINAL_N}");print("DECISION:",DECISION)
print("CAUTION: This validates append-only retention for one deterministic 80-record synthetic factual bank; it is not yet a general capacity bound.")
print("CAUTION: Memory cartridge remains augmented H6 residual + L0-L6 KV; strict KV-only memory is not claimed.")
section(10,"Final research record...")
WEIGHT_OK=sentinel()==S0;TRAINABLE=sum(int(p.requires_grad) for p in model.parameters())
LOCK={"test":TEST,"model":MODEL_ID,"seed":SEED,"bank_sha":BANK_SHA,"probe_sha":PROBE_SHA,"bank_sizes":BANK_SIZES,"final_n":FINAL_N,"cut":CUT,"checkpoint":"independent H6 residual + L0-L6 KV","growth":"append-only incremental consolidation","source_replay_during_append":False,"prior_memory_recompute":False,"training":False,"retrieval":False,"router":False,"qsf":False}
LOCK_SHA=sha_obj(LOCK);FINAL={"test":TEST,"lock_sha":LOCK_SHA,"bank_sha":BANK_SHA,"probe_sha":PROBE_SHA,"stage_scores":S,"final_score":FS,"failures":FAIL,"cost":COST,"decision":DECISION,"weight_ok":WEIGHT_OK,"trainable":TRAINABLE};RESULT_SHA=sha_obj(FINAL)
print("="*158);print("TEST561 — FINAL RESEARCH RECORD");print("="*158)
print("MODEL                  :",MODEL_ID);print("BANK SHA               :",BANK_SHA);print("PROBE SHA              :",PROBE_SHA);print("LOCK SHA               :",LOCK_SHA);print("RESULT SHA             :",RESULT_SHA)
for n in BANK_SIZES:print(f"N={n:02d}                    : {S[n]}")
print("FINAL EXHAUSTIVE        :",FS);print("ENGINEERING DECISION   :",DECISION);print("WEIGHT SENTINEL        :","PASS" if WEIGHT_OK else "FAIL");print("TRAINABLE              :",TRAINABLE);print("TOTAL TIME             :",f"{time.perf_counter()-T0:.2f}s")
print("INTERPRETATION         : Final validation of long append-only numerical memory growth and retention of early, middle and late records without replaying source text or recomputing prior memories.")
print("NEXT                    : If PASS, freeze research engine version and build demo.");print("="*158)
if not WEIGHT_OK:raise RuntimeError("WEIGHT SENTINEL FAILURE.")
if TRAINABLE!=0:raise RuntimeError("TRAINABLE PARAMETER FAILURE.")

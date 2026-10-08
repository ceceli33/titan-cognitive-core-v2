# TEST563 — AKBASCORE MAM · NATIVE-CONTEXT vs NUMERICAL-MEMORY SCALE CONTROL
# TEST562 WORKING BACKBONE PRESERVED
# SAME 80 FACTS · DIRECT TEXT CONTEXT vs INCREMENTAL DC6 · 5/10/20/40/80 SCALE CURVE
# FROZEN MISTRAL · NO TRAINING / RETRIEVAL / ROUTER / QSF · DIAGNOSTIC ONLY
import os,sys,subprocess,importlib.util,random,time,hashlib,json,re,gc
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
TEST="563";SEED=561561;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";DEVICE=torch.device("cuda");DTYPE=torch.bfloat16;MAX_NEW=16;CUT=6
SIZES=[5,10,20,40,80];FINAL_N=80;EXPECTED_BANK_SHA="c912551daf4d11d460b61a7958c7ce0ea778b82d791d7a713f5f7045e40c5d2a"
if not torch.cuda.is_available():raise RuntimeError("CUDA gerekli.")
os.environ["TOKENIZERS_PARALLELISM"]="false";random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);T0=time.perf_counter()
def sha_obj(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def norm(s):return " ".join(re.sub(r"[^\w\s-]"," ",s.casefold().strip(),flags=re.UNICODE).split())
def hit(s,g):return re.search(r"(?<!\w)"+re.escape(norm(g))+r"(?!\w)",norm(s)) is not None
def section(n,t):print(f"\n[{n}/10] {t}",flush=True)
print("="*158);print("TEST563 — AKBASCORE MAM · NATIVE-CONTEXT vs NUMERICAL-MEMORY SCALE CONTROL");print("TEST562 WORKING BACKBONE PRESERVED · SAME 80 FACTS · DIRECT TEXT vs INCREMENTAL DC6 · 5/10/20/40/80");print("FROZEN MISTRAL · NO TRAINING / RETRIEVAL / ROUTER / QSF · DIAGNOSTIC ONLY");print("="*158)
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
def native_prompt(facts,q):return f"<s>[INST] {SYSTEM}\n\nINFORMATION:\n"+"\n".join(facts)+f"\n\nQUESTION:\n{q}\n\nANSWER: [/INST]"
PREFIX_IDS=tok(canonical_prefix(),add_special_tokens=False).input_ids;P=len(PREFIX_IDS)
SYL1=["Zor","Kel","Tar","Bel","Rav","Nem","Dar","Fer","Lar","Tor","Sel","Mir","Var","Kor","Nal","Der","Cal","Par","Sor","Bar"]
SYL2=["van","vek","nor","ith","en","or","al","ik","eth","ar"]
def name(i,shift=0):
    a=SYL1[(i+shift*7)%len(SYL1)];b=SYL2[((i//len(SYL1))+i*3+shift*5)%len(SYL2)]
    return a+b+str(1000+i+shift*FINAL_N)
CAR=[];used=set()
for i in range(FINAL_N):
    s=name(i,0);g=name(i,1)
    if s in used or g in used or s==g:raise RuntimeError("Synthetic name collision.")
    used.add(s);used.add(g);fact=f"The current capital of {s} is {g}.";q=f"What is the current capital of {s}?";ids=tok(source_prefix(fact),add_special_tokens=False).input_ids
    if ids[:P]!=PREFIX_IDS:raise RuntimeError(f"Canonical prefix mismatch {i}")
    CAR.append({"idx":i,"subject":s,"gold":g,"fact":fact,"query":q,"body":ids[P:]})
BANK_SHA=sha_obj([{"idx":x["idx"],"subject":x["subject"],"gold":x["gold"],"fact":x["fact"],"query":x["query"],"body":x["body"]} for x in CAR])
print("BANK SHA:",BANK_SHA);print("BANK RECORDS:",len(CAR))
if BANK_SHA!=EXPECTED_BANK_SHA:raise RuntimeError("TEST561/562 BANK PROVENANCE FAIL.")
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
def greedy_prompt(text):
    ids=tok(text,add_special_tokens=False,return_tensors="pt").input_ids.to(DEVICE);att=torch.ones_like(ids)
    out=model(input_ids=ids,attention_mask=att,use_cache=True,return_dict=True);cache=out.past_key_values;nxt=out.logits[:,-1,:].argmax(-1,keepdim=True);gen=[];eos=set() if tok.eos_token_id is None else {int(tok.eos_token_id)}
    for _ in range(MAX_NEW):
        gen.append(nxt)
        if int(nxt.item()) in eos:break
        physical=cache_len(cache);pos=torch.tensor([[physical]],dtype=torch.long,device=DEVICE);a=torch.ones((1,physical+1),dtype=torch.long,device=DEVICE)
        out=model(input_ids=nxt,past_key_values=cache,attention_mask=a,position_ids=pos,cache_position=pos.squeeze(0),use_cache=True,return_dict=True);cache=out.past_key_values;nxt=out.logits[:,-1,:].argmax(-1,keepdim=True)
    return tok.decode(torch.cat(gen,dim=1)[0],skip_special_tokens=True).strip()
@torch.no_grad()
def answer_layers(layers,q):
    cache=build_cache(layers);physical=cache_len(cache);qids=tok(query_suffix(q),add_special_tokens=False).input_ids;ids=torch.tensor(qids,dtype=torch.long,device=DEVICE).reshape(1,-1);qlen=ids.shape[1]
    pos=torch.arange(physical,physical+qlen,dtype=torch.long,device=DEVICE).unsqueeze(0);att=torch.ones((1,physical+qlen),dtype=torch.long,device=DEVICE)
    out=model(input_ids=ids,past_key_values=cache,attention_mask=att,position_ids=pos,cache_position=pos.squeeze(0),use_cache=True,return_dict=True);cache=out.past_key_values;nxt=out.logits[:,-1,:].argmax(-1,keepdim=True);gen=[];eos=set() if tok.eos_token_id is None else {int(tok.eos_token_id)}
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
section(2,"Provenance and mechanism contract...")
print("BANK PROVENANCE: PASS");print("PREFIX:",P,"CUT:",CUT)
cp=forge_checkpoint(0);m=init_incremental(0);m2=append_incremental(m,1)
if len(cp["kv"])!=CUT+1 or cp["h"].shape[-1]!=H:raise RuntimeError("Checkpoint contract FAIL.")
if cache_len(build_cache(m2))!=P+len(CAR[0]["body"])+len(CAR[1]["body"]):raise RuntimeError("Append contract FAIL.")
print("CHECKPOINT: PASS");print("INCREMENTAL APPEND: PASS");del cp,m,m2;torch.cuda.empty_cache();gc.collect()
section(3,"Build unchanged incremental DC6 scale snapshots...")
MEM=None;SNAPS={}
for ci in range(FINAL_N):
    if MEM is None:MEM=init_incremental(ci)
    else:
        new=append_incremental(MEM,ci);del MEM;MEM=new
    if ci+1 in SIZES:
        SNAPS[ci+1]=tuple((k.detach().clone(),v.detach().clone()) for k,v in MEM)
        print(f"N={ci+1:02d} CACHE={cache_len(build_cache(MEM))} elapsed={time.perf_counter()-T0:.1f}s",flush=True)
print("MEMORY GROWTH: PASS")
section(4,"Native direct-text scale curve...")
NATIVE={};NATIVE_ROWS={}
for n in SIZES:
    facts=[CAR[i]["fact"] for i in range(n)];rows=[]
    prompt_tokens=len(tok(native_prompt(facts,CAR[0]["query"]),add_special_tokens=False).input_ids)
    for ci in range(n):
        ans=greedy_prompt(native_prompt(facts,CAR[ci]["query"]));ok=bool(hit(ans,CAR[ci]["gold"]));rows.append({"idx":ci,"answer":ans,"ok":ok})
    NATIVE[n]=sum(int(x["ok"]) for x in rows);NATIVE_ROWS[n]=rows
    print(f"NATIVE N={n:02d}: {NATIVE[n]:02d}/{n:02d} | TOKENS≈{prompt_tokens}",flush=True)
section(5,"Incremental DC6 scale curve...")
DC6={};DC6_ROWS={}
for n in SIZES:
    rows=[]
    for ci in range(n):
        ans=answer_layers(SNAPS[n],CAR[ci]["query"]);ok=bool(hit(ans,CAR[ci]["gold"]));rows.append({"idx":ci,"answer":ans,"ok":ok})
    DC6[n]=sum(int(x["ok"]) for x in rows);DC6_ROWS[n]=rows
    print(f"DC6   N={n:02d}: {DC6[n]:02d}/{n:02d}",flush=True)
section(6,"Paired native-vs-memory failures...")
PAIR={}
for n in SIZES:
    nn=sum(int(a["ok"] and not b["ok"]) for a,b in zip(NATIVE_ROWS[n],DC6_ROWS[n]))
    dd=sum(int((not a["ok"]) and b["ok"]) for a,b in zip(NATIVE_ROWS[n],DC6_ROWS[n]))
    both=sum(int(a["ok"] and b["ok"]) for a,b in zip(NATIVE_ROWS[n],DC6_ROWS[n]))
    fail=sum(int((not a["ok"]) and (not b["ok"])) for a,b in zip(NATIVE_ROWS[n],DC6_ROWS[n]))
    PAIR[n]={"native_only":nn,"dc6_only":dd,"both":both,"both_fail":fail}
    print(f"N={n:02d} BOTH={both:02d} NATIVE_ONLY={nn:02d} DC6_ONLY={dd:02d} BOTH_FAIL={fail:02d}")
section(7,"80-record autopsy...")
for ci in range(FINAL_N):
    a=NATIVE_ROWS[80][ci];b=DC6_ROWS[80][ci]
    if a["ok"]!=b["ok"] or not a["ok"]:
        tag="NATIVE_ONLY" if a["ok"] and not b["ok"] else "DC6_ONLY" if b["ok"] and not a["ok"] else "BOTH_FAIL"
        print(f"IDX={ci:02d} {tag:11s} GOLD={CAR[ci]['gold']} | NATIVE={a['answer']} | DC6={b['answer']}")
section(8,"Capacity attribution...")
GAP={n:NATIVE[n]-DC6[n] for n in SIZES}
print("SIZE | NATIVE | DC6 | GAP")
for n in SIZES:print(f"{n:02d}   | {NATIVE[n]:02d}/{n:02d}  | {DC6[n]:02d}/{n:02d} | {GAP[n]:+03d}")
if NATIVE[80]>=72 and DC6[80]<=40:DIAG="NUMERICAL-MEMORY SCALE BOTTLENECK — NATIVE 7B READOUT REMAINS STRONG AT 80"
elif NATIVE[80]<=40 and DC6[80]<=40 and abs(NATIVE[80]-DC6[80])<=10:DIAG="SHARED 7B/READOUT CAPACITY BOTTLENECK — NATIVE AND NUMERICAL MEMORY BOTH COLLAPSE"
elif NATIVE[80]>DC6[80]+20:DIAG="MIXED, MEMORY-DOMINANT SCALE LOSS — NATIVE ALSO LIMITED BUT DC6 ADDS LARGE DEGRADATION"
elif NATIVE[80]<=40 and DC6[80]<NATIVE[80]-10:DIAG="MIXED BOTTLENECK — 7B NATIVE LIMIT PLUS ADDITIONAL DC6 SCALE LOSS"
else:DIAG="MIXED/INCONCLUSIVE — SCALE CURVES REQUIRE TARGETED FOLLOW-UP"
print("DIAGNOSIS:",DIAG)
section(9,"Decision gate...")
print("IMPORTANT: Native control uses the same frozen Mistral, same facts, same questions and same greedy readout.")
print("IMPORTANT: Native text is a diagnostic upper/control condition; it is not part of the memory engine.")
print("IMPORTANT: DC6 mechanism is unchanged from TEST561/562.")
print("IMPORTANT: This test does not introduce retrieval, routing, training or a new memory mechanism.")
print("DECISION:",DIAG)
section(10,"Final research record...")
WEIGHT_OK=sentinel()==S0;TRAINABLE=sum(int(p.requires_grad) for p in model.parameters())
LOCK={"test":TEST,"parent":"TEST562","model":MODEL_ID,"seed":SEED,"bank_sha":BANK_SHA,"sizes":SIZES,"cut":CUT,"control":"same facts/questions native direct-text context","memory":"unchanged incremental DC6","training":False,"retrieval":False,"router":False,"qsf":False}
LOCK_SHA=sha_obj(LOCK);FINAL={"test":TEST,"lock_sha":LOCK_SHA,"bank_sha":BANK_SHA,"native":NATIVE,"dc6":DC6,"gap":GAP,"paired":PAIR,"diagnosis":DIAG,"weight_ok":WEIGHT_OK,"trainable":TRAINABLE};RESULT_SHA=sha_obj(FINAL)
print("="*158);print("TEST563 — FINAL RESEARCH RECORD");print("="*158);print("MODEL                  :",MODEL_ID);print("BANK SHA               :",BANK_SHA);print("LOCK SHA               :",LOCK_SHA);print("RESULT SHA             :",RESULT_SHA)
for n in SIZES:print(f"N={n:02d}                    : NATIVE={NATIVE[n]}/{n} DC6={DC6[n]}/{n} GAP={GAP[n]:+d} {PAIR[n]}")
print("DIAGNOSIS              :",DIAG);print("WEIGHT SENTINEL        :","PASS" if WEIGHT_OK else "FAIL");print("TRAINABLE              :",TRAINABLE);print("TOTAL TIME             :",f"{time.perf_counter()-T0:.2f}s")
print("INTERPRETATION         : Same frozen 7B model is tested on identical facts under native text context and unchanged incremental DC6 numerical memory, isolating model/readout scale limits from memory-specific scale loss.")
print("NEXT                    : Interpret TEST563 before any engine modification.");print("="*158)
if not WEIGHT_OK:raise RuntimeError("WEIGHT SENTINEL FAILURE.")
if TRAINABLE!=0:raise RuntimeError("TRAINABLE PARAMETER FAILURE.")

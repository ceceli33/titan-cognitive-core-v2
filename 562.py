# TEST562 — AKBASCORE MAM · CAPACITY vs CARTRIDGE INTEGRITY X-RAY
# TEST561 WORKING BACKBONE PRESERVED
# SAME 80 CARTRIDGES · SAME INCREMENTAL DC6 · FULL vs PREFIX vs LOCAL-WINDOW READOUT
# FROZEN MISTRAL · NO TRAINING / RETRIEVAL / ROUTER / QSF · DIAGNOSTIC ONLY
import os,sys,subprocess,importlib.util,random,time,hashlib,json,re,gc
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
TEST="562";SEED=561561;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";DEVICE=torch.device("cuda");DTYPE=torch.bfloat16;MAX_NEW=16;CUT=6;FINAL_N=80
WINDOWS=[5,10,20,40];EXPECTED_BANK_SHA="c912551daf4d11d460b61a7958c7ce0ea778b82d791d7a713f5f7045e40c5d2a"
if not torch.cuda.is_available():raise RuntimeError("CUDA gerekli.")
os.environ["TOKENIZERS_PARALLELISM"]="false";random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);T0=time.perf_counter()
def sha_obj(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def norm(s):return " ".join(re.sub(r"[^\w\s-]"," ",s.casefold().strip(),flags=re.UNICODE).split())
def hit(s,g):return re.search(r"(?<!\w)"+re.escape(norm(g))+r"(?!\w)",norm(s)) is not None
def section(n,t):print(f"\n[{n}/10] {t}",flush=True)
print("="*158);print("TEST562 — AKBASCORE MAM · CAPACITY vs CARTRIDGE INTEGRITY X-RAY");print("TEST561 WORKING BACKBONE PRESERVED · SAME 80 CARTRIDGES · FULL vs PREFIX vs LOCAL-WINDOW READOUT");print("FROZEN MISTRAL · NO TRAINING / RETRIEVAL / ROUTER / QSF · DIAGNOSTIC ONLY");print("="*158)
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
if BANK_SHA!=EXPECTED_BANK_SHA:raise RuntimeError("TEST561 BANK PROVENANCE FAIL.")
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
# Diagnostic banks are rebuilt from the SAME independently forged cartridges.
# No query-conditioned selection is used to construct them.
@torch.no_grad()
def build_bank(indices):
    mem=None
    for ci in indices:
        if mem is None:mem=init_incremental(ci)
        else:
            new=append_incremental(mem,ci);del mem;mem=new
    return mem
section(2,"TEST561 provenance and mechanism contract...")
print("BANK PROVENANCE: PASS");print("PREFIX:",P,"CUT:",CUT)
cp=forge_checkpoint(0);m=init_incremental(0);m2=append_incremental(m,1)
if len(cp["kv"])!=CUT+1 or cp["h"].shape[-1]!=H:raise RuntimeError("Checkpoint contract FAIL.")
if cache_len(build_cache(m2))!=P+len(CAR[0]["body"])+len(CAR[1]["body"]):raise RuntimeError("Append contract FAIL.")
print("CHECKPOINT: PASS");print("INCREMENTAL APPEND: PASS");del cp,m,m2;torch.cuda.empty_cache();gc.collect()
section(3,"Rebuild exact 80-record incremental bank...")
FULL=None;SNAPS={}
for ci in range(FINAL_N):
    if FULL is None:FULL=init_incremental(ci)
    else:
        new=append_incremental(FULL,ci);del FULL;FULL=new
    if ci+1 in WINDOWS:SNAPS[ci+1]=tuple((k.detach().clone(),v.detach().clone()) for k,v in FULL)
    if (ci+1)%10==0:print(f"APPEND {ci+1:02d}/{FINAL_N} CACHE={cache_len(build_cache(FULL))} elapsed={time.perf_counter()-T0:.1f}s",flush=True)
print("FULL BANK GROWTH: PASS")
section(4,"Exact TEST561 full-bank replication...")
FULL_RES=[]
for ci in range(FINAL_N):
    ans=answer_layers(FULL,CAR[ci]["query"]);ok=bool(hit(ans,CAR[ci]["gold"]));FULL_RES.append({"idx":ci,"answer":ans,"ok":ok})
    if (ci+1)%10==0:print(f"FULL {ci+1:02d}/{FINAL_N} correct={sum(int(x['ok']) for x in FULL_RES)}/{len(FULL_RES)}",flush=True)
FULL_OK=sum(int(x["ok"]) for x in FULL_RES);print("FULL80:",FULL_OK,"/80")
if FULL_OK!=22:print("WARNING: TEST561 exact score 22/80 not reproduced; continue as diagnostic but provenance differs behaviorally.")
section(5,"Prefix-capacity curve...")
PREFIX_RES={}
for n in WINDOWS:
    rr=[]
    for ci in range(n):
        ans=answer_layers(SNAPS[n],CAR[ci]["query"]);rr.append(bool(hit(ans,CAR[ci]["gold"])))
    PREFIX_RES[n]=sum(rr);print(f"PREFIX N={n:02d}: {PREFIX_RES[n]:02d}/{n:02d}")
section(6,"Independent local-window reconstruction...")
# Fixed contiguous windows. These are NOT retrieval: every window is predetermined before readout.
RANGES=[(0,5),(0,10),(0,20),(20,40),(40,60),(60,80)]
LOCAL={}
for a,b in RANGES:
    mem=build_bank(range(a,b));ok=0;rows=[]
    for ci in range(a,b):
        ans=answer_layers(mem,CAR[ci]["query"]);good=bool(hit(ans,CAR[ci]["gold"]));ok+=int(good);rows.append({"idx":ci,"answer":ans,"ok":good})
    LOCAL[f"{a}:{b}"]={"ok":ok,"n":b-a,"rows":rows};print(f"LOCAL {a:02d}:{b:02d} = {ok:02d}/{b-a:02d}")
    del mem;torch.cuda.empty_cache();gc.collect()
section(7,"Failure recovery matrix...")
FAIL=[x["idx"] for x in FULL_RES if not x["ok"]];REC={}
for ci in FAIL:
    candidates=[]
    for a,b in RANGES:
        if a<=ci<b:candidates.append((a,b))
    if ci<20:candidates.append((0,20))
    seen=set();rows=[]
    for a,b in candidates:
        if (a,b) in seen:continue
        seen.add((a,b));r=LOCAL.get(f"{a}:{b}")
        if r is not None:
            z=next(x for x in r["rows"] if x["idx"]==ci);rows.append({"window":f"{a}:{b}","ok":z["ok"],"answer":z["answer"]})
    REC[ci]=rows
recovered=sum(any(z["ok"] for z in rows) for rows in REC.values())
print("FULL FAILURES:",len(FAIL));print("RECOVERED IN FIXED SMALLER BANK:",recovered,"/",len(FAIL))
for ci in FAIL:
    rows=REC[ci];status="RECOVERED" if any(z["ok"] for z in rows) else "NOT_RECOVERED"
    print(f"IDX={ci:02d} {status:13s} GOLD={CAR[ci]['gold']} FULL={FULL_RES[ci]['answer']} | "+", ".join(f"{z['window']}={'PASS' if z['ok'] else 'FAIL'}:{z['answer']}" for z in rows))
section(8,"Cartridge integrity vs scale diagnosis...")
LOCAL20=sum(LOCAL[f"{a}:{b}"]["ok"] for a,b in [(0,20),(20,40),(40,60),(60,80)])
LOCAL20_N=80
LATE_LOCAL=LOCAL["60:80"]["ok"];LATE_FULL=sum(int(FULL_RES[i]["ok"]) for i in range(60,80))
print(f"FULL80             : {FULL_OK}/80");print(f"FOUR FIXED 20 BANKS: {LOCAL20}/{LOCAL20_N}");print(f"LATE 60:80 LOCAL   : {LATE_LOCAL}/20");print(f"LATE 60:80 FULL80  : {LATE_FULL}/20");print(f"FAILED→RECOVERED   : {recovered}/{len(FAIL)}")
if LOCAL20>=72 and FULL_OK<=40:DIAG="STRONG SCALE/READOUT INTERFERENCE — CARTRIDGES RECOVER IN SMALLER FIXED BANKS"
elif LOCAL20>FULL_OK+20:DIAG="SCALE-SENSITIVE READOUT/COMPOSITION — SUBSTANTIAL RECOVERY IN SMALLER BANKS"
elif LOCAL20<=FULL_OK+5:DIAG="NO STRONG SMALL-BANK RECOVERY — CARTRIDGE/COMPOSITION DAMAGE REMAINS PLAUSIBLE"
else:DIAG="MIXED — BOTH SCALE AND CARTRIDGE/COMPOSITION EFFECTS REMAIN PLAUSIBLE"
print("DIAGNOSIS:",DIAG)
section(9,"Interpretation gate...")
print("IMPORTANT: This test does not change the engine and does not introduce retrieval.")
print("IMPORTANT: Fixed local banks are diagnostic reconstructions from the same cartridge-generation mechanism.")
print("IMPORTANT: A recovery does not prove a 7B parameter-capacity limit; it demonstrates scale-dependent interference/readout failure in this frozen 7B system.")
print("IMPORTANT: Failure to recover points back toward cartridge/consolidation integrity rather than mere bank size.")
print("DECISION:",DIAG)
section(10,"Final research record...")
WEIGHT_OK=sentinel()==S0;TRAINABLE=sum(int(p.requires_grad) for p in model.parameters())
LOCK={"test":TEST,"parent":"TEST561","model":MODEL_ID,"seed":SEED,"bank_sha":BANK_SHA,"n":FINAL_N,"windows":WINDOWS,"ranges":RANGES,"cut":CUT,"engine_change":False,"training":False,"retrieval":False,"router":False,"qsf":False}
LOCK_SHA=sha_obj(LOCK)
FINAL={"test":TEST,"lock_sha":LOCK_SHA,"bank_sha":BANK_SHA,"full80":FULL_OK,"prefix":PREFIX_RES,"local":{k:{"ok":v["ok"],"n":v["n"]} for k,v in LOCAL.items()},"recovered":recovered,"full_failures":len(FAIL),"diagnosis":DIAG,"weight_ok":WEIGHT_OK,"trainable":TRAINABLE}
RESULT_SHA=sha_obj(FINAL)
print("="*158);print("TEST562 — FINAL RESEARCH RECORD");print("="*158);print("MODEL                  :",MODEL_ID);print("BANK SHA               :",BANK_SHA);print("LOCK SHA               :",LOCK_SHA);print("RESULT SHA             :",RESULT_SHA);print("FULL80                 :",f"{FULL_OK}/80")
for n in WINDOWS:print(f"PREFIX {n:02d}               :",f"{PREFIX_RES[n]}/{n}")
for a,b in RANGES:print(f"LOCAL {a:02d}:{b:02d}             :",f"{LOCAL[f'{a}:{b}']['ok']}/{b-a}")
print("FAILED→RECOVERED       :",f"{recovered}/{len(FAIL)}");print("DIAGNOSIS              :",DIAG);print("WEIGHT SENTINEL        :","PASS" if WEIGHT_OK else "FAIL");print("TRAINABLE              :",TRAINABLE);print("TOTAL TIME             :",f"{time.perf_counter()-T0:.2f}s")
print("INTERPRETATION         : Distinguishes scale-dependent readout/composition failure from irreversible cartridge damage using the unchanged TEST561 incremental DC6 mechanism.")
print("NEXT                    : TEST563 only after TEST562 interpretation.");print("="*158)
if not WEIGHT_OK:raise RuntimeError("WEIGHT SENTINEL FAILURE.")
if TRAINABLE!=0:raise RuntimeError("TRAINABLE PARAMETER FAILURE.")

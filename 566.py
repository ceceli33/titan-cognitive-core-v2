# TEST566 — AKBASCORE MAM · QWEN CAUSAL CONSOLIDATION CONTROLS
# TEST564 FROZEN BACKBONE · JOINT / INDEP / DC / NOCROSS · CUTS 1/2/3/6
# SAME 24-CASE PANEL SHA · SAME TOKENS / LOGICAL POSITIONS
# NO TRAINING · NO RETRIEVAL · NO ROUTER · NO SOURCE REPLAY AT ASK
import os,sys,subprocess,importlib.util,time,random,re,hashlib,json,gc
for m in ("torch","transformers","accelerate"):
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",m])
import torch,numpy as np,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
TEST="566";MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SEED=552552;PANEL_SEED=550550;CUTS=[1,2,3,6];MAX_NEW=16;DEVICE=torch.device("cuda");DTYPE=torch.bfloat16
ARMS=["JOINT","INDEP","DC","NOCROSS"];CASES=list(range(8));SLOTS=["FIRST","MIDDLE","LAST"]
EXPECTED_PANEL_SHA="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required")
os.environ["TOKENIZERS_PARALLELISM"]="false";random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
torch.set_grad_enabled(False);T0=time.perf_counter()
def sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def norm(s):return " ".join(re.sub(r"[^\w\s-]"," ",s.casefold()).split())
def hit(s,g):return re.search(r"(?<!\w)"+re.escape(norm(g))+r"(?!\w)",norm(s)) is not None
print("="*132);print("TEST566 — QWEN MAM · CAUSAL CONSOLIDATION CONTROLS");print("="*132)
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",dtype=DTYPE).eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=H//QH
assert (NL,H,QH,KVH,HD)==(28,3584,28,4,128),(NL,H,QH,KVH,HD)
print("GPU:",torch.cuda.get_device_name(0),"TORCH:",torch.__version__,"TRANSFORMERS:",transformers.__version__)
print("ARCH:",NL,H,QH,KVH,HD,"CUTS:",CUTS,"ARMS:",ARMS)
FP=[model.model.layers[i].self_attn.o_proj.weight for i in (0,7,14,21,27)]+[model.model.norm.weight,model.lm_head.weight]
def sentinel():
    h=hashlib.sha256()
    for p in FP:
        a=p.detach().reshape(-1);n=a.numel()
        for j in range(16):
            o=j*(n-256)//15;h.update(a[o:o+256].float().cpu().numpy().tobytes())
    return h.hexdigest()
S0=sentinel()
BASE=[
("Zorvan","Melket","Dravel","Oakhaven","Pelnor"),("Kelvar","Nareth","Solven","Branik","Tarsen"),("Tarev","Luneth","Varos","Cedran","Mireth"),("Belnor","Arven","Dorel","Kesmar","Falven"),
("Ravik","Selora","Terven","Maldor","Nerik"),("Nemor","Calven","Istral","Pareth","Dovren"),("Darsen","Velora","Keldin","Orvek","Sarnel"),("Feron","Talven","Merith","Sovran","Belvik"),
("Larev","Nerith","Calder","Veyron","Tormek"),("Torven","Elsar","Marvek","Dorin","Kaleth"),("Selnor","Kareth","Valen","Ordan","Mervek"),("Mirev","Taldor","Neris","Kelmar","Sorvik"),
("Varen","Solith","Deran","Malvek","Cordan"),("Kelor","Ardin","Velmar","Toren","Narell"),("Narev","Belith","Corven","Sareth","Dorvik"),("Dervan","Mirel","Talvek","Orsen","Kelron"),
("Calnor","Verith","Naldor","Seren","Parvek"),("Parel","Dorven","Kelith","Maros","Tervik"),("Sorven","Tarell","Vindor","Nelmar","Calrek"),("Barel","Corith","Laven","Derik","Solmar"),
("Ralen","Mervor","Talith","Kesven","Noreth"),("Norel","Valdor","Serith","Calvenor","Darvek"),("Tervan","Orel","Mardin","Velos","Karven"),("Karev","Solen","Dareth","Mirven","Talrek")]
W=[];CAR=[];KEY={}
for wi,(s,current,near,former,role) in enumerate(BASE):
    facts=[("CURRENT",f"The current capital of {s} is {current}.",current),("NEAR",f"The largest city of {s} is {near}.",near),("FORMER",f"The former capital of {s} was {former}.",former),("ROLE",f"The current capital of {role} is {s}.",s)]
    queries={"CURRENT":f"What is the current capital of {s}?","FORMER":f"What was the former capital of {s}?","NEAR":f"What is the largest city of {s}?","ROLE":f"What is the current capital of {role}?"}
    W.append(queries)
    for typ,f,gold in facts:
        KEY[(wi,typ)]=len(CAR);CAR.append(dict(world=wi,type=typ,fact=f,gold=gold))
SYSTEM="Answer the question using only the information provided. Give only the requested name and nothing else."
PREFIX=f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\nINFORMATION:\n"
def source(f):return PREFIX+f+"\n\n"
def suffix(q):return f"\nQUESTION:\n{q}\n\nANSWER:\n<|im_end|>\n<|im_start|>assistant\n"
PIDS=tok(PREFIX,add_special_tokens=False).input_ids;P=len(PIDS)
for c in CAR:
    ids=tok(source(c["fact"]),add_special_tokens=False).input_ids
    if ids[:P]!=PIDS:raise RuntimeError("Prefix mismatch")
    c["body"]=ids[P:]
rng=random.Random(PANEL_SEED);pool=list(range(96));rng.shuffle(pool);PANEL=[];cursor=0;cycle=["CURRENT","FORMER","NEAR","ROLE"]
for wi in range(24):
    typ=cycle[wi%4];target=KEY[(wi,typ)];others=[]
    while len(others)<4:
        ci=pool[cursor%96];cursor+=1
        if ci==target or CAR[ci]["world"]==wi or any(CAR[x]["world"]==CAR[ci]["world"] for x in others):continue
        others.append(ci)
    PANEL.append(dict(case=wi,target=target,others=others))
PANEL_SHA=sha(PANEL);assert PANEL_SHA==EXPECTED_PANEL_SHA,(PANEL_SHA,EXPECTED_PANEL_SHA)
print("PANEL SHA:",PANEL_SHA,"PASS","PREFIX:",P)
def ordered(z,slot):
    d=z["others"];t=z["target"]
    return [t]+d if slot=="FIRST" else d[:2]+[t]+d[2:] if slot=="MIDDLE" else d+[t]
def pairs(c):
    if hasattr(c,"layers"):return tuple((l.keys,l.values) for l in c.layers)
    return tuple(zip(c.key_cache,c.value_cache))
def clone(c):return tuple((k.detach().clone(),v.detach().clone()) for k,v in pairs(c))
def cache_build(x):
    c=DynamicCache()
    for i,(k,v) in enumerate(x):c.update(k,v,i)
    return c
def replace(out,h):
    if isinstance(out,tuple):return (h,)+out[1:]
    if isinstance(out,list):return [h]+out[1:]
    return h
def half(x):
    a,b=x.chunk(2,-1);return torch.cat((-b,a),-1)
def rope(pos):
    dummy=torch.zeros((1,len(pos),H),device=DEVICE,dtype=DTYPE)
    c,s=model.model.rotary_emb(dummy,pos.unsqueeze(0))
    return c.unsqueeze(1).float(),s.unsqueeze(1).float()
def rephase(k,old,new):
    if torch.equal(old,new):return k
    co,so=rope(old);cn,sn=rope(new);x=k.float()
    x=x*co-half(x)*so
    return (x*cn+half(x)*sn).to(k.dtype)
@torch.no_grad()
def checkpoint(ci,cut):
    ids=PIDS+CAR[ci]["body"];n=len(ids);x=torch.tensor([ids],device=DEVICE);pos=torch.arange(n,device=DEVICE);state={}
    def capture(module,args,out):state["h"]=(out[0] if isinstance(out,(tuple,list)) else out).detach().clone()
    h=model.model.layers[cut].register_forward_hook(capture)
    try:
        out=model(input_ids=x,attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        kv=clone(out.past_key_values)
        return dict(h=state["h"],kv=kv,pos=pos,body=n-P)
    finally:h.remove()
@torch.no_grad()
def init(ci,cut):
    cp=checkpoint(ci,cut);n=cp["h"].shape[1]
    def inject(module,args,out):return replace(out,cp["h"])
    h=model.model.layers[cut].register_forward_hook(inject)
    try:
        pos=torch.arange(n,device=DEVICE)
        out=model(inputs_embeds=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        upper=clone(out.past_key_values)
        return cp["kv"][:cut+1]+upper[cut+1:]
    finally:h.remove()
@torch.no_grad()
def append(memory,ci,cut):
    cp=checkpoint(ci,cut);oldn=memory[0][0].shape[-2];q=cp["body"];pos=torch.arange(oldn,oldn+q,device=DEVICE);body=cp["h"][:,P:,:]
    if body.shape!=(1,q,H):raise RuntimeError("Body shape mismatch")
    def inject(module,args,out):
        current=out[0] if isinstance(out,(tuple,list)) else out
        if current.shape!=body.shape:raise RuntimeError("Injection shape mismatch")
        return replace(out,body)
    h=model.model.layers[cut].register_forward_hook(inject)
    try:
        out=model(inputs_embeds=torch.zeros((1,q,H),device=DEVICE,dtype=DTYPE),past_key_values=cache_build(memory),attention_mask=torch.ones((1,oldn+q),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        grown=list(clone(out.past_key_values))
        for li in range(cut+1):
            ok,ov=memory[li];k,v=cp["kv"][li];k=rephase(k[:,:,P:,:],cp["pos"][P:],pos);v=v[:,:,P:,:]
            grown[li]=(torch.cat((ok,k),-2),torch.cat((ov,v),-2))
        result=tuple(grown)
        for (ok,ov),(nk,nv) in zip(memory,result):
            if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):raise RuntimeError("APPEND-ONLY BITWISE FAIL")
        if any(k.shape[-2]!=oldn+q for k,v in result):raise RuntimeError("Cache length mismatch")
        return result
    finally:h.remove()
@torch.no_grad()
def answer(memory,q):
    cache=cache_build(memory);n=memory[0][0].shape[-2];ids=tok(suffix(q),add_special_tokens=False).input_ids
    x=torch.tensor([ids],device=DEVICE);pos=torch.arange(n,n+len(ids),device=DEVICE)
    out=model(input_ids=x,past_key_values=cache,attention_mask=torch.ones((1,n+len(ids)),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
    cache=out.past_key_values;next_id=out.logits[:,-1,:].argmax(-1,keepdim=True);gen=[]
    for _ in range(MAX_NEW):
        t=int(next_id.item())
        if t in (tok.eos_token_id,tok.convert_tokens_to_ids("<|im_end|>")):break
        gen.append(t);n=cache.get_seq_length();pos=torch.tensor([n],device=DEVICE)
        out=model(input_ids=next_id,past_key_values=cache,attention_mask=torch.ones((1,n+1),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        cache=out.past_key_values;next_id=out.logits[:,-1,:].argmax(-1,keepdim=True)
    return tok.decode(gen,skip_special_tokens=True).strip()
# TEST566 ADDITIONS — original TEST564 checkpoint/init/append/answer remain unchanged.
@torch.no_grad()
def joint(keys):
    ids=PIDS+sum((CAR[ci]["body"] for ci in keys),[])
    n=len(ids);x=torch.tensor([ids],device=DEVICE);pos=torch.arange(n,device=DEVICE)
    out=model(input_ids=x,attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
    return clone(out.past_key_values)
@torch.no_grad()
def independent(keys,cut,mode):
    if mode not in ("INDEP","NOCROSS"):raise ValueError(mode)
    memory=None;oldn=0
    for j,ci in enumerate(keys):
        cp=checkpoint(ci,cut) if mode=="INDEP" else None
        kv=cp["kv"] if mode=="INDEP" else init(ci,cut)
        if j==0:
            memory=kv;oldn=kv[0][0].shape[-2]
            continue
        q=len(CAR[ci]["body"]);oldpos=torch.arange(P,P+q,device=DEVICE);newpos=torch.arange(oldn,oldn+q,device=DEVICE);grown=[]
        for li,((ok,ov),(k,v)) in enumerate(zip(memory,kv)):
            kk=rephase(k[:,:,P:,:],oldpos,newpos);vv=v[:,:,P:,:]
            grown.append((torch.cat((ok,kk),-2),torch.cat((ov,vv),-2)))
        result=tuple(grown)
        for (ok,ov),(nk,nv) in zip(memory,result):
            if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):raise RuntimeError("INDEPENDENT APPEND BITWISE FAIL")
        memory=result;oldn+=q
    return memory
@torch.no_grad()
def consolidated(keys,cut):
    memory=init(keys[0],cut)
    for ci in keys[1:]:memory=append(memory,ci,cut)
    return memory
def expected_length(keys):return P+sum(len(CAR[ci]["body"]) for ci in keys)
def validate(memory,keys):
    n=expected_length(keys)
    if len(memory)!=NL:raise RuntimeError("Layer count mismatch")
    for li,(k,v) in enumerate(memory):
        if k.shape!=(1,KVH,n,HD) or v.shape!=(1,KVH,n,HD):raise RuntimeError(f"Cache shape mismatch L{li}: {k.shape}/{v.shape}")
        if not torch.isfinite(k).all() or not torch.isfinite(v).all():raise RuntimeError(f"Nonfinite cache L{li}")
    return n
def score_rows(rows):
    return {s:sum(int(r["ok"]) for r in rows if r["slot"]==s) for s in SLOTS}
RESULTS=[];RAW=[];ERRORS=[]
print("\n[1/5] JOINT natural reference — same 24 questions...")
for wi in CASES:
    z=PANEL[wi];typ=CAR[z["target"]]["type"];q=W[wi][typ];gold=CAR[z["target"]]["gold"]
    for slot in SLOTS:
        keys=ordered(z,slot)
        try:
            mem=joint(keys);validate(mem,keys);a=answer(mem,q);ok=hit(a,gold)
            RAW.append(dict(arm="JOINT",cut=None,case=wi,slot=slot,gold=gold,answer=a,ok=ok))
            print(f"JOINT CASE={wi:02d} {slot:6s} {'PASS' if ok else 'FAIL'} | gold={gold} | answer={a!r}",flush=True)
            del mem
        except Exception as e:
            ERRORS.append(dict(arm="JOINT",case=wi,slot=slot,error=repr(e)));print("JOINT ERROR:",repr(e),flush=True);raise
JROWS=[r for r in RAW if r["arm"]=="JOINT"];JS=score_rows(JROWS)
print("JOINT:",JS,"TOTAL:",sum(JS.values()),"/24")
print("\n[2/5] INDEP native independent KV reference...")
for wi in CASES:
    z=PANEL[wi];typ=CAR[z["target"]]["type"];q=W[wi][typ];gold=CAR[z["target"]]["gold"]
    for slot in SLOTS:
        keys=ordered(z,slot)
        try:
            mem=independent(keys,0,"INDEP");validate(mem,keys);a=answer(mem,q);ok=hit(a,gold)
            RAW.append(dict(arm="INDEP",cut=None,case=wi,slot=slot,gold=gold,answer=a,ok=ok))
            print(f"INDEP CASE={wi:02d} {slot:6s} {'PASS' if ok else 'FAIL'} | gold={gold} | answer={a!r}",flush=True)
            del mem
        except Exception as e:
            ERRORS.append(dict(arm="INDEP",case=wi,slot=slot,error=repr(e)));print("INDEP ERROR:",repr(e),flush=True);raise
IROWS=[r for r in RAW if r["arm"]=="INDEP"];IS=score_rows(IROWS)
print("INDEP:",IS,"TOTAL:",sum(IS.values()),"/24")
print("\n[3/5] CUT-wise DC and NOCROSS...")
for cut in CUTS:
    tcut=time.perf_counter()
    for arm in ("DC","NOCROSS"):
        rows=[];failed=None
        print("\n"+"="*110);print("CUT",cut,"ARM",arm,flush=True)
        try:
            for wi in CASES:
                z=PANEL[wi];typ=CAR[z["target"]]["type"];q=W[wi][typ];gold=CAR[z["target"]]["gold"]
                for slot in SLOTS:
                    keys=ordered(z,slot)
                    mem=consolidated(keys,cut) if arm=="DC" else independent(keys,cut,"NOCROSS")
                    validate(mem,keys);a=answer(mem,q);ok=hit(a,gold)
                    r=dict(arm=arm,cut=cut,case=wi,slot=slot,gold=gold,answer=a,ok=ok)
                    rows.append(r);RAW.append(r)
                    print(f"{arm:7s} CUT={cut:02d} CASE={wi:02d} {slot:6s} {'PASS' if ok else 'FAIL'} | gold={gold} | answer={a!r}",flush=True)
                    del mem
        except Exception as e:
            failed=repr(e);ERRORS.append(dict(arm=arm,cut=cut,error=failed));print("ARM ERROR:",failed,flush=True)
        ss=score_rows(rows);complete=len(rows)==24 and failed is None
        RESULTS.append(dict(arm=arm,cut=cut,scores=ss,total=sum(ss.values()),complete=complete,error=failed))
        print(f"ARM={arm} CUT={cut} SCORE={sum(ss.values())}/24 COMPLETE={complete} FIRST={ss['FIRST']}/8 MIDDLE={ss['MIDDLE']}/8 LAST={ss['LAST']}/8",flush=True)
        if not complete:raise RuntimeError(f"INCOMPLETE ARM {arm} CUT {cut}: {failed}")
    print("CUT",cut,"SECONDS",round(time.perf_counter()-tcut,2))
    torch.cuda.empty_cache();gc.collect()
print("\n[4/5] Causal comparison and paired transitions...")
MAP={(r["arm"],r["cut"],r["case"],r["slot"]):r for r in RAW}
DIAG=[]
for cut in CUTS:
    for arm in ("DC","NOCROSS"):
        gain=loss=matchj=matchi=ansj=0
        for wi in CASES:
            for slot in SLOTS:
                a=MAP[(arm,cut,wi,slot)];j=MAP[("JOINT",None,wi,slot)];i=MAP[("INDEP",None,wi,slot)]
                gain+=int(not i["ok"] and a["ok"]);loss+=int(i["ok"] and not a["ok"])
                matchj+=int(a["ok"]==j["ok"]);matchi+=int(a["ok"]==i["ok"]);ansj+=int(norm(a["answer"])==norm(j["answer"]))
        d=dict(arm=arm,cut=cut,gain=gain,loss=loss,net=gain-loss,ok_identity_joint=matchj,ok_identity_indep=matchi,answer_identity_joint=ansj)
        DIAG.append(d)
        print(f"{arm:7s} CUT={cut:02d} GAIN={gain:02d} LOSS={loss:02d} NET={gain-loss:+03d} OK-ID-JOINT={matchj:02d}/24 ANS-ID-JOINT={ansj:02d}/24")
print("\n[5/5] Final research record...")
WEIGHT_OK=sentinel()==S0;TRAINABLE=sum(int(p.requires_grad) for p in model.parameters())
INTEGRITY=len(JROWS)==24 and len(IROWS)==24 and len(RAW)==24*(2+2*len(CUTS)) and len({(r["arm"],r["cut"],r["case"],r["slot"]) for r in RAW})==len(RAW) and not ERRORS
DCROWS=[r for r in RESULTS if r["arm"]=="DC"];NCROWS=[r for r in RESULTS if r["arm"]=="NOCROSS"]
print("="*132);print("TEST566 — FINAL RESEARCH RECORD");print("="*132)
print("MODEL:",MODEL_ID,"GPU:",torch.cuda.get_device_name(0))
print("PANEL SHA:",PANEL_SHA,"INTEGRITY:","PASS" if INTEGRITY else "FAIL")
print(f"JOINT   FIRST={JS['FIRST']:02d}/8 MIDDLE={JS['MIDDLE']:02d}/8 LAST={JS['LAST']:02d}/8 TOTAL={sum(JS.values()):02d}/24")
print(f"INDEP   FIRST={IS['FIRST']:02d}/8 MIDDLE={IS['MIDDLE']:02d}/8 LAST={IS['LAST']:02d}/8 TOTAL={sum(IS.values()):02d}/24")
for cut in CUTS:
    for arm in ("DC","NOCROSS"):
        r=next(x for x in RESULTS if x["arm"]==arm and x["cut"]==cut);s=r["scores"]
        print(f"{arm:7s} CUT={cut:02d} FIRST={s['FIRST']:02d}/8 MIDDLE={s['MIDDLE']:02d}/8 LAST={s['LAST']:02d}/8 TOTAL={r['total']:02d}/24")
print("WEIGHT SENTINEL:","PASS" if WEIGHT_OK else "FAIL","TRAINABLE:",TRAINABLE)
print("TOTAL SECONDS:",round(time.perf_counter()-T0,2))
LOCK={"test":TEST,"model":MODEL_ID,"panel_sha":PANEL_SHA,"seed":SEED,"panel_seed":PANEL_SEED,"cuts":CUTS,"arms":ARMS,"joint":"native concatenated source prefill","indep":"native independently forged complete KV with RoPE rephase","dc":"TEST564 Hcut + early KV + append-only upper consolidation","nocross":"independent Hcut checkpoint upper reconstruction, no cross-cartridge prefill","training":False,"retrieval":False,"router":False}
FINAL={"lock":LOCK,"joint":JS,"indep":IS,"results":RESULTS,"diag":DIAG,"weight_ok":WEIGHT_OK,"trainable":TRAINABLE,"integrity":INTEGRITY}
print("LOCK SHA:",sha(LOCK));print("RESULT SHA:",sha(FINAL))
print("INTERPRETATION: INDEP is native isolated KV concatenation. NOCROSS is independently reconstructed checkpoint KV concatenation; both prohibit cross-cartridge attention during memory construction.")
print("CAUTION: NOCROSS is not an exact DC ablation: independent upper-layer reconstruction may also change local numerical states. Interpret DC-NOCROSS differences as a mechanism-screening result, not isolated proof of cross-attention necessity.")
print("CAUTION: Same panel as TEST564; this is calibration, not independent validation. Correctness alone does not establish numerical identity to JOINT.")
if not INTEGRITY or not WEIGHT_OK or TRAINABLE!=0:raise RuntimeError("FINAL INTEGRITY FAILURE")
print("DECISION: CAUSAL SCREEN COMPLETE — REVIEW DC vs NOCROSS vs INDEP BEFORE TEST567")
print("="*132)

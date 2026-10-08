# TEST552 — AKBASCORE MAM · VARAN 1 · CARTRIDGE-PARALLEL CO-TERMINAL ATTENTION
# TEST548 BASELINE PRESERVED: SOURCE READ ONCE → FROZEN NATIVE PKV
# 2×2 PHYSICS: SEQ-DUP / SEQ-SHR / COT-DUP / COT-SHR
# TARGET FIRST / MIDDLE / LAST · k=5
# NO RETRIEVAL · NO ROUTER · NO SCORER · NO TOP-K · NO THRESHOLD · NO TRAINING
import os,sys,subprocess,importlib.util,random,time,hashlib,json,re,math
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
try:from transformers.cache_utils import DynamicCache
except Exception:DynamicCache=None

TEST="552";SEED=552552;PANEL_SEED=550550;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
PARENT548="68e1594d7095cf7828286572a2ae6250bfcd5a7e3612e6db9c15a429bc4a8a87"
PARENT551="69a6c6275298ee028ad9399dca869128989778fd34c5ad450ec9dc0b6c49c08b"
DEVICE=torch.device("cuda");DTYPE=torch.bfloat16;MAX_NEW=16;KPOP=5
if not torch.cuda.is_available():raise RuntimeError("CUDA gerekli.")
os.environ["TOKENIZERS_PARALLELISM"]="false"
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*178)
print("TEST552 — AKBASCORE MAM · CARTRIDGE-PARALLEL CO-TERMINAL ATTENTION")
print("TEST548 FORGE PRESERVED · SEQ-DUP / SEQ-SHR / COT-DUP / COT-SHR · FIRST/MIDDLE/LAST")
print("NO RETRIEVAL · NO ROUTER · NO SCORER · NO TOP-K · NO THRESHOLD · NO TRAINING")
print("="*178);T0=time.perf_counter()

print("\n[1/10] Donmuş Mistral...")
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2])
DTARG="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DTARG:DTYPE}).eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None) or H//QH
if (NL,H,QH,KVH,HD)!=(32,4096,32,8,128):raise RuntimeError(f"Mimari uyuşmazlığı: {(NL,H,QH,KVH,HD)}")
ROPE_THETA=float(getattr(cfg,"rope_theta",10000.0))
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | Torch {torch.__version__} | Transformers {transformers.__version__}")
print(f"      {NL}L H={H} QH={QH} KVH={KVH} HD={HD} | rope_theta={ROPE_THETA:g} | {DTYPE} | trainable=0")

FP=[model.model.layers[0].self_attn.q_proj.weight,model.model.layers[8].self_attn.o_proj.weight,model.model.layers[16].mlp.down_proj.weight,model.model.layers[24].self_attn.o_proj.weight,model.model.layers[31].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
def sentinel():
    h=hashlib.sha256()
    for p in FP:
        a=p.detach().reshape(-1);n=a.numel()
        for j in range(16):
            o=j*(n-256)//15;h.update(a[o:o+256].float().cpu().numpy().tobytes())
    return h.hexdigest()
S0=sentinel()

BASE=[
("Zorvan","Melket","Dravel","Oakhaven","Pelnor"),("Kelvar","Nareth","Solven","Branik","Tarsen"),
("Tarev","Luneth","Varos","Cedran","Mireth"),("Belnor","Arven","Dorel","Kesmar","Falven"),
("Ravik","Selora","Terven","Maldor","Nerik"),("Nemor","Calven","Istral","Pareth","Dovren"),
("Darsen","Velora","Keldin","Orvek","Sarnel"),("Feron","Talven","Merith","Sovran","Belvik"),
("Larev","Nerith","Calder","Veyron","Tormek"),("Torven","Elsar","Marvek","Dorin","Kaleth"),
("Selnor","Kareth","Valen","Ordan","Mervek"),("Mirev","Taldor","Neris","Kelmar","Sorvik"),
("Varen","Solith","Deran","Malvek","Cordan"),("Kelor","Ardin","Velmar","Toren","Narell"),
("Narev","Belith","Corven","Sareth","Dorvik"),("Dervan","Mirel","Talvek","Orsen","Kelron"),
("Calnor","Verith","Naldor","Seren","Parvek"),("Parel","Dorven","Kelith","Maros","Tervik"),
("Sorven","Tarell","Vindor","Nelmar","Calrek"),("Barel","Corith","Laven","Derik","Solmar"),
("Ralen","Mervor","Talith","Kesven","Noreth"),("Norel","Valdor","Serith","Calvenor","Darvek"),
("Tervan","Orel","Mardin","Velos","Karven"),("Karev","Solen","Dareth","Mirven","Talrek")
]
W=[]
for i,(s,current,near,former,role_target) in enumerate(BASE):
    facts=[
        ("CURRENT",f"The current capital of {s} is {current}.",current),
        ("NEAR",f"The largest city of {s} is {near}.",near),
        ("FORMER",f"The former capital of {s} was {former}.",former),
        ("ROLE",f"The current capital of {role_target} is {s}.",s)
    ]
    queries=[
        ("CURRENT",f"What is the current capital of {s}?",current),
        ("FORMER",f"What was the former capital of {s}?",former),
        ("NEAR",f"What is the largest city of {s}?",near),
        ("ROLE",f"What is the current capital of {role_target}?",s)
    ]
    W.append({"id":i,"subject":s,"facts":facts,"queries":queries})

SYSTEM="Answer the question using only the information provided. Give only the requested name and nothing else."
def source_prefix(fact):return f"<s>[INST] {SYSTEM}\n\nINFORMATION:\n{fact}\n\n"
def canonical_prefix():return f"<s>[INST] {SYSTEM}\n\nINFORMATION:\n"
def query_suffix(q):return f"QUESTION:\n{q}\n\nANSWER: [/INST]"
def norm(s):
    s=s.casefold().strip();s=re.sub(r"[^\w\s-]"," ",s,flags=re.UNICODE);return " ".join(s.split())
def hit(out,gold):return re.search(r"(?<!\w)"+re.escape(norm(gold))+r"(?!\w)",norm(out)) is not None

def cache_layers(pkv):
    if hasattr(pkv,"layers"):
        out=[]
        for layer in pkv.layers:
            k=getattr(layer,"keys",None);v=getattr(layer,"values",None)
            if k is None:k=getattr(layer,"key_cache",None)
            if v is None:v=getattr(layer,"value_cache",None)
            if k is None or v is None:raise RuntimeError(f"DynamicCache layer API tanınmadı: {type(layer)}")
            out.append((k,v))
        if out:return tuple(out)
    if hasattr(pkv,"key_cache") and hasattr(pkv,"value_cache"):
        return tuple((k,v) for k,v in zip(pkv.key_cache,pkv.value_cache))
    if hasattr(pkv,"to_legacy_cache"):
        p=pkv.to_legacy_cache()
        if isinstance(p,(tuple,list)):return tuple(p)
    if isinstance(pkv,(tuple,list)):return tuple(pkv)
    raise RuntimeError(f"Desteklenmeyen cache tipi: {type(pkv)}")

def cache_len(pkv):
    if hasattr(pkv,"get_seq_length"):
        try:return int(pkv.get_seq_length())
        except TypeError:return int(pkv.get_seq_length(0))
    p=cache_layers(pkv)
    if not p:raise RuntimeError("Boş cache.")
    return int(p[0][0].shape[-2])

def clone_layers(pkv):
    return tuple((k.detach().clone(),v.detach().clone()) for k,v in cache_layers(pkv))

def build_cache(layers):
    if DynamicCache is None:raise RuntimeError("DynamicCache bulunamadı.")
    try:
        c=DynamicCache()
        for li,(k,v) in enumerate(layers):c.update(k,v,li)
        return c
    except Exception as e1:
        try:return DynamicCache(ddp_cache_data=[(k,v) for k,v in layers])
        except Exception as e2:raise RuntimeError(f"DynamicCache oluşturulamadı: update={e1!r} ddp={e2!r}")

@torch.no_grad()
def forge(fact):
    x=tok(source_prefix(fact),return_tensors="pt",add_special_tokens=False).to(DEVICE)
    out=model(**x,use_cache=True,return_dict=True)
    layers=clone_layers(out.past_key_values);n=int(x.input_ids.shape[1])
    if len(layers)!=NL:raise RuntimeError(f"Forge layer count FAIL: {len(layers)} != {NL}")
    for li,(k,v) in enumerate(layers):
        if k.ndim!=4 or v.ndim!=4:raise RuntimeError(f"L{li}: rank FAIL")
        if tuple(k.shape)!=tuple(v.shape):raise RuntimeError(f"L{li}: K/V shape FAIL")
        if k.shape[0]!=1 or k.shape[1]!=KVH or k.shape[-1]!=HD or k.shape[-2]!=n:
            raise RuntimeError(f"L{li}: forge shape FAIL K={tuple(k.shape)} V={tuple(v.shape)} n={n}")
    del out,x
    return layers,n

# HF/Mistral RoPE rotate_half convention.
def rotate_half(x):
    a=x[...,:HD//2];b=x[...,HD//2:]
    return torch.cat((-b,a),dim=-1)

def rope_delta(x,delta):
    if delta==0:return x
    inv=1.0/(ROPE_THETA**(torch.arange(0,HD,2,device=x.device,dtype=torch.float32)/HD))
    ang=float(delta)*inv
    c=torch.cat((ang.cos(),ang.cos()),dim=-1).to(dtype=x.dtype)
    s=torch.cat((ang.sin(),ang.sin()),dim=-1).to(dtype=x.dtype)
    while c.ndim<x.ndim:c=c.unsqueeze(0)
    while s.ndim<x.ndim:s=s.unsqueeze(0)
    return x*c+rotate_half(x)*s

PREFIX_IDS=tok(canonical_prefix(),add_special_tokens=False).input_ids
P=len(PREFIX_IDS)
if P<1:raise RuntimeError("Kanonik önek boş.")
print(f"      Canonical prefix tokens={P} | first_id={PREFIX_IDS[0]} | bos_id={tok.bos_token_id}")

LOCK={
"test":TEST,"parent548":PARENT548,"parent551":PARENT551,"model":MODEL_ID,
"seed":SEED,"panel_seed":PANEL_SEED,"k":KPOP,
"arms":["SEQ-DUP","SEQ-SHR","COT-DUP","COT-SHR"],
"target_positions":["FIRST","MIDDLE","LAST"],
"retrieval":False,"router":False,"scorer":False,"topk":False,
"threshold":False,"training":False
}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      LOCK SHA:",LOCK_SHA)

print("\n[2/10] TEST548 sözleşmesiyle 96 bağımsız cartridge forge...")
CAR={}
for wi,w in enumerate(W):
    print(f"      {wi+1:02d}/24 {w['subject']}")
    for typ,fact,gold in w["facts"]:
        ids=tok(source_prefix(fact),add_special_tokens=False).input_ids
        if ids[:P]!=PREFIX_IDS:raise RuntimeError(f"W{wi:02d} {typ}: canonical prefix token mismatch.")
        layers,n=forge(fact)
        CAR[(wi,typ)]={"layers":layers,"n":n,"body_len":n-P,"fact":fact,"gold":gold}

lens=[v["n"] for v in CAR.values()]
blens=[v["body_len"] for v in CAR.values()]
print(f"      Cartridges={len(CAR)} | full min/mean/max={min(lens)}/{np.mean(lens):.2f}/{max(lens)} | body min/mean/max={min(blens)}/{np.mean(blens):.2f}/{max(blens)}")
if min(blens)<1:raise RuntimeError("Body uzunluğu geçersiz.")

print("\n[3/10] Shared-prefix numerical identity audit...")
REF=next(iter(CAR.values()))["layers"]
PFX_MAX_K=PFX_MAX_V=0.0
for c in CAR.values():
    for li in range(NL):
        rk,rv=REF[li];k,v=c["layers"][li]
        PFX_MAX_K=max(PFX_MAX_K,float((rk[:,:,:P,:].float()-k[:,:,:P,:].float()).abs().max().item()))
        PFX_MAX_V=max(PFX_MAX_V,float((rv[:,:,:P,:].float()-v[:,:,:P,:].float()).abs().max().item()))
print(f"      Prefix K maxΔ={PFX_MAX_K:.8f} | V maxΔ={PFX_MAX_V:.8f}")
if PFX_MAX_K!=0.0 or PFX_MAX_V!=0.0:
    print("      NOT: Prefix tokenları aynı fakat sayısal prefix byte-identical değil; shared kol REF prefix kullanacak.")

rng=random.Random(PANEL_SEED)
PANEL=[]
for wi,w in enumerate(W):
    typ,q,gold=w["queries"][wi%4]
    target=(wi,typ)
    pool=[(wj,t) for wj in range(len(W)) if wj!=wi for t in ["CURRENT","FORMER","NEAR","ROLE"]]
    dist=rng.sample(pool,KPOP-1)
    PANEL.append({"world":wi,"type":typ,"q":q,"gold":gold,"target":target,"dist":dist})

PANEL_SHA=hashlib.sha256(json.dumps(PANEL,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      PANEL SHA:",PANEL_SHA)

def ordered_keys(case,pos):
    d=list(case["dist"])
    if pos=="FIRST":return [case["target"]]+d
    if pos=="MIDDLE":return d[:2]+[case["target"]]+d[2:]
    if pos=="LAST":return d+[case["target"]]
    raise ValueError(pos)

# DUP: complete independent cartridges.
# SHR: one canonical prefix + cartridge bodies.
# SEQ: ordinary sequential physical RoPE placement.
# COT: every cartridge/body ends at the same virtual terminal coordinate.
def compose(keys,shared,coterminal):
    cs=[CAR[k] for k in keys]
    body_lens=[c["body_len"] for c in cs]
    LSTAR=max(body_lens)
    out=[]
    for li in range(NL):
        partsK=[];partsV=[]
        if shared:
            pk,pv=REF[li]
            partsK.append(pk[:,:,:P,:]);partsV.append(pv[:,:,:P,:])
            if coterminal:
                for c in cs:
                    k,v=c["layers"][li]
                    delta=LSTAR-c["body_len"]
                    partsK.append(rope_delta(k[:,:,P:,:],delta))
                    partsV.append(v[:,:,P:,:])
            else:
                cursor=P
                for c in cs:
                    k,v=c["layers"][li]
                    delta=cursor-P
                    partsK.append(rope_delta(k[:,:,P:,:],delta))
                    partsV.append(v[:,:,P:,:])
                    cursor+=c["body_len"]
        else:
            if coterminal:
                NMAX=max(c["n"] for c in cs)
                for c in cs:
                    k,v=c["layers"][li]
                    delta=NMAX-c["n"]
                    partsK.append(rope_delta(k,delta))
                    partsV.append(v)
            else:
                cursor=0
                for c in cs:
                    k,v=c["layers"][li]
                    partsK.append(rope_delta(k,cursor))
                    partsV.append(v)
                    cursor+=c["n"]
        out.append((
            torch.cat(partsK,dim=-2).contiguous(),
            torch.cat(partsV,dim=-2).contiguous()
        ))
    if shared:
        virtual_q=P+(LSTAR if coterminal else sum(body_lens))
    else:
        virtual_q=max(c["n"] for c in cs) if coterminal else sum(c["n"] for c in cs)
    return tuple(out),int(virtual_q)

@torch.no_grad()
def answer_layers(layers,virtual_q,q):
    cache=build_cache(layers)
    physical=cache_len(cache)
    x=tok(query_suffix(q),return_tensors="pt",add_special_tokens=False).to(DEVICE)
    ids=x.input_ids;qlen=int(ids.shape[1])
    if qlen<1:raise RuntimeError("Boş query suffix.")
    att=torch.ones((1,physical+qlen),dtype=torch.long,device=DEVICE)
    cpos=torch.arange(physical,physical+qlen,dtype=torch.long,device=DEVICE)
    pos=torch.arange(virtual_q,virtual_q+qlen,dtype=torch.long,device=DEVICE).unsqueeze(0)
    out=model(
        input_ids=ids,
        past_key_values=cache,
        attention_mask=att,
        position_ids=pos,
        cache_position=cpos,
        use_cache=True,
        return_dict=True
    )
    cache=out.past_key_values
    if cache_len(cache)!=physical+qlen:
        raise RuntimeError(f"Initial cache growth FAIL: {cache_len(cache)} != {physical+qlen}")
    first_logits=out.logits[:,-1,:].float().detach().clone()
    nxt=out.logits[:,-1,:].argmax(dim=-1,keepdim=True)
    gen=[]
    eos=tok.eos_token_id
    eosset=set() if eos is None else set(int(z) for z in (eos if isinstance(eos,(list,tuple,set)) else [eos]))
    for step in range(MAX_NEW):
        gen.append(nxt)
        if int(nxt.item()) in eosset:break
        physical=cache_len(cache)
        att=torch.ones((1,physical+1),dtype=torch.long,device=DEVICE)
        cpos=torch.tensor([physical],dtype=torch.long,device=DEVICE)
        # Query suffix occupies virtual_q ... virtual_q+qlen-1.
        # The first generated token is therefore processed at virtual_q+qlen.
        pos=torch.tensor([[virtual_q+qlen+step]],dtype=torch.long,device=DEVICE)
        out=model(
            input_ids=nxt,
            past_key_values=cache,
            attention_mask=att,
            position_ids=pos,
            cache_position=cpos,
            use_cache=True,
            return_dict=True
        )
        cache=out.past_key_values
        if cache_len(cache)!=physical+1:
            raise RuntimeError(f"Generation cache growth FAIL: {cache_len(cache)} != {physical+1}")
        nxt=out.logits[:,-1,:].argmax(dim=-1,keepdim=True)
    g=torch.cat(gen,dim=1)
    ans=tok.decode(g[0],skip_special_tokens=True).strip()
    del x,out,g
    return ans,first_logits

print("\n[4/10] G0 — N=1 TEST548 reduction...")
G0=0;G0_EX=[]
for wi,w in enumerate(W):
    typ,q,gold=w["queries"][wi%4];key=(wi,typ)
    layers,vq=compose([key],shared=False,coterminal=True)
    ans,_=answer_layers(layers,vq,q)
    ok=hit(ans,gold);G0+=int(ok)
    if not ok and len(G0_EX)<8:G0_EX.append((wi,typ,gold,ans))
    del layers
print(f"      N=1 COT-DUP = {G0}/24")
if G0_EX:
    for z in G0_EX:print("      FAIL",z)

print("\n[5/10] 2×2 motor panel — k=5 × FIRST/MIDDLE/LAST...")
ARMS={
    "SEQ-DUP":(False,False),
    "SEQ-SHR":(True,False),
    "COT-DUP":(False,True),
    "COT-SHR":(True,True)
}
RES=[];LOGITS={}
for arm,(shared,cot) in ARMS.items():
    print(f"\n      [{arm}]")
    for pos in ["FIRST","MIDDLE","LAST"]:
        okn=0
        for ci,c in enumerate(PANEL):
            keys=ordered_keys(c,pos)
            layers,vq=compose(keys,shared,cot)
            ans,lg=answer_layers(layers,vq,c["q"])
            ok=hit(ans,c["gold"]);okn+=int(ok)
            RES.append({
                "arm":arm,"pos":pos,"case":ci,"world":c["world"],
                "type":c["type"],"gold":c["gold"],"answer":ans,"ok":ok
            })
            LOGITS[(arm,pos,ci)]=lg.cpu()
            del layers,lg
        print(f"      {pos:6s} {okn:02d}/24 = {okn/24:.6f}")

print("\n[6/10] G2 — permutation invariance...")
for arm in ARMS:
    same_fl=sum(
        norm(next(r for r in RES if r["arm"]==arm and r["pos"]=="FIRST" and r["case"]==i)["answer"])==
        norm(next(r for r in RES if r["arm"]==arm and r["pos"]=="LAST" and r["case"]==i)["answer"])
        for i in range(len(PANEL))
    )
    same_fm=sum(
        norm(next(r for r in RES if r["arm"]==arm and r["pos"]=="FIRST" and r["case"]==i)["answer"])==
        norm(next(r for r in RES if r["arm"]==arm and r["pos"]=="MIDDLE" and r["case"]==i)["answer"])
        for i in range(len(PANEL))
    )
    md_fl=max(
        float((LOGITS[(arm,"FIRST",i)]-LOGITS[(arm,"LAST",i)]).abs().max().item())
        for i in range(len(PANEL))
    )
    md_fm=max(
        float((LOGITS[(arm,"FIRST",i)]-LOGITS[(arm,"MIDDLE",i)]).abs().max().item())
        for i in range(len(PANEL))
    )
    print(f"      {arm:7s} SAME F↔M={same_fm:02d}/24 F↔L={same_fl:02d}/24 | maxlogitΔ F↔M={md_fm:.8f} F↔L={md_fl:.8f}")

print("\n[7/10] Faktöriyel özet...")
SUM={}
for arm in ARMS:
    SUM[arm]={}
    for pos in ["FIRST","MIDDLE","LAST"]:
        rr=[r for r in RES if r["arm"]==arm and r["pos"]==pos]
        n=sum(int(r["ok"]) for r in rr)
        SUM[arm][pos]=n
        print(f"      {arm:7s} {pos:6s}: {n:02d}/24")

print("\n      COT-SHR CASE-BY-CASE:")
for i,c in enumerate(PANEL):
    rr=[
        next(r for r in RES if r["arm"]=="COT-SHR" and r["pos"]==p and r["case"]==i)
        for p in ["FIRST","MIDDLE","LAST"]
    ]
    print(f"      {i:02d} W{c['world']:02d} {c['type']:8s} gold={c['gold']:10s} | F={rr[0]['answer']!r} | M={rr[1]['answer']!r} | L={rr[2]['answer']!r}")

print("\n[8/10] Yasak-mekanizma audit...")
print("      Source text replay at query : YOK")
print("      Candidate search / retrieval: YOK")
print("      Cartridge ID as address     : YOK")
print("      Cosine / ANN / top-k        : YOK")
print("      Router / classifier         : YOK")
print("      Relevance threshold         : YOK")
print("      Learned/fitted scorer       : YOK")
print("      Training / LoRA / optimizer : YOK")
print("      Query-dependent selection   : YOK")
print("      Only intervention           : physical memory geometry / shared-prefix factor")

print("\n[9/10] Sentinel / karar...")
S1=sentinel();SENT=S0==S1
cot=[r for r in RES if r["arm"]=="COT-SHR"]
cot_acc=sum(int(r["ok"]) for r in cot)/len(cot)
cot_same=sum(
    norm(next(r for r in RES if r["arm"]=="COT-SHR" and r["pos"]=="FIRST" and r["case"]==i)["answer"])==
    norm(next(r for r in RES if r["arm"]=="COT-SHR" and r["pos"]=="MIDDLE" and r["case"]==i)["answer"])==
    norm(next(r for r in RES if r["arm"]=="COT-SHR" and r["pos"]=="LAST" and r["case"]==i)["answer"])
    for i in range(len(PANEL))
)
G0_PASS=G0==24
G2_PASS=cot_same==len(PANEL)
MAIN_MIN=min(SUM["COT-SHR"].values())

if G0_PASS and G2_PASS and MAIN_MIN>=22:
    DECISION="TEST552_COTERMINAL_NATIVE_BINDING_PASS"
elif G0_PASS and G2_PASS:
    DECISION="TEST552_ORDER_REMOVED_BUT_NATIVE_BINDING_INSUFFICIENT"
else:
    DECISION="TEST552_OPERATOR_OR_GEOMETRY_GATE_FAIL"

FINAL={
    "test":TEST,"parent548":PARENT548,"parent551":PARENT551,
    "lock_sha":LOCK_SHA,"panel_sha":PANEL_SHA,"g0":G0,
    "summary":SUM,"cot_acc":cot_acc,"cot_same":cot_same,
    "sentinel":SENT,"decision":DECISION
}
RESULT_SHA=hashlib.sha256(json.dumps(FINAL,sort_keys=True,separators=(",",":")).encode()).hexdigest()

print("      G0 N=1 reduction            :",f"{G0}/24","PASS" if G0_PASS else "FAIL")
print("      COT-SHR permutation outputs :",f"{cot_same}/24","PASS" if G2_PASS else "FAIL")
print("      COT-SHR min positional acc  :",f"{MAIN_MIN}/24")
print("      Weight sentinel             :","PASS" if SENT else "FAIL")
print("      Trainable tensors           :",sum(int(p.requires_grad) for p in model.parameters()))
print("      DECISION                    :",DECISION)

print("\n[10/10] Sonuç...")
print("="*178)
print("TEST552 SONUÇ — CARTRIDGE-PARALLEL CO-TERMINAL ATTENTION")
print("="*178)
for arm in ARMS:
    print(f"{arm:8s} FIRST={SUM[arm]['FIRST']:02d}/24 MIDDLE={SUM[arm]['MIDDLE']:02d}/24 LAST={SUM[arm]['LAST']:02d}/24")
print("G0 N=1                       :",f"{G0}/24")
print("COT-SHR SAME F=M=L          :",f"{cot_same}/24")
print("WEIGHT SENTINEL             :","PASS" if SENT else "FAIL")
print("TEST548 RESULT SHA          :",PARENT548)
print("TEST551 RESULT SHA          :",PARENT551)
print("TEST552 ÖN KİLİT SHA        :",LOCK_SHA)
print("TEST552 PANEL SHA           :",PANEL_SHA)
print("TEST552 RESULT SHA          :",RESULT_SHA)
print("TOPLAM SÜRE                 :",f"{time.perf_counter()-T0:.2f}s")
print("KARAR                       :",DECISION)
print("="*178)

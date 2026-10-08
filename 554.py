# TEST554 — AKBASCORE MAM · CROSS-FACT CAUSALITY vs POSITIONAL GEOMETRY
# TEST553 WORKING FORGE/CACHE/ROTARY BASELINE PRESERVED
# T-JOINT / T-BLOCK-SHR / P-SEQ-SHR · FIRST/MIDDLE/LAST
# SAME TOKEN IDS · SAME LOGICAL POSITIONS · DIFFERENT WRITE-TIME CAUSAL ACCESS
# NO RETRIEVAL · NO ROUTER · NO SCORER · NO TOP-K · NO TRAINING · NO QSF
import os,sys,subprocess,importlib.util,random,time,hashlib,json,re
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

TEST="554";SEED=552552;PANEL_SEED=550550
MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
DEVICE=torch.device("cuda");DTYPE=torch.bfloat16
MAX_NEW=16;KPOP=5
FACT_TYPES=["CURRENT","NEAR","FORMER","ROLE"]
QUERY_CYCLE=["CURRENT","FORMER","NEAR","ROLE"]
ARMS=["T-JOINT","T-BLOCK-SHR","P-SEQ-SHR"]
POSITIONS=["FIRST","MIDDLE","LAST"]
if not torch.cuda.is_available():raise RuntimeError("CUDA gerekli.")
os.environ["TOKENIZERS_PARALLELISM"]="false"
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
T0=time.perf_counter()
print("="*170)
print("TEST554 — AKBASCORE MAM · CROSS-FACT CAUSALITY vs POSITIONAL GEOMETRY")
print("T-JOINT / T-BLOCK-SHR / P-SEQ-SHR · SAME TOKENS · SAME LOGICAL POSITIONS")
print("TEST553 NATIVE CARTRIDGE CONTRACT · NO RETRIEVAL · NO ROUTER · NO TRAINING · NO QSF")
print("="*170)

print("\n[1/12] Frozen Mistral...")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2])
DTARG="dtype" if tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DTARG:DTYPE}).eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None) or H//QH
if (NL,H,QH,KVH,HD)!=(32,4096,32,8,128):raise RuntimeError(f"Model architecture mismatch: {(NL,H,QH,KVH,HD)}")
print(f"MODEL={MODEL_ID} | GPU={torch.cuda.get_device_name(0)}")
print(f"Torch={torch.__version__} | Transformers={transformers.__version__}")
print(f"NL={NL} H={H} QH={QH} KVH={KVH} HD={HD} DTYPE={DTYPE}")
print("ROPE:",getattr(cfg,"rope_parameters",None))
print("TRAINABLE:",sum(int(p.requires_grad) for p in model.parameters()))

FP=[model.model.layers[0].self_attn.q_proj.weight,model.model.layers[8].self_attn.o_proj.weight,model.model.layers[16].mlp.down_proj.weight,model.model.layers[24].self_attn.o_proj.weight,model.model.layers[31].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
def sentinel():
    h=hashlib.sha256()
    for p in FP:
        a=p.detach().reshape(-1);n=a.numel()
        for j in range(16):
            o=j*(n-256)//15
            h.update(a[o:o+256].float().cpu().numpy().tobytes())
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
    s=s.casefold().strip()
    s=re.sub(r"[^\w\s-]"," ",s,flags=re.UNICODE)
    return " ".join(s.split())
def hit(out,gold):return re.search(r"(?<!\w)"+re.escape(norm(gold))+r"(?!\w)",norm(out)) is not None
def sha_obj(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

def cache_layers(pkv):
    if hasattr(pkv,"layers"):
        out=[]
        for layer in pkv.layers:
            k=getattr(layer,"keys",None);v=getattr(layer,"values",None)
            if k is None:k=getattr(layer,"key_cache",None)
            if v is None:v=getattr(layer,"value_cache",None)
            if k is None or v is None:raise RuntimeError("DynamicCache layer API mismatch.")
            out.append((k,v))
        if out:return tuple(out)
    if hasattr(pkv,"key_cache") and hasattr(pkv,"value_cache"):return tuple(zip(pkv.key_cache,pkv.value_cache))
    if hasattr(pkv,"to_legacy_cache"):
        z=pkv.to_legacy_cache()
        if isinstance(z,(tuple,list)):return tuple(z)
    if isinstance(pkv,(tuple,list)):return tuple(pkv)
    raise RuntimeError(f"Unsupported cache format: {type(pkv)}")

def cache_len(pkv):
    if hasattr(pkv,"get_seq_length"):
        try:return int(pkv.get_seq_length())
        except TypeError:return int(pkv.get_seq_length(0))
    return int(cache_layers(pkv)[0][0].shape[-2])

def clone_layers(pkv):return tuple((k.detach().clone(),v.detach().clone()) for k,v in cache_layers(pkv))

def new_cache_from_layers(layers):
    c=DynamicCache()
    for li,(k,v) in enumerate(layers):
        try:c.update(k,v,li)
        except TypeError:c.update(k,v,li,{})
    if cache_len(c)!=int(layers[0][0].shape[-2]):raise RuntimeError("DynamicCache length mismatch.")
    return c

def audit_layers(layers,n):
    if len(layers)!=NL:raise RuntimeError("Layer count mismatch.")
    for li,(k,v) in enumerate(layers):
        expected=(1,KVH,n,HD)
        if tuple(k.shape)!=expected or tuple(v.shape)!=expected:raise RuntimeError(f"L{li} shape mismatch K={tuple(k.shape)} V={tuple(v.shape)} expected={expected}")
    return True

def raw_bytes(x):
    x=x.detach().contiguous().cpu()
    if x.dtype==torch.bfloat16:x=x.view(torch.uint16)
    return x.numpy().tobytes()

def tensor_sha(layers):
    h=hashlib.sha256()
    for k,v in layers:h.update(raw_bytes(k));h.update(raw_bytes(v))
    return h.hexdigest()

@torch.no_grad()
def forward_ids(ids,cache=None,virtual_start=0):
    ids=ids.to(DEVICE).reshape(1,-1);n=int(ids.shape[1])
    physical=0 if cache is None else cache_len(cache)
    cp=torch.arange(physical,physical+n,dtype=torch.long,device=DEVICE)
    pos=torch.arange(virtual_start,virtual_start+n,dtype=torch.long,device=DEVICE).unsqueeze(0)
    att=torch.ones((1,physical+n),dtype=torch.long,device=DEVICE)
    return model(input_ids=ids,past_key_values=cache,attention_mask=att,cache_position=cp,position_ids=pos,use_cache=True,return_dict=True)

@torch.no_grad()
def forge(fact):
    ids=torch.tensor(tok(source_prefix(fact),add_special_tokens=False).input_ids,dtype=torch.long,device=DEVICE)
    out=forward_ids(ids,None,0)
    layers=clone_layers(out.past_key_values)
    audit_layers(layers,len(ids))
    del out
    return layers,len(ids)

@torch.no_grad()
def direct_forge_at_offset(fact,offset):
    ids=torch.tensor(tok(source_prefix(fact),add_special_tokens=False).input_ids,dtype=torch.long,device=DEVICE)
    n=len(ids);pos=torch.arange(offset,offset+n,dtype=torch.long,device=DEVICE).unsqueeze(0)
    cp=torch.arange(n,dtype=torch.long,device=DEVICE)
    att=torch.ones((1,n),dtype=torch.long,device=DEVICE)
    out=model(input_ids=ids.reshape(1,-1),attention_mask=att,position_ids=pos,cache_position=cp,use_cache=True,return_dict=True)
    layers=clone_layers(out.past_key_values)
    audit_layers(layers,n)
    del out
    return layers,n

def rotate_half(x):
    a=x[...,:x.shape[-1]//2];b=x[...,x.shape[-1]//2:]
    return torch.cat((-b,a),dim=-1)

@torch.no_grad()
def rope_cos_sin(positions):
    rotary=model.model.rotary_emb
    pos=positions.reshape(1,-1).to(device=DEVICE,dtype=torch.long)
    dummy=torch.zeros((1,pos.shape[1],H),device=DEVICE,dtype=DTYPE)
    try:cos,sin=rotary(dummy,pos)
    except TypeError:cos,sin=rotary(dummy,position_ids=pos)
    if cos.ndim==2:cos=cos.unsqueeze(0);sin=sin.unsqueeze(0)
    if cos.ndim==3:cos=cos.unsqueeze(1);sin=sin.unsqueeze(1)
    return cos.float(),sin.float()

@torch.no_grad()
def rope_rebase_k(k,old_start,new_start):
    if old_start==new_start:return k.clone()
    n=int(k.shape[-2])
    old=torch.arange(old_start,old_start+n,dtype=torch.long,device=DEVICE)
    new=torch.arange(new_start,new_start+n,dtype=torch.long,device=DEVICE)
    co,so=rope_cos_sin(old);cn,sn=rope_cos_sin(new);x=k.float()
    den=co.square()+so.square()
    raw=(x*co-rotate_half(x)*so)/den
    return (raw*cn+rotate_half(raw)*sn).to(k.dtype)

@torch.no_grad()
def answer_from_layers(layers,q,virtual_q):
    cache=new_cache_from_layers(layers)
    ids=torch.tensor(tok(query_suffix(q),add_special_tokens=False).input_ids,dtype=torch.long,device=DEVICE)
    qlen=len(ids)
    out=forward_ids(ids,cache,virtual_q)
    nxt=out.logits[:,-1,:].argmax(-1,keepdim=True)
    cache=out.past_key_values
    eos=tok.eos_token_id
    eosset=set() if eos is None else ({int(eos)} if not isinstance(eos,(tuple,list,set)) else set(map(int,eos)))
    gen=[]
    for step in range(MAX_NEW):
        gen.append(nxt)
        if int(nxt.item()) in eosset:break
        out=forward_ids(nxt.reshape(-1),cache,virtual_q+qlen+step)
        cache=out.past_key_values
        nxt=out.logits[:,-1,:].argmax(-1,keepdim=True)
    ans="" if not gen else tok.decode(torch.cat(gen,dim=1)[0],skip_special_tokens=True).strip()
    del out
    return ans

print("\n[2/12] TEST553 bağımsız kartuşları hazırlanıyor...")
CAR={};IDX_TO_KEY=[];KEY_TO_IDX={}
idx=0
for wi,w in enumerate(W):
    for typ,fact,gold in w["facts"]:
        layers,n=forge(fact)
        key=(wi,typ)
        CAR[key]={"idx":idx,"layers":layers,"n":n,"fact":fact,"gold":gold,"sha":tensor_sha(layers)}
        IDX_TO_KEY.append(key);KEY_TO_IDX[key]=idx;idx+=1
    print(f"      {wi+1:02d}/24 {w['subject']}")
if len(CAR)!=96 or len(IDX_TO_KEY)!=96:raise RuntimeError("Cartridge count mismatch.")
print("CARTRIDGES:",len(CAR))

print("\n[3/12] Canonical prefix / body token identity...")
prefix_ids=tok(canonical_prefix(),add_special_tokens=False).input_ids
P=len(prefix_ids)
PREFIX=tuple((k[:,:,:P,:].clone(),v[:,:,:P,:].clone()) for k,v in CAR[(0,"CURRENT")]["layers"])
audit_layers(PREFIX,P)
BODY={};prefix_max=0.0
for key,c in CAR.items():
    ids=tok(source_prefix(c["fact"]),add_special_tokens=False).input_ids
    if ids[:P]!=prefix_ids:raise RuntimeError(f"Prefix token mismatch: {key}")
    BODY[key]=torch.tensor(ids[P:],dtype=torch.long,device=DEVICE)
    if len(BODY[key])!=c["n"]-P:raise RuntimeError(f"Body length mismatch: {key}")
    for li,(k,v) in enumerate(c["layers"]):
        pk,pv=PREFIX[li]
        prefix_max=max(prefix_max,float((pk.float()-k[:,:,:P,:].float()).abs().max()),float((pv.float()-v[:,:,:P,:].float()).abs().max()))
print(f"PREFIX TOKENS={P} | BODY LENGTH={min(len(v) for v in BODY.values())}..{max(len(v) for v in BODY.values())}")
print(f"PREFIX MAX NUMERICAL DELTA={prefix_max:.8f}")
if prefix_max!=0.0:raise RuntimeError("Shared-prefix numerical identity failed.")

print("\n[4/12] TEST553 panelinin birebir kurulması...")
rng2=random.Random(PANEL_SEED);pool=list(range(96));rng2.shuffle(pool)
PANEL_RAW=[];cursor=0
for wi in range(24):
    typ=QUERY_CYCLE[wi%4]
    target=KEY_TO_IDX[(wi,typ)]
    others=[]
    while len(others)<4:
        ci=pool[cursor%96];cursor+=1
        if ci==target or IDX_TO_KEY[ci][0]==wi or any(IDX_TO_KEY[x][0]==IDX_TO_KEY[ci][0] for x in others):continue
        others.append(ci)
    PANEL_RAW.append({"case":wi,"target":target,"others":others})
PANEL_SHA=hashlib.sha256(json.dumps(PANEL_RAW,sort_keys=True,separators=(",",":")).encode()).hexdigest()
EXPECTED_553="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
print("TEST553 PANEL SHA:",PANEL_SHA)
print("EXPECTED          :",EXPECTED_553)
if PANEL_SHA!=EXPECTED_553:raise RuntimeError("TEST553 panel SHA mismatch; experiment stopped.")
PANEL=[]
for z in PANEL_RAW:
    target=IDX_TO_KEY[z["target"]];others=[IDX_TO_KEY[i] for i in z["others"]]
    wi,typ=target
    q=next(x[1] for x in W[wi]["queries"] if x[0]==typ)
    PANEL.append({"case":z["case"],"target":target,"others":others,"q":q,"gold":CAR[target]["gold"]})
print("PANEL PROVENANCE: PASS — exact TEST553 reconstructed panel")

def order_keys(z,slot):
    d=list(z["others"]);t=z["target"]
    if slot=="FIRST":return [t]+d
    if slot=="MIDDLE":return d[:2]+[t]+d[2:]
    if slot=="LAST":return d+[t]
    raise ValueError(slot)

@torch.no_grad()
def joint_layers(keys):
    ids=torch.cat([torch.tensor(prefix_ids,dtype=torch.long,device=DEVICE)]+[BODY[k] for k in keys])
    out=forward_ids(ids,None,0)
    layers=clone_layers(out.past_key_values)
    audit_layers(layers,len(ids))
    del out
    return layers,len(ids)

@torch.no_grad()
def block_body_direct(key,start):
    body=BODY[key]
    prefix_cache=new_cache_from_layers(PREFIX)
    out=forward_ids(body,prefix_cache,start)
    all_layers=clone_layers(out.past_key_values)
    n=len(body)
    sliced=tuple((k[:,:,P:,:].clone(),v[:,:,P:,:].clone()) for k,v in all_layers)
    for li,(k,v) in enumerate(sliced):
        if tuple(k.shape)!=(1,KVH,n,HD) or tuple(v.shape)!=(1,KVH,n,HD):raise RuntimeError(f"Direct block shape mismatch L{li}")
    del out,prefix_cache,all_layers
    return sliced,n

@torch.no_grad()
def block_body_rebased(key,start):
    c=CAR[key];n=len(BODY[key])
    layers=tuple((rope_rebase_k(k[:,:,P:,:],P,start),v[:,:,P:,:].clone()) for k,v in c["layers"])
    for li,(k,v) in enumerate(layers):
        if tuple(k.shape)!=(1,KVH,n,HD) or tuple(v.shape)!=(1,KVH,n,HD):raise RuntimeError(f"Rebased block shape mismatch L{li}")
    return layers,n

@torch.no_grad()
def compose_independent(keys,mode):
    starts=[];cursor=P
    for key in keys:
        starts.append(cursor);cursor+=len(BODY[key])
    banks=[]
    for key,start in zip(keys,starts):
        if mode=="T-BLOCK-SHR":banks.append(block_body_direct(key,start)[0])
        elif mode=="P-SEQ-SHR":banks.append(block_body_rebased(key,start)[0])
        else:raise ValueError(mode)
    layers=[]
    for li in range(NL):
        pk,pv=PREFIX[li]
        layers.append((torch.cat([pk]+[b[li][0] for b in banks],dim=-2).contiguous(),
                       torch.cat([pv]+[b[li][1] for b in banks],dim=-2).contiguous()))
    layers=tuple(layers);audit_layers(layers,cursor)
    return layers,cursor

print("\n[5/12] G-R — TEST553 model-native full-sequence RoPE doğrulaması...")
probe=(0,"CURRENT");base=CAR[probe]["layers"];GR=[]
for delta in [1,2,7,31]:
    direct,n=direct_forge_at_offset(CAR[probe]["fact"],delta)
    kd=[];vd=[]
    for li in range(NL):
        kr=rope_rebase_k(base[li][0],0,delta)
        kd.append(float((kr.float()-direct[li][0].float()).abs().max()))
        vd.append(float((base[li][1].float()-direct[li][1].float()).abs().max()))
    mk=max(kd);mv=max(vd)
    GR.append({"delta":delta,"kmax":mk,"vmax":mv})
    print(f"δ={delta:2d} K maxΔ={mk:.8f} V maxΔ={mv:.8f}")
    del direct
GR_OK=all(x["kmax"]<=0.5 and x["vmax"]<=0.5 for x in GR)
print("G-R:", "PASS" if GR_OK else "FAIL")
if not GR_OK:raise RuntimeError("Model-native RoPE validation failed.")

print("\n[6/12] G0 — N=1 TEST548 reduction...")
G0={a:0 for a in ARMS};G0_EQUAL=0
for z in PANEL:
    key=z["target"];outs={}
    for arm in ARMS:
        if arm=="T-JOINT":layers,vq=joint_layers([key])
        else:layers,vq=compose_independent([key],arm)
        ans=answer_from_layers(layers,z["q"],vq)
        outs[arm]=ans;G0[arm]+=int(hit(ans,z["gold"]))
        del layers
    G0_EQUAL+=int(len(set(norm(v) for v in outs.values()))==1)
print("G0:",G0,"| SAME OUTPUT:",G0_EQUAL,"/24")
if min(G0.values())<24 or G0_EQUAL<24:raise RuntimeError("G0 N=1 reduction failed. Multi-memory result must not be interpreted.")

print("\n[7/12] MAIN MINIMAL PAIR — 24 × 3 positions × 3 arms...")
RESULTS=[]
for ci,z in enumerate(PANEL):
    print(f"\nCASE {ci:02d} W{z['target'][0]:02d} {z['target'][1]:8s} GOLD={z['gold']} QUERY={z['q']!r}")
    for slot in POSITIONS:
        keys=order_keys(z,slot)
        row={"case":ci,"slot":slot,"target":str(z["target"]),"gold":z["gold"]}
        for arm in ARMS:
            if arm=="T-JOINT":layers,vq=joint_layers(keys)
            else:layers,vq=compose_independent(keys,arm)
            ans=answer_from_layers(layers,z["q"],vq)
            row[arm]={"answer":ans,"ok":bool(hit(ans,z["gold"]))}
            del layers
        RESULTS.append(row)
        print(f"  {slot:6s} | JOINT={row['T-JOINT']['answer']!r} | BLOCK={row['T-BLOCK-SHR']['answer']!r} | PKV={row['P-SEQ-SHR']['answer']!r}")

print("\n[8/12] Accuracy / permutation / behavioral identity...")
SUMMARY={}
for arm in ARMS:
    SUMMARY[arm]={}
    for slot in POSITIONS:
        rr=[r for r in RESULTS if r["slot"]==slot]
        n=sum(int(r[arm]["ok"]) for r in rr)
        SUMMARY[arm][slot]=n
        print(f"{arm:12s} {slot:6s}: {n:02d}/24")
    same=0
    for i in range(24):
        outs=[next(r for r in RESULTS if r["case"]==i and r["slot"]==slot)[arm]["answer"] for slot in POSITIONS]
        same+=int(len(set(norm(x) for x in outs))==1)
    SUMMARY[arm]["SAME_FML"]=same
    print(f"{arm:12s} SAME F=M=L: {same}/24")
IDENTITY={}
for slot in POSITIONS:
    rr=[r for r in RESULTS if r["slot"]==slot]
    n=sum(norm(r["T-BLOCK-SHR"]["answer"])==norm(r["P-SEQ-SHR"]["answer"]) for r in rr)
    IDENTITY[slot]=n
    print(f"BLOCK↔PKV {slot:6s}: {n}/24")

print("\n[9/12] ZERO-RELEVANT CONTROL — same four foreign-world distractors only...")
ZERO=[]
for ci,z in enumerate(PANEL):
    base=list(z["others"])
    zero_orders={"FIRST":base,"MIDDLE":base[2:]+base[:2],"LAST":list(reversed(base))}
    for slot in POSITIONS:
        keys=zero_orders[slot]
        row={"case":ci,"slot":slot,"gold":z["gold"]}
        for arm in ARMS:
            if arm=="T-JOINT":layers,vq=joint_layers(keys)
            else:layers,vq=compose_independent(keys,arm)
            ans=answer_from_layers(layers,z["q"],vq)
            row[arm]=ans;del layers
        ZERO.append(row)
    if (ci+1)%6==0:print(f"ZERO {ci+1}/24 completed")
REFUSAL_TERMS=["not provided","not given","not specified","no information","does not provide","does not specify","does not mention","cannot determine","cannot answer","insufficient","unknown","not available","not mentioned","does not contain","without information","no details"]
def refusal_like(s):
    n=norm(s)
    return any(x in n for x in REFUSAL_TERMS)
for arm in ARMS:
    vals=[r[arm] for r in ZERO]
    refusals=sum(refusal_like(x) for x in vals)
    gold_leak=sum(hit(r[arm],r["gold"]) for r in ZERO)
    print(f"{arm:12s} ZERO refusal-like={refusals}/72 | target-gold mentions={gold_leak}/72")
print("NOTE: refusal-like is diagnostic only; it is NOT used for the main mechanism decision.")

print("\n[10/12] COUNTERFACTUAL TARGET VALUE — 6-case diagnostic...")
CF=[]
CF_CASES=[0,1,2,3,4,5]
for ci in CF_CASES:
    z=PANEL[ci];key=z["target"];old=CAR[key]["gold"]
    replacement=f"Xevora{ci:02d}"
    fact=CAR[key]["fact"]
    if old not in fact:raise RuntimeError(f"Counterfactual replacement source mismatch: case {ci}")
    changed=fact.replace(old,replacement,1)
    if changed==fact:raise RuntimeError("Counterfactual text unchanged.")
    new_layers,new_n=forge(changed)
    new_ids=tok(source_prefix(changed),add_special_tokens=False).input_ids
    if new_ids[:P]!=prefix_ids:raise RuntimeError("Counterfactual prefix mismatch.")
    new_body=torch.tensor(new_ids[P:],dtype=torch.long,device=DEVICE)
    original_car=CAR[key];original_body=BODY[key]
    CAR[key]={"idx":original_car["idx"],"layers":new_layers,"n":new_n,"fact":changed,"gold":replacement,"sha":tensor_sha(new_layers)}
    BODY[key]=new_body
    try:
        for arm in ARMS:
            keys=order_keys(z,"MIDDLE")
            if arm=="T-JOINT":layers,vq=joint_layers(keys)
            else:layers,vq=compose_independent(keys,arm)
            ans=answer_from_layers(layers,z["q"],vq)
            CF.append({"case":ci,"type":key[1],"arm":arm,"old":old,"new":replacement,"answer":ans,"new_ok":bool(hit(ans,replacement)),"old_leak":bool(hit(ans,old))})
            del layers
    finally:
        CAR[key]=original_car;BODY[key]=original_body
    print(f"CF CASE {ci:02d} {key[1]:8s} {old}→{replacement}: "+" | ".join(f"{r['arm']}={r['answer']!r}" for r in CF if r["case"]==ci))
for arm in ARMS:
    rows=[r for r in CF if r["arm"]==arm]
    print(f"{arm:12s} CF new-value={sum(r['new_ok'] for r in rows)}/6 | old-value={sum(r['old_leak'] for r in rows)}/6")

print("\n[11/12] Predeclared mechanism decision...")
GID_OK=all(IDENTITY[s]>=22 for s in POSITIONS)
JOINT_OK=all(SUMMARY["T-JOINT"][s]>=22 for s in POSITIONS)
BLOCK_GOOD=all(SUMMARY["T-BLOCK-SHR"][s]>=20 for s in POSITIONS)
BLOCK_BAD=any(SUMMARY["T-BLOCK-SHR"][s]<=12 for s in POSITIONS)
BLOCK_ORDER=abs(SUMMARY["T-BLOCK-SHR"]["FIRST"]-SUMMARY["T-BLOCK-SHR"]["LAST"])>=6
if not JOINT_OK:
    DECISION="INCONCLUSIVE — JOINT CONTROL BELOW 22/24"
elif not GID_OK:
    DECISION="INCONCLUSIVE — BLOCK/PKV BEHAVIORAL IDENTITY FAILURE"
elif BLOCK_GOOD:
    DECISION="H-POS SUPPORTED — STATIC SEQUENTIAL GEOMETRY SURVIVES"
elif BLOCK_BAD or BLOCK_ORDER:
    DECISION="H-CROSS SUPPORTED — WRITE-TIME CROSS-FACT CAUSAL COMPUTATION MATTERS"
else:
    DECISION="MIXED — NO ARCHITECTURAL CLAIM"
print("JOINT CONTROL :", "PASS" if JOINT_OK else "FAIL")
print("G-ID          :", "PASS" if GID_OK else "FAIL")
print("BLOCK GOOD    :",BLOCK_GOOD)
print("BLOCK BAD     :",BLOCK_BAD)
print("BLOCK ORDER   :",BLOCK_ORDER)
print("DECISION      :",DECISION)

print("\n[12/12] Anti-drift / cartridge seal / weight sentinel...")
SEAL_OK=all(tensor_sha(c["layers"])==c["sha"] for c in CAR.values())
WEIGHT_OK=sentinel()==S0
TRAINABLE=sum(int(p.requires_grad) for p in model.parameters())
print("SOURCE REPLAY AT QUERY : P-SEQ-SHR NO; T-JOINT/T-BLOCK-SHR ARE TEXT-DERIVED CONTROLS")
print("RETRIEVAL              : NO")
print("ROUTER                 : NO")
print("COSINE / ANN / TOP-K   : NO")
print("EXTERNAL SCORER        : NO")
print("TRAINING / LORA        : NO")
print("QSF                    : NO")
print("MODEL-NATIVE RoPE      : YES")
print("CARTRIDGE SEAL         :", "PASS" if SEAL_OK else "FAIL")
print("WEIGHT SENTINEL        :", "PASS" if WEIGHT_OK else "FAIL")
print("TRAINABLE              :",TRAINABLE)

LOCK={
    "test":TEST,"model":MODEL_ID,"panel_sha":PANEL_SHA,"panel_source":"TEST553 exact reconstructed panel",
    "arms":ARMS,"positions":POSITIONS,"k":KPOP,"prefix_tokens":P,"native_rope":True,
    "fact_index_order":FACT_TYPES,"query_cycle":QUERY_CYCLE,
    "same_token_ids":True,"same_logical_positions":True,
    "cross_body_visibility":{"T-JOINT":True,"T-BLOCK-SHR":False,"P-SEQ-SHR":False},
    "training":False,"retrieval":False,"router":False,"qsf":False
}
LOCK_SHA=sha_obj(LOCK)
FINAL={
    "test":TEST,"lock_sha":LOCK_SHA,"panel_sha":PANEL_SHA,"gr":GR,"gr_ok":GR_OK,
    "g0":G0,"g0_equal":G0_EQUAL,"summary":SUMMARY,"identity":IDENTITY,
    "zero":ZERO,"counterfactual":CF,"decision":DECISION,
    "cartridge_seal":SEAL_OK,"weight_sentinel":WEIGHT_OK,"trainable":TRAINABLE
}
RESULT_SHA=sha_obj(FINAL)

print("\n"+"="*170)
print("TEST554 — FINAL RESEARCH RECORD")
print("="*170)
print("MODEL                    :",MODEL_ID)
print("PANEL SHA                :",PANEL_SHA)
print("LOCK SHA                 :",LOCK_SHA)
print("RESULT SHA               :",RESULT_SHA)
print("G-R                      :","PASS" if GR_OK else "FAIL")
print("G0                       :",G0)
print("G0 SAME OUTPUT           :",f"{G0_EQUAL}/24")
for arm in ARMS:
    s=SUMMARY[arm]
    print(f"{arm:24s}: FIRST={s['FIRST']:02d}/24 MIDDLE={s['MIDDLE']:02d}/24 LAST={s['LAST']:02d}/24 SAME={s['SAME_FML']:02d}/24")
print("BLOCK↔PKV IDENTITY       :",IDENTITY)
print("JOINT CONTROL            :","PASS" if JOINT_OK else "FAIL")
print("G-ID                     :","PASS" if GID_OK else "FAIL")
print("CARTRIDGE SEAL           :","PASS" if SEAL_OK else "FAIL")
print("WEIGHT SENTINEL          :","PASS" if WEIGHT_OK else "FAIL")
print("TRAINABLE                :",TRAINABLE)
print("TOTAL TIME               :",f"{time.perf_counter()-T0:.2f}s")
print("RESEARCH DECISION        :",DECISION)
print("NEXT TEST                : TEST555 — ONLY AFTER TEST554 INTERPRETATION")
print("="*170)
if not SEAL_OK or not WEIGHT_OK or TRAINABLE:raise RuntimeError("ANTI-DRIFT FAILURE — RESULTS INVALID")

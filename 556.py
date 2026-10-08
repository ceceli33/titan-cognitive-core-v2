# TEST556 — AKBASCORE MAM · LAYERWISE CAUSAL STATE TRANSPLANT — CORRECTED REPLICATION
# TEST555 ARCHITECTURE PRESERVED
# FIXES:
#   1) FULL/TARGET/HISTORY aggregation is filtered by BOTH layer AND scope.
#   2) Per-layer row cardinality is hard-gated at 72.
#   3) Scope identity is measured at case/slot outcome level, not only aggregate score level.
#   4) No dependence on TEST555 notebook RAM/state.
# JOINT / BLOCK / SWITCH-ONLY / FULL / TARGET / HISTORY
# FROZEN MISTRAL · SAME TOKENS · SAME LOGICAL POSITIONS
# NO TRAINING · NO RETRIEVAL · NO ROUTER · NO QSF

import os,sys,subprocess,importlib.util,random,time,hashlib,json,re,inspect,gc
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(m) is None:
        subprocess.check_call([sys.executable,"-m","pip","install","-q",p])

import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

TEST="556"
SEED=552552
PANEL_SEED=550550
MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
DEVICE=torch.device("cuda")
DTYPE=torch.bfloat16
MAX_NEW=16
NL_EXPECT=32
KPOP=5
FACT_TYPES=["CURRENT","NEAR","FORMER","ROLE"]
QUERY_CYCLE=["CURRENT","FORMER","NEAR","ROLE"]
POSITIONS=["FIRST","MIDDLE","LAST"]
SCOPES=["FULL","TARGET","HISTORY"]
LAYERS=list(range(32))
EXPECTED_PANEL_SHA="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA gerekli.")

os.environ["TOKENIZERS_PARALLELISM"]="false"
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
T0=time.perf_counter()

def sha_obj(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def norm(s):
    s=re.sub(r"[^\w\s-]"," ",s.casefold().strip(),flags=re.UNICODE)
    return " ".join(s.split())

def hit(s,g):
    return re.search(r"(?<!\w)"+re.escape(norm(g))+r"(?!\w)",norm(s)) is not None

print("="*168)
print("TEST556 — AKBASCORE MAM · LAYERWISE CAUSAL STATE TRANSPLANT — CORRECTED REPLICATION")
print("JOINT / BLOCK / SWITCH-ONLY / FULL / TARGET / HISTORY")
print("TEST555 ARCHITECTURE PRESERVED · CORRECT SCOPE AGGREGATION · CASE-LEVEL IDENTITY")
print("FROZEN MISTRAL · SAME TOKENS · SAME LOGICAL POSITIONS · NO TRAINING")
print("="*168)

print("\n[1/11] Frozen Mistral...")
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map={"":0},
    attn_implementation="sdpa",
    dtype=DTYPE
).eval()
for p in model.parameters():
    p.requires_grad_(False)

cfg=model.config
NL=len(model.model.layers)
H=cfg.hidden_size
QH=cfg.num_attention_heads
KVH=cfg.num_key_value_heads
HD=getattr(cfg,"head_dim",None) or H//QH

if (NL,H,QH,KVH,HD)!=(32,4096,32,8,128):
    raise RuntimeError(f"Architecture mismatch: {(NL,H,QH,KVH,HD)}")

print("MODEL:",MODEL_ID)
print("GPU:",torch.cuda.get_device_name(0))
print("TORCH:",torch.__version__,"TRANSFORMERS:",transformers.__version__)
print("NL=%d H=%d QH=%d KVH=%d HD=%d DTYPE=%s"%(NL,H,QH,KVH,HD,DTYPE))
print("ROPE:",getattr(cfg,"rope_parameters",None))
print("TRAINABLE:",sum(int(p.requires_grad) for p in model.parameters()))
print("DECODER SIGNATURE:",inspect.signature(model.model.layers[0].forward))

FP=[
    model.model.layers[0].self_attn.q_proj.weight,
    model.model.layers[8].self_attn.o_proj.weight,
    model.model.layers[16].mlp.down_proj.weight,
    model.model.layers[24].self_attn.o_proj.weight,
    model.model.layers[31].mlp.down_proj.weight,
    model.model.norm.weight,
    model.lm_head.weight
]

def sentinel():
    h=hashlib.sha256()
    for p in FP:
        a=p.detach().reshape(-1)
        n=a.numel()
        for j in range(16):
            o=j*(n-256)//15
            h.update(a[o:o+256].float().cpu().numpy().tobytes())
    return h.hexdigest()

S0=sentinel()

BASE=[
("Zorvan","Melket","Dravel","Oakhaven","Pelnor"),
("Kelvar","Nareth","Solven","Branik","Tarsen"),
("Tarev","Luneth","Varos","Cedran","Mireth"),
("Belnor","Arven","Dorel","Kesmar","Falven"),
("Ravik","Selora","Terven","Maldor","Nerik"),
("Nemor","Calven","Istral","Pareth","Dovren"),
("Darsen","Velora","Keldin","Orvek","Sarnel"),
("Feron","Talven","Merith","Sovran","Belvik"),
("Larev","Nerith","Calder","Veyron","Tormek"),
("Torven","Elsar","Marvek","Dorin","Kaleth"),
("Selnor","Kareth","Valen","Ordan","Mervek"),
("Mirev","Taldor","Neris","Kelmar","Sorvik"),
("Varen","Solith","Deran","Malvek","Cordan"),
("Kelor","Ardin","Velmar","Toren","Narell"),
("Narev","Belith","Corven","Sareth","Dorvik"),
("Dervan","Mirel","Talvek","Orsen","Kelron"),
("Calnor","Verith","Naldor","Seren","Parvek"),
("Parel","Dorven","Kelith","Maros","Tervik"),
("Sorven","Tarell","Vindor","Nelmar","Calrek"),
("Barel","Corith","Laven","Derik","Solmar"),
("Ralen","Mervor","Talith","Kesven","Noreth"),
("Norel","Valdor","Serith","Calvenor","Darvek"),
("Tervan","Orel","Mardin","Velos","Karven"),
("Karev","Solen","Dareth","Mirven","Talrek")
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
    W.append({"id":i,"facts":facts,"queries":queries})

SYSTEM="Answer the question using only the information provided. Give only the requested name and nothing else."

def source_prefix(f):
    return f"<s>[INST] {SYSTEM}\n\nINFORMATION:\n{f}\n\n"

def canonical_prefix():
    return f"<s>[INST] {SYSTEM}\n\nINFORMATION:\n"

def query_suffix(q):
    return f"QUESTION:\n{q}\n\nANSWER: [/INST]"

PREFIX_IDS=tok(canonical_prefix(),add_special_tokens=False).input_ids
P=len(PREFIX_IDS)

CAR=[]
KEY_TO_IDX={}
for wi,w in enumerate(W):
    for typ,fact,gold in w["facts"]:
        idx=len(CAR)
        ids=tok(source_prefix(fact),add_special_tokens=False).input_ids
        if ids[:P]!=PREFIX_IDS:
            raise RuntimeError(f"Canonical prefix mismatch at cartridge {idx}")
        CAR.append({
            "idx":idx,
            "world":wi,
            "type":typ,
            "fact":fact,
            "gold":gold,
            "body":ids[P:]
        })
        KEY_TO_IDX[(wi,typ)]=idx

if len(CAR)!=96:
    raise RuntimeError(f"Cartridge count mismatch: {len(CAR)} != 96")

print("\n[2/11] TEST553 panel provenance...")
rng=random.Random(PANEL_SEED)
pool=list(range(96))
rng.shuffle(pool)
PANEL=[]
cursor=0

for wi in range(24):
    typ=QUERY_CYCLE[wi%4]
    target=KEY_TO_IDX[(wi,typ)]
    others=[]
    while len(others)<4:
        ci=pool[cursor%96]
        cursor+=1
        if ci==target:
            continue
        if CAR[ci]["world"]==wi:
            continue
        if any(CAR[x]["world"]==CAR[ci]["world"] for x in others):
            continue
        others.append(ci)
    PANEL.append({"case":wi,"target":target,"others":others})

PANEL_SHA=sha_obj(PANEL)
print("PANEL SHA:",PANEL_SHA)
print("EXPECTED :",EXPECTED_PANEL_SHA)

if PANEL_SHA!=EXPECTED_PANEL_SHA:
    raise RuntimeError("TEST553 panel provenance FAIL.")

print("PANEL PROVENANCE: PASS")

def order_keys(z,slot):
    d=list(z["others"])
    t=z["target"]
    if slot=="FIRST":
        return [t]+d
    if slot=="MIDDLE":
        return d[:2]+[t]+d[2:]
    if slot=="LAST":
        return d+[t]
    raise ValueError(slot)

def cache_layers(pkv):
    if hasattr(pkv,"layers"):
        out=[]
        for layer in pkv.layers:
            k=getattr(layer,"keys",None)
            v=getattr(layer,"values",None)
            if k is None:
                k=getattr(layer,"key_cache",None)
            if v is None:
                v=getattr(layer,"value_cache",None)
            if k is None or v is None:
                raise RuntimeError("Cache layer API mismatch.")
            out.append((k,v))
        if out:
            return tuple(out)
    if hasattr(pkv,"key_cache"):
        return tuple(zip(pkv.key_cache,pkv.value_cache))
    raise RuntimeError("Cache API mismatch.")

def clone_layers(pkv):
    return tuple((k.detach().clone(),v.detach().clone()) for k,v in cache_layers(pkv))

def build_cache(layers):
    c=DynamicCache()
    for li,(k,v) in enumerate(layers):
        c.update(k,v,li)
    return c

def cache_len(c):
    return int(c.get_seq_length())

@torch.no_grad()
def answer_layers(layers,q):
    cache=build_cache(layers)
    physical=cache_len(cache)
    qids=tok(query_suffix(q),add_special_tokens=False).input_ids
    ids=torch.tensor(qids,dtype=torch.long,device=DEVICE).reshape(1,-1)
    qlen=int(ids.shape[1])
    pos=torch.arange(physical,physical+qlen,dtype=torch.long,device=DEVICE).unsqueeze(0)
    cp=pos.squeeze(0)
    att=torch.ones((1,physical+qlen),dtype=torch.long,device=DEVICE)

    out=model(
        input_ids=ids,
        past_key_values=cache,
        attention_mask=att,
        position_ids=pos,
        cache_position=cp,
        use_cache=True,
        return_dict=True
    )

    cache=out.past_key_values
    nxt=out.logits[:,-1,:].argmax(-1,keepdim=True)
    eos=tok.eos_token_id
    eosset=set() if eos is None else {int(eos)}
    gen=[]

    for step in range(MAX_NEW):
        gen.append(nxt)
        if int(nxt.item()) in eosset:
            break
        physical=cache_len(cache)
        pos=torch.tensor([[physical]],dtype=torch.long,device=DEVICE)
        cp=pos.squeeze(0)
        att=torch.ones((1,physical+1),dtype=torch.long,device=DEVICE)

        out=model(
            input_ids=nxt,
            past_key_values=cache,
            attention_mask=att,
            position_ids=pos,
            cache_position=cp,
            use_cache=True,
            return_dict=True
        )
        cache=out.past_key_values
        nxt=out.logits[:,-1,:].argmax(-1,keepdim=True)

    return tok.decode(torch.cat(gen,dim=1)[0],skip_special_tokens=True).strip()

def make_case(keys,target):
    ids=list(PREFIX_IDS)
    ranges={}

    for ci in keys:
        a=len(ids)
        ids.extend(CAR[ci]["body"])
        ranges[ci]=(a,len(ids))

    n=len(ids)
    group=torch.full((n,),-1,dtype=torch.long,device=DEVICE)

    for j,ci in enumerate(keys):
        a,b=ranges[ci]
        group[a:b]=j

    i=torch.arange(n,device=DEVICE)
    causal=i[:,None]>=i[None,:]

    # BLOCK contract:
    # - prefix is globally visible when causally preceding
    # - each body sees itself
    # - no body sees another body's tokens
    allowed=causal & (
        (group[:,None]==-1) |
        (group[None,:]==-1) |
        (group[:,None]==group[None,:])
    )

    block=torch.zeros((1,1,n,n),dtype=DTYPE,device=DEVICE)
    block.masked_fill_(~allowed,torch.finfo(DTYPE).min)

    joint=torch.zeros((1,1,n,n),dtype=DTYPE,device=DEVICE)
    joint.masked_fill_(~causal,torch.finfo(DTYPE).min)

    a,b=ranges[target]

    target_mask=torch.zeros(n,dtype=torch.bool,device=DEVICE)
    target_mask[a:b]=True

    history_mask=torch.zeros(n,dtype=torch.bool,device=DEVICE)
    for ci in keys:
        if ci!=target:
            x,y=ranges[ci]
            history_mask[x:y]=True

    full_mask=torch.zeros(n,dtype=torch.bool,device=DEVICE)
    full_mask[P:]=True

    return (
        torch.tensor(ids,dtype=torch.long,device=DEVICE).reshape(1,-1),
        joint,
        block,
        {
            "FULL":full_mask,
            "TARGET":target_mask,
            "HISTORY":history_mask
        },
        ranges
    )

# Semantics preserved from TEST555:
#
# JOINT:
#   joint attention at every decoder layer.
#
# BLOCK:
#   cross-body attention blocked at every decoder layer.
#
# SWITCH(cut=L):
#   layers 0..L use BLOCK
#   layers L+1..31 use JOINT.
#
# TRANSPLANT(cut=L):
#   same attention schedule as SWITCH;
#   output hidden state of decoder layer L is patched with the corresponding
#   JOINT reference state on FULL/TARGET/HISTORY token subset.
#
# Important:
#   K/V already written at layers <= L remain BLOCK-derived.
#   The transplant changes the hidden-state stream entering L+1.
#   Therefore comparison TRANSPLANT - SWITCH isolates the added effect of
#   the JOINT hidden-state replacement under the same attention schedule.

@torch.no_grad()
def write_pass(
    ids,
    joint,
    block,
    mode="JOINT",
    cut=None,
    scope=None,
    reference=None,
    mask=None,
    capture=False
):
    handles=[]
    states={}

    if mode in ("SWITCH","TRANSPLANT"):
        if cut is None or cut not in LAYERS:
            raise RuntimeError("Valid cut required.")

    if mode=="TRANSPLANT":
        if scope not in SCOPES:
            raise RuntimeError("Valid transplant scope required.")
        if reference is None or mask is None:
            raise RuntimeError("TRANSPLANT requires reference and mask.")

    try:
        for li,layer in enumerate(model.model.layers):

            def prehook(module,args,kwargs,li=li):
                if mode=="JOINT":
                    chosen=joint
                elif mode=="BLOCK":
                    chosen=block
                elif mode in ("SWITCH","TRANSPLANT"):
                    chosen=block if li<=cut else joint
                else:
                    raise ValueError(mode)

                kwargs["attention_mask"]=chosen
                return args,kwargs

            handles.append(
                layer.register_forward_pre_hook(prehook,with_kwargs=True)
            )

            if capture:
                def record(module,args,output,li=li):
                    h=output[0] if isinstance(output,(tuple,list)) else output
                    states[li]=h.detach().clone()
                handles.append(layer.register_forward_hook(record))

            if mode=="TRANSPLANT" and li==cut:
                def transplant(module,args,output,li=li):
                    if li not in reference:
                        raise RuntimeError(f"Missing JOINT reference state at L{li}")
                    h=output[0] if isinstance(output,(tuple,list)) else output
                    ref=reference[li]

                    if h.shape!=ref.shape:
                        raise RuntimeError(
                            f"Transplant shape mismatch at L{li}: "
                            f"{tuple(h.shape)} != {tuple(ref.shape)}"
                        )

                    if mask.numel()!=h.shape[1]:
                        raise RuntimeError(
                            f"Transplant mask mismatch: {mask.numel()} != {h.shape[1]}"
                        )

                    patched=torch.where(mask.reshape(1,-1,1),ref,h)

                    if isinstance(output,tuple):
                        return (patched,)+output[1:]
                    if isinstance(output,list):
                        return [patched]+output[1:]
                    return patched

                handles.append(layer.register_forward_hook(transplant))

        n=int(ids.shape[1])
        pos=torch.arange(n,dtype=torch.long,device=DEVICE).unsqueeze(0)

        out=model(
            input_ids=ids,
            attention_mask=torch.ones((1,n),dtype=torch.long,device=DEVICE),
            position_ids=pos,
            cache_position=pos.squeeze(0),
            use_cache=True,
            return_dict=True
        )

        layers=clone_layers(out.past_key_values)

        if len(layers)!=NL:
            raise RuntimeError(f"Cache layer count mismatch: {len(layers)} != {NL}")

        for li,(k,v) in enumerate(layers):
            if k.shape[-2]!=n or v.shape[-2]!=n:
                raise RuntimeError(
                    f"Cache length mismatch L{li}: "
                    f"K={k.shape[-2]} V={v.shape[-2]} expected={n}"
                )

        del out
        return layers,states

    finally:
        for h in handles:
            h.remove()

print("\n[3/11] Attention-mask / scope validation...")
z=PANEL[0]
keys=order_keys(z,"MIDDLE")
target=z["target"]
ids,joint,block,masks,ranges=make_case(keys,target)
n=int(ids.shape[1])

if joint.shape!=(1,1,n,n) or block.shape!=(1,1,n,n):
    raise RuntimeError("Attention mask shape mismatch.")

# Exact causal JOINT validation.
expected_joint=torch.zeros((n,n),dtype=DTYPE,device=DEVICE)
expected_joint.masked_fill_(
    ~(torch.arange(n,device=DEVICE)[:,None]>=torch.arange(n,device=DEVICE)[None,:]),
    torch.finfo(DTYPE).min
)
if not torch.equal(joint[0,0],expected_joint):
    raise RuntimeError("JOINT causal mask malformed.")

# Every body must be isolated from every other body in BLOCK.
for ci in keys:
    a,b=ranges[ci]
    for cj in keys:
        if ci==cj:
            continue
        x,y=ranges[cj]
        sub=block[0,0,a:b,x:y]
        if torch.any(sub==0):
            raise RuntimeError(
                f"BLOCK cross-body leakage: target body {ci}, foreign body {cj}"
            )

# Scope contract.
target_n=int(masks["TARGET"].sum().item())
history_n=int(masks["HISTORY"].sum().item())
full_n=int(masks["FULL"].sum().item())

if target_n<=0 or history_n<=0:
    raise RuntimeError("Empty transplant scope.")
if torch.any(masks["TARGET"] & masks["HISTORY"]):
    raise RuntimeError("TARGET/HISTORY overlap.")
if not torch.equal(masks["FULL"],masks["TARGET"]|masks["HISTORY"]):
    raise RuntimeError("FULL != TARGET union HISTORY.")
if int(masks["FULL"][:P].sum().item())!=0:
    raise RuntimeError("Canonical prefix leaked into FULL transplant scope.")

print(
    f"MASK CONTRACT: PASS | TOKENS={n} PREFIX={P} "
    f"TARGET={target_n} HISTORY={history_n} FULL={full_n}"
)

del ids,joint,block,masks,ranges,expected_joint
torch.cuda.empty_cache()

print("\n[4/11] Baseline validation — JOINT vs BLOCK...")
BASELINES=[]

for ci,z in enumerate(PANEL):
    typ=CAR[z["target"]]["type"]
    q=next(x[1] for x in W[ci]["queries"] if x[0]==typ)
    gold=CAR[z["target"]]["gold"]

    case_rows=[]

    for slot in POSITIONS:
        keys=order_keys(z,slot)
        ids,joint,block,masks,ranges=make_case(keys,z["target"])

        jl,_=write_pass(ids,joint,block,"JOINT")
        bl,_=write_pass(ids,joint,block,"BLOCK")

        ja=answer_layers(jl,q)
        ba=answer_layers(bl,q)

        row={
            "case":ci,
            "slot":slot,
            "type":typ,
            "gold":gold,
            "joint":ja,
            "block":ba,
            "joint_ok":bool(hit(ja,gold)),
            "block_ok":bool(hit(ba,gold))
        }

        BASELINES.append(row)
        case_rows.append(row)

        del jl,bl,ids,joint,block,masks,ranges

    print(
        f"CASE {ci:02d} {typ:8s} "
        f"JOINT="+"/".join(str(int(x["joint_ok"])) for x in case_rows)+" "
        f"BLOCK="+"/".join(str(int(x["block_ok"])) for x in case_rows)
    )

BJ={
    s:sum(int(x["joint_ok"]) for x in BASELINES if x["slot"]==s)
    for s in POSITIONS
}
BB={
    s:sum(int(x["block_ok"]) for x in BASELINES if x["slot"]==s)
    for s in POSITIONS
}

print("JOINT:",BJ)
print("BLOCK:",BB)

JOINT_GATE=all(BJ[s]>=22 for s in POSITIONS)
BLOCK_GATE=all(BB[s]<=12 for s in POSITIONS)

print("JOINT GATE:","PASS" if JOINT_GATE else "FAIL")
print("BLOCK GATE:","PASS" if BLOCK_GATE else "FAIL")

if not JOINT_GATE or not BLOCK_GATE:
    raise RuntimeError(
        "Baseline gate failed. TEST556 transplant experiment is uninterpretable."
    )

torch.cuda.empty_cache()

print("\n[5/11] Causal transplant — 32 layers × SWITCH + FULL/TARGET/HISTORY...")
RESULTS=[]
SWITCH=[]

for ci,z in enumerate(PANEL):
    typ=CAR[z["target"]]["type"]
    q=next(x[1] for x in W[ci]["queries"] if x[0]==typ)
    gold=CAR[z["target"]]["gold"]

    print(f"\nCASE {ci:02d} {typ:8s} GOLD={gold}")

    for slot in POSITIONS:
        keys=order_keys(z,slot)
        ids,joint,block,masks,ranges=make_case(keys,z["target"])

        ref_layers,reference=write_pass(
            ids,joint,block,
            mode="JOINT",
            capture=True
        )
        del ref_layers

        if set(reference.keys())!=set(LAYERS):
            raise RuntimeError(
                f"JOINT reference capture incomplete case={ci} slot={slot}: "
                f"{sorted(reference.keys())}"
            )

        for cut in LAYERS:
            sw_layers,_=write_pass(
                ids,joint,block,
                mode="SWITCH",
                cut=cut
            )

            sw_answer=answer_layers(sw_layers,q)

            SWITCH.append({
                "case":ci,
                "slot":slot,
                "layer":cut,
                "type":typ,
                "gold":gold,
                "answer":sw_answer,
                "answer_norm":norm(sw_answer),
                "ok":bool(hit(sw_answer,gold))
            })

            del sw_layers

            for scope in SCOPES:
                layers,_=write_pass(
                    ids,joint,block,
                    mode="TRANSPLANT",
                    cut=cut,
                    scope=scope,
                    reference=reference,
                    mask=masks[scope]
                )

                ans=answer_layers(layers,q)

                RESULTS.append({
                    "case":ci,
                    "slot":slot,
                    "layer":cut,
                    "scope":scope,
                    "type":typ,
                    "gold":gold,
                    "answer":ans,
                    "answer_norm":norm(ans),
                    "ok":bool(hit(ans,gold))
                })

                del layers

        print(
            f"  {slot:6s} complete | "
            f"32 layers | SWITCH + FULL/TARGET/HISTORY"
        )

        del reference,ids,joint,block,masks,ranges

    if (ci+1)%4==0:
        torch.cuda.empty_cache()
        gc.collect()
        print(
            f"PROGRESS {ci+1}/24 | "
            f"elapsed={time.perf_counter()-T0:.1f}s"
        )

print("\n[6/11] Raw-result integrity...")
EXPECTED_SWITCH=24*3*32
EXPECTED_PER_SCOPE=24*3*32
EXPECTED_RESULTS=EXPECTED_PER_SCOPE*3

print("SWITCH ROWS :",len(SWITCH),"EXPECTED:",EXPECTED_SWITCH)
print("RESULT ROWS :",len(RESULTS),"EXPECTED:",EXPECTED_RESULTS)

if len(SWITCH)!=EXPECTED_SWITCH:
    raise RuntimeError(
        f"SWITCH row count mismatch: {len(SWITCH)} != {EXPECTED_SWITCH}"
    )

if len(RESULTS)!=EXPECTED_RESULTS:
    raise RuntimeError(
        f"RESULT row count mismatch: {len(RESULTS)} != {EXPECTED_RESULTS}"
    )

scope_counts={
    s:sum(1 for r in RESULTS if r["scope"]==s)
    for s in SCOPES
}

print("SCOPE COUNTS:",scope_counts)

for s in SCOPES:
    if scope_counts[s]!=EXPECTED_PER_SCOPE:
        raise RuntimeError(
            f"{s} row count mismatch: "
            f"{scope_counts[s]} != {EXPECTED_PER_SCOPE}"
        )

# Every unique experimental cell must occur exactly once.
switch_keys=[
    (r["case"],r["slot"],r["layer"])
    for r in SWITCH
]
result_keys=[
    (r["case"],r["slot"],r["layer"],r["scope"])
    for r in RESULTS
]

if len(set(switch_keys))!=EXPECTED_SWITCH:
    raise RuntimeError("Duplicate/missing SWITCH experimental cells.")

if len(set(result_keys))!=EXPECTED_RESULTS:
    raise RuntimeError("Duplicate/missing transplant experimental cells.")

print("RAW RESULT INTEGRITY: PASS")

print("\n[7/11] Correct layerwise aggregation...")
SUMMARY={"SWITCH":{}}

for cut in LAYERS:
    rr=[r for r in SWITCH if r["layer"]==cut]

    if len(rr)!=72:
        raise RuntimeError(
            f"SWITCH L{cut:02d}: {len(rr)} rows != 72"
        )

    SUMMARY["SWITCH"][cut]=sum(int(r["ok"]) for r in rr)

for scope in SCOPES:
    SUMMARY[scope]={}

    for cut in LAYERS:
        # TEST555 BUG FIX:
        # BOTH layer and scope are mandatory filters.
        rr=[
            r for r in RESULTS
            if r["layer"]==cut and r["scope"]==scope
        ]

        if len(rr)!=72:
            raise RuntimeError(
                f"{scope} L{cut:02d}: {len(rr)} rows != 72"
            )

        SUMMARY[scope][cut]=sum(int(r["ok"]) for r in rr)

for scope in ["SWITCH"]+SCOPES:
    vals=[SUMMARY[scope][l] for l in LAYERS]

    if any(v<0 or v>72 for v in vals):
        raise RuntimeError(
            f"{scope}: impossible layer accuracy count {vals}"
        )

    print(
        f"{scope:8s}: "+
        " ".join(
            f"L{l:02d}={SUMMARY[scope][l]:02d}/72"
            for l in LAYERS
        )
    )

print("\n[8/11] Causal increment over SWITCH-only...")
INCREMENT={}

for scope in SCOPES:
    INCREMENT[scope]={}

    for cut in LAYERS:
        INCREMENT[scope][cut]=(
            SUMMARY[scope][cut]-
            SUMMARY["SWITCH"][cut]
        )

    best=max(
        LAYERS,
        key=lambda l:(
            INCREMENT[scope][l],
            SUMMARY[scope][l],
            -l
        )
    )

    print(
        f"{scope:8s} BEST L={best:02d} | "
        f"transplant={SUMMARY[scope][best]}/72 | "
        f"switch={SUMMARY['SWITCH'][best]}/72 | "
        f"Δ={INCREMENT[scope][best]:+d}"
    )

BEST={
    scope:max(
        LAYERS,
        key=lambda l:(
            INCREMENT[scope][l],
            SUMMARY[scope][l],
            -l
        )
    )
    for scope in SCOPES
}

MAXINC={
    scope:INCREMENT[scope][BEST[scope]]
    for scope in SCOPES
}

print("\n[9/11] FULL / TARGET / HISTORY localization + exact outcome identity...")

def result_map(scope,layer):
    rr=[
        r for r in RESULTS
        if r["scope"]==scope and r["layer"]==layer
    ]

    if len(rr)!=72:
        raise RuntimeError(
            f"{scope} L{layer:02d}: expected 72 rows, got {len(rr)}"
        )

    return {
        (r["case"],r["slot"]):r
        for r in rr
    }

def switch_map(layer):
    rr=[r for r in SWITCH if r["layer"]==layer]

    if len(rr)!=72:
        raise RuntimeError(
            f"SWITCH L{layer:02d}: expected 72 rows, got {len(rr)}"
        )

    return {
        (r["case"],r["slot"]):r
        for r in rr
    }

IDENTITY={
    "FULL_TARGET":{},
    "FULL_HISTORY":{},
    "TARGET_HISTORY":{}
}

ANSWER_IDENTITY={
    "FULL_TARGET":{},
    "FULL_HISTORY":{},
    "TARGET_HISTORY":{}
}

GAIN_OVER_SWITCH={
    s:{} for s in SCOPES
}

LOSS_OVER_SWITCH={
    s:{} for s in SCOPES
}

for cut in LAYERS:
    fm=result_map("FULL",cut)
    tm=result_map("TARGET",cut)
    hm=result_map("HISTORY",cut)
    sm=switch_map(cut)

    keys=set(sm.keys())

    if set(fm.keys())!=keys or set(tm.keys())!=keys or set(hm.keys())!=keys:
        raise RuntimeError(f"Case/slot key mismatch at L{cut:02d}")

    # Outcome identity = same correct/incorrect status on the exact same 72 cells.
    IDENTITY["FULL_TARGET"][cut]=sum(
        int(fm[k]["ok"]==tm[k]["ok"])
        for k in keys
    )
    IDENTITY["FULL_HISTORY"][cut]=sum(
        int(fm[k]["ok"]==hm[k]["ok"])
        for k in keys
    )
    IDENTITY["TARGET_HISTORY"][cut]=sum(
        int(tm[k]["ok"]==hm[k]["ok"])
        for k in keys
    )

    # Exact normalized generated-answer identity.
    ANSWER_IDENTITY["FULL_TARGET"][cut]=sum(
        int(fm[k]["answer_norm"]==tm[k]["answer_norm"])
        for k in keys
    )
    ANSWER_IDENTITY["FULL_HISTORY"][cut]=sum(
        int(fm[k]["answer_norm"]==hm[k]["answer_norm"])
        for k in keys
    )
    ANSWER_IDENTITY["TARGET_HISTORY"][cut]=sum(
        int(tm[k]["answer_norm"]==hm[k]["answer_norm"])
        for k in keys
    )

    for scope,m in [
        ("FULL",fm),
        ("TARGET",tm),
        ("HISTORY",hm)
    ]:
        GAIN_OVER_SWITCH[scope][cut]=sum(
            int((not sm[k]["ok"]) and m[k]["ok"])
            for k in keys
        )

        LOSS_OVER_SWITCH[scope][cut]=sum(
            int(sm[k]["ok"] and (not m[k]["ok"]))
            for k in keys
        )

    print(
        f"L{cut:02d} "
        f"SW={SUMMARY['SWITCH'][cut]:02d} "
        f"F={SUMMARY['FULL'][cut]:02d} "
        f"T={SUMMARY['TARGET'][cut]:02d} "
        f"H={SUMMARY['HISTORY'][cut]:02d} /72 | "
        f"ΔF={INCREMENT['FULL'][cut]:+03d} "
        f"ΔT={INCREMENT['TARGET'][cut]:+03d} "
        f"ΔH={INCREMENT['HISTORY'][cut]:+03d} | "
        f"OK-ID F/T={IDENTITY['FULL_TARGET'][cut]:02d} "
        f"F/H={IDENTITY['FULL_HISTORY'][cut]:02d} "
        f"T/H={IDENTITY['TARGET_HISTORY'][cut]:02d} | "
        f"ANS-ID F/T={ANSWER_IDENTITY['FULL_TARGET'][cut]:02d} "
        f"F/H={ANSWER_IDENTITY['FULL_HISTORY'][cut]:02d} "
        f"T/H={ANSWER_IDENTITY['TARGET_HISTORY'][cut]:02d}"
    )

print("\nBEST FULL   :",
      f"L{BEST['FULL']:02d}",
      f"{SUMMARY['FULL'][BEST['FULL']]}/72",
      f"Δ={MAXINC['FULL']:+d}",
      f"GAIN={GAIN_OVER_SWITCH['FULL'][BEST['FULL']]}",
      f"LOSS={LOSS_OVER_SWITCH['FULL'][BEST['FULL']]}")

print("BEST TARGET :",
      f"L{BEST['TARGET']:02d}",
      f"{SUMMARY['TARGET'][BEST['TARGET']]}/72",
      f"Δ={MAXINC['TARGET']:+d}",
      f"GAIN={GAIN_OVER_SWITCH['TARGET'][BEST['TARGET']]}",
      f"LOSS={LOSS_OVER_SWITCH['TARGET'][BEST['TARGET']]}")

print("BEST HISTORY:",
      f"L{BEST['HISTORY']:02d}",
      f"{SUMMARY['HISTORY'][BEST['HISTORY']]}/72",
      f"Δ={MAXINC['HISTORY']:+d}",
      f"GAIN={GAIN_OVER_SWITCH['HISTORY'][BEST['HISTORY']]}",
      f"LOSS={LOSS_OVER_SWITCH['HISTORY'][BEST['HISTORY']]}")

print("\n[10/11] Conservative mechanism decision...")

mf=MAXINC["FULL"]
mt=MAXINC["TARGET"]
mh=MAXINC["HISTORY"]

# This decision is deliberately conservative.
# A higher maximum alone is not treated as proof of a unique physical bridge.
if mf<=0 and mt<=0 and mh<=0:
    DECISION="NO POSITIVE HIDDEN-STATE TRANSPLANT EFFECT BEYOND SWITCH-ONLY"
elif mt>0 and mh<=0:
    DECISION="TARGET-STATE TRANSPLANT HAS POSITIVE EFFECT; HISTORY-STATE EFFECT NOT ESTABLISHED"
elif mh>0 and mt<=0:
    DECISION="HISTORY-STATE TRANSPLANT HAS POSITIVE EFFECT; TARGET-STATE EFFECT NOT ESTABLISHED"
elif mt>0 and mh>0:
    if mt>mh:
        DECISION="TARGET AND HISTORY TRANSPLANTS BOTH HELP; TARGET HAS LARGER MAXIMUM NET EFFECT"
    elif mh>mt:
        DECISION="TARGET AND HISTORY TRANSPLANTS BOTH HELP; HISTORY HAS LARGER MAXIMUM NET EFFECT"
    else:
        DECISION="TARGET AND HISTORY TRANSPLANTS BOTH HELP; MAXIMUM NET EFFECTS TIED"
elif mf>0:
    DECISION="FULL-STATE TRANSPLANT HELPS; SUBSET LOCALIZATION NOT ESTABLISHED"
else:
    DECISION="POSITIVE TRANSPLANT EFFECT; LOCALIZATION UNRESOLVED"

# Stronger localization flags: require positive effect AND at least one layer
# where the candidate beats the competing subset on exact accuracy.
TARGET_DOMINATES=any(
    INCREMENT["TARGET"][l]>0 and
    SUMMARY["TARGET"][l]>SUMMARY["HISTORY"][l]
    for l in LAYERS
)

HISTORY_DOMINATES=any(
    INCREMENT["HISTORY"][l]>0 and
    SUMMARY["HISTORY"][l]>SUMMARY["TARGET"][l]
    for l in LAYERS
)

print("DECISION:",DECISION)
print("TARGET DOMINATES AT >=1 POSITIVE LAYER:",TARGET_DOMINATES)
print("HISTORY DOMINATES AT >=1 POSITIVE LAYER:",HISTORY_DOMINATES)
print("CAUTION: transplant patches decoder-layer output hidden states only.")
print("CAUTION: cached K/V at and before the cut remain BLOCK-derived.")
print("CAUTION: SWITCH isolates timing of restored cross-fact attention access.")
print("CAUTION: TRANSPLANT-SWITCH isolates added effect of JOINT hidden-state replacement.")
print("CAUTION: positive effects establish causal sufficiency of the intervention, not a unique physical representation.")

print("\n[11/11] Anti-drift / final research record...")

WEIGHT_OK=(sentinel()==S0)
TRAINABLE=sum(int(p.requires_grad) for p in model.parameters())

print("WEIGHT SENTINEL:", "PASS" if WEIGHT_OK else "FAIL")
print("TRAINABLE:",TRAINABLE)
print("RETRIEVAL: NO")
print("ROUTER: NO")
print("COSINE / ANN / TOP-K: NO")
print("EXTERNAL SCORER: NO")
print("TRAINING / LORA: NO")
print("QSF: NO")

LOCK={
    "test":TEST,
    "model":MODEL_ID,
    "panel_sha":PANEL_SHA,
    "seed":SEED,
    "panel_seed":PANEL_SEED,
    "layers":LAYERS,
    "scopes":SCOPES,
    "positions":POSITIONS,
    "attention_modes":["JOINT","BLOCK","SWITCH"],
    "switch_contract":"BLOCK layers 0..cut inclusive; JOINT layers cut+1..31",
    "transplant_boundary":"decoder_layer_output",
    "transplant_scopes":{
        "FULL":"all five fact-body tokens; canonical prefix excluded",
        "TARGET":"target fact-body tokens only",
        "HISTORY":"four non-target fact-body tokens only"
    },
    "cache_at_and_before_cut":"BLOCK-derived",
    "cache_after_cut":"generated under JOINT attention schedule from patched/unpatched hidden stream",
    "aggregation":"layer AND scope",
    "expected_switch_rows":EXPECTED_SWITCH,
    "expected_rows_per_scope":EXPECTED_PER_SCOPE,
    "expected_result_rows":EXPECTED_RESULTS,
    "case_level_identity":True,
    "answer_level_identity":True,
    "training":False,
    "retrieval":False,
    "router":False,
    "qsf":False
}

LOCK_SHA=sha_obj(LOCK)

FINAL={
    "test":TEST,
    "lock_sha":LOCK_SHA,
    "panel_sha":PANEL_SHA,
    "baseline_joint":BJ,
    "baseline_block":BB,
    "summary":SUMMARY,
    "increment":INCREMENT,
    "best_layer":BEST,
    "max_increment":MAXINC,
    "gain_over_switch":GAIN_OVER_SWITCH,
    "loss_over_switch":LOSS_OVER_SWITCH,
    "outcome_identity":IDENTITY,
    "answer_identity":ANSWER_IDENTITY,
    "target_dominates":TARGET_DOMINATES,
    "history_dominates":HISTORY_DOMINATES,
    "decision":DECISION,
    "weight_ok":WEIGHT_OK,
    "trainable":TRAINABLE
}

RESULT_SHA=sha_obj(FINAL)

print("\n"+"="*168)
print("TEST556 — FINAL RESEARCH RECORD")
print("="*168)
print("MODEL                    :",MODEL_ID)
print("PANEL SHA                :",PANEL_SHA)
print("LOCK SHA                 :",LOCK_SHA)
print("RESULT SHA               :",RESULT_SHA)
print("TEST555 AGGREGATION BUG  : FIXED — RESULTS FILTERED BY LAYER + SCOPE")
print("RAW SWITCH ROWS          :",len(SWITCH))
print("RAW TRANSPLANT ROWS      :",len(RESULTS))
print("JOINT BASELINE           :",BJ)
print("BLOCK BASELINE           :",BB)

print(
    "BEST FULL                : "
    f"L{BEST['FULL']:02d} "
    f"{SUMMARY['FULL'][BEST['FULL']]}/72 "
    f"SW={SUMMARY['SWITCH'][BEST['FULL']]}/72 "
    f"Δ={MAXINC['FULL']:+d} "
    f"GAIN={GAIN_OVER_SWITCH['FULL'][BEST['FULL']]} "
    f"LOSS={LOSS_OVER_SWITCH['FULL'][BEST['FULL']]}"
)

print(
    "BEST TARGET              : "
    f"L{BEST['TARGET']:02d} "
    f"{SUMMARY['TARGET'][BEST['TARGET']]}/72 "
    f"SW={SUMMARY['SWITCH'][BEST['TARGET']]}/72 "
    f"Δ={MAXINC['TARGET']:+d} "
    f"GAIN={GAIN_OVER_SWITCH['TARGET'][BEST['TARGET']]} "
    f"LOSS={LOSS_OVER_SWITCH['TARGET'][BEST['TARGET']]}"
)

print(
    "BEST HISTORY             : "
    f"L{BEST['HISTORY']:02d} "
    f"{SUMMARY['HISTORY'][BEST['HISTORY']]}/72 "
    f"SW={SUMMARY['SWITCH'][BEST['HISTORY']]}/72 "
    f"Δ={MAXINC['HISTORY']:+d} "
    f"GAIN={GAIN_OVER_SWITCH['HISTORY'][BEST['HISTORY']]} "
    f"LOSS={LOSS_OVER_SWITCH['HISTORY'][BEST['HISTORY']]}"
)

print("TARGET DOMINATES          :",TARGET_DOMINATES)
print("HISTORY DOMINATES         :",HISTORY_DOMINATES)
print("WEIGHT SENTINEL           :",WEIGHT_OK)
print("TRAINABLE                 :",TRAINABLE)
print("TOTAL TIME                :",f"{time.perf_counter()-T0:.2f}s")
print("RESEARCH DECISION         :",DECISION)
print("NEXT TEST                 : TEST557 — ONLY AFTER TEST556 INTERPRETATION")
print("="*168)

if not WEIGHT_OK:
    raise RuntimeError("WEIGHT SENTINEL FAILURE.")

if TRAINABLE!=0:
    raise RuntimeError("TRAINABLE PARAMETER FAILURE.")

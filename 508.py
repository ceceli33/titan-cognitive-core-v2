# Copyright © 2026 Mustafa Akbaş
# New original AKBASCORE MAM material is source-available under the repository LICENSE.
# See repository LICENSE for complete terms.
#
# AKBASCORE MAM — TEST508
# FIVE-STAGE SELECTIVE ASSOCIATIVE RELAY IN A 32-CARTRIDGE BANK
#
# 32 independently forged RAW BELLEKÖZ per item:
#   5 required memories + 27 independent distractors.
#
# Required chain:
#   QUERY(Entity)
#     → MEMORY A → Seal
#     → MEMORY B → Class
#     → MEMORY C → Sector
#     → MEMORY D → Node
#     → MEMORY E → Code
#
# No required cartridge address is supplied to RANKED arm.
#
# Controls:
#   DIRECT       = all 32 independent memories, one-shot
#   ORACLE_ADDR  = correct A/B/C/D/E addresses; traces remain model-produced
#   RANKED       = exhaustive O(N) model-native YES-vs-NO first-token logit ranking
#   WRONG_S1     = wrong Seal before B lookup
#   WRONG_S2     = wrong Class before C lookup
#   WRONG_S3     = wrong Sector before D lookup
#   WRONG_S4     = wrong Node before E lookup
#
# IMPORTANT:
#   O(N) exhaustive scan remains experimental addressing, not final scalable routing.
#   Intermediate traces remain textual, not latent BELLEKBAĞ.
#
# No compression. No joint re-forging. No DRA/steering.
# No VTOKEN. No ANN/learned router. No graph. No training. No post-hoc selection.

import os,sys,subprocess,importlib.util,random,time,hashlib,json,re,string
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache

TEST="508";SEED=508;BASE_SEED=501;MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
N_ITEMS=32;BANK_N=32;N_REQ=5;N_DIST=27;MAX_NEW=12
DEVICE=torch.device("cuda");FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
TEST501_LOCK_SHA="1a3e6ea2a9f6c72d7b8dacb0c0844e9f2a6f8595bb232f1554eb41af7a20dc41"
TEST502_LOCK_SHA="f852b6efc6ee887bae54db08cbc11f987e749499a51cebeb5132f569e74edb89"
TEST503_LOCK_SHA="2cb70ffa53a98efb249590dba7ef184b9e41cf72a99291bda65e6c93b82cb5de"
TEST504_LOCK_SHA="74c375373666c840105cc165b90977c56c8698cf3d5617c63f619a0139386846"
TEST505_LOCK_SHA="6d093a5d082b43ff56f09bb94cfed1839c0bd0d76a967822937996a5d1948e43"
TEST506_LOCK_SHA="5f14f6660c5ce99813544c93420b00559a9fc4612813d326fa02df4f76d65327"
TEST507_LOCK_SHA="86f9598a595603bfa4f63cb5133ffe1174fa307474f8597420db3bb151d74e8f"

ENTITY_BANK=[
("Aldren","Boreal","Cyrene"),("Darian","Elara","Faron"),("Galen","Hesper","Ilyra"),("Joren","Kaelis","Lorin"),
("Maren","Neris","Orlan"),("Perrin","Quorin","Ralen"),("Saren","Taris","Ulric"),("Valen","Weyra","Xeran"),
("Yorin","Zaren","Avel"),("Brann","Ceris","Dalen"),("Eris","Felis","Gorin"),("Halen","Ivar","Jaris"),
("Koren","Leris","Miran"),("Nolan","Orel","Palis"),("Riven","Solis","Teren"),("Urian","Varen","Wilis"),
("Xorin","Yalen","Zorin"),("Arven","Belis","Coren"),("Derin","Evan","Feris"),("Garin","Heron","Ilven"),
("Jarin","Kelis","Laven"),("Moris","Naven","Orris"),("Parin","Rovis","Selan"),("Torin","Ulen","Veris"),
("Waren","Xelis","Yaris"),("Zelis","Aren","Borin"),("Caren","Dorin","Elen"),("Faren","Gelis","Harin"),
("Iren","Joris","Kalen"),("Laris","Meren","Noren"),("Oris","Peren","Ravin"),("Serin","Toren","Ulis")
]
SEAL_BANK=["KOR","VEL","DAR","MIR","SEN","ROL","FEN","JAL","WEX","NUR","BAV","CIR","DEM","GOS","HIL","KET","LOR","MEV","PIR","RUK","SAV","TOL","VIR","YEK","ZAM","BIR","CAV","DOL","FER","GUL","HAR","JEM","KIR","LEV","MOR","NEX","PEL","RAS","SUL","TIR","VAN","YOR","ZEL","BOS","CER","DIN","FAL","GER","HOV","JUN","KEL","LUM","NAV","POR","REV","SIM","TUR","WAL","XEN","YIL","ZOR","BEK","COR","DUR","EKS","FIR","GAN","HEL","IVO","JOR","KAS","LIN","MUR","NOR","OVA","PAR","RIN","SOL","TEV","URB","VAR","WEN","XAL","YUN","ZEN","BOR","CEN","DAN","ELV","FOR","GIR","HAN","IRV","JEN","KOL","MAR"]
CLASS_RAW=["TAK","BEX","LUM","RAV","SOD","PEK","NIV","GOR","HAX","JUR","KEM","VOL","DAX","FIR","MON","SAL","TEK","WIR","ZUN","COV","HEM","JAX","LIV","NOR","PAK","RUM","SEV","TIX","VOR","YAM","ZEK","BOL","CER","DOV","FEX","GAM","HUR","JIN","KAV","LER","MEX","NUR","PIV","ROK","SUM","TAL","VEK","WON","XIR","YAV","ZOL","BAR","CIX","DEM","FOV","GEL","HIN","JOV","KUR","LEV","MAV","NEX","PUL","RIM","SAV","TOX","VIL","WER","XAN","YER","ZIM","BUN","CAL","DOR","EVI","FAR","GUN","HES","ILM","JER","KON","LAR","MUR","NOL","OVI","PER","RUS","SIN","TUR","VEX","WAL","XEN","YUL","ZAR","BEL","CUM"]
CODE_BANK=["AQ7","BR4","CX9","DM2","EV8","FK3","GL6","HN5","JP7","KR2","LS9","MT4","NV6","PX3","QH8","RJ5",
"SK7","TL2","UM9","VW4","WX6","YB3","ZC8","AD5","BE7","CF2","DG9","EH4","FI6","GJ3","HK8","IL5",
"JM7","KN2","LO9","MP4","NQ6","OR3","PS8","QT5","RU7","SV2","TW9","UX4","VY6","WZ3","XA8","YC5",
"ZD7","AE2","BF9","CG4","DH6","EI3","FJ8","GK5","HL7","IM2","JN9","KO4","LP6","MQ3","NR8","OS5",
"PT7","QU2","RV9","SW4","TX6","UY3","VZ8","WA5","XB7","YC2","ZD9","AF4","BG6","CH3","DI8","EJ5",
"FK7","GL2","HM9","IN4","JO6","KP3","LQ8","MR5","NS7","OT2","PU9","QV4","RW6","SX3","TY8","UZ5"]

# -------------------------------------------------------------------------
# TYPE NAMESPACE CONSTRUCTION
# Seal / Class / Sector / Node are pairwise-disjoint 3-letter namespaces.
# Code is naturally disjoint because it has AA0 form.
# -------------------------------------------------------------------------
def fresh3(n,forbidden,start=0):
    out=[];seen=set(forbidden);k=0
    for a in string.ascii_uppercase:
        for b in string.ascii_uppercase:
            for c in string.ascii_uppercase:
                x=a+b+c
                if k>=start and x not in seen:
                    out.append(x);seen.add(x)
                    if len(out)==n:return out
                k+=1
    raise RuntimeError("3-letter namespace exhausted.")

def sanitize_classes(raw,seals):
    forbidden=set(seals);used=set(raw)|forbidden;pool=fresh3(len(raw)*2,used);it=iter(pool);out=[];remap={}
    for x in raw:
        if x in forbidden:
            y=next(it);remap[x]=y;out.append(y)
        else:out.append(x)
    return out,remap

CLASS_BANK,CLASS_REMAP=sanitize_classes(CLASS_RAW,SEAL_BANK)
SECTOR_BANK=fresh3(96,set(SEAL_BANK)|set(CLASS_BANK))
NODE_BANK=fresh3(96,set(SEAL_BANK)|set(CLASS_BANK)|set(SECTOR_BANK))

assert len(ENTITY_BANK)==N_ITEMS
for x in [SEAL_BANK,CLASS_BANK,SECTOR_BANK,NODE_BANK,CODE_BANK]:
    assert len(x)>=N_ITEMS*3 and len(set(x))==len(x)
assert set(SEAL_BANK).isdisjoint(CLASS_BANK)
assert set(SEAL_BANK).isdisjoint(SECTOR_BANK)
assert set(SEAL_BANK).isdisjoint(NODE_BANK)
assert set(CLASS_BANK).isdisjoint(SECTOR_BANK)
assert set(CLASS_BANK).isdisjoint(NODE_BANK)
assert set(SECTOR_BANK).isdisjoint(NODE_BANK)
assert all(re.fullmatch(r"[A-Z]{3}",x) for x in SEAL_BANK+CLASS_BANK+SECTOR_BANK+NODE_BANK)
assert all(re.fullmatch(r"[A-Z]{2}[0-9]",x) for x in CODE_BANK)

os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
torch.set_grad_enabled(False)
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

def token_present(text,token):
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(str(token))}(?![A-Za-z0-9_])",str(text),re.I) is not None

def reorder(rows,gold,target_pos,seed):
    g=rows[gold];rest=[x for j,x in enumerate(rows) if j!=gold]
    rr=random.Random(seed);rr.shuffle(rest);out=rest[:];out.insert(target_pos,g);return out

def make_base():
    rng=random.Random(BASE_SEED)
    seals=SEAL_BANK.copy();classes=CLASS_BANK.copy();sectors=SECTOR_BANK.copy();nodes=NODE_BANK.copy();codes=CODE_BANK.copy()
    for x in [seals,classes,sectors,nodes,codes]:rng.shuffle(x)
    items=[]
    for i in range(N_ITEMS):
        ents=list(ENTITY_BANK[i]);g=i%3
        ss=seals[3*i:3*i+3];cc=classes[3*i:3*i+3];xx=sectors[3*i:3*i+3];nn=nodes[3*i:3*i+3];dd=codes[3*i:3*i+3]
        if any(len(x)!=3 for x in [ss,cc,xx,nn,dd]):raise RuntimeError(f"Base pool exhaustion item {i+1}.")
        r1=[f"Instrument {ents[j]} carries seal {ss[j]}." for j in range(3)]
        r2=[f"Seal {ss[j]} corresponds to routing class {cc[j]}." for j in range(3)]
        r3=[f"Routing class {cc[j]} corresponds to sector {xx[j]}." for j in range(3)]
        r4=[f"Sector {xx[j]} corresponds to node {nn[j]}." for j in range(3)]
        r5=[f"Node {nn[j]} carries recovery code {dd[j]}." for j in range(3)]
        pos=[i%3,(i+1)%3,(i+2)%3,i%3,(i+1)%3]
        recs=[
            " ".join(reorder(r1,g,pos[0],BASE_SEED+i*101+17)),
            " ".join(reorder(r2,g,pos[1],BASE_SEED+i*101+34)),
            " ".join(reorder(r3,g,pos[2],BASE_SEED+i*101+51)),
            " ".join(reorder(r4,g,pos[3],BASE_SEED+i*101+68)),
            " ".join(reorder(r5,g,pos[4],BASE_SEED+i*101+85))
        ]
        items.append({"id":i+1,"records":recs,
                      "gold":{"entity":ents[g],"seal":ss[g],"class":cc[g],"sector":xx[g],"node":nn[g],"code":dd[g]},
                      "positions":pos})
    return items

BASE=make_base()

# 27 distractors:
#   6 Entity→Seal
#   6 Seal→Class
#   5 Class→Sector
#   5 Sector→Node
#   5 Node→Code
def make_distractors(i,g):
    rng=random.Random(SEED+90000+i)
    ep=[e for j,t in enumerate(ENTITY_BANK) if j!=i for e in t if e.lower()!=g["entity"].lower()]
    sp=[x for x in SEAL_BANK if x!=g["seal"]]
    cp=[x for x in CLASS_BANK if x!=g["class"]]
    xp=[x for x in SECTOR_BANK if x!=g["sector"]]
    npool=[x for x in NODE_BANK if x!=g["node"]]
    dp=[x for x in CODE_BANK if x!=g["code"]]
    for x in [ep,sp,cp,xp,npool,dp]:rng.shuffle(x)
    if len(ep)<18 or len(sp)<36 or len(cp)<33 or len(xp)<30 or len(npool)<30 or len(dp)<15:
        raise RuntimeError(f"Insufficient distractor pool item {i+1}.")
    ds=[]
    for d in range(6):
        e=ep[d*3:d*3+3];s=sp[d*3:d*3+3]
        ds.append(" ".join(f"Instrument {e[j]} carries seal {s[j]}." for j in range(3)))
    for d in range(6):
        s=sp[18+d*3:18+d*3+3];c=cp[d*3:d*3+3]
        ds.append(" ".join(f"Seal {s[j]} corresponds to routing class {c[j]}." for j in range(3)))
    for d in range(5):
        c=cp[18+d*3:18+d*3+3];x=xp[d*3:d*3+3]
        ds.append(" ".join(f"Routing class {c[j]} corresponds to sector {x[j]}." for j in range(3)))
    for d in range(5):
        x=xp[15+d*3:15+d*3+3];n=npool[d*3:d*3+3]
        ds.append(" ".join(f"Sector {x[j]} corresponds to node {n[j]}." for j in range(3)))
    for d in range(5):
        n=npool[15+d*3:15+d*3+3];q=dp[d*3:d*3+3]
        ds.append(" ".join(f"Node {n[j]} carries recovery code {q[j]}." for j in range(3)))
    if len(ds)!=N_DIST:raise RuntimeError(f"Distractor count failure item {i+1}: {len(ds)}")
    for n,x in enumerate(ds):
        for key in ("entity","seal","class","sector","node","code"):
            if token_present(x,g[key]):raise RuntimeError(f"{key} contamination item {i+1}, distractor {n+1}: {x}")
    return ds

ITEMS=[]
for i,b in enumerate(BASE):
    g=b["gold"];ds=make_distractors(i,g)
    mems=[
        {"role":"A","text":b["records"][0]},
        {"role":"B","text":b["records"][1]},
        {"role":"C","text":b["records"][2]},
        {"role":"D","text":b["records"][3]},
        {"role":"E","text":b["records"][4]}
    ]+[{"role":"X","text":x} for x in ds]
    rng=random.Random(SEED+70000+i);rng.shuffle(mems)
    if len(mems)!=BANK_N:raise RuntimeError(f"Bank size failure item {i+1}: {len(mems)}")
    counts={r:sum(m["role"]==r for m in mems) for r in ("A","B","C","D","E","X")}
    if counts!={"A":1,"B":1,"C":1,"D":1,"E":1,"X":N_DIST}:raise RuntimeError(f"Role-count failure item {i+1}: {counts}")
    idx={r:next(j for j,m in enumerate(mems) if m["role"]==r) for r in ("A","B","C","D","E")}
    ia,ib,ic,id_,ie=[idx[r] for r in ("A","B","C","D","E")]
    if len({ia,ib,ic,id_,ie})!=5:raise RuntimeError(f"Required-address collision item {i+1}.")
    expected={
        "entity":[ia],
        "seal":sorted([ia,ib]),
        "class":sorted([ib,ic]),
        "sector":sorted([ic,id_]),
        "node":sorted([id_,ie]),
        "code":[ie]
    }
    for key,want in expected.items():
        hits=[j for j,m in enumerate(mems) if token_present(m["text"],g[key])]
        if sorted(hits)!=sorted(want):raise RuntimeError(f"{key} uniqueness failure item {i+1}: {hits}/{want}")
    ITEMS.append({**b,"bank":mems,"oracle":[ia,ib,ic,id_,ie]})

LOCK={"test":TEST,"model":MODEL_ID,"seed":SEED,"base_panel_seed":BASE_SEED,"items":ITEMS,
      "bank_n":BANK_N,"required":N_REQ,"distractors":N_DIST,
      "class_namespace_remap":CLASS_REMAP,
      "namespace_guard":"Seal/Class/Sector/Node/Code pairwise type-distinct",
      "test501":TEST501_LOCK_SHA,"test502":TEST502_LOCK_SHA,"test503":TEST503_LOCK_SHA,
      "test504":TEST504_LOCK_SHA,"test505":TEST505_LOCK_SHA,"test506":TEST506_LOCK_SHA,"test507":TEST507_LOCK_SHA,
      "chain":"Entity->Seal->Class->Sector->Node->Code",
      "arms":["DIRECT_32","ORACLE_ADDRESS_5_STAGE","RANKED_5_STAGE","WRONG_S1","WRONG_S2","WRONG_S3","WRONG_S4"],
      "rank_signal":"first-answer-token logit(YES)-logit(NO)",
      "used_memory_reselection":"OFF","compression":"OFF","joint_reforge":"OFF","steering":"OFF","vtoken":"OFF",
      "ann_router":"OFF","learned_router":"OFF","graph":"OFF","training":"OFF","posthoc":"OFF"}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

print("="*174)
print("TEST508 — AKBASCORE MAM · FIVE-STAGE SELECTIVE ASSOCIATIVE RELAY IN A 32-CARTRIDGE BANK")
print("32 INDEPENDENT BELLEKÖZ · 5 REQUIRED + 27 DISTRACTORS · ENTITY → SEAL → CLASS → SECTOR → NODE → CODE")
print("="*174)
print("LOCK SHA:",LOCK_SHA)
print("TEST507 LOCK:",TEST507_LOCK_SHA)
print("ITEMS:",N_ITEMS,"| BANK:",BANK_N,"| REQUIRED:",N_REQ,"| DISTRACTORS:",N_DIST)
print("CONTENT ADDRESSING: exhaustive O(N); gold cartridge addresses hidden from RANKED arm")
print("COMPRESSION / JOINT RE-FORGE / DRA / VTOKEN / ANN / LEARNED ROUTER / GRAPH / TRAINING: NONE")
print("TYPE NAMESPACE GUARD: Seal / Class / Sector / Node / Code type-distinct")
print("CLASS COLLISIONS REMAPPED:",len(CLASS_REMAP),CLASS_REMAP)
T0=time.perf_counter()

print("\n[1/7] Loading frozen Qwen...")
_tv=tuple(int("".join(c for c in x if c.isdigit())or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
_ge=model.generation_config.eos_token_id
EOS=sorted({tok.eos_token_id,*(_ge if isinstance(_ge,(list,tuple)) else [_ge])}-{None})
enc=lambda s:tok(s,add_special_tokens=False).input_ids
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=H//QH
if (NL,H,QH,KVH,HD)!=(28,3584,28,4,128):raise RuntimeError("Architecture mismatch.")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
YES_IDS=enc(" YES");NO_IDS=enc(" NO")
if len(YES_IDS)!=1 or len(NO_IDS)!=1:raise RuntimeError(f"YES/NO tokenizer mismatch: YES={YES_IDS}, NO={NO_IDS}")
YES_ID,NO_ID=YES_IDS[0],NO_IDS[0]
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | {NL}L | H={H} | QH={QH} KVH={KVH} HD={HD} | frozen BF16")
print(f"      Ranking tokens: YES={YES_ID} {tok.decode([YES_ID])!r} | NO={NO_ID} {tok.decode([NO_ID])!r}")

@torch.inference_mode()
def kv_from_ids(ids):
    x=torch.tensor([ids],device=DEVICE)
    o=model(input_ids=x,use_cache=False,output_hidden_states=True,return_dict=True)
    out=[]
    for L,layer in enumerate(model.model.layers):
        h=layer.input_layernorm(o.hidden_states[L][0])
        k=layer.self_attn.k_proj(h).view(-1,KVH,HD).transpose(0,1).contiguous()
        v=layer.self_attn.v_proj(h).view(-1,KVH,HD).transpose(0,1).contiguous()
        out.append((k,v))
    return out

@torch.inference_mode()
def forge(s):return kv_from_ids([PAD]+enc(s+SEP))

def rope_k(k,pos,L):
    layer=model.model.layers[L]
    rot=layer.self_attn.rotary_emb if hasattr(layer.self_attn,"rotary_emb") else model.model.rotary_emb
    dummy=torch.zeros((1,k.shape[0],len(pos),HD),device=DEVICE,dtype=k.dtype)
    p=torch.tensor([pos],device=DEVICE,dtype=torch.long)
    try:cos,sin=rot(dummy,p)
    except TypeError:cos,sin=rot(dummy,position_ids=p)
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    _,kr=apply_rotary_pos_emb(dummy,k.unsqueeze(0),cos,sin)
    return kr[0]

@torch.inference_mode()
def install_single(raw):
    T=raw[0][0].shape[1];pos=list(range(T));out=[]
    for L in range(NL):
        k,v=raw[L];out.append((rope_k(k,pos,L),v))
    return out,T,T

@torch.inference_mode()
def install_parallel(raws):
    Ts=[r[0][0].shape[1] for r in raws];out=[]
    for L in range(NL):
        KS=[];VS=[]
        for r,T in zip(raws,Ts):
            k,v=r[L];KS.append(rope_k(k,list(range(T)),L));VS.append(v)
        out.append((torch.cat(KS,1),torch.cat(VS,1)))
    return out,sum(Ts),max(Ts)

def make_cache(installed):
    try:cache=DynamicCache(config=cfg)
    except TypeError:cache=DynamicCache()
    for L,(k,v) in enumerate(installed):cache.update(k.unsqueeze(0),v.unsqueeze(0),L)
    return cache

@torch.inference_mode()
def read_kv(installed,Tm,P,q,max_new=MAX_NEW):
    cache=make_cache(installed);qids=enc(FMT.format(q=q));out=[]
    for step in range(max_new):
        ids=qids if step==0 else [out[-1]]
        nq=len(ids);pos=torch.arange(P,P+nq,device=DEVICE).unsqueeze(0)
        mask=torch.ones((1,Tm+nq),device=DEVICE,dtype=torch.long)
        o=model(input_ids=torch.tensor([ids],device=DEVICE),past_key_values=cache,attention_mask=mask,
                position_ids=pos,use_cache=True,return_dict=True)
        cache=o.past_key_values;nxt=int(o.logits[0,-1].float().argmax())
        if nxt in EOS:break
        out.append(nxt);P+=nq;Tm+=nq
    return tok.decode(out,skip_special_tokens=True).strip()

@torch.inference_mode()
def yesno_margin(installed,Tm,P,q):
    cache=make_cache(installed);qids=enc(FMT.format(q=q));nq=len(qids)
    pos=torch.arange(P,P+nq,device=DEVICE).unsqueeze(0)
    mask=torch.ones((1,Tm+nq),device=DEVICE,dtype=torch.long)
    o=model(input_ids=torch.tensor([qids],device=DEVICE),past_key_values=cache,attention_mask=mask,
            position_ids=pos,use_cache=False,return_dict=True)
    lg=o.logits[0,-1].float()
    return float((lg[YES_ID]-lg[NO_ID]).item())

def firstline(x):return next((z.strip() for z in str(x).splitlines() if z.strip()),"")
def norm3(x):
    s=firstline(x).strip().upper()
    m=re.match(r"^\s*([A-Z]{3})(?=$|[\s\.,;:!?])",s)
    return m.group(1) if m else s.rstrip(".")
def normcode(x):
    s=firstline(x).strip().upper()
    m=re.match(r"^\s*([A-Z]{2}[0-9])(?=$|[\s\.,;:!?])",s)
    return m.group(1) if m else s.rstrip(".")
def exact3(x,t):return norm3(x)==str(t).upper()
def exactcode(x,t):return normcode(x)==str(t).upper()

def q_direct(e):return f"What recovery code ultimately corresponds to instrument {e}? Follow the available memories and give only the exact recovery code."
def q_rel_entity(e):return f"Does this memory explicitly contain a relation stating which seal instrument {e} carries? Answer only YES or NO."
def q_rel_seal(x):return f"Does this memory explicitly contain a relation stating which routing class seal {x} corresponds to? Answer only YES or NO."
def q_rel_class(x):return f"Does this memory explicitly contain a relation stating which sector routing class {x} corresponds to? Answer only YES or NO."
def q_rel_sector(x):return f"Does this memory explicitly contain a relation stating which node sector {x} corresponds to? Answer only YES or NO."
def q_rel_node(x):return f"Does this memory explicitly contain a relation stating which recovery code node {x} carries? Answer only YES or NO."
def q_read_a(e):return f"What seal does instrument {e} carry? Give only the exact seal."
def q_read_b(x):return f"What routing class corresponds to seal {x}? Give only the exact routing class."
def q_read_c(x):return f"What sector corresponds to routing class {x}? Give only the exact sector."
def q_read_d(x):return f"What node corresponds to sector {x}? Give only the exact node."
def q_read_e(x):return f"What recovery code does node {x} carry? Give only the exact recovery code."

def ranked_select(installed,q,excluded=None):
    excluded=set() if excluded is None else set(excluded);scores=[]
    for j,(inst,Tm,P) in enumerate(installed):
        if j in excluded:continue
        scores.append((yesno_margin(inst,Tm,P,q),j))
    if not scores:raise RuntimeError("ranked_select: no candidate memories remain.")
    scores.sort(key=lambda x:(-x[0],x[1]))
    return scores[0][1],scores[0][0],scores

def rank_of(scores,gold):
    for r,(_,j) in enumerate(scores,1):
        if j==gold:return r
    return None

def wrong_symbol(pool,gold,offset):
    if gold not in pool:raise RuntimeError(f"Gold symbol {gold!r} absent from pool.")
    j=pool.index(gold);step=offset%(len(pool)-1)+1;x=pool[(j+step)%len(pool)]
    if x==gold:raise RuntimeError("Wrong-symbol construction failure.")
    return x

def bootstrap_delta(a,b,B=10000,seed=508):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);d=a-b;n=len(d)
    rng=np.random.default_rng(seed);vals=np.empty(B)
    for i in range(B):vals[i]=d[rng.integers(0,n,n)].mean()
    return float(d.mean()),tuple(np.quantile(vals,[.025,.975]))

print("\n[2/7] Bank seal...")
for it in ITEMS[:6]:
    a,b,c,d,e=it["oracle"];g=it["gold"]
    print(f"      ITEM {it['id']:02d} {g['entity']:6s} {g['seal']}->{g['class']}->{g['sector']}->{g['node']}->{g['code']} | A=M{a+1:02d} B=M{b+1:02d} C=M{c+1:02d} D=M{d+1:02d} E=M{e+1:02d}")
print("      32 independent memories/item; A/B/C/D/E addresses hidden from RANKED.")
print("      Gold Entity/Seal/Class/Sector/Node/Code absent from all 27 distractors.")

KEYS=["DIRECT","ORACLE","A","T1","B","T2","C","T3","D","T4","E","VALUE","E2E",
      "W1_NEXT","W1_FINAL","W2_NEXT","W2_FINAL","W3_NEXT","W3_FINAL","W4_NEXT","W4_FINAL"]
R={k:[] for k in KEYS};RANKS=[[] for _ in range(5)]

print("\n[3/7] Forging and running 32-cartridge five-stage banks...")
for z,it in enumerate(ITEMS):
    g=it["gold"];ia,ib,ic,id_,ie=it["oracle"]
    raws=[forge(m["text"]) for m in it["bank"]]
    installed=[install_single(r) for r in raws]

    instD,TmD,PD=install_parallel(raws)
    direct=exactcode(read_kv(instD,TmD,PD,q_direct(g["entity"])),g["code"])

    # ORACLE ADDRESS 5-STAGE
    x=read_kv(*installed[ia],q_read_a(g["entity"]));o1=norm3(x)
    x=read_kv(*installed[ib],q_read_b(o1));o2=norm3(x)
    x=read_kv(*installed[ic],q_read_c(o2));o3=norm3(x)
    x=read_kv(*installed[id_],q_read_d(o3));o4=norm3(x)
    x=read_kv(*installed[ie],q_read_e(o4))
    oracle=exact3(o1,g["seal"]) and exact3(o2,g["class"]) and exact3(o3,g["sector"]) and exact3(o4,g["node"]) and exactcode(x,g["code"])

    # RANKED STAGE 1
    a,_,sc=ranked_select(installed,q_rel_entity(g["entity"]));r1=rank_of(sc,ia);RANKS[0].append(r1)
    t1=norm3(read_kv(*installed[a],q_read_a(g["entity"])))
    A=a==ia;T1=A and t1==g["seal"]

    # RANKED STAGE 2
    b=None;r2=None;t2="";B=T2=False
    if re.fullmatch(r"[A-Z]{3}",t1 or ""):
        b,_,sc=ranked_select(installed,q_rel_seal(t1),{a});r2=rank_of(sc,ib);RANKS[1].append(r2)
        t2=norm3(read_kv(*installed[b],q_read_b(t1)));B=b==ib;T2=T1 and B and t2==g["class"]
    else:RANKS[1].append(None)

    # RANKED STAGE 3
    c=None;r3=None;t3="";C=T3=False
    used={a}|({b} if b is not None else set())
    if re.fullmatch(r"[A-Z]{3}",t2 or ""):
        c,_,sc=ranked_select(installed,q_rel_class(t2),used);r3=rank_of(sc,ic);RANKS[2].append(r3)
        t3=norm3(read_kv(*installed[c],q_read_c(t2)));C=c==ic;T3=T2 and C and t3==g["sector"]
    else:RANKS[2].append(None)

    # RANKED STAGE 4
    d=None;r4=None;t4="";D=T4=False
    used={x for x in (a,b,c) if x is not None}
    if re.fullmatch(r"[A-Z]{3}",t3 or ""):
        d,_,sc=ranked_select(installed,q_rel_sector(t3),used);r4=rank_of(sc,id_);RANKS[3].append(r4)
        t4=norm3(read_kv(*installed[d],q_read_d(t3)));D=d==id_;T4=T3 and D and t4==g["node"]
    else:RANKS[3].append(None)

    # RANKED STAGE 5
    e=None;r5=None;final="";E=VALUE=E2E=False
    used={x for x in (a,b,c,d) if x is not None}
    if re.fullmatch(r"[A-Z]{3}",t4 or ""):
        e,_,sc=ranked_select(installed,q_rel_node(t4),used);r5=rank_of(sc,ie);RANKS[4].append(r5)
        final=read_kv(*installed[e],q_read_e(t4));E=e==ie;VALUE=E and exactcode(final,g["code"]);E2E=T4 and VALUE
    else:RANKS[4].append(None)

    # WRONG TRACE CONTROLS
    # Each control starts immediately after the deliberately corrupted trace.
    def wrong_chain(stage):
        if stage==1:
            w=wrong_symbol(SEAL_BANK,g["seal"],17);u={a}
            j,_,_=ranked_select(installed,q_rel_seal(w),u);nextgold=int(j==ib);u.add(j)
            x=norm3(read_kv(*installed[j],q_read_b(w)))
            if not re.fullmatch(r"[A-Z]{3}",x or ""):return nextgold,0
            j,_,_=ranked_select(installed,q_rel_class(x),u);u.add(j);x=norm3(read_kv(*installed[j],q_read_c(x)))
            if not re.fullmatch(r"[A-Z]{3}",x or ""):return nextgold,0
            j,_,_=ranked_select(installed,q_rel_sector(x),u);u.add(j);x=norm3(read_kv(*installed[j],q_read_d(x)))
            if not re.fullmatch(r"[A-Z]{3}",x or ""):return nextgold,0
            j,_,_=ranked_select(installed,q_rel_node(x),u);y=read_kv(*installed[j],q_read_e(x))
            return nextgold,int(exactcode(y,g["code"]))
        if stage==2:
            w=wrong_symbol(CLASS_BANK,g["class"],19);u={x for x in (a,b) if x is not None}
            j,_,_=ranked_select(installed,q_rel_class(w),u);nextgold=int(j==ic);u.add(j)
            x=norm3(read_kv(*installed[j],q_read_c(w)))
            if not re.fullmatch(r"[A-Z]{3}",x or ""):return nextgold,0
            j,_,_=ranked_select(installed,q_rel_sector(x),u);u.add(j);x=norm3(read_kv(*installed[j],q_read_d(x)))
            if not re.fullmatch(r"[A-Z]{3}",x or ""):return nextgold,0
            j,_,_=ranked_select(installed,q_rel_node(x),u);y=read_kv(*installed[j],q_read_e(x))
            return nextgold,int(exactcode(y,g["code"]))
        if stage==3:
            w=wrong_symbol(SECTOR_BANK,g["sector"],23);u={x for x in (a,b,c) if x is not None}
            j,_,_=ranked_select(installed,q_rel_sector(w),u);nextgold=int(j==id_);u.add(j)
            x=norm3(read_kv(*installed[j],q_read_d(w)))
            if not re.fullmatch(r"[A-Z]{3}",x or ""):return nextgold,0
            j,_,_=ranked_select(installed,q_rel_node(x),u);y=read_kv(*installed[j],q_read_e(x))
            return nextgold,int(exactcode(y,g["code"]))
        w=wrong_symbol(NODE_BANK,g["node"],29);u={x for x in (a,b,c,d) if x is not None}
        j,_,_=ranked_select(installed,q_rel_node(w),u);nextgold=int(j==ie)
        y=read_kv(*installed[j],q_read_e(w))
        return nextgold,int(exactcode(y,g["code"]))

    w1n,w1f=wrong_chain(1);w2n,w2f=wrong_chain(2);w3n,w3f=wrong_chain(3);w4n,w4f=wrong_chain(4)

    vals={"DIRECT":direct,"ORACLE":oracle,"A":A,"T1":T1,"B":B,"T2":T2,"C":C,"T3":T3,"D":D,"T4":T4,
          "E":E,"VALUE":VALUE,"E2E":E2E,"W1_NEXT":w1n,"W1_FINAL":w1f,"W2_NEXT":w2n,"W2_FINAL":w2f,
          "W3_NEXT":w3n,"W3_FINAL":w3f,"W4_NEXT":w4n,"W4_FINAL":w4f}
    for k,v in vals.items():R[k].append(int(v))

    rs=[r1,r2,r3,r4,r5];rs=["-" if x is None else str(x) for x in rs]
    print(f"      [{z+1:02d}/{N_ITEMS}] {g['entity']:6s} {g['seal']}->{g['class']}->{g['sector']}->{g['node']}->{g['code']} | DIR={int(direct)} OR={int(oracle)} | A={int(A)}(r{rs[0]}) T1={int(T1)} B={int(B)}(r{rs[1]}) T2={int(T2)} C={int(C)}(r{rs[2]}) T3={int(T3)} D={int(D)}(r{rs[3]}) T4={int(T4)} E={int(E)}(r{rs[4]}) FINAL={int(VALUE)} E2E={int(E2E)} | W={w1f}/{w2f}/{w3f}/{w4f}")
    if not E2E:
        sel=[a,b,c,d,e];sel=["NONE" if x is None else f"M{x+1:02d}" for x in sel]
        gold=[f"M{x+1:02d}" for x in (ia,ib,ic,id_,ie)]
        print("               hidden="+"->".join(gold))
        print(f"               selected={'->'.join(sel)} | traces={t1!r}->{t2!r}->{t3!r}->{t4!r} | final={normcode(final)!r}")

OK={k:np.asarray(v,dtype=np.int32) for k,v in R.items()}

print("\n[4/7] Accuracy...")
names=[
("DIRECT","DIRECT 32 ONE-SHOT"),("ORACLE","ORACLE ADDRESS 5-STAGE"),
("A","RANKED A ADDRESS"),("T1","RANKED TRACE-1 SEAL"),
("B","RANKED B ADDRESS"),("T2","RANKED TRACE-2 CLASS"),
("C","RANKED C ADDRESS"),("T3","RANKED TRACE-3 SECTOR"),
("D","RANKED D ADDRESS"),("T4","RANKED TRACE-4 NODE"),
("E","RANKED E ADDRESS"),("VALUE","RANKED FINAL CODE"),("E2E","RANKED END-TO-END"),
("W1_NEXT","WRONG S1→GOLD B"),("W1_FINAL","WRONG S1→GOLD FINAL"),
("W2_NEXT","WRONG S2→GOLD C"),("W2_FINAL","WRONG S2→GOLD FINAL"),
("W3_NEXT","WRONG S3→GOLD D"),("W3_FINAL","WRONG S3→GOLD FINAL"),
("W4_NEXT","WRONG S4→GOLD E"),("W4_FINAL","WRONG S4→GOLD FINAL")]
for k,n in names:print(f"      {n:27s}: {OK[k].sum():2d}/{N_ITEMS} = {OK[k].mean():.4f}")

print("\n[5/7] Retrieval ranks and paired effects...")
def rank_stats(name,x):
    a=np.asarray([v for v in x if v is not None],dtype=float)
    if not len(a):print(f"      {name}: no valid ranks");return
    print(f"      {name}: N={len(a)} R1={np.mean(a<=1):.4f} R5={np.mean(a<=5):.4f} R16={np.mean(a<=16):.4f} MRR={np.mean(1/a):.6f} median={np.median(a):.1f}")
for i,n in enumerate(["A rank","B rank","C rank","D rank","E rank"]):rank_stats(n,RANKS[i])
for a,b in [("E2E","DIRECT"),("ORACLE","E2E"),("E2E","W1_FINAL"),("E2E","W2_FINAL"),("E2E","W3_FINAL"),("E2E","W4_FINAL")]:
    dlt,ci=bootstrap_delta(OK[a],OK[b],10000,SEED+sum(map(ord,a+b)))
    print(f"      {a}-{b}: Δ={dlt:+.4f} | bootstrap95=[{ci[0]:+.4f},{ci[1]:+.4f}]")

print("\n[6/7] Failure localization...")
for k,n in [("ORACLE","Oracle 5-stage relay capacity"),("A","Stage-1 A address"),("T1","Stage-1 Seal trace"),
            ("B","Stage-2 B address"),("T2","Stage-2 Class trace"),("C","Stage-3 C address"),("T3","Stage-3 Sector trace"),
            ("D","Stage-4 D address"),("T4","Stage-4 Node trace"),("E","Stage-5 E address"),
            ("VALUE","Final Code correct"),("E2E","Complete 5-memory traversal")]:
    print(f"      {n:34s}: {OK[k].sum()}/{N_ITEMS}")
for i in range(1,5):
    print(f"      Wrong-S{i} next-memory leakage       : {OK[f'W{i}_NEXT'].sum()}/{N_ITEMS}")
    print(f"      Wrong-S{i} gold-final leakage        : {OK[f'W{i}_FINAL'].sum()}/{N_ITEMS}")

print("\n[7/7] Mechanistic classification...")
DIRECT=OK["DIRECT"].mean();ORACLE=OK["ORACLE"].mean();E2E=OK["E2E"].mean()
W=max(OK[f"W{i}_FINAL"].mean() for i in range(1,5))
if ORACLE<.70:VERDICT="FIVE_STAGE_RELAY_CAPACITY_NOT_RELIABLE"
elif OK["T1"].mean()<.75:VERDICT="STAGE1_ASSOCIATIVE_RECALL_NOT_RELIABLE"
elif OK["T2"].mean()<.70:VERDICT="STAGE2_ASSOCIATIVE_RELAY_NOT_RELIABLE"
elif OK["T3"].mean()<.70:VERDICT="STAGE3_ASSOCIATIVE_RELAY_NOT_RELIABLE"
elif OK["T4"].mean()<.70:VERDICT="STAGE4_ASSOCIATIVE_RELAY_NOT_RELIABLE"
elif OK["E"].mean()<.70:VERDICT="STAGE5_ASSOCIATIVE_ADDRESSING_NOT_RELIABLE"
elif E2E>=.70 and W<=.25:VERDICT="FIVE_MEMORY_SELECTIVE_ASSOCIATIVE_RELAY_REPLICATED"
elif E2E>DIRECT+.25 and E2E>W+.25:VERDICT="FIVE_MEMORY_SELECTIVE_ASSOCIATIVE_RELAY_PARTIAL"
else:VERDICT="FIVE_MEMORY_SELECTIVE_ASSOCIATIVE_RELAY_NOT_YET_RELIABLE"

print("\n"+"="*174)
print("TEST508 FINAL RESULT — AKBASCORE MAM · FIVE-STAGE SELECTIVE ASSOCIATIVE RELAY")
print("="*174)
print("MODEL                         : Qwen/Qwen2.5-7B-Instruct · frozen")
print("BANK                          : 32 independently forged RAW BELLEKÖZ")
print("REQUIRED                      : 5")
print("DISTRACTORS                   : 27")
print("CHAIN                         : Entity → Seal → Class → Sector → Node → Code")
print("REQUIRED ADDRESSES TO RANKED  : NO")
print("GOLD CHAIN IN DISTRACTORS     : NO")
print("TYPE NAMESPACES               : Seal/Class/Sector/Node/Code type-distinct")
print("COMPRESSION                   : OFF")
print("JOINT RE-FORGE                : NONE")
print("DRA / STEERING                : NONE")
print("VTOKEN / ADDRESS VECTOR       : OFF")
print("ANN / LEARNED ROUTER          : NONE")
print("GRAPH                          : NONE")
print("TRAINING / LoRA               : NONE")
print("POST-HOC SELECTION            : NONE")
print("CONTENT SEARCH                : exhaustive O(N)")
print("RANK SIGNAL                   : model first-token logit(YES)-logit(NO)")
print("INTERMEDIATES                 : model-produced textual Seal/Class/Sector/Node")
print("USED MEMORIES RESELECTABLE    : NO")
print("-"*174)
for k,n in [("DIRECT","DIRECT · 32 ONE-SHOT"),("ORACLE","ORACLE ADDRESS 5-STAGE"),
            ("A","RANKED · A ADDRESS"),("T1","RANKED · TRACE-1 SEAL"),
            ("B","RANKED · B ADDRESS"),("T2","RANKED · TRACE-2 CLASS"),
            ("C","RANKED · C ADDRESS"),("T3","RANKED · TRACE-3 SECTOR"),
            ("D","RANKED · D ADDRESS"),("T4","RANKED · TRACE-4 NODE"),
            ("E","RANKED · E ADDRESS"),("VALUE","RANKED · FINAL CODE"),("E2E","RANKED · END-TO-END")]:
    print(f"{n:31s}: {OK[k].sum():2d}/{N_ITEMS} = {OK[k].mean():.4f}")
for i,letter in enumerate(["B","C","D","E"],1):
    print(f"WRONG S{i} · GOLD {letter:5s}          : {OK[f'W{i}_NEXT'].sum():2d}/{N_ITEMS} = {OK[f'W{i}_NEXT'].mean():.4f}")
    print(f"WRONG S{i} · GOLD FINAL          : {OK[f'W{i}_FINAL'].sum():2d}/{N_ITEMS} = {OK[f'W{i}_FINAL'].mean():.4f}")
print("-"*174)
print("TEST501 LOCK SHA              :",TEST501_LOCK_SHA)
print("TEST502 LOCK SHA              :",TEST502_LOCK_SHA)
print("TEST503 LOCK SHA              :",TEST503_LOCK_SHA)
print("TEST504 LOCK SHA              :",TEST504_LOCK_SHA)
print("TEST505 LOCK SHA              :",TEST505_LOCK_SHA)
print("TEST506 LOCK SHA              :",TEST506_LOCK_SHA)
print("TEST507 LOCK SHA              :",TEST507_LOCK_SHA)
print("TEST508 LOCK SHA              :",LOCK_SHA)
print(f"TOTAL TEST TIME                : {time.perf_counter()-T0:.2f}s")
print("EXPLORATORY VERDICT           :",VERDICT)
print("="*174)

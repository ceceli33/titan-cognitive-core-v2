# Copyright © 2026 Mustafa Akbaş
# New original AKBASCORE MAM material is source-available under the repository LICENSE.
# See repository LICENSE for complete terms.
#
# AKBASCORE MAM — TEST507
# THREE-STAGE SELECTIVE ASSOCIATIVE RELAY IN A 32-CARTRIDGE BANK
#
# 32 independently forged RAW BELLEKÖZ per item:
#   3 required memories + 29 independent distractors.
#
# Required chain:
#   QUERY(Entity)
#     → find MEMORY A → Seal
#     → find MEMORY B → Class
#     → find MEMORY C → Code
#
# No required cartridge address is supplied to CONTENT/RANKED arms.
#
# Controls:
#   DIRECT       = all 32 independent memories, one-shot
#   ORACLE_ADDR  = correct A/B/C addresses; intermediate traces remain model-produced
#   RANKED       = exhaustive O(N) model-native YES-vs-NO first-token logit ranking
#   WRONG_S1     = wrong Seal injected before B lookup
#   WRONG_S2     = wrong Class injected before C lookup
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

TEST="507";SEED=507;BASE_SEED=501;MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N_ITEMS=32;BANK_N=32;N_REQ=3;N_DIST=29;MAX_NEW=12
DEVICE=torch.device("cuda");FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
TEST501_LOCK_SHA="1a3e6ea2a9f6c72d7b8dacb0c0844e9f2a6f8595bb232f1554eb41af7a20dc41"
TEST502_LOCK_SHA="f852b6efc6ee887bae54db08cbc11f987e749499a51cebeb5132f569e74edb89"
TEST503_LOCK_SHA="2cb70ffa53a98efb249590dba7ef184b9e41cf72a99291bda65e6c93b82cb5de"
TEST504_LOCK_SHA="74c375373666c840105cc165b90977c56c8698cf3d5617c63f619a0139386846"
TEST505_LOCK_SHA="6d093a5d082b43ff56f09bb94cfed1839c0bd0d76a967822937996a5d1948e43"
TEST506_LOCK_SHA="5f14f6660c5ce99813544c93420b00559a9fc4612813d326fa02df4f76d65327"

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
CLASS_BANK_RAW=["TAK","BEX","LUM","RAV","SOD","PEK","NIV","GOR","HAX","JUR","KEM","VOL","DAX","FIR","MON","SAL","TEK","WIR","ZUN","COV","HEM","JAX","LIV","NOR","PAK","RUM","SEV","TIX","VOR","YAM","ZEK","BOL","CER","DOV","FEX","GAM","HUR","JIN","KAV","LER","MEX","NUR","PIV","ROK","SUM","TAL","VEK","WON","XIR","YAV","ZOL","BAR","CIX","DEM","FOV","GEL","HIN","JOV","KUR","LEV","MAV","NEX","PUL","RIM","SAV","TOX","VIL","WER","XAN","YER","ZIM","BUN","CAL","DOR","EVI","FAR","GUN","HES","ILM","JER","KON","LAR","MUR","NOL","OVI","PER","RUS","SIN","TUR","VEX","WAL","XEN","YUL","ZAR","BEL","CUM"]
CODE_BANK=["AQ7","BR4","CX9","DM2","EV8","FK3","GL6","HN5","JP7","KR2","LS9","MT4","NV6","PX3","QH8","RJ5",
"SK7","TL2","UM9","VW4","WX6","YB3","ZC8","AD5","BE7","CF2","DG9","EH4","FI6","GJ3","HK8","IL5",
"JM7","KN2","LO9","MP4","NQ6","OR3","PS8","QT5","RU7","SV2","TW9","UX4","VY6","WZ3","XA8","YC5",
"ZD7","AE2","BF9","CG4","DH6","EI3","FJ8","GK5","HL7","IM2","JN9","KO4","LP6","MQ3","NR8","OS5",
"PT7","QU2","RV9","SW4","TX6","UY3","VZ8","WA5","XB7","YC2","ZD9","AF4","BG6","CH3","DI8","EJ5",
"FK7","GL2","HM9","IN4","JO6","KP3","LQ8","MR5","NS7","OT2","PU9","QV4","RW6","SX3","TY8","UZ5"]

# Type-namespace sanitization.
# TEST507 adds a third relation family, therefore a Seal token must never also be
# a Class token. Only colliding CLASS symbols are deterministically replaced.
def sanitize_class_namespace(raw_classes,seals,codes):
    forbidden=set(seals)|set(codes)
    used=set(raw_classes)|forbidden
    candidates=[]
    for a in "QZXWVUTSRPONMLKJIHGFEDCBA":
        for b in string.ascii_uppercase:
            for c in string.ascii_uppercase:
                x=a+b+c
                if x not in used:
                    candidates.append(x);used.add(x)
    it=iter(candidates);out=[];mapping={}
    for x in raw_classes:
        if x in forbidden:
            y=next(it);mapping[x]=y;out.append(y)
        else:out.append(x)
    return out,mapping

CLASS_BANK,CLASS_NAMESPACE_REMAP=sanitize_class_namespace(CLASS_BANK_RAW,SEAL_BANK,CODE_BANK)

assert len(ENTITY_BANK)==N_ITEMS
assert len(SEAL_BANK)>=N_ITEMS*3 and len(CLASS_BANK)>=N_ITEMS*3 and len(CODE_BANK)>=N_ITEMS*3
assert len(set(SEAL_BANK))==len(SEAL_BANK)
assert len(set(CLASS_BANK))==len(CLASS_BANK)
assert len(set(CODE_BANK))==len(CODE_BANK)
assert set(SEAL_BANK).isdisjoint(CLASS_BANK)
assert set(SEAL_BANK).isdisjoint(CODE_BANK)
assert set(CLASS_BANK).isdisjoint(CODE_BANK)
assert all(re.fullmatch(r"[A-Z]{3}",x) for x in SEAL_BANK)
assert all(re.fullmatch(r"[A-Z]{3}",x) for x in CLASS_BANK)
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
    seals=SEAL_BANK.copy();classes=CLASS_BANK.copy();codes=CODE_BANK.copy()
    rng.shuffle(seals);rng.shuffle(classes);rng.shuffle(codes);items=[]
    for i in range(N_ITEMS):
        ents=list(ENTITY_BANK[i]);g=i%3
        ss=seals[3*i:3*i+3];cc=classes[3*i:3*i+3];dd=codes[3*i:3*i+3]
        if len(ss)!=3 or len(cc)!=3 or len(dd)!=3:raise RuntimeError(f"Base pool exhaustion item {i+1}.")
        r1=[f"Instrument {ents[j]} carries seal {ss[j]}." for j in range(3)]
        r2=[f"Seal {ss[j]} corresponds to routing class {cc[j]}." for j in range(3)]
        r3=[f"Routing class {cc[j]} carries recovery code {dd[j]}." for j in range(3)]
        p1=i%3;p2=(i+1)%3;p3=(i+2)%3
        rec1=" ".join(reorder(r1,g,p1,BASE_SEED+i*101+17))
        rec2=" ".join(reorder(r2,g,p2,BASE_SEED+i*101+34))
        rec3=" ".join(reorder(r3,g,p3,BASE_SEED+i*101+51))
        items.append({"id":i+1,"records":[rec1,rec2,rec3],
                      "gold":{"entity":ents[g],"seal":ss[g],"class":cc[g],"code":dd[g]},
                      "positions":[p1,p2,p3]})
    return items

BASE=make_base()

# 29 distractors:
#   10 Entity→Seal
#   10 Seal→Class
#    9 Class→Code
# Current target Entity/Seal/Class/Code are forbidden from all distractors.
def make_distractors(i,g):
    rng=random.Random(SEED+90000+i)
    forbidden={g["seal"],g["class"],g["code"]}
    ep=[e for j,t in enumerate(ENTITY_BANK) if j!=i for e in t if e.lower()!=g["entity"].lower()]
    sp=[s for s in SEAL_BANK if s not in forbidden]
    cp=[c for c in CLASS_BANK if c not in forbidden]
    dp=[d for d in CODE_BANK if d not in forbidden]
    rng.shuffle(ep);rng.shuffle(sp);rng.shuffle(cp);rng.shuffle(dp)
    if len(ep)<30 or len(sp)<60 or len(cp)<57 or len(dp)<27:
        raise RuntimeError(f"Insufficient distractor pool at item {i+1}: E={len(ep)} S={len(sp)} C={len(cp)} D={len(dp)}")
    ds=[]
    for d in range(10):
        es=ep[d*3:d*3+3];ss=sp[d*3:d*3+3]
        ds.append(" ".join(f"Instrument {es[j]} carries seal {ss[j]}." for j in range(3)))
    for d in range(10):
        ss=sp[30+d*3:30+d*3+3];cc=cp[d*3:d*3+3]
        ds.append(" ".join(f"Seal {ss[j]} corresponds to routing class {cc[j]}." for j in range(3)))
    for d in range(9):
        cc=cp[30+d*3:30+d*3+3];dd=dp[d*3:d*3+3]
        ds.append(" ".join(f"Routing class {cc[j]} carries recovery code {dd[j]}." for j in range(3)))
    if len(ds)!=N_DIST:raise RuntimeError(f"Distractor count failure item {i+1}: {len(ds)}")
    for n,x in enumerate(ds):
        for key in ("entity","seal","class","code"):
            if token_present(x,g[key]):raise RuntimeError(f"{key} contamination item {i+1}, distractor {n+1}: {x}")
    return ds

ITEMS=[]
for i,b in enumerate(BASE):
    g=b["gold"];ds=make_distractors(i,g)
    mems=[
        {"role":"A","text":b["records"][0]},
        {"role":"B","text":b["records"][1]},
        {"role":"C","text":b["records"][2]}
    ]+[{"role":"D","text":x} for x in ds]
    rng=random.Random(SEED+70000+i);rng.shuffle(mems)
    if len(mems)!=BANK_N:raise RuntimeError(f"Bank size failure item {i+1}: {len(mems)}")
    counts={r:sum(m["role"]==r for m in mems) for r in ("A","B","C","D")}
    if counts!={"A":1,"B":1,"C":1,"D":N_DIST}:raise RuntimeError(f"Role-count failure item {i+1}: {counts}")
    ia=next(j for j,m in enumerate(mems) if m["role"]=="A")
    ib=next(j for j,m in enumerate(mems) if m["role"]=="B")
    ic=next(j for j,m in enumerate(mems) if m["role"]=="C")
    if len({ia,ib,ic})!=3:raise RuntimeError(f"Required-address collision item {i+1}.")
    ent_hits=[j for j,m in enumerate(mems) if token_present(m["text"],g["entity"])]
    seal_hits=[j for j,m in enumerate(mems) if token_present(m["text"],g["seal"])]
    class_hits=[j for j,m in enumerate(mems) if token_present(m["text"],g["class"])]
    code_hits=[j for j,m in enumerate(mems) if token_present(m["text"],g["code"])]
    if ent_hits!=[ia]:raise RuntimeError(f"Entity uniqueness failure item {i+1}: {ent_hits}/{[ia]}")
    if sorted(seal_hits)!=sorted([ia,ib]):raise RuntimeError(f"Seal uniqueness failure item {i+1}: {seal_hits}/{[ia,ib]}")
    if sorted(class_hits)!=sorted([ib,ic]):raise RuntimeError(f"Class uniqueness failure item {i+1}: {class_hits}/{[ib,ic]}")
    if code_hits!=[ic]:raise RuntimeError(f"Code uniqueness failure item {i+1}: {code_hits}/{[ic]}")
    ITEMS.append({**b,"bank":mems,"oracle":[ia,ib,ic]})

LOCK={"test":TEST,"model":MODEL_ID,"seed":SEED,"base_panel_seed":BASE_SEED,"items":ITEMS,
      "bank_n":BANK_N,"required":N_REQ,"distractors":N_DIST,
      "class_namespace_remap":CLASS_NAMESPACE_REMAP,
      "namespace_guard":"Seal/Class/Code pairwise disjoint",
      "test501":TEST501_LOCK_SHA,"test502":TEST502_LOCK_SHA,"test503":TEST503_LOCK_SHA,
      "test504":TEST504_LOCK_SHA,"test505":TEST505_LOCK_SHA,"test506":TEST506_LOCK_SHA,
      "chain":"Entity->Seal->Class->Code",
      "arms":["DIRECT_32","ORACLE_ADDRESS_3_STAGE","RANKED_3_STAGE","WRONG_S1","WRONG_S2"],
      "rank_signal":"first-answer-token logit(YES)-logit(NO)",
      "stage2_excludes":["selected_stage1"],"stage3_excludes":["selected_stage1","selected_stage2"],
      "compression":"OFF","joint_reforge":"OFF","steering":"OFF","vtoken":"OFF",
      "ann_router":"OFF","learned_router":"OFF","graph":"OFF","training":"OFF","posthoc":"OFF"}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

print("="*170)
print("TEST507 — AKBASCORE MAM · THREE-STAGE SELECTIVE ASSOCIATIVE RELAY IN A 32-CARTRIDGE BANK")
print("32 INDEPENDENT BELLEKÖZ · 3 REQUIRED + 29 DISTRACTORS · ENTITY → SEAL → CLASS → CODE")
print("="*170)
print("LOCK SHA:",LOCK_SHA)
print("TEST506 LOCK:",TEST506_LOCK_SHA)
print("ITEMS:",N_ITEMS,"| BANK:",BANK_N,"| REQUIRED:",N_REQ,"| DISTRACTORS:",N_DIST)
print("CONTENT ADDRESSING: exhaustive O(N); gold cartridge addresses hidden from RANKED arm")
print("COMPRESSION / JOINT RE-FORGE / DRA / VTOKEN / ANN / LEARNED ROUTER / GRAPH / TRAINING: NONE")
print("TYPE NAMESPACE GUARD: Seal / Class / Code pairwise disjoint")
print("CLASS COLLISIONS REMAPPED:",len(CLASS_NAMESPACE_REMAP),CLASS_NAMESPACE_REMAP)
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

def q_direct(entity):
    return f"What recovery code ultimately corresponds to instrument {entity}? Follow the available memories and give only the exact recovery code."
def q_rel_entity(entity):
    return f"Does this memory explicitly contain a relation stating which seal instrument {entity} carries? Answer only YES or NO."
def q_rel_seal(seal):
    return f"Does this memory explicitly contain a relation stating which routing class seal {seal} corresponds to? Answer only YES or NO."
def q_rel_class(cls):
    return f"Does this memory explicitly contain a relation stating which recovery code routing class {cls} carries? Answer only YES or NO."
def q_read_a(entity):
    return f"What seal does instrument {entity} carry? Give only the exact seal."
def q_read_b(seal):
    return f"What routing class corresponds to seal {seal}? Give only the exact routing class."
def q_read_c(cls):
    return f"What recovery code does routing class {cls} carry? Give only the exact recovery code."

def ranked_select(installed,q,excluded=None):
    excluded=set() if excluded is None else set(excluded);scores=[]
    for j,(inst,Tm,P) in enumerate(installed):
        if j in excluded:continue
        scores.append((yesno_margin(inst,Tm,P,q),j))
    if not scores:raise RuntimeError("ranked_select: no candidate memories remain.")
    scores.sort(key=lambda x:(-x[0],x[1]))
    return scores[0][1],scores[0][0],scores

def rank_of(scores,gold_idx):
    for r,(_,j) in enumerate(scores,1):
        if j==gold_idx:return r
    return None

def wrong_symbol(pool,gold,offset):
    if gold not in pool:raise RuntimeError(f"Gold symbol {gold!r} absent from wrong-symbol pool.")
    if len(pool)<2:raise RuntimeError("Wrong-symbol pool too small.")
    j=pool.index(gold);step=offset%(len(pool)-1)+1
    x=pool[(j+step)%len(pool)]
    if x==gold:raise RuntimeError("Wrong-symbol construction failure.")
    return x

def bootstrap_delta(a,b,B=10000,seed=507):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64);d=a-b;n=len(d)
    rng=np.random.default_rng(seed);vals=np.empty(B)
    for i in range(B):vals[i]=d[rng.integers(0,n,n)].mean()
    return float(d.mean()),tuple(np.quantile(vals,[.025,.975]))

print("\n[2/7] Bank seal...")
for it in ITEMS[:6]:
    ia,ib,ic=it["oracle"];g=it["gold"]
    print(f"      ITEM {it['id']:02d} {g['entity']:6s} {g['seal']}->{g['class']}->{g['code']} | hidden A=M{ia+1:02d} B=M{ib+1:02d} C=M{ic+1:02d}")
print("      32 independent memories/item; A/B/C addresses hidden from RANKED.")
print("      Gold Entity/Seal/Class/Code absent from all 29 distractors.")
print("      Seal/Class/Code namespaces are pairwise disjoint.")

R={k:[] for k in [
    "DIRECT","ORACLE_A","ORACLE_B","ORACLE_C","ORACLE",
    "RANK_A","TRACE1","RANK_B","TRACE2","RANK_C","VALUE","E2E",
    "WRONG_S1_B","WRONG_S1_FINAL","WRONG_S2_C","WRONG_S2_FINAL"
]}
RANK_A=[];RANK_B=[];RANK_C=[]

print("\n[3/7] Forging and running 32-cartridge three-stage banks...")
for z,it in enumerate(ITEMS):
    g=it["gold"];ia,ib,ic=it["oracle"]
    raws=[forge(m["text"]) for m in it["bank"]]
    installed=[install_single(r) for r in raws]

    # DIRECT: all 32 independent memories simultaneously.
    instD,TmD,PD=install_parallel(raws)
    direct=read_kv(instD,TmD,PD,q_direct(g["entity"]))
    direct_ok=exactcode(direct,g["code"])

    # ORACLE ADDRESS: correct A/B/C addresses, traces remain model-generated.
    aI,aT,aP=installed[ia]
    oa=read_kv(aI,aT,aP,q_read_a(g["entity"]));ot1=norm3(oa)
    bI,bT,bP=installed[ib]
    ob=read_kv(bI,bT,bP,q_read_b(ot1));ot2=norm3(ob)
    cI,cT,cP=installed[ic]
    oc=read_kv(cI,cT,cP,q_read_c(ot2))
    oracle_a=exact3(oa,g["seal"])
    oracle_b=oracle_a and exact3(ob,g["class"])
    oracle_c=oracle_b and exactcode(oc,g["code"])
    oracle_ok=oracle_c

    # RANKED STAGE 1: Entity → A → Seal
    ra_idx,_,ra_scores=ranked_select(installed,q_rel_entity(g["entity"]))
    ra_rank=rank_of(ra_scores,ia);RANK_A.append(ra_rank)
    raI,raT,raP=installed[ra_idx]
    t1_raw=read_kv(raI,raT,raP,q_read_a(g["entity"]));t1=norm3(t1_raw)
    rank_a=ra_idx==ia
    trace1=rank_a and t1==g["seal"]

    # RANKED STAGE 2: model-produced Seal → B → Class
    rb_idx=None;rb_rank=None;t2="";rank_b=False;trace2=False
    if re.fullmatch(r"[A-Z]{3}",t1 or ""):
        rb_idx,_,rb_scores=ranked_select(installed,q_rel_seal(t1),excluded={ra_idx})
        rb_rank=rank_of(rb_scores,ib);RANK_B.append(rb_rank)
        rbI,rbT,rbP=installed[rb_idx]
        t2_raw=read_kv(rbI,rbT,rbP,q_read_b(t1));t2=norm3(t2_raw)
        rank_b=rb_idx==ib
        trace2=trace1 and rank_b and t2==g["class"]
    else:RANK_B.append(None)

    # RANKED STAGE 3: model-produced Class → C → Code
    rc_idx=None;rc_rank=None;final="";rank_c=False;value=False;e2e=False
    used={ra_idx}
    if rb_idx is not None:used.add(rb_idx)
    if re.fullmatch(r"[A-Z]{3}",t2 or ""):
        rc_idx,_,rc_scores=ranked_select(installed,q_rel_class(t2),excluded=used)
        rc_rank=rank_of(rc_scores,ic);RANK_C.append(rc_rank)
        rcI,rcT,rcP=installed[rc_idx]
        final=read_kv(rcI,rcT,rcP,q_read_c(t2))
        rank_c=rc_idx==ic
        value=rank_c and exactcode(final,g["code"])
        e2e=trace2 and value
    else:RANK_C.append(None)

    # WRONG STAGE-1 TRACE: deterministic wrong Seal before B lookup.
    ws1=wrong_symbol(SEAL_BANK,g["seal"],17)
    if ws1==g["seal"] or ws1 in CLASS_BANK or ws1 in CODE_BANK:raise RuntimeError(f"Wrong-S1 namespace failure item {z+1}: {ws1}")
    w1b_idx,_,_=ranked_select(installed,q_rel_seal(ws1),excluded={ra_idx})
    w1bI,w1bT,w1bP=installed[w1b_idx]
    w1class=norm3(read_kv(w1bI,w1bT,w1bP,q_read_b(ws1)))
    w1c_idx=None;w1final=""
    if re.fullmatch(r"[A-Z]{3}",w1class or ""):
        w1c_idx,_,_=ranked_select(installed,q_rel_class(w1class),excluded={ra_idx,w1b_idx})
        w1cI,w1cT,w1cP=installed[w1c_idx]
        w1final=read_kv(w1cI,w1cT,w1cP,q_read_c(w1class))
    wrong_s1_b=int(w1b_idx==ib)
    wrong_s1_final=int(exactcode(w1final,g["code"]))

    # WRONG STAGE-2 TRACE: deterministic wrong Class directly before C lookup.
    ws2=wrong_symbol(CLASS_BANK,g["class"],19)
    if ws2==g["class"] or ws2 in SEAL_BANK or ws2 in CODE_BANK:raise RuntimeError(f"Wrong-S2 namespace failure item {z+1}: {ws2}")
    ex2={ra_idx}
    if rb_idx is not None:ex2.add(rb_idx)
    w2c_idx,_,_=ranked_select(installed,q_rel_class(ws2),excluded=ex2)
    w2cI,w2cT,w2cP=installed[w2c_idx]
    w2final=read_kv(w2cI,w2cT,w2cP,q_read_c(ws2))
    wrong_s2_c=int(w2c_idx==ic)
    wrong_s2_final=int(exactcode(w2final,g["code"]))

    R["DIRECT"].append(int(direct_ok))
    R["ORACLE_A"].append(int(oracle_a));R["ORACLE_B"].append(int(oracle_b));R["ORACLE_C"].append(int(oracle_c));R["ORACLE"].append(int(oracle_ok))
    R["RANK_A"].append(int(rank_a));R["TRACE1"].append(int(trace1));R["RANK_B"].append(int(rank_b));R["TRACE2"].append(int(trace2))
    R["RANK_C"].append(int(rank_c));R["VALUE"].append(int(value));R["E2E"].append(int(e2e))
    R["WRONG_S1_B"].append(wrong_s1_b);R["WRONG_S1_FINAL"].append(wrong_s1_final)
    R["WRONG_S2_C"].append(wrong_s2_c);R["WRONG_S2_FINAL"].append(wrong_s2_final)

    ras=str(ra_rank) if ra_rank is not None else "-"
    rbs=str(rb_rank) if rb_rank is not None else "-"
    rcs=str(rc_rank) if rc_rank is not None else "-"
    print(f"      [{z+1:02d}/{N_ITEMS}] {g['entity']:6s} {g['seal']}->{g['class']}->{g['code']} | DIR={int(direct_ok)} OR={int(oracle_ok)} | A={int(rank_a)}(r{ras}) T1={int(trace1)} B={int(rank_b)}(r{rbs}) T2={int(trace2)} C={int(rank_c)}(r{rcs}) V={int(value)} E2E={int(e2e)} | W1={wrong_s1_final} W2={wrong_s2_final}")
    if not e2e:
        rbtxt=f"M{rb_idx+1:02d}" if rb_idx is not None else "NONE"
        rctxt=f"M{rc_idx+1:02d}" if rc_idx is not None else "NONE"
        print(f"               hidden=M{ia+1:02d}->M{ib+1:02d}->M{ic+1:02d}")
        print(f"               selected=M{ra_idx+1:02d} trace1={t1!r} -> {rbtxt} trace2={t2!r} -> {rctxt} final={normcode(final)!r}")

print("\n[4/7] Accuracy...")
OK={k:np.asarray(v,dtype=np.int32) for k,v in R.items()}
for k,name in [
    ("DIRECT","DIRECT 32 ONE-SHOT"),
    ("ORACLE_A","ORACLE A→SEAL"),
    ("ORACLE_B","ORACLE A→B→CLASS"),
    ("ORACLE_C","ORACLE A→B→C→CODE"),
    ("ORACLE","ORACLE ADDR E2E"),
    ("RANK_A","RANKED A ADDRESS"),
    ("TRACE1","RANKED TRACE-1"),
    ("RANK_B","RANKED B ADDRESS"),
    ("TRACE2","RANKED TRACE-2"),
    ("RANK_C","RANKED C ADDRESS"),
    ("VALUE","RANKED FINAL VALUE"),
    ("E2E","RANKED END-TO-END"),
    ("WRONG_S1_B","WRONG S1→GOLD B"),
    ("WRONG_S1_FINAL","WRONG S1→GOLD FINAL"),
    ("WRONG_S2_C","WRONG S2→GOLD C"),
    ("WRONG_S2_FINAL","WRONG S2→GOLD FINAL")
]:
    print(f"      {name:26s}: {OK[k].sum():2d}/{N_ITEMS} = {OK[k].mean():.4f}")

print("\n[5/7] Retrieval ranks and paired effects...")
def rank_stats(name,x):
    a=np.asarray([v for v in x if v is not None],dtype=float)
    if not len(a):
        print(f"      {name}: no valid ranks");return
    print(f"      {name}: N={len(a)} R1={np.mean(a<=1):.4f} R5={np.mean(a<=5):.4f} R16={np.mean(a<=16):.4f} MRR={np.mean(1/a):.6f} median={np.median(a):.1f}")
rank_stats("A rank",RANK_A);rank_stats("B rank",RANK_B);rank_stats("C rank",RANK_C)
for a,b in [("E2E","DIRECT"),("ORACLE","E2E"),("E2E","WRONG_S1_FINAL"),("E2E","WRONG_S2_FINAL")]:
    d,ci=bootstrap_delta(OK[a],OK[b],10000,SEED+sum(map(ord,a+b)))
    print(f"      {a}-{b}: Δ={d:+.4f} | bootstrap95=[{ci[0]:+.4f},{ci[1]:+.4f}]")

print("\n[6/7] Failure localization...")
print(f"      Oracle 3-stage relay capacity     : {OK['ORACLE'].sum()}/{N_ITEMS}")
print(f"      Stage-1 A address                 : {OK['RANK_A'].sum()}/{N_ITEMS}")
print(f"      Stage-1 correct Seal trace        : {OK['TRACE1'].sum()}/{N_ITEMS}")
print(f"      Stage-2 B address                 : {OK['RANK_B'].sum()}/{N_ITEMS}")
print(f"      Stage-2 correct Class trace       : {OK['TRACE2'].sum()}/{N_ITEMS}")
print(f"      Stage-3 C address                 : {OK['RANK_C'].sum()}/{N_ITEMS}")
print(f"      Final Code correct                : {OK['VALUE'].sum()}/{N_ITEMS}")
print(f"      Complete 3-memory traversal       : {OK['E2E'].sum()}/{N_ITEMS}")
print(f"      Wrong-S1 gold-B leakage           : {OK['WRONG_S1_B'].sum()}/{N_ITEMS}")
print(f"      Wrong-S1 gold-final leakage       : {OK['WRONG_S1_FINAL'].sum()}/{N_ITEMS}")
print(f"      Wrong-S2 gold-C leakage           : {OK['WRONG_S2_C'].sum()}/{N_ITEMS}")
print(f"      Wrong-S2 gold-final leakage       : {OK['WRONG_S2_FINAL'].sum()}/{N_ITEMS}")
if OK["TRACE1"].sum():
    n=int(OK["TRACE1"].sum());c=int(((OK["TRACE1"]==1)&(OK["TRACE2"]==1)).sum())
    print(f"      Correct Trace-2 | correct Trace-1 : {c}/{n} = {c/n:.4f}")
if OK["TRACE2"].sum():
    n=int(OK["TRACE2"].sum());c=int(((OK["TRACE2"]==1)&(OK["VALUE"]==1)).sum())
    print(f"      Final value | correct Trace-2     : {c}/{n} = {c/n:.4f}")

print("\n[7/7] Mechanistic classification...")
DIRECT=OK["DIRECT"].mean();ORACLE=OK["ORACLE"].mean()
A=OK["RANK_A"].mean();T1=OK["TRACE1"].mean();B=OK["RANK_B"].mean();T2=OK["TRACE2"].mean()
C=OK["RANK_C"].mean();VALUE=OK["VALUE"].mean();E2E=OK["E2E"].mean()
W1=OK["WRONG_S1_FINAL"].mean();W2=OK["WRONG_S2_FINAL"].mean()
if ORACLE<.70:
    VERDICT="THREE_STAGE_RELAY_CAPACITY_NOT_RELIABLE"
elif T1<.75:
    VERDICT="STAGE1_ASSOCIATIVE_RECALL_NOT_RELIABLE"
elif T2<.70:
    VERDICT="STAGE2_ASSOCIATIVE_RELAY_NOT_RELIABLE"
elif C<.70:
    VERDICT="STAGE3_ASSOCIATIVE_ADDRESSING_NOT_RELIABLE"
elif E2E>=.70 and W1<=.25 and W2<=.25:
    VERDICT="THREE_MEMORY_SELECTIVE_ASSOCIATIVE_RELAY_REPLICATED"
elif E2E>DIRECT+.25 and E2E>max(W1,W2)+.25:
    VERDICT="THREE_MEMORY_SELECTIVE_ASSOCIATIVE_RELAY_PARTIAL"
else:
    VERDICT="THREE_MEMORY_SELECTIVE_ASSOCIATIVE_RELAY_NOT_YET_RELIABLE"

print("\n"+"="*170)
print("TEST507 FINAL RESULT — AKBASCORE MAM · THREE-STAGE SELECTIVE ASSOCIATIVE RELAY")
print("="*170)
print("MODEL                         : Qwen/Qwen2.5-7B-Instruct · frozen")
print("BANK                          : 32 independently forged RAW BELLEKÖZ")
print("REQUIRED                      : 3")
print("DISTRACTORS                   : 29")
print("CHAIN                         : Entity → Seal → Class → Code")
print("REQUIRED ADDRESSES TO RANKED  : NO")
print("TARGET ENTITY IN DISTRACTORS  : NO")
print("BRIDGE SEAL IN DISTRACTORS    : NO")
print("BRIDGE CLASS IN DISTRACTORS   : NO")
print("FINAL CODE IN DISTRACTORS     : NO")
print("TYPE NAMESPACES               : Seal / Class / Code pairwise disjoint")
print("CLASS COLLISIONS REMAPPED     :",len(CLASS_NAMESPACE_REMAP))
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
print("INTERMEDIATES                 : model-produced textual Seal + Class")
print("USED MEMORIES RESELECTABLE    : NO")
print("-"*170)
print(f"DIRECT · 32 ONE-SHOT           : {OK['DIRECT'].sum():2d}/{N_ITEMS} = {DIRECT:.4f}")
print(f"ORACLE ADDRESS 3-STAGE         : {OK['ORACLE'].sum():2d}/{N_ITEMS} = {ORACLE:.4f}")
print(f"RANKED · A ADDRESS             : {OK['RANK_A'].sum():2d}/{N_ITEMS} = {A:.4f}")
print(f"RANKED · TRACE-1 SEAL          : {OK['TRACE1'].sum():2d}/{N_ITEMS} = {T1:.4f}")
print(f"RANKED · B ADDRESS             : {OK['RANK_B'].sum():2d}/{N_ITEMS} = {B:.4f}")
print(f"RANKED · TRACE-2 CLASS         : {OK['TRACE2'].sum():2d}/{N_ITEMS} = {T2:.4f}")
print(f"RANKED · C ADDRESS             : {OK['RANK_C'].sum():2d}/{N_ITEMS} = {C:.4f}")
print(f"RANKED · FINAL CODE            : {OK['VALUE'].sum():2d}/{N_ITEMS} = {VALUE:.4f}")
print(f"RANKED · END-TO-END            : {OK['E2E'].sum():2d}/{N_ITEMS} = {E2E:.4f}")
print(f"WRONG S1 · GOLD B              : {OK['WRONG_S1_B'].sum():2d}/{N_ITEMS} = {OK['WRONG_S1_B'].mean():.4f}")
print(f"WRONG S1 · GOLD FINAL          : {OK['WRONG_S1_FINAL'].sum():2d}/{N_ITEMS} = {W1:.4f}")
print(f"WRONG S2 · GOLD C              : {OK['WRONG_S2_C'].sum():2d}/{N_ITEMS} = {OK['WRONG_S2_C'].mean():.4f}")
print(f"WRONG S2 · GOLD FINAL          : {OK['WRONG_S2_FINAL'].sum():2d}/{N_ITEMS} = {W2:.4f}")
print("-"*170)
print("TEST501 LOCK SHA              :",TEST501_LOCK_SHA)
print("TEST502 LOCK SHA              :",TEST502_LOCK_SHA)
print("TEST503 LOCK SHA              :",TEST503_LOCK_SHA)
print("TEST504 LOCK SHA              :",TEST504_LOCK_SHA)
print("TEST505 LOCK SHA              :",TEST505_LOCK_SHA)
print("TEST506 LOCK SHA              :",TEST506_LOCK_SHA)
print("TEST507 LOCK SHA              :",LOCK_SHA)
print(f"TOTAL TEST TIME                : {time.perf_counter()-T0:.2f}s")
print("EXPLORATORY VERDICT           :",VERDICT)
print("="*170)

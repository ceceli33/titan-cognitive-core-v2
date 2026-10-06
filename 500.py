# Copyright © 2026 Mustafa Akbaş
# New original AKBASCORE MAM material is source-available under the repository LICENSE.
# Earlier MIT-licensed Titan Cognitive Core / AkbasCore material remains under its original license.
# See repository LICENSE for complete terms.
#
# TEST500 — AKBASCORE MAM · WRITE-TIME BRIDGING vs INDEPENDENT BELLEKÖZ
# CLEAN MULTI-ITEM FALSIFICATION PANEL
#
# PURPOSE:
#   Determine whether a composition that frozen Qwen can solve from contiguous text/cache
#   breaks specifically when the same information is written as independent memory packets.
#
# FROZEN FROM TEST496:
#   Qwen/Qwen2.5-7B-Instruct
#   TEST482 K120/V128/OWN compression
#   TEST482 neutral PCA codebook
#   TEST495/496 parallel sequence-dimension cache composition
#   forge / RoPE / DynamicCache / greedy reader architecture
#
# TEST500 CONDITIONS:
#   A = CONTIGUOUS TEXT IN PROMPT
#   B = CONTIGUOUS RAW K/V CACHE
#   C = INDEPENDENT RAW K/V PARALLEL CACHE
#   D = INDEPENDENT K120/V128 BELLEKÖZ PARALLEL CACHE
#   E = D + EXPLICIT HOP-TRACE DECODING
#
# CLEAN DESIGN:
#   32 fresh deterministic items
#   3 competing entities per item
#   target row position rotates across records/items
#   no fixed "gold is first row" shortcut
#   no literal final answer in any memory
#   no VTOKEN / Top-K / router / ANN / graph / training / steering
#   no post-hoc item selection
#
# INTERPRETATION:
#   A fail                -> task/reader problem; do not blame MAM
#   A,B pass; C fail      -> independent-write / write-time-bridging deficit
#   C pass; D fail        -> K120 compression deficit
#   D fail; E rescue      -> explicit read-cycle/hop decoding restores composition
#
# TEST496 remains sealed. TEST500 is a new diagnostic line.

import os,sys,subprocess,importlib.util,random,re,time,hashlib,json
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

TEST="500";SEED=500;MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
K_DIM=120;V_DIM=128;N_ITEMS=32;MAX_NEW=24;TRACE_NEW=48;DEVICE=torch.device("cuda")
FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
TEST482_BLOB_SHA="ecc630144a44479cb35c432c50eb0d09bc6866d7"
TEST482_LOCK_SHA="fa59fd38661e558f6bae22eedff0999f32f8f6e9d4a08932383525323a5fe687"
TEST495_LOCK_SHA="2bae3f2497fefb50b3d18d8fb7035c63a08fdd67d80077dd92f4e9c56eca787e"

CORPUS=[
"Jonas Weber carried the wooden crate across the quiet market square.",
"Priya Nair wrote a long letter to her cousin in the morning.",
"The small boat drifted slowly toward the rocky shore.",
"Omar Haddad fixed the broken clock in the village school.",
"A tired teacher closed the green door of the library.",
"Sofia Rossi baked fresh bread for the harvest festival.",
"The children watched the kites rising above the hill.",
"Liam O'Connor sold his old bicycle to a neighbor.",
"Heavy rain flooded the narrow street near the market.",
"Nadia Petrova translated the ancient manuscript into French.",
"The farmer counted the sheep before sunset.",
"Hiro Sato opened a tiny bakery beside the river.",
"An old dog slept under the kitchen table all afternoon.",
"Carlos Mendes washed the windows of his grandmother's house.",
"The museum guard locked the heavy gate at midnight.",
"Fatima Zahra grew tomatoes in the backyard garden.",
"The pilot announced a short delay because of fog.",
"Anna Kowalski found a lost wallet on the bus.",
"Snow covered the mountain village during the night.",
"Ravi Kumar taught his brother how to play chess.",
"The chef sharpened every knife before dinner service.",
"Lucas Martin cleaned the roof of the barn after the storm.",
"A young violinist practiced scales in the empty hall.",
"Mei Lin delivered the package to the wrong address.",
"Marek Novak placed the glass bottle beneath the wooden bench.",
"Sara Ibrahim carried a red notebook into the quiet classroom.",
"Noah Schmidt repaired the small radio beside the kitchen window.",
"Yuki Mori left a paper envelope near the station entrance.",
"Amira Hassan moved the ceramic bowl onto the upper shelf.",
"Peter Novak opened the metal box behind the old theater.",
"Lucia Costa placed the yellow scarf inside the travel bag.",
"Daniel Kim carried a black umbrella through the central courtyard."
]

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
CLASS_BANK=["TAK","BEX","LUM","RAV","SOD","PEK","NIV","GOR","HAX","JUR","KEM","VOL","DAX","FIR","MON","SAL","TEK","WIR","ZUN","COV","HEM","JAX","LIV","NOR","PAK","RUM","SEV","TIX","VOR","YAM","ZEK","BOL","CER","DOV","FEX","GAM","HUR","JIN","KAV","LER","MEX","NUR","PIV","ROK","SUM","TAL","VEK","WON","XIR","YAV","ZOL","BAR","CIX","DEM","FOV","GEL","HIN","JOV","KUR","LEV","MAV","NEX","PUL","RIM","SAV","TOX","VIL","WER","XAN","YER","ZIM","BUN","CAL","DOR","EVI","FAR","GUN","HES","ILM","JER","KON","LAR","MUR","NOL","OVI","PER","RUS","SIN","TUR","VEX","WAL","XEN","YUL","ZAR","BEL","CUM"]
SECTION_BANK=list("QWERTYUIOPASDFGHJKLZXCVBNM")
BATCH_BANK=["SEL","DOR","NIM","PAZ","RIL","VOK","JEN","KUR","FAS","GEM","HUR","LIX","MON","PEV","RAX","SIV","TOL","WEN","YAR","ZEK","BIM","COV","DUL","FEN","GIR","HAL","JOS","KEM","LUR","NAV","PIR","ROV","SAX","TEV","VIL","WOR","XEN","YUM","ZAL","BER","CIN","DAK","ELM","FOV","GUN","HES","JAR","KIV","LEN","MOR","NEX","PUL","RIM","SEN","TAV","VOR","WIL","XAR","YEN","ZUR","BOL","CER","DIN","FAL","GER","HIN","JOV","KEL","LUM","MAR","NOR","PEL","RUS","SIM","TOR","VEX","WAL","XIR","YAV","ZEN","BUN","CAL","DEM","EVI","FIR","GOS","HAR","IVO","JUN","KAS","LEV","MUR","OVA","PAR","RIN","SOL"]
BAY_BANK=[str(x) for x in range(101,197)]

NEED=N_ITEMS*3
assert len(ENTITY_BANK)>=N_ITEMS
assert len(SEAL_BANK)>=NEED,f"SEAL_BANK requires {NEED}, found {len(SEAL_BANK)}"
assert len(CLASS_BANK)>=NEED,f"CLASS_BANK requires {NEED}, found {len(CLASS_BANK)}"
assert len(BATCH_BANK)>=NEED,f"BATCH_BANK requires {NEED}, found {len(BATCH_BANK)}"
assert len(BAY_BANK)>=NEED,f"BAY_BANK requires {NEED}, found {len(BAY_BANK)}"
assert len(set(SEAL_BANK))==len(SEAL_BANK)
assert len(set(CLASS_BANK))==len(CLASS_BANK)
assert len(set(BATCH_BANK))==len(BATCH_BANK)
assert len(set(BAY_BANK))==len(BAY_BANK)

os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
torch.set_grad_enabled(False)
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

def ordered_rows(rows,gold_index,target_pos):
    assert len(rows)==3 and 0<=gold_index<3 and 0<=target_pos<3
    gold=rows[gold_index];rest=[x for i,x in enumerate(rows) if i!=gold_index]
    rr=random.Random(SEED+1000+target_pos*37+sum(ord(c) for c in str(gold)))
    rr.shuffle(rest);out=rest[:];out.insert(target_pos,gold)
    assert len(out)==3 and out[target_pos]==gold
    return out

def make_items():
    rng=random.Random(SEED);items=[]
    seals=SEAL_BANK.copy();classes=CLASS_BANK.copy();batches=BATCH_BANK.copy();bays=BAY_BANK.copy()
    rng.shuffle(seals);rng.shuffle(classes);rng.shuffle(batches);rng.shuffle(bays)
    for i in range(N_ITEMS):
        ents=list(ENTITY_BANK[i]);target_ix=i%3
        ss=seals[i*3:i*3+3];cc=classes[i*3:i*3+3];bb=batches[i*3:i*3+3];yy=bays[i*3:i*3+3]
        assert len(ents)==len(ss)==len(cc)==len(bb)==len(yy)==3
        sec=[SECTION_BANK[(i*3+j)%len(SECTION_BANK)] for j in range(3)]
        r1=[f"Instrument {ents[j]} carries seal {ss[j]}." for j in range(3)]
        r2=[f"Seal {ss[j]} corresponds to routing class {cc[j]}." for j in range(3)]
        r3=[f"Routing class {cc[j]} is assigned archive section {sec[j]}." for j in range(3)]
        r4=[f"Instrument {ents[j]} belongs to transfer batch {bb[j]}." for j in range(3)]
        r5=[f"Transfer batch {bb[j]} uses active bay {yy[j]}." for j in range(3)]
        positions=[(i+m)%3 for m in range(5)]
        records=[
            " ".join(ordered_rows(r1,target_ix,positions[0])),
            " ".join(ordered_rows(r2,target_ix,positions[1])),
            " ".join(ordered_rows(r3,target_ix,positions[2])),
            " ".join(ordered_rows(r4,target_ix,positions[3])),
            " ".join(ordered_rows(r5,target_ix,positions[4])),
            "The final paired identifier for an instrument consists of its archive section, a hyphen, and its active bay."
        ]
        target=f"{sec[target_ix]}-{yy[target_ix]}"
        q=f"Using the available records, determine the final paired identifier for instrument {ents[target_ix]}. Give only the exact identifier. If it cannot be determined, answer NONE."
        tq=f"Using the available records, trace instrument {ents[target_ix]} through the records. Return exactly one line in this format: SEAL > ROUTING_CLASS > ARCHIVE_SECTION | TRANSFER_BATCH > ACTIVE_BAY | FINAL_IDENTIFIER"
        item={"id":i+1,"entity":ents[target_ix],"target":target,"records":records,"q":q,"trace_q":tq,
              "gold":{"seal":ss[target_ix],"class":cc[target_ix],"section":sec[target_ix],"batch":bb[target_ix],"bay":yy[target_ix]}}
        assert target.casefold() not in " ".join(records).casefold()
        items.append(item)
    return items

ITEMS=make_items()
assert len(ITEMS)==N_ITEMS

LOCK={"test":TEST,"model":MODEL_ID,"seed":SEED,"n_items":N_ITEMS,"k_dim":K_DIM,"v_dim":V_DIM,
"engine":"TEST496 frozen MAM engine","conditions":["A_TEXT","B_CONTIG_RAW_KV","C_INDEPENDENT_RAW_KV","D_INDEPENDENT_K120_V128","E_D_EXPLICIT_TRACE"],
"items":ITEMS,"test482_blob_sha":TEST482_BLOB_SHA,"test482_lock_sha":TEST482_LOCK_SHA,"test495_lock_sha":TEST495_LOCK_SHA,
"address":"OFF","vtoken":"OFF","topk":"NONE","router":"NONE","ann":"NONE","graph":"NONE","training":"NONE","steering":"NONE","posthoc_selection":"NONE"}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

print("="*164)
print("TEST500 — AKBASCORE MAM · WRITE-TIME BRIDGING vs INDEPENDENT BELLEKÖZ")
print("CLEAN 32-ITEM FALSIFICATION PANEL · TEXT → CONTIG RAW → INDEPENDENT RAW → K120/V128 → EXPLICIT TRACE")
print("="*164)
print("LOCK SHA:",LOCK_SHA)
print("TEST482 BLOB:",TEST482_BLOB_SHA)
print("TEST495 LOCK:",TEST495_LOCK_SHA)
print("ITEMS:",N_ITEMS)
print("DRA / STEERING: NONE")
T0=time.perf_counter()

print("\n[1/7] Loading frozen Qwen...")
_tv=tuple(int("".join(c for c in x if c.isdigit())or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;layers=model.model.layers
H=cfg.hidden_size;NH=cfg.num_attention_heads;NKV=cfg.num_key_value_heads
HD=getattr(cfg,"head_dim",None) or H//NH;KVD=NKV*HD;NL=len(layers)
if (NL,H,NH,NKV,HD)!=(28,3584,28,4,128):raise RuntimeError(f"Architecture mismatch: {(NL,H,NH,NKV,HD)}")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
_ge=model.generation_config.eos_token_id
EOS=sorted({tok.eos_token_id,*(_ge if isinstance(_ge,(list,tuple)) else [_ge])}-{None})
enc=lambda s:tok(s,add_special_tokens=False).input_ids
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | {NL}L | H={H} | QH={NH} KVH={NKV} HD={HD} | frozen BF16")

@torch.inference_mode()
def kv_from_ids(ids):
    o=model(input_ids=torch.tensor([ids],device=DEVICE),output_hidden_states=True,use_cache=False,return_dict=True)
    K=[];V=[]
    for L in range(NL):
        z=layers[L].input_layernorm(o.hidden_states[L][0]);a=layers[L].self_attn
        K.append(a.k_proj(z).contiguous());V.append(a.v_proj(z).contiguous())
    return K,V

def forge(s):return kv_from_ids([PAD]+enc(s+SEP))

print("[2/7] Building frozen TEST482 neutral PCA codebook...")
CB=[];CO=[forge(s) for s in CORPUS]
for L in range(NL):
    e={}
    for j,n in enumerate(("K","V")):
        R=torch.cat([c[j][L][1:] for c in CO]).float().view(-1,NKV,HD);MU=[];BB=[]
        for h in range(NKV):
            X=R[:,h];mu=X.mean(0);_,S,Vh=torch.linalg.svd(X-mu,full_matrices=False)
            m=min(128,Vh.shape[0]);b=Vh[:m].T.contiguous()
            if m<128:b=F.pad(b,(0,128-m))
            MU.append(mu);BB.append(b)
        e[n]=(torch.stack(MU),torch.stack(BB))
    CB.append(e)
del CO;torch.cuda.empty_cache()
print("      Codebook ready.")

@torch.inference_mode()
def compress_raw(K,V):
    out={}
    for n,X,d in (("K",K,K_DIM),("V",V,V_DIM)):
        rows=[]
        for L in range(NL):
            mu,B=CB[L][n]
            coeff=torch.einsum("thi,hid->thd",X[L][1:].float().view(-1,NKV,HD)-mu,B[:,:,:d]).to(torch.bfloat16)
            content=(mu+torch.einsum("thd,hid->thi",coeff.float(),B[:,:,:d])).reshape(-1,KVD).to(torch.bfloat16)
            rows.append(torch.cat([X[L][:1],content]))
        out[n]=rows
    return out["K"],out["V"]

def packet(s):
    K,V=forge(s);return compress_raw(K,V)

@torch.inference_mode()
def install_single(K,V):
    T=K[0].shape[0];pos=torch.arange(T,device=DEVICE)[None]
    cos,sin=model.model.rotary_emb(K[0][None],pos);KK=[];VV=[]
    for L in range(NL):
        k=K[L].reshape(1,T,NKV,HD).transpose(1,2);v=V[L].reshape(1,T,NKV,HD).transpose(1,2)
        KK.append(apply_rotary_pos_emb(k,k,cos,sin,unsqueeze_dim=1)[1].contiguous());VV.append(v.contiguous())
    return tuple(KK),tuple(VV),T,T

@torch.inference_mode()
def install_parallel(packets,order=None):
    if order is None:order=list(range(len(packets)))
    KK=[[] for _ in range(NL)];VV=[[] for _ in range(NL)];Tmax=0;total=0
    for ix in order:
        K,V=packets[ix];T=K[0].shape[0];Tmax=max(Tmax,T);total+=T
        pos=torch.arange(T,device=DEVICE)[None];cos,sin=model.model.rotary_emb(K[0][None],pos)
        for L in range(NL):
            k=K[L].reshape(1,T,NKV,HD).transpose(1,2);v=V[L].reshape(1,T,NKV,HD).transpose(1,2)
            KK[L].append(apply_rotary_pos_emb(k,k,cos,sin,unsqueeze_dim=1)[1].contiguous());VV[L].append(v.contiguous())
    return tuple(torch.cat(x,dim=2) for x in KK),tuple(torch.cat(x,dim=2) for x in VV),total,Tmax

def make_cache(kv):
    try:cache=DynamicCache(config=cfg)
    except TypeError:cache=DynamicCache()
    for L in range(NL):cache.update(kv[0][L].clone(),kv[1][L].clone(),L)
    return cache

@torch.inference_mode()
def read_kv(q,kv,max_new=MAX_NEW):
    qids=enc(FMT.format(q=q));Tm=kv[2];P=kv[3];nq=len(qids);cache=make_cache(kv)
    mask=torch.ones(1,Tm+nq,dtype=torch.long,device=DEVICE)
    pos=torch.arange(P,P+nq,device=DEVICE)[None];ids=torch.tensor([qids],device=DEVICE);out=[]
    for _ in range(max_new):
        o=model(input_ids=ids,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=True,return_dict=True)
        nxt=int(o.logits[0,-1].float().argmax())
        if nxt in EOS:break
        out.append(nxt);ids=torch.tensor([[nxt]],device=DEVICE);pos=pos[:,-1:]+1
        mask=torch.cat([mask,torch.ones(1,1,dtype=torch.long,device=DEVICE)],1)
    return tok.decode(out,skip_special_tokens=True).strip()

@torch.inference_mode()
def read_text(records,q,max_new=MAX_NEW):
    context="\n".join(f"RECORD {i+1}: {s}" for i,s in enumerate(records))
    prompt=f"{context}\n\n{FMT.format(q=q)}"
    ids=torch.tensor([enc(prompt)],device=DEVICE);out=[]
    for _ in range(max_new):
        o=model(input_ids=ids,use_cache=False,return_dict=True)
        nxt=int(o.logits[0,-1].float().argmax())
        if nxt in EOS:break
        out.append(nxt);ids=torch.cat([ids,torch.tensor([[nxt]],device=DEVICE)],1)
    return tok.decode(out,skip_special_tokens=True).strip()

def firstline(x):return next((z.strip() for z in str(x).splitlines() if z.strip()),"")
def canon(x):return firstline(x).strip().upper()
def exact(x,target):return canon(x)==target.upper()

def trace_final(x):
    s=firstline(x).upper()
    if "|" not in s:return ""
    z=s.split("|")[-1].strip()
    m=re.search(r"\b([A-Z]-\d{2,3})\b",z)
    return m.group(1) if m else ""

def trace_hops_ok(x,g):
    s=firstline(x).upper()
    vals=[g["seal"],g["class"],g["section"],g["batch"],g["bay"]]
    return all(re.search(rf"(?<![A-Z0-9]){re.escape(str(v).upper())}(?![A-Z0-9])",s) is not None for v in vals)

print("[3/7] Panel seal...")
for it in ITEMS[:4]:
    print(f"      ITEM {it['id']:02d} target={it['target']} entity={it['entity']} gold={it['gold']}")
print("      ...")
print("      Target-row positions rotate across records; no fixed first-row gold shortcut.")

RESULTS=[]
COUNT={"A":0,"B":0,"C":0,"D":0,"E":0};E_HOPS=0
print("\n[4/7] Running 32 fresh items across A/B/C/D/E...")
for z,it in enumerate(ITEMS):
    records=it["records"];q=it["q"];target=it["target"]

    # A — ordinary contiguous text in prompt.
    A=read_text(records,q)

    # Forge each independent record once.
    raw=[forge(s) for s in records]

    # B — same six records written contiguously in one prefill/cache.
    contiguous=SEP.join(records)
    BK,BV=forge(contiguous);BKV=install_single(BK,BV)
    B=read_kv(q,BKV)
    del BK,BV,BKV

    # C — independent uncompressed raw K/V packets in parallel workspace.
    CKV=install_parallel(raw)
    C=read_kv(q,CKV)
    del CKV

    # D — exact TEST482 K120/V128 compression on the same independent packets.
    compressed=[compress_raw(K,V) for K,V in raw]
    DKV=install_parallel(compressed)
    D=read_kv(q,DKV)

    # E — identical D memory cache; only reader protocol exposes intermediate hops.
    E=read_kv(it["trace_q"],DKV,max_new=TRACE_NEW)
    Ef=trace_final(E);Eh=trace_hops_ok(E,it["gold"])

    okA=exact(A,target);okB=exact(B,target);okC=exact(C,target);okD=exact(D,target);okE=(Ef==target.upper())
    COUNT["A"]+=int(okA);COUNT["B"]+=int(okB);COUNT["C"]+=int(okC);COUNT["D"]+=int(okD);COUNT["E"]+=int(okE);E_HOPS+=int(Eh)
    RESULTS.append({"id":it["id"],"target":target,"A":A,"B":B,"C":C,"D":D,"E":E,"Ef":Ef,
                    "okA":okA,"okB":okB,"okC":okC,"okD":okD,"okE":okE,"E_hops":Eh})
    print(f"      [{z+1:02d}/{N_ITEMS}] {it['entity']:6s} target={target:5s} | A={int(okA)} B={int(okB)} C={int(okC)} D={int(okD)} E={int(okE)}")
    print(f"               A={firstline(A)!r}")
    print(f"               B={firstline(B)!r}")
    print(f"               C={firstline(C)!r}")
    print(f"               D={firstline(D)!r}")
    print(f"               E={firstline(E)!r}")
    del DKV,compressed,raw
    if (z+1)%4==0:torch.cuda.empty_cache()

print("\n[5/7] Pairwise transition anatomy...")
AB=sum(r["okA"] and r["okB"] for r in RESULTS)
AC=sum(r["okA"] and r["okC"] for r in RESULTS)
BC=sum(r["okB"] and r["okC"] for r in RESULTS)
B_NOT_C=sum(r["okB"] and not r["okC"] for r in RESULTS)
C_NOT_D=sum(r["okC"] and not r["okD"] for r in RESULTS)
D_NOT_E=sum(r["okD"] and not r["okE"] for r in RESULTS)
E_RESCUE=sum((not r["okD"]) and r["okE"] for r in RESULTS)
A_NOT_B=sum(r["okA"] and not r["okB"] for r in RESULTS)
print(f"      A∩B correct                 : {AB}/{N_ITEMS}")
print(f"      A∩C correct                 : {AC}/{N_ITEMS}")
print(f"      B∩C correct                 : {BC}/{N_ITEMS}")
print(f"      A correct → B fail          : {A_NOT_B}/{N_ITEMS}")
print(f"      B correct → C fail          : {B_NOT_C}/{N_ITEMS}")
print(f"      C correct → D fail          : {C_NOT_D}/{N_ITEMS}")
print(f"      D correct → E fail          : {D_NOT_E}/{N_ITEMS}")
print(f"      D fail → E rescue           : {E_RESCUE}/{N_ITEMS}")
print(f"      E complete-hop trace        : {E_HOPS}/{N_ITEMS}")

print("\n[6/7] Bootstrap paired accuracy differences...")
def bootdiff(a,b,nboot=10000):
    x=np.array([int(r[a]) for r in RESULTS],dtype=float);y=np.array([int(r[b]) for r in RESULTS],dtype=float)
    rng=np.random.default_rng(SEED+77);d=np.empty(nboot)
    for i in range(nboot):
        ix=rng.integers(0,len(x),len(x));d[i]=(x[ix]-y[ix]).mean()
    return float((x-y).mean()),float(np.quantile(d,.025)),float(np.quantile(d,.975))

for name,a,b in [("A-B","okA","okB"),("B-C","okB","okC"),("C-D","okC","okD"),("E-D","okE","okD")]:
    m,lo,hi=bootdiff(a,b)
    print(f"      {name:4s}: Δ={m:+.4f} | bootstrap95=[{lo:+.4f},{hi:+.4f}]")

print("\n[7/7] Mechanistic classification...")
Aacc=COUNT["A"]/N_ITEMS;Bacc=COUNT["B"]/N_ITEMS;Cacc=COUNT["C"]/N_ITEMS;Dacc=COUNT["D"]/N_ITEMS;Eacc=COUNT["E"]/N_ITEMS
if Aacc<0.75:
    VERDICT="BASE_TASK_NOT_RELIABLY_SOLVED"
elif Bacc<0.75:
    VERDICT="CONTIGUOUS_CACHE_TRANSITION_DEFICIT"
elif Cacc+0.20<Bacc:
    VERDICT="INDEPENDENT_WRITE_BRIDGING_DEFICIT_OBSERVED"
elif Dacc+0.20<Cacc:
    VERDICT="K120_COMPRESSION_DEFICIT_OBSERVED"
elif Eacc>=Dacc+0.20:
    VERDICT="EXPLICIT_HOP_DECODING_RESCUE_OBSERVED"
else:
    VERDICT="NO_LARGE_STAGE_SPECIFIC_DEFICIT_OBSERVED"

print("\n"+"="*164)
print("TEST500 FINAL RESULT — CLEAN WRITE-TIME BRIDGING FALSIFICATION")
print("="*164)
print("VALIDATION TYPE             : FRESH 32-ITEM MECHANISTIC DIAGNOSTIC")
print("MODEL                       : Qwen/Qwen2.5-7B-Instruct · frozen")
print("TEST496 MAM ENGINE          : PRESERVED")
print("TEST482 CODEBOOK            : PRESERVED")
print("K/V COMPRESSION             : K120 / V128")
print("TARGET LITERAL IN MEMORY    : NO")
print("GOLD-FIRST SHORTCUT         : REMOVED / ROTATING POSITIONS")
print("DRA / STEERING              : NONE")
print("VTOKEN / ADDRESS            : OFF")
print("TOP-K / ROUTER / ANN        : NONE")
print("GRAPH                        : NONE")
print("TRAINING / LoRA             : NONE")
print("POST-HOC ITEM SELECTION     : NONE")
print("-"*164)
print(f"A · CONTIGUOUS TEXT          : {COUNT['A']:2d}/{N_ITEMS} = {Aacc:.4f}")
print(f"B · CONTIGUOUS RAW K/V       : {COUNT['B']:2d}/{N_ITEMS} = {Bacc:.4f}")
print(f"C · INDEPENDENT RAW K/V      : {COUNT['C']:2d}/{N_ITEMS} = {Cacc:.4f}")
print(f"D · INDEPENDENT K120/V128    : {COUNT['D']:2d}/{N_ITEMS} = {Dacc:.4f}")
print(f"E · D + EXPLICIT HOP TRACE   : {COUNT['E']:2d}/{N_ITEMS} = {Eacc:.4f}")
print(f"E · COMPLETE HOP CONTENT     : {E_HOPS:2d}/{N_ITEMS}")
print("-"*164)
print(f"B PASS → C FAIL              : {B_NOT_C}/{N_ITEMS}")
print(f"C PASS → D FAIL              : {C_NOT_D}/{N_ITEMS}")
print(f"D FAIL → E RESCUE            : {E_RESCUE}/{N_ITEMS}")
print("-"*164)
print("TEST482 BLOB SHA            :",TEST482_BLOB_SHA)
print("TEST482 LOCK SHA            :",TEST482_LOCK_SHA)
print("TEST495 LOCK SHA            :",TEST495_LOCK_SHA)
print("TEST500 LOCK SHA            :",LOCK_SHA)
print(f"TOTAL TEST TIME              : {time.perf_counter()-T0:.2f}s")
print("EXPLORATORY VERDICT          :",VERDICT)
print("="*164)

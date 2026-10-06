# Copyright © 2026 Mustafa Akbaş
# New original AKBASCORE MAM material is source-available under the repository LICENSE.
# See repository LICENSE for complete terms.
#
# AKBASCORE MAM — TEST510
# ÇAĞRIİZ LATENT TRACE ATLAS
# WHERE DOES THE NEXT KEY EXIST?
#
# PURPOSE
# -------
# TEST509 showed that:
#
#   A pre-answer final-token Q trace
#       → raw B K-address header
#
# does NOT provide reliable direct A→B addressing.
#
# TEST510 isolates the unresolved upstream question:
#
#   While MEMORY A is being read, where — if anywhere — does the next
#   associative key (Seal) exist in the frozen model's natural latent state?
#
# QUERY(Entity)
#      ↓
# ORACLE MEMORY A
#      ↓
# one frozen-model forward over A + query
#      ↓
# NO answer token generated
#      ↓
# layerwise latent atlas
#      ↓
# fixed precomputed Seal reference bank
#
# REPRESENTATION FAMILIES
# -----------------------
# Per layer, from the final pre-answer question token:
#
#   HRAW  = raw residual hidden state
#   HNORM = layer input-normalized hidden state
#   QMEAN = q_proj, collapsed from 28 Q heads → 4 GQA groups → flattened
#   KPROJ = k_proj → flattened
#   VPROJ = v_proj → flattened
#
# SEAL REFERENCE BANK
# -------------------
# Each Seal is independently encoded ONCE before query evaluation using the
# frozen model. Reference representations use the same representation family
# and layer as the query trace.
#
# Reference prompt:
#   "Seal <TOKEN>"
#
# No gold Seal is inserted into the query.
# No textual Seal is decoded from MEMORY A.
# No B cartridge is queried.
# No YES/NO scan.
# No per-candidate model forward at retrieval time.
#
# DISCOVERY / EVALUATION
# ----------------------
# Items 01–16 : DISCOVERY
# Items 17–32 : FROZEN EVALUATION
#
# Candidate space is declared before results:
#   28 layers × {HRAW,HNORM,QMEAN,KPROJ,VPROJ}
#
# Discovery chooses exactly one (family, layer).
# That choice is frozen before evaluation.
#
# IMPORTANT
# ---------
# This is an X-RAY / representation-localization experiment.
# It does NOT yet claim direct BELLEKÖZ routing.
#
# No compression.
# No DRA / steering.
# No VTOKEN.
# No ANN.
# No learned router.
# No training / LoRA.
# No post-hoc evaluation-channel selection.

import os,sys,subprocess,importlib.util,random,time,hashlib,json,re
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache

TEST="510";SEED=510;BASE_SEED=501;MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
N_ITEMS=32;N_DISC=16;N_EVAL=16
DEVICE=torch.device("cuda");FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
TEST508_LOCK_SHA="47512c7860268517670ca5b45b6e76fe1582362dd38e447d164f33c756a1a2ca"
TEST509_LOCK_SHA="4b5b78841ebf0d9b458306136e122fd3108b33bdfd89924585f891f748eca0d8"

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
CLASS_BANK=["TAK","BEX","QAA","RAV","SOD","PEK","NIV","GOR","HAX","JUR","KEM","VOL","DAX","QAB","MON","SAL","TEK","WIR","ZUN","COV","HEM","JAX","LIV","QAC","PAK","RUM","SEV","TIX","VOR","YAM","ZEK","BOL","QAD","DOV","FEX","GAM","HUR","JIN","KAV","LER","MEX","QAE","PIV","ROK","SUM","TAL","VEK","WON","XIR","YAV","ZOL","BAR","CIX","QAF","FOV","GEL","HIN","JOV","KUR","QAG","MAV","QAH","PUL","RIM","QAI","TOX","VIL","WER","XAN","YER","ZIM","BUN","CAL","DOR","EVI","FAR","GUN","HES","ILM","JER","KON","LAR","QAJ","NOL","OVI","PER","RUS","SIN","QAK","VEX","QAL","QAM","YUL","ZAR","BEL","CUM"]

assert N_DISC+N_EVAL==N_ITEMS
assert len(ENTITY_BANK)==N_ITEMS
assert len(SEAL_BANK)>=N_ITEMS*3 and len(CLASS_BANK)>=N_ITEMS*3
assert len(set(SEAL_BANK))==len(SEAL_BANK)
assert len(set(CLASS_BANK))==len(CLASS_BANK)
assert set(SEAL_BANK).isdisjoint(CLASS_BANK)
assert all(re.fullmatch(r"[A-Z]{3}",x) for x in SEAL_BANK)
assert all(re.fullmatch(r"[A-Z]{3}",x) for x in CLASS_BANK)

os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
torch.set_grad_enabled(False)
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

def reorder(rows,gold,target_pos,seed):
    if not 0<=gold<len(rows):raise RuntimeError("Invalid gold row.")
    if not 0<=target_pos<len(rows):raise RuntimeError("Invalid target position.")
    g=rows[gold];rest=[x for j,x in enumerate(rows) if j!=gold]
    rr=random.Random(seed);rr.shuffle(rest);out=rest[:];out.insert(target_pos,g);return out

# Same TEST509 base construction.
def make_base():
    rng=random.Random(BASE_SEED);seals=SEAL_BANK.copy();classes=CLASS_BANK.copy()
    rng.shuffle(seals);rng.shuffle(classes);items=[]
    for i in range(N_ITEMS):
        ents=list(ENTITY_BANK[i]);g=i%3;ss=seals[3*i:3*i+3];cc=classes[3*i:3*i+3]
        if len(ss)!=3 or len(cc)!=3:raise RuntimeError(f"Pool exhaustion item {i+1}.")
        r1=[f"Instrument {ents[j]} carries seal {ss[j]}." for j in range(3)]
        r2=[f"Seal {ss[j]} corresponds to routing class {cc[j]}." for j in range(3)]
        p1=i%3;p2=(i+1)%3
        A=" ".join(reorder(r1,g,p1,BASE_SEED+i*101+17))
        B=" ".join(reorder(r2,g,p2,BASE_SEED+i*101+34))
        items.append({"id":i+1,"A":A,"B":B,"gold":{"entity":ents[g],"seal":ss[g],"class":cc[g]}})
    return items

ITEMS=make_base()
USED_SEALS=sorted({it["gold"]["seal"] for it in ITEMS})
if len(USED_SEALS)!=N_ITEMS:raise RuntimeError(f"Gold Seal uniqueness failure: {len(USED_SEALS)}/{N_ITEMS}")
for it in ITEMS:
    g=it["gold"]
    if not re.search(rf"(?<![A-Za-z0-9_]){re.escape(g['entity'])}(?![A-Za-z0-9_])",it["A"]):raise RuntimeError(f"Entity missing from A item {it['id']}.")
    if not re.search(rf"(?<![A-Za-z0-9_]){re.escape(g['seal'])}(?![A-Za-z0-9_])",it["A"]):raise RuntimeError(f"Seal missing from A item {it['id']}.")

LOCK={
"test":TEST,"parent_test509":TEST509_LOCK_SHA,"parent_test508":TEST508_LOCK_SHA,
"model":MODEL_ID,"seed":SEED,"base_seed":BASE_SEED,"items":ITEMS,
"n_items":N_ITEMS,"discovery_items":"01-16","evaluation_items":"17-32",
"task":"A_preanswer_latent_state_to_gold_Seal_reference",
"seal_reference_bank":USED_SEALS,
"seal_reference_prompt":"Seal <TOKEN>",
"query_time_candidate_model_forwards":0,
"intermediate_text_decode":"NONE",
"b_cartridge_search":"NONE",
"candidate_families":["HRAW","HNORM","QMEAN","KPROJ","VPROJ"],
"candidate_layers":list(range(28)),
"selection":"DISCOVERY_ONLY_THEN_FROZEN_EVAL",
"compression":"OFF","steering":"OFF","vtoken":"OFF","ann":"OFF",
"learned_router":"OFF","training":"OFF","posthoc_eval_selection":"OFF"
}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

print("="*174)
print("TEST510 — AKBASCORE MAM · ÇAĞRIİZ LATENT TRACE ATLAS")
print("WHERE DOES THE NEXT KEY EXIST? · ORACLE A → PRE-ANSWER LATENT STATE → FIXED SEAL REFERENCE BANK")
print("="*174)
print("LOCK SHA:",LOCK_SHA)
print("PARENT TEST509 LOCK:",TEST509_LOCK_SHA)
print("PARENT TEST508 LOCK:",TEST508_LOCK_SHA)
print("ITEMS:",N_ITEMS,"| DISCOVERY:",N_DISC,"| FROZEN EVAL:",N_EVAL,"| SEAL REFERENCE BANK:",len(USED_SEALS))
print("TEXTUAL INTERMEDIATE SEAL DECODE: NONE")
print("B-CARTRIDGE SEARCH: NONE")
print("QUERY-TIME MODEL FORWARDS PER SEAL CANDIDATE: 0")
print("CANDIDATES: HRAW / HNORM / QMEAN / KPROJ / VPROJ × layers 0..27")
T0=time.perf_counter()

print("\n[1/7] Loading frozen Qwen...")
_tv=tuple(int("".join(c for c in x if c.isdigit())or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
enc=lambda s:tok(s,add_special_tokens=False).input_ids
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=H//QH
if (NL,H,QH,KVH,HD)!=(28,3584,28,4,128):raise RuntimeError(f"Architecture mismatch: {(NL,H,QH,KVH,HD)}")
if QH%KVH!=0:raise RuntimeError("GQA grouping mismatch.")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
if PAD is None:raise RuntimeError("No PAD/EOS token available.")
GROUP=QH//KVH
MODEL_DTYPE=model.model.layers[0].self_attn.q_proj.weight.dtype
if MODEL_DTYPE not in (torch.bfloat16,torch.float16,torch.float32):raise RuntimeError(f"Unexpected model dtype: {MODEL_DTYPE}")
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | {NL}L | H={H} | QH={QH} KVH={KVH} HD={HD} | {MODEL_DTYPE}")
print(f"      GQA group: {GROUP} query heads per KV head")

@torch.inference_mode()
def kv_from_ids(ids):
    if not ids:raise RuntimeError("kv_from_ids received empty ids.")
    x=torch.tensor([ids],device=DEVICE,dtype=torch.long)
    o=model(input_ids=x,use_cache=False,output_hidden_states=True,return_dict=True)
    if len(o.hidden_states)!=NL+1:raise RuntimeError(f"Unexpected hidden-state count: {len(o.hidden_states)}")
    out=[]
    for L,layer in enumerate(model.model.layers):
        h=o.hidden_states[L][0]
        h=h.to(device=layer.input_layernorm.weight.device,dtype=layer.input_layernorm.weight.dtype)
        hn=layer.input_layernorm(h)
        qdtype=layer.self_attn.k_proj.weight.dtype
        hn=hn.to(device=layer.self_attn.k_proj.weight.device,dtype=qdtype)
        k=layer.self_attn.k_proj(hn).view(-1,KVH,HD).transpose(0,1).contiguous()
        v=layer.self_attn.v_proj(hn).view(-1,KVH,HD).transpose(0,1).contiguous()
        out.append((k,v))
    return out

@torch.inference_mode()
def forge(s):
    ids=[PAD]+enc(s+SEP)
    if len(ids)<2:raise RuntimeError("Forge serialization unexpectedly empty.")
    return kv_from_ids(ids)

def rope_k(k,pos,L):
    if k.ndim!=3 or k.shape[0]!=KVH or k.shape[2]!=HD:raise RuntimeError(f"Invalid K shape L{L}: {tuple(k.shape)}")
    if len(pos)!=k.shape[1]:raise RuntimeError(f"RoPE position mismatch L{L}: {len(pos)} vs {k.shape[1]}")
    layer=model.model.layers[L]
    rot=layer.self_attn.rotary_emb if hasattr(layer.self_attn,"rotary_emb") else model.model.rotary_emb
    dummy=torch.zeros((1,k.shape[0],len(pos),HD),device=k.device,dtype=k.dtype)
    p=torch.tensor([pos],device=k.device,dtype=torch.long)
    try:cos,sin=rot(dummy,p)
    except TypeError:cos,sin=rot(dummy,position_ids=p)
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    _,kr=apply_rotary_pos_emb(dummy,k.unsqueeze(0),cos,sin)
    return kr[0]

@torch.inference_mode()
def install_single(raw):
    if len(raw)!=NL:raise RuntimeError(f"RAW layer count mismatch: {len(raw)}/{NL}")
    T=raw[0][0].shape[1]
    if T<1:raise RuntimeError("Empty memory packet.")
    pos=list(range(T));out=[]
    for L in range(NL):
        k,v=raw[L]
        if k.shape[1]!=T or v.shape[1]!=T:raise RuntimeError(f"Packet length mismatch L{L}.")
        out.append((rope_k(k,pos,L),v))
    return out,T,T

def make_cache(installed):
    if len(installed)!=NL:raise RuntimeError(f"Installed layer count mismatch: {len(installed)}/{NL}")
    try:cache=DynamicCache(config=cfg)
    except TypeError:cache=DynamicCache()
    for L,(k,v) in enumerate(installed):
        cache.update(k.unsqueeze(0),v.unsqueeze(0),L)
    return cache

def q_read_a(e):return f"What seal does instrument {e} carry? Give only the exact seal."

def unit(x):
    x=x.float().reshape(-1)
    n=x.norm()
    if not torch.isfinite(n) or n.item()<=0:raise RuntimeError("Invalid representation norm.")
    return x/n.clamp_min(1e-8)

# -------------------------------------------------------------------------
# REPRESENTATION EXTRACTION
# -------------------------------------------------------------------------
# CRITICAL DTYPE RULE:
#   Hidden → LayerNorm → q/k/v projection remains in the model projection
#   dtype (BF16 here). Only AFTER projection do we cast to FP32 for analysis.
# -------------------------------------------------------------------------
def reps_from_hidden(h,L):
    layer=model.model.layers[L]
    if h.ndim!=1 or h.numel()!=H:raise RuntimeError(f"Invalid hidden shape L{L}: {tuple(h.shape)}")

    raw=h.float()

    ln_dtype=layer.input_layernorm.weight.dtype
    ln_device=layer.input_layernorm.weight.device
    h_model=h.to(device=ln_device,dtype=ln_dtype)
    hn_model=layer.input_layernorm(h_model)
    hn=hn_model.float()

    qproj=layer.self_attn.q_proj
    kproj=layer.self_attn.k_proj
    vproj=layer.self_attn.v_proj

    qin=hn_model.to(device=qproj.weight.device,dtype=qproj.weight.dtype)
    kin=hn_model.to(device=kproj.weight.device,dtype=kproj.weight.dtype)
    vin=hn_model.to(device=vproj.weight.device,dtype=vproj.weight.dtype)

    q=qproj(qin).view(QH,HD).float()
    q=q.view(KVH,GROUP,HD).mean(dim=1)
    k=kproj(kin).view(KVH,HD).float()
    v=vproj(vin).view(KVH,HD).float()

    if q.shape!=(KVH,HD) or k.shape!=(KVH,HD) or v.shape!=(KVH,HD):
        raise RuntimeError(f"Projection shape mismatch L{L}: Q={tuple(q.shape)} K={tuple(k.shape)} V={tuple(v.shape)}")

    return {
        "HRAW":unit(raw),
        "HNORM":unit(hn),
        "QMEAN":unit(q),
        "KPROJ":unit(k),
        "VPROJ":unit(v)
    }

@torch.inference_mode()
def latent_atlas(installed,Tm,P,q):
    if Tm<=0 or P<=0:raise RuntimeError(f"Invalid cache geometry Tm={Tm}, P={P}")
    cache=make_cache(installed);qids=enc(FMT.format(q=q))
    if not qids:raise RuntimeError("Query tokenization returned no tokens.")
    nq=len(qids)
    pos=torch.arange(P,P+nq,device=DEVICE,dtype=torch.long).unsqueeze(0)
    mask=torch.ones((1,Tm+nq),device=DEVICE,dtype=torch.long)
    o=model(input_ids=torch.tensor([qids],device=DEVICE,dtype=torch.long),
            past_key_values=cache,attention_mask=mask,position_ids=pos,
            use_cache=False,output_hidden_states=True,return_dict=True)
    if len(o.hidden_states)!=NL+1:raise RuntimeError(f"Unexpected latent hidden-state count: {len(o.hidden_states)}")
    atlas=[]
    for L in range(NL):
        h=o.hidden_states[L][0,-1]
        atlas.append({k:v.detach().cpu() for k,v in reps_from_hidden(h,L).items()})
    return atlas

# -------------------------------------------------------------------------
# FIXED SEAL REFERENCE BANK
# -------------------------------------------------------------------------
# Each candidate Seal is encoded ONCE, independently, before query evaluation.
# The final serialized token state represents the complete "Seal <TOKEN>"
# prefix context even if <TOKEN> itself consists of more than one tokenizer
# subtoken.
# -------------------------------------------------------------------------
@torch.inference_mode()
def seal_reference(seal):
    ids=[PAD]+enc(f"Seal {seal}")
    if len(ids)<2:raise RuntimeError(f"Seal reference tokenization failure: {seal}")
    x=torch.tensor([ids],device=DEVICE,dtype=torch.long)
    o=model(input_ids=x,use_cache=False,output_hidden_states=True,return_dict=True)
    if len(o.hidden_states)!=NL+1:raise RuntimeError(f"Unexpected Seal hidden-state count: {len(o.hidden_states)}")
    refs=[]
    for L in range(NL):
        h=o.hidden_states[L][0,-1]
        refs.append({k:v.detach().cpu() for k,v in reps_from_hidden(h,L).items()})
    return refs

def cosine(a,b):
    if a.ndim!=1 or b.ndim!=1 or a.shape!=b.shape:raise RuntimeError(f"Cosine shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    v=torch.dot(a.float(),b.float())
    if not torch.isfinite(v):raise RuntimeError("Non-finite cosine score.")
    return float(v.item())

def score_reference_bank(qref,refs,L,family):
    scores=[cosine(qref,refs[s][L][family]) for s in USED_SEALS]
    if len(scores)!=len(USED_SEALS) or not np.all(np.isfinite(scores)):raise RuntimeError("Invalid reference scores.")
    return scores

def ranks_from_scores(scores,gold_idx):
    if not 0<=gold_idx<len(scores):raise RuntimeError("Gold index out of range.")
    if not np.all(np.isfinite(scores)):raise RuntimeError("Non-finite ranking score.")
    order=sorted(range(len(scores)),key=lambda j:(-scores[j],j))
    return order.index(gold_idx)+1,order

def rank_stats(x):
    a=np.asarray(x,dtype=float)
    if not len(a):raise RuntimeError("rank_stats received empty input.")
    if not np.all(np.isfinite(a)) or np.any(a<1):raise RuntimeError("Invalid ranks.")
    return {"R1":float(np.mean(a==1)),"R5":float(np.mean(a<=5)),
            "R16":float(np.mean(a<=16)),"MRR":float(np.mean(1/a)),
            "MED":float(np.median(a))}

print("\n[2/7] Pre-forging oracle A memories...")
ADATA=[]
for z,it in enumerate(ITEMS):
    raw=forge(it["A"]);inst=install_single(raw);ADATA.append(inst)
    if z<6:
        g=it["gold"]
        print(f"      ITEM {it['id']:02d} {g['entity']:6s} | A contains {g['entity']}→{g['seal']}")
print("      Same TEST509 RAW A-memory construction retained.")

print("\n[3/7] Building fixed Seal reference bank and extracting A latent atlas...")
SEAL_REFS={}
for i,s in enumerate(USED_SEALS):
    SEAL_REFS[s]=seal_reference(s)
    if i<6:print(f"      REF {i+1:02d} Seal {s}")
if set(SEAL_REFS)!=set(USED_SEALS):raise RuntimeError("Seal reference bank mismatch.")
print(f"      Fixed reference bank complete: {len(SEAL_REFS)} independently encoded Seals.")

ATLASES=[]
for z,it in enumerate(ITEMS):
    atlas=latent_atlas(*ADATA[z],q_read_a(it["gold"]["entity"]))
    if len(atlas)!=NL:raise RuntimeError(f"Atlas layer count failure item {z+1}.")
    ATLASES.append(atlas);ADATA[z]=None
    if z<6:print(f"      ITEM {it['id']:02d} {it['gold']['entity']:6s} | 28-layer × 5-family atlas captured | Seal decoded: NO")
print("      All A traces captured before generation of any answer token.")

FAMILIES=("HRAW","HNORM","QMEAN","KPROJ","VPROJ")
SEAL_INDEX={s:i for i,s in enumerate(USED_SEALS)}
if len(SEAL_INDEX)!=N_ITEMS:raise RuntimeError("Seal index collision.")

# -------------------------------------------------------------------------
# DISCOVERY
# -------------------------------------------------------------------------
print("\n[4/7] DISCOVERY — items 01–16 only...")
DISC={};DISC_RANKS={}
for fam in FAMILIES:
    for L in range(NL):
        ranks=[]
        for z in range(N_DISC):
            gold=ITEMS[z]["gold"]["seal"];gi=SEAL_INDEX[gold]
            scores=score_reference_bank(ATLASES[z][L][fam],SEAL_REFS,L,fam)
            r,_=ranks_from_scores(scores,gi);ranks.append(r)
        st=rank_stats(ranks);DISC[(fam,L)]=st;DISC_RANKS[(fam,L)]=ranks

# Selection rule fixed before frozen evaluation:
# 1) highest R1
# 2) highest MRR
# 3) highest R5
# 4) projected family preference only on exact metric tie
# 5) lower layer
PREF={"KPROJ":5,"VPROJ":4,"QMEAN":3,"HNORM":2,"HRAW":1}
def selkey(x):
    (fam,L),s=x
    return (s["R1"],s["MRR"],s["R5"],PREF[fam],-L)

ordered=sorted(DISC.items(),key=selkey,reverse=True)
if len(ordered)!=len(FAMILIES)*NL:raise RuntimeError("Discovery candidate count mismatch.")
for n,((fam,L),s) in enumerate(ordered[:15],1):
    print(f"      #{n:02d} {fam:6s} L{L:02d} | R1={s['R1']:.4f} R5={s['R5']:.4f} R16={s['R16']:.4f} MRR={s['MRR']:.6f} median={s['MED']:.1f}")

(FROZEN_FAMILY,FROZEN_LAYER),BEST_DISC=ordered[0]
print(f"\n      FROZEN CHANNEL → {FROZEN_FAMILY} / L{FROZEN_LAYER:02d}")
print("      Evaluation items have NOT participated in channel selection.")

# -------------------------------------------------------------------------
# FROZEN EVALUATION
# -------------------------------------------------------------------------
print("\n[5/7] FROZEN EVALUATION — items 17–32...")
EVAL_RANKS=[];EVAL_SCORE_BANKS=[];GOLD_SCORES=[];TOP_WRONG_SCORES=[]
for z in range(N_DISC,N_ITEMS):
    gold=ITEMS[z]["gold"]["seal"];gi=SEAL_INDEX[gold]
    qref=ATLASES[z][FROZEN_LAYER][FROZEN_FAMILY]
    scores=score_reference_bank(qref,SEAL_REFS,FROZEN_LAYER,FROZEN_FAMILY)
    EVAL_SCORE_BANKS.append(scores)
    r,order=ranks_from_scores(scores,gi);pred=USED_SEALS[order[0]]
    EVAL_RANKS.append(r);GOLD_SCORES.append(scores[gi])
    wrong=max(scores[j] for j in range(len(scores)) if j!=gi);TOP_WRONG_SCORES.append(wrong)
    g=ITEMS[z]["gold"]
    print(f"      [{z+1:02d}/{N_ITEMS}] {g['entity']:6s} | gold={gold} pred={pred} rank={r:2d} | gold={scores[gi]:+.5f} topWrong={wrong:+.5f} margin={scores[gi]-wrong:+.5f}")

if len(EVAL_RANKS)!=N_EVAL:raise RuntimeError(f"Evaluation count mismatch: {len(EVAL_RANKS)}/{N_EVAL}")
EVAL=rank_stats(EVAL_RANKS)
DISC_FROZEN=DISC[(FROZEN_FAMILY,FROZEN_LAYER)]
margins=np.asarray(GOLD_SCORES,dtype=float)-np.asarray(TOP_WRONG_SCORES,dtype=float)

# Random-label null over frozen evaluation score banks.
# Scores/channel remain frozen. Only candidate identity labels are randomized.
rng=np.random.default_rng(SEED)
NULL_R1=[];NULL_MRR=[]
for _ in range(10000):
    rr=[]
    for scores in EVAL_SCORE_BANKS:
        fake=int(rng.integers(0,len(USED_SEALS)))
        r,_=ranks_from_scores(scores,fake);rr.append(r)
    rr=np.asarray(rr,dtype=float)
    NULL_R1.append(float(np.mean(rr==1)))
    NULL_MRR.append(float(np.mean(1/rr)))

print("\n[6/7] Frozen-channel statistics...")
print(f"      CHANNEL             : {FROZEN_FAMILY} / L{FROZEN_LAYER:02d}")
print(f"      DISCOVERY R1        : {DISC_FROZEN['R1']:.4f}")
print(f"      DISCOVERY R5        : {DISC_FROZEN['R5']:.4f}")
print(f"      DISCOVERY R16       : {DISC_FROZEN['R16']:.4f}")
print(f"      DISCOVERY MRR       : {DISC_FROZEN['MRR']:.6f}")
print(f"      EVAL R1             : {EVAL['R1']:.4f}")
print(f"      EVAL R5             : {EVAL['R5']:.4f}")
print(f"      EVAL R16            : {EVAL['R16']:.4f}")
print(f"      EVAL MRR            : {EVAL['MRR']:.6f}")
print(f"      EVAL MEDIAN RANK    : {EVAL['MED']:.1f}")
print(f"      MEAN GOLD MARGIN    : {margins.mean():+.6f}")
print(f"      MEDIAN GOLD MARGIN  : {np.median(margins):+.6f}")
print(f"      POSITIVE MARGIN     : {np.mean(margins>0):.4f}")
print(f"      NULL R1 mean / P99  : {np.mean(NULL_R1):.4f} / {np.quantile(NULL_R1,.99):.4f}")
print(f"      NULL MRR mean / P99 : {np.mean(NULL_MRR):.6f} / {np.quantile(NULL_MRR,.99):.6f}")

# Conservative classification.
if EVAL["R1"]>=.75 and EVAL["MRR"]>=.82 and np.mean(margins>0)>=.75:
    VERDICT="NEXT_KEY_LATENT_TRACE_STRONGLY_LOCALIZED"
elif EVAL["R1"]>=.50 and EVAL["MRR"]>=.65:
    VERDICT="NEXT_KEY_LATENT_TRACE_LOCALIZED"
elif EVAL["R5"]>=.75 and EVAL["MRR"]>=.45:
    VERDICT="NEXT_KEY_LATENT_SIGNAL_OBSERVED_BUT_NOT_YET_SHARP"
else:
    VERDICT="NEXT_KEY_LATENT_TRACE_NOT_YET_LOCALIZED"

print("\n[7/7] Mechanistic classification...")
print("      A address                           : ORACLE")
print("      A latent forwards/item              : 1")
print("      Answer tokens generated             : 0")
print("      Intermediate Seal decoded           : 0")
print("      B cartridges queried                : 0")
print("      Per-Seal query-time model forwards  : 0")
print("      Seal references                     : precomputed")
print("      Discovery/evaluation separation     : YES")
print("      Evaluation channel reselection      : NO")
print("      Projection dtype path               : MODEL DTYPE → FP32 ANALYSIS")
print("      DRA / steering                      : NONE")
print("      Training / learned router           : NONE")

print("\n"+"="*174)
print("TEST510 FINAL RESULT — AKBASCORE MAM · ÇAĞRIİZ LATENT TRACE ATLAS")
print("="*174)
print("MODEL                         : Qwen/Qwen2.5-7B-Instruct · frozen")
print("PARENT                        : TEST509")
print("PARENT TEST509 LOCK           :",TEST509_LOCK_SHA)
print("PARENT TEST508 LOCK           :",TEST508_LOCK_SHA)
print("TASK                          : Oracle A → latent state → next-key Seal")
print("SEAL REFERENCE BANK           :",len(USED_SEALS))
print("A ADDRESS                     : ORACLE — isolation control")
print("ANSWER TOKEN GENERATION       : NONE")
print("INTERMEDIATE SEAL TEXT        : NEVER DECODED")
print("B-CARTRIDGE SEARCH            : NONE")
print("PER-CANDIDATE MODEL FORWARD   : NONE")
print("REFERENCE ENCODING            : frozen model · one-time")
print("PROJECTION COMPUTE DTYPE      :",MODEL_DTYPE)
print("ANALYSIS DTYPE                : FP32")
print("TRAINING                      : NONE")
print("DRA / STEERING                : NONE")
print("VTOKEN                        : NONE")
print("ANN                           : NONE")
print("DISCOVERY                     : items 01–16")
print("FROZEN EVALUATION             : items 17–32")
print("CANDIDATES                    : HRAW/HNORM/QMEAN/KPROJ/VPROJ × 28 layers")
print("FROZEN TRACE CHANNEL          :",f"{FROZEN_FAMILY} / L{FROZEN_LAYER:02d}")
print("-"*174)
print(f"DISCOVERY · R1                : {DISC_FROZEN['R1']:.4f}")
print(f"DISCOVERY · R5                : {DISC_FROZEN['R5']:.4f}")
print(f"DISCOVERY · R16               : {DISC_FROZEN['R16']:.4f}")
print(f"DISCOVERY · MRR               : {DISC_FROZEN['MRR']:.6f}")
print(f"FROZEN EVAL · R1              : {EVAL['R1']:.4f}")
print(f"FROZEN EVAL · R5              : {EVAL['R5']:.4f}")
print(f"FROZEN EVAL · R16             : {EVAL['R16']:.4f}")
print(f"FROZEN EVAL · MRR             : {EVAL['MRR']:.6f}")
print(f"FROZEN EVAL · MEDIAN RANK     : {EVAL['MED']:.1f}")
print(f"FROZEN EVAL · POS MARGIN      : {np.mean(margins>0):.4f}")
print(f"NULL · R1 MEAN                : {np.mean(NULL_R1):.4f}")
print(f"NULL · MRR MEAN               : {np.mean(NULL_MRR):.6f}")
print("-"*174)
print("TEST508 LOCK SHA              :",TEST508_LOCK_SHA)
print("TEST509 LOCK SHA              :",TEST509_LOCK_SHA)
print("TEST510 LOCK SHA              :",LOCK_SHA)
print(f"TOTAL TEST TIME                : {time.perf_counter()-T0:.2f}s")
print("EXPLORATORY VERDICT           :",VERDICT)
print("="*174)

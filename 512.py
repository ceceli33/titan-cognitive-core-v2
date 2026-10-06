# Copyright © 2026 Mustafa Akbaş
# New original AKBASCORE MAM material is source-available under the repository LICENSE.
# See repository LICENSE for complete terms.
#
# AKBASCORE MAM — TEST512
# L27 ÇAĞRIİZ → B-ADDRESS BRIDGE
# ORACLE A → PRE-DECODE L27 TRACE → PRECOMPUTED MODEL-NATIVE B ADDRESS
#
# QUESTION
# --------
# TEST511 established that the next associative key is strongly readable
# immediately before decoding:
#
#   A + query → L27 pre-decode state → Seal
#
# TEST512 asks:
#
#   Can that model-native pre-decode state address the correct B BELLEKÖZ
#   WITHOUT decoding/reinserting Seal text and WITHOUT a model forward
#   for every candidate B cartridge?
#
# TARGET
# ------
# QUERY(Entity)
#      ↓
# ORACLE A
#      ↓
# one frozen forward
#      ↓
# L27 pre-decode state
#      ↓
# frozen final RMSNorm + LM head
#      ↓
# model-native vocabulary fingerprint
#      ↓
# precomputed B-address fingerprints
#      ↓
# correct B BELLEKÖZ
#
# NO generated Seal.
# NO decoded/reinserted intermediate.
# NO B candidate forward at query time.
# NO YES/NO scan.
# NO learned router.
# NO ANN.
# NO DRA / steering.
# NO training.
#
# B DISTRACTOR STANDARD
# ---------------------
# Every B candidate is a real grammatical, meaningful record produced by
# the same dataset and processed through the same frozen-model pipeline:
#
#   "Seal XXX corresponds to routing class YYY."
#
# No placeholders.
# No random strings.
# No malformed/noise negatives.
#
# ADDRESS FINGERPRINTS
# --------------------
# B address fingerprints are precomputed ONCE from each candidate B record.
#
# Query side:
#   TEST511-sealed L27 pre-decode state → final RMSNorm → frozen lm_head.
#
# Address side:
#   full Seal sequence → vocabulary-token fingerprint.
#
# No literal Seal string comparison occurs at retrieval time.
#
# Candidate fingerprint families are declared before results:
#
#   TOKSUM   : sum of normalized LM-head rows for complete Seal token sequence
#   TOKMEAN  : mean of normalized LM-head rows
#   TOKLAST  : normalized LM-head row of final Seal token
#   TOKFIRST : normalized LM-head row of first Seal token
#
# Query fingerprints:
#
#   FULLVOC  : normalized complete L27 vocabulary-logit vector
#   TOP256   : same vector restricted to frozen query top-256 vocabulary values
#
# DISCOVERY 01–16 chooses exactly one family/mode.
# EVALUATION 17–32 is frozen.
#
# IMPORTANT
# ---------
# This is still an address-bridge experiment.
# Success does NOT yet establish sublinear retrieval.
# It establishes whether a reusable model-native address representation
# exists between the TEST511 ÇAĞRIİZ and B BELLEKÖZ identity.

import os,sys,subprocess,importlib.util,random,time,hashlib,json,re
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache

TEST="512";SEED=512;BASE_SEED=501;MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
N_ITEMS=32;N_DISC=16;N_EVAL=16;TRACE_DEPTH=27;TOPK=256
DEVICE=torch.device("cuda");FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
TEST508_LOCK_SHA="47512c7860268517670ca5b45b6e76fe1582362dd38e447d164f33c756a1a2ca"
TEST509_LOCK_SHA="4b5b78841ebf0d9b458306136e122fd3108b33bdfd89924585f891f748eca0d8"
TEST510_LOCK_SHA="cf3aaf955c0cd5019b1b951d185f290528286df82b23da62aaa30657e6bad5ea"
TEST511_LOCK_SHA="31558a2b86b1a1e86cb4f729d7c956781fe9269ff20bb2557a92b3300270e35c"

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

os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
torch.set_grad_enabled(False)
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

def reorder(rows,gold,target_pos,seed):
    g=rows[gold];rest=[x for j,x in enumerate(rows) if j!=gold]
    rr=random.Random(seed);rr.shuffle(rest);out=rest[:];out.insert(target_pos,g);return out

def make_base():
    rng=random.Random(BASE_SEED);seals=SEAL_BANK.copy();classes=CLASS_BANK.copy()
    rng.shuffle(seals);rng.shuffle(classes);items=[]
    for i in range(N_ITEMS):
        ents=list(ENTITY_BANK[i]);g=i%3;ss=seals[3*i:3*i+3];cc=classes[3*i:3*i+3]
        if len(ss)!=3 or len(cc)!=3:raise RuntimeError(f"Pool exhaustion item {i+1}.")
        r1=[f"Instrument {ents[j]} carries seal {ss[j]}." for j in range(3)]
        r2=[f"Seal {ss[j]} corresponds to routing class {cc[j]}." for j in range(3)]
        A=" ".join(reorder(r1,g,i%3,BASE_SEED+i*101+17))
        B=" ".join(reorder(r2,g,(i+1)%3,BASE_SEED+i*101+34))
        items.append({"id":i+1,"A":A,"B":B,"gold":{"entity":ents[g],"seal":ss[g],"class":cc[g]}})
    return items

ITEMS=make_base()
USED_SEALS=sorted({it["gold"]["seal"] for it in ITEMS})
if len(USED_SEALS)!=N_ITEMS:raise RuntimeError("Gold Seal uniqueness failure.")
SEAL_INDEX={s:i for i,s in enumerate(USED_SEALS)}

# Every B must be a complete meaningful record and must contain exactly one
# gold candidate Seal associated with its correct routing class.
for it in ITEMS:
    g=it["gold"]
    if f"Seal {g['seal']} corresponds to routing class {g['class']}." not in it["B"]:
        raise RuntimeError(f"Malformed gold B record item {it['id']}.")

FAMILIES=("TOKSUM","TOKMEAN","TOKLAST","TOKFIRST")
MODES=("FULLVOC","TOP256")

LOCK={
"test":TEST,"parent_test511":TEST511_LOCK_SHA,"parent_test510":TEST510_LOCK_SHA,
"parent_test509":TEST509_LOCK_SHA,"parent_test508":TEST508_LOCK_SHA,
"model":MODEL_ID,"seed":SEED,"base_seed":BASE_SEED,"items":ITEMS,
"trace_depth":TRACE_DEPTH,"topk":TOPK,
"n_items":N_ITEMS,"discovery_items":"01-16","evaluation_items":"17-32",
"task":"TEST511_L27_CAGRIIZ_to_precomputed_B_address_bridge",
"address_families":FAMILIES,"query_modes":MODES,
"query_forward_per_item":1,"query_time_candidate_model_forwards":0,
"intermediate_decode":"NONE","text_reinsertion":"NONE","B_read":"NONE",
"candidate_B_records":"REAL_GRAMMATICAL_MEANINGFUL_MODEL_NATIVE",
"placeholder_distractors":"FORBIDDEN",
"learned_router":"OFF","ann":"OFF","steering":"OFF","training":"OFF",
"selection":"DISCOVERY_ONLY_THEN_FROZEN_EVAL"
}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

print("="*174)
print("TEST512 — AKBASCORE MAM · L27 ÇAĞRIİZ → B-ADDRESS BRIDGE")
print("ORACLE A → PRE-DECODE L27 TRACE → PRECOMPUTED MODEL-NATIVE B ADDRESS")
print("="*174)
print("LOCK SHA:",LOCK_SHA)
print("PARENT TEST511 LOCK:",TEST511_LOCK_SHA)
print("PARENT TEST510 LOCK:",TEST510_LOCK_SHA)
print("PARENT TEST509 LOCK:",TEST509_LOCK_SHA)
print("PARENT TEST508 LOCK:",TEST508_LOCK_SHA)
print("TRACE DEPTH: L27 — SEALED BY TEST511")
print("BANK:",N_ITEMS,"REAL B BELLEKÖZ | DISCOVERY:",N_DISC,"| FROZEN EVAL:",N_EVAL)
print("QUERY-TIME MODEL FORWARDS PER B CANDIDATE: 0")
print("INTERMEDIATE SEAL DECODE: NONE")
print("TEXT REINSERTION: NONE")
print("DISTRACTORS: GRAMMATICAL + MEANINGFUL + SAME MODEL-NATIVE PIPELINE")
T0=time.perf_counter()

print("\n[1/8] Loading frozen Qwen...")
_tv=tuple(int("".join(c for c in x if c.isdigit())or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
enc=lambda s:tok(s,add_special_tokens=False).input_ids
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=H//QH
if (NL,H,QH,KVH,HD)!=(28,3584,28,4,128):raise RuntimeError(f"Architecture mismatch: {(NL,H,QH,KVH,HD)}")
if TRACE_DEPTH>=NL:raise RuntimeError("TRACE_DEPTH must reference a pre-final decoder residual.")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
if PAD is None:raise RuntimeError("No PAD/EOS token.")
VOCAB=model.lm_head.weight.shape[0]
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | {NL}L | H={H} | QH={QH} KVH={KVH} HD={HD} | vocab={VOCAB}")

@torch.inference_mode()
def kv_from_ids(ids):
    x=torch.tensor([ids],device=DEVICE,dtype=torch.long)
    o=model(input_ids=x,use_cache=False,output_hidden_states=True,return_dict=True)
    if len(o.hidden_states)!=NL+1:raise RuntimeError("Hidden-state count mismatch.")
    out=[]
    for L,layer in enumerate(model.model.layers):
        h=o.hidden_states[L][0].to(device=layer.input_layernorm.weight.device,dtype=layer.input_layernorm.weight.dtype)
        hn=layer.input_layernorm(h)
        k=layer.self_attn.k_proj(hn.to(device=layer.self_attn.k_proj.weight.device,dtype=layer.self_attn.k_proj.weight.dtype))
        v=layer.self_attn.v_proj(hn.to(device=layer.self_attn.v_proj.weight.device,dtype=layer.self_attn.v_proj.weight.dtype))
        k=k.view(-1,KVH,HD).transpose(0,1).contiguous()
        v=v.view(-1,KVH,HD).transpose(0,1).contiguous()
        out.append((k,v))
    return out

@torch.inference_mode()
def forge(s):return kv_from_ids([PAD]+enc(s+SEP))

def rope_k(k,pos,L):
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
    if len(raw)!=NL:raise RuntimeError("RAW layer mismatch.")
    T=raw[0][0].shape[1];out=[];pos=list(range(T))
    for L,(k,v) in enumerate(raw):
        if k.shape[1]!=T or v.shape[1]!=T:raise RuntimeError(f"Packet length mismatch L{L}.")
        out.append((rope_k(k,pos,L),v))
    return out,T,T

def make_cache(installed):
    try:cache=DynamicCache(config=cfg)
    except TypeError:cache=DynamicCache()
    for L,(k,v) in enumerate(installed):cache.update(k.unsqueeze(0),v.unsqueeze(0),L)
    return cache

def q_read_a(e):return f"What seal does instrument {e} carry? Give only the exact seal."

def unit(x):
    x=x.float().reshape(-1)
    n=x.norm()
    if not torch.isfinite(n) or n.item()<=0:raise RuntimeError("Invalid vector norm.")
    return x/n.clamp_min(1e-8)

@torch.inference_mode()
def l27_logits(installed,Tm,P,q):
    cache=make_cache(installed);qids=enc(FMT.format(q=q))
    if not qids:raise RuntimeError("Empty query.")
    nq=len(qids)
    pos=torch.arange(P,P+nq,device=DEVICE,dtype=torch.long).unsqueeze(0)
    mask=torch.ones((1,Tm+nq),device=DEVICE,dtype=torch.long)
    o=model(input_ids=torch.tensor([qids],device=DEVICE,dtype=torch.long),
            past_key_values=cache,attention_mask=mask,position_ids=pos,
            use_cache=False,output_hidden_states=True,return_dict=True)
    h=o.hidden_states[TRACE_DEPTH][0,-1]
    fn=model.model.norm
    h=fn(h.to(device=fn.weight.device,dtype=fn.weight.dtype))
    h=h.to(device=model.lm_head.weight.device,dtype=model.lm_head.weight.dtype)
    logits=model.lm_head(h).float()
    if logits.numel()!=VOCAB:raise RuntimeError("Vocabulary-logit size mismatch.")
    return logits.detach().cpu()

print("\n[2/8] Building exact Seal token identities...")
SEAL_TOKENS={}
for s in USED_SEALS:
    ids=enc(s)
    if not ids:raise RuntimeError(f"Empty Seal tokenization: {s}")
    SEAL_TOKENS[s]=ids
lens=[len(x) for x in SEAL_TOKENS.values()]
print("      Seal token lengths:",dict(sorted({n:lens.count(n) for n in set(lens)}.items())))
print("      Complete Seal sequences retained; first-token collisions are NOT treated as identity.")

# -------------------------------------------------------------------------
# PRECOMPUTED MODEL-NATIVE B ADDRESS FINGERPRINTS
# -------------------------------------------------------------------------
# The B records themselves are all real/meaningful and are forged through the
# same frozen-model memory pipeline. Their address identity is the complete
# Seal that naturally indexes each B record.
#
# The bridge representation is built from the frozen LM-head token embedding
# geometry. This is not a literal string comparison and requires no query-time
# candidate forward.
# -------------------------------------------------------------------------
print("\n[3/8] Precomputing real B BELLEKÖZ and address fingerprints...")
W=model.lm_head.weight.detach().float().cpu()
Wn=W/W.norm(dim=1,keepdim=True).clamp_min(1e-8)
B_PACKETS=[];ADDR={f:{} for f in FAMILIES}
for z,it in enumerate(ITEMS):
    # Every distractor B is actually processed into a genuine BELLEKÖZ packet.
    raw=forge(it["B"]);B_PACKETS.append(install_single(raw))
    seal=it["gold"]["seal"];ids=SEAL_TOKENS[seal]
    rows=Wn[ids]
    ADDR["TOKSUM"][seal]=unit(rows.sum(dim=0))
    ADDR["TOKMEAN"][seal]=unit(rows.mean(dim=0))
    ADDR["TOKLAST"][seal]=unit(rows[-1])
    ADDR["TOKFIRST"][seal]=unit(rows[0])
    if z<6:
        print(f"      B{z+1:02d} | Seal {seal} → Class {it['gold']['class']} | real packet tokens={B_PACKETS[-1][1]} | Seal subtokens={len(ids)}")
print(f"      {len(B_PACKETS)} real grammatical B BELLEKÖZ precomputed.")
del W,Wn
torch.cuda.empty_cache()

print("\n[4/8] Pre-forging A and capturing TEST511-sealed L27 ÇAĞRIİZ...")
QLOGITS=[]
for z,it in enumerate(ITEMS):
    A=install_single(forge(it["A"]))
    logits=l27_logits(*A,q_read_a(it["gold"]["entity"]))
    QLOGITS.append(logits)
    if z<6:print(f"      ITEM {it['id']:02d} {it['gold']['entity']:6s} | L27 trace captured | target Seal text decoded: NO")
print("      One A+query forward/item. No B candidate was queried.")

# Query logit → hidden-dimensional model-native address carrier.
# lm_head logits live in vocabulary space. Multiplication by frozen normalized
# LM-head rows maps the vocabulary evidence back into the model's own lexical
# hidden geometry without generating a token.
@torch.inference_mode()
def query_carrier(logits,mode):
    w=model.lm_head.weight.detach().float().cpu()
    if mode=="FULLVOC":
        p=torch.softmax(logits.float(),dim=0)
        h=torch.matmul(p,w)
    elif mode=="TOP256":
        val,idx=torch.topk(logits.float(),min(TOPK,logits.numel()))
        p=torch.softmax(val,dim=0)
        h=torch.matmul(p,w[idx])
    else:raise RuntimeError(mode)
    return unit(h)

print("\n      Building frozen query carriers...")
QCARRIERS={m:[] for m in MODES}
for mode in MODES:
    for z in range(N_ITEMS):QCARRIERS[mode].append(query_carrier(QLOGITS[z],mode))
print("      FULLVOC and TOP256 carriers ready.")

def cosine(a,b):
    if a.shape!=b.shape:raise RuntimeError(f"Shape mismatch {a.shape} vs {b.shape}")
    v=torch.dot(a.float(),b.float())
    if not torch.isfinite(v):raise RuntimeError("Non-finite score.")
    return float(v.item())

def score_bank(q,fam):
    return [cosine(q,ADDR[fam][ITEMS[j]["gold"]["seal"]]) for j in range(N_ITEMS)]

def ranks_from_scores(scores,gold_idx):
    if len(scores)!=N_ITEMS or not np.all(np.isfinite(scores)):raise RuntimeError("Invalid score bank.")
    order=sorted(range(N_ITEMS),key=lambda j:(-scores[j],j))
    return order.index(gold_idx)+1,order

def rank_stats(x):
    a=np.asarray(x,dtype=float)
    return {"R1":float(np.mean(a==1)),"R5":float(np.mean(a<=5)),
            "R16":float(np.mean(a<=16)),"MRR":float(np.mean(1/a)),
            "MED":float(np.median(a))}

# -------------------------------------------------------------------------
# DISCOVERY
# -------------------------------------------------------------------------
print("\n[5/8] DISCOVERY — address bridge selection, items 01–16 only...")
DISC={}
for mode in MODES:
    for fam in FAMILIES:
        ranks=[]
        for z in range(N_DISC):
            scores=score_bank(QCARRIERS[mode][z],fam)
            r,_=ranks_from_scores(scores,z);ranks.append(r)
        DISC[(mode,fam)]=rank_stats(ranks)

# Predeclared tie preference:
# complete-token identities before partial-token identities;
# sparse TOP256 before FULLVOC only on exact metric tie.
FPREF={"TOKSUM":4,"TOKMEAN":3,"TOKLAST":2,"TOKFIRST":1}
MPREF={"TOP256":2,"FULLVOC":1}
def selkey(x):
    (mode,fam),s=x
    return (s["R1"],s["MRR"],s["R5"],FPREF[fam],MPREF[mode])

ordered=sorted(DISC.items(),key=selkey,reverse=True)
for n,((mode,fam),s) in enumerate(ordered,1):
    print(f"      #{n:02d} {mode:7s}/{fam:8s} | R1={s['R1']:.4f} R5={s['R5']:.4f} R16={s['R16']:.4f} MRR={s['MRR']:.6f} median={s['MED']:.1f}")

(FROZEN_MODE,FROZEN_FAMILY),DISC_FROZEN=ordered[0]
print(f"\n      FROZEN BRIDGE → {FROZEN_MODE} / {FROZEN_FAMILY}")
print("      Evaluation items have NOT participated in bridge selection.")

# -------------------------------------------------------------------------
# FROZEN EVALUATION
# -------------------------------------------------------------------------
print("\n[6/8] FROZEN EVALUATION — L27 ÇAĞRIİZ → 32 real B addresses...")
EVAL_RANKS=[];EVAL_BANKS=[];GOLD=[];WRONG=[]
for z in range(N_DISC,N_ITEMS):
    scores=score_bank(QCARRIERS[FROZEN_MODE][z],FROZEN_FAMILY)
    r,order=ranks_from_scores(scores,z);pred=order[0]
    EVAL_RANKS.append(r);EVAL_BANKS.append(scores)
    GOLD.append(scores[z]);WRONG.append(max(scores[j] for j in range(N_ITEMS) if j!=z))
    g=ITEMS[z]["gold"];pg=ITEMS[pred]["gold"]
    print(f"      [{z+1:02d}/{N_ITEMS}] {g['entity']:6s} | gold=B{z+1:02d}({g['seal']}) pred=B{pred+1:02d}({pg['seal']}) rank={r:2d} | margin={scores[z]-WRONG[-1]:+.6f}")

EVAL=rank_stats(EVAL_RANKS)
margins=np.asarray(GOLD)-np.asarray(WRONG)

print("\n[7/8] Frozen bridge statistics + random-label null...")
rng=np.random.default_rng(SEED);NULL_R1=[];NULL_MRR=[]
for _ in range(10000):
    rr=[]
    for scores in EVAL_BANKS:
        fake=int(rng.integers(0,N_ITEMS))
        r,_=ranks_from_scores(scores,fake);rr.append(r)
    rr=np.asarray(rr,dtype=float)
    NULL_R1.append(float(np.mean(rr==1)))
    NULL_MRR.append(float(np.mean(1/rr)))

print(f"      BRIDGE                : {FROZEN_MODE}/{FROZEN_FAMILY}")
print(f"      DISCOVERY R1          : {DISC_FROZEN['R1']:.4f}")
print(f"      DISCOVERY R5          : {DISC_FROZEN['R5']:.4f}")
print(f"      DISCOVERY MRR         : {DISC_FROZEN['MRR']:.6f}")
print(f"      EVAL R1               : {EVAL['R1']:.4f}")
print(f"      EVAL R5               : {EVAL['R5']:.4f}")
print(f"      EVAL R16              : {EVAL['R16']:.4f}")
print(f"      EVAL MRR              : {EVAL['MRR']:.6f}")
print(f"      EVAL MEDIAN RANK      : {EVAL['MED']:.1f}")
print(f"      POSITIVE MARGIN       : {np.mean(margins>0):.4f}")
print(f"      MEAN MARGIN           : {np.mean(margins):+.6f}")
print(f"      NULL R1 mean / P99    : {np.mean(NULL_R1):.4f} / {np.quantile(NULL_R1,.99):.4f}")
print(f"      NULL MRR mean / P99   : {np.mean(NULL_MRR):.6f} / {np.quantile(NULL_MRR,.99):.6f}")

if EVAL["R1"]>=.75 and EVAL["MRR"]>=.82 and np.mean(margins>0)>=.75:
    VERDICT="LATENT_CAGRIIZ_TO_B_ADDRESS_BRIDGE_STRONGLY_OBSERVED"
elif EVAL["R1"]>=.50 and EVAL["MRR"]>=.65:
    VERDICT="LATENT_CAGRIIZ_TO_B_ADDRESS_BRIDGE_OBSERVED"
elif EVAL["R5"]>=.75 and EVAL["MRR"]>=.45:
    VERDICT="LATENT_CAGRIIZ_TO_B_ADDRESS_SIGNAL_OBSERVED_BUT_DIFFUSE"
else:
    VERDICT="LATENT_CAGRIIZ_TO_B_ADDRESS_BRIDGE_NOT_YET_OBSERVED"

print("\n[8/8] Mechanistic classification...")
print("      A address                           : ORACLE")
print("      ÇAĞRIİZ depth                       : L27 — TEST511 sealed")
print("      A+query forwards/item               : 1")
print("      Answer tokens generated             : 0")
print("      Intermediate Seal decoded           : 0")
print("      Intermediate text reinserted        : 0")
print("      B candidates                        : 32 real BELLEKÖZ")
print("      Query-time B candidate forwards     : 0")
print("      B content read before selection     : 0")
print("      Address fingerprints                : precomputed")
print("      Learned router                      : NONE")
print("      ANN                                 : NONE")
print("      DRA / steering                      : NONE")
print("      Training                            : NONE")
print("      Distractor placeholders/noise       : FORBIDDEN")
print("      Discovery/evaluation separation     : YES")
print("      Evaluation bridge reselection       : NO")

print("\n"+"="*174)
print("TEST512 FINAL RESULT — AKBASCORE MAM · L27 ÇAĞRIİZ → B-ADDRESS BRIDGE")
print("="*174)
print("MODEL                         : Qwen/Qwen2.5-7B-Instruct · frozen")
print("PARENT                        : TEST511")
print("PARENT TEST511 LOCK           :",TEST511_LOCK_SHA)
print("PARENT TEST510 LOCK           :",TEST510_LOCK_SHA)
print("PARENT TEST509 LOCK           :",TEST509_LOCK_SHA)
print("PARENT TEST508 LOCK           :",TEST508_LOCK_SHA)
print("TASK                          : L27 ÇAĞRIİZ → precomputed B address")
print("TRACE DEPTH                   : L27")
print("B BANK                        : 32 real grammatical BELLEKÖZ")
print("ANSWER GENERATION             : NONE")
print("INTERMEDIATE TEXT             : NONE")
print("QUERY-TIME CANDIDATE FORWARD  : NONE")
print("B READ BEFORE ADDRESS         : NONE")
print("LEARNED ROUTER                : NONE")
print("ANN                           : NONE")
print("TRAINING                      : NONE")
print("DISCOVERY                     : items 01–16")
print("FROZEN EVALUATION             : items 17–32")
print("FROZEN BRIDGE                 :",f"{FROZEN_MODE}/{FROZEN_FAMILY}")
print("-"*174)
print(f"DISCOVERY · R1                : {DISC_FROZEN['R1']:.4f}")
print(f"DISCOVERY · R5                : {DISC_FROZEN['R5']:.4f}")
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
print("TEST510 LOCK SHA              :",TEST510_LOCK_SHA)
print("TEST511 LOCK SHA              :",TEST511_LOCK_SHA)
print("TEST512 LOCK SHA              :",LOCK_SHA)
print(f"TOTAL TEST TIME               : {time.perf_counter()-T0:.2f}s")
print("EXPLORATORY VERDICT           :",VERDICT)
print("="*174)

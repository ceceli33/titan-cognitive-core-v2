# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST513
# ORDERED FULL-SEAL LATENT ADDRESS
# ORACLE A → SINGLE PRE-DECODE L27 STATE → ORDER-PRESERVING 2-SLOT ADDRESS → REAL B BELLEKÖZ
#
# PURPOSE
# -------
# TEST511:
#   L27 contains strong pre-decode next-key information.
#
# TEST512:
#   L27 → B address showed above-random signal, but TOKFIRST collapsed
#   Seals sharing the same first tokenizer token.
#
# TEST513 asks:
#
#   Does ONE frozen L27 state already contain enough information to form
#   a collision-resistant ORDERED FULL-SEAL address, before any answer
#   token is generated?
#
# STRICT CONSTRAINTS
# ------------------
# ONE A+query model forward/item.
# NO answer token generation.
# NO teacher forcing.
# NO second autoregressive forward.
# NO decoded Seal.
# NO text reinsertion.
# NO B candidate forward at query time.
# NO YES/NO scan.
# NO learned router/probe.
# NO ANN.
# NO DRA/steering.
# NO training.
#
# DISTRACTOR STANDARD
# -------------------
# All 32 B candidates are real grammatical meaningful records processed
# through the same frozen-model BELLEKÖZ forge pipeline.
# No placeholder/noise/malformed distractors.
#
# ADDRESS IDEA
# ------------
# Each Seal has 1 or 2 tokenizer tokens in this locked bank.
#
# Instead of:
#   TOKFIRST = token1
#   TOKSUM   = token1 + token2  (order destroyed)
#
# TEST513 uses ordered slots:
#
#   ADDRESS(Seal) = [token1_embedding || token2_embedding]
#
# One-token Seals use an explicit zero second slot plus a length bit.
#
# Query side must be constructed from the SAME SINGLE L27 vocabulary
# distribution. No first token is fed back.
#
# Slot 1:
#   ordinary L27 token evidence.
#
# Slot 2:
#   residual vocabulary evidence after removing the component explained
#   by slot 1 candidates. This is a target-blind algebraic decomposition
#   of the SAME frozen L27 logits; it is NOT an autoregressive step.
#
# Predeclared query variants:
#   RAW2       : top-1 lexical carrier + residual carrier
#   SOFT2      : soft first-slot carrier + orthogonal residual carrier
#   GROUP2     : first-token-group conditional residual carrier
#
# Discovery 01–16 selects one variant.
# Evaluation 17–32 is frozen.
#
# A positive result means full-key identity is recoverable from one L27
# instant. A negative result, especially with strong slot1 but weak slot2,
# supports the hypothesis that later subtokens emerge autoregressively.

import os,sys,subprocess,importlib.util,random,time,hashlib,json,re
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache

TEST="513";SEED=513;BASE_SEED=501;MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
N_ITEMS=32;N_DISC=16;N_EVAL=16;TRACE_DEPTH=27;TOPK=256
DEVICE=torch.device("cuda");FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
TEST508_LOCK_SHA="47512c7860268517670ca5b45b6e76fe1582362dd38e447d164f33c756a1a2ca"
TEST509_LOCK_SHA="4b5b78841ebf0d9b458306136e122fd3108b33bdfd89924585f891f748eca0d8"
TEST510_LOCK_SHA="cf3aaf955c0cd5019b1b951d185f290528286df82b23da62aaa30657e6bad5ea"
TEST511_LOCK_SHA="31558a2b86b1a1e86cb4f729d7c956781fe9269ff20bb2557a92b3300270e35c"
TEST512_LOCK_SHA="2c9576ce56491489be55f34a1a6d331a29bc4fd8080084693e92d2dab4a43ac2"

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

assert N_DISC+N_EVAL==N_ITEMS and len(ENTITY_BANK)==N_ITEMS
assert len(SEAL_BANK)>=N_ITEMS*3 and len(CLASS_BANK)>=N_ITEMS*3
assert len(set(SEAL_BANK))==len(SEAL_BANK) and len(set(CLASS_BANK))==len(CLASS_BANK)
assert set(SEAL_BANK).isdisjoint(CLASS_BANK)
assert all(re.fullmatch(r"[A-Z]{3}",x) for x in SEAL_BANK+CLASS_BANK)

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
        r1=[f"Instrument {ents[j]} carries seal {ss[j]}." for j in range(3)]
        r2=[f"Seal {ss[j]} corresponds to routing class {cc[j]}." for j in range(3)]
        A=" ".join(reorder(r1,g,i%3,BASE_SEED+i*101+17))
        B=" ".join(reorder(r2,g,(i+1)%3,BASE_SEED+i*101+34))
        items.append({"id":i+1,"A":A,"B":B,"gold":{"entity":ents[g],"seal":ss[g],"class":cc[g]}})
    return items

ITEMS=make_base()
USED_SEALS=sorted({it["gold"]["seal"] for it in ITEMS})
if len(USED_SEALS)!=32:raise RuntimeError("Gold Seal uniqueness failure.")
SEAL_INDEX={s:i for i,s in enumerate(USED_SEALS)}
for it in ITEMS:
    g=it["gold"]
    if f"Seal {g['seal']} corresponds to routing class {g['class']}." not in it["B"]:
        raise RuntimeError(f"Malformed B item {it['id']}.")

VARIANTS=("RAW2","SOFT2","GROUP2")
LOCK={
"test":TEST,"parent_test512":TEST512_LOCK_SHA,"parent_test511":TEST511_LOCK_SHA,
"parent_test510":TEST510_LOCK_SHA,"parent_test509":TEST509_LOCK_SHA,"parent_test508":TEST508_LOCK_SHA,
"model":MODEL_ID,"seed":SEED,"base_seed":BASE_SEED,"items":ITEMS,"trace_depth":TRACE_DEPTH,
"topk":TOPK,"variants":VARIANTS,"address":"ORDERED_TWO_SLOT_COMPLETE_SEAL",
"one_token_address":"slot1_plus_zero_slot2_plus_length_bit",
"query_forwards_per_item":1,"answer_generation":"OFF","teacher_forcing":"OFF",
"second_autoregressive_forward":"OFF","decoded_intermediate":"NONE","text_reinsertion":"NONE",
"query_time_B_candidate_forwards":0,"B_read_before_selection":"OFF",
"candidate_B_records":"REAL_GRAMMATICAL_MEANINGFUL_MODEL_NATIVE",
"placeholder_distractors":"FORBIDDEN","learned_router":"OFF","ann":"OFF",
"steering":"OFF","training":"OFF","selection":"DISCOVERY_01_16_THEN_FROZEN_EVAL_17_32"
}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

print("="*174)
print("TEST513 — AKBASCORE MAM · ORDERED FULL-SEAL LATENT ADDRESS")
print("ORACLE A → SINGLE PRE-DECODE L27 STATE → ORDER-PRESERVING 2-SLOT ADDRESS → REAL B BELLEKÖZ")
print("="*174)
print("LOCK SHA:",LOCK_SHA)
print("PARENT TEST512 LOCK:",TEST512_LOCK_SHA)
print("PARENT TEST511 LOCK:",TEST511_LOCK_SHA)
print("TRACE DEPTH: L27 — SEALED BY TEST511")
print("BANK: 32 REAL B BELLEKÖZ | DISCOVERY: 16 | FROZEN EVAL: 16")
print("A+QUERY FORWARDS/ITEM: 1")
print("ANSWER TOKEN GENERATION: NONE | TEACHER FORCING: NONE | SECOND AUTOREGRESSIVE FORWARD: NONE")
print("QUERY-TIME B CANDIDATE FORWARDS: 0 | TEXT REINSERTION: NONE")
T0=time.perf_counter()

print("\n[1/9] Loading frozen Qwen...")
_tv=tuple(int("".join(c for c in x if c.isdigit())or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
enc=lambda s:tok(s,add_special_tokens=False).input_ids
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=H//QH
if (NL,H,QH,KVH,HD)!=(28,3584,28,4,128):raise RuntimeError(f"Architecture mismatch {(NL,H,QH,KVH,HD)}")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
if PAD is None:raise RuntimeError("No PAD/EOS.")
VOCAB=model.lm_head.weight.shape[0]
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | {NL}L | H={H} | vocab={VOCAB}")

@torch.inference_mode()
def kv_from_ids(ids):
    x=torch.tensor([ids],device=DEVICE,dtype=torch.long)
    o=model(input_ids=x,use_cache=False,output_hidden_states=True,return_dict=True)
    if len(o.hidden_states)!=NL+1:raise RuntimeError("Hidden-state mismatch.")
    out=[]
    for L,layer in enumerate(model.model.layers):
        h=o.hidden_states[L][0].to(layer.input_layernorm.weight.device,layer.input_layernorm.weight.dtype)
        hn=layer.input_layernorm(h)
        k=layer.self_attn.k_proj(hn.to(layer.self_attn.k_proj.weight.device,layer.self_attn.k_proj.weight.dtype))
        v=layer.self_attn.v_proj(hn.to(layer.self_attn.v_proj.weight.device,layer.self_attn.v_proj.weight.dtype))
        out.append((k.view(-1,KVH,HD).transpose(0,1).contiguous(),
                    v.view(-1,KVH,HD).transpose(0,1).contiguous()))
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

def install_single(raw):
    if len(raw)!=NL:raise RuntimeError("RAW layer mismatch.")
    T=raw[0][0].shape[1];pos=list(range(T));out=[]
    for L,(k,v) in enumerate(raw):
        if k.shape[1]!=T or v.shape[1]!=T:raise RuntimeError("Packet length mismatch.")
        out.append((rope_k(k,pos,L),v))
    return out,T,T

def make_cache(installed):
    try:c=DynamicCache(config=cfg)
    except TypeError:c=DynamicCache()
    for L,(k,v) in enumerate(installed):c.update(k.unsqueeze(0),v.unsqueeze(0),L)
    return c

def q_read_a(e):return f"What seal does instrument {e} carry? Give only the exact seal."
def unit(x):
    x=x.float().reshape(-1);n=x.norm()
    if not torch.isfinite(n) or n.item()<=0:raise RuntimeError("Invalid norm.")
    return x/n.clamp_min(1e-8)

print("\n[2/9] Locking complete Seal token sequences...")
SEAL_TOKENS={s:enc(s) for s in USED_SEALS}
if any(len(x)==0 or len(x)>2 for x in SEAL_TOKENS.values()):
    raise RuntimeError("TEST513 locked design requires all candidate Seals to tokenize to 1 or 2 tokens.")
lens=[len(x) for x in SEAL_TOKENS.values()]
print("      Seal token lengths:",dict(sorted({n:lens.count(n) for n in set(lens)}.items())))
FIRST_GROUPS={}
for s,ids in SEAL_TOKENS.items():FIRST_GROUPS.setdefault(ids[0],[]).append(s)
coll={k:v for k,v in FIRST_GROUPS.items() if len(v)>1}
print("      First-token identities:",len(FIRST_GROUPS),"/ 32 | collision groups:",len(coll))
for tid,ss in coll.items():print("      ",tok.convert_ids_to_tokens([tid]),"→",ss)

print("\n[3/9] Precomputing 32 real B BELLEKÖZ...")
B_PACKETS=[]
for z,it in enumerate(ITEMS):
    B_PACKETS.append(install_single(forge(it["B"])))
    if z<6:
        g=it["gold"];print(f"      B{z+1:02d} | {g['seal']}→{g['class']} | packet tokens={B_PACKETS[-1][1]}")
print("      All distractors are grammatical meaningful model-processed BELLEKÖZ.")

print("\n[4/9] Building ordered two-slot B addresses...")
W=model.lm_head.weight.detach().float().cpu()
Wn=W/W.norm(dim=1,keepdim=True).clamp_min(1e-8)
ZERO=torch.zeros(H,dtype=torch.float32)

# Address = [slot1(H),slot2(H),length-bit].
ADDR=[]
for z,it in enumerate(ITEMS):
    ids=SEAL_TOKENS[it["gold"]["seal"]]
    a1=Wn[ids[0]]
    if len(ids)==2:a2=Wn[ids[1]];lb=1.0
    else:a2=ZERO;lb=0.0
    # Length coordinate scaled to same approximate slot norm.
    ADDR.append(unit(torch.cat([a1,a2,torch.tensor([lb],dtype=torch.float32)])))
print("      Ordered addresses ready: dimension",2*H+1)
print("      Token order preserved; one-token/two-token identities explicitly separated.")

@torch.inference_mode()
def capture_l27(installed,Tm,P,q):
    cache=make_cache(installed);qids=enc(FMT.format(q=q));nq=len(qids)
    pos=torch.arange(P,P+nq,device=DEVICE,dtype=torch.long).unsqueeze(0)
    mask=torch.ones((1,Tm+nq),device=DEVICE,dtype=torch.long)
    o=model(input_ids=torch.tensor([qids],device=DEVICE,dtype=torch.long),
            past_key_values=cache,attention_mask=mask,position_ids=pos,
            use_cache=False,output_hidden_states=True,return_dict=True)
    if len(o.hidden_states)!=NL+1:raise RuntimeError("Hidden-state count mismatch.")
    h=o.hidden_states[TRACE_DEPTH][0,-1]
    fn=model.model.norm
    hn=fn(h.to(fn.weight.device,fn.weight.dtype))
    logits=model.lm_head(hn.to(model.lm_head.weight.device,model.lm_head.weight.dtype)).float().cpu()
    return h.detach().float().cpu(),logits

print("\n[5/9] Capturing one-shot L27 ÇAĞRIİZ...")
TRACES=[];LOGITS=[]
for z,it in enumerate(ITEMS):
    A=install_single(forge(it["A"]))
    h,l=capture_l27(*A,q_read_a(it["gold"]["entity"]))
    TRACES.append(h);LOGITS.append(l)
    if z<6:print(f"      ITEM {z+1:02d} {it['gold']['entity']:6s} | one L27 state captured | generated tokens=0")
print("      Exactly one A+query forward/item.")

# -------------------------------------------------------------------------
# TARGET-BLIND QUERY ADDRESS CONSTRUCTION
# -------------------------------------------------------------------------
# All three variants use ONLY the single L27 vocabulary distribution.
#
# RAW2:
#   slot1 = embedding of globally highest-scoring candidate first-token ID.
#   slot2 = weighted carrier of remaining candidate Seal-token evidence.
#
# SOFT2:
#   slot1 = soft mixture over all candidate first-token embeddings.
#   slot2 = soft mixture over all candidate second-token embeddings after
#           removing projection onto slot1.
#
# GROUP2:
#   slot1 = soft first-token carrier.
#   slot2 = second-token carrier weighted by BOTH first-token group evidence
#           and second-token evidence, still from the same static logits.
#
# No gold identity is used in constructing a query vector.
# -------------------------------------------------------------------------

CAND_FIRST=sorted(set(ids[0] for ids in SEAL_TOKENS.values()))
CAND_SECOND=sorted(set(ids[1] for ids in SEAL_TOKENS.values() if len(ids)==2))
FIRST_MAT=Wn[CAND_FIRST]
SECOND_MAT=Wn[CAND_SECOND]
FIRST_ID_TO_ROW={t:i for i,t in enumerate(CAND_FIRST)}
SECOND_ID_TO_ROW={t:i for i,t in enumerate(CAND_SECOND)}

def softmax_np_tensor(x):
    return torch.softmax(x.float(),dim=0)

def orth_residual(v,base):
    base=unit(base);return v-torch.dot(v,base)*base

def query_address(logits,variant):
    # Candidate-restricted evidence; no arbitrary vocabulary noise.
    fl=torch.tensor([float(logits[t]) for t in CAND_FIRST])
    sl=torch.tensor([float(logits[t]) for t in CAND_SECOND]) if CAND_SECOND else torch.empty(0)

    if variant=="RAW2":
        # Slot1 hard first-token winner.
        i=int(torch.argmax(fl).item());q1=FIRST_MAT[i]
        # Slot2 static evidence over second-token vocabulary.
        if len(CAND_SECOND):
            p2=softmax_np_tensor(sl);q2=torch.matmul(p2,SECOND_MAT)
            q2=orth_residual(q2,q1)
            if q2.norm().item()>1e-8:q2=unit(q2)
            else:q2=ZERO
        else:q2=ZERO
        # Length confidence: probability mass of first-token groups that
        # correspond to 2-token candidate Seals.
        pf=softmax_np_tensor(fl);two=0.0
        for j,t in enumerate(CAND_FIRST):
            if any(len(SEAL_TOKENS[s])==2 for s in FIRST_GROUPS[t]):two+=float(pf[j])
        lb=two

    elif variant=="SOFT2":
        pf=softmax_np_tensor(fl);q1=unit(torch.matmul(pf,FIRST_MAT))
        if len(CAND_SECOND):
            p2=softmax_np_tensor(sl);q2=torch.matmul(p2,SECOND_MAT)
            q2=orth_residual(q2,q1)
            q2=unit(q2) if q2.norm().item()>1e-8 else ZERO
        else:q2=ZERO
        two=0.0
        for j,t in enumerate(CAND_FIRST):
            if any(len(SEAL_TOKENS[s])==2 for s in FIRST_GROUPS[t]):two+=float(pf[j])
        lb=two

    elif variant=="GROUP2":
        pf=softmax_np_tensor(fl);q1=unit(torch.matmul(pf,FIRST_MAT))
        # Joint static evidence for every 2-token candidate:
        # score = first-token logit + second-token logit.
        rows=[];weights=[]
        for s,ids in SEAL_TOKENS.items():
            if len(ids)==2:
                rows.append(Wn[ids[1]])
                weights.append(float(logits[ids[0]]+logits[ids[1]]))
        if rows:
            pp=torch.softmax(torch.tensor(weights,dtype=torch.float32),dim=0)
            q2=torch.matmul(pp,torch.stack(rows))
            q2=orth_residual(q2,q1)
            q2=unit(q2) if q2.norm().item()>1e-8 else ZERO
        else:q2=ZERO
        # Static probability that winning identity belongs to two-token family.
        one_scores=[];two_scores=[]
        for s,ids in SEAL_TOKENS.items():
            sc=float(logits[ids[0]])
            if len(ids)==1:one_scores.append(sc)
            else:two_scores.append(sc+float(logits[ids[1]]))
        a=torch.logsumexp(torch.tensor(two_scores),0) if two_scores else torch.tensor(-1e9)
        b=torch.logsumexp(torch.tensor(one_scores),0) if one_scores else torch.tensor(-1e9)
        lb=float(torch.sigmoid(a-b))
    else:raise RuntimeError(variant)

    return unit(torch.cat([unit(q1),q2,torch.tensor([lb],dtype=torch.float32)]))

print("\n[6/9] Building one-shot ordered query addresses...")
QADDR={v:[] for v in VARIANTS}
for v in VARIANTS:
    for z in range(N_ITEMS):QADDR[v].append(query_address(LOGITS[z],v))
print("      RAW2 / SOFT2 / GROUP2 ready.")
print("      Additional model forwards: 0 | generated tokens: 0 | teacher forcing: 0")

def cosine(a,b):
    if a.shape!=b.shape:raise RuntimeError("Address shape mismatch.")
    x=torch.dot(a.float(),b.float())
    if not torch.isfinite(x):raise RuntimeError("Non-finite score.")
    return float(x)

def score_bank(q):return [cosine(q,a) for a in ADDR]

def rank_scores(scores,gold):
    if len(scores)!=N_ITEMS or not np.all(np.isfinite(scores)):raise RuntimeError("Invalid scores.")
    order=sorted(range(N_ITEMS),key=lambda j:(-scores[j],j))
    return order.index(gold)+1,order

def stats(rr):
    a=np.asarray(rr,dtype=float)
    return {"R1":float(np.mean(a==1)),"R5":float(np.mean(a<=5)),
            "R16":float(np.mean(a<=16)),"MRR":float(np.mean(1/a)),
            "MED":float(np.median(a))}

print("\n[7/9] DISCOVERY — items 01–16 only...")
DISC={}
for v in VARIANTS:
    rr=[]
    for z in range(N_DISC):
        r,_=rank_scores(score_bank(QADDR[v][z]),z);rr.append(r)
    DISC[v]=stats(rr)
ordered=sorted(DISC.items(),key=lambda x:(x[1]["R1"],x[1]["MRR"],x[1]["R5"],-VARIANTS.index(x[0])),reverse=True)
for n,(v,s) in enumerate(ordered,1):
    print(f"      #{n:02d} {v:6s} | R1={s['R1']:.4f} R5={s['R5']:.4f} R16={s['R16']:.4f} MRR={s['MRR']:.6f} median={s['MED']:.1f}")
FROZEN,DISC_FROZEN=ordered[0]
print(f"\n      FROZEN ADDRESS → {FROZEN}")
print("      Evaluation items have NOT participated in selection.")

print("\n[8/9] FROZEN EVALUATION...")
RR=[];BANKS=[];GOLD=[];WRONG=[]
for z in range(N_DISC,N_ITEMS):
    sc=score_bank(QADDR[FROZEN][z]);r,order=rank_scores(sc,z);pred=order[0]
    RR.append(r);BANKS.append(sc);GOLD.append(sc[z]);WRONG.append(max(sc[j] for j in range(N_ITEMS) if j!=z))
    g=ITEMS[z]["gold"];p=ITEMS[pred]["gold"]
    print(f"      [{z+1:02d}/32] {g['entity']:6s} | gold=B{z+1:02d}({g['seal']}) pred=B{pred+1:02d}({p['seal']}) rank={r:2d} | margin={sc[z]-WRONG[-1]:+.6f}")
EVAL=stats(RR);margins=np.asarray(GOLD)-np.asarray(WRONG)

# Collision subset diagnostic: items whose gold Seal shares its first token.
COLL_R=[];UNIQ_R=[]
for local,z in enumerate(range(N_DISC,N_ITEMS)):
    ids=SEAL_TOKENS[ITEMS[z]["gold"]["seal"]]
    target=COLL_R if len(FIRST_GROUPS[ids[0]])>1 else UNIQ_R
    target.append(RR[local])
COLL=stats(COLL_R) if COLL_R else None
UNIQ=stats(UNIQ_R) if UNIQ_R else None

rng=np.random.default_rng(SEED);NULL_R1=[];NULL_MRR=[]
for _ in range(10000):
    rr=[]
    for sc in BANKS:
        fake=int(rng.integers(0,N_ITEMS));r,_=rank_scores(sc,fake);rr.append(r)
    a=np.asarray(rr,float);NULL_R1.append(float(np.mean(a==1)));NULL_MRR.append(float(np.mean(1/a)))

print("\n      Frozen statistics:")
print(f"      ADDRESS               : {FROZEN}")
print(f"      DISC R1/R5/MRR        : {DISC_FROZEN['R1']:.4f} / {DISC_FROZEN['R5']:.4f} / {DISC_FROZEN['MRR']:.6f}")
print(f"      EVAL R1/R5/R16        : {EVAL['R1']:.4f} / {EVAL['R5']:.4f} / {EVAL['R16']:.4f}")
print(f"      EVAL MRR/MEDIAN       : {EVAL['MRR']:.6f} / {EVAL['MED']:.1f}")
print(f"      POSITIVE MARGIN       : {np.mean(margins>0):.4f}")
if COLL:print(f"      COLLISION-GROUP R1    : {COLL['R1']:.4f} | MRR={COLL['MRR']:.6f} | n={len(COLL_R)}")
if UNIQ:print(f"      UNIQUE-FIRST R1       : {UNIQ['R1']:.4f} | MRR={UNIQ['MRR']:.6f} | n={len(UNIQ_R)}")
print(f"      NULL R1 mean / P99    : {np.mean(NULL_R1):.4f} / {np.quantile(NULL_R1,.99):.4f}")
print(f"      NULL MRR mean / P99   : {np.mean(NULL_MRR):.6f} / {np.quantile(NULL_MRR,.99):.6f}")

# Interpretation threshold deliberately requires strong collision resolution.
coll_r1=COLL["R1"] if COLL else EVAL["R1"]
if EVAL["R1"]>=.75 and EVAL["MRR"]>=.82 and coll_r1>=.70 and np.mean(margins>0)>=.70:
    VERDICT="ONE_SHOT_FULL_KEY_LATENT_ADDRESS_STRONGLY_OBSERVED"
elif EVAL["R1"]>=.50 and EVAL["MRR"]>=.65 and coll_r1>=.40:
    VERDICT="ONE_SHOT_FULL_KEY_LATENT_ADDRESS_OBSERVED"
elif EVAL["R5"]>=.75 and EVAL["MRR"]>=.45:
    VERDICT="ONE_SHOT_FULL_KEY_SIGNAL_PRESENT_BUT_NOT_SHARP"
else:
    VERDICT="ONE_SHOT_FULL_KEY_LATENT_ADDRESS_NOT_YET_OBSERVED"

print("\n[9/9] Final mechanistic classification...")
print("      A address                           : ORACLE")
print("      ÇAĞRIİZ depth                       : L27 — TEST511 sealed")
print("      A+query forwards/item               : 1")
print("      Answer tokens generated             : 0")
print("      Teacher-forced tokens               : 0")
print("      Second autoregressive forward       : 0")
print("      Intermediate Seal decoded           : 0")
print("      Text reinserted                     : 0")
print("      Query-time B candidate forwards     : 0")
print("      B content read before selection     : 0")
print("      B candidates                        : 32 real grammatical BELLEKÖZ")
print("      Learned router/probe                : NONE")
print("      ANN / DRA / training                : NONE")
print("      Evaluation reselection              : NO")

print("\n"+"="*174)
print("TEST513 FINAL RESULT — AKBASCORE MAM · ORDERED FULL-SEAL LATENT ADDRESS")
print("="*174)
print("MODEL                         : Qwen/Qwen2.5-7B-Instruct · frozen")
print("PARENT                        : TEST512")
print("TRACE DEPTH                   : L27")
print("TASK                          : one-shot L27 → ordered full-Seal → B address")
print("B BANK                        : 32 real grammatical BELLEKÖZ")
print("QUERY FORWARDS/ITEM           : 1")
print("ANSWER GENERATION             : NONE")
print("TEACHER FORCING               : NONE")
print("SECOND AUTOREGRESSIVE FORWARD : NONE")
print("INTERMEDIATE TEXT             : NONE")
print("QUERY-TIME B FORWARDS         : NONE")
print("FROZEN ADDRESS                :",FROZEN)
print("-"*174)
print(f"DISCOVERY · R1                : {DISC_FROZEN['R1']:.4f}")
print(f"DISCOVERY · R5                : {DISC_FROZEN['R5']:.4f}")
print(f"DISCOVERY · MRR               : {DISC_FROZEN['MRR']:.6f}")
print(f"FROZEN EVAL · R1              : {EVAL['R1']:.4f}")
print(f"FROZEN EVAL · R5              : {EVAL['R5']:.4f}")
print(f"FROZEN EVAL · R16             : {EVAL['R16']:.4f}")
print(f"FROZEN EVAL · MRR             : {EVAL['MRR']:.6f}")
print(f"FROZEN EVAL · MEDIAN          : {EVAL['MED']:.1f}")
print(f"FROZEN EVAL · POS MARGIN      : {np.mean(margins>0):.4f}")
if COLL:print(f"COLLISION-GROUP · R1          : {COLL['R1']:.4f}")
if UNIQ:print(f"UNIQUE-FIRST · R1             : {UNIQ['R1']:.4f}")
print(f"NULL · R1 MEAN                : {np.mean(NULL_R1):.4f}")
print(f"NULL · MRR MEAN               : {np.mean(NULL_MRR):.6f}")
print("-"*174)
print("TEST508 LOCK SHA              :",TEST508_LOCK_SHA)
print("TEST509 LOCK SHA              :",TEST509_LOCK_SHA)
print("TEST510 LOCK SHA              :",TEST510_LOCK_SHA)
print("TEST511 LOCK SHA              :",TEST511_LOCK_SHA)
print("TEST512 LOCK SHA              :",TEST512_LOCK_SHA)
print("TEST513 LOCK SHA              :",LOCK_SHA)
print(f"TOTAL TEST TIME               : {time.perf_counter()-T0:.2f}s")
print("EXPLORATORY VERDICT           :",VERDICT)
print("="*174)

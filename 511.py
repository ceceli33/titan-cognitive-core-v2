# Copyright © 2026 Mustafa Akbaş
# New original AKBASCORE MAM material is source-available under the repository LICENSE.
# See repository LICENSE for complete terms.
#
# AKBASCORE MAM — TEST511
# PRE-DECODE NEXT-KEY TRAJECTORY X-RAY
# ORACLE A → PRE-ANSWER LAYERWISE STATE → FROZEN LM-HEAD SEAL READOUT
#
# PURPOSE
# -------
# TEST509:
#   pre-answer Q trace → B K-address header
#   direct address trace NOT observed.
#
# TEST510:
#   pre-answer latent state → independently encoded Seal reference
#   next-key trace NOT localized by static cosine matching.
#
# TEST511 asks a different question:
#
#   Before the first answer token is generated, does the frozen model's
#   own vocabulary readout already contain the correct next key (Seal)?
#
# QUERY(Entity)
#      ↓
# ORACLE MEMORY A
#      ↓
# one frozen-model forward
#      ↓
# NO answer token generated
#      ↓
# layerwise final-question-token residual state
#      ↓
# model final RMSNorm
#      ↓
# frozen lm_head
#      ↓
# score exact Seal token sequence
#
# IMPORTANT
# ---------
# No Seal text is generated or fed back.
# No B cartridge is queried.
# No candidate cartridge forward.
# No YES/NO scan.
# No learned probe.
# No trained classifier.
# No cosine reference matching.
# No DRA / steering.
# No VTOKEN.
# No ANN.
# No training.
#
# MULTI-TOKEN SEALS
# -----------------
# A Seal may tokenize into >1 token.
# Therefore TEST511 has two measurements:
#
# FIRST:
#   layerwise score of the FIRST Seal token from the pre-answer state.
#
# FULL:
#   exact teacher-forced Seal sequence log-probability under the frozen model.
#   This is a diagnostic readout only. Gold Seal tokens are used only AFTER
#   the first-token state for scoring continuation likelihood; they are NEVER
#   used to alter memory, choose a channel, retrieve B, or feed a later memory.
#
# Primary mechanistic localization = FIRST.
# FULL is reported separately as a sequence-readability control.
#
# DISCOVERY / FROZEN EVALUATION
# -----------------------------
# Items 01–16: discovery.
# Items 17–32: frozen evaluation.
#
# Candidate FIRST channels:
#   residual states entering layers 0..27 + final model output state L28.
#
# Discovery selects exactly one depth.
# Evaluation cannot reselect it.
#
# DISTRACTOR STANDARD
# -------------------
# All candidate Seals originate from real, grammatically valid, model-processed
# memory records. No random placeholders, keyboard noise, malformed strings,
# or cheap "broken data" negatives are permitted.

import os,sys,subprocess,importlib.util,random,time,hashlib,json,re,math
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache

TEST="511";SEED=511;BASE_SEED=501;MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
N_ITEMS=32;N_DISC=16;N_EVAL=16
DEVICE=torch.device("cuda");FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
TEST508_LOCK_SHA="47512c7860268517670ca5b45b6e76fe1582362dd38e447d164f33c756a1a2ca"
TEST509_LOCK_SHA="4b5b78841ebf0d9b458306136e122fd3108b33bdfd89924585f891f748eca0d8"
TEST510_LOCK_SHA="cf3aaf955c0cd5019b1b951d185f290528286df82b23da62aaa30657e6bad5ea"

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
        p1=i%3;p2=(i+1)%3
        A=" ".join(reorder(r1,g,p1,BASE_SEED+i*101+17))
        B=" ".join(reorder(r2,g,p2,BASE_SEED+i*101+34))
        items.append({"id":i+1,"A":A,"B":B,"gold":{"entity":ents[g],"seal":ss[g],"class":cc[g]}})
    return items

ITEMS=make_base()
USED_SEALS=sorted({it["gold"]["seal"] for it in ITEMS})
if len(USED_SEALS)!=N_ITEMS:raise RuntimeError("Gold Seal uniqueness failure.")
SEAL_INDEX={s:i for i,s in enumerate(USED_SEALS)}

LOCK={
"test":TEST,"parent_test510":TEST510_LOCK_SHA,"parent_test509":TEST509_LOCK_SHA,"parent_test508":TEST508_LOCK_SHA,
"model":MODEL_ID,"seed":SEED,"base_seed":BASE_SEED,"items":ITEMS,
"n_items":N_ITEMS,"discovery_items":"01-16","evaluation_items":"17-32",
"task":"oracle_A_predecode_layerwise_frozen_lm_head_next_key_xray",
"candidate_seals":USED_SEALS,
"primary":"FIRST_SEAL_TOKEN_LOGIT",
"secondary":"FULL_SEAL_TEACHER_FORCED_LOGPROB",
"candidate_depths":"hidden_states_0_through_28",
"channel_selection":"DISCOVERY_ONLY_THEN_FROZEN_EVAL",
"answer_generation":"OFF","text_trace_reinsertion":"OFF","B_search":"OFF",
"per_candidate_model_forward":"OFF","cosine_reference_matching":"OFF",
"compression":"OFF","steering":"OFF","vtoken":"OFF","ann":"OFF",
"learned_probe":"OFF","training":"OFF","posthoc_eval_selection":"OFF",
"distractor_standard":"MODEL_NATIVE_GRAMMATICAL_MEANINGFUL_NO_PLACEHOLDER_NO_NOISE"
}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

print("="*174)
print("TEST511 — AKBASCORE MAM · PRE-DECODE NEXT-KEY TRAJECTORY X-RAY")
print("ORACLE A → PRE-ANSWER LAYERWISE STATE → FROZEN LM-HEAD SEAL READOUT")
print("="*174)
print("LOCK SHA:",LOCK_SHA)
print("PARENT TEST510 LOCK:",TEST510_LOCK_SHA)
print("PARENT TEST509 LOCK:",TEST509_LOCK_SHA)
print("PARENT TEST508 LOCK:",TEST508_LOCK_SHA)
print("ITEMS:",N_ITEMS,"| DISCOVERY:",N_DISC,"| FROZEN EVAL:",N_EVAL,"| SEAL CANDIDATES:",len(USED_SEALS))
print("ANSWER TOKEN GENERATION: NONE")
print("TEXT TRACE REINSERTION: NONE")
print("B-CARTRIDGE SEARCH: NONE")
print("CANDIDATE MODEL FORWARDS: NONE")
print("PRIMARY: FIRST SEAL TOKEN LOGIT")
print("SECONDARY: FULL SEAL TEACHER-FORCED LOG-PROBABILITY")
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
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
if PAD is None:raise RuntimeError("No PAD/EOS token.")
MODEL_DTYPE=model.lm_head.weight.dtype
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | {NL}L | H={H} | QH={QH} KVH={KVH} HD={HD} | {MODEL_DTYPE}")

@torch.inference_mode()
def kv_from_ids(ids):
    x=torch.tensor([ids],device=DEVICE,dtype=torch.long)
    o=model(input_ids=x,use_cache=False,output_hidden_states=True,return_dict=True)
    if len(o.hidden_states)!=NL+1:raise RuntimeError("Hidden-state count mismatch.")
    out=[]
    for L,layer in enumerate(model.model.layers):
        h=o.hidden_states[L][0].to(layer.input_layernorm.weight.device,layer.input_layernorm.weight.dtype)
        hn=layer.input_layernorm(h)
        k=layer.self_attn.k_proj(hn.to(layer.self_attn.k_proj.weight.dtype)).view(-1,KVH,HD).transpose(0,1).contiguous()
        v=layer.self_attn.v_proj(hn.to(layer.self_attn.v_proj.weight.dtype)).view(-1,KVH,HD).transpose(0,1).contiguous()
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
    if len(raw)!=NL:raise RuntimeError("RAW layer count mismatch.")
    T=raw[0][0].shape[1];pos=list(range(T));out=[]
    for L in range(NL):
        k,v=raw[L]
        if k.shape[1]!=T or v.shape[1]!=T:raise RuntimeError(f"Packet length mismatch L{L}.")
        out.append((rope_k(k,pos,L),v))
    return out,T,T

def make_cache(installed):
    try:cache=DynamicCache(config=cfg)
    except TypeError:cache=DynamicCache()
    for L,(k,v) in enumerate(installed):cache.update(k.unsqueeze(0),v.unsqueeze(0),L)
    return cache

def q_read_a(e):return f"What seal does instrument {e} carry? Give only the exact seal."

print("\n[2/8] Tokenizing candidate Seals...")
# Exact answer continuation is the bare Seal, because the prompt ends at ANSWER:
SEAL_TOKENS={}
for s in USED_SEALS:
    ids=enc(s)
    if not ids:raise RuntimeError(f"Empty Seal tokenization: {s}")
    SEAL_TOKENS[s]=ids
lens=[len(SEAL_TOKENS[s]) for s in USED_SEALS]
print("      Seal token lengths:",dict(sorted({n:lens.count(n) for n in set(lens)}.items())))
for s in USED_SEALS[:8]:print(f"      {s}: {SEAL_TOKENS[s]} → {tok.convert_ids_to_tokens(SEAL_TOKENS[s])}")

# FIRST-token collision groups matter mechanistically.
FIRST_TOKEN={s:SEAL_TOKENS[s][0] for s in USED_SEALS}
groups={}
for s,t in FIRST_TOKEN.items():groups.setdefault(t,[]).append(s)
collisions={t:ss for t,ss in groups.items() if len(ss)>1}
print("      Unique first-token IDs:",len(groups),"/",len(USED_SEALS))
print("      First-token collision groups:",len(collisions))
if collisions:
    for t,ss in list(collisions.items())[:8]:print("       ",t,tok.convert_ids_to_tokens([t]),"→",ss)

print("\n[3/8] Pre-forging oracle A memories...")
ADATA=[]
for z,it in enumerate(ITEMS):
    raw=forge(it["A"]);ADATA.append(install_single(raw))
    if z<6:
        g=it["gold"];print(f"      ITEM {it['id']:02d} {g['entity']:6s} | A contains {g['entity']}→{g['seal']}")
print("      Same TEST509/510 RAW A-memory construction retained.")

@torch.inference_mode()
def preanswer_states(installed,Tm,P,q):
    cache=make_cache(installed);qids=enc(FMT.format(q=q))
    if not qids:raise RuntimeError("Empty query.")
    nq=len(qids)
    pos=torch.arange(P,P+nq,device=DEVICE,dtype=torch.long).unsqueeze(0)
    mask=torch.ones((1,Tm+nq),device=DEVICE,dtype=torch.long)
    o=model(input_ids=torch.tensor([qids],device=DEVICE,dtype=torch.long),
            past_key_values=cache,attention_mask=mask,position_ids=pos,
            use_cache=False,output_hidden_states=True,return_dict=True)
    if len(o.hidden_states)!=NL+1:raise RuntimeError("Pre-answer hidden-state count mismatch.")
    return [h[0,-1].detach().cpu() for h in o.hidden_states]

# Logit-lens style frozen model readout.
# For intermediate residual states, apply the model's final RMSNorm and tied/frozen LM head.
@torch.inference_mode()
def lens_logits(h):
    fn=model.model.norm
    x=h.to(device=fn.weight.device,dtype=fn.weight.dtype)
    x=fn(x)
    x=x.to(device=model.lm_head.weight.device,dtype=model.lm_head.weight.dtype)
    return model.lm_head(x).float().cpu()

def first_scores_from_state(h):
    logits=lens_logits(h)
    return [float(logits[FIRST_TOKEN[s]].item()) for s in USED_SEALS]

def ranks_from_scores(scores,gold_idx):
    if len(scores)!=len(USED_SEALS):raise RuntimeError("Score length mismatch.")
    if not np.all(np.isfinite(scores)):raise RuntimeError("Non-finite score.")
    order=sorted(range(len(scores)),key=lambda j:(-scores[j],j))
    return order.index(gold_idx)+1,order

def rank_stats(x):
    a=np.asarray(x,dtype=float)
    return {"R1":float(np.mean(a==1)),"R5":float(np.mean(a<=5)),
            "R16":float(np.mean(a<=16)),"MRR":float(np.mean(1/a)),
            "MED":float(np.median(a))}

print("\n[4/8] Capturing pre-answer trajectory...")
STATES=[]
for z,it in enumerate(ITEMS):
    st=preanswer_states(*ADATA[z],q_read_a(it["gold"]["entity"]))
    if len(st)!=NL+1:raise RuntimeError(f"State depth mismatch item {z+1}.")
    STATES.append(st);ADATA[z]=None
    if z<6:print(f"      ITEM {it['id']:02d} {it['gold']['entity']:6s} | {NL+1} residual depths captured | answer generated: NO")
print("      Depth L00 = embedding/pre-layer-0 state.")
print("      Depth L28 = final decoder residual before model final RMSNorm.")

print("\n[5/8] DISCOVERY — FIRST Seal-token readout, items 01–16...")
DISC={};DISC_SCORE_BANKS={}
for depth in range(NL+1):
    ranks=[];banks=[]
    for z in range(N_DISC):
        scores=first_scores_from_state(STATES[z][depth]);banks.append(scores)
        gi=SEAL_INDEX[ITEMS[z]["gold"]["seal"]]
        r,_=ranks_from_scores(scores,gi);ranks.append(r)
    DISC[depth]=rank_stats(ranks);DISC_SCORE_BANKS[depth]=banks

ordered=sorted(DISC.items(),key=lambda x:(x[1]["R1"],x[1]["MRR"],x[1]["R5"],-x[0]),reverse=True)
for n,(d,s) in enumerate(ordered[:15],1):
    print(f"      #{n:02d} L{d:02d} | R1={s['R1']:.4f} R5={s['R5']:.4f} R16={s['R16']:.4f} MRR={s['MRR']:.6f} median={s['MED']:.1f}")
FROZEN_DEPTH,BEST_DISC=ordered[0]
print(f"\n      FROZEN DEPTH → L{FROZEN_DEPTH:02d}")
print("      Evaluation items have NOT participated in depth selection.")

print("\n[6/8] FROZEN EVALUATION — FIRST Seal-token readout...")
EVAL_RANKS=[];EVAL_BANKS=[];GOLD_SCORES=[];TOP_WRONG=[]
for z in range(N_DISC,N_ITEMS):
    scores=first_scores_from_state(STATES[z][FROZEN_DEPTH])
    gi=SEAL_INDEX[ITEMS[z]["gold"]["seal"]]
    r,order=ranks_from_scores(scores,gi)
    pred=USED_SEALS[order[0]]
    EVAL_RANKS.append(r);EVAL_BANKS.append(scores)
    GOLD_SCORES.append(scores[gi])
    wrong=max(scores[j] for j in range(len(scores)) if j!=gi);TOP_WRONG.append(wrong)
    g=ITEMS[z]["gold"]
    print(f"      [{z+1:02d}/{N_ITEMS}] {g['entity']:6s} | gold={g['seal']} pred={pred} rank={r:2d} | gold={scores[gi]:+.4f} topWrong={wrong:+.4f} margin={scores[gi]-wrong:+.4f}")

EVAL=rank_stats(EVAL_RANKS);DISC_FROZEN=DISC[FROZEN_DEPTH]
margins=np.asarray(GOLD_SCORES)-np.asarray(TOP_WRONG)

# -------------------------------------------------------------------------
# SECONDARY CONTROL: FULL SEAL SEQUENCE READABILITY
# -------------------------------------------------------------------------
# This does NOT select the frozen layer.
# It measures whether the frozen model can score the full Seal answer sequence
# when the candidate sequence is teacher-forced after the query.
#
# Candidate scoring is done in a batched forward per ITEM, not one forward per
# candidate. This is a diagnostic sequence-readability control, not retrieval.
# -------------------------------------------------------------------------
@torch.inference_mode()
def full_seal_scores(installed,Tm,P,q):
    qids=enc(FMT.format(q=q))
    # Build one branch per candidate Seal from identical A cache.
    # We score log P(seal_tokens | A, query), including first token from query state.
    # To avoid modifying/querying B, only A cache is used.
    base_cache=make_cache(installed)

    # First token score comes from actual final model output for the query.
    nq=len(qids);pos=torch.arange(P,P+nq,device=DEVICE,dtype=torch.long).unsqueeze(0)
    mask=torch.ones((1,Tm+nq),device=DEVICE,dtype=torch.long)
    o=model(input_ids=torch.tensor([qids],device=DEVICE,dtype=torch.long),
            past_key_values=base_cache,attention_mask=mask,position_ids=pos,
            use_cache=True,output_hidden_states=False,return_dict=True)
    first_lp=torch.log_softmax(o.logits[0,-1].float(),dim=-1)
    scores=[]
    # Continuation length is tiny; candidates are diagnostic only.
    for s in USED_SEALS:
        ids=SEAL_TOKENS[s]
        sc=float(first_lp[ids[0]].item())
        if len(ids)>1:
            # Fresh A cache prevents candidate branches contaminating one another.
            c=make_cache(installed)
            prefix=qids+ids[:-1]
            pp=torch.arange(P,P+len(prefix),device=DEVICE,dtype=torch.long).unsqueeze(0)
            mm=torch.ones((1,Tm+len(prefix)),device=DEVICE,dtype=torch.long)
            oo=model(input_ids=torch.tensor([prefix],device=DEVICE,dtype=torch.long),
                     past_key_values=c,attention_mask=mm,position_ids=pp,
                     use_cache=False,return_dict=True)
            # logits after query predicts ids[0]; subsequent positions predict ids[1:].
            start=len(qids)-1
            lp=torch.log_softmax(oo.logits[0].float(),dim=-1)
            sc=0.0
            for j,t in enumerate(ids):
                sc+=float(lp[start+j,t].item())
        scores.append(sc/len(ids))
    return scores

# Re-forge A only for the secondary diagnostic; no B memory exists here.
print("\n      FULL-SEQUENCE readability control...")
FULL_RANKS=[]
for z in range(N_DISC,N_ITEMS):
    raw=forge(ITEMS[z]["A"]);inst=install_single(raw)
    scores=full_seal_scores(*inst,q_read_a(ITEMS[z]["gold"]["entity"]))
    gi=SEAL_INDEX[ITEMS[z]["gold"]["seal"]]
    r,_=ranks_from_scores(scores,gi);FULL_RANKS.append(r)
FULL=rank_stats(FULL_RANKS)
print(f"      FULL EVAL R1={FULL['R1']:.4f} R5={FULL['R5']:.4f} R16={FULL['R16']:.4f} MRR={FULL['MRR']:.6f} median={FULL['MED']:.1f}")

print("\n[7/8] Frozen statistics and null...")
rng=np.random.default_rng(SEED);NULL_R1=[];NULL_MRR=[]
for _ in range(10000):
    rr=[]
    for scores in EVAL_BANKS:
        fake=int(rng.integers(0,len(USED_SEALS)))
        r,_=ranks_from_scores(scores,fake);rr.append(r)
    rr=np.asarray(rr,dtype=float)
    NULL_R1.append(float(np.mean(rr==1)));NULL_MRR.append(float(np.mean(1/rr)))

print(f"      FROZEN DEPTH         : L{FROZEN_DEPTH:02d}")
print(f"      DISCOVERY R1         : {DISC_FROZEN['R1']:.4f}")
print(f"      DISCOVERY R5         : {DISC_FROZEN['R5']:.4f}")
print(f"      DISCOVERY MRR        : {DISC_FROZEN['MRR']:.6f}")
print(f"      EVAL FIRST R1        : {EVAL['R1']:.4f}")
print(f"      EVAL FIRST R5        : {EVAL['R5']:.4f}")
print(f"      EVAL FIRST R16       : {EVAL['R16']:.4f}")
print(f"      EVAL FIRST MRR       : {EVAL['MRR']:.6f}")
print(f"      EVAL FIRST MEDIAN    : {EVAL['MED']:.1f}")
print(f"      FIRST POS MARGIN     : {np.mean(margins>0):.4f}")
print(f"      FULL SEQ R1          : {FULL['R1']:.4f}")
print(f"      FULL SEQ R5          : {FULL['R5']:.4f}")
print(f"      FULL SEQ MRR         : {FULL['MRR']:.6f}")
print(f"      NULL R1 mean / P99   : {np.mean(NULL_R1):.4f} / {np.quantile(NULL_R1,.99):.4f}")
print(f"      NULL MRR mean / P99  : {np.mean(NULL_MRR):.6f} / {np.quantile(NULL_MRR,.99):.6f}")

if EVAL["R1"]>=.75 and EVAL["MRR"]>=.82 and np.mean(margins>0)>=.75:
    VERDICT="PREDECODE_NEXT_KEY_STRONGLY_PRESENT_IN_FROZEN_READOUT"
elif EVAL["R1"]>=.50 and EVAL["MRR"]>=.65:
    VERDICT="PREDECODE_NEXT_KEY_PRESENT_IN_FROZEN_READOUT"
elif EVAL["R5"]>=.75 and EVAL["MRR"]>=.45:
    VERDICT="PREDECODE_NEXT_KEY_SIGNAL_PRESENT_BUT_DIFFUSE"
elif FULL["R1"]>=.75 and EVAL["R1"]<.50:
    VERDICT="NEXT_KEY_SEQUENCE_READABLE_BUT_NOT_LOCALIZED_BEFORE_DECODE"
else:
    VERDICT="PREDECODE_NEXT_KEY_READOUT_NOT_YET_OBSERVED"

print("\n[8/8] Mechanistic classification...")
print("      A address                           : ORACLE")
print("      Answer tokens generated             : 0")
print("      Seal text fed to B                  : 0")
print("      B cartridges queried                : 0")
print("      Learned probes                      : 0")
print("      Primary channel selection           : DISCOVERY ONLY")
print("      Frozen evaluation reselection       : NO")
print("      Primary readout                     : frozen final norm + lm_head")
print("      Secondary full-sequence control     : teacher-forced diagnostic")
print("      Distractors                         : meaningful model-native records only")
print("      DRA / steering                      : NONE")
print("      Training                            : NONE")

print("\n"+"="*174)
print("TEST511 FINAL RESULT — AKBASCORE MAM · PRE-DECODE NEXT-KEY TRAJECTORY X-RAY")
print("="*174)
print("MODEL                         : Qwen/Qwen2.5-7B-Instruct · frozen")
print("PARENT                        : TEST510")
print("PARENT TEST510 LOCK           :",TEST510_LOCK_SHA)
print("PARENT TEST509 LOCK           :",TEST509_LOCK_SHA)
print("PARENT TEST508 LOCK           :",TEST508_LOCK_SHA)
print("TASK                          : Oracle A → pre-decode trajectory → Seal readout")
print("SEAL CANDIDATES               :",len(USED_SEALS))
print("A ADDRESS                     : ORACLE")
print("ANSWER GENERATION             : NONE")
print("TEXT TRACE REINSERTION        : NONE")
print("B-CARTRIDGE SEARCH            : NONE")
print("LEARNED PROBE                 : NONE")
print("COSINE REFERENCE MATCHING     : NONE")
print("PRIMARY                       : FIRST Seal-token frozen LM-head readout")
print("SECONDARY                     : FULL Seal teacher-forced sequence score")
print("DISCOVERY                     : items 01–16")
print("FROZEN EVALUATION             : items 17–32")
print("FROZEN DEPTH                  :",f"L{FROZEN_DEPTH:02d}")
print("-"*174)
print(f"DISCOVERY · R1                : {DISC_FROZEN['R1']:.4f}")
print(f"DISCOVERY · R5                : {DISC_FROZEN['R5']:.4f}")
print(f"DISCOVERY · MRR               : {DISC_FROZEN['MRR']:.6f}")
print(f"FROZEN EVAL · FIRST R1        : {EVAL['R1']:.4f}")
print(f"FROZEN EVAL · FIRST R5        : {EVAL['R5']:.4f}")
print(f"FROZEN EVAL · FIRST R16       : {EVAL['R16']:.4f}")
print(f"FROZEN EVAL · FIRST MRR       : {EVAL['MRR']:.6f}")
print(f"FROZEN EVAL · FIRST MEDIAN    : {EVAL['MED']:.1f}")
print(f"FROZEN EVAL · POS MARGIN      : {np.mean(margins>0):.4f}")
print(f"FROZEN EVAL · FULL R1         : {FULL['R1']:.4f}")
print(f"FROZEN EVAL · FULL R5         : {FULL['R5']:.4f}")
print(f"FROZEN EVAL · FULL MRR        : {FULL['MRR']:.6f}")
print(f"NULL · R1 MEAN                : {np.mean(NULL_R1):.4f}")
print(f"NULL · MRR MEAN               : {np.mean(NULL_MRR):.6f}")
print("-"*174)
print("TEST508 LOCK SHA              :",TEST508_LOCK_SHA)
print("TEST509 LOCK SHA              :",TEST509_LOCK_SHA)
print("TEST510 LOCK SHA              :",TEST510_LOCK_SHA)
print("TEST511 LOCK SHA              :",LOCK_SHA)
print(f"TOTAL TEST TIME               : {time.perf_counter()-T0:.2f}s")
print("EXPLORATORY VERDICT           :",VERDICT)
print("="*174)

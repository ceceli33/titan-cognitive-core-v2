# Copyright © 2026 Mustafa Akbaş
# New original AKBASCORE MAM material is source-available under the repository LICENSE.
# See repository LICENSE for complete terms.
#
# AKBASCORE MAM — TEST506
# 32-CARTRIDGE CONTENT-ADDRESSED RELAY · STRICT vs MODEL-NATIVE RELEVANCE RANKING
#
# 32 independently forged RAW BELLEKÖZ per item:
#   2 required memories + 30 independent distractors.
#
# Required cartridge addresses are hidden from all CONTENT arms.
#
# Arms:
#   DIRECT       = all 32 independent memories, one-shot
#   ORACLE_ADDR  = correct A/B addresses; isolates relay/read capacity
#   STRICT       = TEST505 rule: exhaustive generated NONE/non-NONE scan
#   RANKED       = exhaustive O(N) model-native YES-vs-NO first-token logit ranking
#
# RANKED traversal:
#   QUERY(entity)
#      → rank 32 memories for entity relevance
#      → select A
#      → decode model-produced seal trace
#      → rank remaining 31 memories for that trace
#      → select B
#      → decode final class
#
# Negative control:
#   WRONG_TRACE = replace Stage-1 trace with a deterministic wrong seal before Stage-2 ranking.
#
# IMPORTANT:
#   O(N) exhaustive scan remains experimental addressing, not final scalable routing.
#   Intermediate trace remains textual, not latent BELLEKBAĞ.
#
# No compression. No joint re-forging. No DRA/steering.
# No VTOKEN. No ANN/learned router. No graph. No training. No post-hoc selection.

import os,sys,subprocess,importlib.util,random,time,hashlib,json,re
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache

TEST="506";SEED=506;MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N_ITEMS=32;BANK_N=32;N_DIST=30;MAX_NEW=12
DEVICE=torch.device("cuda");FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
TEST501_LOCK_SHA="1a3e6ea2a9f6c72d7b8dacb0c0844e9f2a6f8595bb232f1554eb41af7a20dc41"
TEST502_LOCK_SHA="f852b6efc6ee887bae54db08cbc11f987e749499a51cebeb5132f569e74edb89"
TEST503_LOCK_SHA="2cb70ffa53a98efb249590dba7ef184b9e41cf72a99291bda65e6c93b82cb5de"
TEST504_LOCK_SHA="74c375373666c840105cc165b90977c56c8698cf3d5617c63f619a0139386846"
TEST505_LOCK_SHA="6d093a5d082b43ff56f09bb94cfed1839c0bd0d76a967822937996a5d1948e43"

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

assert len(ENTITY_BANK)==N_ITEMS
assert len(SEAL_BANK)>=N_ITEMS*3 and len(CLASS_BANK)>=N_ITEMS*3
assert len(set(SEAL_BANK))==len(SEAL_BANK)
assert len(set(CLASS_BANK))==len(CLASS_BANK)

os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
torch.set_grad_enabled(False)
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

def token_present(text,token):
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(str(token))}(?![A-Za-z0-9_])",text,re.I) is not None

def reorder(rows,gold,target_pos,seed):
    g=rows[gold];rest=[x for j,x in enumerate(rows) if j!=gold]
    rr=random.Random(seed);rr.shuffle(rest);out=rest[:];out.insert(target_pos,g);return out

def make_base():
    rng=random.Random(501);seals=SEAL_BANK.copy();classes=CLASS_BANK.copy()
    rng.shuffle(seals);rng.shuffle(classes);items=[]
    for i in range(N_ITEMS):
        ents=list(ENTITY_BANK[i]);g=i%3;ss=seals[3*i:3*i+3];cc=classes[3*i:3*i+3]
        r1=[f"Instrument {ents[j]} carries seal {ss[j]}." for j in range(3)]
        r2=[f"Seal {ss[j]} corresponds to routing class {cc[j]}." for j in range(3)]
        p1=i%3;p2=(i+1)%3
        rec1=" ".join(reorder(r1,g,p1,501+i*101+17))
        rec2=" ".join(reorder(r2,g,p2,501+i*101+34))
        items.append({"id":i+1,"records":[rec1,rec2],"gold":{"entity":ents[g],"seal":ss[g],"class":cc[g]},"positions":[p1,p2]})
    return items

BASE=make_base()

# 15 entity→seal distractors + 15 seal→class distractors.
# Three relations per cartridge preserve the required-memory surface format.
def make_distractors(i,g):
    rng=random.Random(SEED+90000+i)
    ep=[e for j,t in enumerate(ENTITY_BANK) if j!=i for e in t if e.lower()!=g["entity"].lower()]
    sp=[s for s in SEAL_BANK if s!=g["seal"] and s!=g["class"]]
    cp=[c for c in CLASS_BANK if c!=g["class"] and c!=g["seal"]]
    if len(ep)<45 or len(sp)<45 or len(cp)<45:raise RuntimeError(f"Insufficient distractor pool at item {i+1}.")
    rng.shuffle(ep);rng.shuffle(sp);rng.shuffle(cp)
    ds=[]
    # Pools may be reused across the two distractor families, but never contain current gold symbols.
    for d in range(15):
        es=[ep[(d*3+j)%len(ep)] for j in range(3)]
        ss=[sp[(d*3+j)%len(sp)] for j in range(3)]
        ds.append(" ".join(f"Instrument {es[j]} carries seal {ss[j]}." for j in range(3)))
    rng2=random.Random(SEED+120000+i);sp2=sp.copy();cp2=cp.copy();rng2.shuffle(sp2);rng2.shuffle(cp2)
    for d in range(15):
        ss=[sp2[(d*3+j)%len(sp2)] for j in range(3)]
        cc=[cp2[(d*3+j)%len(cp2)] for j in range(3)]
        ds.append(" ".join(f"Seal {ss[j]} corresponds to routing class {cc[j]}." for j in range(3)))
    if len(ds)!=N_DIST:raise RuntimeError(f"Distractor count failure item {i+1}.")
    for x in ds:
        if token_present(x,g["entity"]):raise RuntimeError(f"Target entity contamination item {i+1}: {x}")
        if token_present(x,g["seal"]):raise RuntimeError(f"Gold seal contamination item {i+1}: {x}")
        if token_present(x,g["class"]):raise RuntimeError(f"Gold class contamination item {i+1}: {x}")
    return ds

ITEMS=[]
for i,b in enumerate(BASE):
    g=b["gold"];ds=make_distractors(i,g)
    mems=[{"role":"A","text":b["records"][0]},{"role":"B","text":b["records"][1]}]+[{"role":"D","text":x} for x in ds]
    rng=random.Random(SEED+70000+i);rng.shuffle(mems)
    if len(mems)!=BANK_N:raise RuntimeError(f"Bank size failure item {i+1}.")
    if sum(m["role"]=="A" for m in mems)!=1 or sum(m["role"]=="B" for m in mems)!=1 or sum(m["role"]=="D" for m in mems)!=N_DIST:
        raise RuntimeError(f"Role-count failure item {i+1}.")
    ia=next(j for j,m in enumerate(mems) if m["role"]=="A")
    ib=next(j for j,m in enumerate(mems) if m["role"]=="B")
    ent_hits=[j for j,m in enumerate(mems) if token_present(m["text"],g["entity"])]
    seal_hits=[j for j,m in enumerate(mems) if token_present(m["text"],g["seal"])]
    class_hits=[j for j,m in enumerate(mems) if token_present(m["text"],g["class"])]
    if ent_hits!=[ia]:raise RuntimeError(f"Entity uniqueness failure item {i+1}: {ent_hits}/{ia}")
    if sorted(seal_hits)!=sorted([ia,ib]):raise RuntimeError(f"Bridge uniqueness failure item {i+1}: {seal_hits}/{[ia,ib]}")
    if class_hits!=[ib]:raise RuntimeError(f"Final-class uniqueness failure item {i+1}: {class_hits}/{ib}")
    ITEMS.append({**b,"bank":mems,"oracle":[ia,ib]})

LOCK={"test":TEST,"model":MODEL_ID,"seed":SEED,"base_panel_seed":501,"items":ITEMS,"bank_n":BANK_N,
      "required":2,"distractors":N_DIST,"test501":TEST501_LOCK_SHA,"test502":TEST502_LOCK_SHA,
      "test503":TEST503_LOCK_SHA,"test504":TEST504_LOCK_SHA,"test505":TEST505_LOCK_SHA,
      "arms":["DIRECT_32","ORACLE_ADDRESS_RELAY","STRICT_EXHAUSTIVE","RANKED_LOGIT_RELAY","WRONG_TRACE_CONTROL"],
      "rank_signal":"first-answer-token logit(YES)-logit(NO)","stage2_excludes_stage1":True,
      "compression":"OFF","joint_reforge":"OFF","steering":"OFF","vtoken":"OFF","ann_router":"OFF",
      "learned_router":"OFF","graph":"OFF","training":"OFF","posthoc":"OFF"}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

print("="*166)
print("TEST506 — AKBASCORE MAM · 32-CARTRIDGE CONTENT-ADDRESSED RELAY")
print("STRICT TEST505 REPLICATION × MODEL-NATIVE YES/NO LOGIT RANKING × WRONG-TRACE CONTROL")
print("="*166)
print("LOCK SHA:",LOCK_SHA)
print("TEST505 LOCK:",TEST505_LOCK_SHA)
print("ITEMS:",N_ITEMS,"| BANK:",BANK_N,"| REQUIRED: 2 | DISTRACTORS:",N_DIST)
print("CONTENT ADDRESSING: exhaustive O(N); gold cartridge addresses hidden from CONTENT arms")
print("COMPRESSION / JOINT RE-FORGE / DRA / VTOKEN / ANN / LEARNED ROUTER / GRAPH / TRAINING: NONE")
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

# We score the exact first generation token for " YES" vs " NO".
YES_IDS=enc(" YES");NO_IDS=enc(" NO")
if len(YES_IDS)!=1 or len(NO_IDS)!=1:
    raise RuntimeError(f"YES/NO must each be one tokenizer token; got YES={YES_IDS}, NO={NO_IDS}")
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
def norm(x):
    s=firstline(x).strip().upper()
    m=re.match(r"^\s*(NONE|[A-Z]{3})(?=$|[\s\.,;:!?])",s)
    return m.group(1) if m else s.rstrip(".")
def exact(x,t):return norm(x)==str(t).upper()

def q_direct(entity):
    return f"What routing class corresponds to instrument {entity}? Follow the available memories and give only the exact routing class."
def q_find_entity(entity):
    return f"If this memory explicitly contains instrument {entity}, give only the exact seal carried by {entity}. Otherwise give only NONE."
def q_find_seal(seal):
    return f"If this memory explicitly contains seal {seal} and its routing-class relation, give only the exact routing class corresponding to {seal}. Otherwise give only NONE."
def q_rel_entity(entity):
    return f"Does this memory explicitly contain a relation stating which seal instrument {entity} carries? Answer only YES or NO."
def q_rel_seal(seal):
    return f"Does this memory explicitly contain a relation stating which routing class seal {seal} corresponds to? Answer only YES or NO."
def q_oracle_a(entity):
    return f"What seal does instrument {entity} carry? Give only the exact seal."
def q_oracle_b(seal):
    return f"What routing class corresponds to seal {seal}? Give only the exact routing class."

def strict_scan(installed,q,excluded=None):
    excluded=set() if excluded is None else set(excluded);claims=[];outs=[]
    for j,(inst,Tm,P) in enumerate(installed):
        if j in excluded:outs.append("SKIP");continue
        v=norm(read_kv(inst,Tm,P,q));outs.append(v)
        if v!="NONE":claims.append((j,v))
    if len(claims)==1:return claims[0][0],claims[0][1],claims,outs
    return None,None,claims,outs

def ranked_select(installed,q,excluded=None):
    excluded=set() if excluded is None else set(excluded);scores=[]
    for j,(inst,Tm,P) in enumerate(installed):
        if j in excluded:continue
        scores.append((yesno_margin(inst,Tm,P,q),j))
    scores.sort(key=lambda x:(-x[0],x[1]))
    return scores[0][1],scores[0][0],scores

def rank_of(scores,gold_idx):
    for r,(_,j) in enumerate(scores,1):
        if j==gold_idx:return r
    return None

def bootstrap_delta(a,b,B=10000,seed=506):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64);d=a-b;n=len(d)
    rng=np.random.default_rng(seed);vals=np.empty(B)
    for i in range(B):vals[i]=d[rng.integers(0,n,n)].mean()
    return float(d.mean()),tuple(np.quantile(vals,[.025,.975]))

print("\n[2/7] Bank seal...")
for it in ITEMS[:6]:
    print(f"      ITEM {it['id']:02d} entity={it['gold']['entity']:6s} trace={it['gold']['seal']} target={it['gold']['class']} | hidden A=M{it['oracle'][0]+1:02d} B=M{it['oracle'][1]+1:02d}")
print("      32 independent memories/item; addresses hidden from STRICT and RANKED.")
print("      Gold entity, bridge seal and final class absent from all 30 distractors.")

R={k:[] for k in ["DIRECT","ORACLE","STRICT","RANK_A","RANK_TRACE","RANK_B","RANK_VALUE","RANK_E2E","WRONG_B","WRONG_FINAL"]}
RANKS_A=[];RANKS_B=[];MARG_A=[];MARG_B=[]

print("\n[3/7] Forging and running 32-cartridge banks...")
for z,it in enumerate(ITEMS):
    g=it["gold"];ia,ib=it["oracle"]
    raws=[forge(m["text"]) for m in it["bank"]]
    installed=[install_single(r) for r in raws]

    # DIRECT
    instD,TmD,PD=install_parallel(raws)
    direct=read_kv(instD,TmD,PD,q_direct(g["entity"]))
    direct_ok=exact(direct,g["class"])

    # ORACLE ADDRESS RELAY
    instA,TmA,PA=installed[ia];oa1=read_kv(instA,TmA,PA,q_oracle_a(g["entity"]));otr=norm(oa1)
    instB,TmB,PB=installed[ib];oa2=read_kv(instB,TmB,PB,q_oracle_b(otr))
    oracle_ok=exact(oa1,g["seal"]) and exact(oa2,g["class"])

    # STRICT TEST505-style scan
    s1_idx,s1_trace,sclaims1,_=strict_scan(installed,q_find_entity(g["entity"]))
    strict_ok=False;sclaims2=[]
    if s1_idx is not None and s1_trace is not None:
        s2_idx,s2_val,sclaims2,_=strict_scan(installed,q_find_seal(s1_trace),excluded={s1_idx})
        strict_ok=(s1_idx==ia and s1_trace==g["seal"] and s2_idx==ib and s2_val==g["class"])

    # RANKED STAGE 1: relevance only; no gold symbol is used in scoring.
    ra_idx,ra_margin,ra_scores=ranked_select(installed,q_rel_entity(g["entity"]))
    ra_rank=rank_of(ra_scores,ia);RANKS_A.append(ra_rank);MARG_A.append(ra_margin)
    rinstA,rTmA,rPA=installed[ra_idx]
    rtrace_raw=read_kv(rinstA,rTmA,rPA,q_oracle_a(g["entity"]))
    rtrace=norm(rtrace_raw)
    rank_a_ok=ra_idx==ia
    rank_trace_ok=rank_a_ok and rtrace==g["seal"]

    # RANKED STAGE 2: only model-produced Stage-1 trace goes forward.
    rb_idx=None;rb_margin=float("nan");rb_rank=None;rfinal=""
    rank_b_ok=False;rank_value_ok=False;rank_e2e=False
    if re.fullmatch(r"[A-Z]{3}",rtrace or ""):
        rb_idx,rb_margin,rb_scores=ranked_select(installed,q_rel_seal(rtrace),excluded={ra_idx})
        rb_rank=rank_of(rb_scores,ib);RANKS_B.append(rb_rank);MARG_B.append(rb_margin)
        rinstB,rTmB,rPB=installed[rb_idx]
        rfinal=read_kv(rinstB,rTmB,rPB,q_oracle_b(rtrace))
        rank_b_ok=rb_idx==ib
        rank_value_ok=rank_b_ok and exact(rfinal,g["class"])
        rank_e2e=rank_trace_ok and rank_value_ok
    else:
        RANKS_B.append(None);MARG_B.append(float("nan"))

    # WRONG-TRACE NEGATIVE CONTROL:
    # deterministic wrong seal, absent as the current gold bridge.
    wrong_trace=SEAL_BANK[(SEAL_BANK.index(g["seal"])+17)%len(SEAL_BANK)]
    if wrong_trace==g["seal"]:raise RuntimeError("Wrong-trace construction failure.")
    wb_idx,_,_=ranked_select(installed,q_rel_seal(wrong_trace),excluded={ra_idx})
    winst,wTm,wP=installed[wb_idx]
    wfinal=read_kv(winst,wTm,wP,q_oracle_b(wrong_trace))
    wrong_b=int(wb_idx==ib)
    wrong_final=int(exact(wfinal,g["class"]))

    R["DIRECT"].append(int(direct_ok));R["ORACLE"].append(int(oracle_ok));R["STRICT"].append(int(strict_ok))
    R["RANK_A"].append(int(rank_a_ok));R["RANK_TRACE"].append(int(rank_trace_ok));R["RANK_B"].append(int(rank_b_ok))
    R["RANK_VALUE"].append(int(rank_value_ok));R["RANK_E2E"].append(int(rank_e2e))
    R["WRONG_B"].append(wrong_b);R["WRONG_FINAL"].append(wrong_final)

    print(f"      [{z+1:02d}/{N_ITEMS}] {g['entity']:6s} {g['seal']}->{g['class']} | DIR={int(direct_ok)} OR={int(oracle_ok)} STRICT={int(strict_ok)} | RANK A={int(rank_a_ok)}(r{ra_rank}) TRACE={int(rank_trace_ok)} B={int(rank_b_ok)}(r{rb_rank}) VALUE={int(rank_value_ok)} E2E={int(rank_e2e)} | WRONG={wrong_final}")
    if not rank_e2e:
        topA=" ".join(f"M{j+1}:{s:+.2f}" for s,j in ra_scores[:3])
        topB="NONE"
        if re.fullmatch(r"[A-Z]{3}",rtrace or ""):
            topB=" ".join(f"M{j+1}:{s:+.2f}" for s,j in rb_scores[:3])
        print(f"               hidden=M{ia+1:02d}->M{ib+1:02d} | selected=M{ra_idx+1:02d} trace={rtrace!r} second={('M'+str(rb_idx+1).zfill(2)) if rb_idx is not None else 'NONE'} final={norm(rfinal)!r}")
        print(f"               topA: {topA}")
        print(f"               topB: {topB}")

print("\n[4/7] Accuracy...")
OK={k:np.asarray(v,dtype=np.int32) for k,v in R.items()}
for k,name in [
    ("DIRECT","DIRECT 32 ONE-SHOT"),("ORACLE","ORACLE ADDR E2E"),("STRICT","STRICT CONTENT E2E"),
    ("RANK_A","RANKED A ADDRESS"),("RANK_TRACE","RANKED A+TRACE"),("RANK_B","RANKED B ADDRESS"),
    ("RANK_VALUE","RANKED B+VALUE"),("RANK_E2E","RANKED END-TO-END"),
    ("WRONG_B","WRONG TRACE→GOLD B"),("WRONG_FINAL","WRONG TRACE→GOLD FINAL")
]:
    print(f"      {name:24s}: {OK[k].sum():2d}/{N_ITEMS} = {OK[k].mean():.4f}")

print("\n[5/7] Retrieval ranks and paired effects...")
ra=np.array([x for x in RANKS_A if x is not None],dtype=float)
rb=np.array([x for x in RANKS_B if x is not None],dtype=float)
print(f"      A rank: R1={np.mean(ra<=1):.4f} R5={np.mean(ra<=5):.4f} R16={np.mean(ra<=16):.4f} MRR={np.mean(1/ra):.6f} median={np.median(ra):.1f}")
if len(rb):
    print(f"      B rank: R1={np.mean(rb<=1):.4f} R5={np.mean(rb<=5):.4f} R16={np.mean(rb<=16):.4f} MRR={np.mean(1/rb):.6f} median={np.median(rb):.1f}")
for a,b in [("RANK_E2E","DIRECT"),("RANK_E2E","STRICT"),("ORACLE","RANK_E2E"),("RANK_E2E","WRONG_FINAL")]:
    d,ci=bootstrap_delta(OK[a],OK[b],10000,SEED+sum(map(ord,a+b)))
    print(f"      {a}-{b}: Δ={d:+.4f} | bootstrap95=[{ci[0]:+.4f},{ci[1]:+.4f}]")

print("\n[6/7] Failure localization...")
print(f"      Oracle relay capacity             : {OK['ORACLE'].sum()}/{N_ITEMS}")
print(f"      Strict 32-bank E2E                : {OK['STRICT'].sum()}/{N_ITEMS}")
print(f"      Ranked first address              : {OK['RANK_A'].sum()}/{N_ITEMS}")
print(f"      Ranked correct model trace        : {OK['RANK_TRACE'].sum()}/{N_ITEMS}")
print(f"      Ranked second address             : {OK['RANK_B'].sum()}/{N_ITEMS}")
print(f"      Ranked second value               : {OK['RANK_VALUE'].sum()}/{N_ITEMS}")
print(f"      Ranked complete traversal         : {OK['RANK_E2E'].sum()}/{N_ITEMS}")
print(f"      Wrong-trace gold-final leakage    : {OK['WRONG_FINAL'].sum()}/{N_ITEMS}")
if OK["RANK_TRACE"].sum():
    c=int(((OK["RANK_TRACE"]==1)&(OK["RANK_VALUE"]==1)).sum());n=int(OK["RANK_TRACE"].sum())
    print(f"      B complete | correct Stage-1 trace: {c}/{n} = {c/n:.4f}")

print("\n[7/7] Mechanistic classification...")
DIRECT=OK["DIRECT"].mean();ORACLE=OK["ORACLE"].mean();STRICT=OK["STRICT"].mean()
RA=OK["RANK_A"].mean();RT=OK["RANK_TRACE"].mean();RB=OK["RANK_B"].mean();RV=OK["RANK_VALUE"].mean()
E2E=OK["RANK_E2E"].mean();WRONG=OK["WRONG_FINAL"].mean()
if ORACLE<.75:
    VERDICT="32_BANK_RELAY_CAPACITY_NOT_RELIABLE"
elif RA<.75:
    VERDICT="32_BANK_FIRST_CONTENT_RANKING_NOT_RELIABLE"
elif RT<.75:
    VERDICT="32_BANK_TRACE_EXTRACTION_NOT_RELIABLE"
elif RB<.75:
    VERDICT="32_BANK_TRACE_TO_NEXT_MEMORY_RANKING_NOT_RELIABLE"
elif E2E>=.75 and WRONG<=.25:
    VERDICT="32_BANK_SELECTIVE_ASSOCIATIVE_RELAY_REPLICATED"
elif E2E>DIRECT+.25 and E2E>WRONG+.25:
    VERDICT="32_BANK_SELECTIVE_ASSOCIATIVE_RELAY_PARTIAL"
else:
    VERDICT="32_BANK_SELECTIVE_ASSOCIATIVE_RELAY_NOT_YET_RELIABLE"

print("\n"+"="*166)
print("TEST506 FINAL RESULT — AKBASCORE MAM · 32-CARTRIDGE CONTENT-ADDRESSED RELAY")
print("="*166)
print("MODEL                         : Qwen/Qwen2.5-7B-Instruct · frozen")
print("BANK                          : 32 independently forged RAW BELLEKÖZ")
print("REQUIRED                      : 2")
print("DISTRACTORS                   : 30")
print("REQUIRED ADDRESSES TO CONTENT : NO")
print("TASK                          : Entity → Seal → Class")
print("TARGET ENTITY IN DISTRACTORS  : NO")
print("BRIDGE SEAL IN DISTRACTORS    : NO")
print("FINAL CLASS IN DISTRACTORS    : NO")
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
print("INTERMEDIATE                  : model-produced textual trace")
print("STAGE-2 REUSES STAGE-1 MEMORY : NO")
print("-"*166)
print(f"DIRECT · 32 ONE-SHOT           : {OK['DIRECT'].sum():2d}/{N_ITEMS} = {DIRECT:.4f}")
print(f"ORACLE ADDRESS RELAY           : {OK['ORACLE'].sum():2d}/{N_ITEMS} = {ORACLE:.4f}")
print(f"STRICT CONTENT E2E             : {OK['STRICT'].sum():2d}/{N_ITEMS} = {STRICT:.4f}")
print(f"RANKED · FIRST ADDRESS         : {OK['RANK_A'].sum():2d}/{N_ITEMS} = {RA:.4f}")
print(f"RANKED · FIRST TRACE           : {OK['RANK_TRACE'].sum():2d}/{N_ITEMS} = {RT:.4f}")
print(f"RANKED · SECOND ADDRESS        : {OK['RANK_B'].sum():2d}/{N_ITEMS} = {RB:.4f}")
print(f"RANKED · SECOND VALUE          : {OK['RANK_VALUE'].sum():2d}/{N_ITEMS} = {RV:.4f}")
print(f"RANKED · END-TO-END            : {OK['RANK_E2E'].sum():2d}/{N_ITEMS} = {E2E:.4f}")
print(f"WRONG TRACE · GOLD B           : {OK['WRONG_B'].sum():2d}/{N_ITEMS} = {OK['WRONG_B'].mean():.4f}")
print(f"WRONG TRACE · GOLD FINAL       : {OK['WRONG_FINAL'].sum():2d}/{N_ITEMS} = {WRONG:.4f}")
print("-"*166)
print("TEST501 LOCK SHA              :",TEST501_LOCK_SHA)
print("TEST502 LOCK SHA              :",TEST502_LOCK_SHA)
print("TEST503 LOCK SHA              :",TEST503_LOCK_SHA)
print("TEST504 LOCK SHA              :",TEST504_LOCK_SHA)
print("TEST505 LOCK SHA              :",TEST505_LOCK_SHA)
print("TEST506 LOCK SHA              :",LOCK_SHA)
print(f"TOTAL TEST TIME                : {time.perf_counter()-T0:.2f}s")
print("EXPLORATORY VERDICT           :",VERDICT)
print("="*166)

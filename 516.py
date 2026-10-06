# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST516
# TEXT-FREE NUMERIC HANDOFF INTO AN INDEPENDENT BELLEKÖZ
#
# PARENTS:
#   TEST504 : independent A -> textual trace -> independent B = 31/32
#   TEST503 : independent A||B cache composition = 0/32
#   TEST513 : frozen L27+SOFT2 one-shot numeric B address observed
#   TEST514 : frozen numeric address causally depends on A
#   TEST515 : numeric B selection works, but static A||B tensor relay = 0/32
#
# QUESTION:
# Can the frozen model replace TEST504's decoded textual intermediate with a
# purely numeric model-native handoff while keeping B in a fresh independent
# read session?
#
# RUNTIME:
#   fresh model state
#   A BELLEKÖZ -> natural query -> L27 -> frozen SOFT2 -> numeric B selection
#   discard A runtime cache
#   open selected B alone
#   fixed reader scaffold + NUMERIC carrier embeddings -> final value
#
# CRITICAL:
#   NO A||B cache concatenation
#   NO Seal decoding
#   NO Seal text reinsertion
#   NO gold B in SELECTED arm
#   NO candidate-B forward before selection
#   NO training / DRA / ANN / learned router / post-hoc selection
#
# ORACLE_NUMERIC_HANDOFF is an explicitly labelled mechanistic control only.
# Human-readable source/GT exists only for controlled construction/evaluation.
# Persistent BELLEKÖZ cartridges after forge are tensors.

import os,sys,subprocess,importlib.util,random,time,hashlib,json,gc
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache

TEST="516";SEED=516;BASE_SEED=501;MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
N=32;TRACE_DEPTH=27;MAX_NEW=12;DEVICE=torch.device("cuda")
FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
TEST503="2cb70ffa53a98efb249590dba7ef184b9e41cf72a99291bda65e6c93b82cb5de"
TEST504="74c375373666c840105cc165b90977c56c8698cf3d5617c63f619a0139386846"
TEST508="47512c7860268517670ca5b45b6e76fe1582362dd38e447d164f33c756a1a2ca"
TEST513="2b28137b12a26517261d0d8827c378f7e9e455bd5f4ccb12af47277ed8de01aa"
TEST514="61e83ffe4b57cbafa0f4ef8a3fe516a153818fa43e87176941fd01ec8ad24146"
TEST515="cf9e875f1a8fa8403f688c7150ea9bfec4dd3bea407cf2ce44f27e1961c9f7fa"

ENTITY_BANK=[
("Aldren","Boreal","Cyrene"),("Darian","Elara","Faron"),("Galen","Hesper","Ilyra"),("Joren","Kaelis","Lorin"),
("Maren","Neris","Orlan"),("Perrin","Quorin","Ralen"),("Saren","Taris","Ulric"),("Valen","Weyra","Xeran"),
("Yorin","Zaren","Avel"),("Brann","Ceris","Dalen"),("Eris","Felis","Gorin"),("Halen","Ivar","Jaris"),
("Koren","Leris","Miran"),("Nolan","Orel","Palis"),("Riven","Solis","Teren"),("Urian","Varen","Wilis"),
("Xorin","Yalen","Zorin"),("Arven","Belis","Coren"),("Derin","Evan","Feris"),("Garin","Heron","Ilven"),
("Jarin","Kelis","Laven"),("Moris","Naven","Orris"),("Parin","Rovis","Selan"),("Torin","Ulen","Veris"),
("Waren","Xelis","Yaris"),("Zelis","Aren","Borin"),("Caren","Dorin","Elen"),("Faren","Gelis","Harin"),
("Iren","Joris","Kalen"),("Laris","Meren","Noren"),("Oris","Peren","Ravin"),("Serin","Toren","Ulis")]
SEAL_BANK=["KOR","VEL","DAR","MIR","SEN","ROL","FEN","JAL","WEX","NUR","BAV","CIR","DEM","GOS","HIL","KET","LOR","MEV","PIR","RUK","SAV","TOL","VIR","YEK","ZAM","BIR","CAV","DOL","FER","GUL","HAR","JEM","KIR","LEV","MOR","NEX","PEL","RAS","SUL","TIR","VAN","YOR","ZEL","BOS","CER","DIN","FAL","GER","HOV","JUN","KEL","LUM","NAV","POR","REV","SIM","TUR","WAL","XEN","YIL","ZOR","BEK","COR","DUR","EKS","FIR","GAN","HEL","IVO","JOR","KAS","LIN","MUR","NOR","OVA","PAR","RIN","SOL","TEV","URB","VAR","WEN","XAL","YUN","ZEN","BOR","CEN","DAN","ELV","FOR","GIR","HAN","IRV","JEN","KOL","MAR"]
CLASS_BANK=["TAK","BEX","QAA","RAV","SOD","PEK","NIV","GOR","HAX","JUR","KEM","VOL","DAX","QAB","MON","SAL","TEK","WIR","ZUN","COV","HEM","JAX","LIV","QAC","PAK","RUM","SEV","TIX","VOR","YAM","ZEK","BOL","QAD","DOV","FEX","GAM","HUR","JIN","KAV","LER","MEX","QAE","PIV","ROK","SUM","TAL","VEK","WON","XIR","YAV","ZOL","BAR","CIX","QAF","FOV","GEL","HIN","JOV","KUR","QAG","MAV","QAH","PUL","RIM","QAI","TOX","VIL","WER","XAN","YER","ZIM","BUN","CAL","DOR","EVI","FAR","GUN","HES","ILM","JER","KON","LAR","QAJ","NOL","OVI","PER","RUS","SIN","QAK","VEX","QAL","QAM","YUL","ZAR","BEL","CUM"]

assert len(ENTITY_BANK)==N and len(SEAL_BANK)>=N*3 and len(CLASS_BANK)>=N*3
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
torch.set_grad_enabled(False)
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

def reorder(rows,gold,pos,seed):
    g=rows[gold];rest=[x for j,x in enumerate(rows) if j!=gold]
    rr=random.Random(seed);rr.shuffle(rest);out=rest[:];out.insert(pos,g);return out

def make_items():
    rng=random.Random(BASE_SEED);seals=SEAL_BANK.copy();classes=CLASS_BANK.copy()
    rng.shuffle(seals);rng.shuffle(classes);items=[]
    for i in range(N):
        ents=list(ENTITY_BANK[i]);g=i%3;ss=seals[3*i:3*i+3];cc=classes[3*i:3*i+3]
        ra=[f"Instrument {ents[j]} carries seal {ss[j]}." for j in range(3)]
        rb=[f"Seal {ss[j]} corresponds to routing class {cc[j]}." for j in range(3)]
        A=" ".join(reorder(ra,g,i%3,BASE_SEED+i*101+17))
        B=" ".join(reorder(rb,g,(i+1)%3,BASE_SEED+i*101+34))
        items.append({"id":i+1,"A":A,"B":B,"gold":{"entity":ents[g],"seal":ss[g],"class":cc[g]}})
    return items

ITEMS=make_items();USED=sorted({x["gold"]["seal"] for x in ITEMS})
if len(USED)!=N:raise RuntimeError("Seal uniqueness failure.")
WRONG=[(i+11)%N for i in range(N)]
assert all(0<=x<N for x in WRONG)
assert all(WRONG[i]!=i for i in range(N))

LOCK={"test":TEST,"model":MODEL_ID,"seed":SEED,"base_seed":BASE_SEED,"items":ITEMS,
"parents":{"503":TEST503,"504":TEST504,"508":TEST508,"513":TEST513,"514":TEST514,"515":TEST515},
"trace_depth":27,"address":"SOFT2_FROZEN_FROM_513_514",
"architecture":"INDEPENDENT_B_FRESH_SESSION_NUMERIC_HANDOFF",
"A_B_cache_concat":"FORBIDDEN","intermediate_decode":"NONE","intermediate_text_reinsertion":"NONE",
"query_time_candidate_B_forwards":0,"training":"OFF","dra":"OFF","ann":"OFF","learned_router":"OFF",
"arms":["SELECTED_NUMERIC","ORACLE_NUMERIC","WRONG_NUMERIC","NO_CARRIER"],
"persistence_contract":"FORGE_THEN_FRESH_RUNTIME_STATE"}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

print("="*176)
print("TEST516 — AKBASCORE MAM · TEXT-FREE NUMERIC HANDOFF INTO AN INDEPENDENT BELLEKÖZ")
print("FROZEN L27+SOFT2 ADDRESS → FRESH B SESSION → NUMERIC TWO-SLOT HANDOFF → FINAL VALUE")
print("="*176)
print("LOCK SHA:",LOCK_SHA)
print("PARENT TEST504:",TEST504)
print("PARENT TEST514:",TEST514)
print("PARENT TEST515:",TEST515)
print("A||B CACHE CONCATENATION: FORBIDDEN")
print("INTERMEDIATE DECODE / TEXT REINSERTION: NONE")
T0=time.perf_counter()

print("\n[1/9] Loading frozen Qwen...")
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
EMB=model.model.embed_tokens
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | {NL}L H={H} QH={QH} KVH={KVH} HD={HD} | frozen BF16")

@torch.inference_mode()
def kv_from_ids(ids):
    o=model(input_ids=torch.tensor([ids],device=DEVICE),use_cache=False,output_hidden_states=True,return_dict=True);out=[]
    for L,layer in enumerate(model.model.layers):
        h=o.hidden_states[L][0].to(layer.input_layernorm.weight.device,layer.input_layernorm.weight.dtype)
        hn=layer.input_layernorm(h)
        k=layer.self_attn.k_proj(hn).view(-1,KVH,HD).transpose(0,1).contiguous()
        v=layer.self_attn.v_proj(hn).view(-1,KVH,HD).transpose(0,1).contiguous()
        out.append((k,v))
    return out

def forge(s):return kv_from_ids([PAD]+enc(s+SEP))

def rope_k(k,pos,L):
    layer=model.model.layers[L];rot=layer.self_attn.rotary_emb if hasattr(layer.self_attn,"rotary_emb") else model.model.rotary_emb
    d=torch.zeros((1,k.shape[0],len(pos),HD),device=k.device,dtype=k.dtype);p=torch.tensor([pos],device=k.device,dtype=torch.long)
    try:c,s=rot(d,p)
    except TypeError:c,s=rot(d,position_ids=p)
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    _,kr=apply_rotary_pos_emb(d,k.unsqueeze(0),c,s);return kr[0]

def install(raw):
    T=raw[0][0].shape[1];out=[]
    for L,(k,v) in enumerate(raw):out.append((rope_k(k,list(range(T)),L),v))
    return out,T,T

def cache_of(inst):
    try:c=DynamicCache(config=cfg)
    except TypeError:c=DynamicCache()
    for L,(k,v) in enumerate(inst):c.update(k.unsqueeze(0),v.unsqueeze(0),L)
    return c

def unit(x):
    x=x.float().reshape(-1)
    return x/x.norm().clamp_min(1e-8)

def firstline(x):return next((z.strip() for z in str(x).splitlines() if z.strip()),"")
def normtxt(x):return firstline(x).strip().upper().rstrip(".")
def exact(x,t):return normtxt(x)==str(t).upper()
def qA(e):return f"What seal does instrument {e} carry? Give only the exact seal."

print("\n[2/9] Freezing TEST513/514 SOFT2 geometry...")
TOK={s:enc(s) for s in USED}
if any(len(x)<1 or len(x)>2 for x in TOK.values()):raise RuntimeError("Tokenizer geometry changed.")
W=model.lm_head.weight.detach().float().cpu()
Wn=W/W.norm(dim=1,keepdim=True).clamp_min(1e-8)
E=EMB.weight.detach().float().cpu()
FIRST=sorted(set(x[0] for x in TOK.values()))
SECOND=sorted(set(x[1] for x in TOK.values() if len(x)==2))
FM=Wn[FIRST];SM=Wn[SECOND];FG={}
for s,x in TOK.items():FG.setdefault(x[0],[]).append(s)
ZERO=torch.zeros(H,dtype=torch.float32)

ADDR=[]
for it in ITEMS:
    ids=TOK[it["gold"]["seal"]]
    a1=Wn[ids[0]]
    a2=Wn[ids[1]] if len(ids)==2 else ZERO
    lb=1.0 if len(ids)==2 else 0.0
    ADDR.append(unit(torch.cat([a1,a2,torch.tensor([lb],dtype=torch.float32)])))

def soft2(logits):
    fl=torch.tensor([float(logits[t]) for t in FIRST],dtype=torch.float32)
    pf=torch.softmax(fl,0);q1=unit(pf@FM)
    if SECOND:
        sl=torch.tensor([float(logits[t]) for t in SECOND],dtype=torch.float32)
        p2=torch.softmax(sl,0);q2=p2@SM
        q2=q2-torch.dot(q2,unit(q1))*unit(q1)
        q2=unit(q2) if float(q2.norm())>1e-8 else ZERO
    else:q2=ZERO
    two=sum(float(pf[j]) for j,t in enumerate(FIRST) if any(len(TOK[s])==2 for s in FG[t]))
    return unit(torch.cat([q1,q2,torch.tensor([two],dtype=torch.float32)]))

# Same frozen L27 evidence is converted into two continuous embedding-space
# carrier slots. No Seal string is decoded and no Seal token IDs are inserted.
def latent_carrier(logits):
    fl=torch.tensor([float(logits[t]) for t in FIRST],dtype=torch.float32)
    pf=torch.softmax(fl,0);c1=pf@E[FIRST]
    if SECOND:
        sl=torch.tensor([float(logits[t]) for t in SECOND],dtype=torch.float32)
        p2=torch.softmax(sl,0);c2=p2@E[SECOND]
    else:c2=torch.zeros_like(c1)
    return torch.stack([c1,c2]).to(dtype=EMB.weight.dtype)

# Explicit mechanistic control only: exact gold numeric embedding carrier.
def oracle_carrier(i):
    ids=TOK[ITEMS[i]["gold"]["seal"]]
    c1=E[ids[0]]
    c2=E[ids[1]] if len(ids)==2 else torch.zeros(H,dtype=E.dtype)
    return torch.stack([c1,c2]).to(dtype=EMB.weight.dtype)

def zero_carrier():
    return torch.zeros((2,H),dtype=EMB.weight.dtype)

print("      L27: FROZEN")
print("      SOFT2: FROZEN")
print("      Address dimension:",2*H+1)
print("      Handoff carrier: 2 numeric H-dimensional slots")
print("      No Seal string is decoded into runtime handoff.")

print("\n[3/9] OFFLINE FORGE — building persistent A/B tensor cartridges...")
A_PACK=[];B_PACK=[]
for i,it in enumerate(ITEMS):
    A_PACK.append(install(forge(it["A"])))
    B_PACK.append(install(forge(it["B"])))
    if i<5:print(f"      ITEM {i+1:02d} | A tokens={A_PACK[-1][1]} B tokens={B_PACK[-1][1]}")
assert len(A_PACK)==N and len(B_PACK)==N
print("      64 persistent tensor cartridges forged.")

# PERSISTENCE / RESET CONTRACT:
# No forge-time DynamicCache survives. Every runtime operation reconstructs
# a fresh DynamicCache from the persistent tensor cartridge.
gc.collect();torch.cuda.empty_cache()
print("\n[4/9] PERSISTENCE RESET...")
print("      Forge-time runtime state discarded.")
print("      Persistent BELLEKÖZ memory = tensor packets only.")
print("      Human-readable ITEMS remain host-side construction/evaluation metadata.")
print("      New query sessions create fresh DynamicCache objects.")
print("      Model parameters remain frozen.")

@torch.inference_mode()
def l27_logits(packet,q):
    inst,Tm,P=packet;c=cache_of(inst);ids=enc(FMT.format(q=q));n=len(ids)
    pos=torch.arange(P,P+n,device=DEVICE).unsqueeze(0)
    mask=torch.ones((1,Tm+n),device=DEVICE,dtype=torch.long)
    o=model(input_ids=torch.tensor([ids],device=DEVICE),past_key_values=c,attention_mask=mask,position_ids=pos,
            use_cache=False,output_hidden_states=True,return_dict=True)
    h=o.hidden_states[TRACE_DEPTH][0,-1];fn=model.model.norm
    return model.lm_head(fn(h.to(fn.weight.dtype))).float().cpu()

def address_scores(logits):
    q=soft2(logits)
    return [float(torch.dot(q,a)) for a in ADDR]

def rank(sc,gold):
    if len(sc)!=N:raise RuntimeError(f"Address score length mismatch: {len(sc)} != {N}")
    order=sorted(range(N),key=lambda j:(-sc[j],j))
    return order.index(gold)+1,order[0]

# Fixed global reader scaffold. No item-specific Seal/Class/identifier.
# The two internal positions are supplied as continuous numeric embeddings.
PREFIX="QUESTION:\nUsing the stored memory, the relevant internal key is"
SUFFIX=". What routing class corresponds to that internal key? Give only the exact routing class.\n\nANSWER:"

@torch.inference_mode()
def read_B_numeric(packet,carrier,max_new=MAX_NEW):
    inst,Tm,P=packet;c=cache_of(inst)
    if carrier.shape!=(2,H):raise RuntimeError(f"Carrier shape mismatch: {tuple(carrier.shape)} != {(2,H)}")
    pre=enc(PREFIX);suf=enc(SUFFIX)
    pree=EMB(torch.tensor(pre,device=DEVICE,dtype=torch.long)).detach()
    sufe=EMB(torch.tensor(suf,device=DEVICE,dtype=torch.long)).detach()
    car=carrier.to(device=DEVICE,dtype=EMB.weight.dtype)
    x=torch.cat([pree,car,sufe],dim=0).unsqueeze(0)
    n=x.shape[1];pos=torch.arange(P,P+n,device=DEVICE).unsqueeze(0)
    mask=torch.ones((1,Tm+n),device=DEVICE,dtype=torch.long)
    o=model(inputs_embeds=x,past_key_values=c,attention_mask=mask,position_ids=pos,use_cache=True,return_dict=True)
    c=o.past_key_values;out=[];Tm+=n;P+=n
    nxt=int(o.logits[0,-1].float().argmax())
    for _ in range(max_new):
        if nxt in EOS:break
        out.append(nxt)
        pos=torch.tensor([[P]],device=DEVICE,dtype=torch.long)
        mask=torch.ones((1,Tm+1),device=DEVICE,dtype=torch.long)
        o=model(input_ids=torch.tensor([[nxt]],device=DEVICE,dtype=torch.long),past_key_values=c,
                attention_mask=mask,position_ids=pos,use_cache=True,return_dict=True)
        c=o.past_key_values;Tm+=1;P+=1
        nxt=int(o.logits[0,-1].float().argmax())
    return tok.decode(out,skip_special_tokens=True).strip()

print("\n[5/9] Fresh-session address phase...")
LOGITS=[];CARRIER=[];SEL=[];RANK=[]
for i,it in enumerate(ITEMS):
    l=l27_logits(A_PACK[i],qA(it["gold"]["entity"]))
    sc=address_scores(l);r,p=rank(sc,i)
    LOGITS.append(l);CARRIER.append(latent_carrier(l));SEL.append(int(p));RANK.append(int(r))
    print(f"      [{i+1:02d}/32] {it['gold']['entity']:6s} | B-rank={r:2d} selected=B{p+1:02d}")

assert len(LOGITS)==len(CARRIER)==len(SEL)==len(RANK)==N
a=np.asarray(RANK,dtype=np.float64)
AR1=float(np.mean(a==1));AR5=float(np.mean(a<=5));AMRR=float(np.mean(1.0/a))
print(f"      ADDRESS R1/R5/MRR: {AR1:.4f} / {AR5:.4f} / {AMRR:.6f}")

print("\n[6/9] Independent B numeric-handoff arms...")
RES={k:[] for k in ("SELECTED","ORACLE","WRONG","NO_CARRIER")}
RAW={k:[] for k in RES}
for i,it in enumerate(ITEMS):
    target=it["gold"]["class"]
    ys=read_B_numeric(B_PACK[SEL[i]],CARRIER[i])
    yo=read_B_numeric(B_PACK[i],oracle_carrier(i))
    yw=read_B_numeric(B_PACK[i],oracle_carrier(WRONG[i]))
    yn=read_B_numeric(B_PACK[i],zero_carrier())
    for name,y in (("SELECTED",ys),("ORACLE",yo),("WRONG",yw),("NO_CARRIER",yn)):
        RES[name].append(int(exact(y,target)));RAW[name].append(y)
    print(f"      [{i+1:02d}/32] {it['gold']['entity']:6s} {it['gold']['seal']}->{target} | addr={int(SEL[i]==i)} SEL={RES['SELECTED'][-1]} OR={RES['ORACLE'][-1]} WRONG={RES['WRONG'][-1]} ZERO={RES['NO_CARRIER'][-1]}")
    if not (RES["SELECTED"][-1] and RES["ORACLE"][-1]):
        print(f"               selected B=B{SEL[i]+1:02d} | selected={firstline(ys)!r}")
        print(f"               oracle={firstline(yo)!r} | wrong={firstline(yw)!r} | zero={firstline(yn)!r}")

for name in RES:
    if len(RES[name])!=N:raise RuntimeError(f"{name} result length mismatch: {len(RES[name])} != {N}")
    if len(RAW[name])!=N:raise RuntimeError(f"{name} raw result length mismatch: {len(RAW[name])} != {N}")

print("\n[7/9] Failure localization...")
# FIX: RES.items(), not RES.
OK={name:np.asarray(values,dtype=np.int32) for name,values in RES.items()}
ADDR_OK=np.asarray([1 if SEL[i]==i else 0 for i in range(N)],dtype=np.int32)
COND=int(np.sum((ADDR_OK==1)&(OK["SELECTED"]==1)))
NADDR=int(ADDR_OK.sum())
HANDOFF_FAIL=int(np.sum((ADDR_OK==1)&(OK["SELECTED"]==0)))
ACCIDENTAL=int(np.sum((ADDR_OK==0)&(OK["SELECTED"]==1)))

for name in ("SELECTED","ORACLE","WRONG","NO_CARRIER"):
    n_ok=int(OK[name].sum());rate=float(OK[name].mean())
    print(f"      {name:12s}: {n_ok:2d}/{N} = {rate:.4f}")
print(f"      Correct address                    : {NADDR}/{N} = {float(ADDR_OK.mean()):.4f}")
print(f"      SELECTED success | address correct : {COND}/{NADDR} = {COND/NADDR:.4f}" if NADDR else "      SELECTED success | address correct : N/A")
print(f"      Correct address but handoff fail   : {HANDOFF_FAIL}")
print(f"      Wrong address but accidental pass  : {ACCIDENTAL}")

print("\n[8/9] Causal controls...")
D_OR_WR=float(OK["ORACLE"].mean()-OK["WRONG"].mean())
D_OR_NO=float(OK["ORACLE"].mean()-OK["NO_CARRIER"].mean())
D_SEL_NO=float(OK["SELECTED"].mean()-OK["NO_CARRIER"].mean())
print(f"      ORACLE - WRONG carrier : {D_OR_WR:+.4f}")
print(f"      ORACLE - ZERO carrier  : {D_OR_NO:+.4f}")
print(f"      SELECTED - ZERO        : {D_SEL_NO:+.4f}")

if float(OK["ORACLE"].mean())>=.75 and D_OR_WR>=.40 and D_OR_NO>=.40 and NADDR and COND/NADDR>=.70:
    VERDICT="TEXT_FREE_NUMERIC_HANDOFF_INTO_INDEPENDENT_B_OBSERVED"
elif float(OK["ORACLE"].mean())>=.60 and D_OR_NO>=.25:
    VERDICT="NUMERIC_HANDOFF_SIGNAL_PRESENT_BUT_NOT_YET_RELIABLE"
elif float(OK["ORACLE"].mean())<.40:
    VERDICT="NUMERIC_HANDOFF_MECHANISM_NOT_YET_ESTABLISHED"
else:
    VERDICT="NUMERIC_HANDOFF_PARTIAL_INCONCLUSIVE"

print("\n[9/9] Protocol audit...")
print("      Model weights frozen                  : YES")
print("      Forge/runtime state separated          : YES")
print("      Fresh cache each query/read            : YES")
print("      A and B cache concatenated             : NO")
print("      Gold B used in SELECTED arm            : NO")
print("      Intermediate Seal decoded              : NO")
print("      Intermediate Seal token IDs inserted   : NO")
print("      Intermediate text reinserted           : NO")
print("      Candidate B model forwards to select   : 0")
print("      Learned router / ANN / DRA / training  : NONE")
print("      Item-specific reader prompt            : NO")
print("      Numeric handoff slots                  : 2")
print("      Persistent BELLEKÖZ representation     : TENSOR")
print("      Human-readable ITEMS at runtime host   : EVALUATION METADATA ONLY")
print("      SELECTED path reads gold metadata      : NO")
print("      ORACLE/WRONG controls read gold metadata: YES — CONTROL ARMS ONLY")

print("\n"+"="*176)
print("TEST516 FINAL RESULT — AKBASCORE MAM · TEXT-FREE NUMERIC HANDOFF")
print("="*176)
print("MODEL                              : Qwen/Qwen2.5-7B-Instruct · frozen")
print("PERSISTENCE                        : forge → reset → fresh runtime state")
print("PERSISTENT BELLEKÖZ                : tensor packets")
print("A/B RELATION                       : NEVER CONCATENATED")
print("TRACE                              : L27 · frozen")
print("ADDRESS                            : SOFT2 · frozen")
print("INTERMEDIATE DECODE                : NONE")
print("TEXT REINSERTION                   : NONE")
print("B READ                             : independent fresh session")
print("HANDOFF                            : numeric 2-slot embedding carrier")
print("-"*176)
print(f"ADDRESS R1                         : {AR1:.4f}")
print(f"ADDRESS R5                         : {AR5:.4f}")
print(f"ADDRESS MRR                        : {AMRR:.6f}")
print(f"SELECTED NUMERIC E2E               : {int(OK['SELECTED'].sum()):2d}/{N} = {float(OK['SELECTED'].mean()):.4f}")
print(f"ORACLE NUMERIC HANDOFF             : {int(OK['ORACLE'].sum()):2d}/{N} = {float(OK['ORACLE'].mean()):.4f}")
print(f"WRONG NUMERIC CARRIER              : {int(OK['WRONG'].sum()):2d}/{N} = {float(OK['WRONG'].mean()):.4f}")
print(f"ZERO CARRIER                       : {int(OK['NO_CARRIER'].sum()):2d}/{N} = {float(OK['NO_CARRIER'].mean()):.4f}")
print(f"SELECTED | CORRECT ADDRESS         : {COND}/{NADDR} = {COND/NADDR:.4f}" if NADDR else "SELECTED | CORRECT ADDRESS         : N/A")
print(f"CORRECT ADDRESS / HANDOFF FAIL     : {HANDOFF_FAIL}")
print(f"WRONG ADDRESS / ACCIDENTAL PASS    : {ACCIDENTAL}")
print("-"*176)
print(f"ORACLE - WRONG                     : {D_OR_WR:+.4f}")
print(f"ORACLE - ZERO                      : {D_OR_NO:+.4f}")
print(f"SELECTED - ZERO                    : {D_SEL_NO:+.4f}")
print("-"*176)
print("TEST503 LOCK                       :",TEST503)
print("TEST504 LOCK                       :",TEST504)
print("TEST508 LOCK                       :",TEST508)
print("TEST513 LOCK                       :",TEST513)
print("TEST514 LOCK                       :",TEST514)
print("TEST515 LOCK                       :",TEST515)
print("TEST516 LOCK                       :",LOCK_SHA)
print(f"TOTAL TEST TIME                    : {time.perf_counter()-T0:.2f}s")
print("VERDICT                            :",VERDICT)
print("="*176)

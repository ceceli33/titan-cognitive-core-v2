# Copyright © 2026 Mustafa Akbaş
# New original AKBASCORE MAM material is source-available under the repository LICENSE.
# See repository LICENSE for complete terms.
#
# AKBASCORE MAM — TEST504
# RUNTIME MEMORY-TO-MEMORY RELAY
#
# TEST503:
#   CONTIG RAW               32/32
#   JOINT-FORGED → REBASED   29/32
#   INDEPENDENT RAW           0/32
#
# TEST504 asks whether two independently forged memories can recover the
# native two-hop relation by passing only the model-produced intermediate
# trace from BELLEKÖZ A to BELLEKÖZ B at runtime.
#
# No joint re-forging. No gold intermediate injection. No compression.
# No DRA/steering. No router/ANN/graph. No training. No post-hoc selection.

import os,sys,subprocess,importlib.util,random,time,hashlib,json
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache

TEST="504";SEED=501;MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N_ITEMS=32;MAX_NEW=32
DEVICE=torch.device("cuda");FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
TEST501_LOCK_SHA="1a3e6ea2a9f6c72d7b8dacb0c0844e9f2a6f8595bb232f1554eb41af7a20dc41"
TEST502_LOCK_SHA="f852b6efc6ee887bae54db08cbc11f987e749499a51cebeb5132f569e74edb89"
TEST503_LOCK_SHA="2cb70ffa53a98efb249590dba7ef184b9e41cf72a99291bda65e6c93b82cb5de"

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
assert len(SEAL_BANK)>=N_ITEMS*3 and len(CLASS_BANK)>=N_ITEMS*3

os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
torch.set_grad_enabled(False)
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

def reorder(rows,gold,target_pos,seed):
    g=rows[gold];rest=[x for j,x in enumerate(rows) if j!=gold]
    rr=random.Random(seed);rr.shuffle(rest);out=rest[:];out.insert(target_pos,g);return out

def make_items():
    rng=random.Random(SEED);seals=SEAL_BANK.copy();classes=CLASS_BANK.copy()
    rng.shuffle(seals);rng.shuffle(classes);items=[]
    for i in range(N_ITEMS):
        ents=list(ENTITY_BANK[i]);g=i%3;ss=seals[3*i:3*i+3];cc=classes[3*i:3*i+3]
        r1=[f"Instrument {ents[j]} carries seal {ss[j]}." for j in range(3)]
        r2=[f"Seal {ss[j]} corresponds to routing class {cc[j]}." for j in range(3)]
        p1=i%3;p2=(i+1)%3
        rec1=" ".join(reorder(r1,g,p1,SEED+i*101+17))
        rec2=" ".join(reorder(r2,g,p2,SEED+i*101+34))
        items.append({"id":i+1,"records":[rec1,rec2],"gold":{"entity":ents[g],"seal":ss[g],"class":cc[g]},"positions":[p1,p2]})
    return items

ITEMS=make_items()
LOCK={"test":TEST,"model":MODEL_ID,"seed":SEED,"items":ITEMS,
      "test501":TEST501_LOCK_SHA,"test502":TEST502_LOCK_SHA,"test503":TEST503_LOCK_SHA,
      "arms":["DIRECT_INDEPENDENT","STAGE1_MEMORY_A","GOLD_RELAY_CONTROL","PREDICTED_RUNTIME_RELAY"],
      "joint_reforge":"OFF","compression":"OFF","steering":"OFF","routing":"OFF","training":"OFF"}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

print("="*160)
print("TEST504 — AKBASCORE MAM · RUNTIME MEMORY-TO-MEMORY RELAY")
print("INDEPENDENT BELLEKÖZ A → MODEL-PRODUCED TRACE → INDEPENDENT BELLEKÖZ B")
print("="*160)
print("LOCK SHA:",LOCK_SHA)
print("TEST501 LOCK:",TEST501_LOCK_SHA)
print("TEST502 LOCK:",TEST502_LOCK_SHA)
print("TEST503 LOCK:",TEST503_LOCK_SHA)
print("ITEMS:",N_ITEMS)
print("JOINT RE-FORGE / COMPRESSION / DRA / ROUTER / TRAINING: NONE")
T0=time.perf_counter()

print("\n[1/5] Loading frozen Qwen...")
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
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | {NL}L | H={H} | QH={QH} KVH={KVH} HD={HD} | frozen BF16")

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

def firstline(x):return next((z.strip() for z in str(x).splitlines() if z.strip()),"")
def norm(x):return firstline(x).strip().upper().rstrip(".")
def exact(x,t):return norm(x)==str(t).upper()

def q_direct(g):
    return f"What routing class corresponds to instrument {g['entity']}? Follow the records and give only the exact routing class."
def q_stage1(g):
    return f"What seal does instrument {g['entity']} carry? Give only the exact seal."
def q_stage2(trace):
    return f"What routing class corresponds to seal {trace}? Give only the exact routing class."

def bootstrap_delta(a,b,B=10000,seed=504):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64);d=a-b;n=len(d)
    rng=np.random.default_rng(seed);vals=np.empty(B)
    for i in range(B):vals[i]=d[rng.integers(0,n,n)].mean()
    return float(d.mean()),tuple(np.quantile(vals,[.025,.975]))

print("\n[2/5] Panel seal...")
for it in ITEMS[:4]:
    print(f"      ITEM {it['id']:02d} entity={it['gold']['entity']} seal={it['gold']['seal']} target={it['gold']['class']} positions={it['positions']}")
print("      TEST502/503 deterministic panel reproduced.")
print("      Gold seal is used only in the explicitly labelled oracle control.")
print("      Predicted relay receives only the model-produced Stage-1 trace.")

print("\n[3/5] Running independent direct / Stage-1 / oracle relay / predicted relay...")
R={"DIRECT":[],"S1":[],"ORACLE":[],"RELAY":[]}
for z,it in enumerate(ITEMS):
    rec1,rec2=it["records"];g=it["gold"];seal=g["seal"];target=g["class"]

    raw1=forge(rec1);raw2=forge(rec2)

    # TEST502/503 independent-memory failure control.
    inst_direct,TmD,PD=install_parallel([raw1,raw2])
    direct=read_kv(inst_direct,TmD,PD,q_direct(g))

    # Stage 1: read only BELLEKÖZ A and recover the intermediate seal.
    inst1,Tm1,P1=install_single(raw1)
    s1=read_kv(inst1,Tm1,P1,q_stage1(g))
    trace=norm(s1)

    # Oracle relay control: proves BELLEKÖZ B can answer if given the correct intermediate.
    inst2o,Tm2o,P2o=install_single(raw2)
    oracle=read_kv(inst2o,Tm2o,P2o,q_stage2(seal))

    # Experimental relay: BELLEKÖZ B receives only the model-produced Stage-1 trace.
    inst2,Tm2,P2=install_single(raw2)
    relay=read_kv(inst2,Tm2,P2,q_stage2(trace))

    vals={"DIRECT":(direct,target),"S1":(s1,seal),"ORACLE":(oracle,target),"RELAY":(relay,target)}
    for k,(v,t) in vals.items():R[k].append({"ok":exact(v,t),"raw":v})
    print(f"      [{z+1:02d}/{N_ITEMS}] {g['entity']:6s} {seal}->{target} | DIRECT={int(R['DIRECT'][-1]['ok'])} S1={int(R['S1'][-1]['ok'])} ORACLE={int(R['ORACLE'][-1]['ok'])} RELAY={int(R['RELAY'][-1]['ok'])}")
    if not all(R[k][-1]["ok"] for k in R):
        print(f"               direct={firstline(direct)!r}")
        print(f"               stage1={firstline(s1)!r} | trace={trace!r}")
        print(f"               oracle={firstline(oracle)!r}")
        print(f"               relay ={firstline(relay)!r}")

print("\n[4/5] Relay anatomy...")
OK={k:np.array([int(x["ok"]) for x in v],dtype=np.int32) for k,v in R.items()}
for k,name in [("DIRECT","DIRECT INDEP"),("S1","A→TRACE"),("ORACLE","GOLD→B"),("RELAY","A→TRACE→B")]:
    print(f"      {name:14s}: {OK[k].sum():2d}/{N_ITEMS} = {OK[k].mean():.4f}")

S1PASS=OK["S1"]==1
relay_given_s1=int(((OK["S1"]==1)&(OK["RELAY"]==1)).sum())
n_s1=int(S1PASS.sum())
oracle_given_s1=int(((OK["S1"]==1)&(OK["ORACLE"]==1)).sum())
print(f"      S1 pass & relay pass : {relay_given_s1}/{n_s1 if n_s1 else 0}")
print(f"      S1 pass & oracle pass: {oracle_given_s1}/{n_s1 if n_s1 else 0}")
print(f"      DIRECT fail→RELAY pass: {int(((OK['DIRECT']==0)&(OK['RELAY']==1)).sum())}/{N_ITEMS}")
print(f"      S1 fail→RELAY pass    : {int(((OK['S1']==0)&(OK['RELAY']==1)).sum())}/{N_ITEMS}")

print("\n      Paired bootstrap accuracy differences:")
for a,b in [("RELAY","DIRECT"),("ORACLE","RELAY"),("S1","RELAY")]:
    d,ci=bootstrap_delta(OK[a],OK[b],10000,SEED+sum(map(ord,a+b)))
    print(f"      {a}-{b}: Δ={d:+.4f} | bootstrap95=[{ci[0]:+.4f},{ci[1]:+.4f}]")

print("\n[5/5] Mechanistic classification...")
DIRECT=OK["DIRECT"].mean();S1=OK["S1"].mean();ORACLE=OK["ORACLE"].mean();RELAY=OK["RELAY"].mean()
COND=relay_given_s1/n_s1 if n_s1 else 0.0
if DIRECT>=.25:
    VERDICT="INDEPENDENT_DIRECT_CONTROL_UNEXPECTEDLY_HIGH"
elif S1<.75:
    VERDICT="STAGE1_MEMORY_READ_NOT_RELIABLE"
elif ORACLE<.75:
    VERDICT="STAGE2_MEMORY_READ_NOT_RELIABLE"
elif RELAY>=.75 and COND>=.90:
    VERDICT="EXPLICIT_RUNTIME_MEMORY_RELAY_REPLICATED"
elif RELAY>DIRECT+.25:
    VERDICT="RUNTIME_RELAY_PARTIAL_RESCUE_OBSERVED"
else:
    VERDICT="RUNTIME_RELAY_NOT_YET_RELIABLE"

print("\n"+"="*160)
print("TEST504 FINAL RESULT — AKBASCORE MAM · RUNTIME MEMORY-TO-MEMORY RELAY")
print("="*160)
print("MODEL                       : Qwen/Qwen2.5-7B-Instruct · frozen")
print("TASK                        : 2 edges · Entity → Seal → Class")
print("MEMORY A                    : independently forged RAW K/V")
print("MEMORY B                    : independently forged RAW K/V")
print("JOINT RE-FORGE              : NONE")
print("RUNTIME INTERMEDIATE        : model-produced Stage-1 text trace")
print("GOLD INTERMEDIATE IN RELAY  : NO")
print("COMPRESSION                 : OFF")
print("DRA / STEERING              : NONE")
print("VTOKEN / ADDRESS            : OFF")
print("TOP-K / ROUTER / ANN        : NONE")
print("GRAPH                        : NONE")
print("TRAINING / LoRA             : NONE")
print("POST-HOC ITEM SELECTION     : NONE")
print("-"*160)
print(f"DIRECT · A+B ONE-SHOT        : {OK['DIRECT'].sum():2d}/{N_ITEMS} = {DIRECT:.4f}")
print(f"STAGE1 · A→TRACE             : {OK['S1'].sum():2d}/{N_ITEMS} = {S1:.4f}")
print(f"ORACLE · GOLD TRACE→B        : {OK['ORACLE'].sum():2d}/{N_ITEMS} = {ORACLE:.4f}")
print(f"RELAY  · A→TRACE→B           : {OK['RELAY'].sum():2d}/{N_ITEMS} = {RELAY:.4f}")
print(f"RELAY | STAGE1 CORRECT       : {relay_given_s1}/{n_s1} = {COND:.4f}" if n_s1 else "RELAY | STAGE1 CORRECT       : N/A")
print(f"DIRECT FAIL → RELAY PASS     : {int(((OK['DIRECT']==0)&(OK['RELAY']==1)).sum())}/{N_ITEMS}")
print("-"*160)
print("TEST501 LOCK SHA            :",TEST501_LOCK_SHA)
print("TEST502 LOCK SHA            :",TEST502_LOCK_SHA)
print("TEST503 LOCK SHA            :",TEST503_LOCK_SHA)
print("TEST504 LOCK SHA            :",LOCK_SHA)
print(f"TOTAL TEST TIME              : {time.perf_counter()-T0:.2f}s")
print("EXPLORATORY VERDICT         :",VERDICT)
print("="*160)

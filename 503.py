# Copyright © 2026 Mustafa Akbaş
# New original AKBASCORE MAM material is source-available under the repository LICENSE.
# See repository LICENSE for complete terms.
#
# AKBASCORE MAM — TEST503
# WRITE-TIME CONTEXT vs POSITION/LAYOUT
#
# TEST502 localized a complete two-hop failure at:
# CONTIGUOUS RAW K/V 32/32 → INDEPENDENT RAW K/V 0/32.
#
# TEST503 isolates the cause:
# B  = contiguous jointly-forged RAW K/V, native contiguous positions
# B2 = jointly-forged RAW K/V, split by record and rebased to the same
#      local-position parallel layout used by independent memories
# C  = independently-forged RAW K/V, local-position parallel layout
#
# No compression, steering, routing, training, graph, or post-hoc selection.

import os,sys,subprocess,importlib.util,random,time,hashlib,json
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache

TEST="503";SEED=501;MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N_ITEMS=32;MAX_NEW=32
DEVICE=torch.device("cuda");FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
TEST501_LOCK_SHA="1a3e6ea2a9f6c72d7b8dacb0c0844e9f2a6f8595bb232f1554eb41af7a20dc41"
TEST502_LOCK_SHA="f852b6efc6ee887bae54db08cbc11f987e749499a51cebeb5132f569e74edb89"

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
LOCK={"test":TEST,"model":MODEL_ID,"seed":SEED,"items":ITEMS,"test501":TEST501_LOCK_SHA,"test502":TEST502_LOCK_SHA,
      "arms":["B_CONTIG_RAW","B2_JOINT_FORGED_REBASED_PARALLEL","C_INDEPENDENT_RAW"],
      "compression":"OFF","steering":"OFF","routing":"OFF","training":"OFF"}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

print("="*160)
print("TEST503 — AKBASCORE MAM · WRITE-TIME CONTEXT vs POSITION/LAYOUT")
print("B CONTIG RAW → B2 JOINT-FORGED / REBASED PARALLEL → C INDEPENDENT RAW")
print("="*160)
print("LOCK SHA:",LOCK_SHA)
print("TEST501 LOCK:",TEST501_LOCK_SHA)
print("TEST502 LOCK:",TEST502_LOCK_SHA)
print("ITEMS:",N_ITEMS)
print("COMPRESSION / DRA / STEERING / ROUTER / TRAINING: NONE")
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

# Joint forging for B2:
# The exact two RECORD-labelled strings are processed in one causal forward pass.
# We then retain each record's pre-RoPE projected K/V rows separately and reinstall
# those rows with local positions exactly like C.
@torch.inference_mode()
def forge_joint_segments(records):
    labels=[f"RECORD {i+1}: {r}" for i,r in enumerate(records)]
    pieces=[]
    for i,s in enumerate(labels):
        # Same serialization as B: RECORD 1...\nRECORD 2...
        pieces.append(s if i==len(labels)-1 else s+"\n")
    piece_ids=[enc(x) for x in pieces]
    ids=[PAD]+sum(piece_ids,[])+enc(SEP)
    raw=kv_from_ids(ids)

    # PAD is excluded from record segments. Final SEP belongs to no segment.
    bounds=[];cur=1
    for p in piece_ids:
        bounds.append((cur,cur+len(p)));cur+=len(p)

    segs=[]
    for a,b in bounds:
        seg=[]
        for L in range(NL):
            k,v=raw[L]
            # Each segment gets a neutral leading PAD row so B2 has the same
            # packet form and local-position convention as independently forged C.
            kp=k[:,0:1].clone();vp=v[:,0:1].clone()
            seg.append((torch.cat([kp,k[:,a:b]],1).contiguous(),
                        torch.cat([vp,v[:,a:b]],1).contiguous()))
        segs.append(seg)
    return segs

def firstline(x):return next((z.strip() for z in str(x).splitlines() if z.strip()),"")
def norm(x):return firstline(x).strip().upper().rstrip(".")
def exact(x,t):return norm(x)==str(t).upper()
def q_for(g):return f"What routing class corresponds to instrument {g['entity']}? Follow the records and give only the exact routing class."

def bootstrap_delta(a,b,B=10000,seed=503):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64);d=a-b;n=len(d)
    rng=np.random.default_rng(seed);vals=np.empty(B)
    for i in range(B):vals[i]=d[rng.integers(0,n,n)].mean()
    return float(d.mean()),tuple(np.quantile(vals,[.025,.975]))

print("\n[2/5] Panel seal...")
for it in ITEMS[:4]:
    print(f"      ITEM {it['id']:02d} entity={it['gold']['entity']} seal={it['gold']['seal']} target={it['gold']['class']} positions={it['positions']}")
print("      TEST502 panel reproduced deterministically.")
print("      B2 changes only write-time context while matching C's reader-side local-position parallel layout.")

print("\n[3/5] Running B / B2 / C...")
R={"B":[],"B2":[],"C":[]}
for z,it in enumerate(ITEMS):
    recs=it["records"];g=it["gold"];q=q_for(g);target=g["class"]

    # B — native contiguous raw K/V.
    joint="\n".join(f"RECORD {i+1}: {r}" for i,r in enumerate(recs))
    rawB=forge(joint)
    instB,TmB,PB=install_single(rawB)
    B=read_kv(instB,TmB,PB,q)

    # B2 — records jointly forged, then segmented and locally rebased.
    joint_segments=forge_joint_segments(recs)
    instB2,TmB2,PB2=install_parallel(joint_segments)
    B2=read_kv(instB2,TmB2,PB2,q)

    # C — independently forged records with same local-position parallel reader layout.
    rawC=[forge(r) for r in recs]
    instC,TmC,PC=install_parallel(rawC)
    C=read_kv(instC,TmC,PC,q)

    vals={"B":B,"B2":B2,"C":C}
    for k,v in vals.items():R[k].append({"ok":exact(v,target),"raw":v})
    print(f"      [{z+1:02d}/{N_ITEMS}] {g['entity']:6s} target={target} | B={int(R['B'][-1]['ok'])} B2={int(R['B2'][-1]['ok'])} C={int(R['C'][-1]['ok'])}")
    if not all(R[k][-1]["ok"] for k in R):
        print(f"               B ={firstline(B)!r}")
        print(f"               B2={firstline(B2)!r}")
        print(f"               C ={firstline(C)!r}")

print("\n[4/5] Causal separation...")
OK={k:np.array([int(x["ok"]) for x in v],dtype=np.int32) for k,v in R.items()}
for k,name in [("B","CONTIG RAW"),("B2","JOINT→REBASED"),("C","INDEPENDENT RAW")]:
    print(f"      {name:16s}: {OK[k].sum():2d}/{N_ITEMS} = {OK[k].mean():.4f}")
print(f"      B pass  → B2 fail : {int(((OK['B']==1)&(OK['B2']==0)).sum())}/{N_ITEMS}")
print(f"      B2 pass → C fail  : {int(((OK['B2']==1)&(OK['C']==0)).sum())}/{N_ITEMS}")
print(f"      B∩B2 correct      : {int(((OK['B']==1)&(OK['B2']==1)).sum())}/{N_ITEMS}")
print(f"      B2∩C correct      : {int(((OK['B2']==1)&(OK['C']==1)).sum())}/{N_ITEMS}")

print("\n      Paired bootstrap accuracy differences:")
for a,b in [("B","B2"),("B2","C"),("B","C")]:
    d,ci=bootstrap_delta(OK[a],OK[b],10000,SEED+sum(map(ord,a+b)))
    print(f"      {a}-{b}: Δ={d:+.4f} | bootstrap95=[{ci[0]:+.4f},{ci[1]:+.4f}]")

print("\n[5/5] Mechanistic classification...")
B=OK["B"].mean();B2=OK["B2"].mean();C=OK["C"].mean()
if B<.75:
    VERDICT="CONTIGUOUS_BASELINE_NOT_RELIABLE"
elif B2>=.75 and C<.25:
    VERDICT="WRITE_TIME_CONTEXTUALIZATION_DEFICIT_STRONGLY_SUPPORTED"
elif B2<.25 and C<.25:
    VERDICT="POSITION_OR_READER_LAYOUT_DEFICIT_SUPPORTED"
elif B2>C+.25:
    VERDICT="MIXED_EFFECT_WRITE_TIME_CONTEXT_CONTRIBUTES"
elif C>=.75:
    VERDICT="INDEPENDENT_RAW_TWO_HOP_REPLICATED"
else:
    VERDICT="MIXED_OR_UNRESOLVED"

print("\n"+"="*160)
print("TEST503 FINAL RESULT — AKBASCORE MAM · WRITE-TIME CONTEXT vs POSITION/LAYOUT")
print("="*160)
print("MODEL                       : Qwen/Qwen2.5-7B-Instruct · frozen")
print("TASK                        : 2 edges · Entity → Seal → Class")
print("PANEL                       : TEST502 deterministic 32-item panel")
print("COMPRESSION                 : OFF")
print("DRA / STEERING              : NONE")
print("VTOKEN / ADDRESS            : OFF")
print("TOP-K / ROUTER / ANN        : NONE")
print("GRAPH                        : NONE")
print("TRAINING / LoRA             : NONE")
print("POST-HOC ITEM SELECTION     : NONE")
print("-"*160)
print(f"B  · CONTIGUOUS RAW K/V      : {OK['B'].sum():2d}/{N_ITEMS} = {B:.4f}")
print(f"B2 · JOINT-FORGED / REBASED : {OK['B2'].sum():2d}/{N_ITEMS} = {B2:.4f}")
print(f"C  · INDEPENDENT RAW K/V     : {OK['C'].sum():2d}/{N_ITEMS} = {C:.4f}")
print("-"*160)
print(f"B PASS → B2 FAIL             : {int(((OK['B']==1)&(OK['B2']==0)).sum())}/{N_ITEMS}")
print(f"B2 PASS → C FAIL             : {int(((OK['B2']==1)&(OK['C']==0)).sum())}/{N_ITEMS}")
print("-"*160)
print("TEST501 LOCK SHA            :",TEST501_LOCK_SHA)
print("TEST502 LOCK SHA            :",TEST502_LOCK_SHA)
print("TEST503 LOCK SHA            :",LOCK_SHA)
print(f"TOTAL TEST TIME              : {time.perf_counter()-T0:.2f}s")
print("EXPLORATORY VERDICT         :",VERDICT)
print("="*160)

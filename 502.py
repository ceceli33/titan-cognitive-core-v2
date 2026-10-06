# Copyright © 2026 Mustafa Akbaş
# New original AKBASCORE MAM material is source-available under the repository LICENSE.
# See repository LICENSE for complete terms.
#
# AKBASCORE MAM — TEST502
# TWO-HOP MEMORY TRANSFER
#
# NOTE:
#   TEST501 established the current native Qwen capacity baseline.
#   This test transfers a reliable two-hop relation into AKBASCORE MAM
#   and measures where performance changes across memory representations.

import os,sys,subprocess,importlib.util,random,re,time,hashlib,json
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache

TEST="502";SEED=501;MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
N_ITEMS=32;K_DIM=120;V_DIM=128;MAX_NEW=32;DEVICE=torch.device("cuda")
FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
TEST482_BLOB_SHA="ecc630144a44479cb35c432c50eb0d09bc6866d7"
TEST482_LOCK_SHA="fa59fd38661e558f6bae22eedff0999f32f8f6e9d4a08932383525323a5fe687"
TEST495_LOCK_SHA="2bae3f2497fefb50b3d18d8fb7035c63a08fdd67d80077dd92f4e9c56eca787e"
TEST501_LOCK_SHA="1a3e6ea2a9f6c72d7b8dacb0c0844e9f2a6f8595bb232f1554eb41af7a20dc41"

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
NEED=N_ITEMS*3
assert len(SEAL_BANK)>=NEED and len(CLASS_BANK)>=NEED

NEUTRAL_CORPUS=[
"The morning train arrived beside the quiet platform.","A wooden chair stood near the open window.",
"The notebook remained on the corner of the desk.","Clouds moved slowly above the distant hills.",
"The glass bottle was placed beside the metal tray.","A narrow road crossed the empty field.",
"The lamp illuminated the room during the evening.","Several books were arranged along the shelf.",
"The small garden contained stones and dry leaves.","A clock hung above the doorway in the hall.",
"The river passed beneath the old bridge.","A folded blanket rested on the wooden bench.",
"The cabinet contained several ordinary tools.","The hallway connected the rooms on both sides.",
"The paper envelope was left near the telephone.","A bicycle leaned against the outside wall.",
"The kitchen table had four simple chairs.","The path continued beyond the group of trees.",
"A small box was stored beneath the counter.","The curtain moved slightly near the window.",
"The building had a staircase near the entrance.","The cup remained beside the empty plate.",
"The field extended toward the low hills.","The office contained several desks and cabinets.",
"The door opened into a narrow corridor.","A small shelf was fixed above the sink.",
"The road turned left after the stone wall.","The room had two windows facing the street.",
"The bag was placed under the wooden table.","The ceiling light remained on during the afternoon.",
"The fence continued along the edge of the property.","The old sign stood beside the road."
]

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
LOCK={"test":TEST,"model":MODEL_ID,"seed":SEED,"items":ITEMS,"K":K_DIM,"V":V_DIM,
      "test501":TEST501_LOCK_SHA,"test482":TEST482_LOCK_SHA,"test495":TEST495_LOCK_SHA,
      "arms":["TEXT","CONTIG_RAW","INDEP_RAW","INDEP_K120V128"],"steering":"OFF","training":"OFF"}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

print("="*160)
print("TEST502 — AKBASCORE MAM · TWO-HOP MEMORY TRANSFER")
print("TEXT → CONTIGUOUS RAW K/V → INDEPENDENT RAW K/V → K120/V128 BELLEKÖZ")
print("="*160)
print("LOCK SHA:",LOCK_SHA)
print("TEST501 LOCK:",TEST501_LOCK_SHA)
print("TEST482 BLOB:",TEST482_BLOB_SHA)
print("TEST495 LOCK:",TEST495_LOCK_SHA)
print("ITEMS:",N_ITEMS)
print("DRA / STEERING: NONE")
T0=time.perf_counter()

print("\n[1/6] Loading frozen Qwen...")
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

print("\n[2/6] Building frozen TEST482 neutral PCA codebook...")
BASE=[]
for s in NEUTRAL_CORPUS:BASE.append(forge(s))
KB=[[[] for _ in range(KVH)] for _ in range(NL)]
VB=[[[] for _ in range(KVH)] for _ in range(NL)]
for L in range(NL):
    for h in range(KVH):
        Xk=torch.cat([x[L][0][h,1:].float().cpu() for x in BASE],0)
        Xv=torch.cat([x[L][1][h,1:].float().cpu() for x in BASE],0)
        Xk=Xk-Xk.mean(0,keepdim=True);Xv=Xv-Xv.mean(0,keepdim=True)
        _,_,Vk=torch.linalg.svd(Xk,full_matrices=False)
        _,_,Vv=torch.linalg.svd(Xv,full_matrices=False)
        KB[L][h]=Vk[:min(128,Vk.shape[0])].contiguous().to(DEVICE)
        VB[L][h]=Vv[:min(128,Vv.shape[0])].contiguous().to(DEVICE)
del BASE
torch.cuda.empty_cache()
print("      Codebook ready.")

@torch.inference_mode()
def compress_raw(raw):
    out=[]
    for L in range(NL):
        k,v=raw[L];T=k.shape[1]
        kk=torch.empty_like(k);vv=torch.empty_like(v)
        kk[:,0]=k[:,0];vv[:,0]=v[:,0]
        for h in range(KVH):
            if T>1:
                bk=KB[L][h][:K_DIM];bv=VB[L][h][:V_DIM]
                x=k[h,1:].float();y=v[h,1:].float()
                kk[h,1:]=((x@bk.T)@bk).to(k.dtype)
                vv[h,1:]=((y@bv.T)@bv).to(v.dtype)
        out.append((kk,vv))
    return out

def rope_k(k,pos,L):
    layer=model.model.layers[L];rot=layer.self_attn.rotary_emb if hasattr(layer.self_attn,"rotary_emb") else model.model.rotary_emb
    dummy=torch.zeros((1,k.shape[0],len(pos),HD),device=DEVICE,dtype=k.dtype)
    p=torch.tensor([pos],device=DEVICE,dtype=torch.long)
    try:cos,sin=rot(dummy,p)
    except TypeError:cos,sin=rot(dummy,position_ids=p)
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    q=dummy
    _,kr=apply_rotary_pos_emb(q,k.unsqueeze(0),cos,sin)
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
def read_text(records,q,max_new=MAX_NEW):
    prompt="\n".join(f"RECORD {i+1}: {r}" for i,r in enumerate(records))+"\n\n"+FMT.format(q=q)
    ids=torch.tensor([enc(prompt)],device=DEVICE);out=[]
    for _ in range(max_new):
        o=model(input_ids=ids,use_cache=False,return_dict=True);nxt=int(o.logits[0,-1].float().argmax())
        if nxt in EOS:break
        out.append(nxt);ids=torch.cat([ids,torch.tensor([[nxt]],device=DEVICE)],1)
    return tok.decode(out,skip_special_tokens=True).strip()

def firstline(x):return next((z.strip() for z in str(x).splitlines() if z.strip()),"")
def norm(x):return firstline(x).strip().upper().rstrip(".")
def exact(x,t):return norm(x)==str(t).upper()

def q_for(g):
    return f"What routing class corresponds to instrument {g['entity']}? Follow the records and give only the exact routing class."

def bootstrap_delta(a,b,B=10000,seed=502):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64);d=a-b;n=len(d)
    rng=np.random.default_rng(seed);vals=np.empty(B)
    for i in range(B):vals[i]=d[rng.integers(0,n,n)].mean()
    return float(d.mean()),tuple(np.quantile(vals,[.025,.975]))

print("\n[3/6] Panel seal...")
for it in ITEMS[:4]:
    print(f"      ITEM {it['id']:02d} entity={it['gold']['entity']} seal={it['gold']['seal']} target={it['gold']['class']} positions={it['positions']}")
print("      Same TEST501 two-hop task; target-row positions rotate.")

print("\n[4/6] Running TEXT / CONTIG RAW / INDEPENDENT RAW / K120-V128...")
R={"A":[],"B":[],"C":[],"D":[]}
for z,it in enumerate(ITEMS):
    recs=it["records"];g=it["gold"];q=q_for(g);target=g["class"]
    A=read_text(recs,q)

    # B: both records forged together in one causal sequence.
    # Exact text serialization matches A's memory text.
    joint="\n".join(f"RECORD {i+1}: {r}" for i,r in enumerate(recs))
    raw_joint=forge(joint)
    instB,TmB,PB=install_single(raw_joint)
    B=read_kv(instB,TmB,PB,q)

    # C: each record forged independently, raw K/V, same reader-side parallel composition.
    raw_ind=[forge(r) for r in recs]
    instC,TmC,PC=install_parallel(raw_ind)
    C=read_kv(instC,TmC,PC,q)

    # D: exact same independent packets/layout as C, only TEST482 K120/V128 compression.
    cmp_ind=[compress_raw(r) for r in raw_ind]
    instD,TmD,PD=install_parallel(cmp_ind)
    D=read_kv(instD,TmD,PD,q)

    vals={"A":A,"B":B,"C":C,"D":D}
    for k,v in vals.items():R[k].append({"ok":exact(v,target),"raw":v})
    print(f"      [{z+1:02d}/{N_ITEMS}] {g['entity']:6s} target={target} | A={int(R['A'][-1]['ok'])} B={int(R['B'][-1]['ok'])} C={int(R['C'][-1]['ok'])} D={int(R['D'][-1]['ok'])}")
    if not all(R[k][-1]["ok"] for k in R):
        print(f"               A={firstline(A)!r}")
        print(f"               B={firstline(B)!r}")
        print(f"               C={firstline(C)!r}")
        print(f"               D={firstline(D)!r}")

print("\n[5/6] Transition anatomy...")
OK={k:np.array([int(x["ok"]) for x in v],dtype=np.int32) for k,v in R.items()}
for k,name in [("A","TEXT"),("B","CONTIG RAW"),("C","INDEP RAW"),("D","K120/V128")]:
    print(f"      {name:12s}: {OK[k].sum():2d}/{N_ITEMS} = {OK[k].mean():.4f}")
print(f"      A pass → B fail : {int(((OK['A']==1)&(OK['B']==0)).sum())}/{N_ITEMS}")
print(f"      B pass → C fail : {int(((OK['B']==1)&(OK['C']==0)).sum())}/{N_ITEMS}")
print(f"      C pass → D fail : {int(((OK['C']==1)&(OK['D']==0)).sum())}/{N_ITEMS}")
print(f"      A∩B correct     : {int(((OK['A']==1)&(OK['B']==1)).sum())}/{N_ITEMS}")
print(f"      B∩C correct     : {int(((OK['B']==1)&(OK['C']==1)).sum())}/{N_ITEMS}")
print(f"      C∩D correct     : {int(((OK['C']==1)&(OK['D']==1)).sum())}/{N_ITEMS}")

print("\n      Paired bootstrap accuracy differences:")
for a,b in [("A","B"),("B","C"),("C","D")]:
    d,ci=bootstrap_delta(OK[a],OK[b],10000,SEED+ord(a)+ord(b))
    print(f"      {a}-{b}: Δ={d:+.4f} | bootstrap95=[{ci[0]:+.4f},{ci[1]:+.4f}]")

print("\n[6/6] Mechanistic classification...")
A=OK["A"].mean();B=OK["B"].mean();C=OK["C"].mean();D=OK["D"].mean()
if A<.75:
    VERDICT="BASE_TASK_NOT_RELIABLY_REPLICATED"
elif B<.75:
    VERDICT="CONTIGUOUS_KV_TRANSFER_NOT_RELIABLE"
elif C<.75:
    VERDICT="INDEPENDENT_WRITE_TWO_HOP_DEFICIT_OBSERVED"
elif D<.75:
    VERDICT="K120_V128_COMPRESSION_TWO_HOP_DEFICIT_OBSERVED"
else:
    VERDICT="TWO_HOP_INDEPENDENT_BELLEKOZ_TRANSFER_REPLICATED"

print("\n"+"="*160)
print("TEST502 FINAL RESULT — AKBASCORE MAM · TWO-HOP MEMORY TRANSFER")
print("="*160)
print("MODEL                       : Qwen/Qwen2.5-7B-Instruct · frozen")
print("TEST501 NATIVE BASELINE     : 28/32 = 0.8750")
print("RELATIONAL DEPTH            : 2 edges · Entity → Seal → Class")
print("TARGET LITERAL IN QUERY     : NO")
print("GOLD-FIRST SHORTCUT         : REMOVED / ROTATING POSITIONS")
print("DRA / STEERING              : NONE")
print("VTOKEN / ADDRESS            : OFF")
print("TOP-K / ROUTER / ANN        : NONE")
print("GRAPH                        : NONE")
print("TRAINING / LoRA             : NONE")
print("POST-HOC ITEM SELECTION     : NONE")
print("-"*160)
print(f"A · CONTIGUOUS TEXT          : {OK['A'].sum():2d}/{N_ITEMS} = {A:.4f}")
print(f"B · CONTIGUOUS RAW K/V       : {OK['B'].sum():2d}/{N_ITEMS} = {B:.4f}")
print(f"C · INDEPENDENT RAW K/V      : {OK['C'].sum():2d}/{N_ITEMS} = {C:.4f}")
print(f"D · INDEPENDENT K120/V128    : {OK['D'].sum():2d}/{N_ITEMS} = {D:.4f}")
print("-"*160)
print(f"A PASS → B FAIL              : {int(((OK['A']==1)&(OK['B']==0)).sum())}/{N_ITEMS}")
print(f"B PASS → C FAIL              : {int(((OK['B']==1)&(OK['C']==0)).sum())}/{N_ITEMS}")
print(f"C PASS → D FAIL              : {int(((OK['C']==1)&(OK['D']==0)).sum())}/{N_ITEMS}")
print("-"*160)
print("TEST482 BLOB SHA            :",TEST482_BLOB_SHA)
print("TEST482 LOCK SHA            :",TEST482_LOCK_SHA)
print("TEST495 LOCK SHA            :",TEST495_LOCK_SHA)
print("TEST501 LOCK SHA            :",TEST501_LOCK_SHA)
print("TEST502 LOCK SHA            :",LOCK_SHA)
print(f"TOTAL TEST TIME              : {time.perf_counter()-T0:.2f}s")
print("EXPLORATORY VERDICT         :",VERDICT)
print("="*160)

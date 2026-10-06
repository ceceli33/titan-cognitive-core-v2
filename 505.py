# Copyright © 2026 Mustafa Akbaş
# New original AKBASCORE MAM material is source-available under the repository LICENSE.
# See repository LICENSE for complete terms.
#
# AKBASCORE MAM — TEST505
# SELECTIVE ASSOCIATIVE RECALL IN A 10-CARTRIDGE BANK
#
# 10 independently forged RAW BELLEKÖZ per item:
#   2 required memories + 8 independent distractors.
#
# No cartridge address is supplied to the content-addressed arm.
# The model must:
#   QUERY → find MEMORY A → produce TRACE → find MEMORY B → ANSWER
#
# Controls:
#   DIRECT      = all 10 memories in one-shot parallel cache
#   ORACLE_ADDR = correct cartridge addresses supplied internally to isolate relay capacity
#   CONTENT     = exhaustive model-native content scan; no gold cartridge address
#
# IMPORTANT:
#   CONTENT is an explicit O(N) content-addressed scan, not yet sublinear addressing
#   and not yet latent/invisible traversal. Intermediate trace remains textual.
#
# No compression. No joint re-forging. No DRA/steering.
# No VTOKEN. No ANN/router. No graph. No training. No post-hoc selection.

import os,sys,subprocess,importlib.util,random,time,hashlib,json,re
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache

TEST="505";SEED=501;MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N_ITEMS=32;BANK_N=10;MAX_NEW=12
DEVICE=torch.device("cuda");FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
TEST501_LOCK_SHA="1a3e6ea2a9f6c72d7b8dacb0c0844e9f2a6f8595bb232f1554eb41af7a20dc41"
TEST502_LOCK_SHA="f852b6efc6ee887bae54db08cbc11f987e749499a51cebeb5132f569e74edb89"
TEST503_LOCK_SHA="2cb70ffa53a98efb249590dba7ef184b9e41cf72a99291bda65e6c93b82cb5de"
TEST504_LOCK_SHA="74c375373666c840105cc165b90977c56c8698cf3d5617c63f619a0139386846"

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

BASE=make_base()

# Four instrument→seal + four seal→class distractor memories.
# Current target entity, bridge seal and final class are forbidden everywhere.
def make_distractors(i,g):
    rng=random.Random(SEED+90000+i)
    ep=[e for j,t in enumerate(ENTITY_BANK) if j!=i for e in t if e.lower()!=g["entity"].lower()]
    sp=[s for s in SEAL_BANK if s!=g["seal"] and s!=g["class"]]
    cp=[c for c in CLASS_BANK if c!=g["class"] and c!=g["seal"]]
    rng.shuffle(ep);rng.shuffle(sp);rng.shuffle(cp)
    if len(ep)<12 or len(sp)<24 or len(cp)<12:raise RuntimeError(f"Insufficient distractor pool at item {i+1}.")
    ds=[]
    for d in range(4):
        es=ep[d*3:d*3+3];ss=sp[d*3:d*3+3]
        ds.append(" ".join(f"Instrument {es[j]} carries seal {ss[j]}." for j in range(3)))
    for d in range(4):
        ss=sp[12+d*3:12+d*3+3];cc=cp[d*3:d*3+3]
        ds.append(" ".join(f"Seal {ss[j]} corresponds to routing class {cc[j]}." for j in range(3)))
    if len(ds)!=8:raise RuntimeError(f"Distractor count failure at item {i+1}.")
    for x in ds:
        if token_present(x,g["entity"]):raise RuntimeError(f"Target entity contamination at item {i+1}: {x}")
        if token_present(x,g["seal"]):raise RuntimeError(f"Gold seal contamination at item {i+1}: {x}")
        if token_present(x,g["class"]):raise RuntimeError(f"Gold class contamination at item {i+1}: {x}")
    return ds

ITEMS=[]
for i,b in enumerate(BASE):
    g=b["gold"];ds=make_distractors(i,g)
    mems=[{"role":"A","text":b["records"][0]},{"role":"B","text":b["records"][1]}]+[{"role":"D","text":x} for x in ds]
    rng=random.Random(SEED+70000+i);rng.shuffle(mems)
    if len(mems)!=BANK_N:raise RuntimeError(f"Bank size failure at item {i+1}.")
    if sum(m["role"]=="A" for m in mems)!=1 or sum(m["role"]=="B" for m in mems)!=1 or sum(m["role"]=="D" for m in mems)!=8:
        raise RuntimeError(f"Role-count failure at item {i+1}.")
    ia=next(j for j,m in enumerate(mems) if m["role"]=="A")
    ib=next(j for j,m in enumerate(mems) if m["role"]=="B")
    if ia==ib:raise RuntimeError(f"Oracle address collision at item {i+1}.")
    # Required-memory integrity.
    if not token_present(mems[ia]["text"],g["entity"]) or not token_present(mems[ia]["text"],g["seal"]):
        raise RuntimeError(f"Memory A integrity failure at item {i+1}.")
    if not token_present(mems[ib]["text"],g["seal"]) or not token_present(mems[ib]["text"],g["class"]):
        raise RuntimeError(f"Memory B integrity failure at item {i+1}.")
    # Only A may contain target entity; only A/B may contain bridge; only B may contain final class.
    ent_hits=[j for j,m in enumerate(mems) if token_present(m["text"],g["entity"])]
    seal_hits=[j for j,m in enumerate(mems) if token_present(m["text"],g["seal"])]
    class_hits=[j for j,m in enumerate(mems) if token_present(m["text"],g["class"])]
    if ent_hits!=[ia]:raise RuntimeError(f"Entity uniqueness failure item {i+1}: {ent_hits} expected {[ia]}")
    if sorted(seal_hits)!=sorted([ia,ib]):raise RuntimeError(f"Bridge uniqueness failure item {i+1}: {seal_hits} expected {[ia,ib]}")
    if class_hits!=[ib]:raise RuntimeError(f"Final-class uniqueness failure item {i+1}: {class_hits} expected {[ib]}")
    ITEMS.append({**b,"bank":mems,"oracle":[ia,ib]})

LOCK={"test":TEST,"model":MODEL_ID,"seed":SEED,"items":ITEMS,"bank_n":BANK_N,"required":2,"distractors":8,
      "test501":TEST501_LOCK_SHA,"test502":TEST502_LOCK_SHA,"test503":TEST503_LOCK_SHA,"test504":TEST504_LOCK_SHA,
      "arms":["DIRECT_10","ORACLE_ADDRESS_RELAY","CONTENT_ADDRESSED_EXHAUSTIVE_RELAY"],
      "stage2_excludes_stage1_selected_memory":True,"gold_entity_distractor_contamination":False,
      "gold_bridge_distractor_contamination":False,"gold_final_distractor_contamination":False,
      "compression":"OFF","joint_reforge":"OFF","steering":"OFF","vtoken":"OFF","ann_router":"OFF","graph":"OFF","training":"OFF"}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

print("="*160)
print("TEST505 — AKBASCORE MAM · SELECTIVE ASSOCIATIVE RECALL IN A 10-CARTRIDGE BANK")
print("10 INDEPENDENT BELLEKÖZ · 2 REQUIRED + 8 DISTRACTORS · QUERY → MEMORY → TRACE → MEMORY → ANSWER")
print("="*160)
print("LOCK SHA:",LOCK_SHA)
print("TEST504 LOCK:",TEST504_LOCK_SHA)
print("ITEMS:",N_ITEMS,"| BANK:",BANK_N,"| REQUIRED: 2 | DISTRACTORS: 8")
print("COMPRESSION / JOINT RE-FORGE / DRA / VTOKEN / ANN / GRAPH / TRAINING: NONE")
print("CONTENT ADDRESSING: explicit exhaustive model-native scan O(N)")
print("PANEL GUARDS: target entity / bridge seal / final class distractor contamination = NONE")
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

# Strict first-token parser: exact three-letter symbolic code or NONE.
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
def q_oracle_a(entity):
    return f"What seal does instrument {entity} carry? Give only the exact seal."
def q_oracle_b(seal):
    return f"What routing class corresponds to seal {seal}? Give only the exact routing class."

# Exhaustive O(N) scan. excluded can suppress already-consumed memory from later traversal.
def scan(raws,q,excluded=None):
    excluded=set() if excluded is None else set(excluded)
    claims=[];outputs=[]
    for j,r in enumerate(raws):
        if j in excluded:
            outputs.append("SKIP");continue
        inst,Tm,P=install_single(r)
        y=read_kv(inst,Tm,P,q)
        v=norm(y);outputs.append(v)
        if v!="NONE":claims.append((j,v))
    if len(claims)==1:return claims[0][0],claims[0][1],claims,outputs
    return None,None,claims,outputs

def bootstrap_delta(a,b,B=10000,seed=505):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64);d=a-b;n=len(d)
    rng=np.random.default_rng(seed);vals=np.empty(B)
    for i in range(B):vals[i]=d[rng.integers(0,n,n)].mean()
    return float(d.mean()),tuple(np.quantile(vals,[.025,.975]))

print("\n[2/6] Bank seal...")
for it in ITEMS[:6]:
    print(f"      ITEM {it['id']:02d} entity={it['gold']['entity']:6s} trace={it['gold']['seal']} target={it['gold']['class']} | hidden A=M{it['oracle'][0]+1:02d} B=M{it['oracle'][1]+1:02d}")
print("      Required memories are independently forged and randomly distributed.")
print("      Content-addressed arm receives no A/B cartridge address.")
print("      Stage-2 cannot reuse the Stage-1 selected cartridge.")

print("\n[3/6] Forging and running 10-cartridge banks...")
R={"DIRECT":[],"OA1":[],"OA2":[],"ORACLE":[],"CA1_ADDR":[],"CA1_TRACE":[],"CA2_ADDR":[],"CA2_VALUE":[],"CONTENT":[]}
for z,it in enumerate(ITEMS):
    g=it["gold"];ia,ib=it["oracle"]
    raws=[forge(m["text"]) for m in it["bank"]]

    # DIRECT: all ten independently forged memories presented simultaneously.
    inst,Tm,P=install_parallel(raws)
    direct=read_kv(inst,Tm,P,q_direct(g["entity"]))

    # ORACLE ADDRESS: correct memory addresses known; intermediate remains model-produced.
    instA,TmA,PA=install_single(raws[ia])
    oa1=read_kv(instA,TmA,PA,q_oracle_a(g["entity"]))
    trO=norm(oa1)
    instB,TmB,PB=install_single(raws[ib])
    oa2=read_kv(instB,TmB,PB,q_oracle_b(trO))
    oracle_ok=exact(oa1,g["seal"]) and exact(oa2,g["class"])

    # CONTENT STAGE 1: no address supplied; scan all ten memories.
    ca1_idx,ca1_trace,claims1,outs1=scan(raws,q_find_entity(g["entity"]))
    ca1_addr_ok=ca1_idx==ia
    ca1_trace_ok=ca1_addr_ok and ca1_trace==g["seal"]

    # CONTENT STAGE 2: only Stage-1 model output moves forward.
    # Already-consumed Stage-1 memory is excluded from the second lookup.
    ca2_idx=None;ca2_val=None;claims2=[];outs2=[]
    if ca1_idx is not None and ca1_trace is not None:
        ca2_idx,ca2_val,claims2,outs2=scan(raws,q_find_seal(ca1_trace),excluded={ca1_idx})
    ca2_addr_ok=ca2_idx==ib
    ca2_value_ok=ca2_addr_ok and ca2_val==g["class"]
    content_ok=ca1_trace_ok and ca2_value_ok

    R["DIRECT"].append(int(exact(direct,g["class"])))
    R["OA1"].append(int(exact(oa1,g["seal"])))
    R["OA2"].append(int(exact(oa2,g["class"])))
    R["ORACLE"].append(int(oracle_ok))
    R["CA1_ADDR"].append(int(ca1_addr_ok))
    R["CA1_TRACE"].append(int(ca1_trace_ok))
    R["CA2_ADDR"].append(int(ca2_addr_ok))
    R["CA2_VALUE"].append(int(ca2_value_ok))
    R["CONTENT"].append(int(content_ok))

    print(f"      [{z+1:02d}/{N_ITEMS}] {g['entity']:6s} {g['seal']}->{g['class']} | DIRECT={R['DIRECT'][-1]} ORACLE={R['ORACLE'][-1]} CA1={R['CA1_ADDR'][-1]}/{R['CA1_TRACE'][-1]} CA2={R['CA2_ADDR'][-1]}/{R['CA2_VALUE'][-1]} E2E={R['CONTENT'][-1]}")
    if not content_ok:
        c1=",".join(f"M{j+1}:{v}" for j,v in claims1) if claims1 else "NONE"
        c2=",".join(f"M{j+1}:{v}" for j,v in claims2) if claims2 else "NONE"
        print(f"               hidden=M{ia+1:02d}->M{ib+1:02d} | direct={norm(direct)!r} | oracle={norm(oa1)!r}->{norm(oa2)!r}")
        print(f"               scan1=[{c1}]")
        print(f"               scan2=[{c2}]")

print("\n[4/6] Selective-recall anatomy...")
OK={k:np.asarray(v,dtype=np.int32) for k,v in R.items()}
for k,name in [
    ("DIRECT","DIRECT 10 ONE-SHOT"),
    ("OA1","ORACLE A→TRACE"),
    ("OA2","ORACLE TRACE→B"),
    ("ORACLE","ORACLE ADDR E2E"),
    ("CA1_ADDR","CONTENT A ADDRESS"),
    ("CA1_TRACE","CONTENT A+TRACE"),
    ("CA2_ADDR","CONTENT B ADDRESS"),
    ("CA2_VALUE","CONTENT B+VALUE"),
    ("CONTENT","CONTENT E2E")
]:
    print(f"      {name:20s}: {OK[k].sum():2d}/{N_ITEMS} = {OK[k].mean():.4f}")

print(f"      DIRECT fail → CONTENT pass : {int(((OK['DIRECT']==0)&(OK['CONTENT']==1)).sum())}/{N_ITEMS}")
print(f"      ORACLE pass → CONTENT fail : {int(((OK['ORACLE']==1)&(OK['CONTENT']==0)).sum())}/{N_ITEMS}")

print("\n      Paired bootstrap accuracy differences:")
for a,b in [("CONTENT","DIRECT"),("ORACLE","CONTENT"),("CA1_ADDR","CA2_ADDR")]:
    d,ci=bootstrap_delta(OK[a],OK[b],10000,SEED+sum(map(ord,a+b)))
    print(f"      {a}-{b}: Δ={d:+.4f} | bootstrap95=[{ci[0]:+.4f},{ci[1]:+.4f}]")

print("\n[5/6] Failure localization...")
n_or=int(OK["ORACLE"].sum());n_c1=int(OK["CA1_TRACE"].sum())
print(f"      Relay capacity with known addresses : {n_or}/{N_ITEMS}")
print(f"      First associative lookup complete   : {n_c1}/{N_ITEMS}")
print(f"      Second address correct              : {OK['CA2_ADDR'].sum()}/{N_ITEMS}")
print(f"      Second address + value correct      : {OK['CA2_VALUE'].sum()}/{N_ITEMS}")
if n_c1:
    both=int(((OK["CA1_TRACE"]==1)&(OK["CA2_VALUE"]==1)).sum())
    print(f"      B complete | correct first trace    : {both}/{n_c1} = {both/n_c1:.4f}")
if n_or:
    e2e_or=int(((OK["ORACLE"]==1)&(OK["CONTENT"]==1)).sum())
    print(f"      CONTENT E2E | oracle relay possible : {e2e_or}/{n_or} = {e2e_or/n_or:.4f}")

print("\n[6/6] Mechanistic classification...")
DIRECT=OK["DIRECT"].mean();ORACLE=OK["ORACLE"].mean();AADDR=OK["CA1_ADDR"].mean()
ATRACE=OK["CA1_TRACE"].mean();BADDR=OK["CA2_ADDR"].mean();BVALUE=OK["CA2_VALUE"].mean();CONTENT=OK["CONTENT"].mean()
if ORACLE<.75:
    VERDICT="TEN_BANK_RELAY_CAPACITY_NOT_RELIABLE"
elif ATRACE<.75:
    VERDICT="FIRST_CONTENT_ADDRESSING_NOT_RELIABLE"
elif BVALUE<.75:
    VERDICT="TRACE_TO_NEXT_MEMORY_ADDRESSING_NOT_RELIABLE"
elif CONTENT>=.75:
    VERDICT="TEN_BANK_SELECTIVE_ASSOCIATIVE_RELAY_REPLICATED"
elif CONTENT>DIRECT+.25:
    VERDICT="TEN_BANK_SELECTIVE_ASSOCIATIVE_RELAY_PARTIAL"
else:
    VERDICT="TEN_BANK_CONTENT_ADDRESSED_RELAY_NOT_YET_RELIABLE"

print("\n"+"="*160)
print("TEST505 FINAL RESULT — AKBASCORE MAM · SELECTIVE ASSOCIATIVE RECALL IN A 10-CARTRIDGE BANK")
print("="*160)
print("MODEL                       : Qwen/Qwen2.5-7B-Instruct · frozen")
print("BANK                        : 10 independently forged RAW BELLEKÖZ")
print("REQUIRED                    : 2")
print("DISTRACTORS                 : 8")
print("REQUIRED MEMORY POSITIONS   : deterministic random / hidden from CONTENT arm")
print("TASK                        : Entity → Seal → Class")
print("FINAL ANSWER IN QUERY       : NO")
print("TARGET ENTITY IN DISTRACTORS: NO")
print("BRIDGE SEAL IN DISTRACTORS  : NO")
print("FINAL CLASS IN DISTRACTORS  : NO")
print("JOINT RE-FORGE              : NONE")
print("COMPRESSION                 : OFF")
print("DRA / STEERING              : NONE")
print("VTOKEN / ADDRESS VECTOR     : OFF")
print("ANN / LEARNED ROUTER        : NONE")
print("GRAPH                        : NONE")
print("TRAINING / LoRA             : NONE")
print("POST-HOC ITEM SELECTION     : NONE")
print("CONTENT SEARCH              : explicit exhaustive O(N) model-native scan")
print("INTERMEDIATE                : model-produced textual trace")
print("STAGE-2 REUSES STAGE-1 MEM  : NO")
print("-"*160)
print(f"DIRECT · 10 ONE-SHOT         : {OK['DIRECT'].sum():2d}/{N_ITEMS} = {DIRECT:.4f}")
print(f"ORACLE ADDRESS RELAY         : {OK['ORACLE'].sum():2d}/{N_ITEMS} = {ORACLE:.4f}")
print(f"CONTENT · FIRST ADDRESS      : {OK['CA1_ADDR'].sum():2d}/{N_ITEMS} = {AADDR:.4f}")
print(f"CONTENT · FIRST TRACE        : {OK['CA1_TRACE'].sum():2d}/{N_ITEMS} = {ATRACE:.4f}")
print(f"CONTENT · SECOND ADDRESS     : {OK['CA2_ADDR'].sum():2d}/{N_ITEMS} = {BADDR:.4f}")
print(f"CONTENT · SECOND VALUE       : {OK['CA2_VALUE'].sum():2d}/{N_ITEMS} = {BVALUE:.4f}")
print(f"CONTENT · END-TO-END         : {OK['CONTENT'].sum():2d}/{N_ITEMS} = {CONTENT:.4f}")
print(f"DIRECT FAIL → CONTENT PASS   : {int(((OK['DIRECT']==0)&(OK['CONTENT']==1)).sum())}/{N_ITEMS}")
print("-"*160)
print("TEST501 LOCK SHA            :",TEST501_LOCK_SHA)
print("TEST502 LOCK SHA            :",TEST502_LOCK_SHA)
print("TEST503 LOCK SHA            :",TEST503_LOCK_SHA)
print("TEST504 LOCK SHA            :",TEST504_LOCK_SHA)
print("TEST505 LOCK SHA            :",LOCK_SHA)
print(f"TOTAL TEST TIME              : {time.perf_counter()-T0:.2f}s")
print("EXPLORATORY VERDICT         :",VERDICT)
print("="*160)

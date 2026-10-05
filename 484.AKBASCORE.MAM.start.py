# TEST484 — AKBASCORE MAM · MODEL-NATIVE ADDRESS SIGNAL DISCOVERY
# Model-Native Associative Memory
#
# Copyright © 2026 Mustafa Akbaş
# New original AKBASCORE MAM material is source-available under the repository LICENSE.
# Earlier MIT-licensed Titan Cognitive Core / AkbasCore material remains under its original license.
# See LICENSE for complete terms.
#
# FROZEN FOUNDATION : TEST482 Qwen2.5-7B-Instruct · K120/V128/OWN · exact 32-sentence PCA codebook
# PURPOSE           : test whether frozen Qwen geometry contains a natural signal identifying the relevant cartridge.
# BANK              : 32 controlled TEST482 Stage-1 records, one independent compressed cartridge each.
# QUERY             : natural paraphrased object questions fixed before measurement.
# MEASUREMENT       : passive only — RAW-K, COMP-K, HIDDEN, native Q->K.
# NO cartridge readout · NO full-scan LM generation · NO router · NO ANN · NO training · NO LoRA · NO gradients.
# NO model/motor changes · NO post-hoc query/config tuning.

import os,sys,subprocess,importlib.util,random,re,time,hashlib,json,math
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate"),("numpy","numpy")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM

TEST="484";SEED=484;MODEL_ID="Qwen/Qwen2.5-7B-Instruct";K_DIM=120;V_DIM=128;SEP="\n\n"
LAYERS=[3,7,11,15,19,23,27];DEVICE=torch.device("cuda")
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

# EXACT TEST482 NEUTRAL PCA CORPUS
CORPUS=[
"Jonas Weber carried the wooden crate across the quiet market square.",
"Priya Nair wrote a long letter to her cousin in the morning.",
"The small boat drifted slowly toward the rocky shore.",
"Omar Haddad fixed the broken clock in the village school.",
"A tired teacher closed the green door of the library.",
"Sofia Rossi baked fresh bread for the harvest festival.",
"The children watched the kites rising above the hill.",
"Liam O'Connor sold his old bicycle to a neighbor.",
"Heavy rain flooded the narrow street near the market.",
"Nadia Petrova translated the ancient manuscript into French.",
"The farmer counted the sheep before sunset.",
"Hiro Sato opened a tiny bakery beside the river.",
"An old dog slept under the kitchen table all afternoon.",
"Carlos Mendes washed the windows of his grandmother's house.",
"The museum guard locked the heavy gate at midnight.",
"Fatima Zahra grew tomatoes in the backyard garden.",
"The pilot announced a short delay because of fog.",
"Anna Kowalski found a lost wallet on the bus.",
"Snow covered the mountain village during the night.",
"Ravi Kumar taught his brother how to play chess.",
"The chef sharpened every knife before dinner service.",
"Lucas Martin cleaned the roof of the barn after the storm.",
"A young violinist practiced scales in the empty hall.",
"Mei Lin delivered the package to the wrong address.",
"Marek Novak placed the glass bottle beneath the wooden bench.",
"Sara Ibrahim carried a red notebook into the quiet classroom.",
"Noah Schmidt repaired the small radio beside the kitchen window.",
"Yuki Mori left a paper envelope near the station entrance.",
"Amira Hassan moved the ceramic bowl onto the upper shelf.",
"Peter Novak opened the metal box behind the old theater.",
"Lucia Costa placed the yellow scarf inside the travel bag.",
"Daniel Kim carried a black umbrella through the central courtyard."
]

# EXACT TEST482 HELD-OUT RECORD IDENTITIES
RECORDS=[
("F1","emerald barometer","KR-214","cedar archive"),
("F1","porcelain compass","DM-763","marble annex"),
("F1","violet harmonica","FS-408","amber gallery"),
("F1","bamboo chronometer","JN-951","fern vault"),
("F1","obsidian sextant","RW-326","copper archive"),
("F1","linen telescope","AE-875","quartz annex"),
("F1","coral metronome","LP-143","ivory gallery"),
("F1","willow calculator","XM-692","silver vault"),
("F2","turquoise monocle","HC-517","indigo lighthouse"),
("F2","mahogany notebook","VK-280","pearl pavilion"),
("F2","ceramic astrolabe","SB-934","granite lodge"),
("F2","scarlet blueprint","NT-461","willow gallery"),
("F2","opal kaleidoscope","GY-708","maple tower"),
("F2","bronze hourglass","PD-352","coral pavilion"),
("F2","silk manuscript","ZU-819","birch lodge"),
("F2","crystal chime","EF-625","onyx gallery"),
("F3","jade projector","WL-407","hazel chamber"),
("F3","glass clarinet","CQ-586","lagoon studio"),
("F3","canvas diary","MR-172","acorn room"),
("F3","iron medallion","UX-943","poplar hall"),
("F3","pearl camera","BH-650","cobalt chamber"),
("F3","steel accordion","KO-238","canal studio"),
("F3","cotton almanac","YD-714","walnut room"),
("F3","onyx brooch","TG-569","elm hall"),
("F4","ivory receiver","PS-381","summit workshop"),
("F4","cedar plaque","AL-826","valley depot"),
("F4","azure kettle","RF-504","birch observatory"),
("F4","brass caliper","MW-197","shore conservatory"),
("F4","crystal phonograph","DE-648","forest workshop"),
("F4","maple slate","KI-275","delta depot"),
("F4","white lantern","OV-930","clover observatory"),
("F4","golden protractor","XC-412","island conservatory")
]

def source(fam,obj,cid,place):
    if fam=="F1":return f"The {obj} is stored in container {cid}."
    if fam=="F2":return f"Container {cid} contains the {obj}."
    if fam=="F3":return f"The {obj} can be found inside container {cid}."
    if fam=="F4":return f"Inside container {cid} there is the {obj}."
    raise ValueError(fam)

# FIXED BEFORE MODEL MEASUREMENT.
# Object names remain necessary semantic anchors, while wording differs from source templates.
QUERY_TEMPLATES=[
"Where was the {obj} put?",
"Which container holds the {obj}?",
"Where can the {obj} be found?",
"Do you know the container holding the {obj}?",
"In which container is the {obj} kept?",
"Where is the {obj} being kept?",
"Which container should I look in for the {obj}?",
"Where was the {obj} stored?"
]
QUERIES=[QUERY_TEMPLATES[i%len(QUERY_TEMPLATES)].format(obj=r[1]) for i,r in enumerate(RECORDS)]
SOURCES=[source(*r) for r in RECORDS]

LOCK={"test":TEST,"seed":SEED,"model":MODEL_ID,"K":K_DIM,"V":V_DIM,"layers":LAYERS,
      "corpus":CORPUS,"records":RECORDS,"sources":SOURCES,"queries":QUERIES}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()

print("="*150)
print("TEST484 — AKBASCORE MAM · MODEL-NATIVE ADDRESS SIGNAL DISCOVERY")
print("32 FROZEN NIRVANA CARTRIDGES · PASSIVE GEOMETRY · NO LM CARTRIDGE READOUT")
print("="*150)
print("LOCK SHA:",LOCK_SHA);T0=time.perf_counter()

# ----------------------------------------------------------------------------------------------------------------------
# 1. LOAD EXACT TEST482 MODEL FOUNDATION
# ----------------------------------------------------------------------------------------------------------------------
print("[1/7] Loading frozen Qwen...")
_tv=tuple(int("".join(c for c in x if c.isdigit())or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;layers=model.model.layers
H=cfg.hidden_size;NH=cfg.num_attention_heads;NKV=cfg.num_key_value_heads
HD=getattr(cfg,"head_dim",None) or H//NH;KVD=NKV*HD;NL=len(layers);GQA=NH//NKV
if (NL,H,NH,NKV,HD)!=(28,3584,28,4,128):raise RuntimeError(f"Architecture mismatch: {(NL,H,NH,NKV,HD)}")
if NH%NKV:raise RuntimeError("Unexpected GQA layout.")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
enc=lambda s:tok(s,add_special_tokens=False).input_ids
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | {NL}L | frozen BF16")

# ----------------------------------------------------------------------------------------------------------------------
# 2. TEST482 FORGE + PASSIVE Q/H INSTRUMENTATION
# ----------------------------------------------------------------------------------------------------------------------
@torch.inference_mode()
def states_from_ids(ids):
    o=model(input_ids=torch.tensor([ids],device=DEVICE),output_hidden_states=True,use_cache=False,return_dict=True)
    Q=[];K=[];V=[];HS=[]
    for L in range(NL):
        h=o.hidden_states[L][0];z=layers[L].input_layernorm(h);a=layers[L].self_attn
        Q.append(a.q_proj(z).contiguous())
        K.append(a.k_proj(z).contiguous())
        V.append(a.v_proj(z).contiguous())
        HS.append(h.contiguous())
    return Q,K,V,HS

def forge_full(s):return states_from_ids([PAD]+enc(s+SEP))
def forge(s):
    _,K,V,_=forge_full(s)
    return K,V

# ----------------------------------------------------------------------------------------------------------------------
# 3. EXACT TEST482 PCA CODEBOOK
# ----------------------------------------------------------------------------------------------------------------------
print("[2/7] Building frozen TEST482 32-sentence PCA codebook...")
CB=[];CO=[forge(s) for s in CORPUS]
for L in range(NL):
    e={}
    for j,n in enumerate(("K","V")):
        R=torch.cat([c[j][L][1:] for c in CO]).float().view(-1,NKV,HD);MU=[];BB=[]
        for h in range(NKV):
            X=R[:,h];mu=X.mean(0);_,_,Vh=torch.linalg.svd(X-mu,full_matrices=False)
            m=min(128,Vh.shape[0]);b=Vh[:m].T.contiguous()
            if m<128:b=torch.nn.functional.pad(b,(0,128-m))
            MU.append(mu);BB.append(b)
        e[n]=(torch.stack(MU),torch.stack(BB))
    CB.append(e)
del CO;torch.cuda.empty_cache()
print("      Codebook ready.")

@torch.inference_mode()
def packet_from_raw(K,V):
    out={}
    for n,X,d in (("K",K,K_DIM),("V",V,V_DIM)):
        rows=[]
        for L in range(NL):
            mu,B=CB[L][n]
            coeff=torch.einsum("thi,hid->thd",X[L][1:].float().view(-1,NKV,HD)-mu,B[:,:,:d]).to(torch.bfloat16)
            content=(mu+torch.einsum("thd,hid->thi",coeff.float(),B[:,:,:d])).reshape(-1,KVD).to(torch.bfloat16)
            rows.append(torch.cat([X[L][:1],content]))
        out[n]=rows
    return out["K"],out["V"]

# ----------------------------------------------------------------------------------------------------------------------
# 4. PASSIVE ADDRESS REPRESENTATIONS
# ----------------------------------------------------------------------------------------------------------------------
def kvmean(x):return x[1:].float().view(-1,NKV,HD).mean(0).reshape(-1).cpu()
def hmean(x):return x[1:].float().mean(0).cpu()
def qnative(x):return x[1:].float().view(-1,NH,HD).mean(0).reshape(-1).cpu()
def knative(x):
    z=x[1:].float().view(-1,NKV,HD).mean(0)
    return z.repeat_interleave(GQA,dim=0).reshape(-1).cpu()

FAMS=["KRAW","KCMP","HMEAN","QK_NATIVE"]
BANK={f:{L:[] for L in LAYERS} for f in FAMS}
QRY={f:{L:[] for L in LAYERS} for f in FAMS}

print("[3/7] Forging 32 frozen NIRVANA cartridges...")
for i,s in enumerate(SOURCES):
    Q,K,V,HS=forge_full(s);KC,_=packet_from_raw(K,V)
    for L in LAYERS:
        BANK["KRAW"][L].append(kvmean(K[L]))
        BANK["KCMP"][L].append(kvmean(KC[L]))
        BANK["HMEAN"][L].append(hmean(HS[L]))
        BANK["QK_NATIVE"][L].append(knative(K[L]))
    del Q,K,V,HS,KC
    print(f"      [{i+1:02d}/32] {RECORDS[i][1]}")

for f in FAMS:
    for L in LAYERS:BANK[f][L]=torch.stack(BANK[f][L]).float()

print("[4/7] Encoding 32 cartridge-free queries...")
for i,q in enumerate(QUERIES):
    Q,K,V,HS=forge_full(q);KC,_=packet_from_raw(K,V)
    for L in LAYERS:
        QRY["KRAW"][L].append(kvmean(K[L]))
        QRY["KCMP"][L].append(kvmean(KC[L]))
        QRY["HMEAN"][L].append(hmean(HS[L]))
        QRY["QK_NATIVE"][L].append(qnative(Q[L]))
    del Q,K,V,HS,KC

for f in FAMS:
    for L in LAYERS:QRY[f][L]=torch.stack(QRY[f][L]).float()

# ----------------------------------------------------------------------------------------------------------------------
# 5. FIXED DEV / HELD SPLIT
# ----------------------------------------------------------------------------------------------------------------------
# 16 DEV / 16 HELD. Configuration selection sees DEV only.
idx=np.arange(32);rng=np.random.default_rng(SEED);rng.shuffle(idx)
DEV=sorted(int(x) for x in idx[:16]);HELD=sorted(int(x) for x in idx[16:])
SPLIT_SHA=hashlib.sha256(json.dumps({"DEV":DEV,"HELD":HELD},separators=(",",":")).encode()).hexdigest()
print("[5/7] Fixed split | DEV:",DEV)
print("                    HELD:",HELD)
print("                    SHA :",SPLIT_SHA)

def normalize_rows(x):return torch.nn.functional.normalize(x.float(),dim=-1)
def score(q,B,metric,center):
    if metric=="COS":return normalize_rows(B)@torch.nn.functional.normalize(q.float(),dim=0)
    if metric=="CENTER_COS":return normalize_rows(B-center)@torch.nn.functional.normalize(q.float()-center,dim=0)
    if metric=="L2":return -torch.sum((B.float()-q.float())**2,dim=1)
    raise ValueError(metric)

def rankof(s,gold):
    o=torch.argsort(s,descending=True);p=(o==gold).nonzero(as_tuple=False)
    return int(p[0,0])+1

def summary(r):
    a=np.asarray(r)
    return {"R1":float(np.mean(a<=1)),"R5":float(np.mean(a<=5)),
            "R16":float(np.mean(a<=16)),"MRR":float(np.mean(1/a)),
            "MEDR":float(np.median(a))}

# ----------------------------------------------------------------------------------------------------------------------
# 6. DEV DISCOVERY
# ----------------------------------------------------------------------------------------------------------------------
print("[6/7] DEV address-signal scan...")
RESULTS=[]
for f in FAMS:
    for L in LAYERS:
        B=BANK[f][L];C=B.mean(0)
        for m in ("COS","CENTER_COS","L2"):
            ranks=[];margins=[]
            for i in DEV:
                s=score(QRY[f][L][i],B,m,C);w=torch.cat((s[:i],s[i+1:]))
                ranks.append(rankof(s,i));margins.append(float(s[i]-w.max()))
            st=summary(ranks)
            RESULTS.append({"fam":f,"layer":L,"metric":m,**st,"margin":float(np.median(margins))})

RESULTS.sort(key=lambda x:(x["R1"],x["R5"],x["MRR"],x["margin"]),reverse=True)
TOP=RESULTS[:3]
for n,x in enumerate(TOP,1):print(f"      TOP{n}: {x}")
CONFIG_SHA=hashlib.sha256(json.dumps(TOP,sort_keys=True,separators=(",",":")).encode()).hexdigest()

# ----------------------------------------------------------------------------------------------------------------------
# 7. LOCKED HELD-OUT
# ----------------------------------------------------------------------------------------------------------------------
print("[7/7] Locked HELD-OUT validation...")
REPORT=[]
for n,c in enumerate(TOP,1):
    f,L,m=c["fam"],c["layer"],c["metric"];B=BANK[f][L];C=B.mean(0)
    ranks=[];margins=[];zs=[]
    print(f"\n      CONFIG {n}: {f} L{L:02d} {m}")
    for i in HELD:
        s=score(QRY[f][L][i],B,m,C);w=torch.cat((s[:i],s[i+1:]))
        gs=float(s[i]);wm=float(w.max());sd=float(w.std(unbiased=False))+1e-12
        r=rankof(s,i);ranks.append(r);margins.append(gs-wm);zs.append((gs-float(w.mean()))/sd)
        print(f"        [{i:02d}] rank={r:2d} margin={gs-wm:+.6f} | {RECORDS[i][1]}")
    st=summary(ranks)
    row={"config":n,**c,"HELD":st,"median_margin":float(np.median(margins)),
         "p05_margin":float(np.quantile(margins,.05)),"median_gold_z":float(np.median(zs))}
    REPORT.append(row)
    print("        =>",row)

# PERMUTATION NULL ON LOCKED BEST CONFIG
best=TOP[0];f,L,m=best["fam"],best["layer"],best["metric"];B=BANK[f][L];C=B.mean(0)
rng=np.random.default_rng(SEED+1);pmap=np.arange(32);rng.shuffle(pmap)
REAL=[];NULL=[]
for i in HELD:
    s=score(QRY[f][L][i],B,m,C)
    REAL.append(rankof(s,i));NULL.append(rankof(s,int(pmap[i])))
REAL=summary(REAL);NULL=summary(NULL)

# SIMPLE LEXICAL CONTROL — NOT USED FOR SELECTION
def ws(s):return set(re.findall(r"[a-z0-9]+",s.casefold()))
SW=[ws(s) for s in SOURCES];LEX=[]
for i in HELD:
    q=ws(QUERIES[i]);v=np.array([len(q&x)/max(1,len(q|x)) for x in SW])
    order=np.argsort(-v,kind="stable");LEX.append(int(np.where(order==i)[0][0])+1)
LEX=summary(LEX)

print("\n"+"="*150)
print("TEST484 FINAL RESULT — AKBASCORE MAM")
print("="*150)
print("FOUNDATION          : TEST482 QWEN K120/V128/OWN")
print("TEST482 BLOB SHA    : ecc630144a44479cb35c432c50eb0d09bc6866d7")
print("CARTRIDGES          : 32")
print("DEV / HELD          : 16 / 16")
print("LOCK SHA            :",LOCK_SHA)
print("SPLIT SHA           :",SPLIT_SHA)
print("CONFIG SHA          :",CONFIG_SHA)
print("BEST DEV            :",TOP[0])
print("BEST HELD           :",REPORT[0])
print("HELD REAL           :",REAL)
print("PERMUTATION NULL    :",NULL)
print("LEXICAL CONTROL     :",LEX)
print("-"*150)

if REAL["R1"]>=.50 and REAL["R5"]>=.80 and REAL["MRR"]>NULL["MRR"]+.30:
    VERDICT="STRONG_INITIAL_MODEL_NATIVE_ADDRESS_SIGNAL"
elif REAL["R5"]>=.50 and REAL["MRR"]>NULL["MRR"]+.15:
    VERDICT="PARTIAL_INITIAL_MODEL_NATIVE_ADDRESS_SIGNAL"
else:
    VERDICT="NO_CLEAR_INITIAL_MODEL_NATIVE_ADDRESS_SIGNAL"

print("VERDICT             :",VERDICT)
print("INTERPRETATION      : first-signal discovery only; no scale claim.")
print("CARTRIDGE LM READOUT: NOT USED")
print("ROUTER / ANN / GRAPH: NONE")
print("TRAINING / LoRA     : NONE")
print("MODEL WEIGHT CHANGE : NONE")
print("NIRVANA MOTOR CHANGE: NONE")
print("HELD CONFIG TUNING  : NONE")
print(f"TOTAL TEST TIME     : {time.perf_counter()-T0:.2f}s")
print("="*150)

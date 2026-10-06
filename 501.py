# Copyright © 2026 Mustafa Akbaş
# New original AKBASCORE MAM material is source-available under the repository LICENSE.
# See repository LICENSE for complete terms.
#
# AKBASCORE MAM — TEST501
# QWEN BASE CAPACITY CALIBRATION
#
# NOTE:
#   This test measures the current native capacity of frozen Qwen across increasing
#   relational depths. The resulting capacity boundary will be used as the baseline
#   for subsequent AKBASCORE MAM system tests.

import os,sys,subprocess,importlib.util,random,re,time,hashlib,json
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM

TEST="501";SEED=501;MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
N_ITEMS=32;MAX_NEW=32;DEVICE=torch.device("cuda")
FMT="QUESTION:\n{q}\n\nANSWER:"

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
SECTION_BANK=list("QWERTYUIOPASDFGHJKLZXCVBNM")
BATCH_BANK=["SEL","DOR","NIM","PAZ","RIL","VOK","JEN","KUR","FAS","GEM","HUR","LIX","MON","PEV","RAX","SIV","TOL","WEN","YAR","ZEK","BIM","COV","DUL","FEN","GIR","HAL","JOS","KEM","LUR","NAV","PIR","ROV","SAX","TEV","VIL","WOR","XEN","YUM","ZAL","BER","CIN","DAK","ELM","FOV","GUN","HES","JAR","KIV","LEN","MOR","NEX","PUL","RIM","SEN","TAV","VOR","WIL","XAR","YEN","ZUR","BOL","CER","DIN","FAL","GER","HIN","JOV","KEL","LUM","MAR","NOR","PEL","RUS","SIM","TOR","VEX","WAL","XIR","YAV","ZEN","BUN","CAL","DEM","EVI","FIR","GOS","HAR","IVO","JUN","KAS","LEV","MUR","OVA","PAR","RIN","SOL"]
BAY_BANK=[str(x) for x in range(101,197)]

NEED=N_ITEMS*3
assert len(ENTITY_BANK)>=N_ITEMS
for name,x in [("SEAL",SEAL_BANK),("CLASS",CLASS_BANK),("BATCH",BATCH_BANK),("BAY",BAY_BANK)]:
    assert len(x)>=NEED,f"{name}_BANK requires {NEED}, found {len(x)}"

os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
torch.set_grad_enabled(False)
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

def reorder(rows,gold,target_pos,seed):
    g=rows[gold];rest=[x for j,x in enumerate(rows) if j!=gold]
    rr=random.Random(seed);rr.shuffle(rest);out=rest[:];out.insert(target_pos,g)
    return out

def make_items():
    rng=random.Random(SEED);seals=SEAL_BANK.copy();classes=CLASS_BANK.copy();batches=BATCH_BANK.copy();bays=BAY_BANK.copy()
    rng.shuffle(seals);rng.shuffle(classes);rng.shuffle(batches);rng.shuffle(bays);items=[]
    for i in range(N_ITEMS):
        ents=list(ENTITY_BANK[i]);g=i%3
        ss=seals[3*i:3*i+3];cc=classes[3*i:3*i+3];bb=batches[3*i:3*i+3];yy=bays[3*i:3*i+3]
        sec=[SECTION_BANK[(3*i+j)%len(SECTION_BANK)] for j in range(3)]
        assert all(len(x)==3 for x in [ents,ss,cc,bb,yy,sec])
        rows=[
            [f"Instrument {ents[j]} carries seal {ss[j]}." for j in range(3)],
            [f"Seal {ss[j]} corresponds to routing class {cc[j]}." for j in range(3)],
            [f"Routing class {cc[j]} is assigned archive section {sec[j]}." for j in range(3)],
            [f"Instrument {ents[j]} belongs to transfer batch {bb[j]}." for j in range(3)],
            [f"Transfer batch {bb[j]} uses active bay {yy[j]}." for j in range(3)]
        ]
        records=[]
        for r in range(5):
            pos=(i+r)%3
            records.append(" ".join(reorder(rows[r],g,pos,SEED+i*101+r*17)))
        gold={"entity":ents[g],"seal":ss[g],"class":cc[g],"section":sec[g],"batch":bb[g],"bay":yy[g]}
        items.append({"id":i+1,"records":records,"gold":gold})
    return items

ITEMS=make_items()
LOCK={"test":TEST,"model":MODEL_ID,"seed":SEED,"items":ITEMS,
      "purpose":"Qwen native relational-depth capacity calibration",
      "mam_memory":"OFF","compression":"OFF","steering":"OFF","training":"OFF"}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

print("="*150)
print("AKBASCORE MAM — TEST501")
print("QWEN BASE CAPACITY CALIBRATION")
print("="*150)
print("LOCK SHA:",LOCK_SHA)
print("ITEMS:",N_ITEMS)
T0=time.perf_counter()

print("\n[1/4] Loading frozen Qwen...")
_tv=tuple(int("".join(c for c in x if c.isdigit())or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
_ge=model.generation_config.eos_token_id
EOS=sorted({tok.eos_token_id,*(_ge if isinstance(_ge,(list,tuple)) else [_ge])}-{None})
enc=lambda s:tok(s,add_special_tokens=False).input_ids
cfg=model.config
if (len(model.model.layers),cfg.hidden_size,cfg.num_attention_heads,cfg.num_key_value_heads)!=(28,3584,28,4):
    raise RuntimeError("Architecture mismatch.")
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | frozen BF16")

@torch.inference_mode()
def generate(prompt,max_new=MAX_NEW):
    ids=torch.tensor([enc(prompt)],device=DEVICE);out=[]
    for _ in range(max_new):
        o=model(input_ids=ids,use_cache=False,return_dict=True)
        nxt=int(o.logits[0,-1].float().argmax())
        if nxt in EOS:break
        out.append(nxt);ids=torch.cat([ids,torch.tensor([[nxt]],device=DEVICE)],1)
    return tok.decode(out,skip_special_tokens=True).strip()

def context(records):
    return "\n".join(f"RECORD {i+1}: {x}" for i,x in enumerate(records))

def ask(records,q):
    return generate(context(records)+"\n\n"+FMT.format(q=q))

def firstline(x):return next((z.strip() for z in str(x).splitlines() if z.strip()),"")
def norm(x):return firstline(x).strip().upper().rstrip(".")
def exact(x,target):return norm(x)==str(target).upper()

LEVELS=[
("L1_ENTITY_TO_SEAL",[0],lambda g:f"What seal does instrument {g['entity']} carry? Give only the exact seal.",lambda g:g["seal"]),
("L2_ENTITY_TO_CLASS",[0,1],lambda g:f"What routing class corresponds to instrument {g['entity']}? Follow the records and give only the exact routing class.",lambda g:g["class"]),
("L3_ENTITY_TO_SECTION",[0,1,2],lambda g:f"What archive section is assigned to instrument {g['entity']}? Follow the records and give only the exact section letter.",lambda g:g["section"]),
("B2_ENTITY_TO_BAY",[3,4],lambda g:f"What active bay is assigned to instrument {g['entity']}? Follow the records and give only the exact bay number.",lambda g:g["bay"]),
("JOIN_SECTION_BAY",[0,1,2,3,4],lambda g:f"What are the archive section and active bay for instrument {g['entity']}? Give only the exact code as SECTION-BAY.",lambda g:f"{g['section']}-{g['bay']}")
]

print("\n[2/4] Running increasing relational-depth panel...")
RESULTS={name:[] for name,_,_,_ in LEVELS}
for z,it in enumerate(ITEMS):
    g=it["gold"];parts=[]
    for name,idxs,qf,tf in LEVELS:
        rec=[it["records"][i] for i in idxs];target=tf(g);raw=ask(rec,qf(g));ok=exact(raw,target)
        RESULTS[name].append({"ok":ok,"raw":raw,"target":target})
        parts.append(f"{name}={int(ok)}")
    print(f"      [{z+1:02d}/{N_ITEMS}] {g['entity']:6s} | "+" ".join(parts))

print("\n[3/4] Accuracy and transition anatomy...")
ACC={}
for name,_,_,_ in LEVELS:
    n=sum(r["ok"] for r in RESULTS[name]);ACC[name]=n/N_ITEMS
    print(f"      {name:20s}: {n:2d}/{N_ITEMS} = {ACC[name]:.4f}")

print("\n      Failure examples:")
for name,_,_,_ in LEVELS:
    bad=[(i+1,r) for i,r in enumerate(RESULTS[name]) if not r["ok"]]
    print(f"\n      {name}:")
    if not bad:print("         NONE")
    else:
        for i,r in bad[:5]:
            print(f"         item={i:02d} target={r['target']!r} raw={firstline(r['raw'])!r}")

L1=sum(r["ok"] for r in RESULTS["L1_ENTITY_TO_SEAL"])
L2=sum(r["ok"] for r in RESULTS["L2_ENTITY_TO_CLASS"])
L3=sum(r["ok"] for r in RESULTS["L3_ENTITY_TO_SECTION"])
B2=sum(r["ok"] for r in RESULTS["B2_ENTITY_TO_BAY"])
JN=sum(r["ok"] for r in RESULTS["JOIN_SECTION_BAY"])

# Conditional progression: among items solved at the previous stage,
# how often does the next relational step remain correct?
def conditional(a,b):
    base=[i for i in range(N_ITEMS) if RESULTS[a][i]["ok"]]
    hit=sum(RESULTS[b][i]["ok"] for i in base)
    return hit,len(base)

C12=conditional("L1_ENTITY_TO_SEAL","L2_ENTITY_TO_CLASS")
C23=conditional("L2_ENTITY_TO_CLASS","L3_ENTITY_TO_SECTION")
print(f"\n      L1 PASS -> L2 PASS: {C12[0]}/{C12[1]}")
print(f"      L2 PASS -> L3 PASS: {C23[0]}/{C23[1]}")

print("\n[4/4] Capacity boundary...")
ordered=[("L1",ACC["L1_ENTITY_TO_SEAL"]),("L2",ACC["L2_ENTITY_TO_CLASS"]),("L3",ACC["L3_ENTITY_TO_SECTION"])]
reliable=[n for n,a in ordered if a>=.75]
BOUNDARY=reliable[-1] if reliable else "BELOW_L1"

if ACC["L1_ENTITY_TO_SEAL"]<.75:
    VERDICT="NATIVE_BASELINE_BELOW_ONE_HOP_THRESHOLD"
elif ACC["L2_ENTITY_TO_CLASS"]<.75:
    VERDICT="NATIVE_CAPACITY_BOUNDARY_AT_ONE_HOP"
elif ACC["L3_ENTITY_TO_SECTION"]<.75:
    VERDICT="NATIVE_CAPACITY_BOUNDARY_AT_TWO_HOPS"
elif ACC["B2_ENTITY_TO_BAY"]<.75:
    VERDICT="SECOND_BRANCH_NOT_RELIABLY_SOLVED"
elif ACC["JOIN_SECTION_BAY"]<.75:
    VERDICT="MULTI_BRANCH_JOIN_IS_PRIMARY_DIFFICULTY"
else:
    VERDICT="FULL_NATIVE_TASK_RELIABLY_SOLVED"

print("\n"+"="*150)
print("AKBASCORE MAM — TEST501 FINAL RESULT")
print("="*150)
print("MODEL                       : Qwen/Qwen2.5-7B-Instruct · frozen")
print("MAM MEMORY                  : OFF")
print("BELLEKÖZ COMPRESSION        : OFF")
print("STEERING                    : OFF")
print("TRAINING                    : OFF")
print("-"*150)
print(f"L1 ENTITY → SEAL            : {L1:2d}/{N_ITEMS} = {L1/N_ITEMS:.4f}")
print(f"L2 ENTITY → CLASS           : {L2:2d}/{N_ITEMS} = {L2/N_ITEMS:.4f}")
print(f"L3 ENTITY → SECTION         : {L3:2d}/{N_ITEMS} = {L3/N_ITEMS:.4f}")
print(f"B2 ENTITY → BAY             : {B2:2d}/{N_ITEMS} = {B2/N_ITEMS:.4f}")
print(f"JOIN SECTION + BAY          : {JN:2d}/{N_ITEMS} = {JN/N_ITEMS:.4f}")
print("-"*150)
print("RELIABLE DEPTH BOUNDARY     :",BOUNDARY)
print("TEST501 LOCK SHA            :",LOCK_SHA)
print(f"TOTAL TEST TIME              : {time.perf_counter()-T0:.2f}s")
print("EXPLORATORY VERDICT         :",VERDICT)
print("="*150)

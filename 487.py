# TEST487 — AKBASCORE MAM · MODEL-NATIVE RECALL ATLAS
# Copyright © 2026 Mustafa Akbaş
# Frozen Qwen2.5-7B-Instruct · 32 NAMELESS memories · passive geometry only.
# Goal: map distinct native recall signals + complementarity. NO router/ANN/training/LoRA/readout/model change.
import os,sys,subprocess,importlib.util,random,re,time,hashlib,json,math
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate"),("numpy","numpy")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
TEST="487";SEED=487;MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SEP="\n\n";DEVICE="cuda";LAYERS=list(range(28))
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

RECORDS=[
("F1","emerald barometer","KR-214"),("F1","porcelain compass","DM-763"),("F1","violet harmonica","FS-408"),("F1","bamboo chronometer","JN-951"),
("F1","obsidian sextant","RW-326"),("F1","linen telescope","AE-875"),("F1","coral metronome","LP-143"),("F1","willow calculator","XM-692"),
("F2","turquoise monocle","HC-517"),("F2","mahogany notebook","VK-280"),("F2","ceramic astrolabe","SB-934"),("F2","scarlet blueprint","NT-461"),
("F2","opal kaleidoscope","GY-708"),("F2","bronze hourglass","PD-352"),("F2","silk manuscript","ZU-819"),("F2","crystal chime","EF-625"),
("F3","jade projector","WL-407"),("F3","glass clarinet","CQ-586"),("F3","canvas diary","MR-172"),("F3","iron medallion","UX-943"),
("F3","pearl camera","BH-650"),("F3","steel accordion","KO-238"),("F3","cotton almanac","YD-714"),("F3","onyx brooch","TG-569"),
("F4","ivory receiver","PS-381"),("F4","cedar plaque","AL-826"),("F4","azure kettle","RF-504"),("F4","brass caliper","MW-197"),
("F4","crystal phonograph","DE-648"),("F4","maple slate","KI-275"),("F4","white lantern","OV-930"),("F4","golden protractor","XC-412")]
def source(f,o,c):
    if f=="F1":return f"The {o} is stored in container {c}."
    if f=="F2":return f"Container {c} contains the {o}."
    if f=="F3":return f"The {o} can be found inside container {c}."
    return f"Inside container {c} there is the {o}."
SOURCES=[source(*x) for x in RECORDS]
DESC=[
"green instrument used to measure atmospheric pressure","ceramic instrument used to determine direction",
"purple handheld instrument played by blowing and drawing air","bamboo device used for precise timekeeping",
"dark volcanic-stone instrument historically used for celestial navigation","fabric-covered optical instrument for viewing distant objects",
"reddish device used to keep a steady musical tempo","wooden device used for arithmetic calculations",
"blue-green single-eye optical lens","dark wooden book used for handwritten notes",
"pottery instrument historically used to determine celestial position","bright red technical drawing",
"gemstone-colored optical toy that forms changing symmetrical patterns","metal timer in which sand falls between two glass chambers",
"smooth-fabric document written by hand","transparent decorative object that produces a ringing sound",
"green-stone device used to display images on a surface","transparent woodwind instrument played with a single reed",
"cloth-covered personal book used for daily written entries","metal ornamental disk worn or kept as a keepsake",
"pale gemstone-colored device used to take photographs","metal bellows-driven musical instrument with keys or buttons",
"soft-fabric annual reference book containing facts and dates","black gemstone ornamental pin",
"pale handheld apparatus used to receive transmitted signals","wooden flat sign or commemorative plate",
"blue vessel used to heat water","metal measuring tool with two jaws for measuring dimensions",
"transparent historical machine that reproduces recorded sound","wooden flat board used for writing",
"pale portable light used for illumination","yellow-metal instrument used to measure and draw angles"]
QUERIES=[f"Which container holds the {x}?" for x in DESC]
for i,(_,o,_) in enumerate(RECORDS):
    if o.lower() in QUERIES[i].lower():raise RuntimeError(f"Leak {i}")
LOCK_SHA=hashlib.sha256(json.dumps({"test":TEST,"records":RECORDS,"sources":SOURCES,"queries":QUERIES},sort_keys=True,separators=(",",":")).encode()).hexdigest()

print("="*150);print("TEST487 — AKBASCORE MAM · MODEL-NATIVE RECALL ATLAS")
print("32 NAMELESS MEMORIES · 28-LAYER PASSIVE X-RAY · MULTI-SIGNAL COMPLEMENTARITY");print("="*150)
print("LOCK SHA:",LOCK_SHA);T0=time.perf_counter()
print("[1/6] Loading frozen Qwen...")
tv=tuple(int("".join(c for c in x if c.isdigit())or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;cfg=model.config;NL=len(layers);H=cfg.hidden_size;NH=cfg.num_attention_heads;NKV=cfg.num_key_value_heads;HD=H//NH;G=NH//NKV
if (NL,H,NH,NKV,HD)!=(28,3584,28,4,128):raise RuntimeError("Architecture mismatch")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
enc=lambda s:tok(s,add_special_tokens=False).input_ids
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | 28L | 28Q/4KV | frozen BF16")

@torch.inference_mode()
def probe(text):
    ids=[PAD]+enc(text+SEP);x=torch.tensor([ids],device=DEVICE)
    o=model(input_ids=x,output_hidden_states=True,use_cache=False,return_dict=True)
    out=[]
    for l in range(NL):
        h=o.hidden_states[l][0,1:].float();z=layers[l].input_layernorm(o.hidden_states[l][0])[1:]
        a=layers[l].self_attn
        q=a.q_proj(z).float().view(-1,NH,HD);k=a.k_proj(z).float().view(-1,NKV,HD)
        out.append({"hm":h.mean(0).cpu(),"hl":h[-1].cpu(),"ht":h.cpu(),
                    "qm":q.mean(0).cpu(),"km":k.mean(0).cpu()})
    return out

print("[2/6] Capturing cartridge-side atlas...")
B=[]
for i,s in enumerate(SOURCES):
    B.append(probe(s));print(f"      [{i+1:02d}/32] {RECORDS[i][1]}")
print("[3/6] Capturing NAMELESS query-side atlas...")
Q=[]
for i,q in enumerate(QUERIES):
    Q.append(probe(q));print(f"      [{i+1:02d}/32] {DESC[i][:52]}")

def norm(x):return torch.nn.functional.normalize(x.float(),dim=-1)
def rank(s,g):return int((torch.argsort(s,descending=True)==g).nonzero(as_tuple=False)[0,0])+1
def metrics(rs):
    a=np.asarray(rs,float);return {"R1":float(np.mean(a<=1)),"R5":float(np.mean(a<=5)),"R16":float(np.mean(a<=16)),
                                  "MRR":float(np.mean(1/a)),"MEDR":float(np.median(a))}
def cosine_matrix(A,C):
    A=norm(torch.stack(A));C=norm(torch.stack(C));return A@C.T
def centered_matrix(A,C):
    A=torch.stack(A).float();C=torch.stack(C).float();mu=C.mean(0);return norm(A-mu)@norm(C-mu).T
def l2_matrix(A,C):
    A=torch.stack(A).float();C=torch.stack(C).float();return -torch.cdist(A,C,p=2)
def tokenmax_matrix(l):
    M=torch.empty(32,32)
    for i in range(32):
        q=norm(Q[i][l]["ht"])
        for j in range(32):
            b=norm(B[j][l]["ht"]);M[i,j]=(q@b.T).max(dim=1).values.mean()
    return M
def qk_matrix(l,head=None):
    M=torch.empty(32,32)
    for i in range(32):
        q=Q[i][l]["qm"]
        for j in range(32):
            k=B[j][l]["km"].repeat_interleave(G,0)
            if head is None:M[i,j]=(norm(q)*norm(k)).sum(-1).mean()
            else:M[i,j]=(norm(q[head])*norm(k[head])).sum()
    return M
def kv_matrix(l):
    A=[Q[i][l]["km"].reshape(-1) for i in range(32)];C=[B[i][l]["km"].reshape(-1) for i in range(32)]
    return cosine_matrix(A,C)
def delta_matrix(l):
    if l==0:return None
    A=[Q[i][l]["hm"]-Q[i][l-1]["hm"] for i in range(32)]
    C=[B[i][l]["hm"]-B[i][l-1]["hm"] for i in range(32)]
    return cosine_matrix(A,C)
def trajectory_matrix(a,b):
    A=[];C=[]
    for i in range(32):
        A.append(torch.cat([Q[i][l]["hm"] for l in range(a,b+1)]))
        C.append(torch.cat([B[i][l]["hm"] for l in range(a,b+1)]))
    return cosine_matrix(A,C)
def evalM(name,M):
    rs=[rank(M[i],i) for i in range(32)];x=metrics(rs);return {"name":name,**x,"ranks":rs,"M":M}

print("[4/6] Building recall atlas...")
AT=[]
for l in LAYERS:
    A=[Q[i][l]["hm"] for i in range(32)];C=[B[i][l]["hm"] for i in range(32)]
    AL=[Q[i][l]["hl"] for i in range(32)];CL=[B[i][l]["hl"] for i in range(32)]
    AT.append(evalM(f"HMEAN_COS_L{l:02d}",cosine_matrix(A,C)))
    AT.append(evalM(f"HMEAN_CENTER_L{l:02d}",centered_matrix(A,C)))
    AT.append(evalM(f"HMEAN_L2_L{l:02d}",l2_matrix(A,C)))
    AT.append(evalM(f"HLAST_COS_L{l:02d}",cosine_matrix(AL,CL)))
    AT.append(evalM(f"KMEAN_COS_L{l:02d}",kv_matrix(l)))
    if l>0:AT.append(evalM(f"DELTA_COS_L{l:02d}",delta_matrix(l)))
    AT.append(evalM(f"TOKENMAX_L{l:02d}",tokenmax_matrix(l)))
    AT.append(evalM(f"QK_ALL_L{l:02d}",qk_matrix(l)))
for a,b in [(0,3),(4,7),(8,11),(12,15),(16,19),(20,23),(24,27),(0,27)]:
    AT.append(evalM(f"TRAJECTORY_{a:02d}_{b:02d}",trajectory_matrix(a,b)))
print("      Scanning head-local Q->K resonance...")
for l in LAYERS:
    for h in range(NH):AT.append(evalM(f"QK_HEAD_L{l:02d}_H{h:02d}",qk_matrix(l,h)))

AT.sort(key=lambda x:(x["MRR"],x["R1"],x["R5"]),reverse=True)
print(f"      Atlas channels: {len(AT)}")
print("\n      TOP 30 NATIVE SIGNALS")
for n,x in enumerate(AT[:30],1):print(f"      {n:02d}. {x['name']:24s} R1={x['R1']:.3f} R5={x['R5']:.3f} MRR={x['MRR']:.3f} MEDR={x['MEDR']:.1f}")

print("\n[5/6] Measuring complementarity...")
# Deduplicate near-identical channels by rank-vector; retain strongest representative.
UNIQ=[];seen=set()
for x in AT:
    z=tuple(x["ranks"])
    if z not in seen:seen.add(z);UNIQ.append(x)
POOL=UNIQ[:64]
# Rank-normalized consensus: no learned weights; each channel contributes equally.
def rankscore(M):
    R=torch.empty_like(M)
    for i in range(32):
        o=torch.argsort(M[i],descending=True);r=torch.empty(32)
        r[o]=torch.arange(1,33,dtype=torch.float32);R[i]=1-(r-1)/31
    return R
RS=[rankscore(x["M"]) for x in POOL]
def consensus(ids):
    M=torch.stack([RS[k] for k in ids]).mean(0)
    rr=[rank(M[i],i) for i in range(32)]
    return metrics(rr),rr,M
# Forward greedy atlas composition = discovery diagnostic only, NOT held-out proof.
chosen=[];remaining=list(range(len(POOL)));history=[]
for step in range(min(12,len(POOL))):
    best=None
    for k in remaining:
        m,rr,_=consensus(chosen+[k]);key=(m["MRR"],m["R1"],m["R5"])
        if best is None or key>best[0]:best=(key,k,m,rr)
    chosen.append(best[1]);remaining.remove(best[1]);history.append(best[2])
    print(f"      +{step+1:02d} {POOL[best[1]]['name']:24s} -> R1={best[2]['R1']:.3f} R5={best[2]['R5']:.3f} MRR={best[2]['MRR']:.3f}")
BEST_STEP=max(range(len(history)),key=lambda i:(history[i]["MRR"],history[i]["R1"],history[i]["R5"]))
BEST_IDS=chosen[:BEST_STEP+1];CONS,CR,CM=consensus(BEST_IDS)

# Oracle complementarity: asks only whether ANY top native channel can recover each memory.
ORACLE=[]
for i in range(32):
    rr=[x["ranks"][i] for x in POOL]
    ORACLE.append(min(rr))
ORACLE_M=metrics(ORACLE)

# Error diversity among top 20: how often at least one channel ranks gold top1/top5.
TOP20=POOL[:20]
ANY1=sum(any(x["ranks"][i]<=1 for x in TOP20) for i in range(32))/32
ANY5=sum(any(x["ranks"][i]<=5 for x in TOP20) for i in range(32))/32
print("\n      BEST FIXED-EQUAL CONSENSUS:")
for k in BEST_IDS:print("       ",POOL[k]["name"])
print("      =>",CONS)
print("      TOP20 ANY-R1:",ANY1,"ANY-R5:",ANY5)
print("      TOP64 ORACLE:",ORACLE_M)

print("\n[6/6] Null + per-memory atlas...")
rng=np.random.default_rng(SEED);perm=np.arange(32);rng.shuffle(perm)
NULL=metrics([rank(AT[0]["M"][i],int(perm[i])) for i in range(32)])
for i in range(32):
    best=min((x["ranks"][i],x["name"]) for x in POOL)
    print(f"      [{i:02d}] best-rank={best[0]:2d} via {best[1]:24s} | {RECORDS[i][1]}")

if ORACLE_M["R1"]>=.90 and CONS["R1"]>=.60:VERDICT="STRONG_COMPLEMENTARY_NATIVE_RECALL_STRUCTURE"
elif ORACLE_M["R1"]>=.75 and CONS["R5"]>=.70:VERDICT="SUBSTANTIAL_COMPLEMENTARY_NATIVE_RECALL_STRUCTURE"
elif ORACLE_M["R5"]>=.90:VERDICT="DIVERSE_NATIVE_SIGNALS_EXIST_BUT_CONSENSUS_UNRESOLVED"
else:VERDICT="NO_STRONG_MULTI_SIGNAL_RECALL_ATLAS_FOUND"

print("\n"+"="*150);print("TEST487 FINAL RESULT — AKBASCORE MAM RECALL ATLAS");print("="*150)
print("FOUNDATION          : TEST482 QWEN · frozen")
print("TEST482 BLOB SHA    : ecc630144a44479cb35c432c50eb0d09bc6866d7")
print("INPUT CONDITION     : NAMELESS only")
print("MEMORIES            : 32")
print("LAYERS              : 0-27")
print("ATLAS CHANNELS      :",len(AT))
print("UNIQUE RANK CHANNELS:",len(UNIQ))
print("LOCK SHA            :",LOCK_SHA)
print("-"*150)
print("BEST SINGLE         :",{k:v for k,v in AT[0].items() if k not in ("M","ranks")})
print("BEST CONSENSUS      :",CONS)
print("CONSENSUS CHANNELS  :",[POOL[k]["name"] for k in BEST_IDS])
print("TOP20 ANY-R1        :",ANY1)
print("TOP20 ANY-R5        :",ANY5)
print("TOP64 ORACLE        :",ORACLE_M)
print("PERMUTATION NULL    :",NULL)
print("-"*150)
print("VERDICT             :",VERDICT)
print("INTERPRETATION      : discovery atlas; consensus selection used same 32 examples and is NOT held-out proof.")
print("NEXT TEST           : freeze discovered channel set -> fresh memories/queries -> held-out validation.")
print("CARTRIDGE READOUT   : NOT USED")
print("ROUTER / ANN / GRAPH: NONE")
print("TRAINING / LoRA     : NONE")
print("MODEL CHANGE        : NONE")
print(f"TOTAL TIME          : {time.perf_counter()-T0:.2f}s")
print("="*150)

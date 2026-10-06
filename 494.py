# Copyright © 2026 Mustafa Akbaş
# New original AKBASCORE MAM material is source-available under the repository LICENSE.
# Earlier MIT-licensed Titan Cognitive Core / AkbasCore material remains under its original license.
# See the repository LICENSE for complete terms.
#
# TEST494 — AKBASCORE MAM · KİMLİKİZ → BELLEKÖZ END-TO-END RECALL
# FIRST END-TO-END MODEL-NATIVE ADDRESS → COMPRESSED MEMORY → ANSWER TEST
#
# CANONICAL WORKING TERMINOLOGY:
#   AKBASCORE MAM : complete Model-Native Associative Memory system
#   BELLEKÖZ      : compressed memory core (TEST482 K120/V128/OWN)
#   ÇAĞRIİZ       : model-native recall-trace family
#   KİMLİKİZ      : frozen instance-identity trace
#                   TEST493 seal = VTOKEN_L23
#
# FROZEN FOUNDATIONS:
#   TEST482 BELLEKÖZ : exact K120/V128/OWN compression/install/readout architecture
#   TEST493 KİMLİKİZ : VTOKEN_L23, frozen before this test
#
# TEST494 QUESTION:
#   Can frozen KİMLİKİZ select the correct record from a bank,
#   then can that selected compressed BELLEKÖZ alone produce the answer?
#
# IMPORTANT:
#   Address geometry is extracted from the source BEFORE BELLEKÖZ compression.
#   BELLEKÖZ remains exactly the TEST482 compressed memory representation.
#   This test does NOT claim that the compressed cartridge itself contains
#   the address key. That is a later architectural question.
#
# PRIMARY PATH:
#   nameless semantic query
#      -> KİMLİKİZ / VTOKEN_L23
#      -> select ONE record
#      -> install selected TEST482 BELLEKÖZ
#      -> frozen Qwen readout
#      -> container ID
#
# CONTROLS:
#   1. KİMLİKİZ address R1/R5 over all 32 records
#   2. GOLD-BELLEKÖZ oracle readout
#   3. KİMLİKİZ→BELLEKÖZ actual end-to-end readout
#   4. WRONG-BELLEKÖZ deterministic wrong-address control
#
# FAILURE ANATOMY:
#   ADDRESS_FAIL
#   BELLEKÖZ_READ_FAIL
#   BOTH_FAIL
#
# NO:
#   channel/layer/head search
#   fusion
#   threshold/weight fitting
#   router/index/signature/ANN/graph
#   training/LoRA/gradient
#   new compression motor
#   post-hoc selection

import os,sys,subprocess,importlib.util,random,re,time,hashlib,json
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None: subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

TEST="494";SEED=494;MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
K_DIM=120;V_DIM=128;MAX_NEW=32;ADDRESS_LAYER=23
FMT="QUESTION:\n{q}\n\n ANSWER:";SEP="\n\n"
STYLE=" Give only the answer on the first line; do not explain."
READOUT='Which container contains the described object according to this record? Give only the container identifier. If this record does not contain that object, answer NONE.'+STYLE
DEVICE=torch.device("cuda")
TEST482_BLOB_SHA="ecc630144a44479cb35c432c50eb0d09bc6866d7"
TEST482_LOCK_SHA="fa59fd38661e558f6bae22eedff0999f32f8f6e9d4a08932383525323a5fe687"
TEST492_LOCK_SHA="8de05fc52332806383642618fcf4a9ea4ed29fe2f544cac5beb5369686ee79a6"
TEST493_LOCK_SHA="fec116dff1c97fdc257de5ebbb09ca4c96202b9c6da98f5d5cfe3aae20da5c0a"
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required.")
torch.set_grad_enabled(False)
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

# EXACT TEST482 neutral PCA codebook
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

# FRESH TEST494 BANK.
# New relative to TEST482 and TEST493.
# Every record has a unique concept and unique instance.
# Query deliberately avoids the literal stored object phrase.
RECORDS=[
("F1","bronze inclinometer","MA-731","cedar archive","the brown-metal instrument used to measure slope or inclination relative to gravity"),
("F1","crystal refractometer","MB-284","marble annex","the transparent optical instrument used to determine how strongly a substance bends light"),
("F1","scarlet flowmeter","MC-615","amber gallery","the red instrument used to quantify the rate at which fluid moves through a pipe"),
("F1","bamboo altazimuth","MD-902","fern vault","the wood-bodied surveying instrument used to measure both horizontal and vertical angles"),
("F1","obsidian manometer","ME-347","copper archive","the black pressure instrument that determines fluid pressure from a column or pressure-sensitive mechanism"),
("F1","linen hygroscope","MF-861","quartz annex","the fabric-covered device used to indicate changes in moisture in the surrounding air"),
("F1","coral viscometer","MG-526","ivory gallery","the reddish instrument used to determine a fluid's resistance to flowing"),
("F1","willow micrometer","MH-193","silver vault","the wooden precision measuring tool used for extremely small dimensions"),

("F2","turquoise lactometer","MI-708","indigo lighthouse","the blue-green floating instrument used to estimate the density or purity of milk"),
("F2","mahogany saccharimeter","MJ-452","pearl pavilion","the dark-wood optical instrument used to determine sugar concentration"),
("F2","ceramic polarimeter","MK-819","granite lodge","the pottery-cased optical device used to measure rotation of polarized light"),
("F2","scarlet nepheloscope","ML-264","willow gallery","the red meteorological instrument used for observing cloud movement or characteristics"),
("F2","opal pycnometer","MM-573","maple tower","the gemstone-colored calibrated vessel used to determine density from a known volume"),
("F2","bronze cathetometer","MN-940","coral pavilion","the brown-metal precision instrument used to measure vertical distances between points"),
("F2","silk heliograph","MO-381","birch lodge","the fabric-covered apparatus that records or signals using sunlight"),
("F2","crystal spectroscope","MP-726","onyx gallery","the transparent optical instrument used to separate light so its spectrum can be examined"),

("F3","jade bolometer","MQ-658","hazel chamber","the green detector that measures radiant energy through the heating it produces"),
("F3","glass eudiometer","MR-405","lagoon studio","the transparent graduated tube used to measure changes in gas volume during chemical reactions"),
("F3","canvas caloriscope","MS-917","acorn room","the fabric-covered apparatus used to demonstrate or compare quantities of heat"),
("F3","iron extensometer","MT-236","poplar hall","the heavy-metal instrument used to measure how much a specimen stretches under load"),
("F3","pearl episcope","MU-564","cobalt chamber","the pale optical projector used to display an image of an opaque object"),
("F3","steel pantograph","MV-823","canal studio","the metallic linked-arm device used to copy a drawing at the same or a different scale"),
("F3","cotton curvimeter","MW-190","walnut room","the fabric-covered measuring wheel used to determine the length of curved lines on maps"),
("F3","onyx planisphere","MX-647","elm hall","the black rotating star chart used to show which constellations are visible at a chosen time"),

("F4","ivory alidade","MY-315","summit workshop","the pale sighting rule used with surveying instruments to establish a direction"),
("F4","cedar hydroscope","MZ-782","valley depot","the wooden device used for observing or examining conditions beneath water"),
("F4","azure actinometer","NA-439","birch observatory","the blue instrument used to measure the intensity of solar or other radiation"),
("F4","brass episometer","NB-806","shore conservatory","the yellow-metal measuring device associated with determining specific physical dimensions"),
("F4","crystal gonioscope","NC-251","forest workshop","the transparent optical instrument used to inspect or measure angular optical properties"),
("F4","maple opisometer","ND-694","delta depot","the wooden wheel instrument rolled along a curved line to determine its length"),
("F4","white cyanometer","NE-538","clover observatory","the pale instrument or scale used to estimate the blueness of the sky"),
("F4","golden clinograph","NF-972","island conservatory","the yellow instrument used to record or represent inclination continuously")
]

def source(fam,obj,cid,place):
    if fam=="F1": return f"The {obj} is stored in container {cid}."
    if fam=="F2": return f"Container {cid} contains the {obj}."
    if fam=="F3": return f"The {obj} can be found inside container {cid}."
    if fam=="F4": return f"Inside container {cid} there is the {obj}."
    raise ValueError(fam)

assert len(RECORDS)==32
assert len({x[1] for x in RECORDS})==32
assert len({x[2] for x in RECORDS})==32
VALID_IDS=[x[2] for x in RECORDS]
SOURCES=[source(*x[:4]) for x in RECORDS]
QUERIES=[x[4] for x in RECORDS]

LOCK={
"test":TEST,"model":MODEL_ID,"seed":SEED,
"k_dim":K_DIM,"v_dim":V_DIM,"max_new":MAX_NEW,
"address":"KİMLİKİZ/VTOKEN_L23","address_layer":ADDRESS_LAYER,
"bellekoz":"TEST482 K120/V128/OWN",
"fmt":FMT,"sep":SEP,"readout":READOUT,
"records":RECORDS,"corpus":CORPUS,
"test482_blob_sha":TEST482_BLOB_SHA,
"test482_lock_sha":TEST482_LOCK_SHA,
"test492_lock_sha":TEST492_LOCK_SHA,
"test493_lock_sha":TEST493_LOCK_SHA,
"search":"NONE","fusion":"NONE","training":"NONE"
}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

print("="*154)
print("TEST494 — AKBASCORE MAM · KİMLİKİZ → BELLEKÖZ END-TO-END RECALL")
print("FROZEN VTOKEN_L23 ADDRESS → EXACT TEST482 K120/V128/OWN COMPRESSED MEMORY → ANSWER")
print("="*154)
print("LOCK SHA:",LOCK_SHA)
print("TEST482 BLOB:",TEST482_BLOB_SHA)
print("TEST493 LOCK:",TEST493_LOCK_SHA)
print("BANK: 32 fresh records | one BELLEKÖZ per record | no full-scan readout")
T0=time.perf_counter()

print("[1/9] Loading frozen Qwen...")
_tv=tuple(int("".join(c for c in x if c.isdigit())or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters(): p.requires_grad_(False)
cfg=model.config;layers=model.model.layers
H=cfg.hidden_size;NH=cfg.num_attention_heads;NKV=cfg.num_key_value_heads
HD=getattr(cfg,"head_dim",None) or H//NH;KVD=NKV*HD;NL=len(layers)
if (NL,H,NH,NKV,HD)!=(28,3584,28,4,128): raise RuntimeError(f"Architecture mismatch: {(NL,H,NH,NKV,HD)}")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
_ge=model.generation_config.eos_token_id
EOS=sorted({tok.eos_token_id,*(_ge if isinstance(_ge,(list,tuple)) else [_ge])}-{None})
enc=lambda s:tok(s,add_special_tokens=False).input_ids
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | {NL}L | H={H} | QH={NH} KVH={NKV} | frozen BF16")

@torch.inference_mode()
def kv_from_ids(ids):
    o=model(input_ids=torch.tensor([ids],device=DEVICE),output_hidden_states=True,use_cache=False,return_dict=True)
    K=[];V=[]
    for L in range(NL):
        z=layers[L].input_layernorm(o.hidden_states[L][0]);a=layers[L].self_attn
        K.append(a.k_proj(z).contiguous());V.append(a.v_proj(z).contiguous())
    return K,V

def forge(s): return kv_from_ids([PAD]+enc(s+SEP))

@torch.inference_mode()
def install(K,V):
    T=K[0].shape[0]
    cos,sin=model.model.rotary_emb(K[0][None],torch.arange(T,device=DEVICE)[None])
    KK=[];VV=[]
    for L in range(NL):
        k=K[L].reshape(1,T,NKV,HD).transpose(1,2)
        v=V[L].reshape(1,T,NKV,HD).transpose(1,2)
        KK.append(apply_rotary_pos_emb(k,k,cos,sin,unsqueeze_dim=1)[1].contiguous())
        VV.append(v.contiguous())
    return tuple(KK),tuple(VV),T

print("[2/9] Building exact TEST482 neutral PCA codebook...")
CB=[];CO=[forge(s) for s in CORPUS]
for L in range(NL):
    e={}
    for j,n in enumerate(("K","V")):
        R=torch.cat([c[j][L][1:] for c in CO]).float().view(-1,NKV,HD);MU=[];BB=[]
        for h in range(NKV):
            X=R[:,h];mu=X.mean(0);_,S,Vh=torch.linalg.svd(X-mu,full_matrices=False)
            m=min(128,Vh.shape[0]);b=Vh[:m].T.contiguous()
            if m<128: b=torch.nn.functional.pad(b,(0,128-m))
            MU.append(mu);BB.append(b)
        e[n]=(torch.stack(MU),torch.stack(BB))
    CB.append(e)
del CO;torch.cuda.empty_cache()
print("      BELLEKÖZ codebook ready.")

@torch.inference_mode()
def packet(s):
    K,V=forge(s);out={}
    for n,X,d in (("K",K,K_DIM),("V",V,V_DIM)):
        rows=[]
        for L in range(NL):
            mu,B=CB[L][n]
            coeff=torch.einsum("thi,hid->thd",X[L][1:].float().view(-1,NKV,HD)-mu,B[:,:,:d]).to(torch.bfloat16)
            content=(mu+torch.einsum("thd,hid->thi",coeff.float(),B[:,:,:d])).reshape(-1,KVD).to(torch.bfloat16)
            rows.append(torch.cat([X[L][:1],content]))
        out[n]=rows
    return out["K"],out["V"]

# Exact TEST493 VTOKEN_L23 geometry.
# IMPORTANT: this is source-side address geometry, not compressed BELLEKÖZ geometry.
@torch.inference_mode()
def kimlikiz_v23(text):
    ids=tok(text,return_tensors="pt",add_special_tokens=True).input_ids.to(DEVICE)
    o=model(input_ids=ids,output_hidden_states=True,use_cache=False,return_dict=True)
    z=layers[ADDRESS_LAYER].input_layernorm(o.hidden_states[ADDRESS_LAYER][0])
    v=layers[ADDRESS_LAYER].self_attn.v_proj(z).float().view(-1,NKV,HD)
    return F.normalize(v,dim=-1).cpu()

def kimlikiz_score(qv,mv):
    vals=[]
    for h in range(NKV): vals.append((qv[:,h]@mv[:,h].T).max(1).values.mean())
    return float(torch.stack(vals).mean())

@torch.inference_mode()
def read_one(q,kv):
    qids=enc(FMT.format(q=q));Tm=kv[2];nq=len(qids)
    try: cache=DynamicCache(config=cfg)
    except TypeError: cache=DynamicCache()
    for L in range(NL): cache.update(kv[0][L].clone(),kv[1][L].clone(),L)
    mask=torch.ones(1,Tm+nq,dtype=torch.long,device=DEVICE)
    pos=torch.arange(Tm,Tm+nq,device=DEVICE)[None]
    ids=torch.tensor([qids],device=DEVICE);out=[];done=False
    for _ in range(MAX_NEW):
        o=model(input_ids=ids,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=True,return_dict=True)
        nxt=int(o.logits[0,-1].float().argmax())
        if nxt in EOS: break
        out.append(nxt)
        ids=torch.tensor([[nxt]],device=DEVICE);pos=pos[:,-1:]+1
        mask=torch.cat([mask,torch.ones(1,1,dtype=torch.long,device=DEVICE)],1)
    return tok.decode(out,skip_special_tokens=True).strip()

def norm(s): return re.sub(r"[^\w-]+"," ",str(s).casefold()).strip()
def firstline(text): return next((x.strip() for x in str(text).splitlines() if x.strip()),"")
def is_none(text):
    n=norm(firstline(text))
    return n=="none" or n.startswith("none ")
def parse_id(text):
    line=firstline(text)
    if is_none(line): return None
    n=norm(line);hits=[x for x in VALID_IDS if norm(x) in n]
    return hits[0] if len(set(hits))==1 else None

print("[3/9] Forging 32 fresh BELLEKÖZ records + frozen KİMLİKİZ keys...")
BELLEKOZ=[];MKEY=[];tf=time.perf_counter()
for i,s in enumerate(SOURCES):
    K,V=packet(s);BELLEKOZ.append(install(K,V));del K,V
    MKEY.append(kimlikiz_v23(s))
    print(f"      [{i+1:02d}/32] {RECORDS[i][1]:24s} -> {RECORDS[i][2]}")
FORGE_SEC=time.perf_counter()-tf

print("[4/9] KİMLİKİZ — frozen 32-way model-native addressing...")
QKEY=[];SCORE=np.zeros((32,32),dtype=np.float64);RANKS=[];SELECT=[]
ta=time.perf_counter()
for i,q in enumerate(QUERIES):
    qv=kimlikiz_v23(q);QKEY.append(qv)
    for j in range(32): SCORE[i,j]=kimlikiz_score(qv,MKEY[j])
    order=np.argsort(-SCORE[i],kind="stable")
    rank=int(np.where(order==i)[0][0])+1;sel=int(order[0])
    RANKS.append(rank);SELECT.append(sel)
    print(f"      [{i+1:02d}] rank={rank:02d} selected={sel+1:02d} {RECORDS[sel][1]:24s} | gold={RECORDS[i][1]:24s} | {'PASS' if sel==i else 'MISS'}")
ADDR_SEC=time.perf_counter()-ta
RANKS=np.array(RANKS);ADDR_R1=float(np.mean(RANKS<=1));ADDR_R5=float(np.mean(RANKS<=5));ADDR_R10=float(np.mean(RANKS<=10));ADDR_MRR=float(np.mean(1/RANKS))

print("\n[5/9] GOLD-BELLEKÖZ — oracle compressed-memory readout upper bound...")
GOLD_RAW=[];GOLD_PARSED=[];GOLD_OK=[]
tg=time.perf_counter()
for i,q in enumerate(QUERIES):
    raw=read_one(READOUT+" Description: "+q,BELLEKOZ[i]);pred=parse_id(raw);ok=pred==RECORDS[i][2]
    GOLD_RAW.append(raw);GOLD_PARSED.append(pred);GOLD_OK.append(ok)
    print(f"      [{i+1:02d}] expected={RECORDS[i][2]:6s} pred={str(pred):6s} | {'PASS' if ok else 'FAIL'} | raw={raw!r}")
GOLD_SEC=time.perf_counter()-tg

print("\n[6/9] ÇAĞRIİZ → BELLEKÖZ — actual end-to-end path...")
E2E_RAW=[];E2E_PARSED=[];E2E_OK=[]
te=time.perf_counter()
for i,q in enumerate(QUERIES):
    j=SELECT[i];raw=read_one(READOUT+" Description: "+q,BELLEKOZ[j]);pred=parse_id(raw);ok=pred==RECORDS[i][2]
    E2E_RAW.append(raw);E2E_PARSED.append(pred);E2E_OK.append(ok)
    print(f"      [{i+1:02d}] addr={j+1:02d} rank={RANKS[i]:02d} expected={RECORDS[i][2]:6s} pred={str(pred):6s} | {'PASS' if ok else 'FAIL'}")
E2E_SEC=time.perf_counter()-te

print("\n[7/9] WRONG-BELLEKÖZ deterministic control...")
WRONG_RAW=[];WRONG_EXACT=0;WRONG_NONE=0
tw=time.perf_counter()
for i,q in enumerate(QUERIES):
    j=(i+1)%32
    raw=read_one(READOUT+" Description: "+q,BELLEKOZ[j]);pred=parse_id(raw)
    exact=pred==RECORDS[i][2];none=is_none(raw)
    WRONG_RAW.append(raw);WRONG_EXACT+=int(exact);WRONG_NONE+=int(none)
    print(f"      [{i+1:02d}] wrong={j+1:02d} gold={RECORDS[i][2]:6s} pred={str(pred):6s} NONE={int(none)} | {'LEAK' if exact else 'CONTROL'}")
WRONG_SEC=time.perf_counter()-tw

print("\n[8/9] Failure anatomy / seal diagnostics...")
ADDRESS_FAIL=0;READ_FAIL=0;BOTH_FAIL=0;E2E_PASS=0;ADDR_CORRECT_READ_FAIL=0
for i in range(32):
    a=SELECT[i]==i;g=GOLD_OK[i];e=E2E_OK[i]
    E2E_PASS+=int(e)
    if not a and g: ADDRESS_FAIL+=1
    elif a and not g: READ_FAIL+=1
    elif not a and not g: BOTH_FAIL+=1
    ADDR_CORRECT_READ_FAIL+=int(a and not e)
    if not e:
        print(f"      [{i+1:02d}] {RECORDS[i][1]:24s} | addr={'OK' if a else 'FAIL'} rank={RANKS[i]:02d} | oracle={'OK' if g else 'FAIL'} | e2e={E2E_PARSED[i]!r}")
GOLD_PASS=sum(GOLD_OK)
ADDR_PASS=sum(int(x==i) for i,x in enumerate(SELECT))
COND_DEN=ADDR_PASS
COND_PASS=sum(int(SELECT[i]==i and E2E_OK[i]) for i in range(32))
COND_ACC=COND_PASS/COND_DEN if COND_DEN else 0.

strict=(ADDR_R1>=0.70 and GOLD_PASS>=28 and E2E_PASS>=22 and COND_ACC>=0.90 and WRONG_EXACT==0)

print("\n"+"="*154)
print("[9/9] TEST494 FINAL RESULT — AKBASCORE MAM · ÇAĞRIİZ → BELLEKÖZ")
print("="*154)
print("VALIDATION TYPE        : FRESH END-TO-END / FROZEN ADDRESS + FROZEN COMPRESSION")
print("BANK                   : 32 fresh records")
print("MAM                    : AKBASCORE MAM")
print("ÇAĞRIİZ                : model-native recall trace")
print("KİMLİKİZ               : VTOKEN_L23 · frozen TEST493")
print("BELLEKÖZ               : TEST482 K120/V128/OWN compressed memory")
print("-"*154)
print(f"KİMLİKİZ ADDRESS R1    : {ADDR_R1:.6f} ({ADDR_PASS}/32)")
print(f"KİMLİKİZ ADDRESS R5    : {ADDR_R5:.6f}")
print(f"KİMLİKİZ ADDRESS R10   : {ADDR_R10:.6f}")
print(f"KİMLİKİZ ADDRESS MRR   : {ADDR_MRR:.6f}")
print(f"GOLD-BELLEKÖZ READOUT  : {GOLD_PASS}/32 = {GOLD_PASS/32:.6f}")
print(f"END-TO-END RECALL      : {E2E_PASS}/32 = {E2E_PASS/32:.6f}")
print(f"READOUT | CORRECT ADDR : {COND_PASS}/{COND_DEN} = {COND_ACC:.6f}")
print(f"WRONG-BELLEKÖZ LEAK    : {WRONG_EXACT}/32")
print(f"WRONG-BELLEKÖZ NONE    : {WRONG_NONE}/32")
print("-"*154)
print(f"ADDRESS_FAIL           : {ADDRESS_FAIL}")
print(f"BELLEKÖZ_READ_FAIL     : {READ_FAIL}")
print(f"BOTH_FAIL              : {BOTH_FAIL}")
print(f"ADDR_OK/E2E_READ_FAIL  : {ADDR_CORRECT_READ_FAIL}")
print("-"*154)
print("TEST482 BLOB SHA       :",TEST482_BLOB_SHA)
print("TEST482 LOCK SHA       :",TEST482_LOCK_SHA)
print("TEST492 LOCK SHA       :",TEST492_LOCK_SHA)
print("TEST493 LOCK SHA       :",TEST493_LOCK_SHA)
print("TEST494 LOCK SHA       :",LOCK_SHA)
print("-"*154)
print("CHANNEL SEARCH         : NONE")
print("LAYER SEARCH           : NONE")
print("HEAD SEARCH            : NONE")
print("FUSION                 : NONE")
print("WEIGHT/THRESHOLD FIT   : NONE")
print("ROUTER / ANN / GRAPH   : NONE")
print("TRAINING / LoRA        : NONE")
print("BELLEKÖZ MOTOR CHANGE  : NONE")
print("POST-HOC SELECTION     : NONE")
print("ADDRESS KEY LOCATION   : SOURCE-SIDE MODEL-NATIVE V-SPACE")
print("COMPRESSED ADDRESS KEY : NOT CLAIMED / NOT TESTED")
print("-"*154)
print(f"Forge/key time         : {FORGE_SEC:.2f}s")
print(f"Address time           : {ADDR_SEC:.2f}s")
print(f"Oracle readout time    : {GOLD_SEC:.2f}s")
print(f"End-to-end readout time: {E2E_SEC:.2f}s")
print(f"Wrong-control time     : {WRONG_SEC:.2f}s")
print(f"TOTAL TEST TIME        : {time.perf_counter()-T0:.2f}s")
print("VERDICT                :","PASS_MAM_CALLTRACE_TO_MEMORYCORE_END_TO_END" if strict else "FAIL_MAM_CALLTRACE_TO_MEMORYCORE_END_TO_END")
print("="*154)

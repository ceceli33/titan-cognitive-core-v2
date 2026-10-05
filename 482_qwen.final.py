# TEST482 — AKBASCORE NIRVANA QWEN · FINAL HELD-OUT 32-BANK SEAL
# FINAL independent validation before the 32-bank demo/public release.
#
# FROZEN FROM TEST481:
#   Model       : Qwen/Qwen2.5-7B-Instruct
#   Engine      : K120/V128/OWN
#   Architecture: 32 records -> 64 independent cartridges
#   Retrieval   : FULL SCAN
#   Stage1      : unchanged OBJECT -> ID readout
#   Stage2      : C_COMPACT, selected BEFORE this held-out test
#   Parser      : TEST476 NONE-first fix
#   Aggregation : deterministic unique-result aggregation
#
# HELD-OUT RULE:
#   All 32 object / ID / location triples below are NEW relative to TEST477-481.
#   No readout selection or tuning is performed in this test.
#
# FINAL STRICT SEAL:
#   Stage1                 32/32
#   Linked two-hop         32/32
#   Stage1 off-target FP    0/992
#   Stage2 off-target FP    0/992
#   Stage1 off-target NONE 992/992
#   Stage2 off-target NONE 992/992
#   Stage1 collisions       0/32
#   Stage2 collisions       0/32
#   Unrelated object NONE 128/128
#   Absent-ID NONE        128/128
#
# NO router/index/signature/training/LoRA/gradient/new motor.

import os,sys,subprocess,importlib.util,random,re,time,hashlib,json
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None: subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

TEST="482";SEED=461;MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
K_DIM=120;V_DIM=128;MAX_NEW=32
FMT="QUESTION:\n{q}\n\n ANSWER:";SEP="\n\n"
STYLE=" Give only the answer on the first line; do not explain."
MW1='Which container contains the {obj} according to this record? Give only the container identifier. If this record does not contain the {obj}, answer NONE.'+STYLE
MW2='Container "{cid}" location? Answer only the location if stated; otherwise answer NONE.'
DEVICE=torch.device("cuda")
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required.")
torch.set_grad_enabled(False)
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

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

# FINAL HELD-OUT BANK — unseen triples, balanced 8×F1/F2/F3/F4
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

def sources(fam,obj,cid,place):
    if fam=="F1": return f"The {obj} is stored in container {cid}.",f"Container {cid} is located in the {place}."
    if fam=="F2": return f"Container {cid} contains the {obj}.",f"The {place} houses container {cid}."
    if fam=="F3": return f"The {obj} can be found inside container {cid}.",f"The location of container {cid} is the {place}."
    if fam=="F4": return f"Inside container {cid} there is the {obj}.",f"Container {cid} can be found at the {place}."
    raise ValueError(fam)

ABSENT_OBJECTS=["yellow microscope","paper accordion","granite whistle","violet telescope"]
ABSENT_IDS=["ZZ-999","AA-000","LM-777","RX-111"]

LOCK={
"model":MODEL_ID,"k_dim":K_DIM,"v_dim":V_DIM,"max_new":MAX_NEW,
"fmt":FMT,"sep":SEP,"mw1":MW1,"mw2":MW2,
"records":RECORDS,"corpus":CORPUS
}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()

print("="*140)
print("TEST482 — AKBASCORE NIRVANA QWEN · FINAL HELD-OUT 32-BANK SEAL")
print("32 NEW RECORDS · 64 CARTRIDGES · K120/V128/OWN · FULL SCAN · C_COMPACT FROZEN")
print("="*140)
print("LOCK SHA:",LOCK_SHA)
T0=time.perf_counter()

print("[1/9] Loading model...")
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
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | {NL}L | BF16 | frozen")

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

print("[2/9] Building frozen 32-sentence neutral PCA codebook...")
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
print("      Codebook ready.")

@torch.inference_mode()
def packet(source):
    K,V=forge(source);out={}
    for n,X,d in (("K",K,K_DIM),("V",V,V_DIM)):
        rows=[]
        for L in range(NL):
            mu,B=CB[L][n]
            coeff=torch.einsum("thi,hid->thd",X[L][1:].float().view(-1,NKV,HD)-mu,B[:,:,:d]).to(torch.bfloat16)
            content=(mu+torch.einsum("thd,hid->thi",coeff.float(),B[:,:,:d])).reshape(-1,KVD).to(torch.bfloat16)
            rows.append(torch.cat([X[L][:1],content]))
        out[n]=rows
    return out["K"],out["V"]

@torch.inference_mode()
def batch(q,kvs):
    qids=enc(FMT.format(q=q));BN=len(kvs);Tm=max(kv[2] for kv in kvs);nq=len(qids)
    try: cache=DynamicCache(config=cfg)
    except TypeError: cache=DynamicCache()
    for L in range(NL):
        Ks=[];Vs=[]
        for kv in kvs:
            k,v,p=kv[0][L],kv[1][L],Tm-kv[2]
            if p:
                k=torch.cat([k.new_zeros(1,NKV,p,HD),k],2)
                v=torch.cat([v.new_zeros(1,NKV,p,HD),v],2)
            Ks.append(k);Vs.append(v)
        cache.update(torch.cat(Ks).clone(),torch.cat(Vs).clone(),L)
    mask=torch.zeros(BN,Tm+nq,dtype=torch.long,device=DEVICE)
    for b,kv in enumerate(kvs): mask[b,Tm-kv[2]:]=1
    pos=torch.tensor([kv[2] for kv in kvs],device=DEVICE)[:,None]+torch.arange(nq,device=DEVICE)[None]
    ids=torch.tensor([qids]*BN,device=DEVICE);outs=[[] for _ in range(BN)];done=[False]*BN
    for _ in range(MAX_NEW):
        o=model(input_ids=ids,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=True,return_dict=True)
        nxt=o.logits[:,-1].float().argmax(-1).tolist()
        for b,t in enumerate(nxt):
            if not done[b]:
                if t in EOS: done[b]=True
                else: outs[b].append(t)
        if all(done): break
        feed=[EOS[0] if done[b] and EOS else nxt[b] for b in range(BN)]
        ids=torch.tensor([[t] for t in feed],device=DEVICE);pos=pos[:,-1:]+1
        mask=torch.cat([mask,torch.ones(BN,1,dtype=torch.long,device=DEVICE)],1)
    return [tok.decode(x,skip_special_tokens=True).strip() for x in outs]

def norm(s): return re.sub(r"[^\w-]+"," ",str(s).casefold()).strip()
def firstline(text): return next((x.strip() for x in str(text).splitlines() if x.strip()),"")
def is_none(text):
    n=norm(firstline(text))
    return n=="none" or n.startswith("none ")
VALID_IDS=[x[2] for x in RECORDS];VALID_PLACES=[x[3] for x in RECORDS]

def parse_id(text):
    line=firstline(text)
    if is_none(line): return None
    n=norm(line);hits=[x for x in VALID_IDS if norm(x) in n]
    return hits[0] if len(set(hits))==1 else None

def parse_place(text):
    line=firstline(text)
    if is_none(line): return None
    n=norm(line);hits=[x for x in VALID_PLACES if norm(x) in n]
    return hits[0] if len(set(hits))==1 else None

def decide(vals):
    u=sorted(set(x for x in vals if x is not None))
    return u[0] if len(u)==1 else None

print("[3/9] Forging 64 HELD-OUT independent cartridges...")
A_KVS=[];B_KVS=[];tf=time.perf_counter()
for i,(fam,obj,cid,place) in enumerate(RECORDS,1):
    sa,sb=sources(fam,obj,cid,place)
    KA,VA=packet(sa);KB,VB=packet(sb)
    A_KVS.append(install(KA,VA));B_KVS.append(install(KB,VB))
    del KA,VA,KB,VB
    print(f"      [{i:02d}/32] {fam} | {obj} -> {cid} -> {place}")
FORGE_SEC=time.perf_counter()-tf

print("\n[4/9] HELD-OUT STAGE1 FULL SCAN — OBJECT -> ID")
stage1=[];S1_FP=0;S1_NONE=0;S1_COLL=0;t1=time.perf_counter()
for i,(fam,obj,gold_id,gold_place) in enumerate(RECORDS):
    raw=batch(MW1.format(obj=obj),A_KVS)
    parsed=[parse_id(x) for x in raw];pred=decide(parsed);gold=parsed[i];ok=pred==gold_id
    wrong=[x for j,x in enumerate(parsed) if j!=i and x is not None]
    S1_FP+=len(wrong);S1_NONE+=sum(is_none(x) for j,x in enumerate(raw) if j!=i)
    S1_COLL+=int(pred is None and len(set(x for x in parsed if x is not None))>1)
    stage1.append((pred,ok,gold,len(wrong),raw,parsed))
    print(f"      [{i+1:02d}] {fam} | expected={gold_id:6s} | pred={str(pred):6s} | gold={str(gold):6s} | offFP={len(wrong):2d} | {'PASS' if ok else 'FAIL'}")
S1_SEC=time.perf_counter()-t1

print("\n[5/9] HELD-OUT STAGE2 FULL SCAN — ID -> PLACE · C_COMPACT FROZEN")
stage2=[];S2_FP=0;S2_NONE=0;S2_COLL=0;t2=time.perf_counter()
for i,((fam,obj,gold_id,gold_place),(pred_id,s1ok,_,_,_,_)) in enumerate(zip(RECORDS,stage1)):
    use_id=pred_id if pred_id is not None else gold_id
    raw=batch(MW2.format(cid=use_id),B_KVS)
    parsed=[parse_place(x) for x in raw];pred=decide(parsed);gold=parsed[i]
    linked=s1ok and pred==gold_place
    wrong=[x for j,x in enumerate(parsed) if j!=i and x is not None]
    S2_FP+=len(wrong);S2_NONE+=sum(is_none(x) for j,x in enumerate(raw) if j!=i)
    S2_COLL+=int(pred is None and len(set(x for x in parsed if x is not None))>1)
    stage2.append((pred,linked,gold,len(wrong),raw,parsed))
    print(f"      [{i+1:02d}] {fam} | expected={gold_place:20s} | pred={str(pred):20s} | gold={str(gold):20s} | offFP={len(wrong):2d} | {'PASS' if linked else 'FAIL'}")
S2_SEC=time.perf_counter()-t2

print("\n[6/9] HELD-OUT UNRELATED OBJECT ABSTENTION")
OBJ_NONE=0;OBJ_TOTAL=0
for obj in ABSENT_OBJECTS:
    raw=batch(MW1.format(obj=obj),A_KVS)
    n=sum(is_none(x) for x in raw);OBJ_NONE+=n;OBJ_TOTAL+=32
    print(f"      OBJECT {obj:20s} -> NONE {n}/32 | {'PASS' if n==32 else 'FAIL'}")

print("\n[7/9] HELD-OUT ABSENT-ID ABSTENTION")
ID_NONE=0;ID_TOTAL=0
for cid in ABSENT_IDS:
    raw=batch(MW2.format(cid=cid),B_KVS)
    n=sum(is_none(x) for x in raw);ID_NONE+=n;ID_TOTAL+=32
    print(f"      ID {cid:6s} -> NONE {n}/32 | {'PASS' if n==32 else 'FAIL'}")

print("\n[8/9] FAMILY / SEAL DIAGNOSTICS")
S1=sum(x[1] for x in stage1);LINKED=sum(x[1] for x in stage2);OFF_TOTAL=32*31
for fam in ("F1","F2","F3","F4"):
    idx=[i for i,x in enumerate(RECORDS) if x[0]==fam]
    a=sum(stage1[i][1] for i in idx);b=sum(stage2[i][1] for i in idx)
    print(f"      {fam}: Stage1 {a}/8 | Linked {b}/8")
print(f"      Stage1 off-target explicit outputs : {S1_FP}/{OFF_TOTAL}")
print(f"      Stage2 off-target explicit outputs : {S2_FP}/{OFF_TOTAL}")
print(f"      Stage1 off-target NONE             : {S1_NONE}/{OFF_TOTAL}")
print(f"      Stage2 off-target NONE             : {S2_NONE}/{OFF_TOTAL}")
print(f"      Stage1 aggregation collisions      : {S1_COLL}/32")
print(f"      Stage2 aggregation collisions      : {S2_COLL}/32")
print(f"      Unrelated object abstention        : {OBJ_NONE}/{OBJ_TOTAL}")
print(f"      Absent-ID abstention               : {ID_NONE}/{ID_TOTAL}")
print(f"      Forge time                         : {FORGE_SEC:.2f}s")
print(f"      Stage1 full-scan time              : {S1_SEC:.2f}s")
print(f"      Stage2 full-scan time              : {S2_SEC:.2f}s")
print(f"      Mean two-stage scan / record       : {(S1_SEC+S2_SEC)/32:.3f}s")

strict=(
S1==32 and LINKED==32 and
S1_FP==0 and S2_FP==0 and
S1_NONE==OFF_TOTAL and S2_NONE==OFF_TOTAL and
S1_COLL==0 and S2_COLL==0 and
OBJ_NONE==OBJ_TOTAL and ID_NONE==ID_TOTAL
)

print("\n"+"="*140)
print("[9/9] TEST482 FINAL RESULT")
print("="*140)
print("VALIDATION TYPE       : HELD-OUT / FROZEN READOUT")
print("RECORDS               : 32 NEW")
print("INDEPENDENT CARTRIDGES: 64")
print(f"STAGE1 OBJECT->ID     : {S1}/32")
print(f"LINKED TWO-HOP        : {LINKED}/32")
print(f"STAGE1 FALSE OUTPUTS  : {S1_FP}/{OFF_TOTAL}")
print(f"STAGE2 FALSE OUTPUTS  : {S2_FP}/{OFF_TOTAL}")
print(f"STAGE1 OFF-TARGET NONE: {S1_NONE}/{OFF_TOTAL}")
print(f"STAGE2 OFF-TARGET NONE: {S2_NONE}/{OFF_TOTAL}")
print(f"STAGE1 COLLISIONS     : {S1_COLL}/32")
print(f"STAGE2 COLLISIONS     : {S2_COLL}/32")
print(f"UNRELATED OBJECT NONE : {OBJ_NONE}/{OBJ_TOTAL}")
print(f"ABSENT-ID NONE        : {ID_NONE}/{ID_TOTAL}")
for fam in ("F1","F2","F3","F4"):
    idx=[i for i,x in enumerate(RECORDS) if x[0]==fam]
    print(f"{fam}: Stage1 {sum(stage1[i][1] for i in idx)}/8 | Linked {sum(stage2[i][1] for i in idx)}/8")
print("-"*140)
print("LOCK SHA:",LOCK_SHA)
print("VERDICT:","PASS_FINAL_HELDOUT_32_BANK_SEAL" if strict else "FAIL_FINAL_HELDOUT_32_BANK_SEAL")

if not strict:
    print("\nFAILURE DETAILS")
    for i,((fam,obj,cid,place),s1,s2) in enumerate(zip(RECORDS,stage1,stage2)):
        if not s1[1] or not s2[1] or s1[3] or s2[3]:
            print("-"*140)
            print(f"[{i+1:02d}] {fam} | {obj} -> {cid} -> {place}")
            print(f"Stage1: pred={s1[0]!r} gold_read={s1[2]!r} offFP={s1[3]}")
            if not s1[1] or s1[3]:
                for j,(rr,pp) in enumerate(zip(s1[4],s1[5])):
                    if j==i or pp is not None:
                        print(f"  A[{j:02d}] parsed={pp!r} raw={rr!r}")
            print(f"Stage2: pred={s2[0]!r} gold_read={s2[2]!r} offFP={s2[3]}")
            if not s2[1] or s2[3]:
                for j,(rr,pp) in enumerate(zip(s2[4],s2[5])):
                    if j==i or pp is not None:
                        print(f"  B[{j:02d}] parsed={pp!r} raw={rr!r}")

print("-"*140)
print(f"TOTAL TEST TIME: {time.perf_counter()-T0:.2f}s")
print("ENGINE: Qwen2.5-7B-Instruct · frozen · K120/V128/OWN · full scan")
print("STAGE2 READOUT: C_COMPACT · FROZEN BEFORE HELD-OUT")
print("NONE-FIRST PARSER: FROZEN")
print("MEMORY/RETRIEVAL MECHANISM CHANGES: NONE")
print("POST-HOC READOUT SELECTION/TUNING: NONE")
print("="*140)

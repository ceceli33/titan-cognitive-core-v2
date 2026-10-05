# TEST481 — AKBASCORE NIRVANA QWEN · 32-BANK UNIVERSAL READOUT FINAL SELECTION
# Same TEST479 bank/engine: 32 records · 64 independent cartridges · K120/V128/OWN · FULL SCAN.
# TEST480 strict winners compared on the COMPLETE 32-record bank:
#   C_COMPACT = Container "{cid}" location? Answer only the location if stated; otherwise answer NONE.
#   D_RECORD  = According to this record, where is container "{cid}" located? Answer only the location, or NONE if unknown.
# Stage1 runs ONCE and is shared by both Stage2 candidates.
# TEST476 NONE-first parser fix retained.
# No router/index/signature/training/LoRA/gradient/new motor.
#
# STRICT PASS PER CANDIDATE:
#   Linked 32/32
#   Stage2 off-target FP 0/992
#   Stage2 off-target NONE 992/992
#   Stage2 collisions 0/32
#   Absent-ID NONE 128/128
#
# Stage1 baseline must remain:
#   32/32 · off-target FP 0/992 · NONE 992/992 · collisions 0/32

import os,sys,subprocess,importlib.util,random,re,time
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None: subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

TEST="481";SEED=461;MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
K_DIM=120;V_DIM=128;MAX_NEW=32
FMT="QUESTION:\n{q}\n\n ANSWER:";SEP="\n\n"
STYLE=" Give only the answer on the first line; do not explain."
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

RECORDS=[
("F1","amber compass","AX-731","north archive"),
("F1","silver lantern","KM-204","river annex"),
("F1","copper violin","QP-583","stone gallery"),
("F1","marble clock","HV-619","garden vault"),
("F2","crimson telescope","RQ-415","saffron lighthouse"),
("F2","ivory notebook","BT-962","slate pavilion"),
("F2","bronze astrolabe","MC-916","elm lodge"),
("F2","velvet map","ZX-247","cedar gallery"),
("F3","jade camera","DN-348","willow chamber"),
("F3","glass trumpet","PL-570","harbor studio"),
("F3","linen journal","GF-821","orchard room"),
("F3","iron pendant","WS-436","maple hall"),
("F4","pearl radio","CJ-105","cliff workshop"),
("F4","oak tablet","YU-674","meadow depot"),
("F4","blue teapot","ER-293","pine observatory"),
("F4","golden ruler","NV-852","lake conservatory"),
("F1","opal sextant","LA-317","birch archive"),
("F1","scarlet flute","TG-684","granite annex"),
("F1","ceramic globe","PK-529","rose gallery"),
("F1","walnut mirror","SD-173","canyon vault"),
("F2","indigo prism","JF-804","juniper tower"),
("F2","brass sundial","UC-251","moss pavilion"),
("F2","silk atlas","WB-690","ash lodge"),
("F2","quartz bell","OY-438","laurel gallery"),
("F3","coral camera","IE-725","alder chamber"),
("F3","steel trumpet","XR-364","marina studio"),
("F3","cotton ledger","FK-918","cypress room"),
("F3","onyx pendant","VA-542","spruce hall"),
("F4","crystal radio","GH-286","ridge workshop"),
("F4","maple tablet","TZ-607","prairie depot"),
("F4","white teapot","BL-159","cedar observatory"),
("F4","silver ruler","QD-873","bay conservatory")
]

def sources(fam,obj,cid,place):
    if fam=="F1": return f"The {obj} is stored in container {cid}.",f"Container {cid} is located in the {place}."
    if fam=="F2": return f"Container {cid} contains the {obj}.",f"The {place} houses container {cid}."
    if fam=="F3": return f"The {obj} can be found inside container {cid}.",f"The location of container {cid} is the {place}."
    if fam=="F4": return f"Inside container {cid} there is the {obj}.",f"Container {cid} can be found at the {place}."
    raise ValueError(fam)

MW1='Which container contains the {obj} according to this record? Give only the container identifier. If this record does not contain the {obj}, answer NONE.'+STYLE
READOUTS=[
("C_COMPACT",'Container "{cid}" location? Answer only the location if stated; otherwise answer NONE.'),
("D_RECORD",'According to this record, where is container "{cid}" located? Answer only the location, or NONE if unknown.')
]
ABSENT=["ZZ-999","AA-000","LM-777","RX-111"]

print("="*136)
print("TEST481 — AKBASCORE NIRVANA QWEN · 32-BANK UNIVERSAL READOUT FINAL SELECTION")
print("32 RECORDS · 64 CARTRIDGES · K120/V128/OWN · FULL SCAN · C_COMPACT vs D_RECORD")
print("="*136)
T0=time.perf_counter()

print("[1/8] Loading model...")
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

print("[2/8] Building original 32-sentence neutral PCA codebook...")
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
    line=firstline(text);n=norm(line)
    if n=="none" or n.startswith("none "): return None
    hits=[x for x in VALID_PLACES if norm(x) in n]
    return hits[0] if len(set(hits))==1 else None

def decide(vals):
    u=sorted(set(x for x in vals if x is not None))
    return u[0] if len(u)==1 else None

print("[3/8] Forging 64 independent cartridges...")
A_KVS=[];B_KVS=[];tf=time.perf_counter()
for i,(fam,obj,cid,place) in enumerate(RECORDS,1):
    sa,sb=sources(fam,obj,cid,place)
    KA,VA=packet(sa);KB,VB=packet(sb)
    A_KVS.append(install(KA,VA));B_KVS.append(install(KB,VB))
    del KA,VA,KB,VB
    print(f"      [{i:02d}/32] {fam} | {obj} -> {cid} -> {place}")
FORGE_SEC=time.perf_counter()-tf

print("\n[4/8] SHARED STAGE1 FULL SCAN — OBJECT -> ID")
stage1=[];S1_NONE=0;S1_FP=0;S1_COLL=0;t1=time.perf_counter()
for i,(fam,obj,gold_id,gold_place) in enumerate(RECORDS):
    raw=batch(MW1.format(obj=obj),A_KVS)
    parsed=[parse_id(x) for x in raw];pred=decide(parsed);ok=pred==gold_id;gold_out=parsed[i]
    wrong=[x for j,x in enumerate(parsed) if j!=i and x is not None]
    S1_NONE+=sum(is_none(x) for j,x in enumerate(raw) if j!=i);S1_FP+=len(wrong)
    S1_COLL+=int(pred is None and len(set(x for x in parsed if x is not None))>1)
    stage1.append((pred,ok,gold_out,len(wrong)))
    print(f"      [{i+1:02d}] {fam} | expected={gold_id:6s} | pred={str(pred):6s} | gold={str(gold_out):6s} | offFP={len(wrong):2d} | {'PASS' if ok else 'FAIL'}")
S1_SEC=time.perf_counter()-t1;S1=sum(x[1] for x in stage1);OFF_TOTAL=32*31

print("\n[5/8] COMPLETE STAGE2 COMPARISON — C_COMPACT vs D_RECORD")
results={}
for name,template in READOUTS:
    print("\n"+"-"*136)
    print(name)
    print("-"*136)
    rows=[];FP=0;NONE=0;COLL=0;ts=time.perf_counter()
    for i,((fam,obj,gold_id,gold_place),(pred_id,s1ok,_,_)) in enumerate(zip(RECORDS,stage1)):
        use_id=pred_id if pred_id is not None else gold_id
        raw=batch(template.format(cid=use_id),B_KVS)
        parsed=[parse_place(x) for x in raw];pred=decide(parsed)
        linked=s1ok and pred==gold_place;gold_out=parsed[i]
        wrong=[x for j,x in enumerate(parsed) if j!=i and x is not None]
        FP+=len(wrong);NONE+=sum(is_none(x) for j,x in enumerate(raw) if j!=i)
        COLL+=int(pred is None and len(set(x for x in parsed if x is not None))>1)
        rows.append((pred,linked,gold_out,len(wrong),raw,parsed))
        print(f"      [{i+1:02d}] {fam} | expected={gold_place:20s} | pred={str(pred):20s} | gold={str(gold_out):20s} | offFP={len(wrong):2d} | {'PASS' if linked else 'FAIL'}")
    sec=time.perf_counter()-ts
    results[name]={"rows":rows,"linked":sum(x[1] for x in rows),"fp":FP,"none":NONE,"coll":COLL,"sec":sec}

print("\n[6/8] ABSENT-ID ABSTENTION — COMPLETE 32-BANK")
for name,template in READOUTS:
    okn=0;total=0
    for cid in ABSENT:
        raw=batch(template.format(cid=cid),B_KVS)
        n=sum(is_none(x) for x in raw)
        okn+=n;total+=32
        print(f"      {name:9s} | {cid:6s} -> NONE {n}/32 | {'PASS' if n==32 else 'FAIL'}")
    results[name]["abs_ok"]=okn;results[name]["abs_total"]=total

print("\n[7/8] FAMILY / STRICT REGRESSION SUMMARY")
print(f"      SHARED STAGE1: {S1}/32 | offFP={S1_FP}/{OFF_TOTAL} | offNONE={S1_NONE}/{OFF_TOTAL} | collisions={S1_COLL}/32")
strict=[]
for name,_ in READOUTS:
    r=results[name]
    print("\n      "+name)
    for fam in ("F1","F2","F3","F4"):
        idx=[i for i,x in enumerate(RECORDS) if x[0]==fam]
        n=sum(r["rows"][i][1] for i in idx)
        print(f"        {fam}: Linked {n}/8")
    ok=(S1==32 and S1_FP==0 and S1_NONE==OFF_TOTAL and S1_COLL==0 and
        r["linked"]==32 and r["fp"]==0 and r["none"]==OFF_TOTAL and r["coll"]==0 and
        r["abs_ok"]==r["abs_total"])
    r["strict"]=ok
    if ok: strict.append(name)
    print(f"        Linked               : {r['linked']}/32")
    print(f"        Stage2 off-target FP : {r['fp']}/{OFF_TOTAL}")
    print(f"        Stage2 off-target NONE: {r['none']}/{OFF_TOTAL}")
    print(f"        Stage2 collisions    : {r['coll']}/32")
    print(f"        Absent-ID NONE       : {r['abs_ok']}/{r['abs_total']}")
    print(f"        Stage2 scan time     : {r['sec']:.2f}s")
    print(f"        STRICT              : {'PASS' if ok else 'FAIL'}")

print("\n"+"="*136)
print("[8/8] TEST481 RESULT")
print("="*136)
print("RECORDS               : 32")
print("INDEPENDENT CARTRIDGES: 64")
print(f"SHARED STAGE1         : {S1}/32")
print(f"STAGE1 FALSE OUTPUTS  : {S1_FP}/{OFF_TOTAL}")
print(f"STAGE1 OFF-TARGET NONE: {S1_NONE}/{OFF_TOTAL}")
print(f"STAGE1 COLLISIONS     : {S1_COLL}/32")
print("-"*136)
for name,_ in READOUTS:
    r=results[name]
    print(f"{name:9s} | Linked {r['linked']}/32 | FP {r['fp']}/{OFF_TOTAL} | NONE {r['none']}/{OFF_TOTAL} | Coll {r['coll']}/32 | Absent {r['abs_ok']}/{r['abs_total']} | {'STRICT PASS' if r['strict'] else 'FAIL'}")
print("-"*136)
print("STRICT WINNERS:",strict if strict else "NONE")
if len(strict)==2:
    verdict="BOTH_UNIVERSAL_READOUTS_PASS_32_BANK"
elif len(strict)==1:
    verdict=f"FINAL_READOUT_CANDIDATE_{strict[0]}"
else:
    verdict="NO_UNIVERSAL_READOUT_PASSES_32_BANK"
print("VERDICT:",verdict)

if len(strict)<2:
    print("\nFAILURE DETAILS")
    for name,_ in READOUTS:
        if results[name]["strict"]: continue
        print("\n"+name)
        for i,((fam,obj,cid,place),row) in enumerate(zip(RECORDS,results[name]["rows"])):
            pred,linked,gold_out,offfp,raw,parsed=row
            if not linked or offfp:
                print("-"*136)
                print(f"[{i+1:02d}] {fam} | {obj} -> {cid} -> {place}")
                print(f"pred={pred!r} gold_read={gold_out!r} offFP={offfp}")
                for j,(rr,pp) in enumerate(zip(raw,parsed)):
                    if j==i or pp is not None:
                        print(f"  B[{j:02d}] parsed={pp!r} raw={rr!r}")

print("-"*136)
print(f"Forge time       : {FORGE_SEC:.2f}s")
print(f"Stage1 scan time : {S1_SEC:.2f}s")
print(f"TOTAL TEST TIME  : {time.perf_counter()-T0:.2f}s")
print("ENGINE: Qwen2.5-7B-Instruct · frozen · K120/V128/OWN · full scan")
print("MEMORY/RETRIEVAL MECHANISM CHANGES: NONE")
print("EXPERIMENTAL VARIABLE: C_COMPACT vs D_RECORD Stage2 readout only")
print("TEST476 NONE-FIRST PARSER FIX: RETAINED")
print("="*136)

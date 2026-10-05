# TEST480 — QWEN STAGE2 UNIVERSAL READOUT ISOLATION
# Purpose: Find ONE family-independent Stage2 readout that resolves both known boundary cases.
# Same Qwen2.5-7B-Instruct · same K120/V128/OWN cartridge engine · same neutral PCA codebook.
# No bank/router/index/signature/training/LoRA/gradient/motor change.
#
# Four isolated Stage2 cartridges:
#   F1 AX-731 -> north archive       (stable control)
#   F2 RQ-415 -> saffron lighthouse (DIRECT_LOCATION failure in TEST479)
#   F3 PL-570 -> harbor studio       (VERIFY failure in TEST477)
#   F4 CJ-105 -> cliff workshop      (stable control)
#
# Candidate readouts:
# A = TEST477 VERIFY
# B = TEST479 DIRECT_LOCATION
# C = COMPACT LOCATION
# D = RECORD WHERE
# E = MINIMAL COMPLETION
#
# A universal candidate must:
#   1) read all four gold cartridges correctly
#   2) abstain on all three non-gold cartridges for each query
#   3) produce zero aggregation collisions
#   4) abstain on absent IDs across all four cartridges
#
# Winning wording is NOT authorized globally here.
# It must next pass the complete 32-bank regression.

import os,sys,subprocess,importlib.util,random,re,time
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None: subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

TEST="480";SEED=461;MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
K_DIM=120;V_DIM=128;MAX_NEW=32
FMT="QUESTION:\n{q}\n\n ANSWER:";SEP="\n\n";DEVICE=torch.device("cuda")
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

CASES=[
("F1","AX-731","north archive","Container AX-731 is located in the north archive."),
("F2","RQ-415","saffron lighthouse","The saffron lighthouse houses container RQ-415."),
("F3","PL-570","harbor studio","The location of container PL-570 is the harbor studio."),
("F4","CJ-105","cliff workshop","Container CJ-105 can be found at the cliff workshop.")
]

READOUTS=[
("A_VERIFY",'Does this memory explicitly contain container "{cid}"? If yes, answer only its location. If no, answer exactly NONE.'),
("B_DIRECT",'What location does this memory state for container "{cid}"? If no location is stated, answer exactly NONE.'),
("C_COMPACT",'Container "{cid}" location? Answer only the location if stated; otherwise answer NONE.'),
("D_RECORD",'According to this record, where is container "{cid}" located? Answer only the location, or NONE if unknown.'),
("E_MINIMAL",'Location of container "{cid}":')
]

ABSENT=["ZZ-999","AA-000","LM-777","RX-111"]

print("="*132)
print("TEST480 — QWEN STAGE2 UNIVERSAL READOUT ISOLATION")
print("4 ISOLATED STAGE2 CARTRIDGES · F1/F2/F3/F4 · K120/V128/OWN · 5 READOUT CANDIDATES")
print("="*132)
T0=time.perf_counter()

print("[1/7] Loading model...")
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

print("[2/7] Building original 32-sentence neutral PCA codebook...")
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
PLACES=[x[2] for x in CASES]

def parse_place(text):
    line=firstline(text);n=norm(line)
    if n=="none" or n.startswith("none "): return None
    hits=[x for x in PLACES if norm(x) in n]
    return hits[0] if len(set(hits))==1 else None

def decide(vals):
    u=sorted(set(x for x in vals if x is not None))
    return u[0] if len(u)==1 else None

print("[3/7] Forging four isolated Stage2 cartridges...")
KVS=[]
for i,(fam,cid,place,source) in enumerate(CASES,1):
    K,V=packet(source);KVS.append(install(K,V));del K,V
    print(f"      [{i}/4] {fam} | {cid} -> {place} | {source}")

print("\n[4/7] GOLD + OFF-TARGET READOUT MATRIX")
results={}
for name,template in READOUTS:
    print("\n"+"-"*132)
    print(name)
    print("-"*132)
    gold_ok=0;off_fp=0;off_none=0;collisions=0;rows=[]
    for i,(fam,cid,place,source) in enumerate(CASES):
        q=template.format(cid=cid)
        raw=batch(q,KVS)
        parsed=[parse_place(x) for x in raw]
        pred=decide(parsed)
        gok=parsed[i]==place
        linked=pred==place
        wrong=[x for j,x in enumerate(parsed) if j!=i and x is not None]
        none=sum(is_none(x) for j,x in enumerate(raw) if j!=i)
        collision=int(pred is None and len(set(x for x in parsed if x is not None))>1)
        gold_ok+=int(gok and linked);off_fp+=len(wrong);off_none+=none;collisions+=collision
        rows.append((fam,cid,place,raw,parsed,pred,gok,linked,wrong,none))
        print(f"      {fam} {cid} | expected={place:18s} | gold={str(parsed[i]):18s} | pred={str(pred):18s} | offFP={len(wrong)} | {'PASS' if gok and linked else 'FAIL'}")
    results[name]={"gold":gold_ok,"fp":off_fp,"none":off_none,"coll":collisions,"rows":rows}

print("\n[5/7] ABSENT-ID ABSTENTION")
for name,template in READOUTS:
    total=0;okn=0
    for cid in ABSENT:
        raw=batch(template.format(cid=cid),KVS)
        n=sum(is_none(x) for x in raw)
        total+=4;okn+=n
    results[name]["abs_ok"]=okn
    results[name]["abs_total"]=total
    print(f"      {name:12s} -> NONE {okn}/{total} | {'PASS' if okn==total else 'FAIL'}")

print("\n[6/7] CANDIDATE SUMMARY")
winners=[]
for name,_ in READOUTS:
    r=results[name]
    strict=(r["gold"]==4 and r["fp"]==0 and r["none"]==12 and r["coll"]==0 and r["abs_ok"]==r["abs_total"])
    if strict: winners.append(name)
    print(f"      {name:12s} | gold={r['gold']}/4 | offFP={r['fp']}/12 | offNONE={r['none']}/12 | collisions={r['coll']}/4 | absent={r['abs_ok']}/{r['abs_total']} | {'STRICT PASS' if strict else 'FAIL'}")

print("\n"+"="*132)
print("[7/7] TEST480 RESULT")
print("="*132)
print("KNOWN BOUNDARIES:")
print("      F2 RQ-415 -> saffron lighthouse")
print("      F3 PL-570 -> harbor studio")
print("CONTROLS:")
print("      F1 AX-731 -> north archive")
print("      F4 CJ-105 -> cliff workshop")
print("-"*132)

if len(winners)==1:
    verdict="UNIVERSAL_READOUT_CANDIDATE_FOUND"
elif len(winners)>1:
    verdict="MULTIPLE_UNIVERSAL_CANDIDATES"
else:
    verdict="NO_UNIVERSAL_READOUT_CANDIDATE"

print("STRICT WINNERS:",winners if winners else "NONE")
print("VERDICT:",verdict)

if winners:
    print("-"*132)
    print("Candidate(s) passed both boundary cases, F1/F4 controls, off-target abstention and absent-ID abstention.")
    print("No candidate is authorized as the 32-bank readout until full TEST477-bank regression.")
else:
    print("-"*132)
    print("FAILURE RAW DETAILS")
    for name,_ in READOUTS:
        r=results[name]
        if r["gold"]<4 or r["fp"]>0 or r["none"]<12:
            print("\n"+name)
            for row in r["rows"]:
                fam,cid,place,raw,parsed,pred,gok,linked,wrong,none=row
                if not (gok and linked) or wrong or none<3:
                    print(f"  {fam} {cid} expected={place!r} pred={pred!r}")
                    for j,x in enumerate(raw):
                        print(f"    K[{j}] parsed={parsed[j]!r} raw={x!r}")

print("-"*132)
print(f"TOTAL TEST TIME: {time.perf_counter()-T0:.2f}s")
print("ENGINE: Qwen2.5-7B-Instruct · frozen · K120/V128/OWN")
print("MEMORY MOTOR CHANGES: NONE")
print("EXPERIMENTAL VARIABLE: Stage2 natural-language readout only")
print("="*132)

# TEST475 — QWEN F2 SINGLE-MODULE READOUT DIAGNOSIS
# Existing TEST461 NIRVANA engine unchanged.
# Only variable: MW2 readout wording. No router, scorer, training, LoRA, hooks or new motor.

import os,sys,subprocess,importlib.util,random,re
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None: subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])

import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

TEST="475";SEED=461;MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
K_DIM=120;V_DIM=128;MAX_NEW=32
FMT="QUESTION:\n{q}\n\n ANSWER:"
SEP="\n\n"
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

# Same F2 relation used by the sealed Qwen panel.
# Four independent ID→PLACE cartridges; no multi-cartridge aggregation in TEST475.
CASES=[
("RQ-415","saffron lighthouse"),
("BT-962","slate pavilion"),
("MC-916","elm lodge"),
("ZX-247","cedar gallery"),
]

# A = exact current Qwen MW2.
# B = same memory, same engine; only explicit verification wording.
MW2_A="Where is container {cid} located according to this record? Give the place name. If this record does not say where container {cid} is, answer NONE."+STYLE
MW2_B='Does this memory explicitly contain container "{cid}"? If yes, answer only its location. If no, answer exactly NONE.'

print("="*118)
print("TEST475 — QWEN F2 SINGLE-MODULE READOUT DIAGNOSIS")
print("TEST461 ENGINE UNCHANGED · K120/V128/OWN · F2 ONLY · CURRENT MW2 vs VERIFY MW2")
print("="*118)
print("[1/5] Loading model...")
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

print("[2/5] Building original 32-sentence neutral PCA codebook...")
CB=[]
CO=[forge(s) for s in CORPUS]
for L in range(NL):
    e={}
    for j,n in enumerate(("K","V")):
        R=torch.cat([c[j][L][1:] for c in CO]).float().view(-1,NKV,HD)
        MU=[];B=[]
        for h in range(NKV):
            X=R[:,h];mu=X.mean(0)
            _,S,Vh=torch.linalg.svd(X-mu,full_matrices=False)
            m=min(128,Vh.shape[0]);bb=Vh[:m].T.contiguous()
            if m<128: bb=torch.nn.functional.pad(bb,(0,128-m))
            MU.append(mu);B.append(bb)
        e[n]=(torch.stack(MU),torch.stack(B))
    CB.append(e)
del CO
torch.cuda.empty_cache()
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
        ids=torch.tensor([[t] for t in feed],device=DEVICE)
        pos=pos[:,-1:]+1
        mask=torch.cat([mask,torch.ones(BN,1,dtype=torch.long,device=DEVICE)],1)
    return [tok.decode(x,skip_special_tokens=True).strip() for x in outs]

def norm(s): return re.sub(r"[^\w]+"," ",str(s).casefold()).strip()
def place_hit(text,place):
    line=next((x.strip() for x in str(text).splitlines() if x.strip()),"")
    n=norm(line);p=norm(place)
    return n==p or n.startswith(p+" ") or p in n

print("[3/5] Forging four F2 cartridges with existing K120/V128/OWN engine...")
KVS=[]
for cid,place in CASES:
    source=f"The {place} houses container {cid}."
    K,V=packet(source)
    KVS.append(install(K,V))
    del K,V
    print(f"      {cid} -> {place} | {source}")

print("\n[4/5] SINGLE-MODULE READOUT")
print("-"*118)
results=[]
for (cid,place),kv in zip(CASES,KVS):
    qa=MW2_A.format(cid=cid)
    qb=MW2_B.format(cid=cid)
    a=batch(qa,[kv])[0]
    b=batch(qb,[kv])[0]
    ha=place_hit(a,place);hb=place_hit(b,place)
    results.append((cid,place,a,ha,b,hb))
    print(f"\n{cid} -> expected: {place}")
    print(f"  A CURRENT : {repr(a)}")
    print(f"  A HIT     : {'PASS' if ha else 'FAIL'}")
    print(f"  B VERIFY  : {repr(b)}")
    print(f"  B HIT     : {'PASS' if hb else 'FAIL'}")

A=sum(x[3] for x in results);B=sum(x[5] for x in results)
rescued=sum((not x[3]) and x[5] for x in results)
lost=sum(x[3] and (not x[5]) for x in results)

print("\n"+"="*118)
print("[5/5] TEST475 RESULT")
print("="*118)
print(f"CURRENT MW2 : {A}/{len(CASES)}")
print(f"VERIFY MW2  : {B}/{len(CASES)}")
print(f"RESCUED     : {rescued}")
print(f"REGRESSED   : {lost}")
print("-"*118)

if A<4 and B==4 and rescued>0 and lost==0:
    verdict="READOUT_FRAMING_CAUSAL_CANDIDATE"
elif A==B:
    verdict="NO_FRAMING_RESCUE"
elif B>A:
    verdict="PARTIAL_FRAMING_RESCUE"
else:
    verdict="VERIFY_FRAMING_WORSE"

print("VERDICT:",verdict)
print("="*118)
print("Interpretation:")
print("  - Cartridge compression/forge/install were identical in A and B.")
print("  - Each F2 cartridge was read alone; aggregation/inter-cartridge interference is absent.")
print("  - Therefore an A->B rescue isolates the effect to readout wording under the existing engine.")
print("  - TEST476 should only be designed after this result is observed.")
print("="*118)

# Copyright © 2026 Mustafa Akbaş
# New original AKBASCORE MAM material is source-available under the repository LICENSE.
# Earlier MIT-licensed Titan Cognitive Core / AkbasCore material remains under its original license.
# See repository LICENSE for complete terms.
#
# TEST497 — AKBASCORE MAM × DRA · RELATIONAL-PATH PRESSURE
# FROZEN TEST482 K120/V128/OWN → TEST495 PARALLEL WORKSPACE
# → AKBASCORE 3.0 SEASC/DRA L0-L19 → MULTI-BELLEKÖZ RELATIONAL COMPLETION
#
# QUESTION:
# Can target-blind layerwise activation pressure make frozen Qwen continue
# a multi-memory relational path instead of collapsing onto an intermediate label?
#
# TEST496 BASELINE FAILURE:
# Meridian → ORIN → [shortcut] → ORIN-23
#
# REQUIRED:
# Meridian → ORIN → TAV → Q
# Meridian → SEL → 23
# Q + 23 → Q-23
#
# DRA DOES NOT CONTAIN:
# Meridian / ORIN / TAV / SEL / Q / 23 / Q-23 / Z / 57.
#
# FROZEN MAM:
# TEST482 K120/V128/OWN
# TEST495 parallel sequence-dimension workspace
#
# FROZEN SEASC/DRA REFERENCE:
# L0-L19
# IVME=0.10
# SONUM=0.30
# ZIRVE=0.70
# TABAN=0.20
# A_L = normalize(mean(h_POS)-mean(h_NEG))
# rho_L = IVME * E(L)
# Delta h_L = rho_L * ||h_L|| * A_L
#
# NO:
# training / LoRA / router / Top-K / ANN / graph / VTOKEN / answer vector
# dose search / layer search / head search / post-hoc tuning

import os,sys,subprocess,importlib.util,random,time,hashlib,json,itertools,math
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

TEST="497";SEED=497;MODEL_ID="Qwen/Qwen2.5-7B-Instruct";DEVICE=torch.device("cuda")
K_DIM=120;V_DIM=128;MAX_NEW=20
FMT="QUESTION:\n{q}\n\n ANSWER:";SEP="\n\n";TARGET="Q-23"
QUESTION="According to the available records, what is the complete current recovery code of the missing Meridian instrument? Give only the exact code. If the complete code cannot be logically determined from the available records, answer NONE."
TEST482_BLOB_SHA="ecc630144a44479cb35c432c50eb0d09bc6866d7"
TEST482_LOCK_SHA="fa59fd38661e558f6bae22eedff0999f32f8f6e9d4a08932383525323a5fe687"
TEST495_LOCK_SHA="2bae3f2497fefb50b3d18d8fb7035c63a08fdd67d80077dd92f4e9c56eca787e"
TEST496_LOCK_SHA="5985c4917ad98c83537990b4cb54a7a3a751dbdb1a3b82774839c3cafac8ca67"

# Frozen AkbasCore 3.0 SEASC/DRA reference configuration.
N_STEER=20;IVME=0.10;SONUM=0.30;ZIRVE=0.70;TABAN=0.20

# Target-blind matched-contrastive axis.
# Same linguistic skeleton; only relational-computation policy changes.
POSITIVE=[
"The solver follows every intermediate relation until the terminal mapping is resolved.",
"The solver traces each available link until the final mapped value is reached.",
"The solver continues through intermediate mappings before producing the final result.",
"The solver resolves the complete relation chain rather than stopping at an intermediate label.",
"The reasoner follows successive links until the terminal relation determines the answer.",
"The reasoner carries the relation through every intermediate mapping to its final value.",
"The reasoner completes all available relational steps before returning the result.",
"The reasoner uses intermediate labels only as links toward the terminal mapped value."
]
NEGATIVE=[
"The solver stops at the first intermediate relation before the terminal mapping is resolved.",
"The solver uses the first available link instead of reaching the final mapped value.",
"The solver stops at an intermediate mapping before producing the final result.",
"The solver treats an intermediate label as the result rather than completing the relation chain.",
"The reasoner stops after an early link before the terminal relation determines the answer.",
"The reasoner uses an intermediate mapping directly instead of carrying the relation to its final value.",
"The reasoner returns after an early relational step instead of completing all available steps.",
"The reasoner uses an intermediate label as the answer instead of following it toward the terminal mapped value."
]

MEMORIES=[
("M1","The missing Meridian instrument carries seal ORIN. The Atlas instrument carries seal NERA. The Helix instrument carries seal PAVO."),
("M2","In the recovery ledger, seal ORIN corresponds to routing class TAV; seal NERA corresponds to class BEX; seal PAVO corresponds to class LUM."),
("M3","Routing class TAV is assigned archive section Q; class BEX is assigned section R; class LUM is assigned section V."),
("M4","The missing Meridian instrument belongs to transfer batch SEL. The Atlas instrument belongs to batch DOR. The Helix instrument belongs to batch NIM."),
("M5","Transfer batch SEL uses active recovery bay 23; batch DOR uses bay 41; batch NIM uses bay 68."),
("M6","A complete recovery code is constructed from the instrument's archive section letter, then a hyphen, then its active recovery bay number.")
]
DISTRACTORS=[
("D1","In the coastal weather survey, station Aster uses wind scale 14 while station Brine uses wind scale 37."),
("D2","The botanical archive maps cedar samples to tray J and maple samples to tray W."),
("D3","For the lunar photography project, camera group Rho uses exposure sequence 52 and group Sigma uses sequence 19."),
("D4","The marine catalog assigns coral specimens to shelf Q while shell specimens are assigned to shelf M."),
("D5","In the railway timetable, route Delta reaches platform 23 while route Gamma reaches platform 44."),
("D6","The acoustics laboratory labels resonance family TAV as experiment group C and family BEX as experiment group H."),
("D7","The astronomy notebook associates marker ORIN with star field 71 and marker NERA with star field 16."),
("D8","The greenhouse inventory uses code Q-23 for a fertilizer cabinet unrelated to recovery instruments.")
]
SWAPS=[
("S3","Routing class TAV is assigned archive section Z; class BEX is assigned section R; class LUM is assigned section V."),
("S5","Transfer batch SEL uses active recovery bay 57; batch DOR uses bay 41; batch NIM uses bay 68.")
]
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

# Exact leakage audit for experiment-specific symbols/names.
FORBIDDEN=["meridian","orin","tav","sel","nera","pavo","bex","lum","dor","nim","q-23","z-23","q-57","23","57"]
axis_text=" ".join(POSITIVE+NEGATIVE).casefold()
LEAK=[x for x in FORBIDDEN if x.casefold() in axis_text]
assert not LEAK,f"DRA AXIS TARGET LEAK: {LEAK}"
assert TARGET.casefold() not in " ".join(x[1] for x in MEMORIES).casefold()

LOCK={"test":TEST,"seed":SEED,"model":MODEL_ID,"mam":"TEST496 exact baseline",
"memory_core":"TEST482 K120/V128/OWN","workspace":"TEST495 PARALLEL_SEQUENCE_CONCAT",
"test482_blob":TEST482_BLOB_SHA,"test482_lock":TEST482_LOCK_SHA,"test495_lock":TEST495_LOCK_SHA,"test496_lock":TEST496_LOCK_SHA,
"target":TARGET,"question":QUESTION,"memories":MEMORIES,"distractors":DISTRACTORS,"swaps":SWAPS,"corpus":CORPUS,
"dra":{"layers":"L0-L19","ivme":IVME,"sonum":SONUM,"zirve":ZIRVE,"taban":TABAN,"positive":POSITIVE,"negative":NEGATIVE},
"address":"OFF","retrieval":"ORACLE","router":"NONE","topk":"NONE","ann":"NONE","graph":"NONE",
"training":"NONE","dose_search":"NONE","layer_search":"NONE","head_search":"NONE","posthoc_tuning":"NONE"}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
torch.set_grad_enabled(False)
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*160)
print("TEST497 — AKBASCORE MAM × DRA · RELATIONAL-PATH PRESSURE")
print("TEST496 EXACT MAM → TARGET-BLIND SEASC/DRA L0-L19 → MULTI-BELLEKÖZ RELATIONAL COMPLETION")
print("="*160)
print("LOCK SHA:",LOCK_SHA)
print("TEST496 LOCK:",TEST496_LOCK_SHA)
print("TEST482 BLOB:",TEST482_BLOB_SHA)
print("TARGET:",TARGET)
print("DRA TARGET LEAK: NONE")
print(f"DRA: L0-L{N_STEER-1} | IVME={IVME:.2f} | SONUM={SONUM:.2f} | ZIRVE={ZIRVE:.2f} | TABAN={TABAN:.2f}")
T0=time.perf_counter()

print("\n[1/12] Loading frozen Qwen...")
_tv=tuple(int("".join(c for c in x if c.isdigit())or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;layers=model.model.layers
H=cfg.hidden_size;NH=cfg.num_attention_heads;NKV=cfg.num_key_value_heads
HD=getattr(cfg,"head_dim",None) or H//NH;KVD=NKV*HD;NL=len(layers)
if (NL,H,NH,NKV,HD)!=(28,3584,28,4,128):raise RuntimeError(f"Architecture mismatch: {(NL,H,NH,NKV,HD)}")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
_ge=model.generation_config.eos_token_id
EOS=sorted({tok.eos_token_id,*(_ge if isinstance(_ge,(list,tuple)) else [_ge])}-{None})
enc=lambda s:tok(s,add_special_tokens=False).input_ids
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | {NL}L H={H} QH={NH} KVH={NKV} BF16")

def dra_envelope(L):
    t=float(L);kb=ZIRVE*math.exp(-SONUM*t)*(1.0+SONUM*t)+TABAN
    return kb/(ZIRVE+TABAN)
ENVELOPE=[dra_envelope(L) for L in range(N_STEER)]
DOSES=[IVME*x for x in ENVELOPE]
print("      DRA envelope:",", ".join(f"L{i}:{100*d:.3f}%" for i,d in enumerate(DOSES)))

@torch.inference_mode()
def mean_hidden_states(texts):
    acc=[torch.zeros(H,dtype=torch.float32,device=DEVICE) for _ in range(N_STEER)]
    for text in texts:
        ids=torch.tensor([enc(text)],device=DEVICE)
        o=model(input_ids=ids,output_hidden_states=True,use_cache=False,return_dict=True)
        for L in range(N_STEER):acc[L].add_(o.hidden_states[L+1][0,-1].float())
    return [x/float(len(texts)) for x in acc]

print("\n[2/12] Extracting target-blind RELATIONAL-CONTINUATION ↔ SHORTCUT compass...")
POS=mean_hidden_states(POSITIVE);NEG=mean_hidden_states(NEGATIVE);ACT=[]
for L in range(N_STEER):
    d=POS[L]-NEG[L];n=torch.linalg.vector_norm(d).clamp_min(1e-12)
    ACT.append((d/n).float().contiguous())
    print(f"      L{L:02d} raw={n.item():.6f} unit={ACT[-1].norm().item():.6f} dose={100*DOSES[L]:.3f}%")
del POS,NEG;torch.cuda.empty_cache()

# Pure-PyTorch implementation of the frozen-norm direct SEASC dose.
# Same law as production CUDA core:
# norm0=||h||; delta=rho*norm0*A; h'=h+delta.
DRA_TELEMETRY=None
def install_dra(sign=+1.0):
    global DRA_TELEMETRY
    DRA_TELEMETRY=[{"calls":0,"requested":[],"realized":[]} for _ in range(N_STEER)]
    handles=[]
    for L in range(N_STEER):
        A=ACT[L];dose=DOSES[L]
        def make_hook(li,a,rho,sgn):
            def hook(module,args,output):
                if isinstance(output,tuple):h=output[0];rest=output[1:]
                else:h=output;rest=None
                old=h;norm=torch.linalg.vector_norm(old.float(),dim=-1,keepdim=True).clamp_min(1e-12)
                delta=(float(sgn*rho)*norm*a.view(1,1,-1)).to(old.dtype)
                new=old+delta
                ol=old[:,-1].float();nl=new[:,-1].float()
                realized=(torch.linalg.vector_norm(nl-ol,dim=-1)/torch.linalg.vector_norm(ol,dim=-1).clamp_min(1e-12)).mean().item()
                DRA_TELEMETRY[li]["calls"]+=1;DRA_TELEMETRY[li]["requested"].append(float(abs(rho)));DRA_TELEMETRY[li]["realized"].append(float(realized))
                return new if rest is None else (new,)+rest
            return hook
        handles.append(layers[L].register_forward_hook(make_hook(L,A,dose,sign)))
    return handles

def remove_hooks(handles):
    for h in handles:h.remove()

@torch.inference_mode()
def kv_from_ids(ids):
    o=model(input_ids=torch.tensor([ids],device=DEVICE),output_hidden_states=True,use_cache=False,return_dict=True)
    K=[];V=[]
    for L in range(NL):
        z=layers[L].input_layernorm(o.hidden_states[L][0]);a=layers[L].self_attn
        K.append(a.k_proj(z).contiguous());V.append(a.v_proj(z).contiguous())
    return K,V

def forge(s):return kv_from_ids([PAD]+enc(s+SEP))

print("\n[3/12] Building exact TEST482 neutral PCA codebook...")
CB=[];CO=[forge(s) for s in CORPUS]
for L in range(NL):
    e={}
    for j,n in enumerate(("K","V")):
        R=torch.cat([c[j][L][1:] for c in CO]).float().view(-1,NKV,HD);MU=[];BB=[]
        for h in range(NKV):
            X=R[:,h];mu=X.mean(0);_,S,Vh=torch.linalg.svd(X-mu,full_matrices=False)
            m=min(128,Vh.shape[0]);b=Vh[:m].T.contiguous()
            if m<128:b=F.pad(b,(0,128-m))
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

@torch.inference_mode()
def install_single(K,V):
    T=K[0].shape[0];pos=torch.arange(T,device=DEVICE)[None]
    cos,sin=model.model.rotary_emb(K[0][None],pos);KK=[];VV=[]
    for L in range(NL):
        k=K[L].reshape(1,T,NKV,HD).transpose(1,2);v=V[L].reshape(1,T,NKV,HD).transpose(1,2)
        KK.append(apply_rotary_pos_emb(k,k,cos,sin,unsqueeze_dim=1)[1].contiguous());VV.append(v.contiguous())
    return tuple(KK),tuple(VV),T,T

@torch.inference_mode()
def install_parallel(packets,order=None):
    if order is None:order=list(range(len(packets)))
    KK=[[] for _ in range(NL)];VV=[[] for _ in range(NL)];Tmax=0;total=0
    for ix in order:
        K,V=packets[ix];T=K[0].shape[0];Tmax=max(Tmax,T);total+=T
        pos=torch.arange(T,device=DEVICE)[None];cos,sin=model.model.rotary_emb(K[0][None],pos)
        for L in range(NL):
            k=K[L].reshape(1,T,NKV,HD).transpose(1,2);v=V[L].reshape(1,T,NKV,HD).transpose(1,2)
            KK[L].append(apply_rotary_pos_emb(k,k,cos,sin,unsqueeze_dim=1)[1].contiguous());VV[L].append(v.contiguous())
    return tuple(torch.cat(x,dim=2) for x in KK),tuple(torch.cat(x,dim=2) for x in VV),total,Tmax

def make_cache(kv):
    try:cache=DynamicCache(config=cfg)
    except TypeError:cache=DynamicCache()
    for L in range(NL):cache.update(kv[0][L].clone(),kv[1][L].clone(),L)
    return cache

@torch.inference_mode()
def read_kv(q,kv,max_new=MAX_NEW,dra=0):
    handles=[]
    if dra!=0:handles=install_dra(+1.0 if dra>0 else -1.0)
    try:
        qids=enc(FMT.format(q=q));Tm=kv[2];P=kv[3];nq=len(qids);cache=make_cache(kv)
        mask=torch.ones(1,Tm+nq,dtype=torch.long,device=DEVICE)
        pos=torch.arange(P,P+nq,device=DEVICE)[None];ids=torch.tensor([qids],device=DEVICE);out=[]
        for _ in range(max_new):
            o=model(input_ids=ids,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=True,return_dict=True)
            nxt=int(o.logits[0,-1].float().argmax())
            if nxt in EOS:break
            out.append(nxt);ids=torch.tensor([[nxt]],device=DEVICE);pos=pos[:,-1:]+1
            mask=torch.cat([mask,torch.ones(1,1,dtype=torch.long,device=DEVICE)],1)
        return tok.decode(out,skip_special_tokens=True).strip()
    finally:
        remove_hooks(handles)

def firstline(x):return next((z.strip() for z in str(x).splitlines() if z.strip()),"")
def canon(x):return firstline(x).strip().upper()
def target_ok(x,target=TARGET):return canon(x)==target.upper()

print("\n[4/12] Forging TEST496 BELLEKÖZ packets with DRA OFF...")
MAIN_PACK=[];DIST_PACK=[];SWAP_PACK=[]
for i,(name,s) in enumerate(MEMORIES):
    MAIN_PACK.append(packet(s));print(f"      {name} [{i+1}/6]")
for _,s in DISTRACTORS:DIST_PACK.append(packet(s))
for _,s in SWAPS:SWAP_PACK.append(packet(s))
print("      6 primary + 8 unrelated-axis + 2 causal-swap packets ready.")

print("\n[5/12] TEST496 mechanical regression — DRA OFF...")
REG=[]
for i in range(6):
    old=install_single(*MAIN_PACK[i]);new=install_parallel([MAIN_PACK[i]])
    ro=read_kv(QUESTION,old,dra=0);rn=read_kv(QUESTION,new,dra=0);eq=ro==rn;REG.append(eq)
    print(f"      {MEMORIES[i][0]} equal={int(eq)} | {firstline(rn)!r}")
REG_OK=all(REG)
FULL_KV=install_parallel(MAIN_PACK)
BASE_RAW=read_kv(QUESTION,FULL_KV,dra=0)
print("      TEST496 FULL / DRA OFF ->",repr(BASE_RAW))
print("      expected historical shortcut ORIN-23 =",int(canon(BASE_RAW)=="ORIN-23"))

print("\n[6/12] Directional A/B — DRA OFF / +RELATIONAL / -RELATIONAL...")
OFF_RAW=BASE_RAW
PLUS_RAW=read_kv(QUESTION,FULL_KV,dra=+1)
PLUS_TELEM=[dict(x) for x in DRA_TELEMETRY]
MINUS_RAW=read_kv(QUESTION,FULL_KV,dra=-1)
MINUS_TELEM=[dict(x) for x in DRA_TELEMETRY]
print(f"      OFF       -> {OFF_RAW!r} | target={int(target_ok(OFF_RAW))}")
print(f"      +DRA      -> {PLUS_RAW!r} | target={int(target_ok(PLUS_RAW))}")
print(f"      -DRA CTRL -> {MINUS_RAW!r} | target={int(target_ok(MINUS_RAW))}")

print("\n[7/12] Chain-depth X-Ray — same BELLEKÖZ, no architecture change...")
PROBES=[
("P1","What seal does the missing Meridian instrument carry? Give only the seal.","ORIN"),
("P2","Which routing class corresponds to the seal carried by the missing Meridian instrument? Give only the routing class.","TAV"),
("P3","Which archive section is assigned to the routing class corresponding to the seal carried by the missing Meridian instrument? Give only the section letter.","Q"),
("P4","Which transfer batch does the missing Meridian instrument belong to? Give only the batch.","SEL"),
("P5","Which active recovery bay is used by the transfer batch of the missing Meridian instrument? Give only the bay number.","23"),
("P6",QUESTION,"Q-23")
]
PROBE_RESULTS=[]
for name,q,exp in PROBES:
    off=read_kv(q,FULL_KV,dra=0);on=read_kv(q,FULL_KV,dra=+1)
    oo=target_ok(off,exp);nn=target_ok(on,exp);PROBE_RESULTS.append((name,exp,off,on,oo,nn))
    print(f"      {name} expected={exp:5s} | OFF={firstline(off)!r} [{int(oo)}] | +DRA={firstline(on)!r} [{int(nn)}]")

print("\n[8/12] Causal substitutions — decisive anti-cheating control...")
swap_section=MAIN_PACK.copy();swap_section[2]=SWAP_PACK[0]
swap_bay=MAIN_PACK.copy();swap_bay[4]=SWAP_PACK[1]
SEC_KV=install_parallel(swap_section);BAY_KV=install_parallel(swap_bay)
SEC_OFF=read_kv(QUESTION,SEC_KV,dra=0);SEC_ON=read_kv(QUESTION,SEC_KV,dra=+1)
BAY_OFF=read_kv(QUESTION,BAY_KV,dra=0);BAY_ON=read_kv(QUESTION,BAY_KV,dra=+1)
SEC_OK=target_ok(SEC_ON,"Z-23");BAY_OK=target_ok(BAY_ON,"Q-57")
print(f"      Q→Z | OFF={firstline(SEC_OFF)!r} | +DRA={firstline(SEC_ON)!r} | expected Z-23={int(SEC_OK)}")
print(f"      23→57 | OFF={firstline(BAY_OFF)!r} | +DRA={firstline(BAY_ON)!r} | expected Q-57={int(BAY_OK)}")

print("\n[9/12] Necessity under +DRA — all leave-one-out...")
LOO=[]
for miss in range(6):
    kv=install_parallel([MAIN_PACK[j] for j in range(6) if j!=miss])
    off=read_kv(QUESTION,kv,dra=0);on=read_kv(QUESTION,kv,dra=+1)
    LOO.append((miss,off,on,target_ok(on)))
    print(f"      -{MEMORIES[miss][0]} | OFF={firstline(off)!r} | +DRA={firstline(on)!r} | target-leak={int(target_ok(on))}")
LOO_LEAK=sum(x[3] for x in LOO)

print("\n[10/12] Order invariance under +DRA — 32 deterministic permutations...")
rng=random.Random(SEED);PERMS=[tuple(range(6))];seen={PERMS[0]}
while len(PERMS)<32:
    p=list(range(6));rng.shuffle(p);p=tuple(p)
    if p not in seen:seen.add(p);PERMS.append(p)
PERM_OK=0
for z,p in enumerate(PERMS):
    raw=read_kv(QUESTION,install_parallel(MAIN_PACK,order=p),dra=+1);ok=target_ok(raw);PERM_OK+=int(ok)
    print(f"      [{z+1:02d}/32] {'-'.join(MEMORIES[i][0] for i in p)} -> {firstline(raw)!r} | target={int(ok)}")

print("\n[11/12] Unrelated-axis contamination under +DRA...")
DIST_OK=0
for d in range(8):
    raw=read_kv(QUESTION,install_parallel(MAIN_PACK+[DIST_PACK[d]]),dra=+1);ok=target_ok(raw);DIST_OK+=int(ok)
    literal=" [LITERAL-TARGET-DISTRACTOR]" if d==7 else ""
    print(f"      +{DISTRACTORS[d][0]} -> {firstline(raw)!r} | target={int(ok)}{literal}")

def telem_summary(T):
    req=[v for x in T for v in x["requested"]];real=[v for x in T for v in x["realized"]]
    return (sum(req)/max(1,len(req)),sum(real)/max(1,len(real)),sum(x["calls"] for x in T))

PLUS_REQ,PLUS_REAL,PLUS_CALLS=telem_summary(PLUS_TELEM)
MINUS_REQ,MINUS_REAL,MINUS_CALLS=telem_summary(MINUS_TELEM)
CHAIN_OFF=sum(x[4] for x in PROBE_RESULTS);CHAIN_ON=sum(x[5] for x in PROBE_RESULTS)

# Strict first-pass criterion.
# We do NOT require D8 contamination to be clean because D8 literally contains Q-23;
# it is reported separately and cannot constitute evidence for relational composition.
DIST_NONLIT_OK=0
for d in range(7):
    raw=read_kv(QUESTION,install_parallel(MAIN_PACK+[DIST_PACK[d]]),dra=+1)
    DIST_NONLIT_OK+=int(target_ok(raw))

PHENOMENON=(
    REG_OK and
    canon(OFF_RAW)=="ORIN-23" and
    target_ok(PLUS_RAW) and
    SEC_OK and BAY_OK and
    LOO_LEAK==0 and
    PERM_OK>=30 and
    DIST_NONLIT_OK>=6
)

print("\n"+"="*160)
print("[12/12] TEST497 FINAL RESULT — MAM × DRA RELATIONAL-PATH PRESSURE")
print("="*160)
print("MODEL                         : Qwen/Qwen2.5-7B-Instruct · frozen")
print("BELLEKÖZ                      : TEST482 K120/V128/OWN · unchanged")
print("WORKSPACE                     : TEST495 parallel sequence concat · unchanged")
print("TEST496 TASK                  : unchanged")
print("STEERING                      : AkbasCore 3.0 SEASC/DRA frozen-norm direct dose")
print("STEERED LAYERS                : L0-L19")
print(f"IVME / SONUM / ZIRVE / TABAN  : {IVME:.2f} / {SONUM:.2f} / {ZIRVE:.2f} / {TABAN:.2f}")
print("COMPASS                       : target-blind RELATIONAL-CONTINUATION ↔ SHORTCUT")
print("TARGET DATA IN COMPASS        : NONE")
print("ANSWER VECTOR                 : NONE")
print("ADDRESS / VTOKEN              : OFF")
print("TOP-K / ROUTER / ANN / GRAPH  : NONE")
print("TRAINING / LoRA               : NONE")
print("DOSE SEARCH                   : NONE")
print("LAYER / HEAD SEARCH           : NONE")
print("POST-HOC TUNING               : NONE")
print("-"*160)
print(f"k=1 REGRESSION                : {sum(REG)}/6")
print("DRA OFF                       :",repr(OFF_RAW))
print("DRA +RELATIONAL               :",repr(PLUS_RAW))
print("DRA -RELATIONAL CONTROL       :",repr(MINUS_RAW))
print(f"OFF EXACT TARGET              : {int(target_ok(OFF_RAW))}")
print(f"+DRA EXACT TARGET             : {int(target_ok(PLUS_RAW))}")
print(f"-DRA EXACT TARGET             : {int(target_ok(MINUS_RAW))}")
print(f"CHAIN PROBES OFF              : {CHAIN_OFF}/6")
print(f"CHAIN PROBES +DRA             : {CHAIN_ON}/6")
print(f"CAUSAL SECTION Q→Z            : {'PASS' if SEC_OK else 'FAIL'} | {firstline(SEC_ON)!r}")
print(f"CAUSAL BAY 23→57              : {'PASS' if BAY_OK else 'FAIL'} | {firstline(BAY_ON)!r}")
print(f"+DRA LEAVE-ONE-OUT TARGET LEAK: {LOO_LEAK}/6")
print(f"+DRA ORDER INVARIANCE         : {PERM_OK}/32")
print(f"+DRA NONLITERAL DISTRACTORS   : {DIST_NONLIT_OK}/7")
print(f"+DRA ALL DISTRACTORS          : {DIST_OK}/8")
print("-"*160)
print(f"+DRA requested dose mean      : {100*PLUS_REQ:.4f}%")
print(f"+DRA realized dose mean       : {100*PLUS_REAL:.4f}%")
print(f"+DRA hook calls               : {PLUS_CALLS}")
print(f"-DRA requested dose mean      : {100*MINUS_REQ:.4f}%")
print(f"-DRA realized dose mean       : {100*MINUS_REAL:.4f}%")
print(f"-DRA hook calls               : {MINUS_CALLS}")
print("-"*160)
for name,exp,off,on,oo,nn in PROBE_RESULTS:
    print(f"{name} expected={exp:5s} | OFF={firstline(off)!r} [{int(oo)}] | +DRA={firstline(on)!r} [{int(nn)}]")
print("-"*160)
print("TEST482 BLOB SHA              :",TEST482_BLOB_SHA)
print("TEST482 LOCK SHA              :",TEST482_LOCK_SHA)
print("TEST495 LOCK SHA              :",TEST495_LOCK_SHA)
print("TEST496 LOCK SHA              :",TEST496_LOCK_SHA)
print("TEST497 LOCK SHA              :",LOCK_SHA)
print(f"TOTAL TEST TIME               : {time.perf_counter()-T0:.2f}s")
print("EXPLORATORY VERDICT           :","TARGET_BLIND_DRA_RESCUES_RELATIONAL_COMPOSITION" if PHENOMENON else "TARGET_BLIND_DRA_RESCUE_NOT_YET_OBSERVED")
print("="*160)

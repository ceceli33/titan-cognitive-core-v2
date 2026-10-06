# Copyright © 2026 Mustafa Akbaş
# New original AKBASCORE MAM material is source-available under the repository LICENSE.
# Earlier MIT-licensed Titan Cognitive Core / AkbasCore material remains under its original license.
# See the repository LICENSE for complete terms.
#
# TEST495 — AKBASCORE MAM · MULTI-BELLEKÖZ COMPOSITION / EMERGENT RECONSTRUCTION
# FROZEN TEST482 K120/V128/OWN MEMORY CORES → PARALLEL MEMORY WORKSPACE → SYNTHESIZED ANSWER
#
# PRIMARY QUESTION:
#   Can several independently compressed BELLEKÖZ memories jointly produce an answer
#   that is not explicitly present in any single source memory?
#
# FROZEN:
#   Qwen/Qwen2.5-7B-Instruct
#   TEST482 K120/V128/OWN compression
#   TEST482 neutral PCA codebook
#   model weights
#
# NEW MECHANISM UNDER TEST:
#   independently forged BELLEKÖZ K/V blocks
#      -> each block receives its own local 0..T-1 RoPE coordinates
#      -> blocks concatenated ONLY along sequence/cache dimension
#      -> no K/V averaging
#      -> no learned weights
#      -> no VTOKEN prior
#      -> one shared Qwen attention workspace
#
# IMPORTANT:
#   TEST495 deliberately disables retrieval/addressing.
#   Required memory neighborhood is ORACLE.
#   This isolates MULTI-MEMORY COMPOSITION from MEMORY DISCOVERY.
#
# TARGET:
#   Complete recovery location code = K-17
#
# CRITICAL PROPERTY:
#   Literal string "K-17" appears in NONE of the 8 memory sources.
#   K and 17 are distributed across separate memories.
#
# CONTROLS:
#   no-memory
#   8 single memories
#   all 254 proper non-empty subsets
#   8 leave-one-out conditions
#   full 8-memory composition
#   32 full-set order permutations
#   unrelated-memory contamination controls
#   deterministic swap controls
#   k=1 BELLEKÖZ regression
#
# NO:
#   address search
#   VTOKEN
#   channel/layer/head search
#   fusion fitting
#   learned router
#   ANN/graph
#   LoRA/training
#   K/V averaging
#   post-hoc memory selection

import os,sys,subprocess,importlib.util,random,re,time,hashlib,json,itertools,math
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None: subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

TEST="495";SEED=495;MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
K_DIM=120;V_DIM=128;MAX_NEW=24
FMT="QUESTION:\n{q}\n\n ANSWER:";SEP="\n\n"
DEVICE=torch.device("cuda")
TEST482_BLOB_SHA="ecc630144a44479cb35c432c50eb0d09bc6866d7"
TEST482_LOCK_SHA="fa59fd38661e558f6bae22eedff0999f32f8f6e9d4a08932383525323a5fe687"
TEST494_LOCK_SHA="e8928220100e5e2655d4a7a93e3702ccd08b121f3c1e9fbbff9589b86e2fd384"
TARGET="K-17"
QUESTION="What is the complete recovery location code for the missing Meridian instrument? Give only the complete code. If the available records are insufficient to determine the complete code, answer NONE."
READOUT=QUESTION

# Eight complementary memories. No source contains the literal final answer K-17.
MEMORIES=[
("M1","The missing Meridian instrument was transferred with the instruments assigned to the western archive route."),
("M2","Only mechanical instruments on the western archive route were moved into the protected recovery section."),
("M3","The Meridian instrument is mechanical rather than optical."),
("M4","For protected recovery records, the destination code is formed from the archive section letter followed by a hyphen and the active bay number."),
("M5","During this transfer, the western archive's protected recovery section was assigned the section letter K."),
("M6","The Meridian transfer occurred after the old recovery bay had been permanently closed."),
("M7","After the old bay closed, the protected recovery section used bay number seventeen for mechanical instruments."),
("M8","The transfer ledger states that the Meridian instrument remained in that protected recovery section and was not moved again.")
]
DISTRACTORS=[
("D1","The eastern archive uses section letter R for optical instruments."),
("D2","A temporary calibration bench was numbered twenty-four during the same month."),
("D3","The southern archive moved several glass instruments into section letter P."),
("D4","Bay eleven was reserved for damaged electrical equipment."),
("D5","The museum display inventory uses a completely separate numbering system."),
("D6","A blue transport crate was returned to the central workshop after inspection."),
("D7","The northern archive labels photographic equipment with section letter T."),
("D8","Bay thirty-two is used for chemical storage containers.")
]
SWAPS=[
("S5","During this transfer, the western archive's protected recovery section was assigned the section letter Q."),
("S7","After the old bay closed, the protected recovery section used bay number twenty-three for mechanical instruments.")
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

assert TARGET.casefold() not in " ".join(x[1] for x in MEMORIES).casefold()
assert len(MEMORIES)==8
LOCK={
"test":TEST,"model":MODEL_ID,"seed":SEED,"k_dim":K_DIM,"v_dim":V_DIM,
"memory_core":"TEST482 K120/V128/OWN","mechanism":"PARALLEL_SEQUENCE_CONCAT",
"target":TARGET,"question":QUESTION,"memories":MEMORIES,"distractors":DISTRACTORS,"swaps":SWAPS,
"corpus":CORPUS,"test482_blob_sha":TEST482_BLOB_SHA,"test482_lock_sha":TEST482_LOCK_SHA,
"test494_lock_sha":TEST494_LOCK_SHA,"retrieval":"ORACLE","address":"OFF","prior":"NONE",
"kv_average":"NONE","training":"NONE","posthoc_selection":"NONE"
}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required.")
torch.set_grad_enabled(False)
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*158)
print("TEST495 — AKBASCORE MAM · MULTI-BELLEKÖZ COMPOSITION / EMERGENT RECONSTRUCTION")
print("FROZEN TEST482 K120/V128/OWN → PARALLEL MEMORY WORKSPACE → DISTRIBUTED SYNTHESIS")
print("="*158)
print("LOCK SHA:",LOCK_SHA)
print("TEST482 BLOB:",TEST482_BLOB_SHA)
print("TEST494 LOCK:",TEST494_LOCK_SHA)
print("TARGET:",TARGET)
print("TARGET LITERAL PRESENT IN ANY MEMORY: NO")
print("\nMEMORY PANEL")
for k,s in MEMORIES: print(f"{k}: {s}")
print("\nQUESTION")
print(QUESTION)
T0=time.perf_counter()

print("\n[1/10] Loading frozen Qwen...")
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

print("[2/10] Building exact TEST482 neutral PCA codebook...")
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
    T=K[0].shape[0]
    cos,sin=model.model.rotary_emb(K[0][None],torch.arange(T,device=DEVICE)[None])
    KK=[];VV=[]
    for L in range(NL):
        k=K[L].reshape(1,T,NKV,HD).transpose(1,2)
        v=V[L].reshape(1,T,NKV,HD).transpose(1,2)
        KK.append(apply_rotary_pos_emb(k,k,cos,sin,unsqueeze_dim=1)[1].contiguous())
        VV.append(v.contiguous())
    return tuple(KK),tuple(VV),T

@torch.inference_mode()
def install_parallel(packets,order=None):
    if order is None:order=list(range(len(packets)))
    KK=[[] for _ in range(NL)];VV=[[] for _ in range(NL)]
    Tmax=0;total=0
    for ix in order:
        K,V=packets[ix];T=K[0].shape[0];Tmax=max(Tmax,T);total+=T
        pos=torch.arange(T,device=DEVICE)[None]
        cos,sin=model.model.rotary_emb(K[0][None],pos)
        for L in range(NL):
            k=K[L].reshape(1,T,NKV,HD).transpose(1,2)
            v=V[L].reshape(1,T,NKV,HD).transpose(1,2)
            kr=apply_rotary_pos_emb(k,k,cos,sin,unsqueeze_dim=1)[1].contiguous()
            KK[L].append(kr);VV[L].append(v.contiguous())
    return tuple(torch.cat(x,dim=2) for x in KK),tuple(torch.cat(x,dim=2) for x in VV),total,Tmax

def make_cache(kv):
    try:cache=DynamicCache(config=cfg)
    except TypeError:cache=DynamicCache()
    for L in range(NL):cache.update(kv[0][L].clone(),kv[1][L].clone(),L)
    return cache

@torch.inference_mode()
def read_kv(q,kv,max_new=MAX_NEW):
    qids=enc(FMT.format(q=q));Tm=kv[2];P=kv[3] if len(kv)>3 else Tm;nq=len(qids)
    cache=make_cache(kv)
    mask=torch.ones(1,Tm+nq,dtype=torch.long,device=DEVICE)
    pos=torch.arange(P,P+nq,device=DEVICE)[None]
    ids=torch.tensor([qids],device=DEVICE);out=[]
    for _ in range(max_new):
        o=model(input_ids=ids,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=True,return_dict=True)
        nxt=int(o.logits[0,-1].float().argmax())
        if nxt in EOS:break
        out.append(nxt);ids=torch.tensor([[nxt]],device=DEVICE);pos=pos[:,-1:]+1
        mask=torch.cat([mask,torch.ones(1,1,dtype=torch.long,device=DEVICE)],1)
    return tok.decode(out,skip_special_tokens=True).strip()

@torch.inference_mode()
def read_no_memory(q,max_new=MAX_NEW):
    ids=torch.tensor([enc(FMT.format(q=q))],device=DEVICE);out=[]
    for _ in range(max_new):
        o=model(input_ids=ids,use_cache=False,return_dict=True)
        nxt=int(o.logits[0,-1].float().argmax())
        if nxt in EOS:break
        out.append(nxt);ids=torch.cat([ids,torch.tensor([[nxt]],device=DEVICE)],1)
    return tok.decode(out,skip_special_tokens=True).strip()

def firstline(x):return next((z.strip() for z in str(x).splitlines() if z.strip()),"")
def norm(x):return re.sub(r"[^a-z0-9]+","",firstline(x).casefold())
def target_ok(x):return norm(x).startswith(norm(TARGET))
def none_ok(x):
    n=norm(x)
    return n=="none" or n.startswith("none")

@torch.inference_mode()
def answer_margin(kv):
    prompt=FMT.format(q=QUESTION)
    cands=[TARGET,"NONE"]
    vals=[]
    for ans in cands:
        pids=enc(prompt);aids=enc(ans)
        cache=make_cache(kv);Tm=kv[2];P=kv[3] if len(kv)>3 else Tm
        ids=torch.tensor([pids+aids],device=DEVICE)
        mask=torch.ones(1,Tm+len(pids)+len(aids),dtype=torch.long,device=DEVICE)
        pos=torch.arange(P,P+len(pids)+len(aids),device=DEVICE)[None]
        o=model(input_ids=ids,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=False,return_dict=True)
        logits=o.logits[0]
        start=len(pids)-1
        lp=F.log_softmax(logits[start:start+len(aids)].float(),dim=-1)
        score=sum(float(lp[t,aids[t]]) for t in range(len(aids)))/max(1,len(aids))
        vals.append(score)
    return vals[0]-vals[1],vals[0],vals[1]

print("[3/10] Forging 8 complementary BELLEKÖZ + controls...")
MAIN_PACK=[];DIST_PACK=[];SWAP_PACK=[]
for i,(name,s) in enumerate(MEMORIES):
    MAIN_PACK.append(packet(s));print(f"      {name} [{i+1}/8] {s}")
for name,s in DISTRACTORS:DIST_PACK.append(packet(s))
for name,s in SWAPS:SWAP_PACK.append(packet(s))
print("      8 primary + 8 distractor + 2 swap packets ready.")

print("\n[4/10] k=1 regression — TEST482 single-memory install vs parallel workspace...")
REG=[]
for i in range(8):
    old=install_single(*MAIN_PACK[i]);new=install_parallel([MAIN_PACK[i]])
    ro=read_kv(QUESTION,(old[0],old[1],old[2],old[2]))
    rn=read_kv(QUESTION,new)
    eq=ro==rn;REG.append(eq)
    print(f"      {MEMORIES[i][0]} equal={int(eq)} | old={ro!r} | new={rn!r}")
REG_OK=all(REG)

print("\n[5/10] Human-readable critical conditions...")
NO_RAW=read_no_memory(QUESTION)
print(f"      NO MEMORY -> {NO_RAW!r} | target={int(target_ok(NO_RAW))}")
SINGLE_RAW=[]
for i in range(8):
    kv=install_parallel([MAIN_PACK[i]])
    raw=read_kv(QUESTION,kv);SINGLE_RAW.append(raw)
    print(f"      {MEMORIES[i][0]} ONLY -> {raw!r} | target={int(target_ok(raw))}")
FULL_KV=install_parallel(MAIN_PACK)
FULL_RAW=read_kv(QUESTION,FULL_KV)
print(f"\n      ALL 8 -> {FULL_RAW!r} | target={int(target_ok(FULL_RAW))}")
LOO_RAW=[]
for miss in range(8):
    subset=[MAIN_PACK[j] for j in range(8) if j!=miss]
    raw=read_kv(QUESTION,install_parallel(subset));LOO_RAW.append(raw)
    print(f"      ALL EXCEPT {MEMORIES[miss][0]} -> {raw!r} | target={int(target_ok(raw))}")

print("\n[6/10] Exhaustive subset lattice — all 255 non-empty subsets...")
SUBSET_ROWS=[]
for r in range(1,9):
    for comb in itertools.combinations(range(8),r):
        packs=[MAIN_PACK[i] for i in comb];kv=install_parallel(packs)
        margin,lt,ln=answer_margin(kv)
        SUBSET_ROWS.append((comb,r,margin,lt,ln))
        if r in (1,7,8):
            names="+".join(MEMORIES[i][0] for i in comb)
            print(f"      {names:27s} n={r} margin(TARGET-NONE)={margin:+.5f}")
FULL_MARGIN=next(x[2] for x in SUBSET_ROWS if x[1]==8)
BEST_PROPER=max((x for x in SUBSET_ROWS if x[1]<8),key=lambda x:x[2])
COMP_INDEX=FULL_MARGIN-BEST_PROPER[2]
print(f"      FULL margin       : {FULL_MARGIN:+.6f}")
print(f"      BEST proper subset: {'+'.join(MEMORIES[i][0] for i in BEST_PROPER[0])} | {BEST_PROPER[2]:+.6f}")
print(f"      COMPOSITION INDEX : {COMP_INDEX:+.6f}")

print("\n[7/10] Full-set permutation stability — 32 deterministic orders...")
rng=random.Random(SEED)
PERMS=[tuple(range(8))]
seen={PERMS[0]}
while len(PERMS)<32:
    p=list(range(8));rng.shuffle(p);p=tuple(p)
    if p not in seen:seen.add(p);PERMS.append(p)
PERM_OK=0;PERM_RAW=[]
for z,p in enumerate(PERMS):
    raw=read_kv(QUESTION,install_parallel(MAIN_PACK,order=p));ok=target_ok(raw);PERM_OK+=int(ok);PERM_RAW.append(raw)
    print(f"      [{z+1:02d}/32] {'-'.join(MEMORIES[i][0] for i in p)} -> {firstline(raw)!r} | target={int(ok)}")

print("\n[8/10] Distractor and causal-swap controls...")
DIST_OK=0;DIST_RAW=[]
for d in range(8):
    packs=MAIN_PACK+[DIST_PACK[d]]
    raw=read_kv(QUESTION,install_parallel(packs));ok=target_ok(raw);DIST_OK+=int(ok);DIST_RAW.append(raw)
    print(f"      +{DISTRACTORS[d][0]} -> {raw!r} | target={int(ok)}")
# Two causal interventions: replace section-letter fact and bay-number fact.
swap_letter=MAIN_PACK.copy();swap_letter[4]=SWAP_PACK[0]
swap_bay=MAIN_PACK.copy();swap_bay[6]=SWAP_PACK[1]
SWAP_LETTER_RAW=read_kv(QUESTION,install_parallel(swap_letter))
SWAP_BAY_RAW=read_kv(QUESTION,install_parallel(swap_bay))
print(f"      M5 K→Q SWAP -> {SWAP_LETTER_RAW!r} | original-target={int(target_ok(SWAP_LETTER_RAW))}")
print(f"      M7 17→23 SWAP -> {SWAP_BAY_RAW!r} | original-target={int(target_ok(SWAP_BAY_RAW))}")

print("\n[9/10] Contribution anatomy...")
FULL_LP=FULL_MARGIN
DELTA=[]
for i in range(8):
    comb=tuple(j for j in range(8) if j!=i)
    row=next(x for x in SUBSET_ROWS if x[0]==comb)
    d=FULL_LP-row[2];DELTA.append(d)
    print(f"      remove {MEMORIES[i][0]} | full={FULL_LP:+.6f} minus-one={row[2]:+.6f} Δ={d:+.6f} | greedy={LOO_RAW[i]!r}")
POS_DELTA=sum(d>0 for d in DELTA)
SINGLE_TARGET=sum(target_ok(x) for x in SINGLE_RAW)
LOO_TARGET=sum(target_ok(x) for x in LOO_RAW)

# Conservative exploratory verdict:
# This is NOT a final scientific seal. It only marks whether the intended phenomenon
# appeared in this first frozen mechanism test.
PHENOMENON=(
    REG_OK and
    not target_ok(NO_RAW) and
    SINGLE_TARGET==0 and
    target_ok(FULL_RAW) and
    COMP_INDEX>0 and
    POS_DELTA>=6 and
    PERM_OK>=24 and
    DIST_OK>=6
)

print("\n"+"="*158)
print("[10/10] TEST495 FINAL RESULT — MULTI-BELLEKÖZ COMPOSITION")
print("="*158)
print("VALIDATION TYPE             : FIRST MECHANISM / ORACLE MEMORY NEIGHBORHOOD")
print("MODEL                       : Qwen/Qwen2.5-7B-Instruct · frozen")
print("BELLEKÖZ                    : TEST482 K120/V128/OWN · unchanged")
print("MEMORY COUNT                : 8 complementary compressed memories")
print("TARGET                      :",TARGET)
print("TARGET LITERAL IN SOURCES   : NO")
print("ADDRESS / VTOKEN            : OFF")
print("MEMORY PRIOR                : NONE")
print("K/V AVERAGING               : NONE")
print("MEMORY INTERACTION          : PARALLEL SEQUENCE-DIMENSION CACHE CONCAT")
print("LOCAL MEMORY POSITIONS      : EACH BELLEKÖZ 0..T-1")
print("READER POSITION START       : MAX LOCAL BELLEKÖZ LENGTH")
print("-"*158)
print(f"k=1 REGRESSION              : {sum(REG)}/8 exact greedy-output equality")
print(f"NO-MEMORY TARGET            : {int(target_ok(NO_RAW))}")
print(f"SINGLE-MEMORY TARGET        : {SINGLE_TARGET}/8")
print(f"FULL 8-MEMORY TARGET        : {int(target_ok(FULL_RAW))}")
print(f"LEAVE-ONE-OUT TARGET        : {LOO_TARGET}/8")
print(f"PERMUTATION TARGET          : {PERM_OK}/32")
print(f"DISTRACTOR TARGET           : {DIST_OK}/8")
print(f"POSITIVE REMOVAL Δ          : {POS_DELTA}/8")
print(f"FULL TARGET-NONE MARGIN     : {FULL_MARGIN:+.6f}")
print(f"BEST PROPER-SUBSET MARGIN   : {BEST_PROPER[2]:+.6f}")
print(f"COMPOSITION INDEX           : {COMP_INDEX:+.6f}")
print("-"*158)
print("NO MEMORY RAW               :",repr(NO_RAW))
print("FULL COMPOSITION RAW        :",repr(FULL_RAW))
print("LETTER-SWAP RAW             :",repr(SWAP_LETTER_RAW))
print("BAY-SWAP RAW                :",repr(SWAP_BAY_RAW))
print("-"*158)
print("TEST482 BLOB SHA            :",TEST482_BLOB_SHA)
print("TEST482 LOCK SHA            :",TEST482_LOCK_SHA)
print("TEST494 LOCK SHA            :",TEST494_LOCK_SHA)
print("TEST495 LOCK SHA            :",LOCK_SHA)
print("-"*158)
print("RETRIEVAL                    : ORACLE / NOT TESTED")
print("CHANNEL SEARCH               : NONE")
print("LAYER SEARCH                 : NONE")
print("HEAD SEARCH                  : NONE")
print("VTOKEN                       : OFF")
print("FUSION WEIGHT FIT            : NONE")
print("ROUTER / ANN / GRAPH         : NONE")
print("TRAINING / LoRA              : NONE")
print("BELLEKÖZ COMPRESSION CHANGE  : NONE")
print("POST-HOC MEMORY SELECTION    : NONE")
print(f"TOTAL TEST TIME              : {time.perf_counter()-T0:.2f}s")
print("EXPLORATORY VERDICT          :","MULTI_MEMORY_COMPOSITION_OBSERVED" if PHENOMENON else "MULTI_MEMORY_COMPOSITION_NOT_YET_OBSERVED")
print("="*158)

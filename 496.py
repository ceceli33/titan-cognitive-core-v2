# Copyright © 2026 Mustafa Akbaş
# New original AKBASCORE MAM material is source-available under the repository LICENSE.
# Earlier MIT-licensed Titan Cognitive Core / AkbasCore material remains under its original license.
# See repository LICENSE for complete terms.
#
# TEST496 — AKBASCORE MAM · NECESSARY MULTI-BELLEKÖZ COMPOSITION
# FROZEN TEST482 K120/V128/OWN → TEST495 PARALLEL WORKSPACE → NECESSARY DISTRIBUTED SYNTHESIS
#
# PURPOSE:
#   Test whether a frozen Qwen can synthesize ONE answer from multiple independently
#   compressed BELLEKÖZ memories when every memory is logically necessary.
#
# IMPORTANT:
#   This is NOT retrieval.
#   No VTOKEN / Top-K / router / ANN / graph / filename / record-ID selection.
#   The required memory neighborhood is oracle-loaded only to isolate composition.
#
# FROZEN:
#   Qwen/Qwen2.5-7B-Instruct
#   TEST482 K120/V128/OWN compression
#   TEST482 neutral PCA codebook
#   TEST495 parallel sequence-dimension cache composition
#
# NEW:
#   Harder logically necessary 6-memory task
#   ambiguous alternatives inside each memory
#   exact answer parser (Q23 != Q-23)
#   exhaustive 63-subset lattice
#   all 6 leave-one-out conditions
#   64 full-memory order permutations
#   unrelated-axis contamination
#   causal substitutions
#
# REQUIRED LOGIC:
#   Meridian -> ORIN -> TAV -> Q
#   Meridian -> SEL -> 23
#   code = section-bay
#   therefore Q-23
#
# No individual memory contains "Q-23".
# No individual memory is sufficient.
# Removing any logical link makes the complete answer underdetermined.

import os,sys,subprocess,importlib.util,random,re,time,hashlib,json,itertools
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

TEST="496";SEED=496;MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
K_DIM=120;V_DIM=128;MAX_NEW=20;DEVICE=torch.device("cuda")
FMT="QUESTION:\n{q}\n\n ANSWER:";SEP="\n\n";TARGET="Q-23"
QUESTION="According to the available records, what is the complete current recovery code of the missing Meridian instrument? Give only the exact code. If the complete code cannot be logically determined from the available records, answer NONE."
TEST482_BLOB_SHA="ecc630144a44479cb35c432c50eb0d09bc6866d7"
TEST482_LOCK_SHA="fa59fd38661e558f6bae22eedff0999f32f8f6e9d4a08932383525323a5fe687"
TEST495_LOCK_SHA="2bae3f2497fefb50b3d18d8fb7035c63a08fdd67d80077dd92f4e9c56eca787e"

# Every record contains competing alternatives so an isolated symbol is not enough.
MEMORIES=[
("M1","The missing Meridian instrument carries seal ORIN. The Atlas instrument carries seal NERA. The Helix instrument carries seal PAVO."),
("M2","In the recovery ledger, seal ORIN corresponds to routing class TAV; seal NERA corresponds to class BEX; seal PAVO corresponds to class LUM."),
("M3","Routing class TAV is assigned archive section Q; class BEX is assigned section R; class LUM is assigned section V."),
("M4","The missing Meridian instrument belongs to transfer batch SEL. The Atlas instrument belongs to batch DOR. The Helix instrument belongs to batch NIM."),
("M5","Transfer batch SEL uses active recovery bay 23; batch DOR uses bay 41; batch NIM uses bay 68."),
("M6","A complete recovery code is constructed from the instrument's archive section letter, then a hyphen, then its active recovery bay number.")
]

# Entirely unrelated second axis. It deliberately contains letters, numbers and mappings.
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

# Causal substitutions. Correct response must change accordingly.
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

assert len(MEMORIES)==6
assert TARGET.casefold() not in " ".join(x[1] for x in MEMORIES).casefold()
LOCK={"test":TEST,"model":MODEL_ID,"seed":SEED,"k_dim":K_DIM,"v_dim":V_DIM,
"memory_core":"TEST482 K120/V128/OWN","workspace":"TEST495 PARALLEL_SEQUENCE_CONCAT",
"target":TARGET,"question":QUESTION,"memories":MEMORIES,"distractors":DISTRACTORS,"swaps":SWAPS,
"corpus":CORPUS,"test482_blob_sha":TEST482_BLOB_SHA,"test482_lock_sha":TEST482_LOCK_SHA,
"test495_lock_sha":TEST495_LOCK_SHA,"retrieval":"ORACLE","address":"OFF","prior":"NONE",
"router":"NONE","ann":"NONE","graph":"NONE","training":"NONE","posthoc_selection":"NONE"}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
torch.set_grad_enabled(False)
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*158)
print("TEST496 — AKBASCORE MAM · NECESSARY MULTI-BELLEKÖZ COMPOSITION")
print("FROZEN TEST482 K120/V128/OWN → TEST495 PARALLEL WORKSPACE → NECESSARY DISTRIBUTED SYNTHESIS")
print("="*158)
print("LOCK SHA:",LOCK_SHA)
print("TEST482 BLOB:",TEST482_BLOB_SHA)
print("TEST495 LOCK:",TEST495_LOCK_SHA)
print("TARGET:",TARGET)
print("TARGET LITERAL PRESENT IN PRIMARY MEMORIES: NO")
print("\nPRIMARY MEMORY PANEL")
for k,s in MEMORIES:print(f"{k}: {s}")
print("\nUNRELATED SECOND AXIS")
for k,s in DISTRACTORS:print(f"{k}: {s}")
print("\nQUESTION")
print(QUESTION)
T0=time.perf_counter()

print("\n[1/11] Loading frozen Qwen...")
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
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | {NL}L | H={H} | QH={NH} KVH={NKV} | frozen BF16")

@torch.inference_mode()
def kv_from_ids(ids):
    o=model(input_ids=torch.tensor([ids],device=DEVICE),output_hidden_states=True,use_cache=False,return_dict=True)
    K=[];V=[]
    for L in range(NL):
        z=layers[L].input_layernorm(o.hidden_states[L][0]);a=layers[L].self_attn
        K.append(a.k_proj(z).contiguous());V.append(a.v_proj(z).contiguous())
    return K,V

def forge(s):return kv_from_ids([PAD]+enc(s+SEP))

print("[2/11] Building exact TEST482 neutral PCA codebook...")
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
def read_kv(q,kv,max_new=MAX_NEW):
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
def canon(x):return firstline(x).strip().upper()
def target_ok(x,target=TARGET):return canon(x)==target.upper()
def none_ok(x):return canon(x)=="NONE"

@torch.inference_mode()
def candidate_score(kv,answer):
    pids=enc(FMT.format(q=QUESTION));aids=enc(answer);cache=make_cache(kv)
    Tm,P=kv[2],kv[3];ids=torch.tensor([pids+aids],device=DEVICE)
    mask=torch.ones(1,Tm+len(pids)+len(aids),dtype=torch.long,device=DEVICE)
    pos=torch.arange(P,P+len(pids)+len(aids),device=DEVICE)[None]
    o=model(input_ids=ids,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=False,return_dict=True)
    logits=o.logits[0];start=len(pids)-1;lp=F.log_softmax(logits[start:start+len(aids)].float(),dim=-1)
    return sum(float(lp[t,aids[t]]) for t in range(len(aids)))/max(1,len(aids))

def diagnostic_margin(kv):
    # Diagnostic only; NOT used to choose memories or alter the test.
    answers=[TARGET,"R-41","V-68","Q-41","Q-68","R-23","V-23","NONE"]
    scores={a:candidate_score(kv,a) for a in answers}
    best_wrong=max(v for k,v in scores.items() if k!=TARGET)
    return scores[TARGET]-best_wrong,scores

print("[3/11] Forging primary BELLEKÖZ + unrelated axis + causal controls...")
MAIN_PACK=[];DIST_PACK=[];SWAP_PACK=[]
for i,(name,s) in enumerate(MEMORIES):
    MAIN_PACK.append(packet(s));print(f"      {name} [{i+1}/6] {s}")
for name,s in DISTRACTORS:DIST_PACK.append(packet(s))
for name,s in SWAPS:SWAP_PACK.append(packet(s))
print("      6 primary + 8 unrelated-axis + 2 causal-swap BELLEKÖZ ready.")

print("\n[4/11] k=1 regression — TEST482 install vs TEST495 workspace...")
REG=[]
for i in range(6):
    old=install_single(*MAIN_PACK[i]);new=install_parallel([MAIN_PACK[i]])
    ro=read_kv(QUESTION,old);rn=read_kv(QUESTION,new);eq=ro==rn;REG.append(eq)
    print(f"      {MEMORIES[i][0]} equal={int(eq)} | old={ro!r} | new={rn!r}")
REG_OK=all(REG)

print("\n[5/11] Critical human-readable conditions...")
NO_RAW=read_no_memory(QUESTION)
print(f"      NO MEMORY -> {NO_RAW!r} | exact-target={int(target_ok(NO_RAW))}")
SINGLE_RAW=[]
for i in range(6):
    raw=read_kv(QUESTION,install_parallel([MAIN_PACK[i]]));SINGLE_RAW.append(raw)
    print(f"      {MEMORIES[i][0]} ONLY -> {raw!r} | exact-target={int(target_ok(raw))}")
FULL_KV=install_parallel(MAIN_PACK);FULL_RAW=read_kv(QUESTION,FULL_KV)
print(f"\n      ALL 6 -> {FULL_RAW!r} | exact-target={int(target_ok(FULL_RAW))}")
LOO_RAW=[]
for miss in range(6):
    raw=read_kv(QUESTION,install_parallel([MAIN_PACK[j] for j in range(6) if j!=miss]));LOO_RAW.append(raw)
    print(f"      ALL EXCEPT {MEMORIES[miss][0]} -> {raw!r} | exact-target={int(target_ok(raw))}")

print("\n[6/11] Exhaustive subset lattice — all 63 non-empty subsets...")
SUBSET=[]
for r in range(1,7):
    for comb in itertools.combinations(range(6),r):
        kv=install_parallel([MAIN_PACK[i] for i in comb]);margin,scores=diagnostic_margin(kv)
        SUBSET.append((comb,r,margin,scores))
        names="+".join(MEMORIES[i][0] for i in comb)
        print(f"      {names:20s} n={r} target-v-bestfoil={margin:+.5f}")
FULL_ROW=next(x for x in SUBSET if x[1]==6);FULL_MARGIN=FULL_ROW[2]
BEST_PROPER=max((x for x in SUBSET if x[1]<6),key=lambda x:x[2])
COMP_INDEX=FULL_MARGIN-BEST_PROPER[2]
print(f"      FULL margin       : {FULL_MARGIN:+.6f}")
print(f"      BEST proper subset: {'+'.join(MEMORIES[i][0] for i in BEST_PROPER[0])} | {BEST_PROPER[2]:+.6f}")
print(f"      COMPOSITION INDEX : {COMP_INDEX:+.6f}")

print("\n[7/11] All leave-one-out contribution diagnostics...")
DELTA=[]
for i in range(6):
    comb=tuple(j for j in range(6) if j!=i);row=next(x for x in SUBSET if x[0]==comb)
    d=FULL_MARGIN-row[2];DELTA.append(d)
    print(f"      remove {MEMORIES[i][0]} | full={FULL_MARGIN:+.6f} minus-one={row[2]:+.6f} Δ={d:+.6f} | raw={LOO_RAW[i]!r}")

print("\n[8/11] Full-set order invariance — 64 deterministic permutations...")
rng=random.Random(SEED);PERMS=[tuple(range(6))];seen={PERMS[0]}
while len(PERMS)<64:
    p=list(range(6));rng.shuffle(p);p=tuple(p)
    if p not in seen:seen.add(p);PERMS.append(p)
PERM_OK=0
for z,p in enumerate(PERMS):
    raw=read_kv(QUESTION,install_parallel(MAIN_PACK,order=p));ok=target_ok(raw);PERM_OK+=int(ok)
    print(f"      [{z+1:02d}/64] {'-'.join(MEMORIES[i][0] for i in p)} -> {firstline(raw)!r} | exact={int(ok)}")

print("\n[9/11] Unrelated-axis contamination...")
DIST_OK=0
for d in range(8):
    # All six relevant memories plus one unrelated-axis memory.
    raw=read_kv(QUESTION,install_parallel(MAIN_PACK+[DIST_PACK[d]]));ok=target_ok(raw);DIST_OK+=int(ok)
    print(f"      +{DISTRACTORS[d][0]} -> {raw!r} | exact-target={int(ok)}")

print("\n[10/11] Causal substitutions...")
swap_section=MAIN_PACK.copy();swap_section[2]=SWAP_PACK[0]
swap_bay=MAIN_PACK.copy();swap_bay[4]=SWAP_PACK[1]
SECTION_RAW=read_kv(QUESTION,install_parallel(swap_section))
BAY_RAW=read_kv(QUESTION,install_parallel(swap_bay))
SECTION_OK=target_ok(SECTION_RAW,"Z-23")
BAY_OK=target_ok(BAY_RAW,"Q-57")
print(f"      M3 Q→Z intervention -> {SECTION_RAW!r} | expected Z-23={int(SECTION_OK)}")
print(f"      M5 23→57 intervention -> {BAY_RAW!r} | expected Q-57={int(BAY_OK)}")

SINGLE_TARGET=sum(target_ok(x) for x in SINGLE_RAW)
LOO_TARGET=sum(target_ok(x) for x in LOO_RAW)
POS_DELTA=sum(d>0 for d in DELTA)

# Strict exploratory criterion:
# 1) mechanism regression intact
# 2) no-memory cannot produce target
# 3) no single memory can produce target
# 4) all six together produce exact target
# 5) removing ANY one primary memory destroys exact target
# 6) full set beats every proper subset in diagnostic target-v-foil margin
# 7) all six removals reduce that margin
# 8) >=60/64 order permutations preserve answer
# 9) >=7/8 unrelated-axis additions preserve answer
# 10) both causal substitutions change the corresponding output component correctly
PHENOMENON=(
    REG_OK and
    not target_ok(NO_RAW) and
    SINGLE_TARGET==0 and
    target_ok(FULL_RAW) and
    LOO_TARGET==0 and
    COMP_INDEX>0 and
    POS_DELTA==6 and
    PERM_OK>=60 and
    DIST_OK>=7 and
    SECTION_OK and BAY_OK
)

print("\n"+"="*158)
print("[11/11] TEST496 FINAL RESULT — NECESSARY MULTI-BELLEKÖZ COMPOSITION")
print("="*158)
print("VALIDATION TYPE             : FROZEN MECHANISM / NECESSARY-COMPOSITION TEST")
print("MODEL                       : Qwen/Qwen2.5-7B-Instruct · frozen")
print("BELLEKÖZ                    : TEST482 K120/V128/OWN · unchanged")
print("WORKSPACE                   : TEST495 parallel sequence-dimension concat · unchanged")
print("PRIMARY MEMORY COUNT        : 6")
print("UNRELATED AXIS COUNT        : 8 controls")
print("TARGET                      :",TARGET)
print("TARGET LITERAL IN PRIMARY   : NO")
print("ADDRESS / VTOKEN            : OFF")
print("TOP-K                       : NONE")
print("ROUTER                      : NONE")
print("ANN / GRAPH                 : NONE")
print("MEMORY PRIOR                : NONE")
print("K/V AVERAGING               : NONE")
print("ORACLE NEIGHBORHOOD         : YES · composition isolation only")
print("-"*158)
print(f"k=1 REGRESSION              : {sum(REG)}/6")
print(f"NO-MEMORY EXACT TARGET      : {int(target_ok(NO_RAW))}")
print(f"SINGLE EXACT TARGET         : {SINGLE_TARGET}/6")
print(f"FULL 6 EXACT TARGET         : {int(target_ok(FULL_RAW))}")
print(f"LEAVE-ONE-OUT EXACT TARGET  : {LOO_TARGET}/6")
print(f"POSITIVE REMOVAL Δ          : {POS_DELTA}/6")
print(f"PERMUTATION EXACT TARGET    : {PERM_OK}/64")
print(f"UNRELATED-AXIS EXACT TARGET : {DIST_OK}/8")
print(f"SECTION CAUSAL SWAP         : {'PASS' if SECTION_OK else 'FAIL'}")
print(f"BAY CAUSAL SWAP             : {'PASS' if BAY_OK else 'FAIL'}")
print(f"FULL TARGET-v-FOIL MARGIN   : {FULL_MARGIN:+.6f}")
print(f"BEST PROPER-SUBSET MARGIN   : {BEST_PROPER[2]:+.6f}")
print(f"COMPOSITION INDEX           : {COMP_INDEX:+.6f}")
print("-"*158)
print("NO MEMORY RAW               :",repr(NO_RAW))
print("FULL COMPOSITION RAW        :",repr(FULL_RAW))
for i,x in enumerate(LOO_RAW):print(f"WITHOUT {MEMORIES[i][0]:2s} RAW             :",repr(x))
print("SECTION-SWAP RAW            :",repr(SECTION_RAW))
print("BAY-SWAP RAW                :",repr(BAY_RAW))
print("-"*158)
print("TEST482 BLOB SHA            :",TEST482_BLOB_SHA)
print("TEST482 LOCK SHA            :",TEST482_LOCK_SHA)
print("TEST495 LOCK SHA            :",TEST495_LOCK_SHA)
print("TEST496 LOCK SHA            :",LOCK_SHA)
print("-"*158)
print("HUMAN FILE SYSTEM           : NONE")
print("RETRIEVAL                    : NOT TESTED / ORACLE ISOLATION")
print("CHANNEL SEARCH               : NONE")
print("LAYER SEARCH                 : NONE")
print("HEAD SEARCH                  : NONE")
print("VTOKEN                       : OFF")
print("FUSION WEIGHT FIT            : NONE")
print("TRAINING / LoRA              : NONE")
print("BELLEKÖZ COMPRESSION CHANGE  : NONE")
print("POST-HOC MEMORY SELECTION    : NONE")
print(f"TOTAL TEST TIME              : {time.perf_counter()-T0:.2f}s")
print("EXPLORATORY VERDICT          :","NECESSARY_MULTI_MEMORY_COMPOSITION_OBSERVED" if PHENOMENON else "NECESSARY_MULTI_MEMORY_COMPOSITION_NOT_YET_OBSERVED")
print("="*158)

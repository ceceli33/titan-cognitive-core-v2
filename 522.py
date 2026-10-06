# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST522
# ÇAĞRIİZ HEAD×LAYER X-RAY
#
# PARENT TEST521:
# frozen L02-V uncentered address
# PRIMARY R1=.4375
# UNIFORM=.0312
# SHIFTED=.0156
# NO-A=.0156
# CF=.4531
#
# QUESTION:
# Is the causal ÇAĞRIİZ signal concentrated in particular native
# attention layer/head channels, or is it a distributed 392-channel effect?
#
# ADDRESS PATH IS UNCHANGED:
# query -> native Q·K attention over A BELLEKÖZ
#       -> weighted A L02-V tensor
#       -> max cosine against B L02-V positions
#
# X-RAY ONLY CHANGES HOW POINTER CHANNELS ARE OBSERVED/AGGREGATED.
#
# PANEL:
# fresh 64 items, disjoint construction seed/code region
# DISC 0..31
# INTERNAL-EVAL 32..63
#
# 392 channels:
# PTR layers L14..L27 × 28 Q heads
#
# DISC:
# - evaluate each individual channel's retrieval
# - measure gold-span attention enrichment
# - rank channels ONLY by DISC retrieval MRR, then R1
# - construct fixed top-K ensembles K={1,2,4,8,16,32,64,128}
#
# INTERNAL-EVAL:
# - evaluate TEST521 all-392 baseline
# - evaluate each DISC-frozen top-K ensemble
# - NO re-ranking/re-selection from INTERNAL-EVAL
#
# OUTPUT:
# one TEST523 candidate K chosen ONLY from DISC.
#
# TEST522 IS MECHANISM DISCOVERY.
# TEST523 remains the external held-out confirmation.
#
# NO token-ID address / decoded key / LM-head address / gold address
# NO training / DRA / learned router / query-time B forward.

import os,sys,subprocess,importlib.util,random,time,hashlib,json,gc,re,math
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

TEST="522";SEED=522;MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N=64;NDISC=32;DEVICE=torch.device("cuda")
FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n";PTR_LAYERS=list(range(14,28));BL=2;BK="V"
TOPKS=[1,2,4,8,16,32,64,128]
TEST521="e1eeaa869cdb15532fb99419cf53fc7f1ba922807a925357934ce0fade7784de"
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*176)
print("TEST522 — AKBASCORE MAM · ÇAĞRIİZ HEAD×LAYER X-RAY")
print("392 NATIVE ATTENTION CHANNELS · FROZEN L02-V UNCENTERED ADDRESS · DISC32 / INTERNAL-EVAL32")
print("="*176)
T0=time.perf_counter()

print("\n[1/11] Loading frozen Qwen...")
_tv=tuple(int("".join(c for c in x if c.isdigit())or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
if not tok.is_fast:raise RuntimeError("Fast tokenizer required.")
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
enc=lambda s:tok(s,add_special_tokens=False).input_ids
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=H//QH;REP=QH//KVH
if (NL,H,QH,KVH,HD)!=(28,3584,28,4,128):raise RuntimeError("Architecture mismatch.")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id;VOC=model.model.embed_tokens.weight.shape[0]
CHANNELS=[(L,h) for L in PTR_LAYERS for h in range(QH)];NC=len(CHANNELS)
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | {NL}L H={H} QH={QH} KVH={KVH} HD={HD}")
print(f"      ADDRESS LOCK : L{BL:02d}-{BK} · UNCENTERED")
print(f"      X-RAY        : {len(PTR_LAYERS)} layers × {QH} heads = {NC} channels")

SYL1=["Aev","Bex","Cyr","Dov","Ery","Fyn","Gex","Hov","Ivr","Jex","Kyr","Luv","Mox","Nyr","Ovx","Pyr","Qex","Ruv","Sox","Tyr","Uvx","Vyr","Wex","Xyr","Yuv","Zex"]
SYL2=["adan","bek","cor","dil","eron","fal","gin","hor","isk","jun","kel","lom","mun","nar","os","pir"]
def build_names(n):
    out=[]
    for a in SYL1:
        for b in SYL2:
            x=a+b
            if x not in out:out.append(x)
            if len(out)>=n:return out
    raise RuntimeError("Name pool too small.")

def native_codes(n,exclude=set(),skip=0):
    out=[];seen=0
    for tid in range(VOC):
        s=tok.decode([tid],skip_special_tokens=False)
        if s in exclude:continue
        if not re.fullmatch(r"[A-Z]{3,8}",s):continue
        if enc(s)!=[tid]:continue
        if seen<skip:seen+=1;continue
        if s in out:continue
        out.append(s)
        if len(out)>=n:return out
    raise RuntimeError(f"Need {n} codes, found {len(out)}.")

NAMES=build_names(N*3);SEALS=native_codes(N*3,skip=1400);CLASSES=native_codes(N*3,exclude=set(SEALS),skip=1800)

def make_items():
    items=[]
    for i in range(N):
        ents=NAMES[3*i:3*i+3];aseals=SEALS[3*i:3*i+3];gold=aseals[0]
        j1=(i+13)%N;j2=(i+31)%N
        bseals=[gold,SEALS[3*j1+1],SEALS[3*j2+2]];bclasses=CLASSES[3*i:3*i+3]
        ra=[f"Instrument {ents[j]} carries seal {aseals[j]}." for j in range(3)]
        rb=[f"Seal {bseals[j]} corresponds to routing class {bclasses[j]}." for j in range(3)]
        rr=random.Random(SEED+i*101);rr.shuffle(ra);rr.shuffle(rb)
        items.append({"id":i+1,"A":" ".join(ra),"B":" ".join(rb),
        "gold":{"entity":ents[0],"seal":gold,"class":bclasses[0]},
        "A_seals":aseals,"A_distractors":aseals[1:],"B_seals":bseals,"B_classes":bclasses})
    return items

ITEMS=make_items()
assert len(set(x["gold"]["seal"] for x in ITEMS))==N
for i,it in enumerate(ITEMS):
    assert it["gold"]["seal"] in it["B_seals"]
    assert all(x not in it["B_seals"] for x in it["A_distractors"])
    assert sum(it["gold"]["seal"] in x["B_seals"] for x in ITEMS)==1

LOCK={"test":TEST,"parent521":TEST521,"model":MODEL_ID,"seed":SEED,"N":N,"disc":[0,31],"internal_eval":[32,63],
"address":{"layer":BL,"kind":BK,"center":False},"pointer_layers":PTR_LAYERS,"heads":QH,
"channels":NC,"topks":TOPKS,
"selection":"DISC channel MRR then R1; K chosen by DISC ensemble MRR then R1 then smaller K",
"eval_rule":"internal eval never changes selected channels or K"}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      TEST522 LOCK:",LOCK_SHA)
print("      PARENT521   :",TEST521)
print("      DISC/EVAL   : 32 / 32")
print("      TEST522     : mechanism discovery; TEST523 remains external confirmation")

@torch.inference_mode()
def kv_from_ids(ids):
    o=model(input_ids=torch.tensor([ids],device=DEVICE),use_cache=False,output_hidden_states=True,return_dict=True);out=[]
    for L,layer in enumerate(model.model.layers):
        h=o.hidden_states[L][0].to(layer.input_layernorm.weight.device,layer.input_layernorm.weight.dtype)
        hn=layer.input_layernorm(h)
        k=layer.self_attn.k_proj(hn).view(-1,KVH,HD).transpose(0,1).contiguous()
        v=layer.self_attn.v_proj(hn).view(-1,KVH,HD).transpose(0,1).contiguous()
        out.append((k,v))
    return out

def forge(s):return kv_from_ids([PAD]+enc(s+SEP))

def rope_k(k,pos,L):
    q=torch.zeros((1,QH,len(pos),HD),device=DEVICE,dtype=k.dtype);kk=k.unsqueeze(0)
    layer=model.model.layers[L];rot=layer.self_attn.rotary_emb if hasattr(layer.self_attn,"rotary_emb") else model.model.rotary_emb
    p=torch.tensor([pos],device=DEVICE,dtype=torch.long)
    try:c,s=rot(q,p)
    except TypeError:c,s=rot(q,position_ids=p)
    _,kr=apply_rotary_pos_emb(q,kk,c,s);return kr[0]

def install(raw):
    T=raw[0][0].shape[1]
    return [(rope_k(k,list(range(T)),L),v) for L,(k,v) in enumerate(raw)],T,T

def cache_of(inst):
    try:c=DynamicCache(config=cfg)
    except TypeError:c=DynamicCache()
    for L,(k,v) in enumerate(inst):c.update(k.unsqueeze(0),v.unsqueeze(0),L)
    return c

def qA(e):return f"What seal does instrument {e} carry? Give only the exact seal."

def rank(sc,gold):
    order=sorted(range(N),key=lambda j:(-float(sc[j]),j));return order.index(gold)+1,order[0]

def metrics(r):
    a=np.asarray(r,float);return float(np.mean(a==1)),float(np.mean(a<=5)),float(np.mean(a<=16)),float(np.mean(1/a))

def cos_rows(v,M):
    v=v.float();M=M.float();v=v/v.norm().clamp_min(1e-8);M=M/M.norm(dim=1,keepdim=True).clamp_min(1e-8)
    return M@v

print("\n[2/11] OFFLINE FORGE — fresh 64-item X-ray panel...")
A_RAW=[];B_RAW=[];A_PACK=[]
for i,it in enumerate(ITEMS):
    ar=forge(it["A"]);br=forge(it["B"]);A_RAW.append(ar);B_RAW.append(br);A_PACK.append(install(ar))
    if i<5:print(f"      ITEM {i+1:02d} | A={A_PACK[-1][1]} B={br[0][0].shape[1]} tokens")
print("      128 independent BELLEKÖZ forged.")

def gold_token_positions(text,seal):
    hits=[m.span() for m in re.finditer(re.escape(seal),text)]
    if len(hits)!=1:raise RuntimeError(f"Seal occurrence !=1: {seal}")
    a,b=hits[0];z=tok(text+SEP,add_special_tokens=False,return_offsets_mapping=True);pos=[]
    for j,(x,y) in enumerate(z["offset_mapping"]):
        if y>a and x<b:pos.append(j+1)
    if not pos:raise RuntimeError(f"No span: {seal}")
    return pos
GOLD_SPANS=[gold_token_positions(it["A"],it["gold"]["seal"]) for it in ITEMS]

@torch.inference_mode()
def query_forward(packet,q):
    inst,Tm,P=packet;c=cache_of(inst);ids=enc(FMT.format(q=q));n=len(ids)
    pos=torch.arange(P,P+n,device=DEVICE).unsqueeze(0);mask=torch.ones((1,Tm+n),device=DEVICE,dtype=torch.long)
    o=model(input_ids=torch.tensor([ids],device=DEVICE),past_key_values=c,attention_mask=mask,position_ids=pos,
    use_cache=False,output_hidden_states=True,return_dict=True)
    return o,Tm,P,n

@torch.inference_mode()
def pointer_xray(packet,q):
    inst,Tm,P=packet;o,_,_,n=query_forward(packet,q);rows=[];qpos=P+n-1
    for L in PTR_LAYERS:
        layer=model.model.layers[L];h=o.hidden_states[L][0,-1].to(layer.input_layernorm.weight.dtype);hn=layer.input_layernorm(h)
        qv=layer.self_attn.q_proj(hn).view(1,QH,1,HD)
        ak=inst[L][0].repeat_interleave(REP,dim=0).unsqueeze(0)
        dummy=torch.zeros_like(ak[:,:,:1,:]);rot=layer.self_attn.rotary_emb if hasattr(layer.self_attn,"rotary_emb") else model.model.rotary_emb
        pp=torch.tensor([[qpos]],device=DEVICE,dtype=torch.long)
        try:c,s=rot(dummy,pp)
        except TypeError:c,s=rot(dummy,position_ids=pp)
        qrot,_=apply_rotary_pos_emb(qv,dummy,c,s)
        score=torch.einsum("bhqd,bhkd->bhqk",qrot.float(),ak.float()).squeeze(0).squeeze(1)/math.sqrt(HD)
        score[:,0]=-torch.inf
        rows.append(torch.softmax(score,dim=-1))
    W=torch.cat(rows,dim=0).detach().cpu()
    W[:,0]=0;W=W/W.sum(dim=1,keepdim=True).clamp_min(1e-12)
    return W

def aggregate(W,idxs):
    w=W[idxs].mean(0);w[0]=0;return w/w.sum().clamp_min(1e-12)

def pointer_vector(raw,w):
    x=raw[BL][1].float().cpu()
    return (x*w[None,:,None]).sum(dim=1).reshape(-1)

# Precompute flattened B L02-V position matrices once.
B_MATS=[]
for raw in B_RAW:
    x=raw[BL][1][:,1:,:].float().cpu()
    B_MATS.append(x.permute(1,0,2).reshape(x.shape[1],-1).contiguous())

def B_scores(p):
    return [float(cos_rows(p,M).max()) for M in B_MATS]

print("\n[3/11] Extracting 392-channel ÇAĞRIİZ X-ray...")
XRAY=[]
for i,it in enumerate(ITEMS):
    W=pointer_xray(A_PACK[i],qA(it["gold"]["entity"]));XRAY.append(W)
    wa=W.mean(0);span=GOLD_SPANS[i];mass=float(wa[span].sum());arg=int(torch.argmax(wa))
    print(f"      [{i+1:02d}/64] {it['gold']['entity']:8s} | T={W.shape[1]:2d} all392-arg={arg:2d} gold-mass={mass:.4f}")
print("      X-ray extraction complete.")

print("\n[4/11] DISC — individual channel retrieval atlas (items 01..32)...")
CH_RANKS=[[] for _ in range(NC)]
CH_MASS=[[] for _ in range(NC)]
for i in range(NDISC):
    W=XRAY[i];span=GOLD_SPANS[i]
    for c in range(NC):
        w=W[c];p=pointer_vector(A_RAW[i],w);r,_=rank(B_scores(p),i)
        CH_RANKS[c].append(r);CH_MASS[c].append(float(w[span].sum()))
CH_MET=[metrics(x) for x in CH_RANKS]
ORDER=sorted(range(NC),key=lambda c:(-CH_MET[c][3],-CH_MET[c][0],CHANNELS[c][0],CHANNELS[c][1]))
print("      TOP 20 DISC CHANNELS:")
for z,c in enumerate(ORDER[:20],1):
    L,h=CHANNELS[c];m=CH_MET[c]
    print(f"      #{z:02d} L{L:02d}H{h:02d} | R1={m[0]:.4f} R5={m[1]:.4f} R16={m[2]:.4f} MRR={m[3]:.6f} gold-mass={np.mean(CH_MASS[c]):.4f}")

print("\n[5/11] DISC — frozen Top-K ensembles...")
DISC_K={}
for K in TOPKS:
    idx=ORDER[:K];rr=[]
    for i in range(NDISC):
        w=aggregate(XRAY[i],idx);p=pointer_vector(A_RAW[i],w);r,_=rank(B_scores(p),i);rr.append(r)
    m=metrics(rr);DISC_K[K]=(rr,m)
    print(f"      TOP{K:03d} | R1={m[0]:.4f} R5={m[1]:.4f} R16={m[2]:.4f} MRR={m[3]:.6f}")
BESTK=sorted(TOPKS,key=lambda k:(-DISC_K[k][1][3],-DISC_K[k][1][0],k))[0]
FROZEN=ORDER[:BESTK]
print(f"      TEST523 CANDIDATE FROZEN FROM DISC ONLY: TOP{BESTK}")
print("      CHANNELS:",",".join(f"L{CHANNELS[c][0]:02d}H{CHANNELS[c][1]:02d}" for c in FROZEN))

print("\n[6/11] INTERNAL-EVAL — all392 baseline vs DISC-frozen ensembles...")
EVAL_ALL=[];EVAL_K={K:[] for K in TOPKS}
for i in range(NDISC,N):
    wa=aggregate(XRAY[i],list(range(NC)));r,_=rank(B_scores(pointer_vector(A_RAW[i],wa)),i);EVAL_ALL.append(r)
    for K in TOPKS:
        w=aggregate(XRAY[i],ORDER[:K]);rk,_=rank(B_scores(pointer_vector(A_RAW[i],w)),i);EVAL_K[K].append(rk)
ma=metrics(EVAL_ALL)
print(f"      ALL392 | R1={ma[0]:.4f} R5={ma[1]:.4f} R16={ma[2]:.4f} MRR={ma[3]:.6f}")
for K in TOPKS:
    m=metrics(EVAL_K[K])
    marker="  <-- DISC-FROZEN CANDIDATE" if K==BESTK else ""
    print(f"      TOP{K:03d} | R1={m[0]:.4f} R5={m[1]:.4f} R16={m[2]:.4f} MRR={m[3]:.6f}{marker}")

print("\n[7/11] Layer concentration X-ray...")
LAYER_DISC={}
LAYER_EVAL={}
for L in PTR_LAYERS:
    idx=[c for c,(ll,h) in enumerate(CHANNELS) if ll==L];rd=[];re=[]
    for i in range(NDISC):
        w=aggregate(XRAY[i],idx);r,_=rank(B_scores(pointer_vector(A_RAW[i],w)),i);rd.append(r)
    for i in range(NDISC,N):
        w=aggregate(XRAY[i],idx);r,_=rank(B_scores(pointer_vector(A_RAW[i],w)),i);re.append(r)
    LAYER_DISC[L]=metrics(rd);LAYER_EVAL[L]=metrics(re)
    d=LAYER_DISC[L];e=LAYER_EVAL[L]
    print(f"      L{L:02d} | DISC R1={d[0]:.4f} MRR={d[3]:.4f} | EVAL R1={e[0]:.4f} MRR={e[3]:.4f}")

print("\n[8/11] Head-index concentration across layers...")
HEAD_DISC={}
HEAD_EVAL={}
for h in range(QH):
    idx=[c for c,(L,hh) in enumerate(CHANNELS) if hh==h];rd=[];re=[]
    for i in range(NDISC):
        w=aggregate(XRAY[i],idx);r,_=rank(B_scores(pointer_vector(A_RAW[i],w)),i);rd.append(r)
    for i in range(NDISC,N):
        w=aggregate(XRAY[i],idx);r,_=rank(B_scores(pointer_vector(A_RAW[i],w)),i);re.append(r)
    HEAD_DISC[h]=metrics(rd);HEAD_EVAL[h]=metrics(re)
HO=sorted(range(QH),key=lambda h:(-HEAD_DISC[h][3],-HEAD_DISC[h][0],h))
for h in HO[:10]:
    d=HEAD_DISC[h];e=HEAD_EVAL[h]
    print(f"      H{h:02d} | DISC R1={d[0]:.4f} MRR={d[3]:.4f} | EVAL R1={e[0]:.4f} MRR={e[3]:.4f}")

print("\n[9/11] Gold-span enrichment vs retrieval quality...")
r1=np.array([CH_MET[c][0] for c in range(NC)],float)
mrr=np.array([CH_MET[c][3] for c in range(NC)],float)
gm=np.array([np.mean(CH_MASS[c]) for c in range(NC)],float)
def corr(a,b):
    return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else float("nan")
print(f"      corr(channel gold-mass, DISC R1)  = {corr(gm,r1):+.6f}")
print(f"      corr(channel gold-mass, DISC MRR) = {corr(gm,mrr):+.6f}")
topmass=np.argsort(-gm)[:10]
print("      TOP GOLD-SPAN-MASS CHANNELS:")
for c in topmass:
    L,h=CHANNELS[c];m=CH_MET[c]
    print(f"      L{L:02d}H{h:02d} | mass={gm[c]:.4f} DISC R1={m[0]:.4f} MRR={m[3]:.4f}")

print("\n[10/11] Mechanism classification...")
MB=metrics(EVAL_ALL);MF=metrics(EVAL_K[BESTK]);DB=DISC_K[BESTK][1]
if MF[0]>=MB[0]+.15 and MF[0]>=.50:
    VERDICT="CAGRIIIZ_SIGNAL_CONCENTRATED_IN_DISCOVERED_NATIVE_CHANNEL_SUBSET"
elif MF[0]>=.50 and MF[0]>=MB[0]:
    VERDICT="CAGRIIIZ_CHANNEL_SUBSET_REPLICATES_AND_MATCHES_OR_EXCEEDS_DISTRIBUTED_BASELINE"
elif MB[0]>=.35 and MF[0]<MB[0]-.10:
    VERDICT="CAGRIIIZ_SIGNAL_APPEARS_DISTRIBUTED_NOT_SPARSE"
elif MB[0]<.25 and MF[0]<.25:
    VERDICT="CAGRIIIZ_SIGNAL_DID_NOT_REPLICATE_IN_TEST522_PANEL"
else:
    VERDICT="CAGRIIIZ_CHANNEL_STRUCTURE_PARTIAL_INCONCLUSIVE"
print(f"      DISC TOP{BESTK}: R1={DB[0]:.4f} MRR={DB[3]:.6f}")
print(f"      EVAL ALL392 : R1={MB[0]:.4f} MRR={MB[3]:.6f}")
print(f"      EVAL TOP{BESTK}: R1={MF[0]:.4f} MRR={MF[3]:.6f}")
print("      VERDICT:",VERDICT)

print("\n[11/11] Protocol audit + final...")
print("      Model frozen                         : YES")
print("      Address geometry                     : TEST521 L02-V UNCENTERED")
print("      Address geometry changed             : NO")
print("      New panel                            : YES")
print("      DISC / internal-eval                 : 32 / 32")
print("      Channels inspected                   : 392")
print("      Channel ranking uses DISC only       : YES")
print("      K selection uses DISC only           : YES")
print("      Internal-eval changes selection      : NO")
print("      Gold span used for channel selection : NO")
print("      Gold span diagnostic only            : YES")
print("      Token-ID / text / LM-head address    : NONE")
print("      Training / DRA / learned router      : NONE")
print("      TEST523 still required               : YES")

print("\n"+"="*176)
print("TEST522 FINAL RESULT — ÇAĞRIİZ HEAD×LAYER X-RAY")
print("="*176)
print("MODEL                              : Qwen/Qwen2.5-7B-Instruct · frozen")
print("ADDRESS                            : L02-V · UNCENTERED")
print("XRAY                               : L14-L27 × H00-H27 = 392 channels")
print("DISC / INTERNAL-EVAL               : 32 / 32")
print("-"*176)
print(f"DISC-FROZEN CANDIDATE              : TOP{BESTK}")
print("FROZEN CHANNELS                    :",",".join(f"L{CHANNELS[c][0]:02d}H{CHANNELS[c][1]:02d}" for c in FROZEN))
print(f"DISC TOP{BESTK}                    : R1={DB[0]:.4f} R5={DB[1]:.4f} R16={DB[2]:.4f} MRR={DB[3]:.6f}")
print(f"INTERNAL-EVAL ALL392               : R1={MB[0]:.4f} R5={MB[1]:.4f} R16={MB[2]:.4f} MRR={MB[3]:.6f}")
print(f"INTERNAL-EVAL TOP{BESTK}           : R1={MF[0]:.4f} R5={MF[1]:.4f} R16={MF[2]:.4f} MRR={MF[3]:.6f}")
print(f"GOLD-MASS ↔ CHANNEL R1 CORR        : {corr(gm,r1):+.6f}")
print(f"GOLD-MASS ↔ CHANNEL MRR CORR       : {corr(gm,mrr):+.6f}")
print("-"*176)
print("TEST521 LOCK                       :",TEST521)
print("TEST522 LOCK                       :",LOCK_SHA)
print(f"TOTAL TEST TIME                    : {time.perf_counter()-T0:.2f}s")
print("VERDICT                            :",VERDICT)
print("="*176)

# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST534
# VARAN 1 — POINTER-MISS X-RAY
#
# SEALED CHAIN:
#   TEST528 -> native single-token external seal
#   TEST529 -> multi-token primitive boundary
#   TEST530 -> composition X-ray / CONCATCOS discovered
#   TEST531 -> untouched multi-token validation
#              PRIMARY R1 = 0.9824
#   TEST532 -> span-origin X-ray
#   TEST533 -> PEAK_CONTAIN origin resolution
#              PRIMARY R1 = 0.9932
#              CF R1      = 0.9912
#              11/18 baseline failures recovered
#              0 previously-correct cases broken
#
# TEST534 PURPOSE:
#   Diagnose the remaining failure class where the frozen L28H00 pointer
#   peak does NOT fall inside the gold multi-token identity span.
#
# FROZEN / UNCHANGED:
#   MODEL   = mistralai/Mistral-7B-Instruct-v0.3
#   POINTER = L28H00
#   ADDRESS = L00-V UNCENTERED
#   MATCH   = position-preserving CONCATCOS
#   PANEL   = exact TEST531/532/533 deterministic construction
#   IDENTITY LENGTHS = 2/3/4
#
# IMPORTANT:
#   TEST534 IS DIAGNOSTIC ONLY.
#
#   NO new retrieval law.
#   NO new origin law.
#   NO head/layer/address scan.
#   NO threshold tuning.
#   NO training.
#   NO learned router/scorer.
#
# QUESTION:
#   When pointer ARGMAX misses the gold identity span, is the gold span
#   still strongly represented elsewhere in the frozen pointer distribution?
#
# MEASUREMENTS:
#   1. Pointer ARGMAX hit / miss.
#   2. Gold-span total pointer mass.
#   3. Strongest token inside gold span.
#   4. Rank of strongest gold-span token among all memory positions.
#   5. Rank of gold span among ALL contiguous width-L windows by:
#          a) MAX token pointer weight
#          b) SUM pointer mass
#   6. Gold span versus winning non-gold pointer island.
#   7. Top-K containment diagnostics: K=1,2,3,5,8,16.
#   8. Primary and counterfactual separately.
#
# INTERPRETATION:
#   If residual gold spans remain high-ranked:
#       pointer signal survives; collapse to one peak is the bottleneck.
#
#   If residual gold spans are genuinely low-ranked:
#       L28H00 itself has a residual localization boundary.
#
# Gold spans are used ONLY for post-selection X-ray scoring.

import os,sys,subprocess,importlib.util,random,time,hashlib,json,gc,re,math
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

TEST="534";SEED=531531;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";N=1024
DEVICE=torch.device("cuda");FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
PTR_L=28;PTR_H=0;BL=0;BK="V";CENTER=False;LENS=(2,3,4);TOPKS=(1,2,3,5,8,16)
PARENT528="9235ed38c944f41a9ede0426d3aabaa12b5b20646f30e2be921e618f9471d3c2"
PARENT529="27ba703872f0d8a6ed870a6ba041d0cd0703d2eca467deac97186a90497795a0"
PARENT530="289247c88dfd4358cbe5ffb8499891546b27acc6912caf525a155ae81338db7b"
PARENT531="20257d7d613a49e55b03603946f07ec493009924f40ac26ab74cb817e1466528"
PARENT532="9e37a473a322a19f7b3862a25162835ab6396998dc4fc416bc90bc9b718df188"
PARENT533="5239fb585b87a22f5da778c61a1011379b216a25bf7d05dc5d01952cbb22a5c6"

os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*176)
print("TEST534 — AKBASCORE MAM · VARAN 1 · POINTER-MISS X-RAY")
print("TEST533 FOLLOW-UP · FROZEN L28H00 POINTER RESIDUAL-FAILURE ANATOMY · NO NEW RETRIEVAL LAW")
print("="*176);T0=time.perf_counter()

print("\n[1/11] Loading frozen Mistral...")
_tv=tuple(int("".join(c for c in x if c.isdigit())or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
if not tok.is_fast:raise RuntimeError("Fast tokenizer required.")
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
enc=lambda s:tok(s,add_special_tokens=False).input_ids
cfg=model.config;layers=model.model.layers;NL=len(layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads
HD=getattr(cfg,"head_dim",None) or H//QH;REP=QH//KVH;KVD=KVH*HD
if (NL,H,QH,KVH,HD)!=(32,4096,32,8,128):raise RuntimeError(f"Architecture mismatch: {(NL,H,QH,KVH,HD)}")
if QH%KVH:raise RuntimeError("QH must be divisible by KVH.")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id;VOC=model.model.embed_tokens.weight.shape[0]
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)}")
print(f"      {NL}L H={H} QH={QH} KVH={KVH} HD={HD} ADDRESS={KVD}")
print("      POINTER : L28H00 · FROZEN")
print("      ADDRESS : L00-V UNCENTERED · FROZEN")
print("      MATCH   : CONCATCOS · FROZEN")
print("      PANEL   : exact TEST531/532/533 deterministic construction")
print("      MODE    : POINTER X-RAY ONLY · NO NEW RETRIEVAL RULE")
print("      MODEL   : frozen · BF16 · SDPA · trainable=0")

FP=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[16].mlp.down_proj.weight,
    layers[24].self_attn.o_proj.weight,layers[31].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
def sentinel():
    h=hashlib.sha256()
    for p in FP:
        a=p.detach().reshape(-1);n=a.numel()
        for j in range(16):
            o=j*(n-256)//15;h.update(a[o:o+256].float().cpu().numpy().tobytes())
    return h.hexdigest()
S0=sentinel()

def native_pool():
    out=[]
    for tid in range(VOC):
        s=tok.decode([tid],skip_special_tokens=False)
        if not re.fullmatch(r"[A-Z]{3,8}",s):continue
        if enc(s)!=[tid]:continue
        if s not in out:out.append(s)
    return out

POOL=native_pool()
if len(POOL)<100:raise RuntimeError(f"Native atom pool too small: {len(POOL)}")
random.Random(SEED+11).shuffle(POOL)
print("      Native model-token atom pool:",len(POOL))

def exact_span(text,key):
    hits=[m.span() for m in re.finditer(r"(?<![A-Za-z0-9])"+re.escape(key)+r"(?![A-Za-z0-9])",text)]
    if len(hits)!=1:return None
    a,b=hits[0];z=tok(text+SEP,add_special_tokens=False,return_offsets_mapping=True);p=[]
    for j,(x,y) in enumerate(z["offset_mapping"]):
        if y>a and x<b:p.append(j+1)
    return p

def valid_identity(parts,L):
    if len(set(parts))!=L:return None
    s=" ".join(parts)
    pa=exact_span(f"Instrument Probe carries seal {s}.",s)
    pb=exact_span(f"Seal {s} corresponds to routing class ALPHA.",s)
    return s if pa and pb and len(pa)==L and len(pb)==L else None

print("\n[2/11] Reconstructing exact sealed panel...")
counts={L:sum(1 for i in range(N) if LENS[i%len(LENS)]==L) for L in LENS}
need={L:counts[L]*5+64 for L in LENS}
rr=random.Random(SEED+100);ALL={L:[] for L in LENS};USED={L:set() for L in LENS};tries=0
while any(len(ALL[L])<need[L] for L in LENS) and tries<5000000:
    tries+=1
    for L in LENS:
        if len(ALL[L])>=need[L]:continue
        s=valid_identity([rr.choice(POOL) for _ in range(L)],L)
        if s is None or s in USED[L]:continue
        USED[L].add(s);ALL[L].append(s)
if any(len(ALL[L])<need[L] for L in LENS):
    raise RuntimeError(f"Identity generation exhausted: have={dict((L,len(ALL[L])) for L in LENS)} need={need}")

TARGETS=[];TLEN=[]
for i in range(N):
    L=LENS[i%len(LENS)];TARGETS.append(ALL[L].pop());TLEN.append(L)

def pop_same(L,forbidden):
    while ALL[L]:
        x=ALL[L].pop()
        if x not in forbidden:return x
    raise RuntimeError(f"Identity pool exhausted for L={L}")

SYL1=["Aq","Bryn","Cav","Dex","Eln","Frax","Gyr","Hex","Iln","Jor","Kex","Lyr","Mav","Nex","Oryn","Pax",
      "Qyr","Rex","Savn","Tov","Uln","Vex","Wyr","Xav","Yex","Zyr","Axl","Bov","Cyn","Drex","Evr","Fyn"]
SYL2=["adar","bren","cyr","dax","elor","fyn","grel","hyn","ivar","jor","kyr","lor","myn","nex","or","pyr",
      "qen","rix","sor","tyn","ul","vyr","wen","xir","yor","zen"]
def build_names(n):
    out=[]
    for a in SYL1:
        for b in SYL2:
            for c in ("a","e","i","o"):
                x=a+b+c
                if x not in out:out.append(x)
    random.Random(SEED+77).shuffle(out)
    if len(out)<n:raise RuntimeError(f"Name pool too small: have={len(out)} need={n}")
    return out[:n]

NAMES=build_names(N*3);CL=["ALPHA","BETA","GAMMA"]

def make_items():
    out=[]
    for i in range(N):
        L=TLEN[i];gold=TARGETS[i];forbid={gold}
        a1=pop_same(L,forbid);forbid.add(a1)
        a2=pop_same(L,forbid);forbid.add(a2)
        b1=pop_same(L,forbid);forbid.add(b1)
        b2=pop_same(L,forbid);forbid.add(b2)
        ents=NAMES[3*i:3*i+3]
        ra=[f"Instrument {ents[0]} carries seal {gold}.",
            f"Instrument {ents[1]} carries seal {a1}.",
            f"Instrument {ents[2]} carries seal {a2}."]
        rb=[f"Seal {gold} corresponds to routing class {CL[0]}.",
            f"Seal {b1} corresponds to routing class {CL[1]}.",
            f"Seal {b2} corresponds to routing class {CL[2]}."]
        random.Random(SEED+i*131+17).shuffle(ra);random.Random(SEED+i*137+29).shuffle(rb)
        out.append({"id":i+1,"L":L,"A":" ".join(ra),"B":" ".join(rb),"entity":ents[0],"gold":gold,"A_d":[a1,a2]})
    return out

ITEMS=make_items()
print("      Required identity pools:",need)
for L in LENS:print(f"      L={L}: {sum(x['L']==L for x in ITEMS)}")
print("      Remaining reserve:",{L:len(ALL[L]) for L in LENS})
print("      Sealed panel seed:",SEED)

LOCK={"test":TEST,
"parents":{"528":PARENT528,"529":PARENT529,"530":PARENT530,"531":PARENT531,"532":PARENT532,"533":PARENT533},
"model":MODEL_ID,"seed":SEED,"N":N,"status":"VARAN1_POINTER_MISS_XRAY",
"pointer":{"layer":PTR_L,"head":PTR_H},
"address":{"layer":BL,"kind":BK,"center":CENTER,"dimension":KVD},
"identity":{"lengths":[2,3,4],"composition":"ordered native-token sequence"},
"match":{"name":"CONCATCOS","locked_from":"TEST530"},
"measurements":["ARGMAX_HIT","GOLD_SPAN_MASS","GOLD_TOKEN_MAX_RANK","GOLD_WINDOW_MAX_RANK",
"GOLD_WINDOW_SUM_RANK","TOPK_GOLD_TOKEN_CONTAINMENT","RESIDUAL_PRIMARY","RESIDUAL_COUNTERFACTUAL"],
"selection":"DIAGNOSTIC_ONLY_NO_NEW_RETRIEVAL_LAW",
"gold_usage":"POST_SELECTION_XRAY_ONLY",
"forbidden":["NEW_ORIGIN_RULE","NEW_RETRIEVAL_RULE","HEAD_SCAN","LAYER_SCAN","ADDRESS_SCAN","CENTER_SCAN",
"MATCH_SCAN","TOKEN_ID_ROUTING","DECODED_ID_ROUTING","GOLD_SELECTION","QUERY_TIME_B_FORWARD","TRAINING",
"LORA","DRA","LEARNED_ROUTER","THRESHOLD_TUNING","POSTHOC_RULE_CREATION"]}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      TEST534 LOCK:",LOCK_SHA)

@torch.inference_mode()
def kv_from_ids(ids):
    o=model(input_ids=torch.tensor([ids],device=DEVICE),use_cache=False,output_hidden_states=True,return_dict=True);out=[]
    for li,layer in enumerate(layers):
        h=o.hidden_states[li][0].to(layer.input_layernorm.weight.device,layer.input_layernorm.weight.dtype)
        hn=layer.input_layernorm(h)
        k=layer.self_attn.k_proj(hn).view(-1,KVH,HD).transpose(0,1).contiguous()
        v=layer.self_attn.v_proj(hn).view(-1,KVH,HD).transpose(0,1).contiguous()
        out.append((k,v))
    del o
    return out

def forge(s):return kv_from_ids([PAD]+enc(s+SEP))

def rope_cos_sin(x,pos):
    p=torch.tensor([pos],device=DEVICE,dtype=torch.long)
    try:return model.model.rotary_emb(x,p)
    except TypeError:return model.model.rotary_emb(x,position_ids=p)

def rope_k(k,pos):
    T=len(pos);kk=k.unsqueeze(0);dummy=torch.zeros((1,KVH,T,HD),device=DEVICE,dtype=k.dtype)
    c,s=rope_cos_sin(dummy,pos);_,kr=apply_rotary_pos_emb(dummy,kk,c,s,unsqueeze_dim=1)
    return kr[0]

def install(raw):
    T=raw[0][0].shape[1]
    return [(rope_k(k,list(range(T))),v) for k,v in raw],T,T

def cache_of(inst):
    try:c=DynamicCache(config=cfg)
    except TypeError:c=DynamicCache()
    for li,(k,v) in enumerate(inst):c.update(k.unsqueeze(0),v.unsqueeze(0),li)
    return c

def qA(e):return f"What seal does instrument {e} carry? Give only the exact seal."

print("\n[3/11] OFFLINE FORGE — A memories only...")
A_RAW=[];A_PACK=[];GOLD_SPANS=[]
for i,it in enumerate(ITEMS):
    ar=forge(it["A"]);sp=exact_span(it["A"],it["gold"])
    if sp is None or len(sp)!=it["L"]:raise RuntimeError(f"A span mismatch item {i+1}")
    A_RAW.append(ar);A_PACK.append(install(ar));GOLD_SPANS.append(sp)
    if (i+1)%64==0:print(f"      forged {i+1:4d}/{N}")
gc.collect();torch.cuda.empty_cache()
print("      A memories:",len(A_RAW))
print("      B candidate forwards required: 0")
print("      Retrieval executed: NO")
print("      Pointer X-ray only: YES")

@torch.inference_mode()
def query_forward(packet,q):
    inst,Tm,P=packet;c=cache_of(inst);ids=enc(FMT.format(q=q));n=len(ids)
    pos=torch.arange(P,P+n,device=DEVICE).unsqueeze(0)
    mask=torch.ones((1,Tm+n),device=DEVICE,dtype=torch.long)
    o=model(input_ids=torch.tensor([ids],device=DEVICE),past_key_values=c,attention_mask=mask,
            position_ids=pos,use_cache=False,output_hidden_states=True,return_dict=True)
    return o,Tm,P,n

@torch.inference_mode()
def frozen_pointer(packet,q):
    inst,Tm,P=packet;o,_,_,n=query_forward(packet,q);qpos=P+n-1;layer=layers[PTR_L]
    h=o.hidden_states[PTR_L][0,-1].to(layer.input_layernorm.weight.dtype);hn=layer.input_layernorm(h)
    qv=layer.self_attn.q_proj(hn).view(1,QH,1,HD)
    ak=inst[PTR_L][0].repeat_interleave(REP,dim=0).unsqueeze(0)
    dummy=torch.zeros((1,QH,1,HD),device=DEVICE,dtype=qv.dtype)
    c,s=rope_cos_sin(dummy,[qpos]);qrot,_=apply_rotary_pos_emb(qv,dummy,c,s,unsqueeze_dim=1)
    score=torch.einsum("bhqd,bhkd->bhqk",qrot.float(),ak.float()).squeeze(0).squeeze(1)/math.sqrt(HD)
    score[:,0]=-torch.inf;w=torch.softmax(score,dim=-1).detach().cpu();w[:,0]=0
    del o,qv,ak,dummy,c,s,qrot,score
    z=w[PTR_H]
    return z/z.sum().clamp_min(1e-12)

def rank_desc(vals,index):
    order=torch.argsort(vals,descending=True)
    hit=(order==index).nonzero(as_tuple=False)
    return int(hit[0,0])+1 if len(hit) else len(vals)+1

def pointer_xray(w,sp,L):
    peak=int(torch.argmax(w))
    gold_start=sp[0]
    gold_set=set(sp)
    valid=w[1:]
    token_order=(torch.argsort(valid,descending=True)+1).tolist()
    gold_token_ranks=[token_order.index(p)+1 for p in sp]
    best_gold_token_rank=min(gold_token_ranks)
    best_gold_token=max(sp,key=lambda p:float(w[p]))
    best_gold_weight=float(w[best_gold_token])
    gold_mass=float(w[sp].sum())
    n=len(w)-1
    starts=list(range(1,n-L+2))
    max_scores=torch.tensor([float(w[s:s+L].max()) for s in starts])
    sum_scores=torch.tensor([float(w[s:s+L].sum()) for s in starts])
    gi=gold_start-1
    gold_max_rank=rank_desc(max_scores,gi)
    gold_sum_rank=rank_desc(sum_scores,gi)
    win_max=starts[int(torch.argmax(max_scores))]
    win_sum=starts[int(torch.argmax(sum_scores))]
    topk={}
    for k in TOPKS:
        top=set(token_order[:min(k,len(token_order))])
        topk[k]=int(bool(top & gold_set))
    non_gold=[p for p in range(1,len(w)) if p not in gold_set]
    ng=max(non_gold,key=lambda p:float(w[p])) if non_gold else -1
    ngw=float(w[ng]) if ng>=0 else 0.0
    return {"peak":peak,"inside":int(peak in gold_set),"exact_start":int(peak==gold_start),
            "gold_start":gold_start,"gold_mass":gold_mass,"best_gold_token":best_gold_token,
            "best_gold_weight":best_gold_weight,"best_gold_token_rank":best_gold_token_rank,
            "gold_token_ranks":gold_token_ranks,"gold_max_window_rank":gold_max_rank,
            "gold_sum_window_rank":gold_sum_rank,"winning_max_start":win_max,"winning_sum_start":win_sum,
            "best_non_gold_token":ng,"best_non_gold_weight":ngw,
            "gold_vs_nongold_ratio":best_gold_weight/max(ngw,1e-12),"topk":topk}

def summarize(rows,name):
    miss=[r for r in rows if not r["inside"]]
    print(f"\n      {name} ALL:")
    print(f"      pointer peak inside gold span       : {np.mean([r['inside'] for r in rows]):.4f}")
    print(f"      pointer exact gold-start            : {np.mean([r['exact_start'] for r in rows]):.4f}")
    print(f"      mean gold-span mass                 : {np.mean([r['gold_mass'] for r in rows]):.6f}")
    print(f"      median gold-span mass               : {np.median([r['gold_mass'] for r in rows]):.6f}")
    print(f"      residual pointer misses             : {len(miss)}/{len(rows)}")
    for k in TOPKS:
        print(f"      gold token present in pointer Top-{k:<2d} : {np.mean([r['topk'][k] for r in rows]):.4f}")
    if miss:
        print(f"\n      {name} POINTER-MISS ONLY:")
        print(f"      count                               : {len(miss)}")
        print(f"      mean gold-span mass                 : {np.mean([r['gold_mass'] for r in miss]):.6f}")
        print(f"      median gold-span mass               : {np.median([r['gold_mass'] for r in miss]):.6f}")
        print(f"      mean best-gold-token rank           : {np.mean([r['best_gold_token_rank'] for r in miss]):.3f}")
        print(f"      median best-gold-token rank         : {np.median([r['best_gold_token_rank'] for r in miss]):.3f}")
        print(f"      mean gold MAX-window rank           : {np.mean([r['gold_max_window_rank'] for r in miss]):.3f}")
        print(f"      mean gold SUM-window rank           : {np.mean([r['gold_sum_window_rank'] for r in miss]):.3f}")
        for k in TOPKS:
            print(f"      miss: gold token in Top-{k:<2d}          : {sum(r['topk'][k] for r in miss)}/{len(miss)}")
    return miss

print("\n[4/11] PRIMARY POINTER X-RAY...")
PRIMARY=[]
for i,it in enumerate(ITEMS):
    w=frozen_pointer(A_PACK[i],qA(it["entity"]))
    r=pointer_xray(w,GOLD_SPANS[i],it["L"]);r.update({"id":i+1,"L":it["L"]});PRIMARY.append(r)
    if i<8 or not r["inside"] or (i+1)%64==0:
        print(f"      [{i+1:04d}/{N}] L={it['L']} gold={r['gold_start']:3d} peak={r['peak']:3d} "
              f"inside={r['inside']} mass={r['gold_mass']:.4f} "
              f"tokR={r['best_gold_token_rank']:3d} maxR={r['gold_max_window_rank']:3d} "
              f"sumR={r['gold_sum_window_rank']:3d}")
PMISS=summarize(PRIMARY,"PRIMARY")

print("\n[5/11] PRIMARY RESIDUAL FAILURE DETAIL...")
if not PMISS:print("      No pointer misses.")
for r in PMISS:
    print(f"      ID={r['id']:04d} L={r['L']} gold={r['gold_start']:3d} peak={r['peak']:3d} "
          f"bestGold={r['best_gold_token']:3d} tokenRank={r['best_gold_token_rank']:3d} "
          f"goldMass={r['gold_mass']:.6f} maxWinRank={r['gold_max_window_rank']:3d} "
          f"sumWinRank={r['gold_sum_window_rank']:3d} "
          f"bestNG={r['best_non_gold_token']:3d} ratio={r['gold_vs_nongold_ratio']:.4f} "
          f"TopK={r['topk']}")

print("\n[6/11] COUNTERFACTUAL POINTER X-RAY...")
CF=[]
for i,it in enumerate(ITEMS):
    j=(i+353)%N;target=TARGETS[j];L=TLEN[j];src=ITEMS[j]
    ents=[it["entity"],NAMES[3*i+1],NAMES[3*i+2]]
    rows=[f"Instrument {ents[0]} carries seal {target}.",
          f"Instrument {ents[1]} carries seal {src['A_d'][0]}.",
          f"Instrument {ents[2]} carries seal {src['A_d'][1]}."]
    random.Random(SEED+90000+i*149).shuffle(rows);cfA=" ".join(rows)
    raw=forge(cfA);pack=install(raw);sp=exact_span(cfA,target)
    if sp is None or len(sp)!=L:raise RuntimeError(f"CF span mismatch item {i+1}")
    w=frozen_pointer(pack,qA(it["entity"]))
    r=pointer_xray(w,sp,L);r.update({"id":i+1,"target":j+1,"L":L});CF.append(r)
    if i<5 or not r["inside"] or (i+1)%64==0:
        print(f"      [{i+1:04d}/{N}] target={j+1:04d} L={L} gold={r['gold_start']:3d} "
              f"peak={r['peak']:3d} inside={r['inside']} mass={r['gold_mass']:.4f} "
              f"tokR={r['best_gold_token_rank']:3d} maxR={r['gold_max_window_rank']:3d} "
              f"sumR={r['gold_sum_window_rank']:3d}")
    del raw,pack,w
CFMISS=summarize(CF,"COUNTERFACTUAL")

print("\n[7/11] COUNTERFACTUAL RESIDUAL FAILURE DETAIL...")
if not CFMISS:print("      No counterfactual pointer misses.")
for r in CFMISS:
    print(f"      ID={r['id']:04d} target={r['target']:04d} L={r['L']} gold={r['gold_start']:3d} "
          f"peak={r['peak']:3d} bestGold={r['best_gold_token']:3d} "
          f"tokenRank={r['best_gold_token_rank']:3d} goldMass={r['gold_mass']:.6f} "
          f"maxWinRank={r['gold_max_window_rank']:3d} sumWinRank={r['gold_sum_window_rank']:3d} "
          f"bestNG={r['best_non_gold_token']:3d} ratio={r['gold_vs_nongold_ratio']:.4f} "
          f"TopK={r['topk']}")

print("\n[8/11] RESIDUAL POINTER-RANK STRUCTURE...")
def rank_table(rows,label):
    miss=[r for r in rows if not r["inside"]]
    print(f"      {label}:")
    for k in TOPKS:
        n=sum(r["topk"][k] for r in miss)
        print(f"        Gold identity represented in Top-{k:<2d}: {n}/{len(miss) if miss else 0}")
    if miss:
        print("        Best-gold-token ranks :",[r["best_gold_token_rank"] for r in miss])
        print("        Gold MAX-window ranks :",[r["gold_max_window_rank"] for r in miss])
        print("        Gold SUM-window ranks :",[r["gold_sum_window_rank"] for r in miss])
        print("        Gold-span masses      :",[round(r["gold_mass"],6) for r in miss])
rank_table(PRIMARY,"PRIMARY")
rank_table(CF,"COUNTERFACTUAL")

print("\n[9/11] LENGTH-STRATIFIED POINTER MISS...")
for label,rows in [("PRIMARY",PRIMARY),("CF",CF)]:
    print("      "+label)
    for L in LENS:
        z=[r for r in rows if r["L"]==L];m=[r for r in z if not r["inside"]]
        print(f"        L={L}: miss={len(m):2d}/{len(z):3d} "
              f"peak-hit={np.mean([r['inside'] for r in z]):.4f} "
              f"exact-start={np.mean([r['exact_start'] for r in z]):.4f} "
              f"mass={np.mean([r['gold_mass'] for r in z]):.6f}")

print("\n[10/11] PROTOCOL AUDIT...")
S1=sentinel();WEIGHT_OK=S0==S1
print("      Weight sentinel                     :","PASS" if WEIGHT_OK else "FAIL")
print("      Trainable tensors                   :",sum(int(p.requires_grad) for p in model.parameters()))
print("      Pointer                             : L28H00 unchanged")
print("      Address                             : L00-V uncentered unchanged")
print("      Match                               : CONCATCOS unchanged")
print("      Panel                               : exact sealed deterministic reconstruction")
print("      New retrieval law                   : NONE")
print("      New origin law                      : NONE")
print("      Pointer threshold                   : NONE")
print("      Gold span used for pointer          : NO")
print("      Gold span used for selection        : NO SELECTION PERFORMED")
print("      Gold span used for X-ray scoring    : YES")
print("      Decoded identity routing            : NO")
print("      Token-ID routing                    : NO")
print("      Head/layer/address scan             : NONE")
print("      Match-law scan                      : NONE")
print("      Training / LoRA / optimizer / DRA   : NONE")
print("      Learned router/scorer               : NONE")
print("      Query-time candidate-B forwards     : 0")
print("      TEST533 retroactively changed       : NO")

print("\n[11/11] FINAL...")
def compact(rows):
    miss=[r for r in rows if not r["inside"]]
    return {"n":len(rows),"misses":len(miss),"peak_hit":float(np.mean([r["inside"] for r in rows])),
            "exact_start":float(np.mean([r["exact_start"] for r in rows])),
            "mean_gold_mass":float(np.mean([r["gold_mass"] for r in rows])),
            "miss_best_gold_token_ranks":[r["best_gold_token_rank"] for r in miss],
            "miss_gold_max_window_ranks":[r["gold_max_window_rank"] for r in miss],
            "miss_gold_sum_window_ranks":[r["gold_sum_window_rank"] for r in miss],
            "miss_topk":{str(k):sum(r["topk"][k] for r in miss) for k in TOPKS}}

PR=compact(PRIMARY);CR=compact(CF)
RESULT={"test":TEST,
"parents":{"528":PARENT528,"529":PARENT529,"530":PARENT530,"531":PARENT531,"532":PARENT532,"533":PARENT533},
"lock_sha":LOCK_SHA,"primary":PR,"counterfactual":CR,"weight_sentinel":WEIGHT_OK,
"status":"POINTER_MISS_XRAY_COMPLETE"}
RESULT_SHA=hashlib.sha256(json.dumps(RESULT,sort_keys=True,separators=(",",":")).encode()).hexdigest()

print("\n"+"="*176)
print("TEST534 FINAL RESULT — AKBASCORE MAM · VARAN 1 · POINTER-MISS X-RAY")
print("="*176)
print("MODEL                              :",MODEL_ID,"· frozen")
print("PANEL                              : TEST531/532/533 deterministic reconstruction · N=1024")
print("POINTER                            : L28H00 · UNCHANGED")
print("ADDRESS                            : L00-V · UNCENTERED · 1024D · UNCHANGED")
print("MATCH                              : CONCATCOS · UNCHANGED")
print("NEW RETRIEVAL / ORIGIN LAW         : NONE")
print("-"*176)
print(f"PRIMARY POINTER PEAK HIT           : {PR['peak_hit']:.4f}")
print(f"PRIMARY EXACT GOLD-START           : {PR['exact_start']:.4f}")
print(f"PRIMARY POINTER MISSES             : {PR['misses']}/{N}")
print(f"PRIMARY MEAN GOLD-SPAN MASS        : {PR['mean_gold_mass']:.6f}")
for k in TOPKS:print(f"PRIMARY MISS GOLD IN TOP-{k:<2d}       : {PR['miss_topk'][str(k)]}/{PR['misses']}")
print("PRIMARY MISS TOKEN RANKS           :",PR["miss_best_gold_token_ranks"])
print("PRIMARY MISS MAX-WINDOW RANKS      :",PR["miss_gold_max_window_ranks"])
print("PRIMARY MISS SUM-WINDOW RANKS      :",PR["miss_gold_sum_window_ranks"])
print("-"*176)
print(f"CF POINTER PEAK HIT                : {CR['peak_hit']:.4f}")
print(f"CF EXACT GOLD-START                : {CR['exact_start']:.4f}")
print(f"CF POINTER MISSES                  : {CR['misses']}/{N}")
print(f"CF MEAN GOLD-SPAN MASS             : {CR['mean_gold_mass']:.6f}")
for k in TOPKS:print(f"CF MISS GOLD IN TOP-{k:<2d}            : {CR['miss_topk'][str(k)]}/{CR['misses']}")
print("CF MISS TOKEN RANKS                :",CR["miss_best_gold_token_ranks"])
print("CF MISS MAX-WINDOW RANKS           :",CR["miss_gold_max_window_ranks"])
print("CF MISS SUM-WINDOW RANKS           :",CR["miss_gold_sum_window_ranks"])
print("-"*176)
print("WEIGHT SENTINEL                    :","PASS" if WEIGHT_OK else "FAIL")
print("TRAINABLE TENSORS                  :",sum(int(p.requires_grad) for p in model.parameters()))
print("QUERY-TIME CANDIDATE-B FORWARDS    : 0")
print("TEST528 RESULT SHA                 :",PARENT528)
print("TEST529 RESULT SHA                 :",PARENT529)
print("TEST530 RESULT SHA                 :",PARENT530)
print("TEST531 RESULT SHA                 :",PARENT531)
print("TEST532 RESULT SHA                 :",PARENT532)
print("TEST533 RESULT SHA                 :",PARENT533)
print("TEST534 PRE-RESULT LOCK            :",LOCK_SHA)
print("TEST534 RESULT SHA                 :",RESULT_SHA)
print("TOTAL TEST TIME                    :",f"{time.perf_counter()-T0:.2f}s")
print("VERDICT                            : TEST534_POINTER_MISS_XRAY_COMPLETE")
print("NEXT                               : preserve TEST534 unchanged; choose next mechanism only from this frozen diagnostic evidence")
print("="*176)

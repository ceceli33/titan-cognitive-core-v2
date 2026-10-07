# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST536
# VARAN 1 — MULTI-PEAK DISCRIMINATION X-RAY
#
# SEALED CHAIN:
#   TEST528 -> native single-token external seal
#   TEST529 -> multi-token primitive boundary
#   TEST530 -> composition X-ray / CONCATCOS discovered
#   TEST531 -> untouched multi-token validation
#   TEST532 -> span-origin X-ray
#   TEST533 -> PEAK_CONTAIN resolution
#              PRIMARY R1 = 0.9932
#              CF R1      = 0.9912
#   TEST534 -> pointer-miss X-ray
#              PRIMARY pointer misses: gold token Top-2 = 9/9
#              CF pointer misses: gold token Top-8 = 9/9
#   TEST535 -> TOP8_JOINT
#              PRIMARY R1 = 0.9570
#              CF R1      = 0.9580
#              recovered PRIMARY PEAK failures = 7/7
#              recovered CF PEAK failures      = 8/9
#              broke PRIMARY PEAK-correct      = 44
#              broke CF PEAK-correct           = 42
#
# TEST536 PURPOSE:
#   Diagnose WHY Top-8 contains the missing identity signal but unrestricted
#   TOP8_JOINT introduces false A origins.
#
#   For each frozen Top-8 pointer peak measure, POST-HOC ONLY:
#     - pointer rank
#     - pointer weight
#     - pointer/top1 ratio
#     - local pointer mass
#     - immediate-neighbor mass
#     - best CONCATCOS score obtainable from width-L origins containing that peak
#     - winning B owner for that peak
#     - whether peak lies inside the gold A identity span (diagnostic only)
#     - whether that peak can recover the gold B owner (diagnostic only)
#
#   Special strata:
#     A) TEST533 PEAK_CONTAIN failures
#     B) TEST535 TOP8-recovered cases
#     C) TEST535 regressions: PEAK correct -> TOP8 wrong
#     D) ordinary PEAK-correct / TOP8-correct cases
#
# IMPORTANT:
#   TEST536 DOES NOT CREATE OR TEST A NEW RETRIEVAL LAW.
#   No feature is used for selection.
#   No threshold is selected.
#   No K scan is performed.
#   TOPK=8 remains frozen from TEST535.
#   Gold information is used only AFTER all candidate measurements are produced.
#
# FROZEN / UNCHANGED:
#   MODEL   = mistralai/Mistral-7B-Instruct-v0.3
#   POINTER = L28H00
#   ADDRESS = L00-V UNCENTERED
#   MATCH   = position-preserving CONCATCOS
#   PANEL   = exact TEST531/532/533/534/535 deterministic construction
#   IDENTITY LENGTHS = 2/3/4
#
# NOTE:
#   Identity length L remains protocol-known.
#   Matching remains length-conditioned.
#   TEST536 is diagnostic and is NOT the final Varan 1 seal.

import os,sys,subprocess,importlib.util,random,time,hashlib,json,gc,re,math
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

TEST="536";SEED=531531;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";N=1024
DEVICE=torch.device("cuda");FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
PTR_L=28;PTR_H=0;BL=0;BK="V";CENTER=False;LENS=(2,3,4);PTR_TOPK=8
PARENT528="9235ed38c944f41a9ede0426d3aabaa12b5b20646f30e2be921e618f9471d3c2"
PARENT529="27ba703872f0d8a6ed870a6ba041d0cd0703d2eca467deac97186a90497795a0"
PARENT530="289247c88dfd4358cbe5ffb8499891546b27acc6912caf525a155ae81338db7b"
PARENT531="20257d7d613a49e55b03603946f07ec493009924f40ac26ab74cb817e1466528"
PARENT532="9e37a473a322a19f7b3862a25162835ab6396998dc4fc416bc90bc9b718df188"
PARENT533="5239fb585b87a22f5da778c61a1011379b216a25bf7d05dc5d01952cbb22a5c6"
PARENT534="b25128d17176790b2a8694c8ec9e7db5eaad469ff02a87a46876b2ffd45a9bb1"
PARENT535="fc407885133d48f84a952249b15e884b201749136f6b16a49e7db11b0e9e5087"

os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*176)
print("TEST536 — AKBASCORE MAM · VARAN 1 · MULTI-PEAK DISCRIMINATION X-RAY")
print("TEST535 FOLLOW-UP · FROZEN TOP-8 POINTER CANDIDATES · FEATURE ANATOMY ONLY · NO NEW RETRIEVAL LAW")
print("="*176);T0=time.perf_counter()

print("\n[1/12] Loading frozen Mistral...")
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
print("      MATCH   : CONCATCOS · LOCKED FROM TEST530")
print("      PANEL   : exact TEST531/532/533/534/535 deterministic construction")
print(f"      POINTER CANDIDATES : TOP-{PTR_TOPK} · FROZEN")
print("      MODE    : X-RAY ONLY · NO NEW RETRIEVAL / ORIGIN LAW")
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

print("\n[2/12] Reconstructing exact sealed panel...")
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
"parents":{"528":PARENT528,"529":PARENT529,"530":PARENT530,"531":PARENT531,"532":PARENT532,"533":PARENT533,"534":PARENT534,"535":PARENT535},
"model":MODEL_ID,"seed":SEED,"N":N,"status":"VARAN1_MULTI_PEAK_DISCRIMINATION_XRAY",
"pointer":{"layer":PTR_L,"head":PTR_H,"topk":PTR_TOPK,"topk_frozen_from":"TEST535"},
"address":{"layer":BL,"kind":BK,"center":CENTER,"dimension":KVD},
"identity":{"lengths":[2,3,4],"length_known":True,"composition":"ordered native-token sequence"},
"match":{"name":"CONCATCOS","formula":"mean position-wise cosine","locked_from":"TEST530"},
"measurements":["POINTER_RANK","POINTER_WEIGHT","POINTER_TOP1_RATIO","LOCAL_MASS_R1","LOCAL_MASS_R2",
                "BEST_PEAK_CONTAIN_CONCATCOS","PEAK_OWNER","GOLD_SPAN_MEMBERSHIP","GOLD_OWNER_RECOVERY"],
"selection":"NONE_XRAY_ONLY",
"gold_usage":"POSTHOC_DIAGNOSTIC_ONLY",
"forbidden":["NEW_RETRIEVAL_LAW","NEW_ORIGIN_LAW","TOPK_SCAN","THRESHOLD_SCAN","HEAD_SCAN","LAYER_SCAN",
"ADDRESS_SCAN","CENTER_SCAN","MATCH_SCAN","TOKEN_ID_ROUTING","DECODED_ID_ROUTING","GOLD_A_START_SELECTION",
"GOLD_B_WINDOW_SELECTION","QUERY_TIME_B_FORWARD","TRAINING","LORA","DRA","LEARNED_ROUTER","POSTHOC_SELECTION_RULE"]}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      TEST536 LOCK:",LOCK_SHA)

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

print("\n[3/12] OFFLINE FORGE — exact sealed A/B construction...")
A_RAW=[];A_PACK=[];GOLD_SPANS=[];B_BANK={L:{"vec":[],"own":[]} for L in LENS}
for i,it in enumerate(ITEMS):
    ar=forge(it["A"]);br=forge(it["B"]);asp=exact_span(it["A"],it["gold"])
    if asp is None or len(asp)!=it["L"]:raise RuntimeError(f"A span mismatch item {i+1}")
    A_RAW.append(ar);A_PACK.append(install(ar));GOLD_SPANS.append(asp)
    x=br[BL][1][:,1:,:].float().cpu().permute(1,0,2).reshape(-1,KVD).contiguous();L=it["L"]
    if len(x)<L:raise RuntimeError(f"B memory too short item {i+1}")
    wins=torch.stack([x[j:j+L] for j in range(len(x)-L+1)])
    B_BANK[L]["vec"].append(wins);B_BANK[L]["own"].extend([i]*(len(x)-L+1))
    del br,x,wins
    if (i+1)%64==0:print(f"      forged {i+1:4d}/{N}")
gc.collect();torch.cuda.empty_cache()
print("      A memories:",len(A_RAW))
print("      B memories:",N)
print("      Query-time candidate-B forwards: 0")

print("\n[4/12] Building locked CONCATCOS banks...")
for L in LENS:
    if not B_BANK[L]["vec"]:raise RuntimeError(f"Empty B bank L={L}")
    X=torch.cat(B_BANK[L]["vec"],0);X=X/X.norm(dim=2,keepdim=True).clamp_min(1e-8)
    B_BANK[L]["vec"]=X.to(DEVICE,dtype=torch.float16)
    B_BANK[L]["own"]=torch.tensor(B_BANK[L]["own"],device=DEVICE,dtype=torch.long)
    print(f"      L={L} windows: {len(B_BANK[L]['own'])} | shape={tuple(B_BANK[L]['vec'].shape)}")
    del X
print("      Match law unchanged: CONCATCOS")
print("      Gold B position used: NO")
gc.collect();torch.cuda.empty_cache()

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

def argmax_start(w):return int(torch.argmax(w))

def pointer_topk(w,k):
    x=w.clone()
    if len(x):x[0]=-1
    k=min(k,max(0,len(x)-1))
    return torch.topk(x,k=k,largest=True,sorted=True).indices.tolist() if k else []

def a_sequence(raw,start,L):
    if start is None:return None
    x=raw[BL][1].float().cpu().permute(1,0,2).reshape(-1,KVD)
    if start<1 or start+L>len(x):return None
    return x[start:start+L]

def containing_starts(raw,peak,L):
    T=raw[BL][1].shape[1]
    lo=max(1,peak-L+1);hi=min(peak,T-L)
    return list(range(lo,hi+1)) if hi>=lo else []

def multi_peak_starts(raw,w,L,k):
    peaks=pointer_topk(w,k);starts=set()
    for p in peaks:starts.update(containing_starts(raw,p,L))
    return peaks,sorted(starts)

@torch.inference_mode()
def concat_scores(Aseq,L):
    A=Aseq.to(DEVICE);A=A/A.norm(dim=1,keepdim=True).clamp_min(1e-8)
    B=B_BANK[L]["vec"];own=B_BANK[L]["own"]
    sim=(B.float()*A.float().unsqueeze(0)).sum(dim=2).mean(dim=1)
    out=torch.full((N,),-torch.inf,device=DEVICE)
    out.scatter_reduce_(0,own,sim,reduce="amax",include_self=True)
    return out.cpu()

def invalid_scores():return torch.full((N,),-torch.inf)

def rank(sc,g):
    if not torch.isfinite(sc).any():return N,-1,[]
    order=torch.argsort(sc,descending=True).tolist()
    return order.index(g)+1,order[0],order

def joint_scores(raw,starts,L):
    if not starts:return invalid_scores(),None,float("-inf")
    best=invalid_scores();best_start=torch.full((N,),-1,dtype=torch.long)
    for s in starts:
        seq=a_sequence(raw,s,L)
        if seq is None:continue
        sc=concat_scores(seq,L);mask=sc>best
        best[mask]=sc[mask];best_start[mask]=s
    if not torch.isfinite(best).any():return best,None,float("-inf")
    owner=int(torch.argmax(best));score=float(best[owner])
    return best,int(best_start[owner]),score

def peak_contain_scores(raw,w,L):
    p=argmax_start(w)
    sc,s,_=joint_scores(raw,containing_starts(raw,p,L),L)
    return sc,s

def top8_joint_scores(raw,w,L):
    peaks,starts=multi_peak_starts(raw,w,L,PTR_TOPK)
    sc,s,_=joint_scores(raw,starts,L)
    return sc,s,peaks,starts

def local_mass(w,p,r):
    a=max(1,p-r);b=min(len(w),p+r+1)
    return float(w[a:b].sum())

def peak_xray(raw,w,L,gold_owner,gold_span):
    peaks=pointer_topk(w,PTR_TOPK);top1=float(w[peaks[0]]) if peaks else 0.0;rows=[]
    for pr,p in enumerate(peaks,1):
        starts=containing_starts(raw,p,L)
        sc,s,score=joint_scores(raw,starts,L)
        rr,owner,_=rank(sc,gold_owner)
        rows.append({
            "rank":pr,"pos":int(p),"weight":float(w[p]),
            "top1_ratio":float(w[p]/top1) if top1>0 else 0.0,
            "local1":local_mass(w,p,1),"local2":local_mass(w,p,2),
            "left":float(w[p-1]) if p-1>=1 else 0.0,
            "right":float(w[p+1]) if p+1<len(w) else 0.0,
            "inside_gold":int(p in gold_span),
            "contains_gold_start":int(gold_span[0] in starts),
            "best_start":int(s) if s is not None else -1,
            "best_owner":int(owner) if owner>=0 else -1,
            "gold_rank":int(rr),
            "recovers_gold":int(owner==gold_owner),
            "best_score":float(score)})
    return rows

def metrics(r):
    a=np.asarray(r,float)
    return float(np.mean(a==1)),float(np.mean(a<=5)),float(np.mean(a<=16)),float(np.mean(1/a))

def meanv(rows,key):
    v=[float(x[key]) for x in rows]
    return float(np.mean(v)) if v else float("nan")

def medv(rows,key):
    v=[float(x[key]) for x in rows]
    return float(np.median(v)) if v else float("nan")

def summarize_candidates(name,rows):
    print(f"      {name}: n={len(rows)}")
    if not rows:return
    print(f"        pointer rank mean/median     : {meanv(rows,'rank'):.3f}/{medv(rows,'rank'):.3f}")
    print(f"        pointer weight mean/median   : {meanv(rows,'weight'):.6f}/{medv(rows,'weight'):.6f}")
    print(f"        top1 ratio mean/median       : {meanv(rows,'top1_ratio'):.6f}/{medv(rows,'top1_ratio'):.6f}")
    print(f"        local±1 mass mean/median     : {meanv(rows,'local1'):.6f}/{medv(rows,'local1'):.6f}")
    print(f"        local±2 mass mean/median     : {meanv(rows,'local2'):.6f}/{medv(rows,'local2'):.6f}")
    print(f"        left weight mean             : {meanv(rows,'left'):.6f}")
    print(f"        right weight mean            : {meanv(rows,'right'):.6f}")
    print(f"        best CONCATCOS mean/median   : {meanv(rows,'best_score'):.6f}/{medv(rows,'best_score'):.6f}")
    print(f"        recovers gold owner          : {sum(x['recovers_gold'] for x in rows)}/{len(rows)}")

def case_candidate_rows(allrows,case_map,case_name,inside=None,recover=None):
    out=[]
    for x in allrows:
        if case_map[x["item"]-1]!=case_name:continue
        if inside is not None and x["inside_gold"]!=inside:continue
        if recover is not None and x["recovers_gold"]!=recover:continue
        out.append(x)
    return out

print("\n[5/12] PRIMARY — reproduce TEST533/535 arms + collect frozen Top-8 X-ray...")
RPEAK=[];RNEW=[];PRIMARY_X=[];PCASE=[]
for i,it in enumerate(ITEMS):
    L=it["L"];sp=GOLD_SPANS[i];w=frozen_pointer(A_PACK[i],qA(it["entity"]))
    scp,speak=peak_contain_scores(A_RAW[i],w,L);rp,selp,_=rank(scp,i)
    scn,snew,peaks,starts=top8_joint_scores(A_RAW[i],w,L);rn,seln,_=rank(scn,i)
    xr=peak_xray(A_RAW[i],w,L,i,sp)
    RPEAK.append(rp);RNEW.append(rn)
    PRIMARY_X.extend([dict(x,item=i+1,L=L) for x in xr])
    if rp!=1 and rn==1:case="RECOVERED"
    elif rp==1 and rn!=1:case="REGRESSION"
    elif rp!=1 and rn!=1:case="BOTH_FAIL"
    else:case="STABLE"
    PCASE.append(case)
    if i<8 or case!="STABLE" or (i+1)%64==0:
        goldrows=[x for x in xr if x["inside_gold"]]
        gr=[x["rank"] for x in goldrows]
        print(f"      [{i+1:04d}/{N}] L={L} PEAK={rp:4d} TOP8={rn:4d} case={case:10s} "
              f"goldPeakRanks={gr} peakOrigin={speak if speak is not None else -1:3d} "
              f"top8Origin={snew if snew is not None else -1:3d}")
MP=metrics(RPEAK);MN=metrics(RNEW)
print(f"      PEAK_CONTAIN reproduction : {MP[0]:.4f}/{MP[1]:.4f}/{MP[2]:.4f}/{MP[3]:.6f}")
print(f"      TOP8_JOINT reproduction   : {MN[0]:.4f}/{MN[1]:.4f}/{MN[2]:.4f}/{MN[3]:.6f}")
print("      Case counts               :",{x:PCASE.count(x) for x in ("RECOVERED","REGRESSION","BOTH_FAIL","STABLE")})

print("\n[6/12] PRIMARY MULTI-PEAK FEATURE ANATOMY...")
for case in ("RECOVERED","REGRESSION","BOTH_FAIL","STABLE"):
    items=PCASE.count(case)
    print(f"\n      [{case}] items={items}")
    rows=case_candidate_rows(PRIMARY_X,PCASE,case)
    gold=case_candidate_rows(PRIMARY_X,PCASE,case,inside=1)
    nongold=case_candidate_rows(PRIMARY_X,PCASE,case,inside=0)
    recovering=case_candidate_rows(PRIMARY_X,PCASE,case,recover=1)
    summarize_candidates("ALL TOP8 CANDIDATES",rows)
    summarize_candidates("GOLD-SPAN PEAKS",gold)
    summarize_candidates("NON-GOLD PEAKS",nongold)
    summarize_candidates("PEAKS RECOVERING GOLD OWNER",recovering)

print("\n      PRIMARY item-level discrimination details...")
for i,case in enumerate(PCASE):
    if case not in ("RECOVERED","REGRESSION","BOTH_FAIL"):continue
    rows=[x for x in PRIMARY_X if x["item"]==i+1]
    print(f"      ITEM {i+1:04d} L={ITEMS[i]['L']} case={case} PEAKrank={RPEAK[i]} TOP8rank={RNEW[i]}")
    for x in rows:
        flag="G" if x["inside_gold"] else "-"
        rec="R" if x["recovers_gold"] else "-"
        print(f"        k={x['rank']} pos={x['pos']:3d} {flag}{rec} w={x['weight']:.6f} "
              f"ratio={x['top1_ratio']:.4f} l1={x['local1']:.6f} l2={x['local2']:.6f} "
              f"L/R={x['left']:.6f}/{x['right']:.6f} start={x['best_start']:3d} "
              f"owner={x['best_owner']+1 if x['best_owner']>=0 else 0:04d} "
              f"goldRank={x['gold_rank']:4d} score={x['best_score']:.6f}")

print("\n[7/12] COUNTERFACTUAL — reproduce TEST533/535 arms + collect frozen Top-8 X-ray...")
RCF_PEAK=[];RCF_NEW=[];CF_X=[];CFCASE=[]
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
    scp,speak=peak_contain_scores(raw,w,L);rp,selp,_=rank(scp,j)
    scn,snew,peaks,starts=top8_joint_scores(raw,w,L);rn,seln,_=rank(scn,j)
    xr=peak_xray(raw,w,L,j,sp)
    RCF_PEAK.append(rp);RCF_NEW.append(rn)
    CF_X.extend([dict(x,item=i+1,target=j+1,L=L) for x in xr])
    if rp!=1 and rn==1:case="RECOVERED"
    elif rp==1 and rn!=1:case="REGRESSION"
    elif rp!=1 and rn!=1:case="BOTH_FAIL"
    else:case="STABLE"
    CFCASE.append(case)
    if i<5 or case!="STABLE" or (i+1)%64==0:
        goldrows=[x for x in xr if x["inside_gold"]]
        gr=[x["rank"] for x in goldrows]
        print(f"      [{i+1:04d}/{N}] target={j+1:04d} L={L} PEAK={rp:4d} TOP8={rn:4d} "
              f"case={case:10s} goldPeakRanks={gr} "
              f"peakOrigin={speak if speak is not None else -1:3d} "
              f"top8Origin={snew if snew is not None else -1:3d}")
    del raw,pack,w,scp,scn
MCP=metrics(RCF_PEAK);MCN=metrics(RCF_NEW)
print(f"      CF PEAK_CONTAIN reproduction : {MCP[0]:.4f}/{MCP[1]:.4f}/{MCP[2]:.4f}/{MCP[3]:.6f}")
print(f"      CF TOP8_JOINT reproduction   : {MCN[0]:.4f}/{MCN[1]:.4f}/{MCN[2]:.4f}/{MCN[3]:.6f}")
print("      CF case counts               :",{x:CFCASE.count(x) for x in ("RECOVERED","REGRESSION","BOTH_FAIL","STABLE")})

print("\n[8/12] COUNTERFACTUAL MULTI-PEAK FEATURE ANATOMY...")
for case in ("RECOVERED","REGRESSION","BOTH_FAIL","STABLE"):
    items=CFCASE.count(case)
    print(f"\n      [{case}] items={items}")
    rows=case_candidate_rows(CF_X,CFCASE,case)
    gold=case_candidate_rows(CF_X,CFCASE,case,inside=1)
    nongold=case_candidate_rows(CF_X,CFCASE,case,inside=0)
    recovering=case_candidate_rows(CF_X,CFCASE,case,recover=1)
    summarize_candidates("ALL TOP8 CANDIDATES",rows)
    summarize_candidates("GOLD-SPAN PEAKS",gold)
    summarize_candidates("NON-GOLD PEAKS",nongold)
    summarize_candidates("PEAKS RECOVERING GOLD OWNER",recovering)

print("\n      COUNTERFACTUAL item-level discrimination details...")
for i,case in enumerate(CFCASE):
    if case not in ("RECOVERED","REGRESSION","BOTH_FAIL"):continue
    rows=[x for x in CF_X if x["item"]==i+1]
    target=(i+353)%N
    print(f"      ITEM {i+1:04d} target={target+1:04d} L={TLEN[target]} case={case} "
          f"PEAKrank={RCF_PEAK[i]} TOP8rank={RCF_NEW[i]}")
    for x in rows:
        flag="G" if x["inside_gold"] else "-"
        rec="R" if x["recovers_gold"] else "-"
        print(f"        k={x['rank']} pos={x['pos']:3d} {flag}{rec} w={x['weight']:.6f} "
              f"ratio={x['top1_ratio']:.4f} l1={x['local1']:.6f} l2={x['local2']:.6f} "
              f"L/R={x['left']:.6f}/{x['right']:.6f} start={x['best_start']:3d} "
              f"owner={x['best_owner']+1 if x['best_owner']>=0 else 0:04d} "
              f"goldRank={x['gold_rank']:4d} score={x['best_score']:.6f}")

print("\n[9/12] CROSS-STRATUM DIAGNOSTIC SUMMARY...")
def item_level_stats(X,cases,name):
    print(f"      {name}")
    for case in ("RECOVERED","REGRESSION","BOTH_FAIL","STABLE"):
        ids=[i+1 for i,c in enumerate(cases) if c==case]
        rows=[x for x in X if cases[x["item"]-1]==case]
        gold=[x for x in rows if x["inside_gold"]]
        nongold=[x for x in rows if not x["inside_gold"]]
        rec=[x for x in rows if x["recovers_gold"]]
        gold_items=len(set(x["item"] for x in gold))
        rec_items=len(set(x["item"] for x in rec))
        print(f"        {case:10s}: items={len(ids):4d} gold-peak-items={gold_items:4d} "
              f"gold-recovery-items={rec_items:4d} candidates={len(rows):5d} "
              f"gold={len(gold):4d} nongold={len(nongold):5d}")
item_level_stats(PRIMARY_X,PCASE,"PRIMARY")
item_level_stats(CF_X,CFCASE,"COUNTERFACTUAL")

def best_gold_vs_nongold(X,cases,case):
    ids=[i+1 for i,c in enumerate(cases) if c==case];diffs=[];wins=ties=losses=0
    for iid in ids:
        rows=[x for x in X if x["item"]==iid]
        g=[x["best_score"] for x in rows if x["inside_gold"]]
        ng=[x["best_score"] for x in rows if not x["inside_gold"]]
        if not g or not ng:continue
        d=max(g)-max(ng);diffs.append(d)
        if d>1e-8:wins+=1
        elif d<-1e-8:losses+=1
        else:ties+=1
    return diffs,wins,ties,losses

for label,X,cases in [("PRIMARY",PRIMARY_X,PCASE),("CF",CF_X,CFCASE)]:
    print(f"\n      {label} best gold-peak vs best non-gold-peak CONCATCOS:")
    for case in ("RECOVERED","REGRESSION","BOTH_FAIL","STABLE"):
        d,w,t,l=best_gold_vs_nongold(X,cases,case)
        if d:
            print(f"        {case:10s}: n={len(d):4d} meanΔ={np.mean(d):+.6f} medianΔ={np.median(d):+.6f} "
                  f"gold>non={w} tie={t} gold<non={l}")
        else:
            print(f"        {case:10s}: n=0")

print("\n[10/12] REPRODUCTION / DIAGNOSTIC GATES...")
GATES={
"PRIMARY_PEAK_REPRO":abs(MP[0]-.9932)<=.005,
"PRIMARY_TOP8_REPRO":abs(MN[0]-.9570)<=.005,
"CF_PEAK_REPRO":abs(MCP[0]-.9912)<=.005,
"CF_TOP8_REPRO":abs(MCN[0]-.9580)<=.005,
"PRIMARY_RECOVERED_REPRO":PCASE.count("RECOVERED")==7,
"PRIMARY_REGRESSION_REPRO":PCASE.count("REGRESSION")==44,
"CF_RECOVERED_REPRO":CFCASE.count("RECOVERED")==8,
"CF_REGRESSION_REPRO":CFCASE.count("REGRESSION")==42}
for k,v in GATES.items():print(f"      {k:32s}: {'PASS' if v else 'FAIL'}")

print("\n[11/12] PROTOCOL AUDIT...")
S1=sentinel();WEIGHT_OK=S0==S1;GATES["WEIGHT_SENTINEL"]=WEIGHT_OK
print("      Weight sentinel                     :","PASS" if WEIGHT_OK else "FAIL")
print("      Trainable tensors                   :",sum(int(p.requires_grad) for p in model.parameters()))
print("      Pointer                             : L28H00 unchanged")
print("      Address                             : L00-V uncentered unchanged")
print("      Match                               : CONCATCOS unchanged")
print("      Panel                               : exact sealed deterministic reconstruction")
print(f"      Pointer candidate K                 : {PTR_TOPK} · FROZEN FROM TEST535")
print("      K scan                              : NONE")
print("      Baseline retrieval under diagnosis : PEAK_CONTAIN + TOP8_JOINT reproductions only")
print("      New retrieval law                   : NONE")
print("      New origin law                      : NONE")
print("      Feature-based selection             : NONE")
print("      Threshold selection                 : NONE")
print("      Identity length L                   : PROTOCOL-KNOWN")
print("      Cross-length competition           : NO")
print("      Gold span used for selection       : NO")
print("      Gold span usage                    : POST-HOC DIAGNOSTIC ONLY")
print("      Decoded identity routing           : NO")
print("      Token-ID routing                   : NO")
print("      Gold B position used               : NO")
print("      Head/layer/address discovery       : NONE")
print("      Match-law scan                     : NONE")
print("      Training / LoRA / optimizer / DRA  : NONE")
print("      Learned router/scorer              : NONE")
print("      Query-time candidate-B forwards    : 0")
print("      TEST535 retroactively changed      : NO")

print("\n[12/12] FINAL...")
VERDICT="TEST536_MULTI_PEAK_DISCRIMINATION_XRAY_COMPLETE"
NEXT="preserve TEST536 unchanged; derive any candidate discrimination law only in a new predeclared test"
RESULT={
"test":TEST,
"parents":{"528":PARENT528,"529":PARENT529,"530":PARENT530,"531":PARENT531,"532":PARENT532,
           "533":PARENT533,"534":PARENT534,"535":PARENT535},
"lock_sha":LOCK_SHA,
"pointer_topk":PTR_TOPK,
"primary":{
"peak":{"r1":MP[0],"r5":MP[1],"r16":MP[2],"mrr":MP[3]},
"top8":{"r1":MN[0],"r5":MN[1],"r16":MN[2],"mrr":MN[3]},
"cases":{x:PCASE.count(x) for x in ("RECOVERED","REGRESSION","BOTH_FAIL","STABLE")}},
"counterfactual":{
"peak":{"r1":MCP[0],"r5":MCP[1],"r16":MCP[2],"mrr":MCP[3]},
"top8":{"r1":MCN[0],"r5":MCN[1],"r16":MCN[2],"mrr":MCN[3]},
"cases":{x:CFCASE.count(x) for x in ("RECOVERED","REGRESSION","BOTH_FAIL","STABLE")}},
"gates":GATES,
"verdict":VERDICT}
RESULT_SHA=hashlib.sha256(json.dumps(RESULT,sort_keys=True,separators=(",",":")).encode()).hexdigest()

print("\n"+"="*176)
print("TEST536 FINAL RESULT — AKBASCORE MAM · VARAN 1 · MULTI-PEAK DISCRIMINATION X-RAY")
print("="*176)
print("MODEL                              :",MODEL_ID,"· frozen")
print("PANEL                              : TEST531/532/533/534/535 deterministic reconstruction · N=1024")
print("POINTER                            : L28H00 · UNCHANGED")
print("ADDRESS                            : L00-V · UNCENTERED · 1024D · UNCHANGED")
print("MATCH                              : CONCATCOS · UNCHANGED")
print("POINTER CANDIDATES                 : TOP-8 · FROZEN FROM TEST535")
print("MODE                               : X-RAY ONLY · NO NEW RETRIEVAL LAW")
print("-"*176)
print(f"PRIMARY PEAK_CONTAIN               : R1={MP[0]:.4f} R5={MP[1]:.4f} R16={MP[2]:.4f} MRR={MP[3]:.6f}")
print(f"PRIMARY TOP8_JOINT                 : R1={MN[0]:.4f} R5={MN[1]:.4f} R16={MN[2]:.4f} MRR={MN[3]:.6f}")
print(f"PRIMARY RECOVERED                  : {PCASE.count('RECOVERED')}")
print(f"PRIMARY REGRESSION                 : {PCASE.count('REGRESSION')}")
print(f"PRIMARY BOTH_FAIL                  : {PCASE.count('BOTH_FAIL')}")
print(f"PRIMARY STABLE                     : {PCASE.count('STABLE')}")
print(f"CF PEAK_CONTAIN                    : R1={MCP[0]:.4f} R5={MCP[1]:.4f} R16={MCP[2]:.4f} MRR={MCP[3]:.6f}")
print(f"CF TOP8_JOINT                      : R1={MCN[0]:.4f} R5={MCN[1]:.4f} R16={MCN[2]:.4f} MRR={MCN[3]:.6f}")
print(f"CF RECOVERED                       : {CFCASE.count('RECOVERED')}")
print(f"CF REGRESSION                      : {CFCASE.count('REGRESSION')}")
print(f"CF BOTH_FAIL                       : {CFCASE.count('BOTH_FAIL')}")
print(f"CF STABLE                          : {CFCASE.count('STABLE')}")
print("-"*176)
for k,v in GATES.items():print(f"{k:35s}: {'PASS' if v else 'FAIL'}")
print("-"*176)
print("WEIGHT SENTINEL                    :","PASS" if WEIGHT_OK else "FAIL")
print("TRAINABLE TENSORS                  :",sum(int(p.requires_grad) for p in model.parameters()))
print("QUERY-TIME CANDIDATE-B FORWARDS    : 0")
print("KNOWN IDENTITY LENGTH              : YES")
print("CROSS-LENGTH GLOBAL COMPETITION    : NO")
print("NEW RETRIEVAL LAW                  : NONE")
print("FEATURE-BASED SELECTION            : NONE")
print("TEST528 RESULT SHA                 :",PARENT528)
print("TEST529 RESULT SHA                 :",PARENT529)
print("TEST530 RESULT SHA                 :",PARENT530)
print("TEST531 RESULT SHA                 :",PARENT531)
print("TEST532 RESULT SHA                 :",PARENT532)
print("TEST533 RESULT SHA                 :",PARENT533)
print("TEST534 RESULT SHA                 :",PARENT534)
print("TEST535 RESULT SHA                 :",PARENT535)
print("TEST536 PRE-RESULT LOCK            :",LOCK_SHA)
print("TEST536 RESULT SHA                 :",RESULT_SHA)
print("TOTAL TEST TIME                    :",f"{time.perf_counter()-T0:.2f}s")
print("VERDICT                            :",VERDICT)
print("NEXT                               :",NEXT)
print("="*176)

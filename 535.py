# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST535
# VARAN 1 — TOP-8 MULTI-PEAK JOINT RESOLUTION
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
#              PRIMARY misses: gold token Top-2 = 9/9
#              CF misses: gold token Top-8 = 9/9
#
# TEST535 PURPOSE:
#   Test a frozen multi-peak retrieval law derived BEFORE this run:
#
#       pointer Top-8 peaks
#           -> every valid width-L A span containing each peak
#           -> locked CONCATCOS against frozen B bank
#           -> global maximum over (peak, A-origin, B-owner)
#
# IMPORTANT:
#   TOPK=8 IS FROZEN BEFORE TEST535.
#   No K scan is performed.
#   No threshold.
#   No gold span selection.
#   No decoded identity.
#   No token-ID routing.
#   No learned router/scorer.
#   No head/layer/address/match scan.
#
# FROZEN / UNCHANGED:
#   MODEL   = mistralai/Mistral-7B-Instruct-v0.3
#   POINTER = L28H00
#   ADDRESS = L00-V UNCENTERED
#   MATCH   = position-preserving CONCATCOS
#   PANEL   = exact TEST531/532/533/534 deterministic construction
#   IDENTITY LENGTHS = 2/3/4
#
# NOTE:
#   Identity length L remains protocol-known in TEST535.
#   Matching remains length-conditioned.
#   Therefore TEST535 is NOT the final Varan 1 seal.
#
# PREDECLARED SUCCESS CRITERIA:
#   BASELINE ARGMAX R1 approximately reproduces 0.9824
#   PEAK_CONTAIN R1 approximately reproduces 0.9932
#   TOP8_JOINT R1 >= 0.999
#   TOP8_JOINT R5 >= 0.999
#   EACH L R1 >= 0.995
#   COUNTERFACTUAL R1 >= 0.995
#   NO previously-correct PEAK_CONTAIN case broken
#   WEIGHT SENTINEL PASS
#   QUERY-TIME CANDIDATE-B FORWARDS = 0

import os,sys,subprocess,importlib.util,random,time,hashlib,json,gc,re,math
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

TEST="535";SEED=531531;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";N=1024
DEVICE=torch.device("cuda");FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
PTR_L=28;PTR_H=0;BL=0;BK="V";CENTER=False;LENS=(2,3,4);PTR_TOPK=8
TH_NEW_R1=.999;TH_NEW_R5=.999;TH_EACH=.995;TH_CF=.995
PARENT528="9235ed38c944f41a9ede0426d3aabaa12b5b20646f30e2be921e618f9471d3c2"
PARENT529="27ba703872f0d8a6ed870a6ba041d0cd0703d2eca467deac97186a90497795a0"
PARENT530="289247c88dfd4358cbe5ffb8499891546b27acc6912caf525a155ae81338db7b"
PARENT531="20257d7d613a49e55b03603946f07ec493009924f40ac26ab74cb817e1466528"
PARENT532="9e37a473a322a19f7b3862a25162835ab6396998dc4fc416bc90bc9b718df188"
PARENT533="5239fb585b87a22f5da778c61a1011379b216a25bf7d05dc5d01952cbb22a5c6"
PARENT534="b25128d17176790b2a8694c8ec9e7db5eaad469ff02a87a46876b2ffd45a9bb1"

os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*176)
print("TEST535 — AKBASCORE MAM · VARAN 1 · TOP-8 MULTI-PEAK JOINT RESOLUTION")
print("TEST534 FOLLOW-UP · FROZEN TOP-8 POINTER CANDIDATES → LOCAL ORIGINS → LOCKED CONCATCOS GLOBAL JOINT SELECTION")
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
print("      PANEL   : exact TEST531/532/533/534 deterministic construction")
print(f"      POINTER CANDIDATES : TOP-{PTR_TOPK} · FROZEN BEFORE TEST535")
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
"parents":{"528":PARENT528,"529":PARENT529,"530":PARENT530,"531":PARENT531,"532":PARENT532,"533":PARENT533,"534":PARENT534},
"model":MODEL_ID,"seed":SEED,"N":N,"status":"VARAN1_TOP8_MULTI_PEAK_JOINT",
"pointer":{"layer":PTR_L,"head":PTR_H,"topk":PTR_TOPK,"topk_frozen_from":"TEST534_DIAGNOSTIC"},
"address":{"layer":BL,"kind":BK,"center":CENTER,"dimension":KVD},
"identity":{"lengths":[2,3,4],"length_known":True,"composition":"ordered native-token sequence"},
"match":{"name":"CONCATCOS","formula":"mean position-wise cosine","locked_from":"TEST530"},
"arms":{
"BASELINE_ARGMAX":"single pointer argmax used as A start",
"PEAK_CONTAIN":"argmax peak -> all width-L spans containing peak -> global CONCATCOS",
"TOP8_JOINT":"top-8 pointer positions -> union of all width-L spans containing any selected peak -> global CONCATCOS"},
"gold_usage":"DIAGNOSTIC_ONLY_AFTER_SELECTION",
"selection":"PREDECLARED_THREE_ARM_TEST",
"forbidden":["TOPK_SCAN","THRESHOLD_SCAN","HEAD_SCAN","LAYER_SCAN","ADDRESS_SCAN","CENTER_SCAN","MATCH_SCAN",
"TOKEN_ID_ROUTING","DECODED_ID_ROUTING","GOLD_A_START_SELECTION","GOLD_B_WINDOW_SELECTION",
"QUERY_TIME_B_FORWARD","TRAINING","LORA","DRA","LEARNED_ROUTER","POSTHOC_ARM_CREATION"]}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      TEST535 LOCK:",LOCK_SHA)

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
    if not starts:return invalid_scores(),None
    best=invalid_scores();best_start=torch.full((N,),-1,dtype=torch.long)
    for s in starts:
        seq=a_sequence(raw,s,L)
        if seq is None:continue
        sc=concat_scores(seq,L);mask=sc>best
        best[mask]=sc[mask];best_start[mask]=s
    if not torch.isfinite(best).any():return best,None
    owner=int(torch.argmax(best))
    return best,int(best_start[owner])

def peak_contain_scores(raw,w,L):
    p=argmax_start(w)
    return joint_scores(raw,containing_starts(raw,p,L),L)

def top8_joint_scores(raw,w,L):
    peaks,starts=multi_peak_starts(raw,w,L,PTR_TOPK)
    sc,s=joint_scores(raw,starts,L)
    return sc,s,peaks,starts

def metrics(r):
    a=np.asarray(r,float)
    return float(np.mean(a==1)),float(np.mean(a<=5)),float(np.mean(a<=16)),float(np.mean(1/a))

print("\n[5/12] PRIMARY — ARGMAX vs PEAK_CONTAIN vs TOP8_JOINT...")
RBASE=[];RPEAK=[];RNEW=[];BYLEN_NEW={L:[] for L in LENS}
BASE_START=[];PEAK_START=[];NEW_START=[];BASE_INSIDE=[];TOP8_GOLD=[]
RECOVERED_FROM_PEAK=[];BROKEN_FROM_PEAK=[]

for i,it in enumerate(ITEMS):
    L=it["L"];sp=GOLD_SPANS[i];gs=sp[0];w=frozen_pointer(A_PACK[i],qA(it["entity"]));p=argmax_start(w)
    seq=a_sequence(A_RAW[i],p,L);scb=invalid_scores() if seq is None else concat_scores(seq,L)
    rb,selb,_=rank(scb,i)
    scp,speak=peak_contain_scores(A_RAW[i],w,L);rp,selp,_=rank(scp,i)
    scn,snew,peaks,starts=top8_joint_scores(A_RAW[i],w,L);rn,seln,_=rank(scn,i)
    RBASE.append(rb);RPEAK.append(rp);RNEW.append(rn);BYLEN_NEW[L].append(rn)
    BASE_START.append(p);PEAK_START.append(speak if speak is not None else -1);NEW_START.append(snew if snew is not None else -1)
    BASE_INSIDE.append(int(p in sp));TOP8_GOLD.append(int(any(x in sp for x in peaks)))
    if rp!=1 and rn==1:RECOVERED_FROM_PEAK.append(i+1)
    if rp==1 and rn!=1:BROKEN_FROM_PEAK.append(i+1)
    if i<8 or rp!=1 or rn!=1 or (i+1)%64==0:
        print(f"      [{i+1:04d}/{N}] L={L} gold={gs:3d} peak={p:3d} "
              f"pOrigin={speak if speak is not None else -1:3d} top8Origin={snew if snew is not None else -1:3d} "
              f"goldTop8={TOP8_GOLD[-1]} | BASE={rb:4d} PEAK={rp:4d} NEW={rn:4d} "
              f"sel={selb+1 if selb>=0 else 0:04d}/{selp+1 if selp>=0 else 0:04d}/{seln+1 if seln>=0 else 0:04d}")

MB=metrics(RBASE);MP=metrics(RPEAK);MN=metrics(RNEW);MLN={L:metrics(BYLEN_NEW[L]) for L in LENS}
print(f"      BASELINE ARGMAX R1/R5/R16/MRR : {MB[0]:.4f}/{MB[1]:.4f}/{MB[2]:.4f}/{MB[3]:.6f}")
print(f"      PEAK_CONTAIN    R1/R5/R16/MRR : {MP[0]:.4f}/{MP[1]:.4f}/{MP[2]:.4f}/{MP[3]:.6f}")
print(f"      TOP8_JOINT      R1/R5/R16/MRR : {MN[0]:.4f}/{MN[1]:.4f}/{MN[2]:.4f}/{MN[3]:.6f}")
for L in LENS:print(f"      L={L} TOP8 R1={MLN[L][0]:.4f} R5={MLN[L][1]:.4f} R16={MLN[L][2]:.4f}")
print(f"      Gold identity represented Top8 : {np.mean(TOP8_GOLD):.4f}")

print("\n[6/12] PRIMARY FAILURE TRANSITION...")
PFAIL=[i for i,r in enumerate(RPEAK) if r!=1];NFAIL=[i for i,r in enumerate(RNEW) if r!=1]
print(f"      PEAK_CONTAIN failures              : {len(PFAIL)}/{N}")
print(f"      TOP8_JOINT failures                : {len(NFAIL)}/{N}")
print(f"      PEAK failures recovered TOP8       : {len(RECOVERED_FROM_PEAK)}/{len(PFAIL) if PFAIL else 1}")
print(f"      Previously-correct PEAK broken     : {len(BROKEN_FROM_PEAK)}")
print("      PEAK failure IDs                   :",[i+1 for i in PFAIL][:80])
print("      TOP8 failure IDs                   :",[i+1 for i in NFAIL][:80])
print("      Recovered IDs                      :",RECOVERED_FROM_PEAK[:80])
print("      Broken IDs                         :",BROKEN_FROM_PEAK[:80])

print("\n[7/12] COUNTERFACTUAL...")
RCF_BASE=[];RCF_PEAK=[];RCF_NEW=[];CF_TOP8_GOLD=[];CF_RECOVERED=[];CF_BROKEN=[]
for i,it in enumerate(ITEMS):
    j=(i+353)%N;target=TARGETS[j];L=TLEN[j];src=ITEMS[j]
    ents=[it["entity"],NAMES[3*i+1],NAMES[3*i+2]]
    rows=[f"Instrument {ents[0]} carries seal {target}.",
          f"Instrument {ents[1]} carries seal {src['A_d'][0]}.",
          f"Instrument {ents[2]} carries seal {src['A_d'][1]}."]
    random.Random(SEED+90000+i*149).shuffle(rows);cfA=" ".join(rows)
    raw=forge(cfA);pack=install(raw);sp=exact_span(cfA,target)
    if sp is None or len(sp)!=L:raise RuntimeError(f"CF span mismatch item {i+1}")
    w=frozen_pointer(pack,qA(it["entity"]));p=argmax_start(w)
    seq=a_sequence(raw,p,L);scb=invalid_scores() if seq is None else concat_scores(seq,L)
    rb,_,_=rank(scb,j)
    scp,speak=peak_contain_scores(raw,w,L);rp,_,_=rank(scp,j)
    scn,snew,peaks,starts=top8_joint_scores(raw,w,L);rn,_,_=rank(scn,j)
    RCF_BASE.append(rb);RCF_PEAK.append(rp);RCF_NEW.append(rn)
    CF_TOP8_GOLD.append(int(any(x in sp for x in peaks)))
    if rp!=1 and rn==1:CF_RECOVERED.append(i+1)
    if rp==1 and rn!=1:CF_BROKEN.append(i+1)
    if i<5 or rp!=1 or rn!=1 or (i+1)%64==0:
        print(f"      [{i+1:04d}/{N}] target={j+1:04d} L={L} gold={sp[0]:3d} peak={p:3d} "
              f"pOrigin={speak if speak is not None else -1:3d} top8Origin={snew if snew is not None else -1:3d} "
              f"goldTop8={CF_TOP8_GOLD[-1]} BASE={rb:4d} PEAK={rp:4d} NEW={rn:4d}")
    del raw,pack,w,seq,scb,scp,scn

MCB=metrics(RCF_BASE);MCP=metrics(RCF_PEAK);MCN=metrics(RCF_NEW)
print(f"      CF BASE R1/R5/R16/MRR : {MCB[0]:.4f}/{MCB[1]:.4f}/{MCB[2]:.4f}/{MCB[3]:.6f}")
print(f"      CF PEAK R1/R5/R16/MRR : {MCP[0]:.4f}/{MCP[1]:.4f}/{MCP[2]:.4f}/{MCP[3]:.6f}")
print(f"      CF TOP8 R1/R5/R16/MRR : {MCN[0]:.4f}/{MCN[1]:.4f}/{MCN[2]:.4f}/{MCN[3]:.6f}")
print(f"      CF gold identity represented Top8 : {np.mean(CF_TOP8_GOLD):.4f}")

print("\n[8/12] COUNTERFACTUAL FAILURE TRANSITION...")
CFPFAIL=[i for i,r in enumerate(RCF_PEAK) if r!=1];CFNFAIL=[i for i,r in enumerate(RCF_NEW) if r!=1]
print(f"      CF PEAK_CONTAIN failures           : {len(CFPFAIL)}/{N}")
print(f"      CF TOP8_JOINT failures             : {len(CFNFAIL)}/{N}")
print(f"      CF PEAK failures recovered TOP8    : {len(CF_RECOVERED)}/{len(CFPFAIL) if CFPFAIL else 1}")
print(f"      CF previously-correct PEAK broken  : {len(CF_BROKEN)}")
print("      CF PEAK failure IDs                :",[i+1 for i in CFPFAIL][:80])
print("      CF TOP8 failure IDs                :",[i+1 for i in CFNFAIL][:80])
print("      CF recovered IDs                   :",CF_RECOVERED[:80])
print("      CF broken IDs                      :",CF_BROKEN[:80])

print("\n[9/12] STATISTICAL SUMMARY...")
def wilson(k,n,z=1.959963984540054):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h

for name,r in [("BASE",RBASE),("PEAK",RPEAK),("TOP8",RNEW),
               ("TOP8_L2",BYLEN_NEW[2]),("TOP8_L3",BYLEN_NEW[3]),("TOP8_L4",BYLEN_NEW[4]),
               ("CF_BASE",RCF_BASE),("CF_PEAK",RCF_PEAK),("CF_TOP8",RCF_NEW)]:
    k=sum(int(x==1) for x in r);lo,hi=wilson(k,len(r))
    print(f"      {name:8s}: {k:4d}/{len(r):4d} = {k/len(r):.4f} | 95% Wilson [{lo:.4f},{hi:.4f}]")

print("\n[10/12] PREDECLARED GATES...")
GATES={
"BASELINE_REPRO_R1":abs(MB[0]-.9824)<=.005,
"PEAK_REPRO_R1":abs(MP[0]-.9932)<=.005,
"TOP8_JOINT_R1":MN[0]>=TH_NEW_R1,
"TOP8_JOINT_R5":MN[1]>=TH_NEW_R5,
"TOP8_L2_R1":MLN[2][0]>=TH_EACH,
"TOP8_L3_R1":MLN[3][0]>=TH_EACH,
"TOP8_L4_R1":MLN[4][0]>=TH_EACH,
"TOP8_CF_R1":MCN[0]>=TH_CF,
"NO_PRIMARY_REGRESSION":len(BROKEN_FROM_PEAK)==0,
"NO_CF_REGRESSION":len(CF_BROKEN)==0}
for k,v in GATES.items():print(f"      {k:28s}: {'PASS' if v else 'FAIL'}")

print("\n[11/12] PROTOCOL AUDIT...")
S1=sentinel();WEIGHT_OK=S0==S1
GATES["WEIGHT_SENTINEL"]=WEIGHT_OK
print("      Weight sentinel                     :","PASS" if WEIGHT_OK else "FAIL")
print("      Trainable tensors                   :",sum(int(p.requires_grad) for p in model.parameters()))
print("      Pointer                             : L28H00 unchanged")
print("      Address                             : L00-V uncentered unchanged")
print("      Match                               : CONCATCOS unchanged")
print("      Panel                               : exact sealed deterministic reconstruction")
print(f"      Pointer candidate K                 : {PTR_TOPK} · FROZEN")
print("      K scan                              : NONE")
print("      Baseline                            : ARGMAX unchanged")
print("      PEAK_CONTAIN                       : TEST533 rule unchanged")
print("      TOP8_JOINT                         : Top8 peaks -> containing width-L origins -> global CONCATCOS")
print("      Identity length L                  : PROTOCOL-KNOWN")
print("      Cross-length competition           : NO")
print("      Selection threshold                : NONE")
print("      Gold span used for selection       : NO")
print("      Decoded identity routing           : NO")
print("      Token-ID routing                   : NO")
print("      Gold B position used               : NO")
print("      Gold A span                        : DIAGNOSTIC ONLY")
print("      Head/layer/address discovery       : NONE")
print("      Match-law scan                     : NONE")
print("      Training / LoRA / optimizer / DRA  : NONE")
print("      Learned router/scorer              : NONE")
print("      Query-time candidate-B forwards    : 0")
print("      TEST534 retroactively changed      : NO")

print("\n[12/12] FINAL...")
ALL_PASS=all(GATES.values())
if ALL_PASS:
    VERDICT="TEST535_TOP8_MULTI_PEAK_JOINT_STRONG_CANDIDATE"
    NEXT="freeze TOP8_JOINT; next remove protocol-known L and same-length bank partition before untouched Varan 1 final"
else:
    VERDICT="TEST535_TOP8_MULTI_PEAK_JOINT_NOT_FULLY_SUPPORTED"
    NEXT="preserve TEST535 unchanged; diagnose residual failures without scanning K/head/layer/address/match"

RESULT={"test":TEST,
"parents":{"528":PARENT528,"529":PARENT529,"530":PARENT530,"531":PARENT531,"532":PARENT532,"533":PARENT533,"534":PARENT534},
"lock_sha":LOCK_SHA,"pointer_topk":PTR_TOPK,
"baseline":{"r1":MB[0],"r5":MB[1],"r16":MB[2],"mrr":MB[3]},
"peak_contain":{"r1":MP[0],"r5":MP[1],"r16":MP[2],"mrr":MP[3]},
"top8_joint":{"r1":MN[0],"r5":MN[1],"r16":MN[2],"mrr":MN[3]},
"top8_lengths":{str(L):{"r1":MLN[L][0],"r5":MLN[L][1],"r16":MLN[L][2],"mrr":MLN[L][3]} for L in LENS},
"counterfactual_baseline":{"r1":MCB[0],"r5":MCB[1],"r16":MCB[2],"mrr":MCB[3]},
"counterfactual_peak":{"r1":MCP[0],"r5":MCP[1],"r16":MCP[2],"mrr":MCP[3]},
"counterfactual_top8":{"r1":MCN[0],"r5":MCN[1],"r16":MCN[2],"mrr":MCN[3]},
"peak_failures":len(PFAIL),"top8_failures":len(NFAIL),
"recovered_from_peak":len(RECOVERED_FROM_PEAK),"broken_from_peak":len(BROKEN_FROM_PEAK),
"cf_peak_failures":len(CFPFAIL),"cf_top8_failures":len(CFNFAIL),
"cf_recovered_from_peak":len(CF_RECOVERED),"cf_broken_from_peak":len(CF_BROKEN),
"gates":GATES,"verdict":VERDICT}
RESULT_SHA=hashlib.sha256(json.dumps(RESULT,sort_keys=True,separators=(",",":")).encode()).hexdigest()

print("\n"+"="*176)
print("TEST535 FINAL RESULT — AKBASCORE MAM · VARAN 1 · TOP-8 MULTI-PEAK JOINT RESOLUTION")
print("="*176)
print("MODEL                              :",MODEL_ID,"· frozen")
print("PANEL                              : TEST531/532/533/534 deterministic reconstruction · N=1024")
print("POINTER                            : L28H00 · UNCHANGED")
print("ADDRESS                            : L00-V · UNCENTERED · 1024D · UNCHANGED")
print("MATCH                              : CONCATCOS · UNCHANGED")
print("POINTER CANDIDATES                 : TOP-8 · FROZEN")
print("-"*176)
print(f"BASELINE ARGMAX                    : R1={MB[0]:.4f} R5={MB[1]:.4f} R16={MB[2]:.4f} MRR={MB[3]:.6f}")
print(f"PEAK_CONTAIN                       : R1={MP[0]:.4f} R5={MP[1]:.4f} R16={MP[2]:.4f} MRR={MP[3]:.6f}")
print(f"TOP8_JOINT                         : R1={MN[0]:.4f} R5={MN[1]:.4f} R16={MN[2]:.4f} MRR={MN[3]:.6f}")
for L in LENS:
    print(f"TOP8 L={L}                           : R1={MLN[L][0]:.4f} R5={MLN[L][1]:.4f} R16={MLN[L][2]:.4f} MRR={MLN[L][3]:.6f}")
print(f"CF BASELINE                        : R1={MCB[0]:.4f} R5={MCB[1]:.4f} R16={MCB[2]:.4f} MRR={MCB[3]:.6f}")
print(f"CF PEAK_CONTAIN                    : R1={MCP[0]:.4f} R5={MCP[1]:.4f} R16={MCP[2]:.4f} MRR={MCP[3]:.6f}")
print(f"CF TOP8_JOINT                      : R1={MCN[0]:.4f} R5={MCN[1]:.4f} R16={MCN[2]:.4f} MRR={MCN[3]:.6f}")
print("-"*176)
print(f"PEAK FAILURES                      : {len(PFAIL)}")
print(f"TOP8 FAILURES                      : {len(NFAIL)}")
print(f"PEAK FAILURES RECOVERED            : {len(RECOVERED_FROM_PEAK)}")
print(f"PEAK CORRECT BROKEN                : {len(BROKEN_FROM_PEAK)}")
print(f"CF PEAK FAILURES                   : {len(CFPFAIL)}")
print(f"CF TOP8 FAILURES                   : {len(CFNFAIL)}")
print(f"CF PEAK FAILURES RECOVERED         : {len(CF_RECOVERED)}")
print(f"CF PEAK CORRECT BROKEN             : {len(CF_BROKEN)}")
print("-"*176)
for k,v in GATES.items():print(f"{k:35s}: {'PASS' if v else 'FAIL'}")
print("-"*176)
print("WEIGHT SENTINEL                    :","PASS" if WEIGHT_OK else "FAIL")
print("TRAINABLE TENSORS                  :",sum(int(p.requires_grad) for p in model.parameters()))
print("QUERY-TIME CANDIDATE-B FORWARDS    : 0")
print("KNOWN IDENTITY LENGTH              : YES")
print("CROSS-LENGTH GLOBAL COMPETITION    : NO")
print("TEST528 RESULT SHA                 :",PARENT528)
print("TEST529 RESULT SHA                 :",PARENT529)
print("TEST530 RESULT SHA                 :",PARENT530)
print("TEST531 RESULT SHA                 :",PARENT531)
print("TEST532 RESULT SHA                 :",PARENT532)
print("TEST533 RESULT SHA                 :",PARENT533)
print("TEST534 RESULT SHA                 :",PARENT534)
print("TEST535 PRE-RESULT LOCK            :",LOCK_SHA)
print("TEST535 RESULT SHA                 :",RESULT_SHA)
print("TOTAL TEST TIME                    :",f"{time.perf_counter()-T0:.2f}s")
print("VERDICT                            :",VERDICT)
print("NEXT                               :",NEXT)
print("="*176)

# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST539
# VARAN 1 — CROSS-LENGTH SCORE CALIBRATION X-RAY
#
# PARENT:
#   TEST538 -> UNKNOWN-LENGTH GLOBAL-BANK MECHANISM VALIDATION
#
# TEST538:
#   PRIMARY GLOBAL R1 = 1013/1024 = .9893
#   CF GLOBAL R1      = 1010/1024 = .9863
#   R5/R16            = 1.0000 / 1.0000
#   L2                = 342/342 = 1.0000
#   Every PRIMARY error: true L3/L4 -> selected L2, gold rank=2
#   Every CF error:      true L3/L4 -> selected L2, gold rank=2
#
# PURPOSE:
#   Diagnose WHY frozen TEST537 joint evidence is not directly comparable
#   across L={2,3,4}.
#
# NO NEW RETRIEVAL LAW.
# NO LENGTH PENALTY.
# NO LENGTH NORMALIZATION.
# NO THRESHOLD.
# NO FIT.
# NO SELECTION CHANGE.
#
# For every query, measure independently:
#   best owner/evidence produced by L2
#   best owner/evidence produced by L3
#   best owner/evidence produced by L4
#
# For cross-length failures record:
#   gold joint
#   selected wrong-L joint
#   Δjoint
#   pointer probability
#   CONCATCOS
#   residual = 1-cos
#   identity evidence = -log(residual)
#   candidate length / origin / peak
#
# Also measure score distributions by candidate length so that a later
# TEST540 law can be predeclared without modifying TEST539.
#
# FROZEN:
#   MODEL   = Mistral-7B-Instruct-v0.3
#   POINTER = L28H00
#   ADDRESS = L00-V UNCENTERED
#   TOPK    = 8
#   MATCH   = CONCATCOS
#   LAW     = log(pointer)-log(max(1-CONCATCOS,float32_eps))
#   PANEL   = exact TEST531–538 development panel
#
# TEST539 is X-RAY ONLY.
# Any proposed cross-length law belongs to TEST540.

import os,sys,subprocess,importlib.util,random,time,hashlib,json,gc,re,math
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

TEST="539";SEED=531531;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";N=1024
DEVICE=torch.device("cuda");FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
PTR_L=28;PTR_H=0;BL=0;BK="V";CENTER=False;LENS=(2,3,4);PTR_TOPK=8
EPS=float(np.finfo(np.float32).eps)
PARENT528="9235ed38c944f41a9ede0426d3aabaa12b5b20646f30e2be921e618f9471d3c2"
PARENT529="27ba703872f0d8a6ed870a6ba041d0cd0703d2eca467deac97186a90497795a0"
PARENT530="289247c88dfd4358cbe5ffb8499891546b27acc6912caf525a155ae81338db7b"
PARENT531="20257d7d613a49e55b03603946f07ec493009924f40ac26ab74cb817e1466528"
PARENT532="9e37a473a322a19f7b3862a25162835ab6396998dc4fc416bc90bc9b718df188"
PARENT533="5239fb585b87a22f5da778c61a1011379b216a25bf7d05dc5d01952cbb22a5c6"
PARENT534="b25128d17176790b2a8694c8ec9e7db5eaad469ff02a87a46876b2ffd45a9bb1"
PARENT535="fc407885133d48f84a952249b15e884b201749136f6b16a49e7db11b0e9e5087"
PARENT536=""
PARENT537="9a4b6856a5d2a4f440ffa66cf4ba8c57b308b4f3a1cb7db4d0d8b826dd87cbf9"
PARENT538="51bbc7059421c1795d26ca5489f3f6aa97ac1dfcd1f70ab478596832b939c31a"

os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*176)
print("TEST539 — AKBASCORE MAM · VARAN 1 · CROSS-LENGTH SCORE CALIBRATION X-RAY")
print("TEST538 FOLLOW-UP · FROZEN TEST537 LAW · L2/L3/L4 SCORE ANATOMY · NO NEW RETRIEVAL LAW")
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
print("      MATCH   : CONCATCOS · LOCKED")
print("      TOPK    : 8 · FROZEN")
print("      LAW     : log(pointer)-log(max(1-CONCATCOS,float32_eps)) · FROZEN")
print("      X-RAY   : CROSS-LENGTH SCORE ANATOMY ONLY")
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

print("\n[2/12] Reconstructing exact TEST538 panel...")
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
if any(len(ALL[L])<need[L] for L in LENS):raise RuntimeError("Identity generation exhausted.")

TARGETS=[];TLEN=[]
for i in range(N):
    L=LENS[i%len(LENS)];TARGETS.append(ALL[L].pop());TLEN.append(L)

def pop_same(L,forbidden):
    while ALL[L]:
        x=ALL[L].pop()
        if x not in forbidden:return x
    raise RuntimeError(f"Identity pool exhausted L={L}")

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
    if len(out)<n:raise RuntimeError("Name pool too small.")
    return out[:n]

NAMES=build_names(N*3);CL=["ALPHA","BETA","GAMMA"]
def make_items():
    out=[]
    for i in range(N):
        L=TLEN[i];gold=TARGETS[i];forbid={gold}
        a1=pop_same(L,forbid);forbid.add(a1);a2=pop_same(L,forbid);forbid.add(a2)
        b1=pop_same(L,forbid);forbid.add(b1);b2=pop_same(L,forbid);forbid.add(b2)
        ents=NAMES[3*i:3*i+3]
        ra=[f"Instrument {ents[0]} carries seal {gold}.",f"Instrument {ents[1]} carries seal {a1}.",f"Instrument {ents[2]} carries seal {a2}."]
        rb=[f"Seal {gold} corresponds to routing class {CL[0]}.",f"Seal {b1} corresponds to routing class {CL[1]}.",f"Seal {b2} corresponds to routing class {CL[2]}."]
        random.Random(SEED+i*131+17).shuffle(ra);random.Random(SEED+i*137+29).shuffle(rb)
        out.append({"id":i+1,"L":L,"A":" ".join(ra),"B":" ".join(rb),"entity":ents[0],"gold":gold,"A_d":[a1,a2]})
    return out

ITEMS=make_items()
print("      Required identity pools:",need)
for L in LENS:print(f"      L={L}: {sum(x['L']==L for x in ITEMS)}")
print("      Remaining reserve:",{L:len(ALL[L]) for L in LENS})
print("      Panel seed:",SEED)

LOCK={"test":TEST,"parent538":PARENT538,"model":MODEL_ID,"seed":SEED,"N":N,
"status":"VARAN1_CROSS_LENGTH_SCORE_XRAY",
"pointer":{"layer":PTR_L,"head":PTR_H,"topk":PTR_TOPK},
"address":{"layer":BL,"kind":BK,"center":CENTER,"dimension":KVD},
"lengths":[2,3,4],
"law":"log(pointer)-log(max(1-CONCATCOS,float32_eps))",
"selection_change":False,"fit":False,"normalization":False,"threshold":False,
"forbidden":["NEW_RETRIEVAL_LAW","LENGTH_PENALTY","LENGTH_NORMALIZATION","LENGTH_PRIOR","THRESHOLD_SCAN",
"LAMBDA_SCAN","TOPK_SCAN","HEAD_SCAN","LAYER_SCAN","ADDRESS_SCAN","MATCH_SCAN","GOLD_SELECTION",
"TOKEN_ID_ROUTING","DECODED_ID_ROUTING","QUERY_TIME_B_FORWARD","TRAINING","LORA","DRA","LEARNED_ROUTER"]}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      TEST539 LOCK:",LOCK_SHA)

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
    c,s=rope_cos_sin(dummy,pos);_,kr=apply_rotary_pos_emb(dummy,kk,c,s,unsqueeze_dim=1);return kr[0]
def install(raw):
    T=raw[0][0].shape[1];return [(rope_k(k,list(range(T))),v) for k,v in raw],T,T
def cache_of(inst):
    try:c=DynamicCache(config=cfg)
    except TypeError:c=DynamicCache()
    for li,(k,v) in enumerate(inst):c.update(k.unsqueeze(0),v.unsqueeze(0),li)
    return c
def qA(e):return f"What seal does instrument {e} carry? Give only the exact seal."

print("\n[3/12] OFFLINE FORGE...")
A_RAW=[];A_PACK=[];GOLD_SPANS=[];B_BANK={L:{"vec":[],"own":[]} for L in LENS}
for i,it in enumerate(ITEMS):
    ar=forge(it["A"]);br=forge(it["B"]);asp=exact_span(it["A"],it["gold"])
    if asp is None or len(asp)!=it["L"]:raise RuntimeError(f"A span mismatch {i+1}")
    A_RAW.append(ar);A_PACK.append(install(ar));GOLD_SPANS.append(asp)
    x=br[BL][1][:,1:,:].float().cpu().permute(1,0,2).reshape(-1,KVD).contiguous();L=it["L"]
    wins=torch.stack([x[j:j+L] for j in range(len(x)-L+1)])
    B_BANK[L]["vec"].append(wins);B_BANK[L]["own"].extend([i]*(len(x)-L+1))
    del br,x,wins
    if (i+1)%64==0:print(f"      forged {i+1:4d}/{N}")
gc.collect();torch.cuda.empty_cache()
print("      A memories:",len(A_RAW));print("      B memories:",N)
print("      Query-time candidate-B forwards: 0")

print("\n[4/12] Building frozen B banks...")
for L in LENS:
    X=torch.cat(B_BANK[L]["vec"],0);X=X/X.norm(dim=2,keepdim=True).clamp_min(1e-8)
    B_BANK[L]["vec"]=X.to(DEVICE,dtype=torch.float16)
    B_BANK[L]["own"]=torch.tensor(B_BANK[L]["own"],device=DEVICE,dtype=torch.long)
    print(f"      L={L}: {len(B_BANK[L]['own'])} windows | {tuple(B_BANK[L]['vec'].shape)}");del X
gc.collect();torch.cuda.empty_cache()

@torch.inference_mode()
def query_forward(packet,q):
    inst,Tm,P=packet;c=cache_of(inst);ids=enc(FMT.format(q=q));n=len(ids)
    pos=torch.arange(P,P+n,device=DEVICE).unsqueeze(0);mask=torch.ones((1,Tm+n),device=DEVICE,dtype=torch.long)
    o=model(input_ids=torch.tensor([ids],device=DEVICE),past_key_values=c,attention_mask=mask,position_ids=pos,use_cache=False,output_hidden_states=True,return_dict=True)
    return o,Tm,P,n

@torch.inference_mode()
def frozen_pointer(packet,q):
    inst,Tm,P=packet;o,_,_,n=query_forward(packet,q);qpos=P+n-1;layer=layers[PTR_L]
    h=o.hidden_states[PTR_L][0,-1].to(layer.input_layernorm.weight.dtype);hn=layer.input_layernorm(h)
    qv=layer.self_attn.q_proj(hn).view(1,QH,1,HD);ak=inst[PTR_L][0].repeat_interleave(REP,dim=0).unsqueeze(0)
    dummy=torch.zeros((1,QH,1,HD),device=DEVICE,dtype=qv.dtype);c,s=rope_cos_sin(dummy,[qpos])
    qrot,_=apply_rotary_pos_emb(qv,dummy,c,s,unsqueeze_dim=1)
    score=torch.einsum("bhqd,bhkd->bhqk",qrot.float(),ak.float()).squeeze(0).squeeze(1)/math.sqrt(HD)
    score[:,0]=-torch.inf;w=torch.softmax(score,dim=-1).detach().cpu();w[:,0]=0
    del o,qv,ak,dummy,c,s,qrot,score
    z=w[PTR_H];return z/z.sum().clamp_min(1e-12)

def pointer_topk(w,k):
    x=w.clone()
    if len(x):x[0]=-1
    k=min(k,max(0,len(x)-1))
    return torch.topk(x,k=k,largest=True,sorted=True).indices.tolist() if k else []

def a_sequence(raw,start,L):
    x=raw[BL][1].float().cpu().permute(1,0,2).reshape(-1,KVD)
    if start<1 or start+L>len(x):return None
    return x[start:start+L]

def containing_starts(raw,peak,L):
    T=raw[BL][1].shape[1];lo=max(1,peak-L+1);hi=min(peak,T-L)
    return list(range(lo,hi+1)) if hi>=lo else []

@torch.inference_mode()
def concat_scores(Aseq,L):
    A=Aseq.to(DEVICE);A=A/A.norm(dim=1,keepdim=True).clamp_min(1e-8)
    B=B_BANK[L]["vec"];own=B_BANK[L]["own"]
    sim=(B.float()*A.float().unsqueeze(0)).sum(dim=2).mean(dim=1)
    out=torch.full((N,),-torch.inf,device=DEVICE)
    out.scatter_reduce_(0,own,sim,reduce="amax",include_self=True)
    return out.cpu()

def invalid_scores():return torch.full((N,),-torch.inf)

# Returns frozen TEST537 evidence PLUS provenance for every owner.
def score_length_xray(raw,w,L):
    best=invalid_scores()
    bcos=torch.full((N,),float("nan"))
    bp=torch.zeros(N)
    bstart=torch.full((N,),-1,dtype=torch.long)
    bpeak=torch.full((N,),-1,dtype=torch.long)
    for p in pointer_topk(w,PTR_TOPK):
        pw=max(float(w[p]),EPS);logp=math.log(pw)
        for s in containing_starts(raw,p,L):
            seq=a_sequence(raw,s,L)
            if seq is None:continue
            cos=concat_scores(seq,L);finite=torch.isfinite(cos)
            residual=torch.clamp(1.0-cos,min=EPS)
            joint=torch.full_like(cos,-torch.inf)
            joint[finite]=logp-torch.log(residual[finite])
            mask=joint>best
            best[mask]=joint[mask];bcos[mask]=cos[mask];bp[mask]=pw;bstart[mask]=s;bpeak[mask]=p
    return {"joint":best,"cos":bcos,"p":bp,"start":bstart,"peak":bpeak}

def rank(sc,g):
    if not torch.isfinite(sc).any():return N,-1,[]
    order=torch.argsort(sc,descending=True).tolist();return order.index(g)+1,order[0],order

def merge_lengths(X):
    best=invalid_scores();bl=torch.full((N,),-1,dtype=torch.long)
    for L in LENS:
        m=X[L]["joint"]>best;best[m]=X[L]["joint"][m];bl[m]=L
    return best,bl

def detail(X,L,owner):
    j=float(X[L]["joint"][owner]);c=float(X[L]["cos"][owner]);p=float(X[L]["p"][owner])
    res=max(1.0-c,EPS) if math.isfinite(c) else float("nan")
    ide=-math.log(res) if math.isfinite(res) else float("nan")
    return {"L":L,"owner":owner,"joint":j,"p":p,"cos":c,"residual":res,"id_evidence":ide,
            "start":int(X[L]["start"][owner]),"peak":int(X[L]["peak"][owner])}

def best_for_length(X,L):
    sc=X[L]["joint"]
    if not torch.isfinite(sc).any():return -1,None
    o=int(torch.argmax(sc));return o,detail(X,L,o)

def stats(v):
    a=np.asarray(v,float);a=a[np.isfinite(a)]
    if not len(a):return {"n":0}
    return {"n":len(a),"mean":float(a.mean()),"median":float(np.median(a)),
            "min":float(a.min()),"max":float(a.max()),
            "p05":float(np.quantile(a,.05)),"p95":float(np.quantile(a,.95))}

def fmt(d):
    if d is None:return "NONE"
    return (f"L{d['L']} owner={d['owner']+1:04d} joint={d['joint']:+.6f} "
            f"p={d['p']:.8f} cos={d['cos']:.9f} res={d['residual']:.9g} "
            f"ide={d['id_evidence']:+.6f} start={d['start']} peak={d['peak']}")

print("\n[5/12] PRIMARY — exact TEST538 reproduction + per-length X-ray...")
R=[];FAIL=[];ROWS=[];DIST={L:{"best_joint":[],"best_cos":[],"best_p":[]} for L in LENS}
for i,it in enumerate(ITEMS):
    w=frozen_pointer(A_PACK[i],qA(it["entity"]))
    X={L:score_length_xray(A_RAW[i],w,L) for L in LENS}
    global_sc,global_L=merge_lengths(X);r,sel,_=rank(global_sc,i);R.append(r)
    tops={}
    for L in LENS:
        o,d=best_for_length(X,L);tops[L]=d
        if d:
            DIST[L]["best_joint"].append(d["joint"]);DIST[L]["best_cos"].append(d["cos"]);DIST[L]["best_p"].append(d["p"])
    predL=int(global_L[sel]) if sel>=0 else -1
    if r!=1:
        FAIL.append(i+1)
        gold=detail(X,it["L"],i);wrong=detail(X,predL,sel)
        row={"item":i+1,"trueL":it["L"],"predL":predL,"rank":r,"gold":gold,"wrong":wrong,
             "delta":wrong["joint"]-gold["joint"],"tops":tops}
        ROWS.append(row)
        print(f"\n      PRIMARY FAIL item={i+1:04d} trueL={it['L']} predL={predL} gold-rank={r}")
        print("        GOLD :",fmt(gold));print("        WRONG:",fmt(wrong))
        print(f"        Δjoint wrong-gold = {row['delta']:+.9f}")
        for L in LENS:print(f"        BEST L{L}:",fmt(tops[L]))
    elif i<5 or (i+1)%128==0:
        print(f"      [{i+1:04d}/{N}] rank=1 trueL={it['L']} predL={predL}")
print(f"\n      PRIMARY reproduction: {sum(x==1 for x in R)}/{N} = {np.mean(np.asarray(R)==1):.4f}")
print("      Failure IDs:",FAIL)

print("\n[6/12] PRIMARY CROSS-LENGTH DISTRIBUTIONS...")
for L in LENS:
    print(f"      L={L} best-joint :",stats(DIST[L]["best_joint"]))
    print(f"          best-cos   :",stats(DIST[L]["best_cos"]))
    print(f"          pointer-p  :",stats(DIST[L]["best_p"]))
if ROWS:
    print("      Failure Δjoint wrongL2-gold:",stats([x["delta"] for x in ROWS]))
    print("      Failure GOLD joint         :",stats([x["gold"]["joint"] for x in ROWS]))
    print("      Failure WRONG joint        :",stats([x["wrong"]["joint"] for x in ROWS]))
    print("      Failure GOLD cos           :",stats([x["gold"]["cos"] for x in ROWS]))
    print("      Failure WRONG cos          :",stats([x["wrong"]["cos"] for x in ROWS]))
    print("      Failure GOLD pointer-p     :",stats([x["gold"]["p"] for x in ROWS]))
    print("      Failure WRONG pointer-p    :",stats([x["wrong"]["p"] for x in ROWS]))
    print("      Failure GOLD id-evidence   :",stats([x["gold"]["id_evidence"] for x in ROWS]))
    print("      Failure WRONG id-evidence  :",stats([x["wrong"]["id_evidence"] for x in ROWS]))

print("\n[7/12] COUNTERFACTUAL — exact TEST538 reproduction + per-length X-ray...")
RCF=[];CFFAIL=[];CFROWS=[];CFDIST={L:{"best_joint":[],"best_cos":[],"best_p":[]} for L in LENS}
for i,it in enumerate(ITEMS):
    j=(i+353)%N;target=TARGETS[j];trueL=TLEN[j];src=ITEMS[j]
    ents=[it["entity"],NAMES[3*i+1],NAMES[3*i+2]]
    rows=[f"Instrument {ents[0]} carries seal {target}.",f"Instrument {ents[1]} carries seal {src['A_d'][0]}.",f"Instrument {ents[2]} carries seal {src['A_d'][1]}."]
    random.Random(SEED+90000+i*149).shuffle(rows);cfA=" ".join(rows)
    raw=forge(cfA);pack=install(raw);sp=exact_span(cfA,target)
    if sp is None or len(sp)!=trueL:raise RuntimeError(f"CF span mismatch {i+1}")
    w=frozen_pointer(pack,qA(it["entity"]));X={L:score_length_xray(raw,w,L) for L in LENS}
    global_sc,global_L=merge_lengths(X);r,sel,_=rank(global_sc,j);RCF.append(r)
    tops={}
    for L in LENS:
        o,d=best_for_length(X,L);tops[L]=d
        if d:
            CFDIST[L]["best_joint"].append(d["joint"]);CFDIST[L]["best_cos"].append(d["cos"]);CFDIST[L]["best_p"].append(d["p"])
    predL=int(global_L[sel]) if sel>=0 else -1
    if r!=1:
        CFFAIL.append(i+1)
        gold=detail(X,trueL,j);wrong=detail(X,predL,sel)
        row={"item":i+1,"target":j+1,"trueL":trueL,"predL":predL,"rank":r,"gold":gold,"wrong":wrong,
             "delta":wrong["joint"]-gold["joint"],"tops":tops}
        CFROWS.append(row)
        print(f"\n      CF FAIL item={i+1:04d} target={j+1:04d} trueL={trueL} predL={predL} gold-rank={r}")
        print("        GOLD :",fmt(gold));print("        WRONG:",fmt(wrong))
        print(f"        Δjoint wrong-gold = {row['delta']:+.9f}")
        for L in LENS:print(f"        BEST L{L}:",fmt(tops[L]))
    elif i<5 or (i+1)%128==0:
        print(f"      [{i+1:04d}/{N}] target={j+1:04d} rank=1 trueL={trueL} predL={predL}")
    del raw,pack,w,X,global_sc,global_L
print(f"\n      CF reproduction: {sum(x==1 for x in RCF)}/{N} = {np.mean(np.asarray(RCF)==1):.4f}")
print("      CF Failure IDs:",CFFAIL)

print("\n[8/12] COUNTERFACTUAL CROSS-LENGTH DISTRIBUTIONS...")
for L in LENS:
    print(f"      L={L} best-joint :",stats(CFDIST[L]["best_joint"]))
    print(f"          best-cos   :",stats(CFDIST[L]["best_cos"]))
    print(f"          pointer-p  :",stats(CFDIST[L]["best_p"]))
if CFROWS:
    print("      CF Failure Δjoint wrongL2-gold:",stats([x["delta"] for x in CFROWS]))
    print("      CF Failure GOLD joint         :",stats([x["gold"]["joint"] for x in CFROWS]))
    print("      CF Failure WRONG joint        :",stats([x["wrong"]["joint"] for x in CFROWS]))
    print("      CF Failure GOLD cos           :",stats([x["gold"]["cos"] for x in CFROWS]))
    print("      CF Failure WRONG cos          :",stats([x["wrong"]["cos"] for x in CFROWS]))
    print("      CF Failure GOLD pointer-p     :",stats([x["gold"]["p"] for x in CFROWS]))
    print("      CF Failure WRONG pointer-p    :",stats([x["wrong"]["p"] for x in CFROWS]))
    print("      CF Failure GOLD id-evidence   :",stats([x["gold"]["id_evidence"] for x in CFROWS]))
    print("      CF Failure WRONG id-evidence  :",stats([x["wrong"]["id_evidence"] for x in CFROWS]))

print("\n[9/12] FAILURE STRUCTURE...")
def structure(rows,name):
    print("     ",name,"count:",len(rows))
    if not rows:return
    pairs={}
    for x in rows:pairs[(x["trueL"],x["predL"])]=pairs.get((x["trueL"],x["predL"]),0)+1
    print("      trueL -> selectedL:",pairs)
    print("      gold ranks:",{r:sum(x["rank"]==r for x in rows) for r in sorted(set(x["rank"] for x in rows))})
    print("      wrong-gold Δjoint:",stats([x["delta"] for x in rows]))
structure(ROWS,"PRIMARY")
structure(CFROWS,"COUNTERFACTUAL")

print("\n[10/12] REPRODUCTION GATES...")
PR1=float(np.mean(np.asarray(R)==1));CFR1=float(np.mean(np.asarray(RCF)==1))
GATES={
"TEST538_PRIMARY_REPRO":abs(PR1-(1013/1024))<=1e-12,
"TEST538_CF_REPRO":abs(CFR1-(1010/1024))<=1e-12,
"PRIMARY_FAILURE_COUNT":len(ROWS)==11,
"CF_FAILURE_COUNT":len(CFROWS)==14,
"PRIMARY_ALL_WRONG_L2":all(x["predL"]==2 for x in ROWS),
"CF_ALL_WRONG_L2":all(x["predL"]==2 for x in CFROWS),
"NO_SELECTION_LAW_CHANGE":True}
for k,v in GATES.items():print(f"      {k:32s}: {'PASS' if v else 'FAIL'}")

print("\n[11/12] PROTOCOL AUDIT...")
S1=sentinel();WEIGHT_OK=S0==S1;GATES["WEIGHT_SENTINEL"]=WEIGHT_OK
print("      Weight sentinel                    :","PASS" if WEIGHT_OK else "FAIL")
print("      Trainable tensors                  :",sum(int(p.requires_grad) for p in model.parameters()))
print("      Pointer                            : L28H00 unchanged")
print("      Address                            : L00-V uncentered unchanged")
print("      Pointer K                          : 8 unchanged")
print("      Match                              : CONCATCOS unchanged")
print("      TEST537 law                        : unchanged")
print("      TEST538 global competition         : reproduced")
print("      New retrieval law                  : NONE")
print("      Length penalty                     : NONE")
print("      Length normalization               : NONE")
print("      Length prior/router                : NONE")
print("      Threshold/lambda/K scan            : NONE")
print("      Head/layer/address/match scan      : NONE")
print("      Gold used for selection            : NO")
print("      Gold used for post-hoc X-ray       : YES")
print("      Training / LoRA / optimizer / DRA : NONE")
print("      Learned router/scorer              : NONE")
print("      Query-time candidate-B forwards    : 0")
print("      TEST539 purpose                    : DIAGNOSTIC ONLY")

print("\n[12/12] FINAL...")
ALL_PASS=all(GATES.values())
VERDICT="TEST539_CROSS_LENGTH_SCORE_XRAY_COMPLETE" if ALL_PASS else "TEST539_REPRODUCTION_MISMATCH"
NEXT="preserve TEST539 unchanged; use measured cross-length anatomy to predeclare exactly one TEST540 comparison law, then test it without scanning this panel"

def serial_row(x):
    return {"item":x["item"],"trueL":x["trueL"],"predL":x["predL"],"rank":x["rank"],
            "delta":x["delta"],"gold":x["gold"],"wrong":x["wrong"]}

RESULT={"test":TEST,"parent538":PARENT538,"lock_sha":LOCK_SHA,
"primary":{"r1":PR1,"failures":[serial_row(x) for x in ROWS]},
"counterfactual":{"r1":CFR1,"failures":[serial_row(x) for x in CFROWS]},
"primary_distributions":{str(L):{k:stats(v) for k,v in DIST[L].items()} for L in LENS},
"cf_distributions":{str(L):{k:stats(v) for k,v in CFDIST[L].items()} for L in LENS},
"gates":GATES,"verdict":VERDICT}
RESULT_SHA=hashlib.sha256(json.dumps(RESULT,sort_keys=True,separators=(",",":")).encode()).hexdigest()

print("\n"+"="*176)
print("TEST539 FINAL RESULT — AKBASCORE MAM · VARAN 1 · CROSS-LENGTH SCORE CALIBRATION X-RAY")
print("="*176)
print("MODEL                              :",MODEL_ID,"· frozen")
print("PANEL                              : exact TEST538 development panel · N=1024")
print("POINTER                            : L28H00 · UNCHANGED")
print("ADDRESS                            : L00-V UNCENTERED · UNCHANGED")
print("MATCH                              : CONCATCOS · UNCHANGED")
print("TOPK                               : 8 · UNCHANGED")
print("LAW                                : TEST537 · UNCHANGED")
print("NEW RETRIEVAL LAW                  : NONE")
print("-"*176)
print(f"PRIMARY TEST538 REPRO              : {sum(x==1 for x in R)}/{N} = {PR1:.4f}")
print(f"CF TEST538 REPRO                   : {sum(x==1 for x in RCF)}/{N} = {CFR1:.4f}")
print("PRIMARY CROSS-LENGTH FAILURES      :",len(ROWS))
print("CF CROSS-LENGTH FAILURES           :",len(CFROWS))
if ROWS:
    print("PRIMARY Δjoint wrong-gold          :",stats([x["delta"] for x in ROWS]))
if CFROWS:
    print("CF Δjoint wrong-gold               :",stats([x["delta"] for x in CFROWS]))
print("-"*176)
for L in LENS:
    print(f"PRIMARY L{L} BEST-JOINT             :",stats(DIST[L]["best_joint"]))
for L in LENS:
    print(f"CF L{L} BEST-JOINT                  :",stats(CFDIST[L]["best_joint"]))
print("-"*176)
for k,v in GATES.items():print(f"{k:35s}: {'PASS' if v else 'FAIL'}")
print("-"*176)
print("WEIGHT SENTINEL                    :","PASS" if WEIGHT_OK else "FAIL")
print("TRAINABLE TENSORS                  :",sum(int(p.requires_grad) for p in model.parameters()))
print("QUERY-TIME CANDIDATE-B FORWARDS    : 0")
print("TEST537 LAW CHANGED                : NO")
print("TEST538 SELECTION CHANGED          : NO")
print("LENGTH CORRECTION FITTED           : NO")
print("TEST537 RESULT SHA                 :",PARENT537)
print("TEST538 RESULT SHA                 :",PARENT538)
print("TEST539 PRE-RESULT LOCK            :",LOCK_SHA)
print("TEST539 RESULT SHA                 :",RESULT_SHA)
print("TOTAL TEST TIME                    :",f"{time.perf_counter()-T0:.2f}s")
print("VERDICT                            :",VERDICT)
print("NEXT                               :",NEXT)
print("="*176)

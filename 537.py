# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST537
# VARAN 1 — POINTER-PRIOR × IDENTITY-EVIDENCE JOINT RESOLUTION
#
# PARENT:
#   TEST536 -> MULTI-PEAK DISCRIMINATION X-RAY
#
# TEST536 ESTABLISHED:
#   - missing identity signal exists inside frozen Top-8
#   - unrestricted CONCATCOS global max recovers PRIMARY 7/7 and CF 8/9
#   - but breaks PRIMARY 44 and CF 42 previously-correct PEAK cases
#   - regression false matches beat gold by microscopic CONCATCOS differences
#   - pointer prior strongly favors the correct origin in those regressions
#
# TEST537 PREDECLARED LAW:
#
#   evidence(cos) = -log(max(1-cos, float32_eps))
#   joint          = log(pointer_weight) + evidence(cos)
#
# equivalently:
#
#   joint = log(pointer_weight) - log(max(1-CONCATCOS,float32_eps))
#
# No fitted coefficient.
# No lambda.
# No threshold.
# No K scan.
# No gold-dependent selection.
#
# Candidate generation remains:
#   frozen pointer Top-8 peaks
#     -> every valid width-L A span containing each peak
#     -> locked CONCATCOS against frozen B bank
#
# Selection changes ONLY from:
#   TEST535: max(CONCATCOS)
# to:
#   TEST537: max(log(pointer)-log(identity residual))
#
# FROZEN / UNCHANGED:
#   MODEL   = mistralai/Mistral-7B-Instruct-v0.3
#   POINTER = L28H00
#   ADDRESS = L00-V UNCENTERED
#   MATCH   = position-preserving CONCATCOS
#   TOPK    = 8
#   PANEL   = exact TEST531–536 deterministic construction
#   L       = protocol-known
#   B bank  = length-conditioned
#
# IMPORTANT:
#   TEST537 is NOT yet the untouched final Varan-1 seal because L remains
#   externally supplied and competition remains length-conditioned.
#
# PREDECLARED SUCCESS:
#   PEAK_CONTAIN reproduction ~0.9932 PRIMARY / ~0.9912 CF
#   TOP8_JOINT reproduction   ~0.9570 PRIMARY / ~0.9580 CF
#   PRIOR_EVIDENCE PRIMARY R1 >= .999
#   PRIOR_EVIDENCE PRIMARY R5 >= .999
#   EACH L PRIMARY R1 >= .995
#   PRIOR_EVIDENCE CF R1 >= .995
#   NO PRIMARY PEAK-correct regression
#   NO CF PEAK-correct regression
#   WEIGHT SENTINEL PASS
#   QUERY-TIME CANDIDATE-B FORWARDS = 0

import os,sys,subprocess,importlib.util,random,time,hashlib,json,gc,re,math
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

TEST="537";SEED=531531;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";N=1024
DEVICE=torch.device("cuda");FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
PTR_L=28;PTR_H=0;BL=0;BK="V";CENTER=False;LENS=(2,3,4);PTR_TOPK=8
EPS=float(np.finfo(np.float32).eps)
TH_R1=.999;TH_R5=.999;TH_EACH=.995;TH_CF=.995
PARENT528="9235ed38c944f41a9ede0426d3aabaa12b5b20646f30e2be921e618f9471d3c2"
PARENT529="27ba703872f0d8a6ed870a6ba041d0cd0703d2eca467deac97186a90497795a0"
PARENT530="289247c88dfd4358cbe5ffb8499891546b27acc6912caf525a155ae81338db7b"
PARENT531="20257d7d613a49e55b03603946f07ec493009924f40ac26ab74cb817e1466528"
PARENT532="9e37a473a322a19f7b3862a25162835ab6396998dc4fc416bc90bc9b718df188"
PARENT533="5239fb585b87a22f5da778c61a1011379b216a25bf7d05dc5d01952cbb22a5c6"
PARENT534="b25128d17176790b2a8694c8ec9e7db5eaad469ff02a87a46876b2ffd45a9bb1"
PARENT535="fc407885133d48f84a952249b15e884b201749136f6b16a49e7db11b0e9e5087"
PARENT536=""

os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*176)
print("TEST537 — AKBASCORE MAM · VARAN 1 · POINTER-PRIOR × IDENTITY-EVIDENCE JOINT RESOLUTION")
print("TEST536 FOLLOW-UP · FROZEN TOP-8 · LOCKED CONCATCOS · PARAMETER-FREE LOG PRIOR + LOG IDENTITY EVIDENCE")
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
print(f"      TOPK    : {PTR_TOPK} · FROZEN FROM TEST535")
print(f"      EPS     : float32 machine epsilon = {EPS:.12g}")
print("      LAW     : log(pointer) - log(max(1-CONCATCOS,eps))")
print("      FITTED PARAMETERS : NONE")
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
if any(len(ALL[L])<need[L] for L in LENS):raise RuntimeError(f"Identity generation exhausted: have={dict((L,len(ALL[L])) for L in LENS)} need={need}")

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
        ra=[f"Instrument {ents[0]} carries seal {gold}.",f"Instrument {ents[1]} carries seal {a1}.",f"Instrument {ents[2]} carries seal {a2}."]
        rb=[f"Seal {gold} corresponds to routing class {CL[0]}.",f"Seal {b1} corresponds to routing class {CL[1]}.",f"Seal {b2} corresponds to routing class {CL[2]}."]
        random.Random(SEED+i*131+17).shuffle(ra);random.Random(SEED+i*137+29).shuffle(rb)
        out.append({"id":i+1,"L":L,"A":" ".join(ra),"B":" ".join(rb),"entity":ents[0],"gold":gold,"A_d":[a1,a2]})
    return out

ITEMS=make_items()
print("      Required identity pools:",need)
for L in LENS:print(f"      L={L}: {sum(x['L']==L for x in ITEMS)}")
print("      Remaining reserve:",{L:len(ALL[L]) for L in LENS})
print("      Sealed panel seed:",SEED)

LOCK={"test":TEST,
"parents":{"528":PARENT528,"529":PARENT529,"530":PARENT530,"531":PARENT531,"532":PARENT532,"533":PARENT533,"534":PARENT534,"535":PARENT535,"536":PARENT536},
"model":MODEL_ID,"seed":SEED,"N":N,"status":"VARAN1_POINTER_PRIOR_IDENTITY_EVIDENCE",
"pointer":{"layer":PTR_L,"head":PTR_H,"topk":PTR_TOPK},
"address":{"layer":BL,"kind":BK,"center":CENTER,"dimension":KVD},
"identity":{"lengths":[2,3,4],"length_known":True},
"match":{"name":"CONCATCOS","locked_from":"TEST530"},
"law":{"name":"LOG_PRIOR_PLUS_LOG_IDENTITY_EVIDENCE","formula":"log(pointer_weight)-log(max(1-CONCATCOS,float32_eps))","epsilon":"float32_machine_epsilon","fitted_parameters":0},
"selection":"PREDECLARED",
"forbidden":["LAMBDA_SCAN","THRESHOLD_SCAN","TOPK_SCAN","HEAD_SCAN","LAYER_SCAN","ADDRESS_SCAN","MATCH_SCAN","GOLD_SELECTION","TOKEN_ID_ROUTING","DECODED_ID_ROUTING","QUERY_TIME_B_FORWARD","TRAINING","LORA","DRA","LEARNED_ROUTER"]}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      TEST537 LOCK:",LOCK_SHA)

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

print("\n[3/12] OFFLINE FORGE — exact sealed A/B construction...")
A_RAW=[];A_PACK=[];GOLD_SPANS=[];B_BANK={L:{"vec":[],"own":[]} for L in LENS}
for i,it in enumerate(ITEMS):
    ar=forge(it["A"]);br=forge(it["B"]);asp=exact_span(it["A"],it["gold"])
    if asp is None or len(asp)!=it["L"]:raise RuntimeError(f"A span mismatch item {i+1}")
    A_RAW.append(ar);A_PACK.append(install(ar));GOLD_SPANS.append(asp)
    x=br[BL][1][:,1:,:].float().cpu().permute(1,0,2).reshape(-1,KVD).contiguous();L=it["L"]
    wins=torch.stack([x[j:j+L] for j in range(len(x)-L+1)])
    B_BANK[L]["vec"].append(wins);B_BANK[L]["own"].extend([i]*(len(x)-L+1))
    del br,x,wins
    if (i+1)%64==0:print(f"      forged {i+1:4d}/{N}")
gc.collect();torch.cuda.empty_cache()
print("      A memories:",len(A_RAW));print("      B memories:",N);print("      Query-time candidate-B forwards: 0")

print("\n[4/12] Building locked CONCATCOS banks...")
for L in LENS:
    X=torch.cat(B_BANK[L]["vec"],0);X=X/X.norm(dim=2,keepdim=True).clamp_min(1e-8)
    B_BANK[L]["vec"]=X.to(DEVICE,dtype=torch.float16)
    B_BANK[L]["own"]=torch.tensor(B_BANK[L]["own"],device=DEVICE,dtype=torch.long)
    print(f"      L={L} windows: {len(B_BANK[L]['own'])} | shape={tuple(B_BANK[L]['vec'].shape)}");del X
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
    T=raw[BL][1].shape[1];lo=max(1,peak-L+1);hi=min(peak,T-L)
    return list(range(lo,hi+1)) if hi>=lo else []

@torch.inference_mode()
def concat_scores(Aseq,L):
    A=Aseq.to(DEVICE);A=A/A.norm(dim=1,keepdim=True).clamp_min(1e-8)
    B=B_BANK[L]["vec"];own=B_BANK[L]["own"]
    sim=(B.float()*A.float().unsqueeze(0)).sum(dim=2).mean(dim=1)
    out=torch.full((N,),-torch.inf,device=DEVICE);out.scatter_reduce_(0,own,sim,reduce="amax",include_self=True)
    return out.cpu()

def invalid_scores():return torch.full((N,),-torch.inf)
def rank(sc,g):
    if not torch.isfinite(sc).any():return N,-1,[]
    order=torch.argsort(sc,descending=True).tolist();return order.index(g)+1,order[0],order

def joint_concat(raw,starts,L):
    best=invalid_scores()
    for s in starts:
        seq=a_sequence(raw,s,L)
        if seq is None:continue
        best=torch.maximum(best,concat_scores(seq,L))
    return best

def peak_contain_scores(raw,w,L):
    p=argmax_start(w);return joint_concat(raw,containing_starts(raw,p,L),L)

def top8_concat_scores(raw,w,L):
    starts=set()
    for p in pointer_topk(w,PTR_TOPK):starts.update(containing_starts(raw,p,L))
    return joint_concat(raw,sorted(starts),L)

# TEST537 LAW — frozen before run.
# Each pointer peak keeps its own prior. A span admitted by multiple peaks may
# compete through each peak independently; final owner receives max joint evidence.
def prior_evidence_scores(raw,w,L):
    best=invalid_scores()
    for p in pointer_topk(w,PTR_TOPK):
        pw=max(float(w[p]),EPS);logp=math.log(pw)
        for s in containing_starts(raw,p,L):
            seq=a_sequence(raw,s,L)
            if seq is None:continue
            cos=concat_scores(seq,L)
            finite=torch.isfinite(cos)
            residual=torch.clamp(1.0-cos,min=EPS)
            joint=torch.full_like(cos,-torch.inf)
            joint[finite]=logp-torch.log(residual[finite])
            best=torch.maximum(best,joint)
    return best

def metrics(r):
    a=np.asarray(r,float);return float(np.mean(a==1)),float(np.mean(a<=5)),float(np.mean(a<=16)),float(np.mean(1/a))

print("\n[5/12] PRIMARY — PEAK / TOP8 reproduction + TEST537 law...")
RPEAK=[];RTOP8=[];R537=[];BYLEN={L:[] for L in LENS};REC=[];BROKEN=[]
for i,it in enumerate(ITEMS):
    L=it["L"];w=frozen_pointer(A_PACK[i],qA(it["entity"]))
    sp=peak_contain_scores(A_RAW[i],w,L);rp,_,_=rank(sp,i)
    st=top8_concat_scores(A_RAW[i],w,L);rt,_,_=rank(st,i)
    sn=prior_evidence_scores(A_RAW[i],w,L);rn,sel,_=rank(sn,i)
    RPEAK.append(rp);RTOP8.append(rt);R537.append(rn);BYLEN[L].append(rn)
    if rp!=1 and rn==1:REC.append(i+1)
    if rp==1 and rn!=1:BROKEN.append(i+1)
    if i<8 or rn!=1 or rp!=1 or (i+1)%64==0:
        print(f"      [{i+1:04d}/{N}] L={L} PEAK={rp:4d} TOP8={rt:4d} TEST537={rn:4d} sel={sel+1 if sel>=0 else 0:04d}")
MP=metrics(RPEAK);MT=metrics(RTOP8);M537=metrics(R537);ML={L:metrics(BYLEN[L]) for L in LENS}
print(f"      PEAK_CONTAIN : {MP[0]:.4f}/{MP[1]:.4f}/{MP[2]:.4f}/{MP[3]:.6f}")
print(f"      TOP8_JOINT   : {MT[0]:.4f}/{MT[1]:.4f}/{MT[2]:.4f}/{MT[3]:.6f}")
print(f"      TEST537 LAW  : {M537[0]:.4f}/{M537[1]:.4f}/{M537[2]:.4f}/{M537[3]:.6f}")
for L in LENS:print(f"      L={L}: R1={ML[L][0]:.4f} R5={ML[L][1]:.4f} R16={ML[L][2]:.4f} MRR={ML[L][3]:.6f}")
print("      PEAK failures recovered :",len(REC))
print("      PEAK-correct broken     :",len(BROKEN))
print("      TEST537 failure IDs     :",[i+1 for i,r in enumerate(R537) if r!=1][:100])

print("\n[6/12] PRIMARY TRANSITION...")
PFAIL=[i for i,r in enumerate(RPEAK) if r!=1];NFAIL=[i for i,r in enumerate(R537) if r!=1]
print(f"      PEAK failures           : {len(PFAIL)}/{N}")
print(f"      TEST537 failures        : {len(NFAIL)}/{N}")
print(f"      recovered               : {len(REC)}/{len(PFAIL) if PFAIL else 1}")
print(f"      previously-correct broken: {len(BROKEN)}")
print("      recovered IDs           :",REC)
print("      broken IDs              :",BROKEN)

print("\n[7/12] COUNTERFACTUAL...")
RCFP=[];RCFT=[];RCF537=[];CFREC=[];CFBROKEN=[]
for i,it in enumerate(ITEMS):
    j=(i+353)%N;target=TARGETS[j];L=TLEN[j];src=ITEMS[j]
    ents=[it["entity"],NAMES[3*i+1],NAMES[3*i+2]]
    rows=[f"Instrument {ents[0]} carries seal {target}.",f"Instrument {ents[1]} carries seal {src['A_d'][0]}.",f"Instrument {ents[2]} carries seal {src['A_d'][1]}."]
    random.Random(SEED+90000+i*149).shuffle(rows);cfA=" ".join(rows)
    raw=forge(cfA);pack=install(raw);sp=exact_span(cfA,target)
    if sp is None or len(sp)!=L:raise RuntimeError(f"CF span mismatch item {i+1}")
    w=frozen_pointer(pack,qA(it["entity"]))
    a=peak_contain_scores(raw,w,L);rp,_,_=rank(a,j)
    b=top8_concat_scores(raw,w,L);rt,_,_=rank(b,j)
    c=prior_evidence_scores(raw,w,L);rn,sel,_=rank(c,j)
    RCFP.append(rp);RCFT.append(rt);RCF537.append(rn)
    if rp!=1 and rn==1:CFREC.append(i+1)
    if rp==1 and rn!=1:CFBROKEN.append(i+1)
    if i<5 or rn!=1 or rp!=1 or (i+1)%64==0:
        print(f"      [{i+1:04d}/{N}] target={j+1:04d} L={L} PEAK={rp:4d} TOP8={rt:4d} TEST537={rn:4d} sel={sel+1 if sel>=0 else 0:04d}")
    del raw,pack,w,a,b,c
MCFP=metrics(RCFP);MCFT=metrics(RCFT);MCF537=metrics(RCF537)
print(f"      CF PEAK_CONTAIN : {MCFP[0]:.4f}/{MCFP[1]:.4f}/{MCFP[2]:.4f}/{MCFP[3]:.6f}")
print(f"      CF TOP8_JOINT   : {MCFT[0]:.4f}/{MCFT[1]:.4f}/{MCFT[2]:.4f}/{MCFT[3]:.6f}")
print(f"      CF TEST537 LAW  : {MCF537[0]:.4f}/{MCF537[1]:.4f}/{MCF537[2]:.4f}/{MCF537[3]:.6f}")
print("      CF PEAK failures recovered :",len(CFREC))
print("      CF PEAK-correct broken     :",len(CFBROKEN))
print("      CF TEST537 failure IDs     :",[i+1 for i,r in enumerate(RCF537) if r!=1][:100])

print("\n[8/12] COUNTERFACTUAL TRANSITION...")
CFPFAIL=[i for i,r in enumerate(RCFP) if r!=1];CFNFAIL=[i for i,r in enumerate(RCF537) if r!=1]
print(f"      CF PEAK failures           : {len(CFPFAIL)}/{N}")
print(f"      CF TEST537 failures        : {len(CFNFAIL)}/{N}")
print(f"      CF recovered               : {len(CFREC)}/{len(CFPFAIL) if CFPFAIL else 1}")
print(f"      CF previously-correct broken: {len(CFBROKEN)}")
print("      CF recovered IDs           :",CFREC)
print("      CF broken IDs              :",CFBROKEN)

print("\n[9/12] STATISTICAL SUMMARY...")
def wilson(k,n,z=1.959963984540054):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h
for name,r in [("PEAK",RPEAK),("TOP8",RTOP8),("TEST537",R537),("537_L2",BYLEN[2]),("537_L3",BYLEN[3]),("537_L4",BYLEN[4]),
               ("CF_PEAK",RCFP),("CF_TOP8",RCFT),("CF_537",RCF537)]:
    k=sum(int(x==1) for x in r);lo,hi=wilson(k,len(r))
    print(f"      {name:8s}: {k:4d}/{len(r):4d} = {k/len(r):.4f} | 95% Wilson [{lo:.4f},{hi:.4f}]")

print("\n[10/12] PREDECLARED GATES...")
GATES={
"PRIMARY_PEAK_REPRO":abs(MP[0]-.9932)<=.005,
"PRIMARY_TOP8_REPRO":abs(MT[0]-.9570)<=.005,
"CF_PEAK_REPRO":abs(MCFP[0]-.9912)<=.005,
"CF_TOP8_REPRO":abs(MCFT[0]-.9580)<=.005,
"TEST537_R1":M537[0]>=TH_R1,
"TEST537_R5":M537[1]>=TH_R5,
"TEST537_L2_R1":ML[2][0]>=TH_EACH,
"TEST537_L3_R1":ML[3][0]>=TH_EACH,
"TEST537_L4_R1":ML[4][0]>=TH_EACH,
"TEST537_CF_R1":MCF537[0]>=TH_CF,
"NO_PRIMARY_PEAK_REGRESSION":len(BROKEN)==0,
"NO_CF_PEAK_REGRESSION":len(CFBROKEN)==0}
for k,v in GATES.items():print(f"      {k:32s}: {'PASS' if v else 'FAIL'}")

print("\n[11/12] PROTOCOL AUDIT...")
S1=sentinel();WEIGHT_OK=S0==S1;GATES["WEIGHT_SENTINEL"]=WEIGHT_OK
print("      Weight sentinel                    :","PASS" if WEIGHT_OK else "FAIL")
print("      Trainable tensors                  :",sum(int(p.requires_grad) for p in model.parameters()))
print("      Pointer                            : L28H00 unchanged")
print("      Address                            : L00-V uncentered unchanged")
print("      Match                              : CONCATCOS unchanged")
print("      Pointer candidate K                : 8 unchanged")
print("      Candidate law                      : log(pointer)-log(max(1-cos,eps))")
print("      Fitted coefficient                 : NONE")
print("      Lambda                             : NONE")
print("      Threshold                          : NONE")
print("      K scan                             : NONE")
print("      Head/layer/address/match scan      : NONE")
print("      Identity length L                  : PROTOCOL-KNOWN")
print("      Cross-length competition           : NO")
print("      Gold span used for selection       : NO")
print("      Gold B position used               : NO")
print("      Token-ID / decoded-ID routing      : NO")
print("      Training / LoRA / optimizer / DRA : NONE")
print("      Learned router/scorer              : NONE")
print("      Query-time candidate-B forwards    : 0")
print("      TEST536 retroactively changed      : NO")

print("\n[12/12] FINAL...")
ALL_PASS=all(GATES.values())
if ALL_PASS:
    VERDICT="TEST537_POINTER_PRIOR_IDENTITY_EVIDENCE_STRONG_CANDIDATE"
    NEXT="freeze TEST537 law; next remove protocol-known L and same-length bank partition, then run a fresh untouched Varan 1 final validation"
else:
    VERDICT="TEST537_POINTER_PRIOR_IDENTITY_EVIDENCE_NOT_FULLY_SUPPORTED"
    NEXT="preserve TEST537 unchanged; diagnose failures without scanning lambda/K/head/layer/address/match on this panel"

RESULT={"test":TEST,
"parents":{"528":PARENT528,"529":PARENT529,"530":PARENT530,"531":PARENT531,"532":PARENT532,"533":PARENT533,"534":PARENT534,"535":PARENT535,"536":PARENT536},
"lock_sha":LOCK_SHA,"law":"log(pointer)-log(max(1-CONCATCOS,float32_eps))","eps":EPS,
"primary":{"peak":MP,"top8":MT,"test537":M537,"by_length":{str(L):ML[L] for L in LENS},"recovered":len(REC),"broken":len(BROKEN)},
"counterfactual":{"peak":MCFP,"top8":MCFT,"test537":MCF537,"recovered":len(CFREC),"broken":len(CFBROKEN)},
"gates":GATES,"verdict":VERDICT}
RESULT_SHA=hashlib.sha256(json.dumps(RESULT,sort_keys=True,separators=(",",":")).encode()).hexdigest()

print("\n"+"="*176)
print("TEST537 FINAL RESULT — AKBASCORE MAM · VARAN 1 · POINTER-PRIOR × IDENTITY-EVIDENCE JOINT RESOLUTION")
print("="*176)
print("MODEL                              :",MODEL_ID,"· frozen")
print("PANEL                              : TEST531–536 deterministic reconstruction · N=1024")
print("POINTER                            : L28H00 · UNCHANGED")
print("ADDRESS                            : L00-V · UNCENTERED · 1024D · UNCHANGED")
print("MATCH                              : CONCATCOS · UNCHANGED")
print("POINTER CANDIDATES                 : TOP-8 · UNCHANGED")
print("LAW                                : log(pointer)-log(max(1-CONCATCOS,float32_eps))")
print("FITTED PARAMETERS                  : 0")
print("-"*176)
print(f"PRIMARY PEAK_CONTAIN               : R1={MP[0]:.4f} R5={MP[1]:.4f} R16={MP[2]:.4f} MRR={MP[3]:.6f}")
print(f"PRIMARY TOP8_JOINT                 : R1={MT[0]:.4f} R5={MT[1]:.4f} R16={MT[2]:.4f} MRR={MT[3]:.6f}")
print(f"PRIMARY TEST537                    : R1={M537[0]:.4f} R5={M537[1]:.4f} R16={M537[2]:.4f} MRR={M537[3]:.6f}")
for L in LENS:print(f"TEST537 L={L}                       : R1={ML[L][0]:.4f} R5={ML[L][1]:.4f} R16={ML[L][2]:.4f} MRR={ML[L][3]:.6f}")
print(f"CF PEAK_CONTAIN                    : R1={MCFP[0]:.4f} R5={MCFP[1]:.4f} R16={MCFP[2]:.4f} MRR={MCFP[3]:.6f}")
print(f"CF TOP8_JOINT                      : R1={MCFT[0]:.4f} R5={MCFT[1]:.4f} R16={MCFT[2]:.4f} MRR={MCFT[3]:.6f}")
print(f"CF TEST537                         : R1={MCF537[0]:.4f} R5={MCF537[1]:.4f} R16={MCF537[2]:.4f} MRR={MCF537[3]:.6f}")
print("-"*176)
print("PRIMARY PEAK FAILURES RECOVERED    :",len(REC))
print("PRIMARY PEAK-CORRECT BROKEN        :",len(BROKEN))
print("CF PEAK FAILURES RECOVERED         :",len(CFREC))
print("CF PEAK-CORRECT BROKEN             :",len(CFBROKEN))
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
print("TEST535 RESULT SHA                 :",PARENT535)
print("TEST536 RESULT SHA                 :",PARENT536 if PARENT536 else "INSERT TEST536 RESULT SHA")
print("TEST537 PRE-RESULT LOCK            :",LOCK_SHA)
print("TEST537 RESULT SHA                 :",RESULT_SHA)
print("TOTAL TEST TIME                    :",f"{time.perf_counter()-T0:.2f}s")
print("VERDICT                            :",VERDICT)
print("NEXT                               :",NEXT)
print("="*176)

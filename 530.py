# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST530
# VARAN 1 — MULTI-TOKEN ADDRESS COMPOSITION X-RAY
#
# SEALED PARENTS:
#   TEST528 -> native single-token external seal PASS
#   TEST529 -> multi-token primitive boundary:
#              pointer hit 0.9980
#              span R1 0.0420
#              L2 span R1 0.0380
#
# FROZEN:
#   MODEL   = mistralai/Mistral-7B-Instruct-v0.3
#   POINTER = L28H00
#   ADDRESS = L00-V UNCENTERED
#   DIM     = 1024
#   MODEL WEIGHTS = FROZEN
#
# TEST530 QUESTION:
#   Pointer already finds the correct multi-token identity.
#   What parameter-free numerical composition maps that
#   pointer-selected L=2 identity onto the same B identity?
#
# IMPORTANT:
#   L=2 ONLY. Diagnostic experiment.
#   No layer/head/address scan.
#   No training/LoRA/DRA.
#   No learned router/scorer.
#   No decoded identity/token-ID routing.
#   No query-time candidate-B model forward.
#   Every contiguous B width-2 window remains a candidate.
#
# PREDECLARED COMPOSITION ARMS:
#   SUM       = v1 + v2
#   MEAN      = (v1 + v2)/2
#   PTR_RATIO = B span composed using normalized A pointer
#               mass ratio across the two selected identity tokens
#   MAX_TOKEN = use the B token corresponding to the stronger
#               of the two A pointer positions
#   CONCATCOS = mean of position-wise cosine(A1,B1), cosine(A2,B2)
#
# NOTE:
#   Gold A span is used to measure/diagnose composition geometry.
#   It does NOT choose a B cartridge or B candidate window.
#   This is an X-RAY test, not a final retrieval seal.
#
# SUCCESS CONDITION FOR NEXT TEST:
#   A predeclared arm must show a decisive L2 recovery,
#   not merely a small gain over TEST529.

import os,sys,subprocess,importlib.util,random,time,hashlib,json,gc,re,math
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

TEST="530";SEED=530530;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
N=512;DEVICE=torch.device("cuda");FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
PTR_L=28;PTR_H=0;BL=0;BK="V";CENTER=False
PARENT528="9235ed38c944f41a9ede0426d3aabaa12b5b20646f30e2be921e618f9471d3c2"
PARENT529="27ba703872f0d8a6ed870a6ba041d0cd0703d2eca467deac97186a90497795a0"
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*176)
print("TEST530 — AKBASCORE MAM · VARAN 1 · MULTI-TOKEN ADDRESS COMPOSITION X-RAY")
print("TEST529 BOUNDARY FOLLOW-UP · FROZEN L28H00 → L00-V · L=2 ONLY · PARAMETER-FREE")
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
if QH%KVH:raise RuntimeError("QH must divide KVH.")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id;VOC=model.model.embed_tokens.weight.shape[0]
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)}")
print(f"      {NL}L H={H} QH={QH} KVH={KVH} HD={HD} ADDRESS={KVD}")
print("      POINTER : L28H00 FROZEN")
print("      ADDRESS : L00-V UNCENTERED FROZEN")
print("      PANEL   : L=2 ONLY · N=512")
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

def valid_pair(a,b):
    if a==b:return None
    s=a+" "+b
    pa=exact_span(f"Instrument Probe carries seal {s}.",s)
    pb=exact_span(f"Seal {s} corresponds to routing class ALPHA.",s)
    return s if pa and pb and len(pa)==2 and len(pb)==2 else None

print("\n[2/12] Building deterministic L=2 panel...")
rr=random.Random(SEED+100);ALL=[];used=set();tries=0
while len(ALL)<N*5 and tries<2000000:
    tries+=1;a,b=rr.choice(POOL),rr.choice(POOL);s=valid_pair(a,b)
    if s is None or s in used:continue
    used.add(s);ALL.append(s)
if len(ALL)<N*5:raise RuntimeError(f"Pair pool too small: {len(ALL)}")
TARGETS=ALL[:N];AD1=ALL[N:2*N];AD2=ALL[2*N:3*N];BD1=ALL[3*N:4*N];BD2=ALL[4*N:5*N]
print("      Targets     :",len(TARGETS))
print("      A distractors:",2*N)
print("      B distractors:",2*N)

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
        ents=NAMES[3*i:3*i+3];gold=TARGETS[i]
        ra=[f"Instrument {ents[0]} carries seal {gold}.",
            f"Instrument {ents[1]} carries seal {AD1[i]}.",
            f"Instrument {ents[2]} carries seal {AD2[i]}."]
        rb=[f"Seal {gold} corresponds to routing class {CL[0]}.",
            f"Seal {BD1[i]} corresponds to routing class {CL[1]}.",
            f"Seal {BD2[i]} corresponds to routing class {CL[2]}."]
        random.Random(SEED+i*131+17).shuffle(ra);random.Random(SEED+i*137+29).shuffle(rb)
        out.append({"id":i+1,"A":" ".join(ra),"B":" ".join(rb),"entity":ents[0],"gold":gold})
    return out
ITEMS=make_items()

LOCK={"test":TEST,"parents":{"528":PARENT528,"529":PARENT529},"model":MODEL_ID,"seed":SEED,"N":N,
      "purpose":"L2_MULTI_TOKEN_ADDRESS_COMPOSITION_XRAY",
      "pointer":{"layer":PTR_L,"head":PTR_H,"frozen":True},
      "address":{"layer":BL,"kind":BK,"center":CENTER,"dimension":KVD,"frozen":True},
      "identity":{"length":2},
      "arms":["SUM","MEAN","PTR_RATIO","MAX_TOKEN","CONCATCOS"],
      "candidate_policy":"ALL_CONTIGUOUS_B_WIDTH2_WINDOWS",
      "selection":"PREDECLARED_ARMS_ONLY",
      "forbidden":["HEAD_SCAN","LAYER_SCAN","ADDRESS_SCAN","CENTER_SCAN","TOKEN_ID_ROUTING",
                   "DECODED_ID_ROUTING","GOLD_B_WINDOW_SELECTION","QUERY_TIME_B_FORWARD",
                   "TRAINING","LORA","DRA","LEARNED_ROUTER"]}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      TEST530 LOCK:",LOCK_SHA)

@torch.inference_mode()
def kv_from_ids(ids):
    o=model(input_ids=torch.tensor([ids],device=DEVICE),use_cache=False,output_hidden_states=True,return_dict=True);out=[]
    for L,layer in enumerate(layers):
        h=o.hidden_states[L][0].to(layer.input_layernorm.weight.device,layer.input_layernorm.weight.dtype)
        hn=layer.input_layernorm(h)
        k=layer.self_attn.k_proj(hn).view(-1,KVH,HD).transpose(0,1).contiguous()
        v=layer.self_attn.v_proj(hn).view(-1,KVH,HD).transpose(0,1).contiguous()
        out.append((k,v))
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
    for L,(k,v) in enumerate(inst):c.update(k.unsqueeze(0),v.unsqueeze(0),L)
    return c

def qA(e):return f"What seal does instrument {e} carry? Give only the exact seal."

print("\n[3/12] OFFLINE FORGE...")
A_RAW=[];A_PACK=[];A_GOLD=[];B_RAW=[];B_GOLD=[]
for i,it in enumerate(ITEMS):
    ar=forge(it["A"]);br=forge(it["B"])
    asp=exact_span(it["A"],it["gold"]);bsp=exact_span(it["B"],it["gold"])
    if asp is None or bsp is None or len(asp)!=2 or len(bsp)!=2:raise RuntimeError(f"Span mismatch item {i+1}")
    A_RAW.append(ar);A_PACK.append(install(ar));A_GOLD.append(asp);B_RAW.append(br);B_GOLD.append(bsp)
    if (i+1)%64==0:print(f"      forged {i+1:4d}/{N}")
print("      A memories:",len(A_RAW))
print("      B memories:",len(B_RAW))
print("      Query-time B forwards: 0")

print("\n[4/12] Building ALL contiguous B width-2 candidates...")
B1=[];B2=[];OWN=[]
for i,raw in enumerate(B_RAW):
    x=raw[BL][1][:,1:,:].float().cpu().permute(1,0,2).reshape(-1,KVD).contiguous()
    if len(x)<2:continue
    B1.append(x[:-1]);B2.append(x[1:]);OWN.extend([i]*(len(x)-1))
B1=torch.cat(B1,0);B2=torch.cat(B2,0);OWN=torch.tensor(OWN,dtype=torch.long)
print("      Candidate windows:",len(OWN))
print("      Gold B window selection: NO")
print("      Every contiguous width-2 B window indexed.")

def normrows(x):return x/x.norm(dim=1,keepdim=True).clamp_min(1e-8)
B1N=normrows(B1);B2N=normrows(B2)
BSUMN=normrows(B1+B2)
B1N=B1N.to(DEVICE,dtype=torch.float16);B2N=B2N.to(DEVICE,dtype=torch.float16)
BSUMN=BSUMN.to(DEVICE,dtype=torch.float16);OWN=OWN.to(DEVICE)
del B_RAW
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
    w=w/w.sum(dim=1,keepdim=True).clamp_min(1e-12)
    return w[PTR_H]

def rank(sc,g):
    order=torch.argsort(sc,descending=True).tolist()
    return order.index(g)+1,order[0],order

def metrics(r):
    a=np.asarray(r,float)
    return float(np.mean(a==1)),float(np.mean(a<=5)),float(np.mean(a<=16)),float(np.mean(1/a))

@torch.inference_mode()
def reduce_candidate_scores(sim):
    out=torch.full((N,),-torch.inf,device=DEVICE)
    out.scatter_reduce_(0,OWN,sim,reduce="amax",include_self=True)
    return out.cpu()

@torch.inference_mode()
def score_sum(a1,a2):
    p=(a1+a2).to(DEVICE);p=p/p.norm().clamp_min(1e-12)
    return reduce_candidate_scores(torch.mv(BSUMN.float(),p.float()))

@torch.inference_mode()
def score_ratio(a1,a2,r1,r2):
    p=(r1*a1+r2*a2).to(DEVICE);p=p/p.norm().clamp_min(1e-12)
    b=(r1*B1.to(DEVICE)+r2*B2.to(DEVICE));b=normrows(b)
    return reduce_candidate_scores(torch.mv(b.float(),p.float()))

@torch.inference_mode()
def score_max_token(a1,a2,r1,r2):
    p=(a1 if r1>=r2 else a2).to(DEVICE);p=p/p.norm().clamp_min(1e-12)
    bank=B1N if r1>=r2 else B2N
    return reduce_candidate_scores(torch.mv(bank.float(),p.float()))

@torch.inference_mode()
def score_concat(a1,a2):
    x1=a1.to(DEVICE);x2=a2.to(DEVICE)
    x1=x1/x1.norm().clamp_min(1e-12);x2=x2/x2.norm().clamp_min(1e-12)
    sim=.5*(torch.mv(B1N.float(),x1.float())+torch.mv(B2N.float(),x2.float()))
    return reduce_candidate_scores(sim)

print("\n[5/12] POINTER X-RAY + COMPOSITION ARMS...")
RSUM=[];RMEAN=[];RRATIO=[];RMAX=[];RCON=[];PHIT=[];PMASS=[];RATIOS=[]
for i,it in enumerate(ITEMS):
    w=frozen_pointer(A_PACK[i],qA(it["entity"]));sp=A_GOLD[i]
    arg=int(torch.argmax(w));PHIT.append(int(arg in sp));PMASS.append(float(w[sp].sum()))
    raw=A_RAW[i];x=raw[BL][1].float().cpu().permute(1,0,2).reshape(-1,KVD)
    a1=x[sp[0]];a2=x[sp[1]]
    ww=w[sp].float();ww=ww/ww.sum().clamp_min(1e-12);r1=float(ww[0]);r2=float(ww[1]);RATIOS.append((r1,r2))
    ss=score_sum(a1,a2);sr=score_ratio(a1,a2,r1,r2);sm=score_max_token(a1,a2,r1,r2);sc=score_concat(a1,a2)
    rs,_,_=rank(ss,i);rr,_,_=rank(sr,i);rm,_,_=rank(sm,i);rc,_,_=rank(sc,i)
    RSUM.append(rs);RMEAN.append(rs);RRATIO.append(rr);RMAX.append(rm);RCON.append(rc)
    if i<8 or (i+1)%16==0:
        print(f"      [{i+1:03d}/{N}] ptr={PHIT[-1]} mass={PMASS[-1]:.4f} ratio={r1:.3f}/{r2:.3f} | SUM={rs:4d} PTR_RATIO={rr:4d} MAX={rm:4d} CONCAT={rc:4d}")

MSUM=metrics(RSUM);MMEAN=metrics(RMEAN);MRATIO=metrics(RRATIO);MMAX=metrics(RMAX);MCON=metrics(RCON)
print(f"      SUM       R1/R5/R16/MRR : {MSUM[0]:.4f}/{MSUM[1]:.4f}/{MSUM[2]:.4f}/{MSUM[3]:.6f}")
print(f"      MEAN      R1/R5/R16/MRR : {MMEAN[0]:.4f}/{MMEAN[1]:.4f}/{MMEAN[2]:.4f}/{MMEAN[3]:.6f}")
print(f"      PTR_RATIO R1/R5/R16/MRR : {MRATIO[0]:.4f}/{MRATIO[1]:.4f}/{MRATIO[2]:.4f}/{MRATIO[3]:.6f}")
print(f"      MAX_TOKEN R1/R5/R16/MRR : {MMAX[0]:.4f}/{MMAX[1]:.4f}/{MMAX[2]:.4f}/{MMAX[3]:.6f}")
print(f"      CONCATCOS R1/R5/R16/MRR : {MCON[0]:.4f}/{MCON[1]:.4f}/{MCON[2]:.4f}/{MCON[3]:.6f}")
print(f"      POINTER HIT             : {np.mean(PHIT):.4f}")
print(f"      POINTER MASS            : {np.mean(PMASS):.6f}")

print("\n[6/12] POINTER RATIO GEOMETRY...")
R1=np.asarray([x[0] for x in RATIOS]);R2=np.asarray([x[1] for x in RATIOS])
DOM=np.maximum(R1,R2)
print(f"      Mean token-1 mass ratio : {R1.mean():.6f}")
print(f"      Mean token-2 mass ratio : {R2.mean():.6f}")
print(f"      Mean dominant ratio     : {DOM.mean():.6f}")
print(f"      Median dominant ratio   : {np.median(DOM):.6f}")
print(f"      Dominant >= .70         : {np.mean(DOM>=.70):.4f}")
print(f"      Dominant >= .80         : {np.mean(DOM>=.80):.4f}")
print(f"      Dominant >= .90         : {np.mean(DOM>=.90):.4f}")

print("\n[7/12] POSITIONAL ORIENTATION DIAGNOSTIC...")
RREV=[]
@torch.inference_mode()
def score_concat_reverse(a1,a2):
    x1=a1.to(DEVICE);x2=a2.to(DEVICE)
    x1=x1/x1.norm().clamp_min(1e-12);x2=x2/x2.norm().clamp_min(1e-12)
    sim=.5*(torch.mv(B2N.float(),x1.float())+torch.mv(B1N.float(),x2.float()))
    return reduce_candidate_scores(sim)
for i,it in enumerate(ITEMS):
    sp=A_GOLD[i];x=A_RAW[i][BL][1].float().cpu().permute(1,0,2).reshape(-1,KVD)
    r,_,_=rank(score_concat_reverse(x[sp[0]],x[sp[1]]),i);RREV.append(r)
MREV=metrics(RREV)
print(f"      CONCAT normal  R1/R5/MRR : {MCON[0]:.4f}/{MCON[1]:.4f}/{MCON[3]:.6f}")
print(f"      CONCAT reverse R1/R5/MRR : {MREV[0]:.4f}/{MREV[1]:.4f}/{MREV[3]:.6f}")
print(f"      Orientation gap R1       : {MCON[0]-MREV[0]:+.4f}")

print("\n[8/12] GOLD-CANDIDATE GEOMETRY DIAGNOSTIC...")
# Diagnostic only: locate the true B span after retrieval scoring to measure
# whether A and B representations of the same two tokens are geometrically aligned.
GCOS1=[];GCOS2=[];GSUM=[];GCON=[]
for i,it in enumerate(ITEMS):
    sp=A_GOLD[i];bp=B_GOLD[i]
    ax=A_RAW[i][BL][1].float().cpu().permute(1,0,2).reshape(-1,KVD)
    # B raw was deleted, reconstruct once OFFLINE for diagnostic only.
    br=forge(it["B"]);bx=br[BL][1].float().cpu().permute(1,0,2).reshape(-1,KVD)
    a1,a2=ax[sp[0]],ax[sp[1]];b1,b2=bx[bp[0]],bx[bp[1]]
    def coss(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-12))
    GCOS1.append(coss(a1,b1));GCOS2.append(coss(a2,b2));GSUM.append(coss(a1+a2,b1+b2))
    GCON.append(.5*(coss(a1,b1)+coss(a2,b2)))
    del br,bx
print(f"      Gold same-token cos #1  : {np.mean(GCOS1):.6f}")
print(f"      Gold same-token cos #2  : {np.mean(GCOS2):.6f}")
print(f"      Gold summed-pair cosine : {np.mean(GSUM):.6f}")
print(f"      Gold positional cosine  : {np.mean(GCON):.6f}")

print("\n[9/12] ARM COMPARISON...")
ARMS={"SUM":MSUM,"MEAN":MMEAN,"PTR_RATIO":MRATIO,"MAX_TOKEN":MMAX,"CONCATCOS":MCON}
for k,m in ARMS.items():print(f"      {k:10s}: R1={m[0]:.4f} R5={m[1]:.4f} R16={m[2]:.4f} MRR={m[3]:.6f}")
best=max(ARMS,key=lambda k:(ARMS[k][0],ARMS[k][1],ARMS[k][3]))
BEST=ARMS[best]
print("      BEST PREDECLARED ARM:",best)
print(f"      BEST R1            : {BEST[0]:.4f}")
print(f"      Gain vs TEST529 L2 : {BEST[0]-0.0380:+.4f}")

print("\n[10/12] DIAGNOSTIC GATES...")
GATES={
"POINTER_STILL_INTACT":float(np.mean(PHIT))>=.95,
"DECISIVE_RECOVERY":BEST[0]>=.85,
"STRONG_RECOVERY":BEST[0]>=.95,
"R5_RECOVERY":BEST[1]>=.99}
for k,v in GATES.items():print(f"      {k:24s}: {'PASS' if v else 'FAIL'}")

print("\n[11/12] PROTOCOL AUDIT...")
S1=sentinel();WEIGHT_OK=S0==S1
print("      Weight sentinel                     :","PASS" if WEIGHT_OK else "FAIL")
print("      Trainable tensors                   :",sum(int(p.requires_grad) for p in model.parameters()))
print("      Pointer                             : L28H00 unchanged")
print("      Address                             : L00-V uncentered unchanged")
print("      Identity                            : L=2 only")
print("      Head/layer/address discovery        : NONE")
print("      Candidate B windows                 : ALL contiguous width-2")
print("      Gold B window used for retrieval    : NO")
print("      Gold B span used in diagnostics     : YES, after scoring only")
print("      Token-ID routing                    : NO")
print("      Decoded identity routing            : NO")
print("      Training / LoRA / DRA               : NONE")
print("      Learned router/scorer               : NONE")
print("      Query-time candidate-B forwards     : 0")
GATES["WEIGHT_SENTINEL"]=WEIGHT_OK

print("\n[12/12] FINAL...")
if not WEIGHT_OK:
    VERDICT="TEST530_PROTOCOL_FAIL"
elif BEST[0]>=.95 and BEST[1]>=.99:
    VERDICT="TEST530_L2_COMPOSITION_LAW_STRONG_CANDIDATE"
elif BEST[0]>=.85:
    VERDICT="TEST530_L2_COMPOSITION_LAW_CANDIDATE"
else:
    VERDICT="TEST530_SIMPLE_COMPOSITION_REJECTED"

RESULT={"test":TEST,"parents":{"528":PARENT528,"529":PARENT529},"lock_sha":LOCK_SHA,
"arms":{k:{"r1":v[0],"r5":v[1],"r16":v[2],"mrr":v[3]} for k,v in ARMS.items()},
"reverse":{"r1":MREV[0],"r5":MREV[1],"mrr":MREV[3]},
"pointer_hit":float(np.mean(PHIT)),"pointer_mass":float(np.mean(PMASS)),
"pointer_dominance":{"mean":float(DOM.mean()),"median":float(np.median(DOM))},
"gold_geometry":{"token1":float(np.mean(GCOS1)),"token2":float(np.mean(GCOS2)),
                 "sum":float(np.mean(GSUM)),"positional":float(np.mean(GCON))},
"best_arm":best,"gates":GATES,"verdict":VERDICT}
RESULT_SHA=hashlib.sha256(json.dumps(RESULT,sort_keys=True,separators=(",",":")).encode()).hexdigest()

print("\n"+"="*176)
print("TEST530 FINAL RESULT — AKBASCORE MAM · VARAN 1 · L2 COMPOSITION X-RAY")
print("="*176)
print("MODEL                              :",MODEL_ID,"· frozen")
print("PARENTS                            : TEST528 + TEST529")
print("PANEL                              :",N,"L=2 independent memories")
print("POINTER                            : L28H00 · UNCHANGED")
print("ADDRESS                            : L00-V · UNCENTERED · 1024D · UNCHANGED")
print("-"*176)
for k,m in ARMS.items():print(f"{k:35s}: R1={m[0]:.4f} R5={m[1]:.4f} R16={m[2]:.4f} MRR={m[3]:.6f}")
print(f"{'CONCAT_REVERSE':35s}: R1={MREV[0]:.4f} R5={MREV[1]:.4f} R16={MREV[2]:.4f} MRR={MREV[3]:.6f}")
print("-"*176)
print(f"POINTER HIT                        : {np.mean(PHIT):.4f}")
print(f"POINTER MASS                       : {np.mean(PMASS):.6f}")
print(f"MEAN DOMINANT TOKEN RATIO          : {DOM.mean():.6f}")
print(f"GOLD TOKEN-1 COSINE                : {np.mean(GCOS1):.6f}")
print(f"GOLD TOKEN-2 COSINE                : {np.mean(GCOS2):.6f}")
print(f"GOLD SUM COSINE                    : {np.mean(GSUM):.6f}")
print(f"GOLD POSITIONAL COSINE             : {np.mean(GCON):.6f}")
print("-"*176)
print("BEST PREDECLARED ARM               :",best)
print(f"BEST R1                            : {BEST[0]:.4f}")
print(f"GAIN VS TEST529 L2                 : {BEST[0]-0.0380:+.4f}")
for k,v in GATES.items():print(f"{k:35s}: {'PASS' if v else 'FAIL'}")
print("-"*176)
print("WEIGHT SENTINEL                    :","PASS" if WEIGHT_OK else "FAIL")
print("QUERY-TIME CANDIDATE-B FORWARDS    : 0")
print("TEST528 RESULT SHA                 :",PARENT528)
print("TEST529 RESULT SHA                 :",PARENT529)
print("TEST530 PRE-RESULT LOCK            :",LOCK_SHA)
print("TEST530 RESULT SHA                 :",RESULT_SHA)
print("TOTAL TEST TIME                    :",f"{time.perf_counter()-T0:.2f}s")
print("VERDICT                            :",VERDICT)
if VERDICT=="TEST530_L2_COMPOSITION_LAW_STRONG_CANDIDATE":
    print("NEXT                               : validate winning composition independently in TEST531")
elif VERDICT=="TEST530_L2_COMPOSITION_LAW_CANDIDATE":
    print("NEXT                               : preserve result; independently validate candidate before extending to L3/L4")
else:
    print("NEXT                               : simple L00-V composition rejected; preserve result and inspect contextual transformation")
print("="*176)

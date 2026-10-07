# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST527
# MISTRAL NATIVE SINGLE-TOKEN 128-WAY SCALE DIAGNOSTIC
#
# PARENT:
#   TEST525 -> L28H00 -> L00-V UNCENTERED calibrated
#   TEST526 -> failed after identity representation changed to
#              highly structured 8-token ZXxxxxxQ codes
#
# PURPOSE:
#   Keep TEST525 Mistral coordinates FROZEN.
#   Restore native single-token seal geometry.
#   Scale to 128 candidates without requiring 768 unique tokens.
#
# DESIGN:
#   128 globally unique TARGET seals
#   remaining native single-token seals = controlled distractor pool
#   B target seal unique per candidate
#   B distractors shared/reused under deterministic balanced schedule
#   fixed natural class labels; class identity is NOT retrieval address
#
# FROZEN:
#   POINTER = L28H00
#   ADDRESS = L00-V UNCENTERED
#   MATCH   = max cosine
#
# NO:
#   layer/head discovery
#   address discovery
#   tuning
#   training
#   D120/PCA/router/scorer
#   token-ID/text/LM-head address
#   query-time B forward
#
# PRIMARY QUESTION:
#   Does TEST525's native single-token MAM geometry survive 128-way competition?
#
# CONTROLS:
#   uniform
#   shifted
#   no-A zero
#   counterfactual
#
# IMPORTANT:
#   TEST527 is a diagnostic/calibration continuation, NOT the final external seal.

import os,sys,subprocess,importlib.util,random,time,hashlib,json,gc,re,math
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

TEST="527";SEED=527;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";N=128;DEVICE=torch.device("cuda")
FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
PTR_L=28;PTR_H=0;BL=0;CENTER=False;BK="V"
PARENT525="a656140e1b484212ef8956c2cbde6fe2df3b76eb19ca901eccd5f66f8442f209"
PARENT526="8df772191e27c4cee99bfe48aa1d15fb2869c9f2d732b5bc88c0a60829ebff98"
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*176)
print("TEST527 — AKBASCORE MAM · MISTRAL NATIVE SINGLE-TOKEN 128-WAY SCALE DIAGNOSTIC")
print("FROZEN L28H00 → L00-V UNCENTERED · 128 UNIQUE TARGET SEALS · NO COORDINATE DISCOVERY")
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
print(f"      {NL}L H={H} QH={QH} KVH={KVH} HD={HD} REP={REP} ADDRESS={KVD}")
print(f"      POINTER LOCK : L{PTR_L:02d}H{PTR_H:02d}")
print(f"      ADDRESS LOCK : L{BL:02d}-{BK} · UNCENTERED")
print("      Model frozen · BF16 · SDPA · trainable=0")

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

SYL1=["Aq","Bryn","Cav","Dex","Eln","Frax","Gyr","Hex","Iln","Jor","Kex","Lyr","Mav","Nex","Oryn","Pax",
"Qyr","Rex","Savn","Tov","Uln","Vex","Wyr","Xav","Yex","Zyr","Axl","Bov","Cyn","Drex","Evr","Fyn"]
SYL2=["adar","bren","cyr","dax","elor","fyn","grel","hyn","ivar","jor","kyr","lor","myn","nex","or","pyr",
"qen","rix","sor","tyn","ul","vyr","wen","xir","yor","zen"]

def build_names(n):
    out=[]
    for a in SYL1:
        for b in SYL2:
            x=a+b
            if x not in out:out.append(x)
            if len(out)>=n:return out
    raise RuntimeError(f"Name pool too small: need {n}, found {len(out)}.")

def native_code_pool():
    out=[]
    for tid in range(VOC):
        s=tok.decode([tid],skip_special_tokens=False)
        if not re.fullmatch(r"[A-Z]{3,8}",s):continue
        if enc(s)!=[tid]:continue
        if s not in out:out.append(s)
    return out

NAMES=build_names(N*3)
POOL=native_code_pool()
rr=random.Random(SEED+100000);rr.shuffle(POOL)
if len(POOL)<N+32:raise RuntimeError(f"Need at least {N+32} native codes, found {len(POOL)}.")
TARGETS=POOL[:N]
DISTRACT=POOL[N:]
if len(set(TARGETS))!=N or set(TARGETS)&set(DISTRACT):raise RuntimeError("Native code partition failure.")
print(f"      Native single-token pool : {len(POOL)}")
print(f"      Unique target seals      : {len(TARGETS)}")
print(f"      Distractor pool          : {len(DISTRACT)}")
print("      All target seals         : exactly 1 native token")

CLASS_LABELS=["ALPHA","BETA","GAMMA"]
if not all(len(enc(x))>=1 for x in CLASS_LABELS):raise RuntimeError("Class label tokenization failure.")

def dseal(i,k):
    x=DISTRACT[(i*7+k*29)%len(DISTRACT)]
    if x==TARGETS[i]:raise RuntimeError("Target/distractor collision.")
    return x

def make_items():
    items=[]
    for i in range(N):
        ents=NAMES[3*i:3*i+3]
        gold=TARGETS[i];d1=dseal(i,0);d2=dseal(i,1)
        if len({gold,d1,d2})!=3:d2=DISTRACT[(i*7+31)%len(DISTRACT)]
        if len({gold,d1,d2})!=3:raise RuntimeError(f"A seal collision at item {i}.")
        aseals=[gold,d1,d2]
        bd1=DISTRACT[(i*11+17)%len(DISTRACT)]
        bd2=DISTRACT[(i*13+43)%len(DISTRACT)]
        tries=0
        while len({gold,bd1,bd2})<3:
            tries+=1;bd2=DISTRACT[(i*13+43+tries)%len(DISTRACT)]
            if tries>len(DISTRACT):raise RuntimeError("B distractor construction failure.")
        bseals=[gold,bd1,bd2]
        ra=[f"Instrument {ents[j]} carries seal {aseals[j]}." for j in range(3)]
        rb=[f"Seal {bseals[j]} corresponds to routing class {CLASS_LABELS[j]}." for j in range(3)]
        r=random.Random(SEED+i*101);r.shuffle(ra);r.shuffle(rb)
        items.append({"id":i+1,"A":" ".join(ra),"B":" ".join(rb),
        "gold":{"entity":ents[0],"seal":gold},"A_seals":aseals,"A_distractors":[d1,d2],"B_seals":bseals})
    return items

ITEMS=make_items()
assert len(set(x["gold"]["seal"] for x in ITEMS))==N
for i,it in enumerate(ITEMS):
    assert len(enc(it["gold"]["seal"]))==1
    assert it["gold"]["seal"] in it["B_seals"]
    assert all(x!=it["gold"]["seal"] for x in it["A_distractors"])
    assert sum(it["gold"]["seal"]==x["gold"]["seal"] for x in ITEMS)==1

LOCK={"test":TEST,"parent525":PARENT525,"parent526":PARENT526,"model":MODEL_ID,"seed":SEED,"N":N,
"purpose":"native_single_token_128_way_scale_diagnostic",
"pointer":{"layer":PTR_L,"head":PTR_H},"address":{"layer":BL,"kind":BK,"center":CENTER,"dimension":KVD,"match":"max_cosine"},
"identity":{"target_seals":"128 globally unique native single-token codes","distractors":"reused deterministic native single-token pool",
"class_labels":"fixed non-routing semantic filler"},
"selection":"NONE; TEST525 coordinates frozen","controls":["uniform","shifted","noA_zero","counterfactual"],
"forbidden":["HEAD_SCAN","LAYER_SCAN","ADDRESS_SCAN","TOKEN_ID_ADDRESS","DECODED_SEAL_ADDRESS","LM_HEAD_ADDRESS",
"GOLD_POSITION_ADDRESS","QUERY_TIME_B_FORWARD","TRAINING","DRA","LEARNED_ROUTER","POSTHOC_SELECTION"]}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      TEST527 LOCK:",LOCK_SHA)

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

def rope_k(k,pos,L):
    T=len(pos);kk=k.unsqueeze(0)
    dummy=torch.zeros((1,KVH,T,HD),device=DEVICE,dtype=k.dtype)
    c,s=rope_cos_sin(dummy,pos)
    _,kr=apply_rotary_pos_emb(dummy,kk,c,s,unsqueeze_dim=1)
    return kr[0]

def install(raw):
    T=raw[0][0].shape[1]
    return [(rope_k(k,list(range(T)),L),v) for L,(k,v) in enumerate(raw)],T,T

def cache_of(inst):
    try:c=DynamicCache(config=cfg)
    except TypeError:c=DynamicCache()
    for L,(k,v) in enumerate(inst):c.update(k.unsqueeze(0),v.unsqueeze(0),L)
    return c

def qA(e):return f"What seal does instrument {e} carry? Give only the exact seal."

# Diagnostic-only span locator.
# Exact standalone ASCII seal match prevents e.g. COP matching inside another word.
# Retrieval itself never uses this function.
def gold_token_positions(text,seal):
    pat=r"(?<![A-Za-z])"+re.escape(seal)+r"(?![A-Za-z])"
    hits=[m.span() for m in re.finditer(pat,text)]
    if len(hits)!=1:raise RuntimeError(f"Exact seal occurrence !=1: {seal} | hits={hits}")
    a,b=hits[0]
    z=tok(text+SEP,add_special_tokens=False,return_offsets_mapping=True)
    pos=[]
    for j,(x,y) in enumerate(z["offset_mapping"]):
        if y>a and x<b:pos.append(j+1)
    if not pos:raise RuntimeError(f"No gold span: {seal}")
    return pos

print("\n[2/11] OFFLINE FORGE — 128 native-single-token items...")
A_RAW=[];B_RAW=[];A_PACK=[];GOLD_SPANS=[];META=[]
for i,it in enumerate(ITEMS):
    ar=forge(it["A"]);br=forge(it["B"])
    A_RAW.append(ar);B_RAW.append(br);A_PACK.append(install(ar))
    GOLD_SPANS.append(gold_token_positions(it["A"],it["gold"]["seal"]))
    META.append({"entity":it["gold"]["entity"],"gold_seal":it["gold"]["seal"],
                 "d1":it["A_distractors"][0],"d2":it["A_distractors"][1]})
    if i<5:print(f"      ITEM {i+1:03d} | A={A_PACK[-1][1]} B={br[0][0].shape[1]} | target={it['gold']['seal']}")
print("      256 independent raw numeric memories forged.")
print("      Coordinates unchanged from TEST525.")

print("\n[3/11] Building frozen L00-V B bank...")
B_MATS=[]
for raw in B_RAW:
    x=raw[BL][1][:,1:,:].float().cpu()
    B_MATS.append(x.permute(1,0,2).reshape(x.shape[1],-1).contiguous())
assert len(B_MATS)==N and all(M.shape[1]==KVD for M in B_MATS)
del B_RAW
for it in ITEMS:it["A"]=None;it["B"]=None
gc.collect();torch.cuda.empty_cache()
print(f"      B matrices              : {len(B_MATS)}")
print(f"      Address dimension       : {KVD}")
print("      Query-time B forwards   : 0")

@torch.inference_mode()
def query_forward(packet,q):
    inst,Tm,P=packet;c=cache_of(inst);ids=enc(FMT.format(q=q));n=len(ids)
    pos=torch.arange(P,P+n,device=DEVICE).unsqueeze(0)
    mask=torch.ones((1,Tm+n),device=DEVICE,dtype=torch.long)
    o=model(input_ids=torch.tensor([ids],device=DEVICE),past_key_values=c,
            attention_mask=mask,position_ids=pos,use_cache=False,
            output_hidden_states=True,return_dict=True)
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
    score[:,0]=-torch.inf;W=torch.softmax(score,dim=-1).detach().cpu()
    W[:,0]=0;W=W/W.sum(dim=1,keepdim=True).clamp_min(1e-12)
    return W[PTR_H]

def cos_rows(v,M):
    v=v.float();M=M.float();nv=v.norm()
    if float(nv)<=1e-12:return torch.zeros(M.shape[0],dtype=torch.float32)
    v=v/nv;M=M/M.norm(dim=1,keepdim=True).clamp_min(1e-8);return M@v

def pointer_vector(raw,w):
    x=raw[BL][1].float().cpu()
    if x.shape[1]!=len(w):raise RuntimeError(f"Pointer length {len(w)} != V length {x.shape[1]}")
    return (x*w[None,:,None]).sum(dim=1).reshape(-1)

def B_scores(p):return [float(cos_rows(p,M).max()) for M in B_MATS]

def rank(sc,gold):
    order=sorted(range(N),key=lambda j:(-float(sc[j]),j))
    return order.index(gold)+1,order[0],order

def metrics(r):
    a=np.asarray(r,float)
    return float(np.mean(a==1)),float(np.mean(a<=5)),float(np.mean(a<=16)),float(np.mean(1/a))

def uniform_pointer(T):
    w=torch.ones(T,dtype=torch.float32);w[0]=0
    return w/w.sum().clamp_min(1e-12)

def shifted_pointer(w):
    n=len(w)-1
    if n<=1:return w.clone()
    z=torch.zeros_like(w);z[1:]=torch.roll(w[1:].clone(),shifts=max(1,n//2),dims=0)
    return z/z.sum().clamp_min(1e-12)

print("\n[4/11] PRIMARY — frozen L28H00 → L00-V...")
PTR=[];RPRI=[];SEL=[];PHIT=[];PMASS=[];PRANK=[];MARG=[]
for i in range(N):
    w=frozen_pointer(A_PACK[i],qA(META[i]["entity"]));PTR.append(w)
    p=pointer_vector(A_RAW[i],w);sc=B_scores(p);r,se,order=rank(sc,i)
    RPRI.append(r);SEL.append(se);MARG.append(float(sc[order[0]]-sc[order[1]]))
    sp=GOLD_SPANS[i];arg=int(torch.argmax(w));PHIT.append(int(arg in sp));PMASS.append(float(w[sp].sum()))
    best=float(w[sp].max());PRANK.append(1+int((w[1:]>best).sum()))
    print(f"      [{i+1:03d}/128] rank={r:3d} selected={se+1:03d} expected={i+1:03d} | ptr-hit={PHIT[-1]} mass={PMASS[-1]:.4f} margin={MARG[-1]:+.6f}")
MP=metrics(RPRI)
print(f"      PRIMARY R1/R5/R16/MRR : {MP[0]:.4f} / {MP[1]:.4f} / {MP[2]:.4f} / {MP[3]:.6f}")
print(f"      POINTER HIT            : {sum(PHIT)}/{N} = {np.mean(PHIT):.4f}")
print(f"      POINTER MASS           : {np.mean(PMASS):.6f}")
print(f"      POINTER MEDIAN RANK    : {np.median(PRANK):.1f}")
print(f"      MEAN TOP1-TOP2 MARGIN : {np.mean(MARG):+.6f}")

print("\n[5/11] CONTROLS — uniform / shifted / no-A...")
RU=[];RS=[];RN=[];ZERO=torch.zeros(KVD,dtype=torch.float32)
for i in range(N):
    r,_,_=rank(B_scores(pointer_vector(A_RAW[i],uniform_pointer(len(PTR[i])))),i);RU.append(r)
    r,_,_=rank(B_scores(pointer_vector(A_RAW[i],shifted_pointer(PTR[i]))),i);RS.append(r)
    r,_,_=rank(B_scores(ZERO),i);RN.append(r)
MU=metrics(RU);MS=metrics(RS);MN=metrics(RN)
print(f"      PRIMARY : R1={MP[0]:.4f} R5={MP[1]:.4f} MRR={MP[3]:.6f}")
print(f"      UNIFORM : R1={MU[0]:.4f} R5={MU[1]:.4f} MRR={MU[3]:.6f}")
print(f"      SHIFTED : R1={MS[0]:.4f} R5={MS[1]:.4f} MRR={MS[3]:.6f}")
print(f"      NO-A    : R1={MN[0]:.4f} R5={MN[1]:.4f} MRR={MN[3]:.6f}")

print("\n[6/11] COUNTERFACTUAL FOLLOW — same frozen mechanism...")
RCF=[];CFHIT=[];CFMASS=[]
for i in range(N):
    j=(i+37)%N;target=META[j]["gold_seal"]
    ents=[META[i]["entity"],NAMES[3*i+1],NAMES[3*i+2]]
    seals=[target,META[i]["d1"],META[i]["d2"]]
    rows=[f"Instrument {ents[x]} carries seal {seals[x]}." for x in range(3)]
    rgen=random.Random(SEED+9000+i);rgen.shuffle(rows);cfA=" ".join(rows)
    raw=forge(cfA);pack=install(raw);w=frozen_pointer(pack,qA(META[i]["entity"]))
    p=pointer_vector(raw,w);r,se,_=rank(B_scores(p),j);RCF.append(r)
    sp=gold_token_positions(cfA,target);arg=int(torch.argmax(w))
    CFHIT.append(int(arg in sp));CFMASS.append(float(w[sp].sum()))
    print(f"      [{i+1:03d}/128] target={j+1:03d} rank={r:3d} selected={se+1:03d} follow={int(se==j)} | ptr-hit={CFHIT[-1]} mass={CFMASS[-1]:.4f}")
    del raw,pack,w,p
MC=metrics(RCF)
print(f"      CF R1/R5/R16/MRR       : {MC[0]:.4f} / {MC[1]:.4f} / {MC[2]:.4f} / {MC[3]:.6f}")
print(f"      CF POINTER HIT          : {sum(CFHIT)}/{N} = {np.mean(CFHIT):.4f}")
print(f"      CF POINTER MASS         : {np.mean(CFMASS):.6f}")

print("\n[7/11] Scale diagnostics...")
R=np.asarray(RPRI);HIT=np.asarray(PHIT,dtype=bool);OK=R==1
def sm(x):return float(np.mean(x)) if len(x) else float("nan")
print(f"      Retrieval R1 | ptr-hit       : {sm(OK[HIT]):.4f} n={int(HIT.sum())}")
print(f"      Retrieval R1 | ptr-miss      : {sm(OK[~HIT]):.4f} n={int((~HIT).sum())}")
print(f"      PRIMARY - UNIFORM            : {MP[0]-MU[0]:+.4f}")
print(f"      PRIMARY - SHIFTED            : {MP[0]-MS[0]:+.4f}")
print(f"      PRIMARY - NO-A               : {MP[0]-MN[0]:+.4f}")
print(f"      Unique selected B candidates : {len(set(SEL))}/{N}")
print(f"      Most selected candidate count: {max(SEL.count(x) for x in set(SEL))}")

print("\n[8/11] Wilson intervals...")
def wilson(k,n,z=1.959963984540054):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h
for name,r in (("PRIMARY",RPRI),("UNIFORM",RU),("SHIFTED",RS),("NO_A",RN),("CF",RCF)):
    k=sum(int(x==1) for x in r);lo,hi=wilson(k,len(r))
    print(f"      {name:8s}: {k:3d}/{len(r)} = {k/len(r):.4f} | 95% Wilson [{lo:.4f},{hi:.4f}]")

print("\n[9/11] Diagnostic decision...")
if MP[0]>=.85 and MC[0]>=.80 and MP[0]-MS[0]>=.60:
    VERDICT="NATIVE_SINGLE_TOKEN_128WAY_SCALE_ESTABLISHED"
elif MP[0]>=.65 and MP[0]-MS[0]>=.40:
    VERDICT="NATIVE_SINGLE_TOKEN_128WAY_SIGNAL_STRONG_BUT_NOT_SEALED"
elif MP[0]>=.30 and MP[0]-MS[0]>=.20:
    VERDICT="NATIVE_SINGLE_TOKEN_SCALE_SIGNAL_PRESENT"
else:
    VERDICT="NATIVE_SINGLE_TOKEN_128WAY_SCALE_NOT_ESTABLISHED"
print("      VERDICT:",VERDICT)

print("\n[10/11] Protocol audit...")
S1=sentinel()
print("      Model frozen                         :",S0==S1)
print("      Trainable tensors                    :",sum(int(p.requires_grad) for p in model.parameters()))
print("      Pointer                              : L28H00 frozen")
print("      Address                              : L00-V uncentered frozen")
print("      Coordinate discovery                 : NONE")
print("      Address discovery                    : NONE")
print("      Target seals                         : 128 unique native single-token")
print("      Distractor seals                     : native single-token reused")
print("      Retrieval uses token IDs/text        : NO")
print("      Gold span used in retrieval          : NO")
print("      Gold span diagnostic                 : exact standalone seal only")
print("      Query-time candidate-B forwards      : 0")
print("      Training / LoRA / optimizer / DRA    : NONE")
print("      TEST526 structured 8-token identities: REMOVED")

print("\n[11/11] FINAL...")
print("\n"+"="*176)
print("TEST527 FINAL RESULT — MISTRAL NATIVE SINGLE-TOKEN 128-WAY SCALE DIAGNOSTIC")
print("="*176)
print("MODEL                              :",MODEL_ID,"· frozen")
print("POINTER                            : L28H00")
print("ADDRESS                            : L00-V · UNCENTERED · 1024D")
print("TARGET IDENTITY                    : 128 UNIQUE NATIVE SINGLE-TOKEN SEALS")
print("CANDIDATES                         : 128")
print("-"*176)
print(f"PRIMARY                            : R1={MP[0]:.4f} R5={MP[1]:.4f} R16={MP[2]:.4f} MRR={MP[3]:.6f}")
print(f"UNIFORM                            : R1={MU[0]:.4f} R5={MU[1]:.4f} R16={MU[2]:.4f} MRR={MU[3]:.6f}")
print(f"SHIFTED                            : R1={MS[0]:.4f} R5={MS[1]:.4f} R16={MS[2]:.4f} MRR={MS[3]:.6f}")
print(f"NO-A                               : R1={MN[0]:.4f} R5={MN[1]:.4f} R16={MN[2]:.4f} MRR={MN[3]:.6f}")
print(f"COUNTERFACTUAL                     : R1={MC[0]:.4f} R5={MC[1]:.4f} R16={MC[2]:.4f} MRR={MC[3]:.6f}")
print(f"POINTER HIT                        : {sum(PHIT)}/{N} = {np.mean(PHIT):.4f}")
print(f"POINTER MASS                       : {np.mean(PMASS):.6f}")
print(f"CF POINTER HIT                     : {sum(CFHIT)}/{N} = {np.mean(CFHIT):.4f}")
print(f"CF POINTER MASS                    : {np.mean(CFMASS):.6f}")
print(f"UNIQUE SELECTED B                  : {len(set(SEL))}/{N}")
print(f"MEAN TOP1-TOP2 MARGIN              : {np.mean(MARG):+.6f}")
print("-"*176)
print("WEIGHT SENTINEL                    :","PASS" if S0==S1 else"FAIL")
print("QUERY-TIME CANDIDATE-B FORWARDS    : 0")
print("TEST525 LOCK                       :",PARENT525)
print("TEST526 LOCK                       :",PARENT526)
print("TEST527 LOCK                       :",LOCK_SHA)
print("TOTAL TEST TIME                    :",f"{time.perf_counter()-T0:.2f}s")
print("VERDICT                            :",VERDICT)
if VERDICT=="NATIVE_SINGLE_TOKEN_128WAY_SCALE_ESTABLISHED":
    print("NEXT                               : preserve this identity regime and build final untouched external seal")
else:
    print("NEXT                               : diagnose scale/address competition without changing L28H00 → L00-V")
print("="*176)

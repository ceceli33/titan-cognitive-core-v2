# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST528
# MISTRAL FINAL UNTOUCHED EXTERNAL SEAL
#
# LOCKED BEFORE THIS TEST:
#   POINTER = L28H00
#   ADDRESS = L00-V UNCENTERED
#   ADDRESS DIM = 1024
#   MATCH = max cosine over precomputed B address rows
#   IDENTITY REGIME = native single-token seals
#
# PARENT RESULTS:
#   TEST525 -> Mistral calibration lock
#   TEST526 -> structured 8-token identity failure
#   TEST527 -> native single-token 128-way scale established
#              PRIMARY 126/128 = 98.44%
#              CF      126/128 = 98.44%
#
# TEST528 PURPOSE:
#   Final untouched external replication.
#   Fresh deterministic seed/panel.
#   No coordinate discovery.
#   No tuning.
#   No post-hoc selection.
#
# PREDECLARED SEAL GATES:
#   PRIMARY R1 >= 0.90
#   COUNTERFACTUAL R1 >= 0.90
#   PRIMARY - SHIFTED R1 >= 0.70
#   PRIMARY - NO-A R1 >= 0.70
#   POINTER HIT >= 0.85
#   CF POINTER HIT >= 0.85
#   WEIGHT SENTINEL PASS
#   QUERY-TIME B FORWARDS = 0
#
# NOTE:
#   Gold spans are diagnostic only and never participate in retrieval.

import os,sys,subprocess,importlib.util,random,time,hashlib,json,gc,re,math
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

TEST="528";SEED=528528;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";N=128;DEVICE=torch.device("cuda")
FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
PTR_L=28;PTR_H=0;BL=0;CENTER=False;BK="V"
TH_PRIMARY=.90;TH_CF=.90;TH_CTRL_GAP=.70;TH_PTR=.85;TH_CF_PTR=.85
PARENT525="a656140e1b484212ef8956c2cbde6fe2df3b76eb19ca901eccd5f66f8442f209"
PARENT526="8df772191e27c4cee99bfe48aa1d15fb2869c9f2d732b5bc88c0a60829ebff98"
PARENT527="eed9acbdc7d307935872c9d268fc99a71520bc47dbb1abf107bf53a26c2e89d7"
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*176)
print("TEST528 — AKBASCORE MAM · MISTRAL FINAL UNTOUCHED EXTERNAL SEAL")
print("FRESH 128 · FROZEN L28H00 → L00-V UNCENTERED · NATIVE SINGLE-TOKEN · NO DISCOVERY / NO TUNING")
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
print(f"      {NL}L H={H} QH={QH} KVH={KVH} HD={HD} REP={REP} ADDRESS={KVD}")
print(f"      POINTER LOCK : L{PTR_L:02d}H{PTR_H:02d}")
print(f"      ADDRESS LOCK : L{BL:02d}-{BK} · UNCENTERED")
print(f"      PREDECLARED  : primary>={TH_PRIMARY:.2f} cf>={TH_CF:.2f} gaps>={TH_CTRL_GAP:.2f} ptr>={TH_PTR:.2f} cfptr>={TH_CF_PTR:.2f}")
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
    for a in reversed(SYL1):
        for b in reversed(SYL2):
            x=a+b
            if x not in out:out.append(x)
    rr=random.Random(SEED+70001);rr.shuffle(out)
    if len(out)<n:raise RuntimeError(f"Name pool too small: need {n}, found {len(out)}.")
    return out[:n]

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
TARGETS=POOL[:N];DISTRACT=POOL[N:]
if len(set(TARGETS))!=N or set(TARGETS)&set(DISTRACT):raise RuntimeError("Native code partition failure.")
if not all(len(enc(x))==1 for x in TARGETS+DISTRACT):raise RuntimeError("Non-single-token seal detected.")
print(f"      Native single-token pool : {len(POOL)}")
print(f"      Unique target seals      : {len(TARGETS)}")
print(f"      Distractor pool          : {len(DISTRACT)}")
print("      Fresh deterministic panel seed:",SEED)

CLASS_LABELS=["ALPHA","BETA","GAMMA"]
if not all(len(enc(x))>=1 for x in CLASS_LABELS):raise RuntimeError("Class label tokenization failure.")

def pick_distinct(pool,indices,forbidden):
    for z in indices:
        x=pool[z%len(pool)]
        if x not in forbidden:return x
    raise RuntimeError("Unable to select distinct distractor.")

def make_items():
    items=[]
    for i in range(N):
        ents=NAMES[3*i:3*i+3];gold=TARGETS[i]
        d1=pick_distinct(DISTRACT,[i*17+5+j for j in range(len(DISTRACT))],{gold})
        d2=pick_distinct(DISTRACT,[i*19+31+j for j in range(len(DISTRACT))],{gold,d1})
        bd1=pick_distinct(DISTRACT,[i*23+11+j for j in range(len(DISTRACT))],{gold})
        bd2=pick_distinct(DISTRACT,[i*29+47+j for j in range(len(DISTRACT))],{gold,bd1})
        aseals=[gold,d1,d2];bseals=[gold,bd1,bd2]
        ra=[f"Instrument {ents[j]} carries seal {aseals[j]}." for j in range(3)]
        rb=[f"Seal {bseals[j]} corresponds to routing class {CLASS_LABELS[j]}." for j in range(3)]
        random.Random(SEED+i*131+17).shuffle(ra)
        random.Random(SEED+i*137+29).shuffle(rb)
        items.append({"id":i+1,"A":" ".join(ra),"B":" ".join(rb),
        "gold":{"entity":ents[0],"seal":gold},"A_seals":aseals,"A_distractors":[d1,d2],"B_seals":bseals})
    return items

ITEMS=make_items()
assert len(ITEMS)==N
assert len(set(x["gold"]["seal"] for x in ITEMS))==N
for it in ITEMS:
    assert len(enc(it["gold"]["seal"]))==1
    assert it["gold"]["seal"] in it["B_seals"]
    assert len(set(it["A_seals"]))==3 and len(set(it["B_seals"]))==3
    assert all(x!=it["gold"]["seal"] for x in it["A_distractors"])

LOCK={"test":TEST,"parents":{"525":PARENT525,"526":PARENT526,"527":PARENT527},"model":MODEL_ID,"seed":SEED,"N":N,
"status":"FINAL_UNTOUCHED_EXTERNAL_SEAL","pointer":{"layer":PTR_L,"head":PTR_H},
"address":{"layer":BL,"kind":BK,"center":CENTER,"dimension":KVD,"match":"max_cosine"},
"identity":{"targets":"128 globally unique native single-token","distractors":"deterministic native single-token reuse",
"classes":"fixed semantic filler"},"thresholds":{"primary_r1":TH_PRIMARY,"counterfactual_r1":TH_CF,
"primary_shifted_gap":TH_CTRL_GAP,"primary_noA_gap":TH_CTRL_GAP,"pointer_hit":TH_PTR,"cf_pointer_hit":TH_CF_PTR},
"selection":"NONE","controls":["uniform","shifted","noA_zero","counterfactual"],
"forbidden":["HEAD_SCAN","LAYER_SCAN","ADDRESS_SCAN","CENTER_SCAN","TOKEN_ID_ADDRESS","DECODED_SEAL_ADDRESS",
"LM_HEAD_ADDRESS","GOLD_POSITION_ADDRESS","QUERY_TIME_B_FORWARD","TRAINING","LORA","DRA","LEARNED_ROUTER",
"POSTHOC_SELECTION","THRESHOLD_TUNING"]}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      TEST528 LOCK:",LOCK_SHA)

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
    T=len(pos);kk=k.unsqueeze(0);dummy=torch.zeros((1,KVH,T,HD),device=DEVICE,dtype=k.dtype)
    c,s=rope_cos_sin(dummy,pos);_,kr=apply_rotary_pos_emb(dummy,kk,c,s,unsqueeze_dim=1)
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

def gold_token_positions(text,seal):
    pat=r"(?<![A-Za-z])"+re.escape(seal)+r"(?![A-Za-z])"
    hits=[m.span() for m in re.finditer(pat,text)]
    if len(hits)!=1:raise RuntimeError(f"Exact seal occurrence !=1: {seal} | hits={hits}")
    a,b=hits[0];z=tok(text+SEP,add_special_tokens=False,return_offsets_mapping=True);pos=[]
    for j,(x,y) in enumerate(z["offset_mapping"]):
        if y>a and x<b:pos.append(j+1)
    if not pos:raise RuntimeError(f"No gold span: {seal}")
    return pos

print("\n[2/12] OFFLINE FORGE — fresh external 128...")
A_RAW=[];B_RAW=[];A_PACK=[];GOLD_SPANS=[];META=[]
for i,it in enumerate(ITEMS):
    ar=forge(it["A"]);br=forge(it["B"]);A_RAW.append(ar);B_RAW.append(br);A_PACK.append(install(ar))
    GOLD_SPANS.append(gold_token_positions(it["A"],it["gold"]["seal"]))
    META.append({"entity":it["gold"]["entity"],"gold_seal":it["gold"]["seal"],"d1":it["A_distractors"][0],"d2":it["A_distractors"][1]})
    if i<5:print(f"      ITEM {i+1:03d} | A={A_PACK[-1][1]} B={br[0][0].shape[1]} | target={it['gold']['seal']}")
print("      256 independent raw numeric memories forged.")
print("      No TEST528 result used for mechanism selection.")

print("\n[3/12] Building frozen L00-V B bank...")
B_MATS=[]
for raw in B_RAW:
    x=raw[BL][1][:,1:,:].float().cpu()
    B_MATS.append(x.permute(1,0,2).reshape(x.shape[1],-1).contiguous())
assert len(B_MATS)==N and all(M.shape[1]==KVD for M in B_MATS)
del B_RAW
for it in ITEMS:it["A"]=None;it["B"]=None
gc.collect();torch.cuda.empty_cache()
print(f"      B matrices            : {len(B_MATS)}")
print(f"      Address dimension     : {KVD}")
print("      Candidate-B live calls: 0")

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
    score[:,0]=-torch.inf;W=torch.softmax(score,dim=-1).detach().cpu();W[:,0]=0
    W=W/W.sum(dim=1,keepdim=True).clamp_min(1e-12)
    return W[PTR_H]

def cos_rows(v,M):
    v=v.float();M=M.float();nv=v.norm()
    if float(nv)<=1e-12:return torch.zeros(M.shape[0],dtype=torch.float32)
    return (M/M.norm(dim=1,keepdim=True).clamp_min(1e-8))@(v/nv)

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

print("\n[4/12] PRIMARY — untouched external evaluation...")
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

print("\n[5/12] NEGATIVE CONTROLS...")
RU=[];RS=[];RN=[];ZERO=torch.zeros(KVD,dtype=torch.float32)
for i in range(N):
    r,_,_=rank(B_scores(pointer_vector(A_RAW[i],uniform_pointer(len(PTR[i])))),i);RU.append(r)
    r,_,_=rank(B_scores(pointer_vector(A_RAW[i],shifted_pointer(PTR[i]))),i);RS.append(r)
    r,_,_=rank(B_scores(ZERO),i);RN.append(r)
MU=metrics(RU);MS=metrics(RS);MN=metrics(RN)
print(f"      PRIMARY : R1={MP[0]:.4f} R5={MP[1]:.4f} R16={MP[2]:.4f} MRR={MP[3]:.6f}")
print(f"      UNIFORM : R1={MU[0]:.4f} R5={MU[1]:.4f} R16={MU[2]:.4f} MRR={MU[3]:.6f}")
print(f"      SHIFTED : R1={MS[0]:.4f} R5={MS[1]:.4f} R16={MS[2]:.4f} MRR={MS[3]:.6f}")
print(f"      NO-A    : R1={MN[0]:.4f} R5={MN[1]:.4f} R16={MN[2]:.4f} MRR={MN[3]:.6f}")

print("\n[6/12] COUNTERFACTUAL FOLLOW...")
RCF=[];CFSEL=[];CFHIT=[];CFMASS=[]
for i in range(N):
    j=(i+53)%N;target=META[j]["gold_seal"]
    ents=[META[i]["entity"],NAMES[3*i+1],NAMES[3*i+2]]
    seals=[target,META[i]["d1"],META[i]["d2"]]
    rows=[f"Instrument {ents[x]} carries seal {seals[x]}." for x in range(3)]
    random.Random(SEED+90000+i*149).shuffle(rows);cfA=" ".join(rows)
    raw=forge(cfA);pack=install(raw);w=frozen_pointer(pack,qA(META[i]["entity"]))
    p=pointer_vector(raw,w);r,se,_=rank(B_scores(p),j);RCF.append(r);CFSEL.append(se)
    sp=gold_token_positions(cfA,target);arg=int(torch.argmax(w));CFHIT.append(int(arg in sp));CFMASS.append(float(w[sp].sum()))
    print(f"      [{i+1:03d}/128] target={j+1:03d} rank={r:3d} selected={se+1:03d} follow={int(se==j)} | ptr-hit={CFHIT[-1]} mass={CFMASS[-1]:.4f}")
    del raw,pack,w,p
MC=metrics(RCF)
print(f"      CF R1/R5/R16/MRR       : {MC[0]:.4f} / {MC[1]:.4f} / {MC[2]:.4f} / {MC[3]:.6f}")
print(f"      CF POINTER HIT          : {sum(CFHIT)}/{N} = {np.mean(CFHIT):.4f}")
print(f"      CF POINTER MASS         : {np.mean(CFMASS):.6f}")

print("\n[7/12] CAUSAL / FAILURE DIAGNOSTICS...")
R=np.asarray(RPRI);HIT=np.asarray(PHIT,dtype=bool);OK=R==1
def sm(x):return float(np.mean(x)) if len(x) else float("nan")
print(f"      Retrieval R1 | ptr-hit       : {sm(OK[HIT]):.4f} n={int(HIT.sum())}")
print(f"      Retrieval R1 | ptr-miss      : {sm(OK[~HIT]):.4f} n={int((~HIT).sum())}")
print(f"      PRIMARY - UNIFORM            : {MP[0]-MU[0]:+.4f}")
print(f"      PRIMARY - SHIFTED            : {MP[0]-MS[0]:+.4f}")
print(f"      PRIMARY - NO-A               : {MP[0]-MN[0]:+.4f}")
print(f"      Unique primary selections    : {len(set(SEL))}/{N}")
print(f"      Unique CF selections         : {len(set(CFSEL))}/{N}")
print(f"      Primary failures             : {[i+1 for i,r in enumerate(RPRI) if r!=1]}")
print(f"      Counterfactual failures      : {[i+1 for i,r in enumerate(RCF) if r!=1]}")

print("\n[8/12] STATISTICAL SUMMARY...")
def wilson(k,n,z=1.959963984540054):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h
for name,r in (("PRIMARY",RPRI),("UNIFORM",RU),("SHIFTED",RS),("NO_A",RN),("CF",RCF)):
    k=sum(int(x==1) for x in r);lo,hi=wilson(k,len(r))
    print(f"      {name:8s}: {k:3d}/{len(r)} = {k/len(r):.4f} | 95% Wilson [{lo:.4f},{hi:.4f}]")

print("\n[9/12] PREDECLARED GATES...")
GATES={
"PRIMARY_R1":MP[0]>=TH_PRIMARY,
"COUNTERFACTUAL_R1":MC[0]>=TH_CF,
"PRIMARY_SHIFTED_GAP":MP[0]-MS[0]>=TH_CTRL_GAP,
"PRIMARY_NOA_GAP":MP[0]-MN[0]>=TH_CTRL_GAP,
"POINTER_HIT":float(np.mean(PHIT))>=TH_PTR,
"CF_POINTER_HIT":float(np.mean(CFHIT))>=TH_CF_PTR}
for k,v in GATES.items():print(f"      {k:24s}: {'PASS' if v else 'FAIL'}")

print("\n[10/12] REPLAY / DETERMINISM CHECK...")
REPLAY_IDX=[0,31,63,95,127];REPLAY_OK=True
for i in REPLAY_IDX:
    w=frozen_pointer(A_PACK[i],qA(META[i]["entity"]));p=pointer_vector(A_RAW[i],w);sc=B_scores(p);r,se,_=rank(sc,i)
    ok=(r==RPRI[i] and se==SEL[i] and torch.equal(w,PTR[i]));REPLAY_OK&=ok
    print(f"      ITEM {i+1:03d} | rank {RPRI[i]}->{r} | selected {SEL[i]+1:03d}->{se+1:03d} | pointer_exact={torch.equal(w,PTR[i])} | {'PASS' if ok else 'FAIL'}")

print("\n[11/12] PROTOCOL AUDIT...")
S1=sentinel();WEIGHT_OK=S0==S1
print("      Model frozen                         :",WEIGHT_OK)
print("      Trainable tensors                    :",sum(int(p.requires_grad) for p in model.parameters()))
print("      Pointer                              : L28H00 frozen before TEST528")
print("      Address                              : L00-V uncentered frozen before TEST528")
print("      Pointer/head discovery               : NONE")
print("      Address/layer discovery              : NONE")
print("      Centering discovery                  : NONE")
print("      Threshold tuning                     : NONE")
print("      Native target seals                  : 128 unique single-token")
print("      Gold span used in retrieval          : NO")
print("      Gold span diagnostic                 : exact standalone seal only")
print("      Retrieval uses decoded answer        : NO")
print("      Retrieval uses token-ID address      : NO")
print("      Query-time candidate-B forwards      : 0")
print("      Training / LoRA / optimizer / DRA    : NONE")
print("      Post-hoc mechanism selection         : NONE")
print("      Deterministic replay                 :",REPLAY_OK)
GATES["WEIGHT_SENTINEL"]=WEIGHT_OK
GATES["REPLAY"]=REPLAY_OK

print("\n[12/12] FINAL SEAL...")
ALL_PASS=all(GATES.values())
VERDICT="MISTRAL_MAM_FINAL_EXTERNAL_SEAL_PASS" if ALL_PASS else "MISTRAL_MAM_FINAL_EXTERNAL_SEAL_FAIL"
RESULT={"test":TEST,"lock_sha":LOCK_SHA,"primary":{"r1":MP[0],"r5":MP[1],"r16":MP[2],"mrr":MP[3]},
"uniform":{"r1":MU[0],"r5":MU[1],"r16":MU[2],"mrr":MU[3]},
"shifted":{"r1":MS[0],"r5":MS[1],"r16":MS[2],"mrr":MS[3]},
"noA":{"r1":MN[0],"r5":MN[1],"r16":MN[2],"mrr":MN[3]},
"counterfactual":{"r1":MC[0],"r5":MC[1],"r16":MC[2],"mrr":MC[3]},
"pointer_hit":float(np.mean(PHIT)),"cf_pointer_hit":float(np.mean(CFHIT)),
"pointer_mass":float(np.mean(PMASS)),"cf_pointer_mass":float(np.mean(CFMASS)),
"gates":GATES,"verdict":VERDICT}
RESULT_SHA=hashlib.sha256(json.dumps(RESULT,sort_keys=True,separators=(",",":")).encode()).hexdigest()

print("\n"+"="*176)
print("TEST528 FINAL RESULT — AKBASCORE MAM · MISTRAL FINAL UNTOUCHED EXTERNAL SEAL")
print("="*176)
print("MODEL                              :",MODEL_ID,"· frozen")
print("EXTERNAL PANEL                     : 128 fresh deterministic records")
print("POINTER                            : L28H00 · FROZEN")
print("ADDRESS                            : L00-V · UNCENTERED · 1024D · FROZEN")
print("IDENTITY                           : NATIVE SINGLE-TOKEN")
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
print(f"PRIMARY-SHIFTED GAP                : {MP[0]-MS[0]:+.4f}")
print(f"PRIMARY-NO-A GAP                   : {MP[0]-MN[0]:+.4f}")
print(f"MEAN TOP1-TOP2 MARGIN              : {np.mean(MARG):+.6f}")
print("-"*176)
for k,v in GATES.items():print(f"{k:35s}: {'PASS' if v else 'FAIL'}")
print("-"*176)
print("WEIGHT SENTINEL                    :","PASS" if WEIGHT_OK else "FAIL")
print("DETERMINISTIC REPLAY               :","PASS" if REPLAY_OK else "FAIL")
print("QUERY-TIME CANDIDATE-B FORWARDS    : 0")
print("TEST525 LOCK                       :",PARENT525)
print("TEST526 LOCK                       :",PARENT526)
print("TEST527 LOCK                       :",PARENT527)
print("TEST528 PRE-RESULT LOCK            :",LOCK_SHA)
print("TEST528 RESULT SHA                 :",RESULT_SHA)
print("TOTAL TEST TIME                    :",f"{time.perf_counter()-T0:.2f}s")
print("VERDICT                            :",VERDICT)
if ALL_PASS:
    print("STATUS                             : MISTRAL MAM EXTERNAL REPLICATION SEALED")
    print("NEXT                               : archive TEST525-528 chain; no further calibration on this sealed result")
else:
    print("STATUS                             : EXTERNAL SEAL NOT ESTABLISHED")
    print("NEXT                               : preserve failure unchanged; diagnose only in a new test")
print("="*176)

# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST526
# MISTRAL FROZEN EXTERNAL REPLICATION SEAL
#
# DIRECT CHILD OF TEST525
#
# TEST525 CALIBRATION:
#   POINTER: L28H00
#   ADDRESS: L00-V UNCENTERED
#   DISC:          16/16 = 1.0000
#   INTERNAL-EVAL: 16/16 = 1.0000
#   POINTER EVAL:  16/16 = 1.0000
#
# TEST526:
#   NO DISCOVERY
#   NO HEAD/LAYER SCAN
#   NO ADDRESS SCAN
#   NO TOP-K SELECTION
#   NO POST-HOC TUNING
#
# FROZEN POINTER = L28H00
# FROZEN ADDRESS = L00-V UNCENTERED
# FROZEN MATCH   = max cosine
#
# EXTERNAL PANEL:
#   128 completely fresh items
#   256 independently forged A/B raw numeric memories
#
# PRIMARY:
#   active A + natural question
#   -> native Mistral L28H00 Q·K + RoPE
#   -> pointer-weighted A L00-V
#   -> max cosine against persistent B L00-V rows
#
# CONTROLS:
#   UNIFORM
#   SHIFTED
#   NO-A ZERO
#   COUNTERFACTUAL FOLLOW
#
# GOLD SPAN:
#   DIAGNOSTIC ONLY
#
# IMPORTANT:
#   Mistral has only 220 suitable unique single-token uppercase codes.
#   128-item external replication requires 768 unique seal/class identities.
#   TEST526 therefore uses deterministic unique multi-token synthetic codes.
#   These codes are ordinary source text only; token IDs/text are NEVER used
#   by retrieval. Retrieval remains purely numeric MAM geometry.
#
# PREDECLARED EXTERNAL SEAL:
#   PRIMARY R1 >= .85
#   PRIMARY R5 >= .95
#   CF FOLLOW R1 >= .80
#   UNIFORM R1 <= .10
#   SHIFTED R1 <= .10
#   NO-A R1 <= .10
#
# STRONG:
#   PRIMARY R1 >= .90
#   CF FOLLOW R1 >= .85
#   all controls <= .05

import os,sys,subprocess,importlib.util,random,time,hashlib,json,gc,re,math
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

TEST="526";SEED=526;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";N=128;DEVICE=torch.device("cuda")
FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
PTR_L=28;PTR_H=0;BL=0;CENTER=False;BK="V"
TEST525="a656140e1b484212ef8956c2cbde6fe2df3b76eb19ca901eccd5f66f8442f209"
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*176)
print("TEST526 — AKBASCORE MAM · MISTRAL FROZEN EXTERNAL REPLICATION SEAL")
print("128 FRESH EXTERNAL ITEMS · NO DISCOVERY · L28H00 → L00-V UNCENTERED")
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
if not (0<=PTR_L<NL and 0<=PTR_H<QH and 0<=BL<NL):raise RuntimeError("Frozen coordinate outside architecture.")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)}")
print(f"      {NL}L H={H} QH={QH} KVH={KVH} HD={HD} REP={REP} ADDRESS={KVD}")
print(f"      POINTER LOCK : L{PTR_L:02d}H{PTR_H:02d}")
print(f"      ADDRESS LOCK : L{BL:02d}-{BK} · UNCENTERED")
print("      DISCOVERY    : NONE")
print("      PANEL        : FRESH EXTERNAL 128")
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

def build_codes(n,offset=0):
    out=[];i=offset
    while len(out)<n:
        s=f"ZX{i:05d}Q"
        ids=enc(s)
        if len(ids)>=2 and tok.decode(ids,skip_special_tokens=False)==s and s not in out:out.append(s)
        i+=1
        if i>offset+n*100:raise RuntimeError(f"Could not build {n} stable multi-token codes.")
    return out

NAMES=build_names(N*3)
ALL_CODES=build_codes(N*6,10000)
SEALS=ALL_CODES[:N*3];CLASSES=ALL_CODES[N*3:]
assert len(SEALS)==N*3 and len(CLASSES)==N*3
assert len(set(SEALS))==N*3 and len(set(CLASSES))==N*3
assert set(SEALS).isdisjoint(CLASSES)
CODE_LENS=[len(enc(x)) for x in ALL_CODES]
print(f"      Multi-token identity pool: {len(ALL_CODES)} unique")
print(f"      Token lengths: min={min(CODE_LENS)} max={max(CODE_LENS)} mean={np.mean(CODE_LENS):.2f}")
print(f"      seals={len(SEALS)} classes={len(CLASSES)}")

def make_items():
    items=[]
    for i in range(N):
        ents=NAMES[3*i:3*i+3];aseals=SEALS[3*i:3*i+3];gold=aseals[0]
        j1=(i+23)%N;j2=(i+57)%N
        bseals=[gold,SEALS[3*j1+1],SEALS[3*j2+2]]
        bclasses=CLASSES[3*i:3*i+3]
        ra=[f"Instrument {ents[j]} carries seal {aseals[j]}." for j in range(3)]
        rb=[f"Seal {bseals[j]} corresponds to routing class {bclasses[j]}." for j in range(3)]
        r=random.Random(SEED+i*101);r.shuffle(ra);r.shuffle(rb)
        items.append({"id":i+1,"A":" ".join(ra),"B":" ".join(rb),
        "gold":{"entity":ents[0],"seal":gold,"class":bclasses[0]},
        "A_seals":list(aseals),"A_distractors":list(aseals[1:]),"B_seals":list(bseals)})
    return items

ITEMS=make_items()
assert len(set(x["gold"]["seal"] for x in ITEMS))==N
for i,it in enumerate(ITEMS):
    assert it["gold"]["seal"] in it["B_seals"]
    assert all(x not in it["B_seals"] for x in it["A_distractors"])
    assert sum(it["gold"]["seal"] in x["B_seals"] for x in ITEMS)==1

LOCK={"test":TEST,"parent525":TEST525,"model":MODEL_ID,"seed":SEED,"N":N,"external":True,"discovery":False,
"pointer":{"layer":PTR_L,"head":PTR_H},
"address":{"layer":BL,"kind":BK,"center":CENTER,"dimension":KVD,"match":"max_cosine"},
"identity":{"type":"stable_unique_multitoken_text","count":N*6,"retrieval_use":"NONE"},
"controls":["uniform","shifted","noA_zero","counterfactual"],
"seal":{"primary_r1":.85,"primary_r5":.95,"cf_r1":.80,"uniform_max":.10,"shifted_max":.10,"noA_max":.10},
"strong":{"primary_r1":.90,"cf_r1":.85,"controls_max":.05},
"selection":"NONE; L28H00 and L00-V UNCENTERED frozen from TEST525 before TEST526 panel evaluation",
"forbidden":["HEAD_SCAN","LAYER_SCAN","ADDRESS_SCAN","TOPK_SCAN","TOKEN_ID_ADDRESS","DECODED_SEAL_ADDRESS",
"LM_HEAD_ADDRESS","GOLD_POSITION_ADDRESS","QUERY_TIME_B_FORWARD","TRAINING","DRA","LEARNED_ROUTER","POSTHOC_SELECTION"]}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      TEST526 LOCK:",LOCK_SHA)
print("      PARENT525   :",TEST525)
print("      Frozen before panel evaluation: L28H00 → L00-V UNCENTERED")

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

def gold_token_positions(text,seal):
    hits=[m.span() for m in re.finditer(re.escape(seal),text)]
    if len(hits)!=1:raise RuntimeError(f"Seal occurrence !=1: {seal}")
    a,b=hits[0]
    z=tok(text+SEP,add_special_tokens=False,return_offsets_mapping=True);pos=[]
    for j,(x,y) in enumerate(z["offset_mapping"]):
        if y>a and x<b:pos.append(j+1)
    if not pos:raise RuntimeError(f"No gold span: {seal}")
    return pos

def rank(sc,gold):
    if len(sc)!=N:raise RuntimeError(f"Score vector must contain {N} candidates, got {len(sc)}.")
    order=sorted(range(N),key=lambda j:(-float(sc[j]),j))
    return order.index(gold)+1,order[0],order

def metrics(r):
    a=np.asarray(r,float)
    return float(np.mean(a==1)),float(np.mean(a<=5)),float(np.mean(a<=16)),float(np.mean(1/a))

def cos_rows(v,M):
    v=v.float();M=M.float();nv=v.norm()
    if float(nv)<=1e-12:return torch.zeros(M.shape[0],dtype=torch.float32)
    v=v/nv;M=M/M.norm(dim=1,keepdim=True).clamp_min(1e-8)
    return M@v

print("\n[2/11] OFFLINE FORGE — 128 fresh external items...")
A_RAW=[];B_RAW=[];A_PACK=[];GOLD_SPANS=[];META=[]
for i,it in enumerate(ITEMS):
    ar=forge(it["A"]);br=forge(it["B"])
    A_RAW.append(ar);B_RAW.append(br);A_PACK.append(install(ar))
    GOLD_SPANS.append(gold_token_positions(it["A"],it["gold"]["seal"]))
    META.append({"entity":it["gold"]["entity"],"gold_seal":it["gold"]["seal"],
                 "d1":it["A_distractors"][0],"d2":it["A_distractors"][1]})
    if i<5:print(f"      ITEM {i+1:03d} | A={A_PACK[-1][1]} B={br[0][0].shape[1]} tokens")
print("      256 independent raw numeric memories forged.")
print("      No TEST526 parameter selected from this panel.")

print("\n[3/11] Building persistent frozen B address bank...")
B_MATS=[]
for raw in B_RAW:
    x=raw[BL][1][:,1:,:].float().cpu()
    B_MATS.append(x.permute(1,0,2).reshape(x.shape[1],-1).contiguous())
assert len(B_MATS)==N and all(M.shape[1]==KVD for M in B_MATS)
del B_RAW
for it in ITEMS:
    it["A"]=None;it["B"]=None
del ITEMS,ALL_CODES,SEALS,CLASSES
gc.collect();torch.cuda.empty_cache()
print(f"      Persistent B bank              : {N} numeric address matrices")
print(f"      Address dimension              : {KVD}")
print("      B source text                  : REMOVED")
print("      Query-time candidate-B forward : 0")
print("      Gold span                      : diagnostic only")

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
    inst,Tm,P=packet;o,_,_,n=query_forward(packet,q);qpos=P+n-1
    layer=layers[PTR_L]
    h=o.hidden_states[PTR_L][0,-1].to(layer.input_layernorm.weight.dtype)
    hn=layer.input_layernorm(h)
    qv=layer.self_attn.q_proj(hn).view(1,QH,1,HD)
    ak=inst[PTR_L][0].repeat_interleave(REP,dim=0).unsqueeze(0)
    dummy=torch.zeros((1,QH,1,HD),device=DEVICE,dtype=qv.dtype)
    c,s=rope_cos_sin(dummy,[qpos])
    qrot,_=apply_rotary_pos_emb(qv,dummy,c,s,unsqueeze_dim=1)
    score=torch.einsum("bhqd,bhkd->bhqk",qrot.float(),ak.float()).squeeze(0).squeeze(1)/math.sqrt(HD)
    score[:,0]=-torch.inf
    W=torch.softmax(score,dim=-1).detach().cpu()
    W[:,0]=0;W=W/W.sum(dim=1,keepdim=True).clamp_min(1e-12)
    return W[PTR_H]

def uniform_pointer(T):
    w=torch.ones(T,dtype=torch.float32);w[0]=0
    return w/w.sum().clamp_min(1e-12)

def shifted_pointer(w):
    n=len(w)-1
    if n<=1:return w.clone()
    z=torch.zeros_like(w);z[1:]=torch.roll(w[1:].clone(),shifts=max(1,n//2),dims=0)
    return z/z.sum().clamp_min(1e-12)

def pointer_vector(raw,w):
    x=raw[BL][1].float().cpu()
    if x.shape[1]!=len(w):raise RuntimeError(f"Pointer length {len(w)} != V length {x.shape[1]}")
    return (x*w[None,:,None]).sum(dim=1).reshape(-1)

def B_scores(p):
    sc=[float(cos_rows(p,M).max()) for M in B_MATS]
    if len(sc)!=N:raise RuntimeError("Incomplete B score vector.")
    return sc

print("\n[4/11] EXTERNAL PRIMARY — frozen L28H00 → L00-V UNCENTERED...")
PTR=[];RPRI=[];SELPRI=[];PHIT=[];PMASS=[];PRANK=[];PRI_SCORES=[];PRI_MARGIN=[]
for i in range(N):
    w=frozen_pointer(A_PACK[i],qA(META[i]["entity"]));PTR.append(w)
    p=pointer_vector(A_RAW[i],w);sc=B_scores(p);r,se,order=rank(sc,i)
    RPRI.append(r);SELPRI.append(se);PRI_SCORES.append(sc)
    PRI_MARGIN.append(float(sc[order[0]]-sc[order[1]]))
    span=GOLD_SPANS[i];arg=int(torch.argmax(w));hit=int(arg in span);mass=float(w[span].sum())
    best=float(w[span].max());sr=1+int((w[1:]>best).sum())
    PHIT.append(hit);PMASS.append(mass);PRANK.append(sr)
    print(f"      [{i+1:03d}/128] rank={r:3d} selected={se+1:03d} | ptr={arg:2d} span={span} hit={hit} mass={mass:.4f} margin={PRI_MARGIN[-1]:+.5f}")
MP=metrics(RPRI)
print(f"      PRIMARY R1/R5/R16/MRR: {MP[0]:.4f} / {MP[1]:.4f} / {MP[2]:.4f} / {MP[3]:.6f}")
print(f"      POINTER argmax gold-span: {sum(PHIT)}/{N} = {np.mean(PHIT):.4f}")
print(f"      POINTER mean gold mass  : {np.mean(PMASS):.6f}")
print(f"      POINTER median span rank: {np.median(PRANK):.1f}")
print(f"      Mean Top1-Top2 margin   : {np.mean(PRI_MARGIN):+.6f}")

print("\n[5/11] EXTERNAL CONTROLS — UNIFORM / SHIFTED / NO-A ZERO...")
RUNI=[];RSHIFT=[];RNO=[];ZERO=torch.zeros(KVD,dtype=torch.float32)
for i in range(N):
    wu=uniform_pointer(len(PTR[i]))
    ru,_,_=rank(B_scores(pointer_vector(A_RAW[i],wu)),i);RUNI.append(ru)
    ws=shifted_pointer(PTR[i])
    rs,_,_=rank(B_scores(pointer_vector(A_RAW[i],ws)),i);RSHIFT.append(rs)
    rn,_,_=rank(B_scores(ZERO),i);RNO.append(rn)
MU=metrics(RUNI);MS=metrics(RSHIFT);MN=metrics(RNO)
print(f"      UNIFORM R1/R5/R16/MRR: {MU[0]:.4f} / {MU[1]:.4f} / {MU[2]:.4f} / {MU[3]:.6f}")
print(f"      SHIFTED R1/R5/R16/MRR: {MS[0]:.4f} / {MS[1]:.4f} / {MS[2]:.4f} / {MS[3]:.6f}")
print(f"      NO-A    R1/R5/R16/MRR: {MN[0]:.4f} / {MN[1]:.4f} / {MN[2]:.4f} / {MN[3]:.6f}")

print("\n[6/11] EXTERNAL COUNTERFACTUAL FOLLOW...")
RCF=[];CFSEL=[];CFHIT=[];CFMASS=[];CF_SCORES=[]
for i in range(N):
    j=(i+37)%N;targetseal=META[j]["gold_seal"]
    ents=[META[i]["entity"],NAMES[3*i+1],NAMES[3*i+2]]
    seals=[targetseal,META[i]["d1"],META[i]["d2"]]
    rows=[f"Instrument {ents[x]} carries seal {seals[x]}." for x in range(3)]
    rgen=random.Random(SEED+9000+i);rgen.shuffle(rows);cfA=" ".join(rows)
    raw=forge(cfA);pack=install(raw)
    w=frozen_pointer(pack,qA(META[i]["entity"]))
    p=pointer_vector(raw,w);sc=B_scores(p);r,se,_=rank(sc,j)
    RCF.append(r);CFSEL.append(se);CF_SCORES.append(sc)
    span=gold_token_positions(cfA,targetseal);arg=int(torch.argmax(w))
    hit=int(arg in span);mass=float(w[span].sum());CFHIT.append(hit);CFMASS.append(mass)
    print(f"      [{i+1:03d}/128] target={j+1:03d} rank={r:3d} selected={se+1:03d} follow={int(se==j)} | ptr-hit={hit} mass={mass:.4f}")
    del raw,pack,w,p
MC=metrics(RCF)
print(f"      CF FOLLOW R1/R5/R16/MRR: {MC[0]:.4f} / {MC[1]:.4f} / {MC[2]:.4f} / {MC[3]:.6f}")
print(f"      CF pointer gold-span hit : {sum(CFHIT)}/{N} = {np.mean(CFHIT):.4f}")
print(f"      CF mean gold-span mass   : {np.mean(CFMASS):.6f}")

print("\n[7/11] Mechanistic diagnostics...")
CORRECT=np.asarray([x==1 for x in RPRI],dtype=bool)
HIT=np.asarray(PHIT,dtype=bool);MASS=np.asarray(PMASS,float)
def safe_mean(x):return float(np.mean(x)) if len(x) else float("nan")
print(f"      Retrieval R1 | pointer-hit      : {safe_mean(CORRECT[HIT]):.4f} n={int(HIT.sum())}")
print(f"      Retrieval R1 | pointer-miss     : {safe_mean(CORRECT[~HIT]):.4f} n={int((~HIT).sum())}")
print(f"      Mean gold mass | retrieval R1   : {safe_mean(MASS[CORRECT]):.6f}")
print(f"      Mean gold mass | retrieval fail : {safe_mean(MASS[~CORRECT]):.6f}")
print(f"      PRIMARY - UNIFORM R1            : {MP[0]-MU[0]:+.4f}")
print(f"      PRIMARY - SHIFTED R1            : {MP[0]-MS[0]:+.4f}")
print(f"      PRIMARY - NO-A R1               : {MP[0]-MN[0]:+.4f}")

print("\n[8/11] Wilson intervals...")
def wilson(k,n,z=1.959963984540054):
    if n==0:return float("nan"),float("nan")
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h

for name,r in (("PRIMARY",RPRI),("UNIFORM",RUNI),("SHIFTED",RSHIFT),("NO_A",RNO),("CF",RCF)):
    k=sum(int(x==1) for x in r);lo,hi=wilson(k,len(r))
    print(f"      {name:8s}: {k:3d}/{len(r)} = {k/len(r):.4f} | 95% Wilson [{lo:.4f},{hi:.4f}]")

print("\n[9/11] PREDECLARED EXTERNAL SEAL...")
PASS=(MP[0]>=.85 and MP[1]>=.95 and MC[0]>=.80 and MU[0]<=.10 and MS[0]<=.10 and MN[0]<=.10)
STRONG=(MP[0]>=.90 and MC[0]>=.85 and MU[0]<=.05 and MS[0]<=.05 and MN[0]<=.05)
if STRONG:VERDICT="STRONG_MISTRAL_EXTERNAL_REPLICATION_SEAL_L28H00_CAGRIIIZ"
elif PASS:VERDICT="MISTRAL_EXTERNAL_REPLICATION_SEAL_L28H00_CAGRIIIZ"
elif MP[0]>=.75 and MC[0]>=.70 and MU[0]<=.10 and MS[0]<=.10:VERDICT="MISTRAL_EXTERNAL_REPLICATION_STRONG_BUT_BELOW_PREDECLARED_SEAL"
elif MP[0]>=.50 and MP[0]-MS[0]>=.30:VERDICT="MISTRAL_CAUSAL_SIGNAL_REPLICATES_BUT_RELIABILITY_INSUFFICIENT"
else:VERDICT="MISTRAL_EXTERNAL_REPLICATION_NOT_ESTABLISHED"
print("      PASS   :",PASS)
print("      STRONG :",STRONG)
print("      VERDICT:",VERDICT)

print("\n[10/11] Protocol / replay / weight audit...")
S1=sentinel()
REPLAY_OK=True
for i in range(N):
    r,se,_=rank(PRI_SCORES[i],i)
    if r!=RPRI[i] or se!=SELPRI[i]:REPLAY_OK=False;break
print("      Model frozen                              :",S0==S1)
print("      Trainable parameter tensors               :",sum(int(p.requires_grad) for p in model.parameters()))
print("      External panel                            : YES — 128 fresh")
print("      DISC in TEST526                           : NONE")
print("      Layer/head scan                           : NONE")
print("      Address scan                              : NONE")
print("      Top-K scan                                : NONE")
print("      Pointer                                   : FROZEN L28H00")
print("      Address                                   : FROZEN L00-V UNCENTERED")
print("      Address dimension                         :",KVD)
print("      Match                                     : max cosine")
print("      Gold span used by retrieval               : NO")
print("      Gold span diagnostic only                 : YES")
print("      Token-ID/text/LM-head address             : NONE")
print("      Query-time candidate-B forwards           : 0")
print("      B source text live                        : NO")
print("      Training / DRA / learned router           : NONE")
print("      Post-hoc TEST526 selection                : NONE")
print("      Counterfactual same frozen mechanism      : YES")
print("      NO-A                                      : zero-address negative control")
print("      Raw-score replay                          :",REPLAY_OK)
print("      Weight sentinel                           :","PASS" if S0==S1 else"FAIL")

print("\n[11/11] FINAL...")
print("\n"+"="*176)
print("TEST526 FINAL RESULT — MISTRAL FROZEN EXTERNAL REPLICATION SEAL")
print("="*176)
print("MODEL                              : mistralai/Mistral-7B-Instruct-v0.3 · frozen")
print("EXTERNAL BANK                      : 128 items / 256 raw numeric memories")
print("POINTER                            : L28H00 · native Q·K + RoPE")
print("ADDRESS                            : L00-V · UNCENTERED · 1024D")
print("MATCH                              : max cosine")
print("DISCOVERY / SELECTION              : NONE")
print("-"*176)
print(f"PRIMARY                            : R1={MP[0]:.4f} R5={MP[1]:.4f} R16={MP[2]:.4f} MRR={MP[3]:.6f}")
print(f"UNIFORM                            : R1={MU[0]:.4f} R5={MU[1]:.4f} R16={MU[2]:.4f} MRR={MU[3]:.6f}")
print(f"SHIFTED                            : R1={MS[0]:.4f} R5={MS[1]:.4f} R16={MS[2]:.4f} MRR={MS[3]:.6f}")
print(f"NO-A                               : R1={MN[0]:.4f} R5={MN[1]:.4f} R16={MN[2]:.4f} MRR={MN[3]:.6f}")
print(f"COUNTERFACTUAL FOLLOW              : R1={MC[0]:.4f} R5={MC[1]:.4f} R16={MC[2]:.4f} MRR={MC[3]:.6f}")
print(f"POINTER GOLD-SPAN ARGMAX HIT       : {sum(PHIT)}/{N} = {np.mean(PHIT):.4f}")
print(f"POINTER MEAN GOLD-SPAN MASS        : {np.mean(PMASS):.6f}")
print(f"POINTER MEDIAN GOLD-SPAN RANK      : {np.median(PRANK):.1f}")
print(f"CF POINTER GOLD-SPAN HIT           : {sum(CFHIT)}/{N} = {np.mean(CFHIT):.4f}")
print(f"CF POINTER MEAN GOLD-SPAN MASS     : {np.mean(CFMASS):.6f}")
print(f"MEAN TOP1-TOP2 MARGIN              : {np.mean(PRI_MARGIN):+.6f}")
print("-"*176)
print("PREDECLARED SEAL                   : PRIMARY R1>=.85 R5>=.95; CF>=.80; UNIFORM/SHIFTED/NO-A<=.10")
print("STRONG SEAL                        : PRIMARY R1>=.90; CF>=.85; all controls<=.05")
print("PASS                               :",PASS)
print("STRONG                             :",STRONG)
print("WEIGHT SENTINEL                    :","PASS" if S0==S1 else"FAIL")
print("RAW-SCORE REPLAY                   :",REPLAY_OK)
print("QUERY-TIME CANDIDATE-B FORWARDS    : 0")
print("TEST525 LOCK                       :",TEST525)
print("TEST526 LOCK                       :",LOCK_SHA)
print(f"TOTAL TEST TIME                    : {time.perf_counter()-T0:.2f}s")
print("VERDICT                            :",VERDICT)
if PASS:print("NEXT                               : freeze Mistral MAM external seal; then TEST527 live demo")
else:print("NEXT                               : diagnose external failure without changing frozen coordinates")
print("="*176)

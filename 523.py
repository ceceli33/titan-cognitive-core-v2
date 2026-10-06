# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST523
# FROZEN L23H12 EXTERNAL REPLICATION SEAL
#
# DIRECT CHILD OF CONFIRMED-WORKING TEST522
#
# TEST522 DISCOVERY:
#   ADDRESS: L02-V UNCENTERED
#   DISC-frozen channel: L23H12
#   DISC:          32/32 = 1.0000
#   INTERNAL-EVAL: 31/32 = .9688
#
# TEST523:
#   NO DISCOVERY
#   NO HEAD/LAYER SCAN
#   NO TOP-K SELECTION
#   NO POST-HOC TUNING
#   FROZEN POINTER = L23H12
#   FROZEN ADDRESS = L02-V UNCENTERED
#   FROZEN MATCH   = max cosine
#
# EXTERNAL PANEL:
#   128 new items
#   256 independently forged A/B tensor memories
#
# PRIMARY:
# query -> native L23H12 Q·K over A
#       -> weighted A L02-V
#       -> max cosine against persistent B L02-V positions
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
# PREDECLARED SEAL:
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
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

TEST="523";SEED=523;MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N=128;DEVICE=torch.device("cuda")
FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
PTR_L=23;PTR_H=12;BL=2;BK="V"
TEST521="e1eeaa869cdb15532fb99419cf53fc7f1ba922807a925357934ce0fade7784de"
TEST522="bb82c131b79d1ef4c2f384f4f5d49776a43a1f55bada5f40372d411105077833"
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*176)
print("TEST523 — AKBASCORE MAM · FROZEN L23H12 EXTERNAL REPLICATION SEAL")
print("128 EXTERNAL ITEMS · NO DISCOVERY · L23H12 → L02-V UNCENTERED")
print("="*176)
T0=time.perf_counter()

print("\n[1/11] Loading frozen Qwen...")
_tv=tuple(int("".join(c for c in x if c.isdigit())or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
if not tok.is_fast:raise RuntimeError("Fast tokenizer required.")
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
enc=lambda s:tok(s,add_special_tokens=False).input_ids
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=H//QH;REP=QH//KVH
if (NL,H,QH,KVH,HD)!=(28,3584,28,4,128):raise RuntimeError("Architecture mismatch.")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id;VOC=model.model.embed_tokens.weight.shape[0]
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | {NL}L H={H} QH={QH} KVH={KVH} HD={HD}")
print(f"      POINTER LOCK : L{PTR_L:02d}H{PTR_H:02d}")
print(f"      ADDRESS LOCK : L{BL:02d}-{BK} · UNCENTERED")
print("      DISCOVERY    : NONE")
print("      PANEL        : EXTERNAL 128")

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
if len(POOL)<N*6:raise RuntimeError(f"Need {N*6} distinct native codes, found {len(POOL)}.")
rr=random.Random(SEED+100000);rr.shuffle(POOL)
SEALS=POOL[:N*3];CLASSES=POOL[N*3:N*6]
assert len(SEALS)==N*3 and len(CLASSES)==N*3
assert len(set(SEALS))==N*3 and len(set(CLASSES))==N*3
assert set(SEALS).isdisjoint(CLASSES)
print(f"      Native code pool: {len(POOL)} | seals={len(SEALS)} classes={len(CLASSES)}")

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
        "A_seals":aseals,"A_distractors":aseals[1:],"B_seals":bseals,"B_classes":bclasses})
    return items

ITEMS=make_items()
assert len(set(x["gold"]["seal"] for x in ITEMS))==N
for i,it in enumerate(ITEMS):
    assert it["gold"]["seal"] in it["B_seals"]
    assert all(x not in it["B_seals"] for x in it["A_distractors"])
    assert sum(it["gold"]["seal"] in x["B_seals"] for x in ITEMS)==1

LOCK={"test":TEST,"parents":{"521":TEST521,"522":TEST522},"model":MODEL_ID,"seed":SEED,"N":N,
"external":True,"discovery":False,
"pointer":{"layer":PTR_L,"head":PTR_H},
"address":{"layer":BL,"kind":BK,"center":False,"match":"max_cosine"},
"controls":["uniform","shifted","noA_zero","counterfactual"],
"seal":{"primary_r1":.85,"primary_r5":.95,"cf_r1":.80,"uniform_max":.10,"shifted_max":.10,"noA_max":.10},
"strong":{"primary_r1":.90,"cf_r1":.85,"controls_max":.05},
"selection":"NONE; L23H12 frozen from TEST522 before TEST523 panel",
"forbidden":["HEAD_SCAN","LAYER_SCAN","TOPK_SCAN","TOKEN_ID_ADDRESS","DECODED_SEAL_ADDRESS",
"LM_HEAD_ADDRESS","GOLD_POSITION_ADDRESS","QUERY_TIME_B_FORWARD","TRAINING","DRA","LEARNED_ROUTER","POSTHOC_SELECTION"]}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      TEST523 LOCK:",LOCK_SHA)
print("      PARENT521   :",TEST521)
print("      PARENT522   :",TEST522)
print("      Frozen before panel evaluation: L23H12 → L02-V UNCENTERED")

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
    layer=model.model.layers[L]
    rot=layer.self_attn.rotary_emb if hasattr(layer.self_attn,"rotary_emb") else model.model.rotary_emb
    p=torch.tensor([pos],device=DEVICE,dtype=torch.long)
    try:c,s=rot(q,p)
    except TypeError:c,s=rot(q,position_ids=p)
    _,kr=apply_rotary_pos_emb(q,kk,c,s)
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

def rank(sc,gold):
    order=sorted(range(N),key=lambda j:(-float(sc[j]),j))
    return order.index(gold)+1,order[0]

def metrics(r):
    a=np.asarray(r,float)
    return float(np.mean(a==1)),float(np.mean(a<=5)),float(np.mean(a<=16)),float(np.mean(1/a))

def cos_rows(v,M):
    v=v.float();M=M.float()
    v=v/v.norm().clamp_min(1e-8)
    M=M/M.norm(dim=1,keepdim=True).clamp_min(1e-8)
    return M@v

print("\n[2/11] OFFLINE FORGE — 128 completely new external items...")
A_RAW=[];B_RAW=[];A_PACK=[]
for i,it in enumerate(ITEMS):
    ar=forge(it["A"]);br=forge(it["B"])
    A_RAW.append(ar);B_RAW.append(br);A_PACK.append(install(ar))
    if i<5:print(f"      ITEM {i+1:03d} | A={A_PACK[-1][1]} B={br[0][0].shape[1]} tokens")
print("      256 independent BELLEKÖZ forged.")
print("      No TEST523 parameter selected from this panel.")

def gold_token_positions(text,seal):
    hits=[m.span() for m in re.finditer(re.escape(seal),text)]
    if len(hits)!=1:raise RuntimeError(f"Seal occurrence !=1: {seal}")
    a,b=hits[0]
    z=tok(text+SEP,add_special_tokens=False,return_offsets_mapping=True);pos=[]
    for j,(x,y) in enumerate(z["offset_mapping"]):
        if y>a and x<b:pos.append(j+1)
    if not pos:raise RuntimeError(f"No span: {seal}")
    return pos

GOLD_SPANS=[gold_token_positions(it["A"],it["gold"]["seal"]) for it in ITEMS]

@torch.inference_mode()
def query_forward(packet,q):
    # Kept identical to confirmed-working TEST522.
    inst,Tm,P=packet;c=cache_of(inst);ids=enc(FMT.format(q=q));n=len(ids)
    pos=torch.arange(P,P+n,device=DEVICE).unsqueeze(0)
    mask=torch.ones((1,Tm+n),device=DEVICE,dtype=torch.long)
    o=model(input_ids=torch.tensor([ids],device=DEVICE),past_key_values=c,
            attention_mask=mask,position_ids=pos,use_cache=False,
            output_hidden_states=True,return_dict=True)
    return o,Tm,P,n

@torch.inference_mode()
def frozen_pointer(packet,q):
    # Exact TEST522 pointer_xray mathematics, restricted to frozen L23H12.
    inst,Tm,P=packet;o,_,_,n=query_forward(packet,q);qpos=P+n-1
    L=PTR_L;layer=model.model.layers[L]
    h=o.hidden_states[L][0,-1].to(layer.input_layernorm.weight.dtype)
    hn=layer.input_layernorm(h)
    qv=layer.self_attn.q_proj(hn).view(1,QH,1,HD)
    ak=inst[L][0].repeat_interleave(REP,dim=0).unsqueeze(0)
    dummy=torch.zeros_like(ak[:,:,:1,:])
    rot=layer.self_attn.rotary_emb if hasattr(layer.self_attn,"rotary_emb") else model.model.rotary_emb
    pp=torch.tensor([[qpos]],device=DEVICE,dtype=torch.long)
    try:c,s=rot(dummy,pp)
    except TypeError:c,s=rot(dummy,position_ids=pp)
    qrot,_=apply_rotary_pos_emb(qv,dummy,c,s)
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
    shift=max(1,n//2)
    z=torch.zeros_like(w)
    z[1:]=torch.roll(w[1:].clone(),shifts=shift,dims=0)
    return z/z.sum().clamp_min(1e-12)

def pointer_vector(raw,w):
    x=raw[BL][1].float().cpu()
    if x.shape[1]!=len(w):raise RuntimeError(f"Pointer length {len(w)} != tensor length {x.shape[1]}")
    return (x*w[None,:,None]).sum(dim=1).reshape(-1)

print("\n[3/11] Building persistent B address bank...")
B_MATS=[]
for raw in B_RAW:
    x=raw[BL][1][:,1:,:].float().cpu()
    B_MATS.append(x.permute(1,0,2).reshape(x.shape[1],-1).contiguous())

def B_scores(p):
    return [float(cos_rows(p,M).max()) for M in B_MATS]

print("      B address source             : persistent L02-V tensors only")
print("      Query-time candidate-B forward: 0")
print("      Gold span                    : diagnostic only")
gc.collect();torch.cuda.empty_cache()

print("\n[4/11] EXTERNAL PRIMARY — frozen L23H12 → L02-V...")
PTR=[];RPRI=[];SELPRI=[];PHIT=[];PMASS=[];PRANK=[]
for i,it in enumerate(ITEMS):
    w=frozen_pointer(A_PACK[i],qA(it["gold"]["entity"]));PTR.append(w)
    p=pointer_vector(A_RAW[i],w)
    r,se=rank(B_scores(p),i);RPRI.append(r);SELPRI.append(se)
    span=GOLD_SPANS[i];arg=int(torch.argmax(w))
    hit=int(arg in span);mass=float(w[span].sum())
    best=float(w[span].max());sr=1+int((w[1:]>best).sum())
    PHIT.append(hit);PMASS.append(mass);PRANK.append(sr)
    print(f"      [{i+1:03d}/128] rank={r:3d} selected={se+1:03d} | ptr={arg:2d} span={span} hit={hit} mass={mass:.4f} span-rank={sr:2d}")
MP=metrics(RPRI)
print(f"      PRIMARY R1/R5/R16/MRR: {MP[0]:.4f} / {MP[1]:.4f} / {MP[2]:.4f} / {MP[3]:.6f}")
print(f"      POINTER argmax gold-span: {sum(PHIT)}/{N} = {np.mean(PHIT):.4f}")
print(f"      POINTER mean gold mass  : {np.mean(PMASS):.6f}")
print(f"      POINTER median span rank: {np.median(PRANK):.1f}")

print("\n[5/11] EXTERNAL CONTROLS — UNIFORM / SHIFTED / NO-A ZERO...")
RUNI=[];RSHIFT=[];RNO=[]
ZERO=torch.zeros(KVH*HD,dtype=torch.float32)
for i in range(N):
    wu=uniform_pointer(len(PTR[i]))
    ru,_=rank(B_scores(pointer_vector(A_RAW[i],wu)),i);RUNI.append(ru)
    ws=shifted_pointer(PTR[i])
    rs,_=rank(B_scores(pointer_vector(A_RAW[i],ws)),i);RSHIFT.append(rs)
    rn,_=rank(B_scores(ZERO),i);RNO.append(rn)
MU=metrics(RUNI);MS=metrics(RSHIFT);MN=metrics(RNO)
print(f"      UNIFORM R1/R5/R16/MRR: {MU[0]:.4f} / {MU[1]:.4f} / {MU[2]:.4f} / {MU[3]:.6f}")
print(f"      SHIFTED R1/R5/R16/MRR: {MS[0]:.4f} / {MS[1]:.4f} / {MS[2]:.4f} / {MS[3]:.6f}")
print(f"      NO-A    R1/R5/R16/MRR: {MN[0]:.4f} / {MN[1]:.4f} / {MN[2]:.4f} / {MN[3]:.6f}")

print("\n[6/11] EXTERNAL COUNTERFACTUAL FOLLOW...")
RCF=[];CFSEL=[];CFHIT=[];CFMASS=[]
for i in range(N):
    j=(i+37)%N
    it=ITEMS[i];targetseal=ITEMS[j]["gold"]["seal"]
    ents=[it["gold"]["entity"],NAMES[3*i+1],NAMES[3*i+2]]
    seals=[targetseal,it["A_distractors"][0],it["A_distractors"][1]]
    rows=[f"Instrument {ents[x]} carries seal {seals[x]}." for x in range(3)]
    rgen=random.Random(SEED+9000+i);rgen.shuffle(rows);cfA=" ".join(rows)
    raw=forge(cfA);pack=install(raw)
    w=frozen_pointer(pack,qA(it["gold"]["entity"]))
    p=pointer_vector(raw,w)
    r,se=rank(B_scores(p),j);RCF.append(r);CFSEL.append(se)
    span=gold_token_positions(cfA,targetseal);arg=int(torch.argmax(w))
    hit=int(arg in span);mass=float(w[span].sum())
    CFHIT.append(hit);CFMASS.append(mass)
    print(f"      [{i+1:03d}/128] target={j+1:03d} rank={r:3d} selected={se+1:03d} follow={int(se==j)} | ptr-hit={hit} mass={mass:.4f}")
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
if STRONG:VERDICT="STRONG_EXTERNAL_REPLICATION_SEAL_L23H12_CAGRIIIZ"
elif PASS:VERDICT="EXTERNAL_REPLICATION_SEAL_L23H12_CAGRIIIZ"
elif MP[0]>=.75 and MC[0]>=.70 and MU[0]<=.10 and MS[0]<=.10:VERDICT="L23H12_EXTERNAL_REPLICATION_STRONG_BUT_BELOW_PREDECLARED_SEAL"
elif MP[0]>=.50 and MP[0]-MS[0]>=.30:VERDICT="L23H12_CAUSAL_SIGNAL_REPLICATES_BUT_RELIABILITY_INSUFFICIENT"
else:VERDICT="L23H12_EXTERNAL_REPLICATION_NOT_ESTABLISHED"
print("      PASS   :",PASS)
print("      STRONG :",STRONG)
print("      VERDICT:",VERDICT)

print("\n[10/11] Protocol audit...")
print("      Model frozen                              : YES")
print("      External panel                            : YES — 128")
print("      DISC in TEST523                           : NONE")
print("      Layer/head scan                           : NONE")
print("      Top-K scan                                : NONE")
print("      Pointer                                   : FROZEN L23H12")
print("      Pointer math                              : TEST522 unchanged")
print("      Address                                   : FROZEN L02-V UNCENTERED")
print("      Match                                     : max cosine")
print("      Gold span used by retrieval               : NO")
print("      Gold span diagnostic only                 : YES")
print("      Token-ID/text/LM-head address             : NONE")
print("      Query-time candidate-B forwards           : 0")
print("      Training / DRA / learned router           : NONE")
print("      Post-hoc TEST523 selection                : NONE")
print("      Counterfactual same frozen mechanism      : YES")
print("      NO-A                                      : zero-address negative control")

print("\n[11/11] FINAL...")
print("\n"+"="*176)
print("TEST523 FINAL RESULT — FROZEN L23H12 EXTERNAL REPLICATION SEAL")
print("="*176)
print("MODEL                              : Qwen/Qwen2.5-7B-Instruct · frozen")
print("EXTERNAL BANK                      : 128 items / 256 BELLEKÖZ")
print("POINTER                            : L23H12 · native Q·K + RoPE")
print("ADDRESS                            : L02-V · UNCENTERED")
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
print("-"*176)
print("PREDECLARED SEAL                   : PRIMARY R1>=.85 R5>=.95; CF>=.80; UNIFORM/SHIFTED/NO-A<=.10")
print("STRONG SEAL                        : PRIMARY R1>=.90; CF>=.85; all controls<=.05")
print("PASS                               :",PASS)
print("STRONG                             :",STRONG)
print("TEST521 LOCK                       :",TEST521)
print("TEST522 LOCK                       :",TEST522)
print("TEST523 LOCK                       :",LOCK_SHA)
print(f"TOTAL TEST TIME                    : {time.perf_counter()-T0:.2f}s")
print("VERDICT                            :",VERDICT)
if PASS:print("NEXT                               : TEST524 / MAM DEMO PROTOTYPE")
else:print("NEXT                               : diagnose external failure before demo seal")
print("="*176)

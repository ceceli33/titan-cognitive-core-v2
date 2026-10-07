# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST525
# MISTRAL POINTER + ADDRESS CALIBRATION
#
# PURPOSE:
#   Qwen TEST523 mechanism -> Mistral-7B-Instruct-v0.3
#   NO QWEN COORDINATE TRANSFER
#   DISC 16 -> pointer L/H + address V layer/mode selection
#   INTERNAL-EVAL 16 -> frozen-coordinate validation
#   FINAL EXTERNAL PANEL IS NOT TOUCHED
#
# MECHANISM:
#   active numeric A + natural question
#   -> native Mistral Q·K + RoPE
#   -> ÇAĞRIİZ pointer
#   -> pointer-weighted Mistral V address (8 KV heads × 128 = 1024)
#   -> max cosine against persistent B V rows
#
# FORBIDDEN:
#   NO D120 · NO PCA · NO router · NO scorer · NO LM-head route
#   NO decoded seal address · NO token-ID address · NO training
#   NO LoRA · NO optimizer · NO DRA · NO query-time B forward
#   NO tuning on INTERNAL-EVAL

import os,sys,subprocess,importlib.util,random,time,hashlib,json,gc,re,math
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

TEST="525";SEED=525;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";N=32;ND=16;NE=16;DEVICE=torch.device("cuda")
FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
if ND+NE!=N:raise RuntimeError("DISC + INTERNAL-EVAL must equal N.")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*176)
print("TEST525 — AKBASCORE MAM · MISTRAL POINTER + ADDRESS CALIBRATION")
print("DISC 16 + INTERNAL-EVAL 16 · 32x32 NATIVE ATTENTION SCAN · V-LAYER x CENTER MODE")
print("="*176);T0=time.perf_counter()

print("\n[1/10] Loading frozen Mistral...")
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
print("      Model frozen · BF16 · SDPA · trainable=0")

FP=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[16].mlp.down_proj.weight,layers[24].self_attn.o_proj.weight,layers[31].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
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

NAMES=build_names(N*3);POOL=native_code_pool()
NEED=N*6
print(f"      Native code pool: {len(POOL)} | required={NEED}")
if len(POOL)<NEED:raise RuntimeError(f"Need {NEED} distinct native codes, found {len(POOL)}.")
rr=random.Random(SEED+100000);rr.shuffle(POOL);SEALS=POOL[:N*3];CLASSES=POOL[N*3:N*6]
assert len(SEALS)==N*3 and len(CLASSES)==N*3
assert len(set(SEALS))==N*3 and len(set(CLASSES))==N*3
assert set(SEALS).isdisjoint(CLASSES)

def make_items():
    items=[]
    for i in range(N):
        ents=NAMES[3*i:3*i+3];aseals=SEALS[3*i:3*i+3];gold=aseals[0]
        j1=(i+7)%N;j2=(i+13)%N
        bseals=[gold,SEALS[3*j1+1],SEALS[3*j2+2]];bclasses=CLASSES[3*i:3*i+3]
        ra=[f"Instrument {ents[j]} carries seal {aseals[j]}." for j in range(3)]
        rb=[f"Seal {bseals[j]} corresponds to routing class {bclasses[j]}." for j in range(3)]
        r=random.Random(SEED+i*101);r.shuffle(ra);r.shuffle(rb)
        items.append({"id":i+1,"A":" ".join(ra),"B":" ".join(rb),
        "gold":{"entity":ents[0],"seal":gold,"class":bclasses[0]},
        "A_seals":aseals,"A_distractors":aseals[1:],"B_seals":bseals,"B_classes":bclasses})
    return items

ITEMS=make_items();DISC=list(range(ND));EVAL=list(range(ND,N))
assert len(DISC)==ND and len(EVAL)==NE and set(DISC).isdisjoint(EVAL)
assert len(set(x["gold"]["seal"] for x in ITEMS))==N
for i,it in enumerate(ITEMS):
    assert it["gold"]["seal"] in it["B_seals"]
    assert all(x not in it["B_seals"] for x in it["A_distractors"])
    assert sum(it["gold"]["seal"] in x["B_seals"] for x in ITEMS)==1
print(f"      Calibration panel: {N} | DISC={ND} INTERNAL-EVAL={NE}")

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
    a,b=hits[0];z=tok(text+SEP,add_special_tokens=False,return_offsets_mapping=True);pos=[]
    for j,(x,y) in enumerate(z["offset_mapping"]):
        if y>a and x<b:pos.append(j+1)
    if not pos:raise RuntimeError(f"No gold span: {seal}")
    return pos

print("\n[2/10] OFFLINE FORGE — independent raw A/B memories...")
A_RAW=[];B_RAW=[];A_PACK=[];GOLD_SPANS=[]
for i,it in enumerate(ITEMS):
    ar=forge(it["A"]);br=forge(it["B"])
    A_RAW.append(ar);B_RAW.append(br);A_PACK.append(install(ar))
    GOLD_SPANS.append(gold_token_positions(it["A"],it["gold"]["seal"]))
    if i<4:print(f"      ITEM {i+1:02d} | A={A_PACK[-1][1]} B={br[0][0].shape[1]} tokens")
print(f"      {N*2} independent raw numeric memories forged.")
print(f"      Natural Mistral address dimension: {KVD}")

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
def all_pointers(packet,q):
    inst,Tm,P=packet;o,_,_,n=query_forward(packet,q);qpos=P+n-1;OUT=[]
    for L,layer in enumerate(layers):
        h=o.hidden_states[L][0,-1].to(layer.input_layernorm.weight.dtype);hn=layer.input_layernorm(h)
        qv=layer.self_attn.q_proj(hn).view(1,QH,1,HD)
        ak=inst[L][0].repeat_interleave(REP,dim=0).unsqueeze(0)
        dummy=torch.zeros((1,QH,1,HD),device=DEVICE,dtype=qv.dtype)
        c,s=rope_cos_sin(dummy,[qpos])
        qrot,_=apply_rotary_pos_emb(qv,dummy,c,s,unsqueeze_dim=1)
        score=torch.einsum("bhqd,bhkd->bhqk",qrot.float(),ak.float()).squeeze(0).squeeze(1)/math.sqrt(HD)
        score[:,0]=-torch.inf
        W=torch.softmax(score,dim=-1).detach().cpu()
        W[:,0]=0;W=W/W.sum(dim=1,keepdim=True).clamp_min(1e-12);OUT.append(W)
    return OUT

print("\n[3/10] ÇAĞRIİZ DISCOVERY — 32 layers x 32 query heads...")
PTR_DISC=[]
for z,i in enumerate(DISC):
    W=all_pointers(A_PACK[i],qA(ITEMS[i]["gold"]["entity"]));PTR_DISC.append(W)
    print(f"      DISC {z+1:02d}/{ND}",end="\r")
print()

STAT=[]
for L in range(NL):
    for h in range(QH):
        hit=[];mass=[];rank=[]
        for z,i in enumerate(DISC):
            w=PTR_DISC[z][L][h];sp=GOLD_SPANS[i];arg=int(torch.argmax(w))
            hit.append(int(arg in sp));mass.append(float(w[sp].sum()))
            best=float(w[sp].max());rank.append(1+int((w[1:]>best).sum()))
        STAT.append({"L":L,"H":h,"hit":float(np.mean(hit)),"mass":float(np.mean(mass)),"medrank":float(np.median(rank))})
STAT.sort(key=lambda x:(-x["hit"],-x["mass"],x["medrank"],x["L"],x["H"]))
print("      TOP POINTER CHANNELS — DISC")
for x in STAT[:16]:print(f"      L{x['L']:02d}H{x['H']:02d} | hit={x['hit']:.4f} mass={x['mass']:.6f} medrank={x['medrank']:.1f}")
PTR_L=STAT[0]["L"];PTR_H=STAT[0]["H"]
print(f"\n      SELECTED POINTER: L{PTR_L:02d}H{PTR_H:02d} | DISC hit={STAT[0]['hit']:.4f} mass={STAT[0]['mass']:.6f}")

print("\n[4/10] POINTER INTERNAL-EVAL — coordinates frozen...")
PTR_EVAL=[];EH=[];EM=[];ER=[]
for z,i in enumerate(EVAL):
    W=all_pointers(A_PACK[i],qA(ITEMS[i]["gold"]["entity"]))
    w=W[PTR_L][PTR_H];PTR_EVAL.append(w);sp=GOLD_SPANS[i]
    EH.append(int(int(torch.argmax(w)) in sp));EM.append(float(w[sp].sum()))
    best=float(w[sp].max());ER.append(1+int((w[1:]>best).sum()))
    print(f"      EVAL {z+1:02d}/{NE} | hit={EH[-1]} mass={EM[-1]:.4f} span-rank={ER[-1]:2d}")
print(f"      POINTER INTERNAL-EVAL: hit={np.mean(EH):.4f} mass={np.mean(EM):.6f} median-rank={np.median(ER):.1f}")

PTR_ALL=[None]*N
for z,i in enumerate(DISC):PTR_ALL[i]=PTR_DISC[z][PTR_L][PTR_H]
for z,i in enumerate(EVAL):PTR_ALL[i]=PTR_EVAL[z]
assert all(w is not None for w in PTR_ALL)
del PTR_DISC,PTR_EVAL;gc.collect();torch.cuda.empty_cache()

def cos_rows(v,M):
    v=v.float();M=M.float()
    nv=v.norm()
    if float(nv)<=1e-12:return torch.zeros(M.shape[0],dtype=torch.float32)
    v=v/nv;M=M/M.norm(dim=1,keepdim=True).clamp_min(1e-8)
    return M@v

def pointer_vector(raw,w,L,center):
    x=raw[L][1].float().cpu()
    if x.shape[1]!=len(w):raise RuntimeError(f"Pointer length {len(w)} != V length {x.shape[1]}")
    if center:
        mu=x[:,1:,:].mean(dim=1,keepdim=True)
        x=x-mu
    return (x*w[None,:,None]).sum(dim=1).reshape(-1)

def bmat(raw,L,center):
    x=raw[L][1][:,1:,:].float().cpu()
    if center:x=x-x.mean(dim=1,keepdim=True)
    return x.permute(1,0,2).reshape(x.shape[1],-1).contiguous()

def rank_local(sc,gold,indices):
    if len(sc)!=len(indices):raise RuntimeError("Score/index length mismatch.")
    order=sorted(range(len(indices)),key=lambda j:(-float(sc[j]),j))
    g=indices.index(gold)
    return order.index(g)+1,indices[order[0]]

def metrics(r):
    a=np.asarray(r,float)
    return float(np.mean(a==1)),float(np.mean(a<=5)),float(np.mean(a<=16)),float(np.mean(1/a))

print("\n[5/10] ADDRESS DISCOVERY — V layer x UNCENTERED/CENTERED on DISC only...")
ADDR=[]
for L in range(NL):
    for center in (False,True):
        BM=[bmat(B_RAW[i],L,center) for i in DISC];ranks=[]
        for i in DISC:
            p=pointer_vector(A_RAW[i],PTR_ALL[i],L,center)
            sc=[float(cos_rows(p,M).max()) for M in BM]
            r,_=rank_local(sc,i,DISC);ranks.append(r)
        m=metrics(ranks)
        ADDR.append({"L":L,"center":center,"R1":m[0],"R5":m[1],"R16":m[2],"MRR":m[3]})
        print(f"      L{L:02d}-V {'CENTERED  'if center else'UNCENTERED'} | R1={m[0]:.4f} R5={m[1]:.4f} R16={m[2]:.4f} MRR={m[3]:.6f}")

ADDR.sort(key=lambda x:(-x["R1"],-x["R5"],-x["MRR"],x["center"],x["L"]))
BEST=ADDR[0];BL=BEST["L"];CENTER=BEST["center"]
print("\n      TOP ADDRESS CANDIDATES — DISC")
for x in ADDR[:12]:print(f"      L{x['L']:02d}-V {'CENTERED  'if x['center'] else'UNCENTERED'} | R1={x['R1']:.4f} R5={x['R5']:.4f} MRR={x['MRR']:.6f}")
print(f"\n      SELECTED ADDRESS: L{BL:02d}-V · {'CENTERED'if CENTER else'UNCENTERED'}")

print("\n[6/10] ADDRESS INTERNAL-EVAL — address frozen...")
BM_E=[bmat(B_RAW[i],BL,CENTER) for i in EVAL];RE=[];SE=[]
for z,i in enumerate(EVAL):
    p=pointer_vector(A_RAW[i],PTR_ALL[i],BL,CENTER)
    sc=[float(cos_rows(p,M).max()) for M in BM_E]
    r,se=rank_local(sc,i,EVAL);RE.append(r);SE.append(se)
    print(f"      [{z+1:02d}/{NE}] rank={r:2d} selected={se+1:02d} expected={i+1:02d}")
ME=metrics(RE)
print(f"      INTERNAL-EVAL R1/R5/R16/MRR: {ME[0]:.4f} / {ME[1]:.4f} / {ME[2]:.4f} / {ME[3]:.6f}")

def uniform_pointer(T):
    w=torch.ones(T,dtype=torch.float32);w[0]=0
    return w/w.sum().clamp_min(1e-12)

def shifted_pointer(w):
    n=len(w)-1
    if n<=1:return w.clone()
    z=torch.zeros_like(w);z[1:]=torch.roll(w[1:].clone(),shifts=max(1,n//2),dims=0)
    return z/z.sum().clamp_min(1e-12)

print("\n[7/10] INTERNAL-EVAL CONTROLS — frozen pointer/address...")
RU=[];RS=[];RN=[];ZERO=torch.zeros(KVD,dtype=torch.float32)
for i in EVAL:
    wu=uniform_pointer(len(PTR_ALL[i]))
    p=pointer_vector(A_RAW[i],wu,BL,CENTER);sc=[float(cos_rows(p,M).max()) for M in BM_E]
    r,_=rank_local(sc,i,EVAL);RU.append(r)

    ws=shifted_pointer(PTR_ALL[i])
    p=pointer_vector(A_RAW[i],ws,BL,CENTER);sc=[float(cos_rows(p,M).max()) for M in BM_E]
    r,_=rank_local(sc,i,EVAL);RS.append(r)

    sc=[float(cos_rows(ZERO,M).max()) for M in BM_E]
    r,_=rank_local(sc,i,EVAL);RN.append(r)

MU=metrics(RU);MS=metrics(RS);MN=metrics(RN)
print(f"      PRIMARY : R1={ME[0]:.4f} R5={ME[1]:.4f} R16={ME[2]:.4f} MRR={ME[3]:.6f}")
print(f"      UNIFORM : R1={MU[0]:.4f} R5={MU[1]:.4f} R16={MU[2]:.4f} MRR={MU[3]:.6f}")
print(f"      SHIFTED : R1={MS[0]:.4f} R5={MS[1]:.4f} R16={MS[2]:.4f} MRR={MS[3]:.6f}")
print(f"      NO-A    : R1={MN[0]:.4f} R5={MN[1]:.4f} R16={MN[2]:.4f} MRR={MN[3]:.6f}")

print("\n[8/10] SOURCE / MODEL / PROTOCOL AUDIT...")
S1=sentinel()
print("      Model frozen                         :",S0==S1)
print("      Trainable parameter tensors          :",sum(int(p.requires_grad) for p in model.parameters()))
print("      Mistral geometry                     :",f"{NL}L / H{H} / QH{QH} / KVH{KVH} / HD{HD}")
print("      Natural address dimension            :",KVD)
print("      Native code pool                     :",len(POOL))
print("      Native codes consumed                :",NEED)
print("      A/B independent model forwards       :",N*2)
print("      Query-time candidate-B forwards      : 0")
print("      Pointer discovery                    : DISC only")
print("      Address discovery                    : DISC only")
print("      INTERNAL-EVAL used for selection     : NO")
print("      Gold span used in retrieval          : NO")
print("      Gold span use                        : pointer calibration diagnostic only")
print("      Training / LoRA / optimizer / DRA    : NONE")
print("      D120 / PCA / old cartridge readout   : NONE")

LOCK={"test":TEST,"model":MODEL_ID,"seed":SEED,"panel":{"N":N,"disc":ND,"internal_eval":NE},
"architecture":{"layers":NL,"hidden":H,"QH":QH,"KVH":KVH,"HD":HD,"address_dim":KVD},
"pointer":{"layer":PTR_L,"head":PTR_H,"disc_hit":STAT[0]["hit"],"disc_mass":STAT[0]["mass"],
"eval_hit":float(np.mean(EH)),"eval_mass":float(np.mean(EM))},
"address":{"layer":BL,"kind":"V","center":CENTER,"match":"max_cosine",
"disc_R1":BEST["R1"],"disc_R5":BEST["R5"],"disc_MRR":BEST["MRR"],
"eval_R1":ME[0],"eval_R5":ME[1],"eval_R16":ME[2],"eval_MRR":ME[3]},
"controls":{"uniform_R1":MU[0],"shifted_R1":MS[0],"noA_R1":MN[0]},
"selection":"DISC ONLY; INTERNAL-EVAL untouched until coordinates frozen",
"next":"fresh external replication with zero discovery",
"forbidden":["D120","PCA","LEARNED_ROUTER","HIDDEN_SCORER","LM_HEAD_ROUTE","TOKEN_ID_ADDRESS",
"DECODED_SEAL_ADDRESS","DRA","TRAINING","QUERY_TIME_B_FORWARD"]}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()

print("\n[9/10] CALIBRATION DECISION...")
POINTER_OK=float(np.mean(EH))>=.75
ADDRESS_OK=ME[0]>=.75
CAUSAL_OK=(ME[0]-MU[0]>=.40 and ME[0]-MS[0]>=.40 and ME[0]-MN[0]>=.40)
if POINTER_OK and ADDRESS_OK and CAUSAL_OK:VERDICT="MISTRAL_MAM_CALIBRATION_LOCK_READY"
elif ME[0]>=.50 and ME[0]-MS[0]>=.25:VERDICT="MISTRAL_MAM_SIGNAL_FOUND_BUT_CALIBRATION_NOT_YET_LOCK_READY"
else:VERDICT="MISTRAL_MAM_CALIBRATION_NOT_ESTABLISHED"
print("      Pointer eval >= .75              :",POINTER_OK)
print("      Address eval R1 >= .75           :",ADDRESS_OK)
print("      Primary-control gap >= .40       :",CAUSAL_OK)
print("      VERDICT                          :",VERDICT)

print("\n[10/10] FINAL...")
print("\n"+"="*176)
print("TEST525 FINAL RESULT — MISTRAL MAM CALIBRATION")
print("="*176)
print("MODEL                              :",MODEL_ID,"· frozen")
print("CALIBRATION PANEL                  :",f"{N} items · DISC {ND} / INTERNAL-EVAL {NE}")
print("MISTRAL GEOMETRY                   :",f"{NL}L · H={H} · QH={QH} · KVH={KVH} · HD={HD}")
print("ADDRESS DIMENSION                  :",KVD)
print("-"*176)
print(f"POINTER LOCK CANDIDATE             : L{PTR_L:02d}H{PTR_H:02d}")
print(f"POINTER DISC HIT / MASS            : {STAT[0]['hit']:.4f} / {STAT[0]['mass']:.6f}")
print(f"POINTER INTERNAL-EVAL HIT / MASS   : {np.mean(EH):.4f} / {np.mean(EM):.6f}")
print(f"ADDRESS LOCK CANDIDATE             : L{BL:02d}-V · {'CENTERED'if CENTER else'UNCENTERED'}")
print(f"ADDRESS DISC R1/R5/MRR             : {BEST['R1']:.4f} / {BEST['R5']:.4f} / {BEST['MRR']:.6f}")
print(f"ADDRESS EVAL R1/R5/R16/MRR         : {ME[0]:.4f} / {ME[1]:.4f} / {ME[2]:.4f} / {ME[3]:.6f}")
print(f"UNIFORM EVAL R1                    : {MU[0]:.4f}")
print(f"SHIFTED EVAL R1                    : {MS[0]:.4f}")
print(f"NO-A EVAL R1                       : {MN[0]:.4f}")
print("-"*176)
print("WEIGHT SENTINEL                    :","PASS" if S0==S1 else"FAIL")
print("QUERY-TIME CANDIDATE-B FORWARDS    : 0")
print("TEST525 LOCK SHA                   :",LOCK_SHA)
print("TOTAL TEST TIME                    :",f"{time.perf_counter()-T0:.2f}s")
print("VERDICT                            :",VERDICT)
if VERDICT=="MISTRAL_MAM_CALIBRATION_LOCK_READY":
    print(f"NEXT                               : freeze L{PTR_L:02d}H{PTR_H:02d} -> L{BL:02d}-V {'CENTERED'if CENTER else'UNCENTERED'} and run fresh external TEST526")
else:
    print("NEXT                               : inspect TEST525 telemetry; DO NOT touch final external panel")
print("="*176)

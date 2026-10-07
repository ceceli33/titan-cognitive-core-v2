# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST531
# VARAN 1 FINAL — UNTOUCHED MULTI-TOKEN IDENTITY VALIDATION
#
# SEALED CHAIN:
#   TEST528 -> native single-token external seal
#   TEST529 -> multi-token primitive boundary
#   TEST530 -> composition X-ray
#              CONCATCOS = 512/512 R1
#              CONCAT_REVERSE = 0.0039 R1
#
# LOCKED BEFORE TEST531:
#   MODEL   = mistralai/Mistral-7B-Instruct-v0.3
#   POINTER = L28H00
#   ADDRESS = L00-V UNCENTERED
#   ADDRESS DIM = 1024
#   IDENTITY LENGTHS = 2/3/4
#   MATCH = position-preserving CONCATCOS
#
# TEST531 PURPOSE:
#   Fresh untouched external validation of the TEST530 composition law.
#   No arm comparison.
#   No coordinate discovery.
#   No tuning.
#   No post-hoc mechanism selection.
#
# CONCATCOS:
#   For an identity span of length L:
#       score(window) = mean_j cosine(A_j, B_j)
#   All contiguous B windows of the required L are candidates.
#   The pointer determines identity start/span on A numerically.
#   Gold spans are diagnostic only and NEVER used for retrieval.
#
# IMPORTANT:
#   Retrieval does NOT use decoded identity, token IDs, gold B positions,
#   candidate-B model forwards, training, LoRA, DRA or learned routing.
#
# PREDECLARED VARAN-1 FINAL GATES:
#   OVERALL R1 >= 0.95
#   OVERALL R5 >= 0.99
#   EACH L=2/3/4 R1 >= 0.93
#   COUNTERFACTUAL R1 >= 0.93
#   POINTER HIT >= 0.90
#   CF POINTER HIT >= 0.90
#   SHIFTED R1 <= 0.02
#   NO-A R1 <= 0.02
#   WEIGHT SENTINEL PASS
#   QUERY-TIME CANDIDATE-B FORWARDS = 0

import os,sys,subprocess,importlib.util,random,time,hashlib,json,gc,re,math
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

TEST="531";SEED=531531;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";N=1024
DEVICE=torch.device("cuda");FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
PTR_L=28;PTR_H=0;BL=0;BK="V";CENTER=False;LENS=(2,3,4)
TH_R1=.95;TH_R5=.99;TH_EACH=.93;TH_CF=.93;TH_PTR=.90;TH_NEG=.02
PARENT528="9235ed38c944f41a9ede0426d3aabaa12b5b20646f30e2be921e618f9471d3c2"
PARENT529="27ba703872f0d8a6ed870a6ba041d0cd0703d2eca467deac97186a90497795a0"
PARENT530="289247c88dfd4358cbe5ffb8499891546b27acc6912caf525a155ae81338db7b"
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*176)
print("TEST531 — AKBASCORE MAM · VARAN 1 FINAL · UNTOUCHED MULTI-TOKEN IDENTITY VALIDATION")
print("FRESH N=1024 · L=2/3/4 · FROZEN L28H00 → L00-V · LOCKED CONCATCOS · NO SELECTION / NO TUNING")
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
print("      MATCH   : CONCATCOS · FROZEN BEFORE TEST531")
print("      PANEL   : 1024 fresh identities · L=2/3/4")
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

print("\n[2/12] Building fresh deterministic identity panel...")
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
print("      Required identity pools:",need)
print("      Generated identity pools:",{L:len(ALL[L]) for L in LENS})

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
        out.append({"id":i+1,"L":L,"A":" ".join(ra),"B":" ".join(rb),"entity":ents[0],"gold":gold,
                    "A_d":[a1,a2]})
    return out

ITEMS=make_items()
for L in LENS:print(f"      L={L}: {sum(x['L']==L for x in ITEMS)}")
print("      Remaining reserve:",{L:len(ALL[L]) for L in LENS})
print("      Fresh deterministic seed:",SEED)

LOCK={"test":TEST,"parents":{"528":PARENT528,"529":PARENT529,"530":PARENT530},"model":MODEL_ID,"seed":SEED,"N":N,
"status":"VARAN1_FINAL_UNTOUCHED_VALIDATION","pointer":{"layer":PTR_L,"head":PTR_H},
"address":{"layer":BL,"kind":BK,"center":CENTER,"dimension":KVD},
"identity":{"lengths":[2,3,4],"composition":"ordered native-token sequence"},
"match":{"name":"CONCATCOS","formula":"mean position-wise cosine","locked_from":"TEST530"},
"selection":"NONE","gold_usage":"DIAGNOSTIC_ONLY",
"controls":["shifted","noA_random","counterfactual"],
"forbidden":["ARM_SCAN","HEAD_SCAN","LAYER_SCAN","ADDRESS_SCAN","CENTER_SCAN","TOKEN_ID_ROUTING",
"DECODED_ID_ROUTING","GOLD_B_WINDOW_SELECTION","QUERY_TIME_B_FORWARD","TRAINING","LORA","DRA",
"LEARNED_ROUTER","POSTHOC_SELECTION","THRESHOLD_TUNING"]}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      TEST531 LOCK:",LOCK_SHA)

@torch.inference_mode()
def kv_from_ids(ids):
    o=model(input_ids=torch.tensor([ids],device=DEVICE),use_cache=False,output_hidden_states=True,return_dict=True);out=[]
    for L,layer in enumerate(layers):
        h=o.hidden_states[L][0].to(layer.input_layernorm.weight.device,layer.input_layernorm.weight.dtype)
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
    for L,(k,v) in enumerate(inst):c.update(k.unsqueeze(0),v.unsqueeze(0),L)
    return c

def qA(e):return f"What seal does instrument {e} carry? Give only the exact seal."

print("\n[3/12] OFFLINE FORGE — fresh A/B memories...")
A_RAW=[];A_PACK=[];GOLD_SPANS=[];B_BANK={L:{"vec":[],"own":[]} for L in LENS}
for i,it in enumerate(ITEMS):
    ar=forge(it["A"]);br=forge(it["B"]);asp=exact_span(it["A"],it["gold"])
    if asp is None or len(asp)!=it["L"]:raise RuntimeError(f"A span mismatch item {i+1}")
    A_RAW.append(ar);A_PACK.append(install(ar));GOLD_SPANS.append(asp)
    x=br[BL][1][:,1:,:].float().cpu().permute(1,0,2).reshape(-1,KVD).contiguous();L=it["L"]
    if len(x)<L:raise RuntimeError(f"B memory too short item {i+1}: tokens={len(x)} L={L}")
    wins=torch.stack([x[j:j+L] for j in range(len(x)-L+1)])
    B_BANK[L]["vec"].append(wins);B_BANK[L]["own"].extend([i]*(len(x)-L+1))
    del br,x,wins
    if (i+1)%64==0:print(f"      forged {i+1:4d}/{N}")
gc.collect();torch.cuda.empty_cache()
print("      A memories:",len(A_RAW))
print("      B memories:",N)
print("      Query-time candidate-B forwards: 0")

print("\n[4/12] Building locked CONCATCOS candidate banks...")
for L in LENS:
    if not B_BANK[L]["vec"]:raise RuntimeError(f"Empty B bank for L={L}")
    X=torch.cat(B_BANK[L]["vec"],0)
    X=X/X.norm(dim=2,keepdim=True).clamp_min(1e-8)
    B_BANK[L]["vec"]=X.to(DEVICE,dtype=torch.float16)
    B_BANK[L]["own"]=torch.tensor(B_BANK[L]["own"],device=DEVICE,dtype=torch.long)
    print(f"      L={L} windows: {len(B_BANK[L]['own'])} | shape={tuple(B_BANK[L]['vec'].shape)}")
    del X
print("      Every contiguous B window of matching length indexed.")
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
    return w[PTR_H]/w[PTR_H].sum().clamp_min(1e-12)

def a_sequence(raw,start,L):
    x=raw[BL][1].float().cpu().permute(1,0,2).reshape(-1,KVD)
    if start<1 or start+L>len(x):return None
    return x[start:start+L]

@torch.inference_mode()
def concat_scores(Aseq,L):
    A=Aseq.to(DEVICE);A=A/A.norm(dim=1,keepdim=True).clamp_min(1e-8)
    B=B_BANK[L]["vec"];own=B_BANK[L]["own"]
    sim=(B.float()*A.float().unsqueeze(0)).sum(dim=2).mean(dim=1)
    out=torch.full((N,),-torch.inf,device=DEVICE)
    out.scatter_reduce_(0,own,sim,reduce="amax",include_self=True)
    return out.cpu()

def rank(sc,g):
    order=torch.argsort(sc,descending=True).tolist()
    return order.index(g)+1,order[0],order

def metrics(r):
    a=np.asarray(r,float)
    return float(np.mean(a==1)),float(np.mean(a<=5)),float(np.mean(a<=16)),float(np.mean(1/a))

def invalid_scores():
    return torch.full((N,),-torch.inf)

def shifted_start(w,L):
    n=len(w)-1
    if n<L:return 1
    s=int(torch.argmax(w[1:]))+1
    return 1+((s-1+max(L,n//2))%max(1,n-L+1))

print("\n[5/12] PRIMARY — locked CONCATCOS only...")
RPRI=[];SEL=[];PHIT=[];PMASS=[];MARG=[];START=[];BYLEN={L:[] for L in LENS}
for i,it in enumerate(ITEMS):
    L=it["L"];w=frozen_pointer(A_PACK[i],qA(it["entity"]));start=int(torch.argmax(w))
    seq=a_sequence(A_RAW[i],start,L)
    sc=invalid_scores() if seq is None else concat_scores(seq,L)
    r,se,o=rank(sc,i);RPRI.append(r);SEL.append(se);START.append(start);BYLEN[L].append(r)
    sp=GOLD_SPANS[i];PHIT.append(int(start in sp));PMASS.append(float(w[sp].sum()))
    MARG.append(float(sc[o[0]]-sc[o[1]]) if torch.isfinite(sc[o[0]]) and torch.isfinite(sc[o[1]]) else float("nan"))
    if i<8 or (i+1)%16==0:
        print(f"      [{i+1:04d}/{N}] L={L} start={start:3d} gold={sp[0]:3d} rank={r:4d} sel={se+1:04d} ptr={PHIT[-1]} mass={PMASS[-1]:.4f} margin={MARG[-1]:+.6f}")

MP=metrics(RPRI);ML={L:metrics(BYLEN[L]) for L in LENS};FINITE_MARGIN=[x for x in MARG if math.isfinite(x)]
print(f"      PRIMARY R1/R5/R16/MRR : {MP[0]:.4f}/{MP[1]:.4f}/{MP[2]:.4f}/{MP[3]:.6f}")
for L in LENS:print(f"      L={L} R1/R5/MRR       : {ML[L][0]:.4f}/{ML[L][1]:.4f}/{ML[L][3]:.6f} n={len(BYLEN[L])}")
print(f"      POINTER HIT            : {np.mean(PHIT):.4f}")
print(f"      POINTER MASS           : {np.mean(PMASS):.6f}")
print(f"      MEAN MARGIN            : {np.mean(FINITE_MARGIN) if FINITE_MARGIN else float('nan'):+.6f}")

print("\n[6/12] NEGATIVE CONTROLS...")
RSHIFT=[];RNO=[];rr_no=random.Random(SEED+999999)
for i,it in enumerate(ITEMS):
    L=it["L"];w=frozen_pointer(A_PACK[i],qA(it["entity"]))
    ss=shifted_start(w,L);seq=a_sequence(A_RAW[i],ss,L)
    if seq is None:r=N
    else:r,_,_=rank(concat_scores(seq,L),i)
    RSHIFT.append(r)
    order=list(range(N));rr_no.shuffle(order);RNO.append(order.index(i)+1)
MS=metrics(RSHIFT);MN=metrics(RNO)
print(f"      PRIMARY : R1={MP[0]:.4f} R5={MP[1]:.4f}")
print(f"      SHIFTED : R1={MS[0]:.4f} R5={MS[1]:.4f}")
print(f"      NO-A    : R1={MN[0]:.4f} R5={MN[1]:.4f}")

print("\n[7/12] COUNTERFACTUAL FOLLOW...")
RCF=[];CFHIT=[];CFMASS=[]
for i,it in enumerate(ITEMS):
    j=(i+353)%N;target=TARGETS[j];L=TLEN[j];src=ITEMS[j]
    ents=[it["entity"],NAMES[3*i+1],NAMES[3*i+2]]
    rows=[f"Instrument {ents[0]} carries seal {target}.",
          f"Instrument {ents[1]} carries seal {src['A_d'][0]}.",
          f"Instrument {ents[2]} carries seal {src['A_d'][1]}."]
    random.Random(SEED+90000+i*149).shuffle(rows);cfA=" ".join(rows)
    raw=forge(cfA);pack=install(raw);sp=exact_span(cfA,target)
    if sp is None or len(sp)!=L:raise RuntimeError(f"CF span mismatch item {i+1}")
    w=frozen_pointer(pack,qA(it["entity"]));start=int(torch.argmax(w));seq=a_sequence(raw,start,L)
    sc=invalid_scores() if seq is None else concat_scores(seq,L)
    r,se,_=rank(sc,j);RCF.append(r);CFHIT.append(int(start in sp));CFMASS.append(float(w[sp].sum()))
    if i<5 or (i+1)%32==0:
        print(f"      [{i+1:04d}/{N}] L={L} target={j+1:04d} rank={r:4d} sel={se+1:04d} ptr={CFHIT[-1]} mass={CFMASS[-1]:.4f}")
    del raw,pack,w,seq,sc
MC=metrics(RCF)
print(f"      CF R1/R5/R16/MRR : {MC[0]:.4f}/{MC[1]:.4f}/{MC[2]:.4f}/{MC[3]:.6f}")
print(f"      CF POINTER HIT    : {np.mean(CFHIT):.4f}")
print(f"      CF POINTER MASS   : {np.mean(CFMASS):.6f}")

print("\n[8/12] FAILURE / LENGTH DIAGNOSTICS...")
for L in LENS:print(f"      L={L} failures: {sum(r!=1 for r in BYLEN[L])}/{len(BYLEN[L])}")
print(f"      Overall failures       : {sum(r!=1 for r in RPRI)}/{N}")
print(f"      Primary failure IDs    : {[i+1 for i,r in enumerate(RPRI) if r!=1][:50]}")
print(f"      CF failure IDs         : {[i+1 for i,r in enumerate(RCF) if r!=1][:50]}")
print(f"      Unique selections      : {len(set(SEL))}/{N}")

print("\n[9/12] STATISTICAL SUMMARY...")
def wilson(k,n,z=1.959963984540054):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h
for name,r in [("PRIMARY",RPRI),("L2",BYLEN[2]),("L3",BYLEN[3]),("L4",BYLEN[4]),("SHIFTED",RSHIFT),("NO_A",RNO),("CF",RCF)]:
    k=sum(int(x==1) for x in r);lo,hi=wilson(k,len(r))
    print(f"      {name:8s}: {k:4d}/{len(r):4d} = {k/len(r):.4f} | 95% Wilson [{lo:.4f},{hi:.4f}]")

print("\n[10/12] PREDECLARED VARAN-1 FINAL GATES...")
GATES={
"OVERALL_R1":MP[0]>=TH_R1,
"OVERALL_R5":MP[1]>=TH_R5,
"L2_R1":ML[2][0]>=TH_EACH,
"L3_R1":ML[3][0]>=TH_EACH,
"L4_R1":ML[4][0]>=TH_EACH,
"COUNTERFACTUAL_R1":MC[0]>=TH_CF,
"POINTER_HIT":float(np.mean(PHIT))>=TH_PTR,
"CF_POINTER_HIT":float(np.mean(CFHIT))>=TH_PTR,
"SHIFTED_R1":MS[0]<=TH_NEG,
"NOA_R1":MN[0]<=TH_NEG}
for k,v in GATES.items():print(f"      {k:24s}: {'PASS' if v else 'FAIL'}")

print("\n[11/12] PROTOCOL AUDIT...")
S1=sentinel();WEIGHT_OK=S0==S1
print("      Weight sentinel                     :","PASS" if WEIGHT_OK else "FAIL")
print("      Trainable tensors                   :",sum(int(p.requires_grad) for p in model.parameters()))
print("      Pointer                             : L28H00 frozen")
print("      Address                             : L00-V uncentered frozen")
print("      Match                               : CONCATCOS frozen from TEST530")
print("      Alternative arms tested             : NO")
print("      Head/layer/address discovery        : NONE")
print("      Threshold tuning                    : NONE")
print("      Gold span used for retrieval        : NO")
print("      Pointer argmax determines A start   : YES")
print("      All matching-length B windows       : YES")
print("      Gold B position used                : NO")
print("      Token-ID routing                    : NO")
print("      Decoded identity routing            : NO")
print("      Query-time candidate-B forwards     : 0")
print("      Training / LoRA / optimizer / DRA   : NONE")
print("      Learned router/scorer               : NONE")
print("      Post-hoc mechanism selection        : NONE")
GATES["WEIGHT_SENTINEL"]=WEIGHT_OK

print("\n[12/12] FINAL...")
ALL_PASS=all(GATES.values())
VERDICT="MISTRAL_MAM_VARAN1_FINAL_PASS" if ALL_PASS else "MISTRAL_MAM_VARAN1_FINAL_FAIL"
MEAN_MARGIN=float(np.mean(FINITE_MARGIN)) if FINITE_MARGIN else float("nan")
RESULT={"test":TEST,"parents":{"528":PARENT528,"529":PARENT529,"530":PARENT530},"lock_sha":LOCK_SHA,
"primary":{"r1":MP[0],"r5":MP[1],"r16":MP[2],"mrr":MP[3]},
"lengths":{str(L):{"r1":ML[L][0],"r5":ML[L][1],"r16":ML[L][2],"mrr":ML[L][3]} for L in LENS},
"counterfactual":{"r1":MC[0],"r5":MC[1],"r16":MC[2],"mrr":MC[3]},
"shifted":{"r1":MS[0],"r5":MS[1],"r16":MS[2],"mrr":MS[3]},
"noA":{"r1":MN[0],"r5":MN[1],"r16":MN[2],"mrr":MN[3]},
"pointer_hit":float(np.mean(PHIT)),"pointer_mass":float(np.mean(PMASS)),
"cf_pointer_hit":float(np.mean(CFHIT)),"cf_pointer_mass":float(np.mean(CFMASS)),
"mean_margin":MEAN_MARGIN,"gates":GATES,"verdict":VERDICT}
RESULT_SHA=hashlib.sha256(json.dumps(RESULT,sort_keys=True,separators=(",",":")).encode()).hexdigest()

print("\n"+"="*176)
print("TEST531 FINAL RESULT — AKBASCORE MAM · VARAN 1 FINAL")
print("="*176)
print("MODEL                              :",MODEL_ID,"· frozen")
print("EXTERNAL PANEL                     : 1024 fresh deterministic identities")
print("POINTER                            : L28H00 · FROZEN")
print("ADDRESS                            : L00-V · UNCENTERED · 1024D · FROZEN")
print("IDENTITY                           : ORDERED MODEL-NATIVE 2/3/4 TOKEN")
print("MATCH                              : LOCKED CONCATCOS")
print("-"*176)
print(f"PRIMARY                            : R1={MP[0]:.4f} R5={MP[1]:.4f} R16={MP[2]:.4f} MRR={MP[3]:.6f}")
for L in LENS:print(f"L={L:1d}                                : R1={ML[L][0]:.4f} R5={ML[L][1]:.4f} R16={ML[L][2]:.4f} MRR={ML[L][3]:.6f}")
print(f"COUNTERFACTUAL                     : R1={MC[0]:.4f} R5={MC[1]:.4f} R16={MC[2]:.4f} MRR={MC[3]:.6f}")
print(f"SHIFTED                            : R1={MS[0]:.4f} R5={MS[1]:.4f} R16={MS[2]:.4f} MRR={MS[3]:.6f}")
print(f"NO-A                               : R1={MN[0]:.4f} R5={MN[1]:.4f} R16={MN[2]:.4f} MRR={MN[3]:.6f}")
print(f"POINTER HIT                        : {np.mean(PHIT):.4f}")
print(f"POINTER MASS                       : {np.mean(PMASS):.6f}")
print(f"CF POINTER HIT                     : {np.mean(CFHIT):.4f}")
print(f"CF POINTER MASS                    : {np.mean(CFMASS):.6f}")
print(f"MEAN TOP1-TOP2 MARGIN              : {MEAN_MARGIN:+.6f}")
print("-"*176)
for k,v in GATES.items():print(f"{k:35s}: {'PASS' if v else 'FAIL'}")
print("-"*176)
print("WEIGHT SENTINEL                    :","PASS" if WEIGHT_OK else "FAIL")
print("QUERY-TIME CANDIDATE-B FORWARDS    : 0")
print("TEST528 RESULT SHA                 :",PARENT528)
print("TEST529 RESULT SHA                 :",PARENT529)
print("TEST530 RESULT SHA                 :",PARENT530)
print("TEST531 PRE-RESULT LOCK            :",LOCK_SHA)
print("TEST531 RESULT SHA                 :",RESULT_SHA)
print("TOTAL TEST TIME                    :",f"{time.perf_counter()-T0:.2f}s")
print("VERDICT                            :",VERDICT)
if ALL_PASS:
    print("STATUS                             : VARAN 1 SEALED")
    print("NEXT                               : VARAN 2 — rich-memory pointer selectivity")
else:
    print("STATUS                             : VARAN 1 NOT SEALED")
    print("NEXT                               : preserve TEST531 unchanged; diagnose only in a new test")
print("="*176)

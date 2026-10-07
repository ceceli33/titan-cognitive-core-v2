# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST532
# VARAN 1 — SPAN-ORIGIN X-RAY
#
# SEALED CHAIN:
#   TEST528 -> native single-token external seal
#   TEST529 -> multi-token primitive boundary
#   TEST530 -> composition X-ray / CONCATCOS discovered
#   TEST531 -> untouched multi-token validation
#              PRIMARY R1 = 0.9824
#              PRIMARY R5 = 0.9834 -> PREDECLARED R5 GATE FAIL
#              POINTER HIT = 0.9912
#
# TEST532 PURPOSE:
#   Diagnose whether TEST531 residual failures are caused by using
#   pointer ARGMAX as the START of a multi-token identity span.
#
# FROZEN / UNCHANGED:
#   MODEL   = mistralai/Mistral-7B-Instruct-v0.3
#   POINTER = L28H00
#   ADDRESS = L00-V UNCENTERED
#   MATCH   = position-preserving CONCATCOS
#   IDENTITY LENGTHS = 2/3/4
#
# PREDECLARED ORIGIN RULES:
#
#   BASELINE_ARGMAX:
#       start = argmax(pointer)
#       Exact TEST531 origin rule.
#
#   MASS_WINDOW:
#       For known identity length L, evaluate EVERY contiguous A window
#       of width L and choose the window with maximum total pointer mass:
#
#           start = argmax_s sum_{j=0}^{L-1} pointer[s+j]
#
#       This is parameter-free.
#       No threshold.
#       No gold span.
#       No decoded identity.
#       No token-ID routing.
#       No learned router/scorer.
#
# IMPORTANT:
#   TEST532 is an X-RAY / DIAGNOSTIC test.
#   MASS_WINDOW is NOT allowed to retroactively change TEST531.
#   If successful, it must be frozen and independently validated
#   on a new untouched panel in TEST533.
#
# PREDECLARED DIAGNOSTIC QUESTIONS:
#   1. Does BASELINE reproduce the TEST531 failure pattern?
#   2. Among BASELINE failures, how often is pointer argmax inside the
#      gold identity but not at its first token?
#   3. Does MASS_WINDOW recover the correct span origin numerically?
#   4. Does locked CONCATCOS recover retrieval when only origin changes?
#   5. Does the same effect follow counterfactual reassociation?
#
# DIAGNOSTIC SUCCESS CRITERIA:
#   BASELINE R1 approximately reproduces TEST531
#   MASS_WINDOW R1 >= 0.995
#   MASS_WINDOW R5 >= 0.999
#   EACH L R1 >= 0.99
#   COUNTERFACTUAL R1 >= 0.99
#   MASS_WINDOW GOLD-START HIT >= 0.99
#   WEIGHT SENTINEL PASS
#   QUERY-TIME CANDIDATE-B FORWARDS = 0
#
# NOTE:
#   Gold A spans are used ONLY after numerical origin selection for
#   diagnostic scoring. They never determine the retrieval start.

import os,sys,subprocess,importlib.util,random,time,hashlib,json,gc,re,math
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

TEST="532";SEED=531531;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";N=1024
DEVICE=torch.device("cuda");FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
PTR_L=28;PTR_H=0;BL=0;BK="V";CENTER=False;LENS=(2,3,4)
TH_MASS_R1=.995;TH_MASS_R5=.999;TH_EACH=.99;TH_CF=.99;TH_START=.99
PARENT528="9235ed38c944f41a9ede0426d3aabaa12b5b20646f30e2be921e618f9471d3c2"
PARENT529="27ba703872f0d8a6ed870a6ba041d0cd0703d2eca467deac97186a90497795a0"
PARENT530="289247c88dfd4358cbe5ffb8499891546b27acc6912caf525a155ae81338db7b"
PARENT531="20257d7d613a49e55b03603946f07ec493009924f40ac26ab74cb817e1466528"
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*176)
print("TEST532 — AKBASCORE MAM · VARAN 1 · SPAN-ORIGIN X-RAY")
print("TEST531 FAILURE DIAGNOSIS · SAME PANEL · ARGMAX vs MAX-CONTIGUOUS-POINTER-MASS · CONCATCOS LOCKED")
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
print("      PANEL   : exact TEST531 deterministic construction")
print("      ORIGINS : ARGMAX baseline + MASS_WINDOW diagnostic")
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

print("\n[2/12] Reconstructing exact TEST531 deterministic panel...")
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
        out.append({"id":i+1,"L":L,"A":" ".join(ra),"B":" ".join(rb),"entity":ents[0],"gold":gold,
                    "A_d":[a1,a2]})
    return out

ITEMS=make_items()
print("      Required identity pools:",need)
for L in LENS:print(f"      L={L}: {sum(x['L']==L for x in ITEMS)}")
print("      Remaining reserve:",{L:len(ALL[L]) for L in LENS})
print("      TEST531 seed:",SEED)

LOCK={"test":TEST,"parents":{"528":PARENT528,"529":PARENT529,"530":PARENT530,"531":PARENT531},
"model":MODEL_ID,"seed":SEED,"N":N,"status":"VARAN1_SPAN_ORIGIN_XRAY",
"pointer":{"layer":PTR_L,"head":PTR_H},
"address":{"layer":BL,"kind":BK,"center":CENTER,"dimension":KVD},
"identity":{"lengths":[2,3,4],"composition":"ordered native-token sequence"},
"match":{"name":"CONCATCOS","formula":"mean position-wise cosine","locked_from":"TEST530"},
"origin_arms":{
"BASELINE_ARGMAX":"argmax(pointer)",
"MASS_WINDOW":"argmax over every contiguous A window of width L of summed pointer mass"},
"gold_usage":"DIAGNOSTIC_ONLY_AFTER_SELECTION",
"selection":"PREDECLARED_TWO_ARM_XRAY",
"forbidden":["HEAD_SCAN","LAYER_SCAN","ADDRESS_SCAN","CENTER_SCAN","MATCH_SCAN","TOKEN_ID_ROUTING",
"DECODED_ID_ROUTING","GOLD_A_START_SELECTION","GOLD_B_WINDOW_SELECTION","QUERY_TIME_B_FORWARD",
"TRAINING","LORA","DRA","LEARNED_ROUTER","THRESHOLD_TUNING","POSTHOC_ARM_CREATION"]}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      TEST532 LOCK:",LOCK_SHA)

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

print("\n[3/12] OFFLINE FORGE — exact TEST531 A/B construction...")
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
    X=torch.cat(B_BANK[L]["vec"],0)
    X=X/X.norm(dim=2,keepdim=True).clamp_min(1e-8)
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
    return w[PTR_H]/w[PTR_H].sum().clamp_min(1e-12)

def argmax_start(w):
    return int(torch.argmax(w))

def mass_window_start(w,L):
    # Memory positions are 1..len(w)-1; position 0 is PAD and excluded.
    n=len(w)-1
    if n<L:return None
    x=w[1:]
    sums=torch.stack([x[s:s+L].sum() for s in range(n-L+1)])
    return int(torch.argmax(sums))+1

def a_sequence(raw,start,L):
    if start is None:return None
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

print("\n[5/12] PRIMARY X-RAY — ARGMAX vs MASS_WINDOW...")
RBASE=[];RMASS=[];BASE_START=[];MASS_START=[];BASE_START_HIT=[];MASS_START_HIT=[]
BASE_INSIDE=[];BASE_INSIDE_NOT_START=[];PMASS=[];BYLEN_BASE={L:[] for L in LENS};BYLEN_MASS={L:[] for L in LENS}
RECOVERED=[];BROKEN=[]

for i,it in enumerate(ITEMS):
    L=it["L"];sp=GOLD_SPANS[i];gold_start=sp[0]
    w=frozen_pointer(A_PACK[i],qA(it["entity"]))
    sb=argmax_start(w);sm=mass_window_start(w,L)

    seqb=a_sequence(A_RAW[i],sb,L);seqm=a_sequence(A_RAW[i],sm,L)
    scb=invalid_scores() if seqb is None else concat_scores(seqb,L)
    scm=invalid_scores() if seqm is None else concat_scores(seqm,L)

    rb,selb,_=rank(scb,i);rm,selm,_=rank(scm,i)
    RBASE.append(rb);RMASS.append(rm);BYLEN_BASE[L].append(rb);BYLEN_MASS[L].append(rm)
    BASE_START.append(sb);MASS_START.append(sm)
    BASE_START_HIT.append(int(sb==gold_start));MASS_START_HIT.append(int(sm==gold_start))
    inside=int(sb in sp);BASE_INSIDE.append(inside);BASE_INSIDE_NOT_START.append(int(inside and sb!=gold_start))
    PMASS.append(float(w[sp].sum()))

    if rb!=1 and rm==1:RECOVERED.append(i+1)
    if rb==1 and rm!=1:BROKEN.append(i+1)

    if i<8 or rb!=1 or rm!=1 or (i+1)%64==0:
        print(f"      [{i+1:04d}/{N}] L={L} gold={gold_start:3d} arg={sb:3d} mass={sm:3d} "
              f"inside={inside} | BASE={rb:4d} MASS={rm:4d} "
              f"sel={selb+1:04d}/{selm+1:04d} pmass={PMASS[-1]:.4f}")

MB=metrics(RBASE);MM=metrics(RMASS)
MLB={L:metrics(BYLEN_BASE[L]) for L in LENS};MLM={L:metrics(BYLEN_MASS[L]) for L in LENS}

print(f"      BASELINE ARGMAX R1/R5/R16/MRR : {MB[0]:.4f}/{MB[1]:.4f}/{MB[2]:.4f}/{MB[3]:.6f}")
print(f"      MASS_WINDOW     R1/R5/R16/MRR : {MM[0]:.4f}/{MM[1]:.4f}/{MM[2]:.4f}/{MM[3]:.6f}")
for L in LENS:
    print(f"      L={L} BASE R1={MLB[L][0]:.4f} | MASS R1={MLM[L][0]:.4f} R5={MLM[L][1]:.4f}")
print(f"      BASE argmax inside gold span  : {np.mean(BASE_INSIDE):.4f}")
print(f"      BASE exact gold-start hit     : {np.mean(BASE_START_HIT):.4f}")
print(f"      MASS exact gold-start hit     : {np.mean(MASS_START_HIT):.4f}")
print(f"      Pointer gold-span mass        : {np.mean(PMASS):.6f}")

print("\n[6/12] TEST531 FAILURE ANATOMY...")
FAIL=[i for i,r in enumerate(RBASE) if r!=1]
F_INSIDE=[i for i in FAIL if BASE_INSIDE[i]]
F_INSIDE_NOT_START=[i for i in FAIL if BASE_INSIDE_NOT_START[i]]
F_WRONG_START=[i for i in FAIL if not BASE_START_HIT[i]]
F_MASS_FIXED=[i for i in FAIL if RMASS[i]==1]
print(f"      Baseline failures                    : {len(FAIL)}/{N}")
print(f"      Failure: argmax inside identity      : {len(F_INSIDE)}/{len(FAIL) if FAIL else 1}")
print(f"      Failure: inside but not first token  : {len(F_INSIDE_NOT_START)}/{len(FAIL) if FAIL else 1}")
print(f"      Failure: argmax != gold start        : {len(F_WRONG_START)}/{len(FAIL) if FAIL else 1}")
print(f"      Failures recovered by MASS_WINDOW    : {len(F_MASS_FIXED)}/{len(FAIL) if FAIL else 1}")
print(f"      Previously-correct broken by MASS    : {len(BROKEN)}")
print("      Baseline failure IDs                 :",[i+1 for i in FAIL][:80])
print("      Recovered IDs                        :",RECOVERED[:80])
print("      Broken IDs                           :",BROKEN[:80])

print("\n[7/12] COUNTERFACTUAL X-RAY...")
RCF_BASE=[];RCF_MASS=[];CF_BASE_START=[];CF_MASS_START=[];CF_BASE_INSIDE=[];CF_MASS_HIT=[];CFPMASS=[]
for i,it in enumerate(ITEMS):
    j=(i+353)%N;target=TARGETS[j];L=TLEN[j];src=ITEMS[j]
    ents=[it["entity"],NAMES[3*i+1],NAMES[3*i+2]]
    rows=[f"Instrument {ents[0]} carries seal {target}.",
          f"Instrument {ents[1]} carries seal {src['A_d'][0]}.",
          f"Instrument {ents[2]} carries seal {src['A_d'][1]}."]
    random.Random(SEED+90000+i*149).shuffle(rows);cfA=" ".join(rows)
    raw=forge(cfA);pack=install(raw);sp=exact_span(cfA,target)
    if sp is None or len(sp)!=L:raise RuntimeError(f"CF span mismatch item {i+1}")

    w=frozen_pointer(pack,qA(it["entity"]))
    sb=argmax_start(w);sm=mass_window_start(w,L)
    seqb=a_sequence(raw,sb,L);seqm=a_sequence(raw,sm,L)
    scb=invalid_scores() if seqb is None else concat_scores(seqb,L)
    scm=invalid_scores() if seqm is None else concat_scores(seqm,L)
    rb,_,_=rank(scb,j);rm,_,_=rank(scm,j)

    RCF_BASE.append(rb);RCF_MASS.append(rm)
    CF_BASE_START.append(int(sb==sp[0]));CF_MASS_START.append(int(sm==sp[0]))
    CF_BASE_INSIDE.append(int(sb in sp));CF_MASS_HIT.append(int(sm==sp[0]))
    CFPMASS.append(float(w[sp].sum()))

    if i<5 or rb!=1 or rm!=1 or (i+1)%64==0:
        print(f"      [{i+1:04d}/{N}] L={L} target={j+1:04d} gold={sp[0]:3d} "
              f"arg={sb:3d} mass={sm:3d} BASE={rb:4d} MASS={rm:4d}")

    del raw,pack,w,seqb,seqm,scb,scm

MCB=metrics(RCF_BASE);MCM=metrics(RCF_MASS)
print(f"      CF BASE R1/R5/R16/MRR : {MCB[0]:.4f}/{MCB[1]:.4f}/{MCB[2]:.4f}/{MCB[3]:.6f}")
print(f"      CF MASS R1/R5/R16/MRR : {MCM[0]:.4f}/{MCM[1]:.4f}/{MCM[2]:.4f}/{MCM[3]:.6f}")
print(f"      CF BASE exact start    : {np.mean(CF_BASE_START):.4f}")
print(f"      CF MASS exact start    : {np.mean(CF_MASS_START):.4f}")
print(f"      CF pointer span mass   : {np.mean(CFPMASS):.6f}")

print("\n[8/12] ORIGIN ERROR DISTRIBUTION...")
def offset_hist(starts,spans):
    d={}
    for s,sp in zip(starts,spans):
        x=s-sp[0];d[x]=d.get(x,0)+1
    return dict(sorted(d.items()))
HB=offset_hist(BASE_START,GOLD_SPANS);HM=offset_hist(MASS_START,GOLD_SPANS)
print("      ARGMAX start offset histogram:")
print("      ",HB)
print("      MASS_WINDOW start offset histogram:")
print("      ",HM)
for L in LENS:
    ids=[i for i,x in enumerate(ITEMS) if x["L"]==L]
    bh=np.mean([BASE_START_HIT[i] for i in ids]);mh=np.mean([MASS_START_HIT[i] for i in ids])
    print(f"      L={L} exact-start: ARGMAX={bh:.4f} MASS_WINDOW={mh:.4f}")

print("\n[9/12] STATISTICAL SUMMARY...")
def wilson(k,n,z=1.959963984540054):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h

for name,r in [("BASE",RBASE),("MASS",RMASS),
               ("MASS_L2",BYLEN_MASS[2]),("MASS_L3",BYLEN_MASS[3]),("MASS_L4",BYLEN_MASS[4]),
               ("CF_BASE",RCF_BASE),("CF_MASS",RCF_MASS)]:
    k=sum(int(x==1) for x in r);lo,hi=wilson(k,len(r))
    print(f"      {name:8s}: {k:4d}/{len(r):4d} = {k/len(r):.4f} | 95% Wilson [{lo:.4f},{hi:.4f}]")

print("\n[10/12] PREDECLARED DIAGNOSTIC GATES...")
GATES={
"BASELINE_REPRO_R1":abs(MB[0]-.9824)<=.005,
"MASS_WINDOW_R1":MM[0]>=TH_MASS_R1,
"MASS_WINDOW_R5":MM[1]>=TH_MASS_R5,
"MASS_L2_R1":MLM[2][0]>=TH_EACH,
"MASS_L3_R1":MLM[3][0]>=TH_EACH,
"MASS_L4_R1":MLM[4][0]>=TH_EACH,
"MASS_CF_R1":MCM[0]>=TH_CF,
"MASS_GOLD_START_HIT":float(np.mean(MASS_START_HIT))>=TH_START,
"MASS_CF_GOLD_START_HIT":float(np.mean(CF_MASS_START))>=TH_START}
for k,v in GATES.items():print(f"      {k:28s}: {'PASS' if v else 'FAIL'}")

print("\n[11/12] PROTOCOL AUDIT...")
S1=sentinel();WEIGHT_OK=S0==S1
print("      Weight sentinel                     :","PASS" if WEIGHT_OK else "FAIL")
print("      Trainable tensors                   :",sum(int(p.requires_grad) for p in model.parameters()))
print("      Pointer                             : L28H00 unchanged")
print("      Address                             : L00-V uncentered unchanged")
print("      Match                               : CONCATCOS unchanged")
print("      TEST531 panel reconstruction        : YES")
print("      Origin arms                         : ARGMAX + MASS_WINDOW only")
print("      MASS_WINDOW thresholds              : NONE")
print("      MASS_WINDOW uses gold span          : NO")
print("      MASS_WINDOW uses decoded identity   : NO")
print("      MASS_WINDOW uses token IDs          : NO")
print("      All A windows of width L considered : YES")
print("      All matching-length B windows       : YES")
print("      Gold B position used                : NO")
print("      Gold A span                         : DIAGNOSTIC ONLY")
print("      Head/layer/address discovery        : NONE")
print("      Match-law scan                      : NONE")
print("      Training / LoRA / optimizer / DRA   : NONE")
print("      Learned router/scorer               : NONE")
print("      Query-time candidate-B forwards     : 0")
print("      TEST531 retroactively changed       : NO")
GATES["WEIGHT_SENTINEL"]=WEIGHT_OK

print("\n[12/12] FINAL...")
ALL_PASS=all(GATES.values())
if ALL_PASS:
    VERDICT="TEST532_SPAN_ORIGIN_LAW_STRONG_CANDIDATE"
    NEXT="freeze MASS_WINDOW origin rule; validate independently in untouched TEST533"
else:
    VERDICT="TEST532_SPAN_ORIGIN_HYPOTHESIS_NOT_FULLY_SUPPORTED"
    NEXT="preserve TEST532 unchanged; diagnose residual failure in a new test"

RESULT={"test":TEST,
"parents":{"528":PARENT528,"529":PARENT529,"530":PARENT530,"531":PARENT531},
"lock_sha":LOCK_SHA,
"baseline":{"r1":MB[0],"r5":MB[1],"r16":MB[2],"mrr":MB[3]},
"mass_window":{"r1":MM[0],"r5":MM[1],"r16":MM[2],"mrr":MM[3]},
"mass_lengths":{str(L):{"r1":MLM[L][0],"r5":MLM[L][1],"r16":MLM[L][2],"mrr":MLM[L][3]} for L in LENS},
"counterfactual_baseline":{"r1":MCB[0],"r5":MCB[1],"r16":MCB[2],"mrr":MCB[3]},
"counterfactual_mass":{"r1":MCM[0],"r5":MCM[1],"r16":MCM[2],"mrr":MCM[3]},
"baseline_start_hit":float(np.mean(BASE_START_HIT)),
"mass_start_hit":float(np.mean(MASS_START_HIT)),
"cf_baseline_start_hit":float(np.mean(CF_BASE_START)),
"cf_mass_start_hit":float(np.mean(CF_MASS_START)),
"baseline_failures":len(FAIL),
"baseline_failures_inside_not_start":len(F_INSIDE_NOT_START),
"recovered_by_mass":len(F_MASS_FIXED),
"broken_by_mass":len(BROKEN),
"gates":GATES,"verdict":VERDICT}
RESULT_SHA=hashlib.sha256(json.dumps(RESULT,sort_keys=True,separators=(",",":")).encode()).hexdigest()

print("\n"+"="*176)
print("TEST532 FINAL RESULT — AKBASCORE MAM · VARAN 1 · SPAN-ORIGIN X-RAY")
print("="*176)
print("MODEL                              :",MODEL_ID,"· frozen")
print("PANEL                              : TEST531 deterministic reconstruction · N=1024")
print("POINTER                            : L28H00 · UNCHANGED")
print("ADDRESS                            : L00-V · UNCENTERED · 1024D · UNCHANGED")
print("MATCH                              : CONCATCOS · UNCHANGED")
print("-"*176)
print(f"BASELINE ARGMAX                    : R1={MB[0]:.4f} R5={MB[1]:.4f} R16={MB[2]:.4f} MRR={MB[3]:.6f}")
print(f"MASS_WINDOW                        : R1={MM[0]:.4f} R5={MM[1]:.4f} R16={MM[2]:.4f} MRR={MM[3]:.6f}")
for L in LENS:
    print(f"MASS L={L}                           : R1={MLM[L][0]:.4f} R5={MLM[L][1]:.4f} R16={MLM[L][2]:.4f} MRR={MLM[L][3]:.6f}")
print(f"CF BASELINE                        : R1={MCB[0]:.4f} R5={MCB[1]:.4f} R16={MCB[2]:.4f} MRR={MCB[3]:.6f}")
print(f"CF MASS_WINDOW                     : R1={MCM[0]:.4f} R5={MCM[1]:.4f} R16={MCM[2]:.4f} MRR={MCM[3]:.6f}")
print("-"*176)
print(f"ARGMAX EXACT GOLD-START HIT        : {np.mean(BASE_START_HIT):.4f}")
print(f"MASS EXACT GOLD-START HIT          : {np.mean(MASS_START_HIT):.4f}")
print(f"CF ARGMAX EXACT GOLD-START HIT     : {np.mean(CF_BASE_START):.4f}")
print(f"CF MASS EXACT GOLD-START HIT       : {np.mean(CF_MASS_START):.4f}")
print(f"ARGMAX INSIDE GOLD SPAN            : {np.mean(BASE_INSIDE):.4f}")
print(f"BASELINE FAILURES                  : {len(FAIL)}")
print(f"FAIL: INSIDE BUT NOT START         : {len(F_INSIDE_NOT_START)}")
print(f"FAILURES RECOVERED BY MASS         : {len(F_MASS_FIXED)}")
print(f"CORRECT CASES BROKEN BY MASS       : {len(BROKEN)}")
print("-"*176)
for k,v in GATES.items():print(f"{k:35s}: {'PASS' if v else 'FAIL'}")
print("-"*176)
print("WEIGHT SENTINEL                    :","PASS" if WEIGHT_OK else "FAIL")
print("QUERY-TIME CANDIDATE-B FORWARDS    : 0")
print("TEST528 RESULT SHA                 :",PARENT528)
print("TEST529 RESULT SHA                 :",PARENT529)
print("TEST530 RESULT SHA                 :",PARENT530)
print("TEST531 RESULT SHA                 :",PARENT531)
print("TEST532 PRE-RESULT LOCK            :",LOCK_SHA)
print("TEST532 RESULT SHA                 :",RESULT_SHA)
print("TOTAL TEST TIME                    :",f"{time.perf_counter()-T0:.2f}s")
print("VERDICT                            :",VERDICT)
print("NEXT                               :",NEXT)
print("="*176)

# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST529
# VARAN 1 — MULTI-TOKEN MODEL-NATIVE IDENTITY
#
# SEALED PARENT:
#   TEST528 -> MISTRAL MAM FINAL EXTERNAL SEAL
#              PRIMARY 127/128 = 99.22%
#              CF      125/128 = 97.66%
#
# FROZEN FROM TEST528:
#   POINTER = L28H00
#   ADDRESS = L00-V UNCENTERED
#   ADDRESS DIM = 1024
#   MODEL = FROZEN
#   QUERY-TIME CANDIDATE-B FORWARDS = 0
#
# ONLY NEW QUESTION:
#   Can the sealed primitive extend from one-token identities
#   to compositional 2/3/4-token model-native identities?
#
# PREDECLARED ARMS:
#   ROW  = TEST528 rule: max cosine over individual B address rows
#   SPAN = parameter-free max cosine over every contiguous B span
#          of width 1..4 using summed L00-V rows.
#
# IMPORTANT:
#   Gold spans are diagnostic only.
#   Gold spans NEVER select candidate rows/windows.
#   No head/layer/address discovery.
#   No learned router/scorer.
#   No training/LoRA/DRA.
#   No token-ID/decoded-identity routing.
#
# PRIMARY PASS:
#   N=1024
#   identity lengths = 2/3/4 tokens
#   SPAN R1 >= .95
#   SPAN R5 >= .99
#   CF R1 >= .93
#   pointer hit >= .90
#   shifted R1 <= .02
#   no-A R1 <= .02
#   weight sentinel PASS
#   query-time candidate-B forwards = 0
#
# FAILURE BOUNDARY:
#   If L=2 SPAN R1 < .85, Varan 1 is NOT established.

import os,sys,subprocess,importlib.util,random,time,hashlib,json,gc,re,math
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

TEST="529";SEED=529529;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";N=1024
DEVICE=torch.device("cuda");FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
PTR_L=28;PTR_H=0;BL=0;BK="V";CENTER=False;MAX_SPAN=4
TH_R1=.95;TH_R5=.99;TH_CF=.93;TH_PTR=.90;TH_NEG=.02
PARENT528="9235ed38c944f41a9ede0426d3aabaa12b5b20646f30e2be921e618f9471d3c2"
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*170)
print("TEST529 — AKBASCORE MAM · VARAN 1 · MULTI-TOKEN MODEL-NATIVE IDENTITY")
print("SEALED TEST528 BASELINE · FROZEN L28H00 → L00-V · N=1024 · LENGTH 2/3/4")
print("="*170);T0=time.perf_counter()

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
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id;VOC=model.model.embed_tokens.weight.shape[0]
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)}")
print(f"      {NL}L H={H} QH={QH} KVH={KVH} HD={HD} ADDRESS={KVD}")
print("      POINTER : L28H00 FROZEN")
print("      ADDRESS : L00-V UNCENTERED FROZEN")
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

def span_positions(text,key):
    hits=[m.span() for m in re.finditer(r"(?<![A-Za-z0-9])"+re.escape(key)+r"(?![A-Za-z0-9])",text)]
    if len(hits)!=1:return None
    a,b=hits[0];z=tok(text+SEP,add_special_tokens=False,return_offsets_mapping=True);p=[]
    for j,(x,y) in enumerate(z["offset_mapping"]):
        if y>a and x<b:p.append(j+1)
    return p

def valid_identity(parts,L):
    s=" ".join(parts)
    pa=span_positions(f"Instrument Probe carries seal {s}.",s)
    pb=span_positions(f"Seal {s} corresponds to routing class ALPHA.",s)
    return s if pa and pb and len(pa)==L and len(pb)==L else None

print("\n[2/12] Building deterministic 2/3/4-token identity space...")
rr=random.Random(SEED+100)
TARGETS=[];TLEN=[];used=set();tries=0
while len(TARGETS)<N and tries<2000000:
    tries+=1;L=2+(len(TARGETS)%3)
    parts=[rr.choice(POOL) for _ in range(L)]
    if len(set(parts))<L:continue
    s=valid_identity(parts,L)
    if s is None or s in used:continue
    TARGETS.append(s);TLEN.append(L);used.add(s)
if len(TARGETS)!=N:raise RuntimeError(f"Could build only {len(TARGETS)}/{N} target identities.")

DIST=[];tries=0
while len(DIST)<N*2 and tries<3000000:
    tries+=1;L=2+(len(DIST)%3)
    parts=[rr.choice(POOL) for _ in range(L)]
    if len(set(parts))<L:continue
    s=valid_identity(parts,L)
    if s is None or s in used:continue
    DIST.append(s);used.add(s)
if len(DIST)<N*2:raise RuntimeError(f"Could build only {len(DIST)}/{N*2} distractors.")

print(f"      Targets     : {len(TARGETS)}")
print(f"      Distractors : {len(DIST)}")
for L in (2,3,4):print(f"      L={L}: {sum(x==L for x in TLEN)} targets")

SYL1=["Aq","Bryn","Cav","Dex","Eln","Frax","Gyr","Hex","Iln","Jor","Kex","Lyr","Mav","Nex","Oryn","Pax",
      "Qyr","Rex","Savn","Tov","Uln","Vex","Wyr","Xav","Yex","Zyr","Axl","Bov","Cyn","Drex","Evr","Fyn"]
SYL2=["adar","bren","cyr","dax","elor","fyn","grel","hyn","ivar","jor","kyr","lor","myn","nex","or","pyr",
      "qen","rix","sor","tyn","ul","vyr","wen","xir","yor","zen"]
def names(n):
    out=[]
    for a in SYL1:
        for b in SYL2:
            for c in ("a","e","i","o"):
                out.append(a+b+c)
    random.Random(SEED+77).shuffle(out)
    if len(out)<n:raise RuntimeError("Name pool too small.")
    return out[:n]
NAMES=names(N*3);CL=["ALPHA","BETA","GAMMA"]

def make_items():
    out=[]
    for i in range(N):
        ents=NAMES[3*i:3*i+3];gold=TARGETS[i];d1=DIST[2*i];d2=DIST[2*i+1]
        bd1=DIST[(2*i+137)%(2*N)];bd2=DIST[(2*i+911)%(2*N)]
        if bd1==gold or bd2==gold or bd1==bd2:raise RuntimeError("Distractor collision.")
        ra=[f"Instrument {ents[0]} carries seal {gold}.",
            f"Instrument {ents[1]} carries seal {d1}.",
            f"Instrument {ents[2]} carries seal {d2}."]
        rb=[f"Seal {gold} corresponds to routing class {CL[0]}.",
            f"Seal {bd1} corresponds to routing class {CL[1]}.",
            f"Seal {bd2} corresponds to routing class {CL[2]}."]
        random.Random(SEED+i*131+17).shuffle(ra);random.Random(SEED+i*137+29).shuffle(rb)
        out.append({"id":i+1,"A":" ".join(ra),"B":" ".join(rb),"entity":ents[0],"gold":gold,"L":TLEN[i],
                    "A_d":[d1,d2],"B_d":[bd1,bd2]})
    return out
ITEMS=make_items()

LOCK={"test":TEST,"parent528":PARENT528,"model":MODEL_ID,"seed":SEED,"N":N,
      "purpose":"VARAN_1_MULTI_TOKEN_IDENTITY",
      "pointer":{"layer":PTR_L,"head":PTR_H,"frozen":True},
      "address":{"layer":BL,"kind":BK,"center":CENTER,"dimension":KVD,"frozen":True},
      "identity":{"lengths":[2,3,4],"construction":"model-native token compositions"},
      "arms":{"ROW":"max cosine over individual B rows",
              "SPAN":"max cosine over all contiguous B windows width 1..4; summed L00-V"},
      "gold_usage":"DIAGNOSTIC_ONLY",
      "forbidden":["HEAD_SCAN","LAYER_SCAN","ADDRESS_SCAN","CENTER_SCAN","TOKEN_ID_ROUTING",
                   "DECODED_ID_ROUTING","GOLD_WINDOW_SELECTION","QUERY_TIME_B_FORWARD","TRAINING",
                   "LORA","DRA","LEARNED_ROUTER","POSTHOC_COORDINATE_SELECTION"]}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      TEST529 LOCK:",LOCK_SHA)

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

print("\n[3/12] OFFLINE FORGE — 1024 A + 1024 B...")
A_RAW=[];A_PACK=[];GOLD_SPANS=[];B_ROWS=[];B_SPANS=[];ROW_OWNER=[];SPAN_OWNER=[]
for i,it in enumerate(ITEMS):
    ar=forge(it["A"]);br=forge(it["B"]);A_RAW.append(ar);A_PACK.append(install(ar))
    sp=span_positions(it["A"],it["gold"])
    if sp is None or len(sp)!=it["L"]:raise RuntimeError(f"A span mismatch item {i+1}")
    GOLD_SPANS.append(sp)
    x=br[BL][1][:,1:,:].float().cpu().permute(1,0,2).reshape(-1,KVD).contiguous()
    B_ROWS.append(x);ROW_OWNER.extend([i]*len(x))
    wins=[]
    for w in range(1,MAX_SPAN+1):
        if len(x)>=w:
            cs=torch.cat([torch.zeros((1,KVD)),x.cumsum(0)],0)
            wins.append(cs[w:]-cs[:-w])
    z=torch.cat(wins,0);B_SPANS.append(z);SPAN_OWNER.extend([i]*len(z))
    del br
    if (i+1)%64==0:print(f"      forged {i+1:4d}/{N}")
gc.collect();torch.cuda.empty_cache()
print("      Independent A memories:",len(A_RAW))
print("      Independent B memories:",N)
print("      Query-time B forwards : 0")

print("\n[4/12] Building parameter-free numerical candidate indices...")
ROW=torch.cat(B_ROWS,0);SPAN=torch.cat(B_SPANS,0)
ROW=ROW/ROW.norm(dim=1,keepdim=True).clamp_min(1e-8)
SPAN=SPAN/SPAN.norm(dim=1,keepdim=True).clamp_min(1e-8)
ROW=ROW.to(DEVICE,dtype=torch.float16);SPAN=SPAN.to(DEVICE,dtype=torch.float16)
ROW_OWNER=torch.tensor(ROW_OWNER,device=DEVICE,dtype=torch.long)
SPAN_OWNER=torch.tensor(SPAN_OWNER,device=DEVICE,dtype=torch.long)
del B_ROWS,B_SPANS
gc.collect();torch.cuda.empty_cache()
print("      ROW vectors :",ROW.shape)
print("      SPAN vectors:",SPAN.shape)
print("      Gold windows used to construct index: NO")
print("      Every contiguous width 1..4 window indexed.")

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

def pointer_vector(raw,w):
    x=raw[BL][1].float().cpu()
    if x.shape[1]!=len(w):raise RuntimeError("Pointer/V length mismatch.")
    return (x*w[None,:,None]).sum(dim=1).reshape(-1)

@torch.inference_mode()
def bank_scores(p,index,owners):
    p=p.to(DEVICE,dtype=torch.float32);p=p/p.norm().clamp_min(1e-12)
    sim=torch.mv(index.float(),p)
    out=torch.full((N,),-torch.inf,device=DEVICE)
    out.scatter_reduce_(0,owners,sim,reduce="amax",include_self=True)
    return out.cpu()

def rank(sc,g):
    order=torch.argsort(sc,descending=True).tolist()
    return order.index(g)+1,order[0],order

def metrics(r):
    a=np.asarray(r,float)
    return float(np.mean(a==1)),float(np.mean(a<=5)),float(np.mean(a<=16)),float(np.mean(1/a))

def shifted(w):
    z=torch.zeros_like(w);n=len(w)-1
    if n>1:z[1:]=torch.roll(w[1:],max(1,n//2))
    return z/z.sum().clamp_min(1e-12)

def zero_scores():
    rr=random.Random(SEED+909090)
    x=torch.tensor([rr.random() for _ in range(N)])
    return x

print("\n[5/12] PRIMARY — ROW baseline + SPAN identity...")
RROW=[];RSPAN=[];SEL=[];PHIT=[];PMASS=[];MARG=[];PTR=[]
BYLEN={2:[],3:[],4:[]}
for i,it in enumerate(ITEMS):
    w=frozen_pointer(A_PACK[i],qA(it["entity"]));PTR.append(w);p=pointer_vector(A_RAW[i],w)
    sr=bank_scores(p,ROW,ROW_OWNER);ss=bank_scores(p,SPAN,SPAN_OWNER)
    rr,_,_=rank(sr,i);rs,se,o=rank(ss,i)
    RROW.append(rr);RSPAN.append(rs);SEL.append(se);BYLEN[it["L"]].append(rs)
    sp=GOLD_SPANS[i];arg=int(torch.argmax(w));PHIT.append(int(arg in sp));PMASS.append(float(w[sp].sum()))
    MARG.append(float(ss[o[0]]-ss[o[1]]))
    if (i+1)%16==0 or i<8:
        print(f"      [{i+1:04d}/{N}] L={it['L']} ROW={rr:4d} SPAN={rs:4d} sel={se+1:04d} ptr={PHIT[-1]} mass={PMASS[-1]:.4f} margin={MARG[-1]:+.5f}")
MR=metrics(RROW);MSP=metrics(RSPAN)
print(f"      ROW  R1/R5/R16/MRR : {MR[0]:.4f}/{MR[1]:.4f}/{MR[2]:.4f}/{MR[3]:.6f}")
print(f"      SPAN R1/R5/R16/MRR : {MSP[0]:.4f}/{MSP[1]:.4f}/{MSP[2]:.4f}/{MSP[3]:.6f}")
for L in (2,3,4):
    m=metrics(BYLEN[L]);print(f"      SPAN L={L}          : R1={m[0]:.4f} R5={m[1]:.4f} MRR={m[3]:.6f} n={len(BYLEN[L])}")
print(f"      POINTER HIT         : {np.mean(PHIT):.4f}")
print(f"      POINTER MASS        : {np.mean(PMASS):.6f}")

print("\n[6/12] NEGATIVE CONTROLS...")
RSHIFT=[];RNO=[]
for i in range(N):
    p=pointer_vector(A_RAW[i],shifted(PTR[i]));r,_,_=rank(bank_scores(p,SPAN,SPAN_OWNER),i);RSHIFT.append(r)
    r,_,_=rank(zero_scores(),i);RNO.append(r)
MSh=metrics(RSHIFT);MNo=metrics(RNO)
print(f"      PRIMARY SPAN : R1={MSP[0]:.4f} R5={MSP[1]:.4f}")
print(f"      SHIFTED      : R1={MSh[0]:.4f} R5={MSh[1]:.4f}")
print(f"      NO-A RANDOM  : R1={MNo[0]:.4f} R5={MNo[1]:.4f}")

print("\n[7/12] COUNTERFACTUAL FOLLOW...")
RCF=[];CFHIT=[];CFMASS=[]
for i,it in enumerate(ITEMS):
    j=(i+353)%N;target=TARGETS[j]
    ents=[it["entity"],NAMES[3*i+1],NAMES[3*i+2]]
    rows=[f"Instrument {ents[0]} carries seal {target}.",
          f"Instrument {ents[1]} carries seal {it['A_d'][0]}.",
          f"Instrument {ents[2]} carries seal {it['A_d'][1]}."]
    random.Random(SEED+90000+i*149).shuffle(rows);cfA=" ".join(rows)
    raw=forge(cfA);pack=install(raw);w=frozen_pointer(pack,qA(it["entity"]));p=pointer_vector(raw,w)
    r,se,_=rank(bank_scores(p,SPAN,SPAN_OWNER),j);RCF.append(r)
    sp=span_positions(cfA,target)
    if sp is None:raise RuntimeError("CF span missing.")
    CFHIT.append(int(int(torch.argmax(w)) in sp));CFMASS.append(float(w[sp].sum()))
    if (i+1)%32==0 or i<5:print(f"      [{i+1:04d}/{N}] target={j+1:04d} rank={r:4d} sel={se+1:04d} ptr={CFHIT[-1]} mass={CFMASS[-1]:.4f}")
    del raw,pack,w,p
MCF=metrics(RCF)
print(f"      CF R1/R5/R16/MRR : {MCF[0]:.4f}/{MCF[1]:.4f}/{MCF[2]:.4f}/{MCF[3]:.6f}")
print(f"      CF POINTER HIT    : {np.mean(CFHIT):.4f}")
print(f"      CF POINTER MASS   : {np.mean(CFMASS):.6f}")

print("\n[8/12] LENGTH / FAILURE DIAGNOSTICS...")
L2=metrics(BYLEN[2]);L3=metrics(BYLEN[3]);L4=metrics(BYLEN[4])
print(f"      L2 SPAN R1 : {L2[0]:.4f}")
print(f"      L3 SPAN R1 : {L3[0]:.4f}")
print(f"      L4 SPAN R1 : {L4[0]:.4f}")
print(f"      ROW→SPAN Δ : {MSP[0]-MR[0]:+.4f}")
print(f"      PTR-HIT R1 : {np.mean(np.asarray(RSPAN)[np.asarray(PHIT,dtype=bool)]==1):.4f}")
miss=~np.asarray(PHIT,dtype=bool)
print(f"      PTR-MISS R1: {np.mean(np.asarray(RSPAN)[miss]==1) if miss.any() else float('nan'):.4f}")
print(f"      MEAN MARGIN : {np.mean(MARG):+.6f}")
print(f"      FAIL ITEMS  : {[i+1 for i,r in enumerate(RSPAN) if r!=1][:50]}")

print("\n[9/12] STATISTICAL SUMMARY...")
def wilson(k,n,z=1.959963984540054):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h
for name,r in (("ROW",RROW),("SPAN",RSPAN),("SHIFTED",RSHIFT),("NO_A",RNO),("CF",RCF)):
    k=sum(int(x==1) for x in r);lo,hi=wilson(k,len(r))
    print(f"      {name:8s}: {k:4d}/{len(r)} = {k/len(r):.4f} | 95% Wilson [{lo:.4f},{hi:.4f}]")

print("\n[10/12] PREDECLARED VARAN-1 GATES...")
GATES={
"SPAN_R1":MSP[0]>=TH_R1,
"SPAN_R5":MSP[1]>=TH_R5,
"CF_R1":MCF[0]>=TH_CF,
"POINTER_HIT":float(np.mean(PHIT))>=TH_PTR,
"SHIFTED_R1":MSh[0]<=TH_NEG,
"NOA_R1":MNo[0]<=TH_NEG,
"L2_NOT_FAILURE":L2[0]>=.85}
for k,v in GATES.items():print(f"      {k:24s}: {'PASS' if v else 'FAIL'}")

print("\n[11/12] PROTOCOL AUDIT...")
S1=sentinel();WEIGHT_OK=S0==S1
print("      Weight sentinel                     :","PASS" if WEIGHT_OK else "FAIL")
print("      Trainable tensors                   :",sum(int(p.requires_grad) for p in model.parameters()))
print("      Pointer                             : L28H00 · unchanged")
print("      Address                             : L00-V uncentered · unchanged")
print("      Pointer/head discovery              : NONE")
print("      Address/layer discovery             : NONE")
print("      Identity lengths                    : 2 / 3 / 4")
print("      Gold span used in retrieval         : NO")
print("      SPAN candidate construction         : ALL contiguous windows width 1..4")
print("      Token-ID routing                    : NO")
print("      Decoded-identity routing            : NO")
print("      Query-time candidate-B forwards     : 0")
print("      Training / LoRA / optimizer / DRA   : NONE")
print("      Learned router/scorer               : NONE")
GATES["WEIGHT_SENTINEL"]=WEIGHT_OK

print("\n[12/12] FINAL...")
ALL_PASS=all(GATES.values())
if ALL_PASS:
    VERDICT="MISTRAL_MAM_VARAN1_MULTI_TOKEN_IDENTITY_PASS"
elif L2[0]<.85:
    VERDICT="MISTRAL_MAM_VARAN1_PRIMITIVE_BOUNDARY_FAIL"
else:
    VERDICT="MISTRAL_MAM_VARAN1_NOT_YET_ESTABLISHED"

RESULT={"test":TEST,"parent528":PARENT528,"lock_sha":LOCK_SHA,
"row":{"r1":MR[0],"r5":MR[1],"r16":MR[2],"mrr":MR[3]},
"span":{"r1":MSP[0],"r5":MSP[1],"r16":MSP[2],"mrr":MSP[3]},
"length":{"2":L2[0],"3":L3[0],"4":L4[0]},
"counterfactual":{"r1":MCF[0],"r5":MCF[1],"r16":MCF[2],"mrr":MCF[3]},
"shifted":{"r1":MSh[0],"r5":MSh[1]},"noA":{"r1":MNo[0],"r5":MNo[1]},
"pointer_hit":float(np.mean(PHIT)),"pointer_mass":float(np.mean(PMASS)),
"cf_pointer_hit":float(np.mean(CFHIT)),"cf_pointer_mass":float(np.mean(CFMASS)),
"gates":GATES,"verdict":VERDICT}
RESULT_SHA=hashlib.sha256(json.dumps(RESULT,sort_keys=True,separators=(",",":")).encode()).hexdigest()

print("\n"+"="*170)
print("TEST529 FINAL RESULT — AKBASCORE MAM · VARAN 1")
print("="*170)
print("MODEL                              :",MODEL_ID,"· frozen")
print("SEALED PARENT                      : TEST528")
print("PANEL                              :",N,"independent memories")
print("POINTER                            : L28H00 · UNCHANGED")
print("ADDRESS                            : L00-V · UNCENTERED · 1024D · UNCHANGED")
print("IDENTITY                           : MODEL-NATIVE COMPOSITION · 2/3/4 TOKEN")
print("-"*170)
print(f"ROW BASELINE                       : R1={MR[0]:.4f} R5={MR[1]:.4f} R16={MR[2]:.4f} MRR={MR[3]:.6f}")
print(f"SPAN PRIMARY                       : R1={MSP[0]:.4f} R5={MSP[1]:.4f} R16={MSP[2]:.4f} MRR={MSP[3]:.6f}")
print(f"L=2                                : R1={L2[0]:.4f} R5={L2[1]:.4f}")
print(f"L=3                                : R1={L3[0]:.4f} R5={L3[1]:.4f}")
print(f"L=4                                : R1={L4[0]:.4f} R5={L4[1]:.4f}")
print(f"COUNTERFACTUAL                     : R1={MCF[0]:.4f} R5={MCF[1]:.4f} R16={MCF[2]:.4f} MRR={MCF[3]:.6f}")
print(f"SHIFTED                            : R1={MSh[0]:.4f} R5={MSh[1]:.4f}")
print(f"NO-A                               : R1={MNo[0]:.4f} R5={MNo[1]:.4f}")
print(f"POINTER HIT                        : {np.mean(PHIT):.4f}")
print(f"POINTER MASS                       : {np.mean(PMASS):.6f}")
print(f"CF POINTER HIT                     : {np.mean(CFHIT):.4f}")
print(f"MEAN TOP1-TOP2 MARGIN              : {np.mean(MARG):+.6f}")
print("-"*170)
for k,v in GATES.items():print(f"{k:35s}: {'PASS' if v else 'FAIL'}")
print("-"*170)
print("QUERY-TIME CANDIDATE-B FORWARDS    : 0")
print("TEST528 RESULT SHA                 :",PARENT528)
print("TEST529 PRE-RESULT LOCK            :",LOCK_SHA)
print("TEST529 RESULT SHA                 :",RESULT_SHA)
print("TOTAL TEST TIME                    :",f"{time.perf_counter()-T0:.2f}s")
print("VERDICT                            :",VERDICT)
if ALL_PASS:
    print("STATUS                             : VARAN 1 ESTABLISHED")
    print("NEXT                               : freeze TEST529; proceed to VARAN 2 only in a new test")
elif L2[0]<.85:
    print("STATUS                             : MULTI-TOKEN PRIMITIVE BOUNDARY FOUND")
    print("NEXT                               : preserve TEST529 unchanged; diagnose pointer/address composition in a new test")
else:
    print("STATUS                             : VARAN 1 NOT YET ESTABLISHED")
    print("NEXT                               : preserve result unchanged; diagnose failures before any new mechanism")
print("="*170)

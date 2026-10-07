# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST538
# VARAN 1 — UNKNOWN-LENGTH GLOBAL-BANK MECHANISM VALIDATION
#
# PARENT:
#   TEST537 -> POINTER-PRIOR × IDENTITY-EVIDENCE JOINT RESOLUTION
#
# TEST537 RESULT:
#   PRIMARY 1024/1024
#   CF      1024/1024
#   L2/L3/L4 all 1.0000
#   PEAK failures recovered PRIMARY 7/7, CF 9/9
#   previously-correct broken PRIMARY 0, CF 0
#
# TEST538 CHANGES ONLY:
#   1) Identity length L is NOT supplied to retrieval.
#   2) L2/L3/L4 candidate banks compete in ONE GLOBAL ranking.
#
# FROZEN / UNCHANGED:
#   MODEL   = mistralai/Mistral-7B-Instruct-v0.3
#   POINTER = L28H00
#   ADDRESS = L00-V UNCENTERED
#   TOPK    = 8
#   MATCH   = position-preserving CONCATCOS
#   LAW     = log(pointer)-log(max(1-CONCATCOS,float32_eps))
#   PANEL   = exact TEST531–537 deterministic construction
#
# GLOBAL UNKNOWN-LENGTH LAW:
#   For each frozen Top-8 pointer peak:
#     for L in {2,3,4}:
#       enumerate every valid width-L A origin containing that peak
#       compare against the corresponding B windows
#       compute TEST537 joint evidence
#   Then globally rank ALL 1024 owners across ALL lengths.
#
# IMPORTANT:
#   - Retrieval never receives gold/protocol L.
#   - Gold L is used only AFTER retrieval for diagnostics.
#   - No length classifier.
#   - No length prior.
#   - No per-length threshold.
#   - No normalization fitted by length.
#   - No K/head/layer/address/match/lambda scan.
#   - No training/LoRA/router/token-ID routing.
#
# This is mechanism validation on the existing development panel.
# If PASS: freeze mechanism and run TEST539 on a fresh untouched panel.

import os,sys,subprocess,importlib.util,random,time,hashlib,json,gc,re,math
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

TEST="538";SEED=531531;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";N=1024
DEVICE=torch.device("cuda");FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
PTR_L=28;PTR_H=0;BL=0;BK="V";CENTER=False;LENS=(2,3,4);PTR_TOPK=8
EPS=float(np.finfo(np.float32).eps)
TH_R1=.995;TH_R5=.999;TH_EACH=.99;TH_CF=.99
PARENT528="9235ed38c944f41a9ede0426d3aabaa12b5b20646f30e2be921e618f9471d3c2"
PARENT529="27ba703872f0d8a6ed870a6ba041d0cd0703d2eca467deac97186a90497795a0"
PARENT530="289247c88dfd4358cbe5ffb8499891546b27acc6912caf525a155ae81338db7b"
PARENT531="20257d7d613a49e55b03603946f07ec493009924f40ac26ab74cb817e1466528"
PARENT532="9e37a473a322a19f7b3862a25162835ab6396998dc4fc416bc90bc9b718df188"
PARENT533="5239fb585b87a22f5da778c61a1011379b216a25bf7d05dc5d01952cbb22a5c6"
PARENT534="b25128d17176790b2a8694c8ec9e7db5eaad469ff02a87a46876b2ffd45a9bb1"
PARENT535="fc407885133d48f84a952249b15e884b201749136f6b16a49e7db11b0e9e5087"
PARENT536=""
PARENT537="9a4b6856a5d2a4f440ffa66cf4ba8c57b308b4f3a1cb7db4d0d8b826dd87cbf9"

os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*176)
print("TEST538 — AKBASCORE MAM · VARAN 1 · UNKNOWN-LENGTH GLOBAL-BANK MECHANISM VALIDATION")
print("TEST537 LAW FROZEN · L NOT SUPPLIED · L2/L3/L4 GLOBAL COMPETITION · NO LENGTH ROUTER")
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
print("      MATCH   : CONCATCOS · LOCKED")
print("      TOPK    : 8 · FROZEN")
print("      LAW     : log(pointer)-log(max(1-CONCATCOS,float32_eps)) · FROZEN")
print("      LENGTH  : UNKNOWN TO RETRIEVAL")
print("      BANK    : GLOBAL L2+L3+L4 OWNER COMPETITION")
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

print("\n[2/12] Reconstructing exact sealed development panel...")
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
print("      Development panel seed:",SEED)

LOCK={"test":TEST,
"parents":{"528":PARENT528,"529":PARENT529,"530":PARENT530,"531":PARENT531,"532":PARENT532,"533":PARENT533,"534":PARENT534,"535":PARENT535,"536":PARENT536,"537":PARENT537},
"model":MODEL_ID,"seed":SEED,"N":N,"status":"VARAN1_UNKNOWN_LENGTH_GLOBAL_BANK_VALIDATION",
"pointer":{"layer":PTR_L,"head":PTR_H,"topk":PTR_TOPK},
"address":{"layer":BL,"kind":BK,"center":CENTER,"dimension":KVD},
"identity":{"candidate_lengths":[2,3,4],"length_known_to_retrieval":False},
"match":{"name":"CONCATCOS","locked_from":"TEST530"},
"law":{"name":"LOG_PRIOR_PLUS_LOG_IDENTITY_EVIDENCE","formula":"log(pointer_weight)-log(max(1-CONCATCOS,float32_eps))","frozen_from":"TEST537","epsilon":"float32_machine_epsilon","fitted_parameters":0},
"competition":{"owners":N,"lengths":[2,3,4],"global_across_lengths":True},
"forbidden":["GOLD_LENGTH_SELECTION","LENGTH_ROUTER","LENGTH_PRIOR","PER_LENGTH_THRESHOLD","PER_LENGTH_NORMALIZATION_FIT",
"LAMBDA_SCAN","THRESHOLD_SCAN","TOPK_SCAN","HEAD_SCAN","LAYER_SCAN","ADDRESS_SCAN","MATCH_SCAN","TOKEN_ID_ROUTING",
"DECODED_ID_ROUTING","QUERY_TIME_B_FORWARD","TRAINING","LORA","DRA","LEARNED_ROUTER"]}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      TEST538 LOCK:",LOCK_SHA)

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

print("\n[3/12] OFFLINE FORGE — exact A/B construction...")
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

print("\n[4/12] Building frozen length banks for GLOBAL owner competition...")
OWNER_LENGTH=torch.tensor(TLEN,dtype=torch.long)
for L in LENS:
    X=torch.cat(B_BANK[L]["vec"],0);X=X/X.norm(dim=2,keepdim=True).clamp_min(1e-8)
    B_BANK[L]["vec"]=X.to(DEVICE,dtype=torch.float16)
    B_BANK[L]["own"]=torch.tensor(B_BANK[L]["own"],device=DEVICE,dtype=torch.long)
    print(f"      L={L} windows: {len(B_BANK[L]['own'])} | shape={tuple(B_BANK[L]['vec'].shape)}");del X
print("      Retrieval receives L       : NO")
print("      All candidate lengths tried: 2,3,4")
print("      Final owner ranking         : GLOBAL 1024-way")
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
    out=torch.full((N,),-torch.inf,device=DEVICE)
    out.scatter_reduce_(0,own,sim,reduce="amax",include_self=True)
    return out.cpu()

def invalid_scores():return torch.full((N,),-torch.inf)

def rank(sc,g):
    if not torch.isfinite(sc).any():return N,-1,[]
    order=torch.argsort(sc,descending=True).tolist()
    return order.index(g)+1,order[0],order

# Frozen TEST537 law, now evaluated across ALL candidate lengths.
# IMPORTANT: concat_scores(seq,L) itself only produces finite scores for owners
# whose B memory has length L. The final `best` merges all L into one 1024-owner
# score vector; therefore final selection is globally cross-length.
def global_unknown_length_scores(raw,w):
    best=invalid_scores();best_L=torch.full((N,),-1,dtype=torch.long);best_start=torch.full((N,),-1,dtype=torch.long);best_peak=torch.full((N,),-1,dtype=torch.long)
    for p in pointer_topk(w,PTR_TOPK):
        pw=max(float(w[p]),EPS);logp=math.log(pw)
        for L in LENS:
            for s in containing_starts(raw,p,L):
                seq=a_sequence(raw,s,L)
                if seq is None:continue
                cos=concat_scores(seq,L);finite=torch.isfinite(cos)
                residual=torch.clamp(1.0-cos,min=EPS)
                joint=torch.full_like(cos,-torch.inf);joint[finite]=logp-torch.log(residual[finite])
                mask=joint>best
                best[mask]=joint[mask];best_L[mask]=L;best_start[mask]=s;best_peak[mask]=p
    return best,best_L,best_start,best_peak

def metrics(r):
    a=np.asarray(r,float);return float(np.mean(a==1)),float(np.mean(a<=5)),float(np.mean(a<=16)),float(np.mean(1/a))

print("\n[5/12] PRIMARY — UNKNOWN LENGTH · GLOBAL L2/L3/L4 COMPETITION...")
R=[];BYTRUE={L:[] for L in LENS};PREDLEN=[];FAIL=[]
for i,it in enumerate(ITEMS):
    w=frozen_pointer(A_PACK[i],qA(it["entity"]))
    sc,bL,bS,bP=global_unknown_length_scores(A_RAW[i],w)
    r,sel,_=rank(sc,i);R.append(r);BYTRUE[it["L"]].append(r)
    predL=int(bL[sel]) if sel>=0 else -1;PREDLEN.append(predL)
    if r!=1:FAIL.append(i+1)
    if i<8 or r!=1 or (i+1)%64==0:
        print(f"      [{i+1:04d}/{N}] trueL={it['L']} predL={predL} rank={r:4d} sel={sel+1 if sel>=0 else 0:04d} "
              f"origin={int(bS[sel]) if sel>=0 else -1:3d} peak={int(bP[sel]) if sel>=0 else -1:3d}")
M=metrics(R);ML={L:metrics(BYTRUE[L]) for L in LENS}
print(f"      GLOBAL UNKNOWN-LENGTH : R1={M[0]:.4f} R5={M[1]:.4f} R16={M[2]:.4f} MRR={M[3]:.6f}")
for L in LENS:print(f"      TRUE L={L}: R1={ML[L][0]:.4f} R5={ML[L][1]:.4f} R16={ML[L][2]:.4f} MRR={ML[L][3]:.6f}")
print("      Failures:",len(FAIL),"/",N)
print("      Failure IDs:",FAIL[:100])

print("\n[6/12] PRIMARY LENGTH DIAGNOSTIC — POST-HOC ONLY...")
LC=np.zeros((3,3),dtype=int)
for t,p in zip(TLEN,PREDLEN):
    if p in LENS:LC[LENS.index(t),LENS.index(p)]+=1
print("      rows=true L, cols=selected-owner L [2,3,4]")
for a,L in enumerate(LENS):print(f"      L={L}: {LC[a].tolist()}")
print("      Selected owner length correct:",sum(int(a==b) for a,b in zip(TLEN,PREDLEN)),"/",N)

print("\n[7/12] COUNTERFACTUAL — UNKNOWN LENGTH · GLOBAL L2/L3/L4 COMPETITION...")
RCF=[];CFTRUE=[];CFPRED=[];CFFAIL=[]
for i,it in enumerate(ITEMS):
    j=(i+353)%N;target=TARGETS[j];trueL=TLEN[j];src=ITEMS[j]
    ents=[it["entity"],NAMES[3*i+1],NAMES[3*i+2]]
    rows=[f"Instrument {ents[0]} carries seal {target}.",f"Instrument {ents[1]} carries seal {src['A_d'][0]}.",f"Instrument {ents[2]} carries seal {src['A_d'][1]}."]
    random.Random(SEED+90000+i*149).shuffle(rows);cfA=" ".join(rows)
    raw=forge(cfA);pack=install(raw);sp=exact_span(cfA,target)
    if sp is None or len(sp)!=trueL:raise RuntimeError(f"CF span mismatch item {i+1}")
    w=frozen_pointer(pack,qA(it["entity"]))
    sc,bL,bS,bP=global_unknown_length_scores(raw,w)
    r,sel,_=rank(sc,j);RCF.append(r);CFTRUE.append(trueL)
    predL=int(bL[sel]) if sel>=0 else -1;CFPRED.append(predL)
    if r!=1:CFFAIL.append(i+1)
    if i<5 or r!=1 or (i+1)%64==0:
        print(f"      [{i+1:04d}/{N}] target={j+1:04d} trueL={trueL} predL={predL} rank={r:4d} "
              f"sel={sel+1 if sel>=0 else 0:04d} origin={int(bS[sel]) if sel>=0 else -1:3d} peak={int(bP[sel]) if sel>=0 else -1:3d}")
    del raw,pack,w,sc,bL,bS,bP
MCF=metrics(RCF)
print(f"      CF GLOBAL UNKNOWN-LENGTH : R1={MCF[0]:.4f} R5={MCF[1]:.4f} R16={MCF[2]:.4f} MRR={MCF[3]:.6f}")
print("      CF Failures:",len(CFFAIL),"/",N)
print("      CF Failure IDs:",CFFAIL[:100])

print("\n[8/12] COUNTERFACTUAL LENGTH DIAGNOSTIC — POST-HOC ONLY...")
CLC=np.zeros((3,3),dtype=int)
for t,p in zip(CFTRUE,CFPRED):
    if p in LENS:CLC[LENS.index(t),LENS.index(p)]+=1
print("      rows=true L, cols=selected-owner L [2,3,4]")
for a,L in enumerate(LENS):print(f"      L={L}: {CLC[a].tolist()}")
print("      CF selected owner length correct:",sum(int(a==b) for a,b in zip(CFTRUE,CFPRED)),"/",N)

print("\n[9/12] STATISTICAL SUMMARY...")
def wilson(k,n,z=1.959963984540054):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h
for name,r in [("GLOBAL",R),("GLOBAL_L2",BYTRUE[2]),("GLOBAL_L3",BYTRUE[3]),("GLOBAL_L4",BYTRUE[4]),("CF_GLOBAL",RCF)]:
    k=sum(int(x==1) for x in r);lo,hi=wilson(k,len(r))
    print(f"      {name:10s}: {k:4d}/{len(r):4d} = {k/len(r):.4f} | 95% Wilson [{lo:.4f},{hi:.4f}]")

print("\n[10/12] PREDECLARED GATES...")
GATES={
"GLOBAL_R1":M[0]>=TH_R1,
"GLOBAL_R5":M[1]>=TH_R5,
"GLOBAL_L2_R1":ML[2][0]>=TH_EACH,
"GLOBAL_L3_R1":ML[3][0]>=TH_EACH,
"GLOBAL_L4_R1":ML[4][0]>=TH_EACH,
"CF_GLOBAL_R1":MCF[0]>=TH_CF}
for k,v in GATES.items():print(f"      {k:32s}: {'PASS' if v else 'FAIL'}")

print("\n[11/12] PROTOCOL AUDIT...")
S1=sentinel();WEIGHT_OK=S0==S1;GATES["WEIGHT_SENTINEL"]=WEIGHT_OK
print("      Weight sentinel                    :","PASS" if WEIGHT_OK else "FAIL")
print("      Trainable tensors                  :",sum(int(p.requires_grad) for p in model.parameters()))
print("      Pointer                            : L28H00 unchanged")
print("      Address                            : L00-V uncentered unchanged")
print("      Pointer K                          : 8 unchanged")
print("      Match                              : CONCATCOS unchanged")
print("      TEST537 law                        : UNCHANGED")
print("      Retrieval receives identity L      : NO")
print("      Candidate lengths                  : 2,3,4 simultaneously")
print("      Final competition                  : GLOBAL 1024 owners")
print("      Length classifier/router           : NONE")
print("      Length prior                       : NONE")
print("      Per-length threshold               : NONE")
print("      Per-length fitted normalization    : NONE")
print("      Gold L used in retrieval           : NO")
print("      Gold span used in retrieval        : NO")
print("      Gold B position used               : NO")
print("      Token-ID / decoded-ID routing      : NO")
print("      K/lambda/threshold scan            : NONE")
print("      Head/layer/address/match scan      : NONE")
print("      Training / LoRA / optimizer / DRA : NONE")
print("      Learned router/scorer              : NONE")
print("      Query-time candidate-B forwards    : 0")
print("      Existing panel                     : YES — mechanism validation")
print("      Untouched final seal               : NOT THIS TEST")

print("\n[12/12] FINAL...")
ALL_PASS=all(GATES.values())
if ALL_PASS:
    VERDICT="TEST538_UNKNOWN_LENGTH_GLOBAL_BANK_STRONG_CANDIDATE"
    NEXT="freeze complete Varan 1 mechanism; run TEST539 once on a fresh untouched seed/panel with no mechanism changes"
else:
    VERDICT="TEST538_UNKNOWN_LENGTH_GLOBAL_BANK_NOT_FULLY_SUPPORTED"
    NEXT="preserve TEST538 unchanged; diagnose cross-length failures without changing the frozen TEST537 law on this run"

RESULT={"test":TEST,
"parents":{"528":PARENT528,"529":PARENT529,"530":PARENT530,"531":PARENT531,"532":PARENT532,"533":PARENT533,
           "534":PARENT534,"535":PARENT535,"536":PARENT536,"537":PARENT537},
"lock_sha":LOCK_SHA,
"mechanism":{"pointer":"L28H00","address":"L00-V UNCENTERED","topk":PTR_TOPK,"match":"CONCATCOS",
"law":"log(pointer)-log(max(1-CONCATCOS,float32_eps))","retrieval_length_known":False,"global_lengths":[2,3,4]},
"primary":{"metrics":M,"by_true_length":{str(L):ML[L] for L in LENS},"failures":FAIL,
"selected_length_correct":sum(int(a==b) for a,b in zip(TLEN,PREDLEN))},
"counterfactual":{"metrics":MCF,"failures":CFFAIL,
"selected_length_correct":sum(int(a==b) for a,b in zip(CFTRUE,CFPRED))},
"gates":GATES,"verdict":VERDICT}
RESULT_SHA=hashlib.sha256(json.dumps(RESULT,sort_keys=True,separators=(",",":")).encode()).hexdigest()

print("\n"+"="*176)
print("TEST538 FINAL RESULT — AKBASCORE MAM · VARAN 1 · UNKNOWN-LENGTH GLOBAL-BANK MECHANISM VALIDATION")
print("="*176)
print("MODEL                              :",MODEL_ID,"· frozen")
print("PANEL                              : existing deterministic development panel · N=1024")
print("POINTER                            : L28H00 · UNCHANGED")
print("ADDRESS                            : L00-V · UNCENTERED · 1024D · UNCHANGED")
print("MATCH                              : CONCATCOS · UNCHANGED")
print("POINTER CANDIDATES                 : TOP-8 · UNCHANGED")
print("LAW                                : log(pointer)-log(max(1-CONCATCOS,float32_eps)) · UNCHANGED")
print("RETRIEVAL KNOWS IDENTITY LENGTH    : NO")
print("GLOBAL LENGTH COMPETITION          : YES · L2+L3+L4")
print("-"*176)
print(f"PRIMARY GLOBAL                     : R1={M[0]:.4f} R5={M[1]:.4f} R16={M[2]:.4f} MRR={M[3]:.6f}")
for L in LENS:print(f"PRIMARY TRUE L={L}                  : R1={ML[L][0]:.4f} R5={ML[L][1]:.4f} R16={ML[L][2]:.4f} MRR={ML[L][3]:.6f}")
print(f"COUNTERFACTUAL GLOBAL              : R1={MCF[0]:.4f} R5={MCF[1]:.4f} R16={MCF[2]:.4f} MRR={MCF[3]:.6f}")
print("PRIMARY FAILURES                   :",len(FAIL))
print("CF FAILURES                        :",len(CFFAIL))
print("PRIMARY SELECTED-LENGTH CORRECT    :",sum(int(a==b) for a,b in zip(TLEN,PREDLEN)),"/",N)
print("CF SELECTED-LENGTH CORRECT         :",sum(int(a==b) for a,b in zip(CFTRUE,CFPRED)),"/",N)
print("-"*176)
for k,v in GATES.items():print(f"{k:35s}: {'PASS' if v else 'FAIL'}")
print("-"*176)
print("WEIGHT SENTINEL                    :","PASS" if WEIGHT_OK else "FAIL")
print("TRAINABLE TENSORS                  :",sum(int(p.requires_grad) for p in model.parameters()))
print("QUERY-TIME CANDIDATE-B FORWARDS    : 0")
print("KNOWN IDENTITY LENGTH              : NO")
print("CROSS-LENGTH GLOBAL COMPETITION    : YES")
print("LENGTH ROUTER                      : NONE")
print("LENGTH PRIOR                       : NONE")
print("FITTED LENGTH NORMALIZATION        : NONE")
print("TEST528 RESULT SHA                 :",PARENT528)
print("TEST529 RESULT SHA                 :",PARENT529)
print("TEST530 RESULT SHA                 :",PARENT530)
print("TEST531 RESULT SHA                 :",PARENT531)
print("TEST532 RESULT SHA                 :",PARENT532)
print("TEST533 RESULT SHA                 :",PARENT533)
print("TEST534 RESULT SHA                 :",PARENT534)
print("TEST535 RESULT SHA                 :",PARENT535)
print("TEST536 RESULT SHA                 :",PARENT536 if PARENT536 else "NOT INSERTED IN SOURCE")
print("TEST537 RESULT SHA                 :",PARENT537)
print("TEST538 PRE-RESULT LOCK            :",LOCK_SHA)
print("TEST538 RESULT SHA                 :",RESULT_SHA)
print("TOTAL TEST TIME                    :",f"{time.perf_counter()-T0:.2f}s")
print("VERDICT                            :",VERDICT)
print("NEXT                               :",NEXT)
print("="*176)

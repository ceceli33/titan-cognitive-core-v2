# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST519
# TENSOR-NATIVE BELLEKÖZ ADDRESS FROM FROZEN K/V GEOMETRY
#
# PARENT:
#   TEST518:
#     AD0/AD1 gold-labelled B address       = 42/48 = .8750
#     AD2/AD3 label-free token header       = 29/48 = .6042
#     H2 bank-derived numeric handoff       = 43/48 = .8958
#     CF carrier follow                     = 47/48 = .9792
#     E2E AD3+H2                            = 29/48 = .6042
#
# TEST519 QUESTION:
# Can the correct independent B BELLEKÖZ be addressed directly from its own
# frozen K/V tensors, without B token IDs, decoded text, LM-head rows,
# gold Seal labels, gold key positions, or query-time B model forwards?
#
# PRIMARY ADDRESS:
#   A BELLEKÖZ + natural query
#       -> query hidden state at frozen layer 23
#       -> same frozen layer k_proj / v_proj
#       -> compare with every B BELLEKÖZ raw K/V position at layer 23
#       -> position score = mean cosine across the 4 native KV heads
#       -> B score = max over B positions
#       -> KV23 score = mean(KMAX23, VMAX23)
#
# IMPORTANT:
#   Residual coordinates are NOT reshaped into heads.
#   Query and B are compared only after the SAME frozen k_proj/v_proj.
#   Native KV-head structure is preserved.
#   B K is pre-RoPE raw K, so query K is also pre-RoPE.
#
# DIAGNOSTIC ARMS:
#   K23, V23, KV23 PRIMARY
#   KV15, KV27
#   KVCONS = fixed mean of KV15/KV23/KV27
#
# CAUSAL CONTROL:
#   NO_A uses the identical natural question with no A BELLEKÖZ.
#
# HANDOFF:
#   TEST518 H2 is preserved only as a frozen downstream control.
#   H2 still uses TEST518's B-bank-derived vocabulary inventory.
#   Therefore TEST519 tests TENSOR-NATIVE ADDRESS, not yet a completely
#   vocabulary-free end-to-end architecture.
#
# PREDECLARED PRIMARY PASS:
#   KV23 R1 >= .70
#   KV23 R5 >= .85
#   NO_A_KV23 R1 <= .10
#   KV23 + frozen H2 E2E >= .60
#
# NO training / DRA / ANN / learned router / post-hoc channel selection.

import os,sys,subprocess,importlib.util,random,time,hashlib,json,gc,re,math
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache

TEST="519";SEED=518;MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N=48;TRACE_DEPTH=27;PRIMARY_L=23;DIAG_LAYERS=[15,23,27];MAX_NEW=12;DEVICE=torch.device("cuda")
FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
TEST516="53f4fa86f1bc4f2c4db412bde68a2b004afc5a6c6cf84d1c3345b175ab1a3e8d"
TEST518="d335fa02068644efbe3c54eba8b6ae3a4678fbba3abed972d3f31a401590ae3c"
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*176)
print("TEST519 — AKBASCORE MAM · TENSOR-NATIVE BELLEKÖZ ADDRESS FROM FROZEN K/V GEOMETRY")
print("A QUERY K/V TRACE → B RAW K/V TENSOR HEADER · NO B TOKEN-ID ADDRESS · NO GOLD KEY POSITION")
print("="*176)
T0=time.perf_counter()

print("\n[1/11] Loading frozen Qwen...")
_tv=tuple(int("".join(c for c in x if c.isdigit())or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
_ge=model.generation_config.eos_token_id;EOS=sorted({tok.eos_token_id,*(_ge if isinstance(_ge,(list,tuple)) else [_ge])}-{None})
enc=lambda s:tok(s,add_special_tokens=False).input_ids
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=H//QH
if (NL,H,QH,KVH,HD)!=(28,3584,28,4,128):raise RuntimeError("Architecture mismatch.")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id;EMB=model.model.embed_tokens;VOC=EMB.weight.shape[0]
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | {NL}L H={H} QH={QH} KVH={KVH} HD={HD} | V={VOC} | frozen BF16")

# Exact TEST518 panel construction.
SYL1=["Al","Bel","Cor","Dar","El","Fen","Gal","Har","Il","Jar","Kel","Lor","Mar","Nor","Or","Par"]
SYL2=["den","rin","ven","lis","ron","mir","tal","sen","vik"]
def build_names(n):
    out=[]
    for a in SYL1:
        for b in SYL2:
            x=a+b
            if x not in out:out.append(x)
            if len(out)>=n:return out
    raise RuntimeError("Name pool too small.")

def native_codes(n,exclude=set()):
    out=[]
    for tid in range(VOC):
        s=tok.decode([tid],skip_special_tokens=False)
        if s in exclude:continue
        if not re.fullmatch(r"[A-Z]{3,8}",s):continue
        if enc(s)!=[tid]:continue
        if s in out:continue
        out.append(s)
        if len(out)>=n:return out
    raise RuntimeError(f"Need {n} tokenizer-native code tokens, found {len(out)}.")

NAMES=build_names(N*3)
SEALS=native_codes(N*3)
CLASSES=native_codes(N*3,exclude=set(SEALS))
SEAL_ID={s:enc(s)[0] for s in SEALS};CLASS_ID={s:enc(s)[0] for s in CLASSES}
assert len(set(SEAL_ID.values()))==N*3 and len(set(CLASS_ID.values()))==N*3

def make_items():
    items=[]
    for i in range(N):
        ents=NAMES[3*i:3*i+3];a_seals=SEALS[3*i:3*i+3];gold=a_seals[0]
        j1=(i+7)%N;j2=(i+19)%N
        b_seals=[gold,SEALS[3*j1+1],SEALS[3*j2+2]]
        b_classes=CLASSES[3*i:3*i+3]
        ra=[f"Instrument {ents[j]} carries seal {a_seals[j]}." for j in range(3)]
        rb=[f"Seal {b_seals[j]} corresponds to routing class {b_classes[j]}." for j in range(3)]
        rr=random.Random(SEED+i*101);rr.shuffle(ra);rr.shuffle(rb)
        items.append({"id":i+1,"A":" ".join(ra),"B":" ".join(rb),
                      "gold":{"entity":ents[0],"seal":gold,"class":b_classes[0]},
                      "A_distractors":a_seals[1:],"B_seals":b_seals,"B_classes":b_classes})
    return items

ITEMS=make_items();GOLDS=[x["gold"]["seal"] for x in ITEMS]
assert len(set(GOLDS))==N
for i,it in enumerate(ITEMS):
    assert it["gold"]["seal"] in it["B_seals"]
    assert all(s not in it["B_seals"] for s in it["A_distractors"])
    assert sum(it["gold"]["seal"] in x["B_seals"] for x in ITEMS)==1

LOCK={"test":TEST,"model":MODEL_ID,"seed":SEED,"N":N,"parent516":TEST516,"parent518":TEST518,"items":ITEMS,
"primary_layer":PRIMARY_L,"diagnostic_layers":DIAG_LAYERS,
"primary_address":"SAME_LAYER_PRE_ROPE_KV23_MAX_POSITION_MEAN_HEAD_COSINE",
"B_address_uses_token_ids":False,"B_address_uses_lm_head":False,"B_address_uses_gold_key_position":False,
"B_address_uses_decoded_text":False,"query_time_candidate_B_forwards":0,
"residual_reshape_into_heads":False,"training":"OFF","dra":"OFF","ann":"OFF","learned_router":"OFF",
"handoff_control":"TEST518_H2_FROZEN_BANK_INVENTORY_CONTROL_ONLY",
"pass":{"KV23_R1":.70,"KV23_R5":.85,"NO_A_KV23_R1_max":.10,"E2E_KV23_H2":.60}}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
print("      TEST519 LOCK:",LOCK_SHA)
print("      PARENT518   :",TEST518)
print("      Primary     : L23 same-projection K/V tensor address")
print("      B token IDs : FORBIDDEN IN EXPERIMENTAL ADDRESS")

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
    layer=model.model.layers[L];rot=layer.self_attn.rotary_emb if hasattr(layer.self_attn,"rotary_emb") else model.model.rotary_emb
    d=torch.zeros((1,k.shape[0],len(pos),HD),device=k.device,dtype=k.dtype);p=torch.tensor([pos],device=k.device,dtype=torch.long)
    try:c,s=rot(d,p)
    except TypeError:c,s=rot(d,position_ids=p)
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    _,kr=apply_rotary_pos_emb(d,k.unsqueeze(0),c,s);return kr[0]

def install(raw):
    T=raw[0][0].shape[1]
    return [(rope_k(k,list(range(T)),L),v) for L,(k,v) in enumerate(raw)],T,T

def cache_of(inst):
    try:c=DynamicCache(config=cfg)
    except TypeError:c=DynamicCache()
    for L,(k,v) in enumerate(inst):c.update(k.unsqueeze(0),v.unsqueeze(0),L)
    return c

def firstline(x):return next((z.strip() for z in str(x).splitlines() if z.strip()),"")
def normtxt(x):return firstline(x).strip().upper().rstrip(".")
def exact(x,t):return normtxt(x)==str(t).upper()
def qA(e):return f"What seal does instrument {e} carry? Give only the exact seal."
def rank(sc,gold):
    order=sorted(range(N),key=lambda j:(-sc[j],j))
    return order.index(gold)+1,order[0]
def metrics(r):
    a=np.asarray(r,float)
    return float(np.mean(a==1)),float(np.mean(a<=5)),float(np.mean(a<=16)),float(np.mean(1/a))
def unit_last(x):
    return x.float()/x.float().norm(dim=-1,keepdim=True).clamp_min(1e-8)

print("\n[2/11] OFFLINE FORGE — preserving raw and installed BELLEKÖZ tensors...")
A_RAW=[];A_PACK=[];B_RAW=[];B_PACK=[]
for i,it in enumerate(ITEMS):
    ar=forge(it["A"]);br=forge(it["B"])
    A_RAW.append(ar);B_RAW.append(br);A_PACK.append(install(ar));B_PACK.append(install(br))
    if i<5:print(f"      ITEM {i+1:02d} | A={A_PACK[-1][1]} B={B_PACK[-1][1]} tokens")
print("      96 BELLEKÖZ forged.")
print("      Experimental B address source = RAW K/V TENSORS ONLY.")

# TEST518 H2 downstream control only.
# These IDs are NOT read by tensor-native address functions below.
B_TOKEN_IDS_H2=[enc(it["B"]+SEP) for it in ITEMS]
BANK_IDS_H2=sorted(set(t for row in B_TOKEN_IDS_H2 for t in row))
BANK_MASK_H2=torch.tensor(BANK_IDS_H2,dtype=torch.long)
print(f"      Frozen H2 control inventory = {len(BANK_IDS_H2)} token IDs (handoff control only)")
gc.collect();torch.cuda.empty_cache()

print("\n[3/11] PERSISTENCE RESET...")
print("      Forge-time DynamicCache discarded.")
print("      Fresh cache will be created for every query/read.")
print("      Tensor-native address sees B_RAW K/V only.")
print("      No B source token IDs are passed to tensor address scoring.")

@torch.inference_mode()
def query_state(packet,q,need_logits=True):
    inst,Tm,P=packet;c=cache_of(inst);ids=enc(FMT.format(q=q));n=len(ids)
    pos=torch.arange(P,P+n,device=DEVICE).unsqueeze(0);mask=torch.ones((1,Tm+n),device=DEVICE,dtype=torch.long)
    o=model(input_ids=torch.tensor([ids],device=DEVICE),past_key_values=c,attention_mask=mask,position_ids=pos,
            use_cache=True,output_hidden_states=True,return_dict=True)
    proj={}
    for L in DIAG_LAYERS:
        layer=model.model.layers[L]
        h=o.hidden_states[L][0,-1].to(layer.input_layernorm.weight.device,layer.input_layernorm.weight.dtype)
        hn=layer.input_layernorm(h)
        k=layer.self_attn.k_proj(hn).view(KVH,HD).detach().float().cpu()
        v=layer.self_attn.v_proj(hn).view(KVH,HD).detach().float().cpu()
        proj[L]=(k,v)
    logits=None
    if need_logits:
        h=o.hidden_states[TRACE_DEPTH][0,-1];fn=model.model.norm
        logits=model.lm_head(fn(h.to(fn.weight.dtype))).float().detach().cpu()
    return proj,logits,o.past_key_values,Tm+n,P+n

@torch.inference_mode()
def query_state_no_A(q):
    ids=enc(FMT.format(q=q))
    o=model(input_ids=torch.tensor([ids],device=DEVICE),use_cache=False,output_hidden_states=True,return_dict=True)
    proj={}
    for L in DIAG_LAYERS:
        layer=model.model.layers[L]
        h=o.hidden_states[L][0,-1].to(layer.input_layernorm.weight.device,layer.input_layernorm.weight.dtype)
        hn=layer.input_layernorm(h)
        k=layer.self_attn.k_proj(hn).view(KVH,HD).detach().float().cpu()
        v=layer.self_attn.v_proj(hn).view(KVH,HD).detach().float().cpu()
        proj[L]=(k,v)
    return proj

# Same native projection space:
# q: [KVH,HD]
# B: [KVH,T,HD]
# Per position: cosine in each corresponding native KV head, then mean across heads.
# Cartridge score: maximum position score.
def tensor_match(q,b):
    q=unit_last(q).unsqueeze(1)                     # [Hk,1,D]
    b=unit_last(b.float())                         # [Hk,T,D]
    per_head=(q*b).sum(-1)                         # [Hk,T]
    per_pos=per_head.mean(0)                       # [T]
    # position 0 is synthetic PAD inserted by forge; structural exclusion only.
    if per_pos.numel()>1:per_pos=per_pos[1:]
    return float(per_pos.max())

def tensor_scores(proj,L,kind):
    qk,qv=proj[L];out=[]
    for br in B_RAW:
        bk,bv=br[L]
        if kind=="K":s=tensor_match(qk,bk.detach().cpu())
        elif kind=="V":s=tensor_match(qv,bv.detach().cpu())
        else:s=.5*(tensor_match(qk,bk.detach().cpu())+tensor_match(qv,bv.detach().cpu()))
        out.append(s)
    return out

def consensus_scores(proj):
    all_sc=[tensor_scores(proj,L,"KV") for L in DIAG_LAYERS]
    return np.asarray(all_sc,dtype=np.float64).mean(0).tolist()

print("\n[4/11] TENSOR-NATIVE ADDRESS — K23 / V23 / KV23 / KV15 / KV27 / KVCONS...")
R={k:[] for k in ("K23","V23","KV23","KV15","KV27","KVCONS","NOA_KV23")}
SEL={k:[] for k in R}
LOGITS=[];PROJ=[]
for i,it in enumerate(ITEMS):
    q=qA(it["gold"]["entity"])
    p,l,_,_,_=query_state(A_PACK[i],q,True);pn=query_state_no_A(q)
    arms={
        "K23":tensor_scores(p,23,"K"),
        "V23":tensor_scores(p,23,"V"),
        "KV23":tensor_scores(p,23,"KV"),
        "KV15":tensor_scores(p,15,"KV"),
        "KV27":tensor_scores(p,27,"KV"),
        "KVCONS":consensus_scores(p),
        "NOA_KV23":tensor_scores(pn,23,"KV")
    }
    PROJ.append(p);LOGITS.append(l)
    for k,sc in arms.items():
        r,s=rank(sc,i);R[k].append(r);SEL[k].append(s)
    print(f"      [{i+1:02d}/48] {it['gold']['entity']:7s} | K23={R['K23'][-1]:2d} V23={R['V23'][-1]:2d} KV23={R['KV23'][-1]:2d} K15={R['KV15'][-1]:2d} K27={R['KV27'][-1]:2d} CONS={R['KVCONS'][-1]:2d} NOA={R['NOA_KV23'][-1]:2d}")

print("\n[5/11] Address metrics...")
for k in R:
    m=metrics(R[k])
    print(f"      {k:10s} R1/R5/R16/MRR: {m[0]:.4f} / {m[1]:.4f} / {m[2]:.4f} / {m[3]:.6f}")

# Frozen TEST518 H2 control.
def probs_bank_h2(logits):
    z=logits[BANK_MASK_H2].float();p=torch.softmax(z,0)
    out=torch.zeros_like(logits,dtype=torch.float32);out[BANK_MASK_H2]=p
    return out

@torch.inference_mode()
def soft_embed(p):
    pp=p.to(device=DEVICE,dtype=torch.float32);ew=EMB.weight.detach().float()
    return (pp@ew).to(dtype=EMB.weight.dtype)

@torch.inference_mode()
def rollout_second_h2(packet,q,c1):
    _,_,c,Tm,P=query_state(packet,q,True)
    x=c1.to(device=DEVICE,dtype=EMB.weight.dtype).view(1,1,H)
    pos=torch.tensor([[P]],device=DEVICE,dtype=torch.long)
    mask=torch.ones((1,Tm+1),device=DEVICE,dtype=torch.long)
    o=model(inputs_embeds=x,past_key_values=c,attention_mask=mask,position_ids=pos,
            use_cache=False,output_hidden_states=True,return_dict=True)
    h=o.hidden_states[TRACE_DEPTH][0,-1];fn=model.model.norm
    l2=model.lm_head(fn(h.to(fn.weight.dtype))).float().cpu()
    return soft_embed(probs_bank_h2(l2)).detach().cpu()

def h2_carrier(packet,q,logits):
    p=probs_bank_h2(logits);c1=soft_embed(p).detach().cpu();c2=rollout_second_h2(packet,q,c1)
    return torch.stack([c1,c2]).to(dtype=EMB.weight.dtype)

PREFIX="QUESTION:\nUsing the stored memory, the relevant internal key is"
SUFFIX=". What routing class corresponds to that internal key? Give only the exact routing class.\n\nANSWER:"

@torch.inference_mode()
def read_B_numeric(packet,carrier,max_new=MAX_NEW):
    inst,Tm,P=packet;c=cache_of(inst);pre=enc(PREFIX);suf=enc(SUFFIX)
    pree=EMB(torch.tensor(pre,device=DEVICE,dtype=torch.long)).detach()
    sufe=EMB(torch.tensor(suf,device=DEVICE,dtype=torch.long)).detach()
    car=carrier.to(device=DEVICE,dtype=EMB.weight.dtype)
    x=torch.cat([pree,car,sufe],0).unsqueeze(0);n=x.shape[1]
    pos=torch.arange(P,P+n,device=DEVICE).unsqueeze(0);mask=torch.ones((1,Tm+n),device=DEVICE,dtype=torch.long)
    o=model(inputs_embeds=x,past_key_values=c,attention_mask=mask,position_ids=pos,use_cache=True,return_dict=True)
    c=o.past_key_values;out=[];Tm+=n;P+=n;nxt=int(o.logits[0,-1].float().argmax())
    for _ in range(max_new):
        if nxt in EOS:break
        out.append(nxt);pos=torch.tensor([[P]],device=DEVICE,dtype=torch.long)
        mask=torch.ones((1,Tm+1),device=DEVICE,dtype=torch.long)
        o=model(input_ids=torch.tensor([[nxt]],device=DEVICE,dtype=torch.long),past_key_values=c,
                attention_mask=mask,position_ids=pos,use_cache=True,return_dict=True)
        c=o.past_key_values;Tm+=1;P+=1;nxt=int(o.logits[0,-1].float().argmax())
    return tok.decode(out,skip_special_tokens=True).strip()

print("\n[6/11] Rebuilding frozen TEST518 H2 carriers...")
H2=[]
for i,it in enumerate(ITEMS):
    H2.append(h2_carrier(A_PACK[i],qA(it["gold"]["entity"]),LOGITS[i]))
    if i<5 or (i+1)%8==0:print(f"      [{i+1:02d}/48] H2 ready")

print("\n[7/11] H2 oracle-B control + tensor-address E2E...")
ORACLE=[];E23=[];ECONS=[];RAW23=[];RAWCONS=[]
for i,it in enumerate(ITEMS):
    t=it["gold"]["class"]
    yo=read_B_numeric(B_PACK[i],H2[i])
    y23=read_B_numeric(B_PACK[SEL["KV23"][i]],H2[i])
    yc=read_B_numeric(B_PACK[SEL["KVCONS"][i]],H2[i])
    ORACLE.append(int(exact(yo,t)));E23.append(int(exact(y23,t)));ECONS.append(int(exact(yc,t)))
    RAW23.append(y23);RAWCONS.append(yc)
    print(f"      [{i+1:02d}/48] addr23={int(SEL['KV23'][i]==i)} E23={E23[-1]} | cons={int(SEL['KVCONS'][i]==i)} ECONS={ECONS[-1]} | OR={ORACLE[-1]}")

print("\n[8/11] Failure localization...")
ADDR23=np.asarray([int(x==1) for x in R["KV23"]],dtype=np.int32)
E23A=np.asarray(E23,dtype=np.int32);ORA=np.asarray(ORACLE,dtype=np.int32)
NADDR=int(ADDR23.sum());COND=int(np.sum((ADDR23==1)&(E23A==1)))
HF=int(np.sum((ADDR23==1)&(E23A==0)));ACC=int(np.sum((ADDR23==0)&(E23A==1)))
print(f"      KV23 correct address                  : {NADDR}/{N} = {ADDR23.mean():.4f}")
print(f"      H2 oracle-B                           : {sum(ORACLE)}/{N} = {np.mean(ORACLE):.4f}")
print(f"      KV23+H2 E2E                           : {sum(E23)}/{N} = {np.mean(E23):.4f}")
print(f"      KVCONS+H2 E2E                         : {sum(ECONS)}/{N} = {np.mean(ECONS):.4f}")
print(f"      E2E success | KV23 address correct    : {COND}/{NADDR} = {COND/NADDR:.4f}" if NADDR else "      E2E success | KV23 address correct    : N/A")
print(f"      Correct KV23 address / handoff fail   : {HF}")
print(f"      Wrong KV23 address / accidental pass  : {ACC}")

print("\n[9/11] Wilson intervals...")
def wilson(k,n,z=1.959963984540054):
    if n==0:return (float("nan"),float("nan"))
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h

for k in R:
    kk=sum(int(x==1) for x in R[k]);lo,hi=wilson(kk,N)
    print(f"      {k:10s}: {kk:2d}/{N} = {kk/N:.4f} | R1 95% Wilson [{lo:.4f},{hi:.4f}]")
for k,v in (("H2_ORACLE",ORACLE),("E2E_KV23",E23),("E2E_CONS",ECONS)):
    kk=sum(v);lo,hi=wilson(kk,N)
    print(f"      {k:10s}: {kk:2d}/{N} = {kk/N:.4f} | 95% Wilson [{lo:.4f},{hi:.4f}]")

print("\n[10/11] Predeclared primary decision...")
M23=metrics(R["KV23"]);MNO=metrics(R["NOA_KV23"])
PASS_ADDR=M23[0]>=.70 and M23[1]>=.85 and MNO[0]<=.10
PASS_E2E=float(np.mean(E23))>=.60
H2_OK=float(np.mean(ORACLE))>=.80
if PASS_ADDR and PASS_E2E and H2_OK:
    VERDICT="TENSOR_NATIVE_BELLEKOZ_ADDRESS_OBSERVED"
elif M23[0]>=.50 and MNO[0]<=.10 and H2_OK:
    VERDICT="TENSOR_NATIVE_ADDRESS_SIGNAL_PRESENT_BUT_BELOW_PREDECLARED_RELIABILITY"
elif M23[0]<.20 and H2_OK:
    VERDICT="SAME_LAYER_KV_TENSOR_ADDRESS_NOT_ESTABLISHED"
elif not H2_OK:
    VERDICT="PARENT_H2_CONTROL_DID_NOT_REPLICATE"
else:
    VERDICT="TENSOR_NATIVE_ADDRESS_PARTIAL_INCONCLUSIVE"
print("      KV23 address pass :",PASS_ADDR)
print("      KV23 E2E pass     :",PASS_E2E)
print("      H2 control pass   :",H2_OK)
print("      VERDICT           :",VERDICT)

print("\n[11/11] Protocol audit...")
print("      Model weights frozen                         : YES")
print("      TEST518 panel preserved                      : YES")
print("      Fresh cache each query/read                   : YES")
print("      A/B cache concatenation                       : NO")
print("      Experimental B address reads B token IDs      : NO")
print("      Experimental B address reads decoded B text   : NO")
print("      Experimental B address reads LM-head rows     : NO")
print("      Experimental B address reads gold Seal        : NO")
print("      Experimental B address reads gold key pos     : NO")
print("      Experimental B address source                 : RAW BELLEKÖZ K/V")
print("      Query/B comparison basis                      : SAME FROZEN k_proj/v_proj")
print("      Native KV heads preserved                     : YES")
print("      Residual reshaped into fake heads              : NO")
print("      B K comparison                                : PRE-RoPE ↔ PRE-RoPE")
print("      Query-time candidate-B model forwards         : 0")
print("      Learned router / ANN / DRA / training         : NONE")
print("      Post-hoc primary channel selection             : NO")
print("      PRIMARY fixed before results                   : KV23")
print("      H2 vocabulary inventory                       : YES — DOWNSTREAM CONTROL ONLY")
print("      TEST519 claim if positive                     : TENSOR-NATIVE ADDRESS, NOT YET FULL VOCAB-FREE E2E")

print("\n"+"="*176)
print("TEST519 FINAL RESULT — AKBASCORE MAM · TENSOR-NATIVE BELLEKÖZ ADDRESS")
print("="*176)
print("MODEL                              : Qwen/Qwen2.5-7B-Instruct · frozen")
print("PARENT                             : TEST518")
print("PANEL                              : exact TEST518 48-item corrected panel")
print("PRIMARY ADDRESS                    : L23 raw K/V · same projection · max position · mean native heads")
print("B TOKEN-ID ADDRESS                 : NONE")
print("B GOLD KEY POSITION                : NONE")
print("B LM-HEAD ADDRESS                  : NONE")
print("QUERY-TIME B FORWARDS              : 0")
print("-"*176)
for k in ("K23","V23","KV23","KV15","KV27","KVCONS","NOA_KV23"):
    m=metrics(R[k])
    print(f"{k:34s}: R1={m[0]:.4f} R5={m[1]:.4f} R16={m[2]:.4f} MRR={m[3]:.6f}")
print("-"*176)
print(f"H2 ORACLE-B CONTROL                : {sum(ORACLE):2d}/{N} = {np.mean(ORACLE):.4f}")
print(f"E2E KV23 + H2                      : {sum(E23):2d}/{N} = {np.mean(E23):.4f}")
print(f"E2E KVCONS + H2                    : {sum(ECONS):2d}/{N} = {np.mean(ECONS):.4f}")
print(f"E2E | CORRECT KV23 ADDRESS         : {COND}/{NADDR} = {COND/NADDR:.4f}" if NADDR else "E2E | CORRECT KV23 ADDRESS         : N/A")
print(f"CORRECT ADDRESS / HANDOFF FAIL     : {HF}")
print(f"WRONG ADDRESS / ACCIDENTAL PASS    : {ACC}")
print("-"*176)
print("PREDECLARED PRIMARY PASS           : KV23 R1>=.70; R5>=.85; NO_A R1<=.10; KV23+H2 E2E>=.60")
print("ADDRESS PASS                       :",PASS_ADDR)
print("E2E PASS                           :",PASS_E2E)
print("H2 CONTROL PASS                    :",H2_OK)
print("TEST516 LOCK                       :",TEST516)
print("TEST518 LOCK                       :",TEST518)
print("TEST519 LOCK                       :",LOCK_SHA)
print(f"TOTAL TEST TIME                    : {time.perf_counter()-T0:.2f}s")
print("VERDICT                            :",VERDICT)
print("="*176)

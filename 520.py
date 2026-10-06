# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST520
# NATIVE ATTENTION-POINTER → SAME-ROLE TENSOR ADDRESS
#
# PARENTS:
# TEST518: token-derived label-free address R1=.6042; H2 oracle-B=.8958
# TEST519: direct query K/V -> raw B K/V collapsed to chance (KV23 R1=.0208)
#
# PRIMARY HYPOTHESIS:
# TEST519 compared role-mismatched states.
# Instead:
#   natural query
#       -> native Q·K attention over A BELLEKÖZ
#       -> tensor-only pointer over A positions
#       -> pointed A K/V state
#       -> centered same-role comparison against B K/V positions
#       -> B address
#
# PRIMARY ADDRESS CONTAINS:
#   NO token IDs
#   NO decoded Seal
#   NO LM-head rows
#   NO gold key position
#   NO B text
#   NO human labels
#   NO query-time B forward
#
# DISC: items 0..23
# EVAL: items 24..47
#
# DISC selects exactly one of:
#   comparison layer ∈ {1,2,4,8,12,16}
#   representation ∈ {K,V}
# by MRR, then R1, then lower layer, then K before V.
# EVAL is untouched until selection is frozen.
#
# POINTER:
#   layers 14..27 fixed
#   actual frozen q_proj
#   actual RoPE query
#   installed RoPE A keys
#   native Q·K / sqrt(head_dim)
#   softmax over A cartridge positions only
#   PAD excluded
#   average 28 Q heads and 14 fixed layers
#
# SAME-ROLE ADDRESS:
#   p = Σ_t attention(t) * (X_A[t] - bank_mean)
#   score(B) = max_s cosine(flatten(p), flatten(X_B[s]-bank_mean))
#
# CONTROLS:
#   UNIFORM pointer
#   NO_A tensor baseline
#   COUNTERFACTUAL A pointer-follow
#   UNCENTERED diagnostic
#   pointer localization diagnostic (gold used ONLY for scoring)
#
# ACETATE DIAGNOSTIC:
#   fixed L24 / rank4
#   B pre-RoPE K principal subspace
#   natural query's real pre-RoPE Q projection energy
#   diagnostic only; cannot replace PRIMARY.
#
# No training / DRA / learned router / ANN / post-hoc EVAL selection.

import os,sys,subprocess,importlib.util,random,time,hashlib,json,gc,re,math
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

TEST="520";SEED=518;MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N=48;NDISC=24;DEVICE=torch.device("cuda")
FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
PTR_LAYERS=list(range(14,28));CMP_LAYERS=[1,2,4,8,12,16];ACETATE_L=24;ACETATE_R=4
TEST518="d335fa02068644efbe3c54eba8b6ae3a4678fbba3abed972d3f31a401590ae3c"
TEST519="2bcf12382a2341ca84fcb1c21a9b52d4c4de79af50ba21d8d3b853f34c9f0f1a"
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*176)
print("TEST520 — AKBASCORE MAM · NATIVE ATTENTION-POINTER → SAME-ROLE TENSOR ADDRESS")
print("QUERY Q·K → A POSITION POINTER → CENTERED A K/V ↔ B K/V · TOKEN-ID-FREE PRIMARY ADDRESS")
print("="*176)
T0=time.perf_counter()

print("\n[1/12] Loading frozen Qwen...")
_tv=tuple(int("".join(c for c in x if c.isdigit())or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
enc=lambda s:tok(s,add_special_tokens=False).input_ids
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=H//QH;REP=QH//KVH
if (NL,H,QH,KVH,HD)!=(28,3584,28,4,128):raise RuntimeError("Architecture mismatch.")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id;VOC=model.model.embed_tokens.weight.shape[0]
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | {NL}L H={H} QH={QH} KVH={KVH} HD={HD} | frozen BF16")

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
    raise RuntimeError(f"Need {n} native codes, found {len(out)}.")

NAMES=build_names(N*3);SEALS=native_codes(N*3);CLASSES=native_codes(N*3,exclude=set(SEALS))
def make_items():
    items=[]
    for i in range(N):
        ents=NAMES[3*i:3*i+3];aseals=SEALS[3*i:3*i+3];gold=aseals[0];j1=(i+7)%N;j2=(i+19)%N
        bseals=[gold,SEALS[3*j1+1],SEALS[3*j2+2]];bclasses=CLASSES[3*i:3*i+3]
        ra=[f"Instrument {ents[j]} carries seal {aseals[j]}." for j in range(3)]
        rb=[f"Seal {bseals[j]} corresponds to routing class {bclasses[j]}." for j in range(3)]
        rr=random.Random(SEED+i*101);rr.shuffle(ra);rr.shuffle(rb)
        items.append({"id":i+1,"A":" ".join(ra),"B":" ".join(rb),
                      "gold":{"entity":ents[0],"seal":gold,"class":bclasses[0]},
                      "A_seals":aseals,"A_distractors":aseals[1:],"B_seals":bseals,"B_classes":bclasses})
    return items
ITEMS=make_items();GOLDS=[x["gold"]["seal"] for x in ITEMS]
assert len(set(GOLDS))==N
for i,it in enumerate(ITEMS):
    assert it["gold"]["seal"] in it["B_seals"]
    assert all(x not in it["B_seals"] for x in it["A_distractors"])
    assert sum(it["gold"]["seal"] in x["B_seals"] for x in ITEMS)==1

LOCK={"test":TEST,"model":MODEL_ID,"seed":SEED,"N":N,"disc":[0,23],"eval":[24,47],
"parents":{"518":TEST518,"519":TEST519},"pointer_layers":PTR_LAYERS,"cmp_layers":CMP_LAYERS,
"primary":"NATIVE_QK_A_POINTER_THEN_CENTERED_SAME_ROLE_A_TO_B",
"selection":"DISC_MRR_THEN_R1_THEN_LOWER_LAYER_THEN_K",
"acetate_diagnostic":{"layer":ACETATE_L,"rank":ACETATE_R},
"forbidden":["TOKEN_ID_ADDRESS","DECODED_KEY","LM_HEAD_ADDRESS","GOLD_POSITION_ADDRESS","B_TEXT_ADDRESS","QUERY_TIME_B_FORWARD","TRAINING","DRA","LEARNED_ROUTER"],
"pass":{"EVAL_R1":.60,"EVAL_R5":.85,"UNIFORM_R1_MAX":.15,"NOA_R1_MAX":.10,"CF_FOLLOW":.70}}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
print("      TEST520 LOCK:",LOCK_SHA)
print("      PARENT518   :",TEST518)
print("      PARENT519   :",TEST519)
print("      DISC/EVAL   : 24 / 24")
print("      Primary     : native attention pointer → centered same-role tensor address")

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

def rope_pair(q,k,pos,L):
    layer=model.model.layers[L];rot=layer.self_attn.rotary_emb if hasattr(layer.self_attn,"rotary_emb") else model.model.rotary_emb
    p=torch.tensor([pos],device=DEVICE,dtype=torch.long)
    dummy=torch.zeros_like(q)
    try:c,s=rot(dummy,p)
    except TypeError:c,s=rot(dummy,position_ids=p)
    return apply_rotary_pos_emb(q,k,c,s)

def rope_k(k,pos,L):
    q=torch.zeros((1,QH,len(pos),HD),device=DEVICE,dtype=k.dtype)
    kk=k.unsqueeze(0)
    layer=model.model.layers[L];rot=layer.self_attn.rotary_emb if hasattr(layer.self_attn,"rotary_emb") else model.model.rotary_emb
    p=torch.tensor([pos],device=DEVICE,dtype=torch.long)
    try:c,s=rot(q,p)
    except TypeError:c,s=rot(q,position_ids=p)
    _,kr=apply_rotary_pos_emb(q,kk,c,s);return kr[0]

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
    order=sorted(range(N),key=lambda j:(-float(sc[j]),j));return order.index(gold)+1,order[0]
def metrics(r):
    a=np.asarray(r,float);return float(np.mean(a==1)),float(np.mean(a<=5)),float(np.mean(a<=16)),float(np.mean(1/a))
def cos_rows(v,M):
    v=v.float();M=M.float()
    v=v/v.norm().clamp_min(1e-8);M=M/M.norm(dim=1,keepdim=True).clamp_min(1e-8)
    return M@v

print("\n[2/12] OFFLINE FORGE — exact corrected TEST518 panel...")
A_RAW=[];B_RAW=[];A_PACK=[];B_PACK=[]
for i,it in enumerate(ITEMS):
    ar=forge(it["A"]);br=forge(it["B"]);A_RAW.append(ar);B_RAW.append(br);A_PACK.append(install(ar));B_PACK.append(install(br))
    if i<5:print(f"      ITEM {i+1:02d} | A={A_PACK[-1][1]} B={B_PACK[-1][1]} tokens")
print("      96 BELLEKÖZ forged.")
print("      Raw K/V retained only as persistent numeric cartridge state.")

print("\n[3/12] Computing tensor-only bank means...")
MU={}
for L in CMP_LAYERS:
    for kind,ix in (("K",0),("V",1)):
        xs=[]
        for bank in (A_RAW,B_RAW):
            for raw in bank:xs.append(raw[L][ix][:,1:,:].float().cpu())
        cat=torch.cat(xs,dim=1);MU[(L,kind)]=cat.mean(dim=1)
        del cat,xs
print("      Means computed from cartridge tensors only; PAD excluded.")

def locate_gold_pos(it):
    # DIAGNOSTIC ONLY. Never enters address computation.
    ids=[PAD]+enc(it["A"]+SEP);gid=enc(it["gold"]["seal"])[0]
    hit=[j for j,t in enumerate(ids) if t==gid]
    return hit[0] if len(hit)==1 else -1

GOLD_POS=[locate_gold_pos(it) for it in ITEMS]

print("\n[4/12] PERSISTENCE RESET...")
print("      Source text/token IDs are NOT used by PRIMARY address after forge.")
print("      Gold positions exist only for pointer-localization scoring.")
gc.collect();torch.cuda.empty_cache()

@torch.inference_mode()
def query_forward(packet,q):
    inst,Tm,P=packet;c=cache_of(inst);ids=enc(FMT.format(q=q));n=len(ids)
    pos=torch.arange(P,P+n,device=DEVICE).unsqueeze(0);mask=torch.ones((1,Tm+n),device=DEVICE,dtype=torch.long)
    o=model(input_ids=torch.tensor([ids],device=DEVICE),past_key_values=c,attention_mask=mask,position_ids=pos,
            use_cache=False,output_hidden_states=True,return_dict=True)
    return o,Tm,P,n

@torch.inference_mode()
def native_pointer(packet,q):
    inst,Tm,P=packet;o,_,_,n=query_forward(packet,q);ws=[]
    qpos=P+n-1
    for L in PTR_LAYERS:
        layer=model.model.layers[L]
        h=o.hidden_states[L][0,-1].to(layer.input_layernorm.weight.dtype)
        hn=layer.input_layernorm(h)
        qv=layer.self_attn.q_proj(hn).view(1,QH,1,HD)
        # Installed A K already carries RoPE at cartridge positions 0..Tm-1.
        ak=inst[L][0].repeat_interleave(REP,dim=0).unsqueeze(0)
        dummy=torch.zeros_like(ak[:,:,:1,:])
        rot=layer.self_attn.rotary_emb if hasattr(layer.self_attn,"rotary_emb") else model.model.rotary_emb
        pp=torch.tensor([[qpos]],device=DEVICE,dtype=torch.long)
        try:c,s=rot(dummy,pp)
        except TypeError:c,s=rot(dummy,position_ids=pp)
        qrot,_=apply_rotary_pos_emb(qv,dummy,c,s)
        score=torch.einsum("bhqd,bhkd->bhqk",qrot.float(),ak.float()).squeeze(0).squeeze(1)/math.sqrt(HD)
        score[:,0]=-torch.inf
        w=torch.softmax(score,dim=-1).mean(0)
        ws.append(w)
    w=torch.stack(ws).mean(0);w[0]=0;w=w/w.sum().clamp_min(1e-12)
    return w.detach().cpu(),o

def uniform_pointer(T):
    w=torch.ones(T,dtype=torch.float32);w[0]=0;return w/w.sum()

def pointer_vector(i,w,L,kind,center=True):
    ix=0 if kind=="K" else 1;x=A_RAW[i][L][ix].float().cpu()
    mu=MU[(L,kind)] if center else torch.zeros((KVH,HD))
    xc=x-mu[:,None,:];return (xc*w[None,:,None]).sum(dim=1).reshape(-1)

def B_scores_from_vector(p,L,kind,center=True):
    ix=0 if kind=="K" else 1;mu=MU[(L,kind)] if center else torch.zeros((KVH,HD));sc=[]
    for raw in B_RAW:
        x=raw[L][ix][:,1:,:].float().cpu()-mu[:,None,:]
        M=x.permute(1,0,2).reshape(x.shape[1],-1)
        sc.append(float(cos_rows(p,M).max()))
    return sc

print("\n[5/12] Native Q·K pointer extraction...")
PTR=[];PTR_ARG=[];PTR_GOLD=[];QOUT=[]
for i,it in enumerate(ITEMS):
    w,o=native_pointer(A_PACK[i],qA(it["gold"]["entity"]));PTR.append(w);QOUT.append(o)
    a=int(torch.argmax(w));PTR_ARG.append(a);PTR_GOLD.append(int(a==GOLD_POS[i]))
    print(f"      [{i+1:02d}/48] {it['gold']['entity']:7s} | pointer={a:2d} gold={GOLD_POS[i]:2d} hit={PTR_GOLD[-1]} peak={float(w[a]):.4f}")
print(f"      Pointer argmax gold localization: {sum(PTR_GOLD)}/{N} = {np.mean(PTR_GOLD):.4f}")

print("\n[6/12] DISC — selecting comparison layer/role on items 01..24 only...")
ARMS=[(L,k) for L in CMP_LAYERS for k in ("K","V")]
DISC={}
for L,k in ARMS:
    rr=[]
    for i in range(NDISC):
        p=pointer_vector(i,PTR[i],L,k,True);r,_=rank(B_scores_from_vector(p,L,k,True),i);rr.append(r)
    m=metrics(rr);DISC[(L,k)]={"ranks":rr,"metrics":m}
    print(f"      L{L:02d}-{k} | R1={m[0]:.4f} R5={m[1]:.4f} R16={m[2]:.4f} MRR={m[3]:.6f}")
BEST=sorted(ARMS,key=lambda z:(-DISC[z]["metrics"][3],-DISC[z]["metrics"][0],z[0],0 if z[1]=="K" else 1))[0]
BL,BK=BEST
print(f"      FROZEN BEFORE EVAL: L{BL:02d}-{BK} | DISC MRR={DISC[BEST]['metrics'][3]:.6f}")

print("\n[7/12] EVAL — frozen primary + controls on items 25..48...")
RPRI=[];RUNI=[];RUNC=[];SELPRI=[]
for i in range(NDISC,N):
    pp=pointer_vector(i,PTR[i],BL,BK,True);sp=B_scores_from_vector(pp,BL,BK,True);rp,se=rank(sp,i)
    wu=uniform_pointer(len(PTR[i]));pu=pointer_vector(i,wu,BL,BK,True);ru,_=rank(B_scores_from_vector(pu,BL,BK,True),i)
    puc=pointer_vector(i,PTR[i],BL,BK,False);ruc,_=rank(B_scores_from_vector(puc,BL,BK,False),i)
    RPRI.append(rp);RUNI.append(ru);RUNC.append(ruc);SELPRI.append(se)
    print(f"      [{i+1:02d}/48] PRIMARY={rp:2d} UNIFORM={ru:2d} UNCENTERED={ruc:2d} | selected B={se+1:02d}")
for name,r in (("PRIMARY",RPRI),("UNIFORM",RUNI),("UNCENTERED",RUNC)):
    m=metrics(r);print(f"      {name:10s} R1/R5/R16/MRR: {m[0]:.4f} / {m[1]:.4f} / {m[2]:.4f} / {m[3]:.6f}")

print("\n[8/12] NO-A control...")
# No A exists here, therefore no A tensor can legitimately be pointed to.
# The strict tensor-native null is the bank-mean origin vector.
RNO=[]
zero=torch.zeros(KVH*HD)
for i in range(NDISC,N):
    r,_=rank(B_scores_from_vector(zero,BL,BK,True),i);RNO.append(r)
mno=metrics(RNO)
print(f"      NO_A R1/R5/R16/MRR: {mno[0]:.4f} / {mno[1]:.4f} / {mno[2]:.4f} / {mno[3]:.6f}")

print("\n[9/12] Counterfactual pointer-follow control...")
# Build a fresh A cartridge where the queried entity is reassigned to another item's GOLD seal.
# The counterfactual target is therefore B_j. No gold position/key is supplied to the router.
RCF=[];CFGOOD=0;CFTOT=0
for z,i in enumerate(range(NDISC,N)):
    j=NDISC+((z+7)%(N-NDISC))
    it=ITEMS[i];targetseal=ITEMS[j]["gold"]["seal"];ents=[it["gold"]["entity"],NAMES[3*i+1],NAMES[3*i+2]]
    seals=[targetseal,it["A_distractors"][0],it["A_distractors"][1]]
    rows=[f"Instrument {ents[x]} carries seal {seals[x]}." for x in range(3)]
    rr=random.Random(SEED+9000+i);rr.shuffle(rows);cfA=" ".join(rows)
    raw=forge(cfA);pack=install(raw);w,_=native_pointer(pack,qA(it["gold"]["entity"]))
    # Same persistent transformation, but pointer vector must come from the counterfactual A tensor.
    ix=0 if BK=="K" else 1;x=raw[BL][ix].float().cpu();mu=MU[(BL,BK)]
    p=((x-mu[:,None,:])*w[None,:,None]).sum(dim=1).reshape(-1)
    r,se=rank(B_scores_from_vector(p,BL,BK,True),j);RCF.append(r);CFTOT+=1;CFGOOD+=int(se==j)
    print(f"      [{i+1:02d}] target B={j+1:02d} rank={r:2d} selected={se+1:02d} follow={int(se==j)}")
mcf=metrics(RCF)
print(f"      CF FOLLOW R1/R5/R16/MRR: {mcf[0]:.4f} / {mcf[1]:.4f} / {mcf[2]:.4f} / {mcf[3]:.6f}")

print("\n[10/12] ACETATE diagnostic — fixed L24/r4 Q→K-subspace projection energy...")
# Diagnostic only. No DISC tuning, cannot replace PRIMARY.
@torch.inference_mode()
def query_q_pre_rope(o,L):
    layer=model.model.layers[L];h=o.hidden_states[L][0,-1].to(layer.input_layernorm.weight.dtype)
    return layer.self_attn.q_proj(layer.input_layernorm(h)).view(QH,HD).float().cpu()

SUB=[]
for raw in B_RAW:
    heads=[]
    for h in range(KVH):
        X=raw[ACETATE_L][0][h,1:,:].float().cpu()
        X=X-X.mean(0,keepdim=True)
        _,_,vh=torch.linalg.svd(X,full_matrices=False)
        heads.append(vh[:min(ACETATE_R,vh.shape[0])].T.contiguous())
    SUB.append(heads)

def acetate_scores(q):
    sc=[]
    for b in range(N):
        e=0.0
        for g in range(KVH):
            U=SUB[b][g];qs=q[g*REP:(g+1)*REP]
            qs=qs/qs.norm(dim=1,keepdim=True).clamp_min(1e-8)
            e+=float(((qs@U)**2).sum())
        sc.append(e)
    return sc

RAC=[]
for i in range(NDISC,N):
    q=query_q_pre_rope(QOUT[i],ACETATE_L);r,se=rank(acetate_scores(q),i);RAC.append(r)
    print(f"      [{i+1:02d}/48] ACETATE rank={r:2d} selected B={se+1:02d}")
mac=metrics(RAC)
print(f"      ACETATE L24/r4 R1/R5/R16/MRR: {mac[0]:.4f} / {mac[1]:.4f} / {mac[2]:.4f} / {mac[3]:.6f}")

print("\n[11/12] Wilson intervals + predeclared decision...")
def wilson(k,n,z=1.959963984540054):
    if n==0:return float("nan"),float("nan")
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h
for name,r in (("PRIMARY",RPRI),("UNIFORM",RUNI),("NO_A",RNO),("CF",RCF),("ACETATE",RAC)):
    k=sum(int(x==1) for x in r);lo,hi=wilson(k,len(r));print(f"      {name:9s}: {k:2d}/{len(r)} = {k/len(r):.4f} | 95% Wilson [{lo:.4f},{hi:.4f}]")
MP=metrics(RPRI);MU=metrics(RUNI);MN=metrics(RNO);MC=metrics(RCF)
PASS=MP[0]>=.60 and MP[1]>=.85 and MU[0]<=.15 and MN[0]<=.10 and MC[0]>=.70
STRONG=MP[0]>=.75 and PASS
if STRONG:VERDICT="STRONG_LABEL_FREE_NATIVE_ATTENTION_POINTER_ADDRESS_OBSERVED"
elif PASS:VERDICT="LABEL_FREE_NATIVE_ATTENTION_POINTER_ADDRESS_OBSERVED"
elif np.mean(PTR_GOLD)>=.60 and MP[0]<.30:VERDICT="QUERY_LOCALIZES_A_BUT_SAME_ROLE_TENSOR_MATCH_FAILS"
elif np.mean(PTR_GOLD)<.30:VERDICT="HEAD_AVERAGED_NATIVE_ATTENTION_POINTER_NOT_ESTABLISHED"
elif MP[0]>=.30:VERDICT="NATIVE_POINTER_ADDRESS_SIGNAL_PRESENT_BUT_BELOW_PREDECLARED_RELIABILITY"
else:VERDICT="NATIVE_ATTENTION_POINTER_ADDRESS_NOT_ESTABLISHED"
print("      Primary pass :",PASS)
print("      Strong pass  :",STRONG)
print("      VERDICT      :",VERDICT)

print("\n[12/12] Protocol audit...")
print("      Model frozen                              : YES")
print("      TEST518 corrected 48-item panel           : YES")
print("      DISC / EVAL separated                     : YES — 24 / 24")
print("      EVAL used for arm selection               : NO")
print("      Query pointer uses native q_proj           : YES")
print("      Query pointer uses native RoPE             : YES")
print("      Pointer target uses installed RoPE A K     : YES")
print("      Pointer layers                             : FIXED L14-L27")
print("      Gold A position used by address            : NO")
print("      Gold A position used for diagnostic only   : YES")
print("      Primary B address token IDs                : NO")
print("      Primary B address decoded text             : NO")
print("      Primary B address LM-head rows             : NO")
print("      Primary B address gold key position        : NO")
print("      Query-time candidate-B forwards            : 0")
print("      Primary persistent address source          : BELLEKÖZ K/V ONLY")
print("      Bank centering source                      : BELLEKÖZ K/V ONLY")
print("      Training / DRA / learned router            : NONE")
print("      Acetate L24/r4                             : DIAGNOSTIC ONLY")
print("      Acetate allowed to replace primary         : NO")

print("\n"+"="*176)
print("TEST520 FINAL RESULT — AKBASCORE MAM · NATIVE ATTENTION-POINTER ADDRESS")
print("="*176)
print("MODEL                              : Qwen/Qwen2.5-7B-Instruct · frozen")
print("PANEL                              : TEST518 corrected 48-item panel")
print("DISC / EVAL                        : 24 / 24")
print(f"FROZEN PRIMARY ARM                 : L{BL:02d}-{BK}")
print("POINTER                            : native Q·K · RoPE · L14-L27 · A positions")
print("ADDRESS                            : centered pointed-A tensor ↔ B tensor positions")
print("TOKEN-ID / GOLD / TEXT ADDRESS     : NONE")
print("-"*176)
md=DISC[BEST]["metrics"]
print(f"DISC PRIMARY                       : R1={md[0]:.4f} R5={md[1]:.4f} R16={md[2]:.4f} MRR={md[3]:.6f}")
print(f"EVAL PRIMARY                       : R1={MP[0]:.4f} R5={MP[1]:.4f} R16={MP[2]:.4f} MRR={MP[3]:.6f}")
print(f"EVAL UNIFORM POINTER               : R1={MU[0]:.4f} R5={MU[1]:.4f} R16={MU[2]:.4f} MRR={MU[3]:.6f}")
print(f"EVAL NO_A                          : R1={MN[0]:.4f} R5={MN[1]:.4f} R16={MN[2]:.4f} MRR={MN[3]:.6f}")
print(f"EVAL COUNTERFACTUAL FOLLOW         : R1={MC[0]:.4f} R5={MC[1]:.4f} R16={MC[2]:.4f} MRR={MC[3]:.6f}")
print(f"POINTER GOLD LOCALIZATION          : {sum(PTR_GOLD)}/{N} = {np.mean(PTR_GOLD):.4f}")
print(f"ACETATE L24/r4 DIAGNOSTIC          : R1={mac[0]:.4f} R5={mac[1]:.4f} R16={mac[2]:.4f} MRR={mac[3]:.6f}")
print("-"*176)
print("PREDECLARED PASS                   : EVAL R1>=.60; R5>=.85; UNIFORM<=.15; NO_A<=.10; CF>=.70")
print("PRIMARY PASS                       :",PASS)
print("STRONG PASS                        :",STRONG)
print("TEST518 LOCK                       :",TEST518)
print("TEST519 LOCK                       :",TEST519)
print("TEST520 LOCK                       :",LOCK_SHA)
print(f"TOTAL TEST TIME                    : {time.perf_counter()-T0:.2f}s")
print("VERDICT                            :",VERDICT)
print("="*176)

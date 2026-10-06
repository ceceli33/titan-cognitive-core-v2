# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST521
# HELD-OUT CAUSAL VALIDATION OF TEST520 UNCENTERED SIGNAL
#
# PARENT:
# TEST520 unexpectedly produced UNCENTERED L02-V EVAL R1=14/24=.5833,
# while centered PRIMARY failed. Gold-position diagnostic in TEST520 was invalid.
#
# TEST521 asks ONE question:
# Is the TEST520 uncentered signal a genuine query-dependent tensor address,
# or a template/position/common-geometry shortcut?
#
# PRE-REGISTERED PRIMARY:
#   comparison layer = L02
#   representation   = V
#   centering        = OFF
#   pointer layers   = L14-L27
#   pointer          = native Q·K with RoPE, averaged over 28 Q heads and L14-L27
#
# NO DISCOVERY / NO ARM SELECTION / NO POST-HOC LAYER CHOICE.
# Entire panel is NEW and disjoint from TEST520.
#
# PRIMARY ADDRESS:
#   query -> native attention over A BELLEKÖZ
#   -> weighted A L02-V tensor
#   -> max cosine against B L02-V positions
#
# ADDRESS USES:
#   NO token IDs
#   NO decoded Seal
#   NO LM-head rows
#   NO gold key position
#   NO B text
#   NO human labels
#   NO query-time B forward
#
# DIAGNOSTICS ONLY:
#   true gold Seal span from character offsets -> tokenizer offset mapping
#   pointer argmax/span mass/span rank
#   token/span information NEVER enters retrieval.
#
# CAUSAL CONTROLS:
#   UNIFORM     : removes query-dependent pointer
#   SHIFTED     : cyclically shifts the same pointer over A positions
#   NO-A        : zero tensor query
#   CF-FOLLOW   : queried entity is reassigned to another bank item's Seal;
#                 address must move to that item's B.
#
# PREDECLARED PASS:
#   PRIMARY R1 >= .50
#   PRIMARY R5 >= .75
#   CF FOLLOW R1 >= .60
#   UNIFORM R1 <= .15
#   SHIFTED R1 <= .20
#   NO-A R1 <= .10
#
# Strong:
#   PRIMARY R1 >= .60 and all controls pass.
#
# Pointer localization is mechanistic diagnostic, NOT a pass requirement.

import os,sys,subprocess,importlib.util,random,time,hashlib,json,gc,re,math
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

TEST="521";SEED=521;MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N=64;DEVICE=torch.device("cuda")
FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n";PTR_LAYERS=list(range(14,28));BL=2;BK="V"
TEST518="d335fa02068644efbe3c54eba8b6ae3a4678fbba3abed972d3f31a401590ae3c"
TEST519="2bcf12382a2341ca84fcb1c21a9b52d4c4de79af50ba21d8d3b853f34c9f0f1a"
TEST520="fb44f20176ae72ce58d9eaf99436e237f4e0548b893a7ee51d8f4d7768b0b806"
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*176)
print("TEST521 — AKBASCORE MAM · HELD-OUT CAUSAL VALIDATION OF UNCENTERED NATIVE POINTER ADDRESS")
print("FROZEN L02-V UNCENTERED · NEW 64-ITEM BANK · POINTER LOCALIZATION + SHIFT + COUNTERFACTUAL CAUSAL CONTROLS")
print("="*176)
T0=time.perf_counter()

print("\n[1/11] Loading frozen Qwen...")
_tv=tuple(int("".join(c for c in x if c.isdigit())or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
if not tok.is_fast:raise RuntimeError("Fast tokenizer required for diagnostic offset mapping.")
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
enc=lambda s:tok(s,add_special_tokens=False).input_ids
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=H//QH;REP=QH//KVH
if (NL,H,QH,KVH,HD)!=(28,3584,28,4,128):raise RuntimeError("Architecture mismatch.")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id;VOC=model.model.embed_tokens.weight.shape[0]
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | {NL}L H={H} QH={QH} KVH={KVH} HD={HD} | frozen BF16")
print("      PRIMARY LOCK: L02-V · UNCENTERED · PTR L14-L27")

SYL1=["Qua","Ren","Sol","Tor","Ul","Var","Wen","Xan","Yor","Zen","Bra","Cre","Dre","Fre","Gre","Kra","Myr","Pry","Syr","Tyr","Vel","Zor"]
SYL2=["dor","len","mar","nis","pel","ris","ton","vek","wyn","zar","bis","cus","fer","gos","hel","jor"]
def build_names(n):
    out=[]
    for a in SYL1:
        for b in SYL2:
            x=a+b
            if x not in out:out.append(x)
            if len(out)>=n:return out
    raise RuntimeError("Name pool too small.")

def native_codes(n,exclude=set(),skip=0):
    out=[];seen=0
    for tid in range(VOC):
        s=tok.decode([tid],skip_special_tokens=False)
        if s in exclude:continue
        if not re.fullmatch(r"[A-Z]{3,8}",s):continue
        if enc(s)!=[tid]:continue
        if seen<skip:seen+=1;continue
        if s in out:continue
        out.append(s)
        if len(out)>=n:return out
    raise RuntimeError(f"Need {n} native codes, found {len(out)}.")

# Deliberately skip the first native-code region used by TEST520.
# This is panel construction only; codes never enter the runtime address.
NAMES=build_names(N*3)
SEALS=native_codes(N*3,skip=512)
CLASSES=native_codes(N*3,exclude=set(SEALS),skip=1024)

def make_items():
    items=[]
    for i in range(N):
        ents=NAMES[3*i:3*i+3];aseals=SEALS[3*i:3*i+3];gold=aseals[0]
        j1=(i+11)%N;j2=(i+29)%N
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

LOCK={"test":"521","parent520":TEST520,"model":MODEL_ID,"seed":SEED,"N":N,
"primary":{"layer":BL,"kind":BK,"center":False},"pointer_layers":PTR_LAYERS,
"panel":"fresh_64_disjoint_from_TEST520","controls":["uniform","shifted_pointer","noA","counterfactual"],
"pass":{"r1":.50,"r5":.75,"cf":.60,"uniform_max":.15,"shifted_max":.20,"noA_max":.10}}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      TEST521 LOCK:",LOCK_SHA)
print("      PARENT520   :",TEST520)
print("      NEW BANK    : 64 items / 128 independent BELLEKÖZ")
print("      DISCOVERY   : NONE")
print("      Selection   : NONE")

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
    a=np.asarray(r,float)
    return float(np.mean(a==1)),float(np.mean(a<=5)),float(np.mean(a<=16)),float(np.mean(1/a))

def cos_rows(v,M):
    v=v.float();M=M.float();v=v/v.norm().clamp_min(1e-8);M=M/M.norm(dim=1,keepdim=True).clamp_min(1e-8)
    return M@v

print("\n[2/11] OFFLINE FORGE — fresh held-out 64-item panel...")
A_RAW=[];B_RAW=[];A_PACK=[];B_PACK=[]
for i,it in enumerate(ITEMS):
    ar=forge(it["A"]);br=forge(it["B"]);A_RAW.append(ar);B_RAW.append(br);A_PACK.append(install(ar));B_PACK.append(install(br))
    if i<5:print(f"      ITEM {i+1:02d} | A={A_PACK[-1][1]} B={B_PACK[-1][1]} tokens")
print("      128 BELLEKÖZ forged.")
print("      Runtime address will use only persistent tensor state.")

print("\n[3/11] Correct gold-span diagnostic map via character offsets...")
def gold_token_positions(text,seal):
    # DIAGNOSTIC ONLY. Character span -> tokenizer offset mapping.
    # +1 below accounts for the prepended PAD in forge().
    hits=[m.span() for m in re.finditer(re.escape(seal),text)]
    if len(hits)!=1:raise RuntimeError(f"Gold seal occurrence count !=1 for {seal}: {len(hits)}")
    a,b=hits[0]
    z=tok(text+SEP,add_special_tokens=False,return_offsets_mapping=True)
    pos=[]
    for j,(x,y) in enumerate(z["offset_mapping"]):
        if y>a and x<b:pos.append(j+1)
    if not pos:raise RuntimeError(f"No token span found for {seal}")
    return pos

GOLD_SPANS=[gold_token_positions(it["A"],it["gold"]["seal"]) for it in ITEMS]
for i in range(5):print(f"      ITEM {i+1:02d} gold span={GOLD_SPANS[i]}")
print("      Gold spans are DIAGNOSTIC ONLY and never enter address computation.")
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
    inst,Tm,P=packet;o,_,_,n=query_forward(packet,q);ws=[];qpos=P+n-1
    for L in PTR_LAYERS:
        layer=model.model.layers[L];h=o.hidden_states[L][0,-1].to(layer.input_layernorm.weight.dtype);hn=layer.input_layernorm(h)
        qv=layer.self_attn.q_proj(hn).view(1,QH,1,HD)
        ak=inst[L][0].repeat_interleave(REP,dim=0).unsqueeze(0)
        dummy=torch.zeros_like(ak[:,:,:1,:]);rot=layer.self_attn.rotary_emb if hasattr(layer.self_attn,"rotary_emb") else model.model.rotary_emb
        pp=torch.tensor([[qpos]],device=DEVICE,dtype=torch.long)
        try:c,s=rot(dummy,pp)
        except TypeError:c,s=rot(dummy,position_ids=pp)
        qrot,_=apply_rotary_pos_emb(qv,dummy,c,s)
        score=torch.einsum("bhqd,bhkd->bhqk",qrot.float(),ak.float()).squeeze(0).squeeze(1)/math.sqrt(HD)
        score[:,0]=-torch.inf
        ws.append(torch.softmax(score,dim=-1).mean(0))
    w=torch.stack(ws).mean(0);w[0]=0;w=w/w.sum().clamp_min(1e-12)
    return w.detach().cpu()

def uniform_pointer(T):
    w=torch.ones(T,dtype=torch.float32);w[0]=0;return w/w.sum()

def shifted_pointer(w):
    # Deterministic causal ablation. Preserve weight distribution/entropy,
    # destroy alignment to original A positions.
    n=len(w)-1
    if n<=1:return w.clone()
    shift=max(1,n//2);x=w[1:].clone();x=torch.roll(x,shifts=shift,dims=0)
    z=torch.zeros_like(w);z[1:]=x;return z/z.sum().clamp_min(1e-12)

def pointer_vector_from_raw(raw,w):
    # PRE-REGISTERED TEST520 signal: L02-V, UNCENTERED.
    x=raw[BL][1].float().cpu()
    return (x*w[None,:,None]).sum(dim=1).reshape(-1)

def B_scores(p):
    sc=[]
    for raw in B_RAW:
        x=raw[BL][1][:,1:,:].float().cpu()
        M=x.permute(1,0,2).reshape(x.shape[1],-1)
        sc.append(float(cos_rows(p,M).max()))
    return sc

def pointer_diag(w,span):
    mass=float(w[span].sum());arg=int(torch.argmax(w));hit=int(arg in span)
    # Rank of the strongest token in the gold span among all non-PAD positions.
    vals=w[1:];best=float(w[span].max());r=1+int((vals>best).sum())
    return arg,hit,mass,r

print("\n[4/11] Native pointer extraction + corrected localization diagnostics...")
PTR=[];PARG=[];PHIT=[];PMASS=[];PRANK=[]
for i,it in enumerate(ITEMS):
    w=native_pointer(A_PACK[i],qA(it["gold"]["entity"]));PTR.append(w)
    a,h,m,r=pointer_diag(w,GOLD_SPANS[i]);PARG.append(a);PHIT.append(h);PMASS.append(m);PRANK.append(r)
    print(f"      [{i+1:02d}/64] {it['gold']['entity']:8s} | ptr={a:2d} gold={GOLD_SPANS[i]} hit={h} mass={m:.4f} span-rank={r:2d} peak={float(w[a]):.4f}")
print(f"      Argmax-in-gold-span : {sum(PHIT)}/{N} = {np.mean(PHIT):.4f}")
print(f"      Mean gold-span mass : {np.mean(PMASS):.6f}")
print(f"      Median span rank    : {np.median(PRANK):.1f}")

print("\n[5/11] PRIMARY held-out retrieval — frozen L02-V UNCENTERED...")
RPRI=[];SELPRI=[]
for i in range(N):
    p=pointer_vector_from_raw(A_RAW[i],PTR[i]);r,se=rank(B_scores(p),i);RPRI.append(r);SELPRI.append(se)
    print(f"      [{i+1:02d}/64] rank={r:2d} selected B={se+1:02d}")
MP=metrics(RPRI)
print(f"      PRIMARY R1/R5/R16/MRR: {MP[0]:.4f} / {MP[1]:.4f} / {MP[2]:.4f} / {MP[3]:.6f}")

print("\n[6/11] Causal ablations — UNIFORM / SHIFTED / NO-A...")
RUNI=[];RSHIFT=[];RNO=[]
zero=torch.zeros(KVH*HD,dtype=torch.float32)
for i in range(N):
    wu=uniform_pointer(len(PTR[i]));pu=pointer_vector_from_raw(A_RAW[i],wu);ru,_=rank(B_scores(pu),i);RUNI.append(ru)
    ws=shifted_pointer(PTR[i]);ps=pointer_vector_from_raw(A_RAW[i],ws);rs,_=rank(B_scores(ps),i);RSHIFT.append(rs)
    rn,_=rank(B_scores(zero),i);RNO.append(rn)
MU=metrics(RUNI);MS=metrics(RSHIFT);MN=metrics(RNO)
print(f"      UNIFORM R1/R5/R16/MRR: {MU[0]:.4f} / {MU[1]:.4f} / {MU[2]:.4f} / {MU[3]:.6f}")
print(f"      SHIFTED R1/R5/R16/MRR: {MS[0]:.4f} / {MS[1]:.4f} / {MS[2]:.4f} / {MS[3]:.6f}")
print(f"      NO-A    R1/R5/R16/MRR: {MN[0]:.4f} / {MN[1]:.4f} / {MN[2]:.4f} / {MN[3]:.6f}")

print("\n[7/11] Counterfactual causal-follow — UNCENTERED L02-V...")
RCF=[];CFSEL=[];CFHIT=[];CFMASS=[]
for i in range(N):
    j=(i+17)%N
    it=ITEMS[i];targetseal=ITEMS[j]["gold"]["seal"]
    ents=[it["gold"]["entity"],NAMES[3*i+1],NAMES[3*i+2]]
    seals=[targetseal,it["A_distractors"][0],it["A_distractors"][1]]
    rows=[f"Instrument {ents[x]} carries seal {seals[x]}." for x in range(3)]
    rr=random.Random(SEED+9000+i);rr.shuffle(rows);cfA=" ".join(rows)
    raw=forge(cfA);pack=install(raw);w=native_pointer(pack,qA(it["gold"]["entity"]))
    p=pointer_vector_from_raw(raw,w);r,se=rank(B_scores(p),j);RCF.append(r);CFSEL.append(se)
    span=gold_token_positions(cfA,targetseal);_,h,m,_=pointer_diag(w,span);CFHIT.append(h);CFMASS.append(m)
    print(f"      [{i+1:02d}/64] target B={j+1:02d} rank={r:2d} selected={se+1:02d} follow={int(se==j)} ptr-hit={h} mass={m:.4f}")
MC=metrics(RCF)
print(f"      CF FOLLOW R1/R5/R16/MRR: {MC[0]:.4f} / {MC[1]:.4f} / {MC[2]:.4f} / {MC[3]:.6f}")
print(f"      CF pointer span hit       : {sum(CFHIT)}/{N} = {np.mean(CFHIT):.4f}")

print("\n[8/11] Mechanistic association diagnostics...")
CORRECT=np.asarray([r==1 for r in RPRI],dtype=bool);HIT=np.asarray(PHIT,dtype=bool);MASS=np.asarray(PMASS)
def safe_mean(x):return float(np.mean(x)) if len(x) else float("nan")
print(f"      Retrieval R1 | pointer-hit     : {safe_mean(CORRECT[HIT]):.4f}  n={int(HIT.sum())}")
print(f"      Retrieval R1 | pointer-miss    : {safe_mean(CORRECT[~HIT]):.4f}  n={int((~HIT).sum())}")
print(f"      Mean gold mass | retrieval R1  : {safe_mean(MASS[CORRECT]):.6f}  n={int(CORRECT.sum())}")
print(f"      Mean gold mass | retrieval fail: {safe_mean(MASS[~CORRECT]):.6f}  n={int((~CORRECT).sum())}")
print(f"      PRIMARY minus UNIFORM R1       : {MP[0]-MU[0]:+.4f}")
print(f"      PRIMARY minus SHIFTED R1       : {MP[0]-MS[0]:+.4f}")

print("\n[9/11] Wilson intervals...")
def wilson(k,n,z=1.959963984540054):
    if n==0:return float("nan"),float("nan")
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h
for name,r in (("PRIMARY",RPRI),("UNIFORM",RUNI),("SHIFTED",RSHIFT),("NO_A",RNO),("CF",RCF)):
    k=sum(int(x==1) for x in r);lo,hi=wilson(k,len(r))
    print(f"      {name:8s}: {k:2d}/{len(r)} = {k/len(r):.4f} | 95% Wilson [{lo:.4f},{hi:.4f}]")

print("\n[10/11] Predeclared decision...")
PASS=MP[0]>=.50 and MP[1]>=.75 and MC[0]>=.60 and MU[0]<=.15 and MS[0]<=.20 and MN[0]<=.10
STRONG=MP[0]>=.60 and PASS
if STRONG:VERDICT="STRONG_UNCENTERED_NATIVE_POINTER_ADDRESS_CAUSALLY_VALIDATED"
elif PASS:VERDICT="UNCENTERED_NATIVE_POINTER_ADDRESS_CAUSALLY_VALIDATED"
elif MP[0]>=.50 and (MU[0]>.15 or MS[0]>.20):
    VERDICT="HIGH_RETRIEVAL_BUT_POINTER_CAUSALITY_NOT_ESTABLISHED"
elif MP[0]>=.50 and MC[0]<.60:
    VERDICT="HIGH_STATIC_RETRIEVAL_BUT_COUNTERFACTUAL_FOLLOW_FAILED"
elif MP[0]<.25 and MU[0]<.15:
    VERDICT="TEST520_UNCENTERED_SIGNAL_DID_NOT_REPLICATE_ON_HELDOUT_BANK"
else:
    VERDICT="UNCENTERED_NATIVE_POINTER_SIGNAL_PARTIAL_INCONCLUSIVE"
print("      PASS   :",PASS)
print("      STRONG :",STRONG)
print("      VERDICT:",VERDICT)

print("\n[11/11] Protocol audit + final result...")
print("      Model frozen                              : YES")
print("      Panel reused from TEST520                 : NO")
print("      Fresh held-out items                      : 64")
print("      DISC / layer selection                    : NONE")
print("      Primary layer/representation              : L02-V PRE-LOCKED")
print("      Centering                                 : OFF PRE-LOCKED")
print("      Pointer layers                            : L14-L27 PRE-LOCKED")
print("      Primary address token IDs                 : NO")
print("      Primary address decoded Seal/text         : NO")
print("      Primary address LM-head rows              : NO")
print("      Gold Seal span used by address            : NO")
print("      Gold Seal span used diagnostic only       : YES")
print("      Query-time candidate-B forwards           : 0")
print("      Training / DRA / learned router           : NONE")
print("      Counterfactual uses same frozen mechanism : YES")
print("      Shift control preserves pointer weights   : YES")
print("\n"+"="*176)
print("TEST521 FINAL RESULT — HELD-OUT CAUSAL VALIDATION")
print("="*176)
print("MODEL                              : Qwen/Qwen2.5-7B-Instruct · frozen")
print("BANK                               : NEW 64-item held-out panel")
print("PRIMARY                            : L02-V · UNCENTERED")
print("POINTER                            : native Q·K · RoPE · L14-L27")
print("DISCOVERY / POST-HOC SELECTION     : NONE")
print("-"*176)
print(f"PRIMARY                            : R1={MP[0]:.4f} R5={MP[1]:.4f} R16={MP[2]:.4f} MRR={MP[3]:.6f}")
print(f"UNIFORM                            : R1={MU[0]:.4f} R5={MU[1]:.4f} R16={MU[2]:.4f} MRR={MU[3]:.6f}")
print(f"SHIFTED POINTER                    : R1={MS[0]:.4f} R5={MS[1]:.4f} R16={MS[2]:.4f} MRR={MS[3]:.6f}")
print(f"NO-A                               : R1={MN[0]:.4f} R5={MN[1]:.4f} R16={MN[2]:.4f} MRR={MN[3]:.6f}")
print(f"COUNTERFACTUAL FOLLOW              : R1={MC[0]:.4f} R5={MC[1]:.4f} R16={MC[2]:.4f} MRR={MC[3]:.6f}")
print(f"POINTER GOLD-SPAN ARGMAX HIT       : {sum(PHIT)}/{N} = {np.mean(PHIT):.4f}")
print(f"POINTER MEAN GOLD-SPAN MASS        : {np.mean(PMASS):.6f}")
print(f"POINTER MEDIAN GOLD-SPAN RANK      : {np.median(PRANK):.1f}")
print("-"*176)
print("PREDECLARED PASS                   : PRIMARY R1>=.50 R5>=.75; CF>=.60; UNIFORM<=.15; SHIFTED<=.20; NO-A<=.10")
print("PRIMARY PASS                       :",PASS)
print("STRONG PASS                        :",STRONG)
print("TEST518 LOCK                       :",TEST518)
print("TEST519 LOCK                       :",TEST519)
print("TEST520 LOCK                       :",TEST520)
print("TEST521 LOCK                       :",LOCK_SHA)
print(f"TOTAL TEST TIME                    : {time.perf_counter()-T0:.2f}s")
print("VERDICT                            :",VERDICT)
print("="*176)

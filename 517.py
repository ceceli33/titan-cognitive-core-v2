# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST517
# TOKEN-FREE BELLEKÖZ-NATIVE ÇAĞRIİZ BRIDGE
#
# PARENT:
#   TEST516:
#     address R1                  = 23/32
#     oracle numeric handoff      = 31/32
#     selected numeric E2E        = 20/32
#     selected | correct address  = 20/23
#
# QUESTION:
# Can TEST516's numeric relay survive after removing the human-defined
# Seal-token / LM-head / token-embedding address scaffold?
#
# CORE CHANGE FROM TEST516:
#   TEST516:
#       Seal -> tokenizer IDs -> LM-head rows / embedding rows
#
#   TEST517:
#       NO Seal tokenization for routing
#       NO candidate key vocabulary
#       NO LM-head key rows
#       NO oracle Seal embedding carrier
#
#       A BELLEKÖZ + query
#           -> frozen hidden ÇAĞRIİZ
#
#       B BELLEKÖZ
#           -> fingerprint extracted directly from its persistent K/V tensors
#
#       ÇAĞRIİZ <-> B fingerprint
#           -> numeric address
#
#       selected B
#           -> fresh independent B session
#           -> continuous numeric A-trace carrier
#           -> final value
#
# DISCOVERY / EVAL:
#   DISCOVERY = items 0..15
#   EVAL      = items 16..31
#   Discovery chooses ONE predeclared native geometry.
#   EVAL performs NO reselection / threshold tuning / post-hoc adjustment.
#
# CRITICAL:
#   model frozen
#   forge/runtime separated
#   A and B never share a cache
#   no intermediate Seal decode
#   no Seal token IDs in routing/handoff
#   no candidate-B forward at query time
#   no learned router
#   no ANN
#   no DRA
#   no training
#   no eval-time tuning
#
# Human-readable Entity/Seal/Class strings exist only for controlled source
# construction, natural query and final evaluator.
# Persistent memory and routing/handoff state are tensors.

import os,sys,subprocess,importlib.util,random,time,hashlib,json,gc
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache

TEST="517";SEED=517;BASE_SEED=501;MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
N=32;DISC=list(range(16));EVAL=list(range(16,32));MAX_NEW=12;DEVICE=torch.device("cuda")
FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
TEST503="2cb70ffa53a98efb249590dba7ef184b9e41cf72a99291bda65e6c93b82cb5de"
TEST504="74c375373666c840105cc165b90977c56c8698cf3d5617c63f619a0139386846"
TEST513="2b28137b12a26517261d0d8827c378f7e9e455bd5f4ccb12af47277ed8de01aa"
TEST514="61e83ffe4b57cbafa0f4ef8a3fe516a153818fa43e87176941fd01ec8ad24146"
TEST515="cf9e875f1a8fa8403f688c7150ea9bfec4dd3bea407cf2ce44f27e1961c9f7fa"
TEST516="53f4fa86f1bc4f2c4db412bde68a2b004afc5a6c6cf84d1c3345b175ab1a3e8d"

ENTITY_BANK=[
("Aldren","Boreal","Cyrene"),("Darian","Elara","Faron"),("Galen","Hesper","Ilyra"),("Joren","Kaelis","Lorin"),
("Maren","Neris","Orlan"),("Perrin","Quorin","Ralen"),("Saren","Taris","Ulric"),("Valen","Weyra","Xeran"),
("Yorin","Zaren","Avel"),("Brann","Ceris","Dalen"),("Eris","Felis","Gorin"),("Halen","Ivar","Jaris"),
("Koren","Leris","Miran"),("Nolan","Orel","Palis"),("Riven","Solis","Teren"),("Urian","Varen","Wilis"),
("Xorin","Yalen","Zorin"),("Arven","Belis","Coren"),("Derin","Evan","Feris"),("Garin","Heron","Ilven"),
("Jarin","Kelis","Laven"),("Moris","Naven","Orris"),("Parin","Rovis","Selan"),("Torin","Ulen","Veris"),
("Waren","Xelis","Yaris"),("Zelis","Aren","Borin"),("Caren","Dorin","Elen"),("Faren","Gelis","Harin"),
("Iren","Joris","Kalen"),("Laris","Meren","Noren"),("Oris","Peren","Ravin"),("Serin","Toren","Ulis")]
SEAL_BANK=["KOR","VEL","DAR","MIR","SEN","ROL","FEN","JAL","WEX","NUR","BAV","CIR","DEM","GOS","HIL","KET","LOR","MEV","PIR","RUK","SAV","TOL","VIR","YEK","ZAM","BIR","CAV","DOL","FER","GUL","HAR","JEM","KIR","LEV","MOR","NEX","PEL","RAS","SUL","TIR","VAN","YOR","ZEL","BOS","CER","DIN","FAL","GER","HOV","JUN","KEL","LUM","NAV","POR","REV","SIM","TUR","WAL","XEN","YIL","ZOR","BEK","COR","DUR","EKS","FIR","GAN","HEL","IVO","JOR","KAS","LIN","MUR","NOR","OVA","PAR","RIN","SOL","TEV","URB","VAR","WEN","XAL","YUN","ZEN","BOR","CEN","DAN","ELV","FOR","GIR","HAN","IRV","JEN","KOL","MAR"]
CLASS_BANK=["TAK","BEX","QAA","RAV","SOD","PEK","NIV","GOR","HAX","JUR","KEM","VOL","DAX","QAB","MON","SAL","TEK","WIR","ZUN","COV","HEM","JAX","LIV","QAC","PAK","RUM","SEV","TIX","VOR","YAM","ZEK","BOL","QAD","DOV","FEX","GAM","HUR","JIN","KAV","LER","MEX","QAE","PIV","ROK","SUM","TAL","VEK","WON","XIR","YAV","ZOL","BAR","CIX","QAF","FOV","GEL","HIN","JOV","KUR","QAG","MAV","QAH","PUL","RIM","QAI","TOX","VIL","WER","XAN","YER","ZIM","BUN","CAL","DOR","EVI","FAR","GUN","HES","ILM","JER","KON","LAR","QAJ","NOL","OVI","PER","RUS","SIN","QAK","VEX","QAL","QAM","YUL","ZAR","BEL","CUM"]

assert len(ENTITY_BANK)==N and len(SEAL_BANK)>=N*3 and len(CLASS_BANK)>=N*3
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
torch.set_grad_enabled(False)
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

def reorder(rows,gold,pos,seed):
    g=rows[gold];rest=[x for j,x in enumerate(rows) if j!=gold]
    rr=random.Random(seed);rr.shuffle(rest);out=rest[:];out.insert(pos,g);return out

def make_items():
    rng=random.Random(BASE_SEED);seals=SEAL_BANK.copy();classes=CLASS_BANK.copy()
    rng.shuffle(seals);rng.shuffle(classes);items=[]
    for i in range(N):
        ents=list(ENTITY_BANK[i]);g=i%3;ss=seals[3*i:3*i+3];cc=classes[3*i:3*i+3]
        ra=[f"Instrument {ents[j]} carries seal {ss[j]}." for j in range(3)]
        rb=[f"Seal {ss[j]} corresponds to routing class {cc[j]}." for j in range(3)]
        A=" ".join(reorder(ra,g,i%3,BASE_SEED+i*101+17))
        B=" ".join(reorder(rb,g,(i+1)%3,BASE_SEED+i*101+34))
        items.append({"id":i+1,"A":A,"B":B,"gold":{"entity":ents[g],"seal":ss[g],"class":cc[g]}})
    return items

ITEMS=make_items()
LOCK={"test":TEST,"model":MODEL_ID,"seed":SEED,"base_seed":BASE_SEED,"items":ITEMS,
"parents":{"503":TEST503,"504":TEST504,"513":TEST513,"514":TEST514,"515":TEST515,"516":TEST516},
"question":"TOKEN_FREE_BELLEKOZ_NATIVE_BRIDGE",
"discovery":DISC,"eval":EVAL,
"candidate_trace_layers":[20,23,25,27,28],
"candidate_B_layers":[20,23,25,27],
"candidate_families":["KMEAN","VMEAN","KVMEAN"],
"selection":"DISCOVERY_ONLY_THEN_FROZEN",
"seal_tokenization_for_routing":"FORBIDDEN",
"lm_head_key_rows":"FORBIDDEN",
"gold_seal_carrier":"FORBIDDEN",
"A_B_cache_concat":"FORBIDDEN",
"query_time_candidate_B_forwards":0,
"training":"OFF","dra":"OFF","ann":"OFF","learned_router":"OFF",
"persistence_contract":"FORGE_THEN_FRESH_RUNTIME_STATE"}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

print("="*180)
print("TEST517 — AKBASCORE MAM · TOKEN-FREE BELLEKÖZ-NATIVE ÇAĞRIİZ BRIDGE")
print("A LAYER STATE → B'S OWN K/V FINGERPRINT → FRESH B SESSION → CONTINUOUS NUMERIC HANDOFF")
print("="*180)
print("LOCK SHA:",LOCK_SHA)
print("PARENT TEST516:",TEST516)
print("DISCOVERY:",DISC)
print("EVAL     :",EVAL)
print("SEAL TOKEN / LM-HEAD KEY GEOMETRY: FORBIDDEN")
T0=time.perf_counter()

print("\n[1/10] Loading frozen Qwen...")
_tv=tuple(int("".join(c for c in x if c.isdigit())or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
_ge=model.generation_config.eos_token_id
EOS=sorted({tok.eos_token_id,*(_ge if isinstance(_ge,(list,tuple)) else [_ge])}-{None})
enc=lambda s:tok(s,add_special_tokens=False).input_ids
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=H//QH
if (NL,H,QH,KVH,HD)!=(28,3584,28,4,128):raise RuntimeError("Architecture mismatch.")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
EMB=model.model.embed_tokens
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | {NL}L H={H} QH={QH} KVH={KVH} HD={HD} | frozen BF16")

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
    T=raw[0][0].shape[1];out=[]
    for L,(k,v) in enumerate(raw):out.append((rope_k(k,list(range(T)),L),v))
    return out,T,T

def cache_of(inst):
    try:c=DynamicCache(config=cfg)
    except TypeError:c=DynamicCache()
    for L,(k,v) in enumerate(inst):c.update(k.unsqueeze(0),v.unsqueeze(0),L)
    return c

def unit(x):
    x=x.float().reshape(-1)
    return x/x.norm().clamp_min(1e-8)

def cosine(a,b):return float(torch.dot(unit(a),unit(b)))
def firstline(x):return next((z.strip() for z in str(x).splitlines() if z.strip()),"")
def normtxt(x):return firstline(x).strip().upper().rstrip(".")
def exact(x,t):return normtxt(x)==str(t).upper()
def qA(e):return f"What seal does instrument {e} carry? Give only the exact seal."

print("\n[2/10] OFFLINE FORGE — persistent A/B BELLEKÖZ...")
A_RAW=[];B_RAW=[];A_PACK=[];B_PACK=[]
for i,it in enumerate(ITEMS):
    ar=forge(it["A"]);br=forge(it["B"])
    A_RAW.append(ar);B_RAW.append(br);A_PACK.append(install(ar));B_PACK.append(install(br))
    if i<5:print(f"      ITEM {i+1:02d} | A={A_PACK[-1][1]} B={B_PACK[-1][1]} tokens")
assert len(A_RAW)==len(B_RAW)==len(A_PACK)==len(B_PACK)==N
print("      64 tensor cartridges forged.")

print("\n[3/10] Building B-native numeric fingerprints...")
# IMPORTANT:
# PAD row is excluded. Fingerprints come only from B's own pre-RoPE K/V tensors.
# No Seal token position is identified or used.
B_FP={}
for L in [20,23,25,27]:
    for fam in ("KMEAN","VMEAN","KVMEAN"):
        arr=[]
        for raw in B_RAW:
            k,v=raw[L];k=k[:,1:,:].float().cpu();v=v[:,1:,:].float().cpu()
            km=k.mean(dim=1).reshape(-1);vm=v.mean(dim=1).reshape(-1)
            if fam=="KMEAN":z=km
            elif fam=="VMEAN":z=vm
            else:z=torch.cat([km,vm])
            arr.append(unit(z))
        B_FP[(L,fam)]=arr
print("      Fingerprints: KMEAN / VMEAN / KVMEAN × L20/L23/L25/L27")
print("      Seal tokenizer / LM-head rows used: 0")

gc.collect();torch.cuda.empty_cache()
print("\n[4/10] PERSISTENCE RESET...")
print("      Forge-time runtime state discarded.")
print("      Persistent BELLEKÖZ = tensor packets + tensor-derived fingerprints.")
print("      Every query/read starts with a fresh DynamicCache.")

@torch.inference_mode()
def capture_A(packet,q):
    inst,Tm,P=packet;c=cache_of(inst);ids=enc(FMT.format(q=q));n=len(ids)
    pos=torch.arange(P,P+n,device=DEVICE).unsqueeze(0)
    mask=torch.ones((1,Tm+n),device=DEVICE,dtype=torch.long)
    o=model(input_ids=torch.tensor([ids],device=DEVICE),past_key_values=c,attention_mask=mask,
            position_ids=pos,use_cache=False,output_hidden_states=True,return_dict=True)
    out={}
    # hidden_states[d] = residual stream entering layer d; hidden_states[28] = final block output
    for d in [20,23,25,27,28]:
        h=o.hidden_states[d][0,-1].detach().float().cpu()
        out[d]=unit(h)
    return out

print("\n[5/10] Capturing label-free A ÇAĞRIİZ states...")
A_TRACE=[]
for i,it in enumerate(ITEMS):
    A_TRACE.append(capture_A(A_PACK[i],qA(it["gold"]["entity"])))
    print(f"      [{i+1:02d}/32] {it['gold']['entity']:6s} | H20/H23/H25/H27/H28 captured")
assert len(A_TRACE)==N

# Fixed, non-learned dimensional bridge.
# A hidden state is split into 28 head-sized chunks.
# For K/V fingerprints (4×128 or 8×128), chunks are grouped 7→1.
def h_to_4kv(h):
    x=h.float().reshape(QH,HD)
    return unit(x.reshape(KVH,QH//KVH,HD).mean(dim=1).reshape(-1))

def h_to_8kv(h):
    z=h_to_4kv(h).reshape(KVH,HD)
    return unit(torch.cat([z,z],dim=0).reshape(-1))

def trace_for_family(h,fam):
    if fam in ("KMEAN","VMEAN"):return h_to_4kv(h)
    return h_to_8kv(h)

def ranks_for_geometry(td,bl,fam,idxs):
    rr=[];pred=[];margin=[]
    bank=B_FP[(bl,fam)]
    for i in idxs:
        q=trace_for_family(A_TRACE[i][td],fam)
        sc=[cosine(q,b) for b in bank]
        order=sorted(range(N),key=lambda j:(-sc[j],j))
        rr.append(order.index(i)+1);pred.append(order[0])
        best_wrong=max(sc[j] for j in range(N) if j!=i)
        margin.append(sc[i]-best_wrong)
    return rr,pred,margin

def stats(rr):
    a=np.asarray(rr,dtype=np.float64)
    return {"R1":float(np.mean(a==1)),"R5":float(np.mean(a<=5)),
            "R16":float(np.mean(a<=16)),"MRR":float(np.mean(1.0/a)),
            "MED":float(np.median(a))}

print("\n[6/10] DISCOVERY — choosing one B-native geometry...")
CAND=[]
for td in [20,23,25,27,28]:
    for bl in [20,23,25,27]:
        for fam in ("KMEAN","VMEAN","KVMEAN"):
            rr,pp,mm=ranks_for_geometry(td,bl,fam,DISC);s=stats(rr)
            CAND.append((s["R1"],s["MRR"],s["R5"],float(np.mean(mm)),td,bl,fam,s))
CAND.sort(key=lambda x:(-x[0],-x[1],-x[2],-x[3],x[4],x[5],x[6]))
for z in CAND[:10]:
    print(f"      H{z[4]:02d} → B-L{z[5]:02d}/{z[6]:6s} | R1={z[0]:.4f} R5={z[2]:.4f} MRR={z[1]:.6f} margin={z[3]:+.6f}")
BEST=CAND[0];TD,BL,FAM=BEST[4],BEST[5],BEST[6]
print(f"      FROZEN GEOMETRY: H{TD:02d} → B-L{BL:02d}/{FAM}")

print("\n[7/10] FROZEN EVAL — no reselection...")
DR,DP,DM=ranks_for_geometry(TD,BL,FAM,DISC)
ER,EP,EM=ranks_for_geometry(TD,BL,FAM,EVAL)
DS=stats(DR);ES=stats(ER)
for j,i in enumerate(EVAL):
    print(f"      [{i+1:02d}] {ITEMS[i]['gold']['entity']:6s} | rank={ER[j]:2d} selected=B{EP[j]+1:02d}")
print(f"      DISC R1/R5/MRR : {DS['R1']:.4f} / {DS['R5']:.4f} / {DS['MRR']:.6f}")
print(f"      EVAL R1/R5/MRR : {ES['R1']:.4f} / {ES['R5']:.4f} / {ES['MRR']:.6f}")
print(f"      EVAL POS MARGIN: {float(np.mean(np.asarray(EM)>0)):.4f}")

# Carrier is the A hidden trace itself.
# No token vocabulary, Seal IDs, LM-head rows or gold-key embeddings.
# Scale is fixed to the mean norm of ordinary model input embeddings.
with torch.no_grad():
    EMB_NORM=float(EMB.weight.detach().float().norm(dim=1).mean().cpu())

def native_carrier(h):
    z=h.float()
    z=z/z.norm().clamp_min(1e-8)*EMB_NORM
    # Two continuous slots, both derived only from the same model-native trace.
    # Slot2 is a deterministic cyclic coordinate permutation, not learned.
    z2=torch.roll(z,shifts=HD)
    return torch.stack([z,z2]).to(dtype=EMB.weight.dtype)

def zero_carrier():
    return torch.zeros((2,H),dtype=EMB.weight.dtype)

PREFIX="QUESTION:\nUsing the stored memory, use this internal numeric state"
SUFFIX=" to resolve the requested routing class. Give only the exact routing class.\n\nANSWER:"

@torch.inference_mode()
def read_B_numeric(packet,carrier,max_new=MAX_NEW):
    inst,Tm,P=packet;c=cache_of(inst)
    if carrier.shape!=(2,H):raise RuntimeError(f"Carrier shape mismatch: {tuple(carrier.shape)}")
    pre=enc(PREFIX);suf=enc(SUFFIX)
    pree=EMB(torch.tensor(pre,device=DEVICE,dtype=torch.long)).detach()
    sufe=EMB(torch.tensor(suf,device=DEVICE,dtype=torch.long)).detach()
    car=carrier.to(device=DEVICE,dtype=EMB.weight.dtype)
    x=torch.cat([pree,car,sufe],dim=0).unsqueeze(0)
    n=x.shape[1];pos=torch.arange(P,P+n,device=DEVICE).unsqueeze(0)
    mask=torch.ones((1,Tm+n),device=DEVICE,dtype=torch.long)
    o=model(inputs_embeds=x,past_key_values=c,attention_mask=mask,position_ids=pos,use_cache=True,return_dict=True)
    c=o.past_key_values;out=[];Tm+=n;P+=n;nxt=int(o.logits[0,-1].float().argmax())
    for _ in range(max_new):
        if nxt in EOS:break
        out.append(nxt)
        pos=torch.tensor([[P]],device=DEVICE,dtype=torch.long)
        mask=torch.ones((1,Tm+1),device=DEVICE,dtype=torch.long)
        o=model(input_ids=torch.tensor([[nxt]],device=DEVICE,dtype=torch.long),past_key_values=c,
                attention_mask=mask,position_ids=pos,use_cache=True,return_dict=True)
        c=o.past_key_values;Tm+=1;P+=1;nxt=int(o.logits[0,-1].float().argmax())
    return tok.decode(out,skip_special_tokens=True).strip()

print("\n[8/10] FROZEN EVAL relay — selected / oracle-B / wrong-B / zero...")
R={k:[] for k in ("SELECTED","ORACLE_B","WRONG_B","ZERO")}
RAW={k:[] for k in R}
for j,i in enumerate(EVAL):
    target=ITEMS[i]["gold"]["class"];carrier=native_carrier(A_TRACE[i][TD])
    selected=EP[j];wrong=(i+11)%N
    ys=read_B_numeric(B_PACK[selected],carrier)
    yo=read_B_numeric(B_PACK[i],carrier)
    yw=read_B_numeric(B_PACK[wrong],carrier)
    yz=read_B_numeric(B_PACK[i],zero_carrier())
    for name,y in (("SELECTED",ys),("ORACLE_B",yo),("WRONG_B",yw),("ZERO",yz)):
        R[name].append(int(exact(y,target)));RAW[name].append(y)
    print(f"      [{i+1:02d}] {ITEMS[i]['gold']['entity']:6s} | addr={int(selected==i)} SEL={R['SELECTED'][-1]} ORB={R['ORACLE_B'][-1]} WRB={R['WRONG_B'][-1]} ZERO={R['ZERO'][-1]}")
    if not (R["SELECTED"][-1] and R["ORACLE_B"][-1]):
        print(f"               selected={firstline(ys)!r} oracleB={firstline(yo)!r} wrongB={firstline(yw)!r} zero={firstline(yz)!r}")

OK={k:np.asarray(v,dtype=np.int32) for k,v in R.items()}
ADDR=np.asarray([int(EP[j]==i) for j,i in enumerate(EVAL)],dtype=np.int32)
NADDR=int(ADDR.sum());COND=int(np.sum((ADDR==1)&(OK["SELECTED"]==1)))
print("\n[9/10] Mechanistic localization...")
for k in R:print(f"      {k:10s}: {int(OK[k].sum()):2d}/{len(EVAL)} = {float(OK[k].mean()):.4f}")
print(f"      Address R1                     : {NADDR}/{len(EVAL)} = {float(ADDR.mean()):.4f}")
print(f"      SELECTED | correct address     : {COND}/{NADDR} = {COND/NADDR:.4f}" if NADDR else "      SELECTED | correct address     : N/A")
print(f"      Oracle-B minus zero            : {float(OK['ORACLE_B'].mean()-OK['ZERO'].mean()):+.4f}")
print(f"      Oracle-B minus wrong-B         : {float(OK['ORACLE_B'].mean()-OK['WRONG_B'].mean()):+.4f}")

# Conservative classification.
# This test removes the successful token-derived carrier of TEST516.
# Address and relay are judged separately.
if ES["R1"]>=.50 and OK["ORACLE_B"].mean()>=.75 and NADDR and COND/NADDR>=.70:
    VERDICT="TOKEN_FREE_BELLEKOZ_NATIVE_BRIDGE_OBSERVED"
elif ES["R5"]>=.50 and OK["ORACLE_B"].mean()>=.50:
    VERDICT="TOKEN_FREE_NATIVE_SIGNAL_PRESENT_BUT_NOT_YET_RELIABLE"
elif ES["R1"]<.25 and OK["ORACLE_B"].mean()<.40:
    VERDICT="TOKEN_FREE_NATIVE_ADDRESS_AND_HANDOFF_NOT_YET_OBSERVED"
elif ES["R1"]<.25:
    VERDICT="TOKEN_FREE_HANDOFF_SIGNAL_WITH_ADDRESS_DEFICIT"
elif OK["ORACLE_B"].mean()<.40:
    VERDICT="TOKEN_FREE_ADDRESS_SIGNAL_WITH_HANDOFF_DEFICIT"
else:
    VERDICT="TOKEN_FREE_NATIVE_BRIDGE_INCONCLUSIVE"

print("\n[10/10] Protocol audit...")
print("      Model weights frozen                 : YES")
print("      Forge/runtime state separated         : YES")
print("      Fresh cache each query/read           : YES")
print("      A/B cache concatenation               : NO")
print("      Seal tokenizer used for routing       : NO")
print("      Seal token IDs used for routing       : NO")
print("      LM-head key rows                      : NO")
print("      Gold Seal embedding carrier           : NO")
print("      Intermediate key decode               : NO")
print("      Intermediate text reinsertion         : NO")
print("      Query-time candidate-B forwards       : 0")
print("      Learned router / ANN / DRA / training : NONE")
print("      Discovery                             : FIRST 16 ONLY")
print("      Evaluation reselection                : NO")
print("      Persistent BELLEKÖZ                   : TENSOR")
print("      Address fingerprint source            : B'S OWN K/V TENSORS")
print("      Runtime handoff source                : A'S OWN HIDDEN ÇAĞRIİZ")

print("\n"+"="*180)
print("TEST517 FINAL RESULT — AKBASCORE MAM · TOKEN-FREE BELLEKÖZ-NATIVE ÇAĞRIİZ BRIDGE")
print("="*180)
print("MODEL                              : Qwen/Qwen2.5-7B-Instruct · frozen")
print("PARENT                             : TEST516")
print("PERSISTENCE                        : forge → reset → fresh runtime")
print("SEAL TOKEN ADDRESS                 : REMOVED")
print("LM-HEAD KEY GEOMETRY               : REMOVED")
print("GOLD KEY EMBEDDING                 : REMOVED")
print("A/B CACHE CONCAT                   : NONE")
print("FROZEN GEOMETRY                    :",f"H{TD:02d} → B-L{BL:02d}/{FAM}")
print("-"*180)
print(f"DISC ADDRESS R1                    : {DS['R1']:.4f}")
print(f"DISC ADDRESS R5                    : {DS['R5']:.4f}")
print(f"DISC ADDRESS MRR                   : {DS['MRR']:.6f}")
print(f"EVAL ADDRESS R1                    : {ES['R1']:.4f}")
print(f"EVAL ADDRESS R5                    : {ES['R5']:.4f}")
print(f"EVAL ADDRESS R16                   : {ES['R16']:.4f}")
print(f"EVAL ADDRESS MRR                   : {ES['MRR']:.6f}")
print(f"EVAL ADDRESS POS MARGIN            : {float(np.mean(np.asarray(EM)>0)):.4f}")
print("-"*180)
print(f"SELECTED NATIVE E2E                : {int(OK['SELECTED'].sum()):2d}/{len(EVAL)} = {float(OK['SELECTED'].mean()):.4f}")
print(f"ORACLE-B + NATIVE CARRIER          : {int(OK['ORACLE_B'].sum()):2d}/{len(EVAL)} = {float(OK['ORACLE_B'].mean()):.4f}")
print(f"WRONG-B + NATIVE CARRIER           : {int(OK['WRONG_B'].sum()):2d}/{len(EVAL)} = {float(OK['WRONG_B'].mean()):.4f}")
print(f"ORACLE-B + ZERO CARRIER            : {int(OK['ZERO'].sum()):2d}/{len(EVAL)} = {float(OK['ZERO'].mean()):.4f}")
print(f"SELECTED | CORRECT ADDRESS         : {COND}/{NADDR} = {COND/NADDR:.4f}" if NADDR else "SELECTED | CORRECT ADDRESS         : N/A")
print("-"*180)
print("TEST503 LOCK                       :",TEST503)
print("TEST504 LOCK                       :",TEST504)
print("TEST513 LOCK                       :",TEST513)
print("TEST514 LOCK                       :",TEST514)
print("TEST515 LOCK                       :",TEST515)
print("TEST516 LOCK                       :",TEST516)
print("TEST517 LOCK                       :",LOCK_SHA)
print(f"TOTAL TEST TIME                    : {time.perf_counter()-T0:.2f}s")
print("VERDICT                            :",VERDICT)
print("="*180)

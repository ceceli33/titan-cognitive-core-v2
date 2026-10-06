# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST515
# LATENT A→B TENSOR RELAY
# FROZEN TEST514 ADDRESS → SELECT B NUMERICALLY → ATTACH B TENSOR → FINAL ANSWER
#
# RUNTIME:
# QUERY + A_TENSOR
#   → L27 ÇAĞRIİZ
#   → frozen SOFT2 numeric address
#   → select B_TENSOR
#   → [A_TENSOR || B_TENSOR]
#   → final answer
#
# ABSOLUTELY NO INTERMEDIATE HUMAN-LANGUAGE KEY:
# NO Seal generation/decode.
# NO "FEN/KOR/..." reinsertion.
# NO textual relay.
# NO B candidate model forward before selection.
# NO learned router / ANN / DRA / training.
#
# Human-readable source exists ONLY OFFLINE to forge controlled cartridges
# and evaluate ground truth. Runtime cartridges are K/V tensors only.
#
# PRIMARY:
# 32-bank end-to-end address-selected tensor relay.
#
# CONTROLS:
# ORACLE_B   : same tensor relay, gold B selected.
# WRONG_B    : deterministic wrong B tensor.
# NO_B       : A tensor only.
#
# TEST514 L27 + SOFT2 IS FROZEN. NO DISCOVERY / RESELECTION.

import os,sys,subprocess,importlib.util,random,time,hashlib,json,re
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache

TEST="515";SEED=515;BASE_SEED=501;MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
N=32;TRACE_DEPTH=27;DEVICE=torch.device("cuda");SEP="\n\n"
TEST508="47512c7860268517670ca5b45b6e76fe1582362dd38e447d164f33c756a1a2ca"
TEST509="4b5b78841ebf0d9b458306136e122fd3108b33bdfd89924585f891f748eca0d8"
TEST510="cf3aaf955c0cd5019b1b951d185f290528286df82b23da62aaa30657e6bad5ea"
TEST511="31558a2b86b1a1e86cb4f729d7c956781fe9269ff20bb2557a92b3300270e35c"
TEST512="2c9576ce56491489be55f34a1a6d331a29bc4fd8080084693e92d2dab4a43ac2"
TEST513="2b28137b12a26517261d0d8827c378f7e9e455bd5f4ccb12af47277ed8de01aa"
TEST514="61e83ffe4b57cbafa0f4ef8a3fe516a153818fa43e87176941fd01ec8ad24146"

ENTITY_BANK=[
("Aldren","Boreal","Cyrene"),("Darian","Elara","Faron"),("Galen","Hesper","Ilyra"),("Joren","Kaelis","Lorin"),
("Maren","Neris","Orlan"),("Perrin","Quorin","Ralen"),("Saren","Taris","Ulric"),("Valen","Weyra","Xeran"),
("Yorin","Zaren","Avel"),("Brann","Ceris","Dalen"),("Eris","Felis","Gorin"),("Halen","Ivar","Jaris"),
("Koren","Leris","Miran"),("Nolan","Orel","Palis"),("Riven","Solis","Teren"),("Urian","Varen","Wilis"),
("Xorin","Yalen","Zorin"),("Arven","Belis","Coren"),("Derin","Evan","Feris"),("Garin","Heron","Ilven"),
("Jarin","Kelis","Laven"),("Moris","Naven","Orris"),("Parin","Rovis","Selan"),("Torin","Ulen","Veris"),
("Waren","Xelis","Yaris"),("Zelis","Aren","Borin"),("Caren","Dorin","Elen"),("Faren","Gelis","Harin"),
("Iren","Joris","Kalen"),("Laris","Meren","Noren"),("Oris","Peren","Ravin"),("Serin","Toren","Ulis")
]
SEAL_BANK=["KOR","VEL","DAR","MIR","SEN","ROL","FEN","JAL","WEX","NUR","BAV","CIR","DEM","GOS","HIL","KET","LOR","MEV","PIR","RUK","SAV","TOL","VIR","YEK","ZAM","BIR","CAV","DOL","FER","GUL","HAR","JEM","KIR","LEV","MOR","NEX","PEL","RAS","SUL","TIR","VAN","YOR","ZEL","BOS","CER","DIN","FAL","GER","HOV","JUN","KEL","LUM","NAV","POR","REV","SIM","TUR","WAL","XEN","YIL","ZOR","BEK","COR","DUR","EKS","FIR","GAN","HEL","IVO","JOR","KAS","LIN","MUR","NOR","OVA","PAR","RIN","SOL","TEV","URB","VAR","WEN","XAL","YUN","ZEN","BOR","CEN","DAN","ELV","FOR","GIR","HAN","IRV","JEN","KOL","MAR"]
CLASS_BANK=["TAK","BEX","QAA","RAV","SOD","PEK","NIV","GOR","HAX","JUR","KEM","VOL","DAX","QAB","MON","SAL","TEK","WIR","ZUN","COV","HEM","JAX","LIV","QAC","PAK","RUM","SEV","TIX","VOR","YAM","ZEK","BOL","QAD","DOV","FEX","GAM","HUR","JIN","KAV","LER","MEX","QAE","PIV","ROK","SUM","TAL","VEK","WON","XIR","YAV","ZOL","BAR","CIX","QAF","FOV","GEL","HIN","JOV","KUR","QAG","MAV","QAH","PUL","RIM","QAI","TOX","VIL","WER","XAN","YER","ZIM","BUN","CAL","DOR","EVI","FAR","GUN","HES","ILM","JER","KON","LAR","QAJ","NOL","OVI","PER","RUS","SIN","QAK","VEX","QAL","QAM","YUL","ZAR","BEL","CUM"]

assert len(ENTITY_BANK)==N and len(SEAL_BANK)>=3*N and len(CLASS_BANK)>=3*N
assert len(set(SEAL_BANK))==len(SEAL_BANK) and len(set(CLASS_BANK))==len(CLASS_BANK)
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
torch.set_grad_enabled(False)
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

def reorder(rows,gold,pos,seed):
    g=rows[gold];rest=[x for j,x in enumerate(rows) if j!=gold]
    r=random.Random(seed);r.shuffle(rest);out=rest[:];out.insert(pos,g);return out

def make_base():
    rng=random.Random(BASE_SEED);ss=SEAL_BANK.copy();cc=CLASS_BANK.copy();rng.shuffle(ss);rng.shuffle(cc);out=[]
    for i in range(N):
        ents=list(ENTITY_BANK[i]);g=i%3;s=ss[3*i:3*i+3];c=cc[3*i:3*i+3]
        ar=[f"Instrument {ents[j]} carries seal {s[j]}." for j in range(3)]
        br=[f"Seal {s[j]} corresponds to routing class {c[j]}." for j in range(3)]
        A=" ".join(reorder(ar,g,i%3,BASE_SEED+i*101+17))
        B=" ".join(reorder(br,g,(i+1)%3,BASE_SEED+i*101+34))
        out.append({"id":i+1,"A":A,"B":B,"g":{"entity":ents[g],"seal":s[g],"class":c[g]}})
    return out
ITEMS=make_base()
USED=sorted(x["g"]["seal"] for x in ITEMS)
if len(set(USED))!=N:raise RuntimeError("Gold Seal uniqueness failure.")

LOCK={"test":TEST,"parent514":TEST514,"model":MODEL_ID,"seed":SEED,"base_seed":BASE_SEED,
"trace_depth":TRACE_DEPTH,"address":"SOFT2_FROZEN_FROM_TEST513_514",
"runtime":"QUERY+A_TENSOR→NUMERIC_ADDRESS→SELECT_B_TENSOR→A+B_TENSOR→FINAL",
"bank":32,"arms":["SELECTED_B","ORACLE_B","WRONG_B","NO_B"],
"intermediate_seal_generation":"NONE","intermediate_text_reinsertion":"NONE",
"query_time_candidate_forwards":0,"learned_router":"NONE","ann":"NONE","dra":"NONE","training":"NONE",
"discovery":"NONE","reselection":"NONE","runtime_cartridges":"TENSORS_ONLY"}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()

print("="*174)
print("TEST515 — AKBASCORE MAM · LATENT A→B TENSOR RELAY")
print("FROZEN L27+SOFT2 → NUMERIC B SELECTION → DIRECT B TENSOR ATTACH → FINAL ANSWER")
print("="*174)
print("LOCK SHA:",LOCK_SHA)
print("PARENT TEST514:",TEST514)
print("RUNTIME CARTRIDGES: TENSORS ONLY | INTERMEDIATE TEXT: NONE")
print("DISCOVERY: NONE | RESELECTION: NONE")
T0=time.perf_counter()

print("\n[1/10] Loading frozen Qwen...")
_tv=tuple(int("".join(c for c in x if c.isdigit())or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
enc=lambda s:tok(s,add_special_tokens=False).input_ids
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=H//QH
if (NL,H,QH,KVH,HD)!=(28,3584,28,4,128):raise RuntimeError("Architecture mismatch.")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | {NL}L H={H}")

@torch.inference_mode()
def kv_from_ids(ids):
    o=model(input_ids=torch.tensor([ids],device=DEVICE),use_cache=False,output_hidden_states=True,return_dict=True);out=[]
    for L,layer in enumerate(model.model.layers):
        h=o.hidden_states[L][0].to(layer.input_layernorm.weight.device,layer.input_layernorm.weight.dtype);hn=layer.input_layernorm(h)
        k=layer.self_attn.k_proj(hn.to(layer.self_attn.k_proj.weight.device,layer.self_attn.k_proj.weight.dtype))
        v=layer.self_attn.v_proj(hn.to(layer.self_attn.v_proj.weight.device,layer.self_attn.v_proj.weight.dtype))
        out.append((k.view(-1,KVH,HD).transpose(0,1).contiguous(),v.view(-1,KVH,HD).transpose(0,1).contiguous()))
    return out
def forge(s):return kv_from_ids([PAD]+enc(s+SEP))

def rope_k(k,pos,L):
    layer=model.model.layers[L];rot=layer.self_attn.rotary_emb if hasattr(layer.self_attn,"rotary_emb") else model.model.rotary_emb
    d=torch.zeros((1,k.shape[0],len(pos),HD),device=k.device,dtype=k.dtype);p=torch.tensor([pos],device=k.device)
    try:c,s=rot(d,p)
    except TypeError:c,s=rot(d,position_ids=p)
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    _,kr=apply_rotary_pos_emb(d,k.unsqueeze(0),c,s);return kr[0]

def install_raw(raw):
    T=raw[0][0].shape[1];out=[]
    for L,(k,v) in enumerate(raw):out.append((rope_k(k,list(range(T)),L),v))
    return out,T,T

def install_pair(rawA,rawB):
    TA=rawA[0][0].shape[1];TB=rawB[0][0].shape[1];out=[]
    for L in range(NL):
        ka,va=rawA[L];kb,vb=rawB[L]
        ka=rope_k(ka,list(range(TA)),L);kb=rope_k(kb,list(range(TA,TA+TB)),L)
        out.append((torch.cat([ka,kb],dim=1),torch.cat([va,vb],dim=1)))
    return out,TA+TB,TA+TB

def cache_of(inst):
    try:c=DynamicCache(config=cfg)
    except TypeError:c=DynamicCache()
    for L,(k,v) in enumerate(inst):c.update(k.unsqueeze(0),v.unsqueeze(0),L)
    return c

def unit(x):
    x=x.float().reshape(-1);return x/x.norm().clamp_min(1e-8)

print("\n[2/10] Forging A/B sources into runtime tensor cartridges...")
A_RAW=[];B_RAW=[]
for i,it in enumerate(ITEMS):
    A_RAW.append(forge(it["A"]));B_RAW.append(forge(it["B"]))
    if i<5:print(f"      ITEM{i+1:02d} | A/B → frozen K/V tensors")
print("      Source strings are no longer used by runtime.")

print("\n[3/10] Freezing TEST514 numeric address geometry...")
TOK={s:enc(s) for s in USED}
if any(len(x)<1 or len(x)>2 for x in TOK.values()):raise RuntimeError("Tokenizer geometry changed.")
W=model.lm_head.weight.detach().float().cpu();Wn=W/W.norm(dim=1,keepdim=True).clamp_min(1e-8);ZERO=torch.zeros(H)
FIRST=sorted(set(x[0] for x in TOK.values()));SECOND=sorted(set(x[1] for x in TOK.values() if len(x)==2))
FM=Wn[FIRST];SM=Wn[SECOND];FG={}
for s,x in TOK.items():FG.setdefault(x[0],[]).append(s)
ADDR=[]
for it in ITEMS:
    ids=TOK[it["g"]["seal"]];a1=Wn[ids[0]];a2=Wn[ids[1]] if len(ids)==2 else ZERO;lb=1. if len(ids)==2 else 0.
    ADDR.append(unit(torch.cat([a1,a2,torch.tensor([lb])])))

def soft2(logits):
    fl=torch.tensor([float(logits[t]) for t in FIRST]);pf=torch.softmax(fl,0);q1=unit(pf@FM)
    if SECOND:
        sl=torch.tensor([float(logits[t]) for t in SECOND]);p2=torch.softmax(sl,0);q2=p2@SM
        q2=q2-torch.dot(q2,unit(q1))*unit(q1);q2=unit(q2) if q2.norm()>1e-8 else ZERO
    else:q2=ZERO
    two=sum(float(pf[j]) for j,t in enumerate(FIRST) if any(len(TOK[s])==2 for s in FG[t]))
    return unit(torch.cat([q1,q2,torch.tensor([two])]))
print("      L27 + SOFT2 frozen | numeric address dim:",2*H+1)

def q_addr(e):return f"What seal does instrument {e} carry? Give only the exact seal."
def q_final(e):return f"What routing class is associated with instrument {e}? Give only the exact routing class."

@torch.inference_mode()
def address_logits(rawA,e):
    inst,Tm,P=install_raw(rawA);c=cache_of(inst);ids=enc("QUESTION:\n"+q_addr(e)+"\n\nANSWER:");n=len(ids)
    pos=torch.arange(P,P+n,device=DEVICE).unsqueeze(0);mask=torch.ones((1,Tm+n),device=DEVICE,dtype=torch.long)
    o=model(input_ids=torch.tensor([ids],device=DEVICE),past_key_values=c,attention_mask=mask,position_ids=pos,
            use_cache=False,output_hidden_states=True,return_dict=True)
    h=o.hidden_states[TRACE_DEPTH][0,-1];fn=model.model.norm
    return model.lm_head(fn(h.to(fn.weight.device,fn.weight.dtype)).to(model.lm_head.weight.device,model.lm_head.weight.dtype)).float().cpu()

def select_b(rawA,e):
    q=soft2(address_logits(rawA,e));sc=[float(torch.dot(q,a)) for a in ADDR]
    return max(range(N),key=lambda j:(sc[j],-j)),sc

print("\n[4/10] Frozen numeric B selection...")
SELECTED=[];RANKS=[]
for i,it in enumerate(ITEMS):
    p,sc=select_b(A_RAW[i],it["g"]["entity"]);order=sorted(range(N),key=lambda j:(-sc[j],j));r=order.index(i)+1
    SELECTED.append(p);RANKS.append(r)
    print(f"      [{i+1:02d}] numeric-address → B{p+1:02d} | gold=B{i+1:02d} | rank={r:2d}")
ADDR_R1=float(np.mean(np.asarray(RANKS)==1));ADDR_R5=float(np.mean(np.asarray(RANKS)<=5))
print(f"      ADDRESS R1={ADDR_R1:.4f} R5={ADDR_R5:.4f}")

@torch.inference_mode()
def generate_from_inst(inst,Tm,P,q,max_new=12):
    cache=cache_of(inst);ids=enc("QUESTION:\n"+q+"\n\nANSWER:");seq=[];cur=torch.tensor([ids],device=DEVICE)
    pos=torch.arange(P,P+len(ids),device=DEVICE).unsqueeze(0);mask=torch.ones((1,Tm+len(ids)),device=DEVICE,dtype=torch.long)
    o=model(input_ids=cur,past_key_values=cache,attention_mask=mask,position_ids=pos,use_cache=True,return_dict=True)
    nxt=int(o.logits[0,-1].argmax());seq.append(nxt);past=o.past_key_values
    for k in range(max_new-1):
        if nxt==tok.eos_token_id:break
        total=Tm+len(ids)+len(seq)
        o=model(input_ids=torch.tensor([[nxt]],device=DEVICE),past_key_values=past,
                attention_mask=torch.ones((1,total),device=DEVICE,dtype=torch.long),
                position_ids=torch.tensor([[P+len(ids)+len(seq)-1]],device=DEVICE),
                use_cache=True,return_dict=True)
        nxt=int(o.logits[0,-1].argmax());seq.append(nxt);past=o.past_key_values
    return tok.decode(seq,skip_special_tokens=True).strip()

def normtxt(x):return re.sub(r"[^A-Z0-9]","",x.upper())
def hit(out,gold):return normtxt(out).startswith(normtxt(gold))

print("\n[5/10] Building tensor-only relay caches...")
PAIR_SELECTED=[];PAIR_ORACLE=[];PAIR_WRONG=[];A_ONLY=[]
WRONG=[(i+11)%N for i in range(N)]
for i in range(N):
    PAIR_SELECTED.append(install_pair(A_RAW[i],B_RAW[SELECTED[i]]))
    PAIR_ORACLE.append(install_pair(A_RAW[i],B_RAW[i]))
    PAIR_WRONG.append(install_pair(A_RAW[i],B_RAW[WRONG[i]]))
    A_ONLY.append(install_raw(A_RAW[i]))
print("      A+B concatenation performed directly in K/V tensor space.")
print("      Intermediate Seal text generated/reinserted: 0")

print("\n[6/10] SELECTED-B end-to-end tensor relay...")
OUT_SEL=[];H_SEL=[]
for i,it in enumerate(ITEMS):
    out=generate_from_inst(*PAIR_SELECTED[i],q_final(it["g"]["entity"]))
    h=hit(out,it["g"]["class"]);OUT_SEL.append(out);H_SEL.append(h)
    print(f"      [{i+1:02d}] B{SELECTED[i]+1:02d} | target={it['g']['class']} | out={out!r} | {'PASS' if h else 'FAIL'}")

print("\n[7/10] ORACLE-B tensor relay ceiling...")
OUT_OR=[];H_OR=[]
for i,it in enumerate(ITEMS):
    out=generate_from_inst(*PAIR_ORACLE[i],q_final(it["g"]["entity"]))
    h=hit(out,it["g"]["class"]);OUT_OR.append(out);H_OR.append(h)
    print(f"      [{i+1:02d}] target={it['g']['class']} | out={out!r} | {'PASS' if h else 'FAIL'}")

print("\n[8/10] WRONG-B / NO-B controls...")
H_WR=[];H_NO=[]
for i,it in enumerate(ITEMS):
    ow=generate_from_inst(*PAIR_WRONG[i],q_final(it["g"]["entity"]))
    on=generate_from_inst(*A_ONLY[i],q_final(it["g"]["entity"]))
    hw=hit(ow,it["g"]["class"]);hn=hit(on,it["g"]["class"]);H_WR.append(hw);H_NO.append(hn)
    print(f"      [{i+1:02d}] WRONG={ow!r}:{int(hw)} | NO-B={on!r}:{int(hn)}")

SEL=float(np.mean(H_SEL));ORACLE=float(np.mean(H_OR));WR=float(np.mean(H_WR));NO=float(np.mean(H_NO))
COND=float(np.mean([H_SEL[i] for i in range(N) if SELECTED[i]==i])) if any(SELECTED[i]==i for i in range(N)) else 0.
EXPECTED=ADDR_R1*ORACLE

print("\n[9/10] Relay decomposition...")
print(f"      Numeric address R1             : {ADDR_R1:.4f}")
print(f"      Oracle B tensor ceiling        : {ORACLE:.4f}")
print(f"      Selected B end-to-end          : {SEL:.4f}")
print(f"      Conditional on correct address : {COND:.4f}")
print(f"      Address×oracle expectation     : {EXPECTED:.4f}")
print(f"      Wrong-B control                : {WR:.4f}")
print(f"      No-B control                   : {NO:.4f}")

if SEL>=.55 and COND>=.85 and ORACLE>=.85 and WR<=.15 and NO<=.15:
    VERDICT="TEXT_FREE_A_TO_B_TENSOR_RELAY_OBSERVED"
elif SEL>=.40 and COND>=.75 and ORACLE>=.80 and SEL-WR>=.25 and SEL-NO>=.25:
    VERDICT="TEXT_FREE_A_TO_B_TENSOR_RELAY_PARTIALLY_OBSERVED"
else:
    VERDICT="TEXT_FREE_A_TO_B_TENSOR_RELAY_NOT_YET_ESTABLISHED"

print("\n[10/10] Runtime audit...")
print("      Runtime A                        : K/V TENSOR")
print("      A→B address                     : NUMERIC VECTOR")
print("      Runtime B                        : K/V TENSOR")
print("      Intermediate Seal generation    : NONE")
print("      Intermediate Seal decode        : NONE")
print("      Intermediate text reinsertion   : NONE")
print("      Candidate B model forwards      : NONE")
print("      B selected before B read        : YES")
print("      A+B composition                 : DIRECT K/V TENSOR CONCAT")
print("      Learned router / ANN / DRA      : NONE")
print("      Discovery / reselection         : NONE")

print("\n"+"="*174)
print("TEST515 FINAL RESULT — AKBASCORE MAM · TEXT-FREE A→B TENSOR RELAY")
print("="*174)
print("MODEL                           : Qwen/Qwen2.5-7B-Instruct · frozen")
print("PARENT                          : TEST514")
print("TRACE / ADDRESS                 : L27 / SOFT2 · frozen")
print("BANK                            : 32 independent B BELLEKÖZ")
print("RUNTIME CARTRIDGES              : TENSORS ONLY")
print("INTERMEDIATE HUMAN-LANGUAGE KEY : NONE")
print("-"*174)
print(f"NUMERIC ADDRESS R1              : {ADDR_R1:.4f}")
print(f"NUMERIC ADDRESS R5              : {ADDR_R5:.4f}")
print(f"ORACLE-B TENSOR RELAY           : {ORACLE:.4f}")
print(f"SELECTED-B END-TO-END           : {SEL:.4f}")
print(f"CORRECT-ADDRESS CONDITIONAL     : {COND:.4f}")
print(f"WRONG-B CONTROL                 : {WR:.4f}")
print(f"NO-B CONTROL                    : {NO:.4f}")
print(f"ADDRESS×ORACLE EXPECTATION      : {EXPECTED:.4f}")
print("-"*174)
print("TEST508 LOCK SHA                :",TEST508)
print("TEST509 LOCK SHA                :",TEST509)
print("TEST510 LOCK SHA                :",TEST510)
print("TEST511 LOCK SHA                :",TEST511)
print("TEST512 LOCK SHA                :",TEST512)
print("TEST513 LOCK SHA                :",TEST513)
print("TEST514 LOCK SHA                :",TEST514)
print("TEST515 LOCK SHA                :",LOCK_SHA)
print(f"TOTAL TEST TIME                 : {time.perf_counter()-T0:.2f}s")
print("VERDICT                         :",VERDICT)
print("="*174)

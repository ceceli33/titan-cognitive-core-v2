# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST514
# FROZEN ÇAĞRIİZ ADDRESS CAUSALITY
# TEST513 L27+SOFT2 FROZEN · CORRECT-A / NO-A / WRONG-A / COUNTERFACTUAL-A
#
# PRIMARY QUESTION
# ----------------
# Is the TEST513 latent B-address signal CAUSED by the contents of A BELLEKÖZ?
#
# FROZEN FROM TEST513
# -------------------
# Trace depth : L27
# Address     : SOFT2
# Dataset     : BASE_SEED=501
# B bank      : same 32 real BELLEKÖZ
# Selection   : NONE
#
# ARMS
# ----
# CORRECT_A:
#   Original A BELLEKÖZ. Must address original B.
#
# NO_A:
#   No memory cartridge. Same entity question.
#   Must lose the original-address advantage.
#
# WRONG_A:
#   A cartridge from another item. Same entity question.
#   Must lose the original-address advantage.
#
# COUNTERFACTUAL_A:
#   Same A record structure and same queried entity, but the queried entity's
#   Seal is replaced with another existing Seal from the SAME 32-address bank.
#   The address must move to that new target.
#
# STRICT
# ------
# NO discovery.
# NO channel selection.
# NO threshold tuning.
# NO answer token generation.
# NO teacher forcing.
# NO second autoregressive forward.
# NO intermediate Seal decode.
# NO text reinsertion.
# NO query-time B candidate model forward.
# NO learned router.
# NO ANN.
# NO DRA.
# NO training.
#
# NOTE
# ----
# Human-readable Entity/Seal strings exist only in the controlled source
# construction and ground-truth evaluator. Runtime BELLEKÖZ is K/V tensor
# memory; routing uses numeric LM-head-derived vectors only.

import os,sys,subprocess,importlib.util,random,time,hashlib,json,re
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache

TEST="514";SEED=514;BASE_SEED=501;MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
N=32;TRACE_DEPTH=27;DEVICE=torch.device("cuda");FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
TEST508="47512c7860268517670ca5b45b6e76fe1582362dd38e447d164f33c756a1a2ca"
TEST509="4b5b78841ebf0d9b458306136e122fd3108b33bdfd89924585f891f748eca0d8"
TEST510="cf3aaf955c0cd5019b1b951d185f290528286df82b23da62aaa30657e6bad5ea"
TEST511="31558a2b86b1a1e86cb4f729d7c956781fe9269ff20bb2557a92b3300270e35c"
TEST512="2c9576ce56491489be55f34a1a6d331a29bc4fd8080084693e92d2dab4a43ac2"
TEST513="2b28137b12a26517261d0d8827c378f7e9e455bd5f4ccb12af47277ed8de01aa"

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

assert len(ENTITY_BANK)==N and len(SEAL_BANK)>=N*3 and len(CLASS_BANK)>=N*3
assert len(set(SEAL_BANK))==len(SEAL_BANK) and len(set(CLASS_BANK))==len(CLASS_BANK)
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
torch.set_grad_enabled(False)
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

def reorder(rows,gold,pos,seed):
    g=rows[gold];rest=[x for j,x in enumerate(rows) if j!=gold]
    rr=random.Random(seed);rr.shuffle(rest);out=rest[:];out.insert(pos,g);return out

def make_base():
    rng=random.Random(BASE_SEED);seals=SEAL_BANK.copy();classes=CLASS_BANK.copy()
    rng.shuffle(seals);rng.shuffle(classes);items=[]
    for i in range(N):
        ents=list(ENTITY_BANK[i]);g=i%3;ss=seals[3*i:3*i+3];cc=classes[3*i:3*i+3]
        r1=[f"Instrument {ents[j]} carries seal {ss[j]}." for j in range(3)]
        r2=[f"Seal {ss[j]} corresponds to routing class {cc[j]}." for j in range(3)]
        A=" ".join(reorder(r1,g,i%3,BASE_SEED+i*101+17))
        B=" ".join(reorder(r2,g,(i+1)%3,BASE_SEED+i*101+34))
        items.append({"id":i+1,"ents":ents,"g":g,"ss":ss,"cc":cc,"A":A,"B":B,
                      "gold":{"entity":ents[g],"seal":ss[g],"class":cc[g]}})
    return items
ITEMS=make_base()
USED=sorted({x["gold"]["seal"] for x in ITEMS})
if len(USED)!=N:raise RuntimeError("Seal uniqueness failure.")
SEAL_TO_ITEM={x["gold"]["seal"]:i for i,x in enumerate(ITEMS)}

# Counterfactual permutation: deterministic derangement over existing target Seals.
# Each item is rewritten so queried entity points to next item's original Seal.
CF_TARGET=[ITEMS[(i+7)%N]["gold"]["seal"] for i in range(N)]
if any(CF_TARGET[i]==ITEMS[i]["gold"]["seal"] for i in range(N)):raise RuntimeError("CF derangement failure.")

def make_cf_A(i):
    it=ITEMS[i];g=it["g"];rows=[]
    for j,e in enumerate(it["ents"]):
        s=CF_TARGET[i] if j==g else it["ss"][j]
        rows.append(f"Instrument {e} carries seal {s}.")
    return " ".join(reorder(rows,g,i%3,BASE_SEED+i*101+17))

CF_A=[make_cf_A(i) for i in range(N)]
WRONG_IDX=[(i+11)%N for i in range(N)]
if any(WRONG_IDX[i]==i for i in range(N)):raise RuntimeError("Wrong-A derangement failure.")

LOCK={
"test":TEST,"parent513":TEST513,"model":MODEL_ID,"seed":SEED,"base_seed":BASE_SEED,
"trace_depth":27,"address":"SOFT2_FROZEN_FROM_TEST513","items":ITEMS,
"arms":["CORRECT_A","NO_A","WRONG_A","COUNTERFACTUAL_A"],
"wrong_A_shift":11,"counterfactual_target_shift":7,
"discovery":"NONE","selection":"NONE","query_forwards_per_arm_item":1,
"answer_generation":"OFF","teacher_forcing":"OFF","second_autoregressive_forward":"OFF",
"intermediate_decode":"NONE","text_reinsertion":"NONE","query_time_B_forwards":0,
"learned_router":"OFF","ann":"OFF","dra":"OFF","training":"OFF",
"runtime_address":"NUMERIC_VECTOR_ONLY"
}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

print("="*174)
print("TEST514 — AKBASCORE MAM · FROZEN ÇAĞRIİZ ADDRESS CAUSALITY")
print("TEST513 L27+SOFT2 FROZEN · CORRECT-A / NO-A / WRONG-A / COUNTERFACTUAL-A")
print("="*174)
print("LOCK SHA:",LOCK_SHA)
print("PARENT TEST513:",TEST513)
print("L27 + SOFT2: FROZEN | DISCOVERY: NONE | RESELECTION: NONE")
print("RUNTIME ADDRESS: NUMERIC VECTOR ONLY")
T0=time.perf_counter()

print("\n[1/9] Loading frozen Qwen...")
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
        h=o.hidden_states[L][0].to(layer.input_layernorm.weight.device,layer.input_layernorm.weight.dtype)
        hn=layer.input_layernorm(h)
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

def install(raw):
    T=raw[0][0].shape[1];out=[]
    for L,(k,v) in enumerate(raw):out.append((rope_k(k,list(range(T)),L),v))
    return out,T,T

def cache_of(x):
    try:c=DynamicCache(config=cfg)
    except TypeError:c=DynamicCache()
    for L,(k,v) in enumerate(x):c.update(k.unsqueeze(0),v.unsqueeze(0),L)
    return c

def unit(x):
    x=x.float().reshape(-1);return x/x.norm().clamp_min(1e-8)

def question(e):return f"What seal does instrument {e} carry? Give only the exact seal."

print("\n[2/9] Freezing TEST513 numeric address geometry...")
TOK={s:enc(s) for s in USED}
if any(len(x)<1 or len(x)>2 for x in TOK.values()):raise RuntimeError("Tokenizer geometry changed.")
W=model.lm_head.weight.detach().float().cpu()
Wn=W/W.norm(dim=1,keepdim=True).clamp_min(1e-8)
ZERO=torch.zeros(H)
FIRST=sorted(set(x[0] for x in TOK.values()))
SECOND=sorted(set(x[1] for x in TOK.values() if len(x)==2))
FM=Wn[FIRST];SM=Wn[SECOND]
FG={}
for s,x in TOK.items():FG.setdefault(x[0],[]).append(s)

ADDR=[]
for it in ITEMS:
    ids=TOK[it["gold"]["seal"]];a1=Wn[ids[0]]
    a2=Wn[ids[1]] if len(ids)==2 else ZERO;lb=1.0 if len(ids)==2 else 0.0
    ADDR.append(unit(torch.cat([a1,a2,torch.tensor([lb])])))
print("      SOFT2 address geometry frozen. Dimension:",2*H+1)

def soft2(logits):
    fl=torch.tensor([float(logits[t]) for t in FIRST]);pf=torch.softmax(fl,0)
    q1=unit(pf@FM)
    if SECOND:
        sl=torch.tensor([float(logits[t]) for t in SECOND]);p2=torch.softmax(sl,0);q2=p2@SM
        q2=q2-torch.dot(q2,unit(q1))*unit(q1)
        q2=unit(q2) if q2.norm()>1e-8 else ZERO
    else:q2=ZERO
    two=0.0
    for j,t in enumerate(FIRST):
        if any(len(TOK[s])==2 for s in FG[t]):two+=float(pf[j])
    return unit(torch.cat([unit(q1),q2,torch.tensor([two])]))

print("\n[3/9] Precomputing real B BELLEKÖZ bank...")
B=[]
for i,it in enumerate(ITEMS):
    B.append(install(forge(it["B"])))
    if i<5:print(f"      B{i+1:02d} | tensor packet tokens={B[-1][1]}")
print("      32/32 real B cartridges forged.")

print("\n[4/9] Precomputing CORRECT / WRONG / COUNTERFACTUAL A tensor cartridges...")
AC=[];AW=[];AF=[]
for i in range(N):
    AC.append(install(forge(ITEMS[i]["A"])))
    AW.append(install(forge(ITEMS[WRONG_IDX[i]]["A"])))
    AF.append(install(forge(CF_A[i])))
print("      CORRECT-A        : 32")
print("      WRONG-A          : 32")
print("      COUNTERFACTUAL-A : 32")
print("      Runtime inputs after forge are tensors.")

@torch.inference_mode()
def logits_mem(packet,q):
    inst,Tm,P=packet;c=cache_of(inst);ids=enc(FMT.format(q=q));n=len(ids)
    pos=torch.arange(P,P+n,device=DEVICE).unsqueeze(0);mask=torch.ones((1,Tm+n),device=DEVICE,dtype=torch.long)
    o=model(input_ids=torch.tensor([ids],device=DEVICE),past_key_values=c,attention_mask=mask,position_ids=pos,
            use_cache=False,output_hidden_states=True,return_dict=True)
    h=o.hidden_states[TRACE_DEPTH][0,-1];fn=model.model.norm
    return model.lm_head(fn(h.to(fn.weight.device,fn.weight.dtype)).to(model.lm_head.weight.device,model.lm_head.weight.dtype)).float().cpu()

@torch.inference_mode()
def logits_nomem(q):
    ids=enc(FMT.format(q=q))
    o=model(input_ids=torch.tensor([ids],device=DEVICE),use_cache=False,output_hidden_states=True,return_dict=True)
    h=o.hidden_states[TRACE_DEPTH][0,-1];fn=model.model.norm
    return model.lm_head(fn(h.to(fn.weight.device,fn.weight.dtype)).to(model.lm_head.weight.device,model.lm_head.weight.dtype)).float().cpu()

def scores(logits):
    q=soft2(logits);return [float(torch.dot(q,a)) for a in ADDR]

def rank(sc,target):
    order=sorted(range(N),key=lambda j:(-sc[j],j));return order.index(target)+1,order[0]

def stat(rr):
    a=np.asarray(rr,float)
    return {"R1":float(np.mean(a==1)),"R5":float(np.mean(a<=5)),"R16":float(np.mean(a<=16)),
            "MRR":float(np.mean(1/a)),"MED":float(np.median(a))}

print("\n[5/9] Running four frozen causal arms...")
RES={k:[] for k in ["CORRECT_A","NO_A","WRONG_A","COUNTERFACTUAL_A"]}
PRED={k:[] for k in RES};SCORES={k:[] for k in RES}
for i,it in enumerate(ITEMS):
    q=question(it["gold"]["entity"])
    lc=logits_mem(AC[i],q);ln=logits_nomem(q);lw=logits_mem(AW[i],q);lf=logits_mem(AF[i],q)
    for arm,l in [("CORRECT_A",lc),("NO_A",ln),("WRONG_A",lw),("COUNTERFACTUAL_A",lf)]:
        sc=scores(l);target=i if arm!="COUNTERFACTUAL_A" else SEAL_TO_ITEM[CF_TARGET[i]]
        r,p=rank(sc,target);RES[arm].append(r);PRED[arm].append(p);SCORES[arm].append(sc)
    if i<8:
        print(f"      [{i+1:02d}] {it['gold']['entity']:6s} | correct r={RES['CORRECT_A'][-1]:2d} | noA r={RES['NO_A'][-1]:2d} | wrongA r={RES['WRONG_A'][-1]:2d} | CF→{CF_TARGET[i]} r={RES['COUNTERFACTUAL_A'][-1]:2d}")

print("\n[6/9] Arm statistics...")
ST={k:stat(v) for k,v in RES.items()}
for k in RES:
    s=ST[k];print(f"      {k:16s} | R1={s['R1']:.4f} R5={s['R5']:.4f} R16={s['R16']:.4f} MRR={s['MRR']:.6f} median={s['MED']:.1f}")

print("\n[7/9] Direct causal movement tests...")
# For every item compare score of ORIGINAL target under correct vs controls.
correct_gold=[];no_gold=[];wrong_gold=[];cf_old=[];cf_new=[]
for i in range(N):
    correct_gold.append(SCORES["CORRECT_A"][i][i])
    no_gold.append(SCORES["NO_A"][i][i])
    wrong_gold.append(SCORES["WRONG_A"][i][i])
    cf_old.append(SCORES["COUNTERFACTUAL_A"][i][i])
    cf_new.append(SCORES["COUNTERFACTUAL_A"][i][SEAL_TO_ITEM[CF_TARGET[i]]])

correct_gold=np.asarray(correct_gold);no_gold=np.asarray(no_gold);wrong_gold=np.asarray(wrong_gold)
cf_old=np.asarray(cf_old);cf_new=np.asarray(cf_new)

C_GT_NO=float(np.mean(correct_gold>no_gold))
C_GT_WRONG=float(np.mean(correct_gold>wrong_gold))
CF_NEW_GT_OLD=float(np.mean(cf_new>cf_old))
CF_SHIFT=float(np.mean(cf_new-cf_old))
ORIG_DROP_NO=float(np.mean(correct_gold-no_gold))
ORIG_DROP_WRONG=float(np.mean(correct_gold-wrong_gold))

print(f"      correct target score > NO-A       : {C_GT_NO:.4f}")
print(f"      correct target score > WRONG-A    : {C_GT_WRONG:.4f}")
print(f"      CF new target score > old target  : {CF_NEW_GT_OLD:.4f}")
print(f"      mean correct-minus-NO score        : {ORIG_DROP_NO:+.6f}")
print(f"      mean correct-minus-WRONG score     : {ORIG_DROP_WRONG:+.6f}")
print(f"      mean CF new-minus-old score        : {CF_SHIFT:+.6f}")

print("\n[8/9] Counterfactual item audit...")
CF_HIT=0;CF_OLD_HIT=0
for i,it in enumerate(ITEMS):
    p=PRED["COUNTERFACTUAL_A"][i];newidx=SEAL_TO_ITEM[CF_TARGET[i]]
    if p==newidx:CF_HIT+=1
    if p==i:CF_OLD_HIT+=1
    print(f"      [{i+1:02d}] {it['gold']['entity']:6s} | old={it['gold']['seal']} new={CF_TARGET[i]} | pred={ITEMS[p]['gold']['seal']} | new-rank={RES['COUNTERFACTUAL_A'][i]:2d}")

CF_HIT/=N;CF_OLD_HIT/=N

# Primary causal criteria are frozen before result.
# Correct and counterfactual arms should retain strong addressability.
# Controls should materially underperform correct A.
DELTA_NO=ST["CORRECT_A"]["R1"]-ST["NO_A"]["R1"]
DELTA_WRONG=ST["CORRECT_A"]["R1"]-ST["WRONG_A"]["R1"]

if (ST["CORRECT_A"]["R1"]>=.50 and ST["COUNTERFACTUAL_A"]["R1"]>=.50 and
    DELTA_NO>=.25 and DELTA_WRONG>=.25 and CF_NEW_GT_OLD>=.70 and CF_OLD_HIT<=.15):
    VERDICT="FROZEN_LATENT_ADDRESS_IS_CAUSALLY_MEMORY_DEPENDENT"
elif (ST["CORRECT_A"]["R1"]>=.40 and ST["COUNTERFACTUAL_A"]["R1"]>=.40 and
      DELTA_NO>=.15 and DELTA_WRONG>=.15 and CF_NEW_GT_OLD>=.60):
    VERDICT="FROZEN_LATENT_ADDRESS_CAUSALITY_OBSERVED"
else:
    VERDICT="FROZEN_LATENT_ADDRESS_CAUSALITY_NOT_YET_ESTABLISHED"

print("\n[9/9] Final classification...")
print("      L27                               : FROZEN")
print("      SOFT2                             : FROZEN")
print("      Discovery                         : NONE")
print("      Reselection                       : NONE")
print("      Human-readable runtime address    : NONE")
print("      Answer generation                 : NONE")
print("      Teacher forcing                   : NONE")
print("      Second autoregressive forward     : NONE")
print("      Intermediate decode               : NONE")
print("      Query-time B candidate forwards   : NONE")
print("      Learned router / ANN / DRA        : NONE")

print("\n"+"="*174)
print("TEST514 FINAL RESULT — AKBASCORE MAM · FROZEN ÇAĞRIİZ ADDRESS CAUSALITY")
print("="*174)
print("MODEL                          : Qwen/Qwen2.5-7B-Instruct · frozen")
print("PARENT                         : TEST513")
print("TRACE                          : L27 · frozen")
print("ADDRESS                        : SOFT2 · frozen")
print("DISCOVERY / RESELECTION        : NONE / NONE")
print("RUNTIME ADDRESS                : NUMERIC VECTOR")
print("-"*174)
for k in RES:
    s=ST[k]
    print(f"{k:18s} R1/R5/MRR : {s['R1']:.4f} / {s['R5']:.4f} / {s['MRR']:.6f}")
print("-"*174)
print(f"CORRECT − NO-A R1              : {DELTA_NO:+.4f}")
print(f"CORRECT − WRONG-A R1           : {DELTA_WRONG:+.4f}")
print(f"CORRECT SCORE > NO-A           : {C_GT_NO:.4f}")
print(f"CORRECT SCORE > WRONG-A        : {C_GT_WRONG:.4f}")
print(f"CF NEW > OLD SCORE             : {CF_NEW_GT_OLD:.4f}")
print(f"CF NEW TARGET R1               : {CF_HIT:.4f}")
print(f"CF OLD TARGET R1               : {CF_OLD_HIT:.4f}")
print(f"CF NEW − OLD MEAN SCORE        : {CF_SHIFT:+.6f}")
print("-"*174)
print("TEST508 LOCK SHA               :",TEST508)
print("TEST509 LOCK SHA               :",TEST509)
print("TEST510 LOCK SHA               :",TEST510)
print("TEST511 LOCK SHA               :",TEST511)
print("TEST512 LOCK SHA               :",TEST512)
print("TEST513 LOCK SHA               :",TEST513)
print("TEST514 LOCK SHA               :",LOCK_SHA)
print(f"TOTAL TEST TIME                : {time.perf_counter()-T0:.2f}s")
print("VERDICT                        :",VERDICT)
print("="*174)

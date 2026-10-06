# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST518
# HUMAN-SCAFFOLD ABLATION ON A CORRECTED INDEPENDENT-MEMORY PANEL
#
# PURPOSE:
# TEST516 worked, but used (a) a closed list of gold Seal tokens and
# (b) a gold-labelled B address. TEST518 removes those aids one by one while
# preserving TEST516's frozen vocabulary output->input interface.
#
# PRIMARY QUESTION:
# Does independent-session numeric relay survive when the human-defined Seal
# candidate list and gold-labelled B address are removed?
#
# PREDECLARED PRIMARY ARM: AD3 + H2
# PASS: address R1>=.70 with NO_A at chance; H2 oracle-B>=.80 and >=ZERO+.40;
#       distractor carrier follows its own B mapping >=.80; E2E>=.60.

import os,sys,subprocess,importlib.util,random,time,hashlib,json,gc,re,math
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache

TEST="518";SEED=518;MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N=48;TRACE_DEPTH=27;MAX_NEW=12;DEVICE=torch.device("cuda")
FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
TEST516="53f4fa86f1bc4f2c4db412bde68a2b004afc5a6c6cf84d1c3345b175ab1a3e8d"
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*176)
print("TEST518 — AKBASCORE MAM · HUMAN-SCAFFOLD ABLATION ON A CORRECTED PANEL")
print("TEST516 VOCABULARY BRIDGE PRESERVED → CLOSED GOLD LIST / GOLD B ADDRESS REMOVED STEPWISE")
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

# New 48-item panel. Entity names are deterministic synthetic proper names.
# Seals/classes are chosen from tokenizer-native single tokens to avoid any
# FIRST/SECOND candidate machinery. This makes the scaffold ablation stricter:
# full-vocabulary probabilities are used directly and no gold candidate list exists.
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
        # Exact standalone one-token uppercase alphabetic code, 3..8 chars.
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

def reorder(rows,pos,seed):
    rr=random.Random(seed);x=rows[:];rr.shuffle(x);g=x.pop(0);x.insert(pos,g);return x

def make_items():
    # A_i owns three unique seals: gold a_i0 + two private distractors a_i1/a_i2.
    # B_i contains gold a_i0 plus distractor seals borrowed from OTHER items.
    # Therefore A_i's own distractors never occur in B_i, and gold seal occurs in B_i only.
    items=[]
    for i in range(N):
        ents=NAMES[3*i:3*i+3];a_seals=SEALS[3*i:3*i+3];gold=a_seals[0]
        # Borrow other items' private distractors, never their golds.
        j1=(i+7)%N;j2=(i+19)%N
        b_seals=[gold,SEALS[3*j1+1],SEALS[3*j2+2]]
        b_classes=CLASSES[3*i:3*i+3]
        ra=[f"Instrument {ents[j]} carries seal {a_seals[j]}." for j in range(3)]
        rb=[f"Seal {b_seals[j]} corresponds to routing class {b_classes[j]}." for j in range(3)]
        rr=random.Random(SEED+i*101)
        rr.shuffle(ra);rr.shuffle(rb)
        items.append({"id":i+1,"A":" ".join(ra),"B":" ".join(rb),
                      "gold":{"entity":ents[0],"seal":gold,"class":b_classes[0]},
                      "A_distractors":a_seals[1:],"B_seals":b_seals,"B_classes":b_classes})
    return items
ITEMS=make_items()
# Panel invariants.
GOLDS=[x["gold"]["seal"] for x in ITEMS]
assert len(set(GOLDS))==N
for i,it in enumerate(ITEMS):
    assert it["gold"]["seal"] in it["B_seals"]
    assert all(s not in it["B_seals"] for s in it["A_distractors"])
    assert sum(it["gold"]["seal"] in x["B_seals"] for x in ITEMS)==1

LOCK={"test":TEST,"model":MODEL_ID,"seed":SEED,"N":N,"parent516":TEST516,"items":ITEMS,"trace_depth":TRACE_DEPTH,
"panel":"GOLD_UNIQUE_TO_Bi;A_DISTRACTORS_ABSENT_FROM_Bi;B_DISTRACTORS_BORROWED_FROM_OTHER_ITEMS",
"address_arms":{"AD0":"closed_gold_list+gold_labelled_B_reference","AD1":"full_vocab+gold_labelled_B","AD2":"full_vocab+label_free_position_max","AD3":"bank_inventory_renorm+label_free_position_max"},
"handoff_arms":{"H0":"closed_gold_list_reference","H1":"full_vocab_soft_embedding","H2":"bank_inventory_renorm_soft_embedding"},
"controls":["NO_A_ADDRESS","ZERO","WRONG","DISTRACTOR_COUNTERFACTUAL"],"training":"OFF","dra":"OFF","ann":"OFF","learned_router":"OFF",
"pass":{"AD3_R1":.70,"H2":.80,"H2_minus_ZERO":.40,"CF_follow":.80,"E2E_AD3_H2":.60}}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
print("      TEST518 LOCK:",LOCK_SHA);print("      PARENT516   :",TEST516)
print("      Panel       : 48 new items; each gold Seal occurs in exactly one B; A distractors absent from its B")
print("      Seal codes  : tokenizer-native single tokens; no FIRST/SECOND gold candidate machinery")

@torch.inference_mode()
def kv_from_ids(ids):
    o=model(input_ids=torch.tensor([ids],device=DEVICE),use_cache=False,output_hidden_states=True,return_dict=True);out=[]
    for L,layer in enumerate(model.model.layers):
        h=o.hidden_states[L][0].to(layer.input_layernorm.weight.device,layer.input_layernorm.weight.dtype);hn=layer.input_layernorm(h)
        k=layer.self_attn.k_proj(hn).view(-1,KVH,HD).transpose(0,1).contiguous();v=layer.self_attn.v_proj(hn).view(-1,KVH,HD).transpose(0,1).contiguous();out.append((k,v))
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
    T=raw[0][0].shape[1];return [(rope_k(k,list(range(T)),L),v) for L,(k,v) in enumerate(raw)],T,T
def cache_of(inst):
    try:c=DynamicCache(config=cfg)
    except TypeError:c=DynamicCache()
    for L,(k,v) in enumerate(inst):c.update(k.unsqueeze(0),v.unsqueeze(0),L)
    return c
def firstline(x):return next((z.strip() for z in str(x).splitlines() if z.strip()),"")
def normtxt(x):return firstline(x).strip().upper().rstrip(".")
def exact(x,t):return normtxt(x)==str(t).upper()
def qA(e):return f"What seal does instrument {e} carry? Give only the exact seal."

print("\n[2/11] OFFLINE FORGE — persistent A/B BELLEKÖZ...")
A_PACK=[];B_PACK=[]
for i,it in enumerate(ITEMS):
    A_PACK.append(install(forge(it["A"])));B_PACK.append(install(forge(it["B"])))
    if i<5:print(f"      ITEM {i+1:02d} | A={A_PACK[-1][1]} B={B_PACK[-1][1]} tokens | gold={it['gold']['seal']}")
print("      96 tensor cartridges forged.")
gc.collect();torch.cuda.empty_cache()

# Numeric headers are derived automatically from each B source's actual token stream.
# No position is labelled as "the key" for AD2/AD3.
B_TOKEN_IDS=[]
for it in ITEMS:B_TOKEN_IDS.append(enc(it["B"]+SEP))
BANK_IDS=sorted(set(t for row in B_TOKEN_IDS for t in row))
BANK_MASK=torch.tensor(BANK_IDS,dtype=torch.long)
GOLD_IDS=torch.tensor([SEAL_ID[s] for s in GOLDS],dtype=torch.long)
print("\n[3/11] Numeric address headers frozen...")
print(f"      Full vocabulary: {VOC} tokens")
print(f"      Bank inventory : {len(BANK_IDS)} unique token IDs derived from stored B sources")
print("      AD2/AD3 header : every B token position; no gold/key position labels")

print("\n[4/11] PERSISTENCE RESET...")
print("      Forge-time runtime state discarded; fresh DynamicCache per query/read.")
print("      Persistent experiment state: BELLEKÖZ tensors + numeric token-position headers/inventory.")

@torch.inference_mode()
def a_trace(packet,q):
    inst,Tm,P=packet;c=cache_of(inst);ids=enc(FMT.format(q=q));n=len(ids);pos=torch.arange(P,P+n,device=DEVICE).unsqueeze(0);mask=torch.ones((1,Tm+n),device=DEVICE,dtype=torch.long)
    o=model(input_ids=torch.tensor([ids],device=DEVICE),past_key_values=c,attention_mask=mask,position_ids=pos,use_cache=True,output_hidden_states=True,return_dict=True)
    h=o.hidden_states[TRACE_DEPTH][0,-1];fn=model.model.norm;logits=model.lm_head(fn(h.to(fn.weight.dtype))).float().cpu()
    return logits,o.past_key_values,Tm+n,P+n

@torch.inference_mode()
def no_a_logits(q):
    ids=enc(FMT.format(q=q));o=model(input_ids=torch.tensor([ids],device=DEVICE),use_cache=False,output_hidden_states=True,return_dict=True)
    h=o.hidden_states[TRACE_DEPTH][0,-1];fn=model.model.norm;return model.lm_head(fn(h.to(fn.weight.dtype))).float().cpu()

def probs_full(logits):return torch.softmax(logits.float(),0)
def probs_closed(logits):
    z=torch.full_like(logits.float(),-torch.inf);z[GOLD_IDS]=logits[GOLD_IDS].float();return torch.softmax(z,0)
def probs_bank(logits):
    z=torch.full_like(logits.float(),-torch.inf);z[BANK_MASK]=logits[BANK_MASK].float();return torch.softmax(z,0)

def score_gold_labelled(p):
    return [float(p[SEAL_ID[it["gold"]["seal"]]]) for it in ITEMS]
def score_posmax(p):
    return [max(float(p[t]) for t in row) for row in B_TOKEN_IDS]
def rank(sc,gold):
    order=sorted(range(N),key=lambda j:(-sc[j],j));return order.index(gold)+1,order[0]
def metrics(rs):
    a=np.asarray(rs,dtype=np.float64);return float(np.mean(a==1)),float(np.mean(a<=5)),float(np.mean(a<=16)),float(np.mean(1.0/a))

print("\n[5/11] ADDRESS ABLATION — AD0/AD1/AD2/AD3 + NO_A...")
LOGITS=[];P0=[];P1=[];P3=[];R={k:[] for k in ("AD0","AD1","AD2","AD3","NO_A")};SEL={k:[] for k in R}
for i,it in enumerate(ITEMS):
    l,_,_,_=a_trace(A_PACK[i],qA(it["gold"]["entity"]));p0=probs_closed(l);p1=probs_full(l);p3=probs_bank(l);ln=no_a_logits(qA(it["gold"]["entity"]));pn=probs_bank(ln)
    arms={"AD0":score_gold_labelled(p0),"AD1":score_gold_labelled(p1),"AD2":score_posmax(p1),"AD3":score_posmax(p3),"NO_A":score_posmax(pn)}
    LOGITS.append(l);P0.append(p0);P1.append(p1);P3.append(p3)
    for k,sc in arms.items():r,s=rank(sc,i);R[k].append(r);SEL[k].append(s)
    print(f"      [{i+1:02d}/48] {it['gold']['entity']:7s} | AD0={R['AD0'][-1]:2d} AD1={R['AD1'][-1]:2d} AD2={R['AD2'][-1]:2d} AD3={R['AD3'][-1]:2d} NOA={R['NO_A'][-1]:2d}")
for k in R:
    m=metrics(R[k]);print(f"      {k:4s} R1/R5/R16/MRR: {m[0]:.4f} / {m[1]:.4f} / {m[2]:.4f} / {m[3]:.6f}")

# Carrier construction. H0 is TEST516's closed-gold-list reference, now one-token.
# H1 uses full-vocabulary p@E. H2 uses bank-inventory-renormalized p@E.
# A genuine second continuous rollout is computed from c1 through the same fresh A
# session, then L27 readout produces p2; no argmax/decode is used.
@torch.inference_mode()
def soft_embed(p):
    # GPU matmul avoids a float32 CPU copy of the entire embedding matrix.
    pp=p.to(device=DEVICE,dtype=torch.float32);ew=EMB.weight.detach().float();return (pp@ew).to(dtype=EMB.weight.dtype)

@torch.inference_mode()
def rollout_second(packet,q,c1,mode):
    # Recreate the A session deterministically, then append c1 as a continuous input slot.
    _,c,Tm,P=a_trace(packet,q);x=c1.to(device=DEVICE,dtype=EMB.weight.dtype).view(1,1,H);pos=torch.tensor([[P]],device=DEVICE,dtype=torch.long);mask=torch.ones((1,Tm+1),device=DEVICE,dtype=torch.long)
    o=model(inputs_embeds=x,past_key_values=c,attention_mask=mask,position_ids=pos,use_cache=False,output_hidden_states=True,return_dict=True)
    h=o.hidden_states[TRACE_DEPTH][0,-1];fn=model.model.norm;l2=model.lm_head(fn(h.to(fn.weight.dtype))).float().cpu()
    p2=probs_closed(l2) if mode=="closed" else probs_full(l2) if mode=="full" else probs_bank(l2)
    return soft_embed(p2).detach().cpu()

def carrier_from_p(packet,q,p,mode):
    c1=soft_embed(p).detach().cpu();c2=rollout_second(packet,q,c1,mode);return torch.stack([c1,c2]).to(dtype=EMB.weight.dtype)
def exact_seal_carrier(seal):
    c1=EMB.weight.detach()[SEAL_ID[seal]].cpu();return torch.stack([c1,torch.zeros_like(c1)]).to(dtype=EMB.weight.dtype)
def zero_carrier():return torch.zeros((2,H),dtype=EMB.weight.dtype)

PREFIX="QUESTION:\nUsing the stored memory, the relevant internal key is"
SUFFIX=". What routing class corresponds to that internal key? Give only the exact routing class.\n\nANSWER:"
@torch.inference_mode()
def read_B_numeric(packet,carrier,max_new=MAX_NEW):
    inst,Tm,P=packet;c=cache_of(inst);pre=enc(PREFIX);suf=enc(SUFFIX);pree=EMB(torch.tensor(pre,device=DEVICE));sufe=EMB(torch.tensor(suf,device=DEVICE));car=carrier.to(device=DEVICE,dtype=EMB.weight.dtype)
    x=torch.cat([pree,car,sufe],0).unsqueeze(0);n=x.shape[1];pos=torch.arange(P,P+n,device=DEVICE).unsqueeze(0);mask=torch.ones((1,Tm+n),device=DEVICE,dtype=torch.long)
    o=model(inputs_embeds=x,past_key_values=c,attention_mask=mask,position_ids=pos,use_cache=True,return_dict=True);c=o.past_key_values;out=[];Tm+=n;P+=n;nxt=int(o.logits[0,-1].float().argmax())
    for _ in range(max_new):
        if nxt in EOS:break
        out.append(nxt);pos=torch.tensor([[P]],device=DEVICE);mask=torch.ones((1,Tm+1),device=DEVICE,dtype=torch.long)
        o=model(input_ids=torch.tensor([[nxt]],device=DEVICE),past_key_values=c,attention_mask=mask,position_ids=pos,use_cache=True,return_dict=True);c=o.past_key_values;Tm+=1;P+=1;nxt=int(o.logits[0,-1].float().argmax())
    return tok.decode(out,skip_special_tokens=True).strip()

print("\n[6/11] Building H0/H1/H2 continuous carriers with soft rollout...")
C={"H0":[],"H1":[],"H2":[]}
for i,it in enumerate(ITEMS):
    q=qA(it["gold"]["entity"]);C["H0"].append(carrier_from_p(A_PACK[i],q,P0[i],"closed"));C["H1"].append(carrier_from_p(A_PACK[i],q,P1[i],"full"));C["H2"].append(carrier_from_p(A_PACK[i],q,P3[i],"bank"))
    if i<5 or (i+1)%8==0:print(f"      [{i+1:02d}/48] carriers ready")

print("\n[7/11] HANDOFF ABLATION — oracle B: H0/H1/H2/ZERO/WRONG/DISTRACTOR...")
RES={k:[] for k in ("H0","H1","H2","ZERO","WRONG","CF_FOLLOW")};RAW={k:[] for k in RES}
for i,it in enumerate(ITEMS):
    target=it["gold"]["class"];wrong=(i+11)%N
    # Counterfactual uses B_i's first distractor seal and expects that seal's own mapped class.
    cf_seal=it["B_seals"][1];cf_class=it["B_classes"][1]
    ys={"H0":read_B_numeric(B_PACK[i],C["H0"][i]),"H1":read_B_numeric(B_PACK[i],C["H1"][i]),"H2":read_B_numeric(B_PACK[i],C["H2"][i]),
        "ZERO":read_B_numeric(B_PACK[i],zero_carrier()),"WRONG":read_B_numeric(B_PACK[i],C["H2"][wrong]),"CF_FOLLOW":read_B_numeric(B_PACK[i],exact_seal_carrier(cf_seal))}
    for k,y in ys.items():RES[k].append(int(exact(y,cf_class if k=="CF_FOLLOW" else target)));RAW[k].append(y)
    print(f"      [{i+1:02d}/48] H0={RES['H0'][-1]} H1={RES['H1'][-1]} H2={RES['H2'][-1]} Z={RES['ZERO'][-1]} W={RES['WRONG'][-1]} CF={RES['CF_FOLLOW'][-1]}")
for k in RES:print(f"      {k:9s}: {sum(RES[k]):2d}/{N} = {np.mean(RES[k]):.4f}")

print("\n[8/11] E2E — AD2+H1 and primary AD3+H2...")
E2E={"AD2_H1":[],"AD3_H2":[]};E2ERAW={k:[] for k in E2E}
for i,it in enumerate(ITEMS):
    y21=read_B_numeric(B_PACK[SEL["AD2"][i]],C["H1"][i]);y32=read_B_numeric(B_PACK[SEL["AD3"][i]],C["H2"][i]);t=it["gold"]["class"]
    E2E["AD2_H1"].append(int(exact(y21,t)));E2ERAW["AD2_H1"].append(y21);E2E["AD3_H2"].append(int(exact(y32,t)));E2ERAW["AD3_H2"].append(y32)
    print(f"      [{i+1:02d}/48] AD2+H1={E2E['AD2_H1'][-1]} AD3+H2={E2E['AD3_H2'][-1]} | B2={SEL['AD2'][i]+1:02d} B3={SEL['AD3'][i]+1:02d}")

print("\n[9/11] Wilson intervals + causal deltas...")
def wilson(k,n,z=1.959963984540054):
    if n==0:return (float('nan'),float('nan'))
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return c-h,c+h
for name,vals in [(k,[int(x==1) for x in R[k]]) for k in R]+[(k,RES[k]) for k in RES]+[(k,E2E[k]) for k in E2E]:
    kk=sum(vals);lo,hi=wilson(kk,len(vals));print(f"      {name:9s}: {kk:2d}/{len(vals)} = {kk/len(vals):.4f} | 95% Wilson [{lo:.4f},{hi:.4f}]")
AD3_R1=metrics(R["AD3"])[0];NOA_R1=metrics(R["NO_A"])[0];H2=float(np.mean(RES["H2"]));ZERO=float(np.mean(RES["ZERO"]));CF=float(np.mean(RES["CF_FOLLOW"]));E32=float(np.mean(E2E["AD3_H2"]))
print(f"      H2 - ZERO: {H2-ZERO:+.4f}")

print("\n[10/11] Predeclared decision...")
# 48-way chance R1 = .0208. NO_A guard is deliberately loose (.10) to avoid claiming a pass under prompt-only routing.
PASS_ADDR=AD3_R1>=.70 and NOA_R1<=.10
PASS_HAND=H2>=.80 and (H2-ZERO)>=.40 and CF>=.80
PASS_E2E=E32>=.60
if PASS_ADDR and PASS_HAND and PASS_E2E:VERDICT="LABEL_FREE_VOCABULARY_BRIDGE_REPLICATED_WITHOUT_HUMAN_KEY_SCAFFOLD"
elif AD3_R1>=.70 and not PASS_HAND:VERDICT="LABEL_FREE_ADDRESS_OBSERVED_BUT_HANDOFF_SCAFFOLD_DEPENDENCE_REMAINS"
elif AD3_R1<.40 and H2>=.80:VERDICT="HANDOFF_SURVIVES_BUT_LABEL_FREE_ADDRESS_NOT_ESTABLISHED"
elif AD3_R1<.40 and H2<.50:VERDICT="TEST516_SUCCESS_DEPENDS_MATERIALLY_ON_HUMAN_SCAFFOLD"
else:VERDICT="SCAFFOLD_ABLATION_PARTIAL_INCONCLUSIVE"
print("      Address pass :",PASS_ADDR);print("      Handoff pass :",PASS_HAND);print("      E2E pass     :",PASS_E2E);print("      VERDICT      :",VERDICT)

print("\n[11/11] Protocol audit...")
print("      Model weights frozen                    : YES")
print("      Fresh cache each query/read              : YES")
print("      A/B cache concatenation                  : NO")
print("      Intermediate Seal decode/reinsert        : NO")
print("      AD2/AD3 gold B/key-position label        : NO")
print("      AD3 candidate inventory                  : DERIVED FROM B BANK CONTENT")
print("      AD3 gold-only Seal list                  : NO")
print("      H2 gold-only Seal list                   : NO")
print("      Query-time candidate-B model forwards    : 0")
print("      Learned router / ANN / DRA / training    : NONE")
print("      Eval-time tuning/reselection             : NONE")
print("      Corrected panel gold Seal unique to B_i  : YES")
print("      A_i distractor Seals absent from B_i     : YES")
print("      Counterfactual B-distractor carrier      : YES")

print("\n"+"="*176)
print("TEST518 FINAL RESULT — AKBASCORE MAM · HUMAN-SCAFFOLD ABLATION")
print("="*176)
print("MODEL                              : Qwen/Qwen2.5-7B-Instruct · frozen")
print("PARENT                             : TEST516")
print("PANEL                              : 48 new corrected items")
print("TRACE                              : L27 · frozen")
print("VOCABULARY INTERFACE               : PRESERVED")
print("AD3 GOLD KEY LABEL                 : NONE")
print("AD3/H2 GOLD CANDIDATE LIST         : NONE")
print("AD3/H2 INVENTORY                   : B-bank-derived numeric token inventory")
print("-"*176)
for k in ("AD0","AD1","AD2","AD3","NO_A"):
    m=metrics(R[k]);print(f"{k:34s}: R1={m[0]:.4f} R5={m[1]:.4f} R16={m[2]:.4f} MRR={m[3]:.6f}")
print("-"*176)
for k in ("H0","H1","H2","ZERO","WRONG","CF_FOLLOW"):print(f"{k:34s}: {sum(RES[k]):2d}/{N} = {np.mean(RES[k]):.4f}")
print(f"H2 - ZERO                         : {H2-ZERO:+.4f}")
print("-"*176)
print(f"E2E AD2 + H1                      : {sum(E2E['AD2_H1']):2d}/{N} = {np.mean(E2E['AD2_H1']):.4f}")
print(f"E2E AD3 + H2 (PRIMARY)            : {sum(E2E['AD3_H2']):2d}/{N} = {E32:.4f}")
print("-"*176)
print("PREDECLARED PASS                  : AD3 R1>=.70 & NO_A R1<=.10; H2>=.80; H2-ZERO>=.40; CF>=.80; E2E>=.60")
print("ADDRESS PASS                       :",PASS_ADDR)
print("HANDOFF PASS                       :",PASS_HAND)
print("E2E PASS                           :",PASS_E2E)
print("TEST516 LOCK                       :",TEST516)
print("TEST518 LOCK                       :",LOCK_SHA)
print(f"TOTAL TEST TIME                    : {time.perf_counter()-T0:.2f}s")
print("VERDICT                            :",VERDICT)
print("="*176)

# =====================================================================================================================================
# AKBASCORE MAM · MISTRAL-7B ASSOCIATIVE RETRIEVAL INSTRUMENT — PART 1 / 3 · ENGINE
# Run PART 1 → PART 2 → PART 3 as three consecutive Google Colab cells in the SAME runtime (GPU: A100 recommended).
# PART 1 embeds the supplied scientific engine VERBATIM, verifies its SHA-256, loads the frozen model and forges the 128 A / 128 B
# numeric memories (about 1–2 minutes). PART 2 adds the measurement pipeline and figures. PART 3 starts the Gradio instrument.
# Copyright © 2026 Mustafa Akbaş. All rights reserved. · Mersin, Türkiye
#
# SCIENTIFIC ENGINE  : MAM_MISTRAL_128_DEMO_ENGINE.py, embedded byte-for-byte as ENGINE_SRC and executed in its own module namespace.
#                      Its retrieval architecture is frozen; this program does not modify, recalibrate or re-implement it.
# MECHANISM          : derived from the sealed TEST528 record (frozen Mistral-7B-Instruct-v0.3 · pointer L28H00 · address L00-V
#                      uncentered · 1024-D · max cosine over 128 precomputed B memories · 0 query-time candidate-B forwards).
#                      This program is an executable demonstration layer; it is NOT TEST528 and does not recompute TEST528 statistics.
# INSTRUMENTATION    : (1) a counting-only forward pre-hook, attached only while an engine API call runs and removed afterwards; it
#                      records input length and installed-cache length of every model forward and never alters inputs or outputs;
#                      (2) an all-parameter weight guard; (3) re-derivation of every reported quantity from the raw score vectors.
# =====================================================================================================================================
import os,sys,io,re,json,math,time,html,random,shutil,hashlib,zipfile,platform,textwrap,threading,subprocess,importlib.util,traceback,gc,types
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import quote
from collections import Counter
#<<CONST_BEGIN>>
DEMO_TITLE="AKBASCORE MAM · Mistral-7B associative retrieval instrument"
DEMO_SHORT="AKBASCORE MAM · Mistral demo instrument"
AUTHOR="Mustafa Akbaş";AUTHOR_PLACE="Mersin, Türkiye";COPYRIGHT="Copyright © 2026 Mustafa Akbaş. All rights reserved."
ENGINE_FILE="MAM_MISTRAL_128_DEMO_ENGINE.py"
ENGINE_SHA256_EXPECTED="df17ba861d4f585162a190da0cd795d5b110e716f1c4953945489f6b0324d2fe"
N_CAND=128;ADDR_DIM=1024;ARCH=(32,4096,32,8,128)
FROZEN_CONFIG={"PTR_L":28,"PTR_H":0,"ADDR_L":0,"ADDR_KIND":"V","ADDR_CENTERED":False,"N":128,"KVD":1024,"MODEL_ID":"mistralai/Mistral-7B-Instruct-v0.3"}
CANON_GPU="NVIDIA A100-SXM4-40GB"
# ---- archived records (read from the sealed experiment logs; never produced or recomputed by this program) ----
TEST528=dict(name="TEST528",lock="4ddcea61a56eab327afe28f190cef7eec6670c9b55706b49de5d5c6be53a7f77",result_sha="9235ed38c944f41a9ede0426d3aabaa12b5b20646f30e2be921e618f9471d3c2",
    primary=(127,128,"99.22%"),counterfactual=(125,128,"97.66%"),shifted=(0,128,"0.00%"),no_a=(1,128,"0.78%"),
    model="mistralai/Mistral-7B-Instruct-v0.3",pointer="L28H00",address="L00-V uncentered",address_dim=1024,identity_regime="native uppercase single-token seals")
CROSS_MODEL=[dict(model="Qwen2.5-7B-Instruct",record="TEST523",pointer="L23H12",address="L02-V uncentered",dim=512,top1=(118,128,"92.19%")),
             dict(model="Mistral-7B-Instruct-v0.3",record="TEST528",pointer="L28H00",address="L00-V uncentered",dim=1024,top1=(127,128,"99.22%"))]
CROSS_MODEL_NOTE="The mechanism transferred across model families; the internal coordinates did not."
SCOPE={
 "demonstrated":[
  "For the selected item, the active numeric A memory and a natural-language question drive one frozen forward pass of Mistral-7B-Instruct-v0.3.",
  "Attention head L28H00 (query head 0 of layer 28) at the final query token yields a pointer distribution over the A memory slots (ÇAĞRIİZ).",
  "The pointer weights the layer-0 value vectors of the A slots into a 1024-dimensional uncentered address.",
  "The address is compared by maximum cosine similarity with every row of all 128 precomputed B address matrices; the highest score selects the B memory.",
  "Model forwards during retrieval are counted by instrumentation: one forward, with only the A memory installed as cache.",
  "A counterfactual A memory (changed seal association, re-forged) changes the selected B memory under the same frozen mechanism."],
 "archived":[
  "TEST528 aggregate results (128 items) are sealed reference values from the archived experiment log. They are displayed, never recomputed from demo interactions.",
  "Cross-model context: TEST523 (Qwen2.5-7B-Instruct) and TEST528 (Mistral-7B-Instruct-v0.3) used different internal coordinates and different panels; the two values are not a head-to-head comparison."],
 "not_established":[
  "Generalisation to arbitrary token identities: the TEST528 identity regime uses native uppercase single-token seals.",
  "Retrieval of the first (A) memory from an empty bank: A is the active memory in this protocol.",
  "Natural-language answer generation from the selected B memory: this instrument measures associative retrieval only.",
  "Transfer of internal coordinates between models: pointer and address coordinates were identified per model family.",
  "Behaviour beyond the 128-candidate bank used here."],
 "notes":[
  "L00-V is the layer-0 value projection of the RMS-normalised input token embedding at each A slot; the address is therefore a pointer-weighted combination of token-level value vectors.",
  "Gold-span (target seal token) positions are computed by the engine for diagnostics only and are not an input to any retrieval step.",
  "The expected B candidate is defined by panel construction: item i's target seal occurs in B memory i only (target and distractor pools are disjoint).",
  "Candidate-B forwards: the engine reports a constant 0; this instrument additionally measures every model forward during each API call.",
  "The engine keeps its forge-time item records (ITEMS) in memory; retrieve() reads only the A packet, the A tensors, the question string and the B address matrices."]}
class AuditFail(RuntimeError):pass
def jsafe(o):
    if isinstance(o,dict):return{str(k):jsafe(v)for k,v in o.items()}
    if isinstance(o,(list,tuple)):return[jsafe(v)for v in o]
    if hasattr(o,"item")and not isinstance(o,(str,bytes)):
        try:o=o.item()
        except Exception:pass
    if isinstance(o,float):return o if math.isfinite(o)else f"non-finite:{o}"
    if isinstance(o,(str,int,bool))or o is None:return o
    return str(o)
def canon(o):return json.dumps(jsafe(o),sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
def fr(t):return f"{t[0]}/{t[1]}"
def rederive(scores,expected):
    """Independent re-derivation of selection, rank and margins from a raw 128-score vector (ties → lower index, as in the engine)."""
    s=[float(x)for x in scores]
    if len(s)!=N_CAND:raise AuditFail(f"score vector has {len(s)} entries, expected {N_CAND}")
    if not all(math.isfinite(x)for x in s):raise AuditFail("non-finite candidate score")
    order=sorted(range(len(s)),key=lambda j:(-s[j],j));sel=order[0]+1;rest=[s[j]for j in order[1:]];srt=sorted(rest);m=len(srt)
    mu=sum(rest)/m;sd=(sum((x-mu)**2 for x in rest)/(m-1))**.5
    q=lambda f:srt[min(m-1,max(0,int(round(f*(m-1)))))]
    return dict(selected=sel,rank=order.index(expected-1)+1,top1=s[order[0]],top2=s[order[1]],margin=s[order[0]]-s[order[1]],order=[j+1 for j in order],
                expected_score=s[expected-1],rest_mean=mu,rest_sd=sd,rest_median=q(.5),rest_q1=q(.25),rest_q3=q(.75),rest_max=max(rest),
                z_top1=((s[order[0]]-mu)/sd if sd>0 else None),all_equal=bool(max(s)==min(s)))
def pointer_stats(w):
    w=[float(x)for x in w];a=max(range(len(w)),key=lambda t:w[t]);H=-sum(x*math.log2(x)for x in w if x>0)
    return dict(T=len(w),argmax=a,peak=w[a],sum=sum(w),w0=w[0],entropy_bits=H,effective_slots=2**H)
# ---- claim-discipline scanner: applied to every generated text (figures, logs, JSON). Panel strings are removed first. ----
ALLOWED_TESTS={"523","524","528"}
CLAIM_RULES=[("capacity-overclaim",r"(?i)\bunlimited\b|\binfinite\b"),("universal-claim",r"(?i)\buniversal\b"),("human-like-claim",r"(?i)human-like"),
    ("model-independence-claim",r"(?i)model-independent"),("all-transformers-claim",r"(?i)all transformers"),("solved-claim",r"(?i)\bsolved\b"),
    ("legal-status-claim",r"(?i)\bpatent(ed|-pending)\b|non-obvious|\bpatentab"),("metaphor",r"(?i)\bbrain\b|\bquantum\b|\bneuron"),
    ("complexity-claim",r"O\(1\)"),("compression-wording",r"(?i)\bcompress")]
def claim_scan(name,text,allow=()):
    for a in allow:
        if a:text=text.replace(a,"<allowed>")
    hits=[(name,lab,m.group(0))for lab,p in CLAIM_RULES for m in re.finditer(p,text)]
    hits+=[(name,"unexpected-test-number",m.group(0))for m in re.finditer(r"TEST\s?(\d+)",text)if m.group(1)not in ALLOWED_TESTS]
    return hits
def strip_words(text,words):
    ws=sorted({w for w in words if w and len(w)>=2},key=len,reverse=True)
    if not ws:return text
    return re.sub(r"(?<![A-Za-z])(?:"+"|".join(map(re.escape,ws))+r")(?![A-Za-z])","",text)
def cpu_selftest():
    n=0
    R=rederive([0.5-0.001*j for j in range(128)],1);assert R["selected"]==1 and R["rank"]==1 and abs(R["margin"]-0.001)<1e-12;n+=1
    s=[0.1]*128;s[9]=0.9;s[4]=0.8;R=rederive(s,5);assert R["selected"]==10 and R["rank"]==2;n+=1
    R=rederive([0.0]*128,7);assert R["selected"]==1 and R["rank"]==7 and R["all_equal"] and R["z_top1"]is None;n+=1
    try:rederive([0.1]*127,1);raise AssertionError
    except AuditFail:n+=1
    try:rederive([float("nan")]+[0.1]*127,1);raise AssertionError
    except AuditFail:n+=1
    P=pointer_stats([0,.25,.5,.25]);assert P["argmax"]==2 and abs(P["entropy_bits"]-1.5)<1e-12;n+=1
    assert claim_scan("t","Mistral-7B-Instruct-v0.3 L28H00 L00-V TEST528 TEST523 TEST524 128 candidates max cosine")==[];n+=1
    for b in("unlimited memory","universal memory","human-like memory","model-independent","works on all transformers","memory solved","patent-pending","brain","quantum","O(1)","compressed","TEST482"):
        assert claim_scan("t",b),b;n+=1
    assert strip_words("seal BRAIN carried","BRAIN".split())=="seal  carried";n+=1
    assert claim_scan("t",strip_words("seal NEURON here",["NEURON"]))==[];n+=1
    return n
#<<CONST_END>>
#<<ENGINE_BEGIN>>
for _m,_p in[("torch","torch"),("transformers","transformers"),("accelerate","accelerate"),("gradio","gradio"),("matplotlib","matplotlib"),("PIL","pillow")]:
    if importlib.util.find_spec(_m)is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",_p])
import numpy as np,torch,transformers,gradio as gr
QUIET=False
def say(*a):
    if not QUIET:print(*a,flush=True)
def utc_now():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def local_now():return datetime.now().astimezone().isoformat(timespec="milliseconds")
say("="*140);say(DEMO_TITLE+" — PART 1 / 3 · ENGINE");say(f"{AUTHOR} · {AUTHOR_PLACE} · {COPYRIGHT}");say("="*140)
say(f"[1/5] CPU self-test PASS ({cpu_selftest()} checks)")
# ---------------- SCIENTIFIC ENGINE — embedded verbatim (do not edit; its SHA-256 is verified below) ----------------
ENGINE_SRC=r'''# MAM_MISTRAL_128_DEMO_ENGINE.py
# Copyright © 2026 Mustafa Akbaş
#
# AKBASCORE MAM — MISTRAL 128-WAY DEMO ENGINE
# Persistent Associative Machine Memory
#
# Scientific backend derived from sealed TEST528.
#
# SEALED REFERENCE:
# TEST528 PRE-RESULT LOCK:
# 4ddcea61a56eab327afe28f190cef7eec6670c9b55706b49de5d5c6be53a7f77
#
# TEST528 RESULT SHA:
# 9235ed38c944f41a9ede0426d3aabaa12b5b20646f30e2be921e618f9471d3c2
#
# TEST528:
# PRIMARY          127/128 = 99.22% Top-1
# COUNTERFACTUAL   125/128 = 97.66% Top-1
# SHIFTED            0/128 =  0.00% Top-1
# NO-A               1/128 =  0.78% Top-1
#
# FROZEN ARCHITECTURE:
# MODEL    = mistralai/Mistral-7B-Instruct-v0.3
# POINTER  = L28H00
# ADDRESS  = L00-V UNCENTERED
# DIM      = 1024
# MATCH    = MAX COSINE
# MEMORY   = 128 B memories
#
# NO TRAINING
# NO LORA
# NO OPTIMIZER
# NO DRA
# NO LEARNED ROUTER
# NO HEAD/LAYER/ADDRESS DISCOVERY
# NO QUERY-TIME CANDIDATE-B FORWARD
# NO TOKEN-ID/TEXT/LM-HEAD ADDRESS
#
# This file contains the scientific engine only.
# UI / Gradio / plots / JPEG / posters belong in a separate presentation layer.
# Do not modify the frozen retrieval architecture when building the UI.

import os,sys,subprocess,importlib.util,random,time,hashlib,re,math,gc
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])

import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

# ==================================================================================================
# FROZEN CONFIGURATION
# ==================================================================================================

ENGINE="AKBASCORE MAM — MISTRAL 128-WAY DEMO ENGINE"
MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
SEED=528528
N=128
PTR_L=28
PTR_H=0
ADDR_L=0
ADDR_KIND="V"
ADDR_CENTERED=False
FMT="QUESTION:\n{q}\n\nANSWER:"
SEP="\n\n"

TEST528_LOCK="4ddcea61a56eab327afe28f190cef7eec6670c9b55706b49de5d5c6be53a7f77"
TEST528_RESULT_SHA="9235ed38c944f41a9ede0426d3aabaa12b5b20646f30e2be921e618f9471d3c2"

os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device("cuda")
torch.set_grad_enabled(False)
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

# ==================================================================================================
# GLOBAL ENGINE STATE
# ==================================================================================================

tok=None
model=None
cfg=None
layers=None
PAD=None
VOC=None
NL=H=QH=KVH=HD=REP=KVD=None

ITEMS=[]
META=[]
TARGETS=[]
DISTRACT=[]
A_RAW=[]
A_PACK=[]
B_MATS=[]
GOLD_SPANS=[]

INITIALIZED=False
WEIGHT_SENTINEL_INITIAL=None

# ==================================================================================================
# FIXED EXTERNAL PANEL GENERATOR — SAME TEST528 REGIME
# ==================================================================================================

CLASS_LABELS=["ALPHA","BETA","GAMMA"]

SYL1=["Aq","Bryn","Cav","Dex","Eln","Frax","Gyr","Hex","Iln","Jor","Kex","Lyr","Mav","Nex","Oryn","Pax",
"Qyr","Rex","Savn","Tov","Uln","Vex","Wyr","Xav","Yex","Zyr","Axl","Bov","Cyn","Drex","Evr","Fyn"]

SYL2=["adar","bren","cyr","dax","elor","fyn","grel","hyn","ivar","jor","kyr","lor","myn","nex","or","pyr",
"qen","rix","sor","tyn","ul","vyr","wen","xir","yor","zen"]

def _enc(s):
    return tok(s,add_special_tokens=False).input_ids

def _build_names(n):
    out=[]
    for a in reversed(SYL1):
        for b in reversed(SYL2):
            x=a+b
            if x not in out:out.append(x)
    rr=random.Random(SEED+70001);rr.shuffle(out)
    if len(out)<n:raise RuntimeError(f"Name pool too small: need {n}, found {len(out)}.")
    return out[:n]

def _native_code_pool():
    out=[]
    for tid in range(VOC):
        s=tok.decode([tid],skip_special_tokens=False)
        if not re.fullmatch(r"[A-Z]{3,8}",s):continue
        if _enc(s)!=[tid]:continue
        if s not in out:out.append(s)
    return out

def _pick_distinct(pool,indices,forbidden):
    for z in indices:
        x=pool[z%len(pool)]
        if x not in forbidden:return x
    raise RuntimeError("Unable to select distinct distractor.")

def _make_items():
    names=_build_names(N*3)
    items=[]
    for i in range(N):
        ents=names[3*i:3*i+3]
        gold=TARGETS[i]
        d1=_pick_distinct(DISTRACT,[i*17+5+j for j in range(len(DISTRACT))],{gold})
        d2=_pick_distinct(DISTRACT,[i*19+31+j for j in range(len(DISTRACT))],{gold,d1})
        bd1=_pick_distinct(DISTRACT,[i*23+11+j for j in range(len(DISTRACT))],{gold})
        bd2=_pick_distinct(DISTRACT,[i*29+47+j for j in range(len(DISTRACT))],{gold,bd1})
        aseals=[gold,d1,d2]
        bseals=[gold,bd1,bd2]
        ra=[f"Instrument {ents[j]} carries seal {aseals[j]}." for j in range(3)]
        rb=[f"Seal {bseals[j]} corresponds to routing class {CLASS_LABELS[j]}." for j in range(3)]
        random.Random(SEED+i*131+17).shuffle(ra)
        random.Random(SEED+i*137+29).shuffle(rb)
        items.append({
            "id":i+1,
            "entity":ents[0],
            "seal":gold,
            "A":" ".join(ra),
            "B":" ".join(rb),
            "A_seals":aseals,
            "A_distractors":[d1,d2],
            "B_seals":bseals
        })
    return items

# ==================================================================================================
# MODEL / WEIGHT SENTINEL
# ==================================================================================================

def _sentinel():
    fp=[
        layers[0].self_attn.q_proj.weight,
        layers[8].self_attn.o_proj.weight,
        layers[16].mlp.down_proj.weight,
        layers[24].self_attn.o_proj.weight,
        layers[31].mlp.down_proj.weight,
        model.model.norm.weight,
        model.lm_head.weight
    ]
    h=hashlib.sha256()
    for p in fp:
        a=p.detach().reshape(-1);n=a.numel()
        for j in range(16):
            o=j*(n-256)//15
            h.update(a[o:o+256].float().cpu().numpy().tobytes())
    return h.hexdigest()

# ==================================================================================================
# SEALED MAM NUMERIC MEMORY ENGINE
# ==================================================================================================

@torch.inference_mode()
def _kv_from_ids(ids):
    o=model(
        input_ids=torch.tensor([ids],device=DEVICE),
        use_cache=False,
        output_hidden_states=True,
        return_dict=True
    )
    out=[]
    for L,layer in enumerate(layers):
        h=o.hidden_states[L][0].to(layer.input_layernorm.weight.device,layer.input_layernorm.weight.dtype)
        hn=layer.input_layernorm(h)
        k=layer.self_attn.k_proj(hn).view(-1,KVH,HD).transpose(0,1).contiguous()
        v=layer.self_attn.v_proj(hn).view(-1,KVH,HD).transpose(0,1).contiguous()
        out.append((k,v))
    return out

def _forge(s):
    return _kv_from_ids([PAD]+_enc(s+SEP))

def _rope_cos_sin(x,pos):
    p=torch.tensor([pos],device=DEVICE,dtype=torch.long)
    try:return model.model.rotary_emb(x,p)
    except TypeError:return model.model.rotary_emb(x,position_ids=p)

def _rope_k(k,pos,L):
    T=len(pos)
    kk=k.unsqueeze(0)
    dummy=torch.zeros((1,KVH,T,HD),device=DEVICE,dtype=k.dtype)
    c,s=_rope_cos_sin(dummy,pos)
    _,kr=apply_rotary_pos_emb(dummy,kk,c,s,unsqueeze_dim=1)
    return kr[0]

def _install(raw):
    T=raw[0][0].shape[1]
    return [( _rope_k(k,list(range(T)),L),v ) for L,(k,v) in enumerate(raw)],T,T

def _cache_of(inst):
    try:c=DynamicCache(config=cfg)
    except TypeError:c=DynamicCache()
    for L,(k,v) in enumerate(inst):
        c.update(k.unsqueeze(0),v.unsqueeze(0),L)
    return c

def _question(entity):
    return f"What seal does instrument {entity} carry? Give only the exact seal."

def _gold_token_positions(text,seal):
    # Diagnostic only. Never used by retrieval.
    pat=r"(?<![A-Za-z])"+re.escape(seal)+r"(?![A-Za-z])"
    hits=[m.span() for m in re.finditer(pat,text)]
    if len(hits)!=1:raise RuntimeError(f"Exact seal occurrence !=1: {seal} | hits={hits}")
    a,b=hits[0]
    z=tok(text+SEP,add_special_tokens=False,return_offsets_mapping=True)
    pos=[]
    for j,(x,y) in enumerate(z["offset_mapping"]):
        if y>a and x<b:pos.append(j+1)
    if not pos:raise RuntimeError(f"No gold span: {seal}")
    return pos

@torch.inference_mode()
def _query_forward(packet,q):
    inst,Tm,P=packet
    c=_cache_of(inst)
    ids=_enc(FMT.format(q=q))
    n=len(ids)
    pos=torch.arange(P,P+n,device=DEVICE).unsqueeze(0)
    mask=torch.ones((1,Tm+n),device=DEVICE,dtype=torch.long)
    o=model(
        input_ids=torch.tensor([ids],device=DEVICE),
        past_key_values=c,
        attention_mask=mask,
        position_ids=pos,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True
    )
    return o,Tm,P,n

@torch.inference_mode()
def _frozen_pointer(packet,q):
    inst,Tm,P=packet
    o,_,_,n=_query_forward(packet,q)
    qpos=P+n-1
    layer=layers[PTR_L]
    h=o.hidden_states[PTR_L][0,-1].to(layer.input_layernorm.weight.dtype)
    hn=layer.input_layernorm(h)
    qv=layer.self_attn.q_proj(hn).view(1,QH,1,HD)
    ak=inst[PTR_L][0].repeat_interleave(REP,dim=0).unsqueeze(0)
    dummy=torch.zeros((1,QH,1,HD),device=DEVICE,dtype=qv.dtype)
    c,s=_rope_cos_sin(dummy,[qpos])
    qrot,_=apply_rotary_pos_emb(qv,dummy,c,s,unsqueeze_dim=1)
    score=torch.einsum("bhqd,bhkd->bhqk",qrot.float(),ak.float()).squeeze(0).squeeze(1)/math.sqrt(HD)
    score[:,0]=-torch.inf
    W=torch.softmax(score,dim=-1).detach().cpu()
    W[:,0]=0
    W=W/W.sum(dim=1,keepdim=True).clamp_min(1e-12)
    return W[PTR_H]

def _pointer_vector(raw,w):
    x=raw[ADDR_L][1].float().cpu()
    if x.shape[1]!=len(w):raise RuntimeError(f"Pointer length {len(w)} != V length {x.shape[1]}")
    return (x*w[None,:,None]).sum(dim=1).reshape(-1)

def _cos_rows(v,M):
    v=v.float();M=M.float();nv=v.norm()
    if float(nv)<=1e-12:return torch.zeros(M.shape[0],dtype=torch.float32)
    return (M/M.norm(dim=1,keepdim=True).clamp_min(1e-8))@(v/nv)

def _B_scores(pointer):
    return [float(_cos_rows(pointer,M).max()) for M in B_MATS]

def _rank(scores,gold_index=None):
    order=sorted(range(N),key=lambda j:(-float(scores[j]),j))
    if gold_index is None:return None,order[0],order
    return order.index(gold_index)+1,order[0],order

def _uniform_pointer(T):
    w=torch.ones(T,dtype=torch.float32);w[0]=0
    return w/w.sum().clamp_min(1e-12)

def _shifted_pointer(w):
    n=len(w)-1
    if n<=1:return w.clone()
    z=torch.zeros_like(w)
    z[1:]=torch.roll(w[1:].clone(),shifts=max(1,n//2),dims=0)
    return z/z.sum().clamp_min(1e-12)

# ==================================================================================================
# PUBLIC API — UI SHOULD CALL THESE FUNCTIONS
# ==================================================================================================

def initialize_memory(verbose=True):
    global tok,model,cfg,layers,PAD,VOC,NL,H,QH,KVH,HD,REP,KVD
    global ITEMS,META,TARGETS,DISTRACT,A_RAW,A_PACK,B_MATS,GOLD_SPANS
    global INITIALIZED,WEIGHT_SENTINEL_INITIAL

    if INITIALIZED:return get_engine_status()

    t0=time.perf_counter()

    if verbose:
        print("="*120)
        print(ENGINE)
        print("SEALED TEST528 BACKEND · FROZEN MODEL · 128-WAY ASSOCIATIVE MEMORY")
        print("="*120)
        print("[1/4] Loading Mistral...")

    tv=tuple(int("".join(c for c in x if c.isdigit())or 0) for x in transformers.__version__.split(".")[:2])
    dtarg="dtype" if tv>=(4,56) else "torch_dtype"

    tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
    if not tok.is_fast:raise RuntimeError("Fast tokenizer required.")

    model=AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map={"":0},
        attn_implementation="sdpa",
        **{dtarg:torch.bfloat16}
    ).eval()

    for p in model.parameters():p.requires_grad_(False)

    cfg=model.config
    layers=model.model.layers
    NL=len(layers)
    H=cfg.hidden_size
    QH=cfg.num_attention_heads
    KVH=cfg.num_key_value_heads
    HD=getattr(cfg,"head_dim",None) or H//QH
    REP=QH//KVH
    KVD=KVH*HD
    PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    VOC=model.model.embed_tokens.weight.shape[0]

    if (NL,H,QH,KVH,HD)!=(32,4096,32,8,128):
        raise RuntimeError(f"Architecture mismatch: {(NL,H,QH,KVH,HD)}")
    if QH%KVH:raise RuntimeError("QH must be divisible by KVH.")
    if KVD!=1024:raise RuntimeError(f"Address dimension mismatch: {KVD}")

    WEIGHT_SENTINEL_INITIAL=_sentinel()

    pool=_native_code_pool()
    rr=random.Random(SEED+100000);rr.shuffle(pool)
    if len(pool)<N+32:raise RuntimeError(f"Need at least {N+32} native codes, found {len(pool)}.")

    TARGETS=pool[:N]
    DISTRACT=pool[N:]

    if len(set(TARGETS))!=N or set(TARGETS)&set(DISTRACT):
        raise RuntimeError("Native code partition failure.")
    if not all(len(_enc(x))==1 for x in TARGETS+DISTRACT):
        raise RuntimeError("Non-single-token seal detected.")

    ITEMS=_make_items()

    if verbose:
        print(f"      Model       : {MODEL_ID}")
        print(f"      GPU         : {torch.cuda.get_device_name(0)}")
        print(f"      Architecture: {NL}L H={H} QH={QH} KVH={KVH} HD={HD}")
        print(f"      Pointer     : L{PTR_L:02d}H{PTR_H:02d}")
        print(f"      Address     : L{ADDR_L:02d}-{ADDR_KIND} UNCENTERED · {KVD}D")
        print(f"      Candidates  : {N}")
        print(f"      Native pool : {len(pool)}")
        print("[2/4] Forging A/B machine-native memories...")

    A_RAW=[];A_PACK=[];B_MATS=[];GOLD_SPANS=[];META=[]
    B_RAW=[]

    for i,it in enumerate(ITEMS):
        ar=_forge(it["A"])
        br=_forge(it["B"])
        A_RAW.append(ar)
        A_PACK.append(_install(ar))
        B_RAW.append(br)
        GOLD_SPANS.append(_gold_token_positions(it["A"],it["seal"]))
        META.append({
            "id":i+1,
            "entity":it["entity"],
            "gold_seal":it["seal"],
            "d1":it["A_distractors"][0],
            "d2":it["A_distractors"][1]
        })

    if verbose:print("[3/4] Building frozen L00-V B memory field...")

    for raw in B_RAW:
        x=raw[ADDR_L][1][:,1:,:].float().cpu()
        B_MATS.append(x.permute(1,0,2).reshape(x.shape[1],-1).contiguous())

    if len(B_MATS)!=N or not all(M.shape[1]==KVD for M in B_MATS):
        raise RuntimeError("B memory bank construction failure.")

    # Source B text is no longer required by the live retrieval mechanism.
    del B_RAW
    gc.collect()
    torch.cuda.empty_cache()

    INITIALIZED=True

    if verbose:
        print("[4/4] Engine ready.")
        print(f"      128 A memories       : READY")
        print(f"      128 B memories       : READY")
        print(f"      B live forwards      : 0")
        print(f"      Trainable tensors    : {sum(int(p.requires_grad) for p in model.parameters())}")
        print(f"      Weight sentinel      : PASS")
        print(f"      Initialization time  : {time.perf_counter()-t0:.2f}s")
        print("="*120)

    return get_engine_status()

def get_engine_status():
    if not INITIALIZED:
        return {
            "initialized":False,
            "engine":ENGINE,
            "model":MODEL_ID,
            "candidates":N,
            "pointer":"L28H00",
            "address":"L00-V UNCENTERED",
            "address_dim":1024,
            "test528_lock":TEST528_LOCK,
            "test528_result_sha":TEST528_RESULT_SHA
        }

    current=_sentinel()

    return {
        "initialized":True,
        "engine":ENGINE,
        "model":MODEL_ID,
        "gpu":torch.cuda.get_device_name(0),
        "layers":NL,
        "hidden_size":H,
        "query_heads":QH,
        "kv_heads":KVH,
        "head_dim":HD,
        "pointer_layer":PTR_L,
        "pointer_head":PTR_H,
        "pointer":"L28H00",
        "address_layer":ADDR_L,
        "address_kind":ADDR_KIND,
        "address_centered":ADDR_CENTERED,
        "address":"L00-V UNCENTERED",
        "address_dim":KVD,
        "candidate_memories":N,
        "model_frozen":current==WEIGHT_SENTINEL_INITIAL,
        "trainable_tensors":sum(int(p.requires_grad) for p in model.parameters()),
        "query_time_candidate_B_forwards":0,
        "training":"NONE",
        "lora":"NONE",
        "optimizer":"NONE",
        "dra":"NONE",
        "learned_router":"NONE",
        "coordinate_discovery":"NONE",
        "test528_lock":TEST528_LOCK,
        "test528_result_sha":TEST528_RESULT_SHA
    }

def get_memory_field():
    if not INITIALIZED:raise RuntimeError("Call initialize_memory() first.")
    return [{
        "candidate":i+1,
        "entity":META[i]["entity"],
        "target_seal":META[i]["gold_seal"],
        "B_tokens":int(B_MATS[i].shape[0]),
        "address_dim":int(B_MATS[i].shape[1])
    } for i in range(N)]

def get_demo_items():
    if not INITIALIZED:raise RuntimeError("Call initialize_memory() first.")
    return [{
        "item":i+1,
        "entity":META[i]["entity"],
        "question":_question(META[i]["entity"]),
        "expected_candidate":i+1,
        "expected_seal":META[i]["gold_seal"]
    } for i in range(N)]

def retrieve(item_id):
    if not INITIALIZED:raise RuntimeError("Call initialize_memory() first.")

    i=int(item_id)-1
    if not 0<=i<N:raise ValueError(f"item_id must be 1..{N}")

    t0=time.perf_counter()
    entity=META[i]["entity"]
    question=_question(entity)

    w=_frozen_pointer(A_PACK[i],question)
    pointer=_pointer_vector(A_RAW[i],w)
    scores=_B_scores(pointer)
    r,selected,order=_rank(scores,i)

    span=GOLD_SPANS[i]
    argmax_pos=int(torch.argmax(w))
    ptr_hit=argmax_pos in span
    ptr_mass=float(w[span].sum())

    top=[
        {
            "rank":k+1,
            "candidate":j+1,
            "entity":META[j]["entity"],
            "seal":META[j]["gold_seal"],
            "score":float(scores[j]),
            "is_expected":j==i
        }
        for k,j in enumerate(order[:10])
    ]

    margin=float(scores[order[0]]-scores[order[1]])

    return {
        "mode":"PRIMARY",
        "item":i+1,
        "entity":entity,
        "question":question,
        "expected_candidate":i+1,
        "expected_seal":META[i]["gold_seal"],
        "selected_candidate":selected+1,
        "selected_entity":META[selected]["entity"],
        "selected_seal":META[selected]["gold_seal"],
        "correct":selected==i,
        "rank":r,
        "top1_score":float(scores[order[0]]),
        "top2_score":float(scores[order[1]]),
        "top1_top2_margin":margin,
        "pointer_hit":bool(ptr_hit),
        "pointer_mass":ptr_mass,
        "pointer_argmax_position":argmax_pos,
        "pointer_weights":[float(x) for x in w.tolist()],
        "candidate_scores":[float(x) for x in scores],
        "top10":top,
        "latency_seconds":time.perf_counter()-t0,
        "query_time_candidate_B_forwards":0,
        "model_frozen":_sentinel()==WEIGHT_SENTINEL_INITIAL
    }

def counterfactual(item_id,target_id=None):
    if not INITIALIZED:raise RuntimeError("Call initialize_memory() first.")

    i=int(item_id)-1
    if not 0<=i<N:raise ValueError(f"item_id must be 1..{N}")

    j=(i+53)%N if target_id is None else int(target_id)-1
    if not 0<=j<N:raise ValueError(f"target_id must be 1..{N}")
    if j==i:raise ValueError("Counterfactual target must differ from source item.")

    t0=time.perf_counter()
    entity=META[i]["entity"]
    target=META[j]["gold_seal"]

    names=_build_names(N*3)
    ents=[entity,names[3*i+1],names[3*i+2]]
    seals=[target,META[i]["d1"],META[i]["d2"]]
    rows=[f"Instrument {ents[x]} carries seal {seals[x]}." for x in range(3)]
    random.Random(SEED+90000+i*149).shuffle(rows)
    cfA=" ".join(rows)

    raw=_forge(cfA)
    pack=_install(raw)
    question=_question(entity)
    w=_frozen_pointer(pack,question)
    pointer=_pointer_vector(raw,w)
    scores=_B_scores(pointer)
    r,selected,order=_rank(scores,j)

    span=_gold_token_positions(cfA,target)
    argmax_pos=int(torch.argmax(w))
    ptr_hit=argmax_pos in span
    ptr_mass=float(w[span].sum())

    top=[
        {
            "rank":k+1,
            "candidate":x+1,
            "entity":META[x]["entity"],
            "seal":META[x]["gold_seal"],
            "score":float(scores[x]),
            "is_expected":x==j
        }
        for k,x in enumerate(order[:10])
    ]

    result={
        "mode":"COUNTERFACTUAL",
        "source_item":i+1,
        "entity":entity,
        "question":question,
        "original_candidate":i+1,
        "original_seal":META[i]["gold_seal"],
        "counterfactual_candidate":j+1,
        "counterfactual_seal":target,
        "selected_candidate":selected+1,
        "selected_entity":META[selected]["entity"],
        "selected_seal":META[selected]["gold_seal"],
        "followed_counterfactual":selected==j,
        "rank":r,
        "top1_score":float(scores[order[0]]),
        "top2_score":float(scores[order[1]]),
        "top1_top2_margin":float(scores[order[0]]-scores[order[1]]),
        "pointer_hit":bool(ptr_hit),
        "pointer_mass":ptr_mass,
        "pointer_argmax_position":argmax_pos,
        "pointer_weights":[float(x) for x in w.tolist()],
        "candidate_scores":[float(x) for x in scores],
        "top10":top,
        "latency_seconds":time.perf_counter()-t0,
        "query_time_candidate_B_forwards":0,
        "model_frozen":_sentinel()==WEIGHT_SENTINEL_INITIAL
    }

    del raw,pack,pointer
    gc.collect()
    torch.cuda.empty_cache()
    return result

def run_controls(item_id):
    if not INITIALIZED:raise RuntimeError("Call initialize_memory() first.")

    i=int(item_id)-1
    if not 0<=i<N:raise ValueError(f"item_id must be 1..{N}")

    question=_question(META[i]["entity"])
    w=_frozen_pointer(A_PACK[i],question)

    primary_pointer=_pointer_vector(A_RAW[i],w)
    uniform_pointer=_pointer_vector(A_RAW[i],_uniform_pointer(len(w)))
    shifted_pointer=_pointer_vector(A_RAW[i],_shifted_pointer(w))
    zero_pointer=torch.zeros(KVD,dtype=torch.float32)

    modes={
        "PRIMARY":primary_pointer,
        "UNIFORM":uniform_pointer,
        "SHIFTED":shifted_pointer,
        "NO_A":zero_pointer
    }

    out={}
    for name,p in modes.items():
        scores=_B_scores(p)
        r,se,order=_rank(scores,i)
        out[name]={
            "rank":r,
            "selected_candidate":se+1,
            "selected_seal":META[se]["gold_seal"],
            "correct":se==i,
            "top1_score":float(scores[order[0]]),
            "top2_score":float(scores[order[1]]),
            "margin":float(scores[order[0]]-scores[order[1]]),
            "candidate_scores":[float(x) for x in scores]
        }

    return {
        "item":i+1,
        "entity":META[i]["entity"],
        "expected_candidate":i+1,
        "expected_seal":META[i]["gold_seal"],
        "controls":out,
        "query_time_candidate_B_forwards":0,
        "model_frozen":_sentinel()==WEIGHT_SENTINEL_INITIAL
    }

def get_pointer_data(item_id):
    if not INITIALIZED:raise RuntimeError("Call initialize_memory() first.")

    i=int(item_id)-1
    if not 0<=i<N:raise ValueError(f"item_id must be 1..{N}")

    w=_frozen_pointer(A_PACK[i],_question(META[i]["entity"]))
    span=GOLD_SPANS[i]

    return {
        "item":i+1,
        "entity":META[i]["entity"],
        "seal":META[i]["gold_seal"],
        "pointer_layer":PTR_L,
        "pointer_head":PTR_H,
        "weights":[float(x) for x in w.tolist()],
        "argmax_position":int(torch.argmax(w)),
        "gold_span_positions":[int(x) for x in span],
        "pointer_hit":int(torch.argmax(w)) in span,
        "pointer_mass":float(w[span].sum())
    }

def verify_engine():
    if not INITIALIZED:raise RuntimeError("Call initialize_memory() first.")

    s=_sentinel()
    return {
        "weight_sentinel_pass":s==WEIGHT_SENTINEL_INITIAL,
        "initial_weight_sentinel":WEIGHT_SENTINEL_INITIAL,
        "current_weight_sentinel":s,
        "trainable_tensors":sum(int(p.requires_grad) for p in model.parameters()),
        "pointer":"L28H00",
        "address":"L00-V UNCENTERED",
        "address_dim":KVD,
        "candidate_count":len(B_MATS),
        "query_time_candidate_B_forwards":0,
        "test528_lock":TEST528_LOCK,
        "test528_result_sha":TEST528_RESULT_SHA
    }

# ==================================================================================================
# OPTIONAL STANDALONE SMOKE TEST
# ==================================================================================================

if __name__=="__main__":
    initialize_memory(verbose=True)

    print("\nDEMO ENGINE SMOKE TEST")
    print("-"*120)

    x=retrieve(1)
    print(f"PRIMARY ITEM 001")
    print(f"Question : {x['question']}")
    print(f"Expected : candidate {x['expected_candidate']:03d} / {x['expected_seal']}")
    print(f"Selected : candidate {x['selected_candidate']:03d} / {x['selected_seal']}")
    print(f"Rank     : {x['rank']}")
    print(f"Pointer  : hit={x['pointer_hit']} mass={x['pointer_mass']:.4f}")
    print(f"Margin   : {x['top1_top2_margin']:+.6f}")
    print(f"Frozen   : {x['model_frozen']}")

    print("-"*120)

    y=counterfactual(1)
    print(f"COUNTERFACTUAL ITEM 001")
    print(f"Original : candidate {y['original_candidate']:03d} / {y['original_seal']}")
    print(f"Changed  : candidate {y['counterfactual_candidate']:03d} / {y['counterfactual_seal']}")
    print(f"Selected : candidate {y['selected_candidate']:03d} / {y['selected_seal']}")
    print(f"Rank     : {y['rank']}")
    print(f"Follow   : {y['followed_counterfactual']}")
    print(f"Pointer  : hit={y['pointer_hit']} mass={y['pointer_mass']:.4f}")
    print(f"Frozen   : {y['model_frozen']}")

    print("-"*120)

    z=verify_engine()
    print("ENGINE VERIFICATION")
    for k,v in z.items():print(f"{k:32s}: {v}")

    print("="*120)
    print("MAM_MISTRAL_128_DEMO_ENGINE READY")
    print("UI API:")
    print("  initialize_memory()")
    print("  get_engine_status()")
    print("  get_demo_items()")
    print("  get_memory_field()")
    print("  retrieve(item_id)")
    print("  counterfactual(item_id, target_id=None)")
    print("  run_controls(item_id)")
    print("  get_pointer_data(item_id)")
    print("  verify_engine()")
    print("="*120)'''
ENGINE_SHA256=hashlib.sha256(ENGINE_SRC.encode("utf-8")).hexdigest()
if ENGINE_SHA256!=ENGINE_SHA256_EXPECTED:
    raise RuntimeError(f"Embedded engine source differs from the supplied engine file (SHA-256 {ENGINE_SHA256} != {ENGINE_SHA256_EXPECTED}). Re-paste PART 1 unchanged.")
say(f"[2/5] Engine source verified · {ENGINE_FILE} · SHA-256 {ENGINE_SHA256}")
STARTUP_UTC=utc_now()
if"ENG"in globals()and getattr(globals()["ENG"],"INITIALIZED",False)and getattr(globals()["ENG"],"__engine_sha256__",None)==ENGINE_SHA256:
    say("[3/5] Engine already initialised in this runtime — reusing it (no second model load).");ENGINE_INIT_SECONDS=globals().get("ENGINE_INIT_SECONDS",float("nan"))
else:
    say("[3/5] Executing the engine in its own module namespace and calling initialize_memory() …")
    ENG=types.ModuleType("mam_mistral_128_demo_engine");ENG.__file__=ENGINE_FILE+" (embedded verbatim)";sys.modules[ENG.__name__]=ENG
    exec(compile(ENGINE_SRC,ENGINE_FILE,"exec"),ENG.__dict__)
    ENG.__engine_sha256__=ENGINE_SHA256
    torch.cuda.synchronize();_t=time.perf_counter();ENG.initialize_memory(verbose=True);torch.cuda.synchronize();ENGINE_INIT_SECONDS=time.perf_counter()-_t
# ---------------- measurement adapter (read-only with respect to the engine; never alters retrieval) ----------------
def hook_module(fn):
    f=getattr(fn,"func",fn);return str(getattr(f,"__module__",None)or type(f).__module__ or"")
def hook_stats():
    tot=0;foreign=Counter()
    for m in ENG.model.modules():
        for d in(m._forward_hooks,m._forward_pre_hooks,m._backward_hooks):
            for fn in d.values():
                tot+=1;mod=hook_module(fn)
                if not mod.startswith(("transformers","accelerate","torch")):foreign[mod]+=1
    return tot,dict(foreign)
def optimizer_present():return any(isinstance(v,torch.optim.Optimizer)for v in list(globals().values())+list(ENG.__dict__.values()))
class Instrument:
    """Thin adapter around the engine's public API, plus measurement. Every scientific quantity comes from the engine itself."""
    def __init__(self,init_seconds):
        m=ENG.model;self.kind="ENGINE"
        self.info=dict(model_id=ENG.MODEL_ID,arch=[ENG.NL,ENG.H,ENG.QH,ENG.KVH,ENG.HD],rep=ENG.REP,kvd=ENG.KVD,dtype=str(next(m.parameters()).dtype).replace("torch.",""),
            attn=getattr(ENG.cfg,"_attn_implementation",None),gpu=torch.cuda.get_device_name(0),gpu_total_gib=torch.cuda.get_device_properties(0).total_memory/2**30,
            torch=torch.__version__,transformers=transformers.__version__,gradio=gr.__version__,python=sys.version.split()[0],platform=platform.platform(),
            params=sum(p.numel()for p in m.parameters()),vocab=int(ENG.VOC),pad_id=int(ENG.PAD),engine_file=ENGINE_FILE,engine_sha256=ENGINE_SHA256,
            init_seconds=init_seconds,startup_utc=STARTUP_UTC,sentinel0=ENG.WEIGHT_SENTINEL_INITIAL,
            sentinel_method="engine _sentinel(): SHA-256 over 16 slices × 256 values of 7 weight tensors",
            guard_method="all-parameter guard: SHA-256 over (name, shape, dtype, sum, |sum|, sum of squares) of every parameter tensor",
            forward_counter="counting-only forward pre-hook on the causal-LM module, attached only during an engine API call")
        self.info["guard0"]=self.guard();self.info["hooks0"]=hook_stats()[0]
        self._items=ENG.get_demo_items();self._field=ENG.get_memory_field()
    def status(self):return ENG.get_engine_status()
    def verify(self):return ENG.verify_engine()
    def items(self):return self._items
    def field(self):return self._field
    def config(self):
        return dict(PTR_L=ENG.PTR_L,PTR_H=ENG.PTR_H,ADDR_L=ENG.ADDR_L,ADDR_KIND=ENG.ADDR_KIND,ADDR_CENTERED=ENG.ADDR_CENTERED,N=ENG.N,KVD=ENG.KVD,MODEL_ID=ENG.MODEL_ID,
                    n_bmats=len(ENG.B_MATS),bmat_dims=sorted({int(M.shape[1])for M in ENG.B_MATS}),seed=ENG.SEED,fmt=ENG.FMT,
                    engine_sha256=hashlib.sha256(ENGINE_SRC.encode("utf-8")).hexdigest(),test528_lock=ENG.TEST528_LOCK,test528_result_sha=ENG.TEST528_RESULT_SHA)
    def guard(self):
        h=hashlib.sha256()
        with torch.inference_mode():
            for name,p in ENG.model.named_parameters():
                x=p.detach().float();vals=(float(x.sum().item()),float(x.abs().sum().item()),float((x*x).sum().item()))
                h.update(name.encode());h.update(str(tuple(p.shape)).encode());h.update(str(p.dtype).encode());h.update(np.asarray(vals,dtype=np.float64).tobytes());del x
        torch.cuda.empty_cache();return h.hexdigest()
    def frozen(self):
        tot,fo=hook_stats()
        return dict(training=bool(ENG.model.training),trainable_tensors=sum(int(p.requires_grad)for p in ENG.model.parameters()),
                    lora=bool(hasattr(ENG.model,"peft_config")or any("lora"in n.lower()for n,_ in ENG.model.named_modules())),optimizer=optimizer_present(),
                    hooks=tot,foreign_hooks=sum(fo.values()),foreign_modules=fo,sentinel=ENG._sentinel())
    def _call(self,fn,*args):
        rec=[]
        def _count(mod,a,kw):
            ids=kw.get("input_ids",a[0]if a else None);pkv=kw.get("past_key_values");pl=None
            if pkv is not None:
                try:pl=int(pkv.get_seq_length())
                except Exception:pl=-1
            rec.append(dict(input_len=(int(ids.shape[-1])if ids is not None else -1),has_past=pkv is not None,past_len=pl))
        h=ENG.model.register_forward_pre_hook(_count,with_kwargs=True)
        try:torch.cuda.synchronize();t=time.perf_counter();res=fn(*args);torch.cuda.synchronize();dt_=time.perf_counter()-t
        finally:h.remove()
        return res,rec,dt_
    def retrieve(self,i):return self._call(ENG.retrieve,i)
    def pointer_data(self,i):return self._call(ENG.get_pointer_data,i)
    def controls(self,i):return self._call(ENG.run_controls,i)
    def counterfactual(self,i,j):return self._call(ENG.counterfactual,i,j)
    def question_tokens(self,q):return len(ENG._enc(ENG.FMT.format(q=q)))
    def a_slots(self,i):return int(ENG.A_RAW[i-1][0][0].shape[1])
    def extras(self,i,w,sel):
        """Display-only quantities computed with the engine's own functions after retrieval (never fed back into retrieval)."""
        wt=torch.tensor([float(x)for x in w],dtype=torch.float32);p=ENG._pointer_vector(ENG.A_RAW[i-1],wt)
        M=ENG.B_MATS[sel-1];c=ENG._cos_rows(p,M);s=int(torch.argmax(c))
        ids=[ENG.PAD]+ENG._enc(ENG.ITEMS[i-1]["A"]+ENG.SEP);toks=["[PAD]"]+[ENG.tok.decode([t])for t in ids[1:]]
        return dict(p=[float(x)for x in p.tolist()],best_row=[float(x)for x in M[s].tolist()],best_slot=s+1,best_cos=float(c[s]),b_rows=int(M.shape[0]),
                    tokens=(toks if len(toks)==len(w)else None),shifted=[float(x)for x in ENG._shifted_pointer(wt).tolist()],uniform=[float(x)for x in ENG._uniform_pointer(len(w)).tolist()])
    def panel_strings(self):
        words=set(ENG._build_names(ENG.N*3))
        for it in ENG.ITEMS:words|=set(re.findall(r"(?<![A-Za-z])[A-Z]{2,8}(?![A-Za-z])",it["A"]+" "+it["B"]))
        words|=set(ENG.CLASS_LABELS);return sorted(words)
    def sync(self):torch.cuda.synchronize()
    def reset_peak(self):torch.cuda.reset_peak_memory_stats()
    def peak_gib(self):return torch.cuda.max_memory_allocated()/2**30
say("[4/5] Measurement adapter · all-parameter weight guard …")
INSTR=Instrument(ENGINE_INIT_SECONDS)
_cfg=INSTR.config();_st=INSTR.status();_fz=INSTR.frozen()
ENGINE_LOCK=[("engine source SHA-256 = supplied engine file",_cfg["engine_sha256"]==ENGINE_SHA256_EXPECTED),
 ("model = mistralai/Mistral-7B-Instruct-v0.3",_cfg["MODEL_ID"]==FROZEN_CONFIG["MODEL_ID"]),("architecture 32L / H4096 / 32Q / 8KV / HD128",tuple(INSTR.info["arch"])==ARCH),
 ("pointer L28H00",(_cfg["PTR_L"],_cfg["PTR_H"])==(28,0)),("address L00-V uncentered",(_cfg["ADDR_L"],_cfg["ADDR_KIND"],_cfg["ADDR_CENTERED"])==(0,"V",False)),
 ("address dimension 1024",_cfg["KVD"]==1024 and _cfg["bmat_dims"]==[1024]),("128 B memories",_cfg["N"]==128 and _cfg["n_bmats"]==128),
 ("engine status: initialised, pointer and address strings as sealed",_st.get("initialized")and _st.get("pointer")=="L28H00"and _st.get("address")=="L00-V UNCENTERED"),
 ("TEST528 lock and result SHA in engine = archived values",(_cfg["test528_lock"],_cfg["test528_result_sha"])==(TEST528["lock"],TEST528["result_sha"])),
 ("weight sentinel = value at initialisation",_fz["sentinel"]==INSTR.info["sentinel0"]),("trainable tensors = 0",_fz["trainable_tensors"]==0),
 ("model in eval mode, no LoRA, no optimizer",(not _fz["training"])and not _fz["lora"]and not _fz["optimizer"]),("no foreign forward/backward hooks",_fz["foreign_hooks"]==0)]
_bad=[n for n,ok in ENGINE_LOCK if not ok]
if _bad:raise RuntimeError(f"ENGINE LOCK FAILED: {_bad}")
say(f"[5/5] Engine lock PASS · {len(ENGINE_LOCK)} checks · init {ENGINE_INIT_SECONDS:.1f}s · guard {INSTR.info['guard0'][:24]}… · sentinel {INSTR.info['sentinel0'][:24]}…")
say("PART 1 / 3 complete. Now run PART 2 / 3 in the next cell.");say("="*140)
MAM_MISTRAL_PART1_OK=True
#<<ENGINE_END>>
# =====================================================================================================================================
# AKBASCORE MAM · MISTRAL-7B ASSOCIATIVE RETRIEVAL INSTRUMENT — PART 2 / 3 · MEASUREMENT PIPELINE AND FIGURES
# Same demo as PART 1. Run this cell after PART 1 / 3, in the same Colab runtime; then run PART 3 / 3.
# Defines the fail-closed measurement run (engine API calls → re-derivation → sealed payload, logs, ZIP) and six figures
# (JPEG 300 dpi + vector PDF). Nothing in this part changes the engine or its retrieval computation.
# =====================================================================================================================================
if not globals().get("MAM_MISTRAL_PART1_OK"):raise RuntimeError("PART 1 / 3 has not been run in this runtime. Run PART 1, then PART 2, then PART 3.")
#<<CORE_BEGIN>>
import unicodedata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.patches import Rectangle,FancyBboxPatch,FancyArrowPatch,Patch
from matplotlib.lines import Line2D
from PIL import Image
ROOT=Path("/content/AKBASCORE_MAM_MISTRAL")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_MAM_MISTRAL")
ROOT.mkdir(parents=True,exist_ok=True);ALLOWED_PATHS=[str(ROOT)]
N_STAGES=9;CF_OFFSET=53
def default_cf_target(item):return((item-1+CF_OFFSET)%N_CAND)+1
def ev(stage,title,body,done=None,total=None):return dict(stage=stage,title=title,body=body,done=done,total=total,eta=None)
def file_ready(p):return p is not None and Path(p).is_file()and Path(p).stat().st_size>0
def prune_runs(keep=4):
    runs=sorted([p for p in ROOT.glob("MAM528DEMO-*")if p.is_dir()],key=lambda p:p.stat().st_mtime)
    for p in runs[:max(0,len(runs)-keep)]:shutil.rmtree(p,ignore_errors=True)
def maxabs(a,b):return max((abs(float(x)-float(y))for x,y in zip(a,b)),default=0.0)
def fwd_ok_query(rec,nq,TA):return len(rec)==1 and rec[0]["has_past"]and rec[0]["input_len"]==nq and rec[0]["past_len"]==TA
def post_seal_audit(imgs,pdfs,zp,member_names,pp,sha,tp,mp,verdict):
    checks=[]
    def ok(name,cond):
        checks.append(name)
        if not cond:raise AuditFail("POST-SEAL AUDIT FAILED: "+name)
    ok(f"{N_FIGS}/{N_FIGS} figures created (JPEG)",len(imgs)==N_FIGS and all(file_ready(p)for p,_ in imgs));ok(f"{N_FIGS}/{N_FIGS} vector PDF copies",len(pdfs)==N_FIGS and all(file_ready(p)for p in pdfs))
    bad=[]
    for p,_ in imgs:
        with Image.open(p)as im:
            if not(im.format=="JPEG"and im.mode=="RGB"and im.size[0]>=3600):bad.append(p.name)
    ok(f"{N_FIGS}/{N_FIGS} figures are JPEG/RGB at ≥3600 px width",not bad)
    with zipfile.ZipFile(zp)as z:
        ok("ZIP testzip()",z.testzip()is None);nm=z.namelist()
        ok("ZIP is flat (no sub-folders)",all("/"not in n for n in nm));ok("ZIP contents = expected package",sorted(nm)==sorted(member_names))
        jp=sorted(n for n in nm if n.lower().endswith(".jpg"));ok("figure numbering 01..%02d"%N_FIGS,[n[4:6]for n in jp]==[f"{i:02d}"for i in range(1,N_FIGS+1)])
    ok("payload SHA-256 recomputation",hashlib.sha256(pp.read_bytes()).hexdigest()==sha);ok("payload JSON parses",isinstance(json.loads(pp.read_bytes().decode("utf-8")),dict))
    m=json.loads(mp.read_text(encoding="utf-8"));ok("manifest hash and verdict match payload",m["payload_sha256"]==sha and m["verdict"]==verdict)
    ok("readable log exists and is non-empty",file_ready(tp));ok("ZIP exists and is non-empty",file_ready(zp))
    return checks
def make_txt(P,sha,names,sealed_utc):
    o=[];a=o.append;S="="*140;Dd="-"*140;E_=P["environment"];L=P["live"];pr=L["primary"];D=L["primary_rederived"];ps=L["pointer_stats"];cf=L["counterfactual"];Dc=L["counterfactual_rederived"]
    fz=P["frozen"];cfg=P["engine"]["config"]
    a(S);a(f"{DEMO_TITLE} — READABLE RUN LOG");a(f"{AUTHOR} · {AUTHOR_PLACE} · {COPYRIGHT}");a(S)
    a(f"Derived from {names['payload']} (SHA-256 {sha}). Manifest: {names['manifest']}.")
    a("This is a live demonstration run of the frozen TEST528 mechanism. It is not TEST528; archived TEST528 values are reproduced verbatim, never recomputed.")
    for k_,v in(("RUN ID",P["run_id"]),("START UTC",P["run_start_utc"]),("END UTC",P["run_end_utc"]),("SEALED UTC",sealed_utc),("VERDICT",P["verdict"]),
        ("MODEL",f"{cfg['MODEL_ID']} · frozen · {P['model']['dtype']} · attention {P['model']['attn']}"),("ARCHITECTURE","32 layers · hidden 4096 · 32 Q heads · 8 KV heads · head 128 · GQA 4:1"),
        ("GPU",f"{E_['gpu']} (canonical: {CANON_GPU}; same: {P['hardware']['same']})"),("TORCH / TRANSFORMERS / GRADIO",f"{E_['torch']} / {E_['transformers']} / {E_['gradio']}"),
        ("ENGINE FILE",f"{P['engine']['file']} · SHA-256 {P['engine']['sha256']}"),("POINTER",f"L{cfg['PTR_L']:02d}H{cfg['PTR_H']:02d}"),
        ("ADDRESS",f"L{cfg['ADDR_L']:02d}-{cfg['ADDR_KIND']} · centered={cfg['ADDR_CENTERED']} · {cfg['KVD']} dimensions"),("CANDIDATES",f"{cfg['N']} precomputed B address matrices"),
        ("MATCH","max cosine over all rows of each B matrix; argmax over 128 candidates; ties → lower index"),("ENGINE SEED",cfg["seed"]),
        ("TEST528 LOCK",TEST528["lock"]),("TEST528 RESULT SHA",TEST528["result_sha"])):a(f"{k_:<30}: {v}")
    a("");a("ITEM");a(Dd)
    a(f"item {L['item']:03d} · entity {L['entity']} · question: {L['question']}");a(f"question tokens (FMT template) = {L['question_tokens']} · active A memory slots T = {L['a_slots']}")
    a(f"expected B candidate {L['item']:03d} · expected seal {pr['expected_seal']} (audit metadata; not an input to retrieval)")
    a("");a("PRIMARY RETRIEVAL · retrieve(item)  [LIVE]");a(Dd)
    a(f"selected B#{pr['selected_candidate']:03d} (seal {pr['selected_seal']}) · {'CORRECT'if pr['correct']else'INCORRECT'} · rank of expected {pr['rank']}/{N_CAND}")
    a(f"top1 {pr['top1_score']:+.6f} · top2 {pr['top2_score']:+.6f} · margin {pr['top1_top2_margin']:+.6f} · non-selected median {D['rest_median']:+.6f} · (top1−mean)/sd {('%.2f'%D['z_top1'])if D['z_top1']is not None else'n/a'}")
    a(f"pointer L28H00: argmax slot {ps['argmax']} of {ps['T']} · peak {ps['peak']:.4f} · entropy {ps['entropy_bits']:.3f} bits · effective slots {ps['effective_slots']:.2f}")
    a(f"diagnostic only: target-span positions {L['pointer_data']['gold_span_positions']} · pointer mass on span {pr['pointer_mass']:.4f} · pointer hit {pr['pointer_hit']}")
    a(f"latency (engine) {1000*pr['latency_seconds']:.1f} ms · measured model forwards {len(L['forwards']['retrieve'])}: {L['forwards']['retrieve']}")
    a("top-10: "+" ".join(f"#{t['candidate']:03d}:{t['score']:+.6f}"for t in pr["top10"]))
    a(f"repeat retrieve(): selected B#{L['repeat']['selected_candidate']:03d} · max |Δscore| {L['repeat_max_abs_diff']:.3e} · get_pointer_data() max |Δw| {L['pointer_data_max_abs_diff']:.3e}")
    ex=L["extras"];a(f"address p (1024-D) and best-matching row of selected B: slot {ex['best_slot']} of {ex['b_rows']} rows · cos {ex['best_cos']:+.6f} (recomputed with engine functions; equals reported score)")
    a("");a("CONTROLS · run_controls(item)  [LIVE, single item]");a(Dd)
    for k_,v in L["controls"]["controls"].items():
        d=L["controls_rederived"][k_];a(f"{k_:<8} selected B#{v['selected_candidate']:03d} · rank of expected {v['rank']:>3}/{N_CAND} · top1 {v['top1_score']:+.6f} · margin {v['margin']:+.6f}"+(" · all scores equal (zero address): argmax falls to B#001 by index"if d["all_equal"]else""))
    a(f"measured model forwards: {L['forwards']['controls']}")
    a("");a("COUNTERFACTUAL · counterfactual(item, target)  [LIVE]");a(Dd)
    a(f"original association: {L['entity']} → {cf['original_seal']} (B#{cf['original_candidate']:03d}) · counterfactual: {L['entity']} → {cf['counterfactual_seal']} (B#{cf['counterfactual_candidate']:03d})")
    a(f"selected B#{cf['selected_candidate']:03d} · followed counterfactual {cf['followed_counterfactual']} · rank of counterfactual target {cf['rank']}/{N_CAND} · rank of original B under counterfactual A {Dc['order'].index(cf['original_candidate'])+1}/{N_CAND}")
    a(f"top1 {cf['top1_score']:+.6f} · margin {cf['top1_top2_margin']:+.6f} · pointer mass on diagnostic span {cf['pointer_mass']:.4f} · hit {cf['pointer_hit']} · measured forwards {L['forwards']['counterfactual']}")
    a("");a("SEALED TEST528 REFERENCE (archived; not produced by this run)");a(Dd)
    for k_,lab in(("primary","PRIMARY"),("counterfactual","COUNTERFACTUAL"),("shifted","SHIFTED"),("no_a","NO-A")):v=TEST528[k_];a(f"{lab:<15} {fr(v)} = {v[2]} top-1")
    a(f"identity regime: {TEST528['identity_regime']}");a(f"cross-model context (archived): "+" | ".join(f"{c['record']} {c['model']} {c['pointer']} {c['address']} {c['dim']}-D {fr(c['top1'])} = {c['top1'][2]}"for c in CROSS_MODEL));a(CROSS_MODEL_NOTE)
    a("");a("MEASURED MODEL FORWARDS (instrumentation)");a(Dd)
    for k_,v in L["forwards"].items():a(f"{k_:<15}: {v}")
    a(f"every retrieval forward processed only the question tokens ({L['question_tokens']}) over the installed A cache ({L['a_slots']} slots); no B memory was installed or forwarded.")
    a("");a("ENGINE VERIFICATION · verify_engine() after the run");a(Dd)
    for k_,v in P["verify_after"].items():a(f"{k_:<34}: {v}")
    a("");a("FROZEN MODEL");a(Dd)
    for k_ in("sentinel_startup","sentinel_before","sentinel_after","guard_startup","guard_before","guard_after"):a(f"{k_:<18}: {fz[k_]}")
    a(f"trainable tensors={fz['trainable_tensors']} · lora={fz['lora']} · optimizer={fz['optimizer']} · training_mode={fz['training_mode']} · foreign hooks outside API calls {fz['foreign_hooks_before']}→{fz['foreign_hooks_after']}")
    a(fz["guard_method"]);a(fz["sentinel_method"])
    for t_,k_ in(("DEMONSTRATED IN THIS RUN","demonstrated"),("ARCHIVED","archived"),("NOT ESTABLISHED BY THIS RUN OR BY TEST528","not_established"),("TECHNICAL NOTES","notes")):
        a("");a(t_);a(Dd)
        for t in P["scope"][k_]:a("• "+t)
    a("");a("PRE-SEAL CHECKS");a(Dd)
    for c_ in P["checks_pre_seal"]:a("PASS · "+c_)
    a("");a("TIMING (seconds)");a(Dd)
    for k_,v in P["timing"].items():a(f"{k_:<24}: {v:.3f}")
    a(f"peak GPU memory {P['gpu']['peak_allocated_gib']:.2f} GiB");a("");a(f"All raw score vectors, pointer weights and the 1024-D address are in {names['payload']} and {names['jsonl']}.");a(S)
    return"\n".join(o)
def execute_run(I,ctl):
    """Fail-closed wrapper: on a technical audit failure the raw outputs gathered so far are preserved (never sealed, no figures)."""
    state={}
    try:return(yield from _execute_run(I,ctl,state))
    except AuditFail as ex:
        raw=state.get("raw")
        if raw and any(raw.values()):
            try:
                p=Path(ctl["run_dir"])/f"FAILED_AUDIT_RAW_{ctl['run_id']}.json"
                p.write_bytes(canon(dict(status="FAILED AUDIT — NOT SEALED",run_id=ctl["run_id"],failed_check=str(ex),checks_passed=state.get("checks",[]),raw=raw,note="Raw outputs preserved. No figures, no seal and no verdict were produced.")));ex.partial=str(p)
            except Exception:pass
        raise
def _execute_run(I,ctl,state):
    run_id=ctl["run_id"];run_dir=Path(ctl["run_dir"]);item=int(ctl["item"]);cft=int(ctl.get("cf_target")or default_cf_target(item))
    info=I.info;checks=[];tm=Counter();T0=time.perf_counter();state["checks"]=checks;raw=dict(primary=None,repeat=None,pointer=None,controls=None,counterfactual=None);state["raw"]=raw
    def chk(name,cond):
        checks.append(name)
        if not cond:raise AuditFail(name)
    run_start_utc=utc_now();say("="*140);say(f"RUN {run_id} · item {item:03d} · counterfactual target {cft:03d}");say("="*140)
    # ---- 1 integrity ----
    yield ev(1,"Integrity","Engine source hash, frozen configuration, weight sentinel, all-parameter guard, hooks.")
    I.reset_peak();cfg=I.config();st0=I.status();fz0=I.frozen();t=time.perf_counter();g_before=I.guard();tm["guard"]+=time.perf_counter()-t
    chk("engine source SHA-256 = supplied engine file",cfg["engine_sha256"]==ENGINE_SHA256_EXPECTED)
    chk("frozen configuration unchanged (L28H00 · L00-V · uncentered · 1024-D · 128 candidates · Mistral-7B-Instruct-v0.3)",all(cfg[k_]==v for k_,v in FROZEN_CONFIG.items())and cfg["n_bmats"]==128 and cfg["bmat_dims"]==[1024])
    chk("engine status reports L28H00 / L00-V UNCENTERED / 1024 / 128",st0.get("pointer")=="L28H00"and st0.get("address")=="L00-V UNCENTERED"and st0.get("address_dim")==1024 and st0.get("candidate_memories")==128)
    chk("archived TEST528 lock and result SHA match the engine constants",(cfg["test528_lock"],cfg["test528_result_sha"])==(TEST528["lock"],TEST528["result_sha"]))
    chk("weight sentinel before run = value at initialisation",fz0["sentinel"]==info["sentinel0"]);chk("all-parameter guard before run = value at startup",g_before==info["guard0"])
    chk("trainable tensors = 0, eval mode, no LoRA, no optimizer",fz0["trainable_tensors"]==0 and not fz0["training"]and not fz0["lora"]and not fz0["optimizer"])
    chk("no foreign forward/backward hooks outside API calls",fz0["foreign_hooks"]==0)
    chk("item and counterfactual target in 1..128 and distinct",1<=item<=N_CAND and 1<=cft<=N_CAND and cft!=item)
    its=I.items();it=its[item-1];question=it["question"];nq=I.question_tokens(question);TA=I.a_slots(item)
    chk("demo item metadata consistent (expected candidate = item)",it["item"]==item and it["expected_candidate"]==item)
    strip=I.panel_strings();state["strip"]=strip
    for c in checks:say("   PASS ·",c)
    # ---- 2 primary ----
    yield ev(2,"Primary retrieval",f"retrieve({item}) — {question}")
    r,fw_r,dt=I.retrieve(item);raw["primary"]=r;tm["retrieve"]+=dt
    chk("retrieve(): exactly one model forward — question tokens over the installed A cache only",fwd_ok_query(fw_r,nq,TA))
    D=rederive(r["candidate_scores"],item)
    chk("retrieve(): selection, rank, top-1/top-2 and margin re-derived from the 128 raw scores",D["selected"]==r["selected_candidate"]and D["rank"]==r["rank"]and D["top1"]==r["top1_score"]and D["top2"]==r["top2_score"]and abs(D["margin"]-r["top1_top2_margin"])<=1e-12)
    chk("retrieve(): question = engine template for the item; mode PRIMARY",r["question"]==question and r["mode"]=="PRIMARY"and r["item"]==item)
    w=r["pointer_weights"];ps=pointer_stats(w)
    chk("pointer: T weights over the A slots, slot 0 excluded, sum = 1",ps["T"]==TA and ps["w0"]==0.0 and abs(ps["sum"]-1)<1e-5 and ps["argmax"]==r["pointer_argmax_position"])
    chk("retrieve(): engine reports 0 query-time candidate-B forwards and frozen model",r["query_time_candidate_B_forwards"]==0 and r["model_frozen"])
    say(f"   selected B#{r['selected_candidate']:03d} · rank {r['rank']} · top1 {r['top1_score']:+.6f} · margin {r['top1_top2_margin']:+.6f} · {1000*dt:.1f} ms")
    # ---- 3 repeat + pointer telemetry ----
    yield ev(3,"Repeatability and pointer telemetry","retrieve() repeated; get_pointer_data(); address and best-matching B row recomputed with engine functions.")
    r2,fw_r2,dt=I.retrieve(item);raw["repeat"]=r2;tm["repeat"]+=dt;rep_diff=maxabs(r["candidate_scores"],r2["candidate_scores"])
    chk("repeat retrieve(): one forward of the same form; same selected candidate",fwd_ok_query(fw_r2,nq,TA)and r2["selected_candidate"]==r["selected_candidate"])
    pdat,fw_p,dt=I.pointer_data(item);raw["pointer"]=pdat;tm["pointer"]+=dt;w_diff=maxabs(w,pdat["weights"])
    chk("get_pointer_data(): one forward; same pointer argmax as retrieve()",fwd_ok_query(fw_p,nq,TA)and pdat["argmax_position"]==r["pointer_argmax_position"])
    ex=I.extras(item,w,r["selected_candidate"])
    chk("address recomputed with engine functions: 1024-D; best row cosine = reported score of the selected B",len(ex["p"])==ADDR_DIM and abs(ex["best_cos"]-r["candidate_scores"][r["selected_candidate"]-1])<=1e-6)
    # ---- 4 controls ----
    yield ev(4,"Controls","run_controls(): PRIMARY · UNIFORM · SHIFTED · NO-A on the same item.")
    c,fw_c,dt=I.controls(item);raw["controls"]=c;tm["controls"]+=dt
    chk("run_controls(): one forward of the query form",fwd_ok_query(fw_c,nq,TA))
    chk("run_controls(): modes PRIMARY, UNIFORM, SHIFTED, NO_A",sorted(c["controls"])==["NO_A","PRIMARY","SHIFTED","UNIFORM"])
    DC={k_:rederive(v["candidate_scores"],item)for k_,v in c["controls"].items()}
    chk("run_controls(): every mode re-derived from its raw scores",all(DC[k_]["selected"]==v["selected_candidate"]and DC[k_]["rank"]==v["rank"]for k_,v in c["controls"].items()))
    ctl_diff=maxabs(c["controls"]["PRIMARY"]["candidate_scores"],r["candidate_scores"])
    chk("run_controls() PRIMARY selects the same candidate as retrieve()",c["controls"]["PRIMARY"]["selected_candidate"]==r["selected_candidate"])
    # ---- 5 counterfactual ----
    yield ev(5,"Counterfactual",f"counterfactual({item}, {cft}) — A memory re-forged with the seal of candidate {cft:03d}.")
    f,fw_f,dt=I.counterfactual(item,cft);raw["counterfactual"]=f;tm["counterfactual"]+=dt
    chk("counterfactual(): two forwards — forge of the new A (no cache), then the question over that A",len(fw_f)==2 and not fw_f[0]["has_past"]and fw_f[1]["has_past"]and fw_f[1]["input_len"]==nq and fw_f[1]["past_len"]==fw_f[0]["input_len"])
    Dc=rederive(f["candidate_scores"],cft)
    chk("counterfactual(): selection and rank re-derived from raw scores",Dc["selected"]==f["selected_candidate"]and Dc["rank"]==f["rank"]and f["counterfactual_candidate"]==cft and f["original_candidate"]==item)
    # ---- 6 verification ----
    yield ev(6,"Engine verification","verify_engine(), weight sentinel, all-parameter guard, hooks.")
    v=I.verify();fz1=I.frozen();t=time.perf_counter();g_after=I.guard();tm["guard"]+=time.perf_counter()-t
    chk("verify_engine(): sentinel PASS · trainable 0 · 128 candidates · L28H00 · L00-V UNCENTERED · 1024",v["weight_sentinel_pass"]and v["trainable_tensors"]==0 and v["candidate_count"]==128 and v["pointer"]=="L28H00"and v["address"]=="L00-V UNCENTERED"and v["address_dim"]==1024)
    chk("all-parameter guard after run = before run = startup (weights unchanged)",g_after==g_before==info["guard0"]);chk("weight sentinel after run = value at initialisation",fz1["sentinel"]==info["sentinel0"])
    chk("trainable tensors = 0 and no foreign hooks after run",fz1["trainable_tensors"]==0 and fz1["foreign_hooks"]==0)
    fwd=dict(retrieve=fw_r,repeat=fw_r2,pointer_data=fw_p,controls=fw_c,counterfactual=fw_f);tm["engine_total"]=time.perf_counter()-T0
    verdict=f"ITEM_{item:03d}_PRIMARY_{'CORRECT'if r['correct']else'INCORRECT'}_CF_{'FOLLOWED'if f['followed_counterfactual']else'NOT_FOLLOWED'}_INTEGRITY_VERIFIED"
    say(f"   counterfactual → B#{f['selected_candidate']:03d} (target {cft:03d}) · verdict {verdict}")
    # ---- 7 seal ----
    yield ev(7,"Sealing","Payload, manifest, readable log and raw vectors are written and hashed.")
    peak=I.peak_gib();same_hw=info["gpu"]==CANON_GPU
    live=dict(item=item,entity=it["entity"],question=question,question_tokens=nq,a_slots=TA,primary=r,primary_rederived=D,pointer_stats=ps,repeat=r2,repeat_max_abs_diff=rep_diff,
              pointer_data=pdat,pointer_data_max_abs_diff=w_diff,extras=ex,controls=c,controls_rederived=DC,controls_primary_vs_retrieve_max_abs_diff=ctl_diff,
              counterfactual=f,counterfactual_rederived=Dc,cf_target=cft,cf_target_rank_under_original_A=D["order"].index(cft)+1,
              original_rank_under_counterfactual_A=Dc["order"].index(item)+1,forwards=fwd,b_tokens=[x["B_tokens"]for x in I.field()])
    P={"schema":"akbascore.mam.mistral.instrument.run.v1","demo":DEMO_TITLE,"author":AUTHOR,"place":AUTHOR_PLACE,"copyright":COPYRIGHT,"run_id":run_id,"run_start_utc":run_start_utc,"run_end_utc":utc_now(),
       "verdict":verdict,"model":{"id":cfg["MODEL_ID"],"arch":info["arch"],"dtype":info["dtype"],"attn":info["attn"],"params":info["params"],"vocab":info["vocab"],"pad_id":info["pad_id"]},
       "environment":{k_:info[k_]for k_ in("gpu","gpu_total_gib","torch","transformers","gradio","python","platform")},"hardware":{"canonical_gpu":CANON_GPU,"this_run_gpu":info["gpu"],"same":bool(same_hw)},
       "engine":{"file":info["engine_file"],"sha256":cfg["engine_sha256"],"config":cfg,"status_before":st0,"init_seconds":info["init_seconds"],"instrument":I.kind},
       "live":live,"verify_after":v,
       "frozen":{"sentinel_startup":info["sentinel0"],"sentinel_before":fz0["sentinel"],"sentinel_after":fz1["sentinel"],"guard_startup":info["guard0"],"guard_before":g_before,"guard_after":g_after,
                 "trainable_tensors":fz1["trainable_tensors"],"lora":fz1["lora"],"optimizer":fz1["optimizer"],"training_mode":fz1["training"],"foreign_hooks_before":fz0["foreign_hooks"],
                 "foreign_hooks_after":fz1["foreign_hooks"],"guard_method":info["guard_method"],"sentinel_method":info["sentinel_method"],"forward_counter":info["forward_counter"]},
       "archived":{"test528":TEST528,"cross_model":CROSS_MODEL,"cross_model_note":CROSS_MODEL_NOTE,"note":"Archived reference values; not produced or recomputed by this run."},
       "scope":SCOPE,"checks_pre_seal":list(checks),"timing":dict(tm),"gpu":{"peak_allocated_gib":peak}}
    P=jsafe(P);allow=(info["gpu"],CANON_GPU)
    hits=claim_scan("payload",strip_words(json.dumps(P,ensure_ascii=False),strip),allow=allow)
    chk("claim-discipline scan of the payload: 0 hits",not hits);P["claim_scan"]={"rules":len(CLAIM_RULES)+1,"hits":0,"scope":"payload JSON text; panel strings (names, seals) excluded"}
    pb=canon(P);sha=hashlib.sha256(pb).hexdigest();payload_name=f"{run_id}_FULL_RUN_LOG_payload.json";pp=run_dir/payload_name;t_seal=time.perf_counter();pp.write_bytes(pb);say("payload sealed:",sha)
    chk("payload SHA-256 verification after write",hashlib.sha256(pp.read_bytes()).hexdigest()==sha)
    names=dict(payload=payload_name,manifest=f"run_manifest_{run_id}.json",txt=f"{run_id}_READABLE_RUN_LOG.txt",summary=f"{run_id}_SUMMARY.json",jsonl=f"{run_id}_RAW_VECTORS.jsonl",images=f"figures_manifest_{run_id}.json",zip=f"AKBASCORE_MAM_MISTRAL_EVIDENCE_{run_id}.zip")
    mp=run_dir/names["manifest"];sealed_utc=utc_now()
    manifest={"demo":DEMO_TITLE,"author":AUTHOR,"copyright":COPYRIGHT,"run_id":run_id,"payload_file":payload_name,"payload_sha256":sha,"payload_bytes":len(pb),"sealed_utc":sealed_utc,"verdict":verdict,
              "engine_sha256":cfg["engine_sha256"],"test528_lock":TEST528["lock"],"test528_result_sha":TEST528["result_sha"],
              "canonicalization":"json.dumps(sort_keys=True, separators=(',',':'), ensure_ascii=False, allow_nan=False), UTF-8",
              "note":"Artifact integrity seal: confirms the payload is unchanged since sealing. It is not a scientific proof and not a third-party verification."}
    mp.write_text(json.dumps(manifest,indent=2,sort_keys=True,ensure_ascii=False),encoding="utf-8")
    summary={"run_id":run_id,"verdict":verdict,"item":item,"entity":it["entity"],"question":question,"expected_candidate":item,"expected_seal":r["expected_seal"],"selected_candidate":r["selected_candidate"],
             "selected_seal":r["selected_seal"],"rank":r["rank"],"top1":r["top1_score"],"top2":r["top2_score"],"margin":r["top1_top2_margin"],"pointer_hit_diagnostic":r["pointer_hit"],"pointer_mass_diagnostic":r["pointer_mass"],
             "controls":{k_:{"selected":v_["selected_candidate"],"rank":v_["rank"]}for k_,v_ in c["controls"].items()},"counterfactual":{"target":cft,"selected":f["selected_candidate"],"followed":f["followed_counterfactual"],"rank":f["rank"]},
             "measured_model_forwards":{k_:len(v_)for k_,v_ in fwd.items()},"weights_unchanged":True,"engine_sha256":cfg["engine_sha256"],"payload_sha256":sha,
             "archived_test528":{"primary":TEST528["primary"],"counterfactual":TEST528["counterfactual"],"shifted":TEST528["shifted"],"no_a":TEST528["no_a"]}}
    (run_dir/names["summary"]).write_text(json.dumps(jsafe(summary),indent=2,ensure_ascii=False),encoding="utf-8")
    lines=[{"op":"retrieve","item":item,"pointer_weights":w,"candidate_scores":r["candidate_scores"]},{"op":"address","item":item,"p_1024":ex["p"],"best_row_selected_B":ex["best_row"],"best_slot":ex["best_slot"]}]
    lines+=[{"op":"control","mode":k_,"candidate_scores":v_["candidate_scores"]}for k_,v_ in c["controls"].items()]
    lines+=[{"op":"counterfactual","item":item,"target":cft,"pointer_weights":f["pointer_weights"],"candidate_scores":f["candidate_scores"]}]
    (run_dir/names["jsonl"]).write_text("\n".join(json.dumps(jsafe(x),ensure_ascii=False)for x in lines)+"\n",encoding="utf-8")
    (run_dir/names["txt"]).write_text(make_txt(P,sha,names,sealed_utc),encoding="utf-8");seal_s=time.perf_counter()-t_seal
    # ---- 8 figures ----
    yield ev(8,"Rendering figures",f"{N_FIGS} figures at 300 dpi (JPEG) with vector PDF copies; every printed value is checked against the sealed payload.")
    ctx={"P":P,"sha":sha,"run_id":run_id,"payload_name":payload_name,"allow":allow,"strip":strip,"N":N_FIGS,"audit":[],"pdfs":[]}
    t_r=time.perf_counter();imgs=render_all(ctx,run_dir);render_s=time.perf_counter()-t_r;entries=[]
    for n_,(p,cap)in enumerate(imgs,1):
        with Image.open(p)as im:w_,h_=im.size
        q_=ctx["pdfs"][n_-1];entries.append({"index":n_,"jpg":p.name,"pdf":q_.name,"title":cap,"width_px":w_,"height_px":h_,"dpi":OUT_DPI,"jpg_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"pdf_sha256":hashlib.sha256(q_.read_bytes()).hexdigest()})
    imf=run_dir/names["images"]
    imf.write_text(json.dumps({"run_id":run_id,"payload_sha256":sha,"verdict":verdict,"count":len(entries),"render_seconds":render_s,"figures":entries,"figure_text_audit":ctx["audit"],
                               "note":"Per-file SHA-256 is an artifact integrity seal. Figure text was checked against the sealed payload and scanned for claim discipline before saving."},indent=2,ensure_ascii=False),encoding="utf-8")
    # ---- 9 package ----
    yield ev(9,"Packaging","ZIP with figures, PDFs, payload, manifest, logs and raw vectors; post-seal audit.")
    zp=run_dir/names["zip"];members=[p for p,_ in imgs]+list(ctx["pdfs"])+[imf,mp,pp,run_dir/names["txt"],run_dir/names["summary"],run_dir/names["jsonl"]]
    for m in members:
        if m.suffix in(".json",".txt",".jsonl"):chk(f"claim-discipline scan of {m.name}: 0 hits",not claim_scan(m.name,strip_words(m.read_text(encoding="utf-8"),strip),allow=allow))
    with zipfile.ZipFile(zp,"w",compression=zipfile.ZIP_DEFLATED)as z:
        for m in members:z.write(m,arcname=m.name)
    post=post_seal_audit(imgs,ctx["pdfs"],zp,[m.name for m in members],pp,sha,run_dir/names["txt"],mp,verdict)
    say(f"{run_id}: {len(checks)} pre-seal + {len(post)} post-seal checks = {len(checks)+len(post)}/{len(checks)+len(post)} PASS · figures {render_s:.1f}s · seal {seal_s:.2f}s")
    say("VERDICT :",verdict);say("PAYLOAD :",sha);say("ZIP     :",zp);say("="*140)
    ledger=dict(run_id=run_id,utc=sealed_utc,item=item,entity=it["entity"],selected=r["selected_candidate"],rank=r["rank"],correct=r["correct"],margin=r["top1_top2_margin"],cf_target=cft,
                cf_selected=f["selected_candidate"],cf_followed=f["followed_counterfactual"],controls={k_:v_["rank"]for k_,v_ in c["controls"].items()},forwards={k_:len(v_)for k_,v_ in fwd.items()},
                weights_unchanged=True,verdict=verdict,payload_sha256=sha)
    return dict(run_id=run_id,run_dir=run_dir,P=P,sha=sha,imgs=imgs,pdfs=ctx["pdfs"],zip=zp,payload=pp,txt=run_dir/names["txt"],manifest=mp,images_manifest=imf,summary=run_dir/names["summary"],
                jsonl=run_dir/names["jsonl"],checks=len(checks)+len(post),verdict=verdict,ledger=ledger)
#<<CORE_END>>
#<<FIGURES_BEGIN>>
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":.8,"axes.edgecolor":"#374151","axes.labelcolor":"#111827","xtick.color":"#374151","ytick.color":"#374151",
                     "xtick.labelsize":8,"ytick.labelsize":8,"axes.labelsize":9,"axes.titlesize":9.5,"axes.titleweight":"bold","axes.titlelocation":"left","pdf.fonttype":42,"ps.fonttype":42,
                     "legend.fontsize":8,"legend.frameon":False})
FIG_W,FIG_H=13.333,7.5;LAYOUT_DPI=100;OUT_DPI=300
INK,INK2,MUTED,RULE,GRIDC,PANEL="#111827","#374151","#6B7280","#D1D5DB","#E5E7EB","#F9FAFB"
BLUE,BLUE_L,ORANGE,ORANGE_L,TEAL,TEAL_L,GRAYPT,AMBER_L,AMBER="#1D4ED8","#DBEAFE","#C2410C","#FFEDD5","#0F766E","#CCFBF1","#9CA3AF","#FEF3C7","#B45309"
MONO="DejaVu Sans Mono"
def mt(s):return str(s).replace("$",r"\$")
def ascii_(s):return unicodedata.normalize("NFKD",str(s)).encode("ascii","ignore").decode("ascii")
def new_fig():return plt.figure(figsize=(FIG_W,FIG_H),dpi=LAYOUT_DPI,facecolor="white")
def sanitize_dashes(fig):
    """Zero-width artists are forced to a solid style: some matplotlib builds raise on a dashed pattern scaled to zero width."""
    for a in fig.findobj(lambda o:isinstance(o,(Patch,Line2D))):
        try:
            if a.get_linewidth()==0 and a.get_linestyle()not in("-","solid","None","none"," ",""):a.set_linestyle("-")
        except Exception:pass
def finish(fig,path,ctx,expect,desc):
    """Every expected value must appear in the figure text; claim scan must be clean; then JPEG (300 dpi, EXIF provenance) + vector PDF."""
    sanitize_dashes(fig);fig.canvas.draw();texts=[t.get_text()for t in fig.findobj(Text)if t.get_text().strip()];blob="\n".join(texts);nm=Path(path).name
    ws=lambda z:re.sub(r"\s+","",z);bw=ws(blob);miss=[e for e in expect if ws(e)not in bw]
    if miss:plt.close(fig);raise AuditFail(f"FIGURE/PAYLOAD MISMATCH in {nm}: missing {miss}")
    hits=claim_scan(nm,strip_words(re.sub(r"\s*\n\s*"," ",blob),ctx.get("strip",[])),allow=ctx["allow"])
    if hits:plt.close(fig);raise AuditFail(f"CLAIM-DISCIPLINE HIT in {nm}: {hits[:3]}")
    pdf=Path(path).with_suffix(".pdf")
    fig.savefig(pdf,format="pdf",facecolor="white",metadata={"Title":desc,"Author":AUTHOR,"Subject":f"{DEMO_TITLE} · run {ctx['run_id']} · payload SHA-256 {ctx['sha']}","Creator":DEMO_SHORT,
                "Keywords":f"engine sha256 {ENGINE_SHA256}; TEST528 lock {TEST528['lock']}"})
    buf=io.BytesIO();fig.savefig(buf,format="png",dpi=OUT_DPI,facecolor="white");plt.close(fig);buf.seek(0)
    with Image.open(buf)as im:rgb=im.convert("RGB")
    ex=Image.Exif();ex[0x010E]=ascii_(f"{desc} | {DEMO_SHORT} | run {ctx['run_id']} | payload sha256 {ctx['sha']} | engine sha256 {ENGINE_SHA256}");ex[0x013B]=ascii_(AUTHOR);ex[0x8298]=ascii_(COPYRIGHT);ex[0x0131]=ascii_(DEMO_SHORT)
    rgb.save(path,"JPEG",quality=95,optimize=True,progressive=False,subsampling=0,dpi=(OUT_DPI,OUT_DPI),exif=ex.tobytes())
    with Image.open(path)as c_:
        if c_.format!="JPEG"or c_.mode!="RGB":raise RuntimeError("JPEG validation failed")
    ctx["pdfs"].append(pdf);ctx["audit"].append(dict(file=nm,texts=len(texts),expected_values=len(expect),claim_hits=0));return str(path)
def fit_text(fig,x,y,w,h,text,fs_max=10,fs_min=6.5,color=INK,family=None,ls=1.3,weight="normal",ha="left"):
    fig.canvas.draw();r=fig.canvas.get_renderer();Wp=w*fig.get_figwidth()*fig.dpi;Hp=h*fig.get_figheight()*fig.dpi
    if Wp<=4 or Hp<=4:return 0
    paras=str(text if text else"(empty)").replace("\r","").split("\n");x0=x+w/2 if ha=="center"else x
    kw={"va":"top","ha":ha,"color":color,"linespacing":ls,"weight":weight,"multialignment":ha}
    if family:kw["family"]=family
    def wrap(c):
        out=[]
        for p in paras:out.extend(textwrap.wrap(p,c,break_long_words=True,break_on_hyphens=False,subsequent_indent=("  "if p.startswith("• ")else""))or[""])
        return out
    fs=float(fs_max);k=0.52
    for _ in range(150):
        cpl=max(6,int(Wp/(k*fs*fig.dpi/72.0)));ln=wrap(cpl);t=fig.text(x0,y+h,mt("\n".join(ln)),fontsize=fs,**kw);bb=t.get_window_extent(renderer=r)
        if bb.width>Wp*1.002 and cpl>6:t.remove();k*=max(1.01,bb.width/Wp*1.01);continue
        if bb.height<=Hp:return fs
        t.remove()
        if fs>fs_min:fs=max(float(fs_min),fs-0.25);continue
        per=bb.height/max(1,len(ln));keep=max(1,int(Hp/per)-1);fig.text(x0,y+h,mt("\n".join(ln[:keep]+["[… continued in the readable run log]"])),fontsize=fs,**kw);return fs
    fig.text(x0,y+h,"[text omitted — see run log]",fontsize=fs_min,**kw);return fs_min
def header(fig,k,title,sub,tag,tagc):
    fig.text(.03,.955,f"Fig. {k}",fontsize=12,weight="bold",color=MUTED,va="center")
    fig.text(.078,.955,mt(title),fontsize=16.5,weight="bold",color=INK,va="center")
    fig.text(.03,.917,mt(sub),fontsize=9.4,color=INK2,va="center")
    fig.text(.97,.955,tag,fontsize=8.4,weight="bold",color=tagc,va="center",ha="right",bbox=dict(boxstyle="square,pad=0.35",fc="white",ec=tagc,lw=1.0))
    fig.add_artist(Line2D([.03,.97],[.895,.895],transform=fig.transFigure,color=INK2,lw=.8))
def footer(fig,ctx,k,caption):
    fig.add_artist(Line2D([.03,.97],[.132,.132],transform=fig.transFigure,color=RULE,lw=.6))
    fit_text(fig,.03,.04,.94,.085,caption,fs_max=8.6,fs_min=6.6,color=INK2)
    fig.add_artist(Line2D([.03,.97],[.033,.033],transform=fig.transFigure,color=RULE,lw=.6))
    fig.text(.5,.016,mt(f"{DEMO_SHORT} (TEST528 mechanism; not TEST528 itself) · run {ctx['run_id']} · engine SHA-256 {ENGINE_SHA256[:12]}… · TEST528 lock {TEST528['lock'][:12]}… · payload {ctx['sha'][:12]}… · Fig. {k}/{ctx['N']} · © 2026 {AUTHOR}"),
             ha="center",va="center",fontsize=6.6,color=MUTED,family=MONO)
def clean(ax,grid=True):
    for s in("top","right"):ax.spines[s].set_visible(False)
    if grid:ax.grid(True,color=GRIDC,lw=.5);ax.set_axisbelow(True)
def frame(fig,x,y,w,h,ec=INK2,fc="white",lw=.9,ls="-",z=-1):
    fig.add_artist(Rectangle((x,y),w,h,transform=fig.transFigure,facecolor=fc,edgecolor=ec,lw=lw,ls=ls,zorder=z))
def block(fig,x,y,w,h,stage,title,body,ec=INK2,fc="white",mono=True,bfs=8.4):
    frame(fig,x,y,w,h,ec,fc);frame(fig,x,y+h-.032,w,.032,ec,PANEL if fc=="white"else fc,lw=.9)
    fig.text(x+.007,y+h-.016,stage,fontsize=7,weight="bold",color=MUTED,va="center")
    fig.text(x+.007,y+h-.052,mt(title),fontsize=9.6,weight="bold",color=INK,va="center")
    if body:fit_text(fig,x+.007,y+.008,w-.014,h-.078,body,fs_max=bfs,fs_min=6.4,family=MONO if mono else None,color=INK2)
def arrow(fig,x0,y0,x1,y1,color=INK2,lw=1.1,ms=11,cs="arc3"):fig.add_artist(FancyArrowPatch((x0,y0),(x1,y1),transform=fig.transFigure,arrowstyle="-|>",mutation_scale=ms,lw=lw,color=color,connectionstyle=cs,zorder=6))
def L_(ctx):return ctx["P"]["live"]
def score_axes(ax,scores,expected,selected,exp_c=INK,sel_c=BLUE,ylab=True,annotate=True,ms=7):
    ax.set_zorder(3);x=np.arange(1,len(scores)+1);s=np.asarray(scores,dtype=float)
    ax.scatter(x,s,s=ms,color=GRAYPT,zorder=2,lw=0)
    ax.scatter([expected],[s[expected-1]],s=ms*12,facecolors="none",edgecolors=exp_c,linewidths=1.2,zorder=4)
    ax.scatter([selected],[s[selected-1]],s=ms*5,color=sel_c,zorder=5,lw=0)
    ax.set_xlim(0,len(s)+1);ax.set_xticks([1,16,32,48,64,80,96,112,128]);clean(ax)
    lo,hi=float(s.min()),float(s.max());pad=max(1e-3,(hi-lo)*.08);ax.set_ylim(lo-pad,hi+pad*2.2)
    if ylab:ax.set_ylabel("candidate score s_j")
    return s
# ------------------------------------------------------------------ FIG 1 · retrieval path
def f1(ctx,k,path):
    L=L_(ctx);P=ctx["P"];cfg=P["engine"]["config"];r=L["primary"];ex=L["extras"];bt=L["b_tokens"];fig=new_fig()
    header(fig,k,"Retrieval path of the frozen mechanism (as executed in this run)",f"Item {L['item']:03d} · dimensions, pointer trace, address and the 128 candidate scores below are taken from this run.","ARCHITECTURE · VALUES FROM THIS RUN",INK2)
    yT=.845;stages=[(.03,"(a) INPUTS"),(.215,"(b) FROZEN COMPUTATION"),(.425,"(c) RETRIEVAL SIGNAL"),(.615,"(d) ADDRESS"),(.775,"(e) STORED B FIELD · COMPARISON · OUTPUT")]
    for x,s in stages:fig.text(x,yT,s,fontsize=8,weight="bold",color=INK2,va="center")
    q=L["question"]
    block(fig,.03,.585,.165,.235,"INPUT 1","Natural-language query",f"{q}\n\nn = {L['question_tokens']} tokens\n(QUESTION/ANSWER template)",mono=False,bfs=8.6)
    block(fig,.03,.205,.165,.335,"INPUT 2 · STORED","Active A memory (numeric)",f"K^l, V^l ∈ R^(8×T×128)\nl = 0 … 31\nT = {L['a_slots']} slots\nslot 0 = [PAD] anchor\nforged once by a frozen\nforward of the A record\n(K installed with RoPE\npositions 0 … T−1)")
    block(fig,.215,.38,.19,.44,"FROZEN","Mistral-7B-Instruct-v0.3",f"32 layers · d = 4096\n32 Q / 8 KV heads · d_h = 128\nweights frozen (guard ✓)\n\nA installed as KV cache\nquery positions T … T+n−1\none forward per question\n\noutput used:\nh_28 = residual stream at the\nfinal query token entering\nlayer 28")
    block(fig,.425,.43,.17,.39,"ÇAĞRIİZ","Pointer · L28H00","q = RoPE(W_Q^28 RMSNorm(h_28))\n  query head 0\nk_t = RoPE(K^28_t)\n  KV head 0 (GQA 4:1)\nw_t = softmax_{t≥1}(q·k_t/√128)\nw_0 = 0 · Σ_t w_t = 1",bfs=7.9)
    ax=fig.add_axes([.433,.445,.154,.085]);ax.set_zorder(3);w=np.asarray(r["pointer_weights"]);ax.bar(np.arange(len(w)),w,color=BLUE,width=.85);ax.set_xlim(-.6,len(w)-.4);ax.set_yticks([]);ax.set_xticks([0,len(w)-1])
    ax.tick_params(labelsize=6.5,length=2);[ax.spines[s].set_visible(False)for s in("top","right","left")];ax.set_title(f"w over T = {len(w)} slots (this run)",fontsize=7,weight="normal",pad=2)
    block(fig,.615,.43,.145,.39,"ADDRESS","L00-V · uncentered","V^0_t = W_V^0 RMSNorm(e_t)\n  ∈ R^(8×128)\np = Σ_t w_t · V^0_t\n  ∈ R^1024 (8 heads × 128)\nno centering",bfs=7.9)
    ax=fig.add_axes([.622,.445,.131,.085]);ax.set_zorder(3);pv=np.asarray(ex["p"]).reshape(8,128);lim=float(np.percentile(np.abs(pv),98))or 1.0
    ax.imshow(pv,aspect="auto",cmap="RdBu_r",vmin=-lim,vmax=lim,interpolation="nearest");ax.set_xticks([]);ax.set_yticks([]);ax.set_title("p (8 × 128, this run)",fontsize=7,weight="normal",pad=2)
    block(fig,.78,.205,.19,.615,"STORED · NOT FORWARDED AT QUERY TIME","128 precomputed B matrices",f"M_j = V^0(B_j)[slots ≥ 1]\n    ∈ R^(T_j × 1024)\nj = 1 … 128 · T_j = {min(bt)} … {max(bt)}\n\ns_j = max_s cos(p, M_j[s])\nĵ = argmax_j s_j\n(ties → lower index)",bfs=7.9)
    frame(fig,.775,.198,.2,.632,ec=INK2,fc="none",lw=.9,ls=(0,(4,3)),z=0)
    ax=fig.add_axes([.787,.28,.176,.20]);score_axes(ax,r["candidate_scores"],L["item"],r["selected_candidate"],ylab=False,ms=3)
    ax.set_xticks([1,64,128]);ax.set_yticks([]);ax.spines["left"].set_visible(False);ax.tick_params(labelsize=6.5);ax.set_title("s_j for j = 1 … 128 (this run)",fontsize=7,weight="normal",pad=2)
    fig.text(.787,.243,f"selected B#{r['selected_candidate']:03d} · s = {r['top1_score']:.4f}",fontsize=8,weight="bold",color=BLUE,va="center")
    fig.text(.787,.218,f"measured forwards in retrieve(): {len(L['forwards']['retrieve'])}",fontsize=7.6,color=INK2,va="center")
    arrow(fig,.195,.70,.215,.66);arrow(fig,.405,.62,.425,.62);arrow(fig,.595,.62,.615,.62);arrow(fig,.76,.62,.78,.62)
    arrow(fig,.195,.47,.215,.47)
    fig.add_artist(Line2D([.195,.69],[.30,.30],transform=fig.transFigure,color=INK2,lw=1.0));arrow(fig,.51,.30,.51,.43);arrow(fig,.69,.30,.69,.43)
    fig.text(.503,.292,"keys K^28 of the A slots → pointer",fontsize=7.4,color=INK2,ha="right",va="top");fig.text(.683,.292,"values V^0 of the A slots → address",fontsize=7.4,color=INK2,ha="right",va="top")
    fig.text(.408,.632,"h_28",fontsize=7.4,color=INK2,ha="left",va="bottom");fig.text(.597,.632,"w",fontsize=7.4,color=INK2,ha="left",va="bottom");fig.text(.762,.632,"p",fontsize=7.4,color=INK2,ha="left",va="bottom")
    frame(fig,.215,.148,.545,.088,ec=RULE,fc=PANEL,lw=.6)
    fit_text(fig,.222,.152,.533,.08,"• V^0 is the layer-0 value projection of the RMS-normalised input token embedding; p is a pointer-weighted combination of token-level value vectors of the A slots.\n"
             "• Target-span (seal token) positions are computed by the engine for diagnostics only; they are not an input to any step shown.\n• Identity regime of the sealed TEST528 protocol: native uppercase single-token seals.",fs_max=7.8,fs_min=6.4,color=INK2)
    footer(fig,ctx,k,f"Figure {k}. Executed retrieval path of the frozen TEST528 mechanism for item {L['item']:03d}. A natural-language query and the active numeric A memory (installed as KV cache) drive one frozen forward of Mistral-7B-Instruct-v0.3. "
           f"At the final query token, head L28H00 scores the A slots (ÇAĞRIİZ pointer w). The pointer weights the layer-0 value vectors of the A slots into an uncentered 1024-D address p, which is compared by maximum cosine with every row of the 128 precomputed B matrices. "
           f"The B matrices are stored tensors; the instrumentation recorded {len(L['forwards']['retrieve'])} model forward in retrieve(), processing {L['question_tokens']} query tokens over a {L['a_slots']}-slot A cache.")
    return finish(fig,path,ctx,["L28H00","L00-V",f"T = {L['a_slots']} slots",f"selected B#{r['selected_candidate']:03d}",f"{r['top1_score']:.4f}"],"Retrieval path of the frozen mechanism")
# ------------------------------------------------------------------ FIG 2 · 128-way competition
def f2(ctx,k,path):
    L=L_(ctx);r=L["primary"];D=L["primary_rederived"];e=L["item"];sel=r["selected_candidate"];fig=new_fig();ok=r["correct"]
    header(fig,k,"128-way candidate competition for one query","Score of every B memory against the address p; expected candidate (ring) and selected candidate (filled).","LIVE DEMO OBSERVATION",BLUE)
    line=(f"item {e:03d} · {L['entity']} · expected B#{e:03d} ({r['expected_seal']}) · selected B#{sel:03d} ({r['selected_seal']}) · rank of expected {r['rank']}/{N_CAND} · "
          f"s1 = {r['top1_score']:.6f} · s2 = {r['top2_score']:.6f} · margin = {r['top1_top2_margin']:+.6f}")
    fig.text(.03,.865,mt(line),fontsize=8.8,family=MONO,color=INK,va="center");fig.text(.97,.865,"CORRECT"if ok else"INCORRECT",fontsize=11,weight="bold",color=BLUE if ok else ORANGE,ha="right",va="center")
    ax=fig.add_axes([.06,.255,.565,.545]);s=score_axes(ax,r["candidate_scores"],e,sel,ms=10)
    ax.axhline(r["top2_score"],color=MUTED,lw=.7,ls=(0,(3,3)),zorder=1);ax.set_xlabel("B candidate j (audit label; not an input to retrieval)")
    ax.set_ylabel("s_j = max_s cos(p, M_j[s])")
    rng_=float(s.max()-s.min())or 1.0
    ax.annotate(f"selected B#{sel:03d}\ns = {r['top1_score']:.6f}",xy=(sel,s[sel-1]),xytext=(sel+(10 if sel<90 else-34),s[sel-1]+.07*rng_),fontsize=8,color=BLUE,va="center",arrowprops=dict(arrowstyle="-",color=BLUE,lw=.7))
    if not ok:ax.annotate(f"expected B#{e:03d}\nrank {r['rank']}",xy=(e,s[e-1]),xytext=(e+(10 if e<90 else-30),s[e-1]-(s.max()-s.min())*.12),fontsize=8,color=INK,arrowprops=dict(arrowstyle="-",color=INK,lw=.7))
    ax.text(127,r["top2_score"],f"s2 = {r['top2_score']:.4f}",fontsize=7.4,color=MUTED,ha="right",va="bottom")
    ax.legend(handles=[Line2D([0],[0],marker="o",lw=0,markerfacecolor="none",markeredgecolor=INK,markersize=8,label=f"expected B#{e:03d} (panel construction)"),
                       Line2D([0],[0],marker="o",lw=0,color=BLUE,markersize=5,label="selected = argmax_j s_j"),Line2D([0],[0],marker="o",lw=0,color=GRAYPT,markersize=4,label="other candidates")],loc="upper right")
    zt=f"{D['z_top1']:.2f}"if D["z_top1"]is not None else"n/a"
    fig.text(.06,.19,mt(f"non-selected candidates (n = 127): median {D['rest_median']:.4f} · IQR {D['rest_q1']:.4f} – {D['rest_q3']:.4f} · max {D['rest_max']:.4f} · (s1 − mean)/sd = {zt}"),fontsize=8.2,color=INK2,family=MONO)
    fig.text(.06,.162,mt(f"repeat retrieve(): same selection · max |Δs| = {L['repeat_max_abs_diff']:.2e} · measured model forwards in retrieve(): {len(L['forwards']['retrieve'])} (query over A cache only)"),fontsize=8.2,color=INK2,family=MONO)
    ax2=fig.add_axes([.69,.60,.28,.20]);ss=np.sort(s)[::-1];ax2.plot(np.arange(1,129),ss,color=INK2,lw=1);ax2.scatter([r["rank"]],[s[e-1]],s=60,facecolors="none",edgecolors=INK,linewidths=1.1,zorder=4)
    ax2.scatter([1],[ss[0]],s=22,color=BLUE,zorder=5);clean(ax2);ax2.set_xlim(0,129);ax2.set_xlabel("rank");ax2.set_ylabel("score");ax2.set_title("Rank-ordered scores")
    fig.text(.69,.515,"Top 10",fontsize=9.5,weight="bold",color=INK);fig.text(.69,.487,mt(f"{'rank':>4} {'cand.':>6}  {'seal (audit)':<13}{'score':>10}{'Δ to s1':>11}"),fontsize=7.9,family=MONO,color=MUTED)
    for n_,t_ in enumerate(r["top10"]):
        y=.462-n_*.0238;hl=t_["candidate"]==e
        fig.text(.69,y,mt(f"{t_['rank']:>4} {'#'+format(t_['candidate'],'03d'):>6}  {t_['seal']:<13}{t_['score']:>10.6f}{t_['score']-r['top1_score']:>+11.6f}"),fontsize=7.9,family=MONO,color=INK if hl else INK2,weight="bold"if hl else"normal")
    footer(fig,ctx,k,f"Figure {k}. Live 128-way competition for item {L['item']:03d} ({L['entity']}). Each point is s_j = max over the slots of B memory j of the cosine between the 1024-D address p and that slot's L00-V vector. "
           f"The expected candidate is defined by panel construction (item i's target seal occurs only in B memory i); candidate numbers are audit labels and are not available to the retrieval computation. "
           f"Result: selected B#{sel:03d}, rank of expected {r['rank']}/{N_CAND}, margin {r['top1_top2_margin']:+.6f}. A single live item is a demonstration observation, not an estimate of accuracy.")
    return finish(fig,path,ctx,[f"s1 = {r['top1_score']:.6f}",f"s2 = {r['top2_score']:.6f}",f"margin = {r['top1_top2_margin']:+.6f}",f"rank of expected {r['rank']}/{N_CAND}"],"128-way candidate competition")
# ------------------------------------------------------------------ FIG 3 · pointer and address
def f3(ctx,k,path):
    L=L_(ctx);r=L["primary"];ex=L["extras"];ps=L["pointer_stats"];pdat=L["pointer_data"];fig=new_fig();w=np.asarray(r["pointer_weights"]);span=pdat["gold_span_positions"]
    header(fig,k,"ÇAĞRIİZ pointer (L28H00) and the resulting 1024-D address","Pointer distribution over the A memory slots, the address it produces, and the best-matching row of the selected B memory.","LIVE DEMO OBSERVATION",BLUE)
    ax=fig.add_axes([.055,.585,.89,.255]);x=np.arange(len(w))
    ax.axvspan(min(span)-.5,max(span)+.5,facecolor=AMBER_L,edgecolor=AMBER,hatch="////",lw=.6,zorder=0)
    ax.bar(x,w,color=[BLUE if t==ps["argmax"]else"#93A3C4"for t in x],width=.8,zorder=2);clean(ax);ax.set_xlim(-.6,len(w)-.4);ax.set_ylim(0,max(w)*1.18)
    ax.set_ylabel("pointer weight w_t")
    if ex["tokens"]:
        ax.set_xticks(x);ax.set_xticklabels([mt(t.replace("\n","\\n").strip()or"·")for t in ex["tokens"]],rotation=90,fontsize=7,family=MONO)
        ax.set_xlabel("A memory slot t · token labels decoded from the engine's forge-time record for display only (not available to retrieval)",fontsize=8)
    else:ax.set_xlabel("A memory slot t")
    info=(f"head L28H00 · T = {ps['T']} slots · argmax slot {ps['argmax']} · peak w = {ps['peak']:.4f}\nentropy {ps['entropy_bits']:.3f} bits · effective slots {ps['effective_slots']:.2f}\n"
          f"diagnostic span {span}: mass {r['pointer_mass']:.4f} · hit {r['pointer_hit']}")
    ax.text(.995,.97,mt(info),transform=ax.transAxes,ha="right",va="top",fontsize=7.8,family=MONO,color=INK,bbox=dict(boxstyle="square,pad=0.4",fc="white",ec=RULE,lw=.6))
    ax.legend(handles=[Patch(facecolor=AMBER_L,edgecolor=AMBER,hatch="////",lw=.6,label="diagnostic target span (seal token positions; not a retrieval input)"),Patch(color=BLUE,label="pointer argmax")],loc="upper right",bbox_to_anchor=(1.0,.70))
    pv=np.asarray(ex["p"]).reshape(8,128);bv=np.asarray(ex["best_row"]).reshape(8,128)
    for n_,(M,ttl)in enumerate(((pv,"Address p = Σ_t w_t · V^0_t   (8 KV heads × 128 dims, uncentered)"),(bv,f"Best-matching row of selected B#{r['selected_candidate']:03d}: M_ĵ[s*], s* = slot {ex['best_slot']} of {ex['b_rows']}"))):
        a=fig.add_axes([.055+n_*.49,.235,.39,.18]);lim=float(np.percentile(np.abs(M),98))or 1.0
        im=a.imshow(M,aspect="auto",cmap="RdBu_r",vmin=-lim,vmax=lim,interpolation="nearest");a.set_yticks(range(8));a.set_yticklabels([f"h{j}"for j in range(8)],fontsize=7)
        a.set_xticks([0,32,64,96,127]);a.set_xlabel("dimension within KV head",fontsize=8);a.set_title(mt(ttl),fontsize=8.6)
        cb=fig.colorbar(im,cax=fig.add_axes([.452+n_*.49,.235,.006,.18]));cb.ax.tick_params(labelsize=6.5);cb.outline.set_linewidth(.5)
    fig.text(.545,.152,mt(f"cos(p, M_ĵ[s*]) = {ex['best_cos']:+.6f} = reported s_ĵ (engine functions; |Δ| ≤ 1e-6 checked)"),fontsize=7.8,family=MONO,color=INK2)
    fig.text(.055,.152,mt(f"get_pointer_data(): max |Δw| vs retrieve() = {L['pointer_data_max_abs_diff']:.2e}"),fontsize=7.8,family=MONO,color=INK2)
    footer(fig,ctx,k,f"Figure {k}. Top: pointer distribution of attention head L28H00 at the final query token over the {ps['T']} slots of the active A memory (slot 0 excluded by the engine). "
           "The shaded span marks the positions of the target seal token; the engine computes it for diagnostics only and it does not enter the pointer, the address or the scoring. "
           "Bottom: the 1024-D address p formed from the layer-0 value vectors (left) and the B-memory row that attains the maximum cosine for the selected candidate (right); colour scales are per panel (98th percentile of |value|).")
    return finish(fig,path,ctx,[f"peak w = {ps['peak']:.4f}",f"mass {r['pointer_mass']:.4f}",f"{ex['best_cos']:+.6f}","L28H00"],"Pointer and address")
# ------------------------------------------------------------------ FIG 4 · counterfactual
def f4(ctx,k,path):
    L=L_(ctx);r=L["primary"];cf=L["counterfactual"];i=L["item"];j=L["cf_target"];fig=new_fig();D=L["primary_rederived"];Dc=L["counterfactual_rederived"]
    header(fig,k,"Counterfactual association changes the selected B memory","Same frozen weights, same question string, same 128 B matrices; only the A memory's seal association is changed and re-forged.","LIVE DEMO OBSERVATION",BLUE)
    rows=[("ORIGINAL A",f"{L['entity']} carries seal {cf['original_seal']}  →  expected B#{i:03d}",r["pointer_weights"],r["candidate_scores"],r["selected_candidate"],r["pointer_mass"],r["pointer_hit"],INK),
          ("COUNTERFACTUAL A",f"{L['entity']} carries seal {cf['counterfactual_seal']}  →  expected B#{j:03d}",cf["pointer_weights"],cf["candidate_scores"],cf["selected_candidate"],cf["pointer_mass"],cf["pointer_hit"],ORANGE)]
    for n_,(lab,assoc,w,sc,sel,mass,hit,col)in enumerate(rows):
        y0=.535-n_*.325
        fig.text(.03,y0+.255,lab,fontsize=10,weight="bold",color=col,va="center");fig.text(.16,y0+.255,mt(assoc),fontsize=9,family=MONO,color=INK,va="center")
        a=fig.add_axes([.03,y0,.25,.215]);w=np.asarray(w);a.bar(np.arange(len(w)),w,color=col if n_ else"#93A3C4",width=.8);a.set_xlim(-.6,len(w)-.4);clean(a)
        a.set_title(f"L28H00 pointer (T = {len(w)}) · span mass* {mass:.3f} · hit {hit}",fontsize=7.6,weight="normal");a.set_xlabel("A slot t",fontsize=7.8);a.tick_params(labelsize=7)
        b=fig.add_axes([.33,y0,.43,.215]);s=np.asarray(sc,dtype=float);x=np.arange(1,129)
        b.scatter(x,s,s=7,color=GRAYPT,lw=0,zorder=2);b.scatter([i],[s[i-1]],s=80,facecolors="none",edgecolors=INK,linewidths=1.2,zorder=4);b.scatter([j],[s[j-1]],s=80,facecolors="none",edgecolors=ORANGE,linewidths=1.2,zorder=4)
        b.scatter([sel],[s[sel-1]],s=34,color=BLUE if n_==0 else ORANGE,zorder=5,lw=0);clean(b);b.set_xlim(0,129);b.set_xticks([1,16,32,48,64,80,96,112,128]);b.tick_params(labelsize=7)
        b.set_title(f"candidate scores · selected B#{sel:03d}",fontsize=7.8,weight="normal");b.set_xlabel("B candidate j",fontsize=7.8)
    fig.legend(handles=[Line2D([0],[0],marker="o",lw=0,markerfacecolor="none",markeredgecolor=INK,markersize=8,label=f"original target B#{i:03d}"),
                        Line2D([0],[0],marker="o",lw=0,markerfacecolor="none",markeredgecolor=ORANGE,markersize=8,label=f"counterfactual target B#{j:03d}"),
                        Line2D([0],[0],marker="o",lw=0,color=BLUE,markersize=5,label="selected (original A)"),Line2D([0],[0],marker="o",lw=0,color=ORANGE,markersize=5,label="selected (counterfactual A)")],
               loc="center",bbox_to_anchor=(.545,.875),ncol=4,fontsize=7.8)
    tx=.79;fig.text(tx,.78,"Summary",fontsize=10,weight="bold",color=INK)
    tab=[("",f"original",f"counterf."),("selected",f"B#{r['selected_candidate']:03d}",f"B#{cf['selected_candidate']:03d}"),(f"rank of B#{i:03d}",f"{r['rank']}",f"{L['original_rank_under_counterfactual_A']}"),
         (f"rank of B#{j:03d}",f"{L['cf_target_rank_under_original_A']}",f"{cf['rank']}"),("top-1 score",f"{r['top1_score']:.4f}",f"{cf['top1_score']:.4f}"),
         ("margin",f"{r['top1_top2_margin']:+.4f}",f"{cf['top1_top2_margin']:+.4f}"),("span mass*",f"{r['pointer_mass']:.3f}",f"{cf['pointer_mass']:.3f}"),
         ("model forwards",f"{len(L['forwards']['retrieve'])}",f"{len(L['forwards']['counterfactual'])}")]
    for n_,(a_,b_,c_)in enumerate(tab):fig.text(tx,.745-n_*.034,mt(f"{a_:<15}{b_:>9}{c_:>11}"),fontsize=8.2,family=MONO,color=MUTED if n_==0 else INK)
    fol=cf["followed_counterfactual"]
    fig.text(tx,.44,"followed counterfactual:",fontsize=9,color=INK);fig.text(tx,.405,"YES"if fol else"NO",fontsize=15,weight="bold",color=ORANGE if fol else INK2)
    fit_text(fig,tx,.17,.18,.2,"* diagnostic only.\nCounterfactual forwards: 1 forge of the new A memory (no cache) + 1 query forward over it. The counterfactual A is built by the engine: the queried entity's seal is replaced by the target seal of candidate "
             f"{j:03d}; the two other statements keep their seals and the statement order is reshuffled with a fixed engine seed.",fs_max=7.6,fs_min=6.4,color=INK2)
    footer(fig,ctx,k,f"Figure {k}. Counterfactual test for item {i:03d}. Upper row: original A memory; lower row: A memory re-forged with the association {L['entity']} → {cf['counterfactual_seal']} (the target seal of B#{j:03d}). "
           f"With weights, question, pointer coordinate, address coordinate and B field unchanged, the selection moves from B#{r['selected_candidate']:03d} to B#{cf['selected_candidate']:03d}; the original candidate falls to rank {L['original_rank_under_counterfactual_A']}. "
           "This is a single live observation; the archived TEST528 counterfactual result is shown in Fig. 5.")
    return finish(fig,path,ctx,[f"B#{cf['selected_candidate']:03d}","followed counterfactual:",cf["counterfactual_seal"]],"Counterfactual association")
# ------------------------------------------------------------------ FIG 5 · controls + archived reference
CTRL_LAB={"PRIMARY":"PRIMARY · ÇAĞRIİZ pointer w","UNIFORM":"UNIFORM · w_t = 1/(T−1), t ≥ 1","SHIFTED":"SHIFTED · w rolled by ⌊(T−1)/2⌋ over t ≥ 1","NO_A":"NO-A · zero address (p = 0)"}
def f5(ctx,k,path):
    L=L_(ctx);c=L["controls"]["controls"];DC=L["controls_rederived"];ex=L["extras"];i=L["item"];fig=new_fig()
    header(fig,k,"Pointer controls on one item · archived TEST528 reference","Left/centre: live single-item controls from run_controls(). Right: sealed TEST528 values over 128 items, reproduced verbatim (not recomputed).","LIVE CONTROLS + SEALED REFERENCE",INK2)
    wmap={"PRIMARY":L["primary"]["pointer_weights"],"UNIFORM":ex["uniform"],"SHIFTED":ex["shifted"],"NO_A":None}
    for n_,mode in enumerate(("PRIMARY","UNIFORM","SHIFTED","NO_A")):
        y=.70-n_*.165;v=c[mode];d=DC[mode];col={"PRIMARY":BLUE,"UNIFORM":"#6B7280","SHIFTED":AMBER,"NO_A":"#9CA3AF"}[mode]
        a=fig.add_axes([.03,y,.24,.10]);clean(a,grid=False);a.tick_params(labelsize=6.8)
        if wmap[mode]is not None:wv=np.asarray(wmap[mode]);a.bar(np.arange(len(wv)),wv,color=col,width=.8);a.set_xlim(-.6,len(wv)-.4)
        else:a.set_xticks([]);a.set_yticks([]);a.text(.5,.5,"no pointer · address p = 0",transform=a.transAxes,ha="center",va="center",fontsize=8,color=INK2)
        a.set_title(mt(CTRL_LAB[mode]),fontsize=7.9)
        b=fig.add_axes([.31,y,.35,.10]);s=np.asarray(v["candidate_scores"],dtype=float);b.scatter(np.arange(1,129),s,s=5,color=GRAYPT,lw=0)
        b.scatter([i],[s[i-1]],s=55,facecolors="none",edgecolors=INK,linewidths=1.1,zorder=4);b.scatter([v["selected_candidate"]],[s[v["selected_candidate"]-1]],s=22,color=col,zorder=5,lw=0)
        clean(b);b.set_xlim(0,129);b.tick_params(labelsize=6.8);b.set_xticks([1,32,64,96,128])
        if d["all_equal"]:b.set_ylim(-.05,.05)
        b.set_title(mt(f"selected B#{v['selected_candidate']:03d} · rank of expected {v['rank']}/{N_CAND} · margin {v['margin']:+.4f}"),fontsize=7.9,weight="normal")
    tx=.695;fig.text(tx,.82,"Live, this item",fontsize=9.8,weight="bold",color=INK)
    fig.text(tx,.79,mt(f"{'control':<9}{'selected':>9}{'rank':>7}{'top-1':>9}{'margin':>9}"),fontsize=7.9,family=MONO,color=MUTED)
    for n_,mode in enumerate(("PRIMARY","UNIFORM","SHIFTED","NO_A")):
        v=c[mode];fig.text(tx,.763-n_*.026,mt(f"{mode:<9}{'B#'+format(v['selected_candidate'],'03d'):>9}{v['rank']:>7}{v['top1_score']:>9.4f}{v['margin']:>+9.4f}"),fontsize=7.9,family=MONO,color=INK)
    if DC["NO_A"]["all_equal"]:fit_text(fig,tx,.585,.275,.06,"NO-A: all 128 scores are 0 (zero address), so the argmax falls to B#001 by index tie-breaking; NO-A can therefore be 'correct' only for item 001, as an artefact.",fs_max=7.5,fs_min=6.4,color=INK2)
    frame(fig,tx,.19,.275,.37,ec=TEAL,fc=TEAL_L,lw=1.0)
    fig.text(tx+.01,.535,"SEALED TEST528 REFERENCE",fontsize=9.6,weight="bold",color=TEAL);fig.text(tx+.01,.512,"archived · 128 items · not recomputed here",fontsize=7.6,color=INK2)
    for n_,(lab,key)in enumerate((("PRIMARY","primary"),("COUNTERFACTUAL","counterfactual"),("SHIFTED","shifted"),("NO-A","no_a"))):
        v=TEST528[key];y=.47-n_*.052;fig.text(tx+.01,y,lab,fontsize=8.6,color=INK,weight="bold");fig.text(tx+.265,y,f"{fr(v)} = {v[2]}",fontsize=8.6,family=MONO,color=INK,ha="right")
    fig.text(tx+.01,.262,"UNIFORM: not part of the archived reference set",fontsize=7.4,color=INK2)
    fig.text(tx+.01,.236,mt(f"lock {TEST528['lock'][:24]}…"),fontsize=7,family=MONO,color=INK2);fig.text(tx+.01,.214,mt(f"result {TEST528['result_sha'][:24]}…"),fontsize=7,family=MONO,color=INK2)
    footer(fig,ctx,k,f"Figure {k}. Controls for item {i:03d} computed live by run_controls() with the same question and A memory: the ÇAĞRIİZ pointer (PRIMARY), a uniform pointer, the pointer rolled by half the slot range (SHIFTED), and a zero address (NO-A). "
           "Rings mark the expected candidate, filled points the selected one. The panel on the right reproduces the sealed TEST528 aggregate results verbatim; a single interactive item does not reproduce or re-estimate those statistics.")
    return finish(fig,path,ctx,["SEALED TEST528 REFERENCE",f"{fr(TEST528['primary'])} = {TEST528['primary'][2]}",f"{fr(TEST528['no_a'])} = {TEST528['no_a'][2]}","PRIMARY","NO_A"],"Controls and archived reference")
# ------------------------------------------------------------------ FIG 6 · provenance and verification record
def f6(ctx,k,path):
    P=ctx["P"];L=P["live"];cfg=P["engine"]["config"];E_=P["environment"];fz=P["frozen"];v=P["verify_after"];fig=new_fig()
    header(fig,k,"Provenance and verification record","Configuration, measured integrity quantities, hashes and the claim boundary of this run.","PROVENANCE RECORD",INK2)
    cols=[(.03,"CONFIGURATION AND ENGINE",
           f"run           {P['run_id']}\nstart (UTC)   {P['run_start_utc']}\nmodel         {cfg['MODEL_ID'].split('/')[-1]}\nweights       frozen · {P['model']['dtype']} · {P['model']['attn']}\n"
           f"architecture  32 L · 4096 · 32 Q / 8 KV · 128\npointer       L{cfg['PTR_L']:02d}H{cfg['PTR_H']:02d}\naddress       L{cfg['ADDR_L']:02d}-{cfg['ADDR_KIND']} · uncentered · {cfg['KVD']}-D\n"
           f"candidates    {cfg['N']} precomputed B matrices\nmatch         max cosine · argmax\nengine seed   {cfg['seed']}\nGPU           {E_['gpu']}\ntorch         {E_['torch']}\ntransformers  {E_['transformers']}\n"
           f"gradio        {E_['gradio']}\npython        {E_['python']}\nengine init   {(f"{P['engine']['init_seconds']:.1f} s"if isinstance(P['engine']['init_seconds'],(int,float))else"reused")}\nend (UTC)     {P['run_end_utc']}\n\nengine file\n  {P['engine']['file']}\nengine SHA-256\n  {P['engine']['sha256'][:32]}\n  {P['engine']['sha256'][32:]}"),
          (.35,"VERIFICATION MEASURED IN THIS RUN",
           f"weight sentinel  startup = before = after\n  {fz['sentinel_after'][:40]}…\nall-parameter guard  startup = before = after\n  {fz['guard_after'][:40]}…\n"
           f"trainable tensors   {fz['trainable_tensors']}\nLoRA / optimizer    {fz['lora']} / {fz['optimizer']}\nforeign hooks       {fz['foreign_hooks_before']} → {fz['foreign_hooks_after']} (outside API calls)\n\n"
           f"model forwards per API call (measured):\n  retrieve()        {len(L['forwards']['retrieve'])}  (n={L['question_tokens']}, A cache {L['a_slots']})\n  retrieve() again  {len(L['forwards']['repeat'])}\n"
           f"  get_pointer_data  {len(L['forwards']['pointer_data'])}\n  run_controls()    {len(L['forwards']['controls'])}\n  counterfactual()  {len(L['forwards']['counterfactual'])}  (forge + query)\n"
           f"candidate-B forwards: none observed\n\nverify_engine(): sentinel pass {v['weight_sentinel_pass']} ·\n  candidates {v['candidate_count']} · dim {v['address_dim']}\nre-derivation from raw scores: all match\n"
           f"pre-seal checks passed: {len(P['checks_pre_seal'])}\n\nlatency (engine API, ms):\n  retrieve {1000*P['timing'].get('retrieve',0):.1f} · controls {1000*P['timing'].get('controls',0):.1f}\n  counterfactual {1000*P['timing'].get('counterfactual',0):.1f} · guards {P['timing'].get('guard',0):.1f} s"),
          (.67,"ARCHIVED REFERENCES AND CLAIM BOUNDARY",
           f"TEST528 (sealed, archived)\n  lock   {TEST528['lock'][:32]}…\n  result {TEST528['result_sha'][:32]}…\n  primary         {fr(TEST528['primary'])} = {TEST528['primary'][2]}\n  counterfactual  {fr(TEST528['counterfactual'])} = {TEST528['counterfactual'][2]}\n"
           f"  shifted         {fr(TEST528['shifted'])} = {TEST528['shifted'][2]}\n  no-A            {fr(TEST528['no_a'])} = {TEST528['no_a'][2]}\n\ncross-model context\n(archived; different panels)\n"
           +"\n".join(f"  {c_['record']} · {c_['model']}\n    {c_['pointer']} · {c_['address'][:5]} · {c_['dim']}-D · {fr(c_['top1'])} = {c_['top1'][2]}"for c_ in CROSS_MODEL)+
           f"\n  {CROSS_MODEL_NOTE}\n\nnot established:\n"+"\n".join("  • "+t for t in("arbitrary token identities (native uppercase single-token seals)","first-A retrieval from an empty bank","answer generation from the selected B",
                                                                                         "transfer of internal coordinates between models","banks larger than 128 candidates")))]
    for x,ttl,body in cols:
        frame(fig,x,.155,.30,.715,ec=RULE,fc="white",lw=.8);fig.text(x+.01,.85,ttl,fontsize=9,weight="bold",color=INK2,va="center")
        fit_text(fig,x+.01,.165,.282,.665,body,fs_max=9.4,fs_min=6.2,family=MONO,color=INK)
    footer(fig,ctx,k,f"Figure {k}. Provenance record of run {P['run_id']} (verdict {P['verdict']}). Payload SHA-256 {ctx['sha']}. "
           "Hashes are artifact-integrity seals, not scientific proof or third-party verification. All live values are re-derived from the raw vectors stored in the sealed payload; archived values are reproduced verbatim from the TEST528 record.")
    return finish(fig,path,ctx,[P["run_id"],ENGINE_SHA256[:40],"none observed",fz["guard_after"][:40]],"Provenance and verification record")
FIGURES=[("fig_01_retrieval_path","Fig. 1 · Retrieval path (values from this run)",f1),("fig_02_candidate_competition","Fig. 2 · 128-way candidate competition",f2),
         ("fig_03_pointer_and_address","Fig. 3 · ÇAĞRIİZ pointer and 1024-D address",f3),("fig_04_counterfactual","Fig. 4 · Counterfactual association",f4),
         ("fig_05_controls_and_reference","Fig. 5 · Controls and sealed TEST528 reference",f5),("fig_06_provenance","Fig. 6 · Provenance and verification record",f6)]
N_FIGS=len(FIGURES)
def render_all(ctx,run_dir):
    imgs=[]
    try:
        for i,(slug,cap,fn)in enumerate(FIGURES,1):
            p=Path(run_dir)/f"{slug}_{ctx['run_id']}.jpg";fn(ctx,i,p);imgs.append((p,cap))
    finally:plt.close("all")
    return imgs
#<<FIGURES_END>>
say("PART 2 / 3 complete — measurement pipeline and figures defined. Now run PART 3 / 3 in the next cell.")
MAM_MISTRAL_PART2_OK=True
# =====================================================================================================================================
# AKBASCORE MAM · MISTRAL-7B ASSOCIATIVE RETRIEVAL INSTRUMENT — PART 3 / 3 · SELF-TEST AND GRADIO INSTRUMENT
# Same demo as PARTS 1 and 2. Run this cell after PART 2 / 3, in the same Colab runtime. It runs a GPU-free self-test of the full
# pipeline (including fail-closed cases) and then prints a public gradio.live link. Select an item and press RUN.
# =====================================================================================================================================
if not globals().get("MAM_MISTRAL_PART2_OK"):raise RuntimeError("PART 2 / 3 has not been run in this runtime. Run PART 1, then PART 2, then PART 3.")
#<<UI_BEGIN>>
_SYN_SYL1=["Aq","Bryn","Cav","Dex","Eln","Frax","Gyr","Hex","Iln","Jor","Kex","Lyr","Mav","Nex","Oryn","Pax","Qyr","Rex","Savn","Tov","Uln","Vex","Wyr","Xav","Yex","Zyr"]
_SYN_SYL2=["adar","bren","cyr","dax","elor","fyn","grel","hyn","ivar","jor","kyr","lor","myn","nex","or","pyr","qen","rix","sor","tyn"]
class SelfTestInstrument:
    """Synthetic stand-in used ONLY by the service self-test (no GPU, no model). Never used for a real run."""
    def __init__(s,mode="ok"):
        s.kind="SELF-TEST";s.mode=mode;s.calls=0;rng=random.Random(7)
        names=[a+b for a in _SYN_SYL1 for b in _SYN_SYL2];rng.shuffle(names);s.names=names[:N_CAND*3]
        codes=sorted({"".join(rng.choice("BCDFGHJKLMNPQRSTVWXZ")for _ in range(3))for _ in range(2000)});rng.shuffle(codes);s.seals=codes[:N_CAND];s.dis=codes[N_CAND:N_CAND+64]
        s.A=[];s.span=[]
        for i in range(N_CAND):
            st=[(s.names[3*i],s.seals[i]),(s.names[3*i+1],s.dis[i%64]),(s.names[3*i+2],s.dis[(i*7+3)%64])];random.Random(i).shuffle(st)
            s.A.append(st)
        s.info=dict(model_id=FROZEN_CONFIG["MODEL_ID"],arch=list(ARCH),rep=4,kvd=1024,dtype="bfloat16",attn="sdpa",gpu="SELF-TEST INSTRUMENT (no GPU)",gpu_total_gib=0.0,torch="n/a",transformers="n/a",
            gradio="n/a",python="n/a",platform="n/a",params=1,vocab=1,pad_id=0,engine_file=ENGINE_FILE,engine_sha256=ENGINE_SHA256_EXPECTED,init_seconds=0.0,startup_utc="n/a",
            sentinel0="a"*64,guard0="b"*64,hooks0=0,sentinel_method="synthetic",guard_method="synthetic",forward_counter="synthetic")
    def _tok(s,i):
        out=["[PAD]"]
        for e,sl in s.A[i-1]:out+=["Instrument"," "+e[:3],e[3:]," carries"," seal"," "+sl,"."]
        return out+["\n\n"]
    def _span(s,i):t=s._tok(i);return[j for j,x in enumerate(t)if x.strip()==s.seals[i-1]]
    def _w(s,i,target_seal=None):
        t=s._tok(i);sl=target_seal or s.seals[i-1];w=[0.0]+[0.02+0.01*((j*7)%5)for j in range(1,len(t))]
        for j,x in enumerate(t):
            if x.strip()==sl:w[j]=1.4
        z=sum(w);return[x/z for x in w]
    def _scores(s,i,target,miss=False):
        rng=random.Random(1000+i*31+target);sc=[0.30+0.12*rng.random()for _ in range(N_CAND)];sc[target-1]=0.88+0.05*rng.random()
        if miss:sc[(target%N_CAND)]=sc[target-1]+0.01
        if s.mode=="nan":sc[3]=float("nan")
        return sc
    def _rec(s,n):
        nq=s.question_tokens(s.items()[0]["question"]);return[dict(input_len=nq,has_past=True,past_len=s.a_slots(n))]
    def _top(s,sc,exp):
        order=sorted(range(N_CAND),key=lambda j:(-sc[j],j))
        return order,[{"rank":k_+1,"candidate":j+1,"entity":s.names[3*j],"seal":s.seals[j],"score":sc[j],"is_expected":j+1==exp}for k_,j in enumerate(order[:10])]
    def status(s):return{"initialized":True,"engine":"synthetic","model":FROZEN_CONFIG["MODEL_ID"],"pointer":"L28H00"if s.mode!="config"else"L27H00","address":"L00-V UNCENTERED","address_dim":1024,"candidate_memories":128,
                         "model_frozen":True,"trainable_tensors":0,"query_time_candidate_B_forwards":0,"test528_lock":TEST528["lock"],"test528_result_sha":TEST528["result_sha"]}
    def verify(s):return{"weight_sentinel_pass":True,"initial_weight_sentinel":"a"*64,"current_weight_sentinel":"a"*64,"trainable_tensors":0,"pointer":"L28H00","address":"L00-V UNCENTERED","address_dim":1024,
                         "candidate_count":128,"query_time_candidate_B_forwards":0,"test528_lock":TEST528["lock"],"test528_result_sha":TEST528["result_sha"]}
    def items(s):return[{"item":i+1,"entity":s.names[3*i],"question":f"What seal does instrument {s.names[3*i]} carry? Give only the exact seal.","expected_candidate":i+1,"expected_seal":s.seals[i]}for i in range(N_CAND)]
    def field(s):return[{"candidate":i+1,"entity":s.names[3*i],"target_seal":s.seals[i],"B_tokens":21+i%3,"address_dim":1024}for i in range(N_CAND)]
    def config(s):
        c=dict(FROZEN_CONFIG);c.update(n_bmats=128,bmat_dims=[1024],seed=528528,fmt="QUESTION:\n{q}\n\nANSWER:",engine_sha256=ENGINE_SHA256_EXPECTED,test528_lock=TEST528["lock"],test528_result_sha=TEST528["result_sha"])
        if s.mode=="config":c["PTR_L"]=27
        return c
    def guard(s):s.calls+=1;return"c"*64 if(s.mode=="guard_changes"and s.calls>1)else"b"*64
    def frozen(s):return dict(training=False,trainable_tensors=0,lora=False,optimizer=False,hooks=0,foreign_hooks=0,foreign_modules={},sentinel="a"*64)
    def question_tokens(s,q):return len(q.split())+6
    def a_slots(s,i):return len(s._tok(i))
    def retrieve(s,i):
        it=s.items()[i-1];miss=(i%41==0);sc=s._scores(i,i,miss);order,top=s._top(sc,i);w=s._w(i);sp=s._span(i);am=max(range(len(w)),key=lambda t:w[t])
        rec=[dict(input_len=s.question_tokens(it["question"]),has_past=True,past_len=s.a_slots(i))]
        if s.mode=="b_forward":rec+= [dict(input_len=22,has_past=False,past_len=None)]*128
        r={"mode":"PRIMARY","item":i,"entity":it["entity"],"question":it["question"],"expected_candidate":i,"expected_seal":s.seals[i-1],"selected_candidate":order[0]+1,"selected_entity":s.names[3*order[0]],
           "selected_seal":s.seals[order[0]],"correct":order[0]+1==i,"rank":order.index(i-1)+1,"top1_score":sc[order[0]],"top2_score":sc[order[1]],"top1_top2_margin":sc[order[0]]-sc[order[1]],
           "pointer_hit":am in sp,"pointer_mass":sum(w[t]for t in sp),"pointer_argmax_position":am,"pointer_weights":w,"candidate_scores":sc,"top10":top,"latency_seconds":0.05,
           "query_time_candidate_B_forwards":0,"model_frozen":True}
        return r,rec,0.05
    def pointer_data(s,i):
        w=s._w(i);sp=s._span(i);am=max(range(len(w)),key=lambda t:w[t]);it=s.items()[i-1]
        return({"item":i,"entity":it["entity"],"seal":s.seals[i-1],"pointer_layer":28,"pointer_head":0,"weights":w,"argmax_position":am,"gold_span_positions":sp,"pointer_hit":am in sp,"pointer_mass":sum(w[t]for t in sp)},
               [dict(input_len=s.question_tokens(it["question"]),has_past=True,past_len=s.a_slots(i))],0.05)
    def controls(s,i):
        it=s.items()[i-1];out={}
        for mode in("PRIMARY","UNIFORM","SHIFTED","NO_A"):
            if mode=="PRIMARY":sc=s._scores(i,i,i%41==0)
            elif mode=="NO_A":sc=[0.0]*N_CAND
            else:rng=random.Random(sum(map(ord,mode))+i);sc=[0.30+0.12*rng.random()for _ in range(N_CAND)]
            order=sorted(range(N_CAND),key=lambda j:(-sc[j],j))
            out[mode]={"rank":order.index(i-1)+1,"selected_candidate":order[0]+1,"selected_seal":s.seals[order[0]],"correct":order[0]+1==i,"top1_score":sc[order[0]],"top2_score":sc[order[1]],
                       "margin":sc[order[0]]-sc[order[1]],"candidate_scores":sc}
        return({"item":i,"entity":it["entity"],"expected_candidate":i,"expected_seal":s.seals[i-1],"controls":out,"query_time_candidate_B_forwards":0,"model_frozen":True},
               [dict(input_len=s.question_tokens(it["question"]),has_past=True,past_len=s.a_slots(i))],0.05)
    def counterfactual(s,i,j):
        it=s.items()[i-1];sc=s._scores(i,j);order,top=s._top(sc,j);w=s._w(i);T=len(w)
        r={"mode":"COUNTERFACTUAL","source_item":i,"entity":it["entity"],"question":it["question"],"original_candidate":i,"original_seal":s.seals[i-1],"counterfactual_candidate":j,"counterfactual_seal":s.seals[j-1],
           "selected_candidate":order[0]+1,"selected_entity":s.names[3*order[0]],"selected_seal":s.seals[order[0]],"followed_counterfactual":order[0]+1==j,"rank":order.index(j-1)+1,"top1_score":sc[order[0]],
           "top2_score":sc[order[1]],"top1_top2_margin":sc[order[0]]-sc[order[1]],"pointer_hit":True,"pointer_mass":max(w),"pointer_argmax_position":max(range(T),key=lambda t:w[t]),"pointer_weights":w,
           "candidate_scores":sc,"top10":top,"latency_seconds":0.1,"query_time_candidate_B_forwards":0,"model_frozen":True}
        return r,[dict(input_len=T,has_past=False,past_len=None),dict(input_len=s.question_tokens(it["question"]),has_past=True,past_len=T)],0.1
    def extras(s,i,w,sel):
        rng=np.random.default_rng(i);p=(rng.normal(0,.3,1024)).tolist();sc=s._scores(i,i,i%41==0);b=(np.asarray(p)+rng.normal(0,.2,1024)).tolist()
        return dict(p=p,best_row=b,best_slot=7,best_cos=sc[sel-1],b_rows=21,tokens=s._tok(i),shifted=list(np.roll(np.asarray(w),len(w)//2)),uniform=[0.0]+[1/(len(w)-1)]*(len(w)-1))
    def panel_strings(s):return sorted(set(s.names)|set(s.seals)|set(s.dis))
    def sync(s):pass
    def reset_peak(s):pass
    def peak_gib(s):return 0.0
def drain(gen):
    while True:
        try:next(gen)
        except StopIteration as s_:return s_.value
RUN_COUNTER=globals().get("RUN_COUNTER",0);GPU_LOCK=globals().get("GPU_LOCK")or threading.Lock();SESSION_LEDGER=globals().get("SESSION_LEDGER",[])
LEDGER_PATH=ROOT/"session_ledger.jsonl"
SKIP=getattr(gr,"skip",None)or gr.update
_K=object()
FILE_KEYS=["zip","json","txt","man","ledger"]
DL_LABELS=["⬇ EVIDENCE PACKAGE (.zip)","⬇ FULL RUN LOG (.json)","⬇ READABLE RUN LOG (.txt)","⬇ RUN MANIFEST (.json)","⬇ SESSION LEDGER (.jsonl)"]
DL_ALL_LABEL=f"⬇ DOWNLOAD ALL {N_FIGS} FIGURES (JPEG, 300 dpi)"
RAW_KEYS=["txt","json","man","ledger"]
N_OUTPUTS=2+2+len(FILE_KEYS)*2+len(RAW_KEYS)
_GR_MAJOR=int(re.match(r"\d+",gr.__version__).group())
try:
    from gradio.routes import API_PREFIX as _GR_API_PREFIX
except Exception:
    _GR_API_PREFIX="/gradio_api"if _GR_MAJOR>=5 else""
FILE_URL_PREFIX=f"{_GR_API_PREFIX}/file="
def dl_update(path,label):
    if path is None:
        try:return gr.DownloadButton(label=label,value=None,interactive=False)
        except Exception:return gr.update(value=None,interactive=False)
    try:return gr.DownloadButton(label=label,value=str(path),interactive=True)
    except Exception:return gr.update(value=str(path),interactive=True)
def btn_update(active):
    try:return gr.Button(value=DL_ALL_LABEL,interactive=bool(active))
    except Exception:return gr.update(value=DL_ALL_LABEL,interactive=bool(active))
def jpg_urls_json(paths):return json.dumps([{"url":FILE_URL_PREFIX+quote(str(Path(p)),safe="/"),"name":Path(p).name}for p in paths],ensure_ascii=False)
def pack(status=_K,gal=_K,files=_K,jpgs=_K,raw=_K):
    out=[SKIP()if status is _K else status,SKIP()if gal is _K else gal]
    if jpgs is _K:out+=[SKIP(),SKIP()]
    elif jpgs is None:out+=[btn_update(False),""]
    else:
        for pth in jpgs:
            if not file_ready(pth):raise RuntimeError(f"figure missing or empty: {pth}")
        out+=[btn_update(True),jpg_urls_json(jpgs)]
    if files is _K:out+=[SKIP()for _ in range(len(FILE_KEYS)*2)]
    elif files is None:out+=[dl_update(None,l)for l in DL_LABELS]+[None]*len(FILE_KEYS)
    else:
        paths=[files.get(k_)for k_ in FILE_KEYS]
        for pth in paths:
            if pth is not None and not file_ready(pth):raise RuntimeError(f"download artifact missing or empty: {pth}")
        out+=[dl_update(p,l)for p,l in zip(paths,DL_LABELS)]+[None if p is None else str(p)for p in paths]
    if raw is _K:out+=[SKIP()for _ in RAW_KEYS]
    elif raw is None:out+=[""]*len(RAW_KEYS)
    else:out+=[raw.get(k_,"")for k_ in RAW_KEYS]
    return tuple(out)
def card_html(title,body,kind="info"):return f'<div class="kz card {kind}"><div class="h">{html.escape(title)}</div><div class="small">{body}</div></div>'
def pbar_html(done,total):
    pc_=100.0*done/max(1,total);return f'<div style="margin-top:6px;background:#e5e7eb;height:10px"><div style="width:{pc_:.1f}%;height:10px;background:#1d4ed8"></div></div><div class="small">{done}/{total}</div>'
def stage_card(e):return card_html(f"{e['stage']}/{N_STAGES} · {e['title']}",html.escape(e["body"])+pbar_html(e["stage"],N_STAGES),"info")
def item_choices(I):return[f"{it['item']:03d} · {it['entity']}"for it in I.items()]
CF_AUTO=f"auto · item + {CF_OFFSET} (mod 128)"
def cf_choices(I):return[CF_AUTO]+[f"{it['item']:03d} · {it['entity']}"for it in I.items()]
def parse_item(v):
    m=re.match(r"\s*(\d{1,3})",str(v or""));return int(m.group(1))if m else None
def ledger_text():return"\n".join(json.dumps(x,ensure_ascii=False)for x in SESSION_LEDGER)
def item_info_html(item_v,cf_v,I=None):
    I=I or INSTR;i=parse_item(item_v)
    if not i:return card_html("Item","Select an item.","info")
    it=I.items()[i-1];j=parse_item(cf_v)if cf_v and cf_v!=CF_AUTO else default_cf_target(i);jt=I.items()[j-1]
    warn=' <b style="color:#b91c1c">counterfactual target must differ from the item</b>'if j==i else""
    body=(f'<span class="mono">question</span> {html.escape(it["question"])}<br><span class="mono">expected</span> B#{i:03d} · seal {html.escape(it["expected_seal"])} '
          f'<span class="small" style="opacity:.75">(audit metadata; not an input to retrieval)</span><br><span class="mono">counterfactual</span> {html.escape(it["entity"])} → seal {html.escape(jt["expected_seal"])} · target B#{j:03d}{warn}')
    return card_html(f"Item {i:03d} · {it['entity']}",body,"info")
def engine_card_html(I):
    inf=I.info;st=I.status()
    rows=[("model",f"{inf['model_id']} · frozen · {inf['dtype']}"),("GPU",inf["gpu"]),("pointer · address",f"{st.get('pointer')} · {st.get('address')} · {st.get('address_dim')}-D"),
          ("candidates",f"{st.get('candidate_memories')} precomputed B memories · max cosine"),("engine",f"{inf['engine_file']} · SHA-256 {inf['engine_sha256'][:16]}…"),
          ("weights",f"sentinel {inf['sentinel0'][:16]}… · all-parameter guard {inf['guard0'][:16]}…"),("init",f"{inf['init_seconds']:.1f} s" if isinstance(inf['init_seconds'],float)and math.isfinite(inf['init_seconds'])else"reused")]
    body="".join(f'<span class="mono">{html.escape(a)}</span> {html.escape(str(b))}<br>'for a,b in rows)
    body+=(f'<div class="small" style="margin-top:6px"><b>Sealed TEST528 reference (archived, 128 items; not recomputed here):</b> primary {fr(TEST528["primary"])} = {TEST528["primary"][2]} · '
           f'counterfactual {fr(TEST528["counterfactual"])} = {TEST528["counterfactual"][2]} · shifted {fr(TEST528["shifted"])} = {TEST528["shifted"][2]} · no-A {fr(TEST528["no_a"])} = {TEST528["no_a"][2]} · '
           f'identity regime: {TEST528["identity_regime"]}.</div>')
    return card_html("Engine status",body,"on")
READY_HTML=card_html("Ready",f"Select an item and press <b>RUN</b>. One run calls retrieve() (twice), get_pointer_data(), run_controls() and counterfactual() through the engine API, counts every model forward, "
    f"re-derives all reported values from the raw score vectors, seals the payload and renders {N_FIGS} figures (JPEG 300 dpi + vector PDF). Each run is a live demonstration observation; "
    "the TEST528 values shown above are archived reference results.","info")
def is_cuda_error(ex):
    s=f"{type(ex).__name__} {ex}".lower();return"cuda"in s or"device-side assert"in s or"cublas"in s
def run_handler(item_v,cf_v):
    global RUN_COUNTER
    i=parse_item(item_v);j=parse_item(cf_v)if cf_v and cf_v!=CF_AUTO else(default_cf_target(i)if i else None)
    if not i or not j or i==j:
        yield pack(card_html("Invalid selection","Choose an item and a counterfactual target different from the item.","warn"));return
    if not GPU_LOCK.acquire(blocking=False):
        yield pack(card_html("Busy","Another run is in progress. Please try again in a moment.","warn"));return
    stage_name="initialisation"
    try:
        RUN_COUNTER+=1;run_id=f"MAM528DEMO-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}-{RUN_COUNTER:04d}-I{i:03d}"
        prune_runs(4);run_dir=ROOT/run_id;run_dir.mkdir(parents=True,exist_ok=True);gen=execute_run(INSTR,dict(run_id=run_id,run_dir=run_dir,item=i,cf_target=j));first=True
        while True:
            try:e=next(gen)
            except StopIteration as s:B=s.value;break
            stage_name=f"{e['stage']}/{N_STAGES} {e['title']}"
            yield pack(stage_card(e),[],None,None,None)if first else pack(stage_card(e));first=False
        SESSION_LEDGER.append(B["ledger"]);LEDGER_PATH.write_text(ledger_text()+"\n",encoding="utf-8")
        P=B["P"];L=P["live"];r=L["primary"];cf=L["counterfactual"];c=L["controls"]["controls"];ok=r["correct"]
        raw={"txt":B["txt"].read_text(encoding="utf-8"),"json":B["payload"].read_bytes().decode("utf-8"),"man":B["manifest"].read_text(encoding="utf-8"),"ledger":ledger_text()}
        body=(f"<b>{html.escape(B['run_id'])}</b> · LIVE DEMO OBSERVATION<br>"
              f'<span class="mono">primary</span> item {i:03d} → selected B#{r["selected_candidate"]:03d} · {"correct" if ok else "incorrect"} · rank of expected {r["rank"]}/128 · '
              f'top-1 {r["top1_score"]:.6f} · margin {r["top1_top2_margin"]:+.6f} · pointer argmax slot {r["pointer_argmax_position"]} (diagnostic span mass {r["pointer_mass"]:.3f})<br>'
              f'<span class="mono">controls</span> '+" · ".join(f"{k_} rank {v_['rank']}"for k_,v_ in c.items())+"<br>"
              f'<span class="mono">counterfactual</span> target B#{j:03d} → selected B#{cf["selected_candidate"]:03d} · followed: {"yes" if cf["followed_counterfactual"] else "no"}<br>'
              f'<span class="mono">measured forwards</span> '+" · ".join(f"{k_} {len(v_)}"for k_,v_ in L["forwards"].items())+" · weights unchanged (sentinel + all-parameter guard)<br>"
              f"verdict <b>{html.escape(B['verdict'])}</b> · {N_FIGS} figures + PDFs + ZIP · {B['checks']}/{B['checks']} checks PASS<br>"
              f'<span class="mono">payload SHA-256 {B["sha"]}</span>')
        yield pack(card_html(f"{N_STAGES}/{N_STAGES} · Sealed",body,"on"),[(str(p),c_)for p,c_ in B["imgs"]],
                   {"zip":B["zip"],"json":B["payload"],"txt":B["txt"],"man":B["manifest"],"ledger":LEDGER_PATH},[p for p,_ in B["imgs"]],raw)
    except Exception as ex:
        print("="*140);print(f"RUN FAILED — stage: {stage_name}");print(f"{type(ex).__name__}: {ex}");traceback.print_exc();print("="*140)
        kind="AUDIT FAIL — nothing was sealed. "if isinstance(ex,AuditFail)else"";partial=getattr(ex,"partial",None)
        if partial:kind+="The raw outputs gathered so far were preserved (FULL RUN LOG). "
        tip="<br><b>CUDA error: Runtime → Restart runtime, then run PARTS 1–3 again.</b>"if is_cuda_error(ex)else"<br>The full traceback is printed in the Colab console."
        card=card_html("Run failed",f"{html.escape(kind)}Stage: {html.escape(stage_name)}<br>{html.escape(type(ex).__name__)}: {html.escape(str(ex)[:600])}{tip}","err")
        if partial:yield pack(card,[],{"json":partial},None,{"txt":traceback.format_exc(),"json":Path(partial).read_text(encoding="utf-8"),"man":"","ledger":ledger_text()})
        else:yield pack(card,[],None,None,None)
    finally:
        plt.close("all");GPU_LOCK.release()
        try:torch.cuda.empty_cache()
        except Exception:pass
def verify_handler():
    if not GPU_LOCK.acquire(blocking=False):return"Busy: a run is in progress."
    try:
        t=time.perf_counter();g=INSTR.guard();v=INSTR.verify();fz=INSTR.frozen()
        out=dict(verify_engine=v,engine_status=INSTR.status(),all_parameter_guard_now=g,all_parameter_guard_startup=INSTR.info["guard0"],guard_unchanged=g==INSTR.info["guard0"],
                 sentinel_unchanged=fz["sentinel"]==INSTR.info["sentinel0"],trainable_tensors=fz["trainable_tensors"],foreign_hooks=fz["foreign_hooks"],
                 engine_sha256=INSTR.config()["engine_sha256"],engine_sha256_expected=ENGINE_SHA256_EXPECTED,checked_utc=utc_now(),seconds=round(time.perf_counter()-t,3))
        return json.dumps(jsafe(out),indent=2,ensure_ascii=False)
    finally:GPU_LOCK.release()
# ---------------- service self-test (runs before the public link is printed) ----------------
def service_selftest():
    global QUIET;out=[];d=ROOT/"_selftest";shutil.rmtree(d,ignore_errors=True);d.mkdir(parents=True);(d/"w.txt").write_text("ok");out.append("ROOT writable");QUIET=True
    try:
        for it_ in(1,41):
            dd=d/f"ok{it_}";dd.mkdir()
            B=drain(execute_run(SelfTestInstrument("ok"),dict(run_id=f"SELFTEST-I{it_:03d}",run_dir=dd,item=it_,cf_target=None)))
            assert len(B["imgs"])==N_FIGS and len(B["pdfs"])==N_FIGS and file_ready(B["zip"])and B["verdict"].endswith("INTEGRITY_VERIFIED")
        out.append(f"full pipeline on a synthetic instrument (correct and incorrect primary): {N_FIGS}/{N_FIGS} figures (JPEG 300 dpi + PDF), ZIP, {B['checks']} checks")
        for mode,what in(("guard_changes","a changed weight guard"),("b_forward","extra model forwards during retrieve() (candidate-B forwarding)"),("config","a changed pointer coordinate"),("nan","a non-finite candidate score")):
            d2=d/mode;d2.mkdir()
            try:drain(execute_run(SelfTestInstrument(mode),dict(run_id="SELFTEST-"+mode.upper(),run_dir=d2,item=5,cf_target=None)))
            except AuditFail:out.append(f"fail-closed: {what} aborts the run (nothing sealed)")
            else:raise RuntimeError(f"self-test: {what} did not abort the run")
    finally:QUIET=False
    if len(pack(READY_HTML))!=N_OUTPUTS or len(pack(READY_HTML,[],None,None,None))!=N_OUTPUTS:raise RuntimeError("callback output count mismatch")
    out.append(f"callback output count = {N_OUTPUTS}");out.append(f"file route {FILE_URL_PREFIX}");shutil.rmtree(d,ignore_errors=True);return out
say("="*140);say(DEMO_TITLE+" — PART 3 / 3 · SELF-TEST AND INTERFACE");say("[1/3] SERVICE SELF-TEST")
for c_ in service_selftest():say(" PASS ·",c_)
# ---------------- interface ----------------
say("[2/3] INTERFACE")
CSS="""
:root{--kz-on:#0f766e;--kz-fg:#111827;--kz-card:#ffffff;--kz-bd:#d1d5db;--kz-a:#1d4ed8;--kz-err:#b91c1c;--kz-warn:#b45309}
.dark{--kz-on:#2dd4bf;--kz-fg:#f3f4f6;--kz-card:#111827;--kz-bd:#4b5563;--kz-a:#60a5fa;--kz-err:#f87171;--kz-warn:#fbbf24}
html,body{overflow-x:hidden!important}
.gradio-container{width:100%!important;max-width:900px!important;margin:0 auto!important;padding-left:10px!important;padding-right:10px!important;box-sizing:border-box!important}
.gradio-container *{box-sizing:border-box}
.kz{color:var(--kz-fg)!important;max-width:100%;overflow-wrap:anywhere;line-height:1.45}
.kz *{color:inherit}
.hero{padding:8px 2px 2px;border-bottom:1px solid var(--kz-bd);margin-bottom:6px}
.brand{font-size:clamp(20px,5.6vw,28px);font-weight:700;letter-spacing:.3px;line-height:1.15}
.sub{font-size:clamp(13px,3.6vw,15px);margin-top:4px}
.by{font-size:12px;opacity:.75;margin-top:4px}
.card{background:var(--kz-card);border:1px solid var(--kz-bd);border-left:4px solid var(--kz-bd);border-radius:4px;padding:10px 12px;margin:6px 0}
.card.on{border-left-color:var(--kz-on)}.card.err{border-left-color:var(--kz-err)}.card.warn{border-left-color:var(--kz-warn)}.card.info{border-left-color:var(--kz-a)}
.card .h{font-weight:700;font-size:15px;margin-bottom:4px}
.small{font-size:13.5px}.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;opacity:.8;margin-right:4px}
#kz_run button,#kz_run{font-size:clamp(15px,4.4vw,18px)!important;font-weight:700!important;min-height:54px!important}
"""
DL_ALL_JS="""(u)=>{let items=[];try{items=JSON.parse(u||"[]")}catch(e){items=[]}
let i=0;const next=()=>{if(i>=items.length)return;const it=items[i++];const a=document.createElement('a');a.href=it.url;a.download=it.name;a.rel='noopener';
document.body.appendChild(a);a.click();setTimeout(()=>{a.remove();next()},900)};next();return u;}"""
HERO=(f'<div class="kz hero"><div class="brand">AKBASCORE MAM · Mistral-7B-Instruct-v0.3</div>'
      '<div class="sub">Executable instrument for the frozen TEST528 associative-retrieval mechanism: ÇAĞRIİZ pointer L28H00 → L00-V uncentered 1024-D address → max cosine over 128 precomputed B memories '
      '(0 query-time candidate-B forwards). Live runs are demonstration observations; TEST528 values are archived reference results.</div>'
      f'<div class="by">{html.escape(AUTHOR)} · {html.escape(AUTHOR_PLACE)} · {html.escape(COPYRIGHT)}</div></div>')
def _tb(**kw):
    for extra in(([{"buttons":["copy"]}]if _GR_MAJOR>=6 else[])+[{"show_copy_button":True},{}]):
        try:return gr.Textbox(**extra,**kw)
        except Exception:pass
def _gallery(**kw):
    try:return gr.Gallery(columns=1,height="auto",object_fit="contain",preview=False,**kw)
    except TypeError:return gr.Gallery(**kw)
if"demo"in globals():
    try:demo.close()
    except Exception:pass
if _GR_MAJOR>=6:_blocks=gr.Blocks(title="AKBASCORE MAM · Mistral instrument")
else:
    try:_blocks=gr.Blocks(css=CSS,title="AKBASCORE MAM · Mistral instrument")
    except TypeError:_blocks=gr.Blocks(title="AKBASCORE MAM · Mistral instrument")
_ICH=item_choices(INSTR);_CCH=cf_choices(INSTR)
with _blocks as demo:
    gr.HTML(HERO);gr.HTML(engine_card_html(INSTR))
    with gr.Row():
        item_dd=gr.Dropdown(choices=_ICH,value=_ICH[0],label="Item (active A memory · question)",interactive=True)
        cf_dd=gr.Dropdown(choices=_CCH,value=CF_AUTO,label="Counterfactual target B memory",interactive=True)
    item_info=gr.HTML(item_info_html(_ICH[0],CF_AUTO))
    run_btn=gr.Button("RUN · retrieve · controls · counterfactual · seal",variant="primary",elem_id="kz_run")
    status=gr.HTML(READY_HTML)
    gallery=_gallery(label=f"{N_FIGS} figures (tap to open · JPEG 300 dpi; vector PDFs in the ZIP)",show_label=True)
    dl_all=gr.Button(DL_ALL_LABEL,interactive=False)
    jpg_urls=gr.Textbox(value="",visible=False)
    with gr.Row():
        dlb=[gr.DownloadButton(label=l,value=None,interactive=False)for l in DL_LABELS]
    with gr.Accordion("Direct file links (fallback)",open=False):
        fls=[gr.File(label=n,interactive=False)for n in("ZIP · figures, PDFs, payload, logs","FULL RUN LOG (.json)","READABLE RUN LOG (.txt)","RUN MANIFEST (.json)","SESSION LEDGER (.jsonl)")]
    with gr.Tabs():
        with gr.Tab("READABLE RUN LOG (.txt)"):raw_txt=_tb(lines=18,max_lines=40,label="readable run log")
        with gr.Tab("FULL RUN LOG (.json)"):raw_json=_tb(lines=18,max_lines=40,label="sealed canonical payload")
        with gr.Tab("MANIFEST (.json)"):raw_man=_tb(lines=12,max_lines=30,label="run manifest")
        with gr.Tab("SESSION LEDGER"):raw_led=_tb(lines=10,max_lines=30,label="one line per sealed run in this session")
    with gr.Accordion("Engine verification · verify_engine() + weight guard",open=False):
        ver_btn=gr.Button("Run verification now")
        ver_out=_tb(lines=14,max_lines=40,label="verification record")
    OUTS=[status,gallery,dl_all,jpg_urls]+dlb+fls+[raw_txt,raw_json,raw_man,raw_led]
    if len(OUTS)!=N_OUTPUTS:raise RuntimeError(f"UI outputs {len(OUTS)} != {N_OUTPUTS}")
    item_dd.change(lambda a,b:item_info_html(a,b),inputs=[item_dd,cf_dd],outputs=item_info)
    cf_dd.change(lambda a,b:item_info_html(a,b),inputs=[item_dd,cf_dd],outputs=item_info)
    run_btn.click(run_handler,inputs=[item_dd,cf_dd],outputs=OUTS)
    ver_btn.click(verify_handler,inputs=None,outputs=ver_out)
    try:dl_all.click(None,inputs=[jpg_urls],outputs=None,js=DL_ALL_JS)
    except TypeError:dl_all.click(None,inputs=[jpg_urls],outputs=None,_js=DL_ALL_JS)
try:demo.queue(default_concurrency_limit=1)
except TypeError:demo.queue(concurrency_count=1)
say("[3/3] LAUNCH (public share link) — open the printed gradio.live link, choose an item and press RUN.")
_LAUNCH=dict(share=True,allowed_paths=ALLOWED_PATHS,show_error=True,inline=False)
if _GR_MAJOR>=6:_LAUNCH["css"]=CSS
try:demo.launch(**_LAUNCH)
except TypeError as _ex:
    if"css"not in str(_ex):raise
    _LAUNCH.pop("css",None);demo.launch(**_LAUNCH)

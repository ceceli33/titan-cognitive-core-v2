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

# AKBASCORE MAM · PERSISTENT ASSOCIATIVE MACHINE MEMORY — WORLD LAUNCH DEMO (single Colab cell · paste PART 1 + PART 2 + PART 3 into ONE cell)
# Copyright © 2026 Mustafa Akbaş. All rights reserved. · Mersin, Türkiye · 06 October 2026
# ENGINE = TEST524 (direct child of sealed TEST523), transplanted without semantic change:
#   frozen Qwen/Qwen2.5-7B-Instruct · BF16 · SDPA · ÇAĞRIİZ pointer = native L23H12 Q·K + RoPE · address = L02-V UNCENTERED · match = max cosine
#   128 records → 256 independently forged numeric A/B memories · 5 fixed showcase questions · every retrieval scores all 128 B memories.
#   NO training · NO LoRA · NO optimizer · NO weight update · NO gold B ID to the retriever · NO token-ID / decoded-text / LM-head router · NO learned router.
# DEMO LAYER (measurement / presentation only): telemetry read from tensors the engine already produced, result re-derivation from raw scores,
#   all-parameter weight guard, source-removal audit, poster rendering, stale-reference scan, sealing, ZIP, Gradio interface.
# Result classes: LIVE (measured in this run) · SEALED (TEST524 launch log / TEST523 seal) · BY CONSTRUCTION (property of the code path) — never merged.
import os,sys,io,re,json,math,time,html,random,shutil,hashlib,zipfile,platform,textwrap,threading,subprocess,importlib.util,traceback,gc
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import quote
from collections import Counter
#<<CONST_BEGIN>>
TEST="524";SEED=524;MODEL_ID="Qwen/Qwen2.5-7B-Instruct";MODEL_SHORT="Qwen2.5-7B-Instruct";N=128
FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
PTR_L=23;PTR_H=12;BL=2;BK="V";ARCH=(28,3584,28,4,128)
PARENT523="9cd1072abbb5b5e7063e48a72d1d32dc90560b2d36aeb2a196ff47be9d6835b8"
SEALED524_LOCK="9e8b36b25b311c631e73d9fc44947fcf9833a255b3263315f4ec807c95128126"
SHOWCASE_IDX=[6,37,63,90,122]
CANON_GPU="NVIDIA A100-SXM4-40GB"
AUTHOR="Mustafa Akbaş";AUTHOR_PLACE="Mersin, Türkiye";LAUNCH_DATE="06 October 2026";COPYRIGHT="Copyright © 2026 Mustafa Akbaş. All rights reserved."
PRODUCT="AKBASCORE MAM";PRODUCT_LONG="Persistent Associative Machine Memory"
# ---------------- TEST524 panel definitions (verbatim) ----------------
SYL1=["Aq","Bryn","Cav","Dex","Eln","Frax","Gyr","Hex","Iln","Jor","Kex","Lyr","Mav","Nex","Oryn","Pax",
      "Qyr","Rex","Savn","Tov","Uln","Vex","Wyr","Xav","Yex","Zyr","Axl","Bov","Cyn","Drex","Evr","Fyn"]
SYL2=["adar","bren","cyr","dax","elor","fyn","grel","hyn","ivar","jor","kyr","lor","myn","nex","or","pyr",
      "qen","rix","sor","tyn","ul","vyr","wen","xir","yor","zen"]
def build_names(n):
    out=[]
    for a in SYL1:
        for b in SYL2:
            x=a+b
            if x not in out:out.append(x)
            if len(out)>=n:return out
    raise RuntimeError(f"Name pool too small: need {n}, found {len(out)}.")
LOCK={"test":TEST,"parent523":PARENT523,"model":MODEL_ID,"N":N,"pointer":[PTR_L,PTR_H],
      "address":[BL,BK,"UNCENTERED"],"match":"max_cosine","showcase_records":[i+1 for i in SHOWCASE_IDX],
      "forbidden":["HEAD_SCAN","LAYER_SCAN","TOPK_SCAN","TOKEN_ID_ADDRESS","DECODED_SEAL_ROUTER",
      "LM_HEAD_ROUTER","GOLD_MEMORY_ID_TO_RETRIEVER","QUERY_TIME_B_FORWARD","TRAINING","DRA",
      "LEARNED_ROUTER","POSTHOC_SELECTION"]}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def qA(e):return f"What seal does instrument {e} carry? Give only the exact seal."
# ---------------- SEALED HISTORICAL RECORDS (read from the named logs; never produced by the live run) ----------------
SEALED524=dict(log="TEST524 world-launch log (06 October 2026)",gpu=CANON_GPU,lock=SEALED524_LOCK,parent=PARENT523,showcase=(4,5),runtime_s=69.18,integrity="VERIFIED",
    guard="a24946e5cd9766c40d0e901234de018701195dc1d74804525f1fe9b19fa19e8e",
    queries=[dict(k=1,record=7,entity="Aqsor",selected=7,rank=1,top1=0.728781,margin=0.129290,peak=0.4400,pos=7,
                  top5=[[7,0.728781],[22,0.599491],[20,0.599313],[108,0.595058],[92,0.591225]]),
             dict(k=2,record=38,entity="Elnhyn",selected=10,rank=19,top1=0.556228,margin=0.002424,peak=0.4117,pos=17,
                  top5=[[10,0.556228],[98,0.553804],[103,0.540995],[47,0.540648],[46,0.540589]]),
             dict(k=3,record=64,entity="Hexhyn",selected=64,rank=1,top1=0.959768,margin=0.313096,peak=0.8928,pos=16,
                  top5=[[64,0.959768],[19,0.646671],[21,0.630443],[90,0.630038],[121,0.629823]]),
             dict(k=4,record=91,entity="Kexkyr",selected=91,rank=1,top1=0.898095,margin=0.196410,peak=0.7662,pos=9,
                  top5=[[91,0.898095],[81,0.701685],[59,0.692537],[107,0.651887],[16,0.640404]]),
             dict(k=5,record=123,entity="Oryncyr",selected=123,rank=1,top1=0.865097,margin=0.202585,peak=0.7686,pos=27,
                  top5=[[123,0.865097],[74,0.662512],[39,0.648305],[46,0.648246],[114,0.646589]])])
SEALED523=dict(log="TEST523 external replication seal",lock=PARENT523,r1=(118,128),r1_pct="92.19%",cf=(117,128),cf_pct="91.41%",
    verdict="STRONG_EXTERNAL_REPLICATION_SEAL_L23H12_CAGRIIIZ",pointer="L23H12",address="L02-V UNCENTERED",match="max cosine")
SCOPE={
 "demonstrated":[
  "An active numeric A memory and a natural-language question drive the frozen model's L23H12 attention channel, producing a question-dependent recall trace (ÇAĞRIİZ).",
  "The recall trace weights the layer-2 V tensor state of the active A memory into a 512-number address.",
  "That address is compared by maximum cosine similarity with the address structures of all 128 independently forged B memories; the highest score selects the associated B memory.",
  "The original source text is removed from the live retrieval path before any question is asked.",
  "Frozen weights. No training, no gold B identifier, no decoded-text, token-ID or LM-head router, no learned router."],
 "not_claimed":[
  "Retrieval of the first (A) memory from an empty global bank: in this experiment A is the active memory.",
  "A separately sealed natural-language answer-generation stage: this demo covers associative retrieval.",
  "Capacity: 128 is the experimental bank size, not a limit. Larger banks and hierarchical/indexed retrieval are a research direction, not demonstrated here."],
 "researcher_notes":[
  "Interpretation (not a measured result): layer 2 is an early transformer layer, and its V representation may be strongly weighted by the identity of the token at each position. The demonstrated contribution is that the frozen model's own question-dependent attention (L23H12) selects which positions form the address.",
  "Persistence: the numeric memories are held as tensors for the duration of the run; this run does not write them to disk and reload them.",
  "Candidate-B LLM forwards = 0 is a property of the code path: B scoring operates on precomputed address matrices and makes no model call. It is not instrumented with a hook, so the engine is unchanged.",
  "TEST524 (5 fixed live questions) and TEST523 (128-question external replication) are separate experiments and are never added together.",
  "Score bars and glow on the posters are visual metaphors for numeric similarity values."]}
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
# ---- exact binomial (Clopper-Pearson) interval, pure Python ----
def _lpmf(i,n,p):return math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)+i*math.log(p)+(n-i)*math.log1p(-p)
def _tail(lo,hi,n,p):
    ls=[_lpmf(i,n,p)for i in range(lo,hi+1)]
    if not ls:return 0.0
    m=max(ls);return math.exp(m)*sum(math.exp(x-m)for x in ls)
def cp_interval(k,n,alpha=0.05):
    a=alpha/2
    if n<=0:return(0.0,1.0)
    def bis(f,incr):
        l,h=1e-12,1-1e-12
        for _ in range(100):
            m=(l+h)/2
            if (f(m)>=a)==incr:h=m
            else:l=m
        return(l+h)/2
    lo=0.0 if k<=0 else bis(lambda p:_tail(k,n,n,p),True)
    hi=1.0 if k>=n else bis(lambda p:_tail(0,k,n,p),False)
    return(lo,hi)
def pc(x):return f"{100*x:.2f}%"
def ci_text(k,n):
    lo,hi=cp_interval(k,n);return f"{k}/{n} · 95% CI [{pc(lo)}, {pc(hi)}]"
def derive(raw):
    """Re-derive every retrieval result from the raw live scores (128 per question) and ÇAĞRIİZ weights, with the TEST524 ordering rule."""
    qs=[];sv={q["k"]:q for q in SEALED524["queries"]}
    for e in sorted(raw["queries"],key=lambda z:z["k"]):
        s=[float(x)for x in e["scores"]]
        if len(s)!=N:raise AuditFail(f"query {e['k']}: {len(s)} scores, expected {N}")
        if e["question"]!=qA(e["entity"]):raise AuditFail(f"query {e['k']}: question text is not the TEST524 template")
        order=sorted(range(N),key=lambda j:(-s[j],j));g=e["record"]-1
        w=[float(x)for x in e["w"]];peak=max(w);pos=w.index(peak);ent=-sum(x*math.log(x)for x in w if x>0)
        sel=order[0]+1;rank=order.index(g)+1
        qs.append(dict(k=e["k"],record=e["record"],entity=e["entity"],question=e["question"],selected=sel,rank=rank,correct=bool(sel==e["record"]),
            top1=s[order[0]],top2=s[order[1]],margin=s[order[0]]-s[order[1]],gold_score=s[g],top10=[[j+1,s[j]]for j in order[:10]],
            peak=peak,pos=pos,entropy=ent,T=len(w),secs=e.get("secs")))
    ok=sum(int(q["correct"])for q in qs)
    same_set=sorted(sv)==[q["k"]for q in qs]
    sel_match=bool(same_set and all(q["selected"]==sv[q["k"]]["selected"]for q in qs))
    rank_match=bool(same_set and all(q["rank"]==sv[q["k"]]["rank"]for q in qs))
    dmax=max((abs(q["top1"]-sv[q["k"]]["top1"])for q in qs if q["k"]in sv),default=float("nan"))
    return dict(queries=qs,correct=ok,total=len(qs),misses=[q["k"]for q in qs if not q["correct"]],
        replay=dict(selected_match=sel_match,rank_match=rank_match,top1_max_abs_diff=dmax,match=bool(sel_match and rank_match)))
# ---- stale-reference scanner: scans GENERATED output (poster text, JSON, TXT, file names), never its own source ----
ALLOWED_TESTS={"523","524"}
STALE=[("other-model",r"[Mm]istral"),("other-arch-layers",r"\b32[ -]?(?:transformer )?(?:layers?|L)\b"),("other-arch-hidden",r"(?i:hidden[^0-9]{0,15}4096|\bH\s?=?\s?4096\b)"),
       ("other-arch-kv",r"\b8\s?KV\b"),("old-codec",r"\b[KVD]\s?120\b"),("compression-wording",r"(?i)\bcompress"),("capacity-overclaim",r"(?i)\bunlimited\b|\binfinite\b"),
       ("complexity-claim",r"O\(1\)"),("neuron-wording",r"(?i)\bneuron"),("doi",r"zenodo|10\.5281")]
def stale_scan(name,text,allow=()):
    for a in allow:
        if a:text=text.replace(a,"<allowed>")
    hits=[(name,lab,m.group(0))for lab,p in STALE for m in re.finditer(p,text)]
    hits+=[(name,"old-test-number",m.group(0))for m in re.finditer(r"TEST\s?(\d+)",text)if m.group(1)not in ALLOWED_TESTS]
    return hits
def strip_strings(text,strings):
    """Remove panel strings (random native-token seals, classes, names) before the stale scan: the scan targets our labels, not panel data."""
    for s in sorted(set(strings),key=len,reverse=True):
        if len(s)>=2:
            for v in(s,json.dumps(s,ensure_ascii=False)[1:-1]):text=text.replace(v,"")
    return text
def synth_raw(variant="sealed"):
    """CPU-only synthetic raw scores. 'sealed' reproduces the TEST524 log's selections and ranks exactly."""
    raw=dict(queries=[])
    for sq in SEALED524["queries"]:
        g=sq["record"]-1;s=[0.5-0.001*j for j in range(N)]
        if sq["selected"]==sq["record"]and variant!="all_miss":s[g]=0.95
        else:
            sel=(sq["selected"]-1)if variant=="sealed"else(g+1)%N;others=[sel]+[j for j in range(N)if j not in(g,sel)][:sq["rank"]-2 if variant=="sealed"else 0]
            for t,j in enumerate(others):s[j]=0.9-0.001*t
            s[g]=0.55
        w=[0.0]+[0.01]*29;w[sq["pos"]]=0.7;tot=sum(w);w=[x/tot for x in w]
        raw["queries"].append(dict(k=sq["k"],record=sq["record"],entity=sq["entity"],question=qA(sq["entity"]),scores=s,w=w,secs=0.05))
    return raw
def cpu_selftest():
    n=0
    assert LOCK_SHA==SEALED524_LOCK,"TEST524 LOCK SHA differs from the sealed value";n+=1
    nm=build_names(N*3);assert len(nm)==N*3 and len(set(nm))==N*3;n+=1
    for sq in SEALED524["queries"]:assert nm[3*(sq["record"]-1)]==sq["entity"],(sq["entity"],nm[3*(sq["record"]-1)]);n+=1
    assert [i+1 for i in SHOWCASE_IDX]==[q["record"]for q in SEALED524["queries"]];n+=1
    assert qA("Hexhyn")=="What seal does instrument Hexhyn carry? Give only the exact seal.";n+=1
    R=derive(synth_raw("sealed"));assert R["correct"]==4 and R["misses"]==[2] and R["replay"]["match"];n+=3
    assert [q["rank"]for q in R["queries"]]==[1,19,1,1,1] and [q["selected"]for q in R["queries"]]==[7,10,64,91,123];n+=2
    assert [q["pos"]for q in R["queries"]]==[q["pos"]for q in SEALED524["queries"]];n+=1
    R=derive(synth_raw("other"));assert R["correct"]==4 and not R["replay"]["match"];n+=2
    R=derive(synth_raw("all_miss"));assert R["correct"]==0 and all(q["rank"]==2 for q in R["queries"]);n+=2
    bad=synth_raw();bad["queries"][0]["question"]="Which memory is #007?"
    try:derive(bad);raise AssertionError("modified question accepted")
    except AuditFail:n+=1
    lo,hi=cp_interval(4,5);assert 0.28<lo<0.29 and 0.99<hi<1.0;n+=1
    assert stale_scan("t",f"{MODEL_SHORT} 28 layers TEST523 TEST524 L23H12 L02-V 128-MEMORY EXPERIMENTAL FIELD #032 {CANON_GPU} 512 numbers")==[];n+=1
    for b in("Mistral-7B","32 layers","hidden 4096","8 KV heads","K120","compressed memory","unlimited memory","O(1)","neuron memory","zenodo","TEST482"):assert stale_scan("t",b),b;n+=1
    assert strip_strings('{"seal":"MISTRAL"}',["MISTRAL"])=='{"seal":""}';n+=1
    return n
#<<CONST_END>>
#<<ENGINE_BEGIN>>
for _m,_p in[("torch","torch"),("transformers","transformers"),("accelerate","accelerate"),("gradio","gradio"),("matplotlib","matplotlib"),("PIL","pillow")]:
    if importlib.util.find_spec(_m)is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",_p])
import numpy as np,torch,transformers,gradio as gr,matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch,FancyArrowPatch,Circle
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from PIL import Image
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
DEVICE=torch.device("cuda")
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required (Runtime → Change runtime type → GPU).")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
def _cell_source():
    try:
        s_=get_ipython().user_ns.get("_ih",[""])[-1]
        return s_ if isinstance(s_,str)and"AKBASCORE MAM"in s_ else None
    except Exception:return None
CELL_SOURCE=_cell_source();CELL_SOURCE_SHA=hashlib.sha256(CELL_SOURCE.encode("utf-8")).hexdigest()if CELL_SOURCE else None
STARTUP_UTC=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
ROOT=Path("/content/AKBASCORE_MAM_TEST524")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_MAM_TEST524")
ROOT.mkdir(parents=True,exist_ok=True);ALLOWED_PATHS=[str(ROOT)]
def utc_now():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def local_now():return datetime.now().astimezone().isoformat(timespec="milliseconds")
QUIET=False
def say(*a):
    if not QUIET:print(*a,flush=True)
say("="*140);say(f"{PRODUCT} · {PRODUCT_LONG} · WORLD LAUNCH DEMO");say(f"{AUTHOR} · {AUTHOR_PLACE} · {LAUNCH_DATE} · {COPYRIGHT}");say("="*140)
say(f"[0/7] CPU self-test PASS ({cpu_selftest()} checks) · TEST524 LOCK SHA recomputed = sealed value");say("LOCK SHA  :",LOCK_SHA);say("START UTC :",STARTUP_UTC)
say("Model :",MODEL_ID);say("GPU :",torch.cuda.get_device_name(0),"| canonical sealed hardware:",CANON_GPU)
say("Gradio :",gr.__version__,"| Transformers:",transformers.__version__,"| Torch:",torch.__version__)
say("[1/7] MODEL LOAD")
torch.cuda.synchronize();_t=time.perf_counter()
# ---------------- TEST524 engine (verbatim) ----------------
_tv=tuple(int("".join(c for c in x if c.isdigit())or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
if not tok.is_fast:raise RuntimeError("Fast tokenizer required.")
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
enc=lambda s:tok(s,add_special_tokens=False).input_ids
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=H//QH;REP=QH//KVH
if (NL,H,QH,KVH,HD)!=(28,3584,28,4,128):raise RuntimeError("Architecture mismatch.")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id;VOC=model.model.embed_tokens.weight.shape[0]
torch.cuda.synchronize();MODEL_LOAD_S=time.perf_counter()-_t
def model_guard():
    # Every parameter tensor participates. This is an integrity guard,
    # not a byte-for-byte cryptographic hash of all 15GB of weights.
    h=hashlib.sha256()
    with torch.inference_mode():
        for name,p in model.named_parameters():
            x=p.detach().float()
            vals=(float(x.sum().item()),float(x.abs().sum().item()),float((x*x).sum().item()))
            h.update(name.encode());h.update(str(tuple(p.shape)).encode());h.update(str(p.dtype).encode())
            h.update(np.asarray(vals,dtype=np.float64).tobytes())
    return h.hexdigest()
def native_code_pool():
    out=[]
    for tid in range(VOC):
        s=tok.decode([tid],skip_special_tokens=False)
        if not re.fullmatch(r"[A-Z]{3,8}",s):continue
        if enc(s)!=[tid]:continue
        if s not in out:out.append(s)
    return out
def make_sources():
    src=[];audit=[]
    for i in range(N):
        ents=NAMES[3*i:3*i+3];aseals=SEALS[3*i:3*i+3];gold=aseals[0]
        j1=(i+23)%N;j2=(i+57)%N
        bseals=[gold,SEALS[3*j1+1],SEALS[3*j2+2]]
        bclasses=CLASSES[3*i:3*i+3]
        ra=[f"Instrument {ents[j]} carries seal {aseals[j]}." for j in range(3)]
        rb=[f"Seal {bseals[j]} corresponds to routing class {bclasses[j]}." for j in range(3)]
        r=random.Random(SEED+i*101);r.shuffle(ra);r.shuffle(rb)
        src.append((" ".join(ra)," ".join(rb)))
        audit.append({"record":i+1,"entity":ents[0],"seal":gold,"class":bclasses[0]})
    return src,audit
@torch.inference_mode()
def kv_from_ids(ids):
    o=model(input_ids=torch.tensor([ids],device=DEVICE),use_cache=False,output_hidden_states=True,return_dict=True);out=[]
    for L,layer in enumerate(model.model.layers):
        h=o.hidden_states[L][0].to(layer.input_layernorm.weight.device,layer.input_layernorm.weight.dtype)
        hn=layer.input_layernorm(h)
        k=layer.self_attn.k_proj(hn).view(-1,KVH,HD).transpose(0,1).contiguous()
        v=layer.self_attn.v_proj(hn).view(-1,KVH,HD).transpose(0,1).contiguous()
        out.append((k,v))
    del o
    return out
def forge(s):return kv_from_ids([PAD]+enc(s+SEP))
def rope_k(k,pos,L):
    q=torch.zeros((1,QH,len(pos),HD),device=DEVICE,dtype=k.dtype);kk=k.unsqueeze(0)
    layer=model.model.layers[L]
    rot=layer.self_attn.rotary_emb if hasattr(layer.self_attn,"rotary_emb") else model.model.rotary_emb
    p=torch.tensor([pos],device=DEVICE,dtype=torch.long)
    try:c,s=rot(q,p)
    except TypeError:c,s=rot(q,position_ids=p)
    _,kr=apply_rotary_pos_emb(q,kk,c,s)
    return kr[0]
def install(raw):
    T=raw[0][0].shape[1]
    return [(rope_k(k,list(range(T)),L),v) for L,(k,v) in enumerate(raw)],T,T
def cache_of(inst):
    try:c=DynamicCache(config=cfg)
    except TypeError:c=DynamicCache()
    for L,(k,v) in enumerate(inst):c.update(k.unsqueeze(0),v.unsqueeze(0),L)
    return c
def cos_rows(v,M):
    v=v.float();M=M.float();v=v/v.norm().clamp_min(1e-8)
    M=M/M.norm(dim=1,keepdim=True).clamp_min(1e-8)
    return M@v
@torch.inference_mode()
def query_forward(packet,q):
    # IDENTICAL TEST523 QUERY PATH.
    inst,Tm,P=packet;c=cache_of(inst);ids=enc(FMT.format(q=q));n=len(ids)
    pos=torch.arange(P,P+n,device=DEVICE).unsqueeze(0)
    mask=torch.ones((1,Tm+n),device=DEVICE,dtype=torch.long)
    o=model(input_ids=torch.tensor([ids],device=DEVICE),past_key_values=c,
            attention_mask=mask,position_ids=pos,use_cache=False,
            output_hidden_states=True,return_dict=True)
    return o,Tm,P,n
@torch.inference_mode()
def frozen_pointer(packet,q):
    # EXACT TEST523 POINTER MATHEMATICS.
    inst,Tm,P=packet;o,_,_,n=query_forward(packet,q);qpos=P+n-1
    L=PTR_L;layer=model.model.layers[L]
    h=o.hidden_states[L][0,-1].to(layer.input_layernorm.weight.dtype)
    hn=layer.input_layernorm(h)
    qv=layer.self_attn.q_proj(hn).view(1,QH,1,HD)
    ak=inst[L][0].repeat_interleave(REP,dim=0).unsqueeze(0)
    dummy=torch.zeros_like(ak[:,:,:1,:])
    rot=layer.self_attn.rotary_emb if hasattr(layer.self_attn,"rotary_emb") else model.model.rotary_emb
    pp=torch.tensor([[qpos]],device=DEVICE,dtype=torch.long)
    try:c,s=rot(dummy,pp)
    except TypeError:c,s=rot(dummy,position_ids=pp)
    qrot,_=apply_rotary_pos_emb(qv,dummy,c,s)
    score=torch.einsum("bhqd,bhkd->bhqk",qrot.float(),ak.float()).squeeze(0).squeeze(1)/math.sqrt(HD)
    score[:,0]=-torch.inf
    W=torch.softmax(score,dim=-1).detach().cpu();W[:,0]=0
    W=W/W.sum(dim=1,keepdim=True).clamp_min(1e-12)
    del o
    return W[PTR_H]
def pointer_vector(raw,w):
    x=raw[BL][1].float().cpu()
    if x.shape[1]!=len(w):raise RuntimeError(f"Pointer length {len(w)} != tensor length {x.shape[1]}")
    return (x*w[None,:,None]).sum(dim=1).reshape(-1)
QUERY_TIME_B_FORWARDS=0
def B_scores(p):
    # Pure numeric comparison. No candidate B is passed through Qwen here.
    global QUERY_TIME_B_FORWARDS
    return [float(cos_rows(p,M).max()) for M in B_MATS]
def retrieve(active_record,question):
    if not 0<=active_record<N:raise ValueError("active_record out of range")
    w=frozen_pointer(A_PACK[active_record],question)
    p=pointer_vector(A_MEM[active_record],w)
    scores=B_scores(p)
    order=sorted(range(N),key=lambda j:(-scores[j],j))
    return w,p,scores,order
# ---------------- end of TEST524 engine ----------------
A_MEM=[];B_MEM=[];B_MATS=[];A_PACK=[]
WIPED_NAMES=("SOURCES","SEALS","CLASSES","NAMES","POOL","rr")
# ---------------- demo-side read-only measurements (never alter memories or weights) ----------------
def hook_module(fn):
    f=getattr(fn,"func",fn);return str(getattr(f,"__module__",None)or type(f).__module__ or"")
def hook_stats():
    tot=0;foreign=Counter()
    for m in model.modules():
        for d in(m._forward_hooks,m._forward_pre_hooks,m._backward_hooks):
            for fn in d.values():
                tot+=1;mod=hook_module(fn)
                if not mod.startswith(("transformers","accelerate","torch")):foreign[mod]+=1
    return tot,dict(foreign)
def lora_present():return hasattr(model,"peft_config")or any("lora"in n.lower()for n,_ in model.named_modules())
def trainable_tensors():return sum(int(p.requires_grad)for p in model.parameters())
def optimizer_present():return any(isinstance(v,torch.optim.Optimizer)for v in list(globals().values()))
def _mem_numbers(raw):return sum(k.numel()+v.numel()for k,v in raw)
class Engine:
    def __init__(self):
        self.counters=Counter()
        g0=model_guard()
        self.info=dict(model_id=MODEL_ID,arch=(NL,H,QH,KVH,HD),dtype="bfloat16",attn=getattr(cfg,"_attn_implementation",None),gpu=torch.cuda.get_device_name(0),canonical_gpu=CANON_GPU,
            gpu_total_gib=torch.cuda.get_device_properties(0).total_memory/2**30,torch=torch.__version__,transformers=transformers.__version__,gradio=gr.__version__,python=sys.version.split()[0],
            platform=platform.platform(),params=sum(p.numel()for p in model.parameters()),pad_id=PAD,vocab=VOC,model_load_seconds=MODEL_LOAD_S,startup_utc=STARTUP_UTC,
            guard0=g0,hooks0=hook_stats()[0],cell_source_sha256=CELL_SOURCE_SHA,
            guard_method="TEST524 all-parameter guard: SHA-256 over (name, shape, dtype, sum, |sum|, sum of squares) of every parameter tensor")
        if model_guard()!=g0:raise RuntimeError("All-parameter weight guard is not repeatable")
    def frozen_state(self):
        tot,fo=hook_stats()
        return dict(training=bool(model.training),trainable_tensors=trainable_tensors(),requires_grad_disabled=all(not p.requires_grad for p in model.parameters()),
                    lora=lora_present(),optimizer=optimizer_present(),hooks=tot,foreign_hooks=sum(fo.values()),foreign_modules=fo,guard=model_guard())
    def prepare_panel(self):
        """TEST524 panel construction, verbatim order: names → native code pool → seeded shuffle → seals/classes → sources/audit → showcase."""
        global NAMES,POOL,SEALS,CLASSES,rr,SOURCES,AUDIT,SHOWCASE,A_MEM,B_MEM,B_MATS,A_PACK
        A_MEM=[];B_MEM=[];B_MATS=[];A_PACK=[];gc.collect();torch.cuda.empty_cache();t0=time.perf_counter()
        NAMES=build_names(N*3);POOL=native_code_pool()
        if len(POOL)<N*6:raise RuntimeError(f"Need {N*6} distinct native codes, found {len(POOL)}.")
        pool_n=len(POOL);rr=random.Random(SEED+100000);rr.shuffle(POOL)
        SEALS=POOL[:N*3];CLASSES=POOL[N*3:N*6]
        assert len(set(SEALS))==N*3 and len(set(CLASSES))==N*3 and set(SEALS).isdisjoint(CLASSES)
        SOURCES,AUDIT=make_sources()
        SHOWCASE=[{"record":i+1,"entity":AUDIT[i]["entity"],"question":f"What seal does instrument {AUDIT[i]['entity']} carry? Give only the exact seal."} for i in SHOWCASE_IDX]
        return dict(pool=pool_n,records=len(SOURCES),showcase=[dict(x)for x in SHOWCASE],audit=[dict(x)for x in AUDIT],seconds=time.perf_counter()-t0,
                    audit_sha256=hashlib.sha256(canon(AUDIT)).hexdigest())
    def source_sentence_lengths(self):return[(len(a),len(b))for a,b in SOURCES]
    def forge_one(self,i):
        torch.cuda.synchronize();t0=time.perf_counter();ar=forge(SOURCES[i][0]);br=forge(SOURCES[i][1]);torch.cuda.synchronize();dt_=time.perf_counter()-t0
        A_MEM.append(ar);B_MEM.append(br);self.counters["forge_passes"]+=2
        return dict(TA=int(ar[0][0].shape[1]),TB=int(br[0][0].shape[1]),numbers_A=_mem_numbers(ar),numbers_B=_mem_numbers(br),secs=dt_,
                    finite=bool(all(torch.isfinite(t).all()for kv in ar+br for t in kv)))
    def build_address_field(self):
        """TEST524 step 3, verbatim: persistent L02-V address matrices for every B memory; installed (RoPE'd) A packets."""
        global B_MATS,A_PACK
        t0=time.perf_counter();B_MATS=[]
        for raw in B_MEM:
            x=raw[BL][1][:,1:,:].float().cpu()
            B_MATS.append(x.permute(1,0,2).reshape(x.shape[1],-1).contiguous())
        A_PACK=[install(x) for x in A_MEM];torch.cuda.synchronize()
        return dict(b_mats=len(B_MATS),rows=[int(M.shape[0])for M in B_MATS],dim=int(B_MATS[0].shape[1]),a_packs=len(A_PACK),seconds=time.perf_counter()-t0,
                    storages=len({id(M)for M in B_MATS}))
    def wipe(self):
        """TEST524 step 4 (SIL BASTAN): source strings and panel construction containers are deleted before any live question."""
        global SOURCES,SEALS,CLASSES,NAMES,POOL,rr
        n=len(SOURCES)
        del SOURCES,SEALS,CLASSES,NAMES,POOL,rr
        gc.collect();torch.cuda.empty_cache()
        present={k:(k in globals())for k in WIPED_NAMES}
        return dict(source_records_before=n,names=list(WIPED_NAMES),present=present,source_text_present=bool(any(present.values())))
    def retrieve_live(self,k,record,question):
        torch.cuda.synchronize();t=time.perf_counter();w,p,scores,order=retrieve(record-1,question);torch.cuda.synchronize();dt_=time.perf_counter()-t
        self.counters["query_retrievals"]+=1
        return dict(k=k,record=record,question=question,w=[float(x)for x in w.tolist()],p=[round(float(x),6)for x in p.tolist()],scores=[float(x)for x in scores],
                    order=[int(j)+1 for j in order[:10]],secs=dt_,b_forwards_counter=int(QUERY_TIME_B_FORWARDS))
    def tensor_excerpt(self,record):
        """Real values read from this run's tensors: layer-2 V of active A memory and the matching B memory's L02-V address rows."""
        i=record-1;v=A_MEM[i][BL][1];M=B_MATS[i];k=A_MEM[i][BL][0]
        return dict(record=record,layer=BL,head=0,a_shape=list(v.shape),b_shape=list(M.shape),
                    a_v=[[round(float(x),3)for x in row]for row in v[0,1:9,:8].float().cpu().tolist()],
                    a_k=[[round(float(x),3)for x in row]for row in k[0,1:5,:8].float().cpu().tolist()],
                    b_rows=[[round(float(x),3)for x in row]for row in M[:8,:8].tolist()],
                    a_heat=[[round(float(x),4)for x in row]for row in v[0,1:,:64].float().cpu().tolist()],
                    numbers_A=_mem_numbers(A_MEM[i]),numbers_B=_mem_numbers(B_MEM[i]))
    def counts(self):return dict(A=len(A_MEM),B=len(B_MEM),B_MATS=len(B_MATS),A_PACK=len(A_PACK))
    def sync(self):torch.cuda.synchronize()
    def reset_peak(self):torch.cuda.reset_peak_memory_stats()
    def peak_gib(self):return torch.cuda.max_memory_allocated()/2**30
ENGINE=Engine()
say(f"Model loaded in {MODEL_LOAD_S:.2f}s | layers={NL} hidden={H} Q={QH} KV={KVH} head={HD} | params={ENGINE.info['params']:,} trainable={trainable_tensors()}")
say("[2/7] ALL-PARAMETER WEIGHT GUARD");say("Guard:",ENGINE.info["guard0"]);say("Sealed TEST524 guard (reference):",SEALED524["guard"])
say("[3/7] ENGINE LOCK")
ENGINE_LOCK=[("MODEL_ID = Qwen/Qwen2.5-7B-Instruct",MODEL_ID=="Qwen/Qwen2.5-7B-Instruct"),("architecture 28 layers / hidden 3584 / 28 Q / 4 KV / head 128",(NL,H,QH,KVH,HD)==ARCH),
 ("ÇAĞRIİZ pointer L23H12 · address L02-V UNCENTERED · match max cosine",(PTR_L,PTR_H,BL,BK)==(23,12,2,"V")),("TEST524 LOCK SHA = sealed value",LOCK_SHA==SEALED524_LOCK),
 ("128 records · showcase records 7/38/64/91/123",N==128 and [i+1 for i in SHOWCASE_IDX]==[7,38,64,91,123]),("fast tokenizer",bool(tok.is_fast)),
 ("BF16 weights",next(model.parameters()).dtype==torch.bfloat16),("SDPA attention",getattr(cfg,"_attn_implementation",None)=="sdpa"),
 ("no forward/backward hooks from this cell (foreign hooks = 0)",not hook_stats()[1]),("frozen eval model",(not model.training)and trainable_tensors()==0 and not lora_present()),
 ("no optimizer",not optimizer_present()),("all-parameter weight guard repeatable",model_guard()==ENGINE.info["guard0"])]
_bad=[n for n,ok in ENGINE_LOCK if not ok]
if _bad:raise RuntimeError(f"ENGINE LOCK FAILED: {_bad}")
say("Engine lock: PASS ·",len(ENGINE_LOCK),"checks")
#<<ENGINE_END>>
#<<CORE_BEGIN>>
N_STAGES=10;TOTAL_STEPS=N+5
def ev(stage,title,body,done=None,total=None,eta=None):return dict(stage=stage,title=title,body=body,done=done,total=total,eta=eta)
def file_ready(p):return p is not None and Path(p).is_file()and Path(p).stat().st_size>0
def prune_runs(keep=2):
    runs=sorted([p for p in ROOT.glob("MAM524-*")if p.is_dir()],key=lambda p:p.stat().st_mtime)
    for p in runs[:max(0,len(runs)-keep)]:shutil.rmtree(p,ignore_errors=True)
def post_seal_audit(imgs,zp,member_names,pp,sha,tp,mp,verdict):
    checks=[]
    def ok(name,cond):
        checks.append(name)
        if not cond:raise AuditFail("POST-SEAL AUDIT FAILED: "+name)
    ok(f"{N_POSTERS}/{N_POSTERS} posters created",len(imgs)==N_POSTERS and all(file_ready(p)for p,_ in imgs))
    bad=[]
    for p,_ in imgs:
        with Image.open(p)as im:
            if not(im.format=="JPEG"and im.mode=="RGB"):bad.append(p.name)
    ok(f"{N_POSTERS}/{N_POSTERS} posters are JPEG/RGB",not bad)
    with zipfile.ZipFile(zp)as z:
        ok("ZIP testzip()",z.testzip()is None);nm=z.namelist()
        ok("ZIP is flat (no sub-folders)",all("/"not in n for n in nm));ok("ZIP contents = expected package",sorted(nm)==sorted(member_names))
        jp=sorted(n for n in nm if n.lower().endswith(".jpg"));ok("poster numbering 01..%02d"%N_POSTERS,[n[:2]for n in jp]==[f"{i:02d}"for i in range(1,N_POSTERS+1)])
    ok("payload SHA-256 recomputation",hashlib.sha256(pp.read_bytes()).hexdigest()==sha);ok("payload JSON parses",isinstance(json.loads(pp.read_bytes().decode("utf-8")),dict))
    m=json.loads(mp.read_text(encoding="utf-8"));ok("manifest hash and verdict match payload",m["payload_sha256"]==sha and m["verdict"]==verdict)
    ok("readable log exists and is non-empty",file_ready(tp));ok("ZIP exists and is non-empty",file_ready(zp))
    return checks
def make_txt(P,R,sha,names,sealed_utc):
    o=[];a=o.append;S="="*140;Dd="-"*140;E_=P["environment"];fz=P["frozen"];eng=P["engine"];pan=P["panel"];wp=P["wipe"]
    a(S);a(f"{PRODUCT} · {PRODUCT_LONG} — WORLD LAUNCH DEMO · READABLE RUN LOG (TEST524 ENGINE)");a(f"{AUTHOR} · {AUTHOR_PLACE} · {LAUNCH_DATE} · {COPYRIGHT}");a(S)
    a(f"Derived from {names['payload']} (SHA-256 {sha}). Manifest: {names['manifest']}.");a("The SHA-256 is an artifact integrity seal, not a scientific proof and not a third-party verification.")
    for k_,v in(("RUN ID",P["run_id"]),("START UTC",P["run_start_utc"]),("END UTC",P["run_end_utc"]),("SEALED UTC",sealed_utc),("MODEL",MODEL_ID+" · frozen"),
        ("ARCHITECTURE","28 layers / hidden 3584 / 28 Q heads / 4 KV heads / head 128"),("DTYPE / ATTENTION",f"{P['model']['dtype']} / {P['model']['attn']}"),
        ("GPU (this run)",E_["gpu"]),("CANONICAL SEALED GPU",CANON_GPU),("SAME HARDWARE AS SEALED RUN",str(P["hardware"]["same"])),("TORCH / TRANSFORMERS",f"{E_['torch']} / {E_['transformers']}"),
        ("SEED",SEED),("ÇAĞRIİZ POINTER",eng["pointer"]),("ADDRESS",eng["address"]),("MATCH",eng["match"]),("TEST524 LOCK SHA (recomputed)",eng["lock_sha_recomputed"]),
        ("TEST524 LOCK SHA (sealed)",eng["lock_sha_sealed"]),("PARENT TEST523 LOCK",eng["parent523"]),("VERDICT (live)",P["verdict"])):a(f"{k_:<32}: {v}")
    for title,key in(("WHAT THIS RUN DEMONSTRATES","demonstrated"),("NOT CLAIMED BY THIS RUN","not_claimed"),("RESEARCHER NOTES","researcher_notes")):
        a("");a(title);a(Dd)
        for t in P["scope"][key]:a("• "+t)
    a("");a("TEST BANK");a(Dd)
    a(f"records={pan['records']} · independent numeric A/B memories={pan['memories']} · native single-token codes found={pan['native_code_pool']} · panel audit SHA-256={pan['audit_sha256']}")
    a(f"A memory slots {min(pan['a_slots'])}–{max(pan['a_slots'])} · B memory slots {min(pan['b_slots'])}–{max(pan['b_slots'])} · B address rows {min(P['address_field']['rows'])}–{max(P['address_field']['rows'])} × {P['address_field']['dim']} numbers")
    a("128 is the experimental bank size, not a capacity limit.")
    a("");a("SHOWCASE AUDIT METADATA (used only after retrieval to verify it; never given to the retriever)");a(Dd)
    for x in pan["showcase_audit"]:a(f"record #{x['record']:03d} | entity {x['entity']} | seal {x['seal']} | class {x['class']} | question: {x['question']}")
    a("");a("LIVE RETRIEVAL (this run, re-derived from the 128 raw scores of every question)");a(Dd)
    for q in R["queries"]:
        a(f"Q{q['k']} active A#{q['record']:03d} · selected B#{q['selected']:03d} · gold rank {q['rank']}/{N} · {'PASS'if q['correct']else'MISS'} · top1 {q['top1']:+.6f} · margin {q['margin']:+.6f} · "
          f"ÇAĞRIİZ peak {q['peak']:.4f} @ position {q['pos']} of {q['T']} · entropy {q['entropy']:.4f} · {1000*(q['secs']or 0):.2f} ms")
        a("     top10: "+" ".join(f"#{j:03d}:{s:+.6f}"for j,s in q["top10"]))
    a(f"LIVE SHOWCASE (TEST524, 5 fixed questions): {ci_text(R['correct'],R['total'])}")
    rp=R["replay"];a(f"REPLAY vs SEALED TEST524 LOG: selections identical={rp['selected_match']} · ranks identical={rp['rank_match']} · max |top1 difference|={rp['top1_max_abs_diff']:.6f}")
    a("");a("SEALED TEST523 EXTERNAL REPLICATION (historical record, NOT produced by this run)");a(Dd)
    a(f"Top-1 {fr(SEALED523['r1'])} = {SEALED523['r1_pct']} · counterfactual follow {fr(SEALED523['cf'])} = {SEALED523['cf_pct']} · {SEALED523['verdict']} · lock {SEALED523['lock']}")
    a("");a("SOURCE REMOVED FROM THE LIVE RETRIEVAL PATH");a(Dd)
    a(f"source records before removal: {wp['source_records_before']} · containers deleted: {', '.join(wp['names'])} · any still present: {wp['source_text_present']}")
    a("");a("FROZEN MODEL");a(Dd)
    for k_ in("guard_startup","guard_pre_run","guard_after_forge","guard_after","guard_sealed_reference"):a(f"{k_:<24}: {fz[k_]}")
    a(f"trainable tensors={fz['trainable_tensors']} lora={fz['lora']} optimizer={fz['optimizer']} training_mode={fz['training_mode']} hooks {fz['hooks_before']}→{fz['hooks_after']} (foreign: {fz['foreign_hooks_before']}→{fz['foreign_hooks_after']})")
    a(fz["guard_method"])
    a("");a("WHAT THE LIVE RETRIEVAL PATH USES (verification class in brackets)");a(Dd)
    for e in P["protocol"]:a(f"{e['key']:<44}: {e['value']}  [{e['cls']}]  {e['note']}")
    a("");a("PRE-SEAL CHECKS");a(Dd)
    for c_ in P["checks_pre_seal"]:a("PASS · "+c_)
    a("");a("TIMING (seconds)");a(Dd)
    for k_,v in P["timing"].items():a(f"{k_:<24}: {v:.3f}")
    c=P["counters"];a(f"forge passes: {c['forge_passes']} · live retrievals: {c['query_retrievals']} · candidate-B LLM forwards: {c['candidate_B_llm_forwards']} · peak GPU memory: {P['gpu']['peak_allocated_gib']:.2f} GiB")
    a("");a("All 128 raw scores, ÇAĞRIİZ weights and 512-number addresses of every question are in "+names["payload"]+" and "+names["jsonl"]+".");a(S)
    return "\n".join(o)
def execute_run(E,ctl):
    """Fail-closed wrapper: on a technical audit failure the raw outputs gathered so far are preserved (never sealed, no posters)."""
    state={}
    try:return(yield from _execute_run(E,ctl,state))
    except AuditFail as ex:
        raw=state.get("raw")
        if raw and any(raw.values()):
            try:
                p=Path(ctl["run_dir"])/f"FAILED_AUDIT_RAW_{ctl['run_id']}.json"
                p.write_bytes(canon(dict(status="FAILED AUDIT — NOT SEALED",run_id=ctl["run_id"],failed_check=str(ex),checks_passed=state.get("checks",[]),raw=raw,note="Raw outputs preserved so the run is not lost. No posters, no seal and no verdict were produced.")));ex.partial=str(p)
            except Exception:pass
        raise
# ===== END PART 1 / 3 — CONTINUE WITH PART 2 =====
FEATURE_RECORD=SHOWCASE_IDX[2]+1   # fixed before the run: the middle showcase question (record #064) is the worked example on posters 03/04/06/07/08
ADDR_DIM=ARCH[3]*ARCH[4]
WIPE_LABELS={"SOURCES":"source records (256 texts)","SEALS":"seal list","CLASSES":"routing-class list","NAMES":"instrument names","POOL":"code pool","rr":"shuffle state"}
def _execute_run(E,ctl,state):
    """Generator. Yields progress events, returns the result bundle. Any failed technical check raises AuditFail (no package is sealed)."""
    run_id=ctl["run_id"];run_dir=Path(ctl["run_dir"]);I=E.info;checks=[];tm=Counter();T0=time.perf_counter();state["checks"]=checks;c0=Counter(E.counters)
    def chk(name,cond):
        checks.append(name)
        if not cond:raise AuditFail(name)
    run_start_utc=utc_now();run_start_local=local_now();say("="*140);say(f"RUN {run_id}");say("="*140)
    # ---- 1/10 integrity ----
    say("[1/10] INTEGRITY · TEST524 LOCK · FROZEN-MODEL PRE-CHECK");yield ev(1,"Integrity check","TEST524 lock, model, architecture, frozen weights and the all-parameter weight guard.")
    fs0=E.frozen_state();same_hw=I["gpu"]==CANON_GPU
    chk("TEST524 LOCK SHA recomputed = sealed value",LOCK_SHA==SEALED524_LOCK);chk("model = Qwen/Qwen2.5-7B-Instruct",I["model_id"]==MODEL_ID);chk("architecture 28L / H3584 / 28Q / 4KV / HD128",tuple(I["arch"])==ARCH)
    chk("dtype bfloat16",I["dtype"]=="bfloat16");chk("attention SDPA",I["attn"]=="sdpa");chk("ÇAĞRIİZ L23H12 · address L02-V · match max cosine",(PTR_L,PTR_H,BL,BK)==(23,12,2,"V"))
    chk("model in eval mode",not fs0["training"]);chk("trainable parameter tensors = 0",fs0["trainable_tensors"]==0 and fs0["requires_grad_disabled"]);chk("no LoRA / PEFT adapter",not fs0["lora"]);chk("no optimizer object",not fs0["optimizer"])
    chk("pre-run all-parameter guard = startup guard",fs0["guard"]==I["guard0"])
    if fs0["foreign_hooks"]:raise AuditFail("foreign forward/backward hooks present before run: "+str(fs0["foreign_modules"]))
    chk("no foreign forward/backward hooks before run",fs0["foreign_hooks"]==0)
    for c in checks:say("   PASS ·",c)
    say(f"   hardware: {I['gpu']} | canonical sealed hardware: {CANON_GPU} | same: {same_hw}"+("" if same_hw else"  (different GPU: scores may differ slightly from the sealed log; any difference is reported as measured)"))
    # ---- 2/10 test bank ----
    say("[2/10] TEST BANK · 128 RECORDS (TEST524 panel, rebuilt deterministically)");yield ev(2,"Building the test bank","128 records, rebuilt exactly as in TEST524. 128 is the experimental bank size, not a capacity limit.")
    E.reset_peak();pan=E.prepare_panel();tm["panel"]=pan["seconds"];audit=pan["audit"];showcase=pan["showcase"]
    chk("test bank = 128 records",pan["records"]==N and len(audit)==N)
    chk("showcase = records 7/38/64/91/123 with the TEST524 question template",[s["record"]for s in showcase]==[i+1 for i in SHOWCASE_IDX]and all(s["question"]==qA(s["entity"])for s in showcase))
    chk("showcase instrument names identical to the sealed TEST524 log",[s["entity"]for s in showcase]==[q["entity"]for q in SEALED524["queries"]])
    strip=[x[k_]for x in audit for k_ in("entity","seal","class")];state["strip"]=strip
    say(f"   {pan['records']} records · native single-token code pool {pan['pool']} · {pan['seconds']:.1f}s")
    # ---- 3/10 forge ----
    say("[3/10] READ ONCE · 128 RECORDS → 256 INDEPENDENT NUMERIC A/B MEMORIES");tels=[];t0=time.perf_counter()
    for i in range(N):
        tel=E.forge_one(i);tels.append(tel)
        if i in SHOWCASE_IDX:say(f"   SHOWCASE RECORD #{i+1:03d} forged · A slots {tel['TA']} · B slots {tel['TB']}")
        elif(i+1)%32==0:say(f"   {i+1:03d}/{N} records forged")
        if i%4==3 or i==N-1:yield ev(3,f"Read once · record {i+1:03d}/{N}","Each record is read once by the frozen AI and kept as two independent numeric memories (A and B). No training occurs.",done=i+1,total=N)
    tm["forge"]=time.perf_counter()-t0;cnt=E.counts()
    chk("256 independent numeric memories (128 A + 128 B)",cnt["A"]==N and cnt["B"]==N);chk("all memory tensors finite",all(t["finite"]for t in tels))
    fs_m=E.frozen_state()
    if fs_m["foreign_hooks"]:raise AuditFail("foreign forward/backward hooks present after forging: "+str(fs_m["foreign_modules"]))
    chk("all-parameter guard unchanged after reading the 128 records (no training)",fs_m["guard"]==I["guard0"]);chk("no foreign hooks and no trainable tensors after forging",fs_m["foreign_hooks"]==0 and fs_m["trainable_tensors"]==0)
    # ---- 4/10 address field ----
    say("[4/10] NUMERIC ADDRESS FIELD · L02-V STRUCTURES FOR ALL 128 B MEMORIES");yield ev(4,"Building the numeric address field","Every B memory gets a numeric address structure taken from its own layer-2 V tensors. No labels, no text.",done=N,total=N)
    af=E.build_address_field();tm["address"]=af["seconds"]
    chk(f"128 B address structures from L02-V ({ADDR_DIM} numbers per row)",af["b_mats"]==N and af["dim"]==ADDR_DIM);chk("128 distinct address structures",af["storages"]==N);chk("128 active-A packets installed",af["a_packs"]==N)
    say(f"   {af['b_mats']} address structures · rows {min(af['rows'])}–{max(af['rows'])} × {af['dim']} · {af['seconds']:.1f}s")
    # ---- 5/10 source removed ----
    say("[5/10] SIL BASTAN · SOURCE REMOVED FROM THE LIVE RETRIEVAL PATH");yield ev(5,"Source removed from the live retrieval path","Source records and panel containers are deleted. From here the retriever receives only a question and numeric memories.")
    t0=time.perf_counter();wp=E.wipe();tm["wipe"]=time.perf_counter()-t0
    chk("source containers deleted before the first question",not wp["source_text_present"]and wp["source_records_before"]==N)
    codes=set(x["seal"]for x in audit)|set(x["class"]for x in audit)
    def leak(q):return any(re.search(r"(?<![A-Za-z])"+re.escape(c)+r"(?![A-Za-z])",q)for c in codes)or bool(re.search(r"\d",q))
    chk("questions contain no seal, no class and no memory number",not any(leak(s["question"])for s in showcase))
    for k_,v in wp["present"].items():say(f"   {k_:<8}: {'PRESENT'if v else'DELETED'}")
    # ---- 6/10 live retrieval ----
    say("[6/10] LIVE RETRIEVAL · 5 FIXED QUESTIONS · EVERY QUESTION SCORES ALL 128 B MEMORIES");raw=dict(queries=[]);state["raw"]=raw;t0=time.perf_counter()
    for k_,s in enumerate(showcase,1):
        yield ev(6,f"Live question {k_}/5",s["question"]+" — ÇAĞRIİZ → numeric fingerprint → 128-memory field.",done=k_-1,total=5)
        r=E.retrieve_live(k_,s["record"],s["question"]);r["entity"]=s["entity"];raw["queries"].append(r)
        sel=r["order"][0];ok_=sel==s["record"]
        say(f" Q{k_} active A#{s['record']:03d} · selected B#{sel:03d} · {'PASS'if ok_ else'MISS'} · top {max(r['scores']):+.6f} · {1000*r['secs']:.1f} ms")
    tm["retrieval"]=time.perf_counter()-t0;ex=E.tensor_excerpt(FEATURE_RECORD)
    # ---- 7/10 re-derivation ----
    say("[7/10] RESULT RE-DERIVATION FROM THE RAW SCORES");yield ev(7,"Re-deriving the results","Every selection and rank is recomputed from the 128 raw scores of each question.",done=5,total=5)
    R=derive(raw);chk("results re-derived from raw scores are stable",derive(raw)==R);chk("every question scored all 128 B memories",all(len(r["scores"])==N for r in raw["queries"]))
    chk("candidate-B LLM forward counter = 0",all(r["b_forwards_counter"]==0 for r in raw["queries"]))
    for q in R["queries"]:say(f"   Q{q['k']} → selected #{q['selected']:03d} | gold rank {q['rank']:3d}/{N} | {'PASS'if q['correct']else'MISS'}")
    say(f"   LIVE SHOWCASE {R['correct']}/{R['total']} · replay of sealed TEST524 selections/ranks identical: {R['replay']['match']}")
    # ---- 8/10 integrity after ----
    say("[8/10] MODEL INTEGRITY AFTER THE RUN");yield ev(8,"Model integrity","The all-parameter weight guard is recomputed after the run.")
    fs1=E.frozen_state();tm["engine"]=time.perf_counter()-T0
    chk("all-parameter guard after run = startup guard (weights unchanged)",fs1["guard"]==I["guard0"]);chk("model eval mode and trainable tensors = 0 after run",(not fs1["training"])and fs1["trainable_tensors"]==0 and fs1["requires_grad_disabled"])
    chk("no LoRA / optimizer after run",(not fs1["lora"])and(not fs1["optimizer"]))
    if fs1["foreign_hooks"]:raise AuditFail("foreign forward/backward hooks present after run: "+str(fs1["foreign_modules"]))
    chk("no foreign forward/backward hooks after run (demo installs none)",fs1["foreign_hooks"]==0)
    verdict=f"TEST524_LIVE_SHOWCASE_{R['correct']}_OF_{R['total']}_INTEGRITY_VERIFIED";say("VERDICT:",verdict)
    # ---- 9/10 seal ----
    say("[9/10] SEAL");yield ev(9,"Sealing the evidence","Payload, manifest, readable log and raw scores are written and hashed.")
    E.sync();peak=E.peak_gib();dc=E.counters-c0
    protocol=[
     dict(key="model weights",value="FROZEN",cls="CHECKED",note="all-parameter guard identical at startup, pre-run, after reading and after the run"),
     dict(key="training / optimizer / LoRA",value="NONE",cls="CHECKED",note="0 trainable tensors, no optimizer object, no adapter"),
     dict(key="source text in the live retrieval path",value="ABSENT",cls="CHECKED",note="source containers deleted before the first question"),
     dict(key="memory number, seal or class in the question",value="NO",cls="CHECKED",note="each question scanned against all 768 panel codes and for digits"),
     dict(key="gold B memory ID supplied to the retriever",value="NO",cls="BY CONSTRUCTION",note="retrieve() receives only the active A index and the question text"),
     dict(key="active A memory",value="GIVEN",cls="BY CONSTRUCTION",note="A is the active memory; first-A retrieval from an empty bank is not part of this experiment"),
     dict(key="decoded-text / token-ID / LM-head router",value="NONE",cls="BY CONSTRUCTION",note="B scoring uses only L02-V numeric matrices"),
     dict(key="learned router",value="NONE",cls="BY CONSTRUCTION",note="no fitted parameters anywhere in the retrieval path"),
     dict(key="candidate-B LLM forwards",value="0",cls="BY CONSTRUCTION",note="B_scores() makes no model call; engine counter = 0; not hook-instrumented"),
     dict(key="matching",value="MAX COSINE · 128 B MEMORIES",cls="CHECKED",note="128 raw scores recorded for every question"),
     dict(key="query cache",value="FRESH PER QUESTION",cls="BY CONSTRUCTION",note="query_forward() builds a new DynamicCache for every question"),
     dict(key="pointer / address / match",value="L23H12 · L02-V · MAX COS",cls="CHECKED",note="L02-V uncentered, max cosine; frozen by the TEST524 lock SHA"),
     dict(key="post-hoc selection",value="NONE",cls="CHECKED",note="showcase records fixed inside the TEST524 lock SHA")]
    show_aud=[dict(record=s["record"],entity=s["entity"],seal=audit[s["record"]-1]["seal"],**{"class":audit[s["record"]-1]["class"]},question=s["question"])for s in showcase]
    P={"schema":"akbascore.mam.test524.worldlaunch.run.v1","project":f"{PRODUCT} · {PRODUCT_LONG}","author":AUTHOR,"place":AUTHOR_PLACE,"launch_date":LAUNCH_DATE,"copyright":COPYRIGHT,
       "run_id":run_id,"run_start_utc":run_start_utc,"run_start_local":run_start_local,"run_end_utc":utc_now(),"verdict":verdict,
       "model":{"id":MODEL_ID,"arch":list(ARCH),"dtype":I["dtype"],"attn":I["attn"],"params":I["params"],"pad_id":I["pad_id"],"vocab":I["vocab"]},
       "environment":{k_:I[k_]for k_ in("gpu","gpu_total_gib","torch","transformers","gradio","python","platform")},
       "hardware":{"canonical_sealed_gpu":CANON_GPU,"this_run_gpu":I["gpu"],"same":bool(same_hw)},
       "engine":{"source":"TEST524 engine functions, unchanged (direct child of sealed TEST523)","pointer":"ÇAĞRIİZ = frozen L23H12 native Q·K + RoPE over the active A memory",
                 "address":f"ÇAĞRIİZ-weighted L02-V, UNCENTERED ({ADDR_DIM} numbers)","match":"max cosine over every row of all 128 B address structures","fmt":FMT,
                 "lock_sha_recomputed":LOCK_SHA,"lock_sha_sealed":SEALED524_LOCK,"parent523":PARENT523,"showcase_records":[i+1 for i in SHOWCASE_IDX],"forbidden":LOCK["forbidden"],
                 "compute_path":"PyTorch CUDA backend; no custom kernel; no hooks"},
       "panel":{"records":N,"memories":2*N,"native_code_pool":pan["pool"],"audit_sha256":pan["audit_sha256"],"a_slots":[t["TA"]for t in tels],"b_slots":[t["TB"]for t in tels],
                "showcase_audit":show_aud,"feature_record":FEATURE_RECORD,"capacity_note":"128 is the experimental bank size, not a capacity limit."},
       "bank":{"numbers_A":sum(t["numbers_A"]for t in tels),"numbers_B":sum(t["numbers_B"]for t in tels),"address_rows":sum(af["rows"]),"address_numbers":sum(af["rows"])*af["dim"]},
       "address_field":{"rows":af["rows"],"dim":af["dim"],"structures":af["b_mats"]},
       "wipe":wp,"raw":raw,"results":R,"tensor_excerpt":ex,
       "frozen":{"guard_startup":I["guard0"],"guard_pre_run":fs0["guard"],"guard_after_forge":fs_m["guard"],"guard_after":fs1["guard"],"guard_sealed_reference":SEALED524["guard"],
                 "guard_method":I["guard_method"],"trainable_tensors":fs1["trainable_tensors"],"lora":fs1["lora"],"optimizer":fs1["optimizer"],"training_mode":fs1["training"],
                 "hooks_before":I["hooks0"],"hooks_after":fs1["hooks"],"foreign_hooks_before":fs0["foreign_hooks"],"foreign_hooks_after":fs1["foreign_hooks"],
                 "hook_note":"hook totals include hooks installed by transformers/accelerate; only hooks from other code count as foreign"},
       "protocol":protocol,
       "counters":{"forge_passes":dc["forge_passes"],"query_retrievals":dc["query_retrievals"],"candidate_B_llm_forwards":max([r["b_forwards_counter"]for r in raw["queries"]]+[0])},
       "timing":{"model_load_seconds":I["model_load_seconds"],"test_bank":tm["panel"],"read_once_forge":tm["forge"],"address_field":tm["address"],"source_removal":tm["wipe"],
                 "live_retrieval":tm["retrieval"],"engine_total":tm["engine"]},
       "gpu":{"peak_allocated_gib":peak},"checks_pre_seal":list(checks),
       "sealed_record":{"note":"Historical logs. NOT produced by this run.","test524":SEALED524,"test523":SEALED523},
       "reproduction":{"seed":SEED,"cell_source_sha256":I.get("cell_source_sha256"),"note":"SHA-256 of the executed cell text; the text itself is not embedded because it contains the stale-reference deny-list."},
       "scope":SCOPE}
    P=jsafe(P);P["stale_scan"]={"patterns":len(STALE)+1,"hits":0,"scope":"payload JSON text, panel strings (names, seals, classes) excluded"}
    allow=(I["gpu"],CANON_GPU)
    chk("stale-reference scan of the payload: 0 hits",not stale_scan("payload",strip_strings(json.dumps(P,ensure_ascii=False),strip),allow=allow))
    pb=canon(P);sha=hashlib.sha256(pb).hexdigest();payload_name=f"{run_id}_FULL_RUN_LOG_payload.json";pp=run_dir/payload_name;t_seal=time.perf_counter();pp.write_bytes(pb);say("payload sealed:",sha)
    chk("payload SHA-256 verification after write",hashlib.sha256(pp.read_bytes()).hexdigest()==sha)
    names=dict(payload=payload_name,manifest=f"run_manifest_{run_id}.json",txt=f"{run_id}_READABLE_RUN_LOG.txt",summary=f"{run_id}_SUMMARY.json",jsonl=f"{run_id}_RESULTS.jsonl",images=f"images_manifest_{run_id}.json",zip=f"AKBASCORE_MAM_WORLD_LAUNCH_ALL_{run_id}.zip")
    mp=run_dir/names["manifest"];sealed_utc=utc_now()
    manifest={"demo":f"{PRODUCT} · {PRODUCT_LONG} — WORLD LAUNCH DEMO (TEST524 engine)","author":AUTHOR,"copyright":COPYRIGHT,"run_id":run_id,"payload_file":payload_name,"payload_sha256":sha,"payload_bytes":len(pb),
              "sealed_utc":sealed_utc,"verdict":verdict,"canonicalization":"json.dumps(sort_keys=True, separators=(',',':'), ensure_ascii=False, allow_nan=False), UTF-8",
              "note":"Artifact integrity seal: confirms the payload is unchanged since sealing. It is not a scientific proof and not an independent third-party verification."}
    mp.write_text(json.dumps(manifest,indent=2,sort_keys=True,ensure_ascii=False),encoding="utf-8")
    summary={"engine":"TEST524","model":MODEL_ID,"test_bank_records":N,"numeric_memories":2*N,"live_showcase":[R["correct"],R["total"]],"live_showcase_ci95":list(cp_interval(R["correct"],R["total"])),
             "queries":[{k_:q[k_]for k_ in("k","record","entity","selected","rank","correct","top1","margin","peak","pos")}for q in R["queries"]],
             "replay_of_sealed_test524":R["replay"],"sealed_test523_top1":list(SEALED523["r1"]),"sealed_test523_top1_pct":SEALED523["r1_pct"],
             "verdict":verdict,"guard_before":I["guard0"],"guard_after":fs1["guard"],"hardware":P["hardware"],"lock_sha":LOCK_SHA,"run_id":run_id,"payload_sha256":sha}
    (run_dir/names["summary"]).write_text(json.dumps(jsafe(summary),indent=2,ensure_ascii=False),encoding="utf-8")
    lines=[{"phase":"LIVE_RETRIEVAL","question_number":r["k"],"active_A_memory":r["record"],"question":r["question"],"selected_B_memory":R["queries"][j]["selected"],"gold_rank":R["queries"][j]["rank"],
            "cagriiz_weights":r["w"],"numeric_fingerprint":r["p"],"all_128_scores":r["scores"]}for j,r in enumerate(raw["queries"])]
    (run_dir/names["jsonl"]).write_text("\n".join(json.dumps(jsafe(x),ensure_ascii=False)for x in lines)+"\n",encoding="utf-8")
    (run_dir/names["txt"]).write_text(make_txt(P,R,sha,names,sealed_utc),encoding="utf-8");seal_s=time.perf_counter()-t_seal
    # ---- 10/10 posters + ZIP ----
    say("[10/10] POSTERS · IMAGE MANIFEST · ZIP");yield ev(10,"Rendering posters","Every number on every poster is checked against the re-derived runtime result before the package is sealed.")
    ctx={"P":P,"R":R,"run_id":run_id,"sha":sha,"payload_name":payload_name,"seal_seconds":seal_s,"allow":allow,"strip":strip,"N":N_POSTERS,"poster_audit":[]}
    t_r=time.perf_counter();imgs=render_all(ctx,run_dir);render_s=time.perf_counter()-t_r;entries=[]
    for n_,(p,cap)in enumerate(imgs,1):
        with Image.open(p)as im:w_,h_=im.size;fm_=im.format;md=im.mode
        entries.append({"index":n_,"filename":p.name,"title":cap,"width":w_,"height":h_,"format":fm_,"mode":md,"size_bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    imf=run_dir/names["images"]
    imf.write_text(json.dumps({"run_id":run_id,"payload_sha256":sha,"verdict":verdict,"count":len(entries),"render_seconds":render_s,"images":entries,"poster_text_audit":ctx["poster_audit"],
                               "note":"Per-image SHA-256 is an artifact integrity seal. Poster text was checked against re-derived results and scanned for stale references before saving."},indent=2,ensure_ascii=False),encoding="utf-8")
    zp=run_dir/names["zip"];members=[p for p,_ in imgs]+[imf,mp,pp,run_dir/names["txt"],run_dir/names["summary"],run_dir/names["jsonl"]]
    for m in members:
        h=stale_scan("filename",m.name)+(stale_scan(m.name,strip_strings(m.read_text(encoding="utf-8"),strip),allow=allow)if m.suffix in(".json",".txt",".jsonl")else[])
        chk(f"stale-reference scan of {m.name}: 0 hits",not h)
    with zipfile.ZipFile(zp,"w",compression=zipfile.ZIP_DEFLATED)as z:
        for m in members:z.write(m,arcname=m.name)
    post=post_seal_audit(imgs,zp,[m.name for m in members],pp,sha,run_dir/names["txt"],mp,verdict)
    say(f"{run_id}: {len(checks)} pre-seal + {len(post)} post-seal checks = {len(checks)+len(post)}/{len(checks)+len(post)} PASS | render {render_s:.1f}s")
    say("="*140);say("LIVE SHOWCASE (TEST524) :",f"{R['correct']}/{R['total']}");say("SEALED TEST523         :",f"{fr(SEALED523['r1'])} = {SEALED523['r1_pct']} top-1 (historical record)")
    say("VERDICT                :",verdict);say("PACKAGE                : SEALED ·",f"{len(checks)+len(post)} checks","· payload SHA-256",sha);say("ZIP                    :",zp);say("="*140)
    return dict(run_id=run_id,run_dir=run_dir,P=P,R=R,sha=sha,imgs=imgs,zip=zp,payload=pp,txt=run_dir/names["txt"],manifest=mp,images_manifest=imf,summary=run_dir/names["summary"],jsonl=run_dir/names["jsonl"],
                checks=len(checks)+len(post),verdict=verdict)
#<<CORE_END>>
#<<POSTERS_BEGIN>>
from matplotlib.text import Text
from matplotlib.patches import Arc,Polygon
plt.rcParams["font.family"]="DejaVu Sans";DPI=120
C_OK,C_ERR,C_CTL,C_CART,C_MOD,C_FG,C_NEU,C_BG,C_SEAL,C_VIO="#047857","#B91C1C","#475569","#B45309","#0369A1","#0F172A","#334155","#F8FAFC","#0F766E","#6D28D9"
C_OKL,C_ERRL,C_CARTL,C_MODL,C_SEALL,C_VIOL="#D1FAE5","#FEE2E2","#FEF3C7","#E0F2FE","#CCFBF1","#EDE9FE"
C_NIGHT,C_NIGHT2,C_CYAN,C_GLOW,C_AI="#0B1220","#16213A","#67E8F9","#FDE68A","#1E293B"
MONO="DejaVu Sans Mono";BRAND=f"{PRODUCT} · {PRODUCT_LONG.upper()}"
def mt(s):return str(s).replace("$",r"\$")
def new_fig():return plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
def save_jpg(fig,path):
    buf=io.BytesIO();fig.savefig(buf,format="png",dpi=fig.dpi,facecolor="white");plt.close(fig);buf.seek(0)
    with Image.open(buf)as im:
        im=im.convert("RGBA");bg=Image.new("RGB",im.size,(255,255,255));bg.paste(im,mask=im.getchannel("A"))
    bg.save(path,"JPEG",quality=92,optimize=True,progressive=False,subsampling=0)
    with Image.open(path)as chk_:
        if chk_.format!="JPEG"or chk_.mode!="RGB":raise RuntimeError("JPEG validation failed")
    return str(path)
def finish(fig,path,ctx,expect):
    """Poster text must contain every expected runtime value and no stale reference; otherwise the package is not sealed."""
    fig.canvas.draw();texts=[t.get_text()for t in fig.findobj(Text)if t.get_text().strip()];blob="\n".join(texts);nm=Path(path).name
    ws=lambda z:re.sub(r"\s+","",z);blob_ws=ws(blob);miss=[e for e in expect if ws(e)not in blob_ws]
    if miss:plt.close(fig);raise AuditFail(f"POSTER/RESULT MISMATCH in {nm}: missing {miss}")
    scan=strip_strings(re.sub(r"[ \t]*\n[ \t]*"," ",blob),ctx.get("strip",[]))
    hits=stale_scan(nm,scan,allow=ctx["allow"])+stale_scan("filename",nm)
    if hits:plt.close(fig);raise AuditFail(f"STALE REFERENCE in {nm}: {hits[:3]}")
    ctx["poster_audit"].append(dict(file=nm,texts=len(texts),expected_values=len(expect),stale_hits=0));return save_jpg(fig,path)
def fit_text(fig,x,y,w,h,text,fs_max=13,fs_min=7,color=C_FG,family=None,ls=1.32,weight="normal",ha="left"):
    fig.canvas.draw();r=fig.canvas.get_renderer();Wp=w*fig.get_figwidth()*fig.dpi;Hp=h*fig.get_figheight()*fig.dpi
    if Wp<=4 or Hp<=4:return 0
    paras=str(text if text else"(empty)").replace("\r","").split("\n");x0=x+w/2 if ha=="center"else x
    kw={"va":"top","ha":ha,"color":color,"linespacing":ls,"weight":weight,"multialignment":ha}
    if family:kw["family"]=family
    def wrap(c):
        out=[]
        for p in paras:out.extend(textwrap.wrap(p,c,break_long_words=True,break_on_hyphens=False)or[""])
        return out
    fs=float(fs_max);k=0.52
    for _ in range(150):
        cpl=max(6,int(Wp/(k*fs*fig.dpi/72.0)));ln=wrap(cpl);t=fig.text(x0,y+h,mt("\n".join(ln)),fontsize=fs,**kw);bb=t.get_window_extent(renderer=r)
        if bb.width>Wp*1.002 and cpl>6:t.remove();k*=max(1.01,bb.width/Wp*1.01);continue
        if bb.height<=Hp:return fs
        t.remove()
        if fs>fs_min:fs=max(float(fs_min),fs-0.5);continue
        per=bb.height/max(1,len(ln));keep=max(1,int(Hp/per)-1);fig.text(x0,y+h,mt("\n".join(ln[:keep]+["[… text shortened — full text in the run log]"])),fontsize=fs,**kw);return fs
    fig.text(x0,y+h,"[text omitted — see run log]",fontsize=fs_min,**kw);return fs_min
TAGC={"THIS":C_MOD,"SEALED":C_SEAL,"RESEARCHER":C_CTL,"RESEARCH":C_CART}
def head(fig,title,sub=None,tag=None):
    Hh=fig.get_figheight();f=lambda inch:1-inch/Hh;tc=next((v for k_,v in TAGC.items()if(tag or"").startswith(k_)),C_MOD)
    fig.text(.05,f(.42),BRAND,fontsize=12,weight="bold",color=C_CART,va="center")
    if tag:fig.text(.95,f(.42),tag,fontsize=12,weight="bold",color=tc,va="center",ha="right",bbox=dict(boxstyle="round,pad=0.35",fc="white",ec=tc,lw=1.6))
    fig.text(.05,f(.95),mt(title),fontsize=29,weight="bold",color=C_FG,va="center")
    if sub:fig.text(.05,f(1.42),mt(sub),fontsize=13.5,color=C_NEU,va="center")
    fig.add_artist(Line2D([.05,.95],[f(1.70),f(1.70)],transform=fig.transFigure,color=C_FG,lw=1.2));return f(1.85)
def foot(fig,ctx,k):
    fig.text(.5,.22/fig.get_figheight(),mt(f"{PRODUCT} · WORLD LAUNCH · © 2026 {AUTHOR} | RUN {ctx['run_id']} | PAYLOAD SHA-256 {ctx['sha'][:16]}… | {k:02d}/{ctx['N']:02d}"),ha="center",va="center",fontsize=9.5,color=C_NEU,family=MONO)
def clean_ax(ax):
    for s in("top","right"):ax.spines[s].set_visible(False)
def panel(fig,x,y,w,h,face,edge,lw=2.2,r=0.012,z=0):
    fig.add_artist(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={r}",transform=fig.transFigure,facecolor=face,edgecolor=edge,lw=lw,zorder=z))
def box(fig,x,y,w,h,title,lines,color,face,tfs=13.5,bfs=11,tcol=None,bcol=C_FG):
    panel(fig,x,y,w,h,face,color,2.4)
    fig.text(x+w/2,y+h*.72,mt(title),ha="center",va="center",fontsize=tfs,weight="bold",color=tcol or color)
    fig.text(x+w/2,y+h*.32,mt(lines),ha="center",va="center",fontsize=bfs,color=bcol,linespacing=1.35)
def arrow(fig,x0,y0,x1,y1,color=C_FG,lw=2.3,ms=24):fig.add_artist(FancyArrowPatch((x0,y0),(x1,y1),transform=fig.transFigure,arrowstyle="-|>",mutation_scale=ms,lw=lw,color=color))
def chip(fig,x,y,w,h,label,value,color,face,vfs=19):
    panel(fig,x,y,w,h,face,color,2,0.008)
    fig.text(x+w/2,y+h*.62,mt(value),ha="center",va="center",fontsize=vfs,weight="bold",color=color);fig.text(x+w/2,y+h*.22,mt(label),ha="center",va="center",fontsize=10.5,color=C_FG)
def okc(b):return C_OK if b else C_ERR
def frozen_ai(fig,x,y,w,h,label="FROZEN AI",sub=MODEL_SHORT,note="weights locked",lines=28):
    """The frozen model drawn as a dark block of 28 layer bands with a padlock."""
    panel(fig,x,y,w,h,C_AI,C_FG,2.4,0.012,z=2)
    for j in range(lines):
        yy=y+h*.12+j*(h*.50/lines);fig.add_artist(Line2D([x+w*.12,x+w*.88],[yy,yy],transform=fig.transFigure,color="#334155",lw=1.1,zorder=3))
    ax=fig.add_axes([x+w*.38,y+h*.64,w*.24,h*.17]);ax.set_zorder(10);ax.patch.set_alpha(0);ax.set_xlim(0,1);ax.set_ylim(0,1.3);ax.set_aspect("equal",adjustable="box");ax.axis("off")
    ax.add_patch(Arc((.5,.62),.5,.6,theta1=0,theta2=180,color=C_GLOW,lw=3));ax.add_patch(FancyBboxPatch((.15,.05),.7,.58,boxstyle="round,pad=0,rounding_size=.08",fc=C_GLOW,ec=C_GLOW))
    fig.text(x+w/2,y+h*.92,label,ha="center",va="center",fontsize=14,weight="bold",color="white",zorder=4)
    fig.text(x+w/2,y+h*.06,mt(sub+(" · "+note if note else"")),ha="center",va="center",fontsize=8.6,color="#CBD5E1",zorder=4)
def doc_icon(fig,x,y,w,h,face="white",edge="#94A3B8",lines=3,alpha=1.0):
    fig.add_artist(Rectangle((x,y),w,h,transform=fig.transFigure,facecolor=face,edgecolor=edge,lw=.8,alpha=alpha))
    for j in range(lines):
        yy=y+h*(.75-j*.22);fig.add_artist(Line2D([x+w*.15,x+w*(.85 if j<lines-1 else .55)],[yy,yy],transform=fig.transFigure,color=edge,lw=.9,alpha=alpha))
def num_row(vals,n=5):return"["+", ".join(f"{v:+.3f}"for v in vals[:n])+", …]"
def silhouette(fig,x,y,w,h,color="#475569"):
    ax=fig.add_axes([x,y,w,h]);ax.set_zorder(10);ax.patch.set_alpha(0);ax.set_xlim(0,1);ax.set_ylim(0,2);ax.set_aspect("equal",adjustable="box");ax.axis("off")
    ax.add_patch(Circle((.5,1.55),.24,fc=color,ec=color));ax.add_patch(Polygon([[.08,0],[.92,0],[.84,.95],[.66,1.22],[.34,1.22],[.16,.95]],closed=True,fc=color,ec=color))
def feat(ctx):
    R=ctx["R"];P=ctx["P"];q=next(z for z in R["queries"]if z["record"]==FEATURE_RECORD);r=next(z for z in P["raw"]["queries"]if z["record"]==FEATURE_RECORD)
    a=next(z for z in P["panel"]["showcase_audit"]if z["record"]==FEATURE_RECORD);return q,r,a
# ------------------------------------------------------------------ 01 WHAT WE BUILT
def p01(ctx,k,path):
    P=ctx["P"];R=ctx["R"];fig=new_fig()
    top=head(fig,"WHAT WE BUILT","A frozen AI keeps model-native numeric memories — and a natural question finds the associated one.",tag="THIS LIVE RUN")
    fig.text(.5,top-.075,PRODUCT,fontsize=58,weight="bold",color=C_FG,ha="center",va="center")
    fig.text(.5,top-.155,PRODUCT_LONG.upper(),fontsize=21,weight="bold",color=C_CART,ha="center",va="center")
    steps=[("READ ONCE","test records enter\na frozen AI"),("LANGUAGE ENDS","memory becomes\nmodel-native numbers"),("SOURCE LEAVES","removed from the live\nretrieval path"),
           ("NEW QUESTION","creates a recall trace:\nÇAĞRIİZ"),("FINGERPRINT","a 512-number\nnumeric address"),("ONE MEMORY LOCKS","strongest match in\nthe memory field")]
    bw=.135;gap=(.9-6*bw)/5;bh=.165;y=top-.42
    panel(fig,.05+3*(bw+gap)-gap/2-.004,y-.035,.955-(.05+3*(bw+gap)-gap/2)+.004,bh+.07,C_NIGHT,C_NIGHT,0,0.014)
    fig.text(.05,y+bh+.02,"HUMAN WORLD",fontsize=10.5,weight="bold",color=C_NEU);fig.text(.94,y+bh+.02,"MACHINE-NATIVE MEMORY SPACE",fontsize=10.5,weight="bold",color=C_CYAN,ha="right")
    for i,(t,b)in enumerate(steps):
        x=.05+i*(bw+gap);dark=i>=3
        panel(fig,x,y,bw,bh,C_NIGHT2 if dark else C_BG,C_CYAN if dark else C_FG,2.0)
        fig.text(x+bw/2,y+bh*.80,str(i+1),ha="center",va="center",fontsize=15,weight="bold",color=C_GLOW if dark else C_CART)
        fig.text(x+bw/2,y+bh*.55,t,ha="center",va="center",fontsize=12.5,weight="bold",color="white"if dark else C_FG)
        fig.text(x+bw/2,y+bh*.22,b,ha="center",va="center",fontsize=9.8,color="#CBD5E1"if dark else C_NEU,linespacing=1.3)
        if i<5:arrow(fig,x+bw+.003,y+bh/2,x+bw+gap-.003,y+bh/2,color=C_CYAN if i>=2 else C_FG,lw=2,ms=18)
    yb=.085;hb=.215
    panel(fig,.05,yb,.425,hb,C_MODL,C_MOD,2.6);panel(fig,.525,yb,.425,hb,C_SEALL,C_SEAL,2.6)
    fig.text(.2625,yb+hb-.032,"WORLD LAUNCH LIVE SHOWCASE · TEST524",ha="center",fontsize=13,weight="bold",color=C_MOD,va="center")
    fig.text(.2625,yb+hb*.47,f"{R['correct']} / {R['total']}",ha="center",va="center",fontsize=50,weight="bold",color=C_MOD)
    fig.text(.2625,yb+.025,"five fixed questions · measured in this run",ha="center",fontsize=11,color=C_FG,va="center")
    fig.text(.7375,yb+hb-.032,"EXTERNAL REPLICATION · TEST523",ha="center",fontsize=13,weight="bold",color=C_SEAL,va="center")
    fig.text(.7375,yb+hb*.47,f"{SEALED523['r1'][0]} / {SEALED523['r1'][1]}",ha="center",va="center",fontsize=50,weight="bold",color=C_SEAL)
    fig.text(.7375,yb+.025,f"{SEALED523['r1_pct']} TOP-1 · sealed historical record",ha="center",fontsize=11,color=C_FG,va="center")
    fig.text(.5,yb+hb/2,"two\nseparate\nexperiments",ha="center",va="center",fontsize=9,color=C_NEU,style="italic",linespacing=1.2)
    fig.text(.5,.045,f"{AUTHOR} · {AUTHOR_PLACE} · {LAUNCH_DATE}",ha="center",va="center",fontsize=11.5,color=C_NEU,weight="bold")
    foot(fig,ctx,k);return finish(fig,path,ctx,[f"{R['correct']} / {R['total']}",f"{SEALED523['r1'][0]} / {SEALED523['r1'][1]}",SEALED523["r1_pct"],PRODUCT,"ÇAĞRIİZ"])
# ------------------------------------------------------------------ 02 READ ONCE
def p02(ctx,k,path):
    P=ctx["P"];fz=P["frozen"];B=P["bank"];fig=new_fig()
    top=head(fig,"STEP 1 — READ ONCE","For this experiment, 128 records are presented to the frozen AI. It reads each one once.",tag="THIS LIVE RUN")
    panel(fig,.05,top-.475,.27,.44,C_BG,C_FG,2.0);fig.text(.185,top-.065,"TEST BANK: 128 RECORDS",ha="center",fontsize=14,weight="bold",color=C_FG)
    for i in range(128):
        r_,c_=divmod(i,16);doc_icon(fig,.065+c_*.0152,top-.115-r_*.042-.033,.0125,.033,lines=3)
    fig.text(.185,top-.46,"human-readable records",ha="center",fontsize=10.5,color=C_NEU,style="italic")
    arrow(fig,.33,top-.255,.375,top-.255)
    frozen_ai(fig,.385,top-.45,.23,.39)
    fig.text(.5,top-.475,"The AI's learned weights are locked\nand are not changed by this experiment.",ha="center",va="top",fontsize=10.5,color=C_FG,linespacing=1.3)
    arrow(fig,.625,top-.255,.67,top-.255,color=C_CYAN)
    panel(fig,.68,top-.475,.27,.44,C_NIGHT,C_CYAN,2.0);fig.text(.815,top-.065,"256 NUMERIC MEMORIES",ha="center",fontsize=14,weight="bold",color="white")
    sl=P["panel"]["a_slots"]+P["panel"]["b_slots"];lo,hi=min(sl),max(sl)
    for i in range(256):
        r_,c_=divmod(i,16);v=(sl[i]-lo)/max(1,hi-lo);col=(0.25+0.4*v,0.75+0.2*v,0.95)
        fig.add_artist(Rectangle((.695+c_*.0152,top-.115-r_*.0198-.016),.0125,.016,transform=fig.transFigure,facecolor=col,edgecolor="none"))
    fig.text(.755,top-.445,"128 A",ha="center",fontsize=10,color=C_CYAN,weight="bold");fig.text(.875,top-.445,"128 B",ha="center",fontsize=10,color=C_CYAN,weight="bold")
    fig.text(.815,top-.46,"independently forged A/B numeric memories",ha="center",fontsize=9.2,color="#CBD5E1",style="italic",va="top")
    panel(fig,.05,.075,.9,.205,C_OKL,C_OK,2.4)
    fig.text(.5,.245,"NO TRAINING OCCURS",ha="center",va="center",fontsize=30,weight="bold",color=C_OK)
    fig.text(.5,.198,"The brain stays frozen. The external memory changes.",ha="center",va="center",fontsize=16,color=C_FG,weight="bold")
    same=fz["guard_pre_run"]==fz["guard_after"]
    fig.text(.08,.145,f"MODEL BEFORE  ·  weight guard {fz['guard_pre_run'][:20]}…",fontsize=10.5,family=MONO,color=C_FG,va="center")
    fig.text(.08,.115,f"MODEL AFTER   ·  weight guard {fz['guard_after'][:20]}…",fontsize=10.5,family=MONO,color=C_FG,va="center")
    fig.text(.92,.13,"✓ IDENTICAL"if same else"✗ CHANGED",fontsize=17,weight="bold",color=okc(same),ha="right",va="center")
    fig.text(.5,.09,f"The memory side holds {B['numbers_A']+B['numbers_B']:,} model-derived numbers. 128 is the size of this experiment, not the architectural memory limit.",ha="center",va="center",fontsize=10.2,color=C_NEU)
    foot(fig,ctx,k);return finish(fig,path,ctx,["NO TRAINING OCCURS","TEST BANK: 128 RECORDS",fz["guard_after"][:20],f"{B['numbers_A']+B['numbers_B']:,}"])
# ------------------------------------------------------------------ 03 SIGNATURE · HUMAN LANGUAGE ENDS HERE
def p03(ctx,k,path):
    P=ctx["P"];ex=P["tensor_excerpt"];q,r,a=feat(ctx);fig=new_fig();rng=random.Random(3)
    top=head(fig,"HUMAN LANGUAGE ENDS HERE","Human-readable source → frozen AI → machine-native memory.",tag="THIS LIVE RUN")
    yb=.265;hb=top-yb-.065
    panel(fig,.05,yb,.36,hb,"white",C_FG,2.0);panel(fig,.565,yb,.385,hb,C_NIGHT,C_NIGHT,2.0)
    for j in range(4):doc_icon(fig,.075+j*.05,yb+.03,.038,.075,lines=4)
    fig.text(.29,yb+.065,"documents ·\nsentences · words",fontsize=9.5,color=C_NEU,va="center",linespacing=1.25)
    fig.text(.07,yb+hb-.035,"HUMAN WORLD",fontsize=15,weight="bold",color=C_FG,va="center");fig.text(.07,yb+hb-.07,"readable text · words · sentences · questions",fontsize=10.5,color=C_NEU,va="center")
    fig.text(.93,yb+hb-.035,"MACHINE-NATIVE MEMORY SPACE",fontsize=15,weight="bold",color=C_CYAN,va="center",ha="right")
    fig.text(.93,yb+hb-.07,"K tensors · V tensors · high-dimensional numbers",fontsize=10.5,color="#CBD5E1",va="center",ha="right")
    sent=f"Instrument {a['entity']} carries seal {a['seal']}."
    fit_text(fig,.075,yb+hb*.30,.31,hb*.38,f"“{sent}”",fs_max=24,fs_min=14,weight="bold",color=C_FG)
    fig.text(.075,yb+hb*.30,f"one of the three sentences read into memory A#{FEATURE_RECORD:03d}",fontsize=9.6,color=C_NEU,style="italic",va="center")
    stream=[c for c in sent if c.strip()]
    for j in range(34):
        t_=j/33;x=.33+t_*.078;y=yb+hb*.52+rng.uniform(-1,1)*hb*(.03+.10*t_);fig.text(x,y,rng.choice(stream),fontsize=11-5*t_,color=C_FG,alpha=max(.05,.75-.7*t_),ha="center",va="center",weight="bold")
    frozen_ai(fig,.415,yb+hb*.18,.145,hb*.64,label="FROZEN AI",sub=MODEL_SHORT,note="")
    for j in range(30):
        t_=j/29;x=.572+t_*.038;y=yb+hb*.52+rng.uniform(-1,1)*hb*.10*(1-t_);fig.text(x,y,rng.choice("0123456789.+-"),fontsize=6+5*t_,color=C_CYAN,alpha=.15+.75*t_,ha="center",va="center",family=MONO)
    fig.add_artist(Line2D([.5625,.5625],[yb-.012,yb+hb+.012],transform=fig.transFigure,color=C_GLOW,lw=3.2,ls=(0,(6,4)),zorder=6))
    fig.text(.5625,yb+hb+.03,"LANGUAGE  →  MODEL-NATIVE NUMBERS",ha="center",va="center",fontsize=12,weight="bold",color=C_CART,zorder=7,bbox=dict(boxstyle="round,pad=0.3",fc="white",ec=C_CART,lw=1.6))
    rows=ex["a_v"]
    for j,row in enumerate(rows):
        t_=j/max(1,len(rows)-1);fig.text(.625,yb+hb*(.78-j*.075),num_row(row),fontsize=12.5-5*t_,color=C_CYAN,alpha=1-.75*t_,family=MONO,va="center")
    fig.text(.625,yb+hb*.15,"…  thousands more values per memory, into depth",fontsize=9.5,color="#94A3B8",family=MONO,va="center")
    fig.text(.625,yb+hb*.07,f"REAL VALUES FROM THIS RUN · memory A#{FEATURE_RECORD:03d} · layer 2 · V tensor",fontsize=9.5,color=C_GLOW,va="center",weight="bold")
    fig.text(.5,.205,"BEYOND THIS POINT, THE MEMORY IS NO LONGER STORED AS HUMAN-READABLE LANGUAGE.",ha="center",va="center",fontsize=17.5,weight="bold",color=C_FG)
    fig.text(.5,.15,"THE AI IS MATCHING ITS OWN NUMERIC REPRESENTATIONS.",ha="center",va="center",fontsize=20,weight="bold",color=C_MOD)
    fit_text(fig,.08,.045,.84,.065,"The live retrieval object is model-derived numeric tensor state rather than ordinary human-readable source text. It was created by processing language, so it carries information derived from it — stored and matched as numbers.",fs_max=10.8,fs_min=8.5,color=C_NEU,ha="center")
    foot(fig,ctx,k);return finish(fig,path,ctx,["HUMAN LANGUAGE ENDS HERE","MACHINE-NATIVE MEMORY SPACE",num_row(rows[0]),"NUMERIC REPRESENTATIONS"])
# ------------------------------------------------------------------ 04 INSIDE A BELLEKÖZ
def p04(ctx,k,path):
    P=ctx["P"];ex=P["tensor_excerpt"];fig=new_fig()
    top=head(fig,"THE RETRIEVAL MEMORY IS NOT A TEXT FILE.","It is model-derived numeric state: K/V tensors produced inside the frozen AI while it read the source. This is a BELLEKÖZ.",tag="THIS LIVE RUN")
    chain=[("SOURCE TEXT",C_FG,C_BG,C_FG),("TOKENS",C_FG,C_BG,C_FG),("FROZEN TRANSFORMER",C_AI,C_AI,"white"),("INTERNAL ACTIVATIONS",C_NIGHT2,C_NIGHT2,"white"),("K / V TENSORS",C_CYAN,C_NIGHT,C_CYAN),("BELLEKÖZ",C_GLOW,C_NIGHT,C_GLOW)]
    bh=.072;g=.024;y0=top-.02
    for i,(t,ec,fc,tc)in enumerate(chain):
        y=y0-(i+1)*bh-i*g;panel(fig,.05,y,.2,bh,fc,ec,2.0);fig.text(.15,y+bh/2,t,ha="center",va="center",fontsize=12,weight="bold",color=tc)
        if i<5:arrow(fig,.15,y-.002,.15,y-g+.002,color=C_NEU,lw=1.8,ms=14)
    yk=top-.03;kw_=.27
    panel(fig,.28,yk-.19,kw_,.19,C_MODL,C_MOD,2.2);fig.text(.295,yk-.03,"K — KEY",fontsize=15,weight="bold",color=C_MOD,va="center")
    fit_text(fig,.295,yk-.18,kw_-.03,.125,"A numerical structure the transformer's attention uses to decide what information is relevant. It is not a filename or a database key.",fs_max=12,fs_min=9)
    panel(fig,.28,yk-.41,kw_,.19,C_VIOL,C_VIO,2.2);fig.text(.295,yk-.25,"V — VALUE",fontsize=15,weight="bold",color=C_VIO,va="center")
    fit_text(fig,.295,yk-.40,kw_-.03,.125,"The numerical information that attention can retrieve and use once relevance has been decided.",fs_max=12,fs_min=9)
    fit_text(fig,.28,yk-.52,kw_,.09,"AKBASCORE MAM keeps the model-derived K/V tensor state as numeric memory. Nobody assigns it a human-readable address.",fs_max=11.5,fs_min=9,weight="bold",color=C_FG)
    H=np.array(ex["a_heat"],dtype=float);ax=fig.add_axes([.665,top-.50,.285,.42]);lim=float(np.percentile(np.abs(H),97))or 1.0
    ax.imshow(H,aspect="auto",cmap="RdBu_r",vmin=-lim,vmax=lim,interpolation="nearest");ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values():s.set_color(C_FG)
    fig.text(.665,top-.045,f"MEMORY A#{FEATURE_RECORD:03d} · layer 2 · V · head 0 · {H.shape[0]} slots × first 64 of 128 dims",fontsize=9.5,color=C_NEU,va="center")
    fig.text(.8075,top-.025,"REAL VALUES FROM THIS RUN",fontsize=11,weight="bold",color=C_MOD,va="center",ha="center")
    silhouette(fig,.585,top-.47,.07,.24)
    fig.text(.8075,top-.535,"A HUMAN CANNOT READ THIS AS A SENTENCE.",ha="center",va="center",fontsize=13.5,weight="bold",color=C_FG)
    fit_text(fig,.585,top-.62,.365,.06,"BELLEKÖZ is numerical model state. Software can inspect its values mathematically, but it is not stored as ordinary human-readable language.",fs_max=9.8,fs_min=8,color=C_NEU)
    panel(fig,.05,.065,.9,.115,C_NIGHT,C_NIGHT,0)
    fig.text(.5,.14,"NO DOCUMENT TO REREAD.     NO SENTENCE TO SEARCH.     ONLY MODEL-NATIVE NUMERIC MEMORY.",ha="center",va="center",fontsize=16.5,weight="bold",color=C_GLOW)
    fig.text(.5,.093,f"One memory = {ex['numbers_A']:,} numbers: 28 layers × K and V × 4 heads × 128 dimensions × each memory slot.",ha="center",va="center",fontsize=11,color="#CBD5E1")
    foot(fig,ctx,k);return finish(fig,path,ctx,["THE RETRIEVAL MEMORY IS NOT A TEXT FILE.","A HUMAN CANNOT READ THIS AS A SENTENCE.","K — KEY","V — VALUE",f"{ex['numbers_A']:,}"])
# ------------------------------------------------------------------ 05 SOURCE REMOVED
def p05(ctx,k,path):
    P=ctx["P"];wp=P["wipe"];fig=new_fig()
    top=head(fig,"SOURCE REMOVED FROM THE LIVE RETRIEVAL PATH","After memory creation, the retrieval mechanism no longer receives the original source text.",tag="THIS LIVE RUN")
    cw=.27;g=(.9-3*cw)/2;yb=.27;hb=top-yb-.02;xs=[.05+i*(cw+g)for i in range(3)]
    panel(fig,xs[0],yb,cw,hb,C_BG,C_FG,2.0);panel(fig,xs[1],yb,cw,hb,C_ERRL,C_ERR,2.0);panel(fig,xs[2],yb,cw,hb,C_NIGHT,C_CYAN,2.0)
    fig.text(xs[0]+cw/2,yb+hb-.035,"1 · READ TIME",ha="center",fontsize=15,weight="bold",color=C_FG,va="center")
    fig.text(xs[1]+cw/2,yb+hb-.035,"2 · REMOVED",ha="center",fontsize=15,weight="bold",color=C_ERR,va="center")
    fig.text(xs[2]+cw/2,yb+hb-.035,"3 · LIVE RETRIEVAL",ha="center",fontsize=15,weight="bold",color=C_CYAN,va="center")
    for j in range(5):doc_icon(fig,xs[0]+.035+j*.042,yb+hb*.52,.034,.085,lines=4)
    fit_text(fig,xs[0]+.02,yb+.03,cw-.04,hb*.36,f"{wp['source_records_before']} source records are read once by the frozen AI to create the numeric memories.",fs_max=13,fs_min=9,ha="center")
    for j,n_ in enumerate(wp["names"]):
        y=yb+hb-.10-j*.058;gone=not wp["present"][n_]
        fig.text(xs[1]+.02,y,WIPE_LABELS.get(n_,n_),fontsize=11.5,color=C_FG,va="center")
        fig.text(xs[1]+cw-.02,y,"✓ deleted"if gone else"✗ PRESENT",fontsize=11.5,weight="bold",color=okc(gone),va="center",ha="right")
    fit_text(fig,xs[1]+.02,yb+.025,cw-.04,.09,"deleted before the first question is asked",fs_max=11,fs_min=8.5,color=C_ERR,weight="bold",ha="center")
    keep=["numeric A memories (K/V tensors)","numeric B memories (K/V tensors)","numeric B address structures","the new question","a fresh cache for every question","conversation history: empty"]
    for j,t in enumerate(keep):fig.text(xs[2]+.02,yb+hb-.10-j*.058,"•  "+t,fontsize=11.5,color="white",va="center")
    fit_text(fig,xs[2]+.02,yb+.025,cw-.04,.09,"what the retriever can use",fs_max=11,fs_min=8.5,color=C_CYAN,weight="bold",ha="center")
    arrow(fig,xs[0]+cw+.004,yb+hb/2,xs[1]-.004,yb+hb/2,lw=2,ms=18);arrow(fig,xs[1]+cw+.004,yb+hb/2,xs[2]-.004,yb+hb/2,color=C_CYAN,lw=2,ms=18)
    ok=not wp["source_text_present"]
    fig.text(.5,.19,"SOURCE TEXT IN THE LIVE RETRIEVAL PATH:  "+("ABSENT"if ok else"PRESENT"),ha="center",va="center",fontsize=21,weight="bold",color=okc(ok))
    fit_text(fig,.08,.05,.84,.1,"Scope: the source is removed from the live retrieval path of this run; it does not mean every copy of the information anywhere has been erased. Audit metadata (record labels and expected seals) is kept apart and used only after retrieval, to verify what happened. Technical flag: source_text_container_destroyed = "+str(ok)+".",fs_max=10.5,fs_min=8.5,color=C_NEU,ha="center")
    foot(fig,ctx,k);return finish(fig,path,ctx,["SOURCE REMOVED FROM THE LIVE RETRIEVAL PATH","ABSENT"if ok else"PRESENT",str(wp["source_records_before"])])
# ------------------------------------------------------------------ 06 HOW A QUESTION FINDS A MEMORY
def p06(ctx,k,path):
    P=ctx["P"];q,r,a=feat(ctx);fig=new_fig();lab=f"#{FEATURE_RECORD:03d}"
    top=head(fig,"HOW A QUESTION FINDS AN ASSOCIATED MEMORY","One numeric memory is already active. A new natural question makes the frozen AI find its associated memory.",tag="THIS LIVE RUN")
    y=top-.25;h=.2
    panel(fig,.05,y+h*.52,.2,h*.48,C_NIGHT,C_CYAN,2.0);fig.text(.15,y+h*.85,"ACTIVE MEMORY A",ha="center",va="center",fontsize=12.5,weight="bold",color=C_CYAN);fig.text(.15,y+h*.64,"numeric K/V only",ha="center",va="center",fontsize=10,color="#CBD5E1")
    fig.text(.15,y+h*.42,"+",ha="center",va="center",fontsize=24,weight="bold",color=C_FG)
    panel(fig,.05,y-h*.18,.2,h*.48,"white",C_FG,2.0);fit_text(fig,.06,y-h*.16,.18,h*.40,"NEW NATURAL QUESTION\n“"+a["question"]+"”",fs_max=10.5,fs_min=8,ha="center")
    arrow(fig,.255,y+h*.3,.3,y+h*.3);frozen_ai(fig,.305,y-h*.18,.17,h*1.18,note="")
    arrow(fig,.48,y+h*.3,.525,y+h*.3,color=C_CYAN)
    panel(fig,.53,y-h*.18,.42,h*1.18,C_NIGHT,C_CYAN,2.0)
    fig.text(.74,y+h*.86,"FIND THE ASSOCIATED MEMORY B",ha="center",va="center",fontsize=14,weight="bold",color="white")
    fig.text(.74,y+h*.68,"among the B memories of the experimental field",ha="center",va="center",fontsize=10.5,color="#CBD5E1")
    for i in range(N):
        rr_,cc=divmod(i,32);fig.add_artist(Rectangle((.55+cc*.0118,y+h*.38-rr_*.022),.0098,.017,transform=fig.transFigure,facecolor=C_GLOW if i+1==q["selected"]else"#1F3A5F",edgecolor="none"))
    yc=.085;hc=(top-.25)-.2*.18-.06-yc
    panel(fig,.05,yc,.36,hc,C_BG,"#94A3B8",1.6);panel(fig,.43,yc,.52,hc,C_MODL,C_MOD,2.2)
    fig.text(.23,yc+hc-.032,"CLASSIC EXPLICIT LOOKUP",ha="center",va="center",fontsize=13,weight="bold",color=C_CTL)
    fig.text(.23,yc+hc*.58,f"memory_id = {FEATURE_RECORD:03d}",ha="center",va="center",fontsize=17,family=MONO,color=C_FG)
    fig.text(.23,yc+hc*.40,"↓",ha="center",va="center",fontsize=16,color=C_NEU)
    fig.text(.23,yc+hc*.24,f"retrieve record {FEATURE_RECORD:03d}",ha="center",va="center",fontsize=14,family=MONO,color=C_FG)
    fig.text(.69,yc+hc-.032,"AKBASCORE MAM",ha="center",va="center",fontsize=13,weight="bold",color=C_MOD)
    chain="natural question  →  frozen-model recall trace  →  model-native numeric address\n→  similarity against numeric memory representations  →  associated memory selected"
    fig.text(.69,yc+hc*.62,chain,ha="center",va="center",fontsize=11.2,color=C_FG,linespacing=1.6)
    fig.text(.69,yc+hc*.33,f'THE QUESTION DOES NOT SAY "{lab}".',ha="center",va="center",fontsize=19,weight="bold",color=C_MOD)
    fig.text(.69,yc+hc*.15,f'"{lab}" IS ONLY AN AUDIT LABEL.',ha="center",va="center",fontsize=14,weight="bold",color=C_FG)
    fig.text(.5,.055,"Memory numbers are display/audit labels only. The retriever is not given the correct number.",ha="center",va="center",fontsize=11.5,color=C_NEU,style="italic")
    foot(fig,ctx,k);return finish(fig,path,ctx,[f'THE QUESTION DOES NOT SAY "{lab}".',"IS ONLY AN AUDIT LABEL",a["entity"],"ACTIVE MEMORY A"])
# ------------------------------------------------------------------ 07 SIGNATURE · NEW QUESTION → ÇAĞRIİZ → NUMERIC FINGERPRINT
def p07(ctx,k,path):
    P=ctx["P"];q,r,a=feat(ctx);fig=new_fig()
    top=head(fig,"NEW QUESTION → ÇAĞRIİZ → NUMERIC FINGERPRINT","Inside the frozen AI, the question becomes a recall trace — and the trace becomes a numeric address.",tag="THIS LIVE RUN")
    yb=.30;hb=top-yb-.02
    panel(fig,.05,yb,.19,hb,"white",C_FG,2.0);fig.text(.145,yb+hb-.035,"A NEW QUESTION",ha="center",va="center",fontsize=13,weight="bold",color=C_FG)
    fit_text(fig,.06,yb+hb*.36,.17,hb*.48,"“"+a["question"]+"”",fs_max=14,fs_min=9,weight="bold",ha="center")
    fig.text(.145,yb+hb*.22,"enters as language",ha="center",va="center",fontsize=10,color=C_NEU,style="italic")
    panel(fig,.065,yb+.02,.16,.06,C_NIGHT,C_CYAN,1.6);fig.text(.145,yb+.05,f"+ active memory A#{FEATURE_RECORD:03d}",ha="center",va="center",fontsize=9.5,color=C_CYAN,weight="bold")
    arrow(fig,.245,yb+hb/2,.27,yb+hb/2,color=C_CYAN)
    panel(fig,.275,yb,.395,hb,C_NIGHT,C_VIO,2.2)
    fig.text(.4725,yb+hb-.035,"ÇAĞRIİZ",ha="center",va="center",fontsize=20,weight="bold",color=C_GLOW)
    fig.text(.4725,yb+hb-.075,"a question-dependent numeric recall trace created inside the frozen AI",ha="center",va="center",fontsize=10,color="#E2E8F0")
    w=np.array(r["w"],dtype=float);xs_=np.arange(len(w));ax=fig.add_axes([.30,yb+.075,.345,hb-.20]);ax.set_facecolor(C_NIGHT)
    cols=[C_GLOW if i==q["pos"]else"#8B5CF6"for i in xs_];ax.bar(xs_,w,color=cols,width=.8);ax.set_xlim(-.6,len(w)-.4);ax.set_ylim(0,max(w)*1.15)
    for s in("top","right"):ax.spines[s].set_visible(False)
    for s in("left","bottom"):ax.spines[s].set_color("#64748B")
    ax.tick_params(colors="#CBD5E1",labelsize=8);ax.set_xlabel("position inside active memory A (numbers only, no words)",fontsize=9,color="#CBD5E1",labelpad=2)
    ax.annotate(f"peak {q['peak']:.3f}",xy=(q["pos"],q["peak"]),xytext=(q["pos"]+(-6 if q["pos"]>len(w)/2 else 3),q["peak"]*1.03),color=C_GLOW,fontsize=10,weight="bold",arrowprops=dict(arrowstyle="->",color=C_GLOW))
    fig.text(.4725,yb+.025,"the AI's own attention decides where to look",ha="center",va="center",fontsize=10,color=C_CYAN,weight="bold")
    arrow(fig,.675,yb+hb/2,.7,yb+hb/2,color=C_GLOW)
    panel(fig,.705,yb,.245,hb,C_NIGHT,C_CYAN,2.2)
    fig.text(.8275,yb+hb-.035,"NUMERIC FINGERPRINT",ha="center",va="center",fontsize=14,weight="bold",color=C_CYAN)
    fig.text(.8275,yb+hb-.075,f"{len(r['p'])} numbers · model-native tensor address",ha="center",va="center",fontsize=10,color="#E2E8F0")
    pv=np.array(r["p"],dtype=float);G=pv.reshape(32,-1)if pv.size%32==0 else pv[None,:];lim=float(np.percentile(np.abs(pv),97))or 1.0
    ax2=fig.add_axes([.72,yb+.075,.215,hb-.20]);ax2.imshow(G,aspect="auto",cmap="RdBu_r",vmin=-lim,vmax=lim,interpolation="nearest");ax2.set_xticks([]);ax2.set_yticks([])
    fig.text(.8275,yb+.025,"real values from this run",ha="center",va="center",fontsize=10,color=C_CYAN,weight="bold")
    fig.text(.5,.235,"THE QUESTION BECOMES A NUMERIC FINGERPRINT.",ha="center",va="center",fontsize=22,weight="bold",color=C_FG)
    fig.text(.5,.185,"Not a memory number · not a filename · not a keyword · not a decoded seal.",ha="center",va="center",fontsize=13,color=C_NEU,weight="bold")
    fit_text(fig,.07,.045,.86,.105,"L23H12 — a specific attention channel inside the frozen model (layer 23, head 12), identified in earlier experiments and externally replicated in TEST523. "
             "LAYER 2 — VALUE TENSOR (L02-V) — the numeric memory representation that the recall trace weights into the address. Technical: frozen L23H12 native Q·K + RoPE attention; pointer-weighted, uncentered L02-V.",fs_max=10.5,fs_min=8.2,color=C_NEU,ha="center")
    foot(fig,ctx,k);return finish(fig,path,ctx,["ÇAĞRIİZ","NUMERIC FINGERPRINT",f"peak {q['peak']:.3f}",f"{len(r['p'])} numbers","THE QUESTION BECOMES A NUMERIC FINGERPRINT."])
# ===== END PART 2 / 3 — CONTINUE WITH PART 3 =====
from matplotlib.colors import LinearSegmentedColormap
FIELD_CMAP=LinearSegmentedColormap.from_list("field",["#16213A","#1E3A8A","#0E7490","#22D3EE","#FDE68A"])
# ------------------------------------------------------------------ 08 SIGNATURE · FINGERPRINT → FIELD → ONE MEMORY LOCKS
def p08(ctx,k,path):
    P=ctx["P"];q,r,a=feat(ctx);fig=new_fig();sc=np.array(r["scores"],dtype=float)
    top=head(fig,"FINGERPRINT → 128-MEMORY FIELD → ONE MEMORY LOCKS","The numeric fingerprint is compared with every B memory of the experimental field. The strongest match locks.",tag="THIS LIVE RUN")
    yb=.33;hb=top-yb-.02
    panel(fig,.05,yb,.15,hb,C_NIGHT,C_CYAN,2.0)
    fig.text(.125,yb+hb-.045,"NUMERIC\nFINGERPRINT",ha="center",va="center",fontsize=12,weight="bold",color=C_CYAN,linespacing=1.15)
    pv=np.array(r["p"],dtype=float);lim=float(np.percentile(np.abs(pv),97))or 1.0;G=pv.reshape(64,-1)if pv.size%64==0 else pv[:,None]
    ax=fig.add_axes([.075,yb+.07,.10,hb-.19]);ax.imshow(G,aspect="auto",cmap="RdBu_r",vmin=-lim,vmax=lim,interpolation="nearest");ax.axis("off")
    fig.text(.125,yb+.035,f"{pv.size} numbers",ha="center",va="center",fontsize=10,color="#E2E8F0")
    arrow(fig,.205,yb+hb/2,.235,yb+hb/2,color=C_GLOW)
    panel(fig,.24,yb,.47,hb,C_NIGHT,C_NIGHT,0)
    fig.text(.475,yb+hb-.032,"128-MEMORY EXPERIMENTAL FIELD",ha="center",va="center",fontsize=14,weight="bold",color="white")
    lo,hi=float(sc.min()),float(sc.max());cols_,rows_=16,8;gx0=.255;gw=.44;gy0=yb+.02;gh=hb-.085;cw=gw/cols_;ch=gh/rows_
    for j in range(N):
        rr_,cc=divmod(j,cols_);x=gx0+cc*cw;y=gy0+gh-(rr_+1)*ch;v=(sc[j]-lo)/max(1e-9,hi-lo);sel=j+1==q["selected"];gold=j+1==q["record"]
        fig.add_artist(FancyBboxPatch((x+cw*.07,y+ch*.09),cw*.86,ch*.82,boxstyle="round,pad=0,rounding_size=0.004",transform=fig.transFigure,facecolor=FIELD_CMAP(v**2),
                                      edgecolor=C_GLOW if sel else("white"if gold else"none"),lw=3.2 if sel else(1.6 if gold else 0),ls="-"if sel else"--",zorder=3))
        fig.text(x+cw/2,y+ch*.5,f"{j+1:03d}",ha="center",va="center",fontsize=6.8,color=C_NIGHT if v**2>.55 else"#94A3B8",zorder=4,family=MONO)
    panel(fig,.725,yb,.225,hb,"white",C_FG,2.0)
    fig.text(.8375,yb+hb-.032,"STRONGEST MATCHES",ha="center",va="center",fontsize=13,weight="bold",color=C_FG)
    for i,(rec,s)in enumerate(q["top10"][:6]):
        y=yb+hb-.09-i*(hb-.16)/6;lock=i==0
        fig.text(.737,y,f"#{rec:03d}",fontsize=12,family=MONO,weight="bold"if lock else"normal",color=C_FG,va="center")
        fig.add_artist(Rectangle((.79,y-.011),.10*max(0,s),.022,transform=fig.transFigure,facecolor=C_GLOW if lock else"#93C5FD",edgecolor="none"))
        fig.text(.893,y,f"{s:.6f}",fontsize=10,family=MONO,color=C_FG,va="center")
        if lock:fig.text(.8375,y-.032,"← LOCKED"+(""if q["correct"]else"  (not the target)"),fontsize=10,weight="bold",color=C_CART if not q["correct"]else C_OK,va="center",ha="center")
    fig.text(.8375,yb+.03,"higher score = closer numeric match",ha="center",va="center",fontsize=9.5,color=C_NEU,style="italic")
    fig.text(.5,.255,"NUMBERS FIND NUMBERS.",ha="center",va="center",fontsize=36,weight="bold",color=C_FG)
    fig.text(.5,.195,"Each memory receives a similarity score. Higher means its numeric pattern is a closer match to the recall address.",ha="center",va="center",fontsize=13,color=C_FG)
    fit_text(fig,.07,.045,.86,.115,"128-MEMORY EXPERIMENTAL FIELD: the experimental bank size — not a hard-coded capacity limit. Brightness and glow are a visual metaphor for numeric similarity. "
             f"Featured: showcase question 3 of 5, fixed before the run (all five on the next poster). Researchers: ÇAĞRIİZ → weighted L02-V address → max-cosine numeric memory matching.",fs_max=10.5,fs_min=8.2,color=C_NEU,ha="center")
    foot(fig,ctx,k);return finish(fig,path,ctx,["NUMBERS FIND NUMBERS.",f"{q['top10'][0][1]:.6f}","128-MEMORY EXPERIMENTAL FIELD",f"#{q['selected']:03d}"])
# ------------------------------------------------------------------ 09 WATCH IT WORK
def p09(ctx,k,path):
    R=ctx["R"];fig=new_fig()
    top=head(fig,"WATCH IT WORK · FIVE LIVE QUESTIONS","Fixed before the run. Every question competes against all 128 B memories.",tag="THIS LIVE RUN")
    rh=(top-.20)/5
    for i,q in enumerate(R["queries"]):
        y=top-.01-(i+1)*rh;ok=q["correct"]
        panel(fig,.05,y+.006,.9,rh-.012,"white",C_OK if ok else C_CART,1.6 if ok else 2.2)
        fig.text(.065,y+rh*.62,f"Q{q['k']}",fontsize=22,weight="bold",color=C_MOD,va="center")
        fig.text(.11,y+rh*.66,f"instrument {q['entity']}",fontsize=13.5,weight="bold",color=C_FG,va="center")
        fig.text(.11,y+rh*.36,f"active memory A#{q['record']:03d} · {1000*(q['secs']or 0):.0f} ms",fontsize=10.5,color=C_NEU,va="center")
        ax=fig.add_axes([.355,y+rh*.15,.33,rh*.70]);t5=q["top10"][:5];vals=[s for _,s in t5]
        ax.barh(range(5),vals,color=[C_GLOW if j==q["selected"]else(C_OK if j==q["record"]else"#BFDBFE")for j,_ in t5],height=.72);ax.invert_yaxis()
        ax.set_yticks(range(5));ax.set_yticklabels([f"#{j:03d}"for j,_ in t5],fontsize=9,family=MONO);ax.set_xlim(0,1.13);ax.set_xticks([])
        for s_ in("top","right","bottom"):ax.spines[s_].set_visible(False)
        for t_,v in enumerate(vals):ax.text(v+.01,t_,f"{v:.4f}",va="center",fontsize=8.5,family=MONO,color=C_FG)
        if ok:
            fig.text(.82,y+rh*.62,f"LOCKED  B#{q['selected']:03d}",fontsize=17,weight="bold",color=C_OK,ha="center",va="center")
            fig.text(.82,y+rh*.32,f"correct · lead {q['margin']:+.3f}",fontsize=10.5,color=C_FG,ha="center",va="center")
        else:
            fig.text(.82,y+rh*.70,"ONE REAL MISS",fontsize=14,weight="bold",color=C_CART,ha="center",va="center")
            fig.text(.82,y+rh*.45,f"selected #{q['selected']:03d} · correct #{q['record']:03d}",fontsize=10.5,color=C_FG,ha="center",va="center")
            fig.text(.82,y+rh*.22,f"correct memory ranked {q['rank']}/{N}",fontsize=10.5,color=C_FG,ha="center",va="center")
    fig.text(.5,.135,f"{R['correct']} / {R['total']}   FIXED LIVE SHOWCASE · TEST524",ha="center",va="center",fontsize=24,weight="bold",color=C_MOD)
    msg="Memory numbers are audit labels; the retriever never receives them. Gold bar = the memory that locked"+("; green bar = the correct memory when it did not lock."if R["misses"]else".")
    if R["misses"]:msg="The system is experimental, not a scripted lookup: one of the five fixed questions misses its target. "+msg
    fit_text(fig,.08,.045,.84,.06,msg,fs_max=11,fs_min=8.5,color=C_NEU,ha="center")
    foot(fig,ctx,k);return finish(fig,path,ctx,[f"{R['correct']} / {R['total']}"]+[f"instrument {q['entity']}"for q in R["queries"]])
# ------------------------------------------------------------------ 10 ONE RECALL PROCESS
def p10(ctx,k,path):
    P=ctx["P"];R=ctx["R"];fig=new_fig();ms=[1000*(q["secs"]or 0)for q in R["queries"]]
    top=head(fig,"0 CANDIDATE-B LLM FORWARDS","The 7B model is not rerun separately for each of the 128 B-memory candidates.",tag="THIS LIVE RUN")
    yb=.30;hb=top-yb-.02
    panel(fig,.05,yb,.42,hb,C_BG,"#94A3B8",1.6);panel(fig,.53,yb,.42,hb,C_MODL,C_MOD,2.4)
    fig.text(.26,yb+hb-.035,"REPEATED LARGE-MODEL PROCESSING",ha="center",va="center",fontsize=13.5,weight="bold",color=C_CTL)
    fig.text(.26,yb+hb-.07,"(the approach this design avoids)",ha="center",va="center",fontsize=10,color=C_NEU,style="italic")
    labs=["Candidate 1","Candidate 2","Candidate 3","…","Candidate 128"]
    for i,l in enumerate(labs):
        y=yb+hb-.14-i*(hb-.20)/5
        if l=="…":fig.text(.26,y,"⋮",ha="center",va="center",fontsize=20,color=C_NEU);continue
        panel(fig,.085,y-.025,.13,.05,"white","#94A3B8",1.2,0.006);fig.text(.15,y,l,ha="center",va="center",fontsize=10.5,color=C_FG)
        arrow(fig,.22,y,.285,y,color="#94A3B8",lw=1.6,ms=14)
        panel(fig,.29,y-.025,.15,.05,"#E2E8F0","#94A3B8",1.2,0.006);fig.text(.365,y,"LLM forward",ha="center",va="center",fontsize=10.5,color=C_CTL,weight="bold")
    fig.text(.74,yb+hb-.035,"AKBASCORE MAM · TEST524",ha="center",va="center",fontsize=13.5,weight="bold",color=C_MOD)
    flow=[("QUESTION + ACTIVE A","white",C_FG,C_FG),("FROZEN MODEL",C_AI,C_AI,"white"),("ONE RECALL / ADDRESS PROCESS",C_NIGHT,C_VIO,C_GLOW),("NUMERIC MEMORY-FIELD COMPARISON",C_NIGHT,C_CYAN,C_CYAN)]
    for i,(t,fc,ec,tc)in enumerate(flow):
        y=yb+hb-.15-i*(hb-.20)/4;panel(fig,.58,y-.03,.32,.06,fc,ec,2.0,0.008);fig.text(.74,y,t,ha="center",va="center",fontsize=11.5,weight="bold",color=tc)
        if i<3:arrow(fig,.74,y-.032,.74,y-(hb-.20)/4+.032,color=C_NEU,lw=1.8,ms=14)
    panel(fig,.05,.105,.9,.165,C_NIGHT,C_NIGHT,0)
    fig.text(.5,.225,"CANDIDATE-B LLM FORWARDS = "+str(P["counters"]["candidate_B_llm_forwards"]),ha="center",va="center",fontsize=26,weight="bold",color=C_GLOW)
    fig.text(.5,.17,"The already-created numeric memory representations are compared mathematically.",ha="center",va="center",fontsize=14,color="white")
    rng_="%.0f ms"%min(ms)if round(min(ms))==round(max(ms))else"%.0f–%.0f ms"%(min(ms),max(ms))
    fig.text(.5,.13,f"Measured live: {rng_} per question for the recall process plus 128 numeric comparisons.",ha="center",va="center",fontsize=11,color=C_CYAN)
    fig.text(.5,.06,"Retrieval still computes: one model pass over the question with the active memory, then 128 numeric similarity comparisons.",ha="center",va="center",fontsize=10.5,color=C_NEU,style="italic")
    foot(fig,ctx,k);return finish(fig,path,ctx,["CANDIDATE-B LLM FORWARDS = "+str(P["counters"]["candidate_B_llm_forwards"]),"REPEATED LARGE-MODEL PROCESSING","ONE RECALL / ADDRESS PROCESS"])
# ------------------------------------------------------------------ 11 MEASURED EVIDENCE
def p11(ctx,k,path):
    P=ctx["P"];R=ctx["R"];S=SEALED524;fig=new_fig();sv={z["k"]:z for z in S["queries"]};rp=R["replay"]
    top=head(fig,"MEASURED EVIDENCE","Two separate experiments, shown separately: the live launch showcase and the larger sealed external replication.")
    yb=.30;hb=top-yb-.02
    panel(fig,.05,yb,.425,hb,C_MODL,C_MOD,2.6);panel(fig,.525,yb,.425,hb,C_SEALL,C_SEAL,2.6)
    fig.text(.2625,yb+hb-.035,"WORLD LAUNCH LIVE SHOWCASE · TEST524",ha="center",va="center",fontsize=13.5,weight="bold",color=C_MOD)
    fig.text(.2625,yb+hb-.068,"measured in this run",ha="center",va="center",fontsize=10.5,color=C_NEU,style="italic")
    fig.text(.2625,yb+hb*.66,f"{R['correct']} / {R['total']}",ha="center",va="center",fontsize=56,weight="bold",color=C_MOD)
    fig.text(.2625,yb+hb*.47,"five fixed launch examples · top-1 correct",ha="center",va="center",fontsize=12,color=C_FG)
    for i,q in enumerate(R["queries"]):
        x=.085+i*.0715;cq=C_OK if q["correct"]else C_CART;panel(fig,x,yb+.06,.062,.085,"white",cq,1.8,0.006)
        fig.text(x+.031,yb+.12,f"Q{q['k']}",ha="center",va="center",fontsize=10,weight="bold",color=C_FG);fig.text(x+.031,yb+.083,f"rank {q['rank']}",ha="center",va="center",fontsize=9.5,color=cq,weight="bold")
    fig.text(.2625,yb+.03,"rank of the correct memory among 128",ha="center",va="center",fontsize=9.5,color=C_NEU)
    fig.text(.7375,yb+hb-.035,"EXTERNAL REPLICATION · TEST523",ha="center",va="center",fontsize=13.5,weight="bold",color=C_SEAL)
    fig.text(.7375,yb+hb-.068,"sealed historical record",ha="center",va="center",fontsize=10.5,color=C_NEU,style="italic")
    fig.text(.7375,yb+hb*.66,f"{SEALED523['r1'][0]} / {SEALED523['r1'][1]}",ha="center",va="center",fontsize=56,weight="bold",color=C_SEAL)
    fig.text(.7375,yb+hb*.47,f"{SEALED523['r1_pct']} TOP-1",ha="center",va="center",fontsize=16,weight="bold",color=C_FG)
    fig.text(.7375,yb+hb*.30,f"counterfactual follow {fr(SEALED523['cf'])} = {SEALED523['cf_pct']}",ha="center",va="center",fontsize=12,color=C_FG)
    fig.text(.7375,yb+hb*.18,"the address followed a changed memory relation",ha="center",va="center",fontsize=10,color=C_NEU,style="italic")
    fig.text(.7375,yb+.03,"same frozen pointer, address and match as this demo",ha="center",va="center",fontsize=9.5,color=C_NEU)
    same="YES"if rp["match"]else"NO"
    fig.text(.5,.24,f"Live selections and ranks identical to the sealed TEST524 launch log: {same}",ha="center",va="center",fontsize=14,weight="bold",color=okc(rp["match"]))
    fig.text(.5,.195,f"largest top-score difference vs the sealed log: {rp['top1_max_abs_diff']:.6f}"+("  ·  same GPU model as the sealed run"if P["hardware"]["same"]else"  ·  different GPU from the sealed run: small numeric differences are expected"),ha="center",va="center",fontsize=10.5,color=C_NEU)
    fig.text(.5,.13,"TEST524 = five fixed launch examples.     TEST523 = larger external replication experiment.",ha="center",va="center",fontsize=14,color=C_FG,weight="bold")
    fig.text(.5,.075,"These two numbers come from different experiments and are never added together.",ha="center",va="center",fontsize=11,color=C_NEU,style="italic")
    foot(fig,ctx,k);return finish(fig,path,ctx,[f"{R['correct']} / {R['total']}",f"{SEALED523['r1'][0]} / {SEALED523['r1'][1]}",SEALED523["r1_pct"],SEALED523["cf_pct"]])
# ------------------------------------------------------------------ 12 EXPERIMENTAL INTEGRITY
def p12(ctx,k,path):
    P=ctx["P"];fz=P["frozen"];wp=P["wipe"];fig=new_fig();same=fz["guard_startup"]==fz["guard_after"]
    top=head(fig,"EXPERIMENTAL INTEGRITY","What the live retrieval path actually uses — transparent properties of the architecture.")
    items=[("Frozen weights",same,"weights are not changed by this experiment"),("No training",True,"no optimizer, no adapter, no weight update"),
           ("Source text in the retrieval path: absent",not wp["source_text_present"],"removed before the first question"),("No gold B memory ID supplied",True,"the retriever receives only the question and the active memory"),
           ("No decoded-text router",True,"no keyword, filename or decoded seal is used to route"),("No learned router",True,"nothing is fitted or trained for retrieval"),
           ("Candidate-B LLM forwards: "+str(P["counters"]["candidate_B_llm_forwards"]),P["counters"]["candidate_B_llm_forwards"]==0,"B memories are compared numerically, not re-run"),
           ("Numeric matching",True,"max cosine similarity over all 128 B memories"),("Fresh cache for every question",True,"no conversation history is carried over"),
           ("Frozen recall channel and address",True,"L23H12 ÇAĞRIİZ · L02-V address — fixed by the TEST524 lock")]
    cw=.43;rh=(top-.30)/5
    for i,(t,ok,sub)in enumerate(items):
        c_,r_=divmod(i,5);x=.05+c_*(cw+.04);y=top-.02-(r_+1)*rh
        panel(fig,x,y+.008,cw,rh-.016,"white","#CBD5E1",1.4,0.01)
        fig.text(x+.03,y+rh/2,"✓"if ok else"✗",fontsize=26,weight="bold",color=okc(ok),va="center",ha="center")
        fig.text(x+.06,y+rh*.62,t,fontsize=14,weight="bold",color=C_FG,va="center");fig.text(x+.06,y+rh*.32,sub,fontsize=10.5,color=C_NEU,va="center")
    panel(fig,.05,.08,.9,.16,C_BG,C_FG,1.6)
    fig.text(.5,.212,"MODEL BEFORE  =  MODEL AFTER",ha="center",va="center",fontsize=17,weight="bold",color=okc(same))
    fig.text(.5,.165,f"all-parameter weight guard · before {fz['guard_startup'][:24]}…",ha="center",va="center",fontsize=10.5,family=MONO,color=C_FG)
    fig.text(.5,.135,f"after  {fz['guard_after'][:24]}…",ha="center",va="center",fontsize=10.5,family=MONO,color=C_FG)
    fig.text(.5,.098,"every parameter tensor of the 7-billion-parameter model participates in this guard",ha="center",va="center",fontsize=10,color=C_NEU,style="italic")
    fig.text(.5,.045,"Verification method for each property (measured in this run or fixed by the code path) is listed in the researcher view.",ha="center",va="center",fontsize=10,color=C_NEU)
    foot(fig,ctx,k);return finish(fig,path,ctx,["EXPERIMENTAL INTEGRITY","Frozen weights","Numeric matching",fz["guard_after"][:24]])
# ------------------------------------------------------------------ 13 WHERE IT COULD LEAD
def p13(ctx,k,path):
    R=ctx["R"];fig=new_fig()
    top=head(fig,"WHERE THIS COULD LEAD","What was demonstrated today — and the research direction it opens.",tag="RESEARCH VISION")
    yb=.10;hb=top-yb-.02
    panel(fig,.05,yb,.32,hb,C_OKL,C_OK,2.6);panel(fig,.40,yb,.55,hb,C_CARTL,C_CART,2.6)
    fig.text(.21,yb+hb-.035,"DEMONSTRATED TODAY",ha="center",va="center",fontsize=16,weight="bold",color=C_OK)
    demo=["128-memory experimental field","frozen AI · no training","source removed from the retrieval path","question → ÇAĞRIİZ → numeric fingerprint","numeric association among 128 B memories",f"{R['correct']}/{R['total']} live · TEST523: {SEALED523['r1'][0]}/{SEALED523['r1'][1]} sealed"]
    for i,t in enumerate(demo):fig.text(.07,yb+hb-.10-i*.075,"✓  "+t,fontsize=12,color=C_FG,va="center")
    fig.text(.675,yb+hb-.035,"RESEARCH VISION / SCALING DIRECTION",ha="center",va="center",fontsize=16,weight="bold",color=C_CART)
    lad=[("128","CURRENT DEMO",True),("1K","",False),("1M","",False),("1B","",False)]
    for i,(n_,lab,real)in enumerate(lad):
        x=.43+i*.13;y=yb+hb-.215
        panel(fig,x,y,.10,.095,C_OK if real else"white",C_OK if real else C_CART,2.2 if real else 1.6)
        fig.text(x+.05,y+.055,n_,ha="center",va="center",fontsize=22,weight="bold",color="white"if real else C_CART)
        fig.text(x+.05,y+.018,lab or"memories",ha="center",va="center",fontsize=8.8,weight="bold",color="white"if real else C_NEU)
        if i<3:arrow(fig,x+.10,y+.047,x+.13,y+.047,color=C_CART,lw=1.8,ms=14)
    fig.text(.675,yb+hb-.25,"1K, 1M and 1B are research directions — not yet experimentally validated.",ha="center",va="center",fontsize=10,color=C_NEU,style="italic")
    vis=["very large persistent memory banks","hierarchical / indexed memory routing","compatible memory transfer between machines","autonomous multi-memory chains","long-lived AI memory","modular machine memory","robotics and distributed agents"]
    for i,t in enumerate(vis):
        c_,r_=divmod(i,4);fig.text(.425+c_*.315,yb+hb-.32-r_*.07,"→  "+t,fontsize=11.5,color=C_FG,va="center",weight="bold")
    fit_text(fig,.43,yb+.02,.5,.08,"These capabilities are a research vision built on today's mechanism. They are not available now and have not been demonstrated by this run.",fs_max=11,fs_min=8.5,color=C_NEU)
    foot(fig,ctx,k);return finish(fig,path,ctx,["DEMONSTRATED TODAY","RESEARCH VISION / SCALING DIRECTION","CURRENT DEMO","1B"])
# ------------------------------------------------------------------ 14 RESEARCHER VIEW · MECHANISM AND RESULTS
def p14(ctx,k,path):
    P=ctx["P"];R=ctx["R"];fig=new_fig()
    top=head(fig,"RESEARCHER VIEW · MECHANISM AND RESULTS","Exact computation of the TEST524 engine and every live value. Frozen by the TEST524 lock; parent: sealed TEST523.",tag="RESEARCHER VIEW")
    panel(fig,.05,top-.235,.9,.215,C_BG,C_FG,1.4)
    eq=["ÇAĞRIİZ   w_t  = softmax_t ( RoPE(q_L23,H12(final question token)) · RoPE(k_L23(A_t)) / sqrt(128) ),   t ≥ 1   (slot 0 excluded)",
        f"ADDRESS   p    = Σ_t  w_t · V_L2(A_t)                       {ADDR_DIM} numbers · uncentered · from the active A memory",
        "SCORE     s(B) = max_s  cos( p , V_L2(B_s) )                over every slot s ≥ 1 of B · all 128 B memories",
        "SELECT    B*   = argmax_B s(B)                              no candidate-B model forward · ties broken by lower index"]
    for i,t in enumerate(eq):fig.text(.065,top-.06-i*.045,t,fontsize=10.6,family=MONO,color=C_FG,va="center")
    hdr=["Q","active A","selected","gold rank","top score","lead","ÇAĞRIİZ peak","peak slot","entropy","ms"];xs=[.06,.10,.18,.26,.34,.43,.52,.62,.71,.80]
    yt=top-.27
    for h_,x in zip(hdr,xs):fig.text(x,yt,h_,fontsize=10,weight="bold",color=C_NEU,va="center")
    for i,q in enumerate(R["queries"]):
        y=yt-.038*(i+1);vals=[f"Q{q['k']}",f"#{q['record']:03d}",f"#{q['selected']:03d}",f"{q['rank']}/{N}",f"{q['top1']:+.6f}",f"{q['margin']:+.6f}",f"{q['peak']:.4f}",f"{q['pos']}/{q['T']}",f"{q['entropy']:.3f}",f"{1000*(q['secs']or 0):.1f}"]
        for v,x in zip(vals,xs):fig.text(x,y,v,fontsize=10,family=MONO,color=(C_OK if q["correct"]else C_CART)if x==.18 else C_FG,va="center")
    fig.text(.06,yt-.038*6-.005,f"live showcase {ci_text(R['correct'],R['total'])} (exact Clopper–Pearson) · replay of sealed TEST524 selections/ranks identical: {R['replay']['match']}",fontsize=9.6,color=C_NEU,va="center")
    fit_text(fig,.05,.045,.9,yt-.038*6-.05-.045,"RESEARCHER NOTES\n"+"\n".join("• "+t for t in P["scope"]["researcher_notes"]),fs_max=10.4,fs_min=7.8,color=C_FG)
    foot(fig,ctx,k);return finish(fig,path,ctx,["RESEARCHER NOTES",f"{R['queries'][0]['top1']:+.6f}",f"{R['queries'][-1]['top1']:+.6f}"])
# ------------------------------------------------------------------ 15 RESEARCHER VIEW · AUDIT
def p15(ctx,k,path):
    P=ctx["P"];fig=new_fig()
    top=head(fig,"RESEARCHER VIEW · VERIFICATION CLASS","Each property of the live retrieval path, and how it is verified: measured in this run, or fixed by the code path.",tag="RESEARCHER VIEW")
    rows=P["protocol"];rh=(top-.10)/len(rows)
    for i,e in enumerate(rows):
        y=top-.01-(i+1)*rh;col=C_OK if e["cls"]=="CHECKED"else C_CTL
        panel(fig,.05,y+.004,.58,rh-.008,"white","#CBD5E1",1.0,0.006)
        fig.text(.06,y+rh*.66,e["key"],fontsize=9.6,weight="bold",color=C_FG,va="center");fig.text(.06,y+rh*.28,e["note"],fontsize=8.1,color=C_NEU,va="center")
        fig.text(.43,y+rh/2,e["value"],fontsize=8.8,weight="bold",family=MONO,color=C_FG,va="center")
        fig.text(.625,y+rh/2,e["cls"],fontsize=8.4,weight="bold",color="white",va="center",ha="right",bbox=dict(boxstyle="round,pad=0.25",fc=col,ec=col))
    panel(fig,.66,.10,.29,top-.11,C_BG,C_FG,1.4)
    txt=("CHECKED = measured or verified during this run.\nBY CONSTRUCTION = a property of the code path, not separately instrumented (no hooks are added to the engine).\n\n"
         "NOT CLAIMED BY THIS RUN\n"+"\n".join("• "+t for t in P["scope"]["not_claimed"])+
         f"\n\n{len(P['checks_pre_seal'])} technical checks passed before sealing; post-seal checks verify posters, ZIP and hashes.\nHashes and the guard are integrity indicators, not third-party verification.")
    fit_text(fig,.675,.115,.26,top-.145,txt,fs_max=10.5,fs_min=7.6,color=C_FG)
    foot(fig,ctx,k);return finish(fig,path,ctx,["CHECKED","BY CONSTRUCTION","NOT CLAIMED BY THIS RUN"])
# ------------------------------------------------------------------ 16 EVIDENCE SEAL
def p16(ctx,k,path):
    P=ctx["P"];R=ctx["R"];E_=P["environment"];tm=P["timing"];H_=P["hardware"];fig=new_fig()
    top=head(fig,"COMPLETE RUN · EVIDENCE SEAL","Everything below was read from the runtime and is contained in the sealed payload.",tag="THIS LIVE RUN")
    tiles=[("RUN ID",P["run_id"]),("MODEL",MODEL_ID+" · frozen"),("GPU · SAME AS SEALED?",f"{E_['gpu']} · {'YES'if H_['same']else'NO'}"),("TORCH / TRANSFORMERS",f"{E_['torch']} / {E_['transformers']}"),
           ("TEST BANK",f"{N} records · {2*N} memories"),("LIVE SHOWCASE",f"{R['correct']}/{R['total']} · TEST524"),("TEST524 LOCK SHA",LOCK_SHA[:22]+"…"),("TECHNICAL CHECKS",f"{len(P['checks_pre_seal'])} passed pre-seal"),
           ("PEAK GPU MEMORY",f"{P['gpu']['peak_allocated_gib']:.2f} GiB"),("VERDICT",P["verdict"])]
    for i,(a_,b_)in enumerate(tiles):
        r_,c_=divmod(i,5);x=.05+c_*.182;y=top-.125-r_*.125;panel(fig,x,y,.172,.105,C_BG,C_MOD,1.8,0.01)
        fig.text(x+.008,y+.083,a_,fontsize=9,weight="bold",color=C_MOD,va="center");fit_text(fig,x+.008,y+.008,.156,.058,b_,fs_max=11.5,fs_min=6.8,family=MONO)
    st=[("model load (startup)",tm["model_load_seconds"]),("test bank",tm["test_bank"]),("read once · 256 memories",tm["read_once_forge"]),("numeric address field",tm["address_field"]),
        ("source removal",tm["source_removal"]),("5 live retrievals",tm["live_retrieval"])]
    ax=fig.add_axes([.25,.30,.67,top-.59]);ax.barh(range(len(st)),[s for _,s in st],color=[C_CTL,C_CART,C_CART,C_MOD,C_ERR,C_VIO]);ax.invert_yaxis();ax.set_yticks(range(len(st)));ax.set_yticklabels([a for a,_ in st],fontsize=10.5)
    for i,(_,s)in enumerate(st):ax.text(s,i,f" {s:.2f} s",va="center",fontsize=10.5,weight="bold")
    ax.set_xlim(0,max(max(s for _,s in st)*1.2,1.0));ax.set_xlabel(f"measured seconds (sealed TEST524 total runtime {SEALED524['runtime_s']:.2f} s on {CANON_GPU})",fontsize=9.5);clean_ax(ax)
    txt=f"PAYLOAD FILE : {ctx['payload_name']}\nPAYLOAD SHA-256 : {ctx['sha']}\nENGINE : TEST524 engine functions, unchanged · parent TEST523 lock {PARENT523[:16]}…\nIMAGE HASHES : images manifest (per-image SHA-256), written after rendering\nArtifact integrity seal ≠ scientific proof and ≠ independent third-party verification.\n{COPYRIGHT}"
    fit_text(fig,.05,.045,.9,.215,txt,fs_max=11.5,fs_min=8,family=MONO)
    foot(fig,ctx,k);return finish(fig,path,ctx,[P["run_id"],E_["gpu"],P["verdict"],f"{R['correct']}/{R['total']}"])
POSTERS=[("01_what_we_built","What we built",p01),("02_read_once","Step 1 — read once",p02),("03_human_language_ends_here","Human language ends here",p03),
 ("04_not_a_text_file","The retrieval memory is not a text file",p04),("05_source_removed","Source removed from the live retrieval path",p05),("06_question_finds_memory","How a question finds an associated memory",p06),
 ("07_question_cagriiz_fingerprint","New question → ÇAĞRIİZ → numeric fingerprint",p07),("08_one_memory_locks","Fingerprint → 128-memory field → one memory locks",p08),
 ("09_watch_it_work","Watch it work · five live questions",p09),("10_zero_candidate_forwards","0 candidate-B LLM forwards",p10),("11_measured_evidence","Measured evidence",p11),
 ("12_experimental_integrity","Experimental integrity",p12),("13_where_it_could_lead","Where this could lead",p13),("14_researcher_mechanism","Researcher view · mechanism and results",p14),
 ("15_researcher_verification","Researcher view · verification class",p15),("16_evidence_seal","Evidence seal",p16)]
N_POSTERS=len(POSTERS)
if N_POSTERS>20 or[s[:2]for s,_,_ in POSTERS]!=[f"{i:02d}"for i in range(1,N_POSTERS+1)]:raise RuntimeError("Poster count/numbering check failed.")
def render_all(ctx,run_dir):
    imgs=[]
    try:
        for i,(slug,cap,fn)in enumerate(POSTERS,1):
            p=Path(run_dir)/f"{slug}_{ctx['run_id']}.jpg";fn(ctx,i,p);imgs.append((p,cap))
    finally:plt.close("all")
    return imgs
#<<POSTERS_END>>
#<<UI_BEGIN>>
class SelfTestEngine:
    """Synthetic engine used ONLY by the service self-test (no GPU, no model). Never used for a real run."""
    def __init__(s,mode="ok"):
        s.counters=Counter();s.mode=mode;s.n=0;s.rng=np.random.default_rng(7);s.g="a"*64
        s.info=dict(model_id=MODEL_ID,arch=ARCH,dtype="bfloat16",attn="sdpa",gpu="SELF-TEST ENGINE (no GPU)",canonical_gpu=CANON_GPU,gpu_total_gib=0.0,torch="n/a",transformers="n/a",gradio="n/a",
            python="n/a",platform="n/a",params=1,pad_id=0,vocab=1,model_load_seconds=0.0,startup_utc="n/a",guard0=s.g,hooks0=5,cell_source_sha256=None,guard_method="synthetic")
    def frozen_state(s):
        fh=3 if s.mode=="foreign_hook"else 0
        return dict(training=False,trainable_tensors=0,requires_grad_disabled=True,lora=False,optimizer=False,hooks=5+fh,foreign_hooks=fh,foreign_modules=({"__main__":fh}if fh else{}),
                    guard=("b"*64 if s.mode=="guard_changes"and s.n>0 else s.g))
    def prepare_panel(s):
        nm=build_names(N*3);codes=[f"Z{chr(65+i//26)}{chr(65+i%26)}Q"for i in range(N*6)];seals=codes[:N*3];classes=codes[N*3:]
        audit=[{"record":i+1,"entity":nm[3*i],"seal":seals[3*i],"class":classes[3*i]}for i in range(N)]
        show=[{"record":i+1,"entity":audit[i]["entity"],"question":qA(audit[i]["entity"])}for i in SHOWCASE_IDX]
        if s.mode=="leak":show[0]["question"]=show[0]["question"]+" "+seals[0]
        return dict(pool=N*6,records=N,showcase=show,audit=audit,seconds=0.0,audit_sha256=hashlib.sha256(canon(audit)).hexdigest())
    def forge_one(s,i):
        s.counters["forge_passes"]+=2;TA=28+i%5;TB=34+i%4
        return dict(TA=TA,TB=TB,numbers_A=2*28*4*128*TA,numbers_B=2*28*4*128*TB,secs=0.0,finite=True)
    def counts(s):return dict(A=N,B=N,B_MATS=N,A_PACK=N)
    def build_address_field(s):return dict(b_mats=N,rows=[33+i%4 for i in range(N)],dim=ARCH[3]*ARCH[4],a_packs=N,seconds=0.0,storages=N)
    def wipe(s):
        present={k_:(s.mode=="no_wipe"and k_=="SOURCES")for k_ in WIPED_NAMES}
        return dict(source_records_before=N,names=list(WIPED_NAMES),present=present,source_text_present=any(present.values()))
    def retrieve_live(s,k_,record,question):
        s.counters["query_retrievals"]+=1;s.n+=1
        sq=next(q for q in SEALED524["queries"]if q["record"]==record);g=record-1;sc=[0.5-0.001*j for j in range(N)]
        if sq["selected"]==record:sc[g]=sq["top1"]
        else:
            sel=sq["selected"]-1;others=[sel]+[j for j in range(N)if j not in(g,sel)][:sq["rank"]-2]
            for t,j in enumerate(others):sc[j]=sq["top1"]-0.001*t
            sc[g]=0.52
        T=30;w=[0.0]+[0.02]*(T-1);w[sq["pos"]]=sq["peak"]*1.6;tot=sum(w);w=[x/tot for x in w]
        order=sorted(range(N),key=lambda j:(-sc[j],j))
        return dict(k=k_,record=record,question=question,w=w,p=[round(float(x),6)for x in s.rng.normal(0,0.3,ARCH[3]*ARCH[4])],scores=sc,order=[j+1 for j in order[:10]],secs=0.05,b_forwards_counter=0)
    def tensor_excerpt(s,record):
        r=s.rng;Hm=(r.normal(0,1,(29,64))*np.exp(-np.arange(64)/40)).round(4)
        return dict(record=record,layer=BL,head=0,a_shape=[4,30,128],b_shape=[35,512],a_v=r.normal(0,.4,(8,8)).round(3).tolist(),a_k=r.normal(0,.4,(4,8)).round(3).tolist(),
                    b_rows=r.normal(0,.4,(8,8)).round(3).tolist(),a_heat=Hm.tolist(),numbers_A=2*28*4*128*30,numbers_B=2*28*4*128*35)
    def sync(s):pass
    def reset_peak(s):pass
    def peak_gib(s):return 0.0
def drain(gen):
    while True:
        try:next(gen)
        except StopIteration as s_:return s_.value
RUN_COUNTER=0;GPU_LOCK=threading.Lock()
SKIP=getattr(gr,"skip",None)or gr.update
_K=object()
FILE_KEYS=["zip","json","txt","man"]
DL_LABELS=["⬇ DOWNLOAD EVERYTHING (.zip)","⬇ DOWNLOAD FULL RUN LOG (.json)","⬇ DOWNLOAD READABLE RUN LOG (.txt)","⬇ DOWNLOAD RUN MANIFEST (.json)"]
DL_ALL_LABEL=f"⬇ DOWNLOAD ALL {N_POSTERS} JPGs"
RAW_KEYS=["txt","json","man"]
N_OUTPUTS=2+2+len(FILE_KEYS)*2+len(RAW_KEYS)
try:
    from gradio.routes import API_PREFIX as _GR_API_PREFIX
except Exception:
    _GR_API_PREFIX="/gradio_api"if int(gr.__version__.split(".")[0])>=5 else""
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
def jpg_urls_json(paths):
    items=[]
    for p in paths:p=Path(p);items.append({"url":FILE_URL_PREFIX+quote(str(p),safe="/"),"name":p.name})
    return json.dumps(items,ensure_ascii=False)
def pack(status=_K,gal=_K,files=_K,jpgs=_K,raw=_K):
    out=[SKIP()if status is _K else status,SKIP()if gal is _K else gal]
    if jpgs is _K:out+=[SKIP(),SKIP()]
    elif jpgs is None:out+=[btn_update(False),""]
    else:
        pl=[Path(p)for p in jpgs]
        for pth in pl:
            if not file_ready(pth):raise RuntimeError(f"JPG artifact missing or empty: {pth}")
        out+=[btn_update(True),jpg_urls_json(pl)]
    if files is _K:out+=[SKIP()for _ in range(len(FILE_KEYS)*2)]
    elif files is None:out+=[dl_update(None,l)for l in DL_LABELS]+[None]*len(FILE_KEYS)
    else:
        paths=[files.get(k_)for k_ in FILE_KEYS]
        for pth in paths:
            if pth is not None and not file_ready(pth):raise RuntimeError(f"Download artifact missing or empty: {pth}")
        out+=[dl_update(p,l)for p,l in zip(paths,DL_LABELS)]+[None if p is None else str(p)for p in paths]
    if raw is _K:out+=[SKIP()for _ in RAW_KEYS]
    elif raw is None:out+=[""]*len(RAW_KEYS)
    else:out+=[raw[k_]for k_ in RAW_KEYS]
    return tuple(out)
def card_html(title,body,kind="info"):return f'<div class="kz card {kind}"><div class="h">{html.escape(title)}</div><div class="small">{body}</div></div>'
def pbar_html(done,total,color="#B45309"):
    pc_=100.0*done/max(1,total);return f'<div style="margin-top:6px;background:#e2e8f0;border-radius:6px;height:14px"><div style="width:{pc_:.1f}%;height:14px;border-radius:6px;background:{color}"></div></div><div class="small">{done}/{total}</div>'
def stage_card(e):
    body=html.escape(e["body"])
    if e.get("total"):body+=pbar_html(e["done"],e["total"],"#0369A1"if e["stage"]>=5 else"#B45309")
    if e.get("eta")is not None:body+=f'<div class="small">estimated time left: {int(e["eta"]//60)} min {int(e["eta"]%60)} s</div>'
    return card_html(f"{e['stage']}/{N_STAGES} · {e['title']}",body,"info")
READY_HTML=card_html("Ready",f"Press <b>RUN WORLD LAUNCH DEMO</b>. The run builds the 128-record test bank, lets the frozen AI read each record once (256 numeric memories), removes the source from the live retrieval path, "
    f"asks the 5 fixed questions — each one scored against all 128 B memories — seals the evidence and renders {N_POSTERS} JPG posters plus one ZIP.<br>"
    f"Canonical sealed hardware: <b>{CANON_GPU}</b>; this run reports the GPU actually used.<br><b>Downloads appear when the run is complete.</b>","info")
def _row_html(q):
    res="LOCKED ✓"if q["correct"]else"miss · correct memory rank %d/%d"%(q["rank"],N)
    return"Q%d · %s · A#%03d → B#%03d · %s<br>"%(q["k"],html.escape(q["entity"]),q["record"],q["selected"],res)
def is_cuda_error(ex):
    s=f"{type(ex).__name__} {ex}".lower();return"cuda"in s or"device-side assert"in s or"cublas"in s
def run_handler():
    global RUN_COUNTER
    if not GPU_LOCK.acquire(blocking=False):
        yield pack(card_html("Busy","Another run is in progress. Please try again in a moment.","warn"));return
    stage_name="initialisation"
    try:
        RUN_COUNTER+=1;run_id=f"MAM524-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}-{RUN_COUNTER:04d}"
        prune_runs(2);run_dir=ROOT/run_id;run_dir.mkdir(parents=True,exist_ok=True);gen=execute_run(ENGINE,dict(run_id=run_id,run_dir=run_dir));first_ev=True
        while True:
            try:e=next(gen)
            except StopIteration as s:B=s.value;break
            stage_name=f"{e['stage']}/{N_STAGES} {e['title']}"
            yield pack(stage_card(e),[],None,None,None)if first_ev else pack(stage_card(e));first_ev=False
        R=B["R"];P=B["P"];rp=R["replay"]
        gallery_items=[(str(p),c_)for p,c_ in B["imgs"]]
        raw_texts={"txt":B["txt"].read_text(encoding="utf-8"),"json":B["payload"].read_bytes().decode("utf-8"),"man":B["manifest"].read_text(encoding="utf-8")}
        rows="".join(_row_html(q)for q in R["queries"])
        body=(f"<b>{html.escape(B['run_id'])}</b><br><b>WORLD LAUNCH LIVE SHOWCASE (TEST524): {R['correct']}/{R['total']}</b> · selections identical to the sealed launch log: {'YES'if rp['match']else'NO'}<br>{rows}"
              f"EXTERNAL REPLICATION (TEST523, sealed record): {SEALED523['r1'][0]}/{SEALED523['r1'][1]} = {SEALED523['r1_pct']} top-1<br>"
              f"verdict: <b>{html.escape(B['verdict'])}</b> · {N_POSTERS} JPGs + ZIP · {B['checks']}/{B['checks']} technical checks PASS<br>"
              f'<span class="mono">payload SHA-256 (artifact integrity seal): {B["sha"]}</span>')
        yield pack(card_html(f"{N_STAGES}/{N_STAGES} · Complete",body,"on"),gallery_items,{"zip":B["zip"],"json":B["payload"],"txt":B["txt"],"man":B["manifest"]},[p for p,_ in B["imgs"]],raw_texts)
    except Exception as ex:
        print("="*140);print(f"WORLD LAUNCH RUN FAILED — stage: {stage_name}");print(f"exception type : {type(ex).__name__}");print(f"exception message: {ex}");traceback.print_exc();print("="*140)
        kind="AUDIT FAIL — no package was sealed. "if isinstance(ex,AuditFail)else"";partial=getattr(ex,"partial",None)
        if partial:kind+="The raw outputs gathered so far were preserved (FULL RUN LOG button / FULL RUN LOG tab). "
        tip="<br><b>CUDA error: Runtime → Restart runtime, then run the cell again.</b>"if is_cuda_error(ex)else"<br>The full traceback is printed in the Colab console."
        card=card_html("Run failed",f"{html.escape(kind)}Stage: {html.escape(stage_name)}<br>{html.escape(type(ex).__name__)}: {html.escape(str(ex)[:600])}{tip}","err")
        if partial:yield pack(card,[],{"json":partial},None,{"txt":traceback.format_exc(),"json":Path(partial).read_text(encoding="utf-8"),"man":""})
        else:yield pack(card,[],None,None,None)
    finally:
        plt.close("all");GPU_LOCK.release()
        try:torch.cuda.empty_cache()
        except Exception:pass
# ---------------- service self-test (runs before the public link is printed) ----------------
def service_selftest():
    global QUIET;out=[];d=ROOT/"_selftest";shutil.rmtree(d,ignore_errors=True);d.mkdir(parents=True);(d/"w.txt").write_text("ok");out.append("ROOT writable");QUIET=True
    try:
        B=drain(execute_run(SelfTestEngine("ok"),dict(run_id="SELFTEST",run_dir=d)))
        assert len(B["imgs"])==N_POSTERS and file_ready(B["zip"])and B["verdict"]=="TEST524_LIVE_SHOWCASE_4_OF_5_INTEGRITY_VERIFIED"and B["R"]["replay"]["match"]
        out.append(f"full pipeline on a synthetic engine: {len(B['imgs'])}/{N_POSTERS} posters (JPEG/RGB), ZIP, {B['checks']} checks")
        for mode,what in(("guard_changes","a changed weight guard"),("leak","a seal inside a question"),("no_wipe","source text left in memory"),("foreign_hook","a foreign hook")):
            d2=d/mode;d2.mkdir()
            try:drain(execute_run(SelfTestEngine(mode),dict(run_id="SELFTEST-"+mode.upper(),run_dir=d2)))
            except AuditFail:out.append(f"fail-closed: {what} aborts the run (no package sealed)")
            else:raise RuntimeError(f"self-test: {what} did not abort the run")
    finally:QUIET=False
    if len(pack(READY_HTML))!=N_OUTPUTS or len(pack(READY_HTML,[],None,None,None))!=N_OUTPUTS:raise RuntimeError("callback output count mismatch")
    out.append(f"callback output count = {N_OUTPUTS}");out.append(f"file route {FILE_URL_PREFIX}");shutil.rmtree(d,ignore_errors=True);return out
say("[4/7] SERVICE SELF-TEST")
for c_ in service_selftest():say(" PASS ·",c_)
# ---------------- interface ----------------
say("[5/7] INTERFACE")
CSS="""
:root{--kz-on:#047857;--kz-fg:#0f172a;--kz-card:#ffffff;--kz-bd:#cbd5e1;--kz-a:#0369a1;--kz-off:#6d28d9;--kz-err:#b91c1c;--kz-warn:#b45309}
.dark{--kz-on:#34d399;--kz-fg:#f8fafc;--kz-card:#0f172a;--kz-bd:#475569;--kz-a:#38bdf8;--kz-off:#c4b5fd;--kz-err:#f87171;--kz-warn:#fbbf24}
html,body{overflow-x:hidden!important}
.gradio-container{width:100%!important;max-width:860px!important;margin:0 auto!important;padding-left:10px!important;padding-right:10px!important;box-sizing:border-box!important}
.gradio-container *{box-sizing:border-box}
.kz{color:var(--kz-fg)!important;max-width:100%;overflow-wrap:anywhere;line-height:1.45}
.kz *{color:inherit}
.hero{text-align:center;padding:10px 2px 2px}
.brand{font-size:clamp(28px,8.5vw,44px);font-weight:800;letter-spacing:1px;line-height:1.05}
.sub{font-size:clamp(13px,4vw,17px);font-weight:700;color:var(--kz-warn)!important;margin-top:4px;letter-spacing:.5px}
.tag{font-size:clamp(13px,3.8vw,15px);opacity:.9;margin-top:8px}
.by{font-size:12px;opacity:.75;margin-top:6px}
.card{background:var(--kz-card);border:2px solid var(--kz-bd);border-radius:12px;padding:12px 14px;margin:6px 0}
.card.on{border-color:var(--kz-on)}.card.err{border-color:var(--kz-err)}.card.warn{border-color:var(--kz-warn)}.card.info{border-color:var(--kz-a)}
.card .h{font-weight:800;font-size:16px;margin-bottom:4px}
.small{font-size:14px}.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;word-break:break-all}
#kz_run button,#kz_run{font-size:clamp(17px,5vw,22px)!important;font-weight:800!important;min-height:64px!important}
"""
DL_ALL_JS="""(u)=>{let items=[];try{items=JSON.parse(u||"[]")}catch(e){items=[]}
let i=0;const next=()=>{if(i>=items.length)return;const it=items[i++];const a=document.createElement('a');a.href=it.url;a.download=it.name;a.rel='noopener';
document.body.appendChild(a);a.click();setTimeout(()=>{a.remove();next()},900)};next();return u;}"""
HERO=('<div class="kz hero"><div class="brand">AKBASCORE MAM</div><div class="sub">PERSISTENT ASSOCIATIVE MACHINE MEMORY · WORLD LAUNCH</div>'
      '<div class="tag">Human language goes in once. A frozen AI keeps model-native numeric memories. A natural question finds the associated one.<br>'
      f'128-record experimental bank · frozen {MODEL_SHORT} · TEST524 engine · numbers find numbers.</div>'
      f'<div class="by">{html.escape(AUTHOR)} · {html.escape(AUTHOR_PLACE)} · {html.escape(LAUNCH_DATE)} · {html.escape(COPYRIGHT)}</div></div>')
def _tb(**kw):
    try:return gr.Textbox(show_copy_button=True,**kw)
    except TypeError:return gr.Textbox(**kw)
def _gallery(**kw):
    try:return gr.Gallery(columns=1,height="auto",object_fit="contain",preview=False,**kw)
    except TypeError:return gr.Gallery(**kw)
try:_blocks=gr.Blocks(css=CSS,title="AKBASCORE MAM · WORLD LAUNCH")
except TypeError:_blocks=gr.Blocks(title="AKBASCORE MAM · WORLD LAUNCH")
with _blocks as demo:
    gr.HTML(HERO)
    run_btn=gr.Button("RUN WORLD LAUNCH DEMO",variant="primary",elem_id="kz_run")
    status=gr.HTML(READY_HTML)
    gallery=_gallery(label=f"{N_POSTERS} JPG posters (tap to open)",show_label=True)
    dl_all=gr.Button(DL_ALL_LABEL,interactive=False)
    jpg_urls=gr.Textbox(value="",visible=False)
    with gr.Row():
        dlb=[gr.DownloadButton(label=l,value=None,interactive=False)for l in DL_LABELS]
    with gr.Accordion("Direct file links (fallback)",open=False):
        fls=[gr.File(label=n,interactive=False)for n in("ZIP · all posters and audit files","FULL RUN LOG (.json)","READABLE RUN LOG (.txt)","RUN MANIFEST (.json)")]
    with gr.Tabs():
        with gr.Tab("RAW LOG / HAM LOG (.txt)"):raw_txt=_tb(lines=18,max_lines=40,label="readable run log")
        with gr.Tab("FULL RUN LOG (.json)"):raw_json=_tb(lines=18,max_lines=40,label="sealed canonical payload")
        with gr.Tab("MANIFEST (.json)"):raw_man=_tb(lines=12,max_lines=30,label="run manifest")
    OUTS=[status,gallery,dl_all,jpg_urls]+dlb+fls+[raw_txt,raw_json,raw_man]
    if len(OUTS)!=N_OUTPUTS:raise RuntimeError(f"UI outputs {len(OUTS)} != {N_OUTPUTS}")
    run_btn.click(run_handler,inputs=None,outputs=OUTS)
    try:dl_all.click(None,inputs=[jpg_urls],outputs=None,js=DL_ALL_JS)
    except TypeError:dl_all.click(None,inputs=[jpg_urls],outputs=None,_js=DL_ALL_JS)
try:demo.queue(default_concurrency_limit=1)
except TypeError:demo.queue(concurrency_count=1)
say("[6/7] LAUNCH (public share link)")
say("[7/7] Open the printed gradio.live link in a new tab and press RUN WORLD LAUNCH DEMO.")
demo.launch(share=True,allowed_paths=ALLOWED_PATHS,show_error=True,inline=False)



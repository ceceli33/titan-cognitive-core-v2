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
